"""GDELT DOC 2.0 news-index connector (Phase 1, off by default).

GDELT is open, unauthenticated and enormous: it indexes worldwide news
continuously and a single loose query answers with more articles per hour than
the rest of Phase 1 combined. Everything specific about this module follows from
those two facts -- free, and far too big to poll carelessly.

**`overlap_seconds = 900`, not the 300 default.** `docs/connector-spec.md` §4.1
rule 3 sets the overlap because "provider indexes lag their own timestamps", and
names GDELT as the eventually-consistent case that needs fifteen minutes. GDELT's
pipeline publishes in 15-minute batches and an article's `seendate` is when GDELT
*saw* it, not when the batch carrying it became queryable -- so a run that
resumed five minutes behind its own watermark would query a range GDELT had not
finished writing, get nothing back, and advance past those records permanently.
Overlap plus dedup is what catches them; the identity key collapses the re-reads
into nothing.

**Identity is rule 2** (`docs/signal-model.md` §4.1): `native_id =
sha256(canonicalize_url(article["url"]))`. GDELT assigns no article id -- the URL
*is* the key it deduplicates on internally -- so the field map declares no
`item_id` and rule 2 is what runs. The same story re-seen in a later batch,
possibly with different tracking parameters attached, canonicalizes to the same
string and therefore to the same Signal.

**Every record is title-only.** GDELT indexes articles; it does not serve them.
`Content.text` is empty and `Content.truncated` is True -- see `_FIELD_MAP`.

**Volume is the design constraint.** `ctx.max_records` and `ctx.max_pages` are
applied *before* the request that would exceed them, not after: `maxrecords` is
narrowed to what is left of the budget, so a run capped at 50 records fetches 50
and not 250. `BaseConnector.run()` enforces the same ceilings after the fact,
which is too late to matter for a source that answers in thousands.

The connector is disabled unless the deployment says otherwise -- see
`from_config` and `GDELT_ENABLED` in `.env.example`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Self
from urllib.parse import urlencode

import httpx

from models.base import utcnow
from models.enums import AuthType, MediaKind, Platform, SourceCategory
from models.signal import MediaRef, Signal
from connectors.base import BaseConnector
from connectors.exceptions import (
    ConnectorConfigurationError,
    NormalizationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.normalize.mapper import (
    FieldMap,
    FieldSpec,
    MappingContext,
    derive_native_id,
)
from connectors.protocol import (
    Credentials,
    Cursor,
    FetchPage,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)

__all__ = ["GdeltConnector"]

DEFAULT_BASE_URL: Final = "https://api.gdeltproject.org/api/v2"
DOC_PATH: Final = "/doc/doc"

MAX_RECORDS_PER_REQUEST: Final = 250
"""The DOC 2.0 ceiling. Asking for more is answered with an error, not a cap."""

DEFAULT_RECORDS_PER_REQUEST: Final = 250
DEFAULT_LOOKBACK_HOURS: Final = 6
"""How far back a first run reaches with no watermark.

Short on purpose: GDELT has no shortage of history, and a cold start that swept a
week would page for hours against an undocumented soft limit before the first
incremental run ever happened.
"""

GDELT_OVERLAP_SECONDS: Final = 900
"""§4.1 rule 3's value for an eventually-consistent provider. See the module docstring."""

QUOTA_RETRY_AFTER_SECONDS: Final = 900.0
"""Above this a 429 becomes a `QuotaError` rather than a held worker (§5.2)."""

MAX_SATURATED_MARKS: Final = 20
"""How many saturated-second markers the cursor carries. See `_advance_start`."""

_RATE_LIMIT_HEADERS: Final = frozenset(
    {"retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}
)
"""The only response headers that leave `fetch()`.

GDELT publishes none of them today, which is precisely why the filter is an
allowlist: the day it starts sending a `Set-Cookie`, that header must not ride
out on a batch and into a log line (`docs/connector-spec.md` §1).
"""

_TIME_FORMAT: Final = "%Y%m%d%H%M%S"
"""`startdatetime`/`enddatetime`, UTC, as the DOC API spells them."""


_FIELD_MAP: Final = FieldMap(
    platform=Platform.GDELT,
    # `seendate` is basic-format ISO ("20260728T140211Z"), which
    # `to_utc_datetime` parses through `datetime.fromisoformat` once the `Z` is
    # rewritten -- no bespoke parser, and therefore no second place for the
    # timestamp of a Signal to be interpreted differently.
    timestamp=FieldSpec.at("seendate", required=True),
    # No `item_id`: GDELT assigns none, so rule 2 runs. `url` is required so a
    # record without one becomes an attributable DLQ entry rather than falling to
    # rule 3, which cannot work here anyway -- rule 3 hashes the cleaned body and
    # this source has no body.
    url=FieldSpec.at("url", required=True),
    title=FieldSpec.at("title"),
    # No `text` path at all. GDELT returns metadata about an article, never the
    # article, and inventing a body from the title would put the headline in
    # `content.text` where the chunker and the language detector would treat it
    # as prose.
    metadata={
        "gdelt.domain": FieldSpec.at("domain"),
        # A language *name* ("English"), not a BCP-47 code, and not the
        # detector's opinion. `Signal.language` is filled by
        # `services/signal_engine/language.py` from the text it actually has;
        # copying a provider's label into it would fabricate a detector result
        # with no confidence behind it (`docs/signal-model.md` §3.3).
        "gdelt.language": FieldSpec.at("language"),
        "gdelt.source_country": FieldSpec.at("sourcecountry"),
        "gdelt.url_mobile": FieldSpec.at("url_mobile"),
    },
    truncated=True,
    # Asserts only the weaker claim the field makes -- "the connector could not
    # obtain the full body". The *stronger* fact, that this is title-only, is
    # carried by `content.text` being empty, which is what the 0.2 tier of
    # `content_integrity` keys off (`docs/signal-model.md` §3.5).
)


class GdeltConnector(BaseConnector):
    """GDELT DOC 2.0 `ArtList` search, walked forward in time."""

    slug: ClassVar[str] = "gdelt"
    platform: ClassVar[Platform] = Platform.GDELT
    category: ClassVar[SourceCategory] = SourceCategory.NEWS
    auth_type: ClassVar[AuthType] = AuthType.NONE
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=12, burst=1, concurrency=1
    )
    """One request every five seconds, one at a time.

    `docs/connector-spec.md` §9.5: the DOC 2.0 API has *undocumented* soft limits,
    so the only safe posture is serialization. `burst=1` and `concurrency=1` are
    the load-bearing parts -- an undocumented limiter answers a burst with a
    silent block, and there is no header to tell us it happened.
    """

    supports_incremental: ClassVar[bool] = True

    supports_backfill: ClassVar[bool] = True
    """History is reachable, but only through `params["lookback_hours"]`.

    `SyncContext` carries no `since`/`until`, so a historical crawl expresses its
    window as a large lookback. That gives it a different `params_hash` and
    therefore its own cursor row, which is exactly the separation §4.1 rule 5
    requires between a backfill and the live watermark.
    """

    overlap_seconds: ClassVar[int] = GDELT_OVERLAP_SECONDS

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._query = _as_str(params.get("query") or params.get("q"))
        self._records_per_request = _clamp(
            _as_int(params.get("max_records_per_request"), DEFAULT_RECORDS_PER_REQUEST),
            1,
            MAX_RECORDS_PER_REQUEST,
        )
        self._lookback_hours = max(1, _as_int(params.get("lookback_hours"), DEFAULT_LOOKBACK_HOURS))
        self._client: httpx.AsyncClient | None = None
        self._mapping = MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=ctx.run_id,
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Refuse to build unless the deployment enabled GDELT explicitly.

        `docs/connector-spec.md` §9.5 ships this connector "off by default", and
        `.env.example` carries `GDELT_ENABLED=false`. A connector may not read
        that setting itself -- `connectors/` cannot import `backend/core/config.py`
        (`docs/architecture.md` §6.2 rule 2) -- so the runtime passes it through as
        a param and the gate is enforced here, at construction, before any socket
        exists.

        A `ConnectorConfigurationError` rather than a silent no-op run: an
        operator who scheduled GDELT without enabling it has a misconfiguration,
        and a run that succeeds with zero records is how that goes unnoticed for a
        month.
        """
        if not _as_bool(ctx.params.get("enabled")):
            raise ConnectorConfigurationError(
                "gdelt is disabled; it is high-volume and ships off by default "
                "(docs/connector-spec.md §9.5). Set GDELT_ENABLED=true, which the "
                "runtime passes through as params['enabled']",
                connector=cls.slug,
                account_id=ctx.account_id,
            )
        connector = cls(ctx, credentials)
        if not connector._query:
            raise ConnectorConfigurationError(
                "gdelt needs params['query']; the DOC 2.0 API rejects an empty "
                "query, and an unbounded one would return the whole news firehose",
                connector=cls.slug,
                account_id=ctx.account_id,
            )
        return connector

    # ------------------------------------------------------------ lifecycle --

    async def authenticate(self) -> None:
        """Build the HTTP client. Idempotent, and there is nothing to authenticate.

        GDELT is open (`auth_type = NONE`), so this stage exists only to create
        the session. The `User-Agent` is not cosmetic: an unattributed client is
        the first thing an undocumented rate limiter throttles, and it is the only
        way the operators of a free service can reach us before blocking us.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.ctx.request_timeout_seconds,
                headers={"User-Agent": self.ctx.user_agent, "Accept": "application/json"},
                follow_redirects=False,
            )

    async def aclose(self) -> None:
        """Release the client. Idempotent: `run()` closes in a `finally`."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ---------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Walk the window forward in time, oldest-first.

        The DOC API has no pagination token: `sort=DateAsc` plus a moving
        `startdatetime` *is* the pager. That is also why oldest-first is honest
        here rather than reconstructed -- every record in a page is older than
        every record in the next, so each page's newest `seendate` is a legal
        watermark the moment the page is durable.

        `enddatetime` is frozen at the first request. A window whose upper bound
        chases `now` never closes, and on a source that produces continuously it
        would never hand the worker back.
        """
        floor = _watermark_floor(cursor, self.overlap_seconds)
        end = utcnow()
        start = cursor.watermark or end - timedelta(hours=self._lookback_hours)
        saturated = _saturated_marks(cursor)
        newest = floor
        fetched = 0
        pages = 0

        while start < end:
            budget = self._request_budget(fetched)
            if budget == 0:
                # A run given a zero record budget. Nothing was fetched, so there
                # is nothing to commit. Unlike the newest-first pager in
                # `news_api.py`, stopping early mid-window is safe here: ascending
                # order means the records this run did not reach are *newer* than
                # the watermark it committed, so the next run picks them up.
                return

            params = self._query_params(start, end, budget)
            await self.acquire_slot(self._base_url)
            body, headers = await self._get(params)

            articles = _articles(body)
            fingerprint = _fingerprint(DOC_PATH, params)
            records = [self._to_record(article, fingerprint) for article in articles]
            fetched += len(records)
            pages += 1
            newest = _max_moment(newest, *(_seen_at(article) for article in articles))

            next_start, saturated = _advance_start(
                start, articles, len(articles) >= budget, saturated
            )
            last = (
                not articles
                or len(articles) < budget
                # The pager did not move. `_advance_start` steps past a saturated
                # second, so the only way to reach this is a full page whose
                # `seendate`s are all unparseable -- every record of which is
                # heading for the DLQ anyway. Stopping repeats the window next run;
                # not stopping requests it forever inside this one.
                or next_start <= start
                or next_start >= end
                or self._budget_reached(fetched, pages)
            )
            yield self._page(records, cursor, newest, saturated, headers=headers)
            if last:
                return
            start = next_start

    def _page(
        self,
        records: Sequence[RawRecord],
        cursor: Cursor,
        newest: datetime | None,
        saturated: Sequence[str],
        *,
        headers: Mapping[str, str] | None = None,
    ) -> FetchPage:
        """One page plus the cursor that resumes after it.

        The watermark moves on *every* page, unlike a newest-first pager: ascending
        order means everything older than `newest` has already been yielded, so
        committing it here loses nothing if the process dies before the next page.
        """
        return FetchPage(
            records=records,
            cursor=cursor.advanced_to(
                watermark=newest,
                # No provider token exists; `watermark` is the entire resume state.
                page_token=None,
                saturated_seconds=list(saturated),
            ),
            raw_headers=dict(headers or {}),
        )

    def _query_params(self, start: datetime, end: datetime, max_records: int) -> dict[str, str]:
        return {
            "query": self._query,
            "mode": "ArtList",
            "format": "json",
            # The reason this connector can page at all: DateAsc makes the result
            # set a timeline, and the timeline is the cursor.
            "sort": "DateAsc",
            "maxrecords": str(max_records),
            "startdatetime": start.astimezone(UTC).strftime(_TIME_FORMAT),
            "enddatetime": end.astimezone(UTC).strftime(_TIME_FORMAT),
        }

    def _request_budget(self, fetched: int) -> int:
        """`maxrecords` for the next request, narrowed by `ctx.max_records`.

        Applied before the request rather than after the page, because this source
        answers in hundreds: `BaseConnector.run()` would stop the loop *after* a
        250-record page had already been fetched, normalized and hashed to serve a
        50-record budget.

        Counted on records fetched rather than emitted, since a connector cannot
        see what survived dedup; fetched is an upper bound on emitted, so the run
        stops at or before the ceiling and the cursor commits either way.
        """
        if self.ctx.max_records is None:
            return self._records_per_request
        return _clamp(self.ctx.max_records - fetched, 0, self._records_per_request)

    def _budget_reached(self, fetched: int, pages: int) -> bool:
        if self.ctx.max_pages is not None and pages >= self.ctx.max_pages:
            return True
        return self.ctx.max_records is not None and fetched >= self.ctx.max_records

    def _to_record(self, article: Mapping[str, Any], fingerprint: str) -> RawRecord:
        """Wrap one article verbatim, with the bytes that will be archived.

        Per-record provider bytes do not exist -- the response is one document per
        request -- so they are synthesized once, here, in a canonical encoding.
        `lineage.raw_sha256` is then taken over exactly the bytes the runtime PUTs
        to R2, which is what content-addressing requires.
        """
        return RawRecord(
            native_id=self._record_identity(article),
            payload=article,
            raw_bytes=json.dumps(
                article, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
            content_type="application/json",
            source_url=_as_str(article.get("url")) or None,
            request_fingerprint=fingerprint,
        )

    def _record_identity(self, article: Mapping[str, Any]) -> str:
        """Rule 2, through the same function `normalize()` uses.

        Sharing `derive_native_id` rather than re-deriving "sha256 of the canonical
        URL" locally is what makes it impossible for the id on a DLQ record to
        disagree with the id of the Signal it failed to become. A payload with no
        usable URL is filed under a digest of itself; it never becomes a Signal,
        because `url` is required in the field map.
        """
        try:
            return derive_native_id(platform=self.platform, url=_as_str(article.get("url")))
        except NormalizationError:
            material = json.dumps(article, sort_keys=True, default=str).encode("utf-8")
            return f"unidentified:{hashlib.sha256(material).hexdigest()}"

    async def _get(self, params: Mapping[str, str]) -> tuple[Mapping[str, Any], dict[str, str]]:
        """One request. Raises; never retries, never sleeps."""
        client = self._client
        if client is None:  # pragma: no cover -- run() always authenticates first
            raise PermanentError(
                "gdelt fetch ran before authenticate(); there is no HTTP client",
                connector=self.slug,
            )

        try:
            response = await client.get(DOC_PATH, params=params)
        except httpx.TransportError as exc:
            raise TransientError(
                f"gdelt request failed: {type(exc).__name__}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                f"gdelt request could not be issued: {type(exc).__name__}",
                connector=self.slug,
                cause=exc,
            ) from exc

        self._raise_for_status(response)
        return self._parse(response), _rate_limit_headers(response.headers)

    def _raise_for_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        if status == 429:
            hint = self.parse_rate_limit(response.headers)
            wait = hint.retry_after_seconds if hint is not None else None
            if wait is not None and wait > QUOTA_RETRY_AFTER_SECONDS:
                raise QuotaError(
                    "gdelt asked for a long wait",
                    connector=self.slug,
                    status_code=status,
                    retry_after_seconds=wait,
                    reset_at=hint.reset_at if hint is not None else None,
                )
            raise TransientError(
                "gdelt throttled the request",
                connector=self.slug,
                status_code=status,
            )
        if status >= 500:
            raise TransientError(
                f"gdelt returned {status}", connector=self.slug, status_code=status
            )
        raise PermanentError(
            f"gdelt returned {status}; the query is wrong and will be wrong again",
            connector=self.slug,
            status_code=status,
        )

    def _parse(self, response: httpx.Response) -> Mapping[str, Any]:
        """Decode the body, tolerating the two non-JSON shapes GDELT sends.

        The DOC API answers `200` with an *empty* body when a window holds nothing
        -- that is zero records, not a failure -- and `200` with a plain-text
        sentence when a query is malformed. The second is a `PermanentError`
        because retrying a bad query is a bad query. Neither branch quotes the
        body: `docs/connector-spec.md` §1 forbids logging it, so the size and the
        content type are what the error carries instead.
        """
        raw = response.content
        if not raw.strip():
            return {"articles": []}
        try:
            body = response.json()
        except ValueError as exc:
            raise PermanentError(
                f"gdelt returned {len(raw)} bytes of non-JSON "
                f"({response.headers.get('content-type', 'unknown')}); the DOC 2.0 API "
                "reports query syntax errors as plain text with status 200",
                connector=self.slug,
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(body, Mapping):
            raise PermanentError(
                f"gdelt returned a JSON {type(body).__name__} where an object was "
                "expected; the response shape changed",
                connector=self.slug,
                status_code=response.status_code,
            )
        return body

    # ------------------------------------------------------------ normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one indexed article onto a title-only Signal."""
        payload = record.payload
        if not _as_str(payload.get("title")):
            # A title-only source with no title carries no observation at all.
            # Dropping is right rather than a DLQ record: the payload is
            # well-formed, it simply says nothing, and emitting it would put an
            # empty document into the embedding queue and the search index.
            return None

        signal = _FIELD_MAP.to_signal(record, self._mapping)
        image = _as_str(payload.get("socialimage"))
        if image:
            # Assigned after mapping because `MediaMap` addresses a list and GDELT
            # carries a single scalar. It is the article's social card image --
            # often the only visual evidence attached to a headline.
            signal.media = [MediaRef(kind=MediaKind.IMAGE, source_url=image)]
        return signal


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _advance_start(
    start: datetime,
    articles: Sequence[Mapping[str, Any]],
    was_full: bool,
    saturated: Sequence[str],
) -> tuple[datetime, list[str]]:
    """Where the next request starts, and what that cost.

    Normally the newest `seendate` in the page: re-reading the boundary second is
    cheap because dedup collapses it, whereas skipping past it would drop every
    article that shares that second with the last one returned.

    The exception is a *saturated second*. GDELT stamps whole 15-minute batches
    with the same `seendate`, so one second can hold more articles than
    `maxrecords` -- and then the newest `seendate` in a full page equals the start
    of the window, and the pager cannot move. Stepping one second past it is the
    only way to make progress, and it loses whatever GDELT did not return. That
    loss is *recorded* in the cursor rather than swallowed: a silent gap in a news
    index is indistinguishable from a quiet news hour.
    """
    newest = _max_moment(*(_seen_at(article) for article in articles))
    if newest is None:
        return start, list(saturated)
    if newest > start:
        return newest, list(saturated)
    if not was_full:
        # No progress and no pressure: the window simply ends here.
        return start, list(saturated)
    marks = [*saturated, start.astimezone(UTC).isoformat()][-MAX_SATURATED_MARKS:]
    return start + timedelta(seconds=1), marks


def _saturated_marks(cursor: Cursor) -> list[str]:
    marks = cursor.checkpoint.get("saturated_seconds")
    if not isinstance(marks, Sequence) or isinstance(marks, (str, bytes)):
        return []
    return [str(mark) for mark in marks][-MAX_SATURATED_MARKS:]


def _watermark_floor(cursor: Cursor, overlap_seconds: int) -> datetime | None:
    """Reconstruct the watermark the runtime is holding.

    `BaseConnector._effective_start` hands `fetch()` a cursor already shifted back
    by `overlap_seconds`, and every cursor derived from it inherits that shifted
    value. This matters far more here than elsewhere: with a 900-second overlap, a
    resumed run that finds nothing new -- the normal case on a quiet query -- would
    otherwise commit a watermark fifteen minutes *older* than the stored one, and
    `docs/connector-spec.md` §4.1 rule 2 says the runtime rejects a watermark that
    moves backwards. The shift is exactly `watermark - overlap`, so undoing it is
    exact too.

    Sound because `run()` is the only caller of `fetch()`: the floor can never
    exceed a watermark the runtime has already committed.
    """
    if cursor.watermark is None:
        return None
    return cursor.watermark + timedelta(seconds=overlap_seconds)


def _articles(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The `articles` array.

    Absent means zero results here, unlike the commercial API: GDELT omits the key
    entirely for an empty window rather than sending `"articles": []`. A present
    but non-list value is a shape change and raises.
    """
    articles = body.get("articles")
    if articles is None:
        return []
    if not isinstance(articles, Sequence) or isinstance(articles, (str, bytes)):
        raise PermanentError("gdelt response has a non-list 'articles'; the shape changed")
    return [item for item in articles if isinstance(item, Mapping)]


def _seen_at(article: Mapping[str, Any]) -> datetime | None:
    """Parse `seendate`, or `None`.

    Never raises. An article with an unparseable date still has to reach
    `normalize()`, which is the stage allowed to DLQ it with its identity
    attached; raising here would abort the page and block every well-formed
    article behind it.
    """
    value = article.get("seendate")
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _max_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return max(present) if present else None


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in _RATE_LIMIT_HEADERS}


def _fingerprint(endpoint: str, params: Mapping[str, str]) -> str:
    """Hash of endpoint plus normalized params; there is no credential to omit."""
    material = endpoint + "?" + urlencode(sorted(params.items()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _as_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    """Read an enablement flag that may have come from an env var.

    `GDELT_ENABLED` is a string long before it is a boolean, and `bool("false")`
    is `True` -- which would turn the off-by-default gate into an on-by-default
    one for every deployment that passed the setting through verbatim.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
