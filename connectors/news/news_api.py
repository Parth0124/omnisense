"""Commercial news API connector, newsapi.org shape (Phase 1).

The one Phase 1 source that bills per request and answers with an *excerpt*
rather than an article. Both facts drive the design of this module.

**Identity is rule 2** (`docs/signal-model.md` §4.1): `native_id =
sha256(canonicalize_url(article["url"]))`. This provider ships no stable item id
-- the `source.id` field names the *outlet*, not the article, and nothing else in
the payload survives a re-poll unchanged. So `_field_map()` deliberately declares
no `item_id`, which is what makes `derive_native_id` fall to rule 2, and `url` is
`required` so that a payload with no usable URL becomes an attributable DLQ
record instead of quietly sliding down to rule 3 and deriving identity from a
truncated body. Two spellings of one article URL -- one carrying `utm_*`, one not
-- therefore produce one Signal, which is the whole point of canonicalizing
before hashing.

**Truncation is a per-record fact, not a per-connector one.** The provider
returns `content` capped at ~200 characters with a `[+2317 chars]` marker glued
on the end, but it returns *some* articles whole, and for others it returns no
`content` at all and only a `description`. `Content.truncated` caps the
`content_integrity` component of confidence (`docs/signal-model.md` §3.5), so
declaring it statically would either understate every full article or overstate
every excerpt. Hence two field maps, chosen per record by `_truncation()`.

**Pagination is newest-first and the provider offers no ascending sort.**
`BaseConnector.fetch` requires oldest-first because "a newest-first pager that
dies mid-run would commit a watermark past records it never emitted". That
failure mode is what the rule protects against, so this connector defeats it
directly instead: the watermark is *pinned* at its incoming value for every page
of a window and only advances on the page that exhausts the window. A run killed
at page 3 of 7 therefore commits the watermark it started with plus a page token,
and the records it did emit are re-fetched next run and collapsed by dedup --
duplicated, never lost.

No retry, no sleep, no writes (`docs/connector-spec.md` §1). Transport failures
and 5xx raise `TransientError` and stop; the runtime owns backoff.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Self
from urllib.parse import urlencode

import httpx

from models.base import utcnow
from models.enums import AuthType, MediaKind, Platform, SourceCategory
from models.signal import MediaRef, Signal
from connectors.auth.apikey import ApiKeyAuth
from connectors.base import BaseConnector
from connectors.exceptions import (
    AuthError,
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

__all__ = ["NewsApiConnector"]

DEFAULT_BASE_URL: Final = "https://newsapi.org/v2"
"""Overridable through `params["base_url"]` for API-compatible mirrors.

Not for tests -- those run under `respx`, which intercepts the real host -- but
for the self-hosted and white-label deployments that speak the same JSON.
"""

EVERYTHING_PATH: Final = "/everything"
MAX_PAGE_SIZE: Final = 100
DEFAULT_PAGE_SIZE: Final = 100

DEFAULT_LOOKBACK_HOURS: Final = 24
"""How far back a first run reaches when there is no watermark.

Deliberately short. The archive is plan-limited and every page costs quota, so a
cold start that tried to sweep a month would spend the day's budget before the
first scheduled poll.
"""

QUOTA_RETRY_AFTER_SECONDS: Final = 900.0
"""`Retry-After` above this becomes a `QuotaError` (`docs/connector-spec.md` §5.2).

Fifteen minutes of held worker to serve one account is worse than checkpointing
and coming back: `QuotaError` is a partial success, so the emitted records stay
emitted and the run is rescheduled at the reset.
"""

_TRUNCATION_MARKER: Final = re.compile(r"\s*\[\+(\d+)\s+chars\]\s*$")
"""The provider's own truncation signal, e.g. `"... [+2317 chars]"`."""

_REMOVED_SENTINEL: Final = "[removed]"
"""What the provider substitutes for an article its licensor withdrew.

Title, description, content and the outlet name all become `"[Removed]"` and the
URL becomes `https://removed.com`. It is a tombstone, not a malformed payload, so
it is *dropped* rather than sent to the DLQ (`connectors/base.py`: dropping is
expected and counted separately from failing to map).
"""

_RATE_LIMIT_HEADERS: Final = frozenset(
    {"retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}
)
"""The only response headers that leave `fetch()`.

`FetchPage.raw_headers` is fed to `parse_rate_limit` and travels with the batch.
A whole header map carries cookies and echoed authorization, and
`docs/connector-spec.md` §1 forbids logging either, so the filter is here rather
than at whatever log line eventually renders a batch.
"""

_AUTH_CODES: Final = frozenset({"apikeyinvalid", "apikeymissing", "apikeydisabled"})
_QUOTA_CODES: Final = frozenset({"ratelimited", "apikeyexhausted"})
_CEILING_CODE: Final = "maximumresultsreached"


class _ResultCeilingError(Exception):
    """The plan's total-results ceiling. A control-flow signal, not a failure.

    Free and developer plans cap how deep a query may page and answer `426` (or a
    `200` carrying `maximumResultsReached`) beyond it. Treating that as an error
    would fail every incremental run whose window grew past the cap, which is all
    of them eventually. Pagination stops and the window closes normally.
    """


# --------------------------------------------------------------------------- #
# Field maps
# --------------------------------------------------------------------------- #


def _strip_truncation_marker(value: Any) -> Any:
    """Remove `[+N chars]` from the body before it becomes `Content.text`.

    The marker is provider bookkeeping, not prose. Leaving it in would put it in
    the embedded text and in the layer-2 content hash, so the same excerpt
    syndicated through a second source -- or the same article re-fetched after the
    remaining-character count changed -- would look like new content.
    """
    if not isinstance(value, str):
        return value
    return _TRUNCATION_MARKER.sub("", value)


def _field_map(*, truncated: bool) -> FieldMap:
    """The payload shape, in the two truncation states the provider returns.

    Identical apart from `truncated`, and built by one function so they cannot
    drift: a path added to one map and forgotten in the other would produce
    Signals whose field set depended on whether the article happened to be
    excerpted.
    """
    return FieldMap(
        platform=Platform.NEWS_API,
        timestamp=FieldSpec.at("publishedAt", required=True),
        # No `item_id`: see the module docstring. Its absence is what selects
        # rule 2, and `url` is required so the ladder can never fall through to
        # rule 3 on a record that is going to be emitted.
        url=FieldSpec.at("url", required=True),
        title=FieldSpec.at("title"),
        text=FieldSpec.at("content", "description", transform=_strip_truncation_marker),
        # Plain text in the documented shape, but publishers inject `<p>` and
        # `<a>` into their own descriptions. `extract_readable` no-ops on text
        # that does not look like markup (`looks_like_html`), so the cost of
        # declaring this is nil and the failure it avoids -- raw tags in the body,
        # and therefore in the embedding -- is permanent.
        text_is_html=True,
        metadata={
            # The byline is deliberately *not* mapped to `Author`. It is a display
            # string with no identifier behind it, and much of the corpus says
            # "Reuters" or "Staff"; keying an author's history on it would merge
            # every staff writer in the world into one node
            # (`docs/signal-model.md` §3.1).
            "news_api.byline": FieldSpec.at("author"),
            "news_api.source_id": FieldSpec.at("source.id"),
            "news_api.source_name": FieldSpec.at("source.name"),
        },
        truncated=truncated,
    )


_FULL_MAP: Final = _field_map(truncated=False)
_EXCERPT_MAP: Final = _field_map(truncated=True)


def _truncation(payload: Mapping[str, Any]) -> tuple[bool, int | None]:
    """Whether this article is an excerpt, and how much of it is missing.

    Three states, because the provider has three behaviours: a full `content`
    (not truncated), a marked excerpt (truncated, count known), and no `content`
    at all, where the body falls back to `description` -- an excerpt by
    construction, with no count to report.
    """
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        match = _TRUNCATION_MARKER.search(content)
        return (True, int(match.group(1))) if match else (False, None)
    return True, None


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #


class NewsApiConnector(BaseConnector):
    """`newsapi.org`-shaped `/v2/everything` search, one query per account."""

    slug: ClassVar[str] = "news_api"
    platform: ClassVar[Platform] = Platform.NEWS_API
    category: ClassVar[SourceCategory] = SourceCategory.NEWS
    auth_type: ClassVar[AuthType] = AuthType.API_KEY
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=20, burst=5, concurrency=1
    )
    """A per-minute bucket cannot express this provider's real limit.

    The binding constraint is a *daily* request cap (100 on the free tier), which
    no bucket sized in requests-per-minute can enforce. The policy here only stops
    a single run from bursting; the actual budget is the poll cadence in
    `workers/scheduler.py`, and that is where a quota overrun gets fixed.
    """

    supports_incremental: ClassVar[bool] = True

    supports_backfill: ClassVar[bool] = False
    """The archive is plan-limited (a trailing month on the free tier) and paging
    stops at the plan's total-results ceiling, so a historical crawl would spend
    quota to discover it cannot reach the history it was asked for."""

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._query = _as_str(params.get("query") or params.get("q"))
        self._sources = _as_csv(params.get("sources"))
        self._domains = _as_csv(params.get("domains"))
        self._language = _as_str(params.get("language"))
        self._page_size = _clamp(
            _as_int(params.get("page_size"), DEFAULT_PAGE_SIZE), 1, MAX_PAGE_SIZE
        )
        self._lookback_hours = max(1, _as_int(params.get("lookback_hours"), DEFAULT_LOOKBACK_HOURS))
        self._auth: ApiKeyAuth | None = None
        self._client: httpx.AsyncClient | None = None
        self._mapping = MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=ctx.run_id,
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Build and validate the query. No I/O, no credential access.

        The query is checked here rather than on the first request because
        `/v2/everything` rejects a search with none of `q`, `sources` or `domains`
        with a `400` -- an authenticated `400`, which spends a request against a
        daily cap to learn something the configuration already knew.
        """
        connector = cls(ctx, credentials)
        if not (connector._query or connector._sources or connector._domains):
            raise ConnectorConfigurationError(
                "news_api needs at least one of params['query'], params['sources'] "
                "or params['domains']; /v2/everything refuses an unbounded search",
                connector=cls.slug,
                account_id=ctx.account_id,
            )
        return connector

    # ------------------------------------------------------------ lifecycle --

    async def authenticate(self) -> None:
        """Prove the key is *present* and build the client. Idempotent, no I/O.

        There is no session endpoint -- an API key is a constant -- so there is
        nothing to call here. Proving the key *valid* would spend a request
        against the daily cap to learn what the first real fetch learns anyway,
        and it would do so on every run.

        A missing secret is an `AuthError` rather than a configuration error: it
        is a credential row an operator has to fix, and `AuthError` is what flags
        the account `needs_reauth` (`docs/connector-spec.md` §2.1).
        """
        if self._auth is None:
            try:
                self._auth = ApiKeyAuth.from_credentials(
                    self.credentials, secret_key="api_key", header="X-Api-Key"
                )
            except KeyError as exc:
                raise AuthError(
                    "news_api account has no 'api_key' secret",
                    connector=self.slug,
                    account_id=self.ctx.account_id,
                    cause=exc,
                ) from exc
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.ctx.request_timeout_seconds,
                headers={
                    "User-Agent": self.ctx.user_agent,
                    "Accept": "application/json",
                    **self._auth.headers(),
                },
                # A redirect off an API endpoint is a captive portal or a login
                # page, never a moved resource. Following it would send the key to
                # whatever answered and parse the result as an article list.
                follow_redirects=False,
            )

    async def aclose(self) -> None:
        """Release the client. Idempotent: `run()` closes in a `finally`."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ---------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Page one time window, pinning the watermark until it is exhausted.

        `to` is frozen at the first request rather than recomputed per page: a
        window whose upper bound chases `now` never closes, and its page numbering
        shifts under the pager as new articles arrive.
        """
        floor = _watermark_floor(cursor, self.overlap_seconds)
        window_start, window_end, page_number = self._resume(cursor)
        newest = floor
        fetched = 0
        pages = 0

        while True:
            page_size = self._page_budget(fetched)
            if page_size == 0:
                # A run given a zero record budget. Nothing was fetched, so there
                # is nothing to commit -- and yielding a cursor here would close a
                # window that was never opened.
                return

            params = self._query_params(window_start, window_end, page_number, page_size)
            await self.acquire_slot(self._base_url)
            try:
                body, headers = await self._get(params)
            except _ResultCeilingError:
                # The plan will not page deeper. The window is as done as it can
                # be, so close it: leaving the watermark where it was would make
                # every later run re-fetch these same first pages forever.
                yield self._closing_page(cursor, newest)
                return

            articles = _articles(body)
            fingerprint = _fingerprint(EVERYTHING_PATH, params)
            records = [self._to_record(a, fingerprint) for a in _sorted_by_published(articles)]
            fetched += len(records)
            pages += 1
            newest = _max_moment(newest, *(_published_at(a) for a in articles))

            # Two different reasons to stop, and they need two different cursors.
            # Running out of *window* releases the watermark; running out of
            # *budget* must not, because the provider pages newest-first and the
            # unfetched pages of this window are older than everything emitted so
            # far. Closing the window here would skip them permanently.
            window_done = (
                not records
                or len(articles) < page_size
                or _window_exhausted(page_number, page_size, body.get("totalResults"))
            )
            yield FetchPage(
                records=records,
                cursor=(
                    self._closing_cursor(cursor, newest)
                    if window_done
                    else cursor.advanced_to(
                        # Pinned at the watermark this run started from, so a crash
                        # -- or a budget stop -- between here and the last page of
                        # the window cannot commit a watermark past records nobody
                        # emitted.
                        watermark=floor,
                        page_token=str(page_number + 1),
                        window={
                            "start": _isoformat(window_start),
                            "end": _isoformat(window_end),
                        },
                    )
                ),
                raw_headers=headers,
            )
            if window_done or self._budget_reached(fetched, pages):
                return
            page_number += 1

    def _resume(self, cursor: Cursor) -> tuple[datetime, datetime, int]:
        """Decide the window and the page to start on.

        A `page_token` is only meaningful against the window it was issued for --
        page 4 of last hour's search addresses different articles than page 4 of
        this one -- so the window travels in the checkpoint beside it. When either
        is unreadable the run re-pages from the top rather than failing:
        `docs/connector-spec.md` §4.1 rule 4 makes the token advisory.
        """
        page_number = _as_int(cursor.page_token, 1)
        window = cursor.checkpoint.get("window")
        if page_number > 1 and isinstance(window, Mapping):
            start = _parse_isoformat(window.get("start"))
            end = _parse_isoformat(window.get("end"))
            if start is not None and end is not None:
                return start, end, page_number

        now = utcnow()
        start = cursor.watermark or now - timedelta(hours=self._lookback_hours)
        return start, now, 1

    def _query_params(
        self, start: datetime, end: datetime, page: int, page_size: int
    ) -> dict[str, str]:
        """Build the query. The credential is *not* here -- it is a header.

        `connectors/auth/apikey.py` has no query-parameter strategy on purpose:
        URLs are logged by every proxy in the path and land in `Referer` on any
        redirect. This provider accepts the header form, so this stays a
        request-shaping function with nothing secret in it.
        """
        params: dict[str, str] = {
            "from": _provider_time(start),
            "to": _provider_time(end),
            # The only sort that is a stable order to page through. `relevancy`
            # and `popularity` re-rank between requests, so page 2 of one run and
            # page 2 of the next overlap in ways no cursor can describe.
            "sortBy": "publishedAt",
            "page": str(page),
            "pageSize": str(page_size),
        }
        if self._query:
            params["q"] = self._query
        if self._sources:
            params["sources"] = self._sources
        if self._domains:
            params["domains"] = self._domains
        if self._language:
            params["language"] = self._language
        return params

    def _page_budget(self, fetched: int) -> int:
        """Page size, narrowed to what is left of `ctx.max_records`.

        Counted on records *fetched* rather than emitted because a connector
        cannot see how many survived dedup. Fetched is an upper bound on emitted,
        so this stops at or before the ceiling `BaseConnector.run()` enforces --
        and it stops before spending quota on a page whose tail would be
        discarded.
        """
        if self.ctx.max_records is None:
            return self._page_size
        return _clamp(self.ctx.max_records - fetched, 0, self._page_size)

    def _budget_reached(self, fetched: int, pages: int) -> bool:
        if self.ctx.max_pages is not None and pages >= self.ctx.max_pages:
            return True
        return self.ctx.max_records is not None and fetched >= self.ctx.max_records

    def _closing_page(self, cursor: Cursor, newest: datetime | None) -> FetchPage:
        """An empty page whose only job is to carry the closing cursor.

        Emitted when pagination ends without records -- the plan's total-results
        ceiling. Without it the window's watermark is never committed and every
        later run repeats this one exactly.
        """
        return FetchPage(records=[], cursor=self._closing_cursor(cursor, newest))

    def _closing_cursor(self, cursor: Cursor, newest: datetime | None) -> Cursor:
        """The window is exhausted: release the watermark and drop the token.

        `window=None` blanks the resume state by merge (`Cursor.advanced_to` unions
        checkpoints, so a key can be emptied but not removed), which is what makes
        the next run compute a fresh window instead of re-paging this one.
        """
        return cursor.advanced_to(watermark=newest, page_token=None, window=None)

    def _to_record(self, article: Mapping[str, Any], fingerprint: str) -> RawRecord:
        """Wrap one article verbatim, with the bytes that will be archived.

        The provider returns one document per *page*, so per-record bytes do not
        exist to be captured. They are synthesized once, here, in a canonical
        encoding, and `lineage.raw_sha256` is taken over exactly the bytes the
        runtime will PUT to R2 -- which is the property content-addressing needs.
        Re-serializing later, in another process on another library version, is
        what `RawRecord.raw_bytes` exists to prevent.
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

        Calling `derive_native_id` rather than re-implementing "sha256 of the
        canonical URL" here is what makes it impossible for the id on a DLQ record
        to disagree with the id on the Signal it failed to become.

        A payload with no usable URL still needs *something* to be filed under, so
        it gets a digest of its own fields. It never becomes a Signal -- `url` is
        required in the field map -- so this string only appears on a DLQ record,
        where its job is to make two arrivals of the same broken payload
        recognisable as one.
        """
        try:
            return derive_native_id(platform=self.platform, url=_as_str(article.get("url")))
        except NormalizationError:
            material = json.dumps(article, sort_keys=True, default=str).encode("utf-8")
            return f"unidentified:{hashlib.sha256(material).hexdigest()}"

    async def _get(self, params: Mapping[str, str]) -> tuple[Mapping[str, Any], dict[str, str]]:
        """One request. Raises; never retries, never sleeps.

        Status is classified before the body is touched, because the body of a
        `401` is the least trustworthy JSON this provider produces.
        """
        client = self._client
        if client is None:  # pragma: no cover -- run() always authenticates first
            raise PermanentError(
                "news_api fetch ran before authenticate(); there is no HTTP client",
                connector=self.slug,
            )

        try:
            response = await client.get(EVERYTHING_PATH, params=params)
        except httpx.TransportError as exc:
            # Timeouts, resets, DNS. Retryable by the runtime, never here.
            raise TransientError(
                f"news_api request failed: {type(exc).__name__}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                f"news_api request could not be issued: {type(exc).__name__}",
                connector=self.slug,
                cause=exc,
            ) from exc

        self._raise_for_status(response)
        try:
            body = response.json()
        except ValueError as exc:
            # A 200 that is not JSON is an intermediary -- a proxy notice, a
            # captive portal -- not this provider. Those recover, so it is
            # transient. The body is measured, never quoted: §1 forbids logging it.
            raise TransientError(
                f"news_api returned {len(response.content)} bytes of non-JSON with "
                f"status {response.status_code}",
                connector=self.slug,
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(body, Mapping):
            raise PermanentError(
                f"news_api returned a JSON {type(body).__name__} where an object was "
                "expected; the response shape changed",
                connector=self.slug,
                status_code=response.status_code,
            )
        if str(body.get("status", "")).lower() == "error":
            raise self._error_from_code(str(body.get("code") or ""), response.status_code)
        return body, _rate_limit_headers(response.headers)

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status onto the four families of `connectors/exceptions.py`."""
        status = response.status_code
        if status < 400:
            return
        if status in (401, 403):
            raise AuthError(
                f"news_api rejected the API key with {status}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status == 426:
            # The plan's total-results ceiling. Not an error; see the exception.
            raise _ResultCeilingError
        if status == 429:
            raise self._throttled(response)
        if status >= 500:
            raise TransientError(
                f"news_api returned {status}",
                connector=self.slug,
                status_code=status,
            )
        raise PermanentError(
            f"news_api returned {status}; the request is wrong and will be wrong again",
            connector=self.slug,
            status_code=status,
        )

    def _throttled(self, response: httpx.Response) -> QuotaError | TransientError:
        """A 429 is transient inside the cap and a quota beyond it (§5.2).

        The split matters operationally: a `TransientError` holds the worker
        through a backoff, while a `QuotaError` commits the cursor and hands the
        worker back. Fifteen minutes is where the second becomes cheaper.
        """
        hint = self.parse_rate_limit(response.headers)
        wait = hint.retry_after_seconds if hint else None
        if wait is not None and wait > QUOTA_RETRY_AFTER_SECONDS:
            return QuotaError(
                "news_api quota exhausted",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=429,
                retry_after_seconds=wait,
                reset_at=hint.reset_at if hint is not None else None,
            )
        return TransientError(
            "news_api throttled the request",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=429,
        )

    def _error_from_code(self, code: str, status: int) -> Exception:
        """Classify a `200 {"status": "error"}` body.

        The provider is mostly honest with HTTP status, but not always -- an
        exhausted key has been observed arriving as a `200`. Classifying by the
        machine-readable `code`, never the human `message` (which echoes the
        request), keeps those on the same paths as their HTTP equivalents.
        """
        normalized = code.strip().lower()
        if normalized == _CEILING_CODE:
            return _ResultCeilingError()
        if normalized in _AUTH_CODES:
            return AuthError(
                f"news_api rejected the API key: {normalized}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details={"provider_code": normalized},
            )
        if normalized in _QUOTA_CODES:
            return QuotaError(
                f"news_api quota exhausted: {normalized}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details={"provider_code": normalized},
            )
        if normalized == "unexpectederror":
            return TransientError(
                "news_api reported an internal error",
                connector=self.slug,
                status_code=status,
                details={"provider_code": normalized},
            )
        return PermanentError(
            f"news_api rejected the query: {normalized or 'unknown code'}",
            connector=self.slug,
            status_code=status,
            details={"provider_code": normalized},
        )

    # ------------------------------------------------------------ normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one article, choosing the field map by its truncation state."""
        payload = record.payload
        if _is_removed(payload):
            # A tombstone, not a defect: the licensor withdrew the article and the
            # provider left a placeholder behind. Dropping is the sanctioned
            # outcome and is counted separately from the DLQ.
            return None

        truncated, remaining = _truncation(payload)
        field_map = _EXCERPT_MAP if truncated else _FULL_MAP
        signal = field_map.to_signal(
            record,
            self._mapping,
            extra_metadata=(
                {"news_api.truncated_chars": remaining} if remaining is not None else None
            ),
        )

        image = _as_str(payload.get("urlToImage"))
        if image:
            # Assigned after mapping because `MediaMap` addresses a *list* and this
            # provider carries a single scalar. Dropping it instead would lose the
            # only media this source has, and `metadata` is the wrong home for a
            # reference `MediaRef` exists to hold.
            signal.media = [MediaRef(kind=MediaKind.IMAGE, source_url=image)]
        return signal


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _watermark_floor(cursor: Cursor, overlap_seconds: int) -> datetime | None:
    """Reconstruct the watermark the runtime is holding.

    `BaseConnector._effective_start` hands `fetch()` a cursor already shifted back
    by `overlap_seconds`, and every cursor derived from it inherits that shifted
    value. A page inside the overlap window whose newest article predates the
    stored watermark would therefore commit a watermark *older* than the one the
    runtime holds -- and `docs/connector-spec.md` §4.1 rule 2 says the runtime
    rejects a watermark that moves backwards. The shift is exactly
    `watermark - overlap`, so undoing it is exact too.

    Sound because `run()` is the only caller of `fetch()`: the floor can never
    exceed a watermark the runtime has already committed.
    """
    if cursor.watermark is None:
        return None
    return cursor.watermark + timedelta(seconds=overlap_seconds)


def _is_removed(payload: Mapping[str, Any]) -> bool:
    title = _as_str(payload.get("title")).lower()
    url = _as_str(payload.get("url")).lower()
    return title == _REMOVED_SENTINEL or url.startswith("https://removed.com")


def _articles(body: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """The `articles` array, or a `PermanentError` if it is not one.

    A missing key is a shape change, not a slow news day: an empty result set is
    `"articles": []`. Defaulting to `[]` here would turn a breaking provider
    change into a run that succeeds forever with zero records.
    """
    articles = body.get("articles")
    if not isinstance(articles, Sequence) or isinstance(articles, (str, bytes)):
        raise PermanentError(
            "news_api response has no 'articles' array; the response shape changed"
        )
    return [item for item in articles if isinstance(item, Mapping)]


def _sorted_by_published(articles: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Oldest-first *within* the page.

    Across pages the provider offers no ascending order at all (see the module
    docstring), but within one page it is free, and it makes the emitted batch
    ordered for anything downstream that reads a page as a timeline.
    """
    return sorted(articles, key=lambda a: _published_at(a) or datetime.min.replace(tzinfo=UTC))


def _published_at(article: Mapping[str, Any]) -> datetime | None:
    """Parse `publishedAt`, or `None` for anything unparseable.

    Never raises: an article with a broken date still has to reach `normalize()`,
    which is the stage allowed to send it to the DLQ with its identity attached.
    Raising here would abort the whole page instead.
    """
    return _parse_isoformat(article.get("publishedAt"))


def _window_exhausted(page: int, page_size: int, total: Any) -> bool:
    """Whether the pager has reached the end of the result set.

    A `totalResults` that is missing or not a number means **stop**, not "keep
    going": the documented shape always carries it, so a response without one came
    from something other than this API, and a pager that ignored its absence would
    request page 2, 3, 4 ... forever against a proxy returning a constant body.
    Stopping costs at most one under-read window, which the next run repeats.
    """
    if not isinstance(total, int) or isinstance(total, bool):
        return True
    return page * page_size >= total


def _max_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return max(present) if present else None


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in _RATE_LIMIT_HEADERS}


def _fingerprint(endpoint: str, params: Mapping[str, str]) -> str:
    """Hash of endpoint plus normalized params. Credentials are in a header.

    `Lineage.request_fingerprint` is what makes a fetch reproducible. It is
    truncated because it identifies a request, and a full-length digest invites
    someone to mistake it for a content hash.
    """
    material = endpoint + "?" + urlencode(sorted(params.items()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _provider_time(moment: datetime) -> str:
    """`from`/`to` in the shape the provider documents: ISO-8601, UTC, seconds."""
    return moment.astimezone(UTC).replace(tzinfo=None).isoformat(timespec="seconds")


def _isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _parse_isoformat(value: Any) -> datetime | None:
    """Parse an ISO-8601 instant to an aware UTC datetime, or `None`.

    Naive input is refused rather than assumed to be UTC. Everything reaching
    this is either the provider's own `publishedAt` (documented as UTC and always
    suffixed) or a checkpoint this module wrote, so a naive value means something
    unexpected produced it -- and guessing would shift a watermark by hours.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo is not None else None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _as_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_csv(value: Any) -> str:
    """Comma-join a list param, or pass a string through.

    Both spellings appear in configuration -- a YAML list and a copied comma
    string -- and the provider only accepts the second.
    """
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence):
        return ",".join(_as_str(item) for item in value if _as_str(item))
    return ""


def _as_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
