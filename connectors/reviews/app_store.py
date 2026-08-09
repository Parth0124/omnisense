"""Apple App Store customer-reviews feed connector (Phase 2).

`docs/connector-spec.md` §9.2 describes this source in one line -- "public review
RSS is documented but legacy" -- and every decision below follows from that
sentence. The feed at `itunes.apple.com/{country}/rss/customerreviews/...` is
genuinely open: no key, no registration, no terms gate. It is also the last
survivor of an RSS generation Apple has otherwise retired, it publishes no
schema, no quota and no rate-limit headers, and it answers a JSON request with
`Content-Type: text/javascript`. Nothing here may assume the shape is stable.

**Newest-first is the sanctioned exception, and it is taken deliberately.**
`sortby=mostrecent` is the only ordering the feed offers, so page 1 holds the
newest reviews and page 10 the oldest. `BaseConnector.fetch` permits exactly this
-- "a connector whose provider only pages newest-first may yield in provider
order *provided* it pins the watermark for the whole window and only advances it
on the closing page" -- and that is what `fetch` does: every non-final page
carries the watermark the run started from plus a `page_token`, and only the page
that ends the descent carries the newest timestamp seen.

`connectors/social/reddit.py` faces the same provider order and solves it the
other way, by buffering the whole descent and yielding it reversed. That is the
better answer *there* and the wrong one here. Reddit's listing is a live API;
this is a cached legacy endpoint that intermittently answers 503, and buffering
means a 503 on page 7 discards the six pages already fetched and emits nothing.
Yielding per page makes each page durable the moment it is read, and the pinned
watermark is what makes that safe: a run killed at page 3 of 7 commits the
watermark it started with, so the next run re-reads pages 1-3 and dedup collapses
them. Duplicated, never lost.

**A truncated descent parks its progress rather than committing it.** If the page
budget runs out before the descent reaches ground a previous run covered, the
newest timestamp seen goes into `checkpoint["pending_watermark"]` and the
watermark stays put -- the same mechanism, and for the same reason, as
`connectors/social/reddit.py`. Committing it would put the watermark above
records between the deepest page read and the old watermark that nobody emitted.
The parked `page_token` is a *page number* against a feed that shifts as reviews
arrive, so it is advisory in the strongest sense (§4.1 rule 4): what makes
resuming mid-descent safe is not the accuracy of the page number but the fact
that the watermark did not move, so the next full descent re-covers whatever slid
past the boundary.

**One connector account is one (app, storefront) pair.** Reviews are
per-storefront: the US feed and the DE feed are disjoint sets written by
different users in different languages. `params["country"]` therefore
participates in `params_hash` and each storefront gets its own cursor row
(§4.1 rule 5). Watching an app in twelve countries is twelve accounts, not one
connector that fans out -- a fan-out would need twelve watermarks and the runtime
persists one.

**The archive is 500 reviews and there is no way past it.** Ten pages of fifty,
most-recent first, and no date parameters at all. `supports_backfill` is False
because a backfill mode could only re-read the same 500 reviews under a different
cursor row.

**Identity is rule 1** (`docs/signal-model.md` §4.1): `native_id` is Apple's own
review id, verbatim, so a DLQ record names something that can be looked up. Rule
2 must never be reached here, which is why `_record_identity` offers the ladder
no URL: every entry in this feed carries the *same* `link` href -- it points at
the app's review page, because Apple publishes no per-review permalink -- so a
fallback to "sha256 of the canonical URL" would file every unidentifiable entry
in a page under one id and collapse them into a single Signal.

**A star rating is not engagement.** `models/signal.py` is explicit: "A 1-star
rating is polarity and belongs in `Sentiment`; `helpful_votes` on that review is
endorsement." A connector may not run the sentiment stage, so the rating is
carried as `metadata["app_store.rating"]` where `services/signal_engine/` can
read it as a prior. `im:voteSum` and `im:voteCount` are real counters and go to
`Engagement.raw`.

**Known limitation: an edited review keeps its id.** Apple re-stamps `updated`
and floats the review back to the top of `mostrecent`, but the review id does not
change, so layer-1 dedup drops the new text for as long as
`SyncContext.dedup_ttl_seconds` remembers the first version. OmniSense therefore
keeps the text it saw first. Changing that means changing what identity dedup
means, which is a decision for `docs/signal-model.md` §4.2 rather than for one
connector.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Self

import httpx

from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal
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

__all__ = ["AppStoreConnector"]


# --------------------------------------------------------------------------- #
# Provider constants
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL: Final = "https://itunes.apple.com"

MAX_FEED_PAGES: Final = 10
"""How many pages the customer-reviews feed serves before it stops.

Ten, of fifty reviews each. Not a policy this connector chose and not something a
parameter can raise: page 11 answers with an empty feed or a 404. It is also the
whole archive -- there is no date range and no offset beyond it.
"""

APP_STORE_OVERLAP_SECONDS: Final = 3600
"""One hour, against the 300-second default of `docs/connector-spec.md` §4.1 rule 3.

A review's `updated` stamp is when the user wrote it; its appearance in the feed
waits on Apple's moderation and then on an edge cache. Tens of minutes between
the two is routine, so a five-minute overlap leaves late arrivals permanently
below the watermark. The overlap is cheap here in a way it is not elsewhere: it
does not widen a query, it only decides how many fifty-review pages the descent
walks before it stops, and dedup collapses everything it re-reads.
"""

QUOTA_RETRY_AFTER_SECONDS: Final = 900.0
"""Above this wait a 429 becomes a `QuotaError` rather than a held worker (§5.2)."""

_RATE_LIMIT_HEADERS: Final = frozenset({"retry-after"})
"""The only response header that leaves `fetch()`.

An allowlist rather than a denylist, exactly as in `connectors/news/gdelt.py`:
`FetchPage.raw_headers` travels with the batch into code that may log what it is
handed, and an Akamai-fronted response carries `Set-Cookie` and edge identifiers
that `docs/connector-spec.md` §1 forbids logging. Apple publishes no
`X-RateLimit-*` on this endpoint, so `Retry-After` is the only thing there is to
carry.
"""

_APP_ID: Final = re.compile(r"^[0-9]{1,20}$")
"""Apple's numeric `trackId`. Validated because it is interpolated into the
request path -- anything else is a 404 against a resource nobody configured, and
the same reasoning `connectors/social/reddit.py` applies to a subreddit name."""

_COUNTRY: Final = re.compile(r"^[A-Za-z]{2}$")
"""ISO 3166-1 alpha-2 storefront code. Also a path segment."""

_REVIEWER_URI: Final = re.compile(r"/id(\d+)")
"""The reviewer id inside `author.uri`, e.g. `.../us/reviews/id123456789`."""

PENDING_WATERMARK_KEY: Final = "pending_watermark"
"""Where an unfinished descent parks progress it is not allowed to commit.

In `Cursor.checkpoint` rather than in `watermark` because the runtime *interprets*
the watermark -- it schedules and detects gaps from it -- while `checkpoint`
round-trips as opaque JSON (`docs/connector-spec.md` §4).
"""


# --------------------------------------------------------------------------- #
# Field map
# --------------------------------------------------------------------------- #


def _star_rating(value: Any) -> int | None:
    """`im:rating` as an integer in 1..5, or `None` for anything else.

    Returning `None` is what makes the spec below fire its `required` check, so a
    feed entry whose rating is missing, non-numeric or out of range becomes an
    attributable DLQ record rather than a Signal that silently lost the only
    quantitative thing an App Store review carries.
    """
    try:
        rating = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return rating if 1 <= rating <= 5 else None


def _reviewer_id(value: Any) -> str:
    """The numeric reviewer id out of `author.uri`, or `""`.

    `author.name` is a nickname the reviewer can change at will, and
    `docs/signal-model.md` §3.1 forbids keying an author's history on a renameable
    handle. The `uri` carries `/id<digits>`, which is the account's stable review
    identity, so that is what becomes `platform_author_id` -- and when the pattern
    does not match, `FieldMap._author` returns no `Author` at all rather than
    promoting the nickname into the id slot.
    """
    match = _REVIEWER_URI.search(value) if isinstance(value, str) else None
    return match.group(1) if match else ""


_FIELD_MAP: Final = FieldMap(
    platform=Platform.APP_STORE,
    # Every scalar in this feed is wrapped as `{"label": ...}` -- an artefact of
    # Apple serializing an Atom document into JSON. `updated` is ISO-8601 with a
    # storefront-local offset ("2026-01-15T09:22:31-07:00"), which arrives aware
    # and needs no `assume_timezone`.
    timestamp=FieldSpec.at("updated.label", required=True),
    item_id=FieldSpec.at("id.label", required=True),
    # The same href on every entry in the feed: Apple publishes no per-review
    # permalink, so this points at the app's review page. Harmless as
    # `Signal.url`, which is a citation target, and deliberately kept away from
    # identity -- see `_record_identity`.
    url=FieldSpec.at("link.attributes.href"),
    title=FieldSpec.at("title.label"),
    # `content` is a two-element array: index 0 is the review body with
    # `attributes.type == "text"`, index 1 is Apple's content-type marker. The
    # second path covers the XML-to-JSON collapse that turns a one-element array
    # into a bare object, which this feed does elsewhere with `entry`.
    text=FieldSpec.at("content.0.label", "content.label"),
    # Not `text_is_html`. Apple hands over the `type="text"` rendering, so there
    # is no markup to strip, and running a readability extractor over a
    # two-sentence review risks it deciding the whole thing is boilerplate.
    engagement={
        # Helpful votes are endorsement and belong here. The star rating does not
        # -- see the module docstring and `models/signal.py`'s `Engagement`.
        "helpful_votes": FieldSpec.at("im:voteSum.label"),
        "total_votes": FieldSpec.at("im:voteCount.label"),
    },
    metadata={
        "app_store.rating": FieldSpec.at("im:rating.label", required=True, transform=_star_rating),
        # Which build the reviewer was running. The single most useful field in
        # this feed for a release post-mortem, and it exists nowhere else.
        "app_store.app_version": FieldSpec.at("im:version.label"),
    },
    author_id=FieldSpec.at("author.uri.label", transform=_reviewer_id),
    author_handle=FieldSpec.at("author.name.label"),
    author_profile_url=FieldSpec.at("author.uri.label"),
)


# --------------------------------------------------------------------------- #
# Configuration validation (no I/O, all of it at construction time)
# --------------------------------------------------------------------------- #


def _validated_app_id(params: Mapping[str, Any]) -> str:
    raw = _as_str(params.get("app_id") or params.get("track_id"))
    if not raw:
        raise ConnectorConfigurationError(
            "app_store needs params['app_id']: Apple's numeric trackId, the digits "
            "in an apps.apple.com/.../id284882215 URL. There is no default and no "
            "way to search for one from this feed",
            connector=AppStoreConnector.slug,
        )
    if not _APP_ID.match(raw):
        raise ConnectorConfigurationError(
            f"params['app_id']={raw!r} is not a numeric Apple trackId. The value is "
            "interpolated into the request path, so anything else is a 404 against a "
            "resource nobody configured -- most often a bundle id pasted in by mistake",
            connector=AppStoreConnector.slug,
        )
    return raw


def _validated_country(params: Mapping[str, Any]) -> str:
    raw = _as_str(params.get("country") or params.get("storefront"))
    if not raw:
        raise ConnectorConfigurationError(
            "app_store needs params['country']: an ISO 3166-1 alpha-2 storefront code. "
            "Defaulting to 'us' would be a silent product decision -- reviews are "
            "per-storefront, so the default would decide which market the account "
            "actually watches",
            connector=AppStoreConnector.slug,
        )
    if not _COUNTRY.match(raw):
        raise ConnectorConfigurationError(
            f"params['country']={raw!r} is not a two-letter storefront code",
            connector=AppStoreConnector.slug,
        )
    return raw.lower()


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #


class AppStoreConnector(BaseConnector):
    """One app's customer reviews in one storefront, newest page first."""

    slug: ClassVar[str] = "app_store"
    platform: ClassVar[Platform] = Platform.APP_STORE
    category: ClassVar[SourceCategory] = SourceCategory.REVIEWS
    auth_type: ClassVar[AuthType] = AuthType.NONE
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=10, burst=1, concurrency=1
    )
    """Serialized, because there is no published limit to size a budget against.

    Apple documents no quota for this endpoint and sends no `X-RateLimit-*`, so
    `parse_rate_limit` has nothing to reconcile and the bucket runs on a local
    estimate forever. `docs/connector-spec.md` §9.5 sets the posture for exactly
    that case -- serialize -- and `burst=1, concurrency=1` are the load-bearing
    parts: an undocumented limiter answers a burst with a silent block or a 503,
    and there is no header to say it happened.

    The per-minute figure is derived from the only real bound that exists rather
    than picked: a complete descent is at most `MAX_FEED_PAGES` requests, so ten a
    minute means a cold start occupies one minute and a steady-state poll spends
    one or two requests.
    """

    supports_incremental: ClassVar[bool] = True

    supports_backfill: ClassVar[bool] = False
    """The feed serves the 500 most recent reviews and takes no date parameters.

    A backfill mode would differ from an incremental run only in which cursor row
    it wrote, while hitting the same wall at page 10. Saying so is more useful
    than a mode that quietly stops there.
    """

    overlap_seconds: ClassVar[int] = APP_STORE_OVERLAP_SECONDS

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        # Both raise `ConnectorConfigurationError` -- a `PermanentError` -- before
        # a socket exists. §6: configuration defects fail fast and no cursor is
        # ever created for one.
        self._app_id = _validated_app_id(params)
        self._country = _validated_country(params)
        self._client: httpx.AsyncClient | None = None
        self._mapping = MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=ctx.run_id,
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct and validate. No I/O: not even the HTTP client is built."""
        return cls(ctx, credentials)

    # ------------------------------------------------------------- lifecycle --

    async def authenticate(self) -> None:
        """Build the HTTP client. Idempotent, and there is nothing to authenticate.

        The feed is open (`auth_type = NONE`), so `self.credentials` is never read
        and no request this connector makes carries a secret. That is precisely
        why the *response* header allowlist in `_get` still exists: the risk here
        is not leaking a credential we send, it is passing an edge cache's
        `Set-Cookie` onward inside `FetchPage.raw_headers`.

        The `User-Agent` is not cosmetic on an undocumented endpoint: an
        unattributed client is the first thing a silent limiter throttles, and it
        is the only way Apple could reach us instead of blocking us.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.ctx.request_timeout_seconds,
                headers={"User-Agent": self.ctx.user_agent, "Accept": "application/json"},
                # A redirect off this path is a storefront interstitial or a
                # captive portal, never a moved feed. Following it would parse
                # whatever answered as a review list.
                follow_redirects=False,
            )

    async def aclose(self) -> None:
        """Release the client. Idempotent: `run()` closes in a `finally`."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ----------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Descend the feed newest page first, pinning the watermark until it ends.

        Three things end a descent, and all three mean the same thing -- there is
        nothing older left worth reading:

        - an empty page, which is how the feed says it has run out;
        - `MAX_FEED_PAGES`, which is where Apple stops serving;
        - a page whose oldest review is at or below the watermark this run
          started from, which is ground a previous run already covered.

        Everything before that point yields a *parked* cursor: the watermark the
        run began with, the next page number, and the newest timestamp seen so far
        in the checkpoint. Only the page that ends the descent promotes it.
        """
        floor = _watermark_floor(cursor, self.overlap_seconds)
        newest = _max_moment(floor, _parse_pending(cursor))
        page_number = _resume_page(cursor)
        fetched = 0
        pages = 0

        while page_number <= MAX_FEED_PAGES:
            if self._budget_reached(fetched, pages):
                # Checked before the request that would exceed the ceiling, not
                # after it. Nothing is yielded here: the page already emitted
                # carries the parked cursor that resumes exactly at
                # `page_number`, and yielding an empty page would only repeat it.
                return

            path = self._feed_path(page_number)
            await self.acquire_slot(self._base_url)
            entries, headers = await self._get(path, page_number)

            fetched += len(entries)
            pages += 1
            moments = [_updated_at(entry) for entry in entries]
            newest = _max_moment(newest, *moments)

            descent_done = (
                not entries
                or page_number >= MAX_FEED_PAGES
                or _crossed(_min_moment(*moments), cursor.watermark)
            )
            yield FetchPage(
                records=[self._to_record(entry, path) for entry in _oldest_first(entries)],
                cursor=(
                    # The descent is complete: everything between the old
                    # watermark and `newest` has been yielded, so the watermark
                    # may finally move and the parked state is cleared.
                    cursor.advanced_to(watermark=newest, page_token=None, **_pending(None))
                    if descent_done
                    # Pinned. The pages still below this one are *older* than
                    # everything emitted so far, so committing `newest` here would
                    # put the watermark above records nobody has read.
                    else cursor.advanced_to(
                        watermark=floor,
                        page_token=str(page_number + 1),
                        **_pending(newest),
                    )
                ),
                raw_headers=headers,
            )
            if descent_done:
                return
            page_number += 1

    def _feed_path(self, page_number: int) -> str:
        """The feed URL for one page.

        Path parameters, not a query string -- this endpoint predates the
        convention. `page` comes first because Apple's own generated `rel="next"`
        links are shaped that way and a differently ordered path has been observed
        answering 404 on some storefront edges.
        """
        return (
            f"/{self._country}/rss/customerreviews"
            f"/page={page_number}/id={self._app_id}/sortby=mostrecent/json"
        )

    def _budget_reached(self, fetched: int, pages: int) -> bool:
        """Whether this run has spent its page or record ceiling.

        Counted on records *fetched* rather than emitted because a connector
        cannot see what survived dedup. Fetched is an upper bound on emitted, so
        the run stops at or before the ceiling `BaseConnector.run()` enforces --
        and it stops before spending a request whose tail would be discarded.
        """
        if self.ctx.max_pages is not None and pages >= self.ctx.max_pages:
            return True
        return self.ctx.max_records is not None and fetched >= self.ctx.max_records

    def _to_record(self, entry: Mapping[str, Any], path: str) -> RawRecord:
        """Wrap one feed entry verbatim, with the bytes that will be archived.

        Per-record provider bytes do not exist -- one response carries fifty
        entries -- so they are synthesized once, here, in a canonical encoding.
        `lineage.raw_sha256` is then taken over exactly the bytes the runtime PUTs
        to R2, which is what content-addressing requires. Re-serializing later, in
        another process on another json library, is the failure
        `RawRecord.raw_bytes` exists to prevent.
        """
        return RawRecord(
            native_id=self._record_identity(entry),
            payload=entry,
            raw_bytes=json.dumps(
                entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
            content_type="application/json",
            source_url=_href(entry.get("link")) or None,
            request_fingerprint=_fingerprint(path),
        )

    def _record_identity(self, entry: Mapping[str, Any]) -> str:
        """Rule 1, through the same function `normalize()` uses.

        No `url` is offered to the ladder, and that omission is the point. Every
        entry in this feed carries the same `link` href, so a fall-through to rule
        2 would hash one URL into one id and file every unidentifiable entry on
        the page under it -- turning "several broken payloads" into "one payload
        that keeps changing". A digest of the entry itself keeps them distinct
        while still collapsing two arrivals of the *same* broken entry into one
        DLQ record.
        """
        try:
            return derive_native_id(platform=self.platform, item_id=_label(entry.get("id")))
        except NormalizationError:
            material = json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
            return f"unidentified:{hashlib.sha256(material).hexdigest()}"

    # --------------------------------------------------------------- request --

    async def _get(
        self, path: str, page_number: int
    ) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
        """One request, returning the entry list. Raises; never retries or sleeps."""
        client = self._client
        if client is None:  # pragma: no cover -- run() always authenticates first
            raise PermanentError(
                "app_store fetch ran before authenticate(); there is no HTTP client",
                connector=self.slug,
            )

        try:
            response = await client.get(path)
        except httpx.TransportError as exc:
            raise TransientError(
                f"app_store request failed: {type(exc).__name__}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                f"app_store request could not be issued: {type(exc).__name__}",
                connector=self.slug,
                cause=exc,
            ) from exc

        headers = _rate_limit_headers(response.headers)
        if self._feed_ended(response, page_number):
            return [], headers
        return self._entries(response), headers

    def _feed_ended(self, response: httpx.Response, page_number: int) -> bool:
        """Classify the status, returning True when this page is past the end.

        Everything that is not "the feed ran out here" raises, mapped onto the
        four families of `connectors/exceptions.py`. The boolean exists because
        one status -- a 404 on a page after the first -- is neither a failure nor
        a body worth parsing, and the caller must not hand its HTML error page to
        the JSON decoder.
        """
        status = response.status_code
        if 300 <= status < 400:
            # Redirects are off, so a 3xx surfaces here rather than being followed
            # into whatever answered. On a legacy endpoint this is the shape a
            # retirement takes, and it deserves a named failure rather than a
            # JSON decode error against an empty body.
            raise PermanentError(
                f"app_store feed answered {status}; the endpoint moved and redirects "
                "are not followed. The RSS review feed is legacy "
                "(docs/connector-spec.md §9.2) and this is how it would be retired",
                connector=self.slug,
                status_code=status,
            )
        if status < 400:
            return False
        if status == 429:
            raise self._throttled(response)
        if status == 404:
            if page_number > 1:
                # Not a misconfiguration: some storefront edges answer a page past
                # the end of the feed with 404 instead of an empty document.
                # Treating it as the end costs nothing -- the pages beyond are
                # older than everything already read.
                return True
            raise ConnectorConfigurationError(
                f"no customer-review feed for app {self._app_id} in storefront "
                f"'{self._country}'. Either the trackId is wrong or the app is not "
                "sold in that storefront; both are configuration, and both answer 404 "
                "identically on every retry",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status == 403:
            raise ConnectorConfigurationError(
                f"storefront '{self._country}' refused the review feed",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status >= 500:
            # 503 from the edge is the normal failure of this endpoint under load.
            raise TransientError(
                f"app_store feed returned {status}",
                connector=self.slug,
                status_code=status,
            )
        raise PermanentError(
            f"app_store feed returned {status}; the request is wrong and will be wrong again",
            connector=self.slug,
            status_code=status,
        )

    def _throttled(self, response: httpx.Response) -> QuotaError | TransientError:
        """A 429 is transient inside the cap and a quota beyond it (§5.2).

        The split is operational: a `TransientError` holds the worker through a
        backoff, a `QuotaError` commits the cursor and hands the worker back.
        Fifteen minutes is where the second becomes cheaper.
        """
        hint = self.parse_rate_limit(response.headers)
        wait = hint.retry_after_seconds if hint is not None else None
        if wait is not None and wait > QUOTA_RETRY_AFTER_SECONDS:
            return QuotaError(
                "app_store asked for a long wait",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                retry_after_seconds=wait,
                reset_at=hint.reset_at if hint is not None else None,
            )
        return TransientError(
            "app_store throttled the request",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=response.status_code,
        )

    def _entries(self, response: httpx.Response) -> list[Mapping[str, Any]]:
        """Decode the body and pull out `feed.entry`.

        Three shapes have to survive here, and only the last is a defect:

        - **No `entry` key at all.** An app with no reviews in this storefront, or
          a page past the end. Zero records, not a failure.
        - **`entry` as a bare object.** Apple serializes Atom into JSON, and a
          one-element list collapses into the element. A connector that only
          handled the list form would silently drop the single review an app has.
        - **`entry` as something else, or no `feed` at all.** A shape change, and
          a `PermanentError` -- retrying returns the same bytes.

        The body is measured, never quoted: §1 forbids logging it. Note also that
        this endpoint answers `Content-Type: text/javascript` for a JSON document,
        which is why nothing here checks the content type.
        """
        try:
            body = response.json()
        except ValueError as exc:
            # A 200 that is not JSON is an intermediary -- an edge error page, a
            # captive portal -- rather than Apple. Those recover, so it is
            # transient.
            raise TransientError(
                f"app_store feed returned {len(response.content)} bytes of non-JSON "
                f"with status {response.status_code}",
                connector=self.slug,
                status_code=response.status_code,
                cause=exc,
            ) from exc

        feed = body.get("feed") if isinstance(body, Mapping) else None
        if not isinstance(feed, Mapping):
            raise PermanentError(
                "app_store response carries no 'feed' object; the shape changed",
                connector=self.slug,
                status_code=response.status_code,
            )

        entry = feed.get("entry")
        if entry is None:
            return []
        if isinstance(entry, Mapping):
            return [entry]
        if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
            raise PermanentError(
                "app_store 'feed.entry' is neither an object nor a list; the shape changed",
                connector=self.slug,
                status_code=response.status_code,
            )
        return [item for item in entry if isinstance(item, Mapping)]

    # ------------------------------------------------------------- normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one feed entry onto a Signal, or drop it.

        Exactly one thing is dropped: the app's own summary entry. Some
        storefronts prepend it to page 1 -- it carries `im:name`, `im:price` and
        an artwork list, and no rating and no review body. It is detected by shape
        rather than by position, because whether it appears at all varies by
        storefront and by page. It is a *drop* and not a DLQ record: the payload
        is well-formed, it simply is not a review, and filing it would put a
        recurring non-defect in the queue where real mapping bugs live.

        A review with a rating and no prose is kept. The star is the observation,
        and dropping text-less reviews would bias every rating distribution
        computed downstream toward the reviewers who happened to write something.
        """
        entry = record.payload
        if _is_app_summary(entry):
            return None

        # The runtime keys the R2 object and the Kafka partition off
        # `RawRecord.native_id`, while every store keys off `Signal.id`, derived
        # from `id.label`. If those disagreed the same review would exist under
        # two identities, so the disagreement is caught here rather than
        # discovered as duplicate rows months later. It also catches the entry
        # that had no id at all, which `_record_identity` filed under a digest.
        if _label(entry.get("id")) != record.native_id:
            raise NormalizationError(
                "feed entry id does not match the fetched record's native_id",
                native_id=record.native_id,
                connector=self.slug,
            )

        return _FIELD_MAP.to_signal(
            record,
            self._mapping,
            # Neither is in the payload: the feed describes a review without ever
            # naming the app or the storefront it came from. Without them a Signal
            # in the store cannot be attributed to a market, and every
            # per-storefront aggregate downstream would need to join back through
            # the connector account to find out.
            extra_metadata={
                "app_store.app_id": self._app_id,
                "app_store.country": self._country,
            },
        )


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #


def _is_app_summary(entry: Mapping[str, Any]) -> bool:
    """Whether this entry describes the app rather than a review.

    Both markers are required. A review always carries `im:rating`, and it
    normally carries `content`; testing only for the missing rating would
    silently drop a genuinely malformed review that deserves the DLQ instead.
    """
    return "im:rating" not in entry and "content" not in entry


def _label(value: Any) -> str:
    """The `label` of one Apple feed node, or `""`.

    Every scalar in this document is wrapped -- `{"label": "5"}`, not `"5"` --
    because it is an Atom document rendered as JSON. Unwrapping in one place keeps
    the pagination arithmetic and the field map reading the same thing.
    """
    if isinstance(value, Mapping):
        return _as_str(value.get("label"))
    return _as_str(value)


def _href(value: Any) -> str:
    """`link.attributes.href`, or `""` when the node is not shaped like a link."""
    if isinstance(value, Mapping):
        attributes = value.get("attributes")
        if isinstance(attributes, Mapping):
            return _as_str(attributes.get("href"))
    return ""


def _updated_at(entry: Mapping[str, Any]) -> datetime | None:
    """Parse `updated`, or `None`.

    Never raises. An entry with an unparseable date still has to reach
    `normalize()`, which is the stage allowed to DLQ it with its identity
    attached; raising here would abort the page and block every well-formed entry
    behind it. Naive values are refused rather than assumed to be UTC -- this feed
    always sends an offset, so a naive value means something other than Apple
    produced it, and guessing would move a watermark by hours.
    """
    raw = _label(entry.get("updated"))
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else None


def _oldest_first(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Sort one page oldest-first.

    Across pages this connector yields in provider order under the exception
    `BaseConnector.fetch` grants. *Within* a page there is no exception -- the
    contract says records are ordered oldest-first, always -- and here it is free.
    """
    return sorted(entries, key=lambda entry: _updated_at(entry) or datetime.min.replace(tzinfo=UTC))


def _crossed(oldest: datetime | None, watermark: datetime | None) -> bool:
    """Whether this page reaches into ground a previous run already covered."""
    if oldest is None or watermark is None:
        return False
    return oldest <= watermark


# --------------------------------------------------------------------------- #
# Cursor helpers
# --------------------------------------------------------------------------- #


def _watermark_floor(cursor: Cursor, overlap_seconds: int) -> datetime | None:
    """Reconstruct the watermark the runtime is actually holding.

    `BaseConnector._effective_start` hands `fetch()` a cursor already shifted back
    by `overlap_seconds`, and every cursor derived from it inherits that shift. A
    truncated descent that pinned the *shifted* value would commit a watermark an
    hour older than the stored one, and §4.1 rule 2 says the runtime rejects a
    watermark that moves backwards. The shift is exactly `watermark - overlap`, so
    undoing it is exact too.

    Sound because `run()` is the only caller of `fetch()`: the floor can never
    exceed a watermark the runtime has already committed.
    """
    if cursor.watermark is None:
        return None
    return cursor.watermark + timedelta(seconds=overlap_seconds)


def _resume_page(cursor: Cursor) -> int:
    """Where a parked descent continues, clamped into the feed's page range.

    `page_token` is advisory (§4.1 rule 4) and here it is advisory twice over: it
    names a page of a feed that shifts as reviews arrive. An unreadable or
    out-of-range value restarts the descent at page 1 rather than failing the run,
    which costs a re-read that dedup collapses.
    """
    raw = cursor.page_token
    if raw is None:
        return 1
    try:
        page = int(str(raw).strip())
    except ValueError:
        return 1
    return page if 1 <= page <= MAX_FEED_PAGES else 1


def _parse_pending(cursor: Cursor) -> datetime | None:
    """Progress parked by an earlier truncated descent, or `None`.

    A checkpoint that cannot be read is not worth failing a run over: the
    watermark is still valid, so the descent simply re-covers ground.
    """
    raw = cursor.checkpoint.get(PENDING_WATERMARK_KEY)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _pending(moment: datetime | None) -> dict[str, Any]:
    """The checkpoint fragment carrying parked progress.

    `None` blanks the key by merge -- `Cursor.advanced_to` unions checkpoints, so
    a key can be emptied but not removed -- which is what stops a completed
    descent from leaving stale progress behind for the next one to promote.
    """
    return {PENDING_WATERMARK_KEY: moment.astimezone(UTC).isoformat() if moment else None}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _max_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return max(present) if present else None


def _min_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return min(present) if present else None


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in _RATE_LIMIT_HEADERS}


def _fingerprint(path: str) -> str:
    """Hash of the request path; there is no credential and no query to omit.

    `lineage.request_fingerprint` is what makes a fetch reproducible -- it names
    the exact request that produced a record. Truncated because it identifies a
    request, and a full-length digest invites someone to read it as a content hash.
    """
    return hashlib.sha256(f"GET {path}".encode()).hexdigest()[:32]


def _as_str(value: Any) -> str:
    """Render a scalar as a stripped string; `""` for anything unusable.

    Booleans and containers are refused rather than stringified: `"True"` is never
    a review id or a storefront code, and letting one through turns a type
    confusion into a plausible-looking value.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (Mapping, Sequence, set, frozenset)):
        return ""
    return str(value).strip()
