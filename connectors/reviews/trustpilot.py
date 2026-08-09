"""Trustpilot public Business Units API connector (Phase 2).

`docs/connector-spec.md` §9.2 puts Trustpilot in the small set of review sources
with a genuinely usable official API: public business-unit reviews are readable
with nothing but an API key on a paid plan. Everything specific about this module
comes from one property of that API -- it sorts, but it does not filter by time.

**The pager *is* the cursor, because there is no `createdAfter`.** The reviews
endpoint accepts a documented `orderBy` (`createdat.asc`) and 1-based `page` /
`perPage`, and nothing else that narrows a request to "reviews since T". So the
connector walks ascending page numbers and parks the page it reached in
`Cursor.page_token`. Ascending order is what makes that safe and what satisfies
`BaseConnector.fetch`'s oldest-first requirement natively rather than by
buffering and reversing the way `connectors/social/reddit.py` has to: new reviews
append to the *end* of an ascending listing, so a page already walked stays
walked, and each page's newest `createdAt` is a legal watermark the moment the
page is durable. Descending order would have inverted that -- every new review
would shift the whole listing by one and page 4 would address different reviews
on every poll.

The one cost of offset pagination is worth naming rather than hiding: a review
deleted *behind* the cursor shifts everything after it back by one position, so
exactly one review can slip across a page boundary unseen. It is bounded at one
per deletion, and it is recoverable -- dropping the cursor re-walks from page 1
and dedup collapses everything already emitted. There is no `createdAfter`
parameter to make it impossible.

**The key rides in the `apikey` header, never the query string.** Trustpilot
documents both forms and recommends the header, which is also what
`connectors/auth/apikey.py` requires: URLs are logged by every proxy in the path
and land in `Referer` on any redirect.

**A star rating is not engagement.** `models/signal.py::Engagement` says so
directly -- "a 1-star rating is polarity and belongs in `Sentiment`" -- and a
connector may not run the sentiment stage (`docs/connector-spec.md` §1). So
`stars` is carried in `metadata` where `services/signal_engine/sentiment.py` can
read it, and `Engagement.raw` holds only `numberOfLikes`, which really is an
endorsement counter.

Identity is rule 1 of `docs/signal-model.md` §4.1: `native_id` is Trustpilot's
own review id, verbatim, so a DLQ record names something a human can paste into
`https://www.trustpilot.com/reviews/<id>`. Rule 3 is never reached, so nothing
here makes identity depend on the text cleaner.

**What this connector cannot see.** Ascending-by-`createdAt` paging never
revisits a review once it is behind the cursor, so a company reply written weeks
later, or a consumer's edit, is invisible. Catching those means re-walking the
whole listing, which costs the same as a cold start; `updatedAt` is carried in
`metadata` so a future reconciliation job can find them without guessing.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar, Final, Self
from urllib.parse import urlencode

import httpx

from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal
from connectors.auth.apikey import ApiKeyAuth
from connectors.base import BaseConnector
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.normalize.mapper import FieldMap, FieldSpec, MappingContext
from connectors.protocol import (
    Credentials,
    Cursor,
    DedupKeys,
    FetchPage,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)

__all__ = ["TrustpilotConnector"]


# --------------------------------------------------------------------------- #
# Endpoints and provider constants
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL: Final = "https://api.trustpilot.com/v1"
"""The public API host. Private (OAuth2) endpoints live on the same host under
different paths; this connector touches none of them."""

WEB_BASE: Final = "https://www.trustpilot.com"
"""Where a human reads a review. The API host serves JSON only, so the `links`
array on a review points at the API resource rather than at a page anybody can
open -- which is why `Signal.url` is composed rather than mapped."""

REVIEWS_PATH: Final = "/business-units/{business_unit_id}/reviews"

ORDER_BY_CREATED_ASC: Final = "createdat.asc"
"""The documented ordering this connector is built on.

`createdat.desc`, `stars.asc` and the rest are equally documented and equally
unusable here: only an ascending *time* order makes a page number a resume point
(see the module docstring), and a star ordering makes the watermark meaningless.
"""

MAX_PER_PAGE: Final = 100
"""Trustpilot's ceiling for `perPage`."""

DEFAULT_PER_PAGE: Final = 100

FIRST_PAGE: Final = 1
"""Trustpilot pages are 1-based. Starting at 0 returns page 1 anyway, which is
the kind of silent off-by-one that makes a resume test pass for the wrong
reason."""

QUOTA_RETRY_AFTER_SECONDS: Final = 900.0
"""Above this a 429 becomes a `QuotaError` rather than a held worker (§5.2)."""

ACTIVE_STATUS: Final = "active"
"""The only `status` a review may carry and still be published.

Anything else -- reported, under moderation, removed -- is content Trustpilot has
taken down. See `normalize`.
"""

_BUSINESS_UNIT_ID: Final = re.compile(r"^[0-9a-fA-F]{24}$")
"""Trustpilot business unit ids are 24 hex characters.

Checked because the value is interpolated into the request path: anything else is
either a 404 or a request for a resource nobody configured
(`connectors/social/reddit.py` refuses a subreddit name for the same reason).
"""

_RATE_LIMIT_HEADERS: Final = frozenset(
    {"retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}
)
"""The only response headers that leave `fetch()`.

An allowlist rather than a denylist: Trustpilot publishes none of these today, and
the day it starts setting a cookie or a CDN request id, that header must not ride
out on a batch and into a log line (`docs/connector-spec.md` §1).
"""


# --------------------------------------------------------------------------- #
# The field map
# --------------------------------------------------------------------------- #


def _review_url(value: Any) -> str:
    """Compose the public permalink from the review id.

    Trustpilot's `links` array addresses the API resource
    (`https://api.trustpilot.com/v1/reviews/<id>`), which needs a key to open and
    is not what a report should cite. The public form is stable and documented,
    and composing it here means `Signal.url` and the citation in a report are the
    same string.
    """
    review_id = _as_str(value)
    return f"{WEB_BASE}/reviews/{review_id}" if review_id else ""


def _consumer_url(value: Any) -> str:
    consumer_id = _as_str(value)
    return f"{WEB_BASE}/users/{consumer_id}" if consumer_id else ""


_FIELD_MAP: Final = FieldMap(
    platform=Platform.TRUSTPILOT,
    # ISO-8601 with a `Z` suffix, which `to_utc_datetime` reads through
    # `datetime.fromisoformat` -- no bespoke parser, and therefore no second
    # place for the timestamp of a Signal to be interpreted differently.
    # `createdAt`, never `updatedAt`: the event is when the review was written.
    timestamp=FieldSpec.at("createdAt", required=True),
    # Rule 1. `required` because a review without an id is not a review -- it is
    # a shape change, and falling through to rule 2 would derive identity from a
    # URL this module composed out of the very field that is missing.
    item_id=FieldSpec.at("id", required=True),
    url=FieldSpec.at("id", transform=_review_url),
    title=FieldSpec.at("title"),
    text=FieldSpec.at("text"),
    engagement={
        # The only counter on a Trustpilot review that describes *this* review's
        # reception. `stars` is polarity and lives in metadata; see below.
        "likes": FieldSpec.at("numberOfLikes"),
    },
    metadata={
        # Polarity, parked where the sentiment stage can find it. A connector may
        # not run enrichment (`docs/connector-spec.md` §1), and putting a rating
        # into `Engagement` would feed it into a percentile cohort where a 1-star
        # review would read as low engagement rather than as a strong negative
        # (`models/signal.py::Engagement`).
        "trustpilot.stars": FieldSpec.at("stars"),
        # The reviewer's declared language, not the detector's opinion.
        # `Signal.language` is filled by `services/signal_engine/language.py`
        # from the text it actually has; copying a provider label into it would
        # fabricate a detector result with no confidence behind it
        # (`docs/signal-model.md` §3.3).
        "trustpilot.language": FieldSpec.at("language"),
        "trustpilot.is_verified": FieldSpec.at("isVerified"),
        "trustpilot.status": FieldSpec.at("status"),
        "trustpilot.source": FieldSpec.at("source"),
        # When the purchase happened, as opposed to when the review was written.
        # The gap between the two is one of the few honesty signals a review
        # platform exposes.
        "trustpilot.experienced_at": FieldSpec.at("experiencedAt"),
        # Carried so a future reconciliation job can find reviews edited after
        # they went behind the ascending cursor -- see the module docstring.
        "trustpilot.updated_at": FieldSpec.at("updatedAt"),
        "trustpilot.company_replied_at": FieldSpec.at("companyReply.createdAt"),
        "trustpilot.consumer_review_count": FieldSpec.at("consumer.numberOfReviews"),
        "trustpilot.consumer_country": FieldSpec.at("consumer.countryCode"),
    },
    # `consumer.id` rather than `consumer.displayName`: display names are
    # renameable and thousands of reviewers share "John D."
    # (`docs/signal-model.md` §3.1). Trustpilot has no handles, so
    # `author_handle` stays unmapped rather than being filled with a display
    # string that would read as one.
    author_id=FieldSpec.at("consumer.id"),
    author_display_name=FieldSpec.at("consumer.displayName"),
    author_profile_url=FieldSpec.at("consumer.id", transform=_consumer_url),
    # Review bodies are plain text; Trustpilot rejects markup at submission.
    # Declaring `text_is_html` would send every review through the readability
    # extractor, which is a no-op on prose but is a lie about the payload.
)


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #


class TrustpilotConnector(BaseConnector):
    """Public reviews for one Trustpilot business unit, walked oldest-first."""

    slug: ClassVar[str] = "trustpilot"
    platform: ClassVar[Platform] = Platform.TRUSTPILOT
    category: ClassVar[SourceCategory] = SourceCategory.REVIEWS
    auth_type: ClassVar[AuthType] = AuthType.API_KEY
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=60, burst=1, concurrency=1
    )
    """One request per second, serialized.

    Trustpilot's rate-limiting guidance is explicit about the threshold -- clients
    are limited above one request per second -- and advisory about the ceilings
    (roughly 833 calls per five minutes, 10,000 per hour, both about 166/min). The
    per-second rule is the one that actually trips, which is why `burst=1` and
    `concurrency=1` are the load-bearing parts rather than the per-minute figure:
    sixty requests delivered in the first two seconds of a minute is a burst that
    respects the average and violates the rule.
    """

    supports_incremental: ClassVar[bool] = True

    supports_backfill: ClassVar[bool] = True
    """History is reachable by construction, not by a special mode.

    An ascending pager *starts* at the oldest review, so a cold run is already a
    backfill and `page_token` is what makes it resumable across runs. Declaring
    backfill support gives such a run its own cursor row and the reduced budget of
    §5.1, which is exactly the separation §4.1 rule 5 asks for -- it does not
    unlock a different code path, because there is not one.
    """

    overlap_seconds: ClassVar[int] = 0
    """No rewind, because the watermark bounds nothing here.

    The default 300 exists to re-read a window whose tail the provider indexed
    late. This connector's window is a *page number*, and the watermark is never
    fed back into a request -- so rewinding it would shift a value that only the
    scheduler reads, and `BaseConnector._guard_watermark` would immediately clamp
    it back. Zero says that plainly instead of relying on the guard.
    """

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._business_unit_id = _validated_business_unit_id(params)
        self._per_page = _validated_per_page(params)
        self._auth: ApiKeyAuth | None = None
        self._client: httpx.AsyncClient | None = None
        self._mapping = MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=ctx.run_id,
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct and validate. No I/O: not even the HTTP client is built.

        Every check happens in `__init__` and raises
        `ConnectorConfigurationError` -- a `PermanentError` -- before a socket
        exists. §6: configuration defects fail fast, and no cursor is ever created
        for one.
        """
        return cls(ctx, credentials)

    # ------------------------------------------------------------ lifecycle --

    async def authenticate(self) -> None:
        """Bind the API key and build the client. Idempotent, and no I/O.

        There is no session to establish -- an API key is a constant -- and
        Trustpilot offers no free validation endpoint, so proving the key valid
        would spend a request against a one-per-second budget to learn what the
        first real fetch learns anyway.

        A missing secret is an `AuthError` rather than a configuration error: it
        is a credential row an operator has to fix, and `AuthError` is what flags
        the account `needs_reauth` (`docs/connector-spec.md` §2.1).
        """
        if self._auth is None:
            try:
                self._auth = ApiKeyAuth.from_credentials(
                    self.credentials,
                    secret_key="api_key",
                    # Trustpilot documents both `?apikey=` and this header and
                    # recommends the header. `connectors/auth/apikey.py` has no
                    # query-parameter strategy at all, for the reason it states:
                    # URLs reach proxy logs and `Referer` headers.
                    header="apikey",
                )
            except KeyError as exc:
                raise AuthError(
                    "trustpilot account has no 'api_key' secret; the public "
                    "Business Units API needs the application key (TRUSTPILOT_API_KEY)",
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
                # A redirect off an API host is a captive portal or a login page,
                # never a moved resource. Following it would send the key to
                # whatever answered and parse the result as a review list.
                follow_redirects=False,
            )

    async def aclose(self) -> None:
        """Release the client. Idempotent: `run()` closes in a `finally`."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ---------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Walk ascending pages from the parked page number forward.

        The parked page is re-read rather than skipped. A page is only "done" when
        it came back full, and the last page of a listing is by definition not
        full, so resuming *after* it would skip every review written since. The
        re-read costs one request and dedup collapses it, which is the same trade
        `connectors/news/gdelt.py` makes at its window boundary.
        """
        page = _resume_page(cursor)
        newest: datetime | None = None
        fetched = 0
        pages = 0

        while True:
            budget = self._request_budget(fetched)
            if budget == 0:
                # A zero record budget. Nothing was fetched, so there is nothing
                # to commit -- and yielding a cursor here would advance a page
                # number past a page nobody read.
                return

            params = self._query_params(page, budget)
            await self.acquire_slot(self._base_url)
            body, headers = await self._get(params)

            reviews = _reviews(body)
            fingerprint = _fingerprint(self._reviews_path(), params)
            records = [self._to_record(review, fingerprint) for review in reviews]
            fetched += len(records)
            pages += 1
            newest = _max_moment(newest, *(_created_at(review) for review in reviews))

            # Full page: everything on it is behind us, so the next run starts
            # after it. Partial page: this is the tail of the listing and it will
            # have grown by the next poll, so stay parked on it.
            full = len(reviews) >= budget
            next_page = page + 1 if full else page

            yield FetchPage(
                records=records,
                # `advanced_to` only ever moves the watermark forward, so a page
                # whose reviews are all older than the running maximum -- which
                # cannot happen in ascending order, but would on a provider-side
                # re-sort -- cannot walk progress backwards.
                cursor=cursor.advanced_to(watermark=newest, page_token=str(next_page)),
                raw_headers=headers,
            )

            if not full or self._budget_reached(fetched, pages):
                return
            page = next_page

    def _reviews_path(self) -> str:
        return REVIEWS_PATH.format(business_unit_id=self._business_unit_id)

    def _query_params(self, page: int, per_page: int) -> dict[str, str]:
        """Build the query. The credential is *not* here -- it is a header."""
        return {
            "orderBy": ORDER_BY_CREATED_ASC,
            "page": str(page),
            "perPage": str(per_page),
            # Trustpilot already defaults this to false, but a default is a thing
            # the provider may change and this connector's correctness depends on
            # it: `normalize` drops non-active reviews, so a plan that started
            # returning reported ones would silently turn every page into drops.
            # Stating it makes the request say what the code assumes.
            "includeReportedReviews": "false",
        }

    def _request_budget(self, fetched: int) -> int:
        """`perPage` for the next request, narrowed by `ctx.max_records`.

        Applied before the request rather than after the page: a run capped at 20
        records should cost one 20-review request, not a 100-review request whose
        tail is normalized, hashed and then discarded by
        `BaseConnector.run()`'s own ceiling.

        Counted on records fetched rather than emitted because a connector cannot
        see what survived dedup; fetched is an upper bound on emitted, so the run
        stops at or before the ceiling and the cursor commits either way.
        """
        if self.ctx.max_records is None:
            return self._per_page
        return _clamp(self.ctx.max_records - fetched, 0, self._per_page)

    def _budget_reached(self, fetched: int, pages: int) -> bool:
        if self.ctx.max_pages is not None and pages >= self.ctx.max_pages:
            return True
        return self.ctx.max_records is not None and fetched >= self.ctx.max_records

    def _to_record(self, review: Mapping[str, Any], fingerprint: str) -> RawRecord:
        """Wrap one review verbatim, with the bytes that will be archived.

        Per-record provider bytes do not exist -- one response carried a hundred
        reviews -- so they are synthesized once, here, in a canonical encoding.
        `lineage.raw_sha256` is then taken over exactly the bytes the runtime PUTs
        to R2, which is what content-addressing requires. Re-serializing later, in
        another process on another json library, is what `RawRecord.raw_bytes`
        exists to prevent.
        """
        review_id = _as_str(review.get("id"))
        return RawRecord(
            native_id=review_id or self._unidentified(review),
            payload=review,
            raw_bytes=json.dumps(
                review, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
            content_type="application/json",
            source_url=_review_url(review_id) or None,
            request_fingerprint=fingerprint,
        )

    def _unidentified(self, review: Mapping[str, Any]) -> str:
        """A filing name for a payload with no review id.

        It never becomes a Signal -- `item_id` is required in the field map -- so
        this string only ever appears on a DLQ record, where its job is to make
        two arrivals of the same broken payload recognisable as one instead of
        two.
        """
        material = json.dumps(review, sort_keys=True, default=str).encode("utf-8")
        return f"unidentified:{hashlib.sha256(material).hexdigest()}"

    # ------------------------------------------------------------ normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one review onto a Signal, or drop it.

        Two drops, both expected rather than defective, which is why they return
        `None` instead of raising (`docs/connector-spec.md` §2.4):

        - **A review Trustpilot has taken down.** `status` is anything but
          `active`. The request asks for published reviews only, so this should be
          unreachable -- but a plan whose entitlements change would start
          returning moderated content, and a removed review quoted in a report is
          the single worst thing this connector could do.
        - **A rating with no words.** Trustpilot allows a star rating with neither
          title nor body. There is no text to clean, embed or cite, and emitting
          one would put an empty document into the embedding queue and the search
          index. The star distribution it belongs to is available in aggregate
          from the business-unit endpoint, which is a different (and cheaper)
          request than paging every review.
        """
        payload = record.payload

        status = _as_str(payload.get("status")).casefold()
        if status and status != ACTIVE_STATUS:
            return None

        if not _as_str(payload.get("text")) and not _as_str(payload.get("title")):
            return None

        return _FIELD_MAP.to_signal(
            record,
            self._mapping,
            # Not in the payload: the reviews endpoint does not echo the business
            # unit it was asked about. Without it every Signal from every
            # competitor's page looks alike downstream, and the graph layer has
            # nothing to attach the review to.
            extra_metadata={"trustpilot.business_unit_id": self._business_unit_id},
        )

    # ---------------------------------------------------------------- dedup --

    def dedup_keys(self, signal: Signal) -> DedupKeys:
        """Identity only. Layer 2 is actively wrong for a review source.

        `BaseConnector.dedup_keys` adds an exact-content key -- a sha256 of the
        cleaned text, scoped to the connector -- to collapse the same article
        syndicated across several feeds. Reviews are not syndicated. "Great
        service, fast delivery" is written independently by thousands of distinct
        consumers, and hashing it would drop every one after the first *within the
        TTL window*, silently, in the drop counter.

        What that deletes is precisely the evidence a reviews source exists to
        provide: the volume of people who independently said the same thing. It
        would also bias the corpus toward verbose reviewers, because long reviews
        collide less. Layer 1 alone is sufficient here -- Trustpilot review ids are
        stable, so a re-fetch of the same review is caught by identity.
        """
        return DedupKeys(identity=f"os:dedup:id:{self.slug}:{signal.id}")

    # -------------------------------------------------------------- request --

    async def _get(self, params: Mapping[str, str]) -> tuple[Mapping[str, Any], dict[str, str]]:
        """One request. Raises; never retries, never sleeps.

        No retry and no sleep: §1 -- a connector that retries privately makes the
        shared limiter's accounting wrong and hides the failure from metrics.
        """
        client = self._client
        if client is None:  # pragma: no cover -- run() always authenticates first
            raise PermanentError(
                "trustpilot fetch ran before authenticate(); there is no HTTP client",
                connector=self.slug,
            )

        try:
            response = await client.get(self._reviews_path(), params=params)
        except httpx.TransportError as exc:
            raise TransientError(
                f"trustpilot request failed: {type(exc).__name__}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                f"trustpilot request could not be issued: {type(exc).__name__}",
                connector=self.slug,
                cause=exc,
            ) from exc

        self._raise_for_status(response)

        try:
            body = response.json()
        except ValueError as exc:
            # A 200 that is not JSON came from something in front of the API -- a
            # proxy notice, a captive portal -- not from Trustpilot. Those
            # recover, so it is transient. The body is measured, never quoted:
            # §1 forbids logging it.
            raise TransientError(
                f"trustpilot returned {len(response.content)} bytes of non-JSON "
                f"with status {response.status_code}",
                connector=self.slug,
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(body, Mapping):
            raise PermanentError(
                f"trustpilot returned a JSON {type(body).__name__} where an object "
                "was expected; the response shape changed",
                connector=self.slug,
                status_code=response.status_code,
            )
        return body, _rate_limit_headers(response.headers)

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map HTTP status onto the four families of `connectors/exceptions.py`."""
        status = response.status_code
        if status < 400:
            return
        if status == httpx.codes.UNAUTHORIZED:
            raise AuthError(
                "trustpilot rejected the API key",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status == httpx.codes.FORBIDDEN:
            # Deliberately *not* an `AuthError`, unlike the 401 above. On a public
            # endpoint a 403 with a syntactically valid key means the key is not
            # entitled to this API -- a plan fact, not a credential fact -- and
            # filing it as an auth failure would flag a working account
            # `needs_reauth` and send an operator to re-issue a key that was never
            # the problem. `connectors/social/reddit.py` splits 403 for the same
            # reason.
            raise ConnectorConfigurationError(
                "trustpilot refused the Business Units API for this key; public "
                "business-unit reviews are a paid-plan entitlement "
                "(docs/connector-spec.md §9.2), so this is a plan or key-scope "
                "problem rather than an expired credential",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status == httpx.codes.NOT_FOUND:
            raise PermanentError(
                f"trustpilot has no business unit {self._business_unit_id!r}; the "
                "configured id is wrong and will be wrong again on retry",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status == httpx.codes.TOO_MANY_REQUESTS:
            raise self._throttled(response)
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise TransientError(
                f"trustpilot returned {status}",
                connector=self.slug,
                status_code=status,
            )
        raise PermanentError(
            f"trustpilot returned {status}; the request is wrong and will be wrong again",
            connector=self.slug,
            status_code=status,
        )

    def _throttled(self, response: httpx.Response) -> QuotaError | TransientError:
        """A 429 is transient inside the cap and a quota beyond it (§5.2).

        The split is operational: a `TransientError` holds the worker through a
        backoff, while a `QuotaError` commits the cursor and hands the worker
        back. Fifteen minutes is where the second becomes cheaper.
        """
        hint = self.parse_rate_limit(response.headers)
        wait = hint.retry_after_seconds if hint is not None else None
        if wait is not None and wait > QUOTA_RETRY_AFTER_SECONDS:
            return QuotaError(
                "trustpilot quota is exhausted",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                retry_after_seconds=wait,
                reset_at=hint.reset_at if hint is not None else None,
            )
        return TransientError(
            "trustpilot throttled the request",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=response.status_code,
        )


# --------------------------------------------------------------------------- #
# Configuration validation (no I/O, all of it at construction time)
# --------------------------------------------------------------------------- #


def _validated_business_unit_id(params: Mapping[str, Any]) -> str:
    """Resolve `params['business_unit_id']` into one path segment.

    Required rather than resolved from a domain name. Trustpilot does offer
    `/business-units/find?name=<domain>`, but calling it would spend one of the
    sixty requests a minute this connector is allowed, on every run, to look up a
    value that never changes -- and caching the answer in the cursor would turn
    resume state into a configuration store.
    """
    raw = params.get("business_unit_id") or params.get("businessUnitId")
    candidate = _as_str(raw)
    if not candidate:
        raise ConnectorConfigurationError(
            "the trustpilot connector needs params['business_unit_id']; there is "
            "no default business unit, and the id is what the reviews path is "
            "built from. Find it once with GET /v1/business-units/find?name="
            "<domain> and store it on the connector account",
            connector=TrustpilotConnector.slug,
        )
    if not _BUSINESS_UNIT_ID.match(candidate):
        raise ConnectorConfigurationError(
            f"{candidate!r} is not a Trustpilot business unit id (24 hex "
            "characters). The value is interpolated into the request path, so "
            "anything else is either a 404 or a request for a resource nobody "
            "configured",
            connector=TrustpilotConnector.slug,
        )
    return candidate


def _validated_per_page(params: Mapping[str, Any]) -> int:
    raw = params.get("per_page", DEFAULT_PER_PAGE)
    try:
        size = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConnectorConfigurationError(
            f"params['per_page'] must be an integer, got {raw!r}",
            connector=TrustpilotConnector.slug,
        ) from exc
    if not 1 <= size <= MAX_PER_PAGE:
        raise ConnectorConfigurationError(
            f"params['per_page'] must be between 1 and {MAX_PER_PAGE}; a larger "
            "value is clamped by the provider, which would leave the page "
            "arithmetic -- and therefore the resume point -- resting on a number "
            "Trustpilot never agreed to",
            connector=TrustpilotConnector.slug,
        )
    return size


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #


def _resume_page(cursor: Cursor) -> int:
    """The page this run starts on.

    An unreadable or nonsensical token restarts from page 1 rather than failing
    the run: `docs/connector-spec.md` §4.1 rule 4 makes `page_token` advisory, and
    re-walking an ascending listing is expensive but correct, while refusing to
    run is neither.
    """
    page = _as_int(cursor.page_token, FIRST_PAGE)
    return page if page >= FIRST_PAGE else FIRST_PAGE


def _reviews(body: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The `reviews` array, or a `PermanentError` if it is not one.

    A missing key is a shape change, not a business unit with no reviews: the
    documented empty result is `"reviews": []`. Defaulting to `[]` here would turn
    a breaking provider change into a run that succeeds forever with zero records
    -- the failure mode nobody notices, because the dashboard stays green.
    """
    reviews = body.get("reviews")
    if not isinstance(reviews, Sequence) or isinstance(reviews, (str, bytes)):
        raise PermanentError(
            "trustpilot response has no 'reviews' array; the response shape changed"
        )
    return [item for item in reviews if isinstance(item, Mapping)]


def _created_at(review: Mapping[str, Any]) -> datetime | None:
    """Parse `createdAt`, or `None`.

    Never raises. A review with an unparseable date still has to reach
    `normalize()`, which is the stage allowed to DLQ it with its identity
    attached; raising here would abort the page and block every well-formed review
    behind it. Naive input is refused rather than assumed to be UTC -- the
    provider always suffixes, so a naive value means something unexpected produced
    it, and guessing would shift a watermark by hours.
    """
    value = review.get("createdAt")
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
    """Hash of endpoint plus normalized params. The credential is in a header.

    `lineage.request_fingerprint` is what makes a fetch reproducible: it names the
    exact request that produced a record without naming who made it. Truncated
    because it identifies a request, and a full-length digest invites someone to
    mistake it for a content hash.
    """
    material = endpoint + "?" + urlencode(sorted(params.items()))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _as_str(value: Any) -> str:
    """Render a scalar as a stripped string; `""` for anything unusable.

    Booleans are refused rather than stringified: `"True"` is never a review id or
    a business unit id, and letting one through turns a type confusion into a
    plausible-looking value.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (Mapping, Sequence, set, frozenset)):
        return ""
    return str(value).strip()


def _as_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
