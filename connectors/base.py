"""`BaseConnector`: the six-stage contract every source integration implements.

Design Doc §5 fixes the order:

    Authentication -> Fetch -> Rate Limit -> Normalize -> Deduplicate -> Emit

`run()` is `@final` because that order is not a connector's decision, and neither
is the decision to publish. A connector supplies four things -- `from_config`,
`authenticate`, `fetch`, `normalize` -- and the template method does the rest:
acquires rate-limit tokens around every page, feeds provider rate-limit headers
back into the bucket, deduplicates, and *yields* batches. It never writes
anywhere. `services/connector_service.py` performs the R2 PUT, the Kafka produce
and the cursor commit, in that order, for each yielded batch
(`docs/connector-spec.md` §2.6).

That split is what keeps `connectors/` free of any `services/` or `backend/`
import (`docs/architecture.md` §6.2 rule 2), which in turn is what lets a
connector be tested with `respx` and two in-memory fakes.

Three prohibitions are enforced structurally rather than by review:

- **No internal retry.** `fetch()` raises `TransientError` and stops. Backoff is
  the runtime's, because a connector that retries privately makes the shared
  limiter's accounting wrong and hides the failure from metrics.
- **No internal sleeping.** The limiter is cross-process; a local `sleep` would
  rate-limit one worker while eight others hammer the provider.
- **No writes.** A connector holds no session, no producer and no object store,
  so a run is replayable by construction.
"""

from __future__ import annotations

import abc
import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import ClassVar, Self, final
from urllib.parse import urlsplit

from models.base import utcnow
from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal
from connectors.exceptions import NormalizationError
from connectors.protocol import (
    Credentials,
    Cursor,
    DedupKeys,
    EmittedBatch,
    FetchPage,
    HealthReport,
    RateLimitHint,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
    SyncMode,
)

__all__ = ["BaseConnector"]


class BaseConnector(abc.ABC):
    """Abstract base for every OmniSense connector."""

    # -- declaration, read by connectors/registry.py before instantiation ----
    slug: ClassVar[str]
    platform: ClassVar[Platform]
    category: ClassVar[SourceCategory]
    auth_type: ClassVar[AuthType]
    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy()
    version: ClassVar[str] = "0.1.0"
    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = False

    requires_tos_review: ClassVar[bool] = False
    """When True the registry refuses to enable this connector.

    Set on every source with no viable official API for this use case --
    Instagram, TikTok, LinkedIn, Amazon reviews (`docs/connector-spec.md` §9).
    Implementing those means scraping, which needs a documented legal review
    first. A boolean in the class declaration is checkable; a comment is not.
    """

    overlap_seconds: ClassVar[int] = 300
    """How far back before the watermark an incremental run restarts.

    Provider indexes lag their own timestamps, so resuming exactly at the
    watermark quietly drops records that were written late. Overlap plus dedup is
    how they are caught (`docs/connector-spec.md` §4.1 rule 3). Eventually
    consistent providers such as GDELT raise this to 900.
    """

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        self.ctx = ctx
        self.credentials = credentials

    # ----------------------------------------------------------- lifecycle --

    @classmethod
    @abc.abstractmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct an instance. **Must not perform I/O.**

        Separate from `authenticate()` so the scheduler can build a connector to
        inspect its declaration without opening a socket, and so construction
        cannot fail for a reason that deserves a retry.
        """

    @abc.abstractmethod
    async def authenticate(self) -> None:
        """Acquire or refresh a usable session. Idempotent.

        Called once at the start of a run and at most once more after a 401.
        Raises `AuthError`, which is terminal -- the runtime never loops on auth.
        """

    async def aclose(self) -> None:
        """Release HTTP clients and provider sessions. Always called.

        Default is a no-op so a connector holding no resources need not implement
        it. `run()` calls it in a `finally`, including on the generator-close path
        that fires when a consumer stops iterating early.
        """
        return None

    @abc.abstractmethod
    def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Yield pages from `cursor` forward, oldest-first.

        Oldest-first is not stylistic. The watermark may only move forward, so a
        newest-first pager that dies mid-run would commit a watermark past
        records it never emitted, and they would never be fetched again.

        What is actually guaranteed, precisely:

        - **Within a page**, records are ordered oldest-first. Always.
        - **Across pages**, oldest-first is the rule, with one sanctioned
          exception: a connector whose provider only pages newest-first may yield
          in provider order *provided* it pins the watermark for the whole window
          and only advances it on the closing page. `connectors/news/news_api.py`
          is that case and says so in its docstring.

        The exception is safe because `run()` clamps every emitted cursor against
        the watermark the run started from (see `_guard_watermark`), so no page
        order can move progress backwards. It is called out here because a
        consumer reading the `EmittedBatch` stream as a strict timeline would
        otherwise be relying on something three of four connectors happen to
        provide rather than something the contract promises.

        No internal sleeping and no internal retry -- raise `TransientError` and
        let the runtime back off.
        """

    @abc.abstractmethod
    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one payload onto the canonical Signal (Design Doc §6).

        Return `None` to **drop** the record -- a deleted comment, an empty feed
        entry. Raise `NormalizationError` when the payload should have mapped and
        did not; that record goes to the DLQ and the run continues.

        The distinction matters: dropping is expected and counted separately,
        while failing to map is a defect. Conflating them buries real mapping
        bugs in a counter nobody reads.
        """

    # ------------------------------------------------------- customisation --

    def rate_limit_keys(self) -> Sequence[str]:
        """Buckets to acquire before every outbound call. All must succeed.

        Three scopes by default: the whole connector (protects a shared app-level
        quota), the individual account (stops one tenant starving others), and --
        for backfill -- a separate reduced bucket so a historical crawl cannot
        crowd out live sync (`docs/connector-spec.md` §5.1).
        """
        keys = [f"os:rl:{self.slug}", f"os:rl:{self.slug}:{self.ctx.account_id}"]
        if self.ctx.mode is SyncMode.BACKFILL:
            keys.append(f"os:rl:{self.slug}:backfill")
        return keys

    def host_rate_limit_key(self, url: str | None) -> str | None:
        """Per-host politeness bucket.

        Matters for RSS, where every feed is a different origin and a
        connector-wide limit would either throttle a thousand hosts as one or
        hammer a single small server.
        """
        if not url:
            return None
        netloc = urlsplit(url).netloc.lower()
        return f"os:rl:host:{netloc}" if netloc else None

    def parse_rate_limit(self, headers: Mapping[str, str]) -> RateLimitHint | None:
        """Translate provider headers into a hint.

        Handles both `Retry-After` forms -- delta-seconds and HTTP-date -- because
        providers use both and the date form is the one that silently parses as
        `None` if you only wrote `int()`.
        """
        lowered = {k.lower(): v for k, v in headers.items()}
        hint = RateLimitHint(
            remaining=_as_int(lowered.get("x-ratelimit-remaining")),
            limit=_as_int(lowered.get("x-ratelimit-limit")),
            reset_at=_as_float(lowered.get("x-ratelimit-reset")),
            retry_after_seconds=_parse_retry_after(lowered.get("retry-after")),
        )
        if all(
            v is None
            for v in (hint.remaining, hint.limit, hint.reset_at, hint.retry_after_seconds)
        ):
            return None
        return hint

    def dedup_keys(self, signal: Signal) -> DedupKeys:
        """Identity, exact-content and near-duplicate keys for one Signal.

        Override only when the provider offers a better identity than the derived
        one -- a DOI for a paper, an ISBN for a book.

        The content hash is taken over the *cleaned* text, so a provider that
        re-serializes its own HTML differently between requests does not present
        as a new record every poll.
        """
        content = signal.content.text.strip()
        return DedupKeys(
            identity=f"os:dedup:id:{self.slug}:{signal.id}",
            content=(
                f"os:dedup:sha:{self.slug}:"
                f"{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
                if content
                else None
            ),
            simhash=None,
        )

    async def health_check(self) -> HealthReport:
        """One cheap authenticated no-op call. Surfaced on `GET /health`.

        Default reports healthy only if `authenticate()` succeeds, which is the
        weakest useful signal. Connectors with a real ping endpoint should
        override.
        """
        try:
            await self.authenticate()
        except Exception as exc:  # noqa: BLE001 -- health never raises
            return HealthReport(
                healthy=False,
                connector_slug=self.slug,
                account_id=self.ctx.account_id,
                detail=type(exc).__name__,
            )
        finally:
            await self.aclose()
        return HealthReport(
            healthy=True, connector_slug=self.slug, account_id=self.ctx.account_id
        )

    # ------------------------------------------------------------ template --

    @final
    async def run(self, cursor: Cursor | None = None) -> AsyncIterator[EmittedBatch]:
        """Drive the six stages and yield one batch per page.

        `@final`: the stage order is the contract. A connector that needed to
        reorder these would be telling us the contract is wrong, and that is a
        change to this file and the spec, not an override.

        Yields rather than returns so the runtime can persist and commit
        incrementally. A connector that accumulated everything and returned once
        would lose an entire multi-hour backfill to a single crash, and would
        hold it all in memory meanwhile.
        """
        floor = self._watermark_floor(cursor)
        start = self._effective_start(cursor)
        stats = {"fetched": 0, "emitted": 0, "dropped": 0, "duplicates": 0, "dlq": 0, "pages": 0}

        await self.authenticate()
        try:
            async for page in self.fetch(start):
                stats["pages"] += 1
                stats["fetched"] += len(page.records)

                if page.raw_headers:
                    hint = self.parse_rate_limit(page.raw_headers)
                    if hint is not None and self.ctx.limiter is not None:
                        # Provider truth beats local estimate: if it says three
                        # requests remain and the bucket believes forty, the
                        # bucket is clamped down.
                        await self.ctx.limiter.observe(self.rate_limit_keys(), hint)

                emitted = await self._normalize_and_dedup(page.records, stats)
                yield EmittedBatch(
                    records=emitted,
                    cursor=self._guard_watermark(page.cursor, floor),
                    stats=dict(stats),
                )

                if self._budget_exhausted(stats):
                    break
        finally:
            # Also runs when a consumer abandons the generator part-way, which is
            # exactly when an unclosed HTTP client would leak.
            await self.aclose()

    @final
    async def acquire_slot(self, url: str | None = None) -> None:
        """Acquire every applicable rate-limit token before one outbound call.

        `fetch()` implementations call this immediately before each request. It
        is here rather than inline in `run()` because pagination is the
        connector's business and only it knows how many calls a page costs.

        A missing limiter is not an error: `docs/architecture.md` §7.3 says
        outbound limiting fails **open** with conservative static limits when
        Redis is down, because keeping ingestion alive matters more than perfect
        pacing -- and because failing closed here would stop ingestion entirely
        every time the cache blipped.
        """
        if self.ctx.limiter is None:
            return
        keys = list(self.rate_limit_keys())
        host_key = self.host_rate_limit_key(url)
        if host_key:
            keys.append(host_key)
        await self.ctx.limiter.acquire(keys, timeout_seconds=self.ctx.request_timeout_seconds)

    # ------------------------------------------------------------ internals --

    def _watermark_floor(self, cursor: Cursor | None) -> datetime | None:
        """The watermark a run must never commit below: the one it started from.

        Captured *before* `_effective_start` applies the overlap shift, because
        that shift is the thing this guards against.
        """
        if cursor is None or not cursor.is_readable():
            return None
        return cursor.watermark

    def _guard_watermark(self, cursor: Cursor, floor: datetime | None) -> Cursor:
        """Stop a run from committing a watermark earlier than it started with.

        This exists because `_effective_start` deliberately hands `fetch()` a
        cursor rewound by `overlap_seconds`, and the obvious connector
        implementation carries that rewound cursor straight back out -- on a 304,
        on a truncated page budget, on any path that has no newer record to
        report. `Cursor.advanced_to` only clamps *upward*, so it cannot repair a
        watermark that arrived already low.

        The failure is not a one-off five-minute slip, it compounds: each poll
        rewinds from the previously-rewound value, so an idle feed walks its
        watermark backwards indefinitely and eventually re-fetches its whole
        history on every cycle. A conditional GET returning 304 is the *normal*
        steady state for RSS, which is precisely when it bites.

        Enforced here rather than in each connector because `run()` is the only
        place that sees both the original cursor and the emitted one, and because
        `docs/connector-spec.md` §4.1 rule 2 makes monotonicity a property of the
        contract -- not something four connector authors should each remember.
        A `None` watermark is also raised to the floor: regressing from a real
        timestamp to "no watermark" would trigger a full re-sync.
        """
        if floor is None:
            return cursor
        if cursor.watermark is not None and cursor.watermark >= floor:
            return cursor
        return Cursor(
            version=cursor.version,
            watermark=floor,
            page_token=cursor.page_token,
            checkpoint=cursor.checkpoint,
        )

    def _effective_start(self, cursor: Cursor | None) -> Cursor:
        """Apply the overlap window and reject a cursor from a future shape."""
        if cursor is None or cursor.is_empty:
            return Cursor()
        if not cursor.is_readable():
            # Unknown shape: start a bounded re-sync rather than misinterpret it.
            return Cursor()
        if cursor.watermark is None:
            return cursor
        return Cursor(
            version=cursor.version,
            watermark=_shift_back(cursor.watermark, self.overlap_seconds),
            page_token=cursor.page_token,
            checkpoint=cursor.checkpoint,
        )

    async def _normalize_and_dedup(
        self, records: Sequence[RawRecord], stats: dict[str, int]
    ) -> list[tuple[RawRecord, Signal]]:
        """Stages 4 and 5 for one page.

        A `NormalizationError` costs one record, not the page: it is counted for
        the DLQ and the loop continues. Aborting the page would let one malformed
        item block every well-formed item behind it, permanently, since the cursor
        would never advance past it.
        """
        emitted: list[tuple[RawRecord, Signal]] = []
        for record in records:
            try:
                signal = await self.normalize(record)
            except NormalizationError:
                stats["dlq"] += 1
                continue

            if signal is None:
                stats["dropped"] += 1
                continue

            if await self._is_duplicate(signal):
                stats["duplicates"] += 1
                continue

            emitted.append((record, signal))
            stats["emitted"] += 1
        return emitted

    async def _is_duplicate(self, signal: Signal) -> bool:
        """Check the identity and exact-content layers.

        Near-duplicate (SimHash) clustering is deliberately absent here: those are
        *clustered, not dropped* (`docs/signal-model.md` §4.3), because six copies
        of a press release across six platforms is evidence of spread and
        deleting five would destroy both the trend signal and the corroboration
        term in confidence. Clustering happens downstream, with the full corpus
        in view.
        """
        if self.ctx.dedup is None:
            return False
        keys = self.dedup_keys(signal)
        ttl = self.ctx.dedup_ttl_seconds
        for key in (keys.identity, keys.content):
            if key is None:
                continue
            if await self.ctx.dedup.seen(key):
                return True
        for key in (keys.identity, keys.content):
            if key is not None:
                await self.ctx.dedup.mark(key, ttl)
        return False

    def _budget_exhausted(self, stats: Mapping[str, int]) -> bool:
        """Whether this run has hit its page or record ceiling.

        Bounds exist so one enormous backfill cannot monopolize a worker
        indefinitely; the cursor is committed, so the next run resumes exactly
        where this one stopped.
        """
        if self.ctx.max_pages is not None and stats["pages"] >= self.ctx.max_pages:
            return True
        if self.ctx.max_records is not None and stats["emitted"] >= self.ctx.max_records:
            return True
        return False


# --------------------------------------------------------------------------- #
# Header parsing helpers
# --------------------------------------------------------------------------- #


def _as_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _as_float(value: str | None) -> float | None:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _parse_retry_after(value: str | None) -> float | None:
    """Parse `Retry-After` in either RFC 9110 form.

    Both are legal and providers use both. Handling only the numeric form means
    the date form parses to `None` and the backoff silently falls back to jitter,
    ignoring an explicit instruction from the provider -- which is how an
    integration earns a ban.
    """
    if not value:
        return None
    numeric = _as_float(value)
    if numeric is not None:
        return max(0.0, numeric)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return max(0.0, (when - utcnow()).total_seconds())


def _shift_back(moment: datetime, seconds: int) -> datetime:
    return moment - timedelta(seconds=seconds)
