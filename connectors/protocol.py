"""The types that cross the connector boundary, and the ports a connector needs.

Two kinds of thing live here.

**Value types** -- `RawRecord`, `Cursor`, `FetchPage`, `EmittedBatch`,
`SyncResult` and friends. All frozen dataclasses with `slots=True`: they are
passed between a connector and the runtime on every page, and immutability means
a connector cannot mutate a cursor the runtime is about to persist.

**Ports** -- `RateLimiter` and `DedupStore`, declared as `Protocol`s. This is the
mechanism that makes `docs/architecture.md` §6.2 rule 2 workable. A connector
needs a *shared* Redis token bucket and a *shared* seen-set, but it may not
import `backend/db/redis.py` to get them. So it receives them on `SyncContext`
as structural types it never imports an implementation of. The production wiring
lives in `services/connector_service.py`; a test passes an in-memory fake and
needs no Redis at all.

Note what a connector is *not* given: no database session, no Kafka producer, no
object store. It cannot write anywhere, which is what makes a run replayable
(`docs/connector-spec.md` §1).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from models.base import utcnow
from models.signal import Signal

__all__ = [
    "CURSOR_VERSION",
    "Credentials",
    "Cursor",
    "DedupKeys",
    "DedupStore",
    "EmittedBatch",
    "FetchPage",
    "HealthReport",
    "RateLimitHint",
    "RateLimitPolicy",
    "RateLimiter",
    "RawRecord",
    "SyncContext",
    "SyncMode",
    "SyncResult",
]

CURSOR_VERSION = 1
"""Shape version of `Cursor.checkpoint`.

The runtime treats an unknown version as "no cursor" and starts a bounded
re-sync, rather than handing a connector state it cannot interpret
(`docs/connector-spec.md` §4.1 rule 6). Bump it when the checkpoint shape changes.
"""


class SyncMode(enum.StrEnum):
    """Whether this run is following the live edge or crawling history.

    Backfill runs against a *separate* cursor row and a reduced rate-limit budget
    (25% of the connector's), so a long historical crawl can never clobber the
    live watermark or starve incremental sync (`docs/connector-spec.md` §4.1
    rule 5, §5.1).
    """

    INCREMENTAL = "incremental"
    BACKFILL = "backfill"


# --------------------------------------------------------------------------- #
# Credentials and policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Credentials:
    """Decrypted credentials for one connector account.

    Constructed by `services/connector_service.py` after Fernet-decrypting the
    ciphertext in `models/orm/connector_account.py`. It exists only in memory,
    only for the duration of a run.

    `__repr__` is overridden because a dataclass would otherwise print every
    secret it holds -- and a `ConnectorError` carrying this object in `details`
    would render straight into a log line.
    """

    account_id: str
    secrets: Mapping[str, str] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def require(self, key: str) -> str:
        """Fetch a secret, raising a clear error when it is absent.

        Better than a `KeyError` from somewhere inside an auth flow: this names
        the connector account and the missing key, which is what an operator
        needs in order to fix it.
        """
        value = self.secrets.get(key)
        if not value:
            raise KeyError(
                f"credential {key!r} is missing for account {self.account_id!r}; "
                "check the connector account configuration"
            )
        return value

    def __repr__(self) -> str:
        return f"Credentials(account_id={self.account_id!r}, secrets=<{len(self.secrets)} redacted>)"


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    """A connector's declared request budget.

    Declared as a `ClassVar` on the connector class so the scheduler can plan a
    run *before* instantiating anything.
    """

    requests_per_minute: int = 60
    burst: int = 10
    concurrency: int = 2
    backfill_fraction: float = 0.25
    """Share of the budget a backfill run may consume. §5.1."""

    def for_mode(self, mode: SyncMode) -> int:
        """Effective per-minute budget for a run in this mode."""
        if mode is SyncMode.BACKFILL:
            return max(1, int(self.requests_per_minute * self.backfill_fraction))
        return self.requests_per_minute


@dataclass(frozen=True, slots=True)
class RateLimitHint:
    """What the provider said about our remaining budget.

    Parsed from response headers by `BaseConnector.parse_rate_limit` and fed back
    into the bucket. Provider truth beats local estimate: if the provider reports
    three requests remaining and the bucket believes forty, the bucket is clamped
    down (`docs/connector-spec.md` §5.2).
    """

    remaining: int | None = None
    limit: int | None = None
    reset_at: float | None = None
    retry_after_seconds: float | None = None


@runtime_checkable
class RateLimiter(Protocol):
    """Shared, cross-process token bucket. Implemented over Redis in `services/`.

    A `Protocol` rather than an import: the limiter must be shared across every
    worker replica, which means Redis, which a connector may not import.
    """

    async def acquire(
        self, keys: Sequence[str], *, timeout_seconds: float | None = None
    ) -> None:
        """Acquire one token from every key, or raise `QuotaError`.

        All-or-nothing: partial acquisition leaks tokens from the buckets that
        did succeed, which over a long run silently tightens the effective limit.
        """
        ...

    async def observe(self, keys: Sequence[str], hint: RateLimitHint) -> None:
        """Reconcile the bucket against what the provider actually reported."""
        ...


@runtime_checkable
class DedupStore(Protocol):
    """Shared seen-set with TTL. Implemented over Redis in `services/`."""

    async def seen(self, key: str) -> bool:
        """Whether this key has been observed inside the TTL window."""
        ...

    async def mark(self, key: str, ttl_seconds: int) -> None:
        """Record a key as seen."""
        ...


# --------------------------------------------------------------------------- #
# Run state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Cursor:
    """Resume state for one (connector, account, params) triple.

    Opaque to the runtime apart from `watermark` and `version`; everything else
    is the connector's private business (`docs/connector-spec.md` §4).
    """

    version: int = CURSOR_VERSION
    watermark: datetime | None = None
    """Event time of the newest record durably emitted. Must move forward only."""

    page_token: str | None = None
    """Provider pagination token, advisory. A connector that finds it rejected
    falls back to `watermark` and re-pages rather than failing the run."""

    checkpoint: Mapping[str, Any] = field(default_factory=dict)
    """ETags, per-feed offsets, anything else the connector needs."""

    @property
    def is_empty(self) -> bool:
        """Whether this is a first run."""
        return self.watermark is None and self.page_token is None and not self.checkpoint

    def is_readable(self) -> bool:
        """Whether this process understands the cursor's shape.

        An unknown version means a bounded re-sync, never a misinterpretation of
        old state.
        """
        return self.version == CURSOR_VERSION

    def advanced_to(
        self,
        *,
        watermark: datetime | None = None,
        page_token: str | None = None,
        **checkpoint: Any,
    ) -> Cursor:
        """Return a new cursor moved forward. Never moves the watermark backwards.

        Silently clamping rather than raising: a provider that pages newest-first
        without re-sorting would otherwise fail every run, and the runtime's own
        monotonicity check is the place that surfaces the problem loudly.
        """
        new_watermark = self.watermark
        if watermark is not None and (self.watermark is None or watermark > self.watermark):
            new_watermark = watermark
        return Cursor(
            version=CURSOR_VERSION,
            watermark=new_watermark,
            page_token=page_token,
            checkpoint={**self.checkpoint, **checkpoint},
        )


@dataclass(frozen=True, slots=True)
class SyncContext:
    """Everything a connector needs that is not a credential.

    Assembled by `services/connector_service.py`. The `limiter` and `dedup` ports
    arrive here so the connector never imports an implementation of either.
    """

    connector_slug: str
    account_id: str
    run_id: str
    mode: SyncMode = SyncMode.INCREMENTAL
    params: Mapping[str, Any] = field(default_factory=dict)
    """Per-account configuration: which subreddits, which feed URLs."""

    params_hash: str = ""
    limiter: RateLimiter | None = None
    dedup: DedupStore | None = None
    max_pages: int | None = None
    max_records: int | None = None
    dedup_ttl_seconds: int = 604_800
    request_timeout_seconds: float = 30.0
    user_agent: str = "omnisense/0.1"

    def rate_limit_budget(self, policy: RateLimitPolicy) -> int:
        return policy.for_mode(self.mode)


@dataclass(frozen=True, slots=True)
class RawRecord:
    """One provider payload, before any interpretation.

    `payload` is kept verbatim so that a mapping bug is repairable by
    reprocessing rather than re-fetching -- re-fetching is lossy because posts get
    deleted and API windows expire (`docs/signal-model.md` §3.2).

    `raw_bytes` is the exact bytes the provider returned, when the connector has
    them. It is what gets PUT to R2 by the runtime and what `raw_sha256` is taken
    over; re-serializing `payload` would produce a different digest on a
    different json library version and break content-addressed keys.
    """

    native_id: str
    payload: Mapping[str, Any]
    fetched_at: datetime = field(default_factory=utcnow)
    raw_bytes: bytes | None = None
    content_type: str = "application/json"
    source_url: str | None = None
    request_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class FetchPage:
    """One page of provider results, plus the cursor that would resume after it.

    The cursor travels with the page rather than being tracked separately so that
    "these records are durable" and "this is where to resume" cannot drift apart.
    """

    records: Sequence[RawRecord]
    cursor: Cursor
    raw_headers: Mapping[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class DedupKeys:
    """The three identities a record is checked against (`docs/signal-model.md` §4.2).

    Layer 1 (`identity`) collapses a re-fetch of the same item. Layer 2
    (`content`) drops an exact repost within the TTL window. Layer 3 (`simhash`)
    finds near-duplicates *across* platforms -- and those are clustered, never
    dropped, because six copies of a press release is evidence of spread.
    """

    identity: str
    content: str | None = None
    simhash: int | None = None


@dataclass(frozen=True, slots=True)
class EmittedBatch:
    """Records that survived normalize and dedup, ready for the runtime to persist.

    The connector never touches R2, never produces to Kafka and never commits a
    cursor: `services/connector_service.py` does all three, in that order, for
    each yielded batch (`docs/connector-spec.md` §2.6). Commit-after-ack is what
    makes a crash duplicate records rather than lose them, and duplicates are
    cheap because of dedup.
    """

    records: Sequence[tuple[RawRecord, Signal]]
    cursor: Cursor
    stats: Mapping[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.records)


@dataclass(frozen=True, slots=True)
class SyncResult:
    """The outcome of one connector run, assembled by the runtime."""

    run_id: str
    connector_slug: str
    account_id: str
    fetched: int = 0
    emitted: int = 0
    dropped: int = 0
    duplicates: int = 0
    dlq: int = 0
    pages: int = 0
    cursor: Cursor = field(default_factory=Cursor)
    started_at: datetime = field(default_factory=utcnow)
    ended_at: datetime | None = None
    error: str | None = None
    error_class: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def is_partial(self) -> bool:
        """A run that hit a quota after emitting real work.

        Distinguished from failure because the cursor was committed and the
        records are durable -- rescheduling continues rather than restarts.
        """
        return self.error_class == "quota" and self.emitted > 0


@dataclass(frozen=True, slots=True)
class HealthReport:
    """Result of one cheap authenticated no-op call, surfaced on `GET /health`."""

    healthy: bool
    connector_slug: str
    account_id: str | None = None
    latency_ms: float | None = None
    detail: str | None = None
    checked_at: datetime = field(default_factory=utcnow)
