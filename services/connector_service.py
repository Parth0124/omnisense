"""The ingestion runtime: where a connector's output becomes durable, in order.

`connectors/` is deliberately powerless. A connector holds no session, no
producer and no object store (`connectors/protocol.py`), so a run is replayable
by construction and a connector is testable with `respx` and two in-memory
fakes. Everything a connector *cannot* do is done here, and the order in which it
is done is the single most important property in the ingestion path.

    for every EmittedBatch yielded by BaseConnector.run():
        1. PUT the raw payload to R2          services/storage/object_store.py
        2. publish RawRecordEvent to Kafka    services/events/producer.py
        3. await the broker's ack, and only then commit the cursor

**Commit after ack, never before** (`docs/connector-spec.md` §4.1 rule 1). The
asymmetry is the whole argument. A cursor committed before the producer
acknowledges silently *loses* records: the next run resumes past a page that was
never durably written, nothing errors, nothing lands in a DLQ, and the only
evidence is a hole in a time series that nobody can explain months later. The
reverse -- acking and then dying before the commit -- merely replays the page,
and a replay is cheap: the R2 key is content-addressed so the `PUT` is a no-op,
`Signal.id` is derived from `(platform, native_id)` so the upsert converges, and
the dedup keys absorb the rest. One direction is unrecoverable, the other is
routine, so the code is written so that the unrecoverable direction is not
expressible: `_persist_batch` performs the commit as its last statement and holds
no `except` that could reach it.

That guarantee rests on `services/events/producer.py` running with `acks="all"`
and `enable_idempotence=True`, which is why neither is a tunable there. Under
`acks=1` an "ack" would only mean the partition leader wrote to its own log, and
committing a cursor on the strength of it would reintroduce exactly the silent
loss above.

Four other decisions are encoded below.

**The watermark may only move forward.** §4.1 rule 2 makes monotonicity the
runtime's job. `connectors/base.py::_guard_watermark` already clamps every
emitted cursor, so a regression seen *here* means the clamp was bypassed -- a
connector overriding `run()`, or a caller driving `_persist_batch` directly. It
is therefore logged at error level with both watermarks and the commit is
dropped, rather than being quietly clamped a second time: a silent second clamp
would make the first one's failure undiscoverable, and a runtime that re-fetches
its whole history every poll is a bug worth waking someone for.

**The error taxonomy decides everything, and it is read off the class.**
`connectors/exceptions.py` fixes four outcomes and two of them are easy to invert:
`QuotaError` is a *partial success* -- the emitted records stay emitted and the
cursor stays committed, because discarding an hour of successful pagination at
the 4,000th request is how a backfill never finishes -- while `AuthError` is
terminal with no partial credit and flags the account `needs_reauth`. Dispatch is
on `ConnectorError.error_class`, not on `isinstance`, so `CircuitOpenError` --
which is deliberately classified `QUOTA` -- gets the quota response without being
a `QuotaError` subclass.

**Anything that is not a `ConnectorError` propagates.** A `KeyError` in a mapper
is our defect, not a source misbehaving in a way we planned for. Folding it into
a `SyncResult` with `error_class="unknown"` would file a bug as a provider fault
and bury the traceback.

**R2 degrades, Kafka halts.** `docs/architecture.md` §7.3: R2 unavailable means
"raw payload archival deferred; enrichment continues", so an object-store failure
publishes the event with `raw_object_key=None` -- which `RawRecordEvent`
explicitly anticipates -- while a broker failure stops the run with the cursor
untouched, because there is no correct local buffer for a message that must be
durable.

Layer note: **L2 service** (`docs/architecture.md` §6.1). It may import `models/`,
`connectors/`, `backend/` and other services, and it is the only place that does
all four -- which is precisely why the wiring `connectors/` refuses to do lives
here. Nothing in this module performs I/O at import.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Protocol

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.config import Settings, get_settings
from backend.core.exceptions import (
    ConfigurationError,
    DependencyUnavailableError,
    ExternalServiceError,
)
from backend.core.logging import get_logger
from connectors.base import BaseConnector
from connectors.dedup.store import RedisDedupStore
from connectors.exceptions import AuthError, ConnectorError, PermanentError
from connectors.protocol import (
    Credentials,
    Cursor,
    EmittedBatch,
    RawRecord,
    SyncContext,
    SyncMode,
    SyncResult,
)
from connectors.ratelimit.backoff import DEFAULT_POLICY, BackoffPolicy, CircuitBreaker, retry_page
from connectors.ratelimit.limiter import BucketPolicy, TokenBucketLimiter
from models.base import utcnow
from models.enums import ConnectorErrorClass
from models.orm.connector_account import (
    ConnectorAccountRow,
    ConnectorAccountStatus,
    ConnectorCursorRow,
)
from models.orm.mixins import DEFAULT_TENANT
from models.signal import Signal
from services.events.schemas import DlqEvent, RawRecordEvent
from services.events.topics import TopicRole, topic_name
from services.storage.object_store import StoredObject, put_raw_payload

__all__ = [
    "RUNTIME_VERSION",
    "AccountWriter",
    "ConnectorRuntime",
    "CursorKey",
    "CursorStore",
    "DlqRecordPublisher",
    "InMemoryCursorStore",
    "RawArchiver",
    "RawRecordPublisher",
    "SqlAccountWriter",
    "SqlCursorStore",
    "archive_raw_payload",
    "build_sync_context",
    "credential_cipher",
    "decrypt_credentials",
    "publish_dlq_record",
    "publish_raw_record",
    "rate_limit_buckets",
    "sync_params_hash",
]

logger = get_logger(__name__)

RUNTIME_VERSION: Final = "1.0.0"
"""Version of this runtime's implementation. Recorded in run logs."""

USER_AGENT: Final = f"omnisense/{RUNTIME_VERSION} (+https://github.com/omnisense)"
"""What every connector identifies itself as. `SyncContext.user_agent`'s value.

Derived from the runtime version rather than from settings: this is how a
provider recognizes us when it decides whether to block or to rate-limit, and a
value an operator can vary per deployment would make "which of our deployments
is hammering you" unanswerable from the provider's side.
"""

_STAT_KEYS: Final[tuple[str, ...]] = ("fetched", "dropped", "duplicates", "dlq", "pages")
"""Counters `BaseConnector.run` reports. `emitted` is deliberately absent.

The connector's `emitted` counts records that survived normalize and dedup and
were handed over; this runtime's `emitted` counts records the broker acknowledged.
Those differ whenever a record is DLQ'd here, and the number a `SyncResult`
reports has to be the durable one -- `SyncResult.is_partial` reads it as "real
work survived", and a count including records that never reached Kafka would
reschedule a quota-halted run as a continuation of work that does not exist.
"""


# --------------------------------------------------------------------------- #
# Ports
# --------------------------------------------------------------------------- #


class RawArchiver(Protocol):
    """Step 1: put one payload in the archive and say where it went.

    A callable protocol rather than the module function, for the same reason
    `services/signal_engine/store.py` declares `SignalPublisher`: the unit suite
    must be able to prove the *ordering* of the three steps, and that requires
    substituting each one independently with something that has no bucket behind
    it.
    """

    async def __call__(
        self, platform: str, fetched_at: datetime, raw_bytes: bytes
    ) -> StoredObject: ...


class RawRecordPublisher(Protocol):
    """Step 2, and step 3's precondition: publish, and return only once acked.

    **Returning from this call is the ack.** The whole ordering guarantee of this
    module is delegated to that statement, so an implementation that buffers and
    returns early -- a fire-and-forget `send()`, a background flush -- silently
    breaks commit-after-ack for every connector at once. The production
    implementation is `publish_raw_record`, which goes through
    `services/events/producer.py`'s `send_and_wait` under `acks="all"`.
    """

    async def __call__(self, event: RawRecordEvent, *, tenant_id: str | None = None) -> None: ...


class DlqRecordPublisher(Protocol):
    """Park one record that can never be archived, and let the run continue.

    Separate from `RawRecordPublisher` rather than one publisher taking an
    `EventPayload`: they fail independently and must be substitutable
    independently. A test that injects a failing raw-record publisher would
    otherwise also disable the DLQ path it is not testing.
    """

    async def __call__(self, event: DlqEvent, *, tenant_id: str | None = None) -> None: ...


class CursorStore(Protocol):
    """Step 3: where resume state lives between runs.

    `commit` returns whether the write was applied, so a caller can tell "the
    watermark was rejected" from "the watermark was written" without reading the
    row back.
    """

    async def load(self, key: CursorKey) -> Cursor | None: ...

    async def commit(self, key: CursorKey, cursor: Cursor) -> bool: ...


class AccountWriter(Protocol):
    """The two account-row effects the error taxonomy requires.

    Optional on `ConnectorRuntime` because a `scripts/sync_connector.py` dry run
    drives a connector with no account row behind it; the runtime then logs the
    same facts and writes nothing.
    """

    async def mark_needs_reauth(self, account_id: str, *, reason: str) -> None: ...

    async def reschedule(self, account_id: str, *, when: datetime) -> None: ...


# --------------------------------------------------------------------------- #
# Production implementations of the ports
# --------------------------------------------------------------------------- #


async def archive_raw_payload(
    platform: str, fetched_at: datetime, raw_bytes: bytes
) -> StoredObject:
    """The production `RawArchiver`. Content-addressed, so a replay is a no-op."""
    return await put_raw_payload(platform, fetched_at, raw_bytes)


async def publish_raw_record(event: RawRecordEvent, *, tenant_id: str | None = None) -> None:
    """The production `RawRecordPublisher`: publish and wait for the broker.

    The import is function-local, matching `services/signal_engine/store.py`:
    `services/events/producer.py` owns a module-level client, and importing it at
    module scope would drag aiokafka into every process that merely imports this
    runtime -- including the unit suite at collection time.
    """
    from services.events.producer import publish

    await publish(event, tenant_id=tenant_id)


async def publish_dlq_record(event: DlqEvent, *, tenant_id: str | None = None) -> None:
    """The production `DlqRecordPublisher`. Routed to `omnisense.dlq` by event type."""
    from services.events.producer import publish

    await publish(event, tenant_id=tenant_id)


# --------------------------------------------------------------------------- #
# Cursor identity
# --------------------------------------------------------------------------- #


def sync_params_hash(params: Mapping[str, Any], mode: SyncMode) -> str:
    """SHA-256 over the canonicalized params *and the sync mode*.

    Folding the mode in is what reconciles two rules that otherwise contradict
    each other: `docs/connector-spec.md` §4 fixes the cursor key as
    `(connector_slug, account_id, params_hash)`, while §4.1 rule 5 requires a
    backfill to run against a *separate* cursor row. With the mode inside the
    hash, both hold with one key -- and a long historical crawl cannot clobber the
    live incremental watermark, which is the failure rule 5 exists to prevent.

    `sort_keys` plus `separators` makes the digest independent of dict ordering
    and of whatever whitespace a JSON encoder feels like emitting, so the same
    configuration hashes the same on every worker and across Python versions.
    """
    canonical = json.dumps(
        {"mode": mode.value, "params": params},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CursorKey:
    """The `(connector_slug, account_id, params_hash)` triple one cursor belongs to."""

    connector_slug: str
    account_id: str
    params_hash: str

    @property
    def row_id(self) -> str:
        """Deterministic primary key for `connector_cursors`.

        Derived rather than random so that a lost `INSERT ... RETURNING`, a
        retried commit, or two workers racing to create the first cursor for an
        account all address the same row instead of relying on the unique
        constraint to reject the loser.
        """
        digest = hashlib.sha256(
            f"{self.connector_slug}:{self.account_id}:{self.params_hash}".encode()
        ).hexdigest()
        return f"cur_{digest[:40]}"

    @classmethod
    def for_context(cls, connector_slug: str, ctx: SyncContext) -> CursorKey:
        """Derive the key from the run's own params and mode.

        `ctx.params_hash` is deliberately *not* trusted as the source of truth. It
        is a denormalization carried on the account row, and `POST
        /connectors/{slug}/sync` may narrow the targets for a single run
        (`models/orm/connector_account.py`), so a caller that forwarded the
        account's hash unchanged would write a narrowed run's progress onto the
        full run's watermark. Recomputing from `ctx.params` and `ctx.mode` makes
        both that and §4.1 rule 5 structural rather than remembered.
        """
        derived = sync_params_hash(ctx.params, ctx.mode)
        if ctx.params_hash and ctx.params_hash != derived:
            # Not fatal: the derived value is authoritative and correct. Logged
            # because a persistent mismatch means whoever computed the stored
            # hash used a different recipe, and every account is then carrying a
            # value that matches no cursor row.
            logger.warning(
                "connector.cursor.params_hash_mismatch",
                connector=connector_slug,
                account_id=ctx.account_id,
                supplied=ctx.params_hash,
                derived=derived,
            )
        return cls(connector_slug=connector_slug, account_id=ctx.account_id, params_hash=derived)


class SqlCursorStore:
    """`CursorStore` over `connector_cursors`. The production implementation.

    Written through the ORM rather than as a Core upsert, unlike
    `services/signal_engine/store.py`. The two have opposite concurrency shapes:
    Signals arrive from six partitions at once and genuinely need `ON CONFLICT`,
    while a cursor has exactly one writer -- the scheduler claims an account
    before a run starts -- so the simpler statement is the honest one, and it
    keeps `updated_at`'s client-side `onupdate` working (see
    `models/orm/mixins.py`, where the upsert path has to set it by hand).

    A second concurrent run of the same account is a scheduling defect, not a
    case to be absorbed here; the unique constraint on
    `(connector_slug, account_id, params_hash)` turns it into a visible
    `IntegrityError` rather than two watermarks fighting.
    """

    __slots__ = ("_session_factory", "_tenant_id")

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    async def load(self, key: CursorKey) -> Cursor | None:
        """Read resume state, or `None` for an account that has never synced."""
        async with self._session_factory() as session:
            row = await session.scalar(self._select(key))
            if row is None:
                return None
            blob = dict(row.cursor or {})
            checkpoint = blob.get("checkpoint")
            return Cursor(
                version=row.cursor_version,
                watermark=_as_utc(row.watermark),
                page_token=blob.get("page_token"),
                checkpoint=dict(checkpoint) if isinstance(checkpoint, dict) else {},
            )

    async def commit(self, key: CursorKey, cursor: Cursor) -> bool:
        """Persist resume state. Always returns `True` -- a rejected commit never
        reaches here, because `ConnectorRuntime` refuses it first."""
        async with self._session_factory() as session:
            row = await session.scalar(self._select(key))
            blob: dict[str, Any] = {
                # The blob is connector-private (`docs/connector-spec.md` §4):
                # stored and returned verbatim, never read into. A value the JSON
                # encoder cannot represent fails loudly here rather than being
                # coerced into something the connector will misread next run.
                "page_token": cursor.page_token,
                "checkpoint": dict(cursor.checkpoint),
            }
            if row is None:
                session.add(
                    ConnectorCursorRow(
                        id=key.row_id,
                        tenant_id=self._tenant_id,
                        connector_slug=key.connector_slug,
                        account_id=key.account_id,
                        params_hash=key.params_hash,
                        cursor=blob,
                        cursor_version=cursor.version,
                        watermark=cursor.watermark,
                    )
                )
            else:
                row.cursor = blob
                row.cursor_version = cursor.version
                row.watermark = cursor.watermark
            await session.commit()
        return True

    def _select(self, key: CursorKey) -> Any:
        return select(ConnectorCursorRow).where(
            ConnectorCursorRow.connector_slug == key.connector_slug,
            ConnectorCursorRow.account_id == key.account_id,
            ConnectorCursorRow.params_hash == key.params_hash,
            ConnectorCursorRow.tenant_id == self._tenant_id,
        )


class InMemoryCursorStore:
    """`CursorStore` in a dict. For `--dry-run` and for tests that are not about SQL.

    Not a production store for the obvious reason: eight replicas would hold eight
    private ideas of where sync had got to, and every restart would re-crawl from
    the beginning.
    """

    __slots__ = ("commits", "cursors")

    def __init__(self, cursors: Mapping[CursorKey, Cursor] | None = None) -> None:
        self.cursors: dict[CursorKey, Cursor] = dict(cursors or {})
        self.commits: list[tuple[CursorKey, Cursor]] = []
        """Every commit in order, so a test can assert on *when* one happened."""

    async def load(self, key: CursorKey) -> Cursor | None:
        return self.cursors.get(key)

    async def commit(self, key: CursorKey, cursor: Cursor) -> bool:
        self.cursors[key] = cursor
        self.commits.append((key, cursor))
        return True


class SqlAccountWriter:
    """`AccountWriter` over `connector_accounts`.

    Both writes are deliberately narrow. `mark_needs_reauth` sets `status` and
    leaves `enabled` alone: those two columns mean different things
    (`models/orm/connector_account.py` -- intent versus observation), and
    disabling the account here would make an expired token indistinguishable from
    a deliberate pause, which is exactly the distinction the reauthentication
    prompt needs.
    """

    __slots__ = ("_session_factory", "_tenant_id")

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    async def mark_needs_reauth(self, account_id: str, *, reason: str) -> None:
        async with self._session_factory() as session:
            row = await self._get(session, account_id)
            if row is None:
                return
            row.status = ConnectorAccountStatus.NEEDS_REAUTH
            # `ConnectorError` messages are authored by us and carry no response
            # body, headers or credentials (`connectors/exceptions.py`), which is
            # what makes this column safe to fill from one.
            row.last_error = reason
            row.consecutive_failures += 1
            await session.commit()

    async def reschedule(self, account_id: str, *, when: datetime) -> None:
        async with self._session_factory() as session:
            row = await self._get(session, account_id)
            if row is None:
                return
            row.next_sync_at = when
            await session.commit()

    async def _get(self, session: AsyncSession, account_id: str) -> ConnectorAccountRow | None:
        row: ConnectorAccountRow | None = await session.scalar(
            select(ConnectorAccountRow).where(
                ConnectorAccountRow.id == account_id,
                ConnectorAccountRow.tenant_id == self._tenant_id,
            )
        )
        return row


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


def credential_cipher(settings: Settings | None = None) -> Fernet:
    """Build the Fernet cipher from `CREDENTIAL_ENCRYPTION_KEY`.

    The key is validated before use rather than at the first `decrypt`. A
    malformed or placeholder key raises `ConfigurationError` -- an operator
    problem, reported as one -- whereas letting it through would surface as
    `InvalidToken` on a real account and be misread as a revoked credential,
    flagging every account in the deployment `needs_reauth` over one wrong
    environment variable.
    """
    resolved = settings or get_settings()
    if not resolved.security.fernet_key_is_wellformed():
        raise ConfigurationError(
            "CREDENTIAL_ENCRYPTION_KEY is not a valid Fernet key (expected 44 "
            "url-safe base64 characters). Generate one with: python -c "
            '"from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())"'
        )
    return Fernet(resolved.security.credential_encryption_key.get_secret_value().encode("ascii"))


def decrypt_credentials(
    account_id: str,
    ciphertext: bytes | None,
    *,
    extra: Mapping[str, Any] | None = None,
    settings: Settings | None = None,
) -> Credentials:
    """Decrypt one account's stored credentials into an in-memory `Credentials`.

    This is the sole decrypt point (`docs/connector-spec.md` §8.1): plaintext
    exists here and on the `SyncContext`'s owner for the duration of one run, and
    nowhere else. Python cannot zero the buffer afterwards, which is why the
    object renders redacted and is never logged.

    A missing ciphertext is not an error. `auth_type=none` connectors (RSS) never
    have one, and an OAuth account legitimately exists with no credential between
    creation and consent, so the caller gets empty secrets and
    `Credentials.require` produces the message naming the account and the key.

    An `InvalidToken` *is* an error, and specifically an `AuthError`: under a
    well-formed key it means this row's ciphertext cannot be opened -- a rotated
    key the row was never re-encrypted under, or corruption -- and the account
    genuinely needs reauthentication. The well-formedness check in
    `credential_cipher` is what keeps a global misconfiguration out of this
    branch.
    """
    if not ciphertext:
        return Credentials(account_id=account_id, secrets={}, extra=dict(extra or {}))

    cipher = credential_cipher(settings)
    try:
        plaintext = cipher.decrypt(bytes(ciphertext))
    except InvalidToken as err:
        raise AuthError(
            f"stored credentials for account {account_id!r} could not be decrypted "
            "under the current CREDENTIAL_ENCRYPTION_KEY",
            account_id=account_id,
        ) from err

    try:
        payload = json.loads(plaintext)
    except json.JSONDecodeError as err:
        raise AuthError(
            f"decrypted credentials for account {account_id!r} are not JSON",
            account_id=account_id,
        ) from err
    if not isinstance(payload, dict):
        raise AuthError(
            f"decrypted credentials for account {account_id!r} are not a JSON object",
            account_id=account_id,
        )

    secrets = {str(k): str(v) for k, v in payload.items() if k != "extra"}
    stored_extra = payload.get("extra")
    merged: dict[str, Any] = dict(stored_extra) if isinstance(stored_extra, dict) else {}
    merged.update(extra or {})
    return Credentials(account_id=account_id, secrets=secrets, extra=merged)


# --------------------------------------------------------------------------- #
# Sync context assembly
# --------------------------------------------------------------------------- #


def rate_limit_buckets(
    connector: type[BaseConnector], account_id: str, mode: SyncMode
) -> dict[str, BucketPolicy]:
    """Per-key bucket policies for the scopes `BaseConnector.rate_limit_keys` names.

    Keys are spelled to match that method exactly, because the limiter resolves
    policies by exact match (`connectors/ratelimit/limiter.py::_PolicyResolver`):
    a key that does not match falls back to the conservative per-host default,
    which would silently throttle a connector to 60 rpm no matter what it
    declared. The backfill bucket is always registered at the reduced rate, not
    only during a backfill run, so the key exists at its correct rate whichever
    mode observes it first (`docs/connector-spec.md` §5.1).
    """
    policy = connector.rate_limit
    return {
        f"os:rl:{connector.slug}": BucketPolicy.from_rate_limit_policy(policy, mode),
        f"os:rl:{connector.slug}:{account_id}": BucketPolicy.from_rate_limit_policy(policy, mode),
        f"os:rl:{connector.slug}:backfill": BucketPolicy.from_rate_limit_policy(
            policy, SyncMode.BACKFILL
        ),
    }


def build_sync_context(
    connector: type[BaseConnector],
    *,
    account_id: str,
    params: Mapping[str, Any] | None = None,
    mode: SyncMode = SyncMode.INCREMENTAL,
    run_id: str | None = None,
    redis: Any | None = None,
    settings: Settings | None = None,
    max_pages: int | None = None,
    max_records: int | None = None,
) -> SyncContext:
    """Assemble the context a connector runs under, with the *shared* ports bound.

    Takes the connector class rather than a slug and a `RateLimitPolicy` because
    both come off the class; passing them separately is what lets a caller pace
    one connector with another's budget.

    The limiter and the dedup store are the Redis-backed implementations, and
    that is the point of this function. Both must be shared across every worker
    replica -- N replicas each enforcing the full rate permit N times the rate,
    which is the provider ban the limiter exists to prevent, and N private
    seen-sets deduplicate nothing across partitions. `connectors/` may not import
    `backend/db/redis.py` (`docs/architecture.md` §6.2 rule 2), so the client is
    obtained here and injected as structural types the connector never names.

    `redis` is injectable so a test, or a caller that already holds a client, need
    not touch the process-wide singleton.
    """
    resolved = settings or get_settings()
    client = redis if redis is not None else _redis_client()
    effective_params = dict(params or {})

    return SyncContext(
        connector_slug=connector.slug,
        account_id=account_id,
        run_id=run_id or f"run_{uuid.uuid4().hex}",
        mode=mode,
        params=effective_params,
        params_hash=sync_params_hash(effective_params, mode),
        limiter=TokenBucketLimiter(
            client,
            policies=rate_limit_buckets(connector, account_id, mode),
        ),
        dedup=RedisDedupStore(client, default_ttl_seconds=resolved.connectors.dedup_ttl_seconds),
        max_pages=max_pages,
        max_records=max_records,
        dedup_ttl_seconds=resolved.connectors.dedup_ttl_seconds,
        request_timeout_seconds=float(resolved.connectors.request_timeout_seconds),
        # Identifies us to every provider we call. A default `python-httpx/x.y`
        # is what gets an integration blocked by a WAF with no way to ask for an
        # exemption, because nothing in the request says who to exempt.
        user_agent=USER_AGENT,
    )


def _redis_client() -> Any:
    """Fetch the process-wide Redis client, imported lazily.

    Function-local for the same reason `publish_raw_record`'s import is: importing
    `backend/db/redis.py` at module scope would make merely importing this runtime
    construct a client, and the unit suite imports it at collection time.
    """
    from backend.db.redis import get_redis

    return get_redis()


# --------------------------------------------------------------------------- #
# Run state
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _RunState:
    """Mutable bookkeeping for one run. Local to `ConnectorRuntime.run`.

    Deliberately not instance state on the runtime: one runtime is shared by a
    worker and may drive several accounts concurrently, so everything about one
    run lives here and dies with the call.
    """

    cursor: Cursor
    """The last cursor *durably committed*. What a retry resumes from, and what
    the `SyncResult` reports -- never the cursor of a batch still in flight."""

    started_at: datetime = field(default_factory=utcnow)
    emitted: int = 0
    totals: dict[str, int] = field(default_factory=lambda: dict.fromkeys(_STAT_KEYS, 0))
    attempt_stats: dict[str, int] = field(default_factory=dict)

    def begin_attempt(self) -> None:
        """Reset the per-attempt baseline.

        `BaseConnector.run` restarts its counters at zero on every call, so after
        a transient retry the incoming stats are a fresh cumulative series. Without
        this reset the first batch of attempt two would be read as a negative
        delta against attempt one's totals.
        """
        self.attempt_stats = {}

    def absorb(self, stats: Mapping[str, int]) -> None:
        """Fold one batch's cumulative counters into the run totals.

        Counters arrive cumulative *within an attempt*, so only the delta is
        added. Across attempts the totals genuinely double-count a re-fetched
        page, and that is correct: the provider really was asked twice, and a
        `fetched` count that hid the retry would make the rate-limit budget
        unexplainable.
        """
        for name in _STAT_KEYS:
            current = stats.get(name, 0)
            self.totals[name] += current - self.attempt_stats.get(name, 0)
        self.attempt_stats = dict(stats)


# --------------------------------------------------------------------------- #
# The runtime
# --------------------------------------------------------------------------- #


class ConnectorRuntime:
    """Drives one `BaseConnector` and makes its output durable, in order.

    Stateless per run -- see `_RunState`. Construct one per process and share it.

    Every collaborator is injectable and every default is the production one, so
    the unit suite substitutes each of the three steps independently and can
    prove the ordering rather than assuming it.
    """

    def __init__(
        self,
        cursors: CursorStore,
        *,
        publisher: RawRecordPublisher | None = None,
        archiver: RawArchiver | None = None,
        dlq_publisher: DlqRecordPublisher | None = None,
        accounts: AccountWriter | None = None,
        tenant_id: str = DEFAULT_TENANT,
        backoff: BackoffPolicy = DEFAULT_POLICY,
        breaker: CircuitBreaker | None = None,
        rng: random.Random | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._cursors = cursors
        self._publish = publisher or publish_raw_record
        self._archive = archiver or archive_raw_payload
        self._publish_dlq = dlq_publisher or publish_dlq_record
        self._accounts = accounts
        self._tenant_id = tenant_id
        self._backoff = backoff
        # In-process by design (`connectors/ratelimit/backoff.py`): a breaker that
        # failed closed because Redis was down would be a self-inflicted outage.
        self._breaker = breaker if breaker is not None else CircuitBreaker()
        self._rng = rng
        self._sleep = sleep

    # ------------------------------------------------------------------ run --

    async def run(self, connector: BaseConnector) -> SyncResult:
        """Drive one connector to completion and report what became durable.

        Raises only for defects. Every anticipated failure -- the four families in
        `connectors/exceptions.py` -- comes back as a `SyncResult` carrying
        `error_class`, because the scheduler's next decision (retry sooner, back
        off, stop scheduling, prompt a human) is a function of that class and of
        nothing else.
        """
        ctx = connector.ctx
        key = CursorKey.for_context(connector.slug, ctx)
        state = _RunState(cursor=await self._cursors.load(key) or Cursor())

        logger.info(
            "connector.run.started",
            connector=connector.slug,
            account_id=ctx.account_id,
            run_id=ctx.run_id,
            mode=ctx.mode.value,
            watermark=_isoformat(state.cursor.watermark),
        )

        try:
            # One retry envelope around the whole drain, not around a page.
            # `BaseConnector.run` is an async generator, and a generator that has
            # raised is closed -- there is no `__anext__` to try again. The
            # recoverable unit is therefore "resume from the last committed
            # cursor", which is exactly what a retry does here, and it is safe
            # precisely because the commit only ever follows an ack.
            await retry_page(
                lambda: self._drain(connector, state, key),
                policy=self._backoff,
                rng=self._rng,
                sleep=self._sleep if self._sleep is not None else _default_sleep,
                breaker=self._breaker,
                breaker_key=f"{connector.slug}:{ctx.account_id}",
                connector=connector.slug,
                account_id=ctx.account_id,
            )
        except ConnectorError as err:
            return await self._handle_failure(connector, state, key, err)

        logger.info(
            "connector.run.finished",
            connector=connector.slug,
            account_id=ctx.account_id,
            run_id=ctx.run_id,
            emitted=state.emitted,
            watermark=_isoformat(state.cursor.watermark),
            **state.totals,
        )
        return self._result(connector, state)

    async def _drain(self, connector: BaseConnector, state: _RunState, key: CursorKey) -> None:
        """Consume every batch the connector yields, persisting each in turn.

        Resumes from `state.cursor` -- the last *committed* position -- so a
        transient retry re-fetches only the page that failed, not the run.
        """
        state.begin_attempt()
        async for batch in connector.run(state.cursor):
            state.absorb(batch.stats)
            await self._persist_batch(connector, batch, state, key)

    # -------------------------------------------------------- the invariant --

    async def _persist_batch(
        self,
        connector: BaseConnector,
        batch: EmittedBatch,
        state: _RunState,
        key: CursorKey,
    ) -> None:
        """Steps 1, 2 and 3 of `docs/connector-spec.md` §2.6, in that order.

        The order is the contract, so it is written as three sequential blocks
        with nothing between them that could reorder or skip one:

            1. archive  -- every payload is in R2 before anything references it
            2. publish  -- and *await the ack*; returning from `_publish` is the
                           broker's durability statement, not a buffer accept
            3. commit   -- the last statement in this function

        **There is deliberately no `try`/`except` here.** A failure anywhere above
        the commit returns control to the caller with the cursor exactly as it
        was, so the page replays on the next run and dedup absorbs the duplicate.
        Wrapping the publish loop in a handler that fell through to the commit is
        the one edit that turns this module from at-least-once into
        silently-lossy, and there is no local structure that catches it -- which
        is why the commit is placed where an early `return` or an escaping
        exception cannot reach it.
        """
        # --- step 1: archive every raw payload to R2 -------------------------
        #
        # Sequential rather than gathered: the archive is the cheap half of the
        # batch and concurrency here would buy a little latency at the cost of
        # making "which record failed" a matter of inspecting an exception group.
        events: list[RawRecordEvent] = []
        for record, signal in batch.records:
            event = await self._archive_record(connector, record, signal, state)
            if event is not None:
                events.append(event)

        # --- step 2: publish, and wait for the broker to acknowledge ---------
        for event in events:
            await self._publish(event, tenant_id=self._tenant_id)
            state.emitted += 1

        # --- step 3: the ack is in; only now may the cursor move -------------
        await self._commit_cursor(connector, key, batch.cursor, state)

    async def _archive_record(
        self,
        connector: BaseConnector,
        record: RawRecord,
        signal: Signal,
        state: _RunState,
    ) -> RawRecordEvent | None:
        """Archive one payload and build the event that points at it.

        Returns `None` when the record went to the DLQ instead, which is the one
        record-level outcome in the taxonomy that does not stop the run
        (`connectors/exceptions.py`): the surrounding page keeps going, because
        aborting on one malformed item would let it block every well-formed item
        behind it forever -- the cursor would never advance past it.
        """
        raw = (
            record.raw_bytes if record.raw_bytes is not None else _canonical_payload(record.payload)
        )
        stored: StoredObject | None = None
        try:
            stored = await self._archive(signal.platform.value, record.fetched_at, raw)
        except (ExternalServiceError, DependencyUnavailableError) as err:
            # `docs/architecture.md` §7.3: R2 unavailable defers archival and
            # lets enrichment continue. The event still carries full provenance,
            # and `RawRecordEvent.raw_object_key` documents `None` as exactly
            # this case, so a consumer defers or re-fetches rather than silently
            # enriching nothing.
            logger.warning(
                "connector.archive.deferred",
                connector=connector.slug,
                account_id=connector.ctx.account_id,
                signal_id=signal.id,
                error=type(err).__name__,
            )
        except (PermanentError, ValueError) as err:
            # A payload that cannot be archived at all -- empty bytes, an
            # unusable key component. Retrying reproduces it exactly, so the
            # record is parked with its bytes and the run continues.
            await self._to_dlq(connector, record, signal, raw, err, state)
            return None

        return RawRecordEvent(
            # Provenance mirrors `models.lineage.Lineage` field for field, so the
            # enrichment worker copies rather than translates -- a translation
            # table between two nearly identical field sets is where `fetched_at`
            # quietly becomes ingestion time.
            platform=signal.platform,
            # The Signal's `native_id`, not the provider's raw one. They differ
            # whenever a connector derives an identity (a URL hash for a feed with
            # no guid), and the event's partition key is
            # `signal_id(platform, native_id)`. Using the provider id would put
            # the raw event on a different partition from the enriched event for
            # the same item, and per-Signal ordering would be gone.
            native_id=signal.lineage.native_id,
            connector_slug=signal.lineage.connector_slug,
            connector_version=signal.lineage.connector_version,
            sync_run_id=signal.lineage.sync_run_id,
            # From the record, not from the lineage: `raw_object_key` embeds the
            # date partition derived from this instant, so an auditor rebuilding
            # the key from the event has to be given the same value the key was
            # built with.
            fetched_at=record.fetched_at,
            raw_object_key=stored.key if stored is not None else None,
            raw_sha256=stored.sha256 if stored is not None else None,
            raw_bytes=stored.size_bytes if stored is not None else len(raw),
            raw_content_type=record.content_type,
            source_url=record.source_url or signal.url,
            request_fingerprint=record.request_fingerprint or signal.lineage.request_fingerprint,
        )

    async def _commit_cursor(
        self,
        connector: BaseConnector,
        key: CursorKey,
        cursor: Cursor,
        state: _RunState,
    ) -> None:
        """Persist the batch cursor, refusing one whose watermark moved backwards.

        `docs/connector-spec.md` §4.1 rule 2. `connectors/base.py::_guard_watermark`
        already clamps this on the way out of `run()`, so reaching this branch
        means the clamp was bypassed -- an overridden `run()`, or a caller driving
        `_persist_batch` directly -- which is a defect in code, not a provider
        misbehaving. Hence error level and both watermarks in the record.

        Refusing the commit rather than raising: the batch's records are already
        durable, and the next page may be perfectly ordered. Failing the run would
        discard work that is already in Kafka, while leaving the cursor where it
        was already prevents the regression from taking effect -- which is the
        entire purpose of the rule.
        """
        if not _advances(cursor, state.cursor):
            logger.error(
                "connector.cursor.watermark_regression",
                connector=connector.slug,
                account_id=connector.ctx.account_id,
                run_id=connector.ctx.run_id,
                committed_watermark=_isoformat(state.cursor.watermark),
                rejected_watermark=_isoformat(cursor.watermark),
                detail=(
                    "connectors/base.py clamps emitted cursors against the run's "
                    "starting watermark; a rejection here means the clamp was "
                    "bypassed"
                ),
            )
            return

        await self._cursors.commit(key, cursor)
        state.cursor = cursor
        logger.debug(
            "connector.cursor.committed",
            connector=connector.slug,
            account_id=connector.ctx.account_id,
            watermark=_isoformat(cursor.watermark),
            emitted=state.emitted,
        )

    async def _to_dlq(
        self,
        connector: BaseConnector,
        record: RawRecord,
        signal: Signal,
        raw: bytes,
        err: Exception,
        state: _RunState,
    ) -> None:
        """Park one unrecoverable record on `omnisense.dlq`, preserving its bytes.

        The original bytes travel with it so `workers/dlq.py` can replay a fixed
        handler against a historical failure without re-hitting the provider
        (`docs/connector-spec.md` §6) -- which matters because re-fetching is
        lossy: posts get deleted and API windows expire.

        A DLQ publish that itself fails is not swallowed. It means the broker is
        unreachable, and continuing would commit a cursor past a record that
        exists nowhere at all.
        """
        state.totals["dlq"] += 1
        logger.warning(
            "connector.record.dlq",
            connector=connector.slug,
            account_id=connector.ctx.account_id,
            signal_id=signal.id,
            native_id=record.native_id,
            error=type(err).__name__,
        )
        await self._publish_dlq(
            DlqEvent.from_failure(
                topic=topic_name(TopicRole.RAW_RECORDS),
                body=raw,
                error=err,
                consumer_group=f"connector:{connector.slug}",
                attempts=1,
                key=signal.id,
            ),
            tenant_id=self._tenant_id,
        )

    # -------------------------------------------------------- error taxonomy --

    async def _handle_failure(
        self,
        connector: BaseConnector,
        state: _RunState,
        key: CursorKey,
        err: ConnectorError,
    ) -> SyncResult:
        """Apply the §6 taxonomy, dispatching on the error's declared class.

        On `error_class` rather than `isinstance`, because the class is the
        contract: `CircuitOpenError` is deliberately classified `QUOTA` without
        being a `QuotaError`, since the right response to an open circuit is to
        stop scheduling and come back later -- the quota response -- and an
        `isinstance` chain would give it the transient one and retry into a source
        already known to be failing.
        """
        fields = err.to_log_fields()

        if err.error_class is ConnectorErrorClass.AUTH:
            # Terminal, with no partial credit: the run is a failure, never a
            # partial success, so nothing reschedules it as a continuation.
            # `SyncResult` still reports the counters honestly -- records already
            # acked are in Kafka whatever this run is called, and a result
            # claiming zero would make the run log disagree with the topic.
            logger.error("connector.run.auth_failed", run_id=connector.ctx.run_id, **fields)
            await self._flag_needs_reauth(connector, err)

        elif err.error_class is ConnectorErrorClass.QUOTA:
            # A partial success. Emitted records stay emitted and the cursor stays
            # committed; throwing away an hour of successful pagination because
            # the 4,000th request hit a wall is how a backfill never finishes.
            logger.warning("connector.run.quota_exhausted", run_id=connector.ctx.run_id, **fields)
            await self._checkpoint(connector, key, state)
            await self._reschedule(connector, err)

        else:
            # Transient with its attempts or budget exhausted, or permanent. Both
            # fail the run with the cursor at its last acked page.
            logger.error("connector.run.failed", run_id=connector.ctx.run_id, **fields)

        return self._result(connector, state, error=err)

    async def _flag_needs_reauth(self, connector: BaseConnector, err: ConnectorError) -> None:
        if self._accounts is None:
            return
        await self._accounts.mark_needs_reauth(connector.ctx.account_id, reason=err.message)

    async def _checkpoint(self, connector: BaseConnector, key: CursorKey, state: _RunState) -> None:
        """Re-commit the last acked cursor on a quota halt.

        Idempotent by construction: every acked page already committed this exact
        cursor, so this normally writes what is already there. It is written again
        rather than assumed because "checkpoint on quota" is a rule of the
        taxonomy, and a future change that stopped committing per page must not
        silently turn a partial success into lost work. The cost is one row-write
        per quota event.

        Skipped for an empty cursor: a run that hit its quota before emitting
        anything has nothing to resume from, and creating a row saying so would
        make a never-synced account indistinguishable from one whose provider
        returned nothing.
        """
        if state.cursor.is_empty:
            return
        await self._cursors.commit(key, state.cursor)
        logger.info(
            "connector.cursor.checkpointed",
            connector=connector.slug,
            account_id=connector.ctx.account_id,
            watermark=_isoformat(state.cursor.watermark),
        )

    async def _reschedule(self, connector: BaseConnector, err: ConnectorError) -> None:
        """Move the account's next run to when the provider says it will serve us."""
        resume_at = _resume_at(err)
        logger.info(
            "connector.run.rescheduled",
            connector=connector.slug,
            account_id=connector.ctx.account_id,
            resume_at=_isoformat(resume_at),
        )
        if self._accounts is None or resume_at is None:
            return
        await self._accounts.reschedule(connector.ctx.account_id, when=resume_at)

    # ---------------------------------------------------------------- result --

    def _result(
        self,
        connector: BaseConnector,
        state: _RunState,
        *,
        error: ConnectorError | None = None,
    ) -> SyncResult:
        return SyncResult(
            run_id=connector.ctx.run_id,
            connector_slug=connector.slug,
            account_id=connector.ctx.account_id,
            fetched=state.totals["fetched"],
            emitted=state.emitted,
            dropped=state.totals["dropped"],
            duplicates=state.totals["duplicates"],
            dlq=state.totals["dlq"],
            pages=state.totals["pages"],
            cursor=state.cursor,
            started_at=state.started_at,
            ended_at=utcnow(),
            error=None if error is None else error.message,
            error_class=None if error is None else error.error_class.value,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _default_sleep(seconds: float) -> None:
    """`asyncio.sleep`, imported lazily so the module has no import-time cost."""
    import asyncio

    await asyncio.sleep(seconds)


def _advances(candidate: Cursor, committed: Cursor) -> bool:
    """Whether `candidate` may replace `committed` under §4.1 rule 2.

    Equality passes: a page that produced no newer record legitimately re-commits
    the same watermark with a fresh `page_token`. Dropping to `None` from a real
    timestamp does not -- that is a regression to "no watermark", which triggers a
    full re-sync on the next run.
    """
    if committed.watermark is None:
        return True
    if candidate.watermark is None:
        return False
    return candidate.watermark >= committed.watermark


def _canonical_payload(payload: Mapping[str, Any]) -> bytes:
    """Serialize a payload the connector gave us no raw bytes for.

    `RawRecord.raw_bytes` is the exact bytes the provider returned and is what
    `raw_sha256` should be taken over; re-serializing produces a different digest
    on a different json library version, and content-addressed keys would drift.
    This path exists for connectors that genuinely never held the bytes (a feed
    parsed by a library that discards them), and its determinism is only as good
    as `json.dumps` -- sorted and separator-pinned here so at least it is stable
    within a build.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def _resume_at(err: ConnectorError) -> datetime | None:
    """When a quota-classified failure says we may run again.

    Reads the three spellings the taxonomy actually produces -- `QuotaError`'s
    `reset_at`, `CircuitOpenError`'s `opens_until`, and a bare
    `retry_after_seconds` -- because looking in only one of them is how a
    provider's own instruction gets parsed correctly, attached correctly and then
    ignored.
    """
    for attribute in ("reset_at", "opens_until"):
        value = getattr(err, attribute, None)
        if isinstance(value, int | float) and value > 0:
            return datetime.fromtimestamp(float(value), tz=UTC)
    delay = getattr(err, "retry_after_seconds", None)
    if delay is None:
        delay = err.details.get("retry_after_seconds")
    if isinstance(delay, int | float) and delay > 0:
        return datetime.fromtimestamp(utcnow().timestamp() + float(delay), tz=UTC)
    return None


def _as_utc(moment: datetime | None) -> datetime | None:
    """Attach UTC to a naive timestamp read back from the database.

    `DateTime(timezone=True)` is `TIMESTAMPTZ` on PostgreSQL and returns an aware
    value; SQLite has no timezone type and hands back a naive one. Comparing a
    naive watermark against an aware one raises `TypeError`, so the monotonicity
    check would fail with a type error instead of a verdict -- on the unit suite's
    database, which is where the check is tested.
    """
    if moment is None or moment.tzinfo is not None:
        return moment
    return moment.replace(tzinfo=UTC)


def _isoformat(moment: datetime | None) -> str | None:
    return None if moment is None else moment.isoformat()
