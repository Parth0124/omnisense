"""Operational helpers for PostgreSQL: health, capability checks and locking.

`backend/db/session.py` owns the engine, the session factory and the readiness
probe. This module owns everything an *operator* needs and an application does
not: how long the round trip took, which server version answered, whether the
extensions the schema depends on are actually installed, what the connection pool
is doing, and the cross-process mutex that keeps two scheduler replicas from
doing the same work twice.

It deliberately does not create an engine. There is exactly one pool per process
(`docs/data-stores.md` §3.1); a second one built here would double the connection
count against a database whose `max_connections` is the binding constraint, and
it would do so invisibly.

Two of the helpers here exist because of failures that are otherwise silent:

- `missing_extensions()`. `pg_trgm` is what makes fuzzy entity-name matching
  work. Without it the similarity operators are simply absent, so the queries
  that use them fail loudly -- but a deployment where `01-extensions.sql` never
  ran looks perfectly healthy until the first entity-resolution query, which may
  be hours after the deploy.
- `advisory_lock()`. See its docstring: it is the thing standing between two
  scheduler replicas and a double connector sync.

Layer note: this is the **L1k kernel** (`docs/architecture.md` §6.1) -- importable
by `services/`, `agents/`, `workers/`, `backend/api/` and `scripts/`, but never by
`connectors/`.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from backend.core.exceptions import ConfigurationError, DependencyUnavailableError
from backend.core.logging import get_logger
from backend.db.session import get_engine

__all__ = [
    "REQUIRED_EXTENSIONS",
    "PoolStats",
    "advisory_lock",
    "advisory_lock_key",
    "installed_extensions",
    "missing_extensions",
    "ping",
    "pool_stats",
    "require_extensions",
    "server_version",
]

logger = get_logger(__name__)

POSTGRES_BACKEND = "postgresql"

REQUIRED_EXTENSIONS: frozenset[str] = frozenset({"uuid-ossp", "pg_trgm", "btree_gin", "unaccent"})
"""Extensions created by `docker/local/postgres/01-extensions.sql`.

Kept in lockstep with that file. Each one is load-bearing:
`uuid-ossp` for server-side id defaults, `pg_trgm` for fuzzy entity-name
matching, `btree_gin` for composite metadata filters over `JSONB`, and
`unaccent` for accent-insensitive search (`docs/data-stores.md` §3.1).
"""

_LOCK_POLL_INTERVAL_SECONDS = 0.25
"""How often a timed `advisory_lock()` retries. See the docstring for why it polls."""


@dataclass(frozen=True, slots=True)
class PoolStats:
    """A snapshot of the connection pool, for gauge metrics.

    Fields are `int | None` because the counters belong to `QueuePool`. The unit
    suite runs on SQLite, which SQLAlchemy serves with `StaticPool`/`NullPool` --
    neither of which tracks any of this. `None` means "this pool does not count
    that", which is honestly different from zero and must not be exported as a
    zero gauge.
    """

    backend: str
    """Dialect name, so a metric from a SQLite test run is identifiable as such."""

    size: int | None
    """Connections the pool is configured to keep open (`POSTGRES_POOL_SIZE`)."""

    checked_in: int | None
    """Idle connections available for checkout."""

    checked_out: int | None
    """Connections currently held by application code."""

    overflow: int | None
    """Connections open beyond `size`. Negative until `size` is reached.

    Saturation is `overflow == POSTGRES_MAX_OVERFLOW`: at that point the next
    checkout waits `POSTGRES_POOL_TIMEOUT_SECONDS` and then raises. That is the
    alertable condition, and it is invisible in latency until it is total.
    """

    def as_metrics(self) -> dict[str, int]:
        """The subset that can be exported, keyed by Prometheus gauge suffix.

        `docs/observability.md` §3.1 names no datastore pool metrics today; the
        four numbers below are what an `omnisense_postgres_pool_*` gauge family
        needs, and dropping the `None`s keeps a SQLite test run from publishing
        meaningless zeroes.
        """
        candidates = {
            "size": self.size,
            "checked_in": self.checked_in,
            "checked_out": self.checked_out,
            "overflow": self.overflow,
        }
        return {name: value for name, value in candidates.items() if value is not None}


def _require_postgres(operation: str) -> None:
    """Refuse a PostgreSQL-only operation on another backend, in plain words.

    The unit suite points `DATABASE_URL` at SQLite so that tests need no Docker
    (`docs/testing-strategy.md`). Without this guard, `SHOW server_version`
    against SQLite surfaces as `OperationalError: near "SHOW": syntax error`,
    which reads like a bug in the query rather than "you are not talking to
    PostgreSQL".
    """
    backend = get_engine().dialect.name
    if backend != POSTGRES_BACKEND:
        raise ConfigurationError(
            f"{operation} is PostgreSQL-only and the configured DATABASE_URL "
            f"resolves to the {backend!r} backend. This helper is unavailable "
            "against SQLite; use it from the integration suite or from a "
            "process configured with a real PostgreSQL DSN.",
            details={"operation": operation, "backend": backend},
        )


async def ping() -> float:
    """Round-trip a `SELECT 1` and return the elapsed milliseconds.

    Measures checkout *and* query, because that is what a request experiences.
    A pool exhausted by a leaked connection shows up here as hundreds of
    milliseconds while the database itself is idle -- splitting the two would
    hide the more common failure.

    Raises:
        DependencyUnavailableError: PostgreSQL did not answer. Use
            `session.check_postgres()` instead where a bool is wanted;
            `/readyz` must never fail because a probe raised.
    """
    started = time.perf_counter()
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise DependencyUnavailableError.for_store("PostgreSQL", cause=exc) from exc
    return (time.perf_counter() - started) * 1000.0


async def server_version() -> str:
    """Return the server version string, e.g. `"16.4"`.

    Recorded at startup and on `/readyz`'s degraded detail, because "which
    PostgreSQL answered?" is the first question after a failover and the last
    one anybody thinks to capture. Never exposed on an unauthenticated endpoint
    -- `docs/observability.md` §9.2 forbids leaking versions from the readiness
    body.

    Raises:
        ConfigurationError: The configured backend is not PostgreSQL.
        DependencyUnavailableError: PostgreSQL did not answer.
    """
    _require_postgres("server_version()")
    try:
        async with get_engine().connect() as conn:
            result = await conn.execute(text("SHOW server_version"))
            return str(result.scalar_one())
    except Exception as exc:
        raise DependencyUnavailableError.for_store("PostgreSQL", cause=exc) from exc


async def installed_extensions() -> frozenset[str]:
    """Return the extension names currently installed in the database.

    Raises:
        ConfigurationError: The configured backend is not PostgreSQL.
        DependencyUnavailableError: PostgreSQL did not answer.
    """
    _require_postgres("installed_extensions()")
    try:
        async with get_engine().connect() as conn:
            result = await conn.execute(text("SELECT extname FROM pg_extension"))
            return frozenset(str(row[0]) for row in result.fetchall())
    except Exception as exc:
        raise DependencyUnavailableError.for_store("PostgreSQL", cause=exc) from exc


async def missing_extensions() -> frozenset[str]:
    """Return the members of `REQUIRED_EXTENSIONS` that are not installed.

    Raises:
        ConfigurationError: The configured backend is not PostgreSQL.
        DependencyUnavailableError: PostgreSQL did not answer.
    """
    return REQUIRED_EXTENSIONS - await installed_extensions()


async def require_extensions() -> None:
    """Fail loudly at startup when a required extension is absent.

    Belongs in the API lifespan and in `scripts/init_databases.py`, not in a
    request path. The failure this prevents is a database that was created
    without `01-extensions.sql` ever running -- restored from a plain `pg_dump`,
    or provisioned by hand. Everything works until the first query that reaches
    for `pg_trgm`, which is entity resolution, which is hours later and in a
    worker.

    Raises:
        ConfigurationError: One or more required extensions are missing, or the
            configured backend is not PostgreSQL.
        DependencyUnavailableError: PostgreSQL did not answer.
    """
    missing = await missing_extensions()
    if missing:
        raise ConfigurationError(
            "required PostgreSQL extensions are not installed: "
            f"{', '.join(sorted(missing))}. They are created by "
            "docker/local/postgres/01-extensions.sql, which runs only on first "
            "container start; run it by hand against an existing database.",
            details={"missing_extensions": sorted(missing)},
        )


def pool_stats() -> PoolStats:
    """Snapshot the connection pool. Synchronous -- it touches no socket.

    Safe to call from a Prometheus collector on a timer. Deliberately not called
    from inside a request handler: `docs/observability.md` §3.2 rules that gauges
    are refreshed by a periodic collector, because sampling a gauge per request
    makes its value a function of traffic rather than of the pool.
    """
    pool = get_engine().pool

    def counter(name: str) -> int | None:
        # `QueuePool` defines these; `NullPool` and `StaticPool` -- what SQLite
        # gets -- define none of them, and one of them (`size`) is a plain
        # attribute on some pool classes rather than a method.
        attribute = getattr(pool, name, None)
        if attribute is None:
            return None
        try:
            value = attribute() if callable(attribute) else attribute
        except (AttributeError, NotImplementedError):
            return None
        return int(value) if isinstance(value, int) else None

    return PoolStats(
        backend=get_engine().dialect.name,
        size=counter("size"),
        checked_in=counter("checkedin"),
        checked_out=counter("checkedout"),
        overflow=counter("overflow"),
    )


def advisory_lock_key(name: str) -> int:
    """Map a human-readable lock name onto the signed 64-bit int PostgreSQL wants.

    Advisory locks are keyed by a bigint, not a string, so every caller would
    otherwise invent its own hash -- and two subsystems disagreeing about how
    `"connector-sync:reddit"` becomes a number is a mutex that does not mutex.
    BLAKE2b is used rather than `hash()`, which is randomized per process by
    `PYTHONHASHSEED` and would therefore produce a different key in every
    replica.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


@asynccontextmanager
async def advisory_lock(
    key: str | int, *, timeout_seconds: float | None = None
) -> AsyncIterator[AsyncConnection]:
    """Hold a PostgreSQL session-level advisory lock for the duration of a block.

    **This is what stops two scheduler replicas from running the same connector
    sync simultaneously.** `workers/scheduler.py` runs with more than one replica
    for availability; without a cross-process mutex both wake on the same cron
    tick, both fetch the same window from the same third-party API, and the
    result is double the rate-limit consumption, duplicate raw objects in R2 and
    two ingestion runs racing for the same `(platform, native_id)` row. The lock
    lives in PostgreSQL because PostgreSQL is the one store the scheduler already
    cannot run without (`docs/architecture.md` §7.3), so it adds no new
    dependency and no new failure mode.

    The lock is held by a dedicated connection, yielded to the caller. Work done
    on a *different* session is still protected -- an advisory lock is global to
    the database, not scoped to a transaction -- so the yielded connection can be
    ignored.

    Two details that are wrong in most implementations of this:

    1. **The connection runs in `AUTOCOMMIT`.** Otherwise SQLAlchemy opens an
       implicit transaction on the first statement and holds it for the entire
       critical section. A scheduler job lasting minutes then shows up as
       `idle in transaction`, blocks `VACUUM` from reclaiming anything newer than
       its snapshot, and is eventually killed by
       `idle_in_transaction_session_timeout` -- which releases the lock mid-job.
    2. **The unlock is explicit.** A session-level advisory lock survives
       `ROLLBACK`, and SQLAlchemy's pool-return reset *is* a `ROLLBACK`. Closing
       the connection without unlocking hands a still-locked session back to the
       pool, and the next borrower -- possibly the next tick of the same job --
       deadlocks against a lock its own process is holding.

    Args:
        key: A lock name, hashed by `advisory_lock_key()`, or a pre-computed
            bigint.
        timeout_seconds: Give up after this long instead of waiting forever.
            `None` waits indefinitely, which is right for a job that must
            eventually run and wrong for anything on a request path.

    Raises:
        ConfigurationError: The configured backend is not PostgreSQL.
        DependencyUnavailableError: The lock could not be acquired within
            `timeout_seconds`, or PostgreSQL did not answer.
    """
    _require_postgres("advisory_lock()")
    lock_key = advisory_lock_key(key) if isinstance(key, str) else key

    conn = await get_engine().connect()
    try:
        conn = await conn.execution_options(isolation_level="AUTOCOMMIT")
        await _acquire_advisory_lock(conn, lock_key, timeout_seconds)
    except Exception:
        await conn.close()
        raise

    try:
        yield conn
    finally:
        await _release_advisory_lock(conn, lock_key)


async def _release_advisory_lock(conn: AsyncConnection, lock_key: int) -> None:
    """Unlock and hand the connection back, or destroy it if unlocking failed.

    Two deliberate departures from `docs/coding-standards.md` §2.7 rule 2, both
    of which come from this running in a `finally`:

    **The exception is logged and not re-raised.** This runs while the caller's
    own exception may be propagating; raising here would replace "the connector
    sync failed because the API returned 500" with "the unlock statement failed",
    and the second one is never the interesting error.

    **The connection is invalidated rather than closed.** `close()` returns the
    session to the pool, and a session whose unlock did not succeed is still
    holding the lock -- the next borrower would then wait on a lock this process
    owns and nobody is going to release. `invalidate()` drops the underlying
    DBAPI connection instead, which ends the PostgreSQL session and lets the
    server release every advisory lock it held.
    """
    try:
        result = await conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": lock_key})
        held = bool(result.scalar_one())
    except Exception:
        logger.warning("postgres.advisory_lock.unlock_failed", lock_key=lock_key, exc_info=True)
        await conn.invalidate()
        await conn.close()
        return

    if not held:
        # False means this session did not hold the lock -- the connection was
        # reset underneath us, or two callers disagree about the key. Either way
        # the mutual exclusion the block promised was not actually in force.
        logger.warning("postgres.advisory_lock.unlock_not_held", lock_key=lock_key)

    await conn.close()


async def _acquire_advisory_lock(
    conn: AsyncConnection, lock_key: int, timeout_seconds: float | None
) -> None:
    """Take the lock, blocking or polling depending on whether a timeout is set.

    The timed path polls `pg_try_advisory_lock` rather than wrapping the blocking
    `pg_advisory_lock` in `asyncio.wait_for`. Cancelling an in-flight query
    cancels it *client-side*: the server may still grant the lock immediately
    afterwards, and nothing is then tracking that this connection holds it. A
    poll can only ever be in one of two states, both of them knowable.
    """
    if timeout_seconds is None:
        await conn.execute(text("SELECT pg_advisory_lock(:key)"), {"key": lock_key})
        return

    deadline = time.monotonic() + timeout_seconds
    while True:
        result = await conn.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": lock_key})
        if result.scalar_one():
            return
        if time.monotonic() >= deadline:
            raise DependencyUnavailableError(
                f"could not acquire PostgreSQL advisory lock {lock_key} within "
                f"{timeout_seconds}s; another process is holding it.",
                details={"lock_key": lock_key, "timeout_seconds": timeout_seconds},
            )
        await asyncio.sleep(min(_LOCK_POLL_INTERVAL_SECONDS, max(deadline - time.monotonic(), 0)))
