"""Async SQLAlchemy engine and session factory.

This module establishes the pattern every other client in `backend/db/` follows:

1. A module-level, lazily-created singleton -- connection pools are expensive and
   must be shared across the process, not rebuilt per request.
2. An `async` accessor that builds on first use, so importing this module never
   opens a socket. Import-time I/O makes the test suite depend on a running
   database just to collect tests.
3. A `dispose_*()` for shutdown, wired into the FastAPI lifespan and the worker
   runtime, so connections close cleanly rather than being reset by the peer.
4. A `check_*()` returning a bool for `/readyz`, which must actually probe the
   dependency rather than assume it.

Layer note: this is the **L1k kernel** (`docs/architecture.md` §6.1) -- importable
by `services/`, `agents/`, `workers/`, `backend/api/` and `scripts/`, but never by
`connectors/`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine.url import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import get_settings
from backend.core.exceptions import DependencyUnavailableError
from backend.db import HEALTH_PROBE_TIMEOUT_SECONDS

__all__ = [
    "check_postgres",
    "dispose_engine",
    "get_engine",
    "get_session",
    "get_sessionmaker",
    "session_scope",
]

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use.

    `pool_pre_ping=True` is not optional here. Connections idle in the pool are
    silently killed by PostgreSQL's `idle_in_transaction_session_timeout`, by
    cloud load balancers, and by container restarts. Without a pre-ping the first
    query after an idle period fails with a confusing `OperationalError` that
    looks like a database outage and is actually a stale socket. The ping costs
    one round trip per checkout.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        url = make_url(settings.postgres.url)

        options: dict[str, Any] = {
            "echo": settings.postgres.echo_sql,
            "pool_pre_ping": True,
        }

        # `pool_size`, `max_overflow` and `pool_timeout` belong to `QueuePool`.
        # SQLite -- used by the unit suite so tests need no Docker -- is served by
        # `StaticPool`/`NullPool`, which reject them outright with a TypeError at
        # engine construction. Guarding on the backend keeps one code path
        # working for both without the caller having to know.
        if url.get_backend_name() != "sqlite":
            options |= {
                "pool_size": settings.postgres.pool_size,
                "max_overflow": settings.postgres.max_overflow,
                "pool_timeout": settings.postgres.pool_timeout_seconds,
                # Bounds the TCP connect and each statement. Without these,
                # asyncpg falls back to its own 60s connect default and no
                # statement timeout at all, so a half-open network path hangs a
                # request until the client gives up rather than failing fast.
                "connect_args": {
                    "timeout": settings.postgres.connect_timeout_seconds,
                    "command_timeout": settings.postgres.command_timeout_seconds,
                },
                # Recycle below the typical 1-hour idle timeout of managed
                # PostgreSQL and most proxies, so we close connections before
                # they are closed for us.
                "pool_recycle": 1800,
            }

        _engine = create_async_engine(url, **options)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory.

    `expire_on_commit=False` because objects are routinely read *after* the
    transaction commits -- serialized into a response, published to Kafka. With
    the default `True`, every attribute access after commit triggers a lazy
    refresh, which in async SQLAlchemy raises `MissingGreenlet` rather than
    quietly issuing a query. That error is thoroughly confusing the first time
    you meet it, and the fix is always this flag.
    """
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope for non-request code: workers, scripts, agent tools.

    Commits on success, rolls back on any exception, always closes. Use this
    rather than `get_session()` outside a request -- FastAPI's dependency system
    is what drives `get_session`, and there is none in a worker.
    """
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session for the life of one request.

    Deliberately does **not** commit. A request handler that changed something
    commits explicitly, which keeps "this endpoint writes" visible at the call
    site instead of implied by the dependency. Rollback on exception is still
    automatic, so a failed request cannot leak a half-applied transaction.
    """
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def check_postgres() -> bool:
    """Probe PostgreSQL for `/readyz`. Never raises, and never blocks for long.

    Returns a bool rather than raising because readiness aggregates several
    dependencies and one being down must not prevent reporting on the others
    (`docs/observability.md`).

    The explicit `asyncio.timeout` is load-bearing, and none of the settings that
    look like they would bound this actually do: `POSTGRES_POOL_TIMEOUT_SECONDS`
    bounds pool *checkout*, not the TCP connect underneath it. Against a
    blackholed host this probe measured 60s before the timeout was added -- the
    slowest of the six by an order of magnitude, and enough on its own to stall
    `/readyz` past the liveness deadline. PostgreSQL is the store
    `docs/architecture.md` §7.3 lists as a hard failure on both paths, so it is
    the probe where a slow answer hurts most.
    """
    try:
        async with asyncio.timeout(HEALTH_PROBE_TIMEOUT_SECONDS):
            async with get_engine().connect() as conn:
                await conn.execute(text("SELECT 1"))
    except (Exception, asyncio.TimeoutError):
        return False
    return True


async def require_postgres() -> None:
    """Assert PostgreSQL is reachable, raising the typed 503 if not.

    For code paths where degrading is not an option -- `docs/architecture.md`
    §7.3 lists PostgreSQL as a hard failure on both the investigation and the
    ingestion path.
    """
    if not await check_postgres():
        raise DependencyUnavailableError.for_store("PostgreSQL")


async def dispose_engine() -> None:
    """Close the pool and reset the singletons. Called from lifespan shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
