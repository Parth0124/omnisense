"""Async Neo4j driver, sessions and query helpers for the knowledge graph.

Shape follows `backend/db/session.py` exactly -- lazy module-level singleton, no
I/O at import, `get_*` / `check_*` / `dispose_*` -- for the reasons documented
there. What is specific to Neo4j is the read/write split.

**Why two session context managers rather than one.**
The graph is reached by whichever URI scheme `NEO4J_URI` carries, and the two
schemes behave differently:

`bolt://` (the local single-instance default)
    No routing. Every session lands on the one server. The access mode is still
    sent to the server, which *enforces* it: a `MERGE` issued inside a
    `READ`-mode transaction is rejected with `Neo.ClientError.Statement.
    AccessMode` rather than quietly mutating the graph. That is the property the
    retrieval path depends on -- `retrieval/graph_retrieval/` is read-only by
    architectural rule (`docs/architecture.md` §6.2 rule 3), and this is the
    mechanism that makes the rule enforceable instead of aspirational.

`neo4j://` (routing, used against a cluster)
    The driver maintains a routing table and picks a server per session *based on
    the access mode*. A `READ` session can be served by any follower or read
    replica; a `WRITE` session must reach the leader. Opening every session in
    `WRITE` mode therefore funnels all graph traffic -- including the read-heavy
    neighbourhood expansion that dominates GraphRAG -- onto the single leader,
    throwing away the read capacity of the rest of the cluster. The distinction
    is a throughput decision, not a formality.

Community Edition has one primary today (`docs/data-stores.md` §3.2, open
question 8), so the routing half of that is latent. Getting the access mode
right now is what makes moving to a cluster a configuration change rather than
an audit of every call site.

**Degradation.** Neo4j is *not* a hard dependency. `docs/architecture.md` §7.3:
graph expansion is skipped, the report notes reduced context, ingestion keeps
running and `omnisense.graph.updates` accumulates lag. Only `/graph/search` and
the Competitor Agent fail outright. So callers on the retrieval path handle the
exception and continue; `require_neo4j()` exists for the two paths that cannot.

Layer note: **L1k kernel** (`docs/architecture.md` §6.1) -- importable by
`services/`, `agents/`, `workers/`, `backend/api/` and `scripts/`, never by
`connectors/`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from neo4j import (
    READ_ACCESS,
    WRITE_ACCESS,
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncManagedTransaction,
    AsyncSession,
    basic_auth,
)
from neo4j.exceptions import ConfigurationError as Neo4jConfigurationError

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError, DependencyUnavailableError
from backend.db import HEALTH_PROBE_TIMEOUT_SECONDS

__all__ = [
    "check_neo4j",
    "dispose_driver",
    "get_driver",
    "read_session",
    "require_neo4j",
    "run_read",
    "run_write",
    "write_session",
]

_driver: AsyncDriver | None = None

# Ping a connection that has been idle longer than this before handing it out.
# This is the Neo4j equivalent of `pool_pre_ping` in `session.py`: a pooled Bolt
# connection is silently dropped by container restarts, NAT tables and load
# balancers, and without the check the first query after an idle period fails
# with a `ServiceUnavailable` that reads like an outage and is actually a stale
# socket. The check costs one RESET round trip, and only on idle connections.
_LIVENESS_CHECK_SECONDS = 30.0

# Retire connections well before the hour-long idle timeout typical of proxies
# in front of Bolt, so we close them rather than discovering they were closed
# for us. Mirrors `pool_recycle` in `session.py`.
_MAX_CONNECTION_LIFETIME_SECONDS = 1800.0

_MAX_TRANSACTION_RETRY_SECONDS = 2.0
"""How long a *managed* transaction may keep retrying inside the driver.

The default is 30s, and leaving it there made `GET /api/v1/graph/search` take a
measured 15 seconds to return 503 against an unreachable Neo4j. The cause is that
retrying happens twice, nested, and the two layers **multiply**: `graph/client.py`
makes up to three attempts, and each one calls `execute_read`, which runs the
driver's own exponential ladder (1.1s, 1.7s, 3.7s, 7.1s ...) before giving up.
Total latency is `client_attempts x driver_window`, not the larger of the two --
so the client's 15s budget was being spent inside its first attempt.

Two seconds is chosen from that arithmetic: three client attempts times two
seconds is a six-second worst case, comfortably inside the 15s request budget and
short enough that a read endpoint fails fast instead of holding a worker. It
still covers what the driver's retry is genuinely for -- a leader election or a
reaped connection, both of which resolve in well under a second -- while leaving
classification, backoff and the give-up decision to the layer that can tell a
transient failure from a syntax error.

`backend/db/session.py` makes the same split for PostgreSQL: the pool handles
sockets, the caller handles semantics. The lesson generalises -- whenever two
layers both retry, measure the product rather than assuming the outer bound wins.
"""


def get_driver() -> AsyncDriver:
    """Return the process-wide async driver, creating it on first use.

    The driver owns a connection pool and, for `neo4j://`, a routing table. It is
    designed to be created once per process and shared -- building one per
    request would re-fetch the routing table and re-authenticate every time.

    Constructing a driver performs no I/O: the first connection is opened lazily
    when a session actually runs something. That is what keeps importing this
    module free of a running database.
    """
    global _driver
    if _driver is None:
        settings = get_settings()
        try:
            _driver = AsyncGraphDatabase.driver(
                settings.neo4j.uri,
                auth=basic_auth(
                    settings.neo4j.user,
                    settings.neo4j.password.get_secret_value(),
                ),
                max_connection_pool_size=settings.neo4j.max_connection_pool_size,
                connection_timeout=float(settings.neo4j.connection_timeout_seconds),
                # Bound the wait for a *pooled* connection too. Left at its
                # default a caller blocks for 60s when the pool is saturated,
                # which turns graph back-pressure into request timeouts far from
                # the cause. Sharing the configured timeout keeps one knob.
                connection_acquisition_timeout=float(settings.neo4j.connection_timeout_seconds),
                liveness_check_timeout=_LIVENESS_CHECK_SECONDS,
                max_connection_lifetime=_MAX_CONNECTION_LIFETIME_SECONDS,
                # Bounded well below `graph/client.py`'s request budget so the
                # two retry layers do not nest. See the constant.
                max_transaction_retry_time=_MAX_TRANSACTION_RETRY_SECONDS,
            )
        except Neo4jConfigurationError as exc:
            # Almost always `NEO4J_URI` pointing at the HTTP port (7474) or at
            # `http://`. The driver's own message is good; what it lacks is which
            # setting to fix, and it would otherwise surface as a 500 from
            # whichever endpoint happened to touch the graph first.
            raise ConfigurationError(
                f"NEO4J_URI is not a valid Bolt URI: {exc}. Use bolt:// or "
                "neo4j:// (port 7687), not the HTTP endpoint on 7474.",
                details={"setting": "NEO4J_URI"},
                cause=exc,
            ) from exc
    return _driver


@asynccontextmanager
async def read_session() -> AsyncIterator[AsyncSession]:
    """Open a read-mode session against the configured database.

    Use this for every traversal, lookup and neighbourhood expansion. Under a
    routing URI it lets the query be served by a follower or read replica; under
    any URI it makes the server reject an accidental write, which is the cheap
    guardrail for the read-only rule on `retrieval/`.
    """
    session = get_driver().session(
        database=get_settings().neo4j.database,
        default_access_mode=READ_ACCESS,
    )
    async with session:
        yield session


@asynccontextmanager
async def write_session() -> AsyncIterator[AsyncSession]:
    """Open a write-mode session against the configured database.

    For `graph/ingest/writer.py` and anything else issuing `MERGE`/`SET`. Under a
    routing URI this is what reaches the leader.

    The session is *not* a transaction. `session.run()` inside it auto-commits
    per statement; when several statements must land together, use
    `session.execute_write(...)` -- which additionally retries on the transient
    errors a leader election produces -- or `session.begin_transaction()`.
    """
    session = get_driver().session(
        database=get_settings().neo4j.database,
        default_access_mode=WRITE_ACCESS,
    )
    async with session:
        yield session


# Both helpers take `parameters` as a separate argument, and callers must use it.
# Never build a query by interpolating or f-stringing a value into `query`:
# Cypher injection is exactly as real as SQL injection, and OmniSense feeds
# LLM-generated and connector-derived strings -- entity names, topic labels,
# search terms -- straight into graph queries. A parameterized query is also
# faster, because the server caches its plan by query text and interpolation
# produces a distinct text every time.
#
# `dict[str, Any]` rather than `dict[str, object]`: a Cypher record is genuinely
# heterogeneous (scalars, nodes, relationships, paths, nested lists), and this
# matches the driver's own `Record.data()` signature. Callers narrow into a
# `models/` type immediately -- see `docs/coding-standards.md` §2.2 rule 2.


async def run_read(
    query: str,
    parameters: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one read query in a managed transaction and return its records.

    Managed rather than auto-commit so the driver retries transient failures --
    a routing-table refresh, a leader election, a connection reaped mid-flight --
    instead of surfacing them to the caller as a graph outage.

    The records are materialized *inside* the transaction function on purpose:
    the result stream is bound to the transaction and is no longer readable once
    it commits, so returning the `AsyncResult` itself would hand back a closed
    cursor.
    """

    async def _work(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
        result = await tx.run(query, parameters=dict(parameters or {}))
        return [record.data() async for record in result]

    async with read_session() as session:
        return await session.execute_read(_work)


async def run_write(
    query: str,
    parameters: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one write query in a managed transaction and return its records.

    Returns records because a `MERGE ... RETURN` is the normal way to learn the
    id of the node that was created or matched.

    The retry the managed transaction gives us is only safe because graph writes
    are `MERGE`-based and therefore idempotent (`docs/data-stores.md` §3.2). Do
    not route a non-idempotent statement -- `SET n.count = n.count + 1` -- through
    here; a retried attempt would apply it twice.
    """

    async def _work(tx: AsyncManagedTransaction) -> list[dict[str, Any]]:
        result = await tx.run(query, parameters=dict(parameters or {}))
        return [record.data() async for record in result]

    async with write_session() as session:
        return await session.execute_write(_work)


async def check_neo4j() -> bool:
    """Probe Neo4j for `/readyz`. Never raises.

    A bool rather than an exception because readiness aggregates several
    dependencies and one being down must not stop the others from being reported
    (`docs/observability.md`). For Neo4j specifically the answer is also not
    fatal: a `False` here means degraded retrieval, not a broken service.

    `RETURN 1` rather than `verify_connectivity()` because it exercises the whole
    path -- routing, connection, authentication, *and* that the configured
    database exists and accepts queries. `verify_connectivity()` stops at the
    handshake and reports healthy against a server where `NEO4J_DATABASE` names
    a database that was never created.

    Deliberately `session.run()` and **not** `run_read()`. The managed
    transaction behind `run_read()` retries transient failures for
    `max_transaction_retry_time` -- 30s by default -- so routing a probe through
    it makes an unreachable Neo4j take half a minute to report as unreachable
    instead of failing on the first attempt. A readiness endpoint that blocks for
    30s per poll is worse than useless: it stalls the whole `/readyz` aggregate
    and can itself trip the liveness probe. Retrying is right for a real query
    and wrong for a probe, so the probe does not share that code path.

    Avoiding the retry loop was necessary but not sufficient. `NEO4J_CONNECTION_
    TIMEOUT_SECONDS` defaults to 30, and against a blackholed host this probe
    still measured exactly 30s -- the connection budget a real query needs is far
    larger than the budget a probe may spend. The explicit `asyncio.timeout` is
    what decouples the two.
    """
    try:
        async with asyncio.timeout(HEALTH_PROBE_TIMEOUT_SECONDS):
            async with read_session() as session:
                result = await session.run("RETURN 1 AS ok")
                await result.consume()
    except (Exception, asyncio.TimeoutError):
        return False
    return True


async def require_neo4j() -> None:
    """Assert Neo4j is reachable, raising the typed 503 if not.

    Only for the two paths that cannot degrade: `/graph/search` and the
    Competitor Agent (`docs/architecture.md` §7.3). Everything else on the
    retrieval path must catch the failure and continue with vector plus keyword
    results, lowering confidence rather than failing the investigation.
    """
    if not await check_neo4j():
        raise DependencyUnavailableError.for_store("Neo4j")


async def dispose_driver() -> None:
    """Close the pool and reset the singleton. Called from lifespan shutdown.

    Awaiting this matters more than it does for a sync driver: an async driver
    left open at interpreter exit leaves connections for the event loop to tear
    down after it has stopped running, which surfaces as `ResourceWarning` and
    "Task was destroyed but it is pending" noise that buries real shutdown
    errors.
    """
    global _driver
    if _driver is not None:
        await _driver.close()
    _driver = None
