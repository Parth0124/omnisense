"""Executes the templates in `graph/queries/cypher.py`, and nothing else.

`backend/db/neo4j.py` owns the driver: the lazy singleton, the auth, the pool,
the read/write session split. This module owns the layer above it -- the part
that is about *graph queries* rather than about Bolt -- and the split is what
keeps `graph/` an L1 library. `graph/` may not import `backend/`
(`docs/architecture.md` §6.1), so the driver arrives as a callable through
`read_runner_from_session_factory()` and this module never names a driver type.

Four things happen here that would otherwise be repeated at every call site.

**Neo4j temporal types become Python ones.** The driver returns
`neo4j.time.DateTime`, not `datetime.datetime`. They look alike, print alike, and
compare *unequal* -- and the difference surfaces at the far end of the system, in
a Pydantic model that rejects the value or a service that subtracts two of them
and gets a `neo4j.time.Duration` it cannot serialise. Converting once, at the
boundary, means nothing downstream of `graph/` ever sees a driver type. Done by
duck-typing `.to_native()` rather than by importing `neo4j`, because importing
the driver for an `isinstance` check is the layer violation this module exists to
avoid.

**Every query is bounded.** A Cypher read has no natural timeout; an unbounded
traversal against a graph that grew a hot node holds a server thread until
someone notices. `DEFAULT_QUERY_TIMEOUT_SECONDS` is applied with
`asyncio.timeout`, so a slow graph degrades the request that asked rather than
the process.

**Transient failures retry, permanent ones do not.** A leader election, a reaped
connection and a `TransientError` are worth one more attempt. A syntax error, a
constraint violation and an unknown procedure are not -- retrying them turns one
clear failure into three identical ones a second apart and buries the real
message under a retry log. Classification is by exception *class name* (again, to
avoid the driver import), which is coarse but wrong in the safe direction: an
unrecognised error is treated as permanent, so a new driver exception fails
loudly instead of being retried into a timeout.

**Reads are reads.** `fetch()` goes through a `READ`-mode session, which the
server enforces: a `MERGE` issued through it is rejected with
`Neo.ClientError.Statement.AccessMode`. That is what makes "`retrieval/` is
read-only over the graph" (`docs/architecture.md` §6.2 rule 3) a mechanism rather
than a convention. Writes have exactly one door, `graph/ingest/writer.py`, and
this module deliberately does not open a second.

Layer note: **L1 library** -- `models/` plus the rest of `graph/`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Final, Protocol, runtime_checkable

import structlog

from graph.queries.cypher import Query

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_QUERY_TIMEOUT_SECONDS",
    "TRANSIENT_ERROR_NAMES",
    "GraphClient",
    "GraphClientError",
    "GraphQueryError",
    "GraphUnavailableError",
    "QueryRunner",
    "is_transient",
    "normalize_record",
    "normalize_value",
    "read_runner_from_session_factory",
]

# `backend.core.logging.get_logger` is a passthrough to this, and `graph/` may
# not import `backend/`. Binding structlog directly gets the same logger and the
# same processor chain, configured by whichever process called
# `backend.core.logging.configure_logging()` at startup.
_log = structlog.get_logger(__name__)


DEFAULT_QUERY_TIMEOUT_SECONDS: Final[float] = 15.0
"""Wall-clock ceiling on one graph read, retries included.

Chosen against the request budget rather than against Neo4j: a `/graph/search`
that takes longer than this has already lost its caller, and holding the
connection past that point costs a server thread the next request needs. Batch
work in `graph/analytics/` sets its own, much larger, bound.
"""

DEFAULT_MAX_ATTEMPTS: Final[int] = 3

_RETRY_BASE_DELAY_SECONDS: Final[float] = 0.1

TRANSIENT_ERROR_NAMES: Final[frozenset[str]] = frozenset(
    {
        "TransientError",
        "ServiceUnavailable",
        "SessionExpired",
        "IncompleteCommit",
        "WriteServiceUnavailable",
        "ReadServiceUnavailable",
        "ConnectionAcquisitionTimeoutError",
        "ClientError",  # conditional -- see `is_transient`
    }
)
"""Driver exception class names worth one more attempt.

Matched by name so this module does not import `neo4j` for an `isinstance`
check. `ClientError` is in the set with a caveat enforced in `is_transient()`:
the class covers both "the leader moved" and "your Cypher does not parse", and
retrying the second three times produces three identical stack traces with the
syntax error buried under them.
"""

_LEADER_HINTS: Final[tuple[str, ...]] = (
    "not the leader",
    "no longer accepts writes",
    "routing table",
    "unable to connect",
    "database is unavailable",
    "cluster member",
)


class GraphClientError(RuntimeError):
    """A graph operation failed for a reason that is not a bad argument.

    A `RuntimeError`, deliberately not the `ValueError` that
    `graph.schema.nodes.GraphSchemaError` is: a malformed query is the caller's
    fault and a 4xx, an unreachable database is nobody's and a 5xx.
    `services/graph_service.py` maps each onto the kernel exception carrying the
    right status code -- a translation `graph/` cannot do for itself, because it
    may not import `backend/core/exceptions.py`.
    """


class GraphQueryError(GraphClientError):
    """The query reached the server and the server rejected it.

    Permanent by construction: a syntax error, a missing procedure, an
    access-mode violation. Retrying changes nothing.
    """


class GraphUnavailableError(GraphClientError):
    """The query could not be executed -- unreachable, timed out, out of retries.

    The one the retrieval path catches. `docs/architecture.md` §7.3: Neo4j is not
    a hard dependency, so graph expansion is skipped, the report records reduced
    context, and the request still answers from vector and keyword hits.
    """


@runtime_checkable
class QueryRunner(Protocol):
    """Executes one statement and materialises its rows. The whole seam.

    Two arguments, one return, no transaction handle and no session -- everything
    a graph *read* needs from a driver, expressed so this module can be tested
    against a dict-backed fake and so the production implementation is
    unambiguously a `READ`-mode managed transaction.

    Rows must be materialised before returning. A Neo4j result stream is bound to
    its transaction and unreadable once that transaction commits, so a runner
    that returned the stream itself would hand back a closed cursor -- and the
    failure appears at first iteration, far from the code that caused it.
    """

    async def __call__(
        self, cypher: str, parameters: Mapping[str, Any] | None = None
    ) -> Sequence[Mapping[str, Any]]: ...


SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]
"""`backend.db.neo4j.read_session` has exactly this shape.

Typed as `Any` inside the context manager rather than as `neo4j.AsyncSession`,
because naming that type would require importing the driver into an L1 library
for the sake of an annotation.
"""


def read_runner_from_session_factory(session_factory: SessionFactory) -> QueryRunner:
    """Build a `QueryRunner` over `backend.db.neo4j.read_session`.

    The entire wiring, and the only place the two layers meet:

        from backend.db.neo4j import read_session
        from graph.client import GraphClient, read_runner_from_session_factory

        client = GraphClient(read_runner_from_session_factory(read_session))

    `execute_read` rather than `session.run` because it is the managed form: the
    driver retries the transient failures a leader election or a reaped
    connection produces, and it sends `READ` access mode, which the server
    enforces. Records are materialised inside the transaction function, for the
    reason given on `QueryRunner`.
    """

    async def _runner(
        cypher: str, parameters: Mapping[str, Any] | None = None
    ) -> Sequence[Mapping[str, Any]]:
        async def _work(tx: Any) -> list[dict[str, Any]]:
            result = await tx.run(cypher, parameters=dict(parameters or {}))
            return [record.data() async for record in result]

        async with session_factory() as session:
            return await session.execute_read(_work)

    return _runner


# --------------------------------------------------------------------------- #
# Driver value normalisation
# --------------------------------------------------------------------------- #


def normalize_value(value: Any) -> Any:
    """Convert a driver-native value into a stdlib one, recursively.

    `neo4j.time.DateTime` is the one that matters. It carries nanosecond
    precision Python's `datetime` does not, prints identically, and compares
    unequal to a `datetime` for the same instant -- so a service that stores one
    and compares it against `datetime.now(UTC)` gets a silently wrong answer, and
    a Pydantic model with a `datetime` field rejects it outright at the API
    boundary several layers from here.

    Detected via `.to_native()` rather than `isinstance`, so `graph/` does not
    import `neo4j`. `neo4j.time.{DateTime,Date,Time,Duration}` and
    `neo4j.spatial.Point` all expose it and nothing in the standard library does
    -- and the `isinstance` fast path above means a stdlib `datetime` that
    somehow grew such a method is still passed through untouched.

    Nested containers are walked, because Cypher returns them: `collect()` yields
    a list, a map projection yields a dict, and a driver `DateTime` inside either
    is exactly as wrong as one at the top level.
    """
    if value is None or isinstance(value, (str, bool, int, float, datetime, date, time)):
        return value
    to_native = getattr(value, "to_native", None)
    if callable(to_native):
        try:
            return normalize_value(to_native())
        except Exception:  # noqa: BLE001 -- a value we cannot convert is passed through
            return value
    if isinstance(value, Mapping):
        return {key: normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [normalize_value(item) for item in value]
    return value


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """`normalize_value` over one row, returning a plain dict."""
    return {key: normalize_value(value) for key, value in record.items()}


# --------------------------------------------------------------------------- #
# Retry classification
# --------------------------------------------------------------------------- #


def is_transient(error: BaseException) -> bool:
    """Whether retrying this error could plausibly succeed.

    Classified by exception class name, walking the MRO so a driver subclass of
    `TransientError` is caught without this module knowing its name. Coarse, and
    coarse in the safe direction: an *unrecognised* error is permanent, so a new
    driver exception surfaces immediately instead of being retried into a timeout
    and then reported as unavailability -- which would misdirect whoever
    investigates towards the network.

    `ClientError` is the awkward one. Neo4j uses it both for "this instance is no
    longer the leader" (retry; the driver re-routes) and for "your Cypher does
    not parse" (do not). The message decides, and when it says nothing
    recognisable, permanent wins.
    """
    names = {klass.__name__ for klass in type(error).__mro__}
    matched = names & TRANSIENT_ERROR_NAMES
    if not matched:
        return False
    if matched == {"ClientError"}:
        message = str(error).casefold()
        return any(hint in message for hint in _LEADER_HINTS)
    return True


# --------------------------------------------------------------------------- #
# The client
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GraphClient:
    """Runs `Query` objects, normalises what comes back, bounds how long it takes.

    Frozen and stateless apart from its configuration, so one instance per
    process serves concurrent requests -- the driver underneath holds the pool
    and is already concurrency-safe.

    It accepts a `Query`, never a string and a dict. That is the point: the only
    way to get a query into this client is to have built it in
    `graph/queries/cypher.py`, where parameterisation is guaranteed. A method
    taking raw Cypher would be used, and the first caller to use it would
    f-string a tenant id into it.
    """

    runner: QueryRunner
    timeout_seconds: float = DEFAULT_QUERY_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    async def fetch(self, query: Query) -> list[dict[str, Any]]:
        """Execute a read and return every row, normalised.

        Raises `GraphQueryError` when the server rejected the statement and
        `GraphUnavailableError` when it could not be reached or did not answer in
        time. The retrieval path catches the second and continues with degraded
        context; nobody should catch the first, because it means the query is
        wrong and degrading would hide that permanently.
        """
        return await self._run(query.cypher, query.parameters)

    async def fetch_one(self, query: Query) -> dict[str, Any] | None:
        """Execute a read and return the first row, or `None`.

        `None` rather than an exception for an empty result: "this entity is not
        in the graph" is an ordinary answer at this layer, and turning it into a
        404 is a decision for `services/graph_service.py`, which knows whether
        the caller asked for something that ought to exist.
        """
        rows = await self._run(query.cypher, query.parameters)
        return rows[0] if rows else None

    async def fetch_value(self, query: Query, key: str, default: Any = None) -> Any:
        """One column of the first row -- for a `count(*)` or an existence probe."""
        row = await self.fetch_one(query)
        return default if row is None else row.get(key, default)

    async def _run(self, cypher: str, parameters: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Execute with a total deadline, retrying only transient failures.

        `asyncio.timeout` wraps the *whole* attempt loop rather than each
        attempt. Bounding each attempt separately would let three attempts plus
        two backoffs take three times the stated timeout, which defeats the point
        of stating it -- the caller budgeted 15 seconds and would wait 45.
        """
        last_error: BaseException | None = None
        try:
            async with asyncio.timeout(self.timeout_seconds):
                for attempt in range(1, self.max_attempts + 1):
                    try:
                        rows = await self.runner(cypher, parameters)
                    except asyncio.CancelledError:
                        # Never swallowed, never retried. Inside `asyncio.timeout`
                        # this is how the deadline arrives; catching it as a
                        # retryable failure would restart the query against an
                        # already-expired budget.
                        raise
                    except Exception as error:  # noqa: BLE001 -- classified below
                        last_error = error
                        if not is_transient(error) or attempt == self.max_attempts:
                            break
                        _log.warning(
                            "graph.query.retry",
                            attempt=attempt,
                            max_attempts=self.max_attempts,
                            error=type(error).__name__,
                        )
                        await asyncio.sleep(_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))
                        continue
                    return [normalize_record(row) for row in rows]
        except TimeoutError as error:
            # `asyncio.timeout` raises `TimeoutError` on expiry. Unavailability,
            # not a query error -- the statement may have been perfectly valid.
            raise GraphUnavailableError(
                f"graph query exceeded {self.timeout_seconds}s"
            ) from error

        # The loop only leaves without returning by breaking, which always sets
        # `last_error` first.
        assert last_error is not None
        if is_transient(last_error):
            raise GraphUnavailableError(
                f"graph unavailable after {self.max_attempts} attempts: {last_error}"
            ) from last_error
        raise GraphQueryError(f"graph rejected the query: {last_error}") from last_error

    @asynccontextmanager
    async def degrade_on_unavailable(self, operation: str) -> AsyncIterator[None]:
        """Swallow `GraphUnavailableError` for a block, log it, and continue.

        The explicit form of "Neo4j is not a hard dependency". A caller writes

            async with client.degrade_on_unavailable("competitor_expansion"):
                facts = await client.fetch(competitors_of(...))

        and gets a request that still answers from vector and keyword hits when
        the graph is down. Only unavailability is swallowed -- a `GraphQueryError`
        propagates, because degrading past a malformed query hides it permanently
        rather than for the duration of an outage.
        """
        try:
            yield
        except GraphUnavailableError as error:
            _log.warning("graph.degraded", operation=operation, error=str(error))
