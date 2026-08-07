"""The only module that writes to Neo4j, and the one that could break the rebuild.

`docs/architecture.md` §5 makes this the single write path for the graph, and
`docs/data-stores.md` §3.2 states the rule that path has to hold up:

    Must never store: signal bodies, vectors, or any property that would make
    Neo4j the only place a fact lives. If an agent can learn something from the
    graph that is not derivable from PostgreSQL plus R2, the rebuild guarantee is
    broken.

That guarantee is what makes the graph disposable. Neo4j Community has no
clustering, offline-only dumps, and a single primary (`docs/data-stores.md`,
open question 8): the operational answer to losing it is to rebuild it from
PostgreSQL and R2, and that answer only works while nothing lives *only* here.
Every convenience that would break it looks reasonable in isolation -- caching a
signal's title on the stub so the graph explorer needs one fewer join, storing
the extracted snippet on the `MENTIONS` edge so a citation renders without a
lookup -- and each one quietly converts a derived store into a system of record.
So `SignalStub` carries five fields, is frozen, and the fragment that writes it
names those five and nothing else. There is no `properties` escape hatch on it,
and that omission is the enforcement.

**Idempotence, and why it needs more than `MERGE`.**
Kafka delivery is at-least-once (`docs/architecture.md` §4.2), so this code will
be handed the same batch twice, and the driver's managed transactions will
themselves retry a statement whose commit acknowledgement was lost. `MERGE` makes
node and edge *identity* replay-safe. It does nothing for counters:
`source_count = source_count + n` applied twice counts twice, which is exactly
what `backend/db/neo4j.py` warns against routing through a managed transaction.

The fix is a content-derived batch id, stamped on every node and edge the batch
touches. A row's counter only advances when the last batch to touch it was a
*different* batch. Because the id is a hash of the batch's contents rather than a
random value, a Kafka replay produces the same id and is suppressed too -- one
mechanism covering both duplication sources. Its limit is worth stating: A, B, A
replayed out of order will double-count, because the guard remembers one batch
and not a set. That is the rare case; the common one is a consumer restart
replaying the batch it had not committed, and that is covered exactly.

**Ordering, and why it is not cosmetic.**
Within one transaction: entity nodes, then signal stubs, then edges. Edges
`MATCH` their endpoints, and a `MATCH` that misses does not error -- the row
simply produces no relationship, and the edge is lost with nothing written
anywhere to say so. Writing nodes first in the same transaction is what makes the
endpoints exist; comparing the returned `count(r)` against the row count is what
turns the remaining case -- an endpoint from an *earlier* batch that never
arrived -- into a loud failure instead of a silent gap in the graph.

Rows are also sorted by id. Two workers writing overlapping entity sets in
opposite orders deadlock on each other's locks; Neo4j resolves it by killing one
transaction, and the batch is retried. Sorting makes every writer take locks in
the same order, so the deadlock does not happen in the first place.

Layer note: **L1 library** -- `models/` and the standard library only. The Neo4j
seam is `TransactionRunner`, supplied by the caller; see
`runner_from_session_factory` for the ten-line adapter over
`backend.db.neo4j.write_session`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

import structlog

from graph.schema import edges as edge_schema
from graph.schema import nodes as node_schema
from graph.schema.edges import SIGNAL_LABEL, edge_key, edge_spec, orient, validate_endpoints
from graph.schema.nodes import SCHEMA_VERSION, GraphSchemaError, node_spec
from models.enums import EdgeType, EntityType

__all__ = [
    "SIGNAL_STUB_PROPERTIES",
    "EdgeWrite",
    "GraphBatch",
    "GraphWriteError",
    "GraphWriter",
    "NodeWrite",
    "PlannedStatement",
    "SignalStub",
    "TransactionRunner",
    "WriteOutcome",
    "WritePlan",
    "runner_from_session_factory",
    "signal_stub_cypher",
]

# `backend.core.logging.get_logger` is a one-line passthrough to this, and `graph/`
# may not import `backend/` (`docs/architecture.md` §6.1). Binding structlog
# directly gets the same logger and the same processor chain, configured by
# whichever process called `backend.core.logging.configure()` at startup.
_log = structlog.get_logger(__name__)

SIGNAL_STUB_PROPERTIES: Final[tuple[str, ...]] = (
    "id",
    "tenant_id",
    "published_at",
    "source",
    "platform",
)
"""Every property a `(:Signal)` node may carry. Exhaustive, and enforced by shape.

`published_at` earns its place because `retrieval/graph_retrieval/traversal.py`
filters signals by date *inside* the traversal, and pulling every mentioned
signal back to PostgreSQL to find out which ones are in the window would defeat
the traversal. `source` and `platform` are there for the same reason -- a
platform filter that cannot be pushed into the graph becomes a post-filter over
an unbounded result set. All three are derivable from PostgreSQL, so none of them
is a fact that lives only here.

Nothing else may be added without an ADR. See the module docstring.
"""


class GraphWriteError(RuntimeError):
    """A write reached the database and did not do what it was asked to.

    Distinct from `GraphSchemaError`, which is a bad argument caught before any
    query runs. This one means the graph's state and the batch disagree -- most
    often an edge whose endpoint node does not exist -- and it is worth
    separating because the responses differ: a schema error is a bug to fix, this
    is a batch to quarantine or an upstream to re-run.
    """


# --------------------------------------------------------------------------- #
# The Neo4j seam
# --------------------------------------------------------------------------- #

Statement = tuple[str, dict[str, Any]]
"""One parameterised statement: `(cypher, parameters)`. Never a formatted string."""


@runtime_checkable
class TransactionRunner(Protocol):
    """Runs a sequence of statements in **one** transaction, in order.

    The contract is deliberately this small. The writer needs exactly two things
    from Neo4j -- ordered execution and one transaction boundary -- and expressing
    the seam at that level keeps driver semantics (result streams that die with
    their transaction, managed-transaction retries, access modes) out of an L1
    library, and makes the unit tests a dozen lines of fake rather than a mock
    driver.

    Returns one record list per statement, positionally. The writer reads
    `count(n)` out of them to verify that what it asked for is what happened.

    There is no default implementation here on purpose: `graph/` may not import
    `backend/db/neo4j.py`. Use `runner_from_session_factory(write_session)`.
    """

    async def __call__(self, statements: Sequence[Statement]) -> list[list[dict[str, Any]]]: ...


SessionFactory = Callable[[], AbstractAsyncContextManager[Any]]
"""`backend.db.neo4j.write_session` has exactly this shape.

Typed as `Any` inside the context manager rather than as a `neo4j.AsyncSession`
because naming that type would require importing the driver into an L1 library
for an annotation. What the adapter needs of it -- `execute_write` taking an
async transaction function -- is documented below and exercised by the tests'
fake, which is the same fake shape `tests/unit/backend/db/test_neo4j.py` uses.
"""


def runner_from_session_factory(session_factory: SessionFactory) -> TransactionRunner:
    """Build a `TransactionRunner` over `backend.db.neo4j.write_session`.

    The whole wiring, and the only place the two layers meet:

        from backend.db.neo4j import write_session
        from graph.ingest.writer import GraphWriter, runner_from_session_factory

        writer = GraphWriter(runner_from_session_factory(write_session))

    `execute_write` rather than `session.run` because it is the managed form: the
    driver retries the transient failures a leader election or a reaped
    connection produces, instead of surfacing them as a graph outage. The retry is
    only safe because every statement this module emits is idempotent -- including
    the counters, thanks to the batch-id guard.

    Records are materialised *inside* the transaction function. A result stream
    is bound to its transaction and is unreadable once it commits, so returning
    the stream itself would hand back a closed cursor.
    """

    async def _runner(statements: Sequence[Statement]) -> list[list[dict[str, Any]]]:
        async def _work(tx: Any) -> list[list[dict[str, Any]]]:
            collected: list[list[dict[str, Any]]] = []
            for query, parameters in statements:
                result = await tx.run(query, parameters=parameters)
                collected.append([record.data() async for record in result])
            return collected

        async with session_factory() as session:
            outcome = await session.execute_write(_work)
            return list(outcome)

    return _runner


# --------------------------------------------------------------------------- #
# What a caller hands in
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NodeWrite:
    """One entity to upsert.

    The four required common properties are named fields rather than entries in
    `properties`, because a write missing one of them is a bug the type system
    can catch instead of a `GraphSchemaError` at run time.
    """

    entity_type: EntityType
    id: str
    tenant_id: str
    canonical_name: str
    normalized_name: str
    observed_at: datetime
    aliases: tuple[str, ...] = ()
    new_signal_count: int = 0
    """How many *newly seen* signals this row evidences. A delta, never a total.

    Zero is the right value for a write that is refreshing an entity's attributes
    rather than recording a new observation -- a description regenerated by an
    LLM, say. Sending the running total instead would replace every other
    writer's accumulation with this writer's view of it.
    """

    properties: Mapping[str, Any] = field(default_factory=dict)
    """Label-specific and optional common properties. Validated against the spec."""

    def __post_init__(self) -> None:
        if self.new_signal_count < 0:
            raise GraphSchemaError(
                f"new_signal_count={self.new_signal_count} for {self.id!r}; a "
                "negative delta would silently decrement a counter that only "
                "ever accumulates"
            )
        spec = node_spec(self.entity_type)
        overlap = sorted(set(self.properties) & _NODE_NAMED_FIELDS)
        if overlap:
            raise GraphSchemaError(
                f"{spec.label} write for {self.id!r} passes "
                f"{', '.join(overlap)} both as a field and in properties; the two "
                "would disagree and the map would win"
            )
        node_schema.validate_node_properties(self.entity_type, self.to_property_map())

    def to_property_map(self) -> dict[str, Any]:
        """The named fields and `properties`, as one map, for validation."""
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "canonical_name": self.canonical_name,
            "normalized_name": self.normalized_name,
            "aliases": list(self.aliases),
            **dict(self.properties),
        }

    def to_row(self) -> dict[str, Any]:
        """The `$rows` element the generated fragment consumes.

        `observed_at` and `new_signal_count` are row inputs rather than
        properties: the fragment folds them into `first_seen`, `last_seen` and
        `source_count`, and neither is stored under its own name.
        """
        row = self.to_property_map()
        row["observed_at"] = self.observed_at
        row["new_signal_count"] = self.new_signal_count
        return row


_NODE_NAMED_FIELDS: Final[frozenset[str]] = frozenset(
    {"id", "tenant_id", "canonical_name", "normalized_name", "aliases"}
)


@dataclass(frozen=True, slots=True)
class SignalStub:
    """A content-free anchor for `MENTIONS` and `COMPLAINS_ABOUT`.

    Five fields, no `properties` map, frozen. That is not minimalism for its own
    sake -- see the module docstring. The absence of an extension point is what
    stops the stub from accreting a title, then a snippet, then the text, at
    which point Neo4j holds signal content that PostgreSQL does not know it needs
    to reproduce.
    """

    id: str
    tenant_id: str
    published_at: datetime
    source: str
    platform: str

    def to_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "published_at": self.published_at,
            # `str()` rather than passing the enum: `Platform` and
            # `SourceCategory` are `StrEnum`s, so the driver would serialise them
            # as strings anyway -- but only by accident of their base class, and
            # a future non-string enum would land in the graph as a repr.
            "source": str(self.source),
            "platform": str(self.platform),
        }


@dataclass(frozen=True, slots=True)
class EdgeWrite:
    """One relationship to upsert, in the orientation the caller extracted it.

    `orient()` is applied by the writer, not by the caller: a symmetric edge has
    to be stored in one canonical direction, and leaving that to every call site
    guarantees that one of them forgets and creates the mirrored duplicate this
    design exists to avoid.
    """

    edge_type: EdgeType
    source_label: str
    source_id: str
    target_label: str
    target_id: str
    tenant_id: str
    valid_from: datetime
    observed_at: datetime
    valid_to: datetime | None = None
    confidence: float | None = None
    extractor: str | None = None
    source_signal_ids: tuple[str, ...] = ()
    new_evidence: int = 0
    evidence_key: str = ""
    """Splits edges that share endpoints and interval but different evidence.

    Empty is right for an extracted fact about the world -- two articles reporting
    the same acquisition are one edge with two citations. It is *not* right for
    `MENTIONS`, where the signal id belongs here: two signals mentioning the same
    entity are two edges, and collapsing them would make `evidence_count` the only
    trace that the second signal existed.
    """

    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.new_evidence < 0:
            raise GraphSchemaError(
                f"new_evidence={self.new_evidence} for a {self.edge_type} edge; a "
                "negative delta would decrement a counter that only accumulates"
            )
        validate_endpoints(self.edge_type, self.source_label, self.target_label)
        overlap = sorted(set(self.properties) & _EDGE_NAMED_FIELDS)
        if overlap:
            raise GraphSchemaError(
                f"{self.edge_type} write passes {', '.join(overlap)} both as a "
                "field and in properties"
            )
        edge_schema.validate_edge_properties(self.edge_type, self.to_property_map())

    # ----------------------------------------------------------- derivation --

    @property
    def oriented(self) -> tuple[str, str, str, str]:
        """`(from_id, from_label, to_id, to_label)` after canonical orientation."""
        return orient(
            self.edge_type,
            self.source_id,
            self.source_label,
            self.target_id,
            self.target_label,
        )

    @property
    def key(self) -> str:
        """The `MERGE` key, computed over the *oriented* endpoints.

        Orientation first is essential. Keying on the raw endpoints would give the
        same symmetric fact two different keys depending on which way the
        extractor happened to emit it, and `MERGE` would create two edges between
        the same pair -- the exact duplication `orient()` exists to prevent.
        """
        from_id, _, to_id, _ = self.oriented
        return edge_key(self.edge_type, from_id, to_id, self.valid_from, self.evidence_key)

    @property
    def group(self) -> tuple[EdgeType, str, str]:
        """The `(type, from_label, to_label)` a query is generated per."""
        _, from_label, _, to_label = self.oriented
        return (self.edge_type, from_label, to_label)

    def to_property_map(self) -> dict[str, Any]:
        return {
            "edge_key": self.key,
            "tenant_id": self.tenant_id,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "observed_at": self.observed_at,
            "confidence": self.confidence,
            "extractor": self.extractor,
            "source_signal_ids": list(self.source_signal_ids),
            **dict(self.properties),
        }

    def to_row(self) -> dict[str, Any]:
        from_id, _, to_id, _ = self.oriented
        row = self.to_property_map()
        row["from_id"] = from_id
        row["to_id"] = to_id
        row["new_evidence"] = self.new_evidence
        return row


_EDGE_NAMED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "edge_key",
        "tenant_id",
        "valid_from",
        "valid_to",
        "observed_at",
        "confidence",
        "extractor",
        "source_signal_ids",
    }
)


@dataclass(frozen=True, slots=True)
class GraphBatch:
    """Everything one transaction writes."""

    nodes: tuple[NodeWrite, ...] = ()
    signals: tuple[SignalStub, ...] = ()
    edges: tuple[EdgeWrite, ...] = ()

    @property
    def row_count(self) -> int:
        """Rows, not statements. The unit `graph/ingest/batcher.py` sizes on."""
        return len(self.nodes) + len(self.signals) + len(self.edges)

    @property
    def is_empty(self) -> bool:
        return self.row_count == 0

    @classmethod
    def of(cls, items: Iterable[NodeWrite | SignalStub | EdgeWrite]) -> GraphBatch:
        """Partition a mixed stream into a batch. Used by the batcher."""
        nodes: list[NodeWrite] = []
        signals: list[SignalStub] = []
        edges: list[EdgeWrite] = []
        for item in items:
            if isinstance(item, NodeWrite):
                nodes.append(item)
            elif isinstance(item, SignalStub):
                signals.append(item)
            elif isinstance(item, EdgeWrite):
                edges.append(item)
            else:  # pragma: no cover - defensive, the union is closed
                raise GraphSchemaError(f"{type(item).__name__} is not a graph write")
        return cls(nodes=tuple(nodes), signals=tuple(signals), edges=tuple(edges))


@dataclass(frozen=True, slots=True)
class WriteOutcome:
    """What one `apply()` did."""

    batch_id: str
    nodes_written: int
    signals_written: int
    edges_written: int
    statement_count: int

    @property
    def total(self) -> int:
        return self.nodes_written + self.signals_written + self.edges_written


# --------------------------------------------------------------------------- #
# The signal stub fragment
# --------------------------------------------------------------------------- #


def signal_stub_cypher() -> str:
    """The `MERGE` for a `(:Signal)` anchor. Five properties, and no way to add more.

    Written out rather than generated from a spec, because there is no spec: the
    point of this label is that it has no schema to grow into. A reader looking
    for "what does the graph store about a signal" gets the whole answer from
    these four lines.

    Note what is missing next to every other fragment in this package: no
    `source_count`, no `last_batch_id`, no replay guard. A stub carries no
    counters, so applying it twice is naturally idempotent -- `MERGE` plus four
    absolute `SET`s of values that do not change.

    **This `MERGE` has no uniqueness constraint behind it.** `Signal` is an eighth
    label that postdates `docker/local/neo4j/01-constraints.cypher`, so two
    concurrent workers can each create a stub for the same id
    (`graph/schema/constraints.py`, `PENDING_CONSTRAINTS`). It is the least
    harmful place for that to happen -- a stub has no properties to lose and the
    duplicates converge the moment the constraint lands -- but it is real, and
    pretending otherwise is how it stays unfixed.
    """
    return (
        "UNWIND $rows AS row\n"
        f"MERGE (s:{SIGNAL_LABEL} {{id: row.id}})\n"
        "ON CREATE SET s.created_at = datetime()\n"
        "SET s.tenant_id = row.tenant_id,\n"
        "    s.published_at = row.published_at,\n"
        "    s.source = row.source,\n"
        "    s.platform = row.platform\n"
        "RETURN count(s) AS written"
    )


# --------------------------------------------------------------------------- #
# The writer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PlannedStatement:
    """One statement, with everything needed to check what it reports back.

    A record per statement rather than four parallel lists. The migrator is
    forced into parallel arrays by Neo4j's property model and has to validate
    them on every read; there is no such excuse in Python, and a mismatch between
    a `kinds` list and a `counts` list would misattribute a write failure to the
    wrong label at exactly the moment someone is trying to debug it.
    """

    kind: str
    label: str
    cypher: str
    parameters: dict[str, Any]
    expected_rows: int
    group: tuple[EdgeType, str, str] | None = None

    def as_statement(self) -> Statement:
        return (self.cypher, self.parameters)


@dataclass(frozen=True, slots=True)
class WritePlan:
    """The full, ordered transaction for one batch, before it runs."""

    batch_id: str
    statements: tuple[PlannedStatement, ...]

    @property
    def is_empty(self) -> bool:
        return not self.statements

    def to_statements(self) -> list[Statement]:
        return [planned.as_statement() for planned in self.statements]


class GraphWriter:
    """Turns a `GraphBatch` into one ordered, parameterised transaction."""

    def __init__(self, runner: TransactionRunner, *, schema_version: int = SCHEMA_VERSION) -> None:
        self._run = runner
        self._schema_version = schema_version

    async def apply(self, batch: GraphBatch) -> WriteOutcome:
        """Write one batch. Nodes, then signal stubs, then edges, atomically.

        Returns without touching the database for an empty batch. That is not an
        optimisation: an empty transaction still costs a round trip and a lock
        acquisition, and the batcher flushes on a timer, so a quiet period would
        otherwise generate one pointless transaction every two seconds forever.
        """
        plan = self.plan(batch)
        if plan.is_empty:
            return WriteOutcome(
                batch_id=plan.batch_id,
                nodes_written=0,
                signals_written=0,
                edges_written=0,
                statement_count=0,
            )

        results = await self._run(plan.to_statements())
        if len(results) != len(plan.statements):
            raise GraphWriteError(
                f"runner returned {len(results)} result sets for "
                f"{len(plan.statements)} statements; the writer cannot tell which "
                "statement produced what, so it cannot tell whether the write "
                "succeeded"
            )

        counts = [_written_count(result) for result in results]
        self._verify(plan, counts, batch)

        totals = {"node": 0, "signal": 0, "edge": 0}
        for planned, count in zip(plan.statements, counts, strict=True):
            totals[planned.kind] += count

        _log.debug(
            "graph.write.applied",
            batch_id=plan.batch_id,
            statements=len(plan.statements),
            nodes=totals["node"],
            signals=totals["signal"],
            edges=totals["edge"],
        )
        return WriteOutcome(
            batch_id=plan.batch_id,
            nodes_written=totals["node"],
            signals_written=totals["signal"],
            edges_written=totals["edge"],
            statement_count=len(plan.statements),
        )

    # ------------------------------------------------------------ planning --

    def plan(self, batch: GraphBatch) -> WritePlan:
        """Build the ordered statement list without running it.

        Public because it is worth being able to inspect a batch's Cypher in a
        test or a `scripts/` dry run without a database, and because that is the
        cheapest way to assert the property this package cares most about: that no
        value is ever interpolated into a query.
        """
        bid = batch_id_for(batch)
        planned: list[PlannedStatement] = []

        for entity_type, rows in _group_nodes(batch.nodes).items():
            planned.append(
                PlannedStatement(
                    kind="node",
                    label=node_spec(entity_type).label,
                    cypher=node_schema.merge_cypher(entity_type),
                    parameters={
                        "rows": rows,
                        "batch_id": bid,
                        "schema_version": self._schema_version,
                    },
                    expected_rows=len(rows),
                )
            )

        signal_rows = _dedupe_signals(batch.signals)
        if signal_rows:
            planned.append(
                PlannedStatement(
                    kind="signal",
                    label=SIGNAL_LABEL,
                    cypher=signal_stub_cypher(),
                    # No `batch_id` and no `schema_version`: a stub carries no
                    # counter to guard and no schema to version. Passing unused
                    # parameters would be harmless and misleading.
                    parameters={"rows": signal_rows},
                    expected_rows=len(signal_rows),
                )
            )

        for group, rows in _group_edges(batch.edges).items():
            edge_type, from_label, to_label = group
            planned.append(
                PlannedStatement(
                    kind="edge",
                    label=f"{from_label}-[{edge_spec(edge_type).type_name}]->{to_label}",
                    cypher=edge_schema.merge_cypher(edge_type, from_label, to_label),
                    parameters={
                        "rows": rows,
                        "batch_id": bid,
                        "schema_version": self._schema_version,
                    },
                    expected_rows=len(rows),
                    group=group,
                )
            )

        return WritePlan(batch_id=bid, statements=tuple(planned))

    def _verify(self, plan: WritePlan, counts: Sequence[int], batch: GraphBatch) -> None:
        """Compare what came back against what was asked for.

        The case this exists for is an edge whose endpoint node is not in the
        graph. Cypher's `MATCH` is not an assertion: a row that matches nothing
        contributes no relationship and raises nothing, so without this check a
        missing endpoint is a permanently absent edge that no log line, metric or
        error ever mentions. It is the one graph failure that is genuinely
        invisible, which is why it is worth a comparison on every write.
        """
        for planned, got in zip(plan.statements, counts, strict=True):
            if got == planned.expected_rows:
                continue
            if planned.group is not None:
                missing = sorted(_missing_endpoint_ids(batch, planned.group))[:10]
                lead = (
                    ", ".join(missing)
                    if missing
                    else "none -- every endpoint in this batch was also written by "
                    "it, so the absent node was expected from an earlier batch"
                )
                raise GraphWriteError(
                    f"{planned.label}: wrote {got} of {planned.expected_rows} "
                    "edges. The rows that produced nothing have an endpoint that "
                    "does not exist in the graph -- MATCH does not raise when it "
                    f"misses, so this check is the only thing that notices. {lead}"
                )
            raise GraphWriteError(
                f"{planned.label}: wrote {got} of {planned.expected_rows} "
                f"{planned.kind} rows. A MERGE either matches or creates, so a "
                "shortfall means the batch and the graph disagree in a way this "
                "writer cannot explain."
            )


# --------------------------------------------------------------------------- #
# Grouping, deduplication and the batch id
# --------------------------------------------------------------------------- #


def _group_nodes(nodes: Sequence[NodeWrite]) -> dict[EntityType, list[dict[str, Any]]]:
    """One row list per label, deduplicated by id and sorted by id.

    Deduplication is not a tidiness measure. Two rows for the same id inside one
    `UNWIND` both run: the counter advances twice for what is one entity, and --
    worse -- `count(n)` returns 2 while only one node exists, which would trip the
    verification above on a batch that is actually fine. Merging them here is the
    only place with enough context to add the deltas rather than lose one.

    Sorting by id is the lock-ordering measure. See the module docstring.
    """
    grouped: dict[EntityType, dict[str, dict[str, Any]]] = {}
    for node in nodes:
        by_id = grouped.setdefault(node.entity_type, {})
        row = node.to_row()
        existing = by_id.get(node.id)
        if existing is None:
            by_id[node.id] = row
            continue
        by_id[node.id] = _merge_rows(existing, row)

    # Registry order across labels so two workers emit their statements in the
    # same sequence, for the same reason rows are sorted within one.
    ordered: dict[EntityType, list[dict[str, Any]]] = {}
    for entity_type in node_schema.NODE_SPECS:
        rows = grouped.get(entity_type)
        if rows:
            ordered[entity_type] = [rows[key] for key in sorted(rows)]
    return ordered


def _merge_rows(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """Fold two rows that address the same node or the same edge into one.

    Deltas add; lists union preserving first-seen order; scalars take the later
    value when it is not None. "Later wins" is the right rule for a scalar
    because the rows arrive in the order the enrichment pipeline produced them,
    so the second one is the more recent observation -- and "not None" keeps a
    row that simply had nothing to say about a property from erasing what the
    first row knew.
    """
    merged = dict(first)
    for key, value in second.items():
        if key in ("new_signal_count", "new_evidence"):
            merged[key] = int(merged.get(key, 0)) + int(value)
        elif isinstance(value, list) and isinstance(merged.get(key), list):
            seen = list(merged[key])
            seen.extend(item for item in value if item not in seen)
            merged[key] = seen
        elif key == "observed_at" and merged.get(key) is not None and value is not None:
            merged[key] = max(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def _dedupe_signals(signals: Sequence[SignalStub]) -> list[dict[str, Any]]:
    """Signal rows, one per id, sorted by id."""
    by_id: dict[str, dict[str, Any]] = {}
    for stub in signals:
        by_id[stub.id] = stub.to_row()
    return [by_id[key] for key in sorted(by_id)]


def _group_edges(
    edges: Sequence[EdgeWrite],
) -> dict[tuple[EdgeType, str, str], list[dict[str, Any]]]:
    """One row list per `(type, from_label, to_label)`, deduplicated by `edge_key`.

    Grouped by the label pair because Cypher cannot parameterise a label and an
    unlabelled endpoint `MATCH` is an all-nodes scan -- see
    `graph/schema/edges.merge_cypher`. The grouping is therefore forced by the
    query shape, and the endpoint validation that comes with it is free.
    """
    grouped: dict[tuple[EdgeType, str, str], dict[str, dict[str, Any]]] = {}
    for edge in edges:
        by_key = grouped.setdefault(edge.group, {})
        row = edge.to_row()
        existing = by_key.get(edge.key)
        by_key[edge.key] = row if existing is None else _merge_rows(existing, row)

    ordered: dict[tuple[EdgeType, str, str], list[dict[str, Any]]] = {}
    for edge_type in edge_schema.EDGE_SPECS:
        for group in sorted(
            (g for g in grouped if g[0] is edge_type),
            key=lambda g: (g[1], g[2]),
        ):
            by_key = grouped[group]
            ordered[group] = [
                by_key[key]
                # Sorted by endpoint ids rather than by `edge_key`: the locks an
                # edge write takes are on its endpoint nodes, and a hash orders
                # them randomly, which is precisely the ordering that deadlocks.
                for key in sorted(by_key, key=lambda k: (by_key[k]["from_id"], by_key[k]["to_id"]))
            ]
    return ordered


def batch_id_for(batch: GraphBatch) -> str:
    """A content hash identifying this batch, stable across processes and replays.

    Content-derived rather than random, and that is the whole idea: a Kafka
    replay reconstructs the same rows, hashes to the same id, and the counter
    guards in the merge fragments suppress the increment. A UUID would make every
    replay look like new evidence and inflate `source_count` on every restart.

    Hashed over the identities *and* the deltas, because a batch that reports two
    new signals for Acme is a genuinely different batch from one that reports
    three, and collapsing them would drop the second increment.

    Sorted before hashing so the id does not depend on the order the rows arrived
    in -- two workers handed the same set in different orders must agree, or the
    guard never fires.
    """
    parts: list[str] = []
    for node in batch.nodes:
        parts.append(f"n\x1f{node.entity_type.value}\x1f{node.id}\x1f{node.new_signal_count}")
    for stub in batch.signals:
        parts.append(f"s\x1f{stub.id}")
    for edge in batch.edges:
        parts.append(f"e\x1f{edge.key}\x1f{edge.new_evidence}")
    digest = hashlib.sha256("\x1e".join(sorted(parts)).encode("utf-8"))
    return digest.hexdigest()[:32]


def _written_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Read `count(...) AS written` out of a result set.

    A statement that returns nothing counts as zero rather than raising: an empty
    result is what a `MATCH` that missed every row produces, which is exactly the
    case `_verify` needs to report well.
    """
    if not records:
        return 0
    value = records[0].get("written")
    return int(value) if isinstance(value, (int, float)) else 0


def _missing_endpoint_ids(batch: GraphBatch, group: tuple[EdgeType, str, str]) -> set[str]:
    """Endpoint ids in this group's edges that no node row in the batch supplies.

    A lead for the error message, not a diagnosis: an endpoint written by an
    *earlier* batch is legitimately absent from this one, so this narrows the
    search rather than naming the culprit. Ids that appear nowhere in the batch's
    own nodes or stubs are the ones worth checking first -- they are the ones the
    upstream never emitted, as opposed to the ones it emitted last week.
    """
    known = {node.id for node in batch.nodes} | {stub.id for stub in batch.signals}
    missing: set[str] = set()
    for edge in batch.edges:
        if edge.group != group:
            continue
        from_id, _, to_id, _ = edge.oriented
        missing.update({from_id, to_id} - known)
    return missing
