"""Unit tests for `graph/ingest/`.

Four properties, and each of them fails silently in production if it breaks --
which is why they are worth a test rather than a code review.

*The same write twice yields one node.* Kafka is at-least-once and managed
transactions retry, so this code is handed the same batch twice as a matter of
routine. `MERGE` covers identity; the counters need the batch-id guard, and a
broken guard produces a `source_count` that drifts upward on every consumer
restart. Nothing errors, and the number that a report cites as its evidence
count is simply wrong.

*An edge between disallowed labels is refused.* There is no query that finds
"Acme competes with Belgium" after the fact -- it is a well-formed relationship
and Neo4j is perfectly happy with it.

*No query contains an interpolated value.* Entity names are scraped from
third-party text, so Cypher injection is a company name away.

*The batcher flushes on both size and age.* Size-only looks correct under load
and loses the tail of every quiet period.

`FakeGraph` below models `MERGE` semantics rather than executing Cypher -- it
keys nodes on `id` and applies the replay guard the way the generated fragment
does. That makes it a model of the write path, not a proof of it; the statements
it is fed are asserted against separately, and the real Cypher is exercised by
`tests/integration/db/`. What it does prove is the part that lives in Python: the
deduplication, the deltas, the ordering and the guard.

No Neo4j, no driver, no network.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from graph.ingest.batcher import BatcherClosedError, WriteBatcher
from graph.ingest.writer import (
    SIGNAL_STUB_PROPERTIES,
    EdgeWrite,
    GraphBatch,
    GraphWriteError,
    GraphWriter,
    NodeWrite,
    SignalStub,
    batch_id_for,
    runner_from_session_factory,
    signal_stub_cypher,
)
from graph.schema.nodes import GraphSchemaError
from models.enums import EdgeType, EntityType

pytestmark = pytest.mark.unit

NOW = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
HOSTILE = "Acme') DETACH DELETE n //"


def company(
    entity_id: str = "ent_acme",
    *,
    name: str = "Acme",
    delta: int = 1,
    aliases: tuple[str, ...] = (),
    observed_at: datetime = NOW,
    **properties: Any,
) -> NodeWrite:
    return NodeWrite(
        entity_type=EntityType.COMPANY,
        id=entity_id,
        tenant_id="t1",
        canonical_name=name,
        normalized_name=name.casefold(),
        observed_at=observed_at,
        aliases=aliases,
        new_signal_count=delta,
        properties=properties,
    )


def stub(signal_id: str = "sig_1") -> SignalStub:
    return SignalStub(
        id=signal_id,
        tenant_id="t1",
        published_at=NOW,
        source="news",
        platform="rss",
    )


def mentions(signal_id: str = "sig_1", entity_id: str = "ent_acme") -> EdgeWrite:
    return EdgeWrite(
        edge_type=EdgeType.MENTIONS,
        source_label="Signal",
        source_id=signal_id,
        target_label="Company",
        target_id=entity_id,
        tenant_id="t1",
        valid_from=NOW,
        observed_at=NOW,
        evidence_key=signal_id,
        new_evidence=1,
        properties={"salience": 0.8},
    )


class FakeGraph:
    """A `TransactionRunner` that models `MERGE` semantics for the two fragments.

    Deliberately not a Cypher engine. It recognises a node merge, a signal merge
    and an edge merge by shape, keys each on the property the fragment merges on,
    and applies the `last_batch_id` guard to the counter exactly as the generated
    Cypher does. Anything it does not recognise raises, so a change to the shape
    of the write path invalidates the fake instead of silently passing.
    """

    def __init__(self) -> None:
        self.nodes: dict[tuple[str, str], dict[str, Any]] = {}
        self.signals: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.transactions: list[list[tuple[str, dict[str, Any]]]] = []
        self.missing_endpoints: set[str] = set()

    async def __call__(
        self, statements: list[tuple[str, dict[str, Any]]]
    ) -> list[list[dict[str, Any]]]:
        self.transactions.append(list(statements))
        results: list[list[dict[str, Any]]] = []
        for query, params in statements:
            results.append([{"written": self._apply(query, params)}])
        return results

    def _apply(self, query: str, params: dict[str, Any]) -> int:
        rows = params["rows"]
        batch_id = params.get("batch_id")

        label_match = re.search(r"MERGE \(n:(\w+) \{id: row\.id\}\)", query)
        if label_match is not None:
            for row in rows:
                self._merge_node(label_match.group(1), row, batch_id)
            return len(rows)

        if "MERGE (s:Signal {id: row.id})" in query:
            for row in rows:
                self.signals[row["id"]] = {
                    k: v for k, v in row.items() if k in SIGNAL_STUB_PROPERTIES
                }
            return len(rows)

        if re.search(r"MERGE \(a\)-\[r:(\w+) \{edge_key: row\.edge_key\}\]->\(b\)", query):
            written = 0
            for row in rows:
                # A MATCH that misses contributes no relationship and raises
                # nothing -- the behaviour `_verify` exists to catch.
                if {row["from_id"], row["to_id"]} & self.missing_endpoints:
                    continue
                self._merge_edge(row, batch_id)
                written += 1
            return written

        raise AssertionError(f"FakeGraph does not model this statement:\n{query}")

    def _merge_node(self, label: str, row: dict[str, Any], batch_id: str | None) -> None:
        key = (label, row["id"])
        node = self.nodes.get(key)
        if node is None:
            node = {"source_count": 0, "last_batch_id": None}
            self.nodes[key] = node
        replayed = node["last_batch_id"] == batch_id
        node["source_count"] += 0 if replayed else row.get("new_signal_count", 0)
        node["last_batch_id"] = batch_id
        for name, value in row.items():
            if name in ("new_signal_count", "observed_at"):
                continue
            if isinstance(value, list):
                merged = list(node.get(name) or [])
                merged.extend(v for v in value if v not in merged)
                node[name] = merged
            elif value is not None:
                node[name] = value
        observed = row["observed_at"]
        node["first_seen"] = min(node.get("first_seen") or observed, observed)
        node["last_seen"] = max(node.get("last_seen") or observed, observed)

    def _merge_edge(self, row: dict[str, Any], batch_id: str | None) -> None:
        edge = self.edges.get(row["edge_key"])
        if edge is None:
            edge = {"evidence_count": 0, "last_batch_id": None}
            self.edges[row["edge_key"]] = edge
        replayed = edge["last_batch_id"] == batch_id
        edge["evidence_count"] += 0 if replayed else row.get("new_evidence", 0)
        edge["last_batch_id"] = batch_id
        edge.update({k: v for k, v in row.items() if v is not None})


class TestIdempotence:
    """The same batch twice must leave the graph exactly as it was."""

    async def test_the_same_merge_twice_yields_one_node(self) -> None:
        graph = FakeGraph()
        writer = GraphWriter(graph)
        batch = GraphBatch(nodes=(company(),))

        await writer.apply(batch)
        await writer.apply(batch)

        assert list(graph.nodes) == [("Company", "ent_acme")]

    async def test_a_replayed_batch_does_not_double_count_evidence(self) -> None:
        """The failure `MERGE` alone does not prevent.

        `backend/db/neo4j.py` warns that `SET n.c = n.c + 1` must not be routed
        through a managed transaction because a retry applies it twice. The
        content-derived batch id is what makes it safe, and it covers Kafka
        redelivery in the same stroke.
        """
        graph = FakeGraph()
        writer = GraphWriter(graph)
        batch = GraphBatch(nodes=(company(delta=3),))

        await writer.apply(batch)
        await writer.apply(batch)

        assert graph.nodes[("Company", "ent_acme")]["source_count"] == 3

    async def test_a_genuinely_new_batch_does_advance_the_counter(self) -> None:
        """The guard must suppress replays without suppressing real evidence."""
        graph = FakeGraph()
        writer = GraphWriter(graph)

        await writer.apply(GraphBatch(nodes=(company(delta=3),)))
        await writer.apply(GraphBatch(nodes=(company(delta=2),)))

        assert graph.nodes[("Company", "ent_acme")]["source_count"] == 5

    def test_the_batch_id_is_content_derived_and_order_independent(self) -> None:
        """A random id would make every replay look like new evidence.

        Order independence matters because two workers handed the same set in
        different orders must agree, or the guard never fires.
        """
        first = GraphBatch(nodes=(company("a"), company("b")))
        second = GraphBatch(nodes=(company("b"), company("a")))
        assert batch_id_for(first) == batch_id_for(second)

    def test_a_different_delta_is_a_different_batch(self) -> None:
        """Two new signals for Acme is genuinely not the same batch as three."""
        assert batch_id_for(GraphBatch(nodes=(company(delta=2),))) != batch_id_for(
            GraphBatch(nodes=(company(delta=3),))
        )

    async def test_duplicate_rows_in_one_batch_collapse_and_sum(self) -> None:
        """Two rows for one id inside one `UNWIND` both run.

        The counter would advance twice for one entity, and `count(n)` would
        report 2 where one node exists -- tripping the write verification on a
        batch that is actually fine.
        """
        graph = FakeGraph()
        writer = GraphWriter(graph)

        outcome = await writer.apply(
            GraphBatch(
                nodes=(
                    company(delta=2, aliases=("Acme Inc",)),
                    company(delta=3, aliases=("ACME",), ticker="ACME"),
                )
            )
        )

        assert outcome.nodes_written == 1
        node = graph.nodes[("Company", "ent_acme")]
        assert node["source_count"] == 5
        assert node["aliases"] == ["Acme Inc", "ACME"]
        assert node["ticker"] == "ACME"

    async def test_duplicate_edges_in_one_batch_collapse(self) -> None:
        graph = FakeGraph()
        writer = GraphWriter(graph)
        outcome = await writer.apply(
            GraphBatch(signals=(stub(),), nodes=(company(),), edges=(mentions(), mentions()))
        )
        assert outcome.edges_written == 1
        assert len(graph.edges) == 1


class TestSignalStubStaysAStub:
    """`docs/data-stores.md` §3.2: nothing may live only in Neo4j."""

    def test_the_stub_carries_exactly_five_properties(self) -> None:
        assert SIGNAL_STUB_PROPERTIES == (
            "id",
            "tenant_id",
            "published_at",
            "source",
            "platform",
        )

    def test_the_stub_fragment_writes_nothing_else(self) -> None:
        """No title, no snippet, no text, no vector -- and no way to add one.

        Every one of those looks reasonable in isolation and each converts a
        derived store into a system of record.
        """
        query = signal_stub_cypher()
        assigned = set(re.findall(r"s\.(\w+) =", query))
        assert assigned == {"tenant_id", "published_at", "source", "platform", "created_at"}
        for forbidden in ("text", "title", "body", "content", "embedding", "snippet"):
            assert forbidden not in query

    def test_the_stub_has_no_extension_point(self) -> None:
        """The absence of a `properties` map is the enforcement."""
        assert not hasattr(stub(), "properties")

    def test_the_stub_is_frozen(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            stub().id = "other"  # type: ignore[misc]

    async def test_applying_a_stub_twice_is_naturally_idempotent(self) -> None:
        """A stub carries no counters, so it needs no replay guard."""
        graph = FakeGraph()
        writer = GraphWriter(graph)
        await writer.apply(GraphBatch(signals=(stub(),)))
        await writer.apply(GraphBatch(signals=(stub(),)))
        assert list(graph.signals) == ["sig_1"]

    def test_no_batch_id_is_sent_with_a_stub(self) -> None:
        """Passing an unused parameter would be harmless and misleading."""
        writer = GraphWriter(FakeGraph())
        (planned,) = writer.plan(GraphBatch(signals=(stub(),))).statements
        assert set(planned.parameters) == {"rows"}


class TestEndpointsAreRefusedAtConstruction:
    """An edge is validated before it can be buffered, not before it is flushed."""

    def test_competes_with_between_a_company_and_a_region_is_refused(self) -> None:
        with pytest.raises(GraphSchemaError, match="cannot connect"):
            EdgeWrite(
                edge_type=EdgeType.COMPETES_WITH,
                source_label="Company",
                source_id="ent_acme",
                target_label="Region",
                target_id="reg_be",
                tenant_id="t1",
                valid_from=NOW,
                observed_at=NOW,
            )

    def test_the_refusal_happens_before_any_query_is_built(self) -> None:
        """Otherwise a bad edge sits in the batcher's buffer until flush time.

        By then the row is one of five hundred, the consumer has moved on, and
        the offending extraction is gone from the logs.
        """
        graph = FakeGraph()
        with pytest.raises(GraphSchemaError):
            EdgeWrite(
                edge_type=EdgeType.USES,
                source_label="Technology",
                source_id="tech_go",
                target_label="Company",
                target_id="ent_acme",
                tenant_id="t1",
                valid_from=NOW,
                observed_at=NOW,
            )
        assert graph.transactions == []

    def test_a_symmetric_edge_is_stored_once_from_either_direction(self) -> None:
        """Two mirrored copies drift the moment one side is updated."""
        forward = EdgeWrite(
            edge_type=EdgeType.COMPETES_WITH,
            source_label="Company",
            source_id="ent_zulu",
            target_label="Company",
            target_id="ent_acme",
            tenant_id="t1",
            valid_from=NOW,
            observed_at=NOW,
        )
        backward = EdgeWrite(
            edge_type=EdgeType.COMPETES_WITH,
            source_label="Company",
            source_id="ent_acme",
            target_label="Company",
            target_id="ent_zulu",
            tenant_id="t1",
            valid_from=NOW,
            observed_at=NOW,
        )
        assert forward.key == backward.key
        assert forward.oriented == backward.oriented

    def test_a_negative_delta_is_refused(self) -> None:
        """A counter that only accumulates must never be handed a decrement."""
        with pytest.raises(GraphSchemaError, match="negative delta"):
            company(delta=-1)

    def test_a_field_repeated_in_properties_is_refused(self) -> None:
        """The map would win and the field would be silently ignored."""
        with pytest.raises(GraphSchemaError, match="both as a field and in properties"):
            company(canonical_name="Other")


class TestStatementsAreParameterised:
    def test_a_hostile_entity_name_never_reaches_the_query_text(self) -> None:
        """The realistic attack: a scraped company name.

        Entity names come from third-party content that nobody in this repository
        controls, and they are written to the graph verbatim.
        """
        writer = GraphWriter(FakeGraph())
        plan = writer.plan(GraphBatch(nodes=(company(name=HOSTILE),)))
        (planned,) = plan.statements
        assert HOSTILE not in planned.cypher
        assert planned.parameters["rows"][0]["canonical_name"] == HOSTILE

    def test_a_hostile_id_never_reaches_the_query_text(self) -> None:
        writer = GraphWriter(FakeGraph())
        plan = writer.plan(GraphBatch(nodes=(company(HOSTILE),)))
        assert HOSTILE not in plan.statements[0].cypher

    def test_every_statement_carries_only_declared_parameters(self) -> None:
        writer = GraphWriter(FakeGraph())
        plan = writer.plan(
            GraphBatch(nodes=(company(),), signals=(stub(),), edges=(mentions(),))
        )
        for planned in plan.statements:
            named = set(re.findall(r"\$(\w+)", planned.cypher))
            assert named == set(planned.parameters)


class TestOrderingAndVerification:
    async def test_nodes_and_stubs_are_written_before_edges_in_one_transaction(
        self,
    ) -> None:
        """An edge whose endpoint does not exist yet is silently dropped.

        `MATCH` is not an assertion: the row contributes no relationship and
        raises nothing.
        """
        graph = FakeGraph()
        await GraphWriter(graph).apply(
            GraphBatch(nodes=(company(),), signals=(stub(),), edges=(mentions(),))
        )

        (transaction,) = graph.transactions
        kinds = [
            "node" if "MERGE (n:" in q else "signal" if "MERGE (s:" in q else "edge"
            for q, _ in transaction
        ]
        assert kinds == ["node", "signal", "edge"]

    async def test_rows_are_sorted_by_id_for_lock_ordering(self) -> None:
        """Two workers taking locks in opposite orders deadlock on each other."""
        writer = GraphWriter(FakeGraph())
        plan = writer.plan(
            GraphBatch(nodes=(company("ent_zulu"), company("ent_acme"), company("ent_mid")))
        )
        ids = [row["id"] for row in plan.statements[0].parameters["rows"]]
        assert ids == sorted(ids)

    async def test_a_missing_endpoint_is_reported_rather_than_lost(self) -> None:
        """The only genuinely invisible graph failure, made loud."""
        graph = FakeGraph()
        graph.missing_endpoints = {"ent_ghost"}
        writer = GraphWriter(graph)

        with pytest.raises(GraphWriteError, match="wrote 0 of 1 edges"):
            await writer.apply(
                GraphBatch(signals=(stub(),), edges=(mentions(entity_id="ent_ghost"),))
            )

    async def test_the_missing_endpoint_error_names_candidate_ids(self) -> None:
        """The message is read by whoever is holding a broken pipeline."""
        graph = FakeGraph()
        graph.missing_endpoints = {"ent_ghost"}
        writer = GraphWriter(graph)
        with pytest.raises(GraphWriteError, match="ent_ghost"):
            await writer.apply(
                GraphBatch(signals=(stub(),), edges=(mentions(entity_id="ent_ghost"),))
            )

    async def test_an_empty_batch_does_not_open_a_transaction(self) -> None:
        """The batcher flushes on a timer; a quiet period must cost nothing."""
        graph = FakeGraph()
        outcome = await GraphWriter(graph).apply(GraphBatch())
        assert graph.transactions == []
        assert outcome.total == 0

    async def test_a_runner_returning_the_wrong_shape_is_caught(self) -> None:
        async def bad_runner(statements: Any) -> list[list[dict[str, Any]]]:
            return []

        with pytest.raises(GraphWriteError, match="result sets"):
            await GraphWriter(bad_runner).apply(GraphBatch(nodes=(company(),)))

    async def test_edges_are_grouped_by_label_pair(self) -> None:
        """Cypher cannot parameterise a label, and an unlabelled MATCH is a scan.

        One statement per `(type, from_label, to_label)` is what lets both
        endpoints seek their id index.
        """
        writer = GraphWriter(FakeGraph())
        plan = writer.plan(
            GraphBatch(
                edges=(
                    mentions(),
                    EdgeWrite(
                        edge_type=EdgeType.MENTIONS,
                        source_label="Signal",
                        source_id="sig_1",
                        target_label="Topic",
                        target_id="top_1",
                        tenant_id="t1",
                        valid_from=NOW,
                        observed_at=NOW,
                        evidence_key="sig_1",
                    ),
                )
            )
        )
        labels = [p.label for p in plan.statements]
        assert labels == [
            "Signal-[MENTIONS]->Company",
            "Signal-[MENTIONS]->Topic",
        ]


class TestSessionAdapter:
    """The ten lines that connect an L1 library to `backend/db/neo4j.py`."""

    async def test_statements_run_in_one_managed_transaction(self) -> None:
        """`execute_write` is the managed form: the driver retries transients.

        The fake mirrors the shape used by `tests/unit/backend/db/test_neo4j.py`.
        """
        seen: dict[str, Any] = {"queries": [], "modes": 0}

        class FakeRecord:
            def data(self) -> dict[str, Any]:
                return {"written": 1}

        class FakeResult:
            def __aiter__(self) -> Any:
                async def gen() -> Any:
                    yield FakeRecord()

                return gen()

        class FakeTx:
            async def run(self, query: str, parameters: Any = None) -> FakeResult:
                seen["queries"].append((query, parameters))
                return FakeResult()

        class FakeSession:
            async def __aenter__(self) -> FakeSession:
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            async def execute_write(self, work: Any) -> Any:
                seen["modes"] += 1
                return await work(FakeTx())

        runner = runner_from_session_factory(FakeSession)
        results = await runner([("RETURN 1", {"a": 1}), ("RETURN 2", {"b": 2})])

        assert results == [[{"written": 1}], [{"written": 1}]]
        assert seen["modes"] == 1, "the statements did not share one transaction"
        assert [q for q, _ in seen["queries"]] == ["RETURN 1", "RETURN 2"]


class RecordingWriter(GraphWriter):
    """A `GraphWriter` that records applied batches and can be made to fail."""

    def __init__(self) -> None:
        super().__init__(FakeGraph())
        self.batches: list[GraphBatch] = []
        self.failures = 0

    async def apply(self, batch: GraphBatch) -> Any:  # type: ignore[override]
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("Neo.TransientError.Transaction.DeadlockDetected")
        self.batches.append(batch)
        return await super().apply(batch)


class TestBatcher:
    async def test_it_flushes_when_the_size_threshold_is_reached(self) -> None:
        writer = RecordingWriter()
        async with WriteBatcher(writer, max_batch_rows=3, max_age_seconds=30.0) as batcher:
            for index in range(3):
                await batcher.submit(company(f"ent_{index}"))
            # The size threshold wakes the flusher task; yield to it. The age is
            # 30s away, so anything that flushes here flushed on size.
            await asyncio.sleep(0.02)
            assert batcher.pending_rows == 0

        assert [b.row_count for b in writer.batches] == [3]

    async def test_it_flushes_on_age_when_the_size_is_never_reached(self) -> None:
        """The bug a size-only flush has.

        Two rows sitting under a 500-row threshold are invisible in the graph
        until traffic resumes -- which, at the end of a busy period, is exactly
        when nobody is looking for something *missing*.
        """
        writer = RecordingWriter()
        async with WriteBatcher(
            writer, max_batch_rows=500, max_age_seconds=0.05
        ) as batcher:
            await batcher.submit(company("ent_a"))
            await batcher.submit(company("ent_b"))

            await asyncio.sleep(0.01)
            assert writer.batches == [], "flushed before the age deadline"

            await asyncio.sleep(0.12)
            assert [b.row_count for b in writer.batches] == [2]

    async def test_the_age_clock_starts_with_the_oldest_row(self) -> None:
        """A steady trickle must not be able to postpone the deadline forever."""
        writer = RecordingWriter()
        async with WriteBatcher(
            writer, max_batch_rows=500, max_age_seconds=0.08
        ) as batcher:
            await batcher.submit(company("ent_a"))
            for index in range(3):
                await asyncio.sleep(0.03)
                await batcher.submit(company(f"ent_{index}"))
            await asyncio.sleep(0.05)
            assert writer.batches, "the deadline was pushed out by later rows"

    async def test_a_flush_takes_one_batch_not_the_whole_buffer(self) -> None:
        """One enormous transaction holds locks for its whole duration."""
        writer = RecordingWriter()
        batcher = WriteBatcher(writer, max_batch_rows=2, max_age_seconds=30.0)
        for index in range(4):
            await batcher.submit(company(f"ent_{index}"))
        await batcher.flush()

        assert [b.row_count for b in writer.batches] == [2]
        assert batcher.pending_rows == 2

    async def test_submit_blocks_when_the_buffer_is_full(self) -> None:
        """Back-pressure, not unbounded growth and not dropped rows.

        The wait is what stops the consumer calling `poll()`, which is what makes
        a slow graph show up as consumer lag instead of as an OOM kill.
        """
        writer = RecordingWriter()
        batcher = WriteBatcher(writer, max_batch_rows=2, max_age_seconds=30.0)
        for index in range(4):  # capacity is 2 batches = 4 rows
            await batcher.submit(company(f"ent_{index}"))
        assert batcher.pending_rows == batcher.capacity

        blocked = asyncio.create_task(batcher.submit(company("ent_late")))
        await asyncio.sleep(0.01)
        assert not blocked.done(), "submit accepted a row past capacity"

        await batcher.flush()
        await asyncio.wait_for(blocked, timeout=1.0)
        assert batcher.pending_rows == 3

    async def test_back_pressure_is_signalled_to_the_consumer(self) -> None:
        """`docs/knowledge-graph.md` §7: pause the consumer above two batches."""
        events: list[bool] = []
        writer = RecordingWriter()
        batcher = WriteBatcher(
            writer,
            max_batch_rows=2,
            max_age_seconds=30.0,
            on_pressure=events.append,
        )
        for index in range(4):
            await batcher.submit(company(f"ent_{index}"))
        blocked = asyncio.create_task(batcher.submit(company("ent_late")))
        await asyncio.sleep(0.01)

        assert events == [True]
        await batcher.flush()
        await asyncio.wait_for(blocked, timeout=1.0)
        await batcher.flush()
        assert events == [True, False], "the pause was never lifted"

    async def test_pressure_is_lifted_at_a_low_water_mark_not_at_capacity(self) -> None:
        """Hysteresis. Each resume costs a consumer-group round trip.

        Clearing the pause the instant the buffer drops below capacity makes a
        saturated pipeline pause and resume on alternate rows. So the pause is
        lifted at half capacity: after the first flush the buffer is below
        capacity and pressure is still asserted, which is the whole point.
        """
        events: list[bool] = []
        writer = RecordingWriter()
        batcher = WriteBatcher(
            writer,
            max_batch_rows=4,
            max_age_seconds=30.0,
            capacity_batches=3,  # capacity 12, low-water mark 6
            on_pressure=events.append,
        )
        for index in range(12):
            await batcher.submit(company(f"ent_{index:02d}"))
        blocked = asyncio.create_task(batcher.submit(company("ent_late")))
        await asyncio.sleep(0.01)
        assert events == [True]

        await batcher.flush()  # 12 -> 8, below capacity but above the mark
        assert events == [True], "the pause was lifted the moment a row left"

        await asyncio.wait_for(blocked, timeout=1.0)
        await batcher.flush()  # 9 -> 5, at last below the mark
        assert events == [True, False]

    async def test_a_failed_flush_keeps_its_rows(self) -> None:
        """Rows are the only copy of updates PostgreSQL has already accepted."""
        writer = RecordingWriter()
        writer.failures = 1
        batcher = WriteBatcher(
            writer,
            max_batch_rows=2,
            max_age_seconds=30.0,
            max_attempts=3,
            sleep=_no_sleep,
        )
        await batcher.submit(company("ent_a"))
        await batcher.flush()

        assert [b.row_count for b in writer.batches] == [1]
        assert batcher.pending_rows == 0

    async def test_a_permanently_failing_flush_raises_and_retains_rows(self) -> None:
        """Without a dead-letter sink there is nowhere safe to put them."""
        writer = RecordingWriter()
        writer.failures = 99
        batcher = WriteBatcher(
            writer, max_batch_rows=2, max_age_seconds=30.0, max_attempts=2, sleep=_no_sleep
        )
        await batcher.submit(company("ent_a"))
        with pytest.raises(RuntimeError, match="Deadlock"):
            await batcher.flush()
        assert batcher.pending_rows == 1

    async def test_a_poison_batch_goes_to_the_dead_letter_hook(self) -> None:
        """So one bad batch cannot stall the stream forever."""
        dead: list[GraphBatch] = []

        async def on_dead_letter(batch: GraphBatch, exc: Exception) -> None:
            dead.append(batch)

        writer = RecordingWriter()
        writer.failures = 99
        batcher = WriteBatcher(
            writer,
            max_batch_rows=2,
            max_age_seconds=30.0,
            max_attempts=2,
            on_dead_letter=on_dead_letter,
            sleep=_no_sleep,
        )
        await batcher.submit(company("ent_a"))
        await batcher.flush()

        assert [b.row_count for b in dead] == [1]
        assert batcher.pending_rows == 0

    async def test_close_drains_the_buffer(self) -> None:
        """Rows buffered at shutdown were accepted and are not in the graph."""
        writer = RecordingWriter()
        batcher = WriteBatcher(writer, max_batch_rows=100, max_age_seconds=30.0)
        batcher.start()
        await batcher.submit(company("ent_a"))
        await batcher.aclose()

        assert [b.row_count for b in writer.batches] == [1]
        assert batcher.pending_rows == 0

    async def test_submit_after_close_raises(self) -> None:
        """A consumer that treats a closed batcher as delivered commits a lie."""
        batcher = WriteBatcher(RecordingWriter(), max_batch_rows=2, max_age_seconds=30.0)
        batcher.start()
        await batcher.aclose()
        with pytest.raises(BatcherClosedError):
            await batcher.submit(company())

    async def test_a_blocked_submitter_is_released_by_close(self) -> None:
        """Leaving it blocked turns a shutdown into a hang."""
        writer = RecordingWriter()
        batcher = WriteBatcher(writer, max_batch_rows=1, max_age_seconds=30.0)
        for index in range(2):
            await batcher.submit(company(f"ent_{index}"))
        blocked = asyncio.create_task(batcher.submit(company("ent_late")))
        await asyncio.sleep(0.01)
        assert not blocked.done()

        batcher.start()
        await batcher.aclose()
        with pytest.raises(BatcherClosedError):
            await asyncio.wait_for(blocked, timeout=1.0)

    async def test_a_mixed_stream_is_partitioned_into_one_batch(self) -> None:
        writer = RecordingWriter()
        batcher = WriteBatcher(writer, max_batch_rows=10, max_age_seconds=30.0)
        await batcher.submit_many([company(), stub(), mentions()])
        await batcher.flush()

        (batch,) = writer.batches
        assert (len(batch.nodes), len(batch.signals), len(batch.edges)) == (1, 1, 1)

    def test_a_zero_age_is_refused(self) -> None:
        """It would flush on every row -- the per-row transaction to be avoided."""
        with pytest.raises(ValueError, match="max_age_seconds must be positive"):
            WriteBatcher(RecordingWriter(), max_age_seconds=0.0)

    def test_a_capacity_below_one_batch_is_refused(self) -> None:
        """`submit` would block before a single batch could be assembled."""
        with pytest.raises(ValueError, match="capacity_batches"):
            WriteBatcher(RecordingWriter(), capacity_batches=0)


async def _no_sleep(_seconds: float) -> None:
    """Injected in place of `asyncio.sleep` so backoff costs no wall time.

    The backoff is exponential with jitter and starts at half a second; a test
    that waited for it would spend seconds proving arithmetic.
    """
    return None


class TestObservedTimes:
    async def test_first_seen_moves_backwards_on_a_backfill(self) -> None:
        """A backfill of older signals must widen the window, not be discarded."""
        graph = FakeGraph()
        writer = GraphWriter(graph)
        await writer.apply(GraphBatch(nodes=(company(observed_at=NOW),)))
        earlier = NOW - timedelta(days=30)
        await writer.apply(GraphBatch(nodes=(company(observed_at=earlier, delta=1),)))

        node = graph.nodes[("Company", "ent_acme")]
        assert node["first_seen"] == earlier
        assert node["last_seen"] == NOW

    async def test_last_seen_does_not_move_backwards_on_an_out_of_order_replay(
        self,
    ) -> None:
        graph = FakeGraph()
        writer = GraphWriter(graph)
        await writer.apply(GraphBatch(nodes=(company(observed_at=NOW),)))
        await writer.apply(
            GraphBatch(nodes=(company(observed_at=NOW - timedelta(days=1), delta=1),))
        )
        assert graph.nodes[("Company", "ent_acme")]["last_seen"] == NOW
