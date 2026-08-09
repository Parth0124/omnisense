"""Neo4j integration: the two properties `docs/knowledge-graph.md` §11 demands.

> Two properties every integration test must assert: **replay idempotence**
> (applying the same batch twice leaves node and edge counts unchanged) and
> **temporal correctness** (an as-of query at time T never returns an edge whose
> interval excludes T).

Both are unprovable in a unit test. Replay idempotence depends on `MERGE`
semantics and on a uniqueness constraint being present — a fake runner will
happily report success for a batch that would have created duplicates against a
real database. Temporal correctness depends on how Neo4j compares a stored
`datetime` against a parameter, which is exactly where a driver-level type
mismatch hides.

Everything here is scoped to a per-run tenant, so these can be run against a
local graph that already has data in it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from models.enums import EdgeType, EntityType

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def _batch(tenant: str, *, evidence: int = 1):
    """Two companies, a signal stub, and a competes-with edge between them."""
    from graph.ingest.writer import EdgeWrite, GraphBatch, NodeWrite, SignalStub

    acme = NodeWrite(
        entity_type=EntityType.COMPANY,
        id=f"{tenant}_acme",
        tenant_id=tenant,
        canonical_name="Acme Corporation",
        normalized_name="acme",
        observed_at=NOW,
        aliases=["Acme", "ACME Corp"],
        new_signal_count=1,
    )
    globex = NodeWrite(
        entity_type=EntityType.COMPANY,
        id=f"{tenant}_globex",
        tenant_id=tenant,
        canonical_name="Globex",
        normalized_name="globex",
        observed_at=NOW,
        new_signal_count=1,
    )
    stub = SignalStub(
        id=f"{tenant}_sig1",
        tenant_id=tenant,
        published_at=NOW,
        source="news",
        platform="rss",
    )
    edge = EdgeWrite(
        edge_type=EdgeType.COMPETES_WITH,
        source_label="Company",
        source_id=f"{tenant}_acme",
        target_label="Company",
        target_id=f"{tenant}_globex",
        tenant_id=tenant,
        valid_from=NOW,
        observed_at=NOW,
        confidence=0.8,
        extractor="integration-test",
        source_signal_ids=[f"{tenant}_sig1"],
        new_evidence=evidence,
        evidence_key=f"{tenant}_sig1:competes",
    )
    return GraphBatch(nodes=(acme, globex), signals=(stub,), edges=(edge,))


async def _counts(client, tenant: str) -> dict[str, int]:
    from graph.queries.cypher import Query

    rows = await client.fetch(
        Query(
            """
MATCH (n {tenant_id: $tenant})
WITH count(n) AS nodes
MATCH ()-[r {tenant_id: $tenant}]->()
RETURN nodes, count(r) AS edges
""".strip(),
            {"tenant": tenant},
        )
    )
    return rows[0] if rows else {"nodes": 0, "edges": 0}


class TestReplayIdempotence:
    """`docs/knowledge-graph.md` §11, property one."""

    async def test_applying_the_same_batch_twice_changes_nothing(
        self, graph_writer, graph_client, clean_graph
    ) -> None:
        """The property Kafka's at-least-once delivery makes non-negotiable.

        Every worker in this system will re-deliver a message after a crash, a
        rebalance or an uncommitted batch. If a second application created a
        second node, the graph would accumulate duplicates in proportion to how
        often workers restart — which is highest exactly when the system is
        under stress.
        """
        tenant = clean_graph

        await graph_writer.apply(_batch(tenant))
        first = await _counts(graph_client, tenant)

        await graph_writer.apply(_batch(tenant))
        second = await _counts(graph_client, tenant)

        assert first == second, (
            f"replay changed the graph: {first} -> {second}. MERGE is not "
            "idempotent here, which means every worker restart duplicates data."
        )
        assert first["nodes"] == 3  # two companies plus the signal stub
        assert first["edges"] >= 1

    async def test_the_source_count_does_not_double_on_replay(
        self, graph_writer, graph_client, clean_graph
    ) -> None:
        """`source_count` is a *delta* accumulation, guarded by the batch id.

        This is the counter a report uses as its cheapest anti-hallucination
        signal. If a redelivery incremented it, an entity mentioned once by one
        article would claim two independent sources after a single worker
        restart — and nothing downstream could detect it.
        """
        from graph.queries.cypher import Query

        tenant = clean_graph
        await graph_writer.apply(_batch(tenant))
        await graph_writer.apply(_batch(tenant))

        rows = await graph_client.fetch(
            Query(
                "MATCH (n:Company {id: $id}) RETURN n.source_count AS count",
                {"id": f"{tenant}_acme"},
            )
        )
        assert rows[0]["count"] == 1, (
            f"source_count is {rows[0]['count']} after two applications of one "
            "batch; the batch-id guard is not holding."
        )


class TestTemporalCorrectness:
    """`docs/knowledge-graph.md` §11, property two."""

    async def test_an_as_of_query_excludes_an_edge_that_had_not_started(
        self, graph_writer, graph_client, clean_graph
    ) -> None:
        """An interval that begins after T must not appear in an as-of T read.

        The failure this catches is a driver-level type mismatch: if `valid_from`
        is stored as a string and compared against a datetime parameter, Neo4j
        returns nothing for *every* as-of query — or, worse, returns everything.
        A fake runner cannot exhibit either.
        """
        from graph.queries.cypher import competitors_of

        tenant = clean_graph
        await graph_writer.apply(_batch(tenant))

        before = await graph_client.fetch(
            competitors_of(
                tenant_id=tenant, name="Acme Corporation", as_of=NOW - timedelta(days=1)
            )
        )
        assert before == [], "an edge valid from NOW was returned for a query at NOW-1d"

    async def test_an_as_of_query_includes_a_currently_valid_edge(
        self, graph_writer, graph_client, clean_graph
    ) -> None:
        from graph.queries.cypher import competitors_of

        tenant = clean_graph
        await graph_writer.apply(_batch(tenant))

        current = await graph_client.fetch(
            competitors_of(
                tenant_id=tenant, name="Acme Corporation", as_of=NOW + timedelta(days=1)
            )
        )
        assert len(current) == 1
        assert current[0]["name"] == "Globex"

    async def test_an_alias_finds_the_entity(
        self, graph_writer, graph_client, clean_graph
    ) -> None:
        """"Big Blue finds IBM" — the reason `aliases` is in the match at all.

        Only provable against a real store: the query does a list membership test
        inside a disjunction, and whether Neo4j evaluates that the way the
        template assumes is not something a fake can tell you.
        """
        from graph.queries.cypher import competitors_of

        tenant = clean_graph
        await graph_writer.apply(_batch(tenant))

        by_alias = await graph_client.fetch(
            competitors_of(tenant_id=tenant, name="ACME Corp", as_of=NOW + timedelta(hours=1))
        )
        assert len(by_alias) == 1, "an alias did not resolve to its entity"


class TestTenantIsolation:
    async def test_a_query_never_crosses_a_tenant(
        self, graph_writer, graph_client, clean_graph
    ) -> None:
        """The property the whole multi-tenancy design rests on.

        Written with two tenants in the same database, because that is the only
        arrangement in which a missing `tenant_id` predicate is visible — with
        one tenant's data present, every query looks correctly scoped.
        """
        from backend.db.neo4j import run_write
        from graph.queries.cypher import competitors_of

        tenant = clean_graph
        other = f"{tenant}_other"
        await graph_writer.apply(_batch(tenant))
        await graph_writer.apply(_batch(other))

        try:
            results = await graph_client.fetch(
                competitors_of(
                    tenant_id=tenant, name="Acme Corporation", as_of=NOW + timedelta(hours=1)
                )
            )
            assert len(results) == 1
            assert results[0]["id"].startswith(tenant)
            assert not results[0]["id"].startswith(other)
        finally:
            await run_write(
                "MATCH (n {tenant_id: $tenant}) DETACH DELETE n", {"tenant": other}
            )


class TestSchemaConstraints:
    async def test_the_uniqueness_constraint_exists(self, graph_client, neo4j_available) -> None:
        """Without it `MERGE` is not a lock and two concurrent writers both create.

        Checked explicitly because the failure is invisible under low
        concurrency: everything works in development and duplicates appear the
        first time two workers process the same entity at once.
        """
        from graph.queries.cypher import Query

        rows = await graph_client.fetch(
            Query("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties", {})
        )
        company = [
            row
            for row in rows
            if "Company" in (row.get("labelsOrTypes") or [])
            and "id" in (row.get("properties") or [])
        ]
        assert company, (
            "no uniqueness constraint on Company.id. Run `make init-db` -- without "
            "it MERGE cannot serialise concurrent writers and the graph will "
            "accumulate duplicate entities under load."
        )
