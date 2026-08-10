"""Unit tests for `services/graph_service.py`.

The service's whole job is translation, so the tests are about translation: a
`GraphSchemaError` must become a 422 and not a 500, a `GraphQueryError` must
become a 502 and not a 503, and an unreachable graph must become either a 503 or
an empty list depending on which call site asked.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from backend.core.exceptions import (
    DependencyUnavailableError,
    ExternalServiceError,
    NotFoundError,
    ValidationError,
)
from graph.client import GraphClient, GraphQueryError, GraphUnavailableError
from models.enums import EdgeType, EntityType
from services.graph_service import (
    Entity,
    GraphService,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)


def _service(rows: list[dict[str, Any]] | None = None) -> GraphService:
    """A service over a real `GraphClient` and a scripted runner.

    The runner is the *driver adapter*, so it is the right place to model a
    successful read. Client-level failures (`GraphUnavailableError`,
    `GraphQueryError`) are produced by the client from driver exceptions, never
    raised by a runner -- `_failing_service` models those at the layer that
    actually emits them.
    """

    async def run(cypher: str, parameters: Any = None) -> list[dict[str, Any]]:
        return list(rows or [])

    return GraphService(GraphClient(run, max_attempts=1))


class _FailingClient:
    """Duck-typed stand-in raising a client-level failure from `fetch`."""

    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def fetch(self, query: Any) -> list[dict[str, Any]]:
        raise self._error


def _failing_service(error: BaseException) -> GraphService:
    return GraphService(_FailingClient(error))  # type: ignore[arg-type]


class TestErrorTranslation:
    async def test_bad_argument_is_a_422_not_a_500(self) -> None:
        """`GraphSchemaError` is raised at *build* time. Building outside the try
        block would let it escape as a raw ValueError and become a 500 -- sending
        an engineer to look at a database that is working perfectly."""
        with pytest.raises(ValidationError):
            await _service().search_entities(tenant_id="", query="x")

    async def test_over_limit_is_a_422(self) -> None:
        with pytest.raises(ValidationError, match="MAX_LIMIT"):
            await _service().search_entities(tenant_id="t", query="x", limit=100_000)

    async def test_unreachable_graph_is_a_503(self) -> None:
        service = _failing_service(GraphUnavailableError("down"))
        with pytest.raises(DependencyUnavailableError):
            await service.search_entities(tenant_id="t", query="x")

    async def test_rejected_query_is_a_502_not_a_503(self) -> None:
        """A 503 is retried by clients, so classifying a syntax error as one
        turns a single bug into a retry loop."""
        service = _failing_service(GraphQueryError("bad cypher"))
        with pytest.raises(ExternalServiceError) as caught:
            await service.search_entities(tenant_id="t", query="x")
        assert caught.value.status_code == 502

    async def test_missing_entity_is_a_404(self) -> None:
        """The layer where `None` becomes a decision. `graph/client.py` cannot
        make it -- 'no rows' is ambiguous there."""
        with pytest.raises(NotFoundError):
            await _service([]).get_entity(tenant_id="t", entity_id="nope")


class TestDegradation:
    async def test_search_never_degrades(self) -> None:
        """`/graph/search` *is* the graph. An empty list returned because Neo4j
        is unreachable is indistinguishable from 'no such company' -- a wrong
        answer wearing the clothes of a right one."""
        service = _failing_service(GraphUnavailableError("down"))
        with pytest.raises(DependencyUnavailableError):
            await service.search_entities(tenant_id="t", query="acme")

    async def test_neighbourhood_degrades_by_default(self) -> None:
        """GraphRAG expansion is one backend of three. Losing it costs recall,
        not correctness."""
        service = _failing_service(GraphUnavailableError("down"))
        result = await service.neighbourhood(
            tenant_id="t", seed_ids=["a"], start=NOW - timedelta(days=1), end=NOW
        )
        assert result == []

    async def test_citations_never_degrade(self) -> None:
        """An empty citation list produces a report claim with no support, which
        is exactly what evidence verification exists to prevent."""
        service = _failing_service(GraphUnavailableError("down"))
        with pytest.raises(DependencyUnavailableError):
            await service.signals_for_entity(tenant_id="t", entity_id="e")

    async def test_topic_activity_degrades_only_when_asked(self) -> None:
        """The two call sites genuinely differ, which is why this is an argument
        rather than a blanket try/except.

        Topic activity feeding an investigation is one input among several, and
        losing it should cost confidence rather than the whole run. The same call
        made without `allow_degraded` must still raise -- a caller that treats an
        outage as "no activity" has turned a wrong answer into a plausible one.
        """
        service = _failing_service(GraphUnavailableError("down"))
        with pytest.raises(DependencyUnavailableError):
            await service.topic_activity(
                tenant_id="t", since=NOW - timedelta(days=1), allow_degraded=False
            )
        assert (
            await service.topic_activity(
                tenant_id="t", since=NOW - timedelta(days=1), allow_degraded=True
            )
            == []
        )

    async def test_a_rejected_query_is_never_degraded(self) -> None:
        """Degrading past a malformed query hides it permanently rather than for
        the duration of an outage."""
        service = _failing_service(GraphQueryError("bad"))
        with pytest.raises(ExternalServiceError):
            await service.neighbourhood(
                tenant_id="t",
                seed_ids=["a"],
                start=NOW - timedelta(days=1),
                end=NOW,
                allow_degraded=True,
            )


class TestRowMapping:
    def test_null_score_stays_none(self) -> None:
        """0.0 means 'assessed and negligible'; None means 'nobody assessed it'.
        A UI that renders None as 0.0 asserts something the graph never said."""
        assert Entity.from_row({"id": "x", "name": "X", "confidence": None}).confidence is None

    def test_zero_score_is_preserved_as_zero(self) -> None:
        assert Entity.from_row({"id": "x", "name": "X", "confidence": 0.0}).confidence == 0.0

    def test_booleans_are_not_mistaken_for_numbers(self) -> None:
        """`bool` is a subclass of `int` in Python, so an unguarded isinstance
        check turns `True` into 1.0 and reports a confidence nobody assessed."""
        assert Entity.from_row({"id": "x", "name": "X", "confidence": True}).confidence is None

    def test_unknown_label_degrades_rather_than_crashing_a_search(self) -> None:
        """Tolerant on the way in is right for a reader: a label written by a
        newer producer must not crash search. Strictness lives on the write and
        filter paths."""
        assert Entity.from_row({"id": "x", "type": "Spaceship"}).type is EntityType.UNKNOWN

    def test_never_computed_analytics_read_as_stale(self) -> None:
        assert Entity.from_row({"id": "x"}).analytics_are_stale

    def test_analytics_older_than_the_last_mention_are_stale(self) -> None:
        """A PageRank computed before the entity absorbed forty new mentions is a
        plausible number describing a graph that no longer exists."""
        entity = Entity.from_row(
            {"id": "x", "computed_at": NOW - timedelta(days=2), "last_seen": NOW}
        )
        assert entity.analytics_are_stale

    def test_fresh_analytics_are_not_stale(self) -> None:
        entity = Entity.from_row(
            {"id": "x", "computed_at": NOW, "last_seen": NOW - timedelta(days=2)}
        )
        assert not entity.analytics_are_stale

    def test_malformed_lists_do_not_crash_the_mapper(self) -> None:
        assert Entity.from_row({"id": "x", "aliases": "not-a-list"}).aliases == ()


class TestQueryWiring:
    async def test_search_returns_typed_entities(self) -> None:
        service = _service([{"id": "e1", "name": "Acme", "type": "Company", "score": 2.5}])
        results = await service.search(tenant_id="t", query="acme")
        assert results[0].type is EntityType.COMPANY
        assert results[0].score == 2.5

    async def test_an_empty_result_is_a_valid_answer(self) -> None:
        """Distinct from an outage, which raises. `graph/client.py` returns `None`
        for an empty result because at that layer "not in the graph" is an
        ordinary answer; this layer decides which absences are 404s and which are
        empty lists, and a search that matched nothing is the latter."""
        assert await _service([]).search(tenant_id="t", query="acme") == []


class TestAgentPort:
    """`GraphService` must satisfy `agents.tools.graph_tools.GraphReader`.

    The three port dataclasses are duplicated across the layer boundary --
    `services/` is L2 and `agents/` is L3, so importing them would invert the
    dependency. These tests are what turns that duplication from a hazard into a
    checked invariant: a field renamed on one side fails here rather than as an
    `AttributeError` inside a running investigation.
    """

    def test_field_names_match_the_agent_side_shapes(self) -> None:
        import dataclasses

        from agents.tools import graph_tools
        from services import graph_service

        for name in ("EntityRef", "GraphFactRecord", "GraphPath"):
            ours = {f.name for f in dataclasses.fields(getattr(graph_service, name))}
            theirs = {f.name for f in dataclasses.fields(getattr(graph_tools, name))}
            assert ours == theirs, f"{name} drifted: {ours ^ theirs}"

    def test_search_entities_returns_the_lean_shape(self) -> None:
        """Agents get a name, a type and a mention count. Sending them a
        description and a pagerank_score on every hit spends tokens on fields no
        prompt references."""
        import asyncio

        service = _service([{"id": "e1", "name": "Acme", "type": "Company", "source_count": 4}])
        refs = asyncio.run(service.search_entities("acme", tenant_id="t"))
        assert refs[0].entity_id == "e1"
        assert refs[0].mention_count == 4

    async def test_neighbours_maps_edges_to_facts(self) -> None:
        service = _service(
            [
                {
                    "subject_id": "a",
                    "subject_name": "Acme",
                    "predicate": "COMPETES_WITH",
                    "object_id": "b",
                    "object_name": "Globex",
                    "confidence": 0.8,
                    "supporting_signal_ids": ["sig_1"],
                }
            ]
        )
        facts = await service.neighbours("a", tenant_id="t")
        assert facts[0].predicate is EdgeType.COMPETES_WITH
        assert facts[0].supporting_signal_ids == ("sig_1",)

    async def test_unknown_predicate_degrades_rather_than_crashing(self) -> None:
        """An edge type written by a newer producer must not end an
        investigation mid-run. Strict rejection lives on the write path."""
        service = _service(
            [{"subject_id": "a", "predicate": "TELEPATHICALLY_LINKED", "object_id": "b"}]
        )
        facts = await service.neighbours("a", tenant_id="t")
        assert facts[0].predicate is EdgeType.UNKNOWN

    async def test_paths_carry_hop_count(self) -> None:
        service = _service(
            [
                {
                    "entity_ids": ["a", "b", "c"],
                    "entity_names": ["A", "B", "C"],
                    "predicates": ["COMPETES_WITH", "ACQUIRED"],
                    "confidence": 0.4,
                }
            ]
        )
        paths = await service.find_paths("a", "c", tenant_id="t")
        assert paths[0].hops == 2

    async def test_neighbourhood_reads_never_degrade(self) -> None:
        """An empty neighbourhood is a meaningful answer -- "this entity has no
        recorded relationships". Returning [] because Neo4j was unreachable makes
        an outage indistinguishable from a finding, and the Competitor agent
        reports the finding."""
        service = _failing_service(GraphUnavailableError("down"))
        with pytest.raises(DependencyUnavailableError):
            await service.neighbours("a", tenant_id="t")
