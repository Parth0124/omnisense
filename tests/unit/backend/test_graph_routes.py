"""Route tests for `/api/v1/graph/*`.

No Neo4j. `get_graph_service` is overridden with a `GraphService` built over a
scripted runner, so every route is exercised end to end -- auth, validation,
serialisation -- against known rows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from backend.api.deps import mint_access_token
from backend.api.v1.graph import get_graph_service
from backend.main import create_app
from graph.client import GraphClient, GraphUnavailableError
from services.graph_service import GraphService

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)
TENANT = "tenant-1"


def _service_returning(rows: list[dict[str, Any]]) -> GraphService:
    async def run(cypher: str, parameters: Any = None) -> list[dict[str, Any]]:
        return list(rows)

    return GraphService(GraphClient(run, max_attempts=1))


class _FailingClient:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def fetch(self, query: Any) -> list[dict[str, Any]]:
        raise self._error


def _client(service: GraphService, *, scopes: tuple[str, ...] = ("graph:read",)):
    app = create_app()
    app.dependency_overrides[get_graph_service] = lambda: service
    token = mint_access_token(subject="u1", tenant_id=TENANT, scopes=scopes)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    )


ENTITY_ROW = {
    "id": "e1",
    "name": "Acme",
    "type": "Company",
    "description": "Battery maker",
    "aliases": ["ACME Corp"],
    "source_count": 12,
    "score": 3.2,
    "confidence": 0.9,
    "pagerank_score": 0.04,
    "computed_at": NOW,
    "last_seen": NOW,
}


class TestAuth:
    async def test_search_requires_a_credential(self) -> None:
        app = create_app()
        app.dependency_overrides[get_graph_service] = lambda: _service_returning([])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/api/v1/graph/search?q=acme")).status_code == 401

    async def test_the_wrong_scope_is_a_403(self) -> None:
        async with _client(_service_returning([]), scopes=("signals:read",)) as client:
            response = await client.get("/api/v1/graph/search?q=acme")
            assert response.status_code == 403
            assert "graph:read" in response.text

    async def test_tenant_is_never_a_query_parameter(self) -> None:
        """The property that makes multi-tenancy an authorization boundary rather
        than a filter: the only way to read another tenant's graph is to hold a
        token for it."""
        import backend.api.v1.graph as module

        assert "tenant_id" not in module.search_entities.__annotations__
        assert "tenant_id" not in module.get_subgraph.__annotations__

    async def test_the_tenant_from_the_token_reaches_the_query(self) -> None:
        seen: dict[str, Any] = {}

        async def run(cypher: str, parameters: Any = None) -> list[dict[str, Any]]:
            seen.update(parameters or {})
            return []

        async with _client(GraphService(GraphClient(run, max_attempts=1))) as client:
            await client.get("/api/v1/graph/search?q=acme")
        assert seen["tenant_id"] == TENANT


class TestSearch:
    async def test_returns_hits(self) -> None:
        async with _client(_service_returning([ENTITY_ROW])) as client:
            body = (await client.get("/api/v1/graph/search?q=acme")).json()
        assert body["total"] == 1
        assert body["results"][0]["name"] == "Acme"
        assert body["results"][0]["score"] == 3.2

    async def test_internal_fields_are_not_published(self) -> None:
        """`normalized_name` and `merged_from` exist so resolution can un-merge.
        Publishing them makes them contract, and the next normalisation change
        breaks somebody's dashboard."""
        async with _client(_service_returning([{**ENTITY_ROW, "normalized_name": "acme"}])) as c:
            hit = (await c.get("/api/v1/graph/search?q=acme")).json()["results"][0]
        assert "normalized_name" not in hit
        assert "merged_from" not in hit

    async def test_an_unknown_type_filter_is_rejected_not_ignored(self) -> None:
        """A tolerant enum would degrade `Compnay` to UNKNOWN, match nothing, and
        let the caller conclude there are no companies."""
        async with _client(_service_returning([])) as client:
            response = await client.get("/api/v1/graph/search?q=x&type=Compnay")
        assert response.status_code == 422
        assert "Compnay" in response.text

    async def test_a_known_type_filter_is_accepted(self) -> None:
        async with _client(_service_returning([ENTITY_ROW])) as client:
            response = await client.get("/api/v1/graph/search?q=x&type=Company&type=Product")
        assert response.status_code == 200

    async def test_an_overlong_query_is_rejected(self) -> None:
        """A pathological Lucene query costs the server time the client
        controls."""
        async with _client(_service_returning([])) as client:
            response = await client.get(f"/api/v1/graph/search?q={'a' * 500}")
        assert response.status_code == 422

    async def test_an_unreachable_graph_is_a_503_not_an_empty_list(self) -> None:
        service = GraphService(_FailingClient(GraphUnavailableError("down")))  # type: ignore[arg-type]
        async with _client(service) as client:
            response = await client.get("/api/v1/graph/search?q=acme")
        assert response.status_code == 503
        assert response.headers["content-type"].startswith("application/problem+json")


class TestEntityDetail:
    async def test_returns_the_entity(self) -> None:
        async with _client(_service_returning([ENTITY_ROW])) as client:
            body = (await client.get("/api/v1/graph/entities/e1")).json()
        assert body["id"] == "e1"
        assert body["pagerank_score"] == 0.04

    async def test_missing_entity_is_a_404(self) -> None:
        async with _client(_service_returning([])) as client:
            assert (await client.get("/api/v1/graph/entities/nope")).status_code == 404

    async def test_staleness_is_published_as_one_boolean(self) -> None:
        """'Is this ranking current' is the actual question; a client should not
        have to compare two timestamps to answer it."""
        stale = {**ENTITY_ROW, "computed_at": None}
        async with _client(_service_returning([stale])) as client:
            body = (await client.get("/api/v1/graph/entities/e1")).json()
        assert body["analytics_are_stale"] is True


class TestCompetitors:
    ROW = {
        "id": "c1",
        "name": "Globex",
        "type": "Company",
        "strength": 0.8,
        "basis": "stated",
        "evidence_count": 4,
        "citations": ["sig_1"],
    }

    async def test_returns_rivals(self) -> None:
        async with _client(_service_returning([self.ROW])) as client:
            body = (await client.get("/api/v1/graph/entities/Acme/competitors")).json()
        assert body["results"][0]["name"] == "Globex"
        assert body["results"][0]["basis"] == "stated"

    async def test_as_of_is_echoed_back(self) -> None:
        """It defaults to now on the server, so a client that omitted it cannot
        otherwise reproduce the result."""
        async with _client(_service_returning([])) as client:
            body = (await client.get("/api/v1/graph/entities/Acme/competitors")).json()
        assert body["as_of"]

    async def test_a_naive_as_of_is_rejected(self) -> None:
        """It would be compared against UTC values in Neo4j and be silently wrong
        by the server's offset."""
        async with _client(_service_returning([])) as client:
            response = await client.get(
                "/api/v1/graph/entities/Acme/competitors?as_of=2026-08-06T00:00:00"
            )
        assert response.status_code == 422

    async def test_an_unrecognised_basis_does_not_500_the_response(self) -> None:
        """One odd edge must not fail the whole payload."""
        async with _client(_service_returning([{**self.ROW, "basis": "vibes"}])) as client:
            body = (await client.get("/api/v1/graph/entities/Acme/competitors")).json()
        assert body["results"][0]["basis"] == "unknown"

    async def test_null_strength_survives_serialisation(self) -> None:
        """None means 'nobody assessed it'; 0.0 means 'assessed and negligible'."""
        async with _client(_service_returning([{**self.ROW, "strength": None}])) as client:
            body = (await client.get("/api/v1/graph/entities/Acme/competitors")).json()
        assert body["results"][0]["strength"] is None


class TestOwnership:
    async def test_independent_company_is_stated_explicitly(self) -> None:
        async with _client(_service_returning([])) as client:
            body = (await client.get("/api/v1/graph/companies/c1/ownership")).json()
        assert body["is_independent"] is True
        assert body["chain"] == []

    async def test_a_chain_names_the_root_first(self) -> None:
        row = {"chain_ids": ["p", "c"], "ownership_chain": ["Parent", "Child"], "hops": 1}
        async with _client(_service_returning([row])) as client:
            body = (await client.get("/api/v1/graph/companies/c/ownership")).json()
        assert body["names"][0] == "Parent"
        assert body["is_independent"] is False


class TestPaths:
    async def test_reports_connected_false_when_there_is_no_path(self) -> None:
        async with _client(_service_returning([])) as client:
            body = (await client.get("/api/v1/graph/paths?source_id=a&target_id=b")).json()
        assert body["connected"] is False

    async def test_more_than_four_hops_is_rejected(self) -> None:
        """Past four hops an entity graph connects everything to everything."""
        async with _client(_service_returning([])) as client:
            response = await client.get(
                "/api/v1/graph/paths?source_id=a&target_id=b&max_hops=9"
            )
        assert response.status_code == 422


class TestSubgraph:
    EDGE = {
        "subject_id": "a",
        "subject_name": "Acme",
        "predicate": "COMPETES_WITH",
        "object_id": "b",
        "object_name": "Globex",
        "confidence": 0.7,
        "supporting_signal_ids": ["sig_1"],
    }

    async def test_nodes_are_deduplicated_and_separate_from_edges(self) -> None:
        """An entity joined by six edges appears once in a node list and six
        times nested inside edges -- and a client rendering the nested form draws
        six overlapping copies."""
        rows = [self.EDGE, {**self.EDGE, "object_id": "c", "object_name": "Initech"}]
        async with _client(_service_returning(rows)) as client:
            body = (
                await client.post(
                    "/api/v1/graph/subgraph", json={"entity_ids": ["a"], "depth": 1}
                )
            ).json()
        assert sorted(node["id"] for node in body["nodes"]) == ["a", "b", "c"]
        assert len(body["edges"]) == 2

    async def test_duplicate_edges_are_collapsed(self) -> None:
        async with _client(_service_returning([self.EDGE, self.EDGE])) as client:
            body = (
                await client.post("/api/v1/graph/subgraph", json={"entity_ids": ["a"]})
            ).json()
        assert len(body["edges"]) == 1

    async def test_truncation_is_reported(self) -> None:
        """Presenting a subset as the full neighbourhood invites a conclusion
        drawn from missing data."""
        rows = [{**self.EDGE, "object_id": f"o{i}"} for i in range(5)]
        async with _client(_service_returning(rows)) as client:
            body = (
                await client.post(
                    "/api/v1/graph/subgraph", json={"entity_ids": ["a"], "limit": 5}
                )
            ).json()
        assert body["truncated"] is True

    async def test_an_empty_seed_set_is_rejected(self) -> None:
        async with _client(_service_returning([])) as client:
            response = await client.post("/api/v1/graph/subgraph", json={"entity_ids": []})
        assert response.status_code == 422

    async def test_too_many_seeds_are_rejected(self) -> None:
        """Each seed expands to a neighbourhood, so cost is multiplicative."""
        async with _client(_service_returning([])) as client:
            response = await client.post(
                "/api/v1/graph/subgraph", json={"entity_ids": [f"e{i}" for i in range(200)]}
            )
        assert response.status_code == 422


class TestOpenApi:
    async def test_every_graph_route_documents_problem_json(self) -> None:
        app = create_app()
        schema = app.openapi()
        graph_paths = {p: v for p, v in schema["paths"].items() if p.startswith("/api/v1/graph")}
        assert graph_paths
        for path, operations in graph_paths.items():
            for method, operation in operations.items():
                assert "401" in operation["responses"], f"{method} {path} has no documented 401"
