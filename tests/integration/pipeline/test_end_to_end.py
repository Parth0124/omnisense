"""The paths that only exist when the pieces are joined.

Everything here spans at least two layers and a real store, which is the whole
selection criterion: if a test can be written with fakes it belongs in
`tests/unit/`, and putting it here only makes the suite slower and more fragile.

The LLM is faked even in these tests, and that is deliberate rather than a
shortcut. A test whose result depends on what a model generated is a test that
fails for reasons unrelated to the code, and the thing worth proving here is that
the *wiring* holds — that a Signal written by the pipeline is the Signal
retrieval reads back, that a report stored by the service is the report the API
serves. Model quality is what `agents/evaluation/` measures, against golden sets,
on a different cadence.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class TestReportPersistence:
    """`services/report_service.py` against a real database."""

    async def test_a_stored_report_reads_back_whole(
        self, pg_sessionmaker, run_namespace
    ) -> None:
        """Sections and citations survive the transaction and the round trip.

        Unit tests cover the projection logic; what they cannot cover is whether
        three tables written in one transaction come back joined correctly —
        which depends on the foreign keys, the flush ordering and the session
        lifecycle, none of which a fake has.
        """
        from services.report_service import ReportService

        service = ReportService(pg_sessionmaker, tenant_id=run_namespace)
        investigation_id = f"{run_namespace}_inv"

        stored = await service.store(
            investigation_id=investigation_id,
            document={
                "title": "Integration report",
                "executive_summary": "A summary.",
                "confidence": 0.62,
                "sections": [
                    {
                        "title": "Findings",
                        "body": "Body text.",
                        "claims": [
                            {
                                "text": "Battery complaints rose",
                                "citations": ["sig_a", "sig_b"],
                                "confidence": 0.7,
                            }
                        ],
                    }
                ],
            },
        )

        fetched = await service.get(stored.id)
        assert fetched is not None
        assert fetched.title == "Integration report"
        assert len(fetched.sections) == 1
        assert fetched.citation_count == 2, (
            "citations did not survive the write; a report whose claims lose "
            "their sources renders as unsourced prose"
        )

    async def test_a_new_version_supersedes_the_previous_one(
        self, pg_sessionmaker, run_namespace
    ) -> None:
        """Reports are versioned, not edited.

        The property that makes a report someone acted on still retrievable
        afterwards. Only checkable against a store, because it is about two rows
        and the pointer between them.
        """
        from services.report_service import ReportService

        service = ReportService(pg_sessionmaker, tenant_id=run_namespace)
        investigation_id = f"{run_namespace}_versioned"
        document = {"title": "v1", "sections": [{"title": "S", "body": "b"}]}

        first = await service.store(investigation_id=investigation_id, document=document)
        second = await service.store(
            investigation_id=investigation_id, document={**document, "title": "v2"}
        )

        assert second.id != first.id
        refetched_first = await service.get(first.id)
        assert refetched_first is not None
        assert refetched_first.superseded_by == second.id, (
            "the earlier version was not marked superseded; a client asking for "
            "'the report' cannot tell which one is current"
        )
        assert refetched_first.title == "v1", "the earlier version was mutated in place"

    async def test_citing_signal_finds_the_reports_that_depend_on_it(
        self, pg_sessionmaker, run_namespace
    ) -> None:
        """The query erasure needs, run against a real index.

        When a source is retracted or a signal must be erased, this is how you
        find the published documents that depended on it. It is an index lookup
        precisely because citations are rows; as JSON inside the body it would be
        a full scan, and it is exactly the query you need under time pressure.
        """
        from services.report_service import ReportService

        service = ReportService(pg_sessionmaker, tenant_id=run_namespace)
        signal_id = f"{run_namespace}_sig_traced"

        stored = await service.store(
            investigation_id=f"{run_namespace}_erasure",
            document={
                "title": "Traceable",
                "sections": [
                    {"title": "S", "body": "b", "claims": [{"text": "t", "citations": [signal_id]}]}
                ],
            },
        )

        assert stored.id in await service.citing_signal(signal_id)


class TestApiAgainstRealStores:
    """The API with its real service layer and real datastores behind it."""

    async def _client(self, scopes: tuple[str, ...]):
        import httpx

        from backend.api.deps import mint_access_token
        from backend.main import create_app

        app = create_app()
        token = mint_access_token(subject="it-user", tenant_id="default", scopes=scopes)
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        )

    async def test_creating_an_investigation_persists_it(
        self, postgres_available
    ) -> None:
        """`POST` then `GET` through the real stack.

        Proves the 202 body's id actually resolves — that the row was committed
        rather than rolled back by a session the handler forgot to commit, which
        is a mistake no unit test with an in-memory session can reproduce.
        """
        async with await self._client(("investigations:write", "investigations:read")) as client:
            created = await client.post(
                "/api/v1/investigations",
                json={"query": f"integration probe {uuid.uuid4().hex[:8]}"},
            )
            assert created.status_code == 202, created.text
            investigation_id = created.json()["id"]

            fetched = await client.get(f"/api/v1/investigations/{investigation_id}")
            assert fetched.status_code == 200
            assert fetched.json()["id"] == investigation_id

    async def test_signals_endpoint_answers_from_the_real_table(
        self, postgres_available
    ) -> None:
        """An empty corpus is a valid answer and must be a 200, not a 500.

        A fresh database is the most common state this endpoint is first hit in,
        so the empty case is the one worth pinning.
        """
        async with await self._client(("signals:read",)) as client:
            response = await client.get("/api/v1/signals", params={"limit": 5})
            assert response.status_code == 200, response.text
            assert "items" in response.json()

    async def test_graph_search_answers_when_neo4j_is_up(
        self, postgres_available, neo4j_available
    ) -> None:
        """Complements the unit test, which proves it 503s when Neo4j is down.

        Both halves are needed. A handler hard-wired to fail would satisfy the
        unit test on its own.
        """
        async with await self._client(("graph:read",)) as client:
            response = await client.get("/api/v1/graph/search", params={"q": "acme"})
            assert response.status_code == 200, response.text
            assert "results" in response.json()


class TestAgentGraphWiring:
    """The composition root, compiled and inspected. No model calls."""

    async def test_the_graph_compiles_with_every_node_bound(self) -> None:
        """Fails if an agent is added without being wired.

        `agents/composition.py` builds the node map from `NODE_AGENT`, so a new
        node with no implementation raises here. Worth an integration test rather
        than a unit one because it imports the whole agent package — which is
        also what makes it catch an import cycle introduced anywhere in it.
        """
        from agents.composition import build_agents, build_nodes, compile_investigation_graph
        from agents.router import NodeName

        class FakeProvider:
            async def complete(self, *args, **kwargs): ...
            async def structured(self, *args, **kwargs): ...
            async def aclose(self): ...

        class FakeRegistry:
            def is_allowed(self, agent, name): return False
            async def invoke(self, **kwargs): ...

        bundle = build_agents(provider=FakeProvider(), registry=FakeRegistry())
        nodes = build_nodes(bundle)

        assert len(bundle.agents) == 10
        assert set(nodes) == set(NodeName)
        assert compile_investigation_graph(bundle) is not None

    async def test_every_agent_prompt_loads_from_disk(self) -> None:
        """Catches a prompt file deleted, renamed or left as a stub.

        The unit suite reads prompts through the same loader, but this asserts
        the *shipped* files specifically — a `v1.md` removed from the package
        would pass a mocked loader and fail every run in production.
        """
        from agents.composition import AGENT_CLASSES
        from prompts.loader import load_prompt

        for agent_cls in AGENT_CLASSES:
            rendered = load_prompt(agent_cls.name, agent_cls.prompt_version)
            assert "TODO" not in rendered.text, f"{agent_cls.name.value} prompt is a stub"
            assert "OMNISENSE_UNTRUSTED_DATA" in rendered.text, (
                f"{agent_cls.name.value} is missing the injection boundary fragment"
            )
