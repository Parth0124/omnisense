"""Unit tests for the FastAPI gateway.

Every test here runs with **no datastore available**, which is the point. The
gateway's most important property is that it degrades rather than crashes: an
unreachable PostgreSQL must produce a 503 from `/readyz`, not a failed import, a
startup crash, or a 200 that lies.
"""

from __future__ import annotations

import httpx
import pytest

from backend.api.v1.health import REQUIRED_DEPENDENCIES
from backend.core.config import get_settings
from backend.core.exceptions import NotFoundError, RateLimitedError
from backend.db import neo4j
from backend.main import create_app

pytestmark = pytest.mark.unit


@pytest.fixture
async def client() -> httpx.AsyncClient:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestLiveness:
    async def test_health_returns_ok_with_no_datastore_running(
        self, client: httpx.AsyncClient
    ) -> None:
        """Liveness must not depend on anything.

        A liveness probe that fails during a database blip makes Kubernetes kill
        and reschedule every replica, converting a degradation into an outage and
        removing the capacity that would have served the unaffected endpoints.
        """
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_health_is_unversioned(self, client: httpx.AsyncClient) -> None:
        """Probes are configured once in a manifest and must survive a version bump."""
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/api/v1/health")).status_code == 404


class TestReadiness:
    async def test_readyz_reports_every_dependency(self, client: httpx.AsyncClient) -> None:
        body = (await client.get("/readyz")).json()
        assert set(body["checks"]) == {
            "postgres",
            "redis",
            "qdrant",
            "neo4j",
            "opensearch",
        }

    async def test_readyz_is_503_when_a_required_dependency_is_down(
        self, client: httpx.AsyncClient
    ) -> None:
        """PostgreSQL is the commit point and the checkpoint store."""
        response = await client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["status"] == "unavailable"

    async def test_readyz_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dead dependency is reported, never raised.

        Neo4j is pointed at TEST-NET-1 (RFC 5737, guaranteed unroutable) instead
        of assuming the configured address has nothing behind it. That assumption
        broke the moment anyone ran `make start`, which is the normal state while
        developing -- and a unit test that passes only when your stack is *down*
        is one people learn to ignore.
        """
        monkeypatch.setenv("NEO4J_URI", "bolt://192.0.2.1:7687")
        get_settings.cache_clear()

        # The driver is a process global while every async test gets its own
        # event loop, so a driver built by an earlier test belongs to a loop that
        # is already closed. Disposing it here raises "Event loop is closed" from
        # inside the neo4j pool -- so it is set aside untouched and put back
        # afterwards, and only the driver this test creates is disposed.
        previous, neo4j._driver = neo4j._driver, None
        try:
            app = create_app()
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as fresh:
                body = (await fresh.get("/readyz")).json()
            assert body["checks"]["neo4j"]["status"] == "fail"
        finally:
            await neo4j.dispose_driver()
            neo4j._driver = previous
            get_settings.cache_clear()

    async def test_probes_run_concurrently(self, client: httpx.AsyncClient) -> None:
        """The property that keeps `/readyz` inside the liveness deadline.

        Six probes bounded at 5s each would take 30s serially -- long enough for
        the liveness probe sharing that deadline to fire and kill the pod, which
        is the failure `/readyz` exists to prevent. Asserting the total is below
        the sum of the parts is what actually catches a regression to a serial
        loop; asserting an absolute number would just be flaky.
        """
        body = (await client.get("/readyz")).json()
        serial = sum(check["latency_ms"] for check in body["checks"].values())
        assert body["total_latency_ms"] < serial

    def test_only_postgres_is_required(self) -> None:
        """Everything else degrades (`docs/architecture.md` §7.3).

        Marking Qdrant required would pull the whole API out of rotation to
        protect keyword-only retrieval, which was designed to survive its loss.
        """
        assert REQUIRED_DEPENDENCIES == frozenset({"postgres"})


class TestProblemResponses:
    async def test_unknown_route_is_problem_json(self, client: httpx.AsyncClient) -> None:
        response = await client.get("/api/v1/nope")
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/problem+json")
        assert response.json()["instance"] == "/api/v1/nope"

    def test_domain_errors_carry_their_own_status(self) -> None:
        assert NotFoundError.for_resource("investigation", "inv_1").status_code == 404

    def test_rate_limit_carries_retry_after(self) -> None:
        """A 429 without Retry-After makes most clients retry immediately."""
        assert RateLimitedError(retry_after_seconds=30).retry_after_seconds == 30


class TestAppConstruction:
    def test_importing_the_app_opens_no_socket(self) -> None:
        """Import-time I/O would make collecting tests require a live database."""
        import socket

        original = socket.socket.connect
        socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[method-assign]
            AssertionError("app construction opened a socket")
        )
        try:
            create_app()
        finally:
            socket.socket.connect = original  # type: ignore[method-assign]

    def test_create_app_is_callable_twice(self) -> None:
        """A factory, not a module singleton, so a test can build an isolated app."""
        assert create_app() is not create_app()

    def test_cors_is_never_a_wildcard(self) -> None:
        from backend.core.config import get_settings

        assert "*" not in get_settings().app.cors_origin_list
