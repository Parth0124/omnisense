"""Unit tests for `backend/db/neo4j.py`.

The module's own docstring argues at length that a readiness probe must not
inherit a real query's retry and connection budgets -- "a readiness endpoint that
blocks for 30s per poll is worse than useless". It then did exactly that, because
`NEO4J_CONNECTION_TIMEOUT_SECONDS` defaults to 30 and bounds the connect
regardless of what the probe wants. Measured against a blackholed host, the probe
took exactly 30.0s while Qdrant's took 5.0s.

That is the kind of defect a docstring cannot prevent and a test can, so the
timing assertion below is the point of this file. The read/write split is the
other thing worth pinning: against a routed `neo4j://` cluster URI a read session
may be served by a replica while writes must reach the leader, so the distinction
is correctness, not decoration.
"""

from __future__ import annotations

import socket
import time
from typing import Any

import pytest
from neo4j import READ_ACCESS, WRITE_ACCESS

from backend.core.config import get_settings
from backend.db import HEALTH_PROBE_TIMEOUT_SECONDS

pytestmark = pytest.mark.unit


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(autouse=True)
async def _reset() -> Any:
    """Clear the driver singleton around each test, and close it afterwards.

    Closing matters: an `AsyncDriver` that is garbage-collected without being
    closed raises inside `__del__`, which pytest surfaces as an unraisable
    exception -- and `pyproject.toml` sets `filterwarnings = ["error"]`, so the
    leak fails the test rather than merely warning.
    """
    import backend.db.neo4j as mod

    mod._driver = None
    get_settings.cache_clear()
    yield
    await mod.dispose_driver()
    mod._driver = None
    get_settings.cache_clear()


class TestImportIsInert:
    def test_import_opens_no_socket(self) -> None:
        import importlib

        import backend.db.neo4j as mod

        original = socket.socket.connect

        def _forbidden(self: Any, *a: Any, **k: Any) -> Any:
            raise AssertionError("import opened a socket")

        socket.socket.connect = _forbidden  # type: ignore[method-assign]
        try:
            importlib.reload(mod)
        finally:
            socket.socket.connect = original  # type: ignore[method-assign]

    def test_driver_is_a_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One driver per process: it owns a connection pool."""
        import backend.db.neo4j as mod

        monkeypatch.setenv("NEO4J_URI", f"bolt://127.0.0.1:{_closed_port()}")
        get_settings.cache_clear()
        assert mod.get_driver() is mod.get_driver()


class TestProbe:
    async def test_returns_false_when_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import backend.db.neo4j as mod

        monkeypatch.setenv("NEO4J_URI", f"bolt://127.0.0.1:{_closed_port()}")
        get_settings.cache_clear()
        assert await mod.check_neo4j() is False

    async def test_is_bounded_by_the_probe_budget_not_the_connection_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The regression this file exists for.

        `NEO4J_CONNECTION_TIMEOUT_SECONDS` is deliberately set high here: a real
        query is allowed to wait that long, a probe is not. If the explicit
        `asyncio.timeout` in `check_neo4j` is ever removed, this fails.
        """
        import backend.db.neo4j as mod

        monkeypatch.setenv("NEO4J_URI", "bolt://192.0.2.1:7687")  # blackholed
        monkeypatch.setenv("NEO4J_CONNECTION_TIMEOUT_SECONDS", "30")
        get_settings.cache_clear()

        started = time.monotonic()
        assert await mod.check_neo4j() is False
        elapsed = time.monotonic() - started
        assert elapsed < HEALTH_PROBE_TIMEOUT_SECONDS + 2.0, (
            f"probe took {elapsed:.1f}s against a 30s connection timeout; the "
            f"probe budget is {HEALTH_PROBE_TIMEOUT_SECONDS}s and must win"
        )


class TestSessionAccessModes:
    """Reads and writes must not open the same kind of session."""

    @pytest.fixture
    def recorded(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
        import backend.db.neo4j as mod

        calls: list[dict[str, Any]] = []

        class FakeResult:
            async def data(self) -> list[dict[str, Any]]:
                return [{"ok": 1}]

            async def consume(self) -> None:
                return None

        class FakeSession:
            def __init__(self, **kwargs: Any) -> None:
                calls.append(kwargs)

            async def __aenter__(self) -> FakeSession:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def run(self, query: str, parameters: Any = None, **kw: Any) -> FakeResult:
                calls[-1]["query"] = query
                calls[-1]["parameters"] = parameters if parameters is not None else kw
                return FakeResult()

            async def execute_read(self, fn: Any, *a: Any, **k: Any) -> Any:
                return await fn(self, *a, **k)

            async def execute_write(self, fn: Any, *a: Any, **k: Any) -> Any:
                return await fn(self, *a, **k)

        class FakeDriver:
            def session(self, **kwargs: Any) -> FakeSession:
                return FakeSession(**kwargs)

        monkeypatch.setattr(mod, "get_driver", lambda: FakeDriver())
        return calls

    async def test_read_session_requests_read_access(
        self, recorded: list[dict[str, Any]]
    ) -> None:
        import backend.db.neo4j as mod

        async with mod.read_session():
            pass
        assert recorded[0]["default_access_mode"] == READ_ACCESS

    async def test_write_session_requests_write_access(
        self, recorded: list[dict[str, Any]]
    ) -> None:
        import backend.db.neo4j as mod

        async with mod.write_session():
            pass
        assert recorded[0]["default_access_mode"] == WRITE_ACCESS

    async def test_sessions_target_the_configured_database(
        self, recorded: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session that omits the database silently uses the server default."""
        import backend.db.neo4j as mod

        monkeypatch.setenv("NEO4J_DATABASE", "omnisense_graph")
        get_settings.cache_clear()
        async with mod.read_session():
            pass
        assert recorded[0].get("database") == "omnisense_graph"


class TestParameterisation:
    """Cypher injection is as real as SQL injection."""

    async def test_run_read_passes_parameters_separately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import backend.db.neo4j as mod

        seen: dict[str, Any] = {}

        class FakeRecord:
            def data(self) -> dict[str, Any]:
                return {"name": "Datadog"}

        class FakeResult:
            def __aiter__(self) -> Any:
                async def gen() -> Any:
                    yield FakeRecord()

                return gen()

        class FakeTx:
            async def run(self, query: str, parameters: Any = None, **kw: Any) -> FakeResult:
                seen["query"] = query
                seen["parameters"] = parameters if parameters is not None else kw
                return FakeResult()

        class FakeSession:
            async def __aenter__(self) -> FakeSession:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def execute_read(self, fn: Any, *a: Any, **k: Any) -> Any:
                # run_read() uses a managed transaction so the driver retries
                # transient failures; the fake must offer the same entry point.
                return await fn(FakeTx(), *a, **k)

        monkeypatch.setattr(mod, "read_session", lambda: FakeSession())

        hostile = "Datadog' OR 1=1 //"
        rows = await mod.run_read(
            "MATCH (c:Company {canonical_name: $name}) RETURN c.canonical_name AS name",
            {"name": hostile},
        )

        assert rows == [{"name": "Datadog"}]
        assert hostile not in seen["query"], (
            "the value was interpolated into the Cypher string instead of being "
            "passed as a parameter"
        )
        assert seen["parameters"]["name"] == hostile
