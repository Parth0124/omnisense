"""Unit tests for `backend/db/qdrant.py`.

Two behaviours here are worth more than the rest of the module put together.

**`ensure_collection` must refuse a geometry mismatch.** A Qdrant collection's
vector size and distance metric are fixed at creation. If the configured
embedding model changes -- `text-embedding-3-small` at 1536 to `bge-m3` at 1024 --
and `ensure_collection` shrugs and proceeds, the mismatch is discovered at the
first upsert, *after* the embedding spend has already happened, and every vector
produced in between is unusable. `docs/signal-model.md` §9 calls this out as the
open decision with the worst blast radius, so the guard is the tested part.

**The probe must be bounded.** `/readyz` aggregates six probes; an unbounded one
does not report late, it stalls the aggregate past the liveness deadline. Against
a blackholed host this probe once measured 5s while PostgreSQL measured 60s -- see
`backend/db/__init__.py`.
"""

from __future__ import annotations

import asyncio
import socket
import time
from typing import Any

import pytest

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError
from backend.db import HEALTH_PROBE_TIMEOUT_SECONDS

pytestmark = pytest.mark.unit


def _closed_port() -> int:
    """A port with nothing listening, so a connect is refused immediately."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point at a dead endpoint and clear both singletons between tests."""
    import backend.db.qdrant as mod

    monkeypatch.setattr(mod, "_client", None, raising=False)
    get_settings.cache_clear()


class TestImportIsInert:
    def test_import_opens_no_socket(self) -> None:
        """Importing a client module must never do I/O.

        Import-time connections make the test suite depend on a running service
        just to *collect* tests, and turn a dead dependency into an ImportError
        during startup rather than a degraded readiness report.
        """
        import importlib

        import backend.db.qdrant as mod

        original = socket.socket.connect

        def _forbidden(self: Any, *a: Any, **k: Any) -> Any:
            raise AssertionError("import opened a socket")

        socket.socket.connect = _forbidden  # type: ignore[method-assign]
        try:
            importlib.reload(mod)
        finally:
            socket.socket.connect = original  # type: ignore[method-assign]

    def test_client_is_a_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import backend.db.qdrant as mod

        monkeypatch.setenv("QDRANT_URL", f"http://127.0.0.1:{_closed_port()}")
        get_settings.cache_clear()
        assert mod.get_qdrant() is mod.get_qdrant()


class TestProbe:
    async def test_returns_false_when_unreachable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """False, not an exception -- readiness aggregates and must not abort."""
        import backend.db.qdrant as mod

        monkeypatch.setenv("QDRANT_URL", f"http://127.0.0.1:{_closed_port()}")
        get_settings.cache_clear()
        assert await mod.check_qdrant() is False

    async def test_is_bounded_by_the_probe_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bounded by our timeout, not by whatever the client library defaults to."""
        import backend.db.qdrant as mod

        # A blackholed address: the connect hangs rather than being refused,
        # which is the case an unbounded probe fails on.
        monkeypatch.setenv("QDRANT_URL", "http://192.0.2.1:6333")
        get_settings.cache_clear()

        started = time.monotonic()
        assert await mod.check_qdrant() is False
        elapsed = time.monotonic() - started
        assert elapsed < HEALTH_PROBE_TIMEOUT_SECONDS + 2.0, (
            f"probe took {elapsed:.1f}s; budget is {HEALTH_PROBE_TIMEOUT_SECONDS}s"
        )


class _FakeVectorParams:
    def __init__(self, size: int, distance: str) -> None:
        self.size = size
        self.distance = distance


class _FakeCollection:
    """Minimal stand-in for a Qdrant collection description."""

    def __init__(self, size: int, distance: str) -> None:
        params = type("P", (), {"vectors": _FakeVectorParams(size, distance)})()
        self.config = type("C", (), {"params": params})()


class TestEnsureCollectionGuardsGeometry:
    """The guard that stops a dimension change from being discovered too late."""

    @pytest.fixture
    def fake_client(self, monkeypatch: pytest.MonkeyPatch):
        import backend.db.qdrant as mod

        class Fake:
            def __init__(self) -> None:
                self.created: list[tuple[str, int, Any]] = []
                self.existing: _FakeCollection | None = None

            async def collection_exists(self, collection_name: str) -> bool:
                return self.existing is not None

            async def get_collection(self, collection_name: str) -> _FakeCollection:
                assert self.existing is not None
                return self.existing

            async def create_collection(
                self, collection_name: str, vectors_config: Any, **kwargs: Any
            ) -> None:
                self.created.append(
                    (collection_name, vectors_config.size, vectors_config.distance)
                )

        fake = Fake()
        monkeypatch.setattr(mod, "get_qdrant", lambda: fake)
        return fake

    async def test_creates_when_absent(self, fake_client: Any) -> None:
        import backend.db.qdrant as mod

        await mod.ensure_collection("omnisense_signals", vector_size=1536)
        assert fake_client.created, "collection was not created"
        name, size, _distance = fake_client.created[0]
        assert (name, size) == ("omnisense_signals", 1536)

    async def test_is_idempotent_when_geometry_matches(self, fake_client: Any) -> None:
        import backend.db.qdrant as mod
        from qdrant_client.models import Distance

        fake_client.existing = _FakeCollection(1536, Distance.COSINE)
        await mod.ensure_collection("omnisense_signals", vector_size=1536)
        assert not fake_client.created, "an existing matching collection was recreated"

    async def test_rejects_a_dimension_mismatch(self, fake_client: Any) -> None:
        """The expensive mistake: re-pointing at a collection of the wrong size."""
        import backend.db.qdrant as mod
        from qdrant_client.models import Distance

        fake_client.existing = _FakeCollection(1024, Distance.COSINE)
        with pytest.raises(ConfigurationError) as excinfo:
            await mod.ensure_collection("omnisense_signals", vector_size=1536)

        message = str(excinfo.value)
        assert "1024" in message and "1536" in message, (
            "the error must name both dimensions, or the operator cannot tell "
            f"which side is wrong: {message}"
        )
        assert not fake_client.created, "must not silently recreate the collection"
