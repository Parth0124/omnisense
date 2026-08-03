"""Unit tests for `backend/db/postgres.py`, against SQLite.

No Docker (`docs/testing-strategy.md`), so the engine here is `aiosqlite`. That
splits the module in two and both halves are worth asserting on:

* `ping()` and `pool_stats()` are backend-agnostic and are exercised for real.
* `server_version()`, `installed_extensions()` and `advisory_lock()` are
  PostgreSQL-only. The thing being tested is the *guard* -- that asking for them
  on the wrong backend produces a sentence naming the operation and the backend,
  rather than `sqlite3.OperationalError: near "SHOW": syntax error`. A confusing
  error costs more engineer-hours than the missing feature does.

Their behaviour against a real server belongs in `tests/integration/db/`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError, DependencyUnavailableError
from backend.db import postgres as postgres_module
from backend.db import session as session_module
from backend.db.postgres import (
    REQUIRED_EXTENSIONS,
    advisory_lock,
    advisory_lock_key,
    installed_extensions,
    missing_extensions,
    ping,
    pool_stats,
    require_extensions,
    server_version,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def sqlite_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    """Point the process-wide engine at a throwaway SQLite file.

    A file rather than `:memory:` on purpose: in-memory SQLite is served by
    `StaticPool`, which exposes none of the pool counters, so the file backend is
    the only way to exercise the populated branch of `pool_stats()`. The
    `None`-returning branch is covered separately.
    """
    db_path = tmp_path_factory.mktemp("pg-helpers") / "unit.sqlite"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    monkeypatch.setattr(session_module, "_engine", None)
    monkeypatch.setattr(session_module, "_sessionmaker", None)
    yield
    get_settings.cache_clear()


@pytest.fixture
async def connected_engine(sqlite_engine: None) -> AsyncIterator[None]:
    """`sqlite_engine`, plus disposal, for tests that actually open a connection.

    Separate from `sqlite_engine` because disposal is `async` and most tests here
    are synchronous. Leaving a pooled aiosqlite connection undisposed does not
    fail the test that created it -- it surfaces later as an unraisable exception
    from `Connection.__del__`, blamed on whichever test happened to trigger the
    garbage collection.
    """
    yield
    await session_module.dispose_engine()


class TestPing:
    async def test_returns_a_positive_latency(self, connected_engine: None) -> None:
        latency_ms = await ping()
        assert latency_ms > 0.0
        # Sanity bound, not a performance assertion: a local round trip that
        # takes over a second means the measurement is wrong, not that the disk
        # is slow.
        assert latency_ms < 1000.0

    async def test_unreachable_database_raises_the_typed_503(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dead socket must become the retryable 503, with the cause attached.

        The failure is injected at the engine rather than by pointing SQLite at
        an unwritable path: what is under test is the translation, and a driver
        that fails in its own idiosyncratic way only obscures that.
        """

        class _DeadEngine:
            def connect(self) -> object:
                raise OSError("connection refused")

        monkeypatch.setattr(postgres_module, "get_engine", _DeadEngine)

        with pytest.raises(DependencyUnavailableError) as excinfo:
            await ping()
        assert excinfo.value.status_code == 503
        assert excinfo.value.details == {"dependency": "PostgreSQL"}
        assert isinstance(excinfo.value.cause, OSError)


class TestPoolStats:
    def test_reports_the_queue_pool_counters(self, sqlite_engine: None) -> None:
        stats = pool_stats()
        assert stats.backend == "sqlite"
        assert stats.size is not None
        assert stats.checked_out == 0
        assert stats.as_metrics().keys() == {"size", "checked_in", "checked_out", "overflow"}

    def test_pools_without_counters_report_none_not_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`StaticPool` counts nothing; publishing zeroes would be a lie."""
        monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
        get_settings.cache_clear()
        monkeypatch.setattr(session_module, "_engine", None)
        monkeypatch.setattr(session_module, "_sessionmaker", None)
        try:
            stats = pool_stats()
            assert stats.size is None
            assert stats.checked_out is None
            assert stats.as_metrics() == {}
        finally:
            get_settings.cache_clear()

    def test_does_not_open_a_connection(self, sqlite_engine: None) -> None:
        """It is called from a metrics collector on a timer; it must be free."""
        before = pool_stats()
        after = pool_stats()
        assert before == after
        assert after.checked_out == 0


class TestPostgresOnlyHelpersAreGuarded:
    async def test_server_version(self, sqlite_engine: None) -> None:
        with pytest.raises(ConfigurationError) as excinfo:
            await server_version()
        message = str(excinfo.value)
        assert "server_version()" in message
        assert "PostgreSQL-only" in message
        assert "'sqlite'" in message
        assert excinfo.value.details["backend"] == "sqlite"

    async def test_installed_extensions(self, sqlite_engine: None) -> None:
        with pytest.raises(ConfigurationError, match="installed_extensions"):
            await installed_extensions()

    async def test_missing_extensions(self, sqlite_engine: None) -> None:
        with pytest.raises(ConfigurationError, match="installed_extensions"):
            await missing_extensions()

    async def test_require_extensions(self, sqlite_engine: None) -> None:
        with pytest.raises(ConfigurationError, match="installed_extensions"):
            await require_extensions()

    async def test_advisory_lock(self, sqlite_engine: None) -> None:
        with pytest.raises(ConfigurationError, match=r"advisory_lock\(\)"):
            async with advisory_lock("connector-sync:reddit"):
                pytest.fail("the guard must fire before the block is entered")


class TestRequiredExtensions:
    def test_matches_the_bootstrap_script(self) -> None:
        """The constant and `01-extensions.sql` must not drift apart.

        If they do, `require_extensions()` either passes a database that is
        actually missing something or fails one that is fine -- and both make the
        check worthless.
        """
        repo_root = Path(__file__).resolve().parents[3]
        sql = (repo_root / "docker/local/postgres/01-extensions.sql").read_text()
        declared = {
            line.split('"')[1] for line in sql.splitlines() if line.startswith("CREATE EXTENSION")
        }
        assert declared == set(REQUIRED_EXTENSIONS)


class TestAdvisoryLockKey:
    def test_is_a_signed_64_bit_int(self) -> None:
        """PostgreSQL takes a bigint; anything wider is rejected at bind time."""
        for name in ("connector-sync:reddit", "scheduler:nightly", "", "x" * 512):
            key = advisory_lock_key(name)
            assert -(2**63) <= key < 2**63

    def test_is_stable_across_processes(self) -> None:
        """Not `hash()`: that is salted per process by PYTHONHASHSEED.

        A key that differs between replicas is a mutex that never excludes
        anything, and it fails open and silently.
        """
        assert advisory_lock_key("connector-sync:reddit") == -4_941_113_323_335_579_958

    def test_distinct_names_get_distinct_keys(self) -> None:
        keys = {advisory_lock_key(f"connector-sync:{slug}") for slug in ("reddit", "rss", "x")}
        assert len(keys) == 3


class _FakeResult:
    def __init__(self, value: object) -> None:
        self._value = value

    def scalar_one(self) -> object:
        return self._value


class _FakeConnection:
    """Records the protocol `advisory_lock()` speaks, without a server.

    The lock/unlock protocol is the whole point of that helper and it is
    PostgreSQL-only, so the unit suite cannot execute it. What it *can* do is
    pin the sequence: AUTOCOMMIT, lock, unlock, close. Every one of those is a
    correctness requirement argued for in the helper's docstring, and a
    regression in any of them is silent -- the lock still appears to work right
    up until two replicas hold it at once.
    """

    def __init__(self, try_lock_results: list[bool] | None = None) -> None:
        self.statements: list[tuple[str, object]] = []
        self.options: dict[str, object] = {}
        self.closed = False
        self._try_lock_results = list(try_lock_results or [])

    async def execution_options(self, **options: object) -> _FakeConnection:
        self.options.update(options)
        return self

    async def execute(self, statement: object, params: object = None) -> _FakeResult:
        sql = str(statement)
        self.statements.append((sql, params))
        if "pg_try_advisory_lock" in sql:
            granted = self._try_lock_results.pop(0) if self._try_lock_results else False
            return _FakeResult(granted)
        if "pg_advisory_unlock" in sql:
            return _FakeResult(True)
        return _FakeResult(None)

    async def close(self) -> None:
        self.closed = True


class _FakeEngine:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection
        self.dialect = SimpleNamespace(name="postgresql")

    async def connect(self) -> _FakeConnection:
        return self._connection


class TestAdvisoryLockProtocol:
    @staticmethod
    def _install(monkeypatch: pytest.MonkeyPatch, conn: _FakeConnection) -> None:
        monkeypatch.setattr(postgres_module, "get_engine", lambda: _FakeEngine(conn))

    async def test_acquires_releases_and_closes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conn = _FakeConnection()
        self._install(monkeypatch, conn)
        key = advisory_lock_key("connector-sync:reddit")

        async with advisory_lock("connector-sync:reddit") as held:
            assert cast(object, held) is conn
            assert [sql for sql, _ in conn.statements] == ["SELECT pg_advisory_lock(:key)"]

        assert [sql for sql, _ in conn.statements] == [
            "SELECT pg_advisory_lock(:key)",
            "SELECT pg_advisory_unlock(:key)",
        ]
        assert all(params == {"key": key} for _, params in conn.statements)
        assert conn.closed

    async def test_runs_in_autocommit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Otherwise the critical section sits `idle in transaction` throughout."""
        conn = _FakeConnection()
        self._install(monkeypatch, conn)

        async with advisory_lock(42):
            pass

        assert conn.options == {"isolation_level": "AUTOCOMMIT"}

    async def test_releases_when_the_body_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed sync must not leave the next tick locked out forever."""
        conn = _FakeConnection()
        self._install(monkeypatch, conn)

        with pytest.raises(RuntimeError):
            async with advisory_lock(42):
                raise RuntimeError("sync blew up")

        assert "SELECT pg_advisory_unlock(:key)" in [sql for sql, _ in conn.statements]
        assert conn.closed

    async def test_timeout_polls_try_lock_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _FakeConnection(try_lock_results=[False, False, True])
        self._install(monkeypatch, conn)
        monkeypatch.setattr(postgres_module, "_LOCK_POLL_INTERVAL_SECONDS", 0.0)

        async with advisory_lock(42, timeout_seconds=5.0):
            pass

        attempts = [sql for sql, _ in conn.statements if "pg_try_advisory_lock" in sql]
        assert len(attempts) == 3
        # The blocking form must not appear: a timed caller asked not to wait.
        assert not [sql for sql, _ in conn.statements if sql == "SELECT pg_advisory_lock(:key)"]

    async def test_timeout_raises_and_closes_the_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        conn = _FakeConnection(try_lock_results=[False])
        self._install(monkeypatch, conn)

        with pytest.raises(DependencyUnavailableError) as excinfo:
            async with advisory_lock(42, timeout_seconds=0.0):
                pytest.fail("the lock was never acquired")

        assert excinfo.value.details["lock_key"] == 42
        # A connection abandoned here would be leaked out of the pool for the
        # life of the process.
        assert conn.closed
        assert "pg_advisory_unlock" not in " ".join(sql for sql, _ in conn.statements)
