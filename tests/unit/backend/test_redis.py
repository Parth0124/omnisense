"""Unit tests for `backend/db/redis.py`.

No Redis server is involved. Two things are worth testing without one, and they
are the two things that are expensive to get wrong:

1. **The degradation contract** (`docs/architecture.md` §7.3). Inbound limiting
   must fail *closed* and outbound limiting must fail *open* under conservative
   static limits. A fake client that raises on every command is a faithful model
   of "Redis is down", and it is the only way to exercise those paths in a unit
   test.
2. **No I/O at import, and `check_redis()` never raises.** Both are pattern
   guarantees from `backend/db/session.py`; both are easy to break later with an
   innocent-looking refactor.
"""

from __future__ import annotations

import importlib
import socket
from typing import Any, cast

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

import backend.db.redis as redis_module
from backend.core.config import get_settings
from backend.db.redis import (
    CACHE_PREFIX,
    DEDUP_PREFIX,
    INBOUND_RATE_PREFIX,
    OUTBOUND_RATE_PREFIX,
    RateLimitDecision,
    cache_delete,
    cache_get_json,
    cache_set_json,
    check_inbound_rate_limit_fail_closed,
    check_outbound_rate_limit_fail_open,
    check_redis,
    dispose_redis,
    get_redis,
    mark_seen,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakePipeline:
    """Buffers commands like `redis.asyncio.client.Pipeline` and applies them in order.

    TTLs are stored as the millisecond value that was set rather than being
    aged against a real clock: these tests assert on limiting arithmetic, not on
    expiry, and a wall-clock dependency would make them flaky for no gain.
    """

    def __init__(self, store: dict[str, Any], ttls: dict[str, int]) -> None:
        self._store = store
        self._ttls = ttls
        self._queue: list[tuple[str, tuple[Any, ...]]] = []

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._queue.clear()

    def incr(self, key: str) -> FakePipeline:
        self._queue.append(("incr", (key,)))
        return self

    def expire(self, key: str, seconds: int, nx: bool = False) -> FakePipeline:
        self._queue.append(("expire", (key, seconds, nx)))
        return self

    def pttl(self, key: str) -> FakePipeline:
        self._queue.append(("pttl", (key,)))
        return self

    async def execute(self) -> list[Any]:
        results: list[Any] = []
        for name, args in self._queue:
            if name == "incr":
                (key,) = args
                self._store[key] = int(self._store.get(key, 0)) + 1
                results.append(self._store[key])
            elif name == "expire":
                key, seconds, nx = args
                if nx and key in self._ttls:
                    results.append(False)
                else:
                    self._ttls[key] = seconds * 1000
                    results.append(True)
            elif name == "pttl":
                (key,) = args
                results.append(self._ttls.get(key, -1))
        self._queue.clear()
        return results


class FakeRedis:
    """The subset of `redis.asyncio.Redis` these helpers actually touch."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}
        self.ttls: dict[str, int] = {}
        self.set_calls: list[dict[str, Any]] = []

    async def get(self, key: str) -> Any:
        return self.store.get(key)

    async def set(
        self, key: str, value: Any, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        self.set_calls.append({"key": key, "value": value, "ex": ex, "nx": nx})
        if nx and key in self.store:
            return None
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex * 1000
        return True

    async def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return FakePipeline(self.store, self.ttls)


class BrokenPipeline(FakePipeline):
    async def execute(self) -> list[Any]:
        raise RedisConnectionError("Error 61 connecting to localhost:6379. Connection refused.")


class BrokenRedis(FakeRedis):
    """Every command fails, exactly as it would with Redis down."""

    async def get(self, key: str) -> Any:
        raise RedisConnectionError("connection refused")

    async def set(
        self, key: str, value: Any, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        raise RedisConnectionError("connection refused")

    async def delete(self, key: str) -> int:
        raise RedisConnectionError("connection refused")

    def pipeline(self, transaction: bool = True) -> FakePipeline:
        return BrokenPipeline(self.store, self.ttls)


def _as_client(fake: FakeRedis) -> Redis:
    """The fakes are structural stand-ins; the annotations want the real type."""
    return cast(Redis, fake)


@pytest.fixture(autouse=True)
async def _reset_module_state() -> Any:
    """Clear the singleton and the process-local fallback windows between tests."""
    await dispose_redis()
    yield
    await dispose_redis()


# --------------------------------------------------------------------------- #
# Client lifecycle
# --------------------------------------------------------------------------- #


class TestClientLifecycle:
    def test_import_performs_no_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing must not read settings or build a pool.

        Enforced by making both operations explode and then re-executing the
        module. If someone later moves `get_settings()` or `ConnectionPool`
        construction to module scope, this fails loudly instead of turning test
        collection into something that needs a running Redis.
        """

        def boom(*args: object, **kwargs: object) -> object:
            raise AssertionError("import-time I/O")

        monkeypatch.setattr("backend.core.config.get_settings", boom)
        monkeypatch.setattr("redis.asyncio.ConnectionPool.from_url", boom)
        try:
            importlib.reload(redis_module)
            assert redis_module._client is None
            assert redis_module._pool is None
        finally:
            # The reload bound the sabotaged `get_settings` into the module's
            # namespace, where `monkeypatch` cannot reach it. Undo first, then
            # reload again, or every later test in the session inherits it.
            monkeypatch.undo()
            importlib.reload(redis_module)

    def test_get_redis_is_a_singleton(self) -> None:
        """Two calls share one pool; rebuilding per call would leak connections."""
        assert get_redis() is get_redis()

    async def test_dispose_resets_the_singleton(self) -> None:
        first = get_redis()
        await dispose_redis()
        assert get_redis() is not first

    async def test_dispose_is_safe_before_any_client_exists(self) -> None:
        """Lifespan shutdown runs even when startup failed before first use."""
        await dispose_redis()

    def test_pool_is_built_from_settings(self) -> None:
        settings = get_settings()
        client = get_redis()
        assert client.connection_pool.max_connections == settings.redis.max_connections
        # Values are decoded to `str`; every call site assumes it.
        assert client.connection_pool.connection_kwargs["decode_responses"] is True

    async def test_check_redis_returns_false_when_nothing_is_listening(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`/readyz` must report a bool, not propagate `ConnectionRefusedError`.

        Pointed at a port the OS just told us is free rather than at the default
        6379, so the result does not depend on whether the developer happens to
        have Redis running.
        """
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]

        monkeypatch.setenv("REDIS_URL", f"redis://127.0.0.1:{free_port}/0")
        get_settings.cache_clear()
        try:
            assert await check_redis() is False
        finally:
            get_settings.cache_clear()

    async def test_check_redis_returns_true_when_ping_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Pinger:
            async def ping(self) -> bool:
                return True

        monkeypatch.setattr(redis_module, "get_redis", lambda: Pinger())
        assert await check_redis() is True


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


class TestCache:
    async def test_round_trip(self) -> None:
        fake = FakeRedis()
        stored = await cache_set_json("sig:42", {"title": "x"}, client=_as_client(fake))
        assert stored is True
        assert await cache_get_json("sig:42", client=_as_client(fake)) == {"title": "x"}
        assert CACHE_PREFIX + "sig:42" in fake.store

    async def test_ttl_defaults_to_settings(self) -> None:
        fake = FakeRedis()
        await cache_set_json("k", 1, client=_as_client(fake))
        assert fake.set_calls[0]["ex"] == get_settings().redis.cache_ttl_seconds

    async def test_non_positive_ttl_writes_nothing(self) -> None:
        """`REDIS_CACHE_TTL_SECONDS=0` disables the cache; it must not write forever-keys."""
        fake = FakeRedis()
        assert await cache_set_json("k", 1, ttl_seconds=0, client=_as_client(fake)) is False
        assert fake.set_calls == []

    async def test_miss_returns_none(self) -> None:
        assert await cache_get_json("absent", client=_as_client(FakeRedis())) is None

    async def test_outage_reads_as_a_miss(self) -> None:
        """§7.3: caches miss through to PostgreSQL rather than failing the request."""
        assert await cache_get_json("k", client=_as_client(BrokenRedis())) is None

    async def test_outage_on_write_is_swallowed(self) -> None:
        assert await cache_set_json("k", 1, client=_as_client(BrokenRedis())) is False

    async def test_corrupt_entry_reads_as_a_miss(self) -> None:
        """An LRU eviction mid-write must not raise inside a request handler."""
        fake = FakeRedis()
        fake.store[CACHE_PREFIX + "k"] = "{not json"
        assert await cache_get_json("k", client=_as_client(fake)) is None

    async def test_delete_reports_whether_a_key_went_away(self) -> None:
        fake = FakeRedis()
        await cache_set_json("k", 1, client=_as_client(fake))
        assert await cache_delete("k", client=_as_client(fake)) is True
        assert await cache_delete("k", client=_as_client(fake)) is False
        assert await cache_delete("k", client=_as_client(BrokenRedis())) is False


# --------------------------------------------------------------------------- #
# Dedup seen-set
# --------------------------------------------------------------------------- #


class TestMarkSeen:
    async def test_first_sight_then_duplicate(self) -> None:
        fake = FakeRedis()
        assert await mark_seen("reddit", "abc", client=_as_client(fake)) is True
        assert await mark_seen("reddit", "abc", client=_as_client(fake)) is False
        assert DEDUP_PREFIX + "reddit:abc" in fake.store

    async def test_uses_set_nx_with_a_ttl(self) -> None:
        """Both flags matter: NX is the atomicity, EX is the eviction guarantee."""
        fake = FakeRedis()
        await mark_seen("rss", "hash", ttl_seconds=60, client=_as_client(fake))
        assert fake.set_calls[0]["nx"] is True
        assert fake.set_calls[0]["ex"] == 60

    async def test_ttl_defaults_to_connector_settings(self) -> None:
        fake = FakeRedis()
        await mark_seen("rss", "hash", client=_as_client(fake))
        assert fake.set_calls[0]["ex"] == get_settings().connectors.dedup_ttl_seconds

    async def test_namespaces_are_independent(self) -> None:
        fake = FakeRedis()
        assert await mark_seen("reddit", "same", client=_as_client(fake)) is True
        assert await mark_seen("rss", "same", client=_as_client(fake)) is True

    async def test_outage_reports_first_sight(self) -> None:
        """§7.3: dedup degrades to the database-level upsert.

        Returning `False` here would silently drop records for the duration of a
        Redis outage, which is unrecoverable data loss. Returning `True` costs a
        duplicate fetch that PostgreSQL's identity layer then absorbs.
        """
        assert await mark_seen("reddit", "abc", client=_as_client(BrokenRedis())) is True

    async def test_disabled_ttl_skips_redis_entirely(self) -> None:
        fake = FakeRedis()
        assert await mark_seen("reddit", "abc", ttl_seconds=0, client=_as_client(fake)) is True
        assert fake.set_calls == []


# --------------------------------------------------------------------------- #
# The degradation contract
# --------------------------------------------------------------------------- #


class TestInboundFailsClosed:
    async def test_allows_up_to_the_limit_then_denies(self) -> None:
        fake = FakeRedis()
        decisions = [
            await check_inbound_rate_limit_fail_closed(
                "tenant-a", limit=3, window_seconds=60, client=_as_client(fake)
            )
            for _ in range(4)
        ]
        assert [d.allowed for d in decisions] == [True, True, True, False]
        assert [d.remaining for d in decisions] == [2, 1, 0, 0]
        assert all(d.degraded is False for d in decisions)
        assert decisions[-1].retry_after_seconds == 60
        assert INBOUND_RATE_PREFIX + "tenant-a" in fake.store

    async def test_buckets_are_independent(self) -> None:
        fake = FakeRedis()
        await check_inbound_rate_limit_fail_closed(
            "tenant-a", limit=1, window_seconds=60, client=_as_client(fake)
        )
        other = await check_inbound_rate_limit_fail_closed(
            "tenant-b", limit=1, window_seconds=60, client=_as_client(fake)
        )
        assert other.allowed is True

    async def test_window_is_not_extended_by_later_requests(self) -> None:
        """`EXPIRE NX`: the first request starts the window, the rest ride it out."""
        fake = FakeRedis()
        for _ in range(3):
            await check_inbound_rate_limit_fail_closed(
                "tenant-a", limit=10, window_seconds=60, client=_as_client(fake)
            )
        assert fake.ttls[INBOUND_RATE_PREFIX + "tenant-a"] == 60_000

    async def test_denies_when_redis_is_down(self) -> None:
        """The whole point: no shared counter means no way to trust the caller."""
        decision = await check_inbound_rate_limit_fail_closed(
            "tenant-a", limit=1000, window_seconds=60, client=_as_client(BrokenRedis())
        )
        assert decision.allowed is False
        assert decision.degraded is True
        assert decision.remaining == 0
        assert decision.retry_after_seconds == 60

    async def test_outage_denial_is_repeatable_and_never_raises(self) -> None:
        broken = _as_client(BrokenRedis())
        for _ in range(5):
            decision = await check_inbound_rate_limit_fail_closed(
                "tenant-a", limit=1000, window_seconds=30, client=broken
            )
            assert decision.allowed is False


class TestOutboundFailsOpen:
    async def test_allows_up_to_the_limit_then_denies(self) -> None:
        fake = FakeRedis()
        decisions = [
            await check_outbound_rate_limit_fail_open(
                "reddit:acct-1", limit=2, window_seconds=60, client=_as_client(fake)
            )
            for _ in range(3)
        ]
        assert [d.allowed for d in decisions] == [True, True, False]
        assert all(d.degraded is False for d in decisions)
        assert OUTBOUND_RATE_PREFIX + "reddit:acct-1" in fake.store

    async def test_keeps_fetching_when_redis_is_down(self) -> None:
        """Ingestion survives the outage instead of halting."""
        decision = await check_outbound_rate_limit_fail_open(
            "reddit:acct-1", limit=10, window_seconds=60, client=_as_client(BrokenRedis())
        )
        assert decision.allowed is True
        assert decision.degraded is True

    async def test_fallback_limit_is_conservative_not_unlimited(self) -> None:
        """Fail *open* still means throttled: half the configured budget, locally.

        Unlimited fetching is the one failure mode that cannot be undone by
        fixing Redis -- the third party bans the credential.
        """
        broken = _as_client(BrokenRedis())
        allowed = [
            (
                await check_outbound_rate_limit_fail_open(
                    "reddit:acct-1", limit=10, window_seconds=60, client=broken
                )
            ).allowed
            for _ in range(8)
        ]
        assert allowed == [True] * 5 + [False] * 3

    async def test_fallback_limit_never_reaches_zero(self) -> None:
        """A limit of 1 must still permit one request, not round down to a stall."""
        decision = await check_outbound_rate_limit_fail_open(
            "reddit:acct-1", limit=1, window_seconds=60, client=_as_client(BrokenRedis())
        )
        assert decision.allowed is True
        assert decision.limit == 1

    async def test_fallback_buckets_are_independent(self) -> None:
        """One account exhausting the local budget must not stall another."""
        broken = _as_client(BrokenRedis())
        for _ in range(4):
            await check_outbound_rate_limit_fail_open(
                "reddit:acct-1", limit=2, window_seconds=60, client=broken
            )
        other = await check_outbound_rate_limit_fail_open(
            "reddit:acct-2", limit=2, window_seconds=60, client=broken
        )
        assert other.allowed is True

    async def test_fallback_window_expires(self) -> None:
        """A zero-length window is the fastest way to prove the window resets."""
        broken = _as_client(BrokenRedis())
        first = await check_outbound_rate_limit_fail_open(
            "reddit:acct-1", limit=1, window_seconds=0, client=broken
        )
        second = await check_outbound_rate_limit_fail_open(
            "reddit:acct-1", limit=1, window_seconds=0, client=broken
        )
        assert first.allowed is True
        assert second.allowed is True

    async def test_fallback_state_is_cleared_on_dispose(self) -> None:
        broken = _as_client(BrokenRedis())
        for _ in range(4):
            await check_outbound_rate_limit_fail_open(
                "reddit:acct-1", limit=2, window_seconds=60, client=broken
            )
        await dispose_redis()
        after = await check_outbound_rate_limit_fail_open(
            "reddit:acct-1", limit=2, window_seconds=60, client=broken
        )
        assert after.allowed is True

    async def test_the_two_directions_disagree_under_the_same_outage(self) -> None:
        """§7.3 in one assertion: same failure, opposite answers."""
        broken = _as_client(BrokenRedis())
        inbound = await check_inbound_rate_limit_fail_closed(
            "tenant-a", limit=10, window_seconds=60, client=broken
        )
        outbound = await check_outbound_rate_limit_fail_open(
            "reddit:acct-1", limit=10, window_seconds=60, client=broken
        )
        assert inbound.allowed is False
        assert outbound.allowed is True
        assert inbound.degraded and outbound.degraded


class TestRateLimitDecision:
    def test_truthiness_follows_allowed(self) -> None:
        """`if decision:` must not silently admit a denied request."""
        denied = RateLimitDecision(
            allowed=False, limit=1, remaining=0, retry_after_seconds=5, degraded=True
        )
        granted = RateLimitDecision(
            allowed=True, limit=1, remaining=0, retry_after_seconds=0, degraded=False
        )
        assert not denied
        assert granted
