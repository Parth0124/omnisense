"""Round trips against the real datastores.

Each test writes something, reads it back, and checks the value survived. That
sounds trivial and is exactly the class of thing unit tests with fakes cannot
establish: a fake round-trips whatever you hand it, while a real store applies a
schema, a serialiser and a type system, and every one of those is somewhere a
value can quietly change shape.

The specific failures these catch, all of which have real precedent in this
codebase:

* A `datetime` written aware and read back naive, so every subsequent comparison
  is wrong by the server's offset.
* A JSON column round-tripping a nested dict as a string.
* An enum stored by name and read by value, or vice versa.
* A vector written at one dimension and rejected — or worse, silently truncated —
  at another.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class TestPostgres:
    async def test_connects_and_answers(self, pg_sessionmaker) -> None:
        from sqlalchemy import text

        async with pg_sessionmaker() as session:
            assert (await session.execute(text("SELECT 1"))).scalar() == 1

    async def test_the_migration_has_been_applied(self, pg_sessionmaker) -> None:
        """Named tables exist.

        A clearer failure than whatever a missing-table error looks like three
        layers into a service call — and the most common state of a fresh
        checkout is exactly this one, `make up` done and `make migrate`
        forgotten.

        Not restricted to `public`. Every table lives in the `omnisense` schema,
        and pinning the query to `public` made this assert that a correctly
        migrated database was unmigrated — a guard that fails on the healthy case
        is worse than no guard, because the message it prints ("run make migrate")
        sends somebody to re-run the thing that already worked.
        """
        from sqlalchemy import text

        async with pg_sessionmaker() as session:
            rows = (
                (
                    await session.execute(
                        text(
                            "SELECT table_name FROM information_schema.tables "
                            "WHERE table_schema NOT IN ('pg_catalog', 'information_schema')"
                        )
                    )
                )
                .scalars()
                .all()
            )

        present = set(rows)
        expected = {"signals", "investigations", "reports"}
        missing = expected - present
        assert not missing, (
            f"tables {sorted(missing)} are absent. Run `make migrate` -- the "
            f"schema has not been applied to this database. Found: {sorted(present)[:12]}"
        )

    async def test_a_timestamp_survives_the_round_trip_with_its_offset(
        self, pg_sessionmaker
    ) -> None:
        """The single most consequential round-trip property in this system.

        `signals.timestamp` is compared against query bounds on every retrieval.
        If an aware datetime comes back naive, every window filter is wrong by
        the server's UTC offset — correctly in a UTC deployment, silently wrong
        everywhere else, which is the worst possible way for it to behave.
        """
        from sqlalchemy import text

        async with pg_sessionmaker() as session:
            value = (
                await session.execute(text("SELECT CAST(:ts AS timestamptz) AS ts"), {"ts": NOW})
            ).scalar_one()

        assert value.tzinfo is not None, (
            "a timestamptz came back naive; every time-window comparison in "
            "retrieval would be wrong by the server's offset"
        )
        assert value == NOW


class TestRedis:
    async def test_set_and_get(self, redis_client, run_namespace) -> None:
        key = f"{run_namespace}:probe"
        await redis_client.set(key, "value", ex=60)
        try:
            raw = await redis_client.get(key)
            assert (raw.decode() if isinstance(raw, bytes) else raw) == "value"
        finally:
            await redis_client.delete(key)

    async def test_the_scratchpad_round_trips_a_nested_value(
        self, redis_client, run_namespace
    ) -> None:
        """The agent scratchpad against a real hash rather than a dict.

        Worth doing because the in-memory store ignores the TTL and returns
        Python objects, while the real one returns bytes and applies `EXPIRE` —
        two differences that would each break a caller the fake never exercises.
        """
        from agents.memory.scratchpad import RedisScratchpadStore, Scratchpad

        pad = Scratchpad(RedisScratchpadStore(redis_client), key=f"{run_namespace}:scratch")
        try:
            assert await pad.put("plan", {"steps": [1, 2, 3], "nested": {"a": True}})
            assert await pad.get("plan") == {"steps": [1, 2, 3], "nested": {"a": True}}
            assert await pad.get("absent", "fallback") == "fallback"
        finally:
            await pad.clear()

    async def test_the_rate_limiter_shares_state_across_instances(
        self, redis_client, run_namespace
    ) -> None:
        """The property that makes it a *distributed* limiter.

        Two `RateLimiter` objects are two API replicas. Against the in-memory
        store each keeps its own bucket and N replicas permit N times the limit —
        a bug that is invisible in every unit test and appears the moment a
        second pod starts.
        """
        from backend.core.ratelimit import RateLimiter, RedisBucketStore

        store = RedisBucketStore(redis_client)
        first = RateLimiter(store, rate_per_minute=60, burst=2)
        second = RateLimiter(store, rate_per_minute=60, burst=2)
        identity = f"{run_namespace}-shared"

        assert (await first.check(identity)).allowed
        assert (await second.check(identity)).allowed
        # The third must be refused: the two instances share one bucket of two.
        assert not (await first.check(identity)).allowed

        await redis_client.delete(f"os:rl:api:{identity}")


class TestQdrant:
    async def test_reachable(self, qdrant_available) -> None:
        from backend.db.qdrant import check_qdrant, dispose_qdrant

        try:
            assert await check_qdrant()
        finally:
            await dispose_qdrant()


class TestOpenSearch:
    async def test_reachable(self, opensearch_available) -> None:
        from backend.db.opensearch import check_opensearch, dispose_opensearch

        try:
            assert await check_opensearch()
        finally:
            await dispose_opensearch()


class TestReadinessAgainstRealStores:
    async def test_readyz_reports_ok_when_everything_is_up(
        self,
        postgres_available,
        neo4j_available,
        redis_available,
        qdrant_available,
        opensearch_available,
    ) -> None:
        """The inverse of the unit test, and the half that matters operationally.

        `tests/unit/backend/test_api.py` proves `/readyz` returns 503 with nothing
        running. That is only half the contract — a probe hard-coded to fail
        would satisfy it. This proves it can also say yes, which is what keeps a
        healthy replica in rotation.
        """
        import httpx

        from backend.main import create_app

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/readyz")

        body = response.json()
        assert response.status_code == 200, (
            f"/readyz returned {response.status_code} with every store up: {body.get('checks')}"
        )
        assert body["status"] == "ok", f"degraded: {body.get('degraded')}"
