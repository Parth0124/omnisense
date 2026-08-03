"""Unit tests for the shared token bucket and the retry policy.

These two modules are the ones whose bugs are invisible until a provider bans
us, so the tests target the specific ways each is normally got wrong:

- a check-and-decrement that is not atomic (there is a test that *demonstrates*
  the double-spend with a naive GET-then-SET, so the Lua is justified by
  evidence rather than by assertion);
- a partial multi-key acquisition that leaks the tokens it did take;
- a provider hint that is allowed to raise a bucket instead of only lowering it;
- jitter asserted against real `random`, which is a flaky test by construction --
  every RNG here is injected and deterministic;
- a `Retry-After` used as an upper bound on jitter rather than as an override;
- a circuit breaker that counts retry *attempts* instead of failed operations,
  and therefore opens on the first flaky page.

Everything runs against `fakeredis`, which executes the Lua scripts for real
through `lupa`. No Redis, no network, no `time.sleep`: the clock and the sleep
are injected, so the suite is deterministic and takes milliseconds.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import pytest
from fakeredis import aioredis

from connectors.exceptions import (
    AuthError,
    CircuitOpenError,
    ConnectorError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.protocol import RateLimiter, RateLimitHint, RateLimitPolicy, SyncMode
from connectors.ratelimit.backoff import (
    DEFAULT_POLICY,
    BackoffPolicy,
    CircuitBreaker,
    CircuitState,
    RetryController,
    full_jitter,
    next_delay,
    retry_after_of,
    retry_page,
    tenacity_wait,
)
from connectors.ratelimit.limiter import (
    MAX_PARK_SECONDS,
    RELEASE_LUA,
    BucketPolicy,
    InMemoryLimiter,
    NullLimiter,
    TokenBucketLimiter,
)

pytestmark = pytest.mark.unit

T0 = 1_800_000_000.0
"""A fixed, realistic UNIX time. Realistic matters: the limiter distinguishes a
`X-RateLimit-Reset` that is a timestamp from one that is delta-seconds by
magnitude, and a clock starting at 0.0 would make every timestamp look like a
delta."""

KEY = "os:rl:demo"


# --------------------------------------------------------------------------- #
# Deterministic doubles
# --------------------------------------------------------------------------- #


class FakeClock:
    """A wall clock the test moves by hand."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleep:
    """Records requested sleeps and advances the clock instead of blocking.

    Advancing is what makes the wait path testable: the bucket refills because
    time moved, and no test spends a real second proving it.
    """

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


class ZeroJitter(random.Random):
    """`random()` -> 0.0, `uniform(a, b)` -> a. The lower edge of every range."""

    def random(self) -> float:
        return 0.0

    def uniform(self, a: float, b: float) -> float:
        return a


class MaxJitter(random.Random):
    """`random()` -> 1.0, `uniform(a, b)` -> b, recording every range it was asked for.

    Returning the top of the range is what lets a test assert the *ceiling* the
    policy computed, which is the part of full jitter that carries the policy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.ranges: list[tuple[float, float]] = []

    def random(self) -> float:
        return 1.0

    def uniform(self, a: float, b: float) -> float:
        self.ranges.append((a, b))
        return b


class BrokenRedis:
    """A client that fails the way a dead Redis does: on every call."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or ConnectionError("connection refused")
        self.calls = 0

    async def script_load(self, script: str) -> str:
        self.calls += 1
        raise self.error

    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: Any) -> Any:
        self.calls += 1
        raise self.error


@pytest.fixture(params=[True, False], ids=["decoded", "bytes"])
async def redis(request: pytest.FixtureRequest) -> Any:
    """A real Lua-executing fake, in both response modes.

    Parametrized over `decode_responses` because this module does not construct
    the client -- a deployment may hand it either, and a `float(b'9.5')` that
    only works in one mode is a production-only failure.
    """
    client = aioredis.FakeRedis(decode_responses=request.param)
    yield client
    await client.aclose()


def limiter(redis: Any, clock: FakeClock, **kwargs: Any) -> TokenBucketLimiter:
    kwargs.setdefault("default_policy", BucketPolicy(capacity=3, refill_per_second=1.0))
    kwargs.setdefault("rng", ZeroJitter())
    return TokenBucketLimiter(redis, clock=clock, **kwargs)


async def tokens_in(redis: Any, key: str) -> float:
    raw = await redis.hget(key, "tokens")
    return float(raw)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


class TestBucketPolicy:
    def test_ttl_is_two_full_refills(self) -> None:
        """§5.1: idle buckets must expire, and an expired bucket cold-starts full."""
        assert BucketPolicy(capacity=10, refill_per_second=1.0).ttl_seconds == 20.0
        assert BucketPolicy(capacity=60, refill_per_second=2.0).ttl_seconds == 60.0

    @pytest.mark.parametrize(
        ("capacity", "refill"),
        [(0, 1.0), (-1, 1.0), (10, 0.0), (10, -1.0)],
    )
    def test_rejects_a_bucket_that_can_never_recover(self, capacity: int, refill: float) -> None:
        """A zero refill is an infinite TTL and a bucket that denies forever.

        Failing at construction rather than at the first acquire, because the
        alternative presents as "the connector silently stopped fetching".
        """
        with pytest.raises(ValueError):
            BucketPolicy(capacity=capacity, refill_per_second=refill)

    def test_backfill_derives_the_reduced_bucket(self) -> None:
        """A historical crawl must not crowd out live sync (§5.1)."""
        policy = RateLimitPolicy(requests_per_minute=60, burst=10)
        live = BucketPolicy.from_rate_limit_policy(policy, SyncMode.INCREMENTAL)
        backfill = BucketPolicy.from_rate_limit_policy(policy, SyncMode.BACKFILL)
        assert live.refill_per_second == 1.0
        assert backfill.refill_per_second == pytest.approx(0.25)

    def test_scaling_never_produces_a_zero_capacity_bucket(self) -> None:
        """The fail-open fallback scales every policy; a 1-token bucket must survive it."""
        assert BucketPolicy(capacity=1, refill_per_second=1.0).scaled(0.5).capacity == 1


# --------------------------------------------------------------------------- #
# Token bucket: the basics
# --------------------------------------------------------------------------- #


class TestAcquire:
    async def test_grants_a_burst_then_denies(self, redis: Any) -> None:
        lim = limiter(redis, FakeClock())
        for _ in range(3):
            await lim.acquire([KEY])
        with pytest.raises(QuotaError) as excinfo:
            await lim.acquire([KEY])
        assert excinfo.value.details["bucket"] == KEY

    async def test_denial_reports_when_the_bucket_recovers(self, redis: Any) -> None:
        """`QuotaError` is a partial success and the runtime reschedules at
        `reset_at`, so a denial that carried no time would be unschedulable."""
        clock = FakeClock()
        lim = limiter(redis, clock)
        for _ in range(3):
            await lim.acquire([KEY])
        with pytest.raises(QuotaError) as excinfo:
            await lim.acquire([KEY])
        assert excinfo.value.retry_after_seconds == pytest.approx(1.0, abs=0.01)
        assert excinfo.value.reset_at == pytest.approx(T0 + 1.0, abs=0.01)

    async def test_state_is_a_hash_of_tokens_and_last_refill(self, redis: Any) -> None:
        clock = FakeClock()
        await limiter(redis, clock).acquire([KEY])
        state = await redis.hgetall(KEY)
        fields = {k.decode() if isinstance(k, bytes) else k for k in state}
        assert fields == {"tokens", "last_refill_ms"}
        assert await tokens_in(redis, KEY) == pytest.approx(2.0)

    async def test_ttl_is_set_from_the_policy(self, redis: Any) -> None:
        """Without the TTL every per-host bucket becomes a permanent key."""
        await limiter(redis, FakeClock()).acquire([KEY])
        assert await redis.pttl(KEY) == pytest.approx(6_000, abs=200)

    async def test_an_evicted_bucket_cold_starts_full(self, redis: Any) -> None:
        """Eviction is Redis's job, so the test performs it; what is under test
        is that an absent key is read as *full* and not as empty."""
        lim = limiter(redis, FakeClock())
        for _ in range(3):
            await lim.acquire([KEY])
        await redis.delete(KEY)
        for _ in range(3):
            await lim.acquire([KEY])  # must not raise

    async def test_refills_at_the_declared_rate(self, redis: Any) -> None:
        clock = FakeClock()
        lim = limiter(redis, clock)
        for _ in range(3):
            await lim.acquire([KEY])

        clock.advance(2.0)
        await lim.acquire([KEY])
        await lim.acquire([KEY])
        with pytest.raises(QuotaError):
            await lim.acquire([KEY])

    async def test_refill_never_exceeds_capacity(self, redis: Any) -> None:
        clock = FakeClock()
        lim = limiter(redis, clock)
        await lim.acquire([KEY])
        clock.advance(10_000.0)
        for _ in range(3):
            await lim.acquire([KEY])
        with pytest.raises(QuotaError):
            await lim.acquire([KEY])

    async def test_a_clock_running_backwards_does_not_drain_the_bucket(self, redis: Any) -> None:
        """`now_ms` comes from the caller's wall clock and callers are different
        hosts; a negative elapsed must clamp to zero rather than remove tokens."""
        clock = FakeClock()
        lim = limiter(redis, clock)
        await lim.acquire([KEY])
        clock.advance(-30.0)  # this replica's clock is behind the last writer's
        assert await tokens_in(redis, KEY) == pytest.approx(2.0)
        await lim.acquire([KEY])
        assert await tokens_in(redis, KEY) == pytest.approx(1.0)

    async def test_no_keys_is_a_no_op(self, redis: Any) -> None:
        await limiter(redis, FakeClock()).acquire([])
        assert await redis.keys("*") == []


# --------------------------------------------------------------------------- #
# Atomicity
# --------------------------------------------------------------------------- #


class TestAtomicity:
    async def test_concurrent_callers_never_overspend_the_bucket(self, redis: Any) -> None:
        """Fifty callers, ten tokens, exactly ten grants.

        This is the invariant the Lua exists for. It holds because Redis runs a
        script to completion with nothing interleaved.
        """
        lim = limiter(
            redis, FakeClock(), default_policy=BucketPolicy(capacity=10, refill_per_second=1.0)
        )

        async def attempt() -> bool:
            try:
                await lim.acquire([KEY])
            except QuotaError:
                return False
            return True

        granted = await asyncio.gather(*(attempt() for _ in range(50)))
        assert sum(granted) == 10
        assert await tokens_in(redis, KEY) == pytest.approx(0.0, abs=1e-9)

    async def test_get_then_set_double_spends_the_same_last_token(self, redis: Any) -> None:
        """The bug the script prevents, demonstrated rather than asserted.

        The `await asyncio.sleep(0)` between the read and the write is not
        cheating: it stands in for the network round trip that unavoidably
        separates two commands issued from Python. In production that gap is a
        millisecond of real time with seven other workers running inside it.
        """
        await redis.hset(KEY, "tokens", "10")

        async def naive_acquire() -> bool:
            tokens = float(await redis.hget(KEY, "tokens"))
            await asyncio.sleep(0)
            if tokens < 1:
                return False
            await redis.hset(KEY, "tokens", tokens - 1)
            return True

        granted = await asyncio.gather(*(naive_acquire() for _ in range(50)))
        assert sum(granted) > 10, "expected the unsafe implementation to over-issue"
        assert float(await redis.hget(KEY, "tokens")) == 9.0

    async def test_recovers_from_a_flushed_script_cache(self, redis: Any) -> None:
        """Redis restarts and operators run SCRIPT FLUSH; a NOSCRIPT that was not
        handled would fail every acquire from then on -- and since the failure
        path fails *open*, silently stop limiting entirely."""
        lim = limiter(redis, FakeClock())
        await lim.acquire([KEY])
        await redis.script_flush()
        await lim.acquire([KEY])
        assert await tokens_in(redis, KEY) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# All-or-nothing
# --------------------------------------------------------------------------- #


class TestAllOrNothing:
    async def test_a_failure_on_the_third_key_releases_the_first_two(self, redis: Any) -> None:
        """Otherwise every contended call leaks a token from the uncontended
        buckets and the effective limit tightens for the life of the process."""
        clock = FakeClock()
        lim = limiter(redis, clock)
        exhausted = "os:rl:demo:acct_1"
        for _ in range(3):
            await lim.acquire([exhausted])

        with pytest.raises(QuotaError):
            await lim.acquire(["a", "b", exhausted])

        assert await tokens_in(redis, "a") == pytest.approx(3.0)
        assert await tokens_in(redis, "b") == pytest.approx(3.0)

    async def test_a_failed_acquisition_leaves_the_bucket_exactly_as_it_was(
        self, redis: Any
    ) -> None:
        """Including the refill that accrued in between.

        The release credits the token back without rewriting `last_refill_ms`,
        so the bucket resumes from its original anchor: a failed call is a no-op
        across time, not merely at the instant it failed.
        """
        clock = FakeClock()
        never_refills = BucketPolicy(capacity=1, refill_per_second=1e-6)
        lim = limiter(redis, clock, policies={"blocked": never_refills})
        await lim.acquire(["blocked"])  # and it stays empty
        await lim.acquire(["free"])  # 3 -> 2 at T0

        clock.advance(1.0)  # "free" is worth 3 again by now
        with pytest.raises(QuotaError):
            await lim.acquire(["free", "blocked"])

        assert await tokens_in(redis, "free") == pytest.approx(3.0)

    async def test_a_repeated_key_is_charged_once(self, redis: Any) -> None:
        """`[connector, account, host]` can collide; a caller means one call."""
        await limiter(redis, FakeClock()).acquire([KEY, KEY, KEY])
        assert await tokens_in(redis, KEY) == pytest.approx(2.0)

    async def test_releasing_a_bucket_that_expired_does_not_lower_it(self, redis: Any) -> None:
        """An absent bucket cold-starts full, so crediting one token back into it
        would write 1 where 3 was implied -- an eviction between the acquire and
        the release would *tighten* the limit instead of restoring it.

        Driven against the script directly because the window it guards is
        exactly the one the public API cannot open on demand: the key is created
        by the acquire microseconds earlier, and only maxmemory eviction or a
        FLUSHDB can take it away in between.
        """
        sha = await redis.script_load(RELEASE_LUA)
        assert float(await redis.evalsha(sha, 1, "gone", 3, 1, 60_000)) == -1.0
        assert await redis.exists("gone") == 0

    async def test_the_release_script_caps_at_capacity(self, redis: Any) -> None:
        """A double release -- or a release racing a refill -- must not mint
        tokens above the burst the provider agreed to."""
        sha = await redis.script_load(RELEASE_LUA)
        await redis.hset("full", mapping={"tokens": "3", "last_refill_ms": "0"})
        assert float(await redis.evalsha(sha, 1, "full", 3, 1, 60_000)) == 3.0


# --------------------------------------------------------------------------- #
# Waiting
# --------------------------------------------------------------------------- #


class TestWaiting:
    async def test_waits_for_a_refill_when_given_a_budget(self, redis: Any) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        lim = limiter(redis, clock, sleep=sleeper)
        for _ in range(3):
            await lim.acquire([KEY])

        await lim.acquire([KEY], timeout_seconds=30.0)

        assert len(sleeper.calls) == 1
        assert sleeper.calls[0] == pytest.approx(1.0, abs=0.01)

    async def test_no_budget_means_no_wait(self, redis: Any) -> None:
        """The default is fail-fast: a `QuotaError` reschedules the run, while an
        unbounded wait pins a worker on a bucket that may not refill for an hour."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        lim = limiter(redis, clock, sleep=sleeper)
        for _ in range(3):
            await lim.acquire([KEY])

        with pytest.raises(QuotaError):
            await lim.acquire([KEY])
        assert sleeper.calls == []

    async def test_a_wait_longer_than_the_budget_fails_immediately(self, redis: Any) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        lim = limiter(redis, clock, sleep=sleeper)
        for _ in range(3):
            await lim.acquire([KEY])

        with pytest.raises(QuotaError):
            await lim.acquire([KEY], timeout_seconds=0.25)
        assert sleeper.calls == [], "must not sleep past the caller's budget"

    async def test_the_wait_is_jittered_upward(self, redis: Any) -> None:
        """Every worker blocked on one bucket is told the same refill instant, so
        an unjittered wait wakes them into the same millisecond."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        lim = limiter(redis, clock, sleep=sleeper, rng=MaxJitter())
        for _ in range(3):
            await lim.acquire([KEY])

        await lim.acquire([KEY], timeout_seconds=30.0)
        assert sleeper.calls[0] == pytest.approx(1.1, abs=0.01)

    async def test_a_partial_acquisition_is_released_before_waiting(self, redis: Any) -> None:
        """The tokens must not be held across the sleep, or a worker waiting on a
        contended bucket starves every other worker on the uncontended ones."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        lim = limiter(redis, clock, sleep=sleeper)
        for _ in range(3):
            await lim.acquire(["blocked"])

        await lim.acquire(["free", "blocked"], timeout_seconds=30.0)

        # One sleep, then both keys taken exactly once on the successful pass.
        assert len(sleeper.calls) == 1
        assert await tokens_in(redis, "free") == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# observe(): downward only
# --------------------------------------------------------------------------- #


class TestObserve:
    async def test_clamps_the_bucket_down_to_the_reported_remaining(self, redis: Any) -> None:
        """Provider truth beats local estimate (§5.2)."""
        lim = limiter(
            redis, FakeClock(), default_policy=BucketPolicy(capacity=40, refill_per_second=1.0)
        )
        await lim.acquire([KEY])
        await lim.observe([KEY], RateLimitHint(remaining=3))
        assert await tokens_in(redis, KEY) == pytest.approx(3.0)

    async def test_never_raises_a_bucket(self, redis: Any) -> None:
        """A stale response must not refund tokens we have already spent."""
        clock = FakeClock()
        lim = limiter(redis, clock)
        for _ in range(3):
            await lim.acquire([KEY])

        await lim.observe([KEY], RateLimitHint(remaining=40))
        assert await tokens_in(redis, KEY) == pytest.approx(0.0, abs=1e-9)
        with pytest.raises(QuotaError):
            await lim.acquire([KEY])

    async def test_compares_against_the_refilled_count_not_the_stored_one(self, redis: Any) -> None:
        """A stored 0 that has been refilling for two seconds is really 2;
        comparing the provider's 1 against the stale 0 would conclude there is
        nothing to tighten and leave the bucket standing at 2."""
        clock = FakeClock()
        lim = limiter(redis, clock)
        for _ in range(3):
            await lim.acquire([KEY])
        clock.advance(2.0)

        await lim.observe([KEY], RateLimitHint(remaining=1))
        assert await tokens_in(redis, KEY) == pytest.approx(1.0)

    async def test_an_observation_above_capacity_is_clamped_to_capacity(self, redis: Any) -> None:
        clock = FakeClock()
        lim = limiter(redis, clock)
        await lim.acquire([KEY])
        await lim.observe([KEY], RateLimitHint(remaining=9_999))
        assert await tokens_in(redis, KEY) == pytest.approx(2.0)

    async def test_a_hint_with_nothing_useful_writes_nothing(self, redis: Any) -> None:
        await limiter(redis, FakeClock()).observe([KEY], RateLimitHint())
        assert await redis.exists(KEY) == 0

    async def test_creates_a_clamped_bucket_that_did_not_exist(self, redis: Any) -> None:
        """The first response of a run can carry a hint before any acquire has
        created the key; treating that as "no bucket, nothing to do" would throw
        away the only authoritative number we get."""
        lim = limiter(redis, FakeClock())
        await lim.observe([KEY], RateLimitHint(remaining=1))
        assert await tokens_in(redis, KEY) == pytest.approx(1.0)

    async def test_retry_after_alone_drains_and_parks_the_bucket(self, redis: Any) -> None:
        """A provider does not send `Retry-After` on a request it was happy to
        serve. Reading it as no-information leaves the bucket handing out tokens
        straight into the next 429."""
        clock = FakeClock()
        lim = limiter(redis, clock)
        await lim.observe([KEY], RateLimitHint(retry_after_seconds=30.0))

        assert await tokens_in(redis, KEY) == pytest.approx(0.0)
        clock.advance(5.0)
        with pytest.raises(QuotaError):
            await lim.acquire([KEY])  # no refill before the provider's reset

        clock.advance(26.0)
        await lim.acquire([KEY])  # ... and it recovers once the reset passes

    async def test_the_park_outlives_the_plain_ttl(self, redis: Any) -> None:
        """A parked bucket that expired before its reset instant would cold-start
        full and undo the clamp, which is the one case where expiry is wrong."""
        lim = limiter(redis, FakeClock())
        await lim.observe([KEY], RateLimitHint(retry_after_seconds=60.0))
        assert await redis.pttl(KEY) >= 60_000

    async def test_reset_at_is_read_as_a_delta_or_a_timestamp(self, redis: Any) -> None:
        """Providers use both spellings and neither labels itself; guessing wrong
        either parks a bucket until 2033 or does nothing at all."""
        clock = FakeClock()
        lim = limiter(redis, clock)

        await lim.observe(["delta"], RateLimitHint(remaining=0, reset_at=30.0))
        await lim.observe(["stamp"], RateLimitHint(remaining=0, reset_at=T0 + 30.0))

        clock.advance(29.0)
        for key in ("delta", "stamp"):
            with pytest.raises(QuotaError):
                await lim.acquire([key])
        clock.advance(2.0)
        for key in ("delta", "stamp"):
            await lim.acquire([key])

    async def test_a_nonsense_reset_cannot_freeze_a_bucket_for_decades(self, redis: Any) -> None:
        """A millisecond timestamp read as seconds is a real provider bug; the
        bucket must self-heal rather than stay dark until someone notices."""
        clock = FakeClock()
        lim = limiter(redis, clock)
        await lim.observe([KEY], RateLimitHint(remaining=0, reset_at=T0 * 1000))

        clock.advance(MAX_PARK_SECONDS + 60.0)
        await lim.acquire([KEY])

    async def test_a_reset_in_the_past_is_ignored(self, redis: Any) -> None:
        clock = FakeClock()
        lim = limiter(redis, clock)
        await lim.acquire([KEY])
        await lim.observe([KEY], RateLimitHint(reset_at=T0 - 5_000))
        assert await tokens_in(redis, KEY) == pytest.approx(2.0)

    async def test_clamps_every_key_it_is_given(self, redis: Any) -> None:
        lim = limiter(redis, FakeClock())
        await lim.acquire(["a", "b"])
        await lim.observe(["a", "b"], RateLimitHint(remaining=0))
        assert await tokens_in(redis, "a") == pytest.approx(0.0)
        assert await tokens_in(redis, "b") == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Failing open
# --------------------------------------------------------------------------- #


class TestFailOpen:
    async def test_a_dead_redis_does_not_stop_ingestion(self) -> None:
        """`docs/architecture.md` §7.3: outbound limiting degrades open."""
        seen: list[str] = []
        lim = TokenBucketLimiter(
            BrokenRedis(),
            default_policy=BucketPolicy(capacity=10, refill_per_second=1.0),
            clock=FakeClock(),
            on_degraded=lambda key, exc: seen.append(key),
        )
        await lim.acquire([KEY])
        assert lim.degraded_calls == 1
        assert seen == [KEY]

    async def test_the_fallback_still_enforces_a_conservative_limit(self) -> None:
        """ "Open" is not "unlimited": a Redis outage must not turn into a ban."""
        lim = TokenBucketLimiter(
            BrokenRedis(),
            default_policy=BucketPolicy(capacity=10, refill_per_second=1.0),
            clock=FakeClock(),
        )
        for _ in range(5):  # half of 10
            await lim.acquire([KEY])
        with pytest.raises(QuotaError) as excinfo:
            await lim.acquire([KEY])
        assert excinfo.value.details["limiter"] == "in-memory"

    async def test_fail_open_can_be_switched_off(self) -> None:
        """For the deployment that would rather stop than risk the provider."""
        lim = TokenBucketLimiter(BrokenRedis(), clock=FakeClock(), fail_open=False)
        with pytest.raises(ConnectionError):
            await lim.acquire([KEY])

    async def test_a_failed_observation_is_not_fatal(self, redis: Any) -> None:
        """Reconciliation is bookkeeping; failing a run over it would be worse
        than the drift it prevents."""
        lim = TokenBucketLimiter(BrokenRedis(), clock=FakeClock())
        await lim.observe([KEY], RateLimitHint(remaining=1))
        assert lim.degraded_calls == 1


# --------------------------------------------------------------------------- #
# Local limiters
# --------------------------------------------------------------------------- #


class TestLocalLimiters:
    def test_all_three_satisfy_the_port(self) -> None:
        """`SyncContext.limiter` is typed as the Protocol; a limiter that does not
        satisfy it fails at the connector, far from the cause."""
        assert isinstance(NullLimiter(), RateLimiter)
        assert isinstance(InMemoryLimiter(), RateLimiter)
        assert isinstance(TokenBucketLimiter(BrokenRedis()), RateLimiter)

    async def test_in_memory_enforces_capacity_and_refills(self) -> None:
        clock = FakeClock()
        lim = InMemoryLimiter(
            default_policy=BucketPolicy(capacity=2, refill_per_second=1.0), clock=clock
        )
        await lim.acquire([KEY])
        await lim.acquire([KEY])
        with pytest.raises(QuotaError):
            await lim.acquire([KEY])
        clock.advance(1.0)
        await lim.acquire([KEY])

    async def test_in_memory_is_all_or_nothing_too(self) -> None:
        clock = FakeClock()
        lim = InMemoryLimiter(
            default_policy=BucketPolicy(capacity=1, refill_per_second=1.0), clock=clock
        )
        await lim.acquire(["blocked"])
        with pytest.raises(QuotaError):
            await lim.acquire(["free", "blocked"])
        assert lim.snapshot()["free"] == pytest.approx(1.0)

    async def test_in_memory_clamps_down_only(self) -> None:
        clock = FakeClock()
        lim = InMemoryLimiter(
            default_policy=BucketPolicy(capacity=10, refill_per_second=1.0), clock=clock
        )
        await lim.acquire([KEY])
        await lim.observe([KEY], RateLimitHint(remaining=2))
        assert lim.snapshot()[KEY] == pytest.approx(2.0)
        await lim.observe([KEY], RateLimitHint(remaining=9))
        assert lim.snapshot()[KEY] == pytest.approx(2.0)

    async def test_in_memory_waits_within_a_budget(self) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        lim = InMemoryLimiter(
            default_policy=BucketPolicy(capacity=1, refill_per_second=1.0),
            clock=clock,
            sleep=sleeper,
        )
        await lim.acquire([KEY])
        await lim.acquire([KEY], timeout_seconds=10.0)
        assert sleeper.calls == [pytest.approx(1.0)]

    async def test_null_limiter_grants_everything(self) -> None:
        lim = NullLimiter()
        for _ in range(100):
            await lim.acquire([KEY])
        assert len(lim.acquired) == 100


# --------------------------------------------------------------------------- #
# Backoff: jitter
# --------------------------------------------------------------------------- #


class TestFullJitter:
    def test_draws_from_zero_to_the_ceiling(self) -> None:
        """Full jitter, not equal jitter: the whole interval is in play, so N
        workers that failed together spread uniformly instead of clustering in
        the top half."""
        rng = MaxJitter()
        full_jitter(3, rng=rng)
        assert rng.ranges == [(0.0, 8.0)]

    def test_the_ceiling_doubles_per_attempt(self) -> None:
        rng = MaxJitter()
        assert [full_jitter(i, rng=rng) for i in range(5)] == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_the_ceiling_is_capped(self) -> None:
        assert full_jitter(20, rng=MaxJitter()) == 60.0

    def test_the_floor_is_zero(self) -> None:
        assert full_jitter(4, rng=ZeroJitter()) == 0.0

    def test_a_huge_attempt_number_does_not_overflow(self) -> None:
        """An attempt counter is the kind of thing a caller passes a loop
        variable to, and `2 ** 1024` is a float overflow, not a big number."""
        assert DEFAULT_POLICY.ceiling_for(5_000) == 60.0

    def test_a_negative_attempt_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError):
            DEFAULT_POLICY.ceiling_for(-1)

    def test_tenacity_wait_offsets_the_attempt_number(self) -> None:
        """tenacity counts attempts from 1; the curve counts from 0, so the first
        retry must wait under `base` rather than under `2 * base`."""

        class State:
            attempt_number = 1

        rng = MaxJitter()
        assert tenacity_wait(rng=rng)(State()) == 1.0


# --------------------------------------------------------------------------- #
# Backoff: Retry-After
# --------------------------------------------------------------------------- #


class TestRetryAfter:
    def test_it_overrides_the_jitter_entirely(self) -> None:
        """Not an upper bound. Retrying before the provider said it would serve
        us is a request we know will be refused, and refused requests are what an
        abuse heuristic counts."""
        rng = MaxJitter()
        assert next_delay(0, retry_after=42.0, rng=rng) == 42.0
        assert rng.ranges == [], "the RNG must not even be consulted"

    def test_it_overrides_upward_past_the_per_attempt_cap(self) -> None:
        assert next_delay(0, retry_after=300.0, rng=MaxJitter()) == 300.0

    def test_above_the_threshold_it_becomes_a_quota_error(self) -> None:
        """Fifteen minutes of sleep is a worker held out of the pool to serve one
        account; `QuotaError` commits the cursor and reschedules instead."""
        with pytest.raises(QuotaError) as excinfo:
            next_delay(0, retry_after=901.0, connector="demo", account_id="acct_1")
        assert excinfo.value.retry_after_seconds == 901.0
        assert excinfo.value.reset_at is not None
        assert excinfo.value.connector == "demo"

    def test_the_threshold_itself_is_still_a_sleep(self) -> None:
        """A boundary that is off by one either holds a worker for 15 minutes or
        reschedules a run that only needed to pause."""
        assert next_delay(0, retry_after=900.0) == 900.0

    def test_zero_is_honoured_rather_than_treated_as_absent(self) -> None:
        assert next_delay(0, retry_after=0.0, rng=MaxJitter()) == 0.0

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (QuotaError("q", retry_after_seconds=12.0), 12.0),
            (TransientError("t", details={"retry_after_seconds": 7.5}), 7.5),
            (TransientError("t"), None),
            (TransientError("t", details={"retry_after_seconds": "nonsense"}), None),
        ],
    )
    def test_reads_the_header_from_either_place_it_can_be_attached(
        self, error: ConnectorError, expected: float | None
    ) -> None:
        """`QuotaError` declares the attribute; a `TransientError` from a
        429-within-cap has only `details`. Looking in one place is how a header
        gets parsed correctly, attached correctly, and then ignored."""
        assert retry_after_of(error) == expected


# --------------------------------------------------------------------------- #
# Backoff: attempts and budget
# --------------------------------------------------------------------------- #


class TestRetryController:
    def controller(self, clock: FakeClock, sleeper: FakeSleep, **kwargs: Any) -> RetryController:
        return RetryController(clock=clock, sleep=sleeper, rng=MaxJitter(), **kwargs)

    async def test_sleeps_four_times_then_re_raises_the_original_error(self) -> None:
        """Five attempts per page (§5.2), and the error the run records is the
        provider's, not "retries exhausted" -- which is never the interesting half."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        controller = self.controller(clock, sleeper)
        error = TransientError("upstream 503")

        for _ in range(4):
            await controller.sleep_before_retry(error)
        with pytest.raises(TransientError) as excinfo:
            await controller.sleep_before_retry(error)

        assert excinfo.value is error
        assert sleeper.calls == [1.0, 2.0, 4.0, 8.0]

    async def test_the_budget_ends_a_page_before_the_attempts_do(self) -> None:
        """Two 200-second `Retry-After`s exhaust five minutes in two attempts."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        controller = self.controller(clock, sleeper)
        error = TransientError("429", details={"retry_after_seconds": 200.0})

        await controller.sleep_before_retry(error)
        with pytest.raises(TransientError):
            await controller.sleep_before_retry(error)
        assert sleeper.calls == [200.0], "must not sleep past the budget and then notice"

    async def test_the_budget_is_measured_from_construction_not_from_the_sleeps(
        self,
    ) -> None:
        """Time spent *in* the request counts: a page that spent four minutes
        timing out has one minute of retrying left, not five."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        controller = self.controller(clock, sleeper)
        clock.advance(299.5)  # 0.5s of budget left, and the next wait wants 1.0
        with pytest.raises(TransientError):
            await controller.sleep_before_retry(TransientError("t"))
        assert sleeper.calls == []

    async def test_a_long_retry_after_converts_on_the_last_attempt_too(self) -> None:
        """The conversion is about the provider's instruction, not about how many
        tries are left. Letting the attempt counter win here would file "come
        back in an hour" as a failed run instead of a rescheduled one, and throw
        away every page already emitted."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        controller = self.controller(clock, sleeper)
        for _ in range(4):
            await controller.sleep_before_retry(TransientError("t"))

        with pytest.raises(QuotaError):
            await controller.sleep_before_retry(
                TransientError("429", details={"retry_after_seconds": 3_600.0})
            )

    async def test_a_long_retry_after_converts_mid_retry(self) -> None:
        clock = FakeClock()
        controller = self.controller(clock, FakeSleep(clock))
        with pytest.raises(QuotaError):
            await controller.sleep_before_retry(
                TransientError("429", details={"retry_after_seconds": 1_800.0})
            )

    async def test_a_custom_policy_is_honoured(self) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        controller = self.controller(
            clock, sleeper, policy=BackoffPolicy(max_attempts=2, base_seconds=0.5)
        )
        await controller.sleep_before_retry(TransientError("t"))
        with pytest.raises(TransientError):
            await controller.sleep_before_retry(TransientError("t"))
        assert sleeper.calls == [0.5]


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


class TestCircuitBreaker:
    def breaker(self, clock: FakeClock) -> CircuitBreaker:
        return CircuitBreaker(clock=clock)

    def test_four_failures_leave_it_closed(self) -> None:
        breaker = self.breaker(FakeClock())
        for _ in range(4):
            breaker.record_failure("demo:acct_1")
        assert breaker.state("demo:acct_1") is CircuitState.CLOSED
        breaker.check("demo:acct_1")

    def test_the_fifth_failure_opens_it(self) -> None:
        clock = FakeClock()
        breaker = self.breaker(clock)
        for _ in range(5):
            breaker.record_failure("demo:acct_1")

        assert breaker.state("demo:acct_1") is CircuitState.OPEN
        with pytest.raises(CircuitOpenError) as excinfo:
            breaker.check("demo:acct_1")
        assert excinfo.value.opens_until == T0 + 600.0
        assert excinfo.value.error_class.value == "quota"

    def test_a_success_resets_the_count(self) -> None:
        """Consecutive means consecutive: four failures spread across a working
        integration must never accumulate into an outage."""
        breaker = self.breaker(FakeClock())
        for _ in range(4):
            breaker.record_failure("k")
        breaker.record_success("k")
        for _ in range(4):
            breaker.record_failure("k")
        assert breaker.state("k") is CircuitState.CLOSED

    def test_it_half_opens_after_the_timer_and_closes_on_a_good_probe(self) -> None:
        """Without the probe the breaker would need a separate healthcheck to
        ever recover, and the integration would stay dark until someone noticed."""
        clock = FakeClock()
        breaker = self.breaker(clock)
        for _ in range(5):
            breaker.record_failure("k")

        clock.advance(600.0)
        assert breaker.state("k") is CircuitState.HALF_OPEN
        breaker.check("k")  # the probe is allowed through
        breaker.record_success("k")
        assert breaker.state("k") is CircuitState.CLOSED

    def test_a_failed_probe_re_opens_immediately(self) -> None:
        """The probe is the one call allowed through to test the water. If its
        failure did not re-open on its own, four more would follow it into a
        source we just watched fail."""
        clock = FakeClock()
        breaker = self.breaker(clock)
        for _ in range(5):
            breaker.record_failure("k")
        clock.advance(600.0)
        breaker.check("k")  # half-open, and the old count is cleared

        breaker.record_failure("k")
        assert breaker.state("k") is CircuitState.OPEN
        with pytest.raises(CircuitOpenError) as excinfo:
            breaker.check("k")
        assert excinfo.value.opens_until == T0 + 1_200.0
        assert "probe" in excinfo.value.message

    def test_a_probe_is_judged_on_its_own_result(self) -> None:
        """The count clears when the circuit half-opens, so what reopens it is
        the failed probe and not five-plus-one arithmetic against an outage that
        has already been counted -- and the number an operator reads describes
        one outage rather than every outage added together."""
        clock = FakeClock()
        breaker = self.breaker(clock)
        for _ in range(5):
            breaker.record_failure("k")
        assert breaker.consecutive_failures("k") == 5

        clock.advance(600.0)
        breaker.check("k")
        assert breaker.consecutive_failures("k") == 0

        breaker.record_failure("k")
        assert breaker.consecutive_failures("k") == 1
        assert breaker.state("k") is CircuitState.OPEN

    def test_the_failure_count_is_visible_before_it_trips(self) -> None:
        """An account approaching the threshold should be visible on /health,
        not only once the integration has gone dark."""
        breaker = self.breaker(FakeClock())
        assert breaker.consecutive_failures("k") == 0
        for expected in (1, 2, 3, 4):
            breaker.record_failure("k")
            assert breaker.consecutive_failures("k") == expected
            assert breaker.state("k") is CircuitState.CLOSED

    def test_accounts_are_independent(self) -> None:
        """Per account, not per connector: one revoked token must not stop every
        tenant on that source."""
        breaker = self.breaker(FakeClock())
        for _ in range(5):
            breaker.record_failure("demo:acct_1")
        breaker.check("demo:acct_2")

    def test_a_threshold_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            CircuitBreaker(failure_threshold=0)


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #


class TestRetryPage:
    def kwargs(self, clock: FakeClock, sleeper: FakeSleep, **extra: Any) -> dict[str, Any]:
        return {"clock": clock, "sleep": sleeper, "rng": MaxJitter(), **extra}

    async def test_retries_a_transient_failure_and_returns(self) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        calls = 0

        async def flaky() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TransientError("connection reset")
            return "page"

        assert await retry_page(flaky, **self.kwargs(clock, sleeper)) == "page"
        assert calls == 3
        assert sleeper.calls == [1.0, 2.0]

    async def test_auth_and_permanent_failures_are_not_retried(self) -> None:
        """The taxonomy exists so the runtime decides from the class it caught."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)

        for error in (AuthError("401"), PermanentError("404")):
            calls = 0

            async def failing(exc: ConnectorError = error) -> str:
                nonlocal calls
                calls += 1
                raise exc

            with pytest.raises(ConnectorError):
                await retry_page(failing, **self.kwargs(clock, sleeper))
            assert calls == 1
        assert sleeper.calls == []

    async def test_one_page_costs_the_breaker_one_failure_not_five(self) -> None:
        """Counting attempts would open the circuit on the first genuinely flaky
        page, inverting the design: retries absorb blips, the breaker exists for
        the case where retrying is pointless."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        breaker = CircuitBreaker(clock=clock)

        async def always_fails() -> str:
            raise TransientError("503")

        with pytest.raises(TransientError):
            await retry_page(
                always_fails, breaker=breaker, breaker_key="k", **self.kwargs(clock, sleeper)
            )
        assert breaker.state("k") is CircuitState.CLOSED

    async def test_five_failed_pages_open_the_circuit(self) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        breaker = CircuitBreaker(clock=clock)

        async def always_fails() -> str:
            raise TransientError("503")

        for _ in range(5):
            with pytest.raises(TransientError):
                await retry_page(
                    always_fails, breaker=breaker, breaker_key="k", **self.kwargs(clock, sleeper)
                )

        with pytest.raises(CircuitOpenError):
            await retry_page(
                always_fails, breaker=breaker, breaker_key="k", **self.kwargs(clock, sleeper)
            )

    async def test_an_open_circuit_is_checked_before_the_call_is_made(self) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        breaker = CircuitBreaker(clock=clock)
        for _ in range(5):
            breaker.record_failure("k")
        called = False

        async def operation() -> str:
            nonlocal called
            called = True
            return "page"

        with pytest.raises(CircuitOpenError):
            await retry_page(
                operation, breaker=breaker, breaker_key="k", **self.kwargs(clock, sleeper)
            )
        assert not called, "the point of the breaker is that the call is not made"

    async def test_a_success_closes_a_half_open_circuit(self) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        breaker = CircuitBreaker(clock=clock)
        for _ in range(5):
            breaker.record_failure("k")
        clock.advance(600.0)

        async def operation() -> str:
            return "page"

        assert (
            await retry_page(
                operation, breaker=breaker, breaker_key="k", **self.kwargs(clock, sleeper)
            )
            == "page"
        )
        assert breaker.state("k") is CircuitState.CLOSED

    async def test_our_own_bugs_do_not_open_the_circuit(self) -> None:
        """A `KeyError` in a mapper is our defect, not the provider being
        unhealthy. Tripping the breaker on it would take the integration down for
        ten minutes over a typo and hide the traceback behind a circuit message."""
        clock = FakeClock()
        sleeper = FakeSleep(clock)
        breaker = CircuitBreaker(failure_threshold=1, clock=clock)

        async def buggy() -> str:
            raise KeyError("title")

        with pytest.raises(KeyError):
            await retry_page(buggy, breaker=breaker, breaker_key="k", **self.kwargs(clock, sleeper))
        assert breaker.state("k") is CircuitState.CLOSED

    async def test_a_long_retry_after_ends_the_page_as_a_quota(self) -> None:
        clock = FakeClock()
        sleeper = FakeSleep(clock)

        async def throttled() -> str:
            raise TransientError("429", details={"retry_after_seconds": 3_600.0})

        with pytest.raises(QuotaError):
            await retry_page(throttled, **self.kwargs(clock, sleeper))
        assert sleeper.calls == []
