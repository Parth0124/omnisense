"""Cross-process token bucket: the only thing standing between us and a ban.

`docs/connector-spec.md` §5.1 puts four buckets in front of every outbound call
-- connector-wide, per-account, per-host and a reduced backfill bucket -- and
requires that check-and-decrement be atomic *across processes*. Not across tasks,
across processes: eight worker replicas, the API and the scheduler all draw on
one provider quota, so the arbiter has to live where they can all see it, which
means Redis, which means a Lua script (§5.1 again).

Three decisions are encoded here and each has a failure mode attached.

**Lua, not GET-then-SET.** See `ACQUIRE_LUA` for the interleaving that lets two
workers spend the same last token.

**All-or-nothing acquisition.** `acquire()` takes from every key or from none. A
partial acquisition that raised without giving the taken tokens back would leak
one token per contended call, and the effective limit of the *uncontended*
buckets would drift downwards for the life of the process -- a bug that presents
months later as "ingestion got slow", with nothing in the logs.

**Provider truth beats local estimate.** `observe()` only ever clamps a bucket
*down*. Our count is an estimate assembled from our own requests; theirs is
authoritative and includes the requests we never saw (another deployment, a
retry that timed out on our side but landed on theirs). Letting a header raise
the count would let one stale response refund tokens we already spent.

The Redis client arrives as a constructor argument typed by a local `Protocol`.
`connectors/` may not import `backend/db/redis.py` (`docs/architecture.md` §6.2
rule 2), and naming `redis.asyncio.Redis` directly would make the whole package
unimportable without the driver installed. Structural typing costs nothing and
lets the tests hand it a fake.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from connectors.exceptions import QuotaError
from connectors.protocol import RateLimiter, RateLimitHint, RateLimitPolicy, SyncMode

__all__ = [
    "ACQUIRE_LUA",
    "OBSERVE_LUA",
    "RELEASE_LUA",
    "BucketPolicy",
    "InMemoryLimiter",
    "NullLimiter",
    "RedisScriptClient",
    "TokenBucketLimiter",
]


MIN_WAIT_SECONDS = 0.005
"""Floor on a computed wait.

A bucket can report a retry-after of zero when the deficit rounds away, and a
zero-length sleep in the acquire loop is a hot spin against Redis rather than a
wait.
"""

MAX_PARK_SECONDS = 3600.0
"""Ceiling on how far into the future a provider header may freeze a bucket.

`X-RateLimit-Reset` is attacker-adjacent input in the sense that matters here:
it is a number we did not compute, and a mis-set one (a millisecond timestamp
read as seconds, a clock-skewed server) would otherwise wedge a bucket for
decades. An hour is longer than any real reset window and short enough to
self-heal.
"""

_EPOCH_THRESHOLD_SECONDS = 1_000_000_000.0
"""Above this, a reset value is a UNIX timestamp; below it, delta-seconds.

Providers use both spellings for `X-RateLimit-Reset` and neither labels itself.
Guessing wrong in one direction parks a bucket until 2033; in the other it does
nothing at all. The split is unambiguous in practice: no real reset window is
thirty-one years long, and no real timestamp predates 2001.
"""

FAIL_OPEN_FRACTION = 0.5
"""How much of the nominal rate the local fallback allows when Redis is down.

`docs/architecture.md` §7.3: outbound limiting fails **open with conservative
static limits**, because halting ingestion every time the cache blips is worse
than pacing imperfectly. "Open" is not "unlimited" -- each replica keeps a local
bucket at half rate, so N replicas overshoot by at most N/2x rather than
without bound.
"""


# --------------------------------------------------------------------------- #
# Ports and policy
# --------------------------------------------------------------------------- #


@runtime_checkable
class RedisScriptClient(Protocol):
    """The four Redis calls this module makes, as a structural type.

    Satisfied by `redis.asyncio.Redis` as-is. Declared here rather than imported
    so `connectors/` keeps its promise of importing only `models/` and itself,
    and so the unit suite can pass `fakeredis` with no shim.
    """

    async def script_load(self, script: str) -> Any: ...

    async def evalsha(self, sha: str, numkeys: int, *keys_and_args: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class BucketPolicy:
    """Capacity and refill rate for one bucket key.

    `capacity` is the burst -- how many calls may be made back-to-back after an
    idle period -- and `refill_per_second` is the sustained rate. Separating them
    is what lets a connector page hard for a few seconds without exceeding a
    per-minute budget over the hour.
    """

    capacity: int
    refill_per_second: float

    def __post_init__(self) -> None:
        # Both are fatal at construction rather than at the first acquire: a zero
        # refill rate is a bucket that never recovers and an infinite TTL, and a
        # zero capacity denies every call forever. Either would present as "the
        # connector silently stopped fetching", which is the hardest class of
        # bug to trace back to a config value.
        if self.capacity < 1:
            raise ValueError(f"bucket capacity must be >= 1, got {self.capacity}")
        if self.refill_per_second <= 0:
            raise ValueError(f"bucket refill_per_second must be > 0, got {self.refill_per_second}")

    @property
    def ttl_seconds(self) -> float:
        """`2 * capacity / refill_rate` (`docs/connector-spec.md` §5.1).

        Two full refills' worth. Long enough that a bucket in active use is never
        evicted mid-flight, short enough that an idle bucket disappears -- which
        is the point: an expired bucket cold-starts full, so a key we have not
        touched in ten minutes costs nothing to keep and nothing to recreate.
        Storing every per-host bucket forever would otherwise turn an RSS fleet
        of a thousand feeds into a thousand permanent keys.
        """
        return 2.0 * self.capacity / self.refill_per_second

    def scaled(self, fraction: float) -> BucketPolicy:
        """The same bucket at a fraction of its rate. Capacity floors at 1."""
        return BucketPolicy(
            capacity=max(1, int(self.capacity * fraction)),
            refill_per_second=max(self.refill_per_second * fraction, 1e-6),
        )

    @classmethod
    def from_rate_limit_policy(
        cls, policy: RateLimitPolicy, mode: SyncMode = SyncMode.INCREMENTAL
    ) -> BucketPolicy:
        """Derive a bucket from a connector's declared `RateLimitPolicy`.

        `for_mode` is what applies the 25% backfill fraction, so a backfill bucket
        built through here is automatically the reduced one §5.1 asks for.
        """
        return cls(
            capacity=max(1, policy.burst),
            refill_per_second=max(policy.for_mode(mode) / 60.0, 1e-6),
        )


DEFAULT_BUCKET_POLICY = BucketPolicy(capacity=10, refill_per_second=1.0)
"""60 requests/minute with a burst of 10 -- `CONNECTOR_DEFAULT_RATE_LIMIT_PER_MINUTE`.

The per-host politeness default from `docs/connector-spec.md` §5.1. Deliberately
conservative: it is the value that applies to a feed host nobody configured.
"""


@dataclass(frozen=True, slots=True)
class _Grant:
    """What one evaluation of `ACQUIRE_LUA` decided."""

    allowed: bool
    tokens: float
    retry_after_seconds: float


class _PolicyResolver:
    """Shared key-to-policy lookup for the two real limiters.

    Exact-match overrides plus a default, and `policy_for` is a method so a
    deployment that wants prefix matching (`os:rl:host:*`) subclasses rather than
    forks. Exact-match is the default because a wrong prefix rule silently
    applies the wrong quota to a bucket, and that is invisible until a provider
    complains.
    """

    def __init__(
        self,
        *,
        default_policy: BucketPolicy = DEFAULT_BUCKET_POLICY,
        policies: Mapping[str, BucketPolicy] | None = None,
    ) -> None:
        self._default_policy = default_policy
        self._policies: dict[str, BucketPolicy] = dict(policies or {})

    @property
    def default_policy(self) -> BucketPolicy:
        return self._default_policy

    def policy_for(self, key: str) -> BucketPolicy:
        return self._policies.get(key, self._default_policy)

    def set_policy(self, key: str, policy: BucketPolicy) -> None:
        """Register a per-key override. Used when an account declares its own budget."""
        self._policies[key] = policy


# --------------------------------------------------------------------------- #
# The scripts
# --------------------------------------------------------------------------- #


ACQUIRE_LUA = """
-- KEYS[1] bucket hash. ARGV: capacity, refill_per_second, now_ms, cost, ttl_ms
-- Returns {allowed, tokens_after_as_string, retry_after_ms}
local key         = KEYS[1]
local capacity    = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now_ms      = tonumber(ARGV[3])
local cost        = tonumber(ARGV[4])
local ttl_ms      = tonumber(ARGV[5])

local state   = redis.call('HMGET', key, 'tokens', 'last_refill_ms')
local tokens  = tonumber(state[1])
local last_ms = tonumber(state[2])

if tokens == nil or last_ms == nil then
  -- No bucket, or a half-written one. Cold-start full: the TTL is two refills
  -- long, so an absent key means nobody has spent from it recently.
  tokens  = capacity
  last_ms = now_ms
end

-- now_ms comes from the caller's wall clock, and callers are different hosts.
-- A last_refill_ms written by a host running a second fast would otherwise make
-- elapsed negative here and *remove* tokens from a bucket nobody spent from.
local elapsed_ms = now_ms - last_ms
if elapsed_ms < 0 then elapsed_ms = 0 end

tokens = math.min(capacity, tokens + (elapsed_ms / 1000.0) * refill_rate)

local allowed = 0
local retry_after_ms = 0
if tokens >= cost then
  allowed = 1
  tokens = tokens - cost
else
  retry_after_ms = math.ceil(((cost - tokens) / refill_rate) * 1000.0)
end

redis.call('HSET', key, 'tokens', tokens, 'last_refill_ms', now_ms)
redis.call('PEXPIRE', key, ttl_ms)

-- tokens is fractional and Redis truncates Lua numbers to integers on the way
-- out, so it goes back as a string. Returning it as a number would round 0.9
-- down to 0 and make every partial bucket look empty to the caller.
return {allowed, tostring(tokens), retry_after_ms}
"""
"""Atomic check-and-decrement.

Why this cannot be `HGETALL` in Python followed by `HSET`: those are two round
trips, and between them the other seven workers are running. With one token left
in the bucket,

    worker A: HMGET -> tokens = 1        worker B: HMGET -> tokens = 1
    worker A: 1 >= 1, allow              worker B: 1 >= 1, allow
    worker A: HSET tokens = 0            worker B: HSET tokens = 0

Both issue a request; the bucket records one spend. The overshoot is not a
rounding error -- under sustained contention it is proportional to the number of
workers, which is exactly the situation the limiter exists for. `WATCH`/`MULTI`
would also be correct but costs three round trips and livelocks under the same
contention; Redis runs a script to completion with nothing interleaved, in one.
"""


RELEASE_LUA = """
-- KEYS[1] bucket hash. ARGV: capacity, amount, ttl_ms
-- Returns the token count after the credit, or '-1' when the bucket is gone.
local key      = KEYS[1]
local capacity = tonumber(ARGV[1])
local amount   = tonumber(ARGV[2])
local ttl_ms   = tonumber(ARGV[3])

local tokens = tonumber(redis.call('HGET', key, 'tokens'))
if tokens == nil then
  -- The bucket expired between the acquire and this release. An absent bucket
  -- cold-starts full, so there is nothing to give back; writing one here would
  -- *lower* it from capacity to amount.
  return '-1'
end

tokens = math.min(capacity, tokens + amount)
redis.call('HSET', key, 'tokens', tokens)
redis.call('PEXPIRE', key, ttl_ms)
return tostring(tokens)
"""
"""Return an unused token.

Deliberately does not touch `last_refill_ms`. That field anchors the refill
accounting to the last acquire; moving it to now would silently discard the time
elapsed since, which is a second, subtler way to leak capacity -- one that
compounds every time a multi-key acquire fails.
"""


OBSERVE_LUA = """
-- KEYS[1] bucket hash.
-- ARGV: capacity, refill_per_second, now_ms, observed (-1 unknown),
--       not_before_ms (0 none), ttl_ms
-- Returns the token count after reconciliation, as a string.
local key           = KEYS[1]
local capacity      = tonumber(ARGV[1])
local refill_rate   = tonumber(ARGV[2])
local now_ms        = tonumber(ARGV[3])
local observed      = tonumber(ARGV[4])
local not_before_ms = tonumber(ARGV[5])
local ttl_ms        = tonumber(ARGV[6])

local state   = redis.call('HMGET', key, 'tokens', 'last_refill_ms')
local tokens  = tonumber(state[1])
local last_ms = tonumber(state[2])
if tokens == nil or last_ms == nil then
  tokens  = capacity
  last_ms = now_ms
end

local elapsed_ms = now_ms - last_ms
if elapsed_ms < 0 then elapsed_ms = 0 end

-- Compare against what the bucket holds *now*, not what was last written. A
-- stored 2 that has been refilling for ten seconds is really 12, and comparing
-- the provider's 5 against the stale 2 would conclude there is nothing to
-- tighten and leave 12 tokens standing.
local current = math.min(capacity, tokens + (elapsed_ms / 1000.0) * refill_rate)

local target = current
if observed >= 0 then
  local clamped = math.min(observed, capacity)
  if clamped < current then target = clamped end
end

local park = (not_before_ms > now_ms) and (target < 1)
if target >= current and not park then
  -- Nothing to tighten. Returning without a write is what makes this
  -- downward-only: there is no branch here that can raise a token count.
  return tostring(current)
end

local new_last = now_ms
if park then
  -- The provider told us when it resets. Anchor the refill clock there so the
  -- bucket does not hand out a token one second from now and earn a second 429;
  -- the negative-elapsed guard in every script keeps a future anchor safe.
  new_last = not_before_ms
end

redis.call('HSET', key, 'tokens', target, 'last_refill_ms', new_last)
redis.call('PEXPIRE', key, ttl_ms)
return tostring(target)
"""
"""Reconcile a bucket against the provider's own count. Downward only."""


# --------------------------------------------------------------------------- #
# The Redis limiter
# --------------------------------------------------------------------------- #


class TokenBucketLimiter(_PolicyResolver):
    """Shared token bucket over Redis. Satisfies `protocol.RateLimiter`.

    One instance per process is enough and is the intended deployment: the state
    is entirely in Redis, so nothing about an instance is worth sharing except
    the cached script SHAs.
    """

    def __init__(
        self,
        redis: RedisScriptClient,
        *,
        default_policy: BucketPolicy = DEFAULT_BUCKET_POLICY,
        policies: Mapping[str, BucketPolicy] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Any] = asyncio.sleep,
        rng: random.Random | None = None,
        fail_open: bool = True,
        fallback: RateLimiter | None = None,
        on_degraded: Callable[[str, Exception], None] | None = None,
    ) -> None:
        super().__init__(default_policy=default_policy, policies=policies)
        self._redis = redis
        # One clock for both the Lua timestamps and the acquire deadline, and it
        # must be a *wall* clock: `last_refill_ms` is compared across hosts, and
        # one host's `monotonic()` is meaningless to another. The cost is that a
        # clock step during an acquire perturbs one wait, which the negative-
        # elapsed guard in the scripts absorbs.
        self._clock = clock
        self._sleep = sleep
        self._rng = rng if rng is not None else random.Random()
        self._fail_open = fail_open
        self._shas: dict[str, str] = {}
        self._on_degraded = on_degraded
        self.degraded_calls = 0
        """Times a Redis failure sent an acquire down the fail-open path.

        Exposed as a counter rather than a log line because the interesting
        signal is the rate, and because `docs/connector-spec.md` §1 forbids this
        layer from logging anything carrying request context.
        """

        if fallback is not None:
            self._fallback: RateLimiter = fallback
        elif fail_open:
            # §7.3's "conservative static limits". Per-replica, so it cannot be
            # exact -- it only has to keep a Redis outage from turning into a
            # provider ban.
            self._fallback = InMemoryLimiter(
                default_policy=default_policy.scaled(FAIL_OPEN_FRACTION),
                policies={k: p.scaled(FAIL_OPEN_FRACTION) for k, p in (policies or {}).items()},
                clock=clock,
                sleep=sleep,
            )
        else:
            self._fallback = NullLimiter()

    # ------------------------------------------------------------- acquire --

    async def acquire(self, keys: Sequence[str], *, timeout_seconds: float | None = None) -> None:
        """Take one token from every key, or from none of them.

        `timeout_seconds` is a budget for *waiting*, not a socket timeout. `None`
        means do not wait at all: fail fast with `QuotaError`. That is the safe
        default because a `QuotaError` is a partial success -- the cursor is
        committed and the run is rescheduled (`connectors/exceptions.py`) --
        whereas an unbounded wait pins a worker on a bucket that may not refill
        for an hour. `BaseConnector.acquire_slot` passes the request timeout, so
        the normal path does wait, briefly.
        """
        ordered = _ordered_unique(keys)
        if not ordered:
            return

        deadline = None if timeout_seconds is None else self._clock() + timeout_seconds

        while True:
            taken: list[str] = []
            blocked: tuple[str, float] | None = None

            for key in ordered:
                policy = self.policy_for(key)
                try:
                    grant = await self._acquire_one(key, policy)
                except Exception as exc:  # Redis is unreachable; see below
                    if not self._fail_open:
                        await self._release(taken)
                        raise
                    # Deliberately not releasing: the release would use the same
                    # dead connection, and the bucket's TTL reclaims the tokens
                    # anyway. Losing a token is cheaper than a second failure.
                    self._degrade(key, exc)
                    await self._fallback.acquire(ordered, timeout_seconds=timeout_seconds)
                    return

                if grant.allowed:
                    taken.append(key)
                    continue
                blocked = (key, grant.retry_after_seconds)
                break

            if blocked is None:
                return

            # All-or-nothing. Without this the two buckets that did grant would
            # be one token poorer for nothing, every time the third is contended.
            await self._release(taken)

            key, retry_after = blocked
            wait = self._wait_for(retry_after)
            remaining = None if deadline is None else deadline - self._clock()
            if remaining is None or remaining <= 0 or wait > remaining:
                raise QuotaError(
                    f"rate limit bucket {key!r} exhausted",
                    reset_at=self._clock() + retry_after,
                    retry_after_seconds=retry_after,
                    details={"bucket": key},
                )
            await self._sleep(wait)

    async def observe(self, keys: Sequence[str], hint: RateLimitHint) -> None:
        """Clamp every bucket down to what the provider reports. Never up.

        A hint carrying only `Retry-After` is read as "zero remaining until
        then". That is not an inference too far: a provider does not send
        `Retry-After` on a request it was happy to serve, and treating it as
        no-information would leave the bucket handing out tokens straight into a
        429.
        """
        remaining = hint.remaining
        if remaining is None and hint.retry_after_seconds:
            remaining = 0

        now = self._clock()
        not_before = self._not_before(hint, now)
        if remaining is None and not_before is None:
            return

        for key in _ordered_unique(keys):
            policy = self.policy_for(key)
            ttl_seconds = policy.ttl_seconds
            if not_before is not None:
                # The park is only real if the bucket survives to see it; with
                # the plain TTL a long reset window would expire the key and
                # cold-start it full, undoing the clamp.
                ttl_seconds += max(0.0, not_before - now)
            try:
                await self._run_script(
                    OBSERVE_LUA,
                    key,
                    policy.capacity,
                    policy.refill_per_second,
                    int(now * 1000),
                    -1 if remaining is None else max(0, remaining),
                    0 if not_before is None else int(not_before * 1000),
                    int(ttl_seconds * 1000),
                )
            except Exception as exc:  # reconciliation is advisory
                if not self._fail_open:
                    raise
                # An unreconciled bucket is merely optimistic; failing the run
                # over a bookkeeping write would be worse than the drift.
                self._degrade(key, exc)
                return

    # ----------------------------------------------------------- internals --

    async def _acquire_one(self, key: str, policy: BucketPolicy) -> _Grant:
        raw = await self._run_script(
            ACQUIRE_LUA,
            key,
            policy.capacity,
            policy.refill_per_second,
            int(self._clock() * 1000),
            1,
            int(policy.ttl_seconds * 1000),
        )
        allowed, tokens, retry_after_ms = raw
        return _Grant(
            allowed=bool(int(allowed)),
            tokens=_as_float(tokens),
            retry_after_seconds=_as_float(retry_after_ms) / 1000.0,
        )

    async def _release(self, keys: Sequence[str]) -> None:
        """Give back tokens taken by a partial acquisition. Best effort.

        A failure here leaks one token from one bucket, which self-heals when the
        bucket expires. Raising instead would replace a recoverable `QuotaError`
        with an opaque Redis error and lose the reason the acquire failed.
        """
        for key in keys:
            policy = self.policy_for(key)
            try:
                await self._run_script(
                    RELEASE_LUA, key, policy.capacity, 1, int(policy.ttl_seconds * 1000)
                )
            except Exception as exc:  # see the docstring above
                self._degrade(key, exc)

    async def _run_script(self, script: str, key: str, *args: Any) -> Any:
        sha = self._shas.get(script)
        if sha is None:
            sha = _as_text(await self._redis.script_load(script))
            self._shas[script] = sha
        try:
            return await self._redis.evalsha(sha, 1, key, *args)
        except Exception as exc:  # narrowed immediately below
            if not _is_no_script(exc):
                raise
        # Redis restarted, or an operator ran SCRIPT FLUSH. Reload and retry
        # once; a second NOSCRIPT would mean something is wrong with the script
        # itself and deserves to propagate rather than loop.
        sha = _as_text(await self._redis.script_load(script))
        self._shas[script] = sha
        return await self._redis.evalsha(sha, 1, key, *args)

    def _wait_for(self, retry_after_seconds: float) -> float:
        """How long to wait before re-attempting a contended bucket.

        Jittered upward by up to 10%. Every worker blocked on one bucket is told
        the same refill instant, so an unjittered wait wakes them all into the
        same millisecond to fight over one token -- the same thundering herd
        `backoff.py` avoids, one layer down.
        """
        base = max(retry_after_seconds, MIN_WAIT_SECONDS)
        return base + self._rng.random() * base * 0.1

    def _not_before(self, hint: RateLimitHint, now: float) -> float | None:
        """The instant the provider says our budget returns, as UNIX seconds."""
        if hint.retry_after_seconds:
            return now + min(hint.retry_after_seconds, MAX_PARK_SECONDS)
        reset_at = hint.reset_at
        if not reset_at or reset_at <= 0:
            return None
        instant = reset_at if reset_at >= _EPOCH_THRESHOLD_SECONDS else now + reset_at
        if instant <= now:
            return None
        return min(instant, now + MAX_PARK_SECONDS)

    def _degrade(self, key: str, exc: Exception) -> None:
        self.degraded_calls += 1
        if self._on_degraded is not None:
            self._on_degraded(key, exc)


# --------------------------------------------------------------------------- #
# Local limiters
# --------------------------------------------------------------------------- #


class InMemoryLimiter(_PolicyResolver):
    """A real token bucket held in one process. Two legitimate uses.

    1. Tests, where a Redis is an unreasonable thing to require of a unit suite.
    2. The fail-open path of `TokenBucketLimiter` -- `docs/architecture.md` §7.3
       wants "conservative static limits" when Redis is unavailable, not an open
       door.

    It is **not** a substitute for the Redis limiter in a multi-replica
    deployment: N replicas each enforcing the full rate permit N times the rate,
    which is precisely the ban this module exists to prevent. Hence the halving
    in `FAIL_OPEN_FRACTION` when it is used as a fallback.

    The lock makes check-and-decrement atomic against other *tasks*; the whole
    point of the Lua script is that no in-process lock can do that against other
    *processes*.
    """

    def __init__(
        self,
        *,
        default_policy: BucketPolicy = DEFAULT_BUCKET_POLICY,
        policies: Mapping[str, BucketPolicy] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        super().__init__(default_policy=default_policy, policies=policies)
        self._clock = clock
        self._sleep = sleep
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, keys: Sequence[str], *, timeout_seconds: float | None = None) -> None:
        ordered = _ordered_unique(keys)
        if not ordered:
            return
        deadline = None if timeout_seconds is None else self._clock() + timeout_seconds

        while True:
            async with self._lock:
                taken: list[str] = []
                blocked: tuple[str, float] | None = None
                for key in ordered:
                    policy = self.policy_for(key)
                    tokens = self._refilled(key, policy)
                    if tokens >= 1.0:
                        self._buckets[key] = (tokens - 1.0, self._clock())
                        taken.append(key)
                        continue
                    blocked = (key, (1.0 - tokens) / policy.refill_per_second)
                    break

                if blocked is None:
                    return

                for key in taken:  # all-or-nothing, same rule as the Redis path
                    policy = self.policy_for(key)
                    tokens, last = self._buckets[key]
                    self._buckets[key] = (min(policy.capacity, tokens + 1.0), last)

            key, retry_after = blocked
            remaining = None if deadline is None else deadline - self._clock()
            wait = max(retry_after, MIN_WAIT_SECONDS)
            if remaining is None or remaining <= 0 or wait > remaining:
                raise QuotaError(
                    f"rate limit bucket {key!r} exhausted",
                    reset_at=self._clock() + retry_after,
                    retry_after_seconds=retry_after,
                    details={"bucket": key, "limiter": "in-memory"},
                )
            await self._sleep(wait)

    async def observe(self, keys: Sequence[str], hint: RateLimitHint) -> None:
        """Downward-only clamp, matching the Redis implementation."""
        remaining = hint.remaining
        if remaining is None and hint.retry_after_seconds:
            remaining = 0
        if remaining is None:
            return
        async with self._lock:
            for key in _ordered_unique(keys):
                policy = self.policy_for(key)
                current = self._refilled(key, policy)
                target = min(float(max(0, remaining)), float(policy.capacity))
                if target < current:
                    self._buckets[key] = (target, self._clock())

    def snapshot(self) -> dict[str, float]:
        """Current token counts, for assertions and for `GET /health`."""
        return {key: self._refilled(key, self.policy_for(key)) for key in tuple(self._buckets)}

    def _refilled(self, key: str, policy: BucketPolicy) -> float:
        """Tokens available now. Absent buckets cold-start full, as in Lua."""
        now = self._clock()
        tokens, last = self._buckets.get(key, (float(policy.capacity), now))
        elapsed = max(0.0, now - last)
        return min(float(policy.capacity), tokens + elapsed * policy.refill_per_second)


class NullLimiter:
    """Grants everything, records nothing. Satisfies `protocol.RateLimiter`.

    For tests that are not about rate limiting, and for the deliberate
    `fail_open=False` wiring where a Redis failure must surface rather than be
    absorbed locally. Not a production limiter: a connector running behind this
    against a real provider is one traffic spike away from an application-level
    ban, which is the failure this whole module is here to prevent.
    """

    def __init__(self) -> None:
        self.acquired: list[list[str]] = []
        self.observed: list[RateLimitHint] = []

    async def acquire(self, keys: Sequence[str], *, timeout_seconds: float | None = None) -> None:
        self.acquired.append(list(keys))

    async def observe(self, keys: Sequence[str], hint: RateLimitHint) -> None:
        self.observed.append(hint)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ordered_unique(keys: Iterable[str]) -> list[str]:
    """Drop blanks and repeats, keep order.

    A repeated key would be charged twice for one call, which is not what a
    caller assembling `[connector, account, host]` means when two of them
    coincide.
    """
    return list(dict.fromkeys(key for key in keys if key))


def _as_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes | bytearray) else str(value)


def _as_float(value: Any) -> float:
    """Read a Lua return value that may arrive as bytes, str, int or float.

    Whether Redis replies in bytes depends on `decode_responses` on a client this
    module does not construct, so both spellings have to work.
    """
    if isinstance(value, bytes | bytearray):
        return float(value.decode())
    return float(value)


def _is_no_script(exc: Exception) -> bool:
    """Whether this is Redis's NOSCRIPT, without importing the driver.

    Matched on the class name as well as the text because redis-py strips the
    `NOSCRIPT` prefix off the server's message when it maps it to
    `NoScriptError` -- so the obvious substring check alone silently never fires,
    and every script cache miss after a Redis restart would fail open forever.
    """
    return type(exc).__name__ == "NoScriptError" or "NOSCRIPT" in str(exc).upper()
