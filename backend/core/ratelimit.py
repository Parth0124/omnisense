"""Inbound rate limiting: a token bucket in Redis, and what happens without it.

`docs/api-reference.md` §3.6 rate-limits per credential rather than per IP, and
that choice is the whole design. An IP is shared by a corporate NAT, a mobile
carrier and every user behind one office router; limiting by IP means one heavy
user throttles their entire company, and an attacker with a handful of addresses
routes around it. A credential is the actual accountable unit.

**Token bucket, not fixed window.** A fixed window lets a client spend its whole
allowance in the last second of one window and again in the first second of the
next -- double the intended rate at the boundary, reliably, and it is exactly
what a retry loop with jitter finds. A bucket refills continuously, so the
average is the limit no matter how the requests land.

**The bucket lives in Redis and refills lazily.** No background timer: the
refill is computed from elapsed time when the bucket is read. A timer would need
a process to own it, and would drift against a bucket that nothing touched for an
hour.

**It fails open, and that is a deliberate, stated trade.** Redis is not on the
list of dependencies that may take the API out of rotation
(`backend/api/v1/health.py`: only PostgreSQL is required). If a Redis outage
made every request fail closed, an outage in an optional dependency would become
a total API outage -- which is precisely the coupling §7.3 exists to prevent. The
cost is that limits are not enforced during a Redis outage; the alternative cost
is that nothing is served at all. `RateLimiter.degraded` records it so the
decision is visible in metrics rather than silent.

Layer note: **L1k kernel.** Takes a Redis-shaped client as a port; imports no
client itself, so the limiter is testable against a dict.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Final, Protocol

from backend.core.logging import get_logger

__all__ = [
    "DEFAULT_BURST",
    "DEFAULT_RATE_PER_MINUTE",
    "KEY_PREFIX",
    "BucketStore",
    "Decision",
    "InMemoryBucketStore",
    "RateLimiter",
]

logger = get_logger(__name__)

KEY_PREFIX: Final = "os:rl:"
DEFAULT_RATE_PER_MINUTE: Final = 120
DEFAULT_BURST: Final = 30
"""Headroom above the steady rate.

Real clients are bursty in a way that is not abuse: a dashboard opening fires six
requests at once and then goes quiet. A bucket with no burst allowance rejects
that page load while permitting the same six requests spread over three seconds,
which is the same load and a much worse experience.
"""

_LUA_TOKEN_BUCKET: Final = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts = tonumber(state[2])
if tokens == nil then
  tokens = burst
  ts = now
end

tokens = math.min(burst, tokens + (now - ts) * rate)
local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', key, ttl)
return {allowed, tostring(tokens)}
"""
"""The whole decision in one round trip, atomically.

A read-modify-write from Python would be a race: two concurrent requests both
read 1 token, both decide they may proceed, and both write 0. Under load that is
not a rare interleaving -- it is the common case, and the limit silently permits
roughly twice its configured rate. A Lua script executes atomically on the Redis
side, which is the only way to make check-and-decrement one operation.
"""


class BucketStore(Protocol):
    """The narrow port: evaluate one bucket, atomically.

    One method. Everything about Redis -- the script, the key layout, the TTL --
    is on the far side, so the limiter's policy is testable against a dict and the
    production path is a single `EVALSHA`.
    """

    async def consume(
        self, key: str, *, rate_per_second: float, burst: float, cost: float, ttl: int
    ) -> tuple[bool, float]: ...


@dataclass(frozen=True, slots=True)
class Decision:
    """The outcome of one limit check, shaped for the response headers §3.6 requires."""

    allowed: bool
    remaining: int
    retry_after_seconds: int
    limit: int
    degraded: bool = False
    """True when the limiter could not reach its store and allowed the request.

    Surfaced rather than hidden because "we are not rate limiting right now" is
    something an operator needs to know during an incident -- and because a spike
    in this flag is the earliest signal that Redis is unwell.
    """


class InMemoryBucketStore:
    """A process-local bucket store. For tests and single-process development.

    Explicitly **not** correct across replicas: each process keeps its own
    buckets, so N replicas permit N times the limit. Named and documented rather
    than quietly provided, because the failure is invisible in a single-process
    test and only appears under a load balancer.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, float]] = {}

    async def consume(
        self, key: str, *, rate_per_second: float, burst: float, cost: float, ttl: int
    ) -> tuple[bool, float]:
        now = time.monotonic()
        tokens, stamp = self._buckets.get(key, (burst, now))
        tokens = min(burst, tokens + (now - stamp) * rate_per_second)
        allowed = tokens >= cost
        if allowed:
            tokens -= cost
        self._buckets[key] = (tokens, now)
        return allowed, tokens


class RedisBucketStore:
    """The production store: one atomic `EVAL` per check."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def consume(
        self, key: str, *, rate_per_second: float, burst: float, cost: float, ttl: int
    ) -> tuple[bool, float]:
        allowed, tokens = await self._client.eval(
            _LUA_TOKEN_BUCKET,
            1,
            key,
            rate_per_second,
            burst,
            time.time(),
            cost,
            ttl,
        )
        return bool(int(allowed)), float(tokens)


class RateLimiter:
    """Per-credential token bucket. Fails open by design -- see the module docstring."""

    def __init__(
        self,
        store: BucketStore | None = None,
        *,
        rate_per_minute: int = DEFAULT_RATE_PER_MINUTE,
        burst: int = DEFAULT_BURST,
    ) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self._store = store
        self._rate_per_minute = rate_per_minute
        self._burst = max(burst, 1)
        self.degraded = False

    @property
    def limit(self) -> int:
        return self._rate_per_minute

    async def check(self, identity: str, *, cost: float = 1.0, scope: str = "api") -> Decision:
        """Consume one token for `identity`, or report that it must wait.

        `identity` is a credential subject, never an IP. See the module docstring:
        an IP is shared by a NAT, a carrier and an office, so limiting by it
        throttles a whole company for one heavy user while an attacker with a few
        addresses routes around it entirely.
        """
        if self._store is None:
            return self._open("no bucket store configured")

        key = f"{KEY_PREFIX}{scope}:{identity}"
        rate_per_second = self._rate_per_minute / 60.0
        # TTL is generous relative to the refill time so an idle bucket expires
        # rather than accumulating one key per credential forever -- and a key
        # that expires simply starts full, which is the correct state for a
        # client that has not been seen in an hour.
        ttl = max(60, int(math.ceil(self._burst / rate_per_second)) * 2)

        try:
            allowed, tokens = await self._store.consume(
                key, rate_per_second=rate_per_second, burst=float(self._burst),
                cost=cost, ttl=ttl,
            )
        except Exception as error:  # noqa: BLE001 -- fail open, loudly
            return self._open(f"{type(error).__name__}: {error}")

        self.degraded = False
        if allowed:
            return Decision(
                allowed=True,
                remaining=int(tokens),
                retry_after_seconds=0,
                limit=self._rate_per_minute,
            )

        # Rounded *up*, and never zero. A `Retry-After: 0` invites an immediate
        # retry that is guaranteed to fail, so the client spins at full rate
        # against an endpoint that is already refusing it -- turning a limit into
        # a load amplifier.
        wait = max(1, int(math.ceil((cost - tokens) / rate_per_second)))
        return Decision(
            allowed=False,
            remaining=0,
            retry_after_seconds=wait,
            limit=self._rate_per_minute,
        )

    def _open(self, reason: str) -> Decision:
        """Allow the request and record that the limiter is not enforcing."""
        if not self.degraded:
            # Logged on the transition only. Logging every request during a Redis
            # outage would add a line per request to a pipeline that is already
            # under stress, which is the last thing an incident needs.
            logger.warning("ratelimit.degraded_open", reason=reason)
        self.degraded = True
        return Decision(
            allowed=True,
            remaining=self._rate_per_minute,
            retry_after_seconds=0,
            limit=self._rate_per_minute,
            degraded=True,
        )


def build_rate_limiter(client: Any | None = None, **kwargs: Any) -> RateLimiter:
    """Wire a limiter over a Redis client, or an in-process one when absent."""
    store: BucketStore | None = RedisBucketStore(client) if client is not None else None
    return RateLimiter(store, **kwargs)
