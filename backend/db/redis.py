"""Redis client: cache, rate-limit buckets and the connector dedup seen-set.

Redis is the one deliberately **disposable** store in OmniSense
(`docs/data-stores.md` §3.4). The container runs with
`--maxmemory-policy allkeys-lru`, so any key can be evicted at any moment,
including one written a millisecond ago. Every helper below is written on that
assumption: nothing here is authoritative, nothing is written without a TTL, and
a miss is always a legal answer rather than an error.

The module follows the shape established by `backend/db/session.py` -- a lazily
created module-level singleton, no I/O at import time, and
`get_redis()` / `check_redis()` / `dispose_redis()`. There is deliberately **no**
`require_redis()`: `docs/architecture.md` §7.3 lists no path on which Redis is
fatal, so a helper that turns "Redis is down" into a 503 would only invite
callers to violate the degradation contract.

The degradation contract (`docs/architecture.md` §7.3)
------------------------------------------------------
When Redis is unavailable the two rate-limiting directions must fail in
**opposite** directions, and getting it backwards is a real outage either way:

*Inbound* API limiting fails **closed**. Without a shared counter we cannot tell
a polite client from a hostile one, and letting everyone through is how a Redis
blip becomes a PostgreSQL overload. Denying is recoverable; the client retries.

*Outbound* connector limiting fails **open**, under conservative process-local
static limits. Blocking ingestion because our own cache is down helps nobody,
but fetching unthrottled risks a permanent ban from a third-party source that no
amount of fixing Redis will undo. So we keep fetching, slowly.

Because those two behaviours are indistinguishable at the call site, they are
exposed as `check_inbound_rate_limit_fail_closed()` and
`check_outbound_rate_limit_fail_open()`. The direction is in the name: reaching
for the wrong one requires typing the wrong word.

Layer note: this is the **L1k kernel** (`docs/architecture.md` §6.1). `services/`,
`agents/`, `workers/`, `backend/api/` and `scripts/` may import it; `connectors/`
may **not** (§6.2 rule 2). That is why every helper takes an optional `client`:
`services/connector_service.py` obtains the client here and passes it to
`connectors/dedup/store.py` and `connectors/ratelimit/limiter.py` as a
constructor argument, so the connector package keeps its "no `backend/` import"
guarantee and stays testable without this module.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Final, cast

from redis.asyncio import ConnectionPool, Redis

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger

__all__ = [
    "RateLimitDecision",
    "cache_delete",
    "cache_get_json",
    "cache_set_json",
    "check_inbound_rate_limit_fail_closed",
    "check_outbound_rate_limit_fail_open",
    "check_redis",
    "dispose_redis",
    "get_redis",
    "mark_seen",
]

# Via the kernel accessor rather than `structlog.get_logger` directly, so these
# records pass through the redaction processor in `backend/core/logging.py`.
logger = get_logger(__name__)

# Key prefixes. `docs/data-stores.md` §5.2 fixes the cache and dedup shapes
# (`sig:{id}`, `dedup:{platform}:{hash}`); the two rate-limit namespaces are kept
# apart so that flushing inbound buckets during an incident cannot also reset a
# connector's outbound budget and trigger the ban we were avoiding.
CACHE_PREFIX: Final = "cache:"
DEDUP_PREFIX: Final = "dedup:"
INBOUND_RATE_PREFIX: Final = "rl:in:"
OUTBOUND_RATE_PREFIX: Final = "rl:out:"

# Redis is a latency optimization; waiting on it defeats the purpose. A cache
# lookup that takes two seconds has already cost more than the PostgreSQL query
# it was meant to avoid, so we would rather fail fast and degrade.
_SOCKET_TIMEOUT_SECONDS: Final = 2.0
_SOCKET_CONNECT_TIMEOUT_SECONDS: Final = 1.0
_HEALTHCHECK_TIMEOUT_SECONDS: Final = 2.0

# The Redis analogue of session.py's `pool_pre_ping`: connections idle in the
# pool are dropped by Redis's own `timeout` setting and by cloud load balancers,
# and without a periodic health check the first command after an idle period
# fails with a bare `ConnectionResetError` that reads like an outage.
_HEALTH_CHECK_INTERVAL_SECONDS: Final = 30

# Fraction of the configured limit that the outbound fallback allows while Redis
# is down. The shared counter is what makes a limit correct across processes;
# without it each process independently grants the full budget, so N workers
# fetch at N times the intended rate -- precisely the ban risk being avoided.
# Halving is a hedge, not a calculation: the process cannot know N.
_FALLBACK_LIMIT_SHARE: Final = 0.5

# The fallback map is keyed by a caller-supplied bucket, so a pathological caller
# could grow it without bound. Pruning expired windows past this size keeps a
# Redis outage from also becoming a memory leak.
_FALLBACK_MAX_BUCKETS: Final = 4096

_client: Redis | None = None
_pool: ConnectionPool | None = None

# JSON is the only value shape this module writes. Binary blobs belong in R2
# (`docs/data-stores.md`), which is why the pool decodes responses to `str`.
type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None


@dataclass(slots=True)
class _FallbackWindow:
    """One process-local fixed window, used only while Redis is unreachable."""

    expires_at: float
    count: int


_fallback_windows: dict[str, _FallbackWindow] = {}


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """The outcome of one rate-limit check.

    `degraded` is the field that makes an outage visible: it is `True` only when
    the decision came from a fallback rather than from the shared counter, so
    `backend/core/ratelimit.py` can emit a metric and an operator can tell "we
    are denying because a client is noisy" from "we are denying because Redis is
    gone". Without it, a fail-closed outage looks exactly like ordinary traffic
    shaping on every dashboard.
    """

    allowed: bool
    limit: int
    remaining: int
    retry_after_seconds: int
    degraded: bool

    def __bool__(self) -> bool:
        """Truthiness is `allowed`.

        Defined on purpose: `if decision:` is the obvious thing to write, and on
        a plain dataclass it would always be true, silently granting every
        request. Making the obvious spelling correct is cheaper than trusting
        review to catch a missing `.allowed`.
        """
        return self.allowed


def get_redis() -> Redis:
    """Return the process-wide Redis client, creating it on first use.

    The client is built from an explicit `ConnectionPool` rather than
    `Redis.from_url()` so `dispose_redis()` has something to disconnect: the
    client's own `aclose()` returns connections to the pool but leaves the
    pool's sockets open, which shows up as file descriptors surviving shutdown.

    `decode_responses=True` because every value this module stores is JSON text
    or an integer counter; without it every read site needs its own `.decode()`
    and the first one to forget produces the silent bug `b"1" != "1"`.
    """
    global _client, _pool
    if _client is None:
        settings = get_settings()
        try:
            _pool = ConnectionPool.from_url(
                settings.redis.url,
                max_connections=settings.redis.max_connections,
                decode_responses=True,
                socket_timeout=_SOCKET_TIMEOUT_SECONDS,
                socket_connect_timeout=_SOCKET_CONNECT_TIMEOUT_SECONDS,
                health_check_interval=_HEALTH_CHECK_INTERVAL_SECONDS,
            )
        except (ValueError, TypeError) as err:
            # Never echo the URL itself: `REDIS_URL` carries the password in any
            # deployment that has one.
            raise ConfigurationError(
                "REDIS_URL is not a valid Redis connection URL "
                "(expected redis://, rediss:// or unix://).",
                cause=err,
            ) from err
        # No client-side retry policy: retrying inside a cache lookup multiplies
        # the very latency the cache exists to remove, and every caller in this
        # module already has a defined degraded path.
        _client = Redis(connection_pool=_pool)
    return _client


async def check_redis() -> bool:
    """Probe Redis for `/readyz`. Never raises.

    Bounded by an explicit timeout as well as by the socket timeouts: a host
    that silently drops packets -- a stale security group, a half-closed NAT --
    accepts the connection attempt and then never answers, and an unbounded
    `PING` would hold the readiness endpoint open until the kernel gives up.

    Returns a bool rather than raising because readiness aggregates several
    dependencies and one being down must not prevent reporting on the others
    (`docs/observability.md`).
    """
    try:
        async with asyncio.timeout(_HEALTHCHECK_TIMEOUT_SECONDS):
            await get_redis().ping()
    except Exception:
        return False
    return True


async def dispose_redis() -> None:
    """Close the pool and reset module state. Called from lifespan shutdown.

    Also clears the outbound fallback windows: they are process-local state with
    no meaning across a restart, and leaving them behind lets a test that
    exercised the degraded path bleed into the next one.
    """
    global _client, _pool
    if _client is not None:
        await _client.aclose()
    if _pool is not None:
        await _pool.disconnect()
    _client = None
    _pool = None
    _fallback_windows.clear()


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


async def cache_get_json(key: str, *, client: Redis | None = None) -> JsonValue:
    """Read a JSON value from the cache. Returns `None` on a miss. Never raises.

    A Redis failure is reported as a miss, which is exactly what §7.3's "caches
    miss through to PostgreSQL" requires: the caller recomputes, more slowly,
    and the request still succeeds.

    Note the deliberate ambiguity: a cached JSON `null` is indistinguishable
    from a miss. Do not cache a value whose absence and whose `null` mean
    different things -- cache a wrapper object instead.
    """
    redis_client = client or get_redis()
    try:
        raw = await redis_client.get(CACHE_PREFIX + key)
    except Exception as err:
        logger.warning("redis.cache.get_failed", cache_entry=key, outcome="skipped", error=str(err))
        return None
    if raw is None:
        return None
    try:
        # `json.loads` is typed `Any`; narrowed here, once, rather than at every
        # call site.
        return cast(JsonValue, json.loads(raw))
    except (TypeError, ValueError) as err:
        # A corrupt or truncated entry -- an eviction mid-write, or a value
        # written by an older schema. Treat it as a miss rather than propagate a
        # decode error into a handler that only asked for a cache.
        logger.warning("redis.cache.decode_failed", cache_entry=key, error=str(err))
        return None


async def cache_set_json(
    key: str,
    value: JsonValue,
    *,
    ttl_seconds: int | None = None,
    client: Redis | None = None,
) -> bool:
    """Write a JSON value with a TTL. Returns whether it was stored. Never raises.

    `ttl_seconds` defaults to `REDIS_CACHE_TTL_SECONDS`. A TTL of zero or less
    means "do not cache" and the write is skipped: `docs/data-stores.md` §3.4
    forbids storing anything without a TTL, so honouring
    `REDIS_CACHE_TTL_SECONDS=0` by writing an immortal key would turn a "disable
    the cache" setting into a slow memory leak.
    """
    ttl = get_settings().redis.cache_ttl_seconds if ttl_seconds is None else ttl_seconds
    if ttl <= 0:
        return False

    redis_client = client or get_redis()
    try:
        await redis_client.set(CACHE_PREFIX + key, json.dumps(value), ex=ttl)
    except Exception as err:
        logger.warning("redis.cache.set_failed", cache_entry=key, outcome="skipped", error=str(err))
        return False
    return True


async def cache_delete(key: str, *, client: Redis | None = None) -> bool:
    """Invalidate one cache entry. Returns whether a key was removed. Never raises.

    Invalidation is best-effort by design. If Redis is unreachable the stale
    entry still cannot outlive its TTL, and that bound is what makes
    "best-effort" acceptable here rather than a correctness hole.
    """
    redis_client = client or get_redis()
    try:
        removed = await redis_client.delete(CACHE_PREFIX + key)
    except Exception as err:
        logger.warning(
            "redis.cache.delete_failed", cache_entry=key, outcome="skipped", error=str(err)
        )
        return False
    return int(removed) > 0


# --------------------------------------------------------------------------- #
# Connector dedup seen-set
# --------------------------------------------------------------------------- #


async def mark_seen(
    namespace: str,
    key: str,
    *,
    ttl_seconds: int | None = None,
    client: Redis | None = None,
) -> bool:
    """Record a dedup key and report whether this is the **first** sighting.

    `SET NX EX` is one command, so two workers racing on the same record cannot
    both be told they are first -- Redis executes commands one at a time, which
    is what makes this a compare-and-set rather than a check-then-write.

    Args:
        namespace: Usually the platform slug, giving `dedup:{platform}:{hash}`
            as specified in `docs/data-stores.md` §5.2.
        key: The record's content or identity hash, from
            `connectors/dedup/hashing.py`.
        ttl_seconds: Retention window for the seen-key. Defaults to
            `CONNECTOR_DEDUP_TTL_SECONDS` (7 days).
        client: Injected by `services/connector_service.py` so that
            `connectors/` never imports this module (`docs/architecture.md`
            §6.2 rule 2).

    Returns:
        `True` if the key was not already present, meaning the caller should
        process the record. `True` is also returned when Redis is unreachable
        **or** when the TTL is disabled: §7.3 says dedup degrades to the
        database-level upsert, and `docs/signal-model.md` §4.2 makes
        PostgreSQL's identity layer the backstop. Dropping records because a
        disposable cache is down would lose data permanently, whereas a
        duplicate fetch costs one wasted request and is then absorbed by
        `ON CONFLICT (id) DO UPDATE`.
    """
    ttl = get_settings().connectors.dedup_ttl_seconds if ttl_seconds is None else ttl_seconds
    if ttl <= 0:
        # A seen-key with no expiry would accumulate forever, and no data in
        # Redis is worth that. Fall through to the PostgreSQL layer instead.
        return True

    redis_client = client or get_redis()
    try:
        stored = await redis_client.set(f"{DEDUP_PREFIX}{namespace}:{key}", "1", ex=ttl, nx=True)
    except Exception as err:
        logger.warning(
            "redis.dedup.unavailable",
            connector=namespace,
            outcome="skipped",
            error=str(err),
        )
        return True
    # redis-py returns True when the key was set and None when NX rejected it.
    return bool(stored)


# --------------------------------------------------------------------------- #
# Rate limiting -- see the degradation contract in the module docstring
# --------------------------------------------------------------------------- #


async def check_inbound_rate_limit_fail_closed(
    bucket: str,
    *,
    limit: int,
    window_seconds: int,
    client: Redis | None = None,
) -> RateLimitDecision:
    """Consume one unit of an **inbound** (API caller) budget. Never raises.

    Fails **closed**: if Redis cannot be reached the request is denied. There is
    no shared counter, therefore no way to distinguish a first request from a
    ten-thousandth, and admitting everything is how a cache outage turns into a
    PostgreSQL outage (`docs/architecture.md` §7.3).

    Args:
        bucket: The identity being limited -- API key, tenant, or client IP.
        limit: Requests permitted per window.
        window_seconds: Fixed-window length.

    Returns:
        A decision whose `retry_after_seconds` is meant for the `Retry-After`
        header. The caller raises `RateLimitedError`; HTTP status codes live
        only in `backend/api/errors.py` (`docs/coding-standards.md` §2.7).
    """
    redis_client = client or get_redis()
    try:
        count, ttl_ms = await _incr_fixed_window(
            redis_client, INBOUND_RATE_PREFIX + bucket, window_seconds
        )
    except Exception as err:
        logger.warning(
            "redis.ratelimit.inbound_fail_closed",
            bucket=bucket,
            outcome="error",
            error=str(err),
        )
        return RateLimitDecision(
            allowed=False,
            limit=limit,
            remaining=0,
            # A Redis outage outlasts one window; asking the client back after a
            # full window is honest and stops it hot-looping on 429s.
            retry_after_seconds=window_seconds,
            degraded=True,
        )
    return _decide(count=count, limit=limit, ttl_ms=ttl_ms, window_seconds=window_seconds)


async def check_outbound_rate_limit_fail_open(
    bucket: str,
    *,
    limit: int,
    window_seconds: int,
    client: Redis | None = None,
) -> RateLimitDecision:
    """Consume one unit of an **outbound** (third-party API) budget. Never raises.

    Fails **open** onto a conservative, process-local static limit. Halting
    ingestion because our own cache is down costs freshness; hammering a source
    unthrottled costs the source. The fallback therefore keeps fetching at a
    fraction of the configured rate rather than at zero or at full speed.

    The fallback is per-process and so is *not* a correct global limit -- which
    is exactly why it is conservative, and why the decision is flagged
    `degraded` so the connector can log it and an operator can see how long
    ingestion has been running without a shared counter.

    Args:
        bucket: Usually `{connector_slug}:{account_id}` -- limits are per
            credential, because that is what the upstream throttles.
        limit: Requests permitted per window while Redis is healthy.
        window_seconds: Fixed-window length.
    """
    redis_client = client or get_redis()
    try:
        count, ttl_ms = await _incr_fixed_window(
            redis_client, OUTBOUND_RATE_PREFIX + bucket, window_seconds
        )
    except Exception as err:
        logger.warning(
            "redis.ratelimit.outbound_fail_open",
            connector=bucket,
            outcome="skipped",
            error=str(err),
        )
        return _fallback_outbound_decision(bucket, limit=limit, window_seconds=window_seconds)
    return _decide(count=count, limit=limit, ttl_ms=ttl_ms, window_seconds=window_seconds)


async def _incr_fixed_window(client: Redis, key: str, window_seconds: int) -> tuple[int, int]:
    """Increment a fixed-window counter and return `(count, remaining_ms)`.

    `EXPIRE ... NX` (Redis 7.0+, which `docker-compose.yml` pins) sets the TTL
    only when the key has none, so the window starts at the first request and is
    not pushed forward by every subsequent one -- with an unconditional `EXPIRE`
    a steady stream of requests keeps the window open indefinitely and the limit
    never resets.

    The three commands run inside `MULTI`/`EXEC`, so no other client observes
    the counter between `INCR` and `EXPIRE`. Without that, a process dying in
    between leaves a counter with no TTL: a permanently exhausted bucket that no
    eviction policy is guaranteed to clear.
    """
    async with client.pipeline(transaction=True) as pipe:
        pipe.incr(key)
        pipe.expire(key, window_seconds, nx=True)
        pipe.pttl(key)
        # `execute()` is typed `list[Any]` by redis-py; the positions below are
        # fixed by the queue order above.
        results: list[Any] = await pipe.execute()

    count = int(results[0])
    ttl_ms = int(results[2])
    if ttl_ms < 0:
        # -1 (no TTL) or -2 (key gone, e.g. LRU eviction between commands).
        # Treat the window as fresh rather than report a negative wait.
        ttl_ms = window_seconds * 1000
    return count, ttl_ms


def _decide(*, count: int, limit: int, ttl_ms: int, window_seconds: int) -> RateLimitDecision:
    """Turn a counter reading into a decision. The shared, non-degraded path."""
    allowed = count <= limit
    retry_after = 0 if allowed else max(1, math.ceil(ttl_ms / 1000))
    return RateLimitDecision(
        allowed=allowed,
        limit=limit,
        remaining=max(0, limit - count),
        # Never advertise a retry longer than the window itself; clock skew or a
        # stale TTL should not park a connector for an hour.
        retry_after_seconds=min(retry_after, window_seconds),
        degraded=False,
    )


def _fallback_outbound_decision(
    bucket: str, *, limit: int, window_seconds: int
) -> RateLimitDecision:
    """Decide from process-local state while Redis is unreachable.

    Synchronous on purpose: with no `await` between reading and writing the
    window, the update is atomic with respect to every other task on this event
    loop, so no lock is needed and none can be forgotten.
    """
    conservative_limit = max(1, int(limit * _FALLBACK_LIMIT_SHARE))
    now = time.monotonic()

    window = _fallback_windows.get(bucket)
    if window is None or now >= window.expires_at:
        window = _FallbackWindow(expires_at=now + window_seconds, count=0)
        _fallback_windows[bucket] = window
    window.count += 1

    if len(_fallback_windows) > _FALLBACK_MAX_BUCKETS:
        _prune_fallback_windows(now)

    allowed = window.count <= conservative_limit
    retry_after = 0 if allowed else max(1, math.ceil(window.expires_at - now))
    return RateLimitDecision(
        allowed=allowed,
        limit=conservative_limit,
        remaining=max(0, conservative_limit - window.count),
        retry_after_seconds=min(retry_after, window_seconds),
        degraded=True,
    )


def _prune_fallback_windows(now: float) -> None:
    """Drop expired fallback windows. Bounded work, and only when oversized."""
    expired = [bucket for bucket, window in _fallback_windows.items() if now >= window.expires_at]
    for bucket in expired:
        del _fallback_windows[bucket]
