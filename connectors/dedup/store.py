"""The shared seen-set behind dedup layers 1 and 2, and its in-memory twin.

`connectors/protocol.py` declares `DedupStore` as a `Protocol` precisely so a
connector never imports an implementation. This module supplies the two
implementations that satisfy it: `RedisDedupStore`, which
`services/connector_service.py` constructs with a client obtained from
`backend/db/redis.py`, and `InMemoryDedupStore`, which lets the whole connector
suite run with nothing installed.

The client arrives as a constructor argument rather than being imported, which
is what keeps `connectors/` free of any `backend/` import
(`docs/architecture.md` §6.2 rule 2) -- and, less abstractly, what lets
`tests/unit/connectors/` pass with no Redis running.

Two shapes, and the difference matters
--------------------------------------
`seen()` then `mark()` is two round trips with a gap between them. Two workers
processing the same record from two partitions can both observe "not seen" in
that gap and both emit. `claim()` is `SET key 1 EX ttl NX`, one command, and
Redis executes commands one at a time -- so exactly one caller is told it is
first. Both are exposed because both are correct for different callers:
`BaseConnector._is_duplicate` checks several keys before marking any of them
(marking a record half-deduplicated and then discovering it was a duplicate
would leave a key behind that suppresses a *different* record), while a single
key check wants the atomic form.

Failure is not an error
-----------------------
`docs/connector-spec.md` §2.5 makes this a "best-effort accelerator in Redis over
an authoritative uniqueness constraint in PostgreSQL", and §7 layer 4 names that
constraint. So every method here degrades **open**: an unreachable Redis reports
"not seen" and lets the record through. The reasoning is asymmetric. A false
"new" costs one redundant emit, absorbed downstream by `ON CONFLICT (id) DO
UPDATE`. A false "duplicate" silently drops an observation forever, and nothing
downstream can recover it because the raw payload was never fetched again --
posts get deleted and API windows expire. Dedup must never fail a run.

Redis also runs `--maxmemory-policy allkeys-lru` (`docs/data-stores.md` §3.4), so
any key can vanish a millisecond after it was written. Nothing here treats a miss
as information.

Logging
-------
Through the standard library, not `backend/core/logging.py` -- that is the
kernel, and importing it here would break the same rule the injected client
exists to satisfy. Which means these records bypass the kernel's redaction
processor, so this module logs key names and exception *classes* only. Never
`str(err)`: a redis-py connection error is built from whatever connection
details it was given, and a module that cannot redact must not repeat them.

Out of scope on purpose: the layer-3 band sets. Near-duplicate candidate lookup
uses the keys `connectors/dedup/hashing.py::simhash_band_keys` produces, and its
in-memory index is `hashing.BandedIndex`. A Redis-backed band index is a
separate piece of work with a different shape (sets, not string keys) and is not
part of the `DedupStore` protocol.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_DEDUP_TTL_SECONDS",
    "InMemoryDedupStore",
    "RedisClient",
    "RedisDedupStore",
]

logger = logging.getLogger(__name__)

SEEN_VALUE = "1"
"""Payload of a seen-key. The key's existence is the whole datum; a value would
be a place for someone to start storing state in a store that may evict it."""

DEFAULT_DEDUP_TTL_SECONDS = 604_800
"""Seven days, mirroring `CONNECTOR_DEDUP_TTL_SECONDS` in `.env.example`.

A constant rather than a settings lookup, because reading settings means
importing `backend/core/config.py` (`docs/architecture.md` §6.2 rule 2). The
runtime passes the configured value through `SyncContext.dedup_ttl_seconds`;
this exists so a connector under test does not have to."""


@runtime_checkable
class RedisClient(Protocol):
    """The three commands this store needs, and nothing else.

    A structural type rather than `redis.asyncio.Redis`, so that importing this
    module does not require the `redis` package and a test can pass a twenty-line
    fake.

    Declared as plain methods returning `Awaitable`, not as `async def`. redis-py
    implements its commands as ordinary methods that hand back an awaitable, so a
    protocol written with `async def` would fail to match the one client it
    exists to describe -- while this form matches both that and any `async def`
    fake.
    """

    def set(
        self,
        name: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> Awaitable[Any]:
        """`SET name value [EX ex] [NX]`. Returns a truthy value when stored."""
        ...

    def exists(self, *names: str) -> Awaitable[Any]:
        """Number of the given keys that exist."""
        ...

    def delete(self, *names: str) -> Awaitable[Any]:
        """Number of keys removed."""
        ...


class RedisDedupStore:
    """`DedupStore` over Redis. Shared across every worker replica.

    Keys arrive fully qualified from `BaseConnector.dedup_keys`
    (`os:dedup:id:{slug}:{id}`, `os:dedup:sha:{slug}:{digest}`) and are used
    verbatim. No prefix is added here: a store that re-namespaced its input would
    make the key in a log line different from the key in Redis, which is the kind
    of divergence that costs an afternoon during an incident.
    """

    __slots__ = ("_client", "_default_ttl_seconds")

    def __init__(
        self,
        client: RedisClient,
        *,
        default_ttl_seconds: int = DEFAULT_DEDUP_TTL_SECONDS,
    ) -> None:
        """
        Args:
            client: An async Redis client, injected by
                `services/connector_service.py`. Not imported, so this module
                stays inside `docs/architecture.md` §6.2 rule 2.
            default_ttl_seconds: Retention for `claim()` when no TTL is given.
                Seven days, matching `CONNECTOR_DEDUP_TTL_SECONDS`. The runtime
                passes the configured value; this default exists so a connector
                test does not have to.
        """
        if default_ttl_seconds <= 0:
            raise ValueError(
                f"default_ttl_seconds must be positive, got {default_ttl_seconds}; "
                "a seen-key with no expiry accumulates forever in a store that is "
                "meant to be disposable"
            )
        self._client = client
        self._default_ttl_seconds = default_ttl_seconds

    async def seen(self, key: str) -> bool:
        """Whether this key was observed inside its TTL window.

        Returns `False` when Redis is unreachable -- "let it through", per the
        degradation reasoning in the module docstring. A `True` on failure would
        drop real observations every time the cache blipped.
        """
        try:
            count = await self._client.exists(key)
        except Exception as err:  # broad on purpose -- dedup must never fail a run
            self._log_degraded("seen", key, err)
            return False
        return int(count) > 0

    async def mark(self, key: str, ttl_seconds: int) -> None:
        """Record a key as seen. Never raises.

        `NX` even though the caller is not reading the result: re-marking an
        existing key without it would reset the TTL, so an item re-fetched on
        every poll inside the overlap window would keep its seen-key alive
        indefinitely and could never be legitimately re-emitted after the
        retention window closed.
        """
        if ttl_seconds <= 0:
            # No expiry means the key outlives its usefulness forever. Fall
            # through to PostgreSQL's unique index instead, exactly as an
            # unreachable Redis does.
            self._log_disabled_ttl("mark", key, ttl_seconds)
            return
        try:
            await self._client.set(key, SEEN_VALUE, ex=ttl_seconds, nx=True)
        except Exception as err:  # broad on purpose -- dedup must never fail a run
            self._log_degraded("mark", key, err)

    async def claim(self, key: str, ttl_seconds: int | None = None) -> bool:
        """Atomically check and mark. `True` means *this caller saw it first*.

        The single-round-trip shape: `SET key 1 EX ttl NX` is one command, so two
        workers racing on the same record cannot both be told they are first.
        The check-then-write pair (`seen()` then `mark()`) has a window between
        the two calls in which both lose.

        Note the polarity -- `True` is the inverse of `seen()`. It reads as "the
        claim succeeded, this record is yours to process".

        `True` is also the answer when Redis is unreachable or the TTL is
        disabled, for the same reason `seen()` answers `False`: both mean
        "proceed".
        """
        ttl = self._default_ttl_seconds if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            self._log_disabled_ttl("claim", key, ttl)
            return True
        try:
            stored = await self._client.set(key, SEEN_VALUE, ex=ttl, nx=True)
        except Exception as err:  # broad on purpose -- dedup must never fail a run
            self._log_degraded("claim", key, err)
            return True
        # redis-py answers True when the key was set and None when NX rejected
        # it; a plain bool() covers both without depending on which.
        return bool(stored)

    async def forget(self, key: str) -> bool:
        """Remove a seen-key. Returns whether one was removed.

        Needed because a run can be repudiated: `lineage.sync_run_id` exists so
        "a bad run can be identified and reverted wholesale", and without this
        the seen-keys from that run would suppress the corrected re-ingest for a
        week. Not part of the `DedupStore` protocol, which is the read path.
        """
        try:
            removed = await self._client.delete(key)
        except Exception as err:  # broad on purpose -- dedup must never fail a run
            self._log_degraded("forget", key, err)
            return False
        return int(removed) > 0

    def _log_degraded(self, operation: str, key: str, err: Exception) -> None:
        """Record a degraded-mode fallback (`docs/connector-spec.md` §2.5).

        The exception *class*, never its message. redis-py builds connection
        errors out of the connection details it was handed, and this logger does
        not pass through the kernel's redaction processor.
        """
        logger.warning(
            "dedup.redis.unavailable operation=%s key=%s error=%s outcome=fail_open",
            operation,
            key,
            type(err).__name__,
        )

    def _log_disabled_ttl(self, operation: str, key: str, ttl_seconds: int) -> None:
        logger.warning(
            "dedup.ttl_disabled operation=%s key=%s ttl_seconds=%d outcome=fail_open",
            operation,
            key,
            ttl_seconds,
        )


class InMemoryDedupStore:
    """`DedupStore` in a dict, with real TTL expiry. For tests and `--dry-run`.

    Expiry is measured against an injectable clock rather than wall time so a
    test can prove that a key stops suppressing records after its window closes
    without sleeping for a week -- or for a second. A suite that sleeps to test a
    TTL either takes minutes or tests a TTL nobody uses.

    Not thread-safe and not shared between processes, which is the point: it
    exists so `tests/unit/connectors/` needs no Redis, and using it in a worker
    would give every replica its own private idea of what it had already seen.
    """

    __slots__ = ("_deadlines", "_now")

    def __init__(self, *, time_source: Callable[[], float] | None = None) -> None:
        self._deadlines: dict[str, float] = {}
        self._now: Callable[[], float] = time_source or time.monotonic
        # Monotonic, not `time.time`: an NTP step backwards would resurrect
        # expired keys and one forwards would evict live ones.

    async def seen(self, key: str) -> bool:
        return self._live(key)

    async def mark(self, key: str, ttl_seconds: int) -> None:
        """Record a key. A non-positive TTL stores nothing, matching Redis.

        Existing keys keep their original deadline, mirroring `SET ... NX`, so a
        record re-fetched on every poll cannot hold its own seen-key open.
        """
        if ttl_seconds <= 0:
            return
        if not self._live(key):
            self._deadlines[key] = self._now() + ttl_seconds

    async def claim(self, key: str, ttl_seconds: int | None = None) -> bool:
        """Atomic by virtue of being single-threaded. `True` if seen first."""
        ttl = DEFAULT_DEDUP_TTL_SECONDS if ttl_seconds is None else ttl_seconds
        if ttl <= 0:
            return True
        if self._live(key):
            return False
        self._deadlines[key] = self._now() + ttl
        return True

    async def forget(self, key: str) -> bool:
        return self._deadlines.pop(key, None) is not None

    def _live(self, key: str) -> bool:
        """Whether the key exists and has not expired, evicting it if it has.

        Lazy eviction rather than a sweep: this store lives for the length of one
        test or one dry run, so a background task would cost more than the keys
        it reclaims.
        """
        deadline = self._deadlines.get(key)
        if deadline is None:
            return False
        if deadline <= self._now():
            del self._deadlines[key]
            return False
        return True

    def __len__(self) -> int:
        """Live keys. Expired-but-unreclaimed entries are excluded, so this
        matches what `seen()` would answer rather than the dict's size."""
        return sum(1 for key in list(self._deadlines) if self._live(key))
