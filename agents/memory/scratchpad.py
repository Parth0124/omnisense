"""The per-run scratchpad: where large intermediate values live instead of the state.

`agents/state.py` says it in its first paragraph -- a checkpoint should be
kilobytes. The state is written to PostgreSQL after *every node*, so anything
stored there is serialised, written and read back on every step and on every
resume. A run that put its retrieved passages in the state would checkpoint
megabytes a dozen times and make a resume a full re-read of everything the run
had ever seen.

So the state carries `scratchpad_key`, and the bulk lives here, in Redis, behind
that key. What goes in the state is a reference; what goes in the scratchpad is
whatever the reference points at.

**Everything is namespaced under one key and expires together.** A run's
scratchpad is a Redis hash at `os:scratch:{investigation_id}` with a TTL. One key
rather than one-key-per-entry means the whole run's working memory is deleted by
a single `DEL` when the investigation is cancelled -- and, more importantly, that
an abandoned run cannot leave scattered keys nothing will ever collect. A
per-entry TTL would expire pieces of a live run's memory at different times,
which produces the worst possible failure: a scratchpad that is *partially*
there.

**It is a cache, not a store, and callers must treat it as one.** A resume after
a Redis restart finds an empty scratchpad. That has to be survivable, so every
`get` returns `None` rather than raising and every caller has a path that
recomputes. The alternative -- treating the scratchpad as durable -- means Redis
becomes a hard dependency of every investigation, which `docs/architecture.md`
§7.3 explicitly does not allow.

**Values are JSON, and size is capped per entry.** A cap because the failure it
prevents is specific: an agent that writes its entire retrieved corpus into one
scratchpad entry turns a Redis instance shared by every run into one run's
working set, and evicts everybody else's memory to hold it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

from backend.core.logging import get_logger

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "MAX_ENTRIES",
    "MAX_VALUE_BYTES",
    "SCRATCHPAD_PREFIX",
    "InMemoryScratchpadStore",
    "Scratchpad",
    "ScratchpadStore",
    "scratchpad_key_for",
]

logger = get_logger(__name__)

SCRATCHPAD_PREFIX: Final = "os:scratch:"
"""Matches `agents.state.new_state`'s default. Restated as a constant here so the
two cannot drift into writing and reading different keys -- a failure that
presents as a scratchpad that is always empty, with no error anywhere."""

DEFAULT_TTL_SECONDS: Final = 24 * 60 * 60
"""One day.

Comfortably longer than any investigation -- the longest budget in
`docs/api-reference.md` is measured in minutes -- and short enough that a crashed
run's memory is gone by the next day rather than accumulating until someone
notices Redis is full. The TTL is refreshed on every write, so a long run does
not expire underneath itself.
"""

MAX_VALUE_BYTES: Final = 256 * 1024
"""Ceiling on one serialised entry.

256 KB holds any reasonable intermediate -- a plan, a set of scores, a few dozen
passage references. What it does not hold is a run's entire retrieved corpus,
which is exactly the write this cap exists to refuse: one run doing that evicts
every other run's working memory from a shared instance.
"""

MAX_ENTRIES: Final = 200
"""Entries per run.

A bound on the *shape* of the failure rather than the size. An agent looping and
writing `note_1`, `note_2`, ... will hit this long before it hits any memory
limit, and hitting a named cap produces a diagnosable error instead of a Redis
instance that slowly fills.
"""


def scratchpad_key_for(investigation_id: str) -> str:
    """The Redis key for one run's scratchpad."""
    return f"{SCRATCHPAD_PREFIX}{investigation_id}"


class ScratchpadStore(Protocol):
    """The narrow port. A hash with a TTL -- nothing Redis-specific escapes it."""

    async def hset(self, key: str, field: str, value: str, *, ttl: int) -> None: ...

    async def hget(self, key: str, field: str) -> str | None: ...

    async def hgetall(self, key: str) -> Mapping[str, str]: ...

    async def hdel(self, key: str, *fields: str) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def hlen(self, key: str) -> int: ...


class InMemoryScratchpadStore:
    """A process-local store, for tests and single-process development.

    Ignores the TTL, which is correct for its purpose and worth stating: a test
    that depended on scratchpad expiry would be a test with a sleep in it, and
    expiry is Redis's behaviour to get right rather than this class's.
    """

    def __init__(self) -> None:
        self._data: dict[str, dict[str, str]] = {}

    async def hset(self, key: str, field: str, value: str, *, ttl: int) -> None:
        self._data.setdefault(key, {})[field] = value

    async def hget(self, key: str, field: str) -> str | None:
        return self._data.get(key, {}).get(field)

    async def hgetall(self, key: str) -> Mapping[str, str]:
        return dict(self._data.get(key, {}))

    async def hdel(self, key: str, *fields: str) -> None:
        bucket = self._data.get(key)
        if bucket is None:
            return
        for field in fields:
            bucket.pop(field, None)

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def hlen(self, key: str) -> int:
        return len(self._data.get(key, {}))


class RedisScratchpadStore:
    """The production store. One hash per run, TTL refreshed on write."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def hset(self, key: str, field: str, value: str, *, ttl: int) -> None:
        await self._client.hset(key, field, value)
        # Refreshed on every write rather than set once at creation, so a run
        # longer than the TTL does not lose its memory halfway through. The cost
        # is one extra command per write; the alternative is a class of bug that
        # only appears on the slowest runs.
        await self._client.expire(key, ttl)

    async def hget(self, key: str, field: str) -> str | None:
        value = await self._client.hget(key, field)
        return value.decode() if isinstance(value, bytes) else value

    async def hgetall(self, key: str) -> Mapping[str, str]:
        raw = await self._client.hgetall(key)
        return {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode() if isinstance(v, bytes) else v
            )
            for k, v in (raw or {}).items()
        }

    async def hdel(self, key: str, *fields: str) -> None:
        if fields:
            await self._client.hdel(key, *fields)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def hlen(self, key: str) -> int:
        return int(await self._client.hlen(key) or 0)


@dataclass(slots=True)
class Scratchpad:
    """One investigation's working memory. A cache with a run-scoped lifetime."""

    store: ScratchpadStore
    key: str
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    max_value_bytes: int = MAX_VALUE_BYTES
    max_entries: int = MAX_ENTRIES

    @classmethod
    def for_investigation(
        cls, store: ScratchpadStore, investigation_id: str, **kwargs: Any
    ) -> Scratchpad:
        return cls(store=store, key=scratchpad_key_for(investigation_id), **kwargs)

    async def put(self, field: str, value: Any) -> bool:
        """Store a value. Returns whether it was written.

        Returns `False` rather than raising when the value is too large or the
        run has too many entries. The scratchpad is a cache: a failed write means
        the caller recomputes, which is a slower run rather than a failed one --
        and raising would turn a Redis size limit into a failed investigation,
        which is the coupling the whole cache-not-store design avoids.

        The refusal is logged with the size, because a caller silently taking the
        slow path forever is the outcome nobody notices.
        """
        try:
            encoded = json.dumps(value, separators=(",", ":"), default=str)
        except (TypeError, ValueError) as error:
            logger.warning(
                "scratchpad.unserialisable", field=field, error=type(error).__name__
            )
            return False

        size = len(encoded.encode("utf-8"))
        if size > self.max_value_bytes:
            logger.warning(
                "scratchpad.value_too_large",
                field=field,
                size_bytes=size,
                limit_bytes=self.max_value_bytes,
                consequence="not cached; the caller will recompute",
            )
            return False

        try:
            if await self.store.hlen(self.key) >= self.max_entries:
                # Checked before the write rather than after, so the cap is a
                # ceiling and not a ceiling-plus-one. An agent looping on
                # `note_N` hits this and gets a named error instead of filling
                # the instance.
                logger.warning(
                    "scratchpad.entry_limit_reached",
                    key=self.key,
                    limit=self.max_entries,
                )
                return False
            await self.store.hset(self.key, field, encoded, ttl=self.ttl_seconds)
        except Exception as error:  # noqa: BLE001 -- a cache write must not fail a run
            logger.warning(
                "scratchpad.write_failed", field=field, error=type(error).__name__
            )
            return False
        return True

    async def get(self, field: str, default: Any = None) -> Any:
        """Read a value, or `default`.

        Never raises. A resume after a Redis restart finds an empty scratchpad,
        and that has to be survivable -- so an unreachable store is
        indistinguishable from a cache miss, which is exactly how a cache should
        behave.
        """
        try:
            raw = await self.store.hget(self.key, field)
        except Exception as error:  # noqa: BLE001 -- a cache miss, not a failure
            logger.debug("scratchpad.read_failed", field=field, error=type(error).__name__)
            return default
        if raw is None:
            return default
        try:
            return json.loads(raw)
        except ValueError:
            # Corrupt entry. Dropped rather than returned, because a caller that
            # got half a JSON document would fail somewhere much less obvious.
            logger.warning("scratchpad.corrupt_entry", field=field)
            return default

    async def all(self) -> dict[str, Any]:
        """Everything in the scratchpad, best-effort.

        Individually decoded so one corrupt entry does not lose the rest -- which
        matters because this is what a resume reads.
        """
        try:
            raw = await self.store.hgetall(self.key)
        except Exception as error:  # noqa: BLE001
            logger.debug("scratchpad.read_all_failed", error=type(error).__name__)
            return {}
        decoded: dict[str, Any] = {}
        for field, value in raw.items():
            try:
                decoded[field] = json.loads(value)
            except ValueError:
                logger.warning("scratchpad.corrupt_entry", field=field)
        return decoded

    async def drop(self, *fields: str) -> None:
        """Remove specific entries."""
        try:
            await self.store.hdel(self.key, *fields)
        except Exception as error:  # noqa: BLE001
            logger.debug("scratchpad.drop_failed", error=type(error).__name__)

    async def clear(self) -> None:
        """Delete the whole scratchpad.

        One `DEL`, because the whole run's memory is one key. Called when an
        investigation is cancelled or completes -- and the reason for the
        single-key layout: with one key per entry, cancelling a run would need a
        scan, and a scan that is skipped leaves keys nothing collects.
        """
        try:
            await self.store.delete(self.key)
        except Exception as error:  # noqa: BLE001
            logger.debug("scratchpad.clear_failed", error=type(error).__name__)


def build_scratchpad(investigation_id: str, client: Any | None = None) -> Scratchpad:
    """Wire a scratchpad over Redis, falling back to process memory.

    The fallback is deliberate and safe: a single-process development run gets a
    working scratchpad with no Redis, and because the scratchpad is a cache by
    contract, the semantics are identical -- just not shared between processes.
    """
    store: ScratchpadStore = (
        RedisScratchpadStore(client) if client is not None else InMemoryScratchpadStore()
    )
    return Scratchpad.for_investigation(store, investigation_id)
