"""Completion cache: the same question, asked twice, billed once.

Agent runs repeat themselves. A re-run after a code change replays the same
planner prompt, a Critic revision re-asks the same question about the same
evidence, and an eval suite hammers one fixture all afternoon. Every one of
those is a full-price call for an answer we already have.

**The key is a hash of everything that could change the answer**: prompt,
system, model, output schema and the sampling parameters. Leaving any of them
out is not a smaller key, it is a wrong one:

- *model* -- serving an `claude-opus-5` answer to a `claude-haiku-4-5-*` request
  silently changes both the result and the cost attribution, and the cheap tier
  starts posting suspiciously good eval numbers that nobody can reproduce.
- *schema* -- by name alone, a `Plan` whose fields changed yesterday still hits
  yesterday's entry; the JSON Schema itself is hashed so a field rename is a new
  key.
- *params* -- `max_tokens` decides whether the answer is truncated.

**A cache hit costs nothing, and the returned response says so.** `LLMResponse`
has no "this was cached" flag to set, so the token counts carry the truth
instead: `input_tokens` and `output_tokens` come back as zero, and the whole
original total lands in `cached_tokens`. Anything charging `billable_tokens`
therefore charges nothing, anything reading `total_tokens` still sees the real
context pressure, and the run budget cannot be consumed by an answer that was
free. Returning the stored counts verbatim would inflate every cost dashboard by
exactly the amount the cache saved.

**Redis is disposable** (`docs/data-stores.md` §3.4, `docs/architecture.md`
§7.3): it runs `allkeys-lru`, any key can vanish a millisecond after it is
written, and when Redis is down the documented behaviour is "LLM cache misses",
not "the investigation fails". So every operation here swallows client failures
and degrades to a miss. The client arrives as a constructor argument typed by a
local `Protocol` -- the same shape `connectors/ratelimit/limiter.py` uses -- so
this module stays testable with `fakeredis` and never reaches for a global.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Final, Protocol, runtime_checkable

from backend.core.config import get_settings
from backend.core.logging import get_logger
from services.llm.provider import BaseModelT, LLMResponse

__all__ = [
    "CACHE_PREFIX",
    "CompletionCache",
    "RedisLike",
    "completion_cache_key",
]

logger = get_logger(__name__)

CACHE_PREFIX: Final = "llm:c1:"
"""Key namespace, carrying the *payload* version.

Bumping `c1` is how the stored shape changes: a deploy that reads yesterday's
payload with today's parser is a `KeyError` on a path documented to degrade
quietly, and renaming the namespace makes the old entries unreachable instead of
poisonous.
"""


@runtime_checkable
class RedisLike(Protocol):
    """The two commands this module needs.

    Structural rather than `redis.asyncio.Redis` so the cache can be exercised
    against `fakeredis` -- and so a future move to a different client is a
    wiring change rather than an edit here.
    """

    async def get(self, name: str) -> Any: ...

    async def set(self, name: str, value: str, ex: int | None = None) -> Any: ...


class CompletionCache:
    """Read-through cache for completions, keyed by the whole request."""

    def __init__(
        self,
        redis: RedisLike,
        *,
        enabled: bool | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        # `get_settings()` is only reached for the arguments that were not
        # supplied, so a caller that pins both never touches process
        # configuration -- which is what keeps this constructible in a test that
        # has no environment to speak of.
        self._redis = redis
        self._enabled = get_settings().llm.cache_enabled if enabled is None else enabled
        self._ttl = get_settings().redis.cache_ttl_seconds if ttl_seconds is None else ttl_seconds

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def get(
        self,
        *,
        prompt: str,
        model: str,
        system: str | None = None,
        schema: type[BaseModelT] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> LLMResponse | None:
        """Return a cached response, or `None` for a miss.

        `None` is the answer for every failure mode -- disabled, absent, Redis
        unreachable, payload unreadable -- because each of them means the same
        thing to the caller: make the call.
        """
        if not self._enabled:
            # Not a lookup that misses: no command is issued at all, so turning
            # the cache off also removes Redis from the request path.
            return None

        key = completion_cache_key(
            prompt=prompt, model=model, system=system, schema=schema, params=params
        )
        try:
            raw = await self._redis.get(key)
        except Exception as exc:  # noqa: BLE001 -- see the module docstring
            # Broad on purpose: the client is injected and typed structurally, so
            # this module cannot name `redis.RedisError` without importing the
            # driver it deliberately does not depend on. `CancelledError` is a
            # `BaseException` and still propagates.
            logger.debug("llm_cache_get_failed", error=type(exc).__name__)
            return None
        if raw is None:
            return None

        payload = _decode(raw)
        if payload is None:
            logger.warning("llm_cache_payload_unreadable", key=key)
            return None
        return _as_response(payload)

    async def set(
        self,
        response: LLMResponse,
        *,
        prompt: str,
        model: str,
        system: str | None = None,
        schema: type[BaseModelT] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> bool:
        """Store a response. Returns whether it was actually written.

        A write always carries a TTL. Nothing in Redis is authoritative and
        nothing here is written without an expiry -- an un-expiring completion
        would outlive the prompt version that produced it and quietly serve
        last quarter's behaviour.
        """
        if not self._enabled:
            return False
        if response.stop_reason == "max_tokens":
            # Caching a truncated answer means every future caller inherits the
            # truncation, including after LLM_MAX_OUTPUT_TOKENS is raised to fix
            # it. Cheaper to re-ask than to serve a known-broken result forever.
            return False

        key = completion_cache_key(
            prompt=prompt, model=model, system=system, schema=schema, params=params
        )
        try:
            await self._redis.set(key, json.dumps(_as_payload(response)), ex=self._ttl)
        except Exception as exc:  # noqa: BLE001 -- see the module docstring
            logger.debug("llm_cache_set_failed", error=type(exc).__name__)
            return False
        return True


# --------------------------------------------------------------------------- #
# Key derivation
# --------------------------------------------------------------------------- #


def completion_cache_key(
    *,
    prompt: str,
    model: str,
    system: str | None = None,
    schema: type[BaseModelT] | None = None,
    params: Mapping[str, Any] | None = None,
) -> str:
    """`sha256` over every input that can change the answer.

    A module-level function rather than a method so a test -- or an operator
    with `redis-cli` -- can compute a key without constructing a cache, and so
    the "what is in the key" question has exactly one answer to read.

    The material is serialized with `sort_keys=True`: dict ordering is an
    implementation detail of whoever built `params`, and letting it into the
    hash would produce two keys for one request and a cache that never hits.
    """
    material = {
        "prompt": prompt,
        "system": system,
        "model": model,
        # The full schema, not `schema.__name__`. A model whose shape changed
        # under an unchanged name is a different request.
        "schema": schema.model_json_schema() if schema is not None else None,
        "params": dict(params) if params else {},
    }
    encoded = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{CACHE_PREFIX}{digest}"


# --------------------------------------------------------------------------- #
# Payload
# --------------------------------------------------------------------------- #


def _as_payload(response: LLMResponse) -> dict[str, Any]:
    """Serialize a response. The original counts are kept for forensics only.

    They are stored so "how much did this entry save?" is answerable, and they
    are *not* what `get()` hands back -- see the module docstring.
    """
    return {
        "text": response.text,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cached_tokens": response.cached_tokens,
    }


def _as_response(payload: Mapping[str, Any]) -> LLMResponse:
    """Rebuild a response with the cost zeroed and the size preserved."""
    saved = (
        _as_int(payload.get("input_tokens"))
        + _as_int(payload.get("output_tokens"))
        + _as_int(payload.get("cached_tokens"))
    )
    return LLMResponse(
        text=str(payload.get("text", "")),
        model=str(payload.get("model", "")),
        input_tokens=0,
        output_tokens=0,
        cached_tokens=saved,
        stop_reason=payload.get("stop_reason"),
    )


def _decode(raw: Any) -> dict[str, Any] | None:
    """Parse a stored value, tolerating both `bytes` and `str` clients.

    `decode_responses` is a client-construction detail this module does not own,
    and a cache that only works under one setting is a cache that silently stops
    working when somebody tunes the pool.
    """
    if isinstance(raw, bytes | bytearray):
        raw = raw.decode("utf-8", errors="replace")
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _as_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0
