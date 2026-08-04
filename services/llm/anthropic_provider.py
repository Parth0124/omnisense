"""Anthropic Messages API, through the official `anthropic` SDK.

The previous version of this module hand-rolled the wire format over `httpx`:
a pinned `anthropic-version` header, hand-written auth headers, hand-assembled
payloads, hand-written content-block shredding, a hand-written tool-use schema
encoding and a hand-written validation-retry loop. Every one of those is a copy
of something the vendor already maintains, against an API that ships breaking
changes -- so each release quietly widened the gap between what we send and what
the provider expects, and nothing in this repository could detect the drift.
The SDK is the correct client whenever one exists, and it was already declared
in `requirements.txt` while being imported nowhere.

Four decisions in this module are load-bearing.

**Structured output goes through the SDK's native parse path.** `messages.parse`
sends the schema as `output_config.format` and validates the reply against the
Pydantic model before returning it, which is the same guarantee the old
forced-tool-call plus local-validation dance produced, minus the encoding. What
is deliberately *gone* is the retry-with-the-error-fed-back loop: constrained
decoding makes a near-miss value impossible, so the only realistic failure left
is truncation mid-object, and re-asking does not fix a budget that is too small.
`LLMSchemaError` still exists, and now means something actionable.

**Every response records tokens, including cache reads.** Cost is a runtime
signal, not a monthly invoice (`docs/observability.md` §8). The usage block is
only available on the response that produced it, so a call whose usage is
dropped is a call whose cost can never be reconstructed. One gap remains and is
named here rather than hidden: when the SDK raises on unparseable output the
`Message` never reaches us, so that call's tokens are unrecoverable. It is the
price of not re-implementing the parse, and it is bounded -- that path raises.

**Sampling parameters are model-gated.** The current Claude generation removed
`temperature`/`top_p`/`top_k`; sending `temperature` to `claude-opus-5` is a 400,
while `claude-haiku-4-5-*` still accepts it. Since `LLM_TEMPERATURE` defaults to
`0.0` and is therefore always "set", a provider that forwarded it
unconditionally would fail every planner and worker call in this deployment on
the first request. `_accepts_sampling()` is that guard, and it fails *open* for
unknown ids so a model released tomorrow is not silently stripped of a parameter
it does support. The SDK does not know this rule; it forwards what it is given.

**Retries are the SDK's, bounded by `LLM_MAX_RETRIES`.** This module used to
refuse to retry at all, on the grounds that private retries make the router's
overload signal wrong. That argument was about *unbounded, hand-written* retries
of everything; the SDK's policy is narrower and better than what we would write
-- connection failures, 408/409/429 and 5xx only, exponential backoff, and it
honours `Retry-After` -- and it never touches a 400, which is the class where a
retry is pure waste. What survives the budget still reaches
`services/llm/router.py` as `LLMRateLimited` or `LLMTimeout` and is still shed a
tier. A deployment that wants the old behaviour sets `LLM_MAX_RETRIES=0`, and
should also know that `LLM_TIMEOUT_SECONDS` bounds one *attempt*, so the
wall-clock ceiling is roughly `timeout x (retries + 1)`.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any, Final, TypeVar

import anthropic
from anthropic.types import Message
from pydantic import ValidationError as PydanticValidationError

from backend.core.config import LLMSettings, get_settings
from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger
from services.llm.provider import (
    BaseModelT,
    LLMError,
    LLMRateLimited,
    LLMResponse,
    LLMSchemaError,
    LLMTimeout,
)

__all__ = ["DEFAULT_BASE_URL", "AnthropicProvider"]

logger = get_logger(__name__)

_T = TypeVar("_T")

DEFAULT_BASE_URL: Final = "https://api.anthropic.com"
"""Passed to the SDK explicitly, even though it is also the SDK's own default.

Left unset, the client falls back to `ANTHROPIC_BASE_URL` from the process
environment -- a second, undeclared way to redirect every LLM call, invisible to
`Settings.describe()` and to `.env.example`. `backend/core/config.py` is the only
module allowed to read the environment (`docs/coding-standards.md` §2.9), so the
endpoint is resolved here from settings and handed over as a value.
"""

_RETIRED_MODEL_PREFIXES: Final = ("claude-3", "claude-2", "claude-instant")
"""Model ids that no longer exist and answer 404.

Worth a startup-time guard because the 404 arrives at the *first request*,
looking exactly like a wrong URL or a revoked key, and the fix (edit
`LLM_MODEL_*`) is nowhere near where the symptom shows up.
"""

_SAMPLING_FREE_PREFIXES: Final = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable",
    "claude-mythos",
)
"""Models that reject `temperature`/`top_p`/`top_k` outright. See module docstring."""

_TIMEOUT_STATUSES: Final = frozenset({408, 504})
"""Provider-side timeouts. Distinct from a refusal: the request may have been billed."""


class AnthropicProvider:
    """`LLMProvider` backed by the Anthropic Messages API.

    Stateless per call and safe to share: one instance per process, driven
    concurrently, holding a single pooled `anthropic.AsyncAnthropic`. Nothing
    per-request is stored on `self`.
    """

    def __init__(
        self,
        *,
        settings: LLMSettings | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings().llm

        key = self._settings.anthropic_api_key
        if key is None or not key.get_secret_value():
            # Boot-time, not request-time. A worker that starts happily and then
            # fails every enrichment an hour later is far more expensive to
            # diagnose than one that refuses to start.
            raise ConfigurationError(
                "ANTHROPIC_API_KEY is not set; AnthropicProvider cannot be constructed.",
                details={"setting": "ANTHROPIC_API_KEY"},
            )

        self._base_url = (self._settings.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._default_model = _validated_model(self._settings.model_worker)

        # An injected client is somebody else's to close; ours is ours. Getting
        # this backwards leaks sockets in one direction and closes a shared pool
        # out from under other callers in the other.
        self._owns_client = client is None
        self._client = client or anthropic.AsyncAnthropic(
            api_key=key.get_secret_value(),
            base_url=self._base_url,
            timeout=float(self._settings.timeout_seconds),
            max_retries=self._settings.max_retries,
        )

    # ------------------------------------------------------------- protocol --

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """One user turn in, one text response out."""
        resolved = _validated_model(model or self._default_model)
        message = await self._send(
            self._client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                **self._request_kwargs(
                    model=resolved,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
        )
        return _response_of(message, model=resolved, text=_text_of(message))

    async def structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModelT],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> BaseModelT:
        """Schema-constrained output, validated before it is returned.

        **Why a schema on the request and not "reply with JSON".** Asking for
        JSON in the prompt and parsing the reply is a different mechanism with
        strictly worse failure modes:

        - The reply is unconstrained text. It can arrive wrapped in a markdown
          fence, prefixed with "Here is the JSON:", or with a trailing comma --
          so the caller ends up owning an ad-hoc repair parser, and every fix to
          it is a guess about what the next model version will emit.
        - A field can be *plausibly wrong* rather than absent: `"sentiment":
          "positive!"` parses as JSON and fails our enum. With the schema on the
          request the provider constrains generation to it, so the whole class
          of near-miss values never occurs.
        - Every parse failure costs a full second call at full price and is
          indistinguishable, in metrics, from a model-quality regression.
        - Constrained output is the one schema mechanism every candidate backend
          has, so it survives the vendor swap that Design Doc §15 exists to allow.

        Validation still happens locally -- the SDK does it -- because
        "constrained" is a provider claim and a truncated response can end
        mid-object. What changes is that a failure is now rare and *actionable*
        rather than routine.
        """
        value, _ = await self.structured_metered(
            prompt=prompt,
            schema=schema,
            system=system,
            model=model,
            max_tokens=max_tokens,
        )
        return value

    async def structured_metered(
        self,
        *,
        prompt: str,
        schema: type[BaseModelT],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[BaseModelT, LLMResponse]:
        """`structured()`, plus the usage the call cost.

        One request, no schema retry. The retry the hand-rolled version made --
        re-asking with the validation error fed back -- was worth its price when
        the schema was advisory and a near-miss value was routine. It is not
        worth it now: the provider constrains generation to the schema, so the
        surviving failure is a response cut off by `max_tokens`, and re-asking
        with the same budget reproduces it exactly while billing twice.
        """
        resolved = _validated_model(model or self._default_model)
        kwargs = self._request_kwargs(
            model=resolved,
            system=system,
            max_tokens=max_tokens,
            # None, not 0.0: `_request_kwargs` still applies the configured
            # default under the model gate. Structured calls are not exempt from
            # the deployment's sampling settings, only from the models that
            # reject them.
            temperature=None,
        )
        try:
            message = await self._send(
                self._client.messages.parse(
                    messages=[{"role": "user", "content": prompt}],
                    output_format=schema,
                    **kwargs,
                )
            )
        except PydanticValidationError as exc:
            # The SDK validates inside response parsing, so there is no `Message`
            # to read usage or `stop_reason` off. `_schema_detail` recovers the
            # important half of that from the validation error itself.
            raise LLMSchemaError(
                f"{schema.__name__} could not be produced: {_schema_detail(exc)}",
                schema=schema.__name__,
                attempts=1,
                details={"model": resolved},
                cause=exc,
            ) from exc

        value = message.parsed_output
        if value is None:
            # 200 with no JSON: a refusal, or content the SDK left unparsed.
            # A content outcome, not a transport one -- so it must not look like
            # something the caller should retry.
            raise LLMSchemaError(
                f"{schema.__name__} could not be produced: {_empty_detail(message)}",
                schema=schema.__name__,
                attempts=1,
                details={"model": resolved, "stop_reason": message.stop_reason},
            )
        return value, _response_of(message, model=resolved, text=_text_of(message))

    async def aclose(self) -> None:
        """Close the pooled client, if it is ours to close. Idempotent."""
        if self._owns_client:
            await self._client.close()

    # ------------------------------------------------------------ internals --

    def _request_kwargs(
        self,
        *,
        model: str,
        system: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        """The parameters shared by `create` and `parse`.

        Keys are *omitted* rather than set to `None`: the SDK forwards an
        explicit `None` as a JSON `null`, which several of these fields reject.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
        }
        if system:
            kwargs["system"] = system

        effective = temperature if temperature is not None else self._settings.temperature
        if _accepts_sampling(model):
            kwargs["temperature"] = effective

        if self._settings.effort is not None:
            # `budget_tokens` is deliberately absent: it was removed on the
            # current models and now answers 400. Effort is its replacement, and
            # unset means "the provider's own default" rather than a level we
            # guessed (`backend/core/config.py`, `LLMEffort`).
            kwargs["output_config"] = {"effort": self._settings.effort.value}
        return kwargs

    async def _send(self, pending: Awaitable[_T]) -> _T:
        """Await one SDK call and map every failure onto the taxonomy.

        Ordered most-specific-first, which is not cosmetic: `APITimeoutError`
        subclasses `APIConnectionError`, and every 4xx/5xx class subclasses
        `APIStatusError`, so a broader clause placed earlier silently swallows
        the distinction the caller branches on.
        """
        try:
            return await pending
        except anthropic.RateLimitError as exc:
            raise LLMRateLimited(
                "the LLM provider rate limited the request.",
                retry_after_seconds=_retry_after(exc.response.headers.get("retry-after")),
                provider_error_type=exc.type,
                request_id=exc.request_id,
                cause=exc,
            ) from exc
        except anthropic.APITimeoutError as exc:
            raise LLMTimeout(
                f"no response from the LLM provider within {self._settings.timeout_seconds}s "
                f"per attempt ({self._settings.max_retries} retries allowed).",
                cause=exc,
            ) from exc
        except anthropic.APIConnectionError as exc:
            # Transport-level: DNS, TLS, connection reset. Distinct from a
            # provider *response*, hence no provider_status.
            raise LLMError(
                f"transport failure talking to the LLM provider ({type(exc).__name__}).",
                cause=exc,
            ) from exc
        except (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        ) as exc:
            # A bad key, a revoked scope, a retired model id or a malformed
            # request. Permanent by construction: the router must not shed a
            # tier for these, because the smaller model fails identically.
            raise LLMError(
                f"the LLM provider rejected the request with HTTP {exc.status_code}; "
                "the credential, the model id or the request shape is wrong.",
                provider_status=exc.status_code,
                provider_error_type=exc.type,
                request_id=exc.request_id,
                cause=exc,
            ) from exc
        except anthropic.APIStatusError as exc:
            raise _status_error(exc) from exc
        except anthropic.APIError as exc:
            # No HTTP status to report: a response the SDK could not validate,
            # or a client-side failure it raised on our behalf.
            raise LLMError(
                f"the LLM provider client failed ({type(exc).__name__}).",
                cause=exc,
            ) from exc


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #


def _response_of(message: Message, *, model: str, text: str) -> LLMResponse:
    """Carry the usage block across, losing nothing.

    Cache *writes* are folded into `input_tokens` rather than dropped. The API
    reports them separately and they bill at a premium, so a provider that
    ignored the field would under-report spend on precisely the largest prompts.
    `LLMResponse` has no fourth counter to put them in, and losing tokens is
    worse than merging them. Cache *reads* stay separate: they are the saving,
    and `docs/observability.md` §8.2 tracks their share.
    """
    usage = message.usage
    request_id = getattr(message, "_request_id", None)
    if request_id is not None:
        # The one identifier Anthropic can trace on. Cheap here, unobtainable
        # afterwards.
        logger.debug("anthropic_request", request_id=request_id, model=message.model)

    return LLMResponse(
        text=text,
        model=str(message.model or model),
        input_tokens=usage.input_tokens + (usage.cache_creation_input_tokens or 0),
        output_tokens=usage.output_tokens,
        cached_tokens=usage.cache_read_input_tokens or 0,
        stop_reason=message.stop_reason,
    )


def _text_of(message: Message) -> str:
    """Concatenate the text blocks.

    Returns `""` rather than raising when there are none. A safety refusal
    answers 200 with `stop_reason="refusal"` and no text blocks, and a provider
    that raised on it would turn a *content* outcome the caller must handle into
    a transport error the caller will retry. Thinking blocks are skipped: they
    are reasoning, not the answer, and callers that concatenated them would put
    the model's scratchpad into a report.
    """
    return "".join(block.text for block in message.content if block.type == "text")


def _empty_detail(message: Message) -> str:
    """Why a 200 carried no parsed object. The `stop_reason` is the whole story."""
    if message.stop_reason == "max_tokens":
        return (
            "the response hit max_tokens before any output was produced; "
            "raise LLM_MAX_OUTPUT_TOKENS"
        )
    if message.stop_reason == "refusal":
        return "the model refused the request"
    return f"the model returned no JSON object (stop_reason={message.stop_reason!r})"


def _schema_detail(exc: PydanticValidationError, *, limit: int = 3) -> str:
    """Field paths and messages only -- never the offending values.

    The rejected value is model output derived from fetched content, and this
    string ends up in an exception that is logged and serialized
    (`docs/security-and-privacy.md`). The field path is what a human needs to
    fix a schema or a prompt anyway.

    `json_invalid` is special-cased because it is the diagnosis, not a symptom:
    generation was constrained to the schema, so text that is not even *JSON*
    means the object was cut off mid-emit, and the fix is a budget rather than a
    prompt. Saying so here is the only place that distinction can be made -- the
    `Message` carrying `stop_reason="max_tokens"` never reaches us.
    """
    errors = exc.errors()
    if errors and errors[0].get("type") == "json_invalid":
        return (
            "the model's output was not parseable JSON, which under constrained "
            "generation means it was cut off mid-object; raise LLM_MAX_OUTPUT_TOKENS"
        )
    parts = [
        f"{'.'.join(str(p) for p in err.get('loc', ())) or '<root>'}: {err.get('msg', '')}"
        for err in errors[:limit]
    ]
    if len(errors) > limit:
        parts.append(f"(+{len(errors) - limit} more)")
    return "; ".join(parts)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


def _status_error(exc: anthropic.APIStatusError) -> LLMError:
    """Map a status the named SDK classes do not cover.

    The provider's free-text `message` is deliberately left behind: providers
    echo the offending request back inside it, requests carry fetched content,
    and this object is serialized into logs and into HTTP responses
    (`docs/security-and-privacy.md`). `exc.type` is a closed vocabulary and is safe.
    """
    if exc.status_code in _TIMEOUT_STATUSES:
        return LLMTimeout(
            "the LLM provider timed out server-side.",
            provider_status=exc.status_code,
            provider_error_type=exc.type,
            request_id=exc.request_id,
            cause=exc,
        )
    return LLMError(
        f"the LLM provider returned HTTP {exc.status_code}.",
        provider_status=exc.status_code,
        provider_error_type=exc.type,
        request_id=exc.request_id,
        cause=exc,
    )


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        # Anthropic sends delta-seconds. An HTTP-date would parse to None here,
        # which is honest: the router then falls back to its own pacing rather
        # than acting on a number it invented.
        return None


# --------------------------------------------------------------------------- #
# Model ids
# --------------------------------------------------------------------------- #


def _validated_model(model: str) -> str:
    """Reject model ids that are known to be gone.

    Fails on the *retired* prefixes rather than allow-listing the three ids we
    ship with: an allowlist would reject the next model on the day it is
    released and turn a config change into a code change.
    """
    normalized = model.strip()
    if not normalized:
        raise ConfigurationError(
            "an empty model id reached the Anthropic provider; check LLM_MODEL_*.",
            details={"setting": "LLM_MODEL_PLANNER|LLM_MODEL_WORKER|LLM_MODEL_FAST"},
        )
    if normalized.startswith(_RETIRED_MODEL_PREFIXES):
        raise ConfigurationError(
            f"model {normalized!r} is retired and will answer 404. Use one of "
            "claude-opus-5, claude-sonnet-5, claude-haiku-4-5-20251001.",
            details={"model": normalized},
        )
    return normalized


def _accepts_sampling(model: str) -> bool:
    """Whether this model still accepts `temperature`. See the module docstring."""
    return not model.startswith(_SAMPLING_FREE_PREFIXES)
