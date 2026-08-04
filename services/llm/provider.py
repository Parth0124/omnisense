"""The pinned LLM contract -- one interface, any vendor.

Design Doc §15 makes the AI layer swappable. That is not aspiration: the model
market re-prices and re-ranks itself every few months, and a codebase that names
a vendor SDK in thirty places cannot follow it. So exactly two shapes are
allowed to cross into the rest of OmniSense -- `LLMProvider` and `LLMResponse`
-- and everything that needs a model takes the provider as a **constructor
argument**. Nothing constructs a provider for itself.

That rule buys three things:

- **Testability.** Every test in `agents/` and `services/` runs against
  `FakeLLMProvider` below. No network, no key, no recorded cassette, no flake.
- **Substitution.** Swapping Anthropic for Bedrock is one wiring change in the
  composition root, not a migration.
- **Honest accounting.** Usage is carried on the response, so cost is recorded
  where it is incurred (`docs/observability.md` §8.2) rather than estimated
  afterwards from a prompt we no longer have.

Why the error taxonomy lives here rather than per-provider: the *caller* decides
what to do about a failure -- an enrichment stage degrades, the router sheds a
tier, an agent parks its run -- and it cannot decide from an
`httpx.HTTPStatusError`. The classes below are exactly the distinctions that
change behaviour: "come back later with the same request" (`LLMRateLimited`),
"we never heard back" (`LLMTimeout`), "the prompt or schema is wrong"
(`LLMSchemaError`), and "the provider said no" (`LLMError`). Anything finer
belongs in `details`, not in a new class.

They derive from `ExternalServiceError`, whose docstring already claims the LLM
provider, so an LLM failure that reaches the API surfaces as a 502 without a
translation table.

One note on `structured()`: it returns the parsed model, which leaves the
signature nowhere to put token usage. That is a real hole in an interface whose
whole point is cost visibility, so providers may *additionally* implement
`MeteredLLMProvider` and hand usage back alongside the value. The router prefers
it, and counts a call it cannot meter rather than pretending it was free.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from backend.core.exceptions import ExternalServiceError

__all__ = [
    "BaseModelT",
    "FakeCall",
    "FakeLLMProvider",
    "LLMError",
    "LLMProvider",
    "LLMRateLimited",
    "LLMResponse",
    "LLMSchemaError",
    "LLMTimeout",
    "MeteredLLMProvider",
    "ScriptItem",
    "ScriptValue",
]

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One completion plus the accounting that comes with it.

    Frozen because it is handed to metrics emitters, span attributes and the run
    record; a mutable response is an invitation to "fix up" a token count
    downstream, after which two dashboards disagree about what a run cost.

    `cached_tokens` is provider-side prompt-cache *reads* -- tokens billed at
    roughly a tenth of list price. Kept separate from `input_tokens` because a
    falling cache-read share is a cost regression that no latency metric will
    reveal (`docs/observability.md` §8.2).
    """

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    stop_reason: str | None = None

    @property
    def billable_tokens(self) -> int:
        """Tokens processed or generated at full price on this call.

        Deliberately excludes `cached_tokens`: counting cache reads at full rate
        would make prompt caching look like it *increased* spend.
        """
        return self.input_tokens + self.output_tokens

    @property
    def total_tokens(self) -> int:
        """Everything the model saw or produced. The context-pressure number."""
        return self.input_tokens + self.output_tokens + self.cached_tokens

    @property
    def truncated(self) -> bool:
        """Whether the provider stopped because it ran out of output budget.

        Worth a first-class name: a `max_tokens` stop is a *configuration*
        defect, and it presents identically to a model-quality problem unless
        something checks for it (`docs/observability.md` §5).
        """
        return self.stop_reason == "max_tokens"


# --------------------------------------------------------------------------- #
# Failures
# --------------------------------------------------------------------------- #


class LLMError(ExternalServiceError):
    """The provider failed in a way we cannot paper over.

    `provider_status`, `provider_error_type` and `request_id` ride in `details`
    so an incident can be correlated with the vendor's own logs. The provider's
    free-text error *message* is deliberately **not** carried: providers echo
    the offending request back inside it, requests carry fetched content, and
    this object is serialized into logs and into HTTP responses
    (`docs/security-and-privacy.md`). The error *type* is a closed vocabulary
    and is safe.
    """

    code = "llm_error"
    default_message = "The LLM provider returned an error."

    def __init__(
        self,
        message: str | None = None,
        *,
        provider_status: int | None = None,
        provider_error_type: str | None = None,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.provider_status = provider_status
        self.provider_error_type = provider_error_type
        self.request_id = request_id
        for key, value in (
            ("provider_status", provider_status),
            ("provider_error_type", provider_error_type),
            ("request_id", request_id),
        ):
            if value is not None:
                self.details.setdefault(key, value)


class LLMTimeout(LLMError):  # noqa: N818 -- pinned interface name, see below
    """No response arrived inside `LLM_TIMEOUT_SECONDS`.

    The missing `Error` suffix is deliberate: this name and `LLMRateLimited` are
    part of the pinned contract every module in this layer imports, so renaming
    them for the convention would break more than the convention protects.

    504 rather than 502, and a class of its own, because the request may well
    have completed on the provider's side and been billed. A retry policy that
    cannot tell "never happened" from "happened, we stopped listening" will
    double-spend.
    """

    status_code = 504
    code = "llm_timeout"
    default_message = "The LLM provider did not respond in time."


class LLMRateLimited(LLMError):  # noqa: N818 -- pinned interface name
    """The provider refused for quota reasons (HTTP 429).

    503, not 429. A 429 to *our* caller says "you are asking too often", which
    is false and actively unhelpful: the client then backs off a budget that is
    not the constrained one, while the real remedy is on our side (shed a tier,
    wait out the window). `retry_after_seconds` is the provider's own
    instruction and is what `services/llm/router.py` and the agent runtime pace
    against.
    """

    status_code = 503
    code = "llm_rate_limited"
    default_message = "The LLM provider is rate limiting us."

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_seconds: float | None = None,
        provider_status: int | None = 429,
        provider_error_type: str | None = None,
        request_id: str | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            message,
            provider_status=provider_status,
            provider_error_type=provider_error_type,
            request_id=request_id,
            details=details,
            cause=cause,
        )
        self.retry_after_seconds = retry_after_seconds
        if retry_after_seconds is not None:
            self.details.setdefault("retry_after_seconds", retry_after_seconds)


class LLMSchemaError(LLMError):
    """Structured output failed validation, including after the retry.

    A distinct class because a repeated schema failure is a *different* defect
    from a provider outage and demands a different response: the prompt or the
    schema is wrong, retrying forever will not fix it, and it must be countable
    on its own (`docs/observability.md` §5 -- "schema-validation pass/fail,
    retry count"). Folding it into `LLMError` buries a prompt regression inside
    the provider-reliability metric, where nobody will look for it.
    """

    code = "llm_schema_error"
    default_message = "The model did not produce output matching the schema."

    def __init__(
        self,
        message: str | None = None,
        *,
        schema: str | None = None,
        attempts: int = 1,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.schema = schema
        self.attempts = attempts
        self.details.setdefault("attempts", attempts)
        if schema is not None:
            self.details.setdefault("schema", schema)


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #


@runtime_checkable
class LLMProvider(Protocol):
    """What every model backend must offer. Pinned -- do not widen.

    Three methods, because three is what the system actually needs: free text,
    schema-constrained output, and an orderly shutdown. Streaming is absent on
    purpose -- nothing in Phase 1 renders tokens as they arrive, and putting it
    in the protocol would force every fake and every alternative backend to
    implement it for a caller that does not exist yet.

    `model=None` means "the provider's configured default". Callers that care
    about tiering go through `services/llm/router.py` instead of passing model
    ids around by hand.
    """

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Free-text completion."""
        ...

    async def structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModelT],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> BaseModelT:
        """Completion constrained to `schema`, returned already validated."""
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP client. Idempotent."""
        ...


@runtime_checkable
class MeteredLLMProvider(LLMProvider, Protocol):
    """Optional extension: hand back usage for a structured call.

    `structured()` returns the parsed model, so its usage has nowhere to go. For
    a system whose per-run cost is the metric that decides viability
    (`docs/observability.md` §8), silently losing the token count of every
    planning call is not acceptable -- and neither is changing the pinned
    signature, which every other module in this layer already codes against.

    So the seam is explicit. Providers that can report usage implement this and
    `ModelRouter` prefers it; a provider that cannot still works, and the router
    records an *unmetered* call rather than a free one, so the gap shows up in
    the run record instead of disappearing into the total.
    """

    async def structured_metered(
        self,
        *,
        prompt: str,
        schema: type[BaseModelT],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[BaseModelT, LLMResponse]:
        """As `structured()`, plus the usage the call incurred.

        `LLMResponse.text` carries the raw JSON the model emitted, which is what
        a trace needs in order to explain a validation failure after the fact.
        """
        ...


# --------------------------------------------------------------------------- #
# The fake every other test in this layer uses
# --------------------------------------------------------------------------- #

ScriptValue = str | LLMResponse | BaseModel | Mapping[str, Any]
"""A scripted answer that is *returned*. See `ScriptItem` for the raised kind."""

ScriptItem = ScriptValue | Exception
"""One scripted answer.

`str` is text (or raw JSON for a structured call), `LLMResponse` pins the token
counts a test wants to assert on, a `BaseModel` or mapping is a structured
result, and an `Exception` is *raised* -- which is how a test scripts "rate
limited, then fine" without patching anything.
"""


@dataclass(frozen=True, slots=True)
class FakeCall:
    """One recorded call. Tests assert on these rather than on mock internals."""

    kind: str
    prompt: str
    system: str | None = None
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    schema: str | None = None


class FakeLLMProvider:
    """Scripted, deterministic, offline `LLMProvider`.

    It lives in the production package rather than under `tests/` because every
    other module in Layer 3 needs one -- the router, the agents, the enrichment
    stages -- and a fake that lives in a test module gets copy-pasted into four
    slightly-different versions the moment a second suite wants it.

    Exhausting the script **raises** rather than returning a bland default. A
    subject that made three calls where the author scripted two has changed
    behaviour, and a silent placeholder would let that pass as green. `default=`
    exists for the tests that genuinely do not care how many calls happen.
    """

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        default: ScriptItem | None = None,
        model: str = "fake-model-1",
    ) -> None:
        self._script: list[ScriptItem] = list(script or ())
        self._default = default
        self.model = model
        self.calls: list[FakeCall] = []
        self.closed = False

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
        self.calls.append(
            FakeCall(
                kind="complete",
                prompt=prompt,
                system=system,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        item = self._next()
        if isinstance(item, LLMResponse):
            return item
        text = item if isinstance(item, str) else _to_json(item)
        return self._synthesize(text, prompt=prompt, system=system, model=model)

    async def structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModelT],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> BaseModelT:
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
        self.calls.append(
            FakeCall(
                kind="structured",
                prompt=prompt,
                system=system,
                model=model,
                max_tokens=max_tokens,
                schema=schema.__name__,
            )
        )
        item = self._next()
        value = self._coerce(item, schema)
        if isinstance(item, LLMResponse):
            return value, item
        usage = self._synthesize(value.model_dump_json(), prompt=prompt, system=system, model=model)
        return value, usage

    async def aclose(self) -> None:
        self.closed = True

    # ------------------------------------------------------------ internals --

    def _next(self) -> ScriptValue:
        """Pop the next scripted item, raising it when it is an exception.

        The narrowed return type is the point: everything downstream knows an
        exception can never reach it, so no caller carries a defensive branch
        for a case `raise` already handled.
        """
        if self._script:
            item = self._script.pop(0)
        elif self._default is not None:
            item = self._default
        else:
            raise LLMError(
                f"FakeLLMProvider script exhausted after {len(self.calls)} call(s); "
                "the subject made more calls than the test scripted. Add items to "
                "`script=`, or pass `default=` when the call count is not the point."
            )
        if isinstance(item, Exception):
            raise item
        return item

    def _coerce(self, item: ScriptValue, schema: type[BaseModelT]) -> BaseModelT:
        """Turn a scripted item into a validated instance of `schema`.

        A scripted value that does not fit surfaces as `LLMSchemaError` -- the
        same class the real provider raises -- so a caller's failure path can be
        exercised without the test knowing that the fake used Pydantic to get
        there.
        """
        if isinstance(item, schema):
            return item
        try:
            if isinstance(item, LLMResponse):
                return schema.model_validate_json(item.text)
            if isinstance(item, str):
                return schema.model_validate_json(item)
            if isinstance(item, BaseModel):
                return schema.model_validate(item.model_dump())
            return schema.model_validate(dict(item))
        except (PydanticValidationError, ValueError) as exc:
            raise LLMSchemaError(
                f"scripted response does not satisfy {schema.__name__}.",
                schema=schema.__name__,
                attempts=1,
                cause=exc,
            ) from exc

    def _synthesize(
        self,
        text: str,
        *,
        prompt: str,
        system: str | None,
        model: str | None,
    ) -> LLMResponse:
        """Invent plausible, *deterministic* token counts.

        Word counts rather than a fixed constant: a budget-guard test needs
        spend to grow with what it sends, and a constant would let a broken
        accumulator pass.
        """
        return LLMResponse(
            text=text,
            model=model or self.model,
            input_tokens=_word_count(prompt) + _word_count(system or ""),
            output_tokens=_word_count(text),
            stop_reason="end_turn",
        )


def _word_count(text: str) -> int:
    return len(text.split())


def _to_json(item: ScriptValue) -> str:
    if isinstance(item, BaseModel):
        return item.model_dump_json()
    if isinstance(item, Mapping):
        return json.dumps(dict(item), sort_keys=True)
    return str(item)
