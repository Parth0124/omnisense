"""An OpenAI chat-completions provider. Covers OpenRouter, OpenAI, Ollama, LiteLLM.

`services/llm/anthropic_provider.py` speaks Anthropic's Messages API. This speaks
the OpenAI chat-completions shape, which is what almost everything else in the
ecosystem exposes -- OpenRouter, vLLM, Ollama, LiteLLM, Azure OpenAI and OpenAI
itself all accept the same request body at some URL. One provider therefore
covers `LLM_PROVIDER` values `openai`, `ollama` and `litellm`, and the difference
between them is entirely `LLM_BASE_URL`.

**Written against `httpx` rather than the `openai` SDK.** The SDK would be a new
runtime dependency to send one well-documented JSON body, and it carries its own
retry and timeout behaviour that would sit underneath ours and interact with it
-- the same nested-retry multiplication that cost 15 seconds on the graph read
path. What is actually needed here is one POST, careful error classification and
honest token accounting, and that is what this is.

**Structured output degrades in three steps, and the order matters.** Not every
model behind an OpenAI-compatible endpoint supports strict schemas, and the ones
that do not fail in different ways:

1. `response_format: json_schema` with `strict: true` -- generation is
   constrained, so a near-miss value cannot occur. Preferred, and what
   `anthropic_provider.py` gets natively.
2. `response_format: json_object` -- valid JSON is guaranteed, the shape is not.
   The schema goes in the system prompt and Pydantic checks the result.
3. Neither -- the schema goes in the prompt and the reply is scraped for JSON.

The provider *remembers* which step worked for a given model, so it pays the
discovery cost once rather than on every call. Without that memory a deployment
on a model that rejects `json_schema` would burn a failed request per structured
call, forever, and the failures would look like a model-quality problem.

**Validation is always local.** Even under a strict schema: "constrained" is a
provider claim, and a response truncated by a token limit ends mid-object and
satisfies nothing. A failure at that point is rare and actionable rather than
routine, which is exactly the property that makes it worth distinguishing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Final

import httpx
from pydantic import ValidationError as PydanticValidationError

from backend.core.config import LLMSettings, get_settings
from backend.core.logging import get_logger
from services.llm.provider import (
    BaseModelT,
    LLMError,
    LLMRateLimited,
    LLMResponse,
    LLMTimeout,
)

__all__ = [
    "OPENROUTER_BASE_URL",
    "OpenAICompatibleProvider",
    "StructuredMode",
]

logger = get_logger(__name__)

OPENROUTER_BASE_URL: Final = "https://openrouter.ai/api/v1"
"""The default when no base URL is configured.

OpenRouter rather than OpenAI, because a deployment that sets
`LLM_PROVIDER=openai` without a base URL is far more likely to be pointing at an
aggregator than at OpenAI direct -- and if it is OpenAI direct, that is one
setting away and the error message says so.
"""

_JSON_BLOCK: Final = re.compile(r"\{.*\}", re.DOTALL)


class StructuredMode:
    """Which structured-output mechanism a model was found to support.

    Cached per model id rather than per provider. One OpenRouter key reaches
    dozens of models with different capabilities, so a provider-wide flag would
    downgrade every model to the weakest one the deployment happened to try
    first.
    """

    SCHEMA = "json_schema"
    OBJECT = "json_object"
    PROMPT = "prompt"


class OpenAICompatibleProvider:
    """Chat completions over any OpenAI-shaped endpoint.

    Satisfies `services.llm.provider.LLMProvider` and, through
    `structured_metered`, `MeteredLLMProvider` -- so `agents/base.py` can bill a
    structured call without a second round trip.
    """

    def __init__(
        self,
        *,
        settings: LLMSettings | None = None,
        client: httpx.AsyncClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        referer: str | None = None,
        app_title: str = "OmniSense",
    ) -> None:
        self._settings = settings or get_settings().llm
        self._base_url = (base_url or self._settings.base_url or OPENROUTER_BASE_URL).rstrip("/")

        resolved_key = api_key or (
            self._settings.anthropic_api_key.get_secret_value()
            if self._settings.anthropic_api_key
            else None
        )
        if not resolved_key:
            raise LLMError(
                "No API key. Set OPENROUTER_API_KEY (or LLM_API_KEY) in .env. "
                f"The provider is pointed at {self._base_url}."
            )
        self._api_key = resolved_key

        # OpenRouter attributes usage to an app when these are present and, more
        # practically, shows the app name in its dashboard -- which is the
        # difference between a spend line you can attribute and one you cannot.
        self._referer = referer
        self._app_title = app_title

        self._client = client
        self._owns_client = client is None
        self._modes: dict[str, str] = {}

    # ------------------------------------------------------------- plumbing --

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-Title": self._app_title,
            }
            if self._referer:
                headers["HTTP-Referer"] = self._referer
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers=headers,
                timeout=httpx.Timeout(self._settings.timeout_seconds),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        """One completion request, with failures classified into our taxonomy.

        The classification decides what the caller does: `LLMRateLimited` is
        parked and retried, `LLMTimeout` is retried immediately, and a plain
        `LLMError` stops. Collapsing them would either loop on a revoked key or
        abandon a run over a blip.
        """
        try:
            response = await self._http().post("/chat/completions", json=body)
        except httpx.TimeoutException as error:
            raise LLMTimeout(
                f"{self._base_url} timed out after {self._settings.timeout_seconds}s"
            ) from error
        except httpx.HTTPError as error:
            raise LLMError(f"request to {self._base_url} failed: {error}") from error

        if response.status_code == 401:
            raise LLMError(
                "The API key was rejected. For OpenRouter, check the key at "
                "openrouter.ai/keys and that it has credit."
            )
        if response.status_code == 402:
            # OpenRouter's own code for an exhausted balance. Distinguished
            # because "out of credit" and "rate limited" need different actions
            # and both otherwise read as 'the model stopped answering'.
            raise LLMError(
                "Out of credit. OpenRouter returns 402 when the account balance "
                "is exhausted -- top up at openrouter.ai/credits."
            )
        if response.status_code == 429:
            raise LLMRateLimited(
                f"rate limited; retry-after={response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise LLMError(f"{self._base_url} returned {response.status_code}")
        if response.status_code >= 400:
            raise LLMError(
                f"{self._base_url} rejected the request ({response.status_code}): "
                f"{response.text[:300]}"
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise LLMError("the endpoint returned a non-JSON body") from error

        # OpenRouter reports upstream provider failures inside a 200 body, which
        # a status check alone would read as success and then fail on a missing
        # `choices` key several frames away.
        if isinstance(payload.get("error"), dict):
            message = payload["error"].get("message", "unknown upstream error")
            raise LLMError(f"upstream model error: {message}")
        return payload

    def _to_response(self, payload: dict[str, Any]) -> LLMResponse:
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError("the endpoint returned no choices")
        message = choices[0].get("message") or {}
        usage = payload.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return LLMResponse(
            text=str(message.get("content") or ""),
            model=str(payload.get("model") or "unknown"),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            # Reported by providers that support prompt caching. Absent is 0
            # rather than an error: most models behind an aggregator do not.
            cached_tokens=int(details.get("cached_tokens") or 0),
            stop_reason=choices[0].get("finish_reason"),
        )

    def _body(
        self, *, prompt: str, system: str | None, model: str | None, max_tokens: int | None
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": model or self._settings.model_worker,
            "messages": messages,
            "max_tokens": max_tokens or self._settings.max_output_tokens,
        }
        # Sent only when non-default. The current Claude generation returns 400
        # for `temperature` at all, and those models are reachable *through*
        # OpenRouter -- so a provider that always sent it would fail on exactly
        # the models this system is designed around.
        if self._settings.temperature:
            body["temperature"] = self._settings.temperature
        return body

    # --------------------------------------------------------------- public --

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        body = self._body(prompt=prompt, system=system, model=model, max_tokens=max_tokens)
        if temperature is not None:
            body["temperature"] = temperature
        return self._to_response(await self._post(body))

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
            prompt=prompt, schema=schema, system=system, model=model, max_tokens=max_tokens
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
        """Schema-constrained output plus its usage, degrading as needed.

        Returns the usage alongside the value so `agents/base.py` can bill the
        call without a second request -- which is what `MeteredLLMProvider`
        exists for.

        One retry, and only one. A model that produced unparseable output twice
        against the same prompt will produce it a third time; the second attempt
        is worth paying for because a truncated or fenced reply is genuinely
        transient, and a third is just spending money to confirm a bad prompt.
        """
        resolved_model = model or self._settings.model_worker
        mode = self._modes.get(resolved_model, StructuredMode.SCHEMA)

        for attempt in (1, 2):
            body = self._body(
                prompt=prompt,
                system=self._system_for(mode, system, schema),
                model=resolved_model,
                max_tokens=max_tokens,
            )
            self._apply_format(body, mode, schema)

            try:
                payload = await self._post(body)
            except LLMError as error:
                downgraded = self._downgrade(mode)
                # A rejected `response_format` looks like an ordinary 400. If
                # there is a weaker mode left, try it once rather than failing --
                # this is the discovery path, and it runs at most twice per model
                # for the life of the process.
                if downgraded and _looks_like_format_rejection(error):
                    logger.info(
                        "llm.structured_mode_downgraded",
                        model=resolved_model,
                        from_mode=mode,
                        to_mode=downgraded,
                        reason=str(error)[:200],
                    )
                    self._modes[resolved_model] = downgraded
                    mode = downgraded
                    continue
                raise

            response = self._to_response(payload)
            try:
                value = schema.model_validate_json(_extract_json(response.text))
            except (PydanticValidationError, ValueError) as error:
                if attempt == 1:
                    logger.info(
                        "llm.structured_retry",
                        model=resolved_model,
                        mode=mode,
                        error=type(error).__name__,
                    )
                    continue
                raise LLMError(
                    f"{resolved_model} did not produce valid {schema.__name__} after "
                    f"two attempts in {mode} mode: {error}"
                ) from error

            self._modes[resolved_model] = mode
            return value, response

        raise LLMError("unreachable: the structured loop always returns or raises")

    # -------------------------------------------------------------- helpers --

    def _apply_format(self, body: dict[str, Any], mode: str, schema: type[BaseModelT]) -> None:
        if mode == StructuredMode.SCHEMA:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": _strict_schema(schema),
                },
            }
        elif mode == StructuredMode.OBJECT:
            body["response_format"] = {"type": "json_object"}

    def _system_for(self, mode: str, system: str | None, schema: type[BaseModelT]) -> str:
        """Append the schema to the system prompt in the weaker modes.

        Not in `json_schema` mode: generation is already constrained there, and
        restating the schema would spend input tokens on every call to repeat
        something the decoder is enforcing.
        """
        if mode == StructuredMode.SCHEMA:
            return system or ""
        instruction = (
            "Respond with a single JSON object matching this schema exactly. "
            "No prose, no markdown fence.\n\n"
            f"{json.dumps(schema.model_json_schema(), separators=(',', ':'))}"
        )
        return f"{system}\n\n{instruction}" if system else instruction

    @staticmethod
    def _downgrade(mode: str) -> str | None:
        return {
            StructuredMode.SCHEMA: StructuredMode.OBJECT,
            StructuredMode.OBJECT: StructuredMode.PROMPT,
        }.get(mode)


def _looks_like_format_rejection(error: Exception) -> bool:
    """Whether a 400 is the endpoint refusing `response_format`.

    Matched on message text, which is coarse -- but the alternative is treating
    every 400 as a capability problem and silently downgrading a model that
    actually rejected a malformed prompt, which would hide a real bug behind a
    quieter mode.
    """
    text = str(error).casefold()
    return any(
        marker in text
        for marker in ("response_format", "json_schema", "not supported", "unsupported")
    )


def _extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may be wrapped.

    Needed only in the weaker modes, where the model may fence the object or
    prefix it with a sentence. Takes the outermost braces rather than the first
    balanced object: a reply containing prose *and* JSON has the object last, and
    a non-greedy match would stop at the first nested `}`.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        return stripped
    match = _JSON_BLOCK.search(stripped)
    if match is None:
        raise ValueError("no JSON object found in the reply")
    return match.group(0)


_UNSUPPORTED_BOUNDS: Final[frozenset[str]] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
    }
)
"""JSON Schema keywords stripped before a schema is sent to a provider.

Every one is a *bound* rather than a *shape*, and no provider is obliged to
support them. Anthropic refuses the whole request when it sees one -- the entire
call fails, so the cost of leaving them in is total rather than partial.

**Dropping them loses nothing that was ever enforced here.** These express
"confidence is between 0 and 1", and the place that actually holds a model to
that is `BaseModel.model_validate` when the response is parsed back -- which runs
regardless of what the schema said, and is the only enforcement that was ever
real. What is lost is a *hint* to the model, and a hint is worth precisely
nothing when including it means the request is rejected outright.

`enum`, `const`, `format` and `description` are deliberately **not** in this set:
they describe what a value may *be* rather than how large it may get, they are
what actually steer the model toward a valid answer, and both providers accept
them.
"""


def _strict_schema(schema: type[BaseModelT]) -> dict[str, Any]:
    """A JSON Schema acceptable to strict structured-output implementations.

    Strict mode rejects a schema that permits unlisted properties, so every
    object needs `additionalProperties: false` and must list every property as
    required. Pydantic emits neither by default -- optional fields are simply
    absent from `required` -- so a schema handed over unmodified is refused with
    a message about the schema rather than about the model, which sends the
    reader in the wrong direction.
    """
    raw = schema.model_json_schema()

    def tighten(node: Any) -> Any:
        if isinstance(node, dict):
            tightened = {key: tighten(value) for key, value in node.items()}

            # A `$ref` may carry no siblings. Pydantic emits
            # `{"$ref": "#/$defs/Foo", "default": "bar"}` for any field whose
            # type is a model or enum and which has a default, and OpenAI
            # refuses it outright: "$ref cannot have keywords {'default'}".
            # Dropping the siblings is safe rather than lossy -- strict mode
            # marks every property required below, so a default can never be
            # consulted; and `description` on a `$ref` is documentation the
            # target already carries.
            if "$ref" in tightened and len(tightened) > 1:
                tightened = {"$ref": tightened["$ref"]}

            # Range and length bounds are not universally supported, and
            # Anthropic rejects the entire request rather than ignoring them:
            #   "For 'array' type, property 'maxItems' is not supported"
            #   "For 'number' type, properties maximum, minimum are not supported"
            # They arrive from ordinary `Field(ge=0.0, le=1.0)` and
            # `Field(max_length=10)` declarations, so they appear on almost every
            # agent schema without anyone opting into them.
            #
            # Moved into the description rather than simply deleted, because the
            # first version of this deleted them and broke the Report agent: with
            # `minItems: 1` gone, the model had no reason to know `sections` may
            # not be empty, returned `[]`, and Pydantic then rejected every
            # attempt. The bound was doing real work as a *hint* even though it
            # was never the thing that enforced anything. Prose survives both
            # providers' validators; the keyword does not.
            dropped = {k: tightened.pop(k) for k in _UNSUPPORTED_BOUNDS if k in tightened}
            if dropped:
                spelled = ", ".join(f"{key} {value}" for key, value in sorted(dropped.items()))
                existing = tightened.get("description")
                tightened["description"] = (
                    f"{existing} (constraints: {spelled})"
                    if existing
                    else f"Constraints: {spelled}."
                )

            if tightened.get("type") == "object" and "properties" in tightened:
                tightened["additionalProperties"] = False
                # Strict mode requires every property listed. Optionality is
                # then expressed by the field's own type permitting null, which
                # is how Pydantic models it anyway.
                tightened["required"] = sorted(tightened["properties"])
            return tightened
        if isinstance(node, list):
            return [tighten(item) for item in node]
        return node

    return tighten(raw)
