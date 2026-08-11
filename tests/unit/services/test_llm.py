"""Unit tests for the model-agnostic AI layer (Design Doc §15).

This layer is where money is spent and where a wrong answer is cheapest to
produce, so the tests target the mistakes that are invisible in review and
expensive in production:

- **accounting that quietly under-reports.** Cache-creation tokens dropped, a
  cache hit charged as if it were a call, usage read off something other than
  the response that produced it. Each of those makes the cost dashboard wrong
  in the direction nobody investigates (`docs/observability.md` §8).
- **a cache key missing an input.** Serving an opus answer to a haiku request is
  not a stale read, it is a different answer at a different price, and nothing
  downstream can tell.
- **shedding that sheds work instead of quality.** `docs/architecture.md` §7.2
  requires the router to answer with a smaller model rather than fail the step,
  and to say so afterwards.
- **sampling parameters sent to a model that rejects them.** `LLM_TEMPERATURE`
  defaults to `0.0` and is therefore always present; forwarding it to
  `claude-opus-5` would 400 every planner call in this deployment.
- **a vector width mismatch discovered at the Qdrant upsert**, which is after
  the spend and after the stage reported success.

**How the Anthropic tests are wired.** `AnthropicProvider` drives the official
`anthropic` SDK, so these tests drive the real SDK too -- the same client class,
the same request serialization, the same status-to-exception mapping, the same
response models -- with only the socket replaced by `httpx.MockTransport`.
Stubbing the SDK's own methods instead would assert that we call a shape we
invented, which is precisely the failure the SDK port removed: a test that stays
green against a wire format the vendor no longer speaks. Assertions therefore
run on the **serialized request**, not on the keyword arguments handed to the
client, because everything interesting -- a parameter renamed, nested one level
deeper, or dropped -- happens between those two points.

Everything runs offline. The Anthropic transport is a stub that raises rather
than dials; embeddings still go through `respx`, since they speak a hand-rolled
OpenAI-compatible protocol with no vendor SDK behind it; Redis is `fakeredis`.
No key, no network, no clock dependency.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import anthropic
import httpx
import pytest
import respx
from fakeredis import aioredis
from pydantic import BaseModel

from backend.core.config import EmbeddingSettings, LLMSettings
from backend.core.exceptions import ConfigurationError, ExternalServiceError
from services.llm.anthropic_provider import AnthropicProvider
from services.llm.cache import CACHE_PREFIX, CompletionCache, completion_cache_key
from services.llm.embeddings import (
    EmbeddingDimensionMismatch,
    EmbeddingProvider,
    FakeEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from services.llm.openai_compatible import _UNSUPPORTED_BOUNDS, _strict_schema
from services.llm.provider import (
    BaseModelT,
    FakeLLMProvider,
    LLMError,
    LLMProvider,
    LLMRateLimited,
    LLMResponse,
    LLMSchemaError,
    LLMTimeout,
    MeteredLLMProvider,
)
from services.llm.router import (
    ModelRouter,
    ModelTier,
    RunBudget,
    TokenBudgetExceeded,
)

pytestmark = pytest.mark.unit

BASE_URL = "https://api.anthropic.test"

EMBED_BASE_URL = "https://embeddings.test/v1"
EMBED_URL = f"{EMBED_BASE_URL}/embeddings"

PLANNER = "claude-opus-5"
WORKER = "claude-sonnet-5"
FAST = "claude-haiku-4-5-20251001"


class Plan(BaseModel):
    """Return the investigation plan."""

    steps: list[str]
    confidence: float


PLAN_JSON = '{"steps": ["a"], "confidence": 0.9}'
"""A reply that satisfies `Plan`.

Kept as the literal text the model would emit, rather than serialized from a
`Plan`, so the constrained-decoding path is exercised against a string the
provider has to parse -- which is where truncation and near-miss values live.
"""


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def llm_settings(
    *,
    api_key: str | None = "k-secret",
    base_url: str | None = BASE_URL,
    planner: str = PLANNER,
    worker: str = WORKER,
    fast: str = FAST,
    max_output_tokens: int = 1024,
    timeout_seconds: int = 30,
    max_retries: int = 0,
    temperature: float = 0.0,
    effort: str | None = None,
    cache_enabled: bool = True,
) -> LLMSettings:
    """Fully-pinned settings. Every field the assertions touch is explicit.

    Constructed with the **environment aliases**, not the field names. Every
    field on `LLMSettings` carries an explicit `alias` and the model is
    configured `extra="ignore"`, so `LLMSettings(model_planner=...)` is accepted
    and then silently discarded -- the object comes back holding the default.
    A test built that way passes against settings it never actually set.

    `_env_file=None` plus explicit values means a developer's exported
    `ANTHROPIC_API_KEY` or a stray `.env` cannot change what these tests assert.

    `max_retries` defaults to **0** here while the shipped default is 3. The
    SDK's retry policy covers 429 and 5xx, so a failure-mapping test would
    otherwise issue four requests and assert the mapping of the last one; the
    tests that care about retrying ask for it explicitly.
    """
    return LLMSettings(
        _env_file=None,
        ANTHROPIC_API_KEY=api_key,
        LLM_BASE_URL=base_url,
        LLM_MODEL_PLANNER=planner,
        LLM_MODEL_WORKER=worker,
        LLM_MODEL_FAST=fast,
        LLM_MAX_OUTPUT_TOKENS=max_output_tokens,
        LLM_TIMEOUT_SECONDS=timeout_seconds,
        LLM_MAX_RETRIES=max_retries,
        LLM_TEMPERATURE=temperature,
        LLM_EFFORT=effort,
        LLM_CACHE_ENABLED=cache_enabled,
    )


def embedding_settings(**overrides: Any) -> EmbeddingSettings:
    values: dict[str, Any] = {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "api_key": "k-embed",
        "base_url": EMBED_BASE_URL,
        "dimensions": 4,
        "batch_size": 2,
    }
    values.update(overrides)
    return EmbeddingSettings(_env_file=None, **values)


def message(
    *,
    text: str = "ok",
    usage: dict[str, int] | None = None,
    stop_reason: str | None = "end_turn",
    model: str = WORKER,
    content: list[dict[str, Any]] | None = None,
    request_id: str = "req_stub",
) -> httpx.Response:
    """A 200 shaped exactly like the Messages API reply the SDK parses.

    Returned as an `httpx.Response` rather than a dict because the SDK's own
    response models do the parsing: a body this suite invents but the vendor
    would never send fails here, not silently in production.
    """
    return httpx.Response(
        200,
        headers={"request-id": request_id},
        json={
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}] if content is None else content,
            "stop_reason": stop_reason,
            "usage": usage or {"input_tokens": 10, "output_tokens": 5},
        },
    )


def error_response(
    status: int,
    *,
    error_type: str = "invalid_request_error",
    error_message: str = "no",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """A non-2xx in the provider's documented error envelope.

    The envelope matters: the SDK reads `error.type` out of it and puts it on
    the exception, which is the one piece of provider error detail this codebase
    is willing to carry (`docs/security-and-privacy.md`).
    """
    return httpx.Response(
        status,
        headers=headers or {},
        json={"type": "error", "error": {"type": error_type, "message": error_message}},
    )


class Transcript:
    """The stub socket, plus every request that reached it.

    Replies are consumed in order and the final one repeats, so a single reply
    serves any number of attempts -- which is what a retry test needs and what
    keeps the common case to one argument.
    """

    def __init__(self, replies: Sequence[httpx.Response | Exception]) -> None:
        self._replies = list(replies) or [message()]
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        # MockTransport hands over a request whose body has not been consumed;
        # reading it here is what makes `.content` available to assertions.
        request.read()
        self.requests.append(request)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def payload(self, index: int = 0) -> dict[str, Any]:
        """The JSON body the SDK actually serialized."""
        return json.loads(self.requests[index].content)

    def headers(self, index: int = 0) -> httpx.Headers:
        return self.requests[index].headers


@asynccontextmanager
async def anthropic_stub(
    *replies: httpx.Response | Exception, **overrides: Any
) -> AsyncIterator[tuple[AnthropicProvider, Transcript]]:
    """An `AnthropicProvider` on a real SDK client whose transport is a stub.

    Injecting the client means `aclose()` deliberately will not close it -- that
    is the ownership rule under test elsewhere -- so this closes it here, and a
    leaked pool cannot quietly accumulate across the suite.
    """
    transcript = Transcript(replies)
    settings = llm_settings(**overrides)
    client = anthropic.AsyncAnthropic(
        api_key="k-secret",
        base_url=BASE_URL,
        max_retries=settings.max_retries,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transcript)),
    )
    provider = AnthropicProvider(settings=settings, client=client)
    try:
        yield provider, transcript
    finally:
        await provider.aclose()
        await client.close()


def embedding_body(*rows: list[float], first_index: int = 0) -> dict[str, Any]:
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": first_index + i, "embedding": row}
            for i, row in enumerate(rows)
        ],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }


def request_json(route: respx.Route, index: int = 0) -> dict[str, Any]:
    return json.loads(route.calls[index].request.content)


def embedding_provider(**overrides: Any) -> OpenAICompatibleEmbeddingProvider:
    return OpenAICompatibleEmbeddingProvider(settings=embedding_settings(**overrides))


@pytest.fixture
async def redis() -> Any:
    client = aioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


class ExplodingRedis:
    """A client that fails every command, and records that it was reached."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or RuntimeError("redis is down")
        self.calls = 0

    async def get(self, name: str) -> Any:
        self.calls += 1
        raise self.error

    async def set(self, name: str, value: str, ex: int | None = None) -> Any:
        self.calls += 1
        raise self.error


class BareProvider:
    """A provider that satisfies `LLMProvider` and nothing more.

    Exists to prove the router degrades gracefully for a backend that cannot
    report usage on structured calls, rather than counting those calls as free.
    """

    def __init__(self) -> None:
        self.models: list[str | None] = []

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.models.append(model)
        return LLMResponse(text="ok", model=model or "bare", input_tokens=3, output_tokens=2)

    async def structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModelT],
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> BaseModelT:
        self.models.append(model)
        return schema.model_validate({"steps": ["a"], "confidence": 0.5})

    async def aclose(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# provider.py
# --------------------------------------------------------------------------- #


class TestLLMResponseAccounting:
    """The response is the only place a call's cost is ever recorded."""

    def test_cache_reads_are_excluded_from_billable_tokens(self) -> None:
        """Charging cache reads at par would make caching look like a cost increase.

        `docs/observability.md` §8.2 tracks the cache-read share precisely
        because it is a *saving*; folding it into the billable total erases the
        signal it exists to expose.
        """
        response = LLMResponse(
            text="x", model=WORKER, input_tokens=100, output_tokens=20, cached_tokens=900
        )
        assert response.billable_tokens == 120
        assert response.total_tokens == 1020

    def test_a_max_tokens_stop_is_named_rather_than_inferred(self) -> None:
        """A truncated answer is a config defect, not a quality one.

        Without a name for it, `max_tokens` truncation is indistinguishable from
        a model that simply answered badly, and the two have entirely different
        fixes (`docs/observability.md` §5).
        """
        assert LLMResponse("x", WORKER, 1, 1, stop_reason="max_tokens").truncated is True
        assert LLMResponse("x", WORKER, 1, 1, stop_reason="end_turn").truncated is False


class TestErrorTaxonomy:
    """The class is what the caller branches on, so the classes must differ."""

    def test_every_failure_is_an_external_service_error(self) -> None:
        """`ExternalServiceError` already claims the LLM provider in its docstring.

        Deriving from it means an LLM failure that escapes to the API renders as
        a 502 problem document without any route needing a translation table.
        """
        assert issubclass(LLMError, ExternalServiceError)
        assert issubclass(LLMTimeout, LLMError)
        assert issubclass(LLMRateLimited, LLMError)
        assert issubclass(LLMSchemaError, LLMError)

    def test_provider_rate_limiting_is_not_reported_to_our_caller_as_429(self) -> None:
        """A 429 blames the client for a quota that is not theirs.

        Returning 429 tells the caller to slow down; the actual remedy is a tier
        shed on our side, and the caller obediently backing off makes it worse.
        """
        assert LLMRateLimited().status_code == 503
        assert LLMTimeout().status_code == 504

    def test_the_retry_after_hint_survives_onto_the_exception(self) -> None:
        """Without it the router paces against a number it invented."""
        error = LLMRateLimited(retry_after_seconds=12.0)
        assert error.retry_after_seconds == 12.0
        assert error.details["retry_after_seconds"] == 12.0

    def test_schema_failures_carry_the_attempt_count(self) -> None:
        """Repeated schema failure is a prompt defect and must be countable alone."""
        error = LLMSchemaError("nope", schema="Plan", attempts=2)
        assert error.details == {"attempts": 2, "schema": "Plan"}


class TestFakeLLMProvider:
    """The fake every other suite in Layer 3 will run against."""

    def test_it_satisfies_both_protocols(self) -> None:
        """If the fake drifts from the contract, every test using it lies."""
        fake = FakeLLMProvider()
        assert isinstance(fake, LLMProvider)
        assert isinstance(fake, MeteredLLMProvider)

    async def test_scripted_answers_are_returned_in_order(self) -> None:
        fake = FakeLLMProvider(["first", "second"])
        assert (await fake.complete(prompt="a")).text == "first"
        assert (await fake.complete(prompt="b")).text == "second"

    async def test_a_scripted_exception_is_raised(self) -> None:
        """Scripting a failure is how the router's shed path is tested at all."""
        fake = FakeLLMProvider([LLMRateLimited()])
        with pytest.raises(LLMRateLimited):
            await fake.complete(prompt="a")

    async def test_running_off_the_end_of_the_script_fails_loudly(self) -> None:
        """A silent default would let a subject make extra calls unnoticed.

        Extra calls are exactly the regression this layer must not ship: they
        are invisible in the result and visible only on the invoice.
        """
        fake = FakeLLMProvider(["only"])
        await fake.complete(prompt="a")
        with pytest.raises(LLMError, match="script exhausted"):
            await fake.complete(prompt="b")

    async def test_default_covers_tests_that_do_not_count_calls(self) -> None:
        fake = FakeLLMProvider(default="always")
        assert (await fake.complete(prompt="a")).text == "always"
        assert (await fake.complete(prompt="b")).text == "always"

    async def test_calls_are_recorded_with_their_arguments(self) -> None:
        """Assertions belong on recorded calls, not on mock internals."""
        fake = FakeLLMProvider(["x"])
        await fake.complete(prompt="p", system="s", model="m", max_tokens=7, temperature=0.4)
        call = fake.calls[0]
        assert (call.kind, call.prompt, call.system, call.model) == ("complete", "p", "s", "m")
        assert (call.max_tokens, call.temperature) == (7, 0.4)

    async def test_synthetic_token_counts_grow_with_the_prompt(self) -> None:
        """A constant would let a broken budget accumulator pass its tests."""
        fake = FakeLLMProvider(["one two three"])
        response = await fake.complete(prompt="a b c d")
        assert response.input_tokens == 4
        assert response.output_tokens == 3

    async def test_structured_accepts_a_model_a_mapping_or_raw_json(self) -> None:
        fake = FakeLLMProvider(
            [
                Plan(steps=["a"], confidence=1.0),
                {"steps": ["b"], "confidence": 0.5},
                '{"steps": ["c"], "confidence": 0.25}',
            ]
        )
        for expected in ("a", "b", "c"):
            plan = await fake.structured(prompt="p", schema=Plan)
            assert plan.steps == [expected]

    async def test_a_scripted_value_that_misses_the_schema_raises_the_real_class(
        self,
    ) -> None:
        """The caller's failure path must be reachable without knowing it is a fake."""
        fake = FakeLLMProvider([{"steps": ["a"]}])
        with pytest.raises(LLMSchemaError):
            await fake.structured(prompt="p", schema=Plan)

    async def test_structured_metered_reports_usage(self) -> None:
        fake = FakeLLMProvider([Plan(steps=["a", "b"], confidence=1.0)])
        plan, usage = await fake.structured_metered(prompt="a b", schema=Plan)
        assert plan.steps == ["a", "b"]
        assert usage.billable_tokens > 0


# --------------------------------------------------------------------------- #
# anthropic_provider.py -- the SDK client
# --------------------------------------------------------------------------- #


class TestAnthropicClientConstruction:
    """Everything the SDK needs arrives as a value, and nothing from the environment."""

    async def test_it_satisfies_both_protocols(self) -> None:
        """If it drifts from the contract, every enrichment stage drifts with it."""
        async with anthropic_stub() as (client, _):
            assert isinstance(client, LLMProvider)
            assert isinstance(client, MeteredLLMProvider)

    async def test_settings_configure_the_sdk_client(self) -> None:
        """The endpoint, the per-attempt budget and the retry policy are declared.

        Left to itself the SDK falls back to `ANTHROPIC_API_KEY` and
        `ANTHROPIC_BASE_URL` from the process environment -- a second,
        undeclared way to redirect every LLM call, invisible to
        `Settings.describe()` and to `.env.example`. `backend/core/config.py` is
        the only module allowed to read the environment
        (`docs/coding-standards.md` §2.9), so each of these is resolved there and
        handed over. Reaching into `_client` is the price of asserting it
        without issuing a request, which is the only way to catch a value that
        is silently never passed.
        """
        client = AnthropicProvider(settings=llm_settings(timeout_seconds=42, max_retries=5))
        try:
            sdk = client._client
            assert str(sdk.base_url).rstrip("/") == BASE_URL
            assert sdk.timeout == 42.0
            assert sdk.max_retries == 5
            assert sdk.api_key == "k-secret"
        finally:
            await client.aclose()

    def test_a_missing_api_key_fails_at_construction(self) -> None:
        """A worker that starts happily and fails every enrichment an hour later
        is far more expensive to diagnose than one that refuses to start."""
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider(settings=llm_settings(api_key=None))

    def test_a_blank_api_key_is_treated_as_missing(self) -> None:
        """An exported-but-empty variable is the usual shape of this mistake, and
        it reaches the provider as a present-looking secret."""
        with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY"):
            AnthropicProvider(settings=llm_settings(api_key=""))

    async def test_an_injected_client_is_not_closed_by_aclose(self) -> None:
        """Closing a shared pool out from under other callers is a real outage."""
        shared = anthropic.AsyncAnthropic(api_key="k", base_url=BASE_URL)
        client = AnthropicProvider(settings=llm_settings(), client=shared)

        await client.aclose()

        assert shared.is_closed() is False
        await shared.close()

    async def test_a_client_it_owns_is_closed_by_aclose(self) -> None:
        """The other half of the same rule; getting it wrong here leaks sockets."""
        client = AnthropicProvider(settings=llm_settings())
        owned = client._client

        await client.aclose()

        assert owned.is_closed() is True


# --------------------------------------------------------------------------- #
# anthropic_provider.py -- requests
# --------------------------------------------------------------------------- #


class TestAnthropicRequestShape:
    """What the SDK actually puts on the wire."""

    async def test_the_key_is_a_header_and_never_a_query_parameter(self) -> None:
        """A credential in a URL is logged by every proxy between here and there."""
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello")

        assert wire.headers()["x-api-key"] == "k-secret"
        assert "k-secret" not in str(wire.requests[0].url)

    async def test_the_api_version_header_is_the_sdks_to_pin(self) -> None:
        """This header used to be a constant in this repository.

        Pinning a vendor's wire version by hand makes every bump a code change
        nobody will think to make, against an API that ships breaking changes.
        The assertion is only that a version is sent and the endpoint is right;
        *which* version is the vendor's business now, and handing that back is
        most of the point of using their client.
        """
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello")

        assert wire.headers()["anthropic-version"]
        assert wire.requests[0].url.path == "/v1/messages"

    async def test_the_configured_worker_model_is_the_default(self) -> None:
        """Callers that pass no model get the volume tier, not the expensive one."""
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello")

        assert wire.payload()["model"] == WORKER

    async def test_system_and_max_tokens_reach_the_request(self) -> None:
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello", system="be terse", max_tokens=64)

        payload = wire.payload()
        assert payload["system"] == "be terse"
        assert payload["max_tokens"] == 64
        assert payload["messages"] == [{"role": "user", "content": "hello"}]

    async def test_max_tokens_falls_back_to_the_configured_ceiling(self) -> None:
        """`max_tokens` is required by the API; omitting a default would 400."""
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello")

        assert wire.payload()["max_tokens"] == 1024

    async def test_an_absent_system_prompt_is_omitted_rather_than_nulled(self) -> None:
        """The SDK forwards an explicit `None` as a JSON `null`, which this
        field rejects -- so the keys have to be left out, not set empty."""
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello")

        assert "system" not in wire.payload()

    async def test_effort_is_sent_only_once_a_level_is_configured(self) -> None:
        """Unset means the provider's own default, which is where its tuning lives.

        Pinning a level here would freeze that tuning at whatever was current
        the day it was typed (`backend/core/config.py`, `LLMEffort`).
        """
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello")
        assert "output_config" not in wire.payload()

        async with anthropic_stub(effort="high") as (client, wire):
            await client.complete(prompt="hello")
        assert wire.payload()["output_config"] == {"effort": "high"}

    async def test_budget_tokens_is_never_sent_on_any_call(self) -> None:
        """It was removed on the current models and now answers 400.

        Effort replaced it. The assertion is against the whole serialized body
        rather than one key, because the old parameter lived nested under
        `thinking` and would come back that way if it came back at all.
        """
        async with anthropic_stub(message(text=PLAN_JSON), effort="max") as (client, wire):
            await client.complete(prompt="hello")
            await client.structured(prompt="plan it", schema=Plan)

        assert wire.call_count == 2
        for index in range(wire.call_count):
            body = json.dumps(wire.payload(index))
            assert "budget_tokens" not in body
            assert "thinking" not in body


class TestAnthropicSamplingParameters:
    """`temperature` is model-gated, and the default makes it always present."""

    @pytest.mark.parametrize(
        "model",
        [
            PLANNER,
            WORKER,
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-fable-5",
            "claude-mythos-5",
        ],
    )
    async def test_temperature_is_withheld_from_models_that_reject_it(self, model: str) -> None:
        """The current Claude generation removed sampling parameters.

        `LLM_TEMPERATURE` defaults to `0.0`, so "only send it when set" is not a
        rule that helps -- it is always set. Sending it to `claude-opus-5`
        returns 400, which would fail every planner and worker call in this
        deployment on the first request. The SDK does not know this rule; it
        forwards what it is given, so the guard is ours to keep.
        """
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello", model=model, temperature=0.7)

        payload = wire.payload()
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload

    async def test_temperature_still_reaches_models_that_accept_it(self) -> None:
        """Failing open matters as much: silently dropping it changes results."""
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello", model=FAST, temperature=0.7)

        assert wire.payload()["temperature"] == 0.7

    async def test_the_configured_default_applies_when_the_caller_passes_none(self) -> None:
        """A deployment that tuned `LLM_TEMPERATURE` expects it to be used."""
        async with anthropic_stub(temperature=0.4) as (client, wire):
            await client.complete(prompt="hello", model=FAST)

        assert wire.payload()["temperature"] == 0.4

    async def test_an_unrecognised_model_is_assumed_to_accept_sampling(self) -> None:
        """The guard fails *open*.

        A model released tomorrow must not be silently stripped of a parameter
        it supports. Guessing wrong in this direction costs one loud 400 on the
        first request; guessing wrong in the other changes every result quietly.
        """
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello", model="claude-next-9", temperature=0.3)

        assert wire.payload()["temperature"] == 0.3


class TestAnthropicModelIds:
    """A wrong model id fails at the first request, far from its cause."""

    @pytest.mark.parametrize(
        "model",
        [
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20241022",
            "claude-2.1",
            "claude-instant-1.2",
        ],
    )
    async def test_a_retired_model_is_rejected_before_any_request(self, model: str) -> None:
        """A retired id answers 404, which reads as a bad URL or a revoked key.

        Naming the real problem -- and the replacements -- at the call site is
        the difference between a two-minute fix and an afternoon. Asserting that
        nothing reached the transport is the point: a guard that fires after the
        request has been billed is not a guard.
        """
        async with anthropic_stub() as (client, wire):
            with pytest.raises(ConfigurationError, match="retired"):
                await client.complete(prompt="hello", model=model)

            assert wire.call_count == 0

    async def test_an_empty_model_id_names_the_settings_that_could_be_blank(self) -> None:
        """`LLM_MODEL_WORKER=` in a `.env` is a plausible typo, and an empty
        `model` is a 400 whose message says nothing about configuration."""
        async with anthropic_stub() as (client, wire):
            with pytest.raises(ConfigurationError, match="LLM_MODEL"):
                await client.complete(prompt="hello", model="   ")

            assert wire.call_count == 0

    async def test_an_unreleased_model_id_passes_the_guard(self) -> None:
        """The guard rejects *retired* prefixes rather than allow-listing the
        three ids we ship with: an allowlist would reject the next model on the
        day it is released, turning a config change into a code change."""
        async with anthropic_stub() as (client, wire):
            await client.complete(prompt="hello", model="claude-opus-6-20270101")

        assert wire.payload()["model"] == "claude-opus-6-20270101"


# --------------------------------------------------------------------------- #
# anthropic_provider.py -- accounting and failures
# --------------------------------------------------------------------------- #


class TestAnthropicAccounting:
    """Usage exists only on the response that produced it."""

    async def test_usage_is_read_off_every_response(self) -> None:
        async with anthropic_stub(
            message(
                text="answer",
                usage={"input_tokens": 40, "output_tokens": 12, "cache_read_input_tokens": 900},
            )
        ) as (client, _):
            response = await client.complete(prompt="hello")

        assert response.text == "answer"
        assert (response.input_tokens, response.output_tokens) == (40, 12)
        assert response.cached_tokens == 900
        assert response.stop_reason == "end_turn"

    async def test_the_model_that_answered_is_recorded_not_the_one_requested(self) -> None:
        """Aliases resolve server-side, so the requested id is a guess and the
        response's is the fact -- and the fact is what a cost attribution needs."""
        async with anthropic_stub(message(model="claude-sonnet-5-20260101")) as (client, _):
            response = await client.complete(prompt="hello", model=WORKER)

        assert response.model == "claude-sonnet-5-20260101"

    async def test_cache_reads_and_writes_are_not_confused_for_each_other(self) -> None:
        """Writes bill at a premium; reads are the saving.

        `LLMResponse` has no fourth counter, so cache *creation* is folded into
        `input_tokens` -- dropping it would under-report spend on exactly the
        largest prompts, the ones a cost alert is watching. Cache *reads* stay
        separate because `docs/observability.md` §8.2 tracks their share, and
        folding them in would erase the signal that section exists to expose.
        """
        async with anthropic_stub(
            message(
                usage={
                    "input_tokens": 10,
                    "output_tokens": 2,
                    "cache_creation_input_tokens": 500,
                    "cache_read_input_tokens": 900,
                }
            )
        ) as (client, _):
            response = await client.complete(prompt="hello")

        assert response.input_tokens == 510
        assert response.cached_tokens == 900
        assert response.billable_tokens == 512
        assert response.total_tokens == 1412

    async def test_a_refusal_returns_empty_text_rather_than_raising(self) -> None:
        """A refusal is a 200 with `stop_reason="refusal"` and no content blocks.

        Raising would turn a content outcome the caller must handle into a
        transport error the caller will retry -- paying twice for the same no.
        """
        async with anthropic_stub(message(stop_reason="refusal", content=[])) as (client, _):
            response = await client.complete(prompt="hello")

        assert response.text == ""
        assert response.stop_reason == "refusal"

    async def test_thinking_blocks_are_not_mistaken_for_the_answer(self) -> None:
        """Thinking is on by default on the current models, so these blocks
        arrive unasked-for. They are the model's scratchpad, not its answer, and
        a provider that concatenated them would put reasoning into a report.
        """
        async with anthropic_stub(
            message(
                content=[
                    {"type": "thinking", "thinking": "scratchpad musing", "signature": "sig"},
                    {"type": "text", "text": "the answer"},
                ]
            )
        ) as (client, _):
            response = await client.complete(prompt="hello")

        assert response.text == "the answer"

    async def test_multiple_text_blocks_are_joined_rather_than_sampled(self) -> None:
        """Taking only the first would silently truncate a segmented reply."""
        async with anthropic_stub(
            message(content=[{"type": "text", "text": "one "}, {"type": "text", "text": "two"}])
        ) as (client, _):
            response = await client.complete(prompt="hello")

        assert response.text == "one two"


class TestAnthropicFailureMapping:
    """The caller acts on the class, so the mapping is the contract.

    The provider's `except` clauses run most-specific-first, which is not
    cosmetic: `APITimeoutError` subclasses `APIConnectionError`, and every
    4xx/5xx class subclasses `APIStatusError`, so a broader clause moved earlier
    silently swallows the distinction the router branches on. These cases are
    what catches that reordering.
    """

    async def test_429_becomes_rate_limited_with_the_providers_own_delay(self) -> None:
        async with anthropic_stub(
            error_response(
                429,
                error_type="rate_limit_error",
                error_message="slow down",
                headers={"retry-after": "30", "request-id": "req_9"},
            )
        ) as (client, _):
            with pytest.raises(LLMRateLimited) as caught:
                await client.complete(prompt="hello")

        assert caught.value.retry_after_seconds == 30.0
        assert caught.value.request_id == "req_9"
        assert caught.value.provider_error_type == "rate_limit_error"

    async def test_an_unparseable_retry_after_leaves_pacing_to_the_router(self) -> None:
        """Anthropic sends delta-seconds; an HTTP-date parses to nothing here.

        That is honest -- the router then falls back to its own pacing rather
        than acting on a number this module invented.
        """
        async with anthropic_stub(
            error_response(
                429,
                error_type="rate_limit_error",
                headers={"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"},
            )
        ) as (client, _):
            with pytest.raises(LLMRateLimited) as caught:
                await client.complete(prompt="hello")

        assert caught.value.retry_after_seconds is None

    @pytest.mark.parametrize(
        ("status", "error_type"),
        [
            (400, "invalid_request_error"),
            (401, "authentication_error"),
            (403, "permission_error"),
            (404, "not_found_error"),
        ],
    )
    async def test_a_permanent_rejection_keeps_its_status_and_stays_unshed(
        self, status: int, error_type: str
    ) -> None:
        """A bad key, a revoked scope, a retired model id or a malformed request.

        Permanent by construction, so the class must be the plain `LLMError` the
        router refuses to shed on: the smaller model fails identically, and
        shedding would spend a second call to learn the same thing.
        """
        async with anthropic_stub(error_response(status, error_type=error_type)) as (client, _):
            with pytest.raises(LLMError) as caught:
                await client.complete(prompt="hello")

        assert type(caught.value) is LLMError
        assert caught.value.provider_status == status
        assert caught.value.provider_error_type == error_type

    async def test_a_5xx_keeps_the_status_so_the_router_can_shed(self) -> None:
        async with anthropic_stub(error_response(529, error_type="overloaded_error")) as (
            client,
            _,
        ):
            with pytest.raises(LLMError) as caught:
                await client.complete(prompt="hello")

        assert caught.value.provider_status == 529

    @pytest.mark.parametrize("status", [408, 504])
    async def test_a_server_side_timeout_is_a_timeout_not_a_generic_failure(
        self, status: int
    ) -> None:
        """Distinct from a refusal: the request may have been billed, and a
        retry policy that cannot tell those apart double-spends."""
        async with anthropic_stub(error_response(status, error_type="timeout_error")) as (
            client,
            _,
        ):
            with pytest.raises(LLMTimeout) as caught:
                await client.complete(prompt="hello")

        assert caught.value.provider_status == status

    async def test_a_client_side_timeout_is_its_own_class(self) -> None:
        """We stopped listening; the provider may well have finished and billed."""
        async with anthropic_stub(httpx.ReadTimeout("too slow")) as (client, _):
            with pytest.raises(LLMTimeout):
                await client.complete(prompt="hello")

    async def test_a_transport_failure_is_an_llm_error_with_no_status(self) -> None:
        """DNS, TLS, a connection reset: there was no provider *response*, so
        there is no status to report and inventing one would put a number into a
        metric the provider never sent."""
        async with anthropic_stub(httpx.ConnectError("no route")) as (client, _):
            with pytest.raises(LLMError) as caught:
                await client.complete(prompt="hello")

        assert type(caught.value) is LLMError
        assert caught.value.provider_status is None

    async def test_the_providers_error_message_is_not_carried(self) -> None:
        """Providers echo the offending request back inside `error.message`.

        Requests carry fetched content, and this exception is logged and
        serialized into an HTTP response (`docs/security-and-privacy.md`). The
        error *type* is a closed vocabulary and is kept.
        """
        async with anthropic_stub(
            error_response(400, error_message="prompt contained: SECRET-PAYLOAD")
        ) as (client, _):
            with pytest.raises(LLMError) as caught:
                await client.complete(prompt="hello")

        rendered = json.dumps(caught.value.to_problem())
        assert "SECRET-PAYLOAD" not in rendered
        assert caught.value.provider_error_type == "invalid_request_error"

    async def test_retrying_is_the_sdks_job_and_is_bounded_by_settings(self) -> None:
        """This module used to refuse to retry at all, on the grounds that
        private retries make the router's overload signal wrong.

        That argument was about unbounded, hand-written retries of everything.
        The SDK's policy is narrower than what we would write -- connection
        failures, 408/409/429 and 5xx only, with backoff that honours
        `Retry-After` -- and it never touches a 400, which is the class where a
        retry is pure waste.
        """
        async with anthropic_stub(
            error_response(429, error_type="rate_limit_error"),
            message(text="eventually"),
            max_retries=2,
        ) as (client, wire):
            response = await client.complete(prompt="hello")

        assert response.text == "eventually"
        assert wire.call_count == 2

    async def test_zero_retries_sends_the_failure_straight_to_the_router(self) -> None:
        """`LLM_MAX_RETRIES=0` is how a deployment gets the old behaviour back:
        what survives the budget reaches `services/llm/router.py` and is shed."""
        async with anthropic_stub(
            error_response(429, error_type="rate_limit_error"), max_retries=0
        ) as (client, wire):
            with pytest.raises(LLMRateLimited):
                await client.complete(prompt="hello")

        assert wire.call_count == 1


# --------------------------------------------------------------------------- #
# anthropic_provider.py -- structured output
# --------------------------------------------------------------------------- #


class TestAnthropicStructured:
    """Schema conformance is forced by the request, not hoped for in the prompt."""

    async def test_the_schema_travels_on_the_request_as_a_constrained_format(self) -> None:
        """Asking for JSON in the prompt and parsing the reply is a different
        mechanism with strictly worse failure modes: markdown fences, "Here is
        the JSON:" preambles, and values that are *plausibly wrong* rather than
        absent. With the schema on the request the provider constrains
        generation to it, so that whole class never occurs.

        The old implementation achieved this with a forced tool call it encoded
        by hand. The SDK's native path sends `output_config.format` instead --
        same guarantee, minus the encoding -- so the absence of `tools` here is
        as much the assertion as the presence of the schema.
        """
        async with anthropic_stub(message(text=PLAN_JSON)) as (client, wire):
            plan = await client.structured(prompt="plan it", schema=Plan)

        payload = wire.payload()
        assert "tools" not in payload
        assert "tool_choice" not in payload

        schema = payload["output_config"]["format"]["schema"]
        assert payload["output_config"]["format"]["type"] == "json_schema"
        assert sorted(schema["required"]) == ["confidence", "steps"]
        assert schema["additionalProperties"] is False
        assert plan.steps == ["a"]

    async def test_it_returns_a_validated_instance_not_a_mapping(self) -> None:
        """The caller's type annotation has to be true at runtime.

        Every enrichment stage reads attributes off this value; handing back a
        dict that happens to have the right keys defers the failure to whichever
        stage touches a field the model omitted.
        """
        async with anthropic_stub(message(text=PLAN_JSON)) as (client, _):
            plan = await client.structured(prompt="plan it", schema=Plan)

        assert isinstance(plan, Plan)
        assert plan.steps == ["a"]
        assert plan.confidence == 0.9

    async def test_structured_metered_reports_what_the_call_cost(self) -> None:
        """`structured()` returns the parsed model, so its usage has nowhere to
        go; without this seam every planning call would be recorded as free."""
        async with anthropic_stub(
            message(
                text=PLAN_JSON,
                usage={"input_tokens": 30, "output_tokens": 7, "cache_read_input_tokens": 12},
            )
        ) as (client, _):
            plan, usage = await client.structured_metered(prompt="plan it", schema=Plan)

        assert plan.steps == ["a"]
        assert (usage.input_tokens, usage.output_tokens) == (30, 7)
        assert usage.cached_tokens == 12
        # The raw JSON, which is what a trace needs to explain a later failure.
        assert usage.text == PLAN_JSON

    async def test_one_request_and_no_validation_retry_loop(self) -> None:
        """The hand-rolled version re-asked with the validation error fed back.

        That was worth its price when the schema was advisory and a near-miss
        value was routine. Under constrained generation it is not: the surviving
        failure is a response cut off by `max_tokens`, and re-asking with the
        same budget reproduces it exactly while billing twice.
        """
        async with anthropic_stub(
            message(text='{"steps": ["a"], "confid', stop_reason="max_tokens")
        ) as (client, wire):
            with pytest.raises(LLMSchemaError):
                await client.structured(prompt="plan it", schema=Plan)

        assert wire.call_count == 1

    async def test_a_truncated_object_reports_the_real_cause(self) -> None:
        """`max_tokens` truncation and a bad prompt need different fixes.

        The `Message` carrying `stop_reason="max_tokens"` never reaches us --
        the SDK raises inside response parsing -- so unparseable JSON *is* the
        diagnosis: generation was constrained to the schema, so text that is not
        even JSON means the object was cut off mid-emit.
        """
        async with anthropic_stub(
            message(text='{"steps": ["a"], "confid', stop_reason="max_tokens")
        ) as (client, _):
            with pytest.raises(LLMSchemaError, match="LLM_MAX_OUTPUT_TOKENS") as caught:
                await client.structured(prompt="plan it", schema=Plan)

        assert caught.value.schema == "Plan"
        assert caught.value.attempts == 1

    async def test_a_refusal_is_a_schema_error_carrying_the_stop_reason(self) -> None:
        """200 with no JSON: a refusal, or content the SDK left unparsed.

        A content outcome, not a transport one -- so it must not look like
        something the caller should retry.
        """
        async with anthropic_stub(message(content=[], stop_reason="refusal")) as (client, _):
            with pytest.raises(LLMSchemaError, match="refused") as caught:
                await client.structured(prompt="plan it", schema=Plan)

        assert caught.value.details["stop_reason"] == "refusal"

    async def test_a_schema_failure_names_the_field_and_never_the_value(self) -> None:
        """The rejected value is model output derived from fetched content, and
        this exception is logged and serialized (`docs/security-and-privacy.md`).
        The field path is what a human needs to fix a schema or a prompt anyway.
        """
        async with anthropic_stub(
            message(text='{"steps": ["a"], "confidence": "SECRET-PAYLOAD"}')
        ) as (client, _):
            with pytest.raises(LLMSchemaError) as caught:
                await client.structured(prompt="plan it", schema=Plan)

        rendered = json.dumps(caught.value.to_problem())
        assert "SECRET-PAYLOAD" not in rendered
        assert "confidence" in rendered

    async def test_sampling_parameters_stay_model_gated_on_a_structured_call(self) -> None:
        """The gate belongs to the model, not to the method.

        Structured calls are not exempt from the deployment's sampling settings
        -- only from the models that reject them -- so the same request against
        `claude-haiku-4-5-*` must still carry the configured temperature.
        """
        async with anthropic_stub(message(text=PLAN_JSON), temperature=0.3) as (client, wire):
            await client.structured(prompt="plan it", schema=Plan, model=PLANNER)
        assert "temperature" not in wire.payload()

        async with anthropic_stub(message(text=PLAN_JSON), temperature=0.3) as (client, wire):
            await client.structured(prompt="plan it", schema=Plan, model=FAST)
        assert wire.payload()["temperature"] == 0.3

    async def test_effort_and_the_schema_share_output_config_without_evicting_each_other(
        self,
    ) -> None:
        """Both live under `output_config`, and the SDK is what merges them.

        This module passes `output_config={"effort": ...}` and `output_format=`
        as separate arguments; `messages.parse` folds the schema in as
        `output_config.format`. That merge is the vendor's code, not ours, and
        it is the kind of thing an SDK release can change from `{**config,
        "format": ...}` to `{"format": ...}` without anybody calling it a
        breaking change.

        Either direction of that regression is silent. Losing `format` would
        surface eventually as unconstrained output; losing `effort` would not
        surface at all -- structured calls would quietly run at the provider's
        default depth while `LLM_EFFORT` still read `high`, and the only symptom
        would be answers that got slightly worse and a config setting that had
        stopped meaning anything.
        """
        async with anthropic_stub(message(text=PLAN_JSON), effort="high") as (client, wire):
            await client.structured(prompt="plan it", schema=Plan)

        output_config = wire.payload()["output_config"]
        assert output_config["effort"] == "high"
        assert output_config["format"]["type"] == "json_schema"
        assert sorted(output_config["format"]["schema"]["required"]) == ["confidence", "steps"]


# --------------------------------------------------------------------------- #
# router.py
# --------------------------------------------------------------------------- #


class TestModelTiering:
    """A tier is a capability request; the model id lives in configuration."""

    def test_each_tier_resolves_to_its_configured_model(self) -> None:
        router = ModelRouter(FakeLLMProvider(), settings=llm_settings())
        assert router.model_for(ModelTier.PLANNER) == PLANNER
        assert router.model_for(ModelTier.WORKER) == WORKER
        assert router.model_for(ModelTier.FAST) == FAST

    async def test_the_tier_model_is_what_reaches_the_provider(self) -> None:
        fake = FakeLLMProvider(["ok"])
        router = ModelRouter(fake, settings=llm_settings())
        await router.complete(tier=ModelTier.PLANNER, prompt="p")
        assert fake.calls[0].model == PLANNER


class TestOverloadShedding:
    """`docs/architecture.md` §7.2: shed the tier, not the work."""

    async def test_a_rate_limited_planner_is_answered_by_the_worker(self) -> None:
        """A worse answer beats no answer.

        A planner call that raises kills the investigation holding a half-built
        plan; one answered a tier down still produces a report.
        """
        fake = FakeLLMProvider([LLMRateLimited(), "smaller but real"])
        router = ModelRouter(fake, settings=llm_settings())
        budget = RunBudget(limit=10_000)

        response = await router.complete(tier=ModelTier.PLANNER, prompt="p", budget=budget)

        assert response.text == "smaller but real"
        assert [call.model for call in fake.calls] == [PLANNER, WORKER]

    async def test_the_shed_is_recorded_on_the_run(self) -> None:
        """A report produced in degraded mode has to be able to say so."""
        fake = FakeLLMProvider([LLMRateLimited(), "ok"])
        router = ModelRouter(fake, settings=llm_settings())
        budget = RunBudget(limit=10_000)

        await router.complete(tier=ModelTier.PLANNER, prompt="p", budget=budget)

        assert budget.degraded is True
        event = budget.shed_events[0]
        assert (event.requested, event.served, event.reason) == (
            ModelTier.PLANNER,
            ModelTier.WORKER,
            "overload",
        )

    async def test_it_walks_the_whole_ladder_before_giving_up(self) -> None:
        fake = FakeLLMProvider([LLMRateLimited(), LLMTimeout(), "at last"])
        router = ModelRouter(fake, settings=llm_settings())

        response = await router.complete(tier=ModelTier.PLANNER, prompt="p")

        assert response.text == "at last"
        assert [call.model for call in fake.calls] == [PLANNER, WORKER, FAST]

    async def test_the_fast_tier_has_nowhere_to_fall_and_raises(self) -> None:
        """The ladder terminates, so provider misbehaviour cannot become a spin."""
        fake = FakeLLMProvider([LLMRateLimited(), LLMRateLimited(), LLMRateLimited()])
        router = ModelRouter(fake, settings=llm_settings())

        with pytest.raises(LLMRateLimited):
            await router.complete(tier=ModelTier.PLANNER, prompt="p")
        assert len(fake.calls) == 3

    async def test_a_schema_failure_is_not_shed(self) -> None:
        """Retrying a prompt the model could not satisfy on a *weaker* model is
        the least likely repair available, and it hides the real defect."""
        fake = FakeLLMProvider([LLMSchemaError("bad", schema="Plan", attempts=2)])
        router = ModelRouter(fake, settings=llm_settings())

        with pytest.raises(LLMSchemaError):
            await router.structured(tier=ModelTier.PLANNER, prompt="p", schema=Plan)
        assert len(fake.calls) == 1

    async def test_a_client_error_is_not_shed(self) -> None:
        """A malformed request is malformed at every tier."""
        fake = FakeLLMProvider([LLMError("bad request", provider_status=400)])
        router = ModelRouter(fake, settings=llm_settings())

        with pytest.raises(LLMError):
            await router.complete(tier=ModelTier.PLANNER, prompt="p")
        assert len(fake.calls) == 1


class TestRunBudget:
    """`INVESTIGATION_TOKEN_BUDGET` is the ceiling on one run's spend."""

    async def test_spend_accumulates_across_calls(self) -> None:
        fake = FakeLLMProvider(default="a b c")
        router = ModelRouter(fake, settings=llm_settings())
        budget = RunBudget(limit=1_000)

        await router.complete(tier=ModelTier.WORKER, prompt="one two")
        assert budget.spent == 0  # no budget passed -> nothing charged

        await router.complete(tier=ModelTier.WORKER, prompt="one two", budget=budget)
        assert budget.spent == 5
        assert budget.remaining == 995

    async def test_an_exhausted_budget_stops_the_next_call(self) -> None:
        """The guard cannot stop the call that crosses the line.

        A call's cost is unknowable until it returns, so the choice is to
        over-shoot by at most one call or to refuse work on a guess. Over-shoot
        is bounded and auditable; guessing truncates runs that would have
        finished.
        """
        fake = FakeLLMProvider(default="a b c d e")
        router = ModelRouter(fake, settings=llm_settings())
        budget = RunBudget(limit=6)

        await router.complete(tier=ModelTier.WORKER, prompt="one two", budget=budget)
        assert budget.spent == 7  # the overshoot, bounded by one call

        with pytest.raises(TokenBudgetExceeded):
            await router.complete(tier=ModelTier.WORKER, prompt="one two", budget=budget)
        assert len(fake.calls) == 1

    def test_a_budget_stop_is_not_a_provider_error(self) -> None:
        """Generic provider handling retries, sheds and parks -- all wrong here.

        Nothing about the provider is broken, and repeating the call is exactly
        what must not happen, so this must not be catchable as an `LLMError`.
        """
        assert not issubclass(TokenBudgetExceeded, LLMError)
        assert TokenBudgetExceeded().status_code == 429

    def test_cache_reads_do_not_consume_the_budget(self) -> None:
        """They cost about a tenth of list price; charging them at par would
        punish a well-cached prompt for being well cached."""
        budget = RunBudget(limit=100)
        budget.charge(LLMResponse("x", WORKER, 10, 5, cached_tokens=900))
        assert budget.spent == 15

    def test_the_budget_comes_from_settings(self) -> None:
        budget = RunBudget.from_settings()
        assert budget.limit >= 1000

    async def test_structured_calls_are_metered_when_the_provider_allows_it(self) -> None:
        fake = FakeLLMProvider([Plan(steps=["a"], confidence=1.0)])
        router = ModelRouter(fake, settings=llm_settings())
        budget = RunBudget(limit=1_000)

        await router.structured(tier=ModelTier.WORKER, prompt="a b", schema=Plan, budget=budget)

        assert budget.spent > 0
        assert budget.unmetered_calls == 0

    async def test_an_unmeterable_call_is_counted_rather_than_treated_as_free(
        self,
    ) -> None:
        """`structured()` returns the parsed model, so its usage has nowhere to go.

        A backend that cannot report it still works; the gap is recorded so a
        total that under-reports can be recognised as under-reporting.
        """
        bare = BareProvider()
        assert not isinstance(bare, MeteredLLMProvider)

        router = ModelRouter(bare, settings=llm_settings())
        budget = RunBudget(limit=1_000)
        await router.structured(tier=ModelTier.WORKER, prompt="p", schema=Plan, budget=budget)

        assert budget.spent == 0
        assert budget.unmetered_calls == 1


class TestBudgetPressure:
    """`docs/observability.md` §8.2: downgrade the tier before the money runs out."""

    async def test_a_nearly_spent_run_is_routed_one_tier_down(self) -> None:
        """The end of a run is synthesis over evidence already gathered, which
        survives a smaller model far better than the planning that produced it."""
        fake = FakeLLMProvider(default="ok")
        router = ModelRouter(fake, settings=llm_settings(), pressure_fraction=0.2)
        budget = RunBudget(limit=1_000, spent=900)

        await router.complete(tier=ModelTier.PLANNER, prompt="p", budget=budget)

        assert fake.calls[0].model == WORKER
        assert budget.shed_events[0].reason == "budget_pressure"

    async def test_pressure_routing_is_off_when_the_fraction_is_zero(self) -> None:
        fake = FakeLLMProvider(default="ok")
        router = ModelRouter(fake, settings=llm_settings(), pressure_fraction=0.0)
        budget = RunBudget(limit=1_000, spent=999)

        await router.complete(tier=ModelTier.PLANNER, prompt="p", budget=budget)

        assert fake.calls[0].model == PLANNER
        assert budget.shed_events == []


# --------------------------------------------------------------------------- #
# cache.py
# --------------------------------------------------------------------------- #


class TestCacheKey:
    """Everything that can change the answer has to be in the key."""

    def test_the_model_is_part_of_the_key(self) -> None:
        """Serving an opus answer to a haiku request changes the result *and*
        the cost attribution, and nothing downstream can tell."""
        opus = completion_cache_key(prompt="p", model=PLANNER)
        haiku = completion_cache_key(prompt="p", model=FAST)
        assert opus != haiku

    def test_the_system_prompt_is_part_of_the_key(self) -> None:
        assert completion_cache_key(prompt="p", model=FAST, system="a") != completion_cache_key(
            prompt="p", model=FAST, system="b"
        )

    def test_the_schema_shape_is_part_of_the_key_not_just_its_name(self) -> None:
        """A model whose fields changed under an unchanged name is a new request.

        Keying on `__name__` would serve yesterday's answer for today's schema,
        and it would fail validation in a place that says nothing about caching.
        """

        class Plan(BaseModel):  # deliberately shadows the module-level Plan
            steps: list[str]

        renamed = completion_cache_key(prompt="p", model=FAST, schema=Plan)
        original = completion_cache_key(prompt="p", model=FAST, schema=globals()["Plan"])
        assert renamed != original

    def test_parameter_ordering_does_not_change_the_key(self) -> None:
        """Dict ordering belongs to whoever built the mapping; letting it into
        the hash produces two keys for one request and a cache that never hits."""
        first = completion_cache_key(
            prompt="p", model=FAST, params={"max_tokens": 10, "temperature": 0.0}
        )
        second = completion_cache_key(
            prompt="p", model=FAST, params={"temperature": 0.0, "max_tokens": 10}
        )
        assert first == second

    def test_keys_are_namespaced_and_hashed(self) -> None:
        """The prefix carries the payload version, so a shape change orphans old
        entries instead of feeding them to a parser that cannot read them."""
        key = completion_cache_key(prompt="p", model=FAST)
        assert key.startswith(CACHE_PREFIX)
        assert "p" not in key.removeprefix(CACHE_PREFIX) or len(key) == len(CACHE_PREFIX) + 64


class TestCompletionCache:
    """Round-trip behaviour, and the four ways a lookup is allowed to miss."""

    async def test_a_stored_response_comes_back(self, redis: Any) -> None:
        cache = CompletionCache(redis, enabled=True, ttl_seconds=60)
        stored = LLMResponse("the answer", WORKER, 100, 20, stop_reason="end_turn")

        assert await cache.set(stored, prompt="p", model=WORKER) is True
        hit = await cache.get(prompt="p", model=WORKER)

        assert hit is not None
        assert hit.text == "the answer"
        assert hit.model == WORKER
        assert hit.stop_reason == "end_turn"

    async def test_a_hit_costs_nothing_and_says_so(self, redis: Any) -> None:
        """`LLMResponse` has no "cached" flag, so the token counts carry the truth.

        Returning the stored counts verbatim would inflate every cost dashboard
        by exactly the amount the cache saved, and would let a run's budget be
        consumed by answers that were free.
        """
        cache = CompletionCache(redis, enabled=True, ttl_seconds=60)
        await cache.set(
            LLMResponse("x", WORKER, 100, 20, cached_tokens=5), prompt="p", model=WORKER
        )

        hit = await cache.get(prompt="p", model=WORKER)

        assert hit is not None
        assert hit.billable_tokens == 0
        assert hit.cached_tokens == 125
        assert hit.total_tokens == 125  # context pressure is preserved

    async def test_a_different_model_is_a_miss(self, redis: Any) -> None:
        cache = CompletionCache(redis, enabled=True, ttl_seconds=60)
        await cache.set(LLMResponse("opus said this", PLANNER, 1, 1), prompt="p", model=PLANNER)

        assert await cache.get(prompt="p", model=FAST) is None

    async def test_disabling_the_cache_removes_redis_from_the_path(self) -> None:
        """Not a lookup that misses -- no command is issued at all.

        `LLM_CACHE_ENABLED=false` exists for incidents; a "disabled" cache that
        still round-trips to a struggling Redis has not been disabled.
        """
        spy = ExplodingRedis()
        cache = CompletionCache(spy, enabled=False, ttl_seconds=60)

        assert await cache.get(prompt="p", model=WORKER) is None
        assert await cache.set(LLMResponse("x", WORKER, 1, 1), prompt="p", model=WORKER) is False
        assert spy.calls == 0

    async def test_redis_being_down_is_a_miss_not_a_failure(self) -> None:
        """`docs/architecture.md` §7.3: Redis down means "LLM cache misses",
        never "the investigation fails"."""
        cache = CompletionCache(ExplodingRedis(), enabled=True, ttl_seconds=60)

        assert await cache.get(prompt="p", model=WORKER) is None
        assert await cache.set(LLMResponse("x", WORKER, 1, 1), prompt="p", model=WORKER) is False

    async def test_an_unreadable_payload_is_a_miss(self, redis: Any) -> None:
        """Redis is `allkeys-lru` and shared; a corrupt value must not raise."""
        key = completion_cache_key(prompt="p", model=WORKER)
        await redis.set(key, "not json at all")

        cache = CompletionCache(redis, enabled=True, ttl_seconds=60)
        assert await cache.get(prompt="p", model=WORKER) is None

    async def test_a_truncated_answer_is_never_stored(self, redis: Any) -> None:
        """Otherwise every future caller inherits the truncation -- including
        after `LLM_MAX_OUTPUT_TOKENS` is raised to fix it."""
        cache = CompletionCache(redis, enabled=True, ttl_seconds=60)
        cut_off = LLMResponse("half an ans", WORKER, 10, 1024, stop_reason="max_tokens")

        assert await cache.set(cut_off, prompt="p", model=WORKER) is False
        assert await cache.get(prompt="p", model=WORKER) is None

    async def test_every_write_carries_a_ttl(self, redis: Any) -> None:
        """Nothing in Redis is authoritative; an un-expiring completion would
        outlive the prompt version that produced it."""
        cache = CompletionCache(redis, enabled=True, ttl_seconds=90)
        await cache.set(LLMResponse("x", WORKER, 1, 1), prompt="p", model=WORKER)

        ttl = await redis.ttl(completion_cache_key(prompt="p", model=WORKER))
        assert 0 < ttl <= 90

    async def test_structured_responses_are_keyed_by_their_schema(self, redis: Any) -> None:
        cache = CompletionCache(redis, enabled=True, ttl_seconds=60)
        await cache.set(
            LLMResponse('{"steps":[],"confidence":1.0}', WORKER, 1, 1),
            prompt="p",
            model=WORKER,
            schema=Plan,
        )

        assert await cache.get(prompt="p", model=WORKER) is None
        assert await cache.get(prompt="p", model=WORKER, schema=Plan) is not None


# --------------------------------------------------------------------------- #
# embeddings.py
# --------------------------------------------------------------------------- #


class TestEmbeddingBatching:
    """Providers cap inputs per request, and the failure is a 400 for the batch."""

    async def test_requests_are_split_at_the_configured_batch_size(self) -> None:
        """One oversized call loses every chunk in it, not just the last one."""
        client = embedding_provider(batch_size=2)
        row = [0.5, 0.5, 0.5, 0.5]
        with respx.mock:
            route = respx.post(EMBED_URL).mock(
                side_effect=[
                    httpx.Response(200, json=embedding_body(row, row)),
                    httpx.Response(200, json=embedding_body(row, row)),
                    httpx.Response(200, json=embedding_body(row)),
                ]
            )
            vectors = await client.embed(["a", "b", "c", "d", "e"])
        await client.aclose()

        assert len(vectors) == 5
        assert route.call_count == 3
        assert [len(request_json(route, i)["input"]) for i in range(3)] == [2, 2, 1]

    async def test_no_texts_means_no_request(self) -> None:
        """An empty `input` array is a 400 at most providers, and "nothing to
        embed" is a legitimate result from a chunker that found no text."""
        client = embedding_provider()
        with respx.mock:
            route = respx.post(EMBED_URL).mock(return_value=httpx.Response(200, json={}))
            assert await client.embed([]) == []
        await client.aclose()

        assert route.call_count == 0

    async def test_the_model_and_credential_reach_the_request(self) -> None:
        client = embedding_provider()
        with respx.mock:
            route = respx.post(EMBED_URL).mock(
                return_value=httpx.Response(200, json=embedding_body([1.0, 0.0, 0.0, 0.0]))
            )
            await client.embed(["a"])
        await client.aclose()

        assert request_json(route)["model"] == "text-embedding-3-small"
        assert route.calls[0].request.headers["authorization"] == "Bearer k-embed"


class TestEmbeddingResponseHandling:
    """A vector paired with the wrong text is corruption nobody will report."""

    async def test_a_width_mismatch_raises_naming_both_numbers(self) -> None:
        """Otherwise this surfaces at the Qdrant upsert -- after the spend, and
        after the pipeline has already recorded the Embedding stage as OK."""
        client = embedding_provider(dimensions=4)
        with respx.mock:
            respx.post(EMBED_URL).mock(
                return_value=httpx.Response(200, json=embedding_body([1.0, 0.0, 0.0]))
            )
            with pytest.raises(EmbeddingDimensionMismatch) as caught:
                await client.embed(["a"])
        await client.aclose()

        assert caught.value.details["returned"] == 3
        assert caught.value.details["expected"] == 4

    async def test_vectors_are_ordered_by_index_not_by_arrival(self) -> None:
        """The `index` field exists because arrival order is not guaranteed.

        A provider that reorders under load would pair every vector with the
        wrong Signal, which no later check can detect.
        """
        client = embedding_provider()
        shuffled = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0, 0.0, 0.0]},
                {"index": 0, "embedding": [1.0, 0.0, 0.0, 0.0]},
            ]
        }
        with respx.mock:
            respx.post(EMBED_URL).mock(return_value=httpx.Response(200, json=shuffled))
            vectors = await client.embed(["first", "second"])
        await client.aclose()

        assert vectors[0] == [1.0, 0.0, 0.0, 0.0]
        assert vectors[1] == [0.0, 1.0, 0.0, 0.0]

    async def test_a_short_response_is_rejected(self) -> None:
        """Silently short results shift every later vector onto the wrong text."""
        client = embedding_provider()
        with respx.mock:
            respx.post(EMBED_URL).mock(
                return_value=httpx.Response(200, json=embedding_body([1.0, 0.0, 0.0, 0.0]))
            )
            with pytest.raises(LLMError, match="1 vectors for 2"):
                await client.embed(["a", "b"])
        await client.aclose()

    async def test_a_429_uses_the_shared_taxonomy(self) -> None:
        """The Embedding stage catches one family and degrades; a parallel
        hierarchy would mean every caller catching two."""
        client = embedding_provider()
        with respx.mock:
            respx.post(EMBED_URL).mock(return_value=httpx.Response(429))
            with pytest.raises(LLMRateLimited):
                await client.embed(["a"])
        await client.aclose()

    async def test_a_timeout_uses_the_shared_taxonomy(self) -> None:
        client = embedding_provider()
        with respx.mock:
            respx.post(EMBED_URL).mock(side_effect=httpx.ReadTimeout("slow"))
            with pytest.raises(LLMTimeout):
                await client.embed(["a"])
        await client.aclose()

    async def test_an_empty_text_is_refused_before_it_is_sent(self) -> None:
        """An empty string has no meaningful embedding and most providers 400."""
        client = embedding_provider()
        with pytest.raises(ValueError, match=r"texts\[1\]"):
            await client.embed(["a", "   "])
        await client.aclose()


class TestEmbeddingConfiguration:
    """Misconfiguration must fail at boot, not at the first Signal."""

    def test_a_missing_model_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="EMBEDDING_MODEL"):
            embedding_provider(model=None)

    def test_a_self_hosted_provider_must_name_its_endpoint(self) -> None:
        """There is no canonical URL to guess for Modal or a local deployment,
        and guessing wrong embeds against a different corpus's model."""
        with pytest.raises(ConfigurationError, match="EMBEDDING_BASE_URL"):
            embedding_provider(provider="modal", base_url=None)

    def test_a_known_hosted_provider_gets_its_default_endpoint(self) -> None:
        client = embedding_provider(provider="voyage", base_url=None)
        assert isinstance(client, EmbeddingProvider)


class TestFakeEmbeddingProvider:
    """Deterministic vectors are what make a retrieval assertion possible."""

    def test_it_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeEmbeddingProvider(), EmbeddingProvider)

    async def test_the_same_text_always_produces_the_same_vector(self) -> None:
        """Seeded from sha256 rather than `hash()`, which is salted per process:
        a fixture written today would otherwise fail tomorrow for no reason."""
        first = FakeEmbeddingProvider(dimensions=6)
        second = FakeEmbeddingProvider(dimensions=6)
        assert await first.embed(["hello"]) == await second.embed(["hello"])

    async def test_different_texts_produce_different_vectors(self) -> None:
        fake = FakeEmbeddingProvider(dimensions=6)
        vectors = await fake.embed(["hello", "goodbye"])
        assert vectors[0] != vectors[1]

    async def test_vectors_are_unit_length_so_cosine_behaves(self) -> None:
        """A similarity threshold tuned against this fake has to mean something."""
        fake = FakeEmbeddingProvider(dimensions=16)
        (vector,) = await fake.embed(["hello"])
        assert len(vector) == 16
        assert sum(value * value for value in vector) == pytest.approx(1.0)

    async def test_it_records_the_batches_it_was_given(self) -> None:
        """Lets a caller's batching be asserted without reaching for a mock."""
        fake = FakeEmbeddingProvider()
        await fake.embed(["a", "b"])
        await fake.embed(["c"])
        assert fake.batches == [["a", "b"], ["c"]]


class TestStrictSchemaIsAcceptedByRealProviders:
    """Every agent's output schema must survive both providers' validators.

    This exists because it did not. `_strict_schema` handled the two things
    OpenAI documents -- `additionalProperties: false` and every property required
    -- and nothing else, and the result was rejected outright by *both* providers
    for reasons neither shares:

      OpenAI     `$ref cannot have keywords {'default'}`
      Anthropic  `For 'number' type, properties maximum, minimum are not supported`

    Five of the seven agents were affected, including the Report agent, so an
    investigation ran the whole graph and stored nothing. Nothing failed loudly:
    the provider downgraded to a looser mode and the run reported `completed`
    with a null `report_id`.

    Asserted against the schemas the agents actually declare, rather than against
    a fixture, because the failure mode is a *new field* introduced with an
    innocuous `Field(ge=0.0, le=1.0)` -- which is how every one of these arrived.
    A fixture would keep passing while the real schema broke.

    Offline: the provider rules are encoded as assertions rather than probed over
    the network, so this runs in the unit suite. `TestStrictSchemaRejections`
    below documents the exact upstream messages that motivated each rule.
    """

    @staticmethod
    def _agent_output_models() -> list[type[BaseModel]]:
        from agents.collector.schemas import CollectorOutput
        from agents.critic.schemas import CriticOutput
        from agents.insight.schemas import InsightOutput
        from agents.planner.schemas import PlannerOutput
        from agents.report.schemas import ReportOutput
        from agents.retriever.schemas import RetrieverOutput
        from agents.strategy.schemas import StrategyOutput

        return [
            PlannerOutput,
            CollectorOutput,
            RetrieverOutput,
            InsightOutput,
            StrategyOutput,
            CriticOutput,
            ReportOutput,
        ]

    @staticmethod
    def _walk(node: Any) -> Iterator[dict[str, Any]]:
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from TestStrictSchemaIsAcceptedByRealProviders._walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from TestStrictSchemaIsAcceptedByRealProviders._walk(item)

    def test_no_ref_carries_a_sibling_keyword(self) -> None:
        """OpenAI: `$ref cannot have keywords {'default'}`.

        Pydantic emits `{"$ref": ..., "default": ...}` for any field whose type is
        a model or enum and which declares a default -- so this appears without
        anyone writing anything unusual.
        """
        for model in self._agent_output_models():
            for node in self._walk(_strict_schema(model)):
                if "$ref" in node:
                    assert set(node) == {"$ref"}, (
                        f"{model.__name__}: $ref carries {sorted(set(node) - {'$ref'})}; "
                        "OpenAI rejects the whole request"
                    )

    def test_no_range_or_length_bound_survives(self) -> None:
        """Anthropic: `For 'number' type, properties maximum, minimum are not supported`.

        These come from `Field(ge=..., le=...)` and `Field(max_length=...)`, which
        are the normal way to declare a bounded field.
        """
        for model in self._agent_output_models():
            for node in self._walk(_strict_schema(model)):
                offending = sorted(set(node) & _UNSUPPORTED_BOUNDS)
                assert not offending, (
                    f"{model.__name__}: schema still carries {offending}; "
                    "Anthropic rejects the whole request"
                )

    def test_every_object_is_closed_and_fully_required(self) -> None:
        """The original contract, kept: strict mode needs both."""
        for model in self._agent_output_models():
            for node in self._walk(_strict_schema(model)):
                if node.get("type") == "object" and "properties" in node:
                    assert node["additionalProperties"] is False
                    assert node["required"] == sorted(node["properties"])

    def test_a_dropped_bound_survives_in_the_description(self) -> None:
        """Stripping the keyword must not lose the information.

        The first version of the fix deleted bounds outright and broke the Report
        agent: with `minItems: 1` gone from `sections`, the model had no reason to
        know the list may not be empty, returned `[]`, and Pydantic rejected every
        attempt -- so the run produced no report at all. The bound was never what
        *enforced* the rule, but it was what *communicated* it.
        """
        from agents.report.schemas import ReportOutput

        sections = _strict_schema(ReportOutput)["properties"]["sections"]
        assert "minItems" not in sections
        assert "minItems 1" in sections["description"]

    def test_enums_and_descriptions_are_preserved(self) -> None:
        """The stripping must not take the parts that steer the model.

        `enum` is what makes a model answer `high` rather than `quite high`, and
        both providers accept it. A sanitiser that removed everything unfamiliar
        would pass the tests above and produce far worse output.
        """
        from agents.report.schemas import ReportOutput

        nodes = list(self._walk(_strict_schema(ReportOutput)))
        assert any("enum" in node for node in nodes), "enum constraints were stripped"
        assert any("description" in node for node in nodes), "descriptions were stripped"
