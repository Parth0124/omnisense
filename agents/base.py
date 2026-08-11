"""`BaseAgent`: the contract every one of the ten agents implements.

`docs/agent-system.md` §5 fixes four things per agent -- a typed input, a typed
output, a declared tool allowlist and a trace span -- and this module is where
those stop being conventions. A subclass that forgets its allowlist fails at
class-definition time; a subclass whose output does not validate against its
`schemas.py` fails at the node boundary rather than three nodes later, when a
`KeyError` in the Report is the first sign that Insight returned prose.

Four decisions here are load-bearing.

**The node never raises for a classified failure.** `__call__` returns a state
delta, always. A raising node aborts the entire LangGraph run, which would make
one flaky Forecast branch destroy an investigation that had already gathered its
evidence -- and would hang the `{Trend, Competitor, Forecast}` join on a branch
that is never going to write. `docs/agent-system.md` §6 requires the opposite: a
failure is recorded in `errors[]` and the router decides whether the branch was
required.

**Accounting is the wrapper's job, not the agent's.** Every invocation adds
`step_count`, its `TokenLedger` and its `PromptRef` to the delta. Leaving that to
each agent means the one agent that forgets under-reports the run's cost, and
under-reporting is invisible: the number is simply lower than it should be, in
the direction nobody investigates.

**Usage is metered by wrapping the provider, not by re-reading the budget.**
`ModelRouter.structured()` returns the parsed model and charges a `RunBudget`,
which leaves no place for the input/output token split the run record wants. So
the provider handed to the router is a shim that records every `LLMResponse`
into this invocation's ledger. That keeps tier shedding, budget enforcement and
per-node accounting all working at once, none of which is true if you pick one
of them and estimate the others.

**Tool output is data, and is fenced in exactly one place.** Anything a tool
returns is third-party text, which `docs/security-and-privacy.md` treats as an
injection surface. `agents/tools/registry.py` scrubs and fences it at
construction and caps its size; `use_tool()` here deliberately does *not* re-wrap
the result. A second fencing implementation would mean a second sentinel, and the
passage that escapes one of them escapes into a model holding tools. What this
layer adds is the allowlist cross-check: the agent's declared tools and the
registry's grant are written in different files, and a divergence must fail on
the first call rather than be found by reading both.

Prompts come from `prompts/loader.py`, which composes the shared fragments with
the agent template and hashes the composite. The hash is recorded on every
invocation, because §11's claim is that a run can be reconstructed a year later
from what it wrote down -- and a `PromptRef` synthesized from anything other than
the bytes actually sent would make every run *look* reproducible while none was.
"""

from __future__ import annotations

import abc
import hashlib
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from agents.errors import (
    DEFAULT_RETRY_POLICY,
    RetryPolicy,
    StructuredOutputError,
    ToolExecutionError,
    ToolNotAllowedError,
    classify,
    run_with_retry,
    to_state_error,
)
from agents.state import InvestigationState, PromptRef, TokenLedger
from backend.core.config import LLMSettings, Settings, get_settings
from backend.core.logging import get_logger
from models.base import utcnow
from models.enums import AgentName
from prompts.loader import load_prompt
from services.llm.provider import BaseModelT, LLMProvider, LLMResponse, MeteredLLMProvider
from services.llm.router import ModelRouter, ModelTier, RunBudget

__all__ = [
    "EXIT_PATH_AGENTS",
    "EXIT_PATH_OVERDRAFT",
    "AgentContext",
    "AgentTraceEvent",
    "BaseAgent",
    "CollectingTraceSink",
    "LoaderPromptSource",
    "LoggingTraceSink",
    "PromptSource",
    "RenderedPrompt",
    "StateDelta",
    "StaticPromptSource",
    "ToolRegistry",
    "TraceSink",
    "default_prompt_source",
]

_log = get_logger(__name__)

StateDelta = dict[str, Any]
"""What a node returns to LangGraph: only the keys it wrote.

A `dict` rather than a partial `InvestigationState` because the reducer-bearing
keys carry *increments* (`step_count=1`, one node's ledger), not totals, and
typing them as the state would invite someone to return the accumulated value --
which the `operator.add` reducer would then add to itself.
"""

# --------------------------------------------------------------------------- #
# Prompt sourcing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """One composed prompt plus the reference that identifies it.

    `ref` travels into `InvestigationState.prompt_versions` and from there into
    `models/orm/run.py`; `text` is thrown away after the call. Keeping them in
    one object is what stops a node recording the version of a prompt it did not
    actually send.
    """

    text: str
    ref: PromptRef


@runtime_checkable
class PromptSource(Protocol):
    """Where an agent's system prompt comes from.

    A protocol rather than a direct import of `prompts/loader.py` so that the
    loader, a test double and the evaluation harness's pinned-prompt replayer
    are interchangeable -- and so that this layer compiles while the loader is
    still a stub.
    """

    def render(
        self,
        *,
        agent: AgentName,
        version: str,
        context: Mapping[str, Any] | None = None,
    ) -> RenderedPrompt:
        """Compose the agent template with the shared fragments and hash it."""
        ...


class StaticPromptSource:
    """A `PromptSource` over prompt text supplied in-process.

    Real, not a stub: it hashes exactly the bytes it returns, so the `PromptRef`
    it produces is as checkable as the loader's. Used by tests and by the
    evaluation harness, which pins prompt text deliberately rather than reading
    whatever is on disk today.
    """

    def __init__(self, text: str, *, version: str = "static") -> None:
        self._text = text
        self._version = version

    def render(
        self,
        *,
        agent: AgentName,
        version: str,
        context: Mapping[str, Any] | None = None,
    ) -> RenderedPrompt:
        # `context` is deliberately not interpolated: per-run values (tenant id,
        # timestamps) must stay out of the system prompt, because they would
        # break the cache-stable prefix `docs/agent-system.md` §4 requires.
        digest = hashlib.sha256(self._text.encode("utf-8")).hexdigest()
        return RenderedPrompt(
            text=self._text,
            ref=PromptRef(agent=agent, version=version or self._version, sha256=digest),
        )


class LoaderPromptSource:
    """The default source: `prompts/<agent>/vN.md`, composed and hashed on disk.

    An adapter rather than a direct call, for one structural reason:
    `prompts/loader.py` may not import `agents/`, so it returns the three fields
    of a `PromptRef` as a mapping and someone on this side has to assemble them.
    Doing that here means exactly one place knows the shape, instead of ten
    agents each writing `PromptRef(**rendered.ref_fields)`.
    """

    def render(
        self,
        *,
        agent: AgentName,
        version: str,
        context: Mapping[str, Any] | None = None,
    ) -> RenderedPrompt:
        rendered = load_prompt(agent, version or None)
        return RenderedPrompt(text=rendered.text, ref=PromptRef(**rendered.ref_fields))


def default_prompt_source() -> PromptSource:
    """The prompt source an agent gets when its caller did not name one.

    Deliberately not memoised on the class: `prompts/loader.py` does its own
    caching and can be cleared, and a source captured at import time would keep
    serving prompt text a test had already replaced -- passing green against the
    previous test's bytes.
    """
    return LoaderPromptSource()


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

EXIT_PATH_OVERDRAFT = 1.25
EXIT_PATH_AGENTS: frozenset[AgentName] = frozenset({AgentName.REPORT, AgentName.CRITIC})
"""The bounded overdraft the run's exit path is allowed on the token budget.

`agents/graph.py` deliberately runs the Report and the final Critic pass *after*
a budget halt, because §6 requires a partial report rather than none. Enforcing
the exhausted ceiling against them would guarantee the failure that rule exists
to prevent: a run that spent its whole budget and then could not afford to say
what it found. A quarter of the ceiling, spread across at most two nodes, is an
overdraft rather than a waiver -- and it is charged here, in the one place that
constructs a `RunBudget`, instead of by exempting a node from accounting.
"""


@runtime_checkable
class ToolRegistry(Protocol):
    """The only way an agent reaches the outside world.

    Structural, and deliberately narrow: `agents/tools/registry.py` owns the
    implementations, the JSON schemas, the per-agent allowlist, the result size
    ceiling *and* the untrusted-content fence. This layer needs two verbs from it
    and must not import connectors, HTTP clients or credentials to get them
    (`docs/architecture.md` §6.2).

    There is deliberately no second fencing implementation here. Content and
    instructions must be separated exactly once, at the point third-party text is
    constructed, because two fences mean two sentinels -- and a passage that
    escapes one of them escapes into a model holding tools
    (`docs/security-and-privacy.md` §8).
    """

    def is_allowed(self, agent: AgentName, name: str) -> bool:
        """Whether the registry's allowlist grants this agent this tool."""
        ...

    async def invoke(
        self,
        *,
        agent: AgentName,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run the tool, or refuse. Raises inside the `agents/errors.py` taxonomy."""
        ...


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class AgentTraceEvent:
    """One node execution, as the observability layer sees it.

    Mirrors the span attributes `docs/agent-system.md` §12 requires
    (`agent.<name>`, investigation, tenant, prompt version, model tier, tokens)
    without importing an OTel SDK, because `backend/core/telemetry.py` is still
    a stub. When it lands, its exporter becomes one more `TraceSink` and nothing
    in this module changes.
    """

    agent: AgentName
    investigation_id: str
    tenant_id: str
    trace_id: str
    tier: ModelTier
    started_at: datetime
    duration_ms: float
    tokens: TokenLedger
    prompt_ref: PromptRef | None = None
    tool_calls: Sequence[str] = ()
    error_type: str | None = None
    degraded: bool = False
    """Whether the model router served any call below the tier that was asked for."""


@runtime_checkable
class TraceSink(Protocol):
    """Where trace events go. One method, so an exporter is trivial to write."""

    def emit(self, event: AgentTraceEvent) -> None: ...


class LoggingTraceSink:
    """The default sink: one structured log line per node.

    Chosen as the default because it is the only sink that works with nothing
    configured. It is deliberately not silent -- an agent runtime that emits
    nothing until an APM is wired up is one that nobody notices is untraced.
    """

    def emit(self, event: AgentTraceEvent) -> None:
        _log.info(
            "agent.step",
            agent=str(event.agent),
            investigation_id=event.investigation_id,
            tenant_id=event.tenant_id,
            trace_id=event.trace_id,
            tier=str(event.tier),
            duration_ms=round(event.duration_ms, 2),
            input_tokens=event.tokens.input_tokens,
            output_tokens=event.tokens.output_tokens,
            cached_tokens=event.tokens.cached_tokens,
            llm_calls=event.tokens.calls,
            prompt_version=event.prompt_ref.version if event.prompt_ref else None,
            prompt_sha256=event.prompt_ref.sha256 if event.prompt_ref else None,
            tool_calls=list(event.tool_calls),
            error_type=event.error_type,
            degraded=event.degraded,
        )


class CollectingTraceSink:
    """In-memory sink for tests and the evaluation harness.

    Lives beside the production sink, like `FakeLLMProvider`, so the four suites
    that need one do not each grow their own slightly different copy.
    """

    def __init__(self) -> None:
        self.events: list[AgentTraceEvent] = []

    def emit(self, event: AgentTraceEvent) -> None:
        self.events.append(event)


# --------------------------------------------------------------------------- #
# Per-invocation context
# --------------------------------------------------------------------------- #


class _LedgerMeteringProvider:
    """Provider decorator that records every response into one node's ledger.

    Wrapping the provider rather than reading the budget afterwards is what
    preserves the input/output/cached split: `RunBudget` tracks a single
    billable total, and a node that reported only that could never show a cache
    regression (`docs/observability.md` §8.2).

    It satisfies `MeteredLLMProvider` structurally, so `ModelRouter` takes the
    metered path even when the underlying provider cannot -- and when it truly
    cannot, the unmetered call is *counted*, not assumed free.
    """

    def __init__(self, inner: LLMProvider, ledger: _MutableLedger) -> None:
        self._inner = inner
        self._ledger = ledger

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        response = await self._inner.complete(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._ledger.record(response)
        return response

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
        if isinstance(self._inner, MeteredLLMProvider):
            value, usage = await self._inner.structured_metered(
                prompt=prompt, schema=schema, system=system, model=model, max_tokens=max_tokens
            )
            self._ledger.record(usage)
            return value, usage

        value = await self._inner.structured(
            prompt=prompt, schema=schema, system=system, model=model, max_tokens=max_tokens
        )
        # Unknown cost, not zero cost. Counted so a total that under-reports can
        # be recognised as under-reporting rather than trusted.
        usage = LLMResponse(text="", model=model or "unknown", input_tokens=0, output_tokens=0)
        self._ledger.record(usage, unmetered=True)
        return value, usage

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass(slots=True)
class _MutableLedger:
    """Accumulator behind the immutable `TokenLedger` the state carries."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    unmetered_calls: int = 0

    def record(self, response: LLMResponse, *, unmetered: bool = False) -> None:
        self.input_tokens += response.input_tokens
        self.output_tokens += response.output_tokens
        self.cached_tokens += response.cached_tokens
        self.calls += 1
        if unmetered:
            self.unmetered_calls += 1

    def snapshot(self) -> TokenLedger:
        # `cost_usd` stays 0.0: no price table exists anywhere in the codebase
        # yet, and a cost computed from invented prices is worse than an absent
        # one -- it looks authoritative on a dashboard.
        return TokenLedger(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_tokens=self.cached_tokens,
            calls=self.calls,
        )


@dataclass(slots=True)
class AgentContext:
    """Everything one node invocation needs that is not the state itself.

    Built per call and thrown away. Nothing here may be cached on the agent:
    one agent instance serves every concurrent investigation in a worker, and a
    budget or ledger stored on `self` is a cross-run leak -- run A's spend
    cancelling run B's calls -- which is the failure `services/llm/router.py`
    documents at length for exactly the same reason.
    """

    investigation_id: str
    tenant_id: str
    trace_id: str
    deadline_at: datetime | None
    scratchpad_key: str
    budget: RunBudget
    router: ModelRouter
    ledger: _MutableLedger = field(default_factory=_MutableLedger)
    prompt_ref: PromptRef | None = None
    tool_calls: list[str] = field(default_factory=list)

    @property
    def seconds_remaining(self) -> float | None:
        """Wall-clock left before `deadline_at`. `None` when the run is untimed."""
        if self.deadline_at is None:
            return None
        return (self.deadline_at - utcnow()).total_seconds()


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #


class BaseAgent[InputT: BaseModel, OutputT: BaseModel](abc.ABC):
    """One graph node: typed in, typed out, allowlisted tools, traced.

    Subclasses implement three methods and declare four class attributes. The
    split exists so each piece is testable alone: `build_input` is a pure
    projection of the state, `execute` is the part that talks to models and
    tools, and `to_delta` is a pure translation back into state keys. A single
    `run(state) -> state` method would make all three untestable together.

    `tools` is required, even when empty (`frozenset()`), because
    `docs/agent-system.md` §9 is deny-by-default and an *absent* allowlist and an
    *empty* one must not be spelled the same way. The check happens in
    `__init_subclass__`, so the failure is at import rather than at the first
    tool call in production.
    """

    name: ClassVar[AgentName]
    tier: ClassVar[ModelTier]
    output_model: ClassVar[type[BaseModel]]
    tools: ClassVar[frozenset[str]]

    prompt_version: ClassVar[str] = "v1"
    retry_policy: ClassVar[RetryPolicy] = DEFAULT_RETRY_POLICY

    blocking: ClassVar[bool] = False
    """Whether this node's failure leaves the graph unable to continue honestly.

    `False` by default and overridden by the few nodes it is true for. The
    Planner is the obvious one: no plan means no branches, no sub-questions and
    nothing for the Critic to check, so continuing would produce a confident
    report about nothing.
    """

    def __init_subclass__(cls, /, abstract: bool = False, **kwargs: Any) -> None:
        """Refuse to define an agent that is missing part of its contract."""
        super().__init_subclass__(**kwargs)
        if abstract:
            return
        missing = [
            attr for attr in ("name", "tier", "output_model", "tools") if not hasattr(cls, attr)
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} is missing required agent declarations: {', '.join(missing)}. "
                "Every agent must declare its name, model tier, output schema and tool "
                "allowlist -- an implicit allowlist is how an agent quietly acquires a tool."
            )

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        prompts: PromptSource | None = None,
        trace: TraceSink | None = None,
        settings: Settings | None = None,
        llm_settings: LLMSettings | None = None,
    ) -> None:
        resolved = settings if settings is not None else get_settings()
        self._provider = provider
        self._registry = tools
        self._prompts = prompts
        self._trace: TraceSink = trace if trace is not None else LoggingTraceSink()
        self._agent_settings = resolved.agents
        self._llm_settings = llm_settings if llm_settings is not None else resolved.llm

    # ------------------------------------------------------------ subclasses --

    @abc.abstractmethod
    def build_input(self, state: InvestigationState) -> InputT:
        """Project the shared state into this agent's typed request.

        Pure and synchronous on purpose: it is the seam where "the state does
        not contain what this agent needs" is discovered, and discovering that
        before any tokens are spent is the whole point.
        """

    @abc.abstractmethod
    async def execute(self, request: InputT, ctx: AgentContext) -> OutputT:
        """Do the work. The only method allowed to call models or tools."""

    @abc.abstractmethod
    def to_delta(self, output: OutputT, state: InvestigationState) -> StateDelta:
        """Translate the typed output into the state keys this node writes.

        Must return *only* this node's keys, and must respect the reducers in
        `agents/state.py`: append-reduced keys take the increment, never the
        accumulated list.
        """

    # ------------------------------------------------------------ node entry --

    async def __call__(self, state: InvestigationState) -> StateDelta:
        """Run as a LangGraph node. Returns a delta; never raises for a classified failure.

        The bookkeeping added here -- one step, this node's ledger, this node's
        `PromptRef` -- is added on *both* paths, success and failure. A failed
        node still consumed tokens and still moved the run closer to its step
        ceiling, and a delta that omitted that would make the guards in
        `agents/router.py` under-count precisely the runs that are going wrong.
        """
        started = utcnow()
        clock = time.perf_counter()
        ctx = self._context(state)
        error_type: str | None = None
        delta: StateDelta

        try:
            request = self.build_input(state)
            output = await run_with_retry(
                lambda: self.execute(request, ctx),
                policy=self.retry_policy,
            )
            delta = self.to_delta(self._validate_output(output), state)
        except Exception as exc:  # every failure becomes state; see the docstring
            error_type = classify(exc)
            delta = {"errors": [to_state_error(exc, agent=self.name)]}

        delta = {
            **delta,
            "step_count": 1,
            "tokens_spent": ctx.ledger.snapshot(),
        }
        if ctx.prompt_ref is not None:
            delta["prompt_versions"] = {str(self.name): ctx.prompt_ref}

        self._trace.emit(
            AgentTraceEvent(
                agent=self.name,
                investigation_id=ctx.investigation_id,
                tenant_id=ctx.tenant_id,
                trace_id=ctx.trace_id,
                tier=self.tier,
                started_at=started,
                duration_ms=(time.perf_counter() - clock) * 1000.0,
                tokens=ctx.ledger.snapshot(),
                prompt_ref=ctx.prompt_ref,
                tool_calls=tuple(ctx.tool_calls),
                error_type=error_type,
                degraded=ctx.budget.degraded,
            )
        )
        return delta

    # ------------------------------------------------------------- utilities --

    def render_prompt(self, ctx: AgentContext, **context: Any) -> RenderedPrompt:
        """Compose this agent's system prompt and remember which one it was.

        Recording the ref on the context rather than returning it separately is
        what makes "the prompt that produced this output" true by construction:
        there is no path that renders a prompt without registering it.
        """
        source = self._prompts if self._prompts is not None else default_prompt_source()
        rendered = source.render(agent=self.name, version=self.prompt_version, context=context)
        ctx.prompt_ref = rendered.ref
        return rendered

    async def call_model(
        self,
        ctx: AgentContext,
        *,
        prompt: str,
        schema: type[BaseModelT],
        system: str | None = None,
        max_tokens: int | None = None,
        tier: ModelTier | None = None,
    ) -> BaseModelT:
        """A schema-constrained model call at this agent's tier.

        Every LLM call in `agents/` goes through here rather than through the
        provider, so tier shedding, budget enforcement and token accounting are
        not four agents' worth of remembering.
        """
        return await ctx.router.structured(
            tier=tier if tier is not None else self.tier,
            prompt=prompt,
            schema=schema,
            system=system,
            budget=ctx.budget,
            max_tokens=max_tokens,
        )

    async def use_tool(
        self,
        ctx: AgentContext,
        name: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:
        """Invoke a tool this agent declared, through the registry.

        Two allowlists are checked, and the redundancy is the point. `self.tools`
        is what this class *says* it uses; the registry's is what the system
        *grants* it. They are written in different files by different people, and
        a divergence -- an agent that quietly calls a tool it never declared, or
        declares one it was never granted -- is a security-relevant drift that
        should fail on the first call rather than being discovered by reading
        two files side by side.

        The result is returned as the registry produced it, fences and size cap
        included. Re-wrapping it here would create a second rendering of
        third-party text, and the one that skipped a scrub would be the one an
        agent pasted into a prompt.
        """
        if name not in self.tools:
            raise ToolNotAllowedError(
                f"{self.name} may not call {name!r}; its declared allowlist is "
                f"{sorted(self.tools) or '(empty)'}.",
                agent=self.name,
            )
        if not self._registry.is_allowed(self.name, name):
            # Two very different situations reach this line, and conflating them
            # produced a wrong diagnosis for weeks: a tool the deployment could
            # not build is not the same as a tool this agent was refused.
            #
            # `build_default_registry` *trims* the allowlist to whatever toolsets
            # actually constructed, so an unavailable toolset silently removes
            # its tools from every agent's grant. Reporting that as drift sends
            # the reader to compare two files that agree perfectly, while the
            # real cause -- a toolset that raised on construction -- is a warning
            # further up the log.
            if name not in self._registry.names:
                raise ToolExecutionError(
                    f"{name!r} is not registered in this deployment, so "
                    f"{self.name} cannot call it. The toolset that provides it "
                    "failed to construct -- look for `toolset.unavailable` "
                    "earlier in the log for the reason.",
                    agent=self.name,
                    tool=name,
                    transient=False,
                )
            raise ToolNotAllowedError(
                f"{name!r} is declared by {type(self).__name__} but not granted to "
                f"{self.name} in agents/tools/registry.py -- the class and the registry "
                "have drifted.",
                agent=self.name,
            )

        ctx.tool_calls.append(name)
        # No try/except: the registry already raises inside this package's
        # taxonomy (`ToolExecutionError` with transience classified from the
        # tool's own exception), and re-wrapping would only bury the tool name
        # one cause deeper.
        return await self._registry.invoke(
            agent=self.name, tool=name, arguments=dict(arguments or {})
        )

    # ------------------------------------------------------------ internals --

    def _context(self, state: InvestigationState) -> AgentContext:
        """Build the per-invocation context, seeding the budget from the run's spend.

        Seeding matters: `RunBudget` is per *call site* here, but the ceiling it
        enforces is per *investigation*, and a fresh budget every node would let
        a 12-node run spend twelve budgets. The already-spent total lives in the
        checkpointed state, which is the only thing that survives a resume.
        """
        spent = state.get("tokens_spent") or TokenLedger()
        limit = self._agent_settings.token_budget_per_investigation
        if self.name in EXIT_PATH_AGENTS:
            limit = int(limit * EXIT_PATH_OVERDRAFT)
        budget = RunBudget(limit=limit, spent=spent.input_tokens + spent.output_tokens)
        ledger = _MutableLedger()
        router = ModelRouter(
            _LedgerMeteringProvider(self._provider, ledger),
            settings=self._llm_settings,
        )
        return AgentContext(
            investigation_id=state.get("investigation_id", ""),
            tenant_id=state.get("tenant_id", ""),
            trace_id=state.get("trace_id", ""),
            deadline_at=state.get("deadline_at"),
            scratchpad_key=state.get("scratchpad_key", ""),
            budget=budget,
            router=router,
            ledger=ledger,
        )

    def _validate_output(self, output: Any) -> OutputT:
        """Enforce the declared output schema at the node boundary.

        An agent that returns a `dict` "that looks right" is the failure this
        catches: it passes every test that reads one key and breaks the first
        consumer that reads another. Validating here rather than trusting the
        annotation means the guarantee holds for output built by hand as well as
        output parsed from a model.
        """
        if isinstance(output, self.output_model):
            return output  # type: ignore[return-value]
        try:
            return self.output_model.model_validate(output)  # type: ignore[return-value]
        except (PydanticValidationError, ValueError, TypeError) as exc:
            raise StructuredOutputError(
                f"{self.name} returned {type(output).__name__}, which does not validate as "
                f"{self.output_model.__name__}.",
                agent=self.name,
                blocking=self.blocking,
                cause=exc,
            ) from exc
