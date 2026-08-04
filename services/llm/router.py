"""Model tiering, overload shedding and the per-run token budget.

Three logical tiers exist because one model for everything is either too
expensive or too weak: a planner reasons over the whole investigation, a worker
executes one delegated step, and a fast model does classification and reranking
where a wrong answer is cheap to detect. Callers ask for a *tier*; the model id
lives in `LLM_MODEL_PLANNER` / `LLM_MODEL_WORKER` / `LLM_MODEL_FAST` and nowhere
else, so re-tiering a deployment is a config change and never a code change.

Two behaviours here are policy, not plumbing.

**Shed the tier, not the work** (`docs/architecture.md` §7.2). When the provider
rate-limits or reports overload, the router retries the same request one tier
down rather than failing the step. A planner call answered by the worker model
is a worse answer; a planner call that raises is *no* answer, and the
investigation dies holding a half-built plan. The ladder is finite -- the fast
tier has nowhere to fall to -- so this cannot loop.

**Degradation is recorded, never silent** (`docs/observability.md` §8.2). Every
shed is appended to the run's `RunBudget.shed_events`, because a report produced
in degraded mode has to say so; a router that quietly downgraded would make the
same claim with less evidence behind it and no way to tell afterwards.

Why per-run state lives on `RunBudget` rather than on the router: one router
instance serves every concurrent investigation in a worker. Anything remembered
on `self` would be a cross-run leak -- run A's spend cancelling run B's calls,
run B's shed labelling run A's report as degraded. The budget object is created
per investigation and threaded through, which is also exactly the lifetime the
token ceiling has.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any, Final, TypeVar

from backend.core.config import AgentSettings, LLMSettings, get_settings
from backend.core.exceptions import OmniSenseError
from services.llm.provider import (
    BaseModelT,
    LLMError,
    LLMProvider,
    LLMRateLimited,
    LLMResponse,
    LLMSchemaError,
    LLMTimeout,
    MeteredLLMProvider,
)

__all__ = [
    "DEFAULT_PRESSURE_FRACTION",
    "ModelRouter",
    "ModelTier",
    "RunBudget",
    "ShedEvent",
    "TokenBudgetExceeded",
]


_ResultT = TypeVar("_ResultT")
"""Whatever the wrapped call returns -- a response, or a validated model.

Generic rather than `Any` so `complete()` and `structured()` keep their declared
return types through the shed loop; `Any` here would erase them at exactly the
seam where a caller stops being able to see what it is getting.
"""


class ModelTier(enum.StrEnum):
    """What a call needs, expressed as capability rather than as a model id."""

    PLANNER = "planner"
    """Whole-investigation reasoning: decomposition, synthesis, critique."""

    WORKER = "worker"
    """One delegated step. The volume tier -- most calls in a run are these."""

    FAST = "fast"
    """Classification, extraction, reranking. Cheap and disposable."""


_SHED_LADDER: Final[dict[ModelTier, ModelTier | None]] = {
    ModelTier.PLANNER: ModelTier.WORKER,
    ModelTier.WORKER: ModelTier.FAST,
    ModelTier.FAST: None,
}
"""Where each tier falls to under pressure. `None` terminates the descent."""

DEFAULT_PRESSURE_FRACTION: Final = 0.15
"""Remaining-budget fraction below which calls pre-emptively route one tier down.

Chosen so the *end* of a run degrades rather than its middle: the last few steps
of an investigation are usually synthesis over evidence already gathered, which
survives a smaller model far better than the planning that produced it. Set it
to 0.0 to disable the behaviour entirely.
"""

_OVERLOAD_STATUSES: Final = frozenset({429, 500, 502, 503, 504, 529})
"""Provider statuses that mean "try again elsewhere" rather than "you are wrong".

500 is included deliberately: providers return it for transient capacity
problems as well as for genuine bugs, and treating it as fatal would fail a run
for something a different model would have answered.
"""


class TokenBudgetExceeded(OmniSenseError):  # noqa: N818 -- matches the taxonomy it sits beside
    """The investigation spent its `INVESTIGATION_TOKEN_BUDGET`.

    Deliberately **not** an `LLMError`. Generic provider-error handling retries,
    sheds and parks; every one of those is wrong here, because nothing about the
    provider is broken and repeating the call is precisely what must not happen.
    Catching this means "stop and report what you have", which is a different
    branch in the agent runtime and must not be reachable by accident.

    429 with no `Retry-After`: it is a quota, but not one that refills on a
    timer, so it does not derive from `RateLimitedError` -- whose contract
    promises the caller that waiting helps.
    """

    status_code = 429
    code = "token_budget_exhausted"
    default_message = "This investigation exhausted its token budget."


@dataclass(frozen=True, slots=True)
class ShedEvent:
    """One downgrade, kept so the report can admit to it."""

    requested: ModelTier
    served: ModelTier
    reason: str
    """`overload` (the provider refused) or `budget_pressure` (we chose to)."""


@dataclass(slots=True)
class RunBudget:
    """Token ceiling and degradation record for one investigation.

    Created per run and threaded through every call. Mutable on purpose -- it is
    an accumulator -- which is exactly why it must never be stored on the
    router.

    The guard stops the *next* call, not the one that crosses the line. A call's
    cost is unknowable until it returns, so the only alternatives are to
    over-shoot by at most one call or to refuse work on a guess; over-shooting is
    bounded, auditable and does not truncate runs that would have finished.
    """

    limit: int
    spent: int = 0
    unmetered_calls: int = 0
    """Calls whose usage the provider could not report. See `MeteredLLMProvider`.

    Not zero-cost -- unknown-cost. Kept visible so a total that under-reports
    can be recognised as under-reporting rather than trusted.
    """

    shed_events: list[ShedEvent] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: AgentSettings | None = None) -> RunBudget:
        """Build from `INVESTIGATION_TOKEN_BUDGET`."""
        resolved = settings if settings is not None else get_settings().agents
        return cls(limit=resolved.token_budget_per_investigation)

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.spent)

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.limit

    @property
    def remaining_fraction(self) -> float:
        return self.remaining / self.limit if self.limit > 0 else 0.0

    @property
    def degraded(self) -> bool:
        """Whether any call in this run was answered below the tier it asked for."""
        return bool(self.shed_events)

    def check(self) -> None:
        """Raise if there is nothing left to spend. Called before every request."""
        if self.exhausted:
            raise TokenBudgetExceeded(
                f"investigation token budget exhausted: {self.spent} of {self.limit} "
                "tokens spent (INVESTIGATION_TOKEN_BUDGET).",
                details={"limit": self.limit, "spent": self.spent},
            )

    def charge(self, response: LLMResponse) -> None:
        """Record what a call cost.

        Charges `billable_tokens`, so provider-side cache reads do not consume
        the run's budget. They cost roughly a tenth of list price, and counting
        them at par would punish a well-cached prompt for being well cached.
        """
        self.spent += response.billable_tokens

    def record_unmetered(self) -> None:
        self.unmetered_calls += 1

    def record_shed(self, requested: ModelTier, served: ModelTier, reason: str) -> None:
        self.shed_events.append(ShedEvent(requested=requested, served=served, reason=reason))


class ModelRouter:
    """Maps tiers to models, sheds under load, and enforces the run budget.

    Holds no per-run state and is safe to share across concurrent runs.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        settings: LLMSettings | None = None,
        pressure_fraction: float = DEFAULT_PRESSURE_FRACTION,
    ) -> None:
        self._provider = provider
        self._settings = settings if settings is not None else get_settings().llm
        self._pressure_fraction = pressure_fraction
        self._models: dict[ModelTier, str] = {
            ModelTier.PLANNER: self._settings.model_planner,
            ModelTier.WORKER: self._settings.model_worker,
            ModelTier.FAST: self._settings.model_fast,
        }

    def model_for(self, tier: ModelTier) -> str:
        """The configured model id for a tier."""
        return self._models[tier]

    async def complete(
        self,
        *,
        tier: ModelTier,
        prompt: str,
        system: str | None = None,
        budget: RunBudget | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Free-text completion at `tier`, shedding downward under overload."""

        async def call(model: str) -> LLMResponse:
            response = await self._provider.complete(
                prompt=prompt,
                system=system,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if budget is not None:
                budget.charge(response)
            return response

        return await self._dispatch(tier=tier, budget=budget, call=call)

    async def structured(
        self,
        *,
        tier: ModelTier,
        prompt: str,
        schema: type[BaseModelT],
        system: str | None = None,
        budget: RunBudget | None = None,
        max_tokens: int | None = None,
    ) -> BaseModelT:
        """Schema-constrained call at `tier`, metered when the provider allows it."""

        async def call(model: str) -> BaseModelT:
            if isinstance(self._provider, MeteredLLMProvider):
                value, usage = await self._provider.structured_metered(
                    prompt=prompt,
                    schema=schema,
                    system=system,
                    model=model,
                    max_tokens=max_tokens,
                )
                if budget is not None:
                    budget.charge(usage)
                return value

            value = await self._provider.structured(
                prompt=prompt,
                schema=schema,
                system=system,
                model=model,
                max_tokens=max_tokens,
            )
            if budget is not None:
                budget.record_unmetered()
            return value

        return await self._dispatch(tier=tier, budget=budget, call=call)

    # ------------------------------------------------------------ internals --

    async def _dispatch(
        self,
        *,
        tier: ModelTier,
        budget: RunBudget | None,
        call: Callable[[str], Coroutine[Any, Any, _ResultT]],
    ) -> _ResultT:
        """Run `call` at the effective tier, walking the ladder on overload.

        Bounded by the ladder, which is three entries long and terminates, so no
        amount of provider misbehaviour turns this into a spin.
        """
        if budget is not None:
            budget.check()

        requested = tier
        current: ModelTier | None = self._under_pressure(tier, budget)

        while current is not None:
            try:
                return await call(self.model_for(current))
            except Exception as exc:  # noqa: BLE001 -- classification is the point
                if not _is_overload(exc):
                    raise
                nxt = _SHED_LADDER[current]
                if nxt is None:
                    # Nothing cheaper exists. Surfacing the provider's own error
                    # matters: "rate limited at the fast tier" and "rate limited
                    # at the planner" are different incidents.
                    raise
                if budget is not None:
                    budget.record_shed(requested, nxt, "overload")
                current = nxt

        raise AssertionError("unreachable: the shed ladder terminates")

    def _under_pressure(self, tier: ModelTier, budget: RunBudget | None) -> ModelTier:
        """Pre-emptively drop one tier when the run is nearly out of budget."""
        if budget is None or self._pressure_fraction <= 0.0:
            return tier
        if budget.remaining_fraction > self._pressure_fraction:
            return tier
        cheaper = _SHED_LADDER[tier]
        if cheaper is None:
            return tier
        budget.record_shed(tier, cheaper, "budget_pressure")
        return cheaper


def _is_overload(exc: BaseException) -> bool:
    """Whether a failure means "somewhere else might work".

    A timeout counts. It usually means the provider is saturated, and the
    smaller model is both faster and less contended -- which makes shedding the
    single most likely thing to succeed. It is not a free choice: the timed-out
    request may still have been billed, and that is the price of not losing the
    run.

    A schema failure explicitly does not count, and is checked first because it
    *is* an `LLMError`: retrying a prompt the model could not satisfy on a
    weaker model is the least likely repair available. `TokenBudgetExceeded`
    falls through for the same reason -- it is not an `LLMError` at all.
    """
    if isinstance(exc, LLMSchemaError):
        return False
    if isinstance(exc, LLMRateLimited | LLMTimeout):
        return True
    if isinstance(exc, LLMError):
        return exc.provider_status in _OVERLOAD_STATUSES
    return False
