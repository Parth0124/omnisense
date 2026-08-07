"""The Collector: dispatches connector syncs for the steps that need fresh data.

The only agent that causes outbound traffic to third parties, and therefore the
only one whose *input schema* is a security control. See
`agents/collector/schemas.py`: connectors are named by slug, never by URL, so a
prompt-injected instruction arriving in scraped content cannot redirect a fetch
at an attacker's host. That is `docs/security-and-privacy.md` §8.2, made
structural.

**Not blocking, and that is a considered choice.** A run whose collection fails
still has whatever is already in the corpus, and answering from a slightly stale
corpus with the staleness stated is a legitimate and useful result. Failing the
run instead would mean one rate-limited third-party API takes down every
investigation that touches it. So failures are recorded as `CollectionResult`
entries with their errors and the run continues -- and because they are in the
state, the Critic sees them and the report says which sources were unavailable.

**Dispatch is concurrent and bounded.** Connectors are independent and each one
is minutes of wall clock; running them serially would make an eight-source
collection eight times slower for no benefit. The bound exists because the
connectors share a process-wide HTTP pool and a rate limiter, and unbounded
concurrency converts a throughput gain into a queue of timeouts.

`docs/agent-system.md` §5.2.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Final

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.collector.schemas import (
    CollectionRequest,
    CollectorInput,
    CollectorOutput,
    CollectorPlan,
)
from agents.errors import ToolExecutionError
from agents.state import CollectionResult, InvestigationState
from backend.core.logging import get_logger
from models.enums import AgentName
from services.llm.router import ModelTier

__all__ = ["MAX_CONCURRENT_SYNCS", "CollectorAgent"]

_log = get_logger(__name__)

MAX_CONCURRENT_SYNCS: Final = 3
"""Connectors dispatched at once.

Three rather than "all of them": the connectors share one HTTP pool and one
rate-limiter budget, so beyond a small number the extra concurrency converts
into queued timeouts and each sync reports failure having done nothing.
"""


class CollectorAgent(BaseAgent[CollectorInput, CollectorOutput]):
    """Chooses connectors, dispatches them, and reports what came back."""

    name: ClassVar[AgentName] = AgentName.COLLECTOR
    tier: ClassVar[ModelTier] = ModelTier.WORKER
    output_model: ClassVar[type[CollectorOutput]] = CollectorOutput
    tools: ClassVar[frozenset[str]] = frozenset({"list_available", "fetch", "sync_status"})

    def build_input(self, state: InvestigationState) -> CollectorInput:
        """Project the plan's fresh-data steps.

        Only those steps. A Collector that saw the whole plan would collect for
        analysis steps that were meant to run off existing evidence, which is how
        a run that needed one source ends up dispatching six.
        """
        plan = state.get("plan") or []
        deadline = state.get("deadline_at")
        seconds_remaining: float | None = None
        if deadline is not None:
            from models.base import utcnow

            seconds_remaining = (deadline - utcnow()).total_seconds()

        return CollectorInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            fresh_data_steps=[
                step.description for step in plan if getattr(step, "requires_fresh_data", False)
            ][:16],
            seconds_remaining=seconds_remaining,
        )

    async def execute(self, request: CollectorInput, ctx: AgentContext) -> CollectorOutput:
        """Ask what exists, let the model choose, dispatch, report."""
        available = await self._list_connectors(ctx)
        if not available:
            # Nothing to dispatch to. Returning an empty result rather than
            # raising: "this deployment has no connectors configured" is a
            # legitimate state, and the report should say the corpus was not
            # refreshed rather than the run failing.
            _log.info("collector.no_connectors_available")
            return CollectorOutput(rationale="no connectors are configured in this deployment")

        enriched = request.model_copy(update={"available_connectors": available})
        plan = await self._choose(enriched, ctx)

        # The model can only name a slug (schema-enforced), and this is the
        # second gate: a slug that is well-formed but not one this deployment
        # actually has is dropped here rather than dispatched. Without it, a
        # hallucinated-but-valid-looking slug reaches the connector registry.
        known = set(available)
        requests = [item for item in plan.requests if item.connector_slug in known]
        hallucinated = [
            item.connector_slug for item in plan.requests if item.connector_slug not in known
        ]
        if hallucinated:
            _log.warning("collector.unknown_slugs_dropped", slugs=sorted(set(hallucinated)))

        results = await self._dispatch(requests, ctx)
        return CollectorOutput(
            dispatched=len(results),
            emitted=sum(result.emitted for result in results),
            failures=[
                f"{result.connector_slug}: {result.error}"
                for result in results
                if result.error is not None
            ][: len(requests) or 1],
            skipped=sorted({*plan.skipped, *hallucinated})[:32],
            rationale=plan.rationale,
        )

    def to_delta(self, output: CollectorOutput, state: InvestigationState) -> StateDelta:
        """Append this node's collection results.

        `collection_results` carries an `operator.add` reducer, so this returns
        the *increment* -- the results from this invocation only. Returning the
        accumulated list would make the reducer add it to itself and double every
        entry.
        """
        return {
            "collection_results": [
                CollectionResult(
                    connector_slug=failure.split(":", 1)[0],
                    run_id=state["investigation_id"],
                    error=failure.split(":", 1)[1].strip() if ":" in failure else failure,
                )
                for failure in output.failures
            ]
            + (
                [
                    CollectionResult(
                        connector_slug="__aggregate__",
                        run_id=state["investigation_id"],
                        emitted=output.emitted,
                    )
                ]
                if output.emitted
                else []
            )
        }

    # ------------------------------------------------------------ internals --

    async def _list_connectors(self, ctx: AgentContext) -> list[str]:
        try:
            result = await self.use_tool(ctx, "list_available")
        except ToolExecutionError as error:
            _log.warning("collector.connector_list_unavailable", error=str(error))
            return []
        return _slugs_from(result)

    async def _choose(self, request: CollectorInput, ctx: AgentContext) -> CollectorPlan:
        rendered = self.render_prompt(
            ctx,
            query=request.query,
            objective=request.objective,
            available_connectors=request.available_connectors,
            fresh_data_steps=request.fresh_data_steps,
        )
        lines = [
            f"Investigation: {request.query}",
            f"Objective: {request.objective}" if request.objective else "",
            "",
            f"Connectors available: {', '.join(request.available_connectors)}",
            "",
            "Plan steps that asked for fresh data:",
            *(f"- {step}" for step in request.fresh_data_steps),
            "",
            "Choose which connectors to run. Name each by its slug exactly as "
            "listed above. Do not name a source that is not in the list.",
        ]
        return await self.call_model(
            ctx,
            prompt="\n".join(line for line in lines if line != ""),
            schema=CollectorPlan,
            system=rendered.text,
        )

    async def _dispatch(
        self, requests: list[CollectionRequest], ctx: AgentContext
    ) -> list[CollectionResult]:
        """Run the chosen connectors concurrently, bounded, never raising.

        Each sync is wrapped individually. One connector's failure must not
        cancel the others -- they are independent sources, and losing five
        because the sixth was rate-limited is the opposite of resilience.
        """
        if not requests:
            return []

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SYNCS)

        async def run_one(item: CollectionRequest) -> CollectionResult:
            async with semaphore:
                try:
                    result = await self.use_tool(
                        ctx,
                        "fetch",
                        {"connector_slug": item.connector_slug, "query": item.query}
                        if item.query
                        else {"connector_slug": item.connector_slug},
                    )
                except ToolExecutionError as error:
                    _log.warning(
                        "collector.sync_failed",
                        connector=item.connector_slug,
                        error=str(error),
                    )
                    return CollectionResult(
                        connector_slug=item.connector_slug,
                        run_id=ctx.investigation_id,
                        error=str(error)[:500],
                    )
                return CollectionResult(
                    connector_slug=item.connector_slug,
                    run_id=ctx.investigation_id,
                    emitted=_emitted_from(result),
                )

        return list(await asyncio.gather(*(run_one(item) for item in requests)))


def _slugs_from(result: Any) -> list[str]:
    payload = getattr(result, "data", result)
    connectors = getattr(payload, "connectors", None)
    if connectors is None and isinstance(payload, dict):
        connectors = payload.get("connectors")
    if not isinstance(connectors, (list, tuple)):
        return []
    slugs: list[str] = []
    for item in connectors:
        slug = getattr(item, "slug", None)
        if slug is None and isinstance(item, dict):
            slug = item.get("slug")
        enabled = getattr(item, "enabled", True)
        if enabled is None and isinstance(item, dict):
            enabled = item.get("enabled", True)
        # A disabled connector is filtered here rather than at dispatch: offering
        # it to the model produces a plan that names it, a dispatch that fails,
        # and a report explaining an absence that was configuration all along.
        if isinstance(slug, str) and slug and enabled:
            slugs.append(slug)
    return slugs[:64]


def _emitted_from(result: Any) -> int:
    payload = getattr(result, "data", result)
    for attribute in ("emitted", "count", "records"):
        value = getattr(payload, attribute, None)
        if value is None and isinstance(payload, dict):
            value = payload.get(attribute)
        if isinstance(value, int) and not isinstance(value, bool):
            return max(0, value)
    return 0
