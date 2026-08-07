"""The Planner: decomposes a query into a routable plan.

First node in the graph, and the only one marked `blocking`. That flag is not
caution -- it is a statement about what the rest of the run means without this
node. No plan means no branches to dispatch, no sub-questions for the Critic to
check coverage against, and nothing for the router to count steps from. A run
that continued past a failed Planner would visit whatever nodes the default path
happens to contain and produce a fluent report about a question nobody
decomposed, which is worse than failing: it is a confident answer with no
provenance.

**What the Planner is allowed to do, and what it is not.** Its allowlist is
`search_entities` and `list_available` (`agents/tools/registry.py`). It resolves
which companies and products the query names, and asks which connectors exist.
It does *not* fetch and does *not* retrieve. That boundary is the whole design of
this node: a Planner that could retrieve would answer the question itself, in one
model call, with whatever evidence it happened to find -- and the plan would
become a post-hoc rationalisation of an answer already formed.

**Degradation.** Both tools are optional. If entity search is unavailable the
Planner plans without resolved entities, which produces a slightly vaguer plan
and is entirely survivable; the tool failure is recorded so the Critic can see
that the plan was made blind. Neither tool failing is allowed to fail the node,
because the node is blocking and a blocking node that fails on an optional
dependency turns a degraded graph into a dead investigation.

`docs/agent-system.md` §5.1.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.errors import ToolExecutionError
from agents.planner.schemas import PlannerInput, PlannerOutput
from agents.state import InvestigationState, PlanStep, SubQuestion
from backend.core.logging import get_logger
from models.enums import AgentName, InvestigationStatus
from services.llm.router import ModelTier

__all__ = ["PlannerAgent"]

_log = get_logger(__name__)


class PlannerAgent(BaseAgent[PlannerInput, PlannerOutput]):
    """Turns a natural-language query into an executable plan."""

    name: ClassVar[AgentName] = AgentName.PLANNER
    tier: ClassVar[ModelTier] = ModelTier.PLANNER
    output_model: ClassVar[type[PlannerOutput]] = PlannerOutput
    tools: ClassVar[frozenset[str]] = frozenset({"search_entities", "list_available"})

    blocking: ClassVar[bool] = True
    """See the module docstring. No plan means nothing downstream is meaningful."""

    def build_input(self, state: InvestigationState) -> PlannerInput:
        """Project the entry state. Almost nothing exists yet, by construction."""
        deadline = state.get("deadline_at")
        seconds_remaining: float | None = None
        if deadline is not None:
            from models.base import utcnow

            seconds_remaining = (deadline - utcnow()).total_seconds()

        return PlannerInput(
            query=state["query"],
            tenant_id=state["tenant_id"],
            seconds_remaining=seconds_remaining,
        )

    async def execute(self, request: PlannerInput, ctx: AgentContext) -> PlannerOutput:
        """Resolve context with tools, then produce the plan in one model call.

        Tools first, model second, and the ordering matters: a plan written
        before knowing which connectors exist will schedule a Collector step for
        a source this deployment does not have, and the run discovers that
        several minutes later when the Collector reports nothing.
        """
        connectors = await self._list_connectors(ctx)
        entities = await self._resolve_entities(ctx, request.query)

        enriched = request.model_copy(
            update={"available_connectors": connectors, "known_entities": entities}
        )
        rendered = self.render_prompt(
            ctx,
            query=enriched.query,
            available_connectors=connectors,
            known_entities=entities,
            seconds_remaining=enriched.seconds_remaining,
        )
        return await self.call_model(
            ctx,
            prompt=self._user_prompt(enriched),
            schema=PlannerOutput,
            system=rendered.text,
        )

    def to_delta(self, output: PlannerOutput, state: InvestigationState) -> StateDelta:
        """Write the plan, the sub-questions and the objective.

        Converts the model's `PlannedStep` into the state's `PlanStep` rather
        than storing the model's type. The two are deliberately separate: any
        field added to the state for internal bookkeeping would otherwise become
        something a prompt can set.

        `status` moves to `RUNNING` here rather than in the router, because the
        transition means "a plan exists and work can be dispatched", and this is
        the node that makes that true.
        """
        return {
            "objective": output.objective,
            "plan": [
                PlanStep(
                    id=step.id,
                    description=step.description,
                    agent=step.agent,
                    requires_fresh_data=step.requires_fresh_data,
                    depends_on=list(step.depends_on),
                    rationale=step.rationale,
                )
                for step in output.steps
            ],
            "sub_questions": [
                SubQuestion(id=question.id, question=question.question)
                for question in output.sub_questions
            ],
            "status": InvestigationStatus.RUNNING,
        }

    # ------------------------------------------------------------ internals --

    async def _list_connectors(self, ctx: AgentContext) -> list[str]:
        """Which connectors this deployment has, or an empty list.

        Failure is swallowed and logged rather than raised. This node is
        `blocking`, so raising here would convert "the connector registry is
        briefly unreachable" into "this investigation cannot run" -- and the
        plan is still perfectly makeable without the list, just less specific
        about sources.
        """
        try:
            result = await self.use_tool(ctx, "list_available")
        except ToolExecutionError as error:
            _log.warning("planner.connector_list_unavailable", error=str(error))
            return []
        return _slugs_from(result)

    async def _resolve_entities(self, ctx: AgentContext, query: str) -> list[str]:
        """Entity names the query appears to reference, or an empty list.

        Same degradation reasoning as `_list_connectors`. A plan made without
        resolved entities is vaguer, not wrong.
        """
        try:
            result = await self.use_tool(
                ctx, "search_entities", {"query": query[:200], "limit": 8}
            )
        except ToolExecutionError as error:
            _log.warning("planner.entity_search_unavailable", error=str(error))
            return []
        return _entity_names_from(result)

    def _user_prompt(self, request: PlannerInput) -> str:
        """The turn text. The system prompt carries the instructions.

        The query is interpolated without a fence, and that is correct here and
        nowhere else in this system: this string is the *user's own request*, not
        third-party content. `agents/tools/registry.py` fences tool results
        because those contain scraped text; a user's query is an instruction by
        definition, and fencing it would tell the model to treat the thing it was
        asked to do as data.
        """
        lines = [f"Investigation request: {request.query}", ""]
        if request.known_entities:
            lines.append(f"Entities already in the knowledge graph: {', '.join(request.known_entities)}")
        else:
            lines.append(
                "No entities were resolved from the knowledge graph. Plan without "
                "assuming any entity already exists."
            )
        if request.available_connectors:
            lines.append(f"Connectors available for fresh collection: {', '.join(request.available_connectors)}")
        else:
            lines.append(
                "The connector list is unavailable. Do not mark steps "
                "requires_fresh_data unless the question cannot be answered from "
                "existing evidence."
            )
        if request.seconds_remaining is not None:
            lines.append(
                f"Wall-clock budget remaining: {int(request.seconds_remaining)} seconds. "
                "A plan that cannot finish in the time available is worse than a shorter one."
            )
        lines.append("")
        lines.append("Produce the objective, the ordered steps and the sub-questions.")
        return "\n".join(lines)


def _slugs_from(result: Any) -> list[str]:
    """Pull connector slugs out of whatever shape the tool returned.

    Defensive because the tool result crosses a registry boundary that renders
    and caps it, and because a tool whose payload shape changes should degrade
    the plan rather than raise inside a blocking node.
    """
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
        if isinstance(slug, str) and slug:
            slugs.append(slug)
    return slugs[:64]


def _entity_names_from(result: Any) -> list[str]:
    payload = getattr(result, "data", result)
    entities = getattr(payload, "entities", None)
    if entities is None and isinstance(payload, dict):
        entities = payload.get("entities")
    if not isinstance(entities, (list, tuple)):
        return []
    names: list[str] = []
    for item in entities:
        name = getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names[:32]
