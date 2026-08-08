"""The Strategy agent: turns insights into recommendations, or declines to.

The only node that asks the reader to act, which makes declining a first-class
outcome rather than a failure. `StrategyOutput` requires a `withheld_reason` when
the recommendation list is empty, and this agent uses it: an investigation whose
evidence supports observations but not actions should say so. The alternative --
producing three plausible recommendations because the section expects three -- is
how an intelligence system becomes a generator of confident advice uncorrelated
with what it found.

**Every recommendation descends from a stated insight.** Enforced twice: the
schema requires `based_on_insight_ids`, and this module drops any recommendation
whose parent ids do not appear in the run's insight set. The second check is what
catches the model inventing a plausible-looking id, which it does, and which
would otherwise produce an action whose provenance chain silently ends nowhere.

**Unanswered questions constrain the advice.** They are passed into the prompt
because a recommendation that depends on a question the evidence could not answer
is exactly the one to hedge or withhold, and the model cannot know which those
are unless told.

`docs/agent-system.md` §5.8.
"""

from __future__ import annotations

from typing import ClassVar

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.state import InvestigationState
from agents.strategy.schemas import (
    MAX_RECOMMENDATIONS,
    Recommendation,
    StrategyInput,
    StrategyOutput,
)
from backend.core.logging import get_logger
from models.enums import AgentName
from services.llm.router import ModelTier

__all__ = ["StrategyAgent"]

_log = get_logger(__name__)


class StrategyAgent(BaseAgent[StrategyInput, StrategyOutput]):
    """Produces actionable recommendations, each traceable to an insight."""

    name: ClassVar[AgentName] = AgentName.STRATEGY
    tier: ClassVar[ModelTier] = ModelTier.PLANNER
    output_model: ClassVar[type[StrategyOutput]] = StrategyOutput
    tools: ClassVar[frozenset[str]] = frozenset({"hybrid_search", "aggregate"})

    def build_input(self, state: InvestigationState) -> StrategyInput:
        return StrategyInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            insights=list(state.get("insights") or [])[:12],
            trends=list(state.get("trends") or [])[:10],
            forecasts=list(state.get("forecasts") or [])[:6],
            competitor_view=state.get("competitor_view"),
        )

    async def execute(self, request: StrategyInput, ctx: AgentContext) -> StrategyOutput:
        if not request.insights:
            # No insights means no chain of provenance to hang an action on.
            # Withholding explicitly rather than returning nothing, so the report
            # can say why the section is absent.
            return StrategyOutput(
                withheld_reason=(
                    "No insights were produced, so there is nothing to recommend "
                    "from. A recommendation must descend from a stated insight."
                )
            )

        rendered = self.render_prompt(
            ctx, query=request.query, objective=request.objective
        )
        produced = await self.call_model(
            ctx,
            prompt=self._build_prompt(request),
            schema=StrategyOutput,
            system=rendered.text,
        )

        known = {
            str(insight.get("id"))
            for insight in request.insights
            if isinstance(insight, dict) and insight.get("id")
        }
        kept: list[Recommendation] = []
        for recommendation in produced.recommendations:
            resolvable = [
                insight_id
                for insight_id in recommendation.based_on_insight_ids
                if insight_id in known
            ]
            if not resolvable:
                _log.warning(
                    "strategy.dropped_unresolvable_provenance",
                    recommendation_id=recommendation.id,
                    cited=recommendation.based_on_insight_ids[:5],
                )
                continue
            kept.append(recommendation.model_copy(update={"based_on_insight_ids": resolvable}))

        if not kept:
            return StrategyOutput(
                summary=produced.summary,
                withheld_reason=(
                    produced.withheld_reason
                    or "Every proposed action cited an insight that does not exist in "
                    "this run, so none could be traced back to evidence."
                ),
            )

        return StrategyOutput(
            recommendations=kept[:MAX_RECOMMENDATIONS],
            summary=produced.summary,
        )

    def to_delta(self, output: StrategyOutput, state: InvestigationState) -> StateDelta:
        """`recommendations` is `operator.add`-reduced: return the increment."""
        return {
            "recommendations": [item.model_dump(mode="json") for item in output.recommendations]
        }

    # ------------------------------------------------------------ internals --

    def _build_prompt(self, request: StrategyInput) -> str:
        lines = [
            f"Investigation: {request.query}",
            f"Objective: {request.objective}" if request.objective else "",
            "",
            "Insights established by this investigation:",
        ]
        for insight in request.insights:
            if not isinstance(insight, dict):
                continue
            lines.append(
                f"- [{insight.get('id')}] ({insight.get('kind')}, "
                f"confidence {insight.get('confidence', 0):.2f}) {insight.get('statement')}"
            )
        lines.append("")

        if request.trends:
            lines.append("Measured trends:")
            lines.extend(
                f"- {trend.get('topic')}: {trend.get('direction')}"
                for trend in request.trends
                if isinstance(trend, dict)
            )
            lines.append("")
        if request.unanswered_questions:
            lines.append(
                "The evidence could NOT answer these questions. Any recommendation "
                "depending on them must be hedged or withheld:"
            )
            lines.extend(f"- {question}" for question in request.unanswered_questions)
            lines.append("")

        lines.append(
            "Recommend actions. Every one must cite the insight ids it rests on, "
            "state what must be true for it to be right, and state what happens "
            "if it is wrong. Use 'monitor' where watching is the correct call -- "
            "it is a real recommendation, not a filler. If the evidence supports "
            "observations but not actions, recommend nothing and say why."
        )
        return "\n".join(line for line in lines if line != "")
