"""The Insight agent: synthesises evidence into claims that carry their support.

Reasons over material already gathered and retrieves nothing. Its allowlist is
`fetch_passage` and `find_paths` -- it may re-read a passage it was given and
trace a graph connection, but it cannot search. That boundary is deliberate: an
Insight agent that could retrieve would answer whatever question it found
interesting rather than the one the plan asked, and the sub-question coverage the
Critic checks against would stop meaning anything.

**Every insight is validated against the evidence that exists.** The schema
requires `signal_ids`; this module additionally drops any insight citing a signal
that is not in the run's evidence set. That second check is the one that matters,
because a model will cite a plausible-looking id it never saw -- and a citation
that points at nothing is worse than no citation, since it survives every check
that does not actually resolve it.

**Unanswered questions are output, not omission.** A report that silently skips
two of six sub-questions looks identical to one that answered all six. Naming
them is what lets the Critic and the reader see the difference.

`docs/agent-system.md` §5.7.
"""

from __future__ import annotations

from typing import ClassVar

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.insight.schemas import (
    MAX_INSIGHTS,
    Insight,
    InsightInput,
    InsightOutput,
)
from agents.state import InvestigationState
from backend.core.logging import get_logger
from models.enums import AgentName
from services.llm.router import ModelTier

__all__ = ["InsightAgent"]

_log = get_logger(__name__)


class InsightAgent(BaseAgent[InsightInput, InsightOutput]):
    """Turns gathered evidence into cited claims."""

    name: ClassVar[AgentName] = AgentName.INSIGHT
    tier: ClassVar[ModelTier] = ModelTier.PLANNER
    """Planner tier, not worker. Synthesis across a hundred references is the
    reasoning-hardest step in the run, and shedding it to a smaller model
    produces insights that restate individual passages instead of connecting
    them -- which is the entire value of the node."""

    output_model: ClassVar[type[InsightOutput]] = InsightOutput
    tools: ClassVar[frozenset[str]] = frozenset({"fetch_passage", "find_paths"})

    def build_input(self, state: InvestigationState) -> InsightInput:
        questions = state.get("sub_questions") or []
        evidence = state.get("evidence") or []
        return InsightInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            sub_questions=[question.question for question in questions][:8],
            sub_question_ids=[question.id for question in questions][:8],
            evidence_ids=[ref.signal_id for ref in evidence][:60],
            trends=list(state.get("trends") or [])[:10],
            forecasts=list(state.get("forecasts") or [])[:6],
            competitor_view=state.get("competitor_view"),
            prior_critique=state.get("critique"),
        )

    async def execute(self, request: InsightInput, ctx: AgentContext) -> InsightOutput:
        if not request.evidence_ids:
            # No evidence means no insight. Producing one anyway would be the
            # model reasoning from its training data about the industry, which
            # is exactly the output this system exists not to produce.
            _log.warning("insight.no_evidence")
            return InsightOutput(
                unanswered_sub_questions=list(request.sub_questions),
                notes=(
                    "No evidence was retrieved, so no insight is offered. Every "
                    "claim in this system must attach to a signal."
                ),
            )

        rendered = self.render_prompt(
            ctx, query=request.query, objective=request.objective
        )
        produced = await self.call_model(
            ctx,
            prompt=self._build_prompt(request),
            schema=InsightOutput,
            system=rendered.text,
        )

        known = set(request.evidence_ids)
        kept: list[Insight] = []
        for insight in produced.insights:
            resolvable = [signal_id for signal_id in insight.signal_ids if signal_id in known]
            if not resolvable:
                # A citation that resolves to nothing is worse than none: it
                # survives every check short of actually looking the signal up,
                # and it makes an unsupported claim look sourced.
                _log.warning(
                    "insight.dropped_unresolvable_citations",
                    insight_id=insight.id,
                    cited=insight.signal_ids[:5],
                )
                continue
            kept.append(insight.model_copy(update={"signal_ids": resolvable}))

        answered = {
            question_id for insight in kept for question_id in insight.sub_question_ids
        }
        unanswered = [
            question
            for question, question_id in zip(
                request.sub_questions, request.sub_question_ids, strict=False
            )
            if question_id not in answered
        ]

        return InsightOutput(
            insights=kept[:MAX_INSIGHTS],
            unanswered_sub_questions=unanswered[:8],
            notes=produced.notes,
        )

    def to_delta(self, output: InsightOutput, state: InvestigationState) -> StateDelta:
        """`insights` is `operator.add`-reduced: return this pass's increment only."""
        return {"insights": [insight.model_dump(mode="json") for insight in output.insights]}

    # ------------------------------------------------------------ internals --

    def _build_prompt(self, request: InsightInput) -> str:
        lines = [
            f"Investigation: {request.query}",
            f"Objective: {request.objective}" if request.objective else "",
            "",
            "Questions this investigation must answer:",
            *(
                f"- [{question_id}] {question}"
                for question, question_id in zip(
                    request.sub_questions, request.sub_question_ids, strict=False
                )
            ),
            "",
            f"Evidence available: {len(request.evidence_ids)} signals.",
            f"Signal ids you may cite: {', '.join(request.evidence_ids[:60])}",
            "",
        ]
        if request.trends:
            lines.append("Measured trends:")
            lines.extend(
                f"- {trend.get('topic')}: {trend.get('direction')} "
                f"({trend.get('observation_count', 0)} observations)"
                for trend in request.trends
                if isinstance(trend, dict)
            )
            lines.append("")
        if request.forecasts:
            lines.append("Projections:")
            lines.extend(
                f"- {item.get('subject')}: {item.get('method')}"
                for item in request.forecasts
                if isinstance(item, dict)
            )
            lines.append("")
        if request.competitor_view:
            names = [
                entry.get("name")
                for entry in (request.competitor_view.get("competitors") or [])
                if isinstance(entry, dict)
            ]
            if names:
                lines.append(f"Competitors identified: {', '.join(str(n) for n in names[:12])}")
                lines.append("")
        if request.prior_critique:
            lines.append(
                "This is a revision. The previous pass was criticised as follows; "
                "address it rather than restating the same claims:"
            )
            lines.append(str(request.prior_critique.get("summary", ""))[:1000])
            lines.append("")

        lines.append(
            "Produce insights. Cite only signal ids from the list above -- a "
            "citation that does not resolve makes an unsupported claim look "
            "sourced. Mark a claim causal only with two independent signals. "
            "Where the evidence contradicts itself, record both and link them "
            "with `contradicts` rather than choosing. Name every sub-question "
            "the evidence cannot answer."
        )
        return "\n".join(line for line in lines if line != "")
