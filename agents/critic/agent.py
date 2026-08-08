"""The Critic: verifies citations mechanically, then judges what remains.

Two passes, and the ordering is the design.

**The mechanical pass runs first and does not involve a model.** Every citation
in every insight and recommendation is resolved against the run's evidence set,
and every quote is verified against its stored source through `resolve_citation`.
These are facts, not opinions: either the signal exists or it does not, either
the quoted sentence appears in the document or it does not. Asking a model to
judge them wastes tokens and gets the answer wrong -- a model reading a
plausible-looking citation has no way to know it is fabricated, and will say it
looks fine.

**The judgement pass runs second, and is told what the mechanical pass found.**
Coverage, overstatement, contradiction and source concentration are genuine
matters of degree that need reading. Running them second means the model is
judging a report whose factual defects are already identified, so it spends its
attention on the questions only reading can answer.

**Source concentration is computed, not judged.** Forty citations that all trace
to one signal look like abundant evidence in every other check. The ratio is
arithmetic, so it is arithmetic here rather than a thing the model is asked to
notice -- which it does not, reliably, because the citation list looks long.

`docs/agent-system.md` §5.9 and §13.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any, ClassVar, Final

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.critic.schemas import (
    MAX_FINDINGS,
    CriticInput,
    CriticOutput,
    Finding,
    FindingKind,
    Severity,
)
from agents.errors import ToolExecutionError
from agents.state import InvestigationState
from backend.core.logging import get_logger
from models.enums import AgentName
from services.llm.router import ModelTier

__all__ = ["SOURCE_CONCENTRATION_THRESHOLD", "CriticAgent"]

_log = get_logger(__name__)

SOURCE_CONCENTRATION_THRESHOLD: Final = 0.5
"""Above this share from one signal, the evidence base is concentrated.

Half. Chosen because a conclusion where one document supplies most of the support
is a conclusion about that document -- and because a lower threshold fires on
every focused investigation, which trains readers to ignore the finding.
"""

MAX_QUOTES_VERIFIED: Final = 30
"""Quotes checked per pass.

Each is a store read. Thirty covers the citations a report actually renders;
verifying three hundred would triple the node's latency to check quotes nobody
will see. The cap is logged when it bites -- a silent cap here would read as "all
citations verified".
"""


class CriticAgent(BaseAgent[CriticInput, CriticOutput]):
    """Mechanically verifies citations, then judges the reasoning."""

    name: ClassVar[AgentName] = AgentName.CRITIC
    tier: ClassVar[ModelTier] = ModelTier.PLANNER
    output_model: ClassVar[type[CriticOutput]] = CriticOutput
    tools: ClassVar[frozenset[str]] = frozenset({"fetch_passage", "resolve_citation"})

    def build_input(self, state: InvestigationState) -> CriticInput:
        questions = state.get("sub_questions") or []
        evidence = state.get("evidence") or []
        return CriticInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            insights=list(state.get("insights") or [])[:12],
            recommendations=list(state.get("recommendations") or [])[:8],
            trends=list(state.get("trends") or [])[:10],
            report=state.get("report"),
            evidence_ids=[ref.signal_id for ref in evidence][:60],
            sub_questions=[q.question for q in questions][:8],
            unanswered_sub_questions=[q.question for q in questions if not q.answered][:8],
            revision_count=state.get("revision_count", 0),
        )

    async def execute(self, request: CriticInput, ctx: AgentContext) -> CriticOutput:
        mechanical = await self._mechanical_pass(request, ctx)
        judged = await self._judgement_pass(request, mechanical, ctx)

        findings = [*mechanical, *judged.findings][:MAX_FINDINGS]
        blocking = [f for f in findings if f.severity is Severity.BLOCKING]

        # Approval is recomputed here rather than trusted from the model. The
        # schema already forbids approving over a blocking finding, but the model
        # only saw the findings *it* produced -- the mechanical ones are merged
        # afterwards, so its approval was made without them.
        approved = judged.approved and not blocking
        confidence = judged.confidence
        if blocking:
            # A report with a fabricated citation is not "0.6 confident". Capping
            # rather than zeroing, because the non-defective parts may still be
            # sound and the number feeds a badge, not a binary.
            confidence = min(confidence, 0.25)

        return CriticOutput(
            findings=findings,
            confidence=confidence,
            summary=judged.summary,
            approved=approved,
        )

    def to_delta(self, output: CriticOutput, state: InvestigationState) -> StateDelta:
        """Write the critique and append it to the history.

        `critique` is the single-writer current verdict the router reads;
        `critique_history` is `operator.add`-reduced and keeps every pass. Both
        exist because a run that revised three times should show what changed --
        a single overwritten verdict makes an improving report and a stuck one
        look identical.
        """
        payload = output.model_dump(mode="json")
        return {
            "critique": payload,
            "critique_history": [payload],
            "confidence": output.confidence,
        }

    # ------------------------------------------------------------ internals --

    async def _mechanical_pass(
        self, request: CriticInput, ctx: AgentContext
    ) -> list[Finding]:
        """Resolve every citation. Facts only -- no model involved."""
        findings: list[Finding] = []
        known = set(request.evidence_ids)

        cited: list[tuple[str, str]] = []
        for insight in request.insights:
            if isinstance(insight, dict):
                for signal_id in insight.get("signal_ids") or []:
                    cited.append((str(insight.get("id", "?")), str(signal_id)))

        # 1. Citations that do not resolve to anything in the run's evidence.
        for target, signal_id in cited:
            if signal_id not in known:
                findings.append(
                    Finding(
                        kind=FindingKind.BROKEN_CITATION,
                        severity=Severity.BLOCKING,
                        target=target,
                        detail=(
                            f"Signal {signal_id!r} is cited but is not in this run's "
                            "evidence set. A citation that resolves to nothing makes "
                            "an unsupported claim look sourced."
                        ),
                        signal_ids=[signal_id],
                    )
                )

        # 2. Source concentration -- arithmetic, not judgement.
        if cited:
            counts = Counter(signal_id for _, signal_id in cited)
            top_signal, top_count = counts.most_common(1)[0]
            share = top_count / len(cited)
            if share > SOURCE_CONCENTRATION_THRESHOLD and len(cited) >= 4:
                findings.append(
                    Finding(
                        kind=FindingKind.SOURCE_CONCENTRATION,
                        severity=Severity.MAJOR,
                        target="evidence base",
                        detail=(
                            f"{share:.0%} of citations trace to a single signal "
                            f"({top_signal}). A conclusion resting mostly on one "
                            "document is a conclusion about that document."
                        ),
                        signal_ids=[top_signal],
                    )
                )

        # 3. Quote verification against the stored source.
        findings.extend(await self._verify_quotes(request, ctx))

        # 4. Coverage -- also arithmetic.
        if request.unanswered_sub_questions:
            findings.append(
                Finding(
                    kind=FindingKind.MISSING_COVERAGE,
                    severity=Severity.MAJOR
                    if len(request.unanswered_sub_questions) > 1
                    else Severity.MINOR,
                    target="sub-question coverage",
                    detail=(
                        f"{len(request.unanswered_sub_questions)} planned question(s) "
                        "were not answered by the evidence: "
                        + "; ".join(request.unanswered_sub_questions[:4])
                    ),
                )
            )

        # 5. Degraded retrieval lowers what the evidence can support.
        if request.degraded_backends:
            findings.append(
                Finding(
                    kind=FindingKind.OVERSTATED_CONFIDENCE,
                    severity=Severity.MINOR,
                    target="retrieval",
                    detail=(
                        "Retrieval ran degraded ("
                        + ", ".join(request.degraded_backends)
                        + "), so recall is lower than normal and absence of evidence "
                        "is weaker evidence of absence than usual."
                    ),
                )
            )

        return findings

    async def _verify_quotes(self, request: CriticInput, ctx: AgentContext) -> list[Finding]:
        """Check each quoted span against its stored source.

        The failure this catches is the one nothing else can: a model that
        paraphrases while believing it is quoting. It produces this constantly
        and reports it never, and the paraphrase reads perfectly.
        """
        quotes: list[tuple[str, str, str]] = []
        for insight in request.insights:
            if not isinstance(insight, dict):
                continue
            quote = insight.get("quote")
            signal_ids = insight.get("signal_ids") or []
            if isinstance(quote, str) and quote and signal_ids:
                quotes.append((str(insight.get("id", "?")), str(signal_ids[0]), quote))

        if len(quotes) > MAX_QUOTES_VERIFIED:
            # Logged rather than silent: a truncated verification that reports
            # nothing reads as "all citations verified".
            _log.warning(
                "critic.quote_verification_capped",
                total=len(quotes),
                verified=MAX_QUOTES_VERIFIED,
            )
            quotes = quotes[:MAX_QUOTES_VERIFIED]

        semaphore = asyncio.Semaphore(4)

        async def verify(target: str, signal_id: str, quote: str) -> Finding | None:
            async with semaphore:
                try:
                    result = await self.use_tool(
                        ctx, "resolve_citation", {"signal_id": signal_id, "quote": quote}
                    )
                except ToolExecutionError as error:
                    # Unverifiable is not the same as false. A store that could
                    # not be reached must not turn every citation into a
                    # fabrication finding and block the whole report.
                    _log.warning(
                        "critic.quote_unverifiable", signal_id=signal_id, error=str(error)
                    )
                    return None
                if _is_verified(result):
                    return None
                return Finding(
                    kind=FindingKind.MISQUOTE,
                    severity=Severity.BLOCKING,
                    target=target,
                    detail=(
                        f"The quoted span does not appear in signal {signal_id}. A "
                        "paraphrase presented as a quotation points a reader at a "
                        "sentence nobody wrote."
                    ),
                    signal_ids=[signal_id],
                )

        checked = await asyncio.gather(*(verify(*item) for item in quotes))
        return [finding for finding in checked if finding is not None]

    async def _judgement_pass(
        self, request: CriticInput, mechanical: list[Finding], ctx: AgentContext
    ) -> CriticOutput:
        """Ask the model the questions only reading can answer."""
        rendered = self.render_prompt(
            ctx, query=request.query, objective=request.objective
        )
        lines = [
            f"Investigation: {request.query}",
            f"Objective: {request.objective}" if request.objective else "",
            "",
        ]
        if mechanical:
            lines.append(
                "A mechanical verification pass already found the following. Do not "
                "repeat these; judge what they imply for the report's overall "
                "trustworthiness:"
            )
            lines.extend(
                f"- [{f.severity.value}] {f.kind.value}: {f.detail}" for f in mechanical
            )
            lines.append("")

        lines.append("Insights under review:")
        for insight in request.insights:
            if isinstance(insight, dict):
                lines.append(
                    f"- [{insight.get('id')}] ({insight.get('kind')}, "
                    f"confidence {insight.get('confidence', 0):.2f}) "
                    f"{insight.get('statement')}"
                )
        lines.append("")

        if request.recommendations:
            lines.append("Recommendations under review:")
            lines.extend(
                f"- [{item.get('id')}] ({item.get('urgency')}) {item.get('action')}"
                for item in request.recommendations
                if isinstance(item, dict)
            )
            lines.append("")

        lines.append(
            "Judge the reasoning: unsupported claims, overstated confidence, "
            "contradictions between insights, and bias in framing. Do not report "
            "broken citations or misquotes -- those were checked mechanically. "
            "Give an overall confidence and state whether this can be published."
        )
        if request.is_final_pass:
            lines.append(
                "This is the final pass; the run has no revisions left. Report "
                "honestly -- the findings will be published alongside the report "
                "rather than triggering another attempt."
            )

        return await self.call_model(
            ctx,
            prompt="\n".join(line for line in lines if line != ""),
            schema=CriticOutput,
            system=rendered.text,
        )


def _is_verified(result: Any) -> bool:
    """Read the verification outcome, defaulting to *unverified*.

    The default direction is the whole point. A payload shape this cannot read
    must not be treated as a passing check -- that would turn a parsing bug into
    silent approval of every citation in the system.
    """
    data = getattr(result, "data", result)
    for attribute in ("verified", "is_verified", "matched"):
        value = getattr(data, attribute, None)
        if value is None and isinstance(data, dict):
            value = data.get(attribute)
        if isinstance(value, bool):
            return value
    outcome = getattr(data, "outcome", None)
    if outcome is None and isinstance(data, dict):
        outcome = data.get("outcome")
    if isinstance(outcome, str):
        return outcome.casefold() in {"verified", "exact", "match"}
    return False
