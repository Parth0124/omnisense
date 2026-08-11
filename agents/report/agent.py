"""The Report agent: renders the finished document, gaps included.

Last node in the graph. Every number it prints must already be in the state
(`docs/agent-system.md` §5.10) -- it has no analysis tools, only
`fetch_passage` and `resolve_citation`, because by this point synthesis is over
and the only legitimate reason to touch a store is to check a quotation before
printing it.

**The gaps section is assembled here, not written by the model.** It is built
from the Critic's findings, the unanswered sub-questions, the degraded retrieval
backends and the failed collections -- all of which are facts already in the
state. A model asked to write its own limitations section writes a graceful
paragraph that omits the most damaging one, every time, and there is no prompt
that reliably fixes that. Assembling it from state makes omission impossible.

**Confidence comes from the Critic, not from here.** An author scoring their own
work grades generously, and the Report agent has just spent a model call
constructing the most persuasive possible presentation of the findings. Taking
the number from the node whose job was to find fault is the only version of this
that means anything.

**Citations are filtered against the run's evidence before rendering.** A claim
whose citations do not resolve is dropped rather than printed uncited, because a
sentence in a finished report is the last place a fabricated reference can still
be caught.

`docs/agent-system.md` §5.10.
"""

from __future__ import annotations

from typing import ClassVar

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.report.schemas import (
    MAX_SECTIONS,
    ConfidenceBand,
    ReportClaim,
    ReportInput,
    ReportOutput,
    ReportSection,
    SectionKind,
)
from agents.state import InvestigationState
from backend.core.logging import get_logger
from models.enums import AgentName, InvestigationStatus
from services.llm.router import ModelTier

__all__ = ["ReportAgent"]

_log = get_logger(__name__)


class ReportAgent(BaseAgent[ReportInput, ReportOutput]):
    """Renders the document. Adds no findings and invents no numbers."""

    name: ClassVar[AgentName] = AgentName.REPORT
    tier: ClassVar[ModelTier] = ModelTier.PLANNER
    output_model: ClassVar[type[ReportOutput]] = ReportOutput
    tools: ClassVar[frozenset[str]] = frozenset({"fetch_passage", "resolve_citation"})

    def _nothing_to_report(self, request: ReportInput, gaps: list[str]) -> ReportOutput:
        """A valid report saying that nothing was found.

        `ReportOutput.sections` requires at least one section, which is right --
        a report with no sections is not a report. But it made "there is nothing
        to report" *inexpressible*: with no evidence, no insights and no
        recommendations, the model had nothing to write a section about, returned
        an empty list, and validation rejected every attempt. The investigation
        then finished `completed` with no report at all, which tells the reader
        nothing about why.

        Built here rather than asked of the model, for the same reason
        `InsightAgent` short-circuits on the same condition: the honest content of
        this report is already known, and asking a model to write it invites it to
        fill the space with plausible prose about a corpus it never read. It also
        makes the outcome deterministic, where the model version failed or
        succeeded depending on whether it decided to invent a section that run.

        Confidence is 0.0 and the only section is `GAPS`, so nothing downstream
        can mistake this for a finding.
        """
        _log.warning(
            "report.nothing_to_report",
            unanswered=len(request.unanswered_sub_questions),
            degraded=list(request.degraded_backends),
        )
        reasons = gaps or ["No evidence was retrieved for this investigation."]
        body = (
            "This investigation produced no evidence, so there is nothing to "
            "report and no claim can be made.\n\nWhy:\n"
            + "\n".join(f"- {reason}" for reason in reasons)
        )
        return ReportOutput(
            title=f"No findings: {request.query[:120]}",
            executive_summary=(
                "The investigation ran to completion but retrieved no evidence, so "
                "no findings are reported. The reasons are listed under Gaps."
            ),
            sections=[
                ReportSection(
                    kind=SectionKind.GAPS,
                    title="Why there are no findings",
                    body=body,
                    order=1,
                )
            ],
            confidence=0.0,
            gaps=reasons,
            citation_count=0,
        )

    def build_input(self, state: InvestigationState) -> ReportInput:
        questions = state.get("sub_questions") or []
        evidence = state.get("evidence") or []
        collection = state.get("collection_results") or []
        critique = state.get("critique")
        return ReportInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            investigation_id=state["investigation_id"],
            insights=list(state.get("insights") or [])[:12],
            recommendations=list(state.get("recommendations") or [])[:8],
            trends=list(state.get("trends") or [])[:10],
            forecasts=list(state.get("forecasts") or [])[:6],
            competitor_view=state.get("competitor_view"),
            critique=critique,
            evidence_ids=[ref.signal_id for ref in evidence][:60],
            unanswered_sub_questions=[q.question for q in questions if not q.answered][:8],
            collection_failures=[
                f"{result.connector_slug}: {result.error}"
                for result in collection
                if getattr(result, "error", None)
            ][:8],
            confidence=float(state.get("confidence") or 0.0),
        )

    async def execute(self, request: ReportInput, ctx: AgentContext) -> ReportOutput:
        gaps = self._assemble_gaps(request)

        if not request.evidence_ids and not request.insights and not request.recommendations:
            return self._nothing_to_report(request, gaps)

        rendered = self.render_prompt(ctx, query=request.query, objective=request.objective)
        drafted = await self.call_model(
            ctx,
            prompt=self._build_prompt(request, gaps),
            schema=ReportOutput,
            system=rendered.text,
        )

        known = set(request.evidence_ids)
        sections = [self._clean_section(section, known) for section in drafted.sections]
        sections = [section for section in sections if section is not None]

        if gaps:
            sections = [section for section in sections if section.kind is not SectionKind.GAPS]
            sections.append(self._gaps_section(gaps, order=len(sections)))

        if not sections:
            # `ReportOutput` requires at least one section, and a document with
            # none means every claim failed citation resolution. Saying that is
            # far more useful than a validation error in a worker log.
            sections = [
                ReportSection(
                    kind=SectionKind.GAPS,
                    title="No supportable findings",
                    body=(
                        "Every drafted claim cited evidence that could not be "
                        "resolved, so nothing could be published. This is a "
                        "failure of the investigation, not a finding about the "
                        "subject."
                    ),
                    order=0,
                )
            ]

        # Confidence is the Critic's number. See the module docstring.
        confidence = request.confidence
        return ReportOutput(
            title=drafted.title,
            executive_summary=drafted.executive_summary,
            sections=sections[:MAX_SECTIONS],
            confidence=confidence,
            confidence_band=ConfidenceBand.from_score(confidence),
            gaps=gaps,
            citation_count=sum(
                len(claim.citations) for section in sections for claim in section.claims
            ),
        )

    def to_delta(self, output: ReportOutput, state: InvestigationState) -> StateDelta:
        """Write the report and close the run.

        `COMPLETED_WITH_FINDINGS` rather than `COMPLETED` when gaps exist. The
        distinction is what lets a caller filter for runs that answered fully
        without opening each one -- and it is the state-level expression of the
        same honesty the gaps section provides to a human reader.
        """
        return {
            "report": output.model_dump(mode="json"),
            "status": (
                InvestigationStatus.COMPLETED_WITH_FINDINGS
                if output.gaps
                else InvestigationStatus.COMPLETED
            ),
        }

    # ------------------------------------------------------------ internals --

    def _assemble_gaps(self, request: ReportInput) -> list[str]:
        """Build the limitations list from state. Never from the model.

        Four sources, all facts already established: questions the evidence could
        not answer, backends that were unavailable, collections that failed, and
        the Critic's own unresolved findings. A model writing this section
        produces a graceful paragraph that omits the most damaging item.
        """
        gaps: list[str] = []

        for question in request.unanswered_sub_questions:
            gaps.append(f"Not established: {question}")

        if request.degraded_backends:
            gaps.append(
                "Retrieval ran degraded ("
                + ", ".join(request.degraded_backends)
                + "), so recall was reduced and absence of evidence is weaker than usual."
            )

        for failure in request.collection_failures:
            gaps.append(f"Source unavailable during collection -- {failure}")

        critique = request.critique or {}
        for finding in critique.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            if finding.get("severity") in {"blocking", "major"}:
                gaps.append(
                    f"Unresolved {finding.get('kind', 'finding')}: {finding.get('detail', '')}"
                )

        # Deduplicated while preserving order: a question that went unanswered
        # because its source failed to collect generates two near-identical
        # entries, and a limitations list that repeats itself gets skimmed.
        return list(dict.fromkeys(gaps))[:12]

    def _gaps_section(self, gaps: list[str], *, order: int) -> ReportSection:
        return ReportSection(
            kind=SectionKind.GAPS,
            title="What this investigation could not establish",
            body="\n".join(f"- {gap}" for gap in gaps),
            order=order,
        )

    def _clean_section(self, section: ReportSection, known: set[str]) -> ReportSection | None:
        """Drop claims whose citations do not resolve.

        The last place a fabricated reference can be caught. A claim citing only
        unknown signals is removed entirely rather than printed uncited -- an
        uncited sentence in a report that cites everything else reads as a
        general statement rather than as an unsupported one.
        """
        kept: list[ReportClaim] = []
        for claim in section.claims:
            resolvable = [signal_id for signal_id in claim.citations if signal_id in known]
            if not resolvable:
                _log.warning(
                    "report.dropped_unresolvable_claim",
                    claim_id=claim.id,
                    cited=claim.citations[:5],
                )
                continue
            kept.append(claim.model_copy(update={"citations": resolvable}))

        if section.claims and not kept and section.kind is not SectionKind.GAPS:
            # Every claim in the section was unsupported. Keeping the prose would
            # leave a section that reads as authoritative with nothing behind it.
            _log.warning("report.dropped_unsupported_section", kind=section.kind.value)
            return None
        return section.model_copy(update={"claims": kept})

    def _build_prompt(self, request: ReportInput, gaps: list[str]) -> str:
        lines = [
            f"Investigation: {request.query}",
            f"Objective: {request.objective}" if request.objective else "",
            "",
            "Findings to render (cite the signal ids given):",
        ]
        for insight in request.insights:
            if isinstance(insight, dict):
                lines.append(
                    f"- [{insight.get('id')}] {insight.get('statement')} "
                    f"(cites: {', '.join(str(s) for s in insight.get('signal_ids') or [])})"
                )
        lines.append("")

        if request.trends:
            lines.append("Measured trends:")
            lines.extend(
                f"- {t.get('topic')}: {t.get('direction')} "
                f"({t.get('observation_count', 0)} observations)"
                for t in request.trends
                if isinstance(t, dict)
            )
            lines.append("")
        if request.forecasts:
            lines.append("Projections (do not restate or adjust the numbers):")
            lines.extend(
                f"- {f.get('subject')}: {f.get('method')}"
                for f in request.forecasts
                if isinstance(f, dict)
            )
            lines.append("")
        if request.recommendations:
            lines.append("Recommendations:")
            lines.extend(
                f"- [{r.get('id')}] ({r.get('urgency')}) {r.get('action')}"
                for r in request.recommendations
                if isinstance(r, dict)
            )
            lines.append("")
        if gaps:
            lines.append(
                "The following limitations will be appended automatically. Do not "
                "restate them, and do not write around them:"
            )
            lines.extend(f"- {gap}" for gap in gaps)
            lines.append("")

        lines.append(f"Signal ids you may cite: {', '.join(request.evidence_ids[:60])}")
        lines.append("")
        lines.append(
            "Write the report. Every claim must cite at least one signal id from "
            "the list above. Introduce no number that is not given here. Hedge "
            "anything derived from a causal hypothesis. Do not soften a finding to "
            "make the document read better."
        )
        return "\n".join(line for line in lines if line != "")
