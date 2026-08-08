"""The Competitor agent: builds the competitive picture from graph plus text.

Two sources, and they fail differently, which is the whole reason this node
combines them rather than picking one.

The **graph** knows rivalries that were extracted over the entire corpus history,
including ones nobody wrote about in the window this investigation retrieved. It
is the only source that can say "these two have competed since 2019" from a
document read eighteen months ago.

**Retrieved text** knows what is being said right now, including about companies
resolution has never seen -- a startup named once in a forum post has no graph
node and is still a competitor.

Neither alone is the answer. Graph-only misses the new entrant; text-only misses
the established rival who simply was not mentioned this quarter and presents the
gap as an absence of competition.

**When the graph is unavailable the output says so.** `graph_available=False`
propagates into the report, because a competitive list built from text alone is
systematically incomplete in a way the list itself cannot show -- every entry
looks equally solid, and the missing ones are invisible. This is the single most
important degradation signal in the system, since competitive positioning is what
a reader acts on.

`docs/agent-system.md` §5.5.
"""

from __future__ import annotations

from typing import Any, ClassVar

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.competitor.schemas import (
    MAX_COMPETITORS,
    CompetitiveBasis,
    CompetitorFinding,
    CompetitorInput,
    CompetitorOutput,
)
from agents.errors import ToolExecutionError
from agents.state import InvestigationState
from backend.core.logging import get_logger
from models.enums import AgentName
from services.llm.router import ModelTier

__all__ = ["CompetitorAgent"]

_log = get_logger(__name__)


class CompetitorAgent(BaseAgent[CompetitorInput, CompetitorOutput]):
    """Combines graph relationships with retrieved text into a rival list."""

    name: ClassVar[AgentName] = AgentName.COMPETITOR
    tier: ClassVar[ModelTier] = ModelTier.WORKER
    output_model: ClassVar[type[CompetitorOutput]] = CompetitorOutput
    tools: ClassVar[frozenset[str]] = frozenset({"find_paths", "hybrid_search", "aggregate"})

    def build_input(self, state: InvestigationState) -> CompetitorInput:
        graph_context = state.get("graph_context")
        seeds = list(getattr(graph_context, "seed_entity_ids", ()) or ())
        return CompetitorInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            seed_entity_ids=seeds[:32],
            evidence_count=len(state.get("evidence") or []),
        )

    async def execute(self, request: CompetitorInput, ctx: AgentContext) -> CompetitorOutput:
        graph_findings, graph_available = await self._from_graph(request, ctx)
        text_context = await self._from_text(request, ctx)

        rendered = self.render_prompt(
            ctx,
            query=request.query,
            objective=request.objective,
            graph_available=graph_available,
        )
        analysed = await self.call_model(
            ctx,
            prompt=self._build_prompt(request, graph_findings, text_context, graph_available),
            schema=CompetitorOutput,
            system=rendered.text,
        )

        merged = self._merge(graph_findings, analysed.competitors)
        return analysed.model_copy(
            update={
                "competitors": merged[:MAX_COMPETITORS],
                "graph_available": graph_available,
                "subject": analysed.subject or request.subject,
            }
        )

    def to_delta(self, output: CompetitorOutput, state: InvestigationState) -> StateDelta:
        """`competitor_view` is a single-writer scalar, so it is written whole."""
        return {"competitor_view": output.model_dump(mode="json")}

    # ------------------------------------------------------------ internals --

    async def _from_graph(
        self, request: CompetitorInput, ctx: AgentContext
    ) -> tuple[list[CompetitorFinding], bool]:
        """Rivals the graph already knows about.

        Returns `(findings, graph_available)`. The flag is separate from an empty
        list on purpose: "the graph has no competitors for this entity" and "the
        graph could not be read" are different facts, and only the second should
        make the report hedge.
        """
        if not request.seed_entity_ids:
            return [], True

        findings: list[CompetitorFinding] = []
        available = True
        for entity_id in request.seed_entity_ids[:4]:
            try:
                result = await self.use_tool(
                    ctx, "find_paths", {"source_id": entity_id, "target_id": entity_id}
                )
            except ToolExecutionError as error:
                _log.warning("competitor.graph_unavailable", error=str(error))
                available = False
                break
            findings.extend(_findings_from_graph(result))
        return findings, available

    async def _from_text(self, request: CompetitorInput, ctx: AgentContext) -> str:
        """A focused search for competitive language.

        Separate from the Retriever's general evidence gathering because the
        phrasing that surfaces rivalries -- "compared to", "versus", "alternative
        to" -- is not what a general query retrieves, and the passages that
        contain it are the ones that support a `stated` basis.
        """
        subject = request.subject or request.query
        try:
            result = await self.use_tool(
                ctx,
                "hybrid_search",
                {"query": f"{subject} competitors alternatives versus compared to"},
            )
        except ToolExecutionError as error:
            _log.warning("competitor.text_search_failed", error=str(error))
            return ""
        return _render_passages(result)

    def _build_prompt(
        self,
        request: CompetitorInput,
        graph_findings: list[CompetitorFinding],
        text_context: str,
        graph_available: bool,
    ) -> str:
        lines = [
            f"Investigation: {request.query}",
            f"Objective: {request.objective}" if request.objective else "",
            "",
        ]
        if graph_findings:
            lines.append("Rivalries already recorded in the knowledge graph:")
            lines.extend(
                f"- {finding.name} (strength {finding.strength:.2f})"
                for finding in graph_findings
            )
        elif graph_available:
            lines.append("The knowledge graph records no competitive relationships here.")
        else:
            lines.append(
                "The knowledge graph could not be read. Build the picture from the "
                "passages alone and note that established rivals may be missing."
            )
        lines.extend(["", text_context or "(no competitive passages retrieved)", ""])
        lines.append(
            "Identify the competitors. Mark a rivalry 'stated' only when a passage "
            "says so, and cite the signal. Mark it 'inferred' when it rests on "
            "co-occurrence -- co-occurrence is equally true of a company and its "
            "own supplier."
        )
        return "\n".join(line for line in lines if line != "")

    def _merge(
        self, graph_findings: list[CompetitorFinding], model_findings: list[CompetitorFinding]
    ) -> list[CompetitorFinding]:
        """Union the two sources, preferring the stronger basis on a collision.

        Name-keyed, case-folded. A graph entry and a text entry for the same
        company must collapse to one row -- listing "Globex" twice because one
        came from each source is the same false-corroboration failure that
        `retrieval/rerank/fusion.py` exists to prevent, arriving through a
        different door.
        """
        by_name: dict[str, CompetitorFinding] = {}
        for finding in [*graph_findings, *model_findings]:
            key = finding.name.strip().casefold()
            existing = by_name.get(key)
            if existing is None:
                by_name[key] = finding
                continue
            # A stated basis outranks graph, which outranks inferred: keep the
            # entry that licenses the strongest honest claim, but take the
            # citations from both so the merged row is at least as evidenced.
            rank = {
                CompetitiveBasis.STATED: 3,
                CompetitiveBasis.GRAPH: 2,
                CompetitiveBasis.INFERRED: 1,
            }
            winner = finding if rank[finding.basis] > rank[existing.basis] else existing
            merged_signals = list(dict.fromkeys([*existing.signal_ids, *finding.signal_ids]))[:10]
            by_name[key] = winner.model_copy(
                update={
                    "signal_ids": merged_signals,
                    "entity_id": winner.entity_id or existing.entity_id or finding.entity_id,
                    "strength": max(existing.strength, finding.strength),
                }
            )
        return sorted(
            by_name.values(), key=lambda item: (-item.strength, -item.confidence, item.name)
        )


def _findings_from_graph(result: Any) -> list[CompetitorFinding]:
    data = getattr(result, "data", result)
    raw = getattr(data, "paths", None)
    if raw is None and isinstance(data, dict):
        raw = data.get("paths") or data.get("competitors")
    if not isinstance(raw, (list, tuple)):
        return []

    findings: list[CompetitorFinding] = []
    for entry in raw:
        name = _attr(entry, "name")
        if not isinstance(name, str) or not name:
            names = _attr(entry, "entity_names")
            name = names[-1] if isinstance(names, (list, tuple)) and names else None
        if not isinstance(name, str) or not name:
            continue
        confidence = _attr(entry, "confidence")
        findings.append(
            CompetitorFinding(
                entity_id=_attr(entry, "entity_id"),
                name=name,
                basis=CompetitiveBasis.GRAPH,
                strength=_clamp(_attr(entry, "strength")),
                confidence=_clamp(confidence),
            )
        )
    return findings


def _render_passages(result: Any) -> str:
    data = getattr(result, "data", result)
    raw = getattr(data, "results", None)
    if raw is None and isinstance(data, dict):
        raw = data.get("results") or data.get("passages")
    if not isinstance(raw, (list, tuple)):
        return ""
    lines = ["Retrieved passages:"]
    for entry in raw[:15]:
        signal_id = _attr(entry, "signal_id") or "?"
        text = _attr(entry, "quote") or _attr(entry, "text") or ""
        if isinstance(text, str) and text:
            lines.append(f"[{signal_id}] {text[:400]}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        return obj.get(name)
    return value


def _clamp(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(1.0, max(0.0, float(value)))
    return 0.0
