"""The Retriever: turns sub-questions into evidence references.

Runs hybrid retrieval once per sub-question rather than once for the whole
query, and that is the design decision this node exists to make. A single search
for "how is Acme's battery strategy performing against competitors" returns
passages about Acme, passages about batteries, and passages about competitors --
ranked together, so whichever facet the corpus is densest in crowds out the
others. Retrieving per sub-question and tagging each result with the question it
serves means the Critic can check coverage per question, and a report that
answers four of six is visibly incomplete rather than merely thin.

**Degradation is reported, not hidden.** `docs/architecture.md` §7.3 lets
retrieval continue keyword-only when Qdrant is unavailable. That produces a real
answer built on weaker evidence, and the difference is invisible in the results
themselves -- the passages look identical. So `degraded_backends` is carried into
the state, the Critic reads it, and the report says the confidence is lower
because semantic search was unavailable. Without that, an outage silently becomes
a worse answer that nobody can distinguish from a good one.

**Deduplication happens here, across sub-questions.** Two sub-questions about the
same company retrieve overlapping passages, and the same passage returned twice
reads downstream as two independent sources -- which is exactly the false
corroboration that `retrieval/rerank/fusion.py` collapses within a single search
and cannot see across separate ones.

`docs/agent-system.md` §5.3.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Final

from agents.base import AgentContext, BaseAgent, StateDelta
from agents.errors import ToolExecutionError
from agents.retriever.schemas import (
    MAX_EVIDENCE_ITEMS,
    MAX_SUB_QUERIES,
    RetrievalQuery,
    RetrievedItem,
    RetrieverInput,
    RetrieverOutput,
)
from agents.state import EvidenceRef, InvestigationState
from backend.core.logging import get_logger
from models.enums import AgentName
from services.llm.router import ModelTier

__all__ = ["MAX_CONCURRENT_SEARCHES", "RetrieverAgent", "RetrievalPlan"]

_log = get_logger(__name__)

MAX_CONCURRENT_SEARCHES: Final = 3
"""Hybrid searches in flight at once.

Each fans out to three backends, so three concurrent searches is nine concurrent
backend queries. Beyond that the vector store's own concurrency limit starts
queueing and the latency gain disappears into waiting.
"""


from pydantic import Field  # noqa: E402 -- kept next to the model it annotates

from models.base import StrictModel  # noqa: E402


class RetrievalPlan(StrictModel):
    """The searches the model decided to run."""

    queries: list[RetrievalQuery] = Field(min_length=1, max_length=MAX_SUB_QUERIES)
    rationale: str | None = Field(default=None, max_length=1000)


class RetrieverAgent(BaseAgent[RetrieverInput, RetrieverOutput]):
    """Plans searches, runs them concurrently, deduplicates the evidence."""

    name: ClassVar[AgentName] = AgentName.RETRIEVER
    tier: ClassVar[ModelTier] = ModelTier.WORKER
    output_model: ClassVar[type[RetrieverOutput]] = RetrieverOutput
    tools: ClassVar[frozenset[str]] = frozenset(
        {"hybrid_search", "fetch_passage", "rerank", "resolve_citation", "neighbours"}
    )

    def build_input(self, state: InvestigationState) -> RetrieverInput:
        questions = state.get("sub_questions") or []
        graph_context = state.get("graph_context")
        seeds = list(getattr(graph_context, "seed_entity_ids", ()) or ())
        return RetrieverInput(
            query=state["query"],
            objective=state.get("objective", ""),
            tenant_id=state["tenant_id"],
            sub_questions=[question.question for question in questions][:8],
            sub_question_ids=[question.id for question in questions][:8],
            seed_entity_ids=seeds[:32],
            already_retrieved=len(state.get("evidence") or []),
        )

    async def execute(self, request: RetrieverInput, ctx: AgentContext) -> RetrieverOutput:
        plan = await self._plan_searches(request, ctx)
        results = await self._run_searches(plan.queries, ctx)

        items: list[RetrievedItem] = []
        degraded: set[str] = set()
        total_candidates = 0
        # Deduplication key is (signal_id, chunk_id). Across sub-questions the
        # same passage is routinely returned twice, and downstream that reads as
        # two independent sources -- the false corroboration fusion collapses
        # inside one search and cannot see between separate ones.
        seen: set[tuple[str, str | None]] = set()

        for query, payload in results:
            if payload is None:
                degraded.add("hybrid_search")
                continue
            total_candidates += _total_candidates_from(payload)
            degraded.update(_degraded_backends_from(payload))
            for hit in _hits_from(payload):
                key = (hit.signal_id, hit.chunk_id)
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    hit.model_copy(update={"sub_question_id": query.sub_question_id})
                )

        # Sorted by relevance before the cap so the truncation keeps the best
        # evidence rather than whichever sub-question happened to finish first.
        items.sort(key=lambda item: (-item.relevance, item.signal_id))

        return RetrieverOutput(
            items=items[:MAX_EVIDENCE_ITEMS],
            queries_run=[query.text for query in plan.queries],
            degraded_backends=sorted(degraded),
            total_candidates=total_candidates,
        )

    def to_delta(self, output: RetrieverOutput, state: InvestigationState) -> StateDelta:
        """Append this pass's evidence and mark the sub-questions it covered.

        `evidence` is `operator.add`-reduced, so only the increment is returned.
        `sub_questions` is a single-writer key, so it is rewritten whole -- and
        rewritten from the *existing* list rather than rebuilt, because the
        Planner's question text is authoritative and this node only sets the
        `answered` flag.
        """
        covered = output.covered_sub_questions
        existing = state.get("sub_questions") or []
        return {
            "evidence": [
                EvidenceRef(
                    signal_id=item.signal_id,
                    chunk_id=item.chunk_id,
                    quote=item.quote,
                    char_start=item.char_start,
                    char_end=item.char_end,
                    relevance=item.relevance,
                    retrieved_by=AgentName.RETRIEVER,
                )
                for item in output.items
            ],
            "sub_questions": [
                question.model_copy(update={"answered": True})
                if question.id in covered
                else question
                for question in existing
            ],
        }

    # ------------------------------------------------------------ internals --

    async def _plan_searches(
        self, request: RetrieverInput, ctx: AgentContext
    ) -> RetrievalPlan:
        """Decide what to search for.

        When the Planner produced sub-questions, they *are* the search plan and
        no model call is made. Asking a model to rephrase six questions it was
        just handed spends tokens to produce paraphrases that retrieve the same
        passages -- and each paraphrase costs a full three-backend search.
        """
        if request.sub_questions:
            paired = zip(
                request.sub_questions,
                request.sub_question_ids or [None] * len(request.sub_questions),
                strict=False,
            )
            return RetrievalPlan(
                queries=[
                    RetrievalQuery(text=text[:500], sub_question_id=question_id)
                    for text, question_id in paired
                ][:MAX_SUB_QUERIES],
                rationale="sub-questions used directly; no decomposition call needed",
            )

        rendered = self.render_prompt(
            ctx, query=request.query, objective=request.objective
        )
        return await self.call_model(
            ctx,
            prompt=(
                f"Investigation: {request.query}\n"
                f"Objective: {request.objective}\n\n"
                "Decompose this into the searches needed to gather evidence. "
                "Each search should target a distinct facet -- near-duplicate "
                "searches retrieve the same passages at full cost."
            ),
            schema=RetrievalPlan,
            system=rendered.text,
        )

    async def _run_searches(
        self, queries: list[RetrievalQuery], ctx: AgentContext
    ) -> list[tuple[RetrievalQuery, Any | None]]:
        """Run every search concurrently, bounded, tolerating individual failure.

        A failed search yields `None` rather than raising. Losing one facet of a
        six-facet investigation is a degraded answer; losing the whole run
        because one backend timed out is not a trade worth making, and the
        degradation is recorded either way.
        """
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

        async def run_one(query: RetrievalQuery) -> tuple[RetrievalQuery, Any | None]:
            async with semaphore:
                try:
                    return query, await self.use_tool(
                        ctx, "hybrid_search", {"query": query.text}
                    )
                except ToolExecutionError as error:
                    _log.warning(
                        "retriever.search_failed", query=query.text[:80], error=str(error)
                    )
                    return query, None

        return list(await asyncio.gather(*(run_one(query) for query in queries)))


# --------------------------------------------------------------------------- #
# Tool-payload extraction
# --------------------------------------------------------------------------- #
#
# Defensive because the payload crosses the registry boundary, which renders and
# caps it. A shape change should thin the evidence, not raise inside a node whose
# failure would cost the whole run's retrieval.


def _hits_from(payload: Any) -> list[RetrievedItem]:
    data = getattr(payload, "data", payload)
    raw = getattr(data, "results", None)
    if raw is None and isinstance(data, dict):
        raw = data.get("results") or data.get("passages")
    if not isinstance(raw, (list, tuple)):
        return []

    items: list[RetrievedItem] = []
    for entry in raw:
        signal_id = _attr(entry, "signal_id")
        if not isinstance(signal_id, str) or not signal_id:
            continue
        quote = _attr(entry, "quote") or _attr(entry, "text")
        relevance = _attr(entry, "score")
        if not isinstance(relevance, (int, float)) or isinstance(relevance, bool):
            relevance = 0.0
        items.append(
            RetrievedItem(
                signal_id=signal_id,
                chunk_id=_as_str_or_none(_attr(entry, "chunk_id")),
                quote=str(quote)[:500] if isinstance(quote, str) else None,
                char_start=_as_int_or_none(_attr(entry, "char_start")),
                char_end=_as_int_or_none(_attr(entry, "char_end")),
                # Clamped: `Score` is 0-1 and a backend that returns a raw
                # distance would otherwise fail validation and lose the whole
                # search rather than one badly-scaled score.
                relevance=min(1.0, max(0.0, float(relevance))),
            )
        )
    return items


def _degraded_backends_from(payload: Any) -> list[str]:
    data = getattr(payload, "data", payload)
    value = _attr(data, "degraded_backends")
    if value is None:
        value = _attr(data, "degraded")
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, str)]
    return []


def _total_candidates_from(payload: Any) -> int:
    data = getattr(payload, "data", payload)
    value = _attr(data, "total_candidates")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _attr(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, dict):
        return obj.get(name)
    return value


def _as_str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
