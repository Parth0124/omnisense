"""Hybrid retrieval: fan out to three backends, fuse, dedupe, rerank.

Design Doc §8 specifies keyword + vector + graph + metadata filtering +
cross-encoder reranking. This module owns the *orchestration*; each backend and
the reranker live behind narrow protocols so any of them can be faked in a test
or swapped in production.

Two properties matter more than the ranking quality, because they are the ones
that fail silently.

**Partial failure must degrade, not abort.** `docs/architecture.md` §7.3 says a
Qdrant outage means keyword-only retrieval with reduced recall, and an OpenSearch
outage means vector-only with reduced exact-term recall -- not a failed
investigation. So a backend that raises is recorded in the diagnostics and
dropped from the fan-out, and the run continues with what is left. The
diagnostics then carry `degraded=True` so the Critic can lower the confidence it
reports rather than presenting a thin answer as a complete one. A retrieval layer
that returned five passages instead of twelve *without saying so* is worse than
one that failed outright.

**Filters are pushed down, never applied afterwards.** Asking Qdrant for the 100
nearest neighbours and then keeping the three that are in-date is not the same as
asking for the 100 nearest in-date neighbours. The difference is invisible until
someone notices the last month of data is unreachable.

Order of operations, and every step of it is load-bearing:

    expand -> fan out (filtered) -> fuse -> dedupe -> rerank -> resolve

Fusion before dedupe, because a chunk found by three backends must accumulate
three RRF terms before collapsing (see `retrieval/rerank/fusion.py`). Rerank
after dedupe, because the cross-encoder is the expensive step and scoring six
copies of one passage wastes the budget that should have gone to six different
ones.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from retrieval.rerank.fusion import (
    DEFAULT_BACKEND_WEIGHTS,
    DEFAULT_RRF_K,
    collapse_near_duplicates,
    reciprocal_rank_fusion,
)
from retrieval.types import (
    Backend,
    Candidate,
    GraphFact,
    Passage,
    RetrievalDiagnostics,
    RetrievalRequest,
    RetrievalResult,
)

__all__ = [
    "GraphExpander",
    "HybridRetriever",
    "PassageResolver",
    "Reranker",
    "SearchBackend",
]


@runtime_checkable
class SearchBackend(Protocol):
    """One candidate generator.

    Returns `Candidate`s -- chunk ids with ranks -- not passages. Resolving a
    chunk id to text is a separate, batched step: doing it per backend would
    fetch the same passage three times when all three backends agree, which is
    exactly the case the design hopes for.
    """

    backend: Backend

    async def search(self, request: RetrievalRequest, *, limit: int) -> Sequence[Candidate]:
        """Return up to `limit` candidates, ranked best-first, filters applied."""
        ...


@runtime_checkable
class GraphExpander(Protocol):
    """Query expansion and graph facts from the knowledge graph."""

    async def expand_query(self, request: RetrievalRequest) -> Sequence[str]:
        """Aliases and canonical names to widen lexical matching.

        "DDOG", "Datadog Inc" and "Datadog" are one entity to the graph and three
        unrelated strings to BM25. Expansion is what closes that gap.
        """
        ...

    async def facts_for(
        self, request: RetrievalRequest, signal_ids: Sequence[str]
    ) -> Sequence[GraphFact]:
        """Relationships worth carrying alongside the retrieved text."""
        ...


@runtime_checkable
class PassageResolver(Protocol):
    """Turns chunk ids into citable passages, in one batched call."""

    async def resolve(self, chunk_ids: Sequence[str]) -> Mapping[str, Passage]:
        """Fetch passage text and provenance. Missing ids are simply absent.

        Absence is normal rather than exceptional: a Signal deleted for an
        erasure request between indexing and retrieval leaves a live chunk id in
        Qdrant pointing at nothing. Raising would make one deleted record break
        every query that happened to match it.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Cross-encoder scoring of query against passage."""

    async def rerank(
        self, query: str, passages: Sequence[Passage], *, top_k: int
    ) -> Sequence[Passage]:
        """Return passages re-scored and re-ordered, best first."""
        ...


class HybridRetriever:
    """Orchestrates the retrieval pipeline.

    Stateless and reusable; one instance per process, driven concurrently.
    """

    def __init__(
        self,
        backends: Sequence[SearchBackend],
        resolver: PassageResolver,
        *,
        expander: GraphExpander | None = None,
        reranker: Reranker | None = None,
        weights: Mapping[Backend, float] | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        simhash_of: object | None = None,
    ) -> None:
        self._backends = {b.backend: b for b in backends}
        if len(self._backends) != len(backends):
            raise ValueError("two backends declared the same Backend value")
        self._resolver = resolver
        self._expander = expander
        self._reranker = reranker
        self._weights = dict(weights or DEFAULT_BACKEND_WEIGHTS)
        self._rrf_k = rrf_k
        self._simhash_of = simhash_of

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Run the full pipeline. Never raises for a backend failure."""
        started = time.perf_counter()
        counts: dict[str, int] = {}
        latencies: dict[str, float] = {}
        failed: list[str] = []

        if self._expander is not None and request.seed_entity_ids:
            try:
                await self._expander.expand_query(request)
            except Exception:  # noqa: BLE001 -- expansion is an optimisation
                # Losing expansion costs recall on aliases; it must not cost the
                # whole query, so this degrades silently rather than failing.
                failed.append("expansion")

        per_backend = await self._fan_out(request, counts, latencies, failed)

        fused = reciprocal_rank_fusion(
            per_backend,
            k=self._rrf_k,
            weights=self._weights,
            pool_max=request.pool_max,
        )

        resolved = await self._resolve(fused, request)
        deduped = self._dedupe(resolved)
        ranked = await self._rerank(request, deduped, failed)
        facts = await self._facts(request, ranked, failed)

        return RetrievalResult(
            request=request,
            passages=ranked[: request.k_final],
            graph_facts=facts,
            diagnostics=RetrievalDiagnostics(
                per_backend_counts=counts,
                per_backend_latency_ms=latencies,
                backends_failed=tuple(failed),
                fused_pool_size=len(fused),
                after_dedupe=len(deduped),
                reranked=len(ranked) if self._reranker and request.rerank else 0,
                total_latency_ms=(time.perf_counter() - started) * 1000,
            ),
        )

    # ------------------------------------------------------------ internals --

    async def _fan_out(
        self,
        request: RetrievalRequest,
        counts: dict[str, int],
        latencies: dict[str, float],
        failed: list[str],
    ) -> dict[Backend, Sequence[Candidate]]:
        """Query every requested backend concurrently, tolerating failures.

        Concurrent because the backends are independent and the slowest one sets
        the latency floor; sequential fan-out would make a hybrid query cost the
        sum rather than the max of three network round trips.
        """
        limits = {
            Backend.KEYWORD: request.k_keyword,
            Backend.VECTOR: request.k_vector,
            Backend.GRAPH: request.k_graph,
        }
        selected = [
            (name, backend)
            for name, backend in self._backends.items()
            if name in request.backends
        ]

        async def run_one(
            name: Backend, backend: SearchBackend
        ) -> tuple[Backend, Sequence[Candidate] | None, float]:
            begin = time.perf_counter()
            try:
                found = await backend.search(request, limit=limits.get(name, 100))
            except Exception:  # noqa: BLE001 -- degradation is the contract
                return name, None, (time.perf_counter() - begin) * 1000
            return name, found, (time.perf_counter() - begin) * 1000

        gathered = await asyncio.gather(*(run_one(n, b) for n, b in selected))

        results: dict[Backend, Sequence[Candidate]] = {}
        for name, found, elapsed in gathered:
            latencies[name.value] = elapsed
            if found is None:
                failed.append(name.value)
                counts[name.value] = 0
                continue
            results[name] = found
            counts[name.value] = len(found)
        return results

    async def _resolve(
        self, fused: Sequence[object], request: RetrievalRequest
    ) -> list[Passage]:
        """Batch-resolve chunk ids to passages, preserving fused order."""
        if not fused:
            return []
        ordered_ids = [c.chunk_id for c in fused]  # type: ignore[attr-defined]
        table = await self._resolver.resolve(ordered_ids)

        passages: list[Passage] = []
        for candidate in fused:
            passage = table.get(candidate.chunk_id)  # type: ignore[attr-defined]
            if passage is None:
                # Indexed but no longer stored -- an erasure between index and
                # query. Skipping is correct; raising would let one deleted
                # record break every query that matched it.
                continue
            import dataclasses

            passages.append(
                dataclasses.replace(
                    passage,
                    fused_score=candidate.score,  # type: ignore[attr-defined]
                    found_by=candidate.found_by,  # type: ignore[attr-defined]
                    ranks={
                        b.value: r
                        for b, r in candidate.ranks.items()  # type: ignore[attr-defined]
                    },
                )
            )
        return passages

    def _dedupe(self, passages: Sequence[Passage]) -> list[Passage]:
        """Collapse near-duplicates, after fusion and before reranking."""
        if self._simhash_of is None:
            return list(passages)
        return collapse_near_duplicates(
            passages,
            simhash_of=self._simhash_of,  # type: ignore[arg-type]
        )

    async def _rerank(
        self, request: RetrievalRequest, passages: Sequence[Passage], failed: list[str]
    ) -> list[Passage]:
        """Cross-encoder rerank of the top slice, tolerating reranker failure.

        Only `rerank_depth` passages are scored: the cross-encoder is quadratic
        in the sense that it processes query and passage together, so it is
        roughly two orders of magnitude more expensive per item than the fusion
        it refines. Reranking the whole pool would dominate query latency for a
        gain that falls off sharply past the top few dozen.
        """
        if not self._reranker or not request.rerank or not passages:
            return list(passages)
        head = list(passages[: request.rerank_depth])
        tail = list(passages[request.rerank_depth :])
        try:
            scored = await self._reranker.rerank(
                request.query, head, top_k=request.k_final
            )
        except Exception:  # noqa: BLE001 -- fused order is a usable fallback
            failed.append("rerank")
            return list(passages)
        return [*scored, *tail]

    async def _facts(
        self, request: RetrievalRequest, passages: Sequence[Passage], failed: list[str]
    ) -> list[GraphFact]:
        """Graph facts for the surviving passages. Never fatal."""
        if self._expander is None or not passages:
            return []
        try:
            signal_ids = list({p.signal_id for p in passages})
            return list(await self._expander.facts_for(request, signal_ids))
        except Exception:  # noqa: BLE001
            failed.append("graph_facts")
            return []
