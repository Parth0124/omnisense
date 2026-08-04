"""Cross-encoder reranking of the fused candidate set.

Fusion (`retrieval/rerank/fusion.py`) ranks by *agreement between backends*,
which is cheap and coarse: it never reads the query and the passage together, so
it cannot tell "Acme acquired Globex" from "Globex acquired Acme". A
cross-encoder does exactly that -- it encodes the pair jointly and emits one
calibrated relevance number -- and pays for it. Encoding 50 pairs jointly is
roughly **two orders of magnitude** more work per item than the arithmetic RRF
does over ranks it already has, whether the cost lands as GPU seconds on a hosted
model or as tokens on an LLM judge. That ratio is the whole reason
`RetrievalRequest.rerank_depth` exists: the top 50 of the fused pool get scored,
the rest keep their fused order, and quality past the top few dozen is not worth
the latency (`docs/retrieval.md` §3, §10).

**The backend is undecided and this module does not choose it.**
`docs/retrieval.md` §10 lists three candidates -- a Modal-hosted cross-encoder,
`claude-haiku-4-5-20251001` as an LLM reranker through `services/llm/router.py`,
and an in-process ONNX model -- and none has an ADR. So scoring arrives as a
constructor argument satisfying `PairScorer`, and this file imports none of them.
That is not only about the pending decision: `retrieval/` is layer L1
(`docs/architecture.md` §6.1) and may not import `services/`, so an LLM reranker
*cannot* be reached from here except through a port the L2 caller supplies.

Three properties are load-bearing.

**Batched, with bounded concurrency.** A scorer is a network call or a GPU
queue. Firing 50 single-pair requests at it converts one batched forward pass
into 50 round trips, and doing that unbounded opens 50 connections to a service
that was sized for a handful. Pairs are grouped into batches and at most
`max_concurrency` batches are in flight.

**Failure is total, never partial.** If one batch fails, the passages it covered
have no cross-encoder score while their neighbours do, and the two scales are not
comparable -- an unscored passage would sink below a scored one for no reason
other than which batch it landed in. So a batch failure raises
`RerankUnavailable` and `HybridRetriever` falls back to the fused order for the
whole list, which is a coherent ranking rather than a mixture of two
(`docs/retrieval.md` §12, "Reranker unavailable -> fall back to fused RRF
order").

**Ordering is deterministic.** Equal scores keep fused order, because an
evaluation harness measuring a one-point nDCG change cannot distinguish a real
regression from a reordering of ties.

Caching is deliberately absent. `docs/retrieval.md` §10 keys the rerank cache on
`(query_hash, chunk_id, reranker_version)`, and the version belongs to whatever
sits behind the port -- this class cannot name it, so it must not be the thing
that builds the key. Cache inside the `PairScorer` implementation, or wrap it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
from collections.abc import Sequence
from typing import Final, Protocol, runtime_checkable

from retrieval.types import Passage

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_TIMEOUT_SECONDS",
    "CrossEncoderReranker",
    "PairScorer",
    "RerankUnavailable",
]


DEFAULT_BATCH_SIZE: Final = 16
"""Pairs per call to the scoring port.

Small enough that one slow batch does not hold the whole rerank hostage, large
enough that a GPU forward pass is not dominated by per-request overhead. Untuned
-- like every other number in `docs/retrieval.md` §3.
"""

DEFAULT_MAX_CONCURRENCY: Final = 4
"""Batches in flight at once.

The ceiling exists for the *scorer*, not for this process: a hosted reranker is a
shared, GPU-bound service, and an unbounded fan-out from one query would queue
ahead of every other query in the system. Four batches of 16 covers the default
`rerank_depth=50` in one wave.
"""

DEFAULT_TIMEOUT_SECONDS: Final = 0.8
"""Whole-rerank budget, from the 800 ms p95 in `docs/retrieval.md` §10.

Applied to the entire call rather than per batch: per-batch timeouts multiply
into a total nobody bounded, and the number the user experiences is the total.
Exceeding it degrades to fused order, which is why the budget can be this tight.
"""


@runtime_checkable
class PairScorer(Protocol):
    """Scores a batch of `(query, passage_text)` pairs. One float per text.

    The narrowest port that admits all three candidate backends: a Modal
    endpoint, a local ONNX session, and an LLM prompted to rate relevance all
    reduce to "given a query and some texts, return a number each". It takes
    *text*, not `Passage`, so an implementation cannot quietly start ranking on
    provenance -- recency and source weighting are fusion's job and the Critic's,
    and burying them in an opaque model score makes a ranking regression
    untraceable.

    Higher is more relevant. The scale is otherwise unconstrained -- scores are
    only ever compared to each other within one call, never across calls or
    against `fused_score`.
    """

    async def __call__(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        """Return one score per text, in the order the texts were given."""
        ...


class RerankUnavailable(RuntimeError):
    """The scorer failed, timed out, or answered incoherently.

    Raised rather than swallowed so the caller records the degradation. Callers
    that must not fail catch it and keep the fused order -- `HybridRetriever`
    does exactly that and appends `"rerank"` to `backends_failed`, which sets
    `RetrievalDiagnostics.degraded` and lowers the confidence the report claims.
    """


class CrossEncoderReranker:
    """A `Reranker` (see `retrieval/hybrid.py`) over an injected scoring port.

    Stateless apart from its configuration; one instance per process, safe to
    drive concurrently.
    """

    def __init__(
        self,
        score: PairScorer,
        *,
        depth: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")
        if timeout_seconds is not None and timeout_seconds <= 0:
            raise ValueError(
                f"timeout_seconds must be positive or None (no deadline), "
                f"got {timeout_seconds}"
            )
        if depth is None:
            # Read late, never at import: `get_settings()` parses `.env`, and a
            # module-level read would freeze whatever the environment looked like
            # when this file happened to be imported.
            from backend.core.config import get_settings

            depth = get_settings().retrieval.rerank_depth
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self._score = score
        self._depth = depth
        self._batch_size = batch_size
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._timeout_seconds = timeout_seconds

    @property
    def depth(self) -> int:
        """How many passages this reranker will score, at most."""
        return self._depth

    async def rerank(
        self, query: str, passages: Sequence[Passage], *, top_k: int
    ) -> Sequence[Passage]:
        """Score the top `depth` passages and return the best `top_k`.

        Input is assumed to be in fused order, best first -- which is what
        `HybridRetriever` passes. Passages beyond `depth` are never scored and
        keep their fused position *behind* every scored passage: they were ranked
        below the head by fusion, and inventing a score for them is the mixing of
        scales the module docstring rules out.

        Raises `RerankUnavailable` if the scorer fails, so the caller can decide
        between degrading and failing. It never returns a partially reranked list.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not passages:
            return []

        head = list(passages[: self._depth])
        tail = list(passages[self._depth :])

        scores = await self._score_all(query, [p.text for p in head])

        scored = [
            dataclasses.replace(passage, rerank_score=score)
            for passage, score in zip(head, scores, strict=True)
        ]
        # Stable sort: equal scores keep fused order. Python's sort guarantees
        # that, which is why there is no explicit tiebreak here -- adding one on
        # `chunk_id` would *lose* information by discarding fusion's opinion.
        scored.sort(key=lambda p: -(p.rerank_score or 0.0))

        return [*scored, *tail][:top_k]

    # ------------------------------------------------------------ internals --

    async def _score_all(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score every text, batched and concurrency-bounded, under one deadline."""
        batches = [
            texts[start : start + self._batch_size]
            for start in range(0, len(texts), self._batch_size)
        ]

        async def run_one(batch: Sequence[str]) -> Sequence[float]:
            async with self._semaphore:
                return await self._score(query, batch)

        try:
            # The deadline wraps the gather, so a cancellation propagates into
            # every in-flight batch instead of leaving them running against a
            # result nobody will read.
            async with asyncio.timeout(self._timeout_seconds):
                results = await asyncio.gather(
                    *(run_one(batch) for batch in batches), return_exceptions=True
                )
        except TimeoutError as exc:
            raise RerankUnavailable(
                f"reranker exceeded its {self._timeout_seconds}s budget scoring "
                f"{len(texts)} passages in {len(batches)} batches"
            ) from exc

        returned: list[Sequence[float]] = []
        failures: list[BaseException] = []
        for result in results:
            if isinstance(result, BaseException):
                failures.append(result)
            else:
                returned.append(result)
        if failures:
            raise RerankUnavailable(
                f"{len(failures)} of {len(batches)} rerank batches failed; "
                f"first failure: {failures[0]!r}"
            ) from failures[0]

        scores: list[float] = []
        for batch, result in zip(batches, returned, strict=True):
            # A short or long answer means scores are about to be zipped onto the
            # wrong passages -- every passage after the gap gets its neighbour's
            # relevance. That misranks silently, so it is fatal here.
            if len(result) != len(batch):
                raise RerankUnavailable(
                    f"scorer returned {len(result)} scores for a batch of {len(batch)}; "
                    "scores would be attributed to the wrong passages"
                )
            for value in result:
                score = float(value)
                if not math.isfinite(score):
                    # NaN compares false against everything and would corrupt the
                    # sort into an arbitrary order rather than an obvious failure.
                    raise RerankUnavailable(f"scorer returned a non-finite score: {value!r}")
                scores.append(score)
        return scores
