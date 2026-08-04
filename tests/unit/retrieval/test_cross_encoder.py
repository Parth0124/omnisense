"""Unit tests for `retrieval/rerank/cross_encoder.py`.

The reranker is the one component in the pipeline that is both expensive and
easy to get *quietly* wrong. Three of its failure modes produce a plausible-
looking ranking rather than an error, so they are tested directly:

- scores zipped onto the wrong passages when the port returns a short batch,
  which silently gives every passage after the gap its neighbour's relevance;
- a partially reranked list, mixing cross-encoder scores with fused scores that
  live on a different scale, so a passage's position depends on which batch it
  landed in;
- a non-deterministic tie order, which makes a one-point nDCG change in the
  evaluation harness indistinguishable between a regression and a reshuffle.

Everything else here is about *cost control*: only `depth` passages are scored,
they are scored in batches, and no more than `max_concurrency` batches are in
flight -- the reranker is roughly two orders of magnitude more expensive per item
than the fusion it refines, and an unbounded fan-out from one query would queue
ahead of every other query on a shared GPU.

No network, no services, no model: scoring arrives as a fake port.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from retrieval.hybrid import Reranker
from retrieval.rerank.cross_encoder import (
    CrossEncoderReranker,
    PairScorer,
    RerankUnavailable,
)
from retrieval.types import Passage, chunk_id_for

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def make_passage(index: int, *, signal: str | None = None, fused: float = 0.0) -> Passage:
    """One passage whose text encodes its index, so scorers can key off it."""
    signal_id = signal or f"sig_{index:03d}"
    return Passage(
        chunk_id=chunk_id_for(signal_id, 0),
        signal_id=signal_id,
        text=f"passage number {index}",
        char_start=0,
        char_end=20,
        fused_score=fused,
    )


def make_passages(count: int) -> list[Passage]:
    """`count` passages in descending fused order, as fusion would hand them over."""
    return [make_passage(i, fused=1.0 - i / 1000) for i in range(count)]


class RecordingScorer:
    """A `PairScorer` that records what it was asked, and answers on demand."""

    def __init__(self, score_of=None) -> None:
        self.calls: list[list[str]] = []
        self.queries: list[str] = []
        self._score_of = score_of or (lambda text: 0.0)

    async def __call__(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        self.queries.append(query)
        self.calls.append(list(texts))
        return [self._score_of(text) for text in texts]


def index_of(text: str) -> int:
    """Recover the index encoded by `make_passage` from a passage's text."""
    return int(text.rsplit(" ", 1)[1])


# --------------------------------------------------------------------------- #
# The protocol contract
# --------------------------------------------------------------------------- #


async def test_satisfies_the_reranker_protocol() -> None:
    """`HybridRetriever` accepts anything shaped like `Reranker`; this must be."""
    reranker = CrossEncoderReranker(RecordingScorer(), depth=10)
    assert isinstance(reranker, Reranker)


async def test_scores_are_attached_without_disturbing_the_fused_score() -> None:
    """`final_score` switches to the rerank score; `fused_score` stays auditable.

    Overwriting `fused_score` would erase the evidence of what fusion thought,
    which is the only way to tell "the reranker disagreed" from "fusion never
    surfaced it".
    """
    scorer = RecordingScorer(lambda text: 0.5)
    reranker = CrossEncoderReranker(scorer, depth=10)

    [passage] = await reranker.rerank("q", [make_passage(0, fused=0.02)], top_k=1)

    assert passage.rerank_score == 0.5
    assert passage.fused_score == 0.02
    assert passage.final_score == 0.5


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


async def test_reorders_by_score_and_truncates_to_top_k() -> None:
    """The point of the whole component: fused order is not the final order."""
    # Reverse the fused order: passage 4 is the most relevant, 0 the least.
    scorer = RecordingScorer(lambda text: float(index_of(text)))
    reranker = CrossEncoderReranker(scorer, depth=10)

    ranked = await reranker.rerank("q", make_passages(5), top_k=3)

    assert [index_of(p.text) for p in ranked] == [4, 3, 2]


async def test_equal_scores_keep_fused_order() -> None:
    """Deterministic ties, or the eval harness cannot attribute a metric change."""
    scorer = RecordingScorer(lambda text: 1.0)
    reranker = CrossEncoderReranker(scorer, depth=10)
    passages = make_passages(6)

    first = await reranker.rerank("q", passages, top_k=6)
    second = await reranker.rerank("q", passages, top_k=6)

    assert [p.chunk_id for p in first] == [p.chunk_id for p in passages]
    assert [p.chunk_id for p in first] == [p.chunk_id for p in second]


async def test_empty_input_is_not_a_call_to_the_scorer() -> None:
    scorer = RecordingScorer()
    reranker = CrossEncoderReranker(scorer, depth=10)

    assert await reranker.rerank("q", [], top_k=5) == []
    assert scorer.calls == []


# --------------------------------------------------------------------------- #
# Cost control
# --------------------------------------------------------------------------- #


async def test_only_depth_passages_are_scored() -> None:
    """The expensive step is bounded, not applied to the whole pool."""
    scorer = RecordingScorer(lambda text: 1.0)
    reranker = CrossEncoderReranker(scorer, depth=4, batch_size=100)

    await reranker.rerank("q", make_passages(30), top_k=12)

    assert sum(len(call) for call in scorer.calls) == 4


async def test_unscored_tail_is_kept_behind_the_scored_head() -> None:
    """Passages past `depth` are not dropped, and are not given a fake score.

    They sit behind everything the cross-encoder saw. Interleaving them by
    `fused_score` would compare two incomparable scales.
    """
    # Score the head *below* what a fused score looks like, so an implementation
    # that mixed the scales would visibly interleave.
    scorer = RecordingScorer(lambda text: -10.0)
    reranker = CrossEncoderReranker(scorer, depth=2, batch_size=2)

    ranked = await reranker.rerank("q", make_passages(5), top_k=5)

    assert [index_of(p.text) for p in ranked] == [0, 1, 2, 3, 4]
    assert [p.rerank_score for p in ranked] == [-10.0, -10.0, None, None, None]


async def test_pairs_are_batched_not_sent_one_by_one() -> None:
    scorer = RecordingScorer(lambda text: 1.0)
    reranker = CrossEncoderReranker(scorer, depth=50, batch_size=16)

    await reranker.rerank("q", make_passages(50), top_k=12)

    assert [len(call) for call in scorer.calls] == [16, 16, 16, 2]
    assert scorer.queries == ["q"] * 4


async def test_concurrency_is_bounded() -> None:
    """At most `max_concurrency` batches in flight, however many batches there are."""
    in_flight = 0
    peak = 0

    async def scorer(query: str, texts: Sequence[str]) -> Sequence[float]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        for _ in range(3):  # yield, so any other admitted batch can start
            await asyncio.sleep(0)
        in_flight -= 1
        return [1.0] * len(texts)

    reranker = CrossEncoderReranker(scorer, depth=50, batch_size=5, max_concurrency=2)
    ranked = await reranker.rerank("q", make_passages(50), top_k=50)

    assert peak == 2
    assert len(ranked) == 50


async def test_batches_do_run_concurrently() -> None:
    """The bound is a ceiling, not a serialisation: latency is max, not sum."""
    started = asyncio.Event()
    release = asyncio.Event()
    seen = 0

    async def scorer(query: str, texts: Sequence[str]) -> Sequence[float]:
        nonlocal seen
        seen += 1
        if seen == 2:
            started.set()
        await release.wait()
        return [1.0] * len(texts)

    reranker = CrossEncoderReranker(
        scorer, depth=4, batch_size=1, max_concurrency=4, timeout_seconds=None
    )
    task = asyncio.create_task(reranker.rerank("q", make_passages(4), top_k=4))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    release.set()

    assert len(await asyncio.wait_for(task, timeout=1.0)) == 4


# --------------------------------------------------------------------------- #
# Failure is total, never partial
# --------------------------------------------------------------------------- #


async def test_a_failing_batch_fails_the_whole_rerank() -> None:
    """No half-reranked list: the caller falls back to a coherent fused order."""
    calls = 0

    async def scorer(query: str, texts: Sequence[str]) -> Sequence[float]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("reranker endpoint refused the connection")
        return [1.0] * len(texts)

    reranker = CrossEncoderReranker(scorer, depth=10, batch_size=2, timeout_seconds=None)

    with pytest.raises(RerankUnavailable, match="1 of 5 rerank batches failed"):
        await reranker.rerank("q", make_passages(10), top_k=5)


async def test_timeout_degrades_rather_than_hangs() -> None:
    """The 800 ms p95 budget in `docs/retrieval.md` §10 is enforced here, not hoped for."""

    async def slow(query: str, texts: Sequence[str]) -> Sequence[float]:
        await asyncio.sleep(5)
        return [1.0] * len(texts)

    reranker = CrossEncoderReranker(slow, depth=4, timeout_seconds=0.01)

    with pytest.raises(RerankUnavailable, match="budget"):
        await reranker.rerank("q", make_passages(4), top_k=4)


async def test_a_short_batch_of_scores_is_fatal() -> None:
    """Because zipping it would give passages their neighbours' relevance."""

    async def short(query: str, texts: Sequence[str]) -> Sequence[float]:
        return [1.0] * (len(texts) - 1)

    reranker = CrossEncoderReranker(short, depth=4, batch_size=4)

    with pytest.raises(RerankUnavailable, match="wrong passages"):
        await reranker.rerank("q", make_passages(4), top_k=4)


async def test_a_non_finite_score_is_fatal() -> None:
    """NaN compares false against everything and would corrupt the sort silently."""

    async def nan_scorer(query: str, texts: Sequence[str]) -> Sequence[float]:
        return [float("nan")] * len(texts)

    reranker = CrossEncoderReranker(nan_scorer, depth=4)

    with pytest.raises(RerankUnavailable, match="non-finite"):
        await reranker.rerank("q", make_passages(4), top_k=4)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


async def test_depth_defaults_to_the_configured_rerank_depth() -> None:
    """Config comes from `get_settings()`, read at construction and never at import."""
    from backend.core.config import get_settings

    reranker = CrossEncoderReranker(RecordingScorer())

    assert reranker.depth == get_settings().retrieval.rerank_depth


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"max_concurrency": 0},
        {"depth": 0},
        {"timeout_seconds": 0},
        {"timeout_seconds": -1.0},
    ],
)
async def test_incoherent_configuration_is_rejected_at_construction(kwargs) -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(RecordingScorer(), **{"depth": 10, **kwargs})


async def test_top_k_must_be_positive() -> None:
    reranker = CrossEncoderReranker(RecordingScorer(), depth=10)

    with pytest.raises(ValueError, match="top_k"):
        await reranker.rerank("q", make_passages(3), top_k=0)


async def test_a_plain_async_function_satisfies_the_port() -> None:
    """The port is deliberately a callable, so an LLM call or a lambda can serve it."""

    async def scorer(query: str, texts: Sequence[str]) -> Sequence[float]:
        return [float(len(text)) for text in texts]

    assert isinstance(scorer, PairScorer)
    ranked = await CrossEncoderReranker(scorer, depth=3).rerank("q", make_passages(3), top_k=1)
    assert len(ranked) == 1
