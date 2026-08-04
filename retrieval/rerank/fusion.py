"""Reciprocal rank fusion across heterogeneous result lists.

Three backends return incomparable numbers. BM25 scores are unbounded and depend
on corpus statistics, cosine similarities sit in [-1, 1], and graph proximity is
a hop count. Adding or averaging them requires a calibration that would have to
be re-derived every time the corpus grows.

RRF sidesteps that entirely by discarding the scores and using only **rank**:

    score(c) = Σ_b  w_b / (k_rrf + r_b(c))

A document at rank 1 in one backend and rank 40 in another scores sensibly
without anyone deciding how many BM25 points a cosine point is worth. Absent
backends contribute nothing -- no imputed rank, no penalty -- which is what keeps
fusion stable when graph traversal returns four results for a sparse entity
while the other two return a hundred each.

The one thing here that is easy to get wrong, and expensive:

    **Fuse first, then dedupe.**

A chunk found by all three backends must accumulate three RRF terms *before*
collapsing, because being found three ways is the strongest evidence of
relevance available. Deduping first throws that away and leaves the chunk with a
single term, ranked as though only one backend ever saw it. The bug is invisible
in tests that use one backend and shows up as mediocre ranking in production.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from retrieval.types import Backend, Candidate, Passage

__all__ = [
    "DEFAULT_BACKEND_WEIGHTS",
    "DEFAULT_RRF_K",
    "FusedCandidate",
    "collapse_near_duplicates",
    "reciprocal_rank_fusion",
]

DEFAULT_RRF_K = 60
"""The conventional RRF constant.

Damps how much rank 1 outweighs rank 5. Lower trusts top ranks more; the value
from the original RRF paper is 60 and `docs/retrieval.md` §3 adopts it as an
untuned starting point.
"""

DEFAULT_BACKEND_WEIGHTS: Mapping[Backend, float] = {
    Backend.KEYWORD: 1.0,
    Backend.VECTOR: 1.0,
    Backend.GRAPH: 0.8,
}
"""Per-backend weights.

Graph is weighted slightly below the other two: its passages are high-precision
but low-recall, so a short graph list would otherwise dominate rank positions
purely by being short. Untuned -- validate against `retrieval/evaluation/`
before changing.
"""


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """One chunk after fusion, carrying how every backend ranked it."""

    chunk_id: str
    signal_id: str
    score: float
    ranks: Mapping[Backend, int]
    raw_scores: Mapping[Backend, float]

    @property
    def found_by(self) -> frozenset[Backend]:
        return frozenset(self.ranks)

    @property
    def backend_count(self) -> int:
        """How many backends independently surfaced this chunk.

        Worth reading directly: agreement across backends is a corroboration
        signal in its own right, not just an input to the score.
        """
        return len(self.ranks)


def reciprocal_rank_fusion(
    results: Mapping[Backend, Sequence[Candidate]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[Backend, float] | None = None,
    pool_max: int | None = None,
) -> list[FusedCandidate]:
    """Fuse per-backend candidate lists into one ranked pool.

    Ties are broken deterministically -- by backend count, then by best rank
    achieved in any backend, then by `chunk_id`. Determinism matters more than
    the specific tiebreak: without it the same query returns different orderings
    across runs, and an evaluation harness measuring a one-point nDCG change
    cannot tell a real regression from dictionary iteration order.

    `pool_max` truncates *after* fusion, never before, so a chunk ranked poorly
    by one backend can still be rescued by agreement from the other two.
    """
    table = dict(weights) if weights is not None else dict(DEFAULT_BACKEND_WEIGHTS)
    accumulated: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[Backend, int]] = defaultdict(dict)
    raw: dict[str, dict[Backend, float]] = defaultdict(dict)
    signal_ids: dict[str, str] = {}

    for backend, candidates in results.items():
        weight = table.get(backend, 1.0)
        if weight == 0.0:
            continue
        for candidate in candidates:
            if candidate.backend is not backend:
                raise ValueError(
                    f"candidate {candidate.chunk_id!r} is tagged {candidate.backend!r} "
                    f"but appears in the {backend!r} list; a mislabelled candidate "
                    "would be weighted as the wrong backend"
                )
            accumulated[candidate.chunk_id] += weight / (k + candidate.rank)
            # A backend returning the same chunk twice keeps the better rank
            # rather than double-counting it, which would let one backend
            # manufacture agreement with itself.
            previous = ranks[candidate.chunk_id].get(backend)
            if previous is None or candidate.rank < previous:
                ranks[candidate.chunk_id][backend] = candidate.rank
                raw[candidate.chunk_id][backend] = candidate.raw_score
            signal_ids.setdefault(candidate.chunk_id, candidate.signal_id)

    fused = [
        FusedCandidate(
            chunk_id=chunk_id,
            signal_id=signal_ids.get(chunk_id, ""),
            score=score,
            ranks=dict(ranks[chunk_id]),
            raw_scores=dict(raw[chunk_id]),
        )
        for chunk_id, score in accumulated.items()
    ]

    fused.sort(
        key=lambda c: (
            -c.score,
            -c.backend_count,
            min(c.ranks.values()) if c.ranks else 1 << 30,
            c.chunk_id,
        )
    )
    return fused[:pool_max] if pool_max is not None else fused


def collapse_near_duplicates(
    passages: Sequence[Passage],
    *,
    simhash_of: Callable[[Passage], int | None],
    max_distance: int = 3,
) -> list[Passage]:
    """Collapse near-identical passages, keeping the best-ranked member.

    Runs **after** fusion, for the reason in the module docstring. Exact
    duplicates are already gone by then -- `chunk_id` is unique in the fused pool
    -- so this catches the same story republished across platforms, which shares
    no chunk id but says the same thing.

    The collapsed siblings are recorded on the survivor rather than discarded.
    Six outlets carrying one press release is corroboration a report should be
    able to state ("reported by 6 sources"), and it feeds the corroboration term
    in Signal confidence. Deleting them would destroy exactly the evidence that
    makes the passage worth trusting -- the same reasoning that makes ingestion
    cluster near-duplicates instead of dropping them
    (`docs/signal-model.md` §4.3).

    O(n·k) over the surviving set rather than O(n²) over everything: the pool is
    capped at ~150 by then, so a full pairwise comparison would be affordable but
    pointless.
    """
    survivors: list[Passage] = []
    survivor_hashes: list[tuple[int, int]] = []  # (index into survivors, simhash)

    for passage in passages:
        fingerprint = simhash_of(passage)
        if fingerprint is None:
            survivors.append(passage)
            continue

        merged_into: int | None = None
        for index, existing in survivor_hashes:
            if _hamming(fingerprint, existing) <= max_distance:
                merged_into = index
                break

        if merged_into is None:
            survivor_hashes.append((len(survivors), fingerprint))
            survivors.append(passage)
            continue

        keeper = survivors[merged_into]
        # Passages arrive in fused order, so the incumbent already outranks the
        # newcomer; the newcomer only contributes corroboration.
        survivors[merged_into] = _absorb(keeper, passage)

    return survivors


def _absorb(keeper: Passage, absorbed: Passage) -> Passage:
    """Fold a near-duplicate into its survivor, preserving the evidence of spread."""
    import dataclasses

    collapsed = [*keeper.collapsed_signal_ids]
    if absorbed.signal_id != keeper.signal_id and absorbed.signal_id not in collapsed:
        collapsed.append(absorbed.signal_id)

    return dataclasses.replace(
        keeper,
        duplicate_of_count=keeper.duplicate_of_count + 1,
        collapsed_signal_ids=tuple(collapsed),
        found_by=keeper.found_by | absorbed.found_by,
    )


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()
