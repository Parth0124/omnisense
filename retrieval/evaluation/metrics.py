"""Retrieval quality metrics: recall@k, nDCG@k, MRR, precision, groundedness.

`docs/retrieval.md` §3 says the tuning defaults -- RRF's `k=60`, the rerank
depth, the fusion weights -- are starting points to be tuned against an
evaluation harness. This module is the arithmetic that harness runs on. Without
it, "we tuned the weights" means "we changed the weights and the output looked
better", which is how a retrieval system acquires numbers nobody can justify and
nobody dares change.

**Every metric here is rank-aware except one, and that is the point.** Precision
and recall treat a result set as a bag: moving the only relevant document from
position 1 to position 20 does not change either of them. For a system that feeds
the top-k into a model's context window, that is exactly the change that matters
-- position 20 falls off the end of the budget. nDCG and MRR are the metrics that
notice, and they are the ones to tune against. Recall is kept because it answers
a different and prior question: *is the answer in the candidate set at all*. A
reranker cannot promote a document that retrieval never returned, so a low
recall@100 with a high nDCG@10 means the reranker is doing well with a bad
candidate set, and the fix is upstream.

**Graded relevance, not binary.** `docs/retrieval.md`'s golden sets label
passages 0-3, and nDCG is defined over those grades. Collapsing them to
relevant/not-relevant discards the distinction between the passage that answers
the question and the passage that mentions the right company -- which is the
distinction the reranker exists to learn.

**Groundedness is a retrieval metric here, not a generation one.** It measures
what fraction of a claim's cited passages actually contain support for it. It
belongs in this module rather than in the agent evaluation because a claim can be
ungrounded for two very different reasons -- the model hallucinated, or retrieval
never surfaced the supporting passage -- and separating them requires measuring
against the retrieved set, which is what happens here.

Layer note: **L1 library** -- `models/` and the standard library. Pure functions
over lists of ids and grades; no clients, no I/O, no dependency on how the
candidates were produced.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

__all__ = [
    "DEFAULT_K_VALUES",
    "MAX_GRADE",
    "EvaluationSummary",
    "QueryEvaluation",
    "average_precision",
    "evaluate_query",
    "evaluate_run",
    "f1",
    "groundedness",
    "hit_rate",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]

MAX_GRADE: Final[int] = 3
"""Top of the relevance scale. 0 irrelevant, 1 marginal, 2 relevant, 3 perfect.

Four levels rather than five or ten because a human annotator cannot reliably
distinguish more than about four, and a scale with more levels than the labeller
can hold produces inconsistency that reads as measurement noise.
"""

DEFAULT_K_VALUES: Final[tuple[int, ...]] = (1, 3, 5, 10, 20, 50, 100)
"""Cutoffs reported by default.

Spans two regimes deliberately: 1-10 is what fits in a context window and is what
the reranker is judged on; 50-100 is the candidate set and is what the retrieval
backends are judged on. Reporting only the first hides an upstream recall
problem behind a reranker that is coping.
"""


def _relevance(grades: Mapping[str, int], doc_id: str) -> int:
    """Grade of a document, defaulting to 0.

    A document absent from the grade map is *unjudged*, and this treats it as
    irrelevant. That is the standard pooled-judgement assumption and it is worth
    being explicit about, because it biases every metric downward for a system
    that surfaces genuinely good documents nobody labelled -- which is precisely
    what happens when a new retrieval backend is added. A drop in nDCG after
    adding a backend should always be checked against the judgement pool before
    it is believed.
    """
    return max(0, min(MAX_GRADE, grades.get(doc_id, 0)))


def _is_relevant(grade: int, threshold: int) -> bool:
    return grade >= threshold


# --------------------------------------------------------------------------- #
# Set metrics
# --------------------------------------------------------------------------- #


def precision_at_k(
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    k: int,
    *,
    threshold: int = 1,
) -> float:
    """Fraction of the top k that are relevant.

    Divided by `k` rather than by `len(ranked_ids[:k])`. The distinction matters
    when a system returns fewer than k results: dividing by the returned count
    gives a system that returned one correct document a precision of 1.0, tying
    it with a system that returned ten correct ones. Dividing by k charges for
    the empty slots, which is the honest accounting -- the caller asked for k.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    top = ranked_ids[:k]
    relevant = sum(1 for doc_id in top if _is_relevant(_relevance(grades, doc_id), threshold))
    return relevant / k


def recall_at_k(
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    k: int,
    *,
    threshold: int = 1,
) -> float:
    """Fraction of all relevant documents that appear in the top k.

    The prior question, and the one to ask first: a reranker cannot promote a
    document retrieval never returned. High nDCG@10 with low recall@100 means the
    reranker is doing well with a bad candidate set, and no amount of reranking
    tuning will fix it.

    Returns 1.0 when nothing is relevant. That is the vacuous-truth convention
    and it is chosen over 0.0 because averaging is the point: a query with no
    relevant documents scored 0.0 drags the mean down for a system that had
    nothing to find and correctly found nothing.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    total_relevant = sum(1 for grade in grades.values() if _is_relevant(grade, threshold))
    if total_relevant == 0:
        return 1.0
    found = sum(
        1 for doc_id in ranked_ids[:k] if _is_relevant(_relevance(grades, doc_id), threshold)
    )
    return found / total_relevant


def f1(precision: float, recall: float) -> float:
    """Harmonic mean. Zero when either side is zero, by definition."""
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hit_rate(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int, *, threshold: int = 1) -> float:
    """1.0 if any relevant document is in the top k, else 0.0.

    Crude, and the right metric for exactly one question: "did we find anything
    at all". Averaged over a query set it reads as the fraction of questions the
    system could have answered, which is the number to quote to someone who does
    not want a lecture on discounted gain.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    return float(
        any(_is_relevant(_relevance(grades, doc_id), threshold) for doc_id in ranked_ids[:k])
    )


# --------------------------------------------------------------------------- #
# Rank-aware metrics
# --------------------------------------------------------------------------- #


def reciprocal_rank(
    ranked_ids: Sequence[str], grades: Mapping[str, int], *, threshold: int = 1
) -> float:
    """1 / rank of the first relevant document, or 0.0 if there is none.

    The metric for a question with one right answer -- "who acquired Acme" -- and
    the wrong one for "what are the complaints about Acme", where the top five
    matter equally. Using MRR for the second kind of query rewards a system that
    puts one good document first and nothing useful after it.
    """
    for position, doc_id in enumerate(ranked_ids, start=1):
        if _is_relevant(_relevance(grades, doc_id), threshold):
            return 1.0 / position
    return 0.0


def mean_reciprocal_rank(
    runs: Iterable[tuple[Sequence[str], Mapping[str, int]]], *, threshold: int = 1
) -> float:
    values = [reciprocal_rank(ranked, grades, threshold=threshold) for ranked, grades in runs]
    return sum(values) / len(values) if values else 0.0


def average_precision(
    ranked_ids: Sequence[str], grades: Mapping[str, int], *, threshold: int = 1
) -> float:
    """Mean of the precisions measured at each relevant document's position.

    Rank-aware and set-complete in a way neither precision@k nor MRR is: it
    rewards putting *all* the relevant documents high, not just the first one.
    The natural single number when a query has several right answers.
    """
    total_relevant = sum(1 for grade in grades.values() if _is_relevant(grade, threshold))
    if total_relevant == 0:
        return 1.0
    hits = 0
    accumulated = 0.0
    for position, doc_id in enumerate(ranked_ids, start=1):
        if _is_relevant(_relevance(grades, doc_id), threshold):
            hits += 1
            accumulated += hits / position
    return accumulated / total_relevant


def ndcg_at_k(ranked_ids: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Normalised discounted cumulative gain over graded relevance.

    The metric to tune against, because it is the only one here that answers the
    question the system actually faces: given a fixed context budget, how much
    total usefulness did the top k deliver, weighted by how early it arrived.

    `(2^grade - 1)` as the gain, `log2(position + 1)` as the discount -- the
    standard formulation. The exponential gain is what makes a grade-3 passage
    worth more than two grade-2 passages (7 versus 6), which matches how a model
    reading the context actually behaves: one passage that answers the question
    is worth more than two that circle it.

    The ideal ranking is computed from the *judgements*, not from the run, so
    a system cannot score 1.0 by returning a single mediocre document and having
    nothing to be compared against.

    Returns 1.0 when no document has a positive grade -- there is nothing to
    rank, and 0.0 would punish a system for a query with no answer.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    def dcg(relevances: Sequence[int]) -> float:
        return sum(
            (2**grade - 1) / math.log2(position + 1)
            for position, grade in enumerate(relevances, start=1)
            if grade > 0
        )

    actual = dcg([_relevance(grades, doc_id) for doc_id in ranked_ids[:k]])
    ideal = dcg(sorted((g for g in grades.values() if g > 0), reverse=True)[:k])
    if ideal <= 0:
        return 1.0
    return actual / ideal


# --------------------------------------------------------------------------- #
# Groundedness
# --------------------------------------------------------------------------- #


def groundedness(
    cited_ids: Sequence[str],
    supporting_ids: Iterable[str],
) -> float:
    """Fraction of cited passages that actually support the claim.

    A retrieval metric rather than a generation one, and deliberately so. A claim
    can be ungrounded for two very different reasons -- the model invented it, or
    retrieval never surfaced the passage that would have supported it -- and only
    measuring against the retrieved set separates them.

    Returns 0.0 for a claim with no citations at all. Not 1.0, and this is the
    one convention here that deliberately breaks the vacuous-truth pattern used
    by `recall_at_k`: an uncited claim is the *worst* case in a system whose
    premise is that every claim carries a citation, and scoring it as perfectly
    grounded would let a model maximise the metric by citing nothing.
    """
    if not cited_ids:
        return 0.0
    supporting = set(supporting_ids)
    return sum(1 for doc_id in cited_ids if doc_id in supporting) / len(cited_ids)


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    """Every metric for one query, at every requested cutoff."""

    query_id: str
    retrieved: int
    relevant_available: int
    precision: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]
    hit: dict[int, float]
    reciprocal_rank: float
    average_precision: float

    @property
    def found_nothing(self) -> bool:
        """True when no relevant document was retrieved at any depth.

        The queries worth reading individually. An aggregate nDCG of 0.6 can be
        an even spread or it can be forty perfect queries and twenty total
        failures, and those call for completely different work.
        """
        return self.reciprocal_rank == 0.0 and self.relevant_available > 0


def evaluate_query(
    query_id: str,
    ranked_ids: Sequence[str],
    grades: Mapping[str, int],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    threshold: int = 1,
) -> QueryEvaluation:
    """Compute every metric for one query.

    Duplicate ids in `ranked_ids` are removed, keeping the first occurrence. A
    duplicate is always a bug -- fusion collapsing two backends' hits, or a
    chunker emitting the same chunk twice -- and left in place it inflates
    precision and nDCG by filling the top k with one document. Removing it here
    means the metric reports the quality of the *distinct* results, which is what
    the context window will actually contain.
    """
    seen: set[str] = set()
    deduped = [doc_id for doc_id in ranked_ids if not (doc_id in seen or seen.add(doc_id))]

    return QueryEvaluation(
        query_id=query_id,
        retrieved=len(deduped),
        relevant_available=sum(1 for g in grades.values() if _is_relevant(g, threshold)),
        precision={k: precision_at_k(deduped, grades, k, threshold=threshold) for k in k_values},
        recall={k: recall_at_k(deduped, grades, k, threshold=threshold) for k in k_values},
        ndcg={k: ndcg_at_k(deduped, grades, k) for k in k_values},
        hit={k: hit_rate(deduped, grades, k, threshold=threshold) for k in k_values},
        reciprocal_rank=reciprocal_rank(deduped, grades, threshold=threshold),
        average_precision=average_precision(deduped, grades, threshold=threshold),
    )


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Means across a query set, plus the failures worth looking at by hand."""

    query_count: int
    precision: dict[int, float]
    recall: dict[int, float]
    ndcg: dict[int, float]
    hit: dict[int, float]
    mrr: float
    map_score: float
    failed_query_ids: tuple[str, ...]

    def format_table(self) -> str:
        """A fixed-width table, for a CI log or a terminal.

        Present because a summary nobody reads is a summary nobody acts on, and
        a dict repr of nested floats is not read by anyone.
        """
        lines = [
            f"queries={self.query_count}  MRR={self.mrr:.4f}  MAP={self.map_score:.4f}",
            f"{'k':>5} {'P@k':>8} {'R@k':>8} {'nDCG@k':>8} {'Hit@k':>8}",
        ]
        for k in sorted(self.ndcg):
            lines.append(
                f"{k:>5} {self.precision[k]:>8.4f} {self.recall[k]:>8.4f} "
                f"{self.ndcg[k]:>8.4f} {self.hit[k]:>8.4f}"
            )
        if self.failed_query_ids:
            shown = ", ".join(self.failed_query_ids[:10])
            more = "" if len(self.failed_query_ids) <= 10 else f" (+{len(self.failed_query_ids) - 10})"
            lines.append(f"found nothing: {shown}{more}")
        return "\n".join(lines)


def evaluate_run(
    evaluations: Sequence[QueryEvaluation],
) -> EvaluationSummary:
    """Average per-query metrics into a run summary.

    **Macro-averaged**: every query counts once, regardless of how many relevant
    documents it has. Micro-averaging (pooling all judgements and computing once)
    would let a handful of queries with fifty relevant documents each dominate a
    set of two hundred queries with two -- so the reported number would describe
    the behaviour of the outliers and be reported as the behaviour of the system.
    """
    if not evaluations:
        return EvaluationSummary(0, {}, {}, {}, {}, 0.0, 0.0, ())

    k_values = sorted(evaluations[0].ndcg)
    count = len(evaluations)

    def mean_over(attribute: str, k: int) -> float:
        return sum(getattr(item, attribute)[k] for item in evaluations) / count

    return EvaluationSummary(
        query_count=count,
        precision={k: mean_over("precision", k) for k in k_values},
        recall={k: mean_over("recall", k) for k in k_values},
        ndcg={k: mean_over("ndcg", k) for k in k_values},
        hit={k: mean_over("hit", k) for k in k_values},
        mrr=sum(item.reciprocal_rank for item in evaluations) / count,
        map_score=sum(item.average_precision for item in evaluations) / count,
        failed_query_ids=tuple(item.query_id for item in evaluations if item.found_nothing),
    )
