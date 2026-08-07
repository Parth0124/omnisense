"""Unit tests for `retrieval/evaluation/metrics.py`.

Every expected value here is computed by hand in the test that asserts it, so a
failure says which arithmetic is wrong rather than that two implementations
disagree.
"""

from __future__ import annotations

import math

import pytest

from retrieval.evaluation.metrics import (
    DEFAULT_K_VALUES,
    average_precision,
    evaluate_query,
    evaluate_run,
    f1,
    groundedness,
    hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

pytestmark = pytest.mark.unit

# Three relevant documents, graded. d4 and d5 are judged irrelevant; d9 is
# unjudged, which the module treats as irrelevant.
GRADES = {"d1": 3, "d2": 2, "d3": 1, "d4": 0, "d5": 0}


class TestPrecision:
    def test_counts_relevant_in_the_top_k(self) -> None:
        assert precision_at_k(["d1", "d4", "d2"], GRADES, 3) == pytest.approx(2 / 3)

    def test_divides_by_k_not_by_what_was_returned(self) -> None:
        """A system returning one correct document must not tie with one
        returning ten. Dividing by the returned count would give both 1.0."""
        assert precision_at_k(["d1"], GRADES, 10) == pytest.approx(0.1)

    def test_threshold_excludes_marginal_grades(self) -> None:
        assert precision_at_k(["d3"], GRADES, 1) == 1.0
        assert precision_at_k(["d3"], GRADES, 1, threshold=2) == 0.0

    def test_zero_k_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            precision_at_k(["d1"], GRADES, 0)


class TestRecall:
    def test_fraction_of_all_relevant_found(self) -> None:
        assert recall_at_k(["d1", "d2"], GRADES, 10) == pytest.approx(2 / 3)

    def test_is_capped_by_the_cutoff(self) -> None:
        assert recall_at_k(["d1", "d2", "d3"], GRADES, 1) == pytest.approx(1 / 3)

    def test_no_relevant_documents_is_vacuously_perfect(self) -> None:
        """Scoring 0.0 would drag the mean down for a system that had nothing to
        find and correctly found nothing."""
        assert recall_at_k(["d4"], {"d4": 0}, 5) == 1.0

    def test_unjudged_documents_count_as_irrelevant(self) -> None:
        """The pooled-judgement assumption, and it biases against a new backend
        that surfaces good documents nobody labelled -- which is why a drop after
        adding one should be checked against the pool before it is believed."""
        assert recall_at_k(["d9"], GRADES, 5) == 0.0


class TestF1:
    def test_harmonic_mean(self) -> None:
        assert f1(0.5, 0.5) == pytest.approx(0.5)

    def test_zero_on_either_side_is_zero(self) -> None:
        assert f1(1.0, 0.0) == 0.0


class TestHitRate:
    def test_one_when_anything_relevant_is_present(self) -> None:
        assert hit_rate(["d4", "d3"], GRADES, 2) == 1.0

    def test_zero_when_nothing_relevant_is_present(self) -> None:
        assert hit_rate(["d4", "d5"], GRADES, 2) == 0.0


class TestReciprocalRank:
    def test_is_the_inverse_of_the_first_relevant_position(self) -> None:
        assert reciprocal_rank(["d4", "d5", "d1"], GRADES) == pytest.approx(1 / 3)

    def test_zero_when_nothing_relevant_is_retrieved(self) -> None:
        assert reciprocal_rank(["d4", "d5"], GRADES) == 0.0

    def test_ignores_everything_after_the_first_hit(self) -> None:
        """The property that makes MRR wrong for a multi-answer question: a
        system with one good result and nothing after it scores identically to
        one that got everything right."""
        assert reciprocal_rank(["d1", "d4"], GRADES) == reciprocal_rank(
            ["d1", "d2", "d3"], GRADES
        )

    def test_mean_averages_over_queries(self) -> None:
        runs = [(["d1"], GRADES), (["d4", "d2"], GRADES)]
        assert mean_reciprocal_rank(runs) == pytest.approx((1.0 + 0.5) / 2)


class TestAveragePrecision:
    def test_rewards_ranking_all_relevant_documents_high(self) -> None:
        perfect = average_precision(["d1", "d2", "d3"], GRADES)
        scattered = average_precision(["d1", "d4", "d2", "d5", "d3"], GRADES)
        assert perfect == pytest.approx(1.0)
        assert scattered < perfect

    def test_matches_a_hand_computation(self) -> None:
        # hits at positions 1 and 3 -> (1/1 + 2/3) / 3 relevant
        assert average_precision(["d1", "d4", "d2"], GRADES) == pytest.approx(
            (1.0 + 2 / 3) / 3
        )


class TestNdcg:
    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["d1", "d2", "d3"], GRADES, 3) == pytest.approx(1.0)

    def test_reversed_ranking_scores_below_one(self) -> None:
        assert ndcg_at_k(["d3", "d2", "d1"], GRADES, 3) < 1.0

    def test_matches_a_hand_computation(self) -> None:
        # gains 2^3-1=7 at pos 1, 2^1-1=1 at pos 2 -> 7/log2(2) + 1/log2(3)
        # ideal: 7/log2(2) + 3/log2(3)   (grades 3 then 2)
        actual = 7 / math.log2(2) + 1 / math.log2(3)
        ideal = 7 / math.log2(2) + 3 / math.log2(3)
        assert ndcg_at_k(["d1", "d3"], GRADES, 2) == pytest.approx(actual / ideal)

    def test_grade_three_outweighs_two_grade_twos(self) -> None:
        """The exponential gain, and why it is right: one passage that answers
        the question is worth more than two that circle it."""
        grades = {"perfect": 3, "ok_a": 2, "ok_b": 2}
        assert ndcg_at_k(["perfect"], grades, 1) > ndcg_at_k(["ok_a"], grades, 1)

    def test_is_rank_aware_where_precision_is_not(self) -> None:
        """The whole reason nDCG is the metric to tune against: moving the only
        relevant document from position 1 to position 10 falls off the context
        budget, and precision@10 does not notice."""
        early = ["d1"] + ["d4"] * 9
        late = ["d4"] * 9 + ["d1"]
        assert precision_at_k(early, GRADES, 10) == precision_at_k(late, GRADES, 10)
        assert ndcg_at_k(early, GRADES, 10) > ndcg_at_k(late, GRADES, 10)

    def test_ideal_comes_from_judgements_not_from_the_run(self) -> None:
        """Otherwise a system returning one mediocre document scores 1.0 by
        having nothing to be compared against."""
        assert ndcg_at_k(["d3"], GRADES, 1) < 1.0

    def test_nothing_relevant_is_vacuously_perfect(self) -> None:
        assert ndcg_at_k(["a"], {"a": 0}, 1) == 1.0


class TestGroundedness:
    def test_fraction_of_citations_that_are_supported(self) -> None:
        assert groundedness(["p1", "p2", "p3"], {"p1", "p3"}) == pytest.approx(2 / 3)

    def test_an_uncited_claim_scores_zero_not_one(self) -> None:
        """The one place the vacuous-truth convention is deliberately broken:
        scoring it 1.0 would let a model maximise groundedness by citing
        nothing."""
        assert groundedness([], {"p1"}) == 0.0

    def test_fully_supported_is_one(self) -> None:
        assert groundedness(["p1"], {"p1", "p2"}) == 1.0


class TestQueryEvaluation:
    def test_duplicates_are_removed_before_scoring(self) -> None:
        """A duplicate is always a bug -- fusion collapsing two backends' hits,
        or a chunker emitting a chunk twice -- and left in place it inflates
        precision by filling the top k with one document."""
        result = evaluate_query("q1", ["d1", "d1", "d1"], GRADES, k_values=(3,))
        assert result.retrieved == 1
        assert result.precision[3] == pytest.approx(1 / 3)

    def test_reports_every_requested_cutoff(self) -> None:
        result = evaluate_query("q1", ["d1"], GRADES)
        assert set(result.ndcg) == set(DEFAULT_K_VALUES)

    def test_total_failure_is_flagged(self) -> None:
        """An aggregate nDCG of 0.6 can be an even spread or forty perfect
        queries and twenty total failures, and those call for different work."""
        assert evaluate_query("q1", ["d4", "d5"], GRADES).found_nothing

    def test_a_query_with_no_answer_is_not_a_failure(self) -> None:
        assert not evaluate_query("q1", ["x"], {"x": 0}).found_nothing


class TestRunSummary:
    def test_macro_averages_over_queries(self) -> None:
        """Micro-averaging would let a handful of queries with fifty relevant
        documents dominate two hundred with two -- so the reported number
        describes the outliers and is quoted as the system."""
        evaluations = [
            evaluate_query("perfect", ["d1", "d2", "d3"], GRADES, k_values=(3,)),
            evaluate_query("empty", ["d4"], GRADES, k_values=(3,)),
        ]
        summary = evaluate_run(evaluations)
        assert summary.query_count == 2
        assert summary.ndcg[3] == pytest.approx(
            (evaluations[0].ndcg[3] + evaluations[1].ndcg[3]) / 2
        )

    def test_collects_the_queries_worth_inspecting(self) -> None:
        evaluations = [
            evaluate_query("good", ["d1"], GRADES, k_values=(3,)),
            evaluate_query("bad", ["d4"], GRADES, k_values=(3,)),
        ]
        assert evaluate_run(evaluations).failed_query_ids == ("bad",)

    def test_empty_run_is_not_a_division_error(self) -> None:
        assert evaluate_run([]).query_count == 0

    def test_table_is_readable(self) -> None:
        summary = evaluate_run([evaluate_query("q", ["d1"], GRADES, k_values=(1, 3))])
        rendered = summary.format_table()
        assert "nDCG@k" in rendered
        assert "MRR=" in rendered
