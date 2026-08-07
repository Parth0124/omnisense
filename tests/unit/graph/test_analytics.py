"""Unit tests for `graph/analytics/`.

Every graph here has an answer that can be worked out by hand, which is the only
way to test a scoring algorithm meaningfully -- asserting that PageRank "returns
some numbers" passes against an implementation that returns the uniform
distribution, and that is a real failure mode for a power iteration whose
dangling-mass handling is wrong.
"""

from __future__ import annotations

import pytest

from graph.analytics.centrality import (
    CentralityResult,
    Projection,
    betweenness_centrality,
    degree_centrality,
    pagerank,
    projection_from_rows,
)
from graph.analytics.communities import (
    community_id_for,
    label_propagation,
    louvain,
    modularity,
)

pytestmark = pytest.mark.unit


def _path(*nodes: str) -> Projection:
    """A---B---C---... with unit weights."""
    projection = Projection()
    for left, right in zip(nodes, nodes[1:], strict=False):
        projection.add_edge(left, right, 1.0)
    return projection


def _clique(prefix: str, size: int) -> list[tuple[str, str]]:
    names = [f"{prefix}{index}" for index in range(size)]
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]


def _two_cliques(bridge: bool = True) -> Projection:
    """Two 5-cliques joined by a single edge. The textbook community graph."""
    projection = Projection()
    for left, right in _clique("a", 5) + _clique("b", 5):
        projection.add_edge(left, right, 1.0)
    if bridge:
        projection.add_edge("a0", "b0", 1.0)
    return projection


class TestProjection:
    def test_edges_are_undirected(self) -> None:
        projection = Projection()
        projection.add_edge("x", "y", 2.0)
        assert projection.adjacency["y"]["x"] == 2.0

    def test_self_loops_are_dropped(self) -> None:
        """A self-loop inflates degree without connecting anything.

        Resolution produces them: merging two nodes that had an edge between them
        leaves the survivor pointing at itself.
        """
        projection = Projection()
        projection.add_edge("x", "x", 5.0)
        assert projection.weighted_degree("x") == 0.0

    def test_parallel_edges_accumulate(self) -> None:
        """Two entities linked by both COMPETES_WITH and MENTIONS are more
        connected than two linked by one, and row order must not decide."""
        projection = Projection()
        projection.add_edge("x", "y", 1.0)
        projection.add_edge("x", "y", 0.5)
        assert projection.adjacency["x"]["y"] == 1.5

    def test_isolated_nodes_survive(self) -> None:
        """An extracted-but-unlinked entity still needs a score.

        Deriving the node set from the edge list would drop it, and 'absent' and
        'scored near zero' are different answers to a dashboard.
        """
        projection = projection_from_rows([], isolated_nodes=["lonely"])
        assert projection.nodes == ["lonely"]
        assert pagerank(projection).scores["lonely"] == pytest.approx(1.0)

    def test_edge_count_halves_the_stored_pairs(self) -> None:
        assert _path("a", "b", "c").edge_count == 2


class TestProjectionFromRows:
    def test_weight_combines_confidence_and_evidence(self) -> None:
        """Neither alone is right: confidence alone ignores corroboration,
        raw evidence alone lets one prolific source dominate."""
        rows = [
            {"source_id": "a", "target_id": "b", "confidence": 0.9, "evidence_count": 1},
            {"source_id": "a", "target_id": "c", "confidence": 0.9, "evidence_count": 40},
        ]
        projection = projection_from_rows(rows)
        assert projection.adjacency["a"]["c"] > projection.adjacency["a"]["b"]

    def test_evidence_is_sublinear(self) -> None:
        """The same *absolute* increase counts for less at a higher base.

        Ten extra mentions on top of one is a real change in how well-evidenced a
        link is; ten on top of four hundred is noise. Compared at equal absolute
        deltas rather than equal ratios -- log gives every tenfold jump the same
        step, so a ratio comparison tests nothing about sublinearity.
        """
        def weight(count: int) -> float:
            rows = [
                {
                    "source_id": "a",
                    "target_id": "b",
                    "confidence": 1.0,
                    "evidence_count": count,
                }
            ]
            return projection_from_rows(rows).adjacency["a"]["b"]

        low_base_step = weight(11) - weight(1)
        high_base_step = weight(410) - weight(400)
        assert high_base_step < low_base_step

    def test_missing_confidence_defaults_below_certainty(self) -> None:
        """Rule-extracted edges carry no confidence; defaulting them to 1.0 would
        rank regex output above an LLM that honestly reported uncertainty."""
        rows = [{"source_id": "a", "target_id": "b", "evidence_count": 1}]
        weight = projection_from_rows(rows).adjacency["a"]["b"]
        certain = projection_from_rows(
            [{"source_id": "a", "target_id": "b", "confidence": 1.0, "evidence_count": 1}]
        ).adjacency["a"]["b"]
        assert weight < certain

    def test_zero_evidence_edge_is_not_deleted(self) -> None:
        """log1p(0) is 0. Floored, because it is still a link somebody extracted."""
        rows = [{"source_id": "a", "target_id": "b", "confidence": 0.8, "evidence_count": 0}]
        assert projection_from_rows(rows).adjacency["a"]["b"] > 0.0

    def test_non_string_endpoints_are_skipped(self) -> None:
        rows = [{"source_id": None, "target_id": "b"}, {"source_id": "a", "target_id": "b"}]
        assert projection_from_rows(rows).edge_count == 1


class TestPageRank:
    def test_scores_sum_to_one(self) -> None:
        result = pagerank(_two_cliques())
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_dangling_mass_is_not_lost(self) -> None:
        """The failure this test exists for: an isolated node has nowhere to send
        its rank, and dropping that mass shrinks every score a little more on
        every iteration. The vector still looks plausible and is uniformly wrong.
        """
        projection = _path("a", "b", "c")
        projection.nodes.append("island")
        projection.adjacency.setdefault("island", {})
        result = pagerank(projection)
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-9)
        assert result.scores["island"] > 0.0

    def test_hub_outranks_spokes_in_a_star(self) -> None:
        projection = Projection()
        for spoke in ("s1", "s2", "s3", "s4", "s5"):
            projection.add_edge("hub", spoke, 1.0)
        scores = pagerank(projection).scores
        assert scores["hub"] > max(scores[s] for s in ("s1", "s2", "s3", "s4", "s5"))

    def test_symmetric_graph_gives_symmetric_scores(self) -> None:
        """Both cliques are structurally identical, so their members must score
        identically -- a directional bug shows up here and nowhere else."""
        scores = pagerank(_two_cliques()).scores
        assert scores["a1"] == pytest.approx(scores["b1"], abs=1e-9)
        assert scores["a0"] == pytest.approx(scores["b0"], abs=1e-9)

    def test_bridge_nodes_outrank_their_own_clique(self) -> None:
        scores = pagerank(_two_cliques()).scores
        assert scores["a0"] > scores["a1"]

    def test_converges_on_a_small_graph(self) -> None:
        result = pagerank(_two_cliques())
        assert result.converged
        assert result.iterations < 100

    def test_personalization_concentrates_rank(self) -> None:
        projection = _two_cliques()
        seeded = pagerank(projection, personalization={"a0": 1.0}).scores
        uniform = pagerank(projection).scores
        assert seeded["a0"] > uniform["a0"]
        assert seeded["b3"] < uniform["b3"]
        assert sum(seeded.values()) == pytest.approx(1.0, abs=1e-9)

    def test_personalization_off_the_graph_is_rejected(self) -> None:
        """Silently ignoring it would return an unpersonalized ranking that the
        caller believes is seeded."""
        with pytest.raises(ValueError, match="nowhere to teleport"):
            pagerank(_two_cliques(), personalization={"not_a_node": 1.0})

    def test_damping_outside_the_open_unit_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="damping"):
            pagerank(_two_cliques(), damping=1.0)

    def test_empty_projection_is_not_an_error(self) -> None:
        result = pagerank(Projection())
        assert result.scores == {}
        assert result.converged

    def test_is_deterministic(self) -> None:
        """Float addition is not associative. An unordered sum reorders the top
        of the ranking between runs, and a dashboard that reshuffles nightly with
        no underlying change is indistinguishable from a real signal."""
        first = pagerank(_two_cliques()).scores
        second = pagerank(_two_cliques()).scores
        assert first == second

    def test_not_converged_is_reported(self) -> None:
        """A capped-out run still writes a clean-looking number to
        `pagerank_score`; the flag is the only way to know."""
        result = pagerank(_two_cliques(), max_iterations=1, tolerance=1e-12)
        assert not result.converged


class TestDegreeCentrality:
    def test_hub_scores_highest(self) -> None:
        projection = Projection()
        for spoke in ("s1", "s2", "s3"):
            projection.add_edge("hub", spoke, 1.0)
        scores = degree_centrality(projection).scores
        assert scores["hub"] > scores["s1"]

    def test_single_node_graph_is_zero_not_a_division_error(self) -> None:
        projection = Projection(nodes=["only"])
        assert degree_centrality(projection).scores == {"only": 0.0}


class TestBetweenness:
    def test_middle_of_a_path_is_the_only_bridge(self) -> None:
        """A---B---C: every shortest path between A and C runs through B, and
        nothing runs through A or C."""
        scores = betweenness_centrality(_path("a", "b", "c"), normalized=False).scores
        assert scores["b"] == pytest.approx(1.0)
        assert scores["a"] == pytest.approx(0.0)
        assert scores["c"] == pytest.approx(0.0)

    def test_bridge_endpoints_dominate_two_cliques(self) -> None:
        """The point of betweenness: a0 and b0 have unremarkable PageRank and
        carry every path between the two clusters."""
        scores = betweenness_centrality(_two_cliques()).scores
        assert scores["a0"] > scores["a1"] * 5
        assert scores["b0"] > scores["b1"] * 5

    def test_clique_has_no_bridges(self) -> None:
        projection = Projection()
        for left, right in _clique("c", 5):
            projection.add_edge(left, right, 1.0)
        assert max(betweenness_centrality(projection).scores.values()) == pytest.approx(0.0)

    def test_tiny_graphs_are_zero(self) -> None:
        assert betweenness_centrality(_path("a", "b")).scores == {"a": 0.0, "b": 0.0}

    def test_sampling_preserves_the_ranking(self) -> None:
        """The estimate need not match exactly; it must not reorder the answer."""
        projection = _two_cliques()
        full = betweenness_centrality(projection).scores
        sampled = betweenness_centrality(projection, sample_size=4).scores
        assert max(full, key=lambda k: full[k]) in {"a0", "b0"}
        assert max(sampled, key=lambda k: sampled[k]) in {"a0", "b0"}

    def test_sampling_reports_itself_as_approximate(self) -> None:
        result = betweenness_centrality(_two_cliques(), sample_size=4)
        assert not result.converged


class TestCommunityId:
    def test_is_content_addressed(self) -> None:
        """A sequential id makes every recomputation look like every community
        changed, and a summary cache keyed by id attaches to the wrong cluster."""
        assert community_id_for({"a", "b"}) == community_id_for({"b", "a"})

    def test_different_membership_gives_a_different_id(self) -> None:
        assert community_id_for({"a", "b"}) != community_id_for({"a", "c"})

    def test_empty_membership_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            community_id_for(set())


class TestModularity:
    def test_correct_partition_scores_well(self) -> None:
        projection = _two_cliques()
        good = {node: (0 if node.startswith("a") else 1) for node in projection.nodes}
        assert modularity(projection, good) > 0.3

    def test_everything_in_one_community_scores_zero(self) -> None:
        """The definition's anchor: a single community has no more internal edge
        weight than a random graph with the same degrees."""
        projection = _two_cliques()
        lumped = dict.fromkeys(projection.nodes, 0)
        assert modularity(projection, lumped) == pytest.approx(0.0, abs=1e-9)

    def test_random_partition_scores_worse_than_the_real_one(self) -> None:
        projection = _two_cliques()
        good = {node: (0 if node.startswith("a") else 1) for node in projection.nodes}
        interleaved = {node: index % 2 for index, node in enumerate(projection.nodes)}
        assert modularity(projection, good) > modularity(projection, interleaved)


class TestLouvain:
    def test_finds_the_two_cliques(self) -> None:
        result = louvain(_two_cliques())
        assert len(result.communities) == 2
        groups = {frozenset(community.members) for community in result.communities}
        assert frozenset(f"a{i}" for i in range(5)) in groups
        assert frozenset(f"b{i}" for i in range(5)) in groups

    def test_reports_real_modularity(self) -> None:
        assert louvain(_two_cliques()).modularity > 0.3

    def test_is_deterministic(self) -> None:
        """Louvain's quality depends on visit order and the reference version
        randomises it. Nightly reshuffling invites a reader to find meaning in
        churn."""
        first = louvain(_two_cliques()).assignment
        second = louvain(_two_cliques()).assignment
        assert first == second

    def test_disconnected_cliques_are_still_separated(self) -> None:
        result = louvain(_two_cliques(bridge=False))
        assert len(result.communities) == 2

    def test_small_groups_are_unassigned_not_communities(self) -> None:
        """A two-node 'community' produces a summary that restates the edge with
        a confident framing around it."""
        projection = _two_cliques()
        projection.add_edge("x", "y", 1.0)
        result = louvain(projection)
        assert "x" in result.unassigned
        assert "y" in result.unassigned
        assert all(community.size >= 3 for community in result.communities)

    def test_edgeless_graph_yields_no_communities(self) -> None:
        result = louvain(Projection(nodes=["a", "b", "c"]))
        assert result.communities == ()
        assert set(result.unassigned) == {"a", "b", "c"}

    def test_empty_graph_is_not_an_error(self) -> None:
        assert louvain(Projection()).communities == ()

    def test_assignment_covers_exactly_the_community_members(self) -> None:
        result = louvain(_two_cliques())
        assert set(result.assignment) == {
            member for community in result.communities for member in community.members
        }

    def test_higher_resolution_does_not_reduce_community_count(self) -> None:
        projection = Projection()
        for prefix in ("a", "b", "c", "d"):
            for left, right in _clique(prefix, 4):
                projection.add_edge(left, right, 1.0)
        for left, right in (("a0", "b0"), ("b1", "c0"), ("c1", "d0"), ("d1", "a1")):
            projection.add_edge(left, right, 1.0)
        coarse = louvain(projection, resolution=0.5)
        fine = louvain(projection, resolution=2.0)
        assert len(fine.communities) >= len(coarse.communities)

    def test_conductance_is_low_for_a_real_cluster(self) -> None:
        """One edge out of a 5-clique: most of its weight stays inside."""
        result = louvain(_two_cliques())
        assert all(community.conductance < 0.2 for community in result.communities)

    def test_internal_weight_is_not_double_counted(self) -> None:
        """A 5-clique has 10 unit edges; counting from both ends gives 20."""
        result = louvain(_two_cliques(bridge=False))
        assert all(
            community.internal_weight == pytest.approx(10.0)
            for community in result.communities
        )


class TestLabelPropagation:
    def test_separates_disconnected_clusters(self) -> None:
        """With no path between them, no label can flood across."""
        assert len(label_propagation(_two_cliques(bridge=False)).communities) == 2

    def test_a_single_bridge_is_enough_to_collapse_it(self) -> None:
        """The documented failure mode, pinned as a fact rather than a warning.

        One edge between two 5-cliques, and label propagation merges them into a
        single community with modularity 0.0 -- a "cluster" containing the entire
        graph, which reads as a confident answer and says nothing. This is why
        `louvain` is the default, and the assertion exists so that claim in the
        module docstring is checked rather than asserted.
        """
        collapsed = label_propagation(_two_cliques())
        assert len(collapsed.communities) == 1
        assert collapsed.modularity == pytest.approx(0.0, abs=1e-9)

        # Louvain, on the identical graph, recovers both.
        assert len(louvain(_two_cliques()).communities) == 2

    def test_is_deterministic(self) -> None:
        assert (
            label_propagation(_two_cliques()).assignment
            == label_propagation(_two_cliques()).assignment
        )

    def test_louvain_is_at_least_as_good_on_a_hub_graph(self) -> None:
        """The documented failure mode, made observable.

        A hub connected to everything is what an entity graph built from mentions
        always has -- some Topic that every signal touches. Label propagation
        tends to let one label flood across it.
        """
        projection = _two_cliques()
        for node in list(projection.nodes):
            projection.add_edge("hub", node, 1.0)
        assert louvain(projection).modularity >= label_propagation(projection).modularity


class TestCentralityResult:
    def test_top_breaks_ties_by_id(self) -> None:
        result = CentralityResult({"b": 1.0, "a": 1.0, "c": 0.5}, 1, True, 3, 0)
        assert [node for node, _ in result.top(2)] == ["a", "b"]
