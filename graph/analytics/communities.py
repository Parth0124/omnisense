"""Community detection over the entity graph.

What this is for: `retrieval/graphrag/community.py` summarises clusters of
related entities so an investigation can be told "there is a cluster here about
battery supply chains" instead of being handed forty individual company nodes.
That summary is only worth reading if the clusters correspond to something --
which is a statement about the algorithm, not about the prompt that renders it.

**Louvain, not label propagation, as the default.** Both are implemented. Label
propagation is faster and near-linear, and it has a failure mode that matters
here: on a graph with a hub -- and an entity graph built from mentions always has
hubs, because some Topic is mentioned by everything -- it collapses most of the
graph into one giant community. A "cluster" containing sixty percent of the
entities tells a reader nothing, and it looks exactly like a correct result.
Louvain optimises modularity, which explicitly penalises that outcome: a
community is only worth forming if it has more internal edges than chance would
put there.

**Determinism.** Louvain's quality depends on the order nodes are visited, and
the reference implementation randomises it. Randomising here would mean the
dashboard's clusters reshuffle every night with no underlying change, and a
reader would look for meaning in the churn. Node order is sorted; ties in the
modularity gain go to the lower community id. The result is reproducible and
slightly worse than a randomised best-of-n, which is the correct trade for
something a human reads repeatedly.

**Resolution.** The `resolution` parameter is the knob that decides whether the
answer is "three industries" or "forty product niches". Neither is more correct;
`docs/knowledge-graph.md` §10 does not fix it, and it is exposed rather than
hidden because the right value depends on the graph and can only be found by
looking at output.

Layer note: **L1 library**. Imports `Projection` from `graph/analytics/centrality.py`
rather than defining a second one -- two projections that drift apart in how they
symmetrise or weight edges would make centrality and communities describe
different graphs while appearing to describe the same one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import structlog

from graph.analytics.centrality import Projection

__all__ = [
    "DEFAULT_MIN_COMMUNITY_SIZE",
    "DEFAULT_RESOLUTION",
    "Community",
    "CommunityResult",
    "community_id_for",
    "label_propagation",
    "louvain",
    "modularity",
]

_log = structlog.get_logger(__name__)

DEFAULT_RESOLUTION: Final[float] = 1.0
"""Modularity resolution. Above 1.0 yields more, smaller communities."""

DEFAULT_MIN_COMMUNITY_SIZE: Final[int] = 3
"""Below this, a "community" is not summarisable.

A two-node community is a pair of entities that happen to be linked, and asking
an LLM to write a thematic summary of it produces a restatement of the edge with
a confident framing around it -- the most expensive possible way to say nothing.
Singletons and pairs are returned as unassigned rather than as communities.
"""

_MAX_PASSES: Final[int] = 20
_MIN_MODULARITY_GAIN: Final[float] = 1.0e-7


def community_id_for(members: frozenset[str] | set[str]) -> str:
    """A stable id derived from a community's membership.

    Content-addressed rather than sequential, and that choice is what makes
    communities comparable across runs. A sequential id means last night's
    `community_3` and tonight's `community_3` are unrelated, so every
    recomputation looks like every community changed -- and a cached summary
    keyed by id would be silently attached to the wrong cluster.

    Hashing the membership means an unchanged cluster keeps its id, and a
    changed one gets a new one, which is exactly the semantics a cache needs.
    Sorted before hashing because a set has no order and `hash()` of a frozenset
    is not stable across processes.
    """
    if not members:
        raise ValueError("a community must have at least one member")
    digest = hashlib.sha256("\x1f".join(sorted(members)).encode("utf-8")).hexdigest()
    return f"com_{digest[:24]}"


@dataclass(frozen=True, slots=True)
class Community:
    """One detected cluster."""

    community_id: str
    members: tuple[str, ...]
    internal_weight: float
    """Edge weight wholly inside the community -- how tightly it holds together."""

    external_weight: float
    """Edge weight crossing the boundary -- how separable it actually is."""

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def conductance(self) -> float:
        """Fraction of the community's edge weight that leaves it. Lower is tighter.

        The honest quality number, and the one to look at before trusting a
        summary. A community with conductance 0.6 sends most of its edges
        elsewhere; calling it a cluster is a stretch, and any summary written
        about it will be vague in a way that reads as thoughtful.
        """
        total = 2.0 * self.internal_weight + self.external_weight
        return self.external_weight / total if total > 0 else 0.0


@dataclass(frozen=True, slots=True)
class CommunityResult:
    """Communities, the leftovers, and the modularity that justifies them."""

    communities: tuple[Community, ...]
    unassigned: tuple[str, ...]
    """Nodes in no community large enough to summarise. Not an error.

    Explicit rather than folded into a catch-all community, because "this entity
    stands alone" is a real and useful answer -- a newly extracted company with
    one mention *should* be unassigned, and burying it in a junk drawer labelled
    'other' invites a summary that pretends the junk drawer is a theme.
    """

    modularity: float
    passes: int

    @property
    def assignment(self) -> dict[str, str]:
        """`{entity_id: community_id}`, ready to write back as `community_id`."""
        return {
            member: community.community_id
            for community in self.communities
            for member in community.members
        }


def modularity(
    projection: Projection,
    assignment: Mapping[str, int | str],
    *,
    resolution: float = DEFAULT_RESOLUTION,
) -> float:
    """Newman-Girvan modularity of a partition. Range roughly (-0.5, 1].

    The number that decides whether a partition is worth anything: it compares
    the edge weight inside communities against what a random graph with the same
    degree sequence would have. Above ~0.3 is a real community structure; near
    zero means the partition found nothing, however confident the labels look.

    Computed here rather than trusted from the optimiser's internal bookkeeping,
    so a bug in the incremental gain calculation cannot report its own success.
    """
    total_weight = projection.total_weight
    if total_weight <= 0:
        return 0.0

    internal: dict[object, float] = {}
    degrees: dict[object, float] = {}
    for node in projection.nodes:
        community = assignment.get(node)
        if community is None:
            continue
        degrees[community] = degrees.get(community, 0.0) + projection.weighted_degree(node)
        for neighbour, weight in projection.neighbours(node):
            if assignment.get(neighbour) == community:
                internal[community] = internal.get(community, 0.0) + weight

    score = 0.0
    for community, degree in sorted(degrees.items(), key=lambda item: str(item[0])):
        inside = internal.get(community, 0.0)
        score += inside / total_weight - resolution * (degree / total_weight) ** 2
    return score


# --------------------------------------------------------------------------- #
# Louvain
# --------------------------------------------------------------------------- #


def louvain(
    projection: Projection,
    *,
    resolution: float = DEFAULT_RESOLUTION,
    min_community_size: int = DEFAULT_MIN_COMMUNITY_SIZE,
    max_passes: int = _MAX_PASSES,
) -> CommunityResult:
    """Modularity-optimising community detection, deterministically.

    Two phases, repeated until modularity stops improving:

    1. **Local moving.** Each node is moved to whichever neighbouring community
       most increases modularity, repeatedly, until nothing moves.
    2. **Aggregation.** Each community collapses into a single node, edges
       between communities become weighted edges between those nodes, and phase
       one runs again on the smaller graph.

    The aggregation step is what makes it fast and what makes it find structure
    at more than one scale -- without it the algorithm is stuck at whatever
    granularity the first pass happened to find.

    The gain formula uses `k_i_in - resolution * k_i * sum_tot / (2m)`, the
    standard incremental form. Note what is *not* in it: the node's own
    contribution to `sum_tot` is excluded by removing it from its community
    before evaluating any move. Leaving it in makes a node's own edges look like
    evidence for staying put, and the partition freezes on the first assignment.
    """
    if projection.size == 0:
        return CommunityResult((), (), 0.0, 0)

    total_weight = projection.total_weight
    if total_weight <= 0:
        # No edges: every node is its own community, none large enough to keep.
        return CommunityResult((), tuple(projection.nodes), 0.0, 0)

    # `placement` maps an *original* node id to the name of the node representing
    # it at the current level. At level 0 that is the id itself; after each
    # aggregation it becomes the string form of the community index. Keeping this
    # indirection explicit is what makes the fold-back correct -- an earlier
    # version indexed the level's assignment by the original node's *position*,
    # which happens to work only when no aggregation has occurred yet.
    placement: dict[str, str] = {node: node for node in projection.nodes}
    level = projection

    best_membership: dict[str, int] = {
        node: index for index, node in enumerate(projection.nodes)
    }
    best_score = modularity(projection, best_membership, resolution=resolution)
    passes = 0

    for pass_index in range(1, max_passes + 1):
        moved, community_of = _local_moving(level, resolution)
        placement = {node: str(community_of[name]) for node, name in placement.items()}

        # Score this level's partition against the *original* graph. Doing it
        # here rather than trusting the optimiser's incremental bookkeeping is
        # what makes the self-loop approximation in `_aggregate` safe: a pass
        # that made the partition worse is measured as worse and discarded,
        # instead of being accepted because the aggregated graph said it helped.
        candidate = _index_membership(placement, projection.nodes)
        score = modularity(projection, candidate, resolution=resolution)
        if score > best_score + _MIN_MODULARITY_GAIN:
            best_score = score
            best_membership = candidate
            passes = pass_index

        if not moved:
            break
        aggregated = _aggregate(level, community_of)
        if aggregated.size >= level.size:
            # Aggregation produced no reduction, so a further pass would repeat
            # this one exactly. Not an approximation -- it is the fixed point.
            break
        level = aggregated

    return _finalize(projection, best_membership, resolution, min_community_size, passes)


def _index_membership(
    placement: Mapping[str, str], nodes: Sequence[str]
) -> dict[str, int]:
    """Turn `{node: level_name}` into `{node: dense_community_index}`.

    Indices are assigned in sorted node order so the same partition always gets
    the same indices -- they are internal, but they reach `_finalize`'s grouping
    order and therefore the order communities are reported in.
    """
    index_of: dict[str, int] = {}
    return {
        node: index_of.setdefault(placement[node], len(index_of))
        for node in nodes
        if node in placement
    }


def _local_moving(level: Projection, resolution: float) -> tuple[bool, dict[str, int]]:
    """Phase one: move nodes between communities while modularity improves.

    Returns `(anything_moved, {node: community_index})`.
    """
    total_weight = level.total_weight
    if total_weight <= 0:
        return False, {node: index for index, node in enumerate(level.nodes)}

    community_of: dict[str, int] = {node: index for index, node in enumerate(level.nodes)}
    # `sum_tot[c]` is the total weighted degree of community c.
    sum_tot: dict[int, float] = {
        index: level.weighted_degree(node) for index, node in enumerate(level.nodes)
    }
    degree = {node: level.weighted_degree(node) for node in level.nodes}

    moved_ever = False
    for _ in range(_MAX_PASSES):
        moved_this_round = False
        for node in level.nodes:  # sorted -- determinism
            current = community_of[node]
            node_degree = degree[node]

            # Remove the node from its own community *before* evaluating moves.
            # Leaving it in lets its own edges argue for staying, and nothing
            # ever moves.
            sum_tot[current] -= node_degree

            # Weight from this node into each candidate community.
            weight_to: dict[int, float] = {}
            for neighbour, weight in level.neighbours(node):
                target = community_of[neighbour]
                weight_to[target] = weight_to.get(target, 0.0) + weight

            best_community = current
            best_gain = weight_to.get(current, 0.0) - resolution * node_degree * sum_tot.get(
                current, 0.0
            ) / total_weight

            for candidate, into in sorted(weight_to.items()):
                if candidate == current:
                    continue
                gain = into - resolution * node_degree * sum_tot.get(candidate, 0.0) / total_weight
                # Strict improvement, with an epsilon. Accepting an equal gain
                # makes two nodes swap communities forever, each move "improving"
                # by zero -- the loop only terminates on the pass cap, and the
                # result depends on where it happened to stop.
                if gain > best_gain + _MIN_MODULARITY_GAIN:
                    best_gain = gain
                    best_community = candidate

            sum_tot[best_community] = sum_tot.get(best_community, 0.0) + node_degree
            if best_community != current:
                community_of[node] = best_community
                moved_this_round = True
                moved_ever = True

        if not moved_this_round:
            break

    return moved_ever, community_of


def _aggregate(level: Projection, community_of: Mapping[str, int]) -> Projection:
    """Phase two: collapse each community into one node.

    Inter-community edges become weighted edges between the new nodes. Intra-
    community edges become self-loops, which `Projection.add_edge` drops -- and
    that is a real, deliberate approximation worth naming: the standard Louvain
    formulation keeps self-loops so that `2m` is preserved across levels. Dropping
    them shrinks the total weight at each level, which makes the resolution term
    slightly more aggressive on aggregated graphs than on the original.

    The alternative is a projection type that carries self-loops through every
    traversal in `centrality.py` -- where a self-loop is unambiguously wrong,
    inflating degree without connecting anything. Given that the final modularity
    is recomputed on the *original* graph by `_finalize`, and that is the number
    reported and acted on, the bias affects how quickly the search converges
    rather than what it converges to.
    """
    aggregated = Projection()
    # Every community becomes a node even if it ends up isolated, so the node
    # count comparison in `louvain` is meaningful.
    for community in sorted({str(index) for index in community_of.values()}):
        if community not in aggregated.nodes:
            aggregated.nodes.append(community)
            aggregated.adjacency.setdefault(community, {})
    aggregated.nodes = sorted(aggregated.nodes)

    seen: set[tuple[str, str]] = set()
    for node in level.nodes:
        source = str(community_of[node])
        for neighbour, weight in level.neighbours(node):
            target = str(community_of[neighbour])
            if source == target:
                continue
            # The projection stores each undirected edge twice, so without this
            # guard every inter-community edge is added twice and the aggregated
            # weights are doubled.
            key = (node, neighbour) if node < neighbour else (neighbour, node)
            if key in seen:
                continue
            seen.add(key)
            aggregated.add_edge(source, target, weight)
    return aggregated


def _finalize(
    projection: Projection,
    membership: Mapping[str, int],
    resolution: float,
    min_community_size: int,
    passes: int,
) -> CommunityResult:
    """Group by community, measure each one, drop the ones too small to summarise."""
    grouped: dict[int, list[str]] = {}
    for node in projection.nodes:
        grouped.setdefault(membership[node], []).append(node)

    communities: list[Community] = []
    unassigned: list[str] = []

    for _, members in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[1][0])):
        if len(members) < min_community_size:
            unassigned.extend(members)
            continue
        member_set = set(members)
        internal = 0.0
        external = 0.0
        for member in members:
            for neighbour, weight in projection.neighbours(member):
                if neighbour in member_set:
                    internal += weight
                else:
                    external += weight
        communities.append(
            Community(
                community_id=community_id_for(member_set),
                members=tuple(sorted(members)),
                # `internal` counted each inside edge from both ends.
                internal_weight=internal / 2.0,
                external_weight=external,
            )
        )

    # Modularity is measured on the partition actually returned -- communities
    # that were dropped for being too small are excluded, so the number
    # describes what the caller receives rather than an internal intermediate.
    kept = {
        member: community.community_id
        for community in communities
        for member in community.members
    }
    score = modularity(projection, kept, resolution=resolution) if kept else 0.0

    if communities and score < 0.05:
        _log.warning(
            "communities.weak_structure",
            modularity=round(score, 4),
            communities=len(communities),
            note="partition is barely better than random; summaries will be vague",
        )

    return CommunityResult(
        communities=tuple(communities),
        unassigned=tuple(sorted(unassigned)),
        modularity=score,
        passes=passes,
    )


# --------------------------------------------------------------------------- #
# Label propagation
# --------------------------------------------------------------------------- #


def label_propagation(
    projection: Projection,
    *,
    max_iterations: int = 30,
    min_community_size: int = DEFAULT_MIN_COMMUNITY_SIZE,
) -> CommunityResult:
    """Near-linear community detection. Fast, and prone to one giant community.

    Kept because it is the right tool when the graph is too large for Louvain and
    the answer only needs to be approximately right -- and because having both
    makes the hub-collapse failure mode observable rather than theoretical:
    run both and compare `modularity`, and a large gap says the graph has a hub
    that label propagation swallowed.

    Deterministic: nodes are visited in sorted order and label ties break towards
    the lexicographically smaller label, rather than the random tie-break the
    original formulation specifies.
    """
    if projection.size == 0:
        return CommunityResult((), (), 0.0, 0)

    labels: dict[str, str] = {node: node for node in projection.nodes}
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        changed = False
        for node in projection.nodes:
            neighbours = projection.neighbours(node)
            if not neighbours:
                continue
            weight_by_label: dict[str, float] = {}
            for neighbour, weight in neighbours:
                label = labels[neighbour]
                weight_by_label[label] = weight_by_label.get(label, 0.0) + weight
            # max by weight, then by lexicographically smallest label
            best = min(weight_by_label.items(), key=lambda item: (-item[1], item[0]))[0]
            if best != labels[node]:
                labels[node] = best
                changed = True
        if not changed:
            break

    index_of: dict[str, int] = {}
    membership: dict[str, int] = {}
    for node in projection.nodes:
        label = labels[node]
        membership[node] = index_of.setdefault(label, len(index_of))

    return _finalize(projection, membership, DEFAULT_RESOLUTION, min_community_size, iterations)
