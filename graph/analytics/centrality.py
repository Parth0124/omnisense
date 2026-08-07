"""Entity importance: PageRank, degree, and sampled betweenness.

Neo4j Community Edition ships without GDS, so there is no
`gds.pageRank.write()` to call. `docs/knowledge-graph.md` §10 leaves the
implementation open between APOC, an in-process pass, and adding GDS; this is the
in-process pass, and it is chosen for a reason that outlives the licence
question: **the algorithms are pure functions over a projection**, so they are
unit-testable against hand-built graphs with known answers, on a laptop, with no
database. An APOC or GDS call is testable only against a running server with the
plugin installed, which in practice means it is tested once by hand and never
again.

The projection is loaded, scored in memory, and written back as node properties
(`pagerank_score`, `community_id`, `computed_at`) so that the request path reads
a number instead of running an algorithm. That is the whole architectural point:
`retrieval/rerank/` wants entity importance as a tie-break, and a tie-break that
costs a graph traversal is not a tie-break, it is the query.

**Scale, honestly stated.** Loading the edge list into Python costs roughly 150
bytes per edge, so ten million edges is about 1.5 GB and this approach stops
being appropriate somewhere below that. `graph/queries/cypher.stale_analytics_nodes`
exists for the incremental path that comes next. Below a million edges -- which
is where this system is and will be for a while -- a full recompute takes
seconds and is simpler than anything incremental.

**Determinism.** Every iteration order here is sorted, and every accumulation is
over a sorted sequence. Float addition is not associative, so an unordered sum
over a `set` produces results that differ in the last bits between runs, and two
entities whose scores differ in the last bits swap places in a ranking. A
dashboard whose top-ten reorders on every batch with no underlying change is
indistinguishable from a real signal, and someone will try to explain it.

Layer note: **L1 library** -- `models/` plus the rest of `graph/`. No numpy: the
dependency is available, but a dense matrix over a sparse graph is the wrong
shape, and a dict-of-lists power iteration is both faster here and readable
without knowing numpy's broadcasting rules.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import structlog

__all__ = [
    "DEFAULT_DAMPING",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_TOLERANCE",
    "CentralityResult",
    "Projection",
    "betweenness_centrality",
    "degree_centrality",
    "pagerank",
    "projection_from_rows",
]

_log = structlog.get_logger(__name__)

DEFAULT_DAMPING: Final[float] = 0.85
"""Probability the random surfer follows an edge rather than teleporting.

0.85 is the original PageRank value and it is kept for a practical reason rather
than a traditional one: the number of iterations to convergence scales roughly
as `log(tolerance) / log(damping)`, so 0.95 roughly triples the work for a
ranking that is more sensitive to the graph's long tail -- which, in a graph
built from scraped mentions, is mostly extraction noise.
"""

DEFAULT_MAX_ITERATIONS: Final[int] = 100
DEFAULT_TOLERANCE: Final[float] = 1.0e-6
"""L1 convergence threshold, summed over all nodes rather than per node.

Per-node would let a large graph converge while a hundred thousand nodes each
still move by just under the threshold.
"""

_BETWEENNESS_SAMPLE_THRESHOLD: Final[int] = 2_000
"""Above this node count, betweenness is estimated from a sample of sources.

Brandes' algorithm is O(V·E). At ten thousand nodes and fifty thousand edges
that is five hundred million operations per full pass -- minutes in Python, for a
number used as a tie-break. Sampling sources gives an unbiased estimate at a
fraction of the cost, and the estimate is scaled back up so the two paths return
comparable magnitudes.
"""


# --------------------------------------------------------------------------- #
# The projection
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Projection:
    """An in-memory, undirected, weighted view of a slice of the graph.

    **Undirected on purpose.** Importance is about connectedness, not about which
    way an extractor happened to write an edge. `LAUNCHED_BY` points from the
    product to the launcher and `ACQUIRED` from the acquirer to the target;
    scoring those directions literally would rank a serial acquirer as
    *unimportant*, because acquisition edges all point away from it. Every
    algorithm here symmetrises.

    **Weighted, and the weight matters.** Two companies with one inferred
    co-occurrence and two companies with forty corroborated mentions are not
    equally connected. `evidence_count` and `confidence` fold into a single edge
    weight in `projection_from_rows`.

    `nodes` is held separately from the adjacency because an isolated entity --
    one real enough to have been extracted but not yet linked -- must still
    receive a score. Deriving the node set from the edge list would silently drop
    it, and "has no PageRank" and "has PageRank 0.0001" are different answers to
    a dashboard.
    """

    nodes: list[str] = field(default_factory=list)
    adjacency: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sorted once, here, so every traversal below inherits a deterministic
        # order without having to re-sort. See the module docstring.
        self.nodes = sorted(set(self.nodes) | set(self.adjacency))
        for node in self.nodes:
            self.adjacency.setdefault(node, {})

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Distinct undirected edges. Each is stored twice, hence the halving."""
        return sum(len(neighbours) for neighbours in self.adjacency.values()) // 2

    @property
    def total_weight(self) -> float:
        """`2m` in modularity notation -- the sum of every stored half-edge."""
        return sum(
            weight
            for neighbours in self.adjacency.values()
            for weight in neighbours.values()
        )

    def neighbours(self, node: str) -> list[tuple[str, float]]:
        """Neighbours in sorted order, so traversal is reproducible."""
        return sorted(self.adjacency.get(node, {}).items())

    def weighted_degree(self, node: str) -> float:
        return sum(self.adjacency.get(node, {}).values())

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        """Add an undirected weighted edge, accumulating parallel edges.

        Accumulation rather than replacement: two entities connected by both a
        `COMPETES_WITH` and a `MENTIONS` path are more connected than two joined
        by one, and taking the last weight written would make the result depend
        on row order.

        Self-loops are dropped. A node that resolution merged with itself, or a
        `MENTIONS` edge from a signal to an entity that resolved *into* that
        signal's own subject, produces one -- and a self-loop inflates a node's
        degree without connecting it to anything, which is precisely the
        opposite of what a centrality score should record.
        """
        if source == target:
            return
        if weight <= 0:
            return
        self.adjacency.setdefault(source, {})
        self.adjacency.setdefault(target, {})
        self.adjacency[source][target] = self.adjacency[source].get(target, 0.0) + weight
        self.adjacency[target][source] = self.adjacency[target].get(source, 0.0) + weight
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)


def projection_from_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    source_key: str = "source_id",
    target_key: str = "target_id",
    confidence_key: str = "confidence",
    evidence_key: str = "evidence_count",
    isolated_nodes: Sequence[str] = (),
) -> Projection:
    """Build a `Projection` from edge rows returned by `graph/client.py`.

    Edge weight is `confidence * log1p(evidence_count)`, and both halves earn
    their place:

    * `confidence` alone would treat one high-confidence extraction as equal to
      forty corroborating ones.
    * `evidence_count` alone would let a single prolific source dominate --
      forty mentions from one scraped aggregator outrank four from four
      independent outlets, which is exactly backwards.
    * `log1p` rather than the raw count because the difference between one and
      ten pieces of evidence is real and the difference between four hundred and
      four hundred and ten is not. Linear weighting makes the top of the graph a
      popularity contest between whichever entities the crawler saw most.

    A missing `confidence` defaults to 0.5 rather than to 1.0. Most rule-extracted
    edges carry no confidence, and defaulting them to certainty would rank
    regex output above LLM output that honestly reported its own uncertainty.
    """
    projection = Projection(nodes=list(isolated_nodes))
    for row in rows:
        source = row.get(source_key)
        target = row.get(target_key)
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        raw_confidence = row.get(confidence_key)
        confidence = (
            float(raw_confidence)
            if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool)
            else 0.5
        )
        raw_evidence = row.get(evidence_key)
        evidence = (
            int(raw_evidence)
            if isinstance(raw_evidence, int) and not isinstance(raw_evidence, bool)
            else 1
        )
        weight = max(confidence, 0.0) * math.log1p(max(evidence, 0))
        # `log1p(0) == 0`, so an edge claiming zero evidence would vanish
        # entirely. It is still a link somebody extracted; floor it at the weight
        # of a single observation rather than deleting it.
        if weight <= 0.0:
            weight = max(confidence, 0.0) * math.log1p(1)
        projection.add_edge(source, target, weight)
    return projection


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CentralityResult:
    """Scores plus the diagnostics needed to trust them.

    `converged` and `iterations` are not decoration. A PageRank that hit the
    iteration cap without converging is still a number, still writes cleanly to
    `pagerank_score`, and is still wrong -- and there is no way to tell from the
    scores themselves. Returning the flag is what lets the worker log it and lets
    a test assert on it.
    """

    scores: dict[str, float]
    iterations: int
    converged: bool
    node_count: int
    edge_count: int

    def top(self, n: int = 10) -> list[tuple[str, float]]:
        """The n highest-scoring nodes, ties broken by id for reproducibility."""
        return sorted(self.scores.items(), key=lambda item: (-item[1], item[0]))[:n]


# --------------------------------------------------------------------------- #
# PageRank
# --------------------------------------------------------------------------- #


def pagerank(
    projection: Projection,
    *,
    damping: float = DEFAULT_DAMPING,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    personalization: Mapping[str, float] | None = None,
) -> CentralityResult:
    """Weighted PageRank by power iteration.

    Two details that separate a correct implementation from one that merely
    produces plausible numbers:

    **Dangling mass is redistributed.** A node with no outgoing weight -- which
    on an undirected projection means a genuinely isolated entity -- has nowhere
    to send its rank. Dropping that mass means the score vector no longer sums to
    one, and it shrinks a little more on every iteration, so the algorithm
    "converges" to a vector of numbers that are all too small by an amount
    nobody can characterise. Collecting it and spreading it over the teleport
    distribution is what keeps the total invariant, and the sum is asserted
    below rather than assumed.

    **The teleport target is the personalization vector, not the uniform
    distribution.** With `personalization=None` they are the same thing. When a
    caller supplies one -- seeding the walk at the entities an investigation
    actually asked about -- the dangling mass has to follow it too, or a fraction
    of the rank leaks back to uniform on every iteration and quietly dilutes the
    personalization towards nothing.

    Returns scores summing to 1.0.
    """
    if not 0.0 < damping < 1.0:
        raise ValueError(f"damping must be in (0, 1), got {damping}")
    if projection.size == 0:
        return CentralityResult({}, 0, True, 0, 0)

    nodes = projection.nodes
    size = len(nodes)

    if personalization is None:
        teleport = {node: 1.0 / size for node in nodes}
    else:
        # Restricted to known nodes: a personalization entry for an id that is
        # not in the projection would silently absorb rank that then never
        # reaches any real node.
        filtered = {
            node: float(personalization[node])
            for node in nodes
            if node in personalization and personalization[node] > 0
        }
        total = sum(filtered.values())
        if total <= 0:
            raise ValueError(
                "personalization has no positive weight on any node in the "
                "projection; the walk would have nowhere to teleport to"
            )
        teleport = {node: filtered.get(node, 0.0) / total for node in nodes}

    scores = {node: 1.0 / size for node in nodes}
    outbound = {node: projection.weighted_degree(node) for node in nodes}

    iterations = 0
    converged = False
    for iterations in range(1, max_iterations + 1):
        incoming = {node: 0.0 for node in nodes}
        dangling_mass = 0.0

        for node in nodes:  # sorted -- see the module docstring
            degree = outbound[node]
            score = scores[node]
            if degree <= 0.0:
                dangling_mass += score
                continue
            share = score / degree
            for neighbour, weight in projection.neighbours(node):
                incoming[neighbour] += share * weight

        delta = 0.0
        for node in nodes:
            updated = (1.0 - damping) * teleport[node] + damping * (
                incoming[node] + dangling_mass * teleport[node]
            )
            delta += abs(updated - scores[node])
            incoming[node] = updated
        scores = incoming

        if delta < tolerance:
            converged = True
            break

    if not converged:
        _log.warning(
            "centrality.pagerank.not_converged",
            iterations=iterations,
            nodes=size,
            tolerance=tolerance,
        )

    return CentralityResult(
        scores=scores,
        iterations=iterations,
        converged=converged,
        node_count=size,
        edge_count=projection.edge_count,
    )


# --------------------------------------------------------------------------- #
# Degree
# --------------------------------------------------------------------------- #


def degree_centrality(projection: Projection, *, weighted: bool = True) -> CentralityResult:
    """Normalised degree -- the cheap importance signal, and a useful control.

    Worth having next to PageRank precisely because the two *disagree*: degree
    counts how many things an entity touches, PageRank counts how important the
    things it touches are. An entity with high degree and low PageRank is
    connected to a lot of unimportant nodes, which is the signature of an
    extraction artefact -- a common word resolved as a Topic, say, mentioned
    everywhere and meaning nothing. Having both makes that detectable.

    Normalised by `n - 1`, the maximum possible degree, so scores are comparable
    across graphs of different sizes.
    """
    if projection.size <= 1:
        return CentralityResult(
            {node: 0.0 for node in projection.nodes},
            1,
            True,
            projection.size,
            projection.edge_count,
        )
    denominator = float(projection.size - 1)
    scores = {
        node: (
            projection.weighted_degree(node) if weighted else float(len(projection.adjacency[node]))
        )
        / denominator
        for node in projection.nodes
    }
    return CentralityResult(scores, 1, True, projection.size, projection.edge_count)


# --------------------------------------------------------------------------- #
# Betweenness
# --------------------------------------------------------------------------- #


def betweenness_centrality(
    projection: Projection,
    *,
    sample_size: int | None = None,
    normalized: bool = True,
) -> CentralityResult:
    """Brandes' algorithm, over unweighted shortest paths, optionally sampled.

    Betweenness answers a question PageRank cannot: which entities are *bridges*.
    A company connecting two otherwise separate industry clusters has modest
    PageRank and high betweenness, and it is the more interesting entity of the
    two -- that is what a market-structure insight looks like.

    **Unweighted shortest paths**, even though the projection is weighted.
    Weighted betweenness needs Dijkstra where this needs BFS, roughly triples the
    constant factor, and answers a subtly different question: "the shortest path
    by accumulated inverse confidence" is not a route anything travels. Hop count
    is what "bridge" means here.

    **Sampling above `_BETWEENNESS_SAMPLE_THRESHOLD` nodes**, scaled back up by
    `V / sample` so the magnitude stays comparable to a full pass. The sample is
    a deterministic stride over the sorted node list rather than a random draw --
    reproducibility matters more than statistical purity for a tie-break, and a
    stride over sorted uuids is already effectively arbitrary with respect to
    graph structure.
    """
    nodes = projection.nodes
    size = len(nodes)
    if size < 3:
        # Betweenness is identically zero below three nodes: there is no vertex
        # that can sit between two others.
        return CentralityResult(
            {node: 0.0 for node in nodes}, 0, True, size, projection.edge_count
        )

    if sample_size is None:
        sample_size = size if size <= _BETWEENNESS_SAMPLE_THRESHOLD else _BETWEENNESS_SAMPLE_THRESHOLD
    sample_size = max(1, min(sample_size, size))

    if sample_size == size:
        sources = nodes
    else:
        stride = size / sample_size
        sources = [nodes[int(index * stride)] for index in range(sample_size)]

    betweenness = {node: 0.0 for node in nodes}

    for source in sources:
        stack: list[str] = []
        predecessors: dict[str, list[str]] = {node: [] for node in nodes}
        path_count = {node: 0.0 for node in nodes}
        distance = {node: -1 for node in nodes}
        path_count[source] = 1.0
        distance[source] = 0
        queue: deque[str] = deque([source])

        while queue:
            current = queue.popleft()
            stack.append(current)
            for neighbour, _weight in projection.neighbours(current):
                if distance[neighbour] < 0:
                    distance[neighbour] = distance[current] + 1
                    queue.append(neighbour)
                if distance[neighbour] == distance[current] + 1:
                    path_count[neighbour] += path_count[current]
                    predecessors[neighbour].append(current)

        dependency = {node: 0.0 for node in nodes}
        while stack:
            node = stack.pop()
            for predecessor in predecessors[node]:
                dependency[predecessor] += (
                    path_count[predecessor] / path_count[node]
                ) * (1.0 + dependency[node])
            if node != source:
                betweenness[node] += dependency[node]

    # Each undirected pair is counted from both ends.
    scale = 0.5
    if sample_size < size:
        scale *= size / sample_size
    if normalized:
        # Divide by the number of ordered pairs not involving the node itself.
        scale /= (size - 1) * (size - 2) / 2.0
    for node in betweenness:
        betweenness[node] *= scale

    return CentralityResult(
        scores=betweenness,
        iterations=len(sources),
        converged=sample_size == size,
        node_count=size,
        edge_count=projection.edge_count,
    )
