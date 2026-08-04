"""Bounded neighbourhood traversal: from seed entities to citable signals.

This is the third `SearchBackend` (`retrieval/hybrid.py`). It answers a question
the other two cannot: not "which text is similar to the query" but "what is
structurally connected to what the caller already knows". A question about a
company's competitive exposure has an answer sitting in `COMPETES_WITH` and
`ACQUIRED` edges that no amount of BM25 or cosine similarity will surface,
because the connecting document never uses the query's words.

**Why every hop is capped, and why that is the whole design.**
The graph is scale-free. A major cloud vendor is `USES`-adjacent to tens of
thousands of products and `MENTIONS`-adjacent to millions of signals. An
uncapped two-hop expansion from that node is the Cartesian product of two
enormous adjacency lists: it does not return slowly, it does not return. Worse,
it fails *asymmetrically* -- the same query against a sparse entity comes back in
15 ms, so the cap is never exercised in development and the first hub entity a
user asks about takes the retrieval path down with it. So:

* fan-out is limited to `fanout_cap` edges **per node per hop**, applied inside
  Cypher as `ORDER BY confidence DESC LIMIT` in a per-row subquery, not by
  truncating in Python -- a Python-side cap still makes the server materialise
  the whole adjacency list;
* depth is clamped to `max_depth`, so the worst case is
  `|seeds| x fanout^depth` nodes and is knowable before the query runs;
* the frontier is additionally truncated to `max_nodes` by path score between
  hops, which bounds a request that arrives with a hundred seeds.

Ordering the fan-out by edge confidence is what makes the cap tolerable rather
than arbitrary: keeping 25 of 40,000 edges at random would be a lottery, and
keeping the 25 the extractor was most sure of is a defensible sample. It is still
a sample, and `Neighbourhood.truncated` records that it happened so the caller
can say so.

**Scoring.** `docs/retrieval.md` §6:

    path_score(c) = SUM over paths of PRODUCT over edges (confidence * 0.6^hop)

Contributions are summed, so an entity reachable by three independent paths
outranks one reachable by a single strong path -- corroboration, which is the
property the graph has and the text backends do not. The exponential hop decay is
what stops a two-hop neighbour of a hub from outranking a direct one.

**Read-only, structurally.** Every query goes through `backend/db/neo4j.py`
`run_read()`, whose session is opened in `READ` access mode; the server rejects a
write issued inside it. That is what makes the "retrieval never mutates the
graph" rule in `docs/architecture.md` §6.2 enforceable rather than aspirational.

**Time.** Entity-to-entity edges are filtered by the as-of predicate from
`docs/knowledge-graph.md` §5, evaluated at the *end of the requested window*, so
an investigation scoped to Q1 sees the graph as it was believed to be in Q1 and
does not back-date an acquisition learned about in August. Signals are filtered
by the compiled metadata filter, which is the same object the other two backends
push down -- the only way three backends fused on `chunk_id` can be answering
over one corpus rather than three.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

from backend.core.logging import get_logger
from models.enums import EdgeType, EntityType
from retrieval.filters.metadata import CypherFilter, as_of_for, compile_cypher
from retrieval.types import (
    Backend,
    Candidate,
    Filter,
    RetrievalRequest,
    chunk_id_for,
)

__all__ = [
    "DEFAULT_EDGE_CONFIDENCE",
    "DEFAULT_SALIENCE",
    "HOP_DECAY",
    "TRAVERSABLE_EDGE_TYPES",
    "GraphReader",
    "GraphTraversalBackend",
    "Neighbour",
    "Neighbourhood",
    "default_reader",
]

_log = get_logger(__name__)


@runtime_checkable
class GraphReader(Protocol):
    """A read-only Cypher executor.

    Narrow on purpose: one method, parameters always separate from the query
    text. It exists so this module can be tested against an in-memory graph with
    no driver and no container, and so the production implementation is
    `backend/db/neo4j.run_read` -- a *read*-mode managed transaction -- rather
    than whatever session happened to be in scope.
    """

    async def __call__(
        self, query: str, parameters: Mapping[str, Any] | None = None
    ) -> Sequence[Mapping[str, Any]]:
        """Run one read query and return its records as dicts."""
        ...


def default_reader() -> GraphReader:
    """`backend.db.neo4j.run_read`, imported lazily.

    Lazily because importing `backend/db/neo4j.py` opens nothing but does pull in
    the Bolt driver, and `retrieval/` is imported by the evaluation harness and
    by tests that never touch a graph. Keeping the import inside the call means a
    keyword-plus-vector deployment does not carry it.
    """
    from backend.db.neo4j import run_read

    return run_read


TRAVERSABLE_EDGE_TYPES: Final[tuple[str, ...]] = (
    EdgeType.COMPETES_WITH.value,
    EdgeType.ACQUIRED.value,
    EdgeType.USES.value,
    EdgeType.COMPLAINS_ABOUT.value,
    EdgeType.LAUNCHED_BY.value,
)
"""Edge types walked during *entity* expansion.

`MENTIONS` is deliberately absent, and it is the one omission worth explaining.
It is not an entity-to-entity edge: it attaches a `(:Signal)` reference node to
an entity, and it is how the second phase collects documents. Walking it during
expansion would put hub Signals in the frontier, whose own expansion is
co-mention noise -- every entity named in the same article, ranked by nothing.
The signals are collected once, at the end, from the whole neighbourhood.

`COMPLAINS_ABOUT` *can* land on a `(:Signal)` (`docs/knowledge-graph.md` §3
allows an unresolvable complainer to be the signal itself), and those are picked
up directly rather than discarded -- see `_collect_direct_signals`.
"""

HOP_DECAY: Final[float] = 0.6
"""Per-hop multiplier from `docs/retrieval.md` §6. Unmeasured, like every default
in that table. What it encodes is that relevance falls off superlinearly with
distance: a competitor of a competitor is weak evidence, and without the decay a
hub two hops out would outscore the seed's own direct neighbours purely on the
number of paths reaching it."""

DEFAULT_EDGE_CONFIDENCE: Final[float] = 0.5
"""Score used for an edge that carries no `confidence`.

Not `0.0`. A missing confidence means "the extractor did not score this", not
"this is certainly false", and treating it as zero multiplies every path through
it to zero -- so a graph written by a rule-based extractor that never sets
confidence contributes *nothing*, silently, while `per_backend_counts` reports a
healthy backend that happened to find no candidates."""

DEFAULT_SALIENCE: Final[float] = 0.5
"""Weight for a `MENTIONS` edge with no `salience`, for the same reason."""

_DEFAULT_MAX_DEPTH: Final[int] = 2
_DEFAULT_FANOUT_CAP: Final[int] = 25
_DEFAULT_MAX_NODES: Final[int] = 1_000
_DEFAULT_SIGNALS_PER_ENTITY: Final[int] = 25

_PARAM_PREFIX: Final[str] = "trv_"
"""Namespace for this module's Cypher parameters.

Compiled filter parameters use `flt_` (`retrieval/filters/metadata.py`). Two
namespaces that cannot collide, because a collision here does not raise -- one
value simply wins and the query quietly means something else."""

_SIGNAL_LABEL: Final[str] = "Signal"


@dataclass(frozen=True, slots=True)
class Neighbour:
    """One entity discovered by the traversal, with how it was reached."""

    entity_id: str
    name: str
    label: str
    hop: int
    path_score: float
    via_edge: str = ""

    @property
    def is_signal(self) -> bool:
        """Whether this "entity" is actually a `(:Signal)` reference node."""
        return self.label == _SIGNAL_LABEL


@dataclass(frozen=True, slots=True)
class Neighbourhood:
    """The bounded subgraph reached from the seeds.

    `truncated` is not decoration. A neighbourhood that hit a cap is a *sample*
    of the entity's surroundings, and a report built on it must not claim to have
    considered the whole competitive landscape. `docs/retrieval.md` §12 requires
    the omission to reach the caller rather than being absorbed here.
    """

    nodes: Mapping[str, Neighbour] = field(default_factory=dict)
    hops_run: int = 0
    truncated: bool = False
    seeds_found: int = 0

    def entities(self) -> list[Neighbour]:
        """Non-signal nodes, best-scoring first."""
        return sorted(
            (n for n in self.nodes.values() if not n.is_signal),
            key=lambda n: (-n.path_score, n.entity_id),
        )

    def signal_nodes(self) -> list[Neighbour]:
        """`(:Signal)` nodes reached directly, best-scoring first."""
        return sorted(
            (n for n in self.nodes.values() if n.is_signal),
            key=lambda n: (-n.path_score, n.entity_id),
        )


# One hop of the expansion. `CALL { WITH a ... }` is a *per-row* subquery: the
# ORDER BY/LIMIT inside it applies once per frontier node, which is what makes
# this a per-node fan-out cap rather than a global one. A global `LIMIT 25` after
# the match would spend the whole budget on the first hub node in the frontier
# and return nothing at all for the other seeds.
_EXPAND_CYPHER: Final[str] = """
MATCH (a{label_expr})
WHERE a.id IN $trv_frontier AND a.tenant_id = $trv_tenant
CALL {{
    WITH a
    MATCH (a)-[r]-(b)
    WHERE type(r) IN $trv_edge_types
      AND b.tenant_id = $trv_tenant
      AND r.valid_from <= $trv_as_of
      AND (r.valid_to IS NULL OR r.valid_to > $trv_as_of)
      AND coalesce(r.confidence, $trv_default_confidence) >= $trv_min_confidence
    RETURN b AS nb, r AS rel
    ORDER BY coalesce(rel.confidence, $trv_default_confidence) DESC, nb.id ASC
    LIMIT $trv_fanout_cap
}}
RETURN a.id                              AS from_id,
       labels(a)[0]                      AS from_label,
       nb.id                             AS to_id,
       coalesce(nb.canonical_name, '')   AS to_name,
       labels(nb)[0]                     AS to_label,
       type(rel)                         AS edge_type,
       coalesce(rel.confidence, $trv_default_confidence) AS confidence
"""
# `from_label` is returned for the seeds' sake. A caller supplies entity *ids*,
# which do not carry a label, so hop 1 has to match unlabelled; learning the
# label here lets hop 2 and the signal query seek the per-label index instead of
# scanning, and lets a seed that is itself a `(:Signal)` take the direct path.

# Phase two: the neighbourhood back to citable documents. The filter clauses are
# the *compiled* ones -- the same object OpenSearch and Qdrant receive -- so the
# graph backend answers over the same slice of the corpus as the other two.
_SIGNALS_CYPHER: Final[str] = """
MATCH (e{label_expr})
WHERE e.id IN $trv_entities
CALL {{
    WITH e
    MATCH (s:Signal)-[m:MENTIONS]->(e)
    {where}
    RETURN s.id AS signal_id,
           coalesce(m.salience, $trv_default_salience) AS salience,
           coalesce(m.chunk_index, -1) AS chunk_index
    ORDER BY salience DESC, signal_id ASC
    LIMIT $trv_signals_per_entity
}}
RETURN e.id AS via_entity, signal_id, salience, chunk_index
"""

# Signals reached *as nodes* by a COMPLAINS_ABOUT edge. They skipped the MENTIONS
# hop, so they also skipped the compiled filter that hangs off it; re-checking
# them here is not belt-and-braces, it is the only place the date window gets
# applied to them at all.
#
# The id restriction is handed to `CypherFilter.where(extra=...)` rather than
# written as its own `WHERE` line above `{where}`. `WHERE` is a sub-clause of
# `MATCH`, not a statement: two of them under one `MATCH` is a parse error, not a
# conjunction. That is why `where()` takes `extra` at all.
_DIRECT_SIGNALS_CYPHER: Final[str] = """
MATCH (s:Signal)
{where}
RETURN s.id AS signal_id
"""


class GraphTraversalBackend:
    """Neighbourhood traversal as a `SearchBackend`. Satisfies that Protocol.

    Stateless apart from its configuration, so one instance per process serves
    concurrent requests. Every bound is an instance attribute rather than a
    module constant because they are the knobs `docs/retrieval.md` §3 says must
    be tuned against the evaluation harness, and a constant cannot be varied per
    deployment while that tuning happens.
    """

    backend: Backend = Backend.GRAPH

    def __init__(
        self,
        reader: GraphReader | None = None,
        *,
        max_depth: int = _DEFAULT_MAX_DEPTH,
        fanout_cap: int = _DEFAULT_FANOUT_CAP,
        max_nodes: int = _DEFAULT_MAX_NODES,
        signals_per_entity: int = _DEFAULT_SIGNALS_PER_ENTITY,
        min_edge_confidence: float = 0.0,
        edge_types: Sequence[str] = TRAVERSABLE_EDGE_TYPES,
        hop_decay: float = HOP_DECAY,
    ) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {max_depth}")
        if fanout_cap < 1:
            raise ValueError(f"fanout_cap must be at least 1, got {fanout_cap}")
        if max_nodes < 1:
            raise ValueError(f"max_nodes must be at least 1, got {max_nodes}")
        if signals_per_entity < 1:
            raise ValueError(f"signals_per_entity must be at least 1, got {signals_per_entity}")
        if not 0.0 < hop_decay <= 1.0:
            raise ValueError(f"hop_decay must be in (0, 1], got {hop_decay}")
        if not edge_types:
            raise ValueError("edge_types is empty; the traversal would reach nothing")
        self._reader = reader if reader is not None else default_reader()
        self._max_depth = max_depth
        self._fanout_cap = fanout_cap
        self._max_nodes = max_nodes
        self._signals_per_entity = signals_per_entity
        self._min_edge_confidence = min_edge_confidence
        self._edge_types = tuple(edge_types)
        self._hop_decay = hop_decay

    # -------------------------------------------------------- SearchBackend --

    async def search(self, request: RetrievalRequest, *, limit: int) -> Sequence[Candidate]:
        """Expand from the request's seeds and rank the signals that surface.

        Returns an empty list when the request carries no seeds, or when the
        seeds resolve to nothing. `docs/retrieval.md` §6 is explicit that this is
        a normal outcome and not an error: the run continues on two backends. It
        must stay distinguishable from a *failure*, which is why nothing here
        catches an exception -- `HybridRetriever` records a raising backend as
        failed and lowers the confidence it reports, and swallowing the error
        here would present a degraded run as a complete one.
        """
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        if not request.seed_entity_ids:
            return []

        # Resolved once and threaded through both phases. `as_of_for` falls back
        # to `utcnow()` for an open-ended window, so calling it twice would read
        # the edge graph and the mention graph at two different instants -- a
        # millisecond apart, which is invisible until an edge closes between them.
        as_of = as_of_for(request.filters)
        hood = await self.neighbourhood(
            request.seed_entity_ids,
            filters=request.filters,
            depth=request.graph_depth,
            fanout_cap=request.graph_fanout_cap,
            as_of=as_of,
        )
        if not hood.nodes:
            return []

        scores = await self._collect_signals(hood, request.filters, as_of=as_of)
        candidates = _rank(scores, limit)
        _log.debug(
            "graph.traversal.search",
            seeds=len(request.seed_entity_ids),
            seeds_found=hood.seeds_found,
            nodes=len(hood.nodes),
            hops=hood.hops_run,
            truncated=hood.truncated,
            candidates=len(candidates),
            tenant_id=request.filters.tenant_id,
        )
        return candidates

    # ------------------------------------------------------------ traversal --

    async def neighbourhood(
        self,
        seed_ids: Sequence[str],
        *,
        filters: Filter | None = None,
        depth: int | None = None,
        fanout_cap: int | None = None,
        as_of: datetime | None = None,
    ) -> Neighbourhood:
        """Breadth-first expansion from `seed_ids`, one query per hop.

        One query per hop rather than a variable-length pattern (`-[*1..2]-`)
        for three reasons that all bite in production: a per-hop query can apply
        a per-node `LIMIT`, which a variable-length match cannot; the edge
        confidences the path score needs are only available hop by hop; and the
        frontier can be truncated between hops, which is the difference between a
        bounded query and one whose cost the server has to discover.

        Each node is expanded exactly once, at the shortest hop that reaches it,
        but scores from *later* paths still accumulate onto it -- dropping them
        would erase the corroboration that is the entire reason to sum over paths.

        `as_of` defaults to the end of the filter's window. Passing it explicitly
        is how a caller reads one instant across several calls instead of a
        slightly different "now" per call.
        """
        effective = filters if filters is not None else Filter()
        depth = _clamp("graph_depth", depth, self._max_depth)
        fanout_cap = _clamp("graph_fanout_cap", fanout_cap, self._fanout_cap)

        seeds = _unique(seed_ids)
        if not seeds:
            return Neighbourhood()

        if as_of is None:
            as_of = as_of_for(effective)
        nodes: dict[str, Neighbour] = {
            seed: Neighbour(entity_id=seed, name="", label="", hop=0, path_score=1.0)
            for seed in seeds
        }
        frontier: list[str] = list(seeds)
        # Seed labels are unknown -- the caller supplies ids only -- so hop 1
        # cannot narrow the match to a label. From hop 2 on the labels are known
        # and the match can use the per-label uniqueness index instead of a scan.
        frontier_labels: set[str] = set()
        truncated = False
        hops_run = 0
        seeds_found = 0

        for hop in range(1, depth + 1):
            rows = await self._reader(
                _EXPAND_CYPHER.format(label_expr=_label_expression(frontier_labels)),
                {
                    f"{_PARAM_PREFIX}frontier": list(frontier),
                    f"{_PARAM_PREFIX}tenant": effective.tenant_id,
                    f"{_PARAM_PREFIX}edge_types": list(self._edge_types),
                    f"{_PARAM_PREFIX}as_of": as_of,
                    f"{_PARAM_PREFIX}fanout_cap": fanout_cap,
                    f"{_PARAM_PREFIX}min_confidence": self._min_edge_confidence,
                    f"{_PARAM_PREFIX}default_confidence": DEFAULT_EDGE_CONFIDENCE,
                },
            )
            hops_run = hop
            if hop == 1:
                seeds_found = len({_text(r.get("from_id")) for r in rows} & set(seeds))

            discovered, hit_cap = self._absorb(rows, nodes, hop, fanout_cap)
            truncated = truncated or hit_cap

            if not discovered:
                # Nothing new: a further hop would re-derive the same set at a
                # lower score. Stopping early is not an optimisation, it is what
                # keeps a small connected component from costing `depth` round
                # trips to learn the same thing twice.
                break
            frontier = [n.entity_id for n in discovered]
            # Every label, including a blank one. Narrowing the match to the
            # labels we *did* see would silently drop the frontier nodes whose
            # label came back empty -- `_label_expression` treats the blank as
            # unrecognised and falls back to an unlabelled match for all of them.
            frontier_labels = {n.label for n in discovered}

        if len(nodes) > self._max_nodes:
            kept = sorted(nodes.values(), key=lambda n: (-n.path_score, n.entity_id))
            nodes = {n.entity_id: n for n in kept[: self._max_nodes]}
            truncated = True
            _log.warning(
                "graph.traversal.node_cap",
                max_nodes=self._max_nodes,
                seeds=len(seeds),
                detail="neighbourhood truncated by path score; results are a sample",
            )

        return Neighbourhood(
            nodes=nodes, hops_run=hops_run, truncated=truncated, seeds_found=seeds_found
        )

    def _absorb(
        self,
        rows: Sequence[Mapping[str, Any]],
        nodes: dict[str, Neighbour],
        hop: int,
        fanout_cap: int,
    ) -> tuple[list[Neighbour], bool]:
        """Fold one hop's edges into the node table; return what was new.

        The contribution of an edge is the parent's accumulated score times the
        edge confidence times the hop decay -- the product form in
        `docs/retrieval.md` §6, computed incrementally so the traversal never has
        to enumerate paths, which is the operation that does not terminate on a
        hub.
        """
        decay = self._hop_decay**hop
        discovered: list[Neighbour] = []
        per_parent: dict[str, int] = {}

        for row in rows:
            from_id = _text(row.get("from_id"))
            to_id = _text(row.get("to_id"))
            if not from_id or not to_id:
                continue
            parent = nodes.get(from_id)
            if parent is None:
                # An edge from a node we did not ask about. Scoring it would mean
                # inventing a parent path score, so it is dropped and counted
                # nowhere -- it can only happen if the reader is not the query.
                continue
            per_parent[from_id] = per_parent.get(from_id, 0) + 1
            if not parent.label:
                # A seed, whose label the caller could not supply. Learning it
                # from the row it appears in is free and makes the next hop an
                # index seek instead of a scan.
                parent = _replace(parent, label=_text(row.get("from_label")))
                nodes[from_id] = parent

            confidence = _number(row.get("confidence"), DEFAULT_EDGE_CONFIDENCE)
            contribution = parent.path_score * confidence * decay
            existing = nodes.get(to_id)
            if existing is None:
                neighbour = Neighbour(
                    entity_id=to_id,
                    name=_text(row.get("to_name")),
                    label=_text(row.get("to_label")),
                    hop=hop,
                    path_score=contribution,
                    via_edge=_text(row.get("edge_type")),
                )
                nodes[to_id] = neighbour
                discovered.append(neighbour)
            else:
                # A second path to a node already seen. Its score grows; its hop
                # stays at the shortest distance and it is not re-expanded, so
                # the node budget cannot be spent twice on one entity.
                nodes[to_id] = _replace(existing, score=existing.path_score + contribution)

        # Exactly `fanout_cap` edges from one node means the LIMIT was reached and
        # there were probably more. Reporting it is the point: a hub whose 40,000
        # edges were sampled down to 25 must not read as an exhaustive answer.
        hit_cap = any(count >= fanout_cap for count in per_parent.values())
        return discovered, hit_cap

    # -------------------------------------------------------------- signals --

    async def _collect_signals(
        self, hood: Neighbourhood, filters: Filter, *, as_of: datetime
    ) -> dict[str, tuple[float, int]]:
        """Signal id -> (score, chunk index) for the whole neighbourhood.

        A signal mentioned by three entities in the neighbourhood accumulates
        three contributions, weighted by each entity's path score and by the
        mention's salience. That is the graph's version of "several independent
        routes lead here", and it is what makes a graph candidate worth fusing
        with a lexical one rather than merely appending to it.
        """
        compiled = compile_cypher(filters, alias="s")
        scores: dict[str, tuple[float, int]] = {}

        entities = hood.entities()
        if entities:
            by_id = {n.entity_id: n for n in entities}
            rows = await self._reader(
                _SIGNALS_CYPHER.format(
                    label_expr=_label_expression({n.label for n in entities}),
                    where=compiled.where(
                        extra=(f"(m.valid_to IS NULL OR m.valid_to > ${_PARAM_PREFIX}as_of)",)
                    ),
                ),
                compiled.merged_parameters(
                    **{
                        f"{_PARAM_PREFIX}entities": list(by_id),
                        f"{_PARAM_PREFIX}as_of": as_of,
                        f"{_PARAM_PREFIX}default_salience": DEFAULT_SALIENCE,
                        f"{_PARAM_PREFIX}signals_per_entity": self._signals_per_entity,
                    }
                ),
            )
            for row in rows:
                signal_id = _text(row.get("signal_id"))
                via = by_id.get(_text(row.get("via_entity")))
                if not signal_id or via is None:
                    continue
                salience = _number(row.get("salience"), DEFAULT_SALIENCE)
                _accumulate(scores, signal_id, via.path_score * salience, row.get("chunk_index"))

        for signal_id, score in await self._collect_direct_signals(hood, compiled):
            _accumulate(scores, signal_id, score, None)
        return scores

    async def _collect_direct_signals(
        self, hood: Neighbourhood, compiled: CypherFilter
    ) -> list[tuple[str, float]]:
        """Signals reached as nodes, re-checked against the compiled filter.

        `COMPLAINS_ABOUT` may start at a `(:Signal)` when the complainer cannot
        be resolved to a person -- which, per `docs/knowledge-graph.md` §3, is the
        common case for throwaway social accounts. Those signals are exactly the
        evidence a competitor question wants, so they are kept rather than
        dropped for having the wrong label; but they arrived without traversing
        the `MENTIONS` hop, so nothing has yet applied the date window to them.
        """
        candidates = hood.signal_nodes()
        if not candidates:
            return []
        by_id = {n.entity_id: n for n in candidates}
        rows = await self._reader(
            _DIRECT_SIGNALS_CYPHER.format(
                where=compiled.where(extra=(f"s.id IN ${_PARAM_PREFIX}signal_ids",))
            ),
            compiled.merged_parameters(**{f"{_PARAM_PREFIX}signal_ids": list(by_id)}),
        )
        found: list[tuple[str, float]] = []
        for row in rows:
            signal_id = _text(row.get("signal_id"))
            node = by_id.get(signal_id)
            if node is not None:
                found.append((signal_id, node.path_score))
        return found


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _rank(scores: Mapping[str, tuple[float, int]], limit: int) -> list[Candidate]:
    """Score table -> ranked candidates, best first.

    Ties break on `signal_id`. Not cosmetic: path scores are products of a small
    number of coarse confidences and therefore tie constantly, and an unstable
    order makes an nDCG movement in the evaluation harness impossible to
    attribute to anything.
    """
    ordered = sorted(scores.items(), key=lambda item: (-item[1][0], item[0]))
    return [
        Candidate(
            chunk_id=chunk_id_for(signal_id, chunk_index),
            backend=Backend.GRAPH,
            rank=position,
            raw_score=score,
            signal_id=signal_id,
        )
        for position, (signal_id, (score, chunk_index)) in enumerate(ordered[:limit], start=1)
    ]


def _accumulate(
    scores: dict[str, tuple[float, int]], signal_id: str, delta: float, chunk_index: Any
) -> None:
    """Add a contribution for a signal, keeping the best-known chunk index.

    The graph names *signals*; fusion joins on *chunks* (`retrieval/types.py`).
    Where `graph/ingest/writer.py` mirrors the mentioning chunk onto the
    `MENTIONS` edge the join is exact; where it does not, the candidate points at
    chunk 0. That is not a silent guess: chunk 0 is the whole document for social
    posts and reviews (never split, `docs/retrieval.md` §8) and the lede of an
    article. But on a long document the graph then votes for a chunk it did not
    choose, and that shows up as poor graph-only precision in the evaluation
    harness long before it is visible anywhere else.
    """
    index = _chunk_index(chunk_index)
    previous = scores.get(signal_id)
    if previous is None:
        scores[signal_id] = (delta, index)
        return
    score, kept = previous
    # A mirrored chunk index beats the chunk-0 fallback whichever order they
    # arrive in, so the result does not depend on row ordering from the server.
    scores[signal_id] = (score + delta, kept if index == 0 else index)


def _chunk_index(value: Any) -> int:
    """A mirrored chunk index, or 0 when the edge does not carry one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    index = int(value)
    return index if index >= 0 else 0


def _clamp(name: str, requested: int | None, ceiling: int) -> int:
    """The requested bound, never above the configured ceiling.

    A `RetrievalRequest` is built from user-facing parameters, so `graph_depth=6`
    is one API call away, and depth is the exponent in the traversal's cost.
    The ceiling belongs to the operator; a request may only ask for less.
    Clamping silently would hide that the answer is not the one asked for, hence
    the log line.
    """
    if requested is None:
        return ceiling
    if requested < 1:
        raise ValueError(f"{name} must be at least 1, got {requested}")
    if requested > ceiling:
        _log.warning(
            "graph.traversal.clamped", parameter=name, requested=requested, applied=ceiling
        )
        return ceiling
    return requested


def _label_expression(labels: set[str]) -> str:
    """`:Company|Product` for a match, or `""` when any label is unrecognised.

    Cypher has no parameter form for a label, so this is the one place a value
    reaches the query *text* -- and it is safe only because every label is
    checked against the closed `EntityType` vocabulary plus `Signal` first, and
    anything else falls back to no label at all. An unrecognised label means the
    graph holds something this code does not know about: an unlabelled match
    keeps the traversal correct at the cost of an index seek, where filtering the
    unknown label out would silently drop those nodes from every result.
    """
    if not labels or not labels <= _KNOWN_LABELS:
        return ""
    return ":" + "|".join(sorted(labels))


_KNOWN_LABELS: Final[frozenset[str]] = frozenset(
    {member.value for member in EntityType if member is not EntityType.UNKNOWN} | {_SIGNAL_LABEL}
)


def _unique(values: Sequence[str]) -> list[str]:
    """De-duplicated, order-preserving, blanks dropped."""
    seen: dict[str, None] = {}
    for value in values:
        text = _text(value)
        if text:
            seen.setdefault(text, None)
    return list(seen)


def _replace(
    neighbour: Neighbour, *, score: float | None = None, label: str | None = None
) -> Neighbour:
    """A modified copy. `Neighbour` is frozen, and it is shared by reference."""
    return Neighbour(
        entity_id=neighbour.entity_id,
        name=neighbour.name,
        label=neighbour.label if label is None else label,
        hop=neighbour.hop,
        path_score=neighbour.path_score if score is None else score,
        via_edge=neighbour.via_edge,
    )


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value: Any, fallback: float) -> float:
    """A float from a Cypher scalar, falling back rather than raising.

    `coalesce()` in the query already covers a missing property; this covers one
    written as a string by an older extractor, which would otherwise raise inside
    a fan-out branch that reports the whole backend as failed for the sake of one
    malformed row.
    """
    if isinstance(value, bool) or value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback
