"""Unit tests for `retrieval/graph_retrieval/`, against an in-memory graph.

The fake here is a small graph *store*, not a stub: it parses the parameters the
modules send, honours them, and refuses to answer a query that did not carry the
predicate it is being asked to apply. That last part is what makes pushdown
testable at all -- a stub returning a fixed list proves the code runs, not that
the tenant reached the server.

Three failures drive most of what is tested:

1. **An uncapped hub.** A major vendor has tens of thousands of edges. Depth 2
   from it is not slow, it is unbounded -- and the same query on a sparse entity
   returns in milliseconds, so the cap is never exercised until production.
   `test_fanout_cap_*` pin the cap as *per node per hop*, the only form of it
   that survives a multi-seed request.
2. **A filter that was compiled and then not sent.** The graph backend would
   answer over the whole corpus while the other two answered over the window,
   and fusion would read the difference as the graph finding more.
3. **A fact that quietly time-travels.** An edge learned in August, returned for
   a question about Q1, reads as though it had been known then.

No network, no driver, no container.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from neo4j.time import DateTime as Neo4jDateTime

from models.enums import EdgeType
from retrieval.graph_retrieval.expansion import (
    FACT_EDGE_TYPES,
    GraphQueryExpander,
    lucene_safe,
)
from retrieval.graph_retrieval.traversal import (
    DEFAULT_EDGE_CONFIDENCE,
    HOP_DECAY,
    TRAVERSABLE_EDGE_TYPES,
    GraphTraversalBackend,
)
from retrieval.hybrid import GraphExpander, SearchBackend
from retrieval.types import Backend, Filter, RetrievalRequest

pytestmark = pytest.mark.unit

TENANT = "tnt_main"
OTHER_TENANT = "tnt_other"
DAWN = datetime(2020, 1, 1, tzinfo=UTC)


def at(day: int, month: int = 1) -> datetime:
    """A UTC instant in 2026."""
    return datetime(2026, month, day, 12, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# The fake graph
# --------------------------------------------------------------------------- #

_WRITE_KEYWORDS = re.compile(r"\b(MERGE|CREATE|DELETE|DETACH|SET|REMOVE|DROP)\b")
_NODE_LABELS = re.compile(r"MATCH \([ase](:[A-Za-z|]+)?\)")

_CLAUSE = re.compile(
    r"\b(OPTIONAL MATCH|MATCH|WHERE|WITH|UNWIND|YIELD|RETURN|ORDER BY|LIMIT|SKIP|CALL)\b"
)
#: `WHERE` is a sub-clause, so it is only legal directly after a clause that
#: opens a scope for it to constrain.
_WHERE_MAY_FOLLOW = frozenset({"MATCH", "OPTIONAL MATCH", "WITH", "UNWIND", "YIELD", "CALL"})


def _scopes(query: str) -> list[str]:
    """The query split into brace-delimited scopes: outer text, then each block.

    `CALL { }` subqueries and `EXISTS { }` predicates each have their own clause
    sequence, so validating the raw text as one stream would let an inner `WHERE`
    excuse an illegal outer one. Map literals like `{limit: $n}` come out as
    scopes holding no clause keywords, which is harmless.
    """
    outer: list[str] = []
    inner: list[str] = []
    block: list[str] = []
    depth = 0
    for character in query:
        if character == "{":
            depth += 1
            if depth == 1:
                continue
        elif character == "}":
            depth -= 1
            assert depth >= 0, f"unbalanced braces in {query!r}"
            if depth == 0:
                inner.extend(_scopes("".join(block)))
                block = []
                outer.append(" ")
                continue
        (block if depth else outer).append(character)
    assert depth == 0, f"unbalanced braces in {query!r}"
    return ["".join(outer), *inner]


def assert_cypher_clause_order(query: str) -> None:
    """Reject a clause sequence the server would refuse to parse.

    Not a Cypher parser -- it checks the one rule these modules build queries by
    string composition and can therefore break silently. `WHERE` is a sub-clause
    of `MATCH`/`WITH`, not a statement, so a template that writes its own `WHERE`
    line above an interpolated `{where}` produces two in a row: a parse error at
    query time, invisible to a fake that dispatches on parameters. That is what
    `CypherFilter.where(extra=...)` exists to prevent, and this is what notices
    when a template stops using it.
    """
    for scope in _scopes(query):
        previous = ""
        for found in _CLAUSE.finditer(scope):
            clause = found.group(1)
            if clause == "WHERE" and previous not in _WHERE_MAY_FOLLOW:
                raise AssertionError(
                    f"`WHERE` follows `{previous or 'nothing'}` in {query!r}. WHERE is a "
                    "sub-clause, not a statement: two in a row is a parse error, not a "
                    "conjunction. Merge them with CypherFilter.where(extra=...)."
                )
            previous = clause


#: Compiled-filter parameter name -> (clause that must be present, predicate).
#: The fake honours a filter only if the query text actually carries its clause,
#: so a parameter passed but never referenced -- a filter compiled and then
#: forgotten -- fails here instead of widening the corpus in production.
_SIGNAL_PREDICATES: dict[str, tuple[str, Any]] = {
    "flt_tenant_id_equals": (
        "s.tenant_id = $flt_tenant_id_equals",
        lambda s, v: s["tenant_id"] == v,
    ),
    "flt_published_at_gte": (
        "s.published_at >= $flt_published_at_gte",
        lambda s, v: s["published_at"] >= v,
    ),
    "flt_published_at_lt": (
        "s.published_at < $flt_published_at_lt",
        lambda s, v: s["published_at"] < v,
    ),
    "flt_platform_any_of": ("s.platform IN $flt_platform_any_of", lambda s, v: s["platform"] in v),
    "flt_source_any_of": ("s.source IN $flt_source_any_of", lambda s, v: s["source"] in v),
    "flt_language_any_of": ("s.language IN $flt_language_any_of", lambda s, v: s["language"] in v),
    "flt_confidence_gte": (
        "s.confidence >= $flt_confidence_gte",
        lambda s, v: s["confidence"] >= v,
    ),
}


class FakeGraph:
    """An in-memory knowledge graph that answers this package's Cypher.

    It dispatches on the parameters a query carries rather than on the query
    text, so a rewrite of the Cypher that preserves the contract keeps passing --
    but it asserts on the text wherever the text *is* the contract: read-only
    access, the label expression, and the pushed-down filter clauses.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.signals: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.mentions: list[dict[str, Any]] = []
        self.queries: list[tuple[str, dict[str, Any]]] = []

    # -- construction ------------------------------------------------------ #

    def entity(
        self,
        entity_id: str,
        *,
        label: str = "Company",
        name: str = "",
        aliases: Sequence[str] = (),
        tenant: str = TENANT,
    ) -> str:
        self.nodes[entity_id] = {
            "id": entity_id,
            "label": label,
            "canonical_name": name or entity_id,
            "aliases": list(aliases),
            "tenant_id": tenant,
        }
        return entity_id

    def signal(
        self,
        signal_id: str,
        *,
        published: datetime | None = None,
        tenant: str = TENANT,
        platform: str = "reddit",
        source: str = "social",
        language: str = "en",
        confidence: float = 0.8,
    ) -> str:
        # A Signal is a node too: `COMPLAINS_ABOUT` may start at one.
        self.nodes[signal_id] = {
            "id": signal_id,
            "label": "Signal",
            "canonical_name": "",
            "aliases": [],
            "tenant_id": tenant,
        }
        self.signals[signal_id] = {
            "id": signal_id,
            "tenant_id": tenant,
            "published_at": published or at(15),
            "platform": platform,
            "source": source,
            "language": language,
            "confidence": confidence,
        }
        return signal_id

    def edge(
        self,
        edge_type: str,
        subject: str,
        obj: str,
        *,
        confidence: float | None = 0.9,
        valid_from: datetime = DAWN,
        valid_to: datetime | None = None,
        signals: Sequence[str] = (),
    ) -> None:
        self.edges.append(
            {
                "type": str(edge_type),
                "from": subject,
                "to": obj,
                "confidence": confidence,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "source_signal_ids": list(signals),
            }
        )

    def mention(
        self,
        signal_id: str,
        entity_id: str,
        *,
        salience: float | None = 0.7,
        chunk_index: int | None = None,
        valid_to: datetime | None = None,
    ) -> None:
        self.mentions.append(
            {
                "signal_id": signal_id,
                "entity_id": entity_id,
                "salience": salience,
                "chunk_index": chunk_index,
                "valid_to": valid_to,
            }
        )

    # -- the reader port --------------------------------------------------- #

    async def __call__(
        self, query: str, parameters: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = dict(parameters or {})
        self.queries.append((query, params))
        assert not _WRITE_KEYWORDS.search(query), (
            f"retrieval issued a mutating clause: {query!r}. The read session "
            "would reject it, but only in an environment with a server."
        )
        assert_cypher_clause_order(query)
        if "trv_frontier" in params:
            return self._expand(query, params)
        if "trv_entities" in params:
            return self._signals_for_entities(query, params)
        if "trv_signal_ids" in params:
            return self._direct_signals(query, params)
        if "exp_ids" in params:
            return self._entities_by_id(params)
        if "exp_query" in params:
            return self._fulltext(params)
        if "exp_signal_ids" in params:
            return self._facts(query, params)
        raise AssertionError(f"fake graph cannot answer: {query!r}")

    # -- traversal --------------------------------------------------------- #

    def _expand(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        assert "CALL {" in query and "LIMIT $trv_fanout_cap" in query, (
            "the fan-out cap must be a per-row LIMIT inside the subquery. A "
            "global LIMIT spends the whole budget on the first hub in the "
            "frontier and returns nothing for the other seeds."
        )
        allowed = _allowed_labels(query)
        tenant = params["trv_tenant"]
        as_of = params["trv_as_of"]
        edge_types = set(params["trv_edge_types"])
        fanout = params["trv_fanout_cap"]
        default_confidence = params["trv_default_confidence"]
        minimum = params["trv_min_confidence"]

        rows: list[dict[str, Any]] = []
        for node_id in params["trv_frontier"]:
            node = self.nodes.get(node_id)
            if node is None or node["tenant_id"] != tenant:
                continue
            assert allowed is None or node["label"] in allowed, (
                f"label expression {allowed} excluded frontier node {node_id!r}; "
                "the traversal would silently lose it"
            )
            matched: list[tuple[float, str, dict[str, Any]]] = []
            for edge in self.edges:
                if edge["type"] not in edge_types:
                    continue
                if node_id == edge["from"]:
                    other_id = edge["to"]
                elif node_id == edge["to"]:
                    other_id = edge["from"]
                else:
                    continue
                other = self.nodes.get(other_id)
                if other is None or other["tenant_id"] != tenant:
                    continue
                if edge["valid_from"] > as_of:
                    continue
                if edge["valid_to"] is not None and edge["valid_to"] <= as_of:
                    continue
                confidence = (
                    default_confidence if edge["confidence"] is None else edge["confidence"]
                )
                if confidence < minimum:
                    continue
                matched.append((confidence, other_id, edge))
            matched.sort(key=lambda item: (-item[0], item[1]))
            for confidence, other_id, edge in matched[:fanout]:
                other = self.nodes[other_id]
                rows.append(
                    {
                        "from_id": node_id,
                        "from_label": node["label"],
                        "to_id": other_id,
                        "to_name": other["canonical_name"],
                        "to_label": other["label"],
                        "edge_type": edge["type"],
                        "confidence": confidence,
                    }
                )
        return rows

    def _signals_for_entities(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        allowed = _allowed_labels(query)
        as_of = params["trv_as_of"]
        per_entity = params["trv_signals_per_entity"]
        default_salience = params["trv_default_salience"]
        assert "m.valid_to IS NULL OR m.valid_to > $trv_as_of" in query, (
            "a retracted mention would be returned as evidence"
        )
        assert "CALL {" in query and "LIMIT $trv_signals_per_entity" in query, (
            "one hub entity's mention list would otherwise fill the whole result"
        )

        rows: list[dict[str, Any]] = []
        for entity_id in params["trv_entities"]:
            node = self.nodes.get(entity_id)
            if node is None:
                continue
            assert allowed is None or node["label"] in allowed
            found: list[tuple[float, str, int]] = []
            for mention in self.mentions:
                if mention["entity_id"] != entity_id:
                    continue
                if mention["valid_to"] is not None and mention["valid_to"] <= as_of:
                    continue
                signal = self.signals.get(mention["signal_id"])
                if signal is None or not self._signal_passes(query, params, signal):
                    continue
                salience = default_salience if mention["salience"] is None else mention["salience"]
                index = -1 if mention["chunk_index"] is None else mention["chunk_index"]
                found.append((salience, mention["signal_id"], index))
            found.sort(key=lambda item: (-item[0], item[1]))
            rows.extend(
                {
                    "via_entity": entity_id,
                    "signal_id": signal_id,
                    "salience": salience,
                    "chunk_index": index,
                }
                for salience, signal_id, index in found[:per_entity]
            )
        return rows

    def _direct_signals(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"signal_id": signal_id}
            for signal_id in params["trv_signal_ids"]
            if signal_id in self.signals
            and self._signal_passes(query, params, self.signals[signal_id])
        ]

    def _signal_passes(
        self, query: str, params: Mapping[str, Any], signal: Mapping[str, Any]
    ) -> bool:
        """Apply exactly the filter clauses the query carried, and no others."""
        applied = 0
        for name, (clause, predicate) in _SIGNAL_PREDICATES.items():
            if name not in params:
                continue
            assert clause in query, (
                f"parameter {name} was sent but {clause!r} is not in the query; "
                "the filter was compiled and then not pushed down"
            )
            applied += 1
            if not predicate(signal, params[name]):
                return False
        assert applied, "no filter reached the query at all; tenant is mandatory"

        if "flt_entity_ids_contains_any" in params:
            wanted = set(params["flt_entity_ids_contains_any"])
            assert "MENTIONS" in query
            mentioned = {m["entity_id"] for m in self.mentions if m["signal_id"] == signal["id"]}
            if not mentioned & wanted:
                return False
        return True

    # -- expansion --------------------------------------------------------- #

    def _entities_by_id(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        tenant = params["exp_tenant"]
        # Returned in storage order, not request order: the caller must not be
        # relying on the server to preserve the seed ordering.
        return [
            _entity_row(node)
            for node in reversed(list(self.nodes.values()))
            if node["id"] in set(params["exp_ids"]) and node["tenant_id"] == tenant
        ]

    def _fulltext(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        tokens = [t.strip('"').casefold() for t in params["exp_query"].split(" OR ")]
        scored: list[tuple[float, dict[str, Any]]] = []
        for node in self.nodes.values():
            if node["tenant_id"] != params["exp_tenant"] or node["label"] == "Signal":
                continue
            haystack = " ".join([node["canonical_name"], *node["aliases"]]).casefold()
            hits = sum(1 for token in tokens if token and token in haystack)
            if hits and float(hits) >= params["exp_min_score"]:
                scored.append((float(hits), node))
        scored.sort(key=lambda item: (-item[0], item[1]["id"]))
        rows = [{**_entity_row(node), "score": score} for score, node in scored]
        return rows[: params["exp_keep"]]

    def _facts(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        tenant = params["exp_tenant"]
        as_of = params["exp_as_of"]
        anchor_signals = set(params["exp_signal_ids"])
        anchors = {
            m["entity_id"]
            for m in self.mentions
            if m["signal_id"] in anchor_signals
            and self.nodes.get(m["entity_id"], {}).get("tenant_id") == tenant
        }
        anchors |= {
            seed
            for seed in params["exp_seed_ids"]
            if self.nodes.get(seed, {}).get("tenant_id") == tenant
        }

        honour_valid_to = "r.valid_to IS NULL OR r.valid_to > $exp_as_of" in query
        rows: list[dict[str, Any]] = []
        for edge in self.edges:
            if edge["type"] not in set(params["exp_edge_types"]):
                continue
            if not ({edge["from"], edge["to"]} & anchors):
                continue
            other_ids = {edge["from"], edge["to"]}
            if any(self.nodes.get(i, {}).get("tenant_id") != tenant for i in other_ids):
                continue
            if edge["valid_from"] > as_of:
                continue
            if honour_valid_to and edge["valid_to"] is not None and edge["valid_to"] <= as_of:
                continue
            confidence = 0.0 if edge["confidence"] is None else edge["confidence"]
            if confidence < params["exp_min_confidence"]:
                continue
            subject, obj = self.nodes[edge["from"]], self.nodes[edge["to"]]
            rows.append(
                {
                    "predicate": edge["type"],
                    "subject_id": subject["id"],
                    "subject_name": subject["canonical_name"],
                    "object_id": obj["id"],
                    "object_name": obj["canonical_name"],
                    "valid_from": edge["valid_from"],
                    "valid_to": edge["valid_to"],
                    "confidence": confidence,
                    "supporting_signal_ids": edge["source_signal_ids"][: params["exp_supporting"]],
                }
            )
        rows.sort(key=lambda r: (-r["confidence"], r["predicate"], r["subject_id"]))
        return rows[: params["exp_limit"]]


def _entity_row(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": node["id"],
        "canonical_name": node["canonical_name"],
        "aliases": list(node["aliases"]),
        "label": node["label"],
    }


def _allowed_labels(query: str) -> set[str] | None:
    """The label expression the query narrowed its match to, if any."""
    found = _NODE_LABELS.search(query)
    if found is None or not found.group(1):
        return None
    return set(found.group(1)[1:].split("|"))


def request(
    *,
    seeds: Sequence[str] = (),
    query: str = "how exposed is acme",
    filters: Filter | None = None,
    depth: int = 2,
    fanout: int = 25,
) -> RetrievalRequest:
    return RetrievalRequest(
        query=query,
        filters=filters or Filter(tenant_id=TENANT),
        seed_entity_ids=tuple(seeds),
        graph_depth=depth,
        graph_fanout_cap=fanout,
    )


# --------------------------------------------------------------------------- #
# Traversal: the caps
# --------------------------------------------------------------------------- #


@pytest.fixture
def hub() -> FakeGraph:
    """One hub entity with 500 competitors, each mentioned by its own signal.

    The shape that breaks an uncapped traversal: fan-out from `acme` is two
    orders of magnitude past the cap, and every neighbour is itself expandable.
    """
    graph = FakeGraph()
    graph.entity("acme", name="Acme Corp")
    for index in range(500):
        rival = graph.entity(f"rival_{index:03d}", name=f"Rival {index}")
        graph.edge(EdgeType.COMPETES_WITH, "acme", rival, confidence=0.5 + index / 2000)
        signal = graph.signal(f"sig_{index:03d}", published=at(15))
        graph.mention(signal, rival)
    return graph


@pytest.mark.asyncio
async def test_fanout_cap_bounds_a_hub(hub: FakeGraph) -> None:
    """25 of 40,000 edges, not 40,000. This is the test the module exists for."""
    backend = GraphTraversalBackend(hub)
    hood = await backend.neighbourhood(["acme"], filters=Filter(tenant_id=TENANT))

    assert len(hood.nodes) == 1 + 25, "fan-out escaped the per-hop cap"
    assert hood.truncated, "a sampled neighbourhood must say that it was sampled"
    # The 25 kept are the highest-confidence edges, not an arbitrary 25.
    kept = {n.entity_id for n in hood.entities() if n.hop == 1}
    assert kept == {f"rival_{i:03d}" for i in range(475, 500)}


@pytest.mark.asyncio
async def test_fanout_cap_is_per_node_not_global(hub: FakeGraph) -> None:
    """Two seeds get 25 each.

    A global cap spends the whole budget on whichever hub the planner reached
    first and returns nothing for the other seed -- which looks like a sparse
    entity rather than an exhausted budget.
    """
    hub.entity("globex", name="Globex")
    for index in range(100):
        hub.edge(EdgeType.COMPETES_WITH, "globex", f"rival_{index:03d}", confidence=0.4)

    backend = GraphTraversalBackend(hub, max_depth=1)
    hood = await backend.neighbourhood(["acme", "globex"], filters=Filter(tenant_id=TENANT))

    from_acme = {n.entity_id for n in hood.entities() if n.hop == 1}
    assert len(from_acme) >= 25
    rows = hub.queries[0][1]
    assert rows["trv_fanout_cap"] == 25
    assert len(hood.nodes) <= 2 + 25 * 2


@pytest.mark.asyncio
async def test_second_hop_is_also_capped(hub: FakeGraph) -> None:
    """The bound is `fanout^depth`, and it holds when every hop-1 node is a hub too."""
    for index in range(475, 500):
        for other in range(100):
            hub.edge(
                EdgeType.USES, f"rival_{index:03d}", hub.entity(f"tech_{other}"), confidence=0.6
            )

    backend = GraphTraversalBackend(hub)
    hood = await backend.neighbourhood(["acme"], filters=Filter(tenant_id=TENANT))

    assert len(hood.nodes) <= 1 + 25 + 25 * 25
    assert hood.hops_run == 2
    assert hood.truncated


@pytest.mark.asyncio
async def test_depth_is_clamped_to_the_operator_ceiling(hub: FakeGraph) -> None:
    """`graph_depth` is one API call away from 6, and depth is the exponent."""
    backend = GraphTraversalBackend(hub, max_depth=2)
    await backend.search(request(seeds=["acme"], depth=9), limit=10)

    expansions = [q for q, p in hub.queries if "trv_frontier" in p]
    assert len(expansions) <= 2


@pytest.mark.asyncio
async def test_traversal_stops_when_a_hop_finds_nothing_new() -> None:
    """A small connected component must not cost `depth` round trips."""
    graph = FakeGraph()
    graph.entity("acme")
    backend = GraphTraversalBackend(graph, max_depth=2)
    hood = await backend.neighbourhood(["acme"], filters=Filter(tenant_id=TENANT))

    assert hood.hops_run == 1
    assert len([q for q, p in graph.queries if "trv_frontier" in p]) == 1


@pytest.mark.asyncio
async def test_node_cap_truncates_by_score() -> None:
    """A hundred-seed request is bounded even before the per-hop cap applies."""
    graph = FakeGraph()
    for index in range(40):
        seed = graph.entity(f"seed_{index:02d}")
        graph.edge(EdgeType.USES, seed, graph.entity(f"tech_{index:02d}"), confidence=0.9)
    backend = GraphTraversalBackend(graph, max_nodes=10, max_depth=1)
    hood = await backend.neighbourhood(
        [f"seed_{i:02d}" for i in range(40)], filters=Filter(tenant_id=TENANT)
    )

    assert len(hood.nodes) == 10
    assert hood.truncated


# --------------------------------------------------------------------------- #
# Traversal: scoring
# --------------------------------------------------------------------------- #


def two_hop_graph() -> FakeGraph:
    """acme -> near (1 hop) -> far (2 hops), plus a signal on each."""
    graph = FakeGraph()
    graph.entity("acme", name="Acme Corp")
    graph.entity("near", name="Near Co")
    graph.entity("far", name="Far Co")
    graph.edge(EdgeType.COMPETES_WITH, "acme", "near", confidence=0.9)
    graph.edge(EdgeType.COMPETES_WITH, "near", "far", confidence=0.9)
    graph.mention(graph.signal("sig_near", published=at(10)), "near", salience=1.0)
    graph.mention(graph.signal("sig_far", published=at(10)), "far", salience=1.0)
    return graph


@pytest.mark.asyncio
async def test_hop_decay_ranks_a_direct_neighbour_above_a_distant_one() -> None:
    """Without the decay, a hub two hops out outranks the seed's own neighbours."""
    backend = GraphTraversalBackend(two_hop_graph())
    candidates = await backend.search(request(seeds=["acme"]), limit=10)

    assert [c.signal_id for c in candidates] == ["sig_near", "sig_far"]
    near, far = candidates
    assert far.raw_score == pytest.approx(near.raw_score * 0.9 * HOP_DECAY**2)


@pytest.mark.asyncio
async def test_paths_accumulate_so_corroboration_outranks_a_single_route() -> None:
    """Two independent routes to one signal beat one route of the same strength.

    This is the property the graph has and the text backends do not; summing
    rather than taking the maximum is what expresses it.
    """
    graph = FakeGraph()
    graph.entity("seed_a")
    graph.entity("seed_b")
    corroborated = graph.entity("corroborated")
    single = graph.entity("single")
    graph.edge(EdgeType.USES, "seed_a", corroborated, confidence=0.5)
    graph.edge(EdgeType.USES, "seed_b", corroborated, confidence=0.5)
    graph.edge(EdgeType.USES, "seed_a", single, confidence=0.5)
    graph.mention(graph.signal("sig_corroborated"), corroborated, salience=1.0)
    graph.mention(graph.signal("sig_single"), single, salience=1.0)

    backend = GraphTraversalBackend(graph, max_depth=1)
    candidates = await backend.search(request(seeds=["seed_a", "seed_b"]), limit=10)

    assert [c.signal_id for c in candidates] == ["sig_corroborated", "sig_single"]
    assert candidates[0].raw_score == pytest.approx(2 * candidates[1].raw_score)


@pytest.mark.asyncio
async def test_an_unscored_edge_does_not_zero_the_path() -> None:
    """A rule-based extractor writes no confidence; zero would silence the backend."""
    graph = FakeGraph()
    graph.entity("acme")
    graph.edge(EdgeType.ACQUIRED, "acme", graph.entity("target"), confidence=None)
    graph.mention(graph.signal("sig_target"), "target", salience=1.0)

    backend = GraphTraversalBackend(graph, max_depth=1)
    candidates = await backend.search(request(seeds=["acme"]), limit=5)

    assert len(candidates) == 1
    assert candidates[0].raw_score == pytest.approx(DEFAULT_EDGE_CONFIDENCE * HOP_DECAY)


@pytest.mark.asyncio
async def test_candidates_are_graph_ranked_and_addressable() -> None:
    """Ranks are dense and 1-based; every candidate names its signal."""
    backend = GraphTraversalBackend(two_hop_graph())
    candidates = await backend.search(request(seeds=["acme"]), limit=10)

    assert [c.rank for c in candidates] == [1, 2]
    assert {c.backend for c in candidates} == {Backend.GRAPH}
    assert all(c.chunk_id.startswith(c.signal_id + ":") for c in candidates)


@pytest.mark.asyncio
async def test_limit_truncates_after_ranking_not_before() -> None:
    backend = GraphTraversalBackend(two_hop_graph())
    candidates = await backend.search(request(seeds=["acme"]), limit=1)
    assert [c.signal_id for c in candidates] == ["sig_near"]


@pytest.mark.asyncio
async def test_ties_break_deterministically() -> None:
    """Path scores are products of coarse confidences and tie constantly."""
    graph = FakeGraph()
    graph.entity("acme")
    for name in ("zeta", "alpha", "mid"):
        entity = graph.entity(name)
        graph.edge(EdgeType.USES, "acme", entity, confidence=0.5)
        graph.mention(graph.signal(f"sig_{name}"), entity, salience=0.5)

    backend = GraphTraversalBackend(graph, max_depth=1)
    first = await backend.search(request(seeds=["acme"]), limit=10)
    second = await backend.search(request(seeds=["acme"]), limit=10)
    assert [c.signal_id for c in first] == [c.signal_id for c in second]
    assert [c.signal_id for c in first] == ["sig_alpha", "sig_mid", "sig_zeta"]


@pytest.mark.asyncio
async def test_chunk_index_is_used_when_the_edge_carries_one() -> None:
    """An exact chunk join when the writer mirrored it; chunk 0 when it did not."""
    graph = FakeGraph()
    graph.entity("acme")
    graph.edge(EdgeType.USES, "acme", graph.entity("tech"), confidence=0.9)
    graph.mention(graph.signal("sig_exact"), "tech", chunk_index=7)
    graph.mention(graph.signal("sig_unknown"), "tech", chunk_index=None)

    backend = GraphTraversalBackend(graph, max_depth=1)
    by_signal = {
        c.signal_id: c.chunk_id for c in await backend.search(request(seeds=["acme"]), limit=5)
    }
    assert by_signal["sig_exact"] == "sig_exact:7"
    assert by_signal["sig_unknown"] == "sig_unknown:0"


# --------------------------------------------------------------------------- #
# Traversal: filters, time and tenancy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_window_is_pushed_into_the_signal_query() -> None:
    """A signal outside the window is never a graph candidate.

    The fake asserts the clause reached the query, so this fails both when the
    window is dropped and when it is applied to the returned list afterwards.
    """
    graph = FakeGraph()
    graph.entity("acme")
    graph.edge(EdgeType.USES, "acme", graph.entity("tech"), confidence=0.9)
    graph.mention(graph.signal("sig_in", published=at(10)), "tech")
    graph.mention(graph.signal("sig_early", published=at(1)), "tech")
    graph.mention(graph.signal("sig_late", published=at(28)), "tech")

    filters = Filter(tenant_id=TENANT, published_after=at(5), published_before=at(20))
    backend = GraphTraversalBackend(graph, max_depth=1)
    candidates = await backend.search(request(seeds=["acme"], filters=filters), limit=10)

    assert [c.signal_id for c in candidates] == ["sig_in"]


@pytest.mark.asyncio
async def test_another_tenants_graph_is_unreachable() -> None:
    graph = FakeGraph()
    graph.entity("acme")
    graph.edge(EdgeType.USES, "acme", graph.entity("theirs", tenant=OTHER_TENANT), confidence=1.0)
    graph.mention(graph.signal("sig_theirs", tenant=OTHER_TENANT), "theirs")

    backend = GraphTraversalBackend(graph, max_depth=1)
    assert await backend.search(request(seeds=["acme"]), limit=10) == []


@pytest.mark.asyncio
async def test_edges_are_read_as_of_the_window_end() -> None:
    """An acquisition learned in August is not a fact about Q1."""
    graph = FakeGraph()
    graph.entity("acme")
    graph.edge(
        EdgeType.ACQUIRED, "acme", graph.entity("later"), valid_from=at(1, month=8), confidence=1.0
    )
    graph.edge(
        EdgeType.ACQUIRED,
        "acme",
        graph.entity("closed"),
        valid_from=DAWN,
        # Closed *before* the window ends, so the as-of instant falls outside
        # `[valid_from, valid_to)`. An edge closed after it is still current at
        # the instant asked about and must stay.
        valid_to=at(5),
        confidence=1.0,
    )
    graph.edge(EdgeType.ACQUIRED, "acme", graph.entity("current"), confidence=1.0)
    for name in ("later", "closed", "current"):
        graph.mention(graph.signal(f"sig_{name}", published=at(10)), name)

    filters = Filter(tenant_id=TENANT, published_before=at(20))
    backend = GraphTraversalBackend(graph, max_depth=1)
    candidates = await backend.search(request(seeds=["acme"], filters=filters), limit=10)

    assert [c.signal_id for c in candidates] == ["sig_current"]


@pytest.mark.asyncio
async def test_mentions_is_not_walked_as_an_entity_edge() -> None:
    """Expanding through MENTIONS puts hub signals in the frontier."""
    assert EdgeType.MENTIONS.value not in TRAVERSABLE_EDGE_TYPES
    graph = FakeGraph()
    graph.entity("acme")
    backend = GraphTraversalBackend(graph)
    await backend.search(request(seeds=["acme"]), limit=5)
    sent = graph.queries[0][1]["trv_edge_types"]
    assert EdgeType.MENTIONS.value not in sent


@pytest.mark.asyncio
async def test_a_signal_reached_by_complains_about_is_kept_and_still_filtered() -> None:
    """An anonymous complainer is a Signal node, and is the evidence being sought.

    It skipped the MENTIONS hop, so the date window has to be re-applied to it --
    otherwise the one path into the graph that bypasses the filter is the one
    carrying the most opinionated content.
    """
    graph = FakeGraph()
    graph.entity("product", label="Product")
    graph.edge(EdgeType.COMPLAINS_ABOUT, graph.signal("sig_rant", published=at(10)), "product")
    graph.edge(EdgeType.COMPLAINS_ABOUT, graph.signal("sig_old", published=at(1)), "product")

    filters = Filter(tenant_id=TENANT, published_after=at(5))
    backend = GraphTraversalBackend(graph, max_depth=1)
    candidates = await backend.search(request(seeds=["product"], filters=filters), limit=10)

    assert [c.signal_id for c in candidates] == ["sig_rant"]


def test_the_clause_order_guard_catches_the_shape_it_is_aimed_at() -> None:
    """Guard on the guard: it has to fail on a doubled `WHERE` and pass real queries."""
    with pytest.raises(AssertionError, match="sub-clause"):
        assert_cypher_clause_order(
            "MATCH (s:Signal)\nWHERE s.id IN $ids\nWHERE s.tenant_id = $t\nRETURN s.id"
        )
    # A second WHERE under a *second* MATCH is ordinary Cypher and must pass, as
    # must one inside a CALL or EXISTS block whose outer scope has none.
    assert_cypher_clause_order(
        "MATCH (a) WHERE a.id IN $x WITH a MATCH (a)-[r]-(b) WHERE type(r) IN $y RETURN b"
    )
    assert_cypher_clause_order(
        "MATCH (e) CALL { WITH e MATCH (s)-[:MENTIONS]->(e) "
        "WHERE EXISTS { MATCH (s)-[:MENTIONS]->(f) WHERE f.id IN $ids } RETURN s } RETURN s"
    )


@pytest.mark.asyncio
async def test_a_signal_reached_directly_is_filtered_by_one_where_clause() -> None:
    """The `COMPLAINS_ABOUT` path composes its own query, so its shape is pinned.

    Its id restriction and the compiled filter have to reach the server as one
    `WHERE`. Written as two lines the query does not narrow the corpus, it fails
    to parse -- and the fake, which dispatches on parameters, would never notice.
    """
    graph = FakeGraph()
    graph.entity("product", label="Product")
    graph.edge(EdgeType.COMPLAINS_ABOUT, graph.signal("sig_rant", published=at(10)), "product")

    backend = GraphTraversalBackend(graph, max_depth=1)
    await backend.search(
        request(seeds=["product"], filters=Filter(tenant_id=TENANT, published_after=at(5))),
        limit=10,
    )

    direct = next(q for q, p in graph.queries if "trv_signal_ids" in p)
    assert direct.count("WHERE") == 1
    assert "s.id IN $trv_signal_ids" in direct
    assert "s.tenant_id = $flt_tenant_id_equals" in direct
    assert "s.published_at >= $flt_published_at_gte" in direct


@pytest.mark.asyncio
async def test_an_entity_filter_is_pushed_into_the_mention_subquery() -> None:
    """`entity_ids` compiles to an `EXISTS` block nested inside the `CALL` block.

    The most fragile composition in the module: a compiled clause interpolated
    into a subquery inside another subquery. If it were applied to the returned
    rows instead, the off-topic signal below would still be a candidate.
    """
    graph = FakeGraph()
    graph.entity("acme")
    graph.edge(EdgeType.USES, "acme", graph.entity("tech"), confidence=0.9)
    graph.entity("ent_wanted")
    graph.mention(graph.signal("sig_on_topic"), "tech")
    graph.mention("sig_on_topic", "ent_wanted")
    graph.mention(graph.signal("sig_off_topic"), "tech")

    filters = Filter(tenant_id=TENANT, entity_ids=frozenset({"ent_wanted"}))
    backend = GraphTraversalBackend(graph, max_depth=1)
    candidates = await backend.search(request(seeds=["acme"], filters=filters), limit=10)

    assert [c.signal_id for c in candidates] == ["sig_on_topic"]


@pytest.mark.asyncio
async def test_a_retracted_mention_is_not_evidence() -> None:
    graph = FakeGraph()
    graph.entity("acme")
    graph.edge(EdgeType.USES, "acme", graph.entity("tech"), confidence=0.9)
    graph.mention(graph.signal("sig_live"), "tech")
    graph.mention(graph.signal("sig_retracted"), "tech", valid_to=at(2))

    backend = GraphTraversalBackend(graph, max_depth=1)
    candidates = await backend.search(
        request(seeds=["acme"], filters=Filter(tenant_id=TENANT)), limit=5
    )
    assert [c.signal_id for c in candidates] == ["sig_live"]


@pytest.mark.asyncio
async def test_labels_learned_on_the_first_hop_narrow_the_later_matches() -> None:
    """A label-less `id` lookup cannot use the per-label uniqueness index.

    Seed ids do not carry a label, so hop 1 has to match unlabelled; every hop
    after it can seek instead of scan, and an unrecognised label falls back to an
    unlabelled match rather than dropping the node.
    """
    graph = FakeGraph()
    graph.entity("acme", label="Company")
    graph.edge(EdgeType.USES, "acme", graph.entity("tech", label="Technology"), confidence=0.9)
    graph.mention(graph.signal("sig_tech"), "tech")

    backend = GraphTraversalBackend(graph)
    await backend.search(request(seeds=["acme"]), limit=5)

    expansions = [q for q, p in graph.queries if "trv_frontier" in p]
    assert "MATCH (a)\n" in expansions[0], "seed labels are unknown; hop 1 must not narrow"
    assert "MATCH (a:Technology)" in expansions[1]
    signals = next(q for q, p in graph.queries if "trv_entities" in p)
    assert "MATCH (e:Company|Technology)" in signals


@pytest.mark.asyncio
async def test_one_request_reads_the_graph_at_one_instant() -> None:
    """An open-ended window resolves `as_of` to `utcnow()`, once per request.

    Resolving it per query would read the edge graph and the mention graph a
    millisecond apart -- invisible until an edge closes in the gap, and then
    unreproducible.
    """
    graph = FakeGraph()
    graph.entity("acme")
    graph.edge(EdgeType.USES, "acme", graph.entity("tech"), confidence=0.9)
    graph.mention(graph.signal("sig_one"), "tech")

    backend = GraphTraversalBackend(graph, max_depth=1)
    await backend.search(request(seeds=["acme"], filters=Filter(tenant_id=TENANT)), limit=5)

    instants = {p["trv_as_of"] for _, p in graph.queries if "trv_as_of" in p}
    assert len(instants) == 1


# --------------------------------------------------------------------------- #
# Traversal: contract
# --------------------------------------------------------------------------- #


def test_traversal_satisfies_the_search_backend_protocol() -> None:
    assert isinstance(GraphTraversalBackend(FakeGraph()), SearchBackend)
    assert GraphTraversalBackend(FakeGraph()).backend is Backend.GRAPH


@pytest.mark.asyncio
async def test_no_seeds_is_a_normal_empty_result_with_no_round_trip() -> None:
    """`docs/retrieval.md` §6: not an error. The run continues on two backends."""
    graph = FakeGraph()
    backend = GraphTraversalBackend(graph)
    assert await backend.search(request(seeds=[]), limit=10) == []
    assert graph.queries == []


@pytest.mark.asyncio
async def test_unknown_seeds_return_nothing_rather_than_everything() -> None:
    graph = FakeGraph()
    graph.entity("acme")
    backend = GraphTraversalBackend(graph)
    assert await backend.search(request(seeds=["ghost"]), limit=10) == []


@pytest.mark.asyncio
async def test_a_reader_failure_propagates() -> None:
    """`HybridRetriever` records the backend as failed and lowers confidence.

    Swallowing the error here would return an empty list indistinguishable from
    a sparse entity, and the run would be reported as complete.
    """

    async def broken(query: str, parameters: Mapping[str, Any] | None = None) -> list[Any]:
        raise RuntimeError("neo4j unreachable")

    backend = GraphTraversalBackend(broken)
    with pytest.raises(RuntimeError, match="unreachable"):
        await backend.search(request(seeds=["acme"]), limit=10)


def test_constructor_refuses_a_bound_that_is_not_a_bound() -> None:
    for kwargs in ({"max_depth": 0}, {"fanout_cap": 0}, {"hop_decay": 0.0}, {"max_nodes": 0}):
        with pytest.raises(ValueError):
            GraphTraversalBackend(FakeGraph(), **kwargs)


@pytest.mark.asyncio
async def test_limit_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="limit"):
        await GraphTraversalBackend(FakeGraph()).search(request(seeds=["acme"]), limit=0)


# --------------------------------------------------------------------------- #
# Expansion: aliases
# --------------------------------------------------------------------------- #


def alias_graph() -> FakeGraph:
    graph = FakeGraph()
    graph.entity("ent_ddog", name="Datadog", aliases=["DDOG", "Datadog Inc", "Datadog, Inc."])
    graph.entity("ent_grafana", name="Grafana Labs", aliases=["Grafana"])
    graph.entity("ent_other", name="Elsewhere", aliases=["ELSE"], tenant=OTHER_TENANT)
    return graph


@pytest.mark.asyncio
async def test_expansion_turns_one_entity_into_its_surface_forms() -> None:
    """The gap the module exists to close: three strings, one entity."""
    expander = GraphQueryExpander(alias_graph(), resolve_query_text=False)
    terms = await expander.expand_query(request(seeds=["ent_ddog"], query="observability spend"))

    assert set(terms) == {"Datadog", "DDOG", "Datadog Inc", "Datadog, Inc."}
    assert terms[0] == "Datadog", "canonical name first, so the cap drops aliases not names"


@pytest.mark.asyncio
async def test_expansion_drops_terms_the_query_already_contains() -> None:
    """A literal term re-added at boost 0.6 changes ranking for no stated reason."""
    expander = GraphQueryExpander(alias_graph(), resolve_query_text=False)
    terms = await expander.expand_query(request(seeds=["ent_ddog"], query="is datadog worth it"))

    assert "Datadog" not in terms
    assert "DDOG" in terms


@pytest.mark.asyncio
async def test_expansion_is_capped_per_entity_and_overall() -> None:
    """An over-merged entity must not starve the other seeds, or break Lucene."""
    graph = FakeGraph()
    graph.entity("ent_blob", name="Blob", aliases=[f"alias_{i:03d}" for i in range(300)])
    graph.entity("ent_small", name="Small Co", aliases=["SC"])

    expander = GraphQueryExpander(
        graph, resolve_query_text=False, max_aliases_per_entity=3, max_terms=8
    )
    terms = await expander.expand_query(request(seeds=["ent_blob", "ent_small"], query="q"))

    assert len(terms) == 6
    assert "Small Co" in terms and "SC" in terms


@pytest.mark.asyncio
async def test_expansion_is_deterministic_and_ordered_by_seed() -> None:
    """Expansion changes ranking; a reshuffling term list reshuffles results."""
    expander = GraphQueryExpander(alias_graph(), resolve_query_text=False)
    seeds = ["ent_grafana", "ent_ddog"]
    once = await expander.expand_query(request(seeds=seeds, query="q"))
    twice = await expander.expand_query(request(seeds=seeds, query="q"))

    assert once == twice
    assert once[0] == "Grafana Labs", "the caller's first seed is the subject of the question"


@pytest.mark.asyncio
async def test_expansion_never_crosses_a_tenant() -> None:
    expander = GraphQueryExpander(alias_graph(), resolve_query_text=False)
    terms = await expander.expand_query(
        RetrievalRequest(
            query="q", filters=Filter(tenant_id=TENANT), seed_entity_ids=("ent_other",)
        )
    )
    assert terms == []


@pytest.mark.asyncio
async def test_query_text_resolves_to_entities_through_the_fulltext_index() -> None:
    """The ticker path: the analyst types DDOG and gets the company's coverage."""
    expander = GraphQueryExpander(alias_graph(), max_resolved_entities=2)
    terms = await expander.expand_query(request(query="how is DDOG doing"))

    assert "Datadog" in terms
    assert "DDOG" not in terms, "already in the query"


@pytest.mark.asyncio
async def test_query_text_resolution_can_be_switched_off() -> None:
    graph = alias_graph()
    expander = GraphQueryExpander(graph, resolve_query_text=False)
    await expander.expand_query(request(query="DDOG"))
    assert not [q for q, p in graph.queries if "exp_query" in p]


@pytest.mark.parametrize(
    "text",
    ['a "quoted" term', "trailing~", "(unbalanced", "a AND OR NOT b", "back\\slash", "café"],
)
def test_lucene_safe_never_produces_a_query_that_cannot_parse(text: str) -> None:
    """A stray quote from user text is a parse error inside an *optimisation*.

    Quoting each token is the small surface: inside a phrase only `"` and `\\`
    are special, whereas an escape table has to track Lucene's operator set
    forever and fails open when it drifts.
    """
    safe = lucene_safe(text)
    assert safe.count('"') % 2 == 0
    for fragment in safe.split(" OR "):
        assert fragment.startswith('"') and fragment.endswith('"')


def test_lucene_safe_returns_nothing_for_text_with_no_usable_tokens() -> None:
    """Empty means "skip the round trip" rather than "match everything"."""
    assert lucene_safe("  ?  ") == ""
    assert lucene_safe("") == ""


@pytest.mark.asyncio
async def test_the_two_lookups_run_concurrently() -> None:
    """Expansion runs *before* fan-out, so its latency is on every query's path.

    Sequential round trips would make widening a query by a handful of terms
    cost the sum of two graph calls before the search has started.
    """
    inflight = 0
    peak = 0

    async def reader(query: str, parameters: Mapping[str, Any] | None = None) -> list[Any]:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        await asyncio.sleep(0)
        inflight -= 1
        return []

    expander = GraphQueryExpander(reader)
    await expander.expand_query(request(seeds=["ent_ddog"], query="datadog pricing"))
    assert peak == 2


@pytest.mark.asyncio
async def test_expansion_skips_the_lookup_when_there_is_nothing_to_look_up() -> None:
    graph = alias_graph()
    expander = GraphQueryExpander(graph)
    assert await expander.expand_query(request(query="!!")) == []
    assert graph.queries == []


# --------------------------------------------------------------------------- #
# Expansion: graph facts
# --------------------------------------------------------------------------- #


def fact_graph() -> FakeGraph:
    graph = FakeGraph()
    graph.entity("acme", name="Acme Corp")
    graph.entity("globex", name="Globex")
    graph.entity("initech", name="Initech")
    graph.signal("sig_1", published=at(10))
    graph.mention("sig_1", "acme")
    graph.edge(
        EdgeType.COMPETES_WITH,
        "acme",
        "globex",
        confidence=0.82,
        valid_from=at(1, month=3),
        signals=["sig_a", "sig_b", "sig_c"],
    )
    graph.edge(
        EdgeType.ACQUIRED,
        "acme",
        "initech",
        confidence=0.4,
        valid_from=DAWN,
        valid_to=at(1, month=2),
    )
    return graph


@pytest.mark.asyncio
async def test_facts_carry_temporal_validity_and_citations() -> None:
    """A fact is rendered with a citation handle; it has to carry one."""
    expander = GraphQueryExpander(fact_graph())
    facts = await expander.facts_for(
        request(seeds=["acme"], filters=Filter(tenant_id=TENANT, published_before=at(1, month=6))),
        ["sig_1"],
    )

    assert len(facts) == 1
    fact = facts[0]
    assert (fact.subject_name, fact.predicate, fact.object_name) == (
        "Acme Corp",
        "COMPETES_WITH",
        "Globex",
    )
    assert fact.valid_from == at(1, month=3)
    assert fact.is_current
    assert fact.confidence == pytest.approx(0.82)
    assert list(fact.supporting_signal_ids) == ["sig_a", "sig_b", "sig_c"]


@pytest.mark.asyncio
async def test_facts_exclude_intervals_that_do_not_cover_the_as_of_instant() -> None:
    """The invariant in `docs/knowledge-graph.md` §11, stated as a test."""
    expander = GraphQueryExpander(fact_graph())
    facts = await expander.facts_for(
        request(seeds=["acme"], filters=Filter(tenant_id=TENANT, published_before=at(1, month=6))),
        [],
    )
    assert [f.predicate for f in facts] == ["COMPETES_WITH"]


@pytest.mark.asyncio
async def test_a_relationship_not_yet_known_at_the_window_end_is_absent() -> None:
    """Reading today's graph for a Q1 question back-dates every edge since."""
    expander = GraphQueryExpander(fact_graph())
    facts = await expander.facts_for(
        request(seeds=["acme"], filters=Filter(tenant_id=TENANT, published_before=at(1, month=2))),
        [],
    )
    assert facts == []


@pytest.mark.asyncio
async def test_history_is_available_when_it_is_asked_for() -> None:
    """ "Who did Acme compete with before the merger" is a real question."""
    expander = GraphQueryExpander(fact_graph(), include_historical_facts=True)
    facts = await expander.facts_for(
        request(seeds=["acme"], filters=Filter(tenant_id=TENANT, published_before=at(1, month=6))),
        [],
    )
    closed = [f for f in facts if not f.is_current]
    assert [f.predicate for f in closed] == ["ACQUIRED"]
    assert closed[0].valid_to == at(1, month=2)


@pytest.mark.asyncio
async def test_a_symmetric_edge_is_reported_once() -> None:
    """Stored once, matched undirected; two orientations read as two facts."""
    graph = fact_graph()
    expander = GraphQueryExpander(graph)
    facts = await expander.facts_for(
        request(
            seeds=["acme", "globex"],
            filters=Filter(tenant_id=TENANT, published_before=at(1, month=6)),
        ),
        [],
    )
    assert len(facts) == 1


@pytest.mark.asyncio
async def test_facts_are_ordered_by_confidence_and_capped() -> None:
    graph = FakeGraph()
    graph.entity("acme")
    for index in range(40):
        graph.edge(EdgeType.USES, "acme", graph.entity(f"tech_{index:02d}"), confidence=index / 100)
    expander = GraphQueryExpander(graph, max_facts=5)
    facts = await expander.facts_for(request(seeds=["acme"]), [])

    assert len(facts) == 5
    assert [f.object_id for f in facts] == [f"tech_{i}" for i in (39, 38, 37, 36, 35)]


@pytest.mark.asyncio
async def test_mentions_is_not_rendered_as_a_fact() -> None:
    """ "This document mentions Acme" is the retrieval, not a fact about the world."""
    assert EdgeType.MENTIONS.value not in FACT_EDGE_TYPES
    graph = fact_graph()
    expander = GraphQueryExpander(graph)
    await expander.facts_for(request(seeds=["acme"]), ["sig_1"])
    sent = next(p for q, p in graph.queries if "exp_signal_ids" in p)
    assert EdgeType.MENTIONS.value not in sent["exp_edge_types"]


@pytest.mark.asyncio
async def test_facts_for_nothing_costs_no_round_trip() -> None:
    graph = fact_graph()
    expander = GraphQueryExpander(graph)
    assert await expander.facts_for(request(seeds=[]), []) == []
    assert graph.queries == []


@pytest.mark.asyncio
async def test_facts_never_cross_a_tenant() -> None:
    graph = fact_graph()
    graph.entity("theirs", name="Theirs", tenant=OTHER_TENANT)
    graph.edge(EdgeType.COMPETES_WITH, "acme", "theirs", confidence=1.0)
    expander = GraphQueryExpander(graph)
    facts = await expander.facts_for(request(seeds=["acme"]), [])
    assert all(f.object_id != "theirs" for f in facts)


@pytest.mark.asyncio
async def test_driver_temporal_types_become_aware_datetimes() -> None:
    """`neo4j.time.DateTime` compared against an aware `as_of` raises, eventually.

    Coercing at the boundary is what keeps `GraphFact.is_current` meaningful
    instead of dependent on which code path produced the value.
    """

    async def reader(query: str, parameters: Mapping[str, Any] | None = None) -> list[Any]:
        return [
            {
                "predicate": "USES",
                "subject_id": "acme",
                "subject_name": "Acme",
                "object_id": "tech",
                "object_name": "Tech",
                "valid_from": Neo4jDateTime(2026, 3, 1, 0, 0, 0),
                "valid_to": None,
                "confidence": 1.4,
                "supporting_signal_ids": ["sig_1"],
            }
        ]

    facts = await GraphQueryExpander(reader).facts_for(request(seeds=["acme"]), [])
    assert facts[0].valid_from == datetime(2026, 3, 1, tzinfo=UTC)
    assert facts[0].confidence == 1.0, "a score above 1 must not dominate the budget"


@pytest.mark.asyncio
async def test_a_fact_that_cannot_be_cited_is_dropped() -> None:
    """A citation handle that resolves to nothing is worse than an absent fact."""

    async def reader(query: str, parameters: Mapping[str, Any] | None = None) -> list[Any]:
        return [
            {"predicate": "USES", "subject_id": "", "object_id": "tech"},
            {"predicate": "", "subject_id": "acme", "object_id": "tech"},
        ]

    assert await GraphQueryExpander(reader).facts_for(request(seeds=["acme"]), []) == []


def test_expander_refuses_a_bound_that_would_silently_drop_citations() -> None:
    """A negative slice bound is a Cypher *feature*: it counts from the end.

    `[..-2]` drops the last two supporting signals from every fact rather than
    raising, so the failure surfaces as facts with thinner provenance than the
    graph holds -- which nothing downstream can distinguish from a sparse edge.
    """
    for kwargs in ({"max_terms": 0}, {"max_facts": 0}, {"max_supporting_signals": -1}):
        with pytest.raises(ValueError):
            GraphQueryExpander(FakeGraph(), **kwargs)


def test_expander_satisfies_the_graph_expander_protocol() -> None:
    assert isinstance(GraphQueryExpander(FakeGraph()), GraphExpander)
