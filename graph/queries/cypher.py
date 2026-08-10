"""Parameterized Cypher templates for the traversals the product actually runs.

Every query in `docs/knowledge-graph.md` §9 lives here, as a function returning a
`Query` -- the text and its parameter map together, never separately. That
pairing is the whole design: a template returning only a string invites the
caller to assemble the parameter dict themselves, and the first time someone
inlines a tenant id "just for this one query" the module stops being a boundary.

**Three things this module refuses to do.**

*Interpolate a value.* Not one. The only things that reach query text are labels,
relationship types and integer path bounds -- labels and types come from closed
enums and pass `validate_label()`, bounds are range-checked ints. Cypher has no
parameter form for a label (`MATCH (n:$label)` is a syntax error) or for the
bounds of a variable-length pattern, so this is unavoidable rather than
convenient, and the validation is what makes it safe.

*Duplicate the as-of predicate.* `graph/temporal/validity.as_of_cypher()` emits
it and every template calls that function. A second hand-written copy is how one
query ends up with `valid_to >= $as_of` and another with `>`, so an edge that
closed at exactly the queried instant is returned by one and not by the other.
Both read correctly in review.

*Depend on APOC.* `docs/knowledge-graph.md` §9 writes the neighbourhood query
with `apoc.path.subgraphNodes`, and the compose file does load the plugin. But
APOC is a *plugin*: a deployment without it gets `Unknown procedure` at runtime,
on the read path, in production, from a query the test suite never exercised
because the test Neo4j had the plugin. The default is pure Cypher; the APOC form
is behind an explicit flag. Same reasoning as
`graph/schema/nodes.list_union_expression()`.

**On `coalesce` around confidence.** Comparing a confidence is three-valued: an
edge whose `confidence` was never set compares `null >= 0.0` as `null`, and most
rule-extracted edges never carried one. No template filters on confidence today
-- the ones that did went with the market traversals -- but every template that
*ranks* by it writes `coalesce(r.confidence, 0.0)`, so an unscored edge sorts as
the weakest rather than at whichever end the server happens to put nulls. Any
filter added later must do the same, or `min_confidence=0.0` will mean "only
edges that carry a confidence" instead of "no minimum".

Layer note: **L1 library** -- `models/` plus the rest of `graph/`. No driver
import and no kernel import; this module builds strings and dicts and executes
nothing. `graph/client.py` runs them.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final, NamedTuple

from graph.schema.constraints import FULLTEXT_INDEX_NAME
from graph.schema.edges import SIGNAL_LABEL, edge_spec
from graph.schema.nodes import GraphSchemaError, entity_labels, node_spec, validate_label
from graph.temporal.validity import AS_OF_PARAM, as_of_cypher, coerce_instant
from models.enums import EdgeType, EntityType

__all__ = [
    "DEFAULT_FANOUT_CAP",
    "DEFAULT_LIMIT",
    "MAX_HOPS",
    "MAX_LIMIT",
    "TRAVERSABLE_EDGE_TYPES",
    "Query",
    "entity_by_id",
    "entity_neighbours",
    "entity_search",
    "neighbourhood_signals",
    "paths_between",
    "signals_mentioning",
    "stale_analytics_nodes",
    "subgraph_edges",
    "topic_activity",
]


class Query(NamedTuple):
    """A Cypher statement and its parameters, inseparable.

    Returned by every builder below, so there is no call shape in which the text
    and the parameters come from two different places -- which is the shape where
    a value gets inlined.
    """

    cypher: str
    parameters: dict[str, Any]

    def with_parameters(self, **extra: Any) -> Query:
        """Return a copy with extra parameters merged in, for the caller that
        needs to add a driver-level hint without rebuilding the query."""
        return Query(self.cypher, {**self.parameters, **extra})


DEFAULT_LIMIT: Final[int] = 25

MAX_LIMIT: Final[int] = 500
"""Hard ceiling on any `LIMIT` these templates emit.

Not politeness. A `LIMIT` is the only bound on the result of a graph query -- the
planner will happily stream a million rows through the Bolt connection, and the
memory that costs is the *server's*, shared with every other query on the
instance. An unbounded neighbourhood read is the single most effective way to
make Neo4j unavailable to everyone else.
"""

MAX_HOPS: Final[int] = 5
"""Ceiling on variable-length path depth.

Path count grows with the branching factor raised to the depth. On a graph where
a popular Topic has ten thousand inbound `MENTIONS`, depth 4 is not slow -- it is
a query that does not finish.
"""

DEFAULT_FANOUT_CAP: Final[int] = 200

TRAVERSABLE_EDGE_TYPES: Final[tuple[EdgeType, ...]] = (
    EdgeType.MENTIONS,
    EdgeType.COMPETES_WITH,
    EdgeType.USES,
    EdgeType.COMPLAINS_ABOUT,
    EdgeType.LAUNCHED_BY,
    EdgeType.ACQUIRED,
)
"""Edge types a GraphRAG neighbourhood walk may cross.

`SAME_AS` and `DUPLICATE_OF` are excluded, and the exclusion is load-bearing.
Both are *bookkeeping* edges: `SAME_AS` records that resolution merged two nodes,
`DUPLICATE_OF` that dedup collapsed two signals. Traversing them makes an
expansion walk into a node's own merge history and return the same entity several
times under different ids, which the fusion layer in `retrieval/rerank/fusion.py`
then reads as independent corroboration. The graph would be manufacturing
agreement with itself.
"""


# --------------------------------------------------------------------------- #
# Argument guards
# --------------------------------------------------------------------------- #


def _require_limit(limit: int, name: str = "limit") -> int:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise GraphSchemaError(f"{name} must be an int, got {type(limit).__name__}")
    if limit < 1:
        raise GraphSchemaError(f"{name} must be at least 1, got {limit}")
    if limit > MAX_LIMIT:
        raise GraphSchemaError(
            f"{name}={limit} exceeds MAX_LIMIT={MAX_LIMIT}. An unbounded graph read "
            "consumes server memory shared with every other query on the instance."
        )
    return limit


def _require_tenant(tenant_id: str) -> str:
    """Refuse an empty tenant.

    An empty string is a perfectly valid Cypher parameter. It matches the nodes of
    whichever tenant wrote one, which in practice is none -- so the query returns
    nothing and the caller reports "no data" for what is actually a wiring bug.
    Failing here names the real problem.
    """
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise GraphSchemaError("tenant_id is required and must be a non-empty string")
    return tenant_id


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GraphSchemaError(f"{name} is required and must be a non-empty string")
    return value


def _instant(value: datetime | None, *, name: str) -> datetime:
    """Normalise an instant to aware UTC. `None` means now.

    Delegated to `graph/temporal/validity.coerce_instant` so the rejection of
    naive datetimes and sentinel years happens in one place. A query that
    accepted a naive datetime would compare a local wall clock against stored UTC
    and be wrong by the server's offset -- silently, and only for deployments
    outside UTC.
    """
    if value is None:
        return datetime.now(UTC)
    return coerce_instant(value, field_name=name)


def _relationship_pattern(edge_types: Sequence[EdgeType]) -> str:
    """`MENTIONS|COMPETES_WITH|...`, validated, in the caller's order.

    Reaches query text, so every member goes through `edge_spec()` -- which
    rejects `UNKNOWN` -- and then `validate_label()`. Deduplicated while
    preserving order, because `[:A|A]` is legal Cypher that reads as a mistake.
    """
    if not edge_types:
        raise GraphSchemaError("at least one edge type is required to traverse")
    seen: list[str] = []
    for edge_type in edge_types:
        name = validate_label(edge_spec(edge_type).type_name)
        if name not in seen:
            seen.append(name)
    return "|".join(seen)


def _label_filter(entity_types: Sequence[EntityType] | None) -> list[str] | None:
    """Turn an entity-type filter into a parameterisable list of label strings.

    A *parameter*, not interpolation: the labels go into `$labels` and the query
    says `any(l IN labels(node) WHERE l IN $labels)`. That is slower than
    `MATCH (n:Company|Product)` and it is the right trade here, because this
    filter arrives from an HTTP query string and the alternative is building
    query text out of request data.
    """
    if entity_types is None:
        return None
    known = set(entity_labels())
    labels: list[str] = []
    for entity_type in entity_types:
        if entity_type is EntityType.UNKNOWN:
            raise GraphSchemaError(
                "EntityType.UNKNOWN cannot be a search filter; it means 'a type "
                "this build does not recognise', which no node carries"
            )
        label = entity_type.value
        if label not in known:
            raise GraphSchemaError(f"{label!r} is not a searchable entity label")
        if label not in labels:
            labels.append(label)
    if not labels:
        # An empty list makes `l IN $labels` false for every node and returns
        # nothing, which reads as "no results" rather than "you filtered
        # everything out". `None` is how a caller says "no filter".
        raise GraphSchemaError("entity_types was empty; pass None for no filter")
    return labels


# --------------------------------------------------------------------------- #
# §9.1  Competitors of X, as of a date
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# §9.2  Topics most complained about for product Y
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# §9.3  Acquisition chain
# --------------------------------------------------------------------------- #




# --------------------------------------------------------------------------- #
# §9.4  Bounded neighbourhood for GraphRAG
# --------------------------------------------------------------------------- #


def neighbourhood_signals(
    *,
    tenant_id: str,
    seed_ids: Sequence[str],
    start: datetime,
    end: datetime,
    max_level: int = 2,
    fanout_cap: int = DEFAULT_FANOUT_CAP,
    limit: int = DEFAULT_LIMIT,
    edge_types: Sequence[EdgeType] = TRAVERSABLE_EDGE_TYPES,
    use_apoc: bool = False,
) -> Query:
    """Signals reachable from a set of seed entities within `max_level` hops.

    The backing query for `retrieval/graph_retrieval/traversal.py`: seeds come
    from vector and keyword hits, and this finds the evidence those entities
    connect to that lexical and semantic search both missed.

    **Pure Cypher by default, APOC by request.** `docs/knowledge-graph.md` §9
    writes this with `apoc.path.subgraphNodes`, and APOC does dedupe the visited
    set as it walks, which the pure form cannot. But APOC is a plugin, and the
    failure mode of assuming it is `Unknown procedure` on the read path in
    production. `use_apoc=True` is available where the plugin is known present
    and the fanout is large enough for the difference to matter.

    The pure form's `WITH DISTINCT node` after the variable-length match is what
    keeps the result honest: without it a node reachable by three paths appears
    three times, and the caller counts three corroborating routes where there is
    one entity.

    `fanout_cap` bounds the entity set *before* the signal match, which is the
    ordering that matters. A Topic with ten thousand inbound `MENTIONS` would
    otherwise expand to ten thousand signals and only then be truncated, having
    already done all the work.
    """
    _require_tenant(tenant_id)
    _require_limit(limit)
    _require_limit(fanout_cap, "fanout_cap")
    if not seed_ids:
        raise GraphSchemaError(
            "seed_ids is empty; an unseeded neighbourhood walk is a full graph scan"
        )
    if not isinstance(max_level, int) or isinstance(max_level, bool):
        raise GraphSchemaError(f"max_level must be an int, got {type(max_level).__name__}")
    if not 1 <= max_level <= MAX_HOPS:
        raise GraphSchemaError(f"max_level must be between 1 and {MAX_HOPS}, got {max_level}")

    resolved_start = _instant(start, name="start")
    resolved_end = _instant(end, name="end")
    if resolved_end <= resolved_start:
        raise GraphSchemaError(
            f"window [{resolved_start.isoformat()}, {resolved_end.isoformat()}) "
            "is empty or inverted"
        )

    relationships = _relationship_pattern(edge_types)

    if use_apoc:
        expansion = f"""
CALL apoc.path.subgraphNodes(seed, {{
  relationshipFilter: '{relationships}',
  maxLevel: $max_level,
  limit: $fanout_cap
}}) YIELD node
WITH DISTINCT node
WHERE node.tenant_id = $tenant_id
""".strip()
    else:
        # `*1..N` is interpolated because Cypher cannot parameterise path bounds.
        # `max_level` is a range-checked int, so there is no string to inject.
        expansion = f"""
MATCH (seed)-[:{relationships}*1..{max_level}]-(node)
WHERE node.tenant_id = $tenant_id
WITH DISTINCT node
LIMIT $fanout_cap
""".strip()

    cypher = f"""
MATCH (seed)
WHERE seed.id IN $seed_ids AND seed.tenant_id = $tenant_id
{expansion}
MATCH (s:{SIGNAL_LABEL})-[m:MENTIONS]->(node)
WHERE s.published_at >= $start AND s.published_at < $end
RETURN s.id                          AS signal_id,
       node.id                       AS via_entity_id,
       node.canonical_name           AS via_entity_name,
       labels(node)[0]               AS via_entity_type,
       max(coalesce(m.salience, 0.0)) AS salience
ORDER BY salience DESC, signal_id ASC
LIMIT $limit
""".strip()

    parameters: dict[str, Any] = {
        "tenant_id": tenant_id,
        "seed_ids": list(seed_ids),
        "start": resolved_start,
        "end": resolved_end,
        "fanout_cap": fanout_cap,
        "limit": limit,
    }
    if use_apoc:
        parameters["max_level"] = max_level
    return Query(cypher, parameters)


# --------------------------------------------------------------------------- #
# §9.5  Entity search -- backs GET /api/v1/graph/search
# --------------------------------------------------------------------------- #


def entity_search(
    *,
    tenant_id: str,
    query: str,
    entity_types: Sequence[EntityType] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Query:
    """Fulltext entity lookup over `canonical_name`, `aliases` and `description`.

    The index name comes from `graph/schema/constraints.FULLTEXT_INDEX_NAME`, so
    a rename cannot leave this query pointing at an index that no longer exists.

    **The tenant filter runs after the index**, and that is a real weakness worth
    naming rather than hiding. `db.index.fulltext.queryNodes` scores across every
    tenant's nodes and `WHERE node.tenant_id = $tenant_id` discards the
    non-matching ones afterwards, so a tenant whose entities rank below another's
    can receive fewer than `limit` results. Neo4j's fulltext index cannot be
    partitioned by a property. The over-fetch below mitigates it; the real fix is
    a database per tenant, `docs/knowledge-graph.md` open question 6.

    No result *leaks* across tenants -- the filter is applied before anything is
    returned. The failure mode is recall, not disclosure.
    """
    _require_tenant(tenant_id)
    _require_text(query, "query")
    _require_limit(limit)

    labels = _label_filter(entity_types)
    over_fetch = min(limit * 4, MAX_LIMIT)
    index = validate_label(FULLTEXT_INDEX_NAME)

    cypher = f"""
CALL db.index.fulltext.queryNodes('{index}', $query, {{limit: $over_fetch}})
YIELD node, score
WITH node, score
WHERE node.tenant_id = $tenant_id
  AND ($labels IS NULL OR any(l IN labels(node) WHERE l IN $labels))
RETURN node.id             AS id,
       node.canonical_name AS name,
       labels(node)[0]     AS type,
       node.description    AS description,
       coalesce(node.aliases, [])[..5] AS aliases,
       node.source_count   AS source_count,
       node.pagerank_score AS pagerank_score,
       score               AS score
ORDER BY score DESC, id ASC
LIMIT $limit
""".strip()

    return Query(
        cypher,
        {
            "tenant_id": tenant_id,
            "query": query,
            "labels": labels,
            "over_fetch": over_fetch,
            "limit": limit,
        },
    )


# --------------------------------------------------------------------------- #
# Supporting reads -- not in §9, required by services/graph_service.py
# --------------------------------------------------------------------------- #


def entity_by_id(*, tenant_id: str, entity_id: str) -> Query:
    """One entity by canonical id, with its analytics properties.

    No label in the pattern. `id` is unique *per label*, so a labelless match is
    the only way to fetch an entity whose type the caller does not know -- which
    is the normal case, because ids travel through this system without their
    labels attached. It costs a union of the seven `id` indexes rather than one
    seek; acceptable for a single-entity read, and the reason
    `stale_analytics_nodes` below takes the label when the caller has it.
    """
    _require_tenant(tenant_id)
    _require_text(entity_id, "entity_id")
    cypher = """
MATCH (n {id: $entity_id, tenant_id: $tenant_id})
RETURN n.id               AS id,
       n.canonical_name   AS name,
       labels(n)[0]       AS type,
       n.normalized_name  AS normalized_name,
       n.description      AS description,
       coalesce(n.aliases, [])     AS aliases,
       coalesce(n.merged_from, []) AS merged_from,
       n.confidence       AS confidence,
       n.source_count     AS source_count,
       n.first_seen       AS first_seen,
       n.last_seen        AS last_seen,
       n.pagerank_score   AS pagerank_score,
       n.community_id     AS community_id,
       n.computed_at      AS computed_at,
       n.schema_version   AS schema_version
LIMIT 1
""".strip()
    return Query(cypher, {"tenant_id": tenant_id, "entity_id": entity_id})


def signals_mentioning(
    *,
    tenant_id: str,
    entity_id: str,
    min_salience: float = 0.0,
    since: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Query:
    """Signal ids that mention an entity, most salient first.

    The citation path. An entity named in a report is only defensible if it walks
    back to the signals that evidenced it, and this is that walk.
    """
    _require_tenant(tenant_id)
    _require_text(entity_id, "entity_id")
    _require_limit(limit)
    cypher = f"""
MATCH (s:{SIGNAL_LABEL})-[m:MENTIONS]->(n {{id: $entity_id, tenant_id: $tenant_id}})
WHERE coalesce(m.salience, 0.0) >= $min_salience
  AND ($since IS NULL OR m.observed_at >= $since)
RETURN s.id           AS signal_id,
       m.salience     AS salience,
       m.sentiment    AS sentiment,
       m.mention_text AS mention_text,
       m.observed_at  AS observed_at
ORDER BY salience DESC, observed_at DESC, signal_id ASC
LIMIT $limit
""".strip()
    return Query(
        cypher,
        {
            "tenant_id": tenant_id,
            "entity_id": entity_id,
            "min_salience": float(min_salience),
            "since": None if since is None else _instant(since, name="since"),
            "limit": limit,
        },
    )


def topic_activity(
    *,
    tenant_id: str,
    since: datetime,
    until: datetime | None = None,
    min_salience: float = 0.3,
    limit: int = DEFAULT_LIMIT,
) -> Query:
    """Topic mention volume over a window -- the graph's input to trend detection.

    Aggregated in the database rather than by streaming rows into Python: the
    counts are over the whole mention set, which for a busy topic is orders of
    magnitude larger than the answer.
    """
    _require_tenant(tenant_id)
    _require_limit(limit)
    resolved_since = _instant(since, name="since")
    resolved_until = _instant(until, name="until")
    if resolved_until <= resolved_since:
        raise GraphSchemaError(
            f"window [{resolved_since.isoformat()}, {resolved_until.isoformat()}) "
            "is empty or inverted"
        )

    cypher = f"""
MATCH (s:{SIGNAL_LABEL})-[m:MENTIONS]->(t:Topic {{tenant_id: $tenant_id}})
WHERE m.observed_at >= $since AND m.observed_at < $until
  AND coalesce(m.salience, 0.0) >= $min_salience
RETURN t.id               AS topic_id,
       t.canonical_name   AS topic,
       count(DISTINCT s)  AS mentions,
       avg(m.sentiment)   AS avg_sentiment,
       max(m.observed_at) AS last_mentioned_at
ORDER BY mentions DESC, topic ASC
LIMIT $limit
""".strip()
    return Query(
        cypher,
        {
            "tenant_id": tenant_id,
            "since": resolved_since,
            "until": resolved_until,
            "min_salience": float(min_salience),
            "limit": limit,
        },
    )


def entity_neighbours(
    *,
    tenant_id: str,
    entity_id: str,
    edge_types: Sequence[EdgeType] = TRAVERSABLE_EDGE_TYPES,
    depth: int = 1,
    as_of: datetime | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Query:
    """Relationships around one entity, as subject-predicate-object rows.

    Backs the agents' `neighbours` tool. Returns *edges*, not nodes, because an
    agent reasoning about a company needs "Acme COMPETES_WITH Globex", and a list
    of neighbouring node ids forces a second round trip per neighbour to find out
    how they are connected.

    `type(r)` yields the relationship type as a string. Read from the edge rather
    than assumed from the query, because with `depth > 1` the returned edges are
    a mixture of types and a caller that guessed from the pattern would label
    them all as the first one.

    The as-of predicate applies to every edge on the path. At depth 2 an edge
    that closed last year sitting between two currently-valid ones would
    otherwise let a stale fact into a current-view answer, wearing the confidence
    of the two live edges around it.
    """
    _require_tenant(tenant_id)
    _require_text(entity_id, "entity_id")
    _require_limit(limit)
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= MAX_HOPS:
        raise GraphSchemaError(f"depth must be an int between 1 and {MAX_HOPS}, got {depth!r}")

    relationships = _relationship_pattern(edge_types)
    cypher = f"""
MATCH (seed {{id: $entity_id, tenant_id: $tenant_id}})
MATCH path = (seed)-[:{relationships}*1..{depth}]-(other)
WHERE other.tenant_id = $tenant_id AND other <> seed
WITH DISTINCT relationships(path) AS rels
UNWIND rels AS r
WITH DISTINCT r
WHERE {as_of_cypher("r")}
WITH r, startNode(r) AS s, endNode(r) AS o
WHERE s.tenant_id = $tenant_id AND o.tenant_id = $tenant_id
RETURN s.id             AS subject_id,
       s.canonical_name AS subject_name,
       type(r)          AS predicate,
       o.id             AS object_id,
       o.canonical_name AS object_name,
       r.valid_from     AS valid_from,
       r.valid_to       AS valid_to,
       r.confidence     AS confidence,
       coalesce(r.source_signal_ids, [])[..5] AS supporting_signal_ids
ORDER BY coalesce(r.confidence, 0.0) DESC, subject_id ASC, object_id ASC
LIMIT $limit
""".strip()

    return Query(
        cypher,
        {
            "tenant_id": tenant_id,
            "entity_id": entity_id,
            AS_OF_PARAM: _instant(as_of, name="as_of"),
            "limit": limit,
        },
    )


def paths_between(
    *,
    tenant_id: str,
    source_id: str,
    target_id: str,
    max_hops: int = 3,
    edge_types: Sequence[EdgeType] = TRAVERSABLE_EDGE_TYPES,
    limit: int = 5,
) -> Query:
    """Shortest paths between two entities. "How are these two connected?"

    `allShortestPaths` rather than `MATCH path = (a)-[*..n]-(b)`, and the
    difference is not stylistic. The unbounded form enumerates *every* path up to
    the hop limit, which on a graph with a hub is combinatorial -- and then
    discards all but a handful. `allShortestPaths` stops at the shortest length
    and returns only paths of that length, which is both the cheaper query and
    the more useful answer: a six-hop connection between two companies through a
    Topic node is technically a path and tells a reader nothing.

    `max_hops` is capped harder than `MAX_HOPS` for the same reason. Beyond about
    four hops in an entity graph, everything is connected to everything, and a
    path that long is an artefact of the graph's density rather than a fact about
    the two endpoints.
    """
    _require_tenant(tenant_id)
    _require_text(source_id, "source_id")
    _require_text(target_id, "target_id")
    _require_limit(limit)
    if not isinstance(max_hops, int) or isinstance(max_hops, bool) or not 1 <= max_hops <= 4:
        raise GraphSchemaError(
            f"max_hops must be an int between 1 and 4, got {max_hops!r}. Past four "
            "hops an entity graph connects everything to everything, and the path "
            "describes the graph's density rather than the two endpoints."
        )

    relationships = _relationship_pattern(edge_types)
    cypher = f"""
MATCH (a {{id: $source_id, tenant_id: $tenant_id}})
MATCH (b {{id: $target_id, tenant_id: $tenant_id}})
MATCH path = allShortestPaths((a)-[:{relationships}*1..{max_hops}]-(b))
WHERE ALL(n IN nodes(path) WHERE n.tenant_id = $tenant_id)
RETURN [n IN nodes(path) | n.id]             AS entity_ids,
       [n IN nodes(path) | n.canonical_name] AS entity_names,
       [r IN relationships(path) | type(r)]  AS predicates,
       reduce(acc = 1.0, r IN relationships(path) |
              acc * coalesce(r.confidence, 0.5)) AS confidence,
       length(path)                          AS hops
ORDER BY hops ASC, confidence DESC, entity_ids ASC
LIMIT $limit
""".strip()

    return Query(
        cypher,
        {
            "tenant_id": tenant_id,
            "source_id": source_id,
            "target_id": target_id,
            "limit": limit,
        },
    )


def subgraph_edges(
    *,
    tenant_id: str,
    entity_ids: Sequence[str],
    depth: int = 1,
    edge_types: Sequence[EdgeType] = TRAVERSABLE_EDGE_TYPES,
    limit: int = DEFAULT_LIMIT,
) -> Query:
    """Every relationship within the neighbourhood of a set of entities.

    What a graph canvas renders. Distinct from `entity_neighbours` in that it
    takes a *set* of seeds and returns the induced edge set, so the frontend gets
    one payload it can lay out rather than n payloads it has to stitch and
    deduplicate -- and stitching client-side is where a UI ends up drawing the
    same edge twice with two different confidences.
    """
    _require_tenant(tenant_id)
    _require_limit(limit)
    if not entity_ids:
        raise GraphSchemaError("entity_ids is empty; an unseeded subgraph is a full scan")
    if not isinstance(depth, int) or isinstance(depth, bool) or not 1 <= depth <= MAX_HOPS:
        raise GraphSchemaError(f"depth must be an int between 1 and {MAX_HOPS}, got {depth!r}")

    relationships = _relationship_pattern(edge_types)
    cypher = f"""
MATCH (seed)
WHERE seed.id IN $entity_ids AND seed.tenant_id = $tenant_id
MATCH path = (seed)-[:{relationships}*1..{depth}]-(other)
WHERE other.tenant_id = $tenant_id
WITH DISTINCT relationships(path) AS rels
UNWIND rels AS r
WITH DISTINCT r
WITH r, startNode(r) AS s, endNode(r) AS o
WHERE s.tenant_id = $tenant_id AND o.tenant_id = $tenant_id
RETURN s.id             AS subject_id,
       s.canonical_name AS subject_name,
       type(r)          AS predicate,
       o.id             AS object_id,
       o.canonical_name AS object_name,
       r.valid_from     AS valid_from,
       r.valid_to       AS valid_to,
       r.confidence     AS confidence,
       coalesce(r.source_signal_ids, [])[..5] AS supporting_signal_ids
ORDER BY coalesce(r.confidence, 0.0) DESC, subject_id ASC, object_id ASC
LIMIT $limit
""".strip()

    return Query(
        cypher,
        {"tenant_id": tenant_id, "entity_ids": list(entity_ids), "limit": limit},
    )


def stale_analytics_nodes(
    *,
    tenant_id: str,
    entity_type: EntityType,
    older_than: datetime,
    limit: int = MAX_LIMIT,
) -> Query:
    """Nodes whose analytics properties predate `older_than`, or are absent.

    Drives the incremental path in `graph/analytics/`: recomputing PageRank over
    the whole graph nightly is fine at ten thousand nodes and is not fine at ten
    million. `n.computed_at IS NULL` comes first in the predicate because a node
    that has never been scored is the case that matters most -- it is invisible
    to every ranking until it is.
    """
    _require_tenant(tenant_id)
    _require_limit(limit)
    # `node_spec()` rather than `validate_label(entity_type.value)`: `EntityType`
    # is a `TolerantStrEnum`, so `EntityType("Spaceship")` degrades to `UNKNOWN`
    # rather than raising, and `"Unknown"` is a perfectly well-formed label that
    # no node carries. The query would run, match nothing, and report "analytics
    # are up to date" for a label that was never scored.
    label = validate_label(node_spec(entity_type).label)
    cypher = f"""
MATCH (n:{label} {{tenant_id: $tenant_id}})
WHERE n.computed_at IS NULL OR n.computed_at < $older_than
RETURN n.id AS id, n.computed_at AS computed_at
ORDER BY coalesce(n.computed_at, datetime('1970-01-01T00:00:00Z')) ASC, id ASC
LIMIT $limit
""".strip()
    return Query(
        cypher,
        {
            "tenant_id": tenant_id,
            "older_than": _instant(older_than, name="older_than"),
            "limit": limit,
        },
    )
