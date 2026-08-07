"""Edge types, their legal endpoints, and the bitemporal block every edge carries.

Two decisions live here, and both of them are the difference between a graph that
can be trusted and one that cannot.

**1. An edge may only connect the labels it is declared to connect.**

Nothing in Neo4j objects to `(:Company)-[:COMPETES_WITH]->(:Region)`. It is a
perfectly well-formed relationship; it is simply meaningless. Written once, it is
invisible -- until the Competitor Agent runs a neighbourhood expansion six weeks
later, finds that Acme competes with Belgium, and writes it into a report a human
sends to their board. There is no query that detects this after the fact, because
there is nothing structurally wrong with it: the only place it can be caught is
the moment before it is written. `validate_endpoints()` is that moment, and
`merge_cypher()` cannot even be *built* without going through it, because the
endpoint labels are part of the query text (Cypher has no parameter form for a
label, and matching an endpoint without one is an all-nodes scan).

That is why endpoint validation is not a lint pass bolted onto the writer. It is
structurally unavoidable: to write an edge you must name its labels, and naming
them checks them.

**2. Every edge carries `valid_from` and `valid_to`.**

That is what makes an as-of query possible, and an as-of query is what makes the
graph a record of what was believed *then* rather than a snapshot of what is
believed *now*. `docs/knowledge-graph.md` §5 fixes the semantics:

* Intervals are half-open, `[valid_from, valid_to)`. `valid_to = null` means
  "still true as far as we know". Never a sentinel like `9999-12-31` -- a
  sentinel silently sorts, compares and aggregates as a real date, and the first
  report to average an end date is the one that discovers it.
* `observed_at` is a *different axis*. `valid_from` is when the fact became true
  in the world; `observed_at` is when OmniSense learned it. Conflating them is
  the most common temporal bug in this layer: "acquisitions in the last 30 days"
  means one thing on each axis and the two answers do not overlap.
* Updates close intervals, they never overwrite. Competition that ends gets a
  `valid_to`, not a `DELETE`.

**On `COMPETES_WITH` being stored once.** It is logically symmetric. Storing both
directions doubles write volume and, worse, lets the two copies drift apart when
one side is updated and the other is not -- and a query that traverses undirected
then returns the same pair twice with different confidences. So it is stored once
in canonical orientation (`from_id < to_id` lexicographically, applied by
`orient()`) and every read traverses it undirected. `SAME_AS` gets the same
treatment for the same reason.

Layer note: **L1 library** -- `models/` and the standard library only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from functools import cache
from typing import Any, Final

from graph.schema.nodes import (
    GraphSchemaError,
    PropertyOwner,
    PropertySpec,
    PropertyType,
    entity_labels,
    list_union_expression,
    validate_label,
    validate_property_map,
)
from graph.schema.nodes import prop as _p
from models.enums import EdgeType, EntityType

__all__ = [
    "COMMON_EDGE_PROPERTIES",
    "EDGE_SPECS",
    "SIGNAL_LABEL",
    "Cardinality",
    "Direction",
    "EdgeSpec",
    "allowed_endpoints",
    "edge_key",
    "edge_spec",
    "edge_types",
    "merge_cypher",
    "orient",
    "validate_edge_properties",
    "validate_endpoints",
]

SIGNAL_LABEL: Final[str] = "Signal"
"""The eighth label: a content-free anchor for `MENTIONS` and `COMPLAINS_ABOUT`.

Not an `EntityType`, and deliberately not in `graph/schema/nodes.py`. The
canonical Signal lives in PostgreSQL, R2, Qdrant and OpenSearch; this is a
foreign key so a mention edge has a subject. `docs/data-stores.md` §3.2 is
explicit that it must never grow content, and `graph/ingest/writer.py` enforces
the property whitelist that keeps it that way.
"""

_ENTITY_LABELS: Final[frozenset[str]] = frozenset(entity_labels())
_ALL_LABELS: Final[frozenset[str]] = _ENTITY_LABELS | {SIGNAL_LABEL}


class Direction(StrEnum):
    """Whether an edge's arrow carries meaning."""

    DIRECTED = "directed"
    """`(a)-[:ACQUIRED]->(b)` means a acquired b. Reversing it inverts the fact."""

    SYMMETRIC = "symmetric"
    """The relation holds both ways. Stored once, canonically oriented, traversed
    undirected. See `orient()`."""


class Cardinality(StrEnum):
    """How many edges of this type may exist between endpoints, per validity interval.

    "Per interval" is the whole subtlety. A `Product` has one launcher, but it has
    one launcher *at a time*: a product transferred between companies has two
    `LAUNCHED_BY` edges whose intervals do not overlap, and both are correct.
    Cardinality that ignored time would force the writer to delete history in
    order to record a change, which is exactly what the bitemporal model exists to
    prevent.
    """

    ONE_TO_MANY = "1:N"
    MANY_TO_ONE = "N:1"
    MANY_TO_MANY = "N:N"


# --------------------------------------------------------------------------- #
# The common edge block (docs/knowledge-graph.md §3)
# --------------------------------------------------------------------------- #


COMMON_EDGE_PROPERTIES: Final[tuple[PropertySpec, ...]] = (
    _p(
        "edge_key",
        PropertyType.STRING,
        required=True,
        doc="Deterministic hash of (type, from_id, to_id, valid_from, "
        "evidence_key). The MERGE key: it is what makes a replayed batch a "
        "no-op instead of a second parallel edge.",
    ),
    _p(
        "tenant_id",
        PropertyType.STRING,
        required=True,
        doc="Mirrors both endpoints. Denormalized onto the edge so a tenant "
        "filter does not have to load two nodes to reject one relationship.",
    ),
    _p(
        "valid_from",
        PropertyType.DATETIME,
        required=True,
        doc="Start of validity in the world, inclusive. Required: an edge "
        "without one is invisible to every as-of query, which means it is "
        "invisible to every query that matters.",
    ),
    _p(
        "valid_to",
        PropertyType.DATETIME,
        doc="End of validity, exclusive. Null means still true as far as we "
        "know -- never a sentinel date, which would compare and aggregate as if "
        "it were real.",
    ),
    _p(
        "valid_from_precision",
        PropertyType.STRING,
        values=("day", "month", "quarter", "year"),
        doc="Many extracted dates are month- or quarter-precise. Recorded so an "
        "as-of query at day granularity does not create confidence the source "
        "never had.",
    ),
    _p(
        "observed_at",
        PropertyType.DATETIME,
        required=True,
        doc="Transaction time: when OmniSense learned this. The second temporal "
        "axis, and never interchangeable with valid_from.",
    ),
    _p("confidence", PropertyType.FLOAT, doc="0-1, from the extractor."),
    _p(
        "extractor",
        PropertyType.STRING,
        doc="Model or rule id that produced the edge, e.g. 'claude-sonnet-5' or "
        "'rule:acquisition_regex_v3'. What lets a bad extractor's output be "
        "found and retracted rather than hunted for.",
    ),
    _p(
        "superseded_by",
        PropertyType.STRING,
        doc="edge_key of the edge that contradicted this one and closed its "
        "interval. The audit trail for contradiction resolution "
        "(docs/knowledge-graph.md §5).",
    ),
    _p(
        "source_signal_ids",
        PropertyType.STRING_LIST,
        max_length=50,
        doc="Supporting signal ids, capped at 50. The full list lives in PostgreSQL; "
        "this is the citation shortlist a report renders without a second hop.",
    ),
    _p(
        "evidence_count",
        PropertyType.INTEGER,
        owner=PropertyOwner.INGEST,
        doc="Distinct supporting signals. Accumulated from per-batch deltas, "
        "with the same replay guard as node source_count.",
    ),
    _p(
        "last_batch_id",
        PropertyType.STRING,
        owner=PropertyOwner.INGEST,
        doc="Content hash of the batch that last touched this edge; suppresses "
        "the evidence_count increment on a replay.",
    ),
    _p("created_at", PropertyType.DATETIME, owner=PropertyOwner.INGEST),
    _p("updated_at", PropertyType.DATETIME, owner=PropertyOwner.INGEST),
    _p("schema_version", PropertyType.INTEGER, owner=PropertyOwner.INGEST),
)


@dataclass(frozen=True)
class EdgeSpec:
    """One relationship type: what it connects, which way, and what it carries."""

    edge_type: EdgeType
    direction: Direction
    cardinality: Cardinality
    endpoints: frozenset[tuple[str, str]]
    """Legal `(source_label, target_label)` pairs. Exhaustive, not illustrative."""

    own_properties: tuple[PropertySpec, ...] = ()
    description: str = ""

    properties: tuple[PropertySpec, ...] = field(init=False, repr=False)
    _by_name: Mapping[str, PropertySpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.edge_type is EdgeType.UNKNOWN:
            raise GraphSchemaError(
                "EdgeType.UNKNOWN has no relationship type. Tolerating an "
                "unrecognised type on read is correct; writing one is how a "
                "graph acquires relationships nobody can interpret."
            )
        validate_label(self.edge_type.value)
        if not self.endpoints:
            raise GraphSchemaError(
                f"{self.edge_type.value} declares no legal endpoints, so no edge "
                "of this type could ever be written"
            )
        for source, target in self.endpoints:
            for label in (source, target):
                if label not in _ALL_LABELS:
                    raise GraphSchemaError(
                        f"{self.edge_type.value} allows unknown label {label!r}; "
                        f"known labels are {', '.join(sorted(_ALL_LABELS))}"
                    )
        if self.direction is Direction.SYMMETRIC:
            # A symmetric edge is stored once and traversed undirected, so its
            # endpoint set must be closed under swapping. Otherwise `orient()`
            # can flip a pair into an orientation the spec calls illegal, and a
            # perfectly valid edge is rejected depending on the alphabetical
            # accident of its two ids.
            missing = {(t, s) for s, t in self.endpoints} - self.endpoints
            if missing:
                raise GraphSchemaError(
                    f"{self.edge_type.value} is symmetric but its endpoint set is "
                    f"not closed under swapping; missing {sorted(missing)}"
                )

        merged = (*COMMON_EDGE_PROPERTIES, *self.own_properties)
        by_name: dict[str, PropertySpec] = {}
        for spec in merged:
            if spec.name in by_name:
                raise GraphSchemaError(
                    f"{self.edge_type.value}.{spec.name} is declared twice"
                )
            by_name[spec.name] = spec

        object.__setattr__(self, "properties", merged)
        object.__setattr__(self, "_by_name", by_name)

    # ------------------------------------------------------------ accessors --

    @property
    def type_name(self) -> str:
        """The Cypher relationship type, verbatim: `COMPETES_WITH`."""
        return self.edge_type.value

    @property
    def is_symmetric(self) -> bool:
        return self.direction is Direction.SYMMETRIC

    def allows(self, source_label: str, target_label: str) -> bool:
        """Whether this edge may connect these two labels, in this order."""
        return (source_label, target_label) in self.endpoints

    def source_labels(self) -> tuple[str, ...]:
        return tuple(sorted({s for s, _ in self.endpoints}))

    def target_labels(self) -> tuple[str, ...]:
        return tuple(sorted({t for _, t in self.endpoints}))

    def property_spec(self, name: str) -> PropertySpec:
        try:
            return self._by_name[name]
        except KeyError:
            raise GraphSchemaError(
                f"{self.type_name} has no property {name!r}. Declared: "
                f"{', '.join(sorted(self._by_name))}"
            ) from None

    def properties_owned_by(self, owner: PropertyOwner) -> tuple[PropertySpec, ...]:
        return tuple(spec for spec in self.properties if spec.owner is owner)

    @property
    def caller_properties(self) -> tuple[PropertySpec, ...]:
        return self.properties_owned_by(PropertyOwner.CALLER)


def _pairs(sources: tuple[str, ...], targets: tuple[str, ...]) -> frozenset[tuple[str, str]]:
    """Cartesian product of two label tuples, as an endpoint set."""
    return frozenset((s, t) for s in sources for t in targets)


_ENTITY_TUPLE: Final[tuple[str, ...]] = entity_labels()
_COMPANY = EntityType.COMPANY.value
_PRODUCT = EntityType.PRODUCT.value
_PERSON = EntityType.PERSON.value
_TECHNOLOGY = EntityType.TECHNOLOGY.value
_EVENT = EntityType.EVENT.value


# --------------------------------------------------------------------------- #
# The eight edge types
# --------------------------------------------------------------------------- #

_MENTIONS = EdgeSpec(
    edge_type=EdgeType.MENTIONS,
    direction=Direction.DIRECTED,
    cardinality=Cardinality.MANY_TO_MANY,
    endpoints=_pairs((SIGNAL_LABEL,), _ENTITY_TUPLE),
    description="A signal talks about an entity. Every one of the seven labels "
    "is a legal target, which is what makes an entity walkable back to citable "
    "evidence.",
    own_properties=(
        _p(
            "salience",
            PropertyType.FLOAT,
            doc="0-1. How central the entity is to this signal. The filter that "
            "keeps a passing mention out of a topic roll-up.",
        ),
        _p("sentiment", PropertyType.FLOAT, doc="-1..1, about this entity specifically."),
        _p(
            "char_spans",
            PropertyType.STRING_LIST,
            doc="'start:end' offsets into Signal.content.text, so a citation can "
            "highlight the exact span. Strings rather than a nested list because "
            "Neo4j has no list-of-lists property type.",
        ),
        _p("mention_text", PropertyType.STRING, doc="The literal surface form seen."),
    ),
)

_COMPETES_WITH = EdgeSpec(
    edge_type=EdgeType.COMPETES_WITH,
    direction=Direction.SYMMETRIC,
    cardinality=Cardinality.MANY_TO_MANY,
    endpoints=_pairs((_COMPANY, _PRODUCT), (_COMPANY, _PRODUCT)),
    description="Two companies or products compete. Stored once in canonical "
    "orientation; traversed undirected.",
    own_properties=(
        _p("market", PropertyType.STRING, doc="The market they compete in."),
        _p(
            "basis",
            PropertyType.STRING,
            values=("stated", "inferred", "analyst"),
            doc="How we know. 'stated' (they named each other) is worth far more "
            "in a report than 'inferred' (they co-occur), and a report that "
            "cannot tell them apart overstates its evidence.",
        ),
        _p("strength", PropertyType.FLOAT, doc="0-1."),
    ),
)

_ACQUIRED = EdgeSpec(
    edge_type=EdgeType.ACQUIRED,
    direction=Direction.DIRECTED,
    cardinality=Cardinality.ONE_TO_MANY,
    endpoints=_pairs((_COMPANY,), (_COMPANY, _PRODUCT)),
    description="Acquirer → target. One acquirer, many targets; a target has at "
    "most one active inbound ACQUIRED per interval.",
    own_properties=(
        _p("announced_at", PropertyType.DATETIME),
        _p("closed_at", PropertyType.DATETIME),
        _p(
            "price",
            PropertyType.FLOAT,
            doc="Deal value in `currency`. A float and not an integer of minor "
            "units because reported deal values are approximations to begin with "
            "-- '$1.2B' carries two significant figures, not twelve.",
        ),
        _p("currency", PropertyType.STRING, doc="ISO 4217."),
        _p("stake_pct", PropertyType.FLOAT, doc="0-100. A minority stake is not an "
           "acquisition, and the ownership-chain query filters on it."),
        _p(
            "status",
            PropertyType.STRING,
            values=("rumoured", "announced", "closed", "terminated"),
            doc="The ownership-chain query traverses only 'closed'. Treating a "
            "rumour as ownership is how a report invents a parent company.",
        ),
    ),
)

_USES = EdgeSpec(
    edge_type=EdgeType.USES,
    direction=Direction.DIRECTED,
    cardinality=Cardinality.MANY_TO_MANY,
    endpoints=_pairs((_COMPANY, _PRODUCT), (_TECHNOLOGY,)),
    description="Adopter → technology.",
    own_properties=(
        _p("role", PropertyType.STRING, values=("core", "optional", "evaluating")),
        _p("since", PropertyType.DATETIME),
        _p(
            "depth",
            PropertyType.STRING,
            values=("mentioned", "documented", "verified"),
            doc="Evidence strength. 'mentioned' is one engineer in a thread; "
            "'verified' is a job posting or a public architecture page.",
        ),
    ),
)

_COMPLAINS_ABOUT = EdgeSpec(
    edge_type=EdgeType.COMPLAINS_ABOUT,
    direction=Direction.DIRECTED,
    cardinality=Cardinality.MANY_TO_MANY,
    endpoints=_pairs((_PERSON, SIGNAL_LABEL), (_PRODUCT, _COMPANY, _TECHNOLOGY)),
    description="Complainant → target. A Signal is a legal subject because most "
    "complaints come from an author who cannot be resolved to a real Person: "
    "forcing a Person node for every throwaway Reddit account would pollute the "
    "graph with millions of singletons that never merge with anything.",
    own_properties=(
        _p("severity", PropertyType.FLOAT, doc="1-5, as reported or inferred."),
        _p("sentiment", PropertyType.FLOAT, doc="-1..1."),
        _p("topic_id", PropertyType.STRING, doc="Topic entity id this complaint is about."),
        _p(
            "resolved",
            PropertyType.BOOLEAN,
            doc="Whether the complaint was subsequently addressed. Distinct from "
            "closing the interval: a complaint that was resolved is still a "
            "complaint that happened.",
        ),
        _p("signal_id", PropertyType.STRING, doc="Originating signal, for citation."),
    ),
)

_LAUNCHED_BY = EdgeSpec(
    edge_type=EdgeType.LAUNCHED_BY,
    direction=Direction.DIRECTED,
    cardinality=Cardinality.MANY_TO_ONE,
    endpoints=_pairs((_PRODUCT, _EVENT), (_COMPANY, _PERSON)),
    description="Thing launched → launcher. Points this way so the edge name "
    "reads left to right, even though ingest more naturally produces "
    "company → product.",
    own_properties=(
        _p("announced_at", PropertyType.DATETIME),
        _p("launched_at", PropertyType.DATETIME),
        _p("region_ids", PropertyType.STRING_LIST, doc="Region entity ids of the launch."),
        _p("launch_type", PropertyType.STRING, values=("ga", "beta", "preview")),
    ),
)

_SAME_AS = EdgeSpec(
    edge_type=EdgeType.SAME_AS,
    direction=Direction.SYMMETRIC,
    cardinality=Cardinality.MANY_TO_MANY,
    endpoints=frozenset((label, label) for label in _ENTITY_TUPLE),
    description="Records an entity-resolution merge reversibly. Same label on "
    "both ends by construction: a Company and a Product with the same name stay "
    "distinct, which is a hard rule in the matcher, so an identity edge across "
    "labels would contradict the resolver that created it.",
    own_properties=(
        _p("score", PropertyType.FLOAT, doc="Match score that justified the merge."),
        _p(
            "decided_by",
            PropertyType.STRING,
            doc="'auto' or a reviewer id. An auto-merge and an adjudicated merge "
            "carry different weight when a merge is questioned later.",
        ),
        _p("must_not_link", PropertyType.BOOLEAN, doc="Set by an un-merge, to stop "
           "the next resolution pass from immediately re-merging the pair."),
    ),
)

_DUPLICATE_OF = EdgeSpec(
    edge_type=EdgeType.DUPLICATE_OF,
    direction=Direction.DIRECTED,
    cardinality=Cardinality.MANY_TO_ONE,
    endpoints=frozenset({(SIGNAL_LABEL, SIGNAL_LABEL)}),
    description="Mirrors Signal.lineage.duplicate_of into the graph so cluster "
    "membership is traversable. Directed from the duplicate to the canonical "
    "member, which is the direction that makes 'give me the canonical signal' a "
    "single hop.",
    own_properties=(
        _p("similarity", PropertyType.FLOAT, doc="0-1, from the dedup stage."),
        _p(
            "method",
            PropertyType.STRING,
            values=("exact", "simhash", "embedding"),
            doc="Which dedup mechanism fired. An exact-hash duplicate and a "
            "near-duplicate by embedding are not equally safe to collapse.",
        ),
    ),
)

EDGE_SPECS: Final[Mapping[EdgeType, EdgeSpec]] = {
    spec.edge_type: spec
    for spec in (
        _MENTIONS,
        _COMPETES_WITH,
        _ACQUIRED,
        _USES,
        _COMPLAINS_ABOUT,
        _LAUNCHED_BY,
        _SAME_AS,
        _DUPLICATE_OF,
    )
}
"""The registry, keyed by `EdgeType`, in `EdgeType` declaration order."""


def edge_types() -> tuple[str, ...]:
    """Every writable relationship type, in registry order."""
    return tuple(spec.type_name for spec in EDGE_SPECS.values())


def edge_spec(edge_type: EdgeType) -> EdgeSpec:
    """Return the spec for an edge type, raising on `UNKNOWN` and on anything absent."""
    try:
        return EDGE_SPECS[edge_type]
    except KeyError:
        raise GraphSchemaError(
            f"{edge_type!r} is not a writable relationship type. Known types are "
            f"{', '.join(edge_types())}."
        ) from None


def allowed_endpoints(edge_type: EdgeType) -> tuple[tuple[str, str], ...]:
    """The legal `(source_label, target_label)` pairs, sorted for a stable message."""
    return tuple(sorted(edge_spec(edge_type).endpoints))


def validate_endpoints(edge_type: EdgeType, source_label: str, target_label: str) -> None:
    """Refuse an edge between labels this type does not connect.

    The whole reason this module exists. `COMPETES_WITH` between a `Company` and
    a `Region` is not a Neo4j error, is not a query error, and is not visible in
    any dashboard: it is a nonsense fact that surfaces months later inside a
    report, by which point nobody can say where it came from. The only place it
    is catchable is here.

    The error names the legal pairs rather than merely saying "invalid", because
    the caller is usually an extractor that got the argument order backwards --
    `LAUNCHED_BY` in particular points from the launched thing to the launcher,
    which is the opposite of how ingest naturally produces it.
    """
    spec = edge_spec(edge_type)
    validate_label(source_label)
    validate_label(target_label)
    if spec.allows(source_label, target_label):
        return

    hint = ""
    if spec.allows(target_label, source_label):
        hint = (
            f" -- but ({target_label}, {source_label}) is legal, so the endpoints "
            "are probably reversed"
        )
    legal = ", ".join(f"({s} -> {t})" for s, t in allowed_endpoints(edge_type))
    raise GraphSchemaError(
        f"{spec.type_name} cannot connect ({source_label} -> {target_label}){hint}. "
        f"Legal endpoints: {legal}."
    )


def orient(
    edge_type: EdgeType,
    source_id: str,
    source_label: str,
    target_id: str,
    target_label: str,
) -> tuple[str, str, str, str]:
    """Put a symmetric edge into canonical orientation. Returns the four values.

    Canonical means `source_id < target_id` lexicographically. Any total order
    would do; what matters is that both writers of the same pair choose the same
    one, so a `MERGE` from either direction lands on the same relationship
    instead of creating a mirrored second copy that then drifts.

    Directed edges are returned untouched -- flipping `ACQUIRED` would invert who
    bought whom.

    The label travels with its id. Swapping ids without swapping labels is a
    subtle, extremely destructive bug: the generated query would `MATCH` the
    acquirer's id against the target's label, find nothing, and silently drop the
    edge.
    """
    spec = edge_spec(edge_type)
    if spec.is_symmetric and target_id < source_id:
        return target_id, target_label, source_id, source_label
    return source_id, source_label, target_id, target_label


def edge_key(
    edge_type: EdgeType,
    source_id: str,
    target_id: str,
    valid_from: datetime,
    evidence_key: str = "",
) -> str:
    """Deterministic `MERGE` key for one edge.

    Hashed rather than concatenated because the components are unbounded strings
    that may contain the separator, and a key collision between
    `("a|b", "c")` and `("a", "b|c")` merges two unrelated facts into one edge.

    `valid_from` is part of the key on purpose: the same pair can hold the same
    relationship over two disjoint intervals -- a company that competed, stopped,
    and competes again -- and those are two edges, not one edge whose interval
    was reopened. It is normalised to UTC ISO-8601 with microsecond precision, so
    the same instant expressed in two timezones yields the same key; a naive
    datetime raises rather than silently keying on a local wall clock.

    `evidence_key` splits edges that share endpoints and interval but come from
    genuinely different evidence -- typically the signal id for a `MENTIONS`
    edge, where one signal mentioning an entity twice must not become two edges
    but two signals mentioning it must.
    """
    if valid_from.tzinfo is None:
        raise GraphSchemaError(
            "valid_from is naive; an edge key derived from a local wall clock is "
            "not reproducible across processes in different timezones"
        )
    parts = (
        edge_spec(edge_type).type_name,
        source_id,
        target_id,
        valid_from.astimezone(UTC).isoformat(timespec="microseconds"),
        evidence_key,
    )
    # Unit separator as the join character: it cannot occur in an id, a type name
    # or an ISO timestamp, so no component can forge a boundary. Truncated to 128
    # bits, which is still far past the point where a collision is less likely
    # than the graph being wrong for some other reason, and keeps the property
    # small enough that the index over it stays cheap.
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:32]


def validate_edge_properties(edge_type: EdgeType, properties: Mapping[str, Any]) -> None:
    """Check a caller-supplied edge property map against the type's declaration.

    Same four failure modes as `validate_node_properties`, and one more that is
    specific to edges: an interval whose end precedes its start. Neo4j stores it
    happily, and every as-of query then returns nothing for that edge at every
    point in time -- a fact that exists in the database and can never be read.
    """
    spec = edge_spec(edge_type)
    validate_property_map(spec.properties, spec.type_name, properties)

    valid_from = properties.get("valid_from")
    valid_to = properties.get("valid_to")
    if (
        isinstance(valid_from, datetime)
        and isinstance(valid_to, datetime)
        and valid_to <= valid_from
    ):
        raise GraphSchemaError(
            f"{spec.type_name} interval [{valid_from.isoformat()}, "
            f"{valid_to.isoformat()}) is empty or inverted; intervals are "
            "half-open and an empty one is invisible to every as-of query"
        )


# --------------------------------------------------------------------------- #
# Cypher generation
# --------------------------------------------------------------------------- #


def _set_expression(spec: PropertySpec) -> str:
    if spec.accumulates:
        return list_union_expression(f"r.{spec.name}", f"row.{spec.name}", spec.max_length)
    if spec.required and spec.name != "edge_key":
        return f"row.{spec.name}"
    return f"coalesce(row.{spec.name}, r.{spec.name})"


@cache
def merge_cypher(edge_type: EdgeType, source_label: str, target_label: str) -> str:
    """Return the idempotent, parameterised `MERGE` fragment for one edge shape.

    The label pair is an argument, not an inferred detail, for two reasons that
    reinforce each other:

    * **Correctness.** Building the query is impossible without naming the
      labels, and naming them runs `validate_endpoints()`. There is no path to a
      query for an edge that connects labels it may not connect.
    * **Performance.** Cypher cannot parameterise a label, and
      `MATCH (a {id: $x})` with no label is an all-nodes scan -- there is no
      global id index in Neo4j. On the write path, at 500 rows a batch, that is
      the difference between an index seek and reading the store. This is why the
      writer groups rows by `(edge_type, source_label, target_label)` before
      calling: the label pair is part of the query's identity.

    `tenant_id` is matched on both endpoints rather than only carried on the
    edge. Ids are UUIDv5 over `(tenant_id, label, resolution_key)` so a
    cross-tenant collision should be impossible -- but "should be impossible" and
    "cannot happen" differ by one bug in id derivation, and the blast radius here
    is one tenant's graph silently linked into another's.

    `ON CREATE` sets `valid_from` and `observed_at` and nothing else touches
    them: an interval's start is a fact about when it began, and a later batch
    re-reporting the same edge must not move it. `valid_to` *is* updated on
    match, because closing an interval is exactly how the model records that
    something stopped being true.

    Parameters: `$rows`, `$batch_id`, `$schema_version`. Nothing else reaches the
    query text but the three validated labels.
    """
    spec = edge_spec(edge_type)
    validate_endpoints(edge_type, source_label, target_label)
    source = validate_label(source_label)
    target = validate_label(target_label)
    rel = validate_label(spec.type_name)

    assignments = [
        f"r.{prop.name} = {_set_expression(prop)}"
        for prop in spec.caller_properties
        if prop.name not in ("edge_key", "valid_from", "observed_at")
    ]
    assignments += [
        "r.evidence_count = coalesce(r.evidence_count, 0) + "
        "CASE WHEN replayed THEN 0 ELSE coalesce(row.new_evidence, 0) END",
        "r.updated_at = datetime()",
        "r.schema_version = $schema_version",
        "r.last_batch_id = $batch_id",
    ]
    body = ",\n    ".join(assignments)

    return (
        "UNWIND $rows AS row\n"
        f"MATCH (a:{source} {{id: row.from_id, tenant_id: row.tenant_id}})\n"
        f"MATCH (b:{target} {{id: row.to_id, tenant_id: row.tenant_id}})\n"
        f"MERGE (a)-[r:{rel} {{edge_key: row.edge_key}}]->(b)\n"
        "ON CREATE SET r.created_at = datetime(),\n"
        "              r.evidence_count = 0,\n"
        "              r.valid_from = row.valid_from,\n"
        "              r.observed_at = row.observed_at\n"
        "WITH r, row, (r.last_batch_id = $batch_id) AS replayed\n"
        f"SET {body}\n"
        "RETURN count(r) AS written"
    )
