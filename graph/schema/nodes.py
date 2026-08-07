"""The seven node labels, expressed as data instead of as prose.

Neo4j is schema-optional. Nothing in the database stops a writer from creating
`(:Company {canonical_nam: "Acme"})` -- it simply creates a new property, on
every node it touches, forever. There is no `ALTER TABLE` moment where the typo
is caught, and no query fails; entity search just quietly stops returning Acme.
That is the failure this module exists to prevent: the schema Neo4j does not
enforce is enforced here, at the one point where a property name reaches a query.

So the seven labels of `models.enums.EntityType` are declared as `NodeSpec`
values, and the `MERGE` fragment that writes each one is *generated from that
declaration*. Adding a property means adding a `PropertySpec`; it cannot be
added to the write path and forgotten in the schema, or vice versa, because
there is only one place it exists.

**Three things a `PropertySpec` records that a comment could not.**

`required`
    Whether the graph is broken without it. `normalized_name` is required not
    because it is convenient but because `graph/resolution/blocking.py` blocks on
    it: a node written without one is invisible to resolution, so it never merges
    with anything, and the graph accumulates a permanent duplicate of a company
    that already exists. The absence is silent at write time and expensive six
    weeks later, which is exactly the class of bug worth failing loudly on.

`owner`
    Who is allowed to write it. `pagerank_score` is computed by
    `graph/analytics/centrality.py` on a batch cadence; `source_count` is a
    counter maintained by `graph/ingest/writer.py`. If the ingest `MERGE` set
    every property it knows about, each mention of a company would reset its
    PageRank to null and the dashboard would flicker between "computed last
    night" and "empty" depending on ingest timing. `owner` is what lets the
    generated fragment set the ingest-owned properties and leave the rest alone.

`allowed_values`
    A closed vocabulary where one exists. `lifecycle_state` is one of four
    strings; `"GA"` instead of `"ga"` is not an error anywhere in Neo4j, it is
    simply a fifth value that no query matches.

**On the `Signal` label.** It is not here. `(:Signal {id})` is a stub -- a
foreign key with no properties of its own beyond the handful needed to anchor and
date a `MENTIONS` edge -- and `docs/data-stores.md` §3.2 forbids it from ever
becoming more than that. It is declared in `graph/ingest/writer.py`, next to the
code that enforces the prohibition, rather than sitting in a registry of *entity*
types where the next person to add a property would find it.

Layer note: **L1 library** (`docs/architecture.md` §6.1) -- imports `models/` and
the standard library, nothing else. That is why the error below is a plain
`ValueError` subclass rather than a `backend.core.exceptions` type: `graph/` may
not import the kernel, and `services/graph_service.py` is the layer that
translates a schema violation into an HTTP-shaped failure.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from functools import cache
from typing import Any, Final

from models.enums import EntityType

__all__ = [
    "COMMON_PROPERTIES",
    "MAX_LIST_PROPERTY_LENGTH",
    "NODE_SPECS",
    "SCHEMA_VERSION",
    "GraphSchemaError",
    "NodeSpec",
    "PropertyOwner",
    "PropertySpec",
    "PropertyType",
    "entity_labels",
    "list_union_expression",
    "merge_cypher",
    "node_spec",
    "prop",
    "validate_label",
    "validate_node_properties",
    "validate_property_map",
]


class GraphSchemaError(ValueError):
    """A write or a query violates the declared graph schema.

    A `ValueError` subclass because that is what it is -- a bad argument, caught
    before it reaches the driver. `graph/` is an L1 library and may not import
    `backend/core/exceptions.py`; `services/graph_service.py` catches this and
    re-raises the kernel's `ValidationError` when it needs to become a 422.
    """


SCHEMA_VERSION: Final[int] = 1
"""The `graph/schema/versions/` version these fragments correspond to.

Stamped onto every node written, so a node can be traced back to the schema that
last touched it. When `v002` changes a property's meaning, the backfill job needs
to know which nodes predate the change, and asking the data is the only way --
Neo4j keeps no history of its own.
"""

MAX_LIST_PROPERTY_LENGTH: Final[int] = 100
"""Cap on every list-valued property.

Neo4j stores a list property inline with the node. An `aliases` list that grows
without bound -- and it will, because every misspelling in every scraped headline
is a candidate alias -- turns a node into a multi-megabyte record that has to be
loaded in full to read its `canonical_name`. Capping loses the tail of a very
long alias list, which is the correct trade: the tail is noise, and the
authoritative list lives in PostgreSQL (`docs/data-stores.md` §3.2).
"""

# A Neo4j label is unquoted in Cypher unless backticked, and Cypher has no
# parameter form for one. Every label that reaches a query therefore passes
# `validate_label()` first. The pattern is deliberately narrower than what Neo4j
# accepts: these labels come from a closed enum, so anything outside
# `[A-Za-z][A-Za-z0-9_]*` means a caller built a label out of data.
_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

# Same reasoning for property names, which are also unparameterisable in Cypher.
_PROPERTY_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def validate_label(label: str) -> str:
    """Assert a label is safe to interpolate into Cypher, and return it.

    Cypher has no parameter form for a label -- `MATCH (n:$label)` is a syntax
    error -- so a label is the one value in this package that reaches query
    *text*. It is safe only because it comes from a closed enum and passes
    through here. If a caller ever builds a label out of a scraped name, this is
    what stops `Company) DETACH DELETE n //` from becoming a query.

    Defined this high in the module because `NodeSpec.__post_init__` calls it,
    and the seven specs are constructed at import time.
    """
    if not _LABEL_PATTERN.match(label):
        raise GraphSchemaError(
            f"{label!r} is not a valid Neo4j label. Labels are interpolated into "
            "Cypher and must come from models.enums.EntityType, never from data."
        )
    return label


class PropertyType(StrEnum):
    """The Neo4j property types this schema uses.

    Deliberately short of Neo4j's full set, and what is *missing* is the point:
    Neo4j property values may only be primitives or homogeneous arrays of
    primitives, so there is no `MAP` member. `docs/knowledge-graph.md` §2
    describes `Person.handles` as a "map of platform → handle", which the
    database cannot store -- writing one raises `Property values can only be of
    primitive types or arrays thereof`. It is modelled below as a list of
    `"platform:handle"` strings instead.
    """

    STRING = "string"
    STRING_LIST = "string[]"
    INTEGER = "integer"
    FLOAT = "float"
    FLOAT_LIST = "float[]"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    POINT = "point"

    @property
    def is_list(self) -> bool:
        """Whether values of this type accumulate rather than replace."""
        return self in (PropertyType.STRING_LIST, PropertyType.FLOAT_LIST)


class PropertyOwner(StrEnum):
    """Which subsystem is allowed to write a property.

    Not a permission system -- there is no enforcement inside Neo4j -- but the
    input to two decisions that *are* enforced here: which properties a caller
    may supply on a write, and which properties the generated `MERGE` fragment
    sets.
    """

    CALLER = "caller"
    """Supplied per write by whoever is ingesting. Extracted facts."""

    INGEST = "ingest"
    """Maintained by `graph/ingest/writer.py`: counters, timestamps, bookkeeping.

    A caller that supplies one of these is rejected. Letting a caller set
    `source_count` directly would break replay safety -- the counter is a
    monotonic accumulation of per-batch deltas, and an absolute value written on
    top of it silently discards every other writer's contribution.
    """

    ANALYTICS = "analytics"
    """Written by `graph/analytics/` batch jobs, never by the ingest path.

    The ingest fragment must not name these. A `MERGE` that set them would reset
    them to null on every mention, because an ingest row has no opinion about
    PageRank.
    """


@dataclass(frozen=True, slots=True)
class PropertySpec:
    """One property on one node label."""

    name: str
    type: PropertyType
    required: bool = False
    owner: PropertyOwner = PropertyOwner.CALLER
    description: str = ""
    allowed_values: frozenset[str] | None = None

    max_length: int = MAX_LIST_PROPERTY_LENGTH
    """Cap applied to an accumulating list. Ignored for non-list properties.

    Per-property rather than global because the right cap depends on what the
    tail is worth: `docs/knowledge-graph.md` §3 caps `source_signal_ids` at 50,
    since it is a citation shortlist and a report renders at most a handful,
    while `aliases` earns a longer tail because resolution matches against all
    of it.
    """

    accumulates: bool | None = None
    """Whether a write *unions* into the existing list instead of replacing it.

    `None` derives it: a `string[]` accumulates, anything else replaces. The
    derivation is right for `aliases` and `merged_from` -- knowledge about an
    entity is additive and a row that mentions one alias must not erase four --
    and catastrophically wrong for `embedding`, which is a vector. Unioning two
    768-dimensional embeddings would concatenate them, deduplicate whichever
    components happened to be bit-identical, and truncate the result to the list
    cap: a silently corrupt vector that still passes every type check and quietly
    ruins resolution. `embedding` therefore sets this to `False` explicitly.
    """

    def __post_init__(self) -> None:
        if self.accumulates is None:
            object.__setattr__(self, "accumulates", self.type is PropertyType.STRING_LIST)
        elif self.accumulates and not self.type.is_list:
            raise GraphSchemaError(
                f"{self.name!r} is declared accumulating but is {self.type.value}; "
                "only a list property can be unioned"
            )
        if not _PROPERTY_PATTERN.match(self.name):
            raise GraphSchemaError(
                f"property name {self.name!r} is not a safe Cypher identifier; "
                "property names are interpolated into query text and must come "
                "from this module, never from data"
            )
        if self.required and self.owner is not PropertyOwner.CALLER:
            # An INGEST- or ANALYTICS-owned property cannot be "required" of a
            # caller who is forbidden from supplying it. Catching the
            # contradiction here beats discovering it as an unsatisfiable
            # validation error on every single write.
            raise GraphSchemaError(
                f"{self.name!r} is required but owned by {self.owner.value}; "
                "only caller-owned properties can be required of a caller"
            )
        if self.allowed_values is not None and self.type is not PropertyType.STRING:
            raise GraphSchemaError(
                f"{self.name!r} declares allowed_values but is {self.type.value}; "
                "a closed vocabulary only makes sense for a string property"
            )


def prop(
    name: str,
    type_: PropertyType,
    *,
    required: bool = False,
    owner: PropertyOwner = PropertyOwner.CALLER,
    values: tuple[str, ...] | None = None,
    accumulates: bool | None = None,
    max_length: int = MAX_LIST_PROPERTY_LENGTH,
    doc: str = "",
) -> PropertySpec:
    """Terse constructor. The property tables are long enough without keywords.

    Shared with `graph/schema/edges.py`, which declares its common block the same
    way -- a second copy of this would be a second place for a default to drift.
    """
    return PropertySpec(
        name=name,
        type=type_,
        required=required,
        owner=owner,
        description=doc,
        allowed_values=None if values is None else frozenset(values),
        accumulates=accumulates,
        max_length=max_length,
    )


_p = prop  # local shorthand for the tables below


# --------------------------------------------------------------------------- #
# The common block -- every one of the seven labels carries all of it
# --------------------------------------------------------------------------- #

COMMON_PROPERTIES: Final[tuple[PropertySpec, ...]] = (
    _p(
        "id",
        PropertyType.STRING,
        required=True,
        doc="Canonical entity id, UNIQUE per label. UUIDv5 over "
        "(tenant_id, label, resolution_key) so a pre-resolution write is "
        "deterministic and a replay lands on the same node.",
    ),
    _p(
        "tenant_id",
        PropertyType.STRING,
        required=True,
        doc="Multi-tenancy is Phase 7, but the property exists from v001. "
        "Retrofitting it onto a populated graph means a backfill across every "
        "node and every edge with no way to know which tenant owned what.",
    ),
    _p(
        "canonical_name",
        PropertyType.STRING,
        required=True,
        doc="Preferred display name. Indexed on Company, Product and Topic.",
    ),
    _p(
        "normalized_name",
        PropertyType.STRING,
        required=True,
        doc="Case-folded, unaccented, legal-suffix-stripped. The blocking key "
        "input for graph/resolution/blocking.py -- a node without one never "
        "becomes a merge candidate and duplicates itself forever.",
    ),
    _p(
        "aliases",
        PropertyType.STRING_LIST,
        doc="Absorbed surface forms. Part of the entity_search fulltext index, "
        "and the reason a query for 'Big Blue' finds IBM.",
    ),
    _p(
        "description",
        PropertyType.STRING,
        doc="One line, LLM-generated. Part of the fulltext index.",
    ),
    _p(
        "embedding",
        PropertyType.FLOAT_LIST,
        accumulates=False,
        doc="Name-plus-context vector used by graph/resolution/matcher.py. "
        "Dimension follows EMBEDDING_DIMENSIONS. This is the one vector the "
        "graph holds, and it is a resolution input rather than retrievable "
        "content -- retrieval vectors live in Qdrant.",
    ),
    _p(
        "confidence",
        PropertyType.FLOAT,
        doc="0-1. Aggregate confidence that this entity is real and correctly "
        "resolved.",
    ),
    _p(
        "merged_from",
        PropertyType.STRING_LIST,
        doc="Entity ids absorbed into this node by resolution. Required for "
        "un-merge: without it a bad merge is unrecoverable, and resolution "
        "errors are corrected by un-merging rather than by re-ingesting.",
    ),
    _p(
        "source_count",
        PropertyType.INTEGER,
        owner=PropertyOwner.INGEST,
        doc="Distinct signals that evidenced this entity. The cheapest "
        "anti-hallucination signal a report has. Accumulated from per-batch "
        "deltas, never recomputed.",
    ),
    _p(
        "first_seen",
        PropertyType.DATETIME,
        owner=PropertyOwner.INGEST,
        doc="Earliest observation. Takes the minimum on every write, so a "
        "backfill of older signals moves it backwards rather than being lost.",
    ),
    _p(
        "last_seen",
        PropertyType.DATETIME,
        owner=PropertyOwner.INGEST,
        doc="Latest observation. Takes the maximum, so an out-of-order replay "
        "cannot move it backwards.",
    ),
    _p(
        "created_at",
        PropertyType.DATETIME,
        owner=PropertyOwner.INGEST,
        doc="Transaction time of node creation, from the server clock. Distinct "
        "from first_seen, which is world time.",
    ),
    _p(
        "updated_at",
        PropertyType.DATETIME,
        owner=PropertyOwner.INGEST,
        doc="Transaction time of the last write, from the server clock.",
    ),
    _p(
        "schema_version",
        PropertyType.INTEGER,
        owner=PropertyOwner.INGEST,
        doc="The graph/schema/versions/ version that last wrote this node.",
    ),
    _p(
        "last_batch_id",
        PropertyType.STRING,
        owner=PropertyOwner.INGEST,
        doc="Content hash of the batch that last touched this node. The guard "
        "that makes the source_count increment idempotent under both a driver "
        "transaction retry and an at-least-once Kafka replay.",
    ),
    _p(
        "pagerank_score",
        PropertyType.FLOAT,
        owner=PropertyOwner.ANALYTICS,
        doc="Written by graph/analytics/centrality.py on a batch cadence.",
    ),
    _p(
        "community_id",
        PropertyType.STRING,
        owner=PropertyOwner.ANALYTICS,
        doc="Written by graph/analytics/communities.py.",
    ),
    _p(
        "computed_at",
        PropertyType.DATETIME,
        owner=PropertyOwner.ANALYTICS,
        doc="When the analytics properties above were last recomputed. Without "
        "it a stale PageRank is indistinguishable from a fresh one.",
    ),
)


@dataclass(frozen=True)
class NodeSpec:
    """One node label: its properties, its indexes and its write fragment."""

    entity_type: EntityType
    own_properties: tuple[PropertySpec, ...] = ()
    indexed: tuple[str, ...] = ()
    """Property names carrying a plain (range) index for this label.

    Not every label gets one. `docker/local/neo4j/01-constraints.cypher` indexes
    `canonical_name` on `Company`, `Product` and `Topic` only, and this mirrors
    it exactly rather than improving on it -- an index this module declares and
    the bootstrap file does not is an index that exists in tests and in nobody's
    database. Adding the missing four is a `v002` change, because an applied
    version is never edited in place.
    """

    description: str = ""

    properties: tuple[PropertySpec, ...] = field(init=False, repr=False)
    _by_name: Mapping[str, PropertySpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.entity_type is EntityType.UNKNOWN:
            raise GraphSchemaError(
                "EntityType.UNKNOWN has no node label. It exists so a reader "
                "running older code tolerates a type a newer producer wrote; "
                "writing an (:Unknown) node instead of failing would turn a "
                "version skew into permanent graph garbage."
            )
        validate_label(self.entity_type.value)

        merged = (*COMMON_PROPERTIES, *self.own_properties)
        by_name: dict[str, PropertySpec] = {}
        for spec in merged:
            if spec.name in by_name:
                raise GraphSchemaError(
                    f"{self.entity_type.value}.{spec.name} is declared twice; a "
                    "label-specific property must not shadow a common one"
                )
            by_name[spec.name] = spec
        for name in self.indexed:
            if name not in by_name:
                raise GraphSchemaError(
                    f"{self.entity_type.value} indexes {name!r}, which it does not declare"
                )

        object.__setattr__(self, "properties", merged)
        object.__setattr__(self, "_by_name", by_name)

    # ------------------------------------------------------------ accessors --

    @property
    def label(self) -> str:
        """The Neo4j label, verbatim. `EntityType` values are already capitalized."""
        return self.entity_type.value

    def property_spec(self, name: str) -> PropertySpec:
        """Look up one property, raising rather than returning None.

        A caller asking for a property that does not exist has a bug; handing
        back `None` propagates it into a query where it becomes a null.
        """
        try:
            return self._by_name[name]
        except KeyError:
            raise GraphSchemaError(
                f"{self.label} has no property {name!r}. Declared: "
                f"{', '.join(sorted(self._by_name))}"
            ) from None

    def has_property(self, name: str) -> bool:
        return name in self._by_name

    def properties_owned_by(self, owner: PropertyOwner) -> tuple[PropertySpec, ...]:
        return tuple(spec for spec in self.properties if spec.owner is owner)

    @property
    def required_properties(self) -> tuple[PropertySpec, ...]:
        return tuple(spec for spec in self.properties if spec.required)

    @property
    def caller_properties(self) -> tuple[PropertySpec, ...]:
        return self.properties_owned_by(PropertyOwner.CALLER)


# --------------------------------------------------------------------------- #
# The seven labels (Design Doc §7, docs/knowledge-graph.md §2)
# --------------------------------------------------------------------------- #

_COMPANY = NodeSpec(
    entity_type=EntityType.COMPANY,
    description="A legal entity. The subject of COMPETES_WITH and ACQUIRED.",
    indexed=("canonical_name",),
    own_properties=(
        _p("legal_name", PropertyType.STRING, doc="Registered name, suffix intact."),
        _p(
            "ticker",
            PropertyType.STRING,
            doc="Exchange ticker. A hard identifier in resolution: matching "
            "tickers force a merge and conflicting ones force a non-merge, "
            "which is why it is worth storing even though it is usually absent.",
        ),
        _p("exchange", PropertyType.STRING, doc="Listing venue, e.g. NASDAQ."),
        _p(
            "domain",
            PropertyType.STRING,
            doc="Primary web domain. The other hard identifier for resolution.",
        ),
        _p("founded_year", PropertyType.INTEGER),
        _p(
            "employee_band",
            PropertyType.STRING,
            values=("1-10", "11-50", "51-200", "201-1000", "1001-10000", "10000+"),
            doc="Banded rather than exact: headcount is stale the day it is "
            "scraped, and a band stays true for a year.",
        ),
        _p(
            "status",
            PropertyType.STRING,
            values=("active", "acquired", "defunct"),
            doc="Not redundant with an inbound ACQUIRED edge: the edge records "
            "the transaction, this records the present state, and a company can "
            "be acquired and still trade under its own name.",
        ),
        _p("industry_codes", PropertyType.STRING_LIST, doc="SIC / NAICS / GICS codes."),
    ),
)

_PRODUCT = NodeSpec(
    entity_type=EntityType.PRODUCT,
    description="A named offering. LAUNCHED_BY a Company, COMPETES_WITH another Product.",
    indexed=("canonical_name",),
    own_properties=(
        _p("category", PropertyType.STRING),
        _p("version", PropertyType.STRING, doc="Free text: '14 Pro', 'v2.1', '2024.3'."),
        _p(
            "lifecycle_state",
            PropertyType.STRING,
            values=("announced", "ga", "deprecated", "discontinued"),
            doc="Drives whether a competitive comparison is still meaningful. A "
            "discontinued product still COMPETES_WITH historically, which is why "
            "the edge is not deleted when this changes.",
        ),
        _p("announced_at", PropertyType.DATETIME),
        _p("released_at", PropertyType.DATETIME),
        _p("pricing_model", PropertyType.STRING, doc="'subscription', 'usage', 'perpetual'."),
    ),
)

_PERSON = NodeSpec(
    entity_type=EntityType.PERSON,
    description="A named individual. Subject of COMPLAINS_ABOUT, target of LAUNCHED_BY.",
    own_properties=(
        _p("given_name", PropertyType.STRING),
        _p("family_name", PropertyType.STRING),
        _p("role_title", PropertyType.STRING),
        _p(
            "handles",
            PropertyType.STRING_LIST,
            doc="Platform handles as 'platform:handle' strings. A list and not a "
            "map because Neo4j property values may only be primitives or arrays "
            "of primitives -- writing a map raises at the driver.",
        ),
        _p(
            "is_public_figure",
            PropertyType.BOOLEAN,
            doc="Gates what may be stored and surfaced about the person "
            "(docs/security-and-privacy.md). False is the safe default, which is "
            "why this is load-bearing rather than merely descriptive.",
        ),
    ),
)

_TOPIC = NodeSpec(
    entity_type=EntityType.TOPIC,
    description="A theme signals cluster around. The unit of trend analysis.",
    indexed=("canonical_name",),
    own_properties=(
        _p("slug", PropertyType.STRING, doc="Stable url-safe key, e.g. 'battery-life'."),
        _p(
            "parent_topic_id",
            PropertyType.STRING,
            doc="Id of the broader topic. A property rather than an edge: the "
            "hierarchy is read on every topic render and a one-hop traversal for "
            "it would be pure overhead.",
        ),
        _p("keywords", PropertyType.STRING_LIST),
        _p(
            "is_emergent",
            PropertyType.BOOLEAN,
            doc="Set when the trend service first flags the topic as rising. "
            "What makes 'what is new' answerable without recomputing trends.",
        ),
        _p("first_trended_at", PropertyType.DATETIME),
    ),
)

_TECHNOLOGY = NodeSpec(
    entity_type=EntityType.TECHNOLOGY,
    description="A language, framework, protocol, model or piece of infrastructure.",
    own_properties=(
        _p(
            "category",
            PropertyType.STRING,
            values=("language", "framework", "protocol", "model", "infra"),
        ),
        _p(
            "maturity",
            PropertyType.STRING,
            values=("experimental", "emerging", "mainstream", "legacy"),
        ),
        _p(
            "vendor_company_id",
            PropertyType.STRING,
            doc="The Company that owns it, when one does. Not every technology "
            "has a vendor, which is why this is a nullable property rather than "
            "a required edge.",
        ),
    ),
)

_REGION = NodeSpec(
    entity_type=EntityType.REGION,
    description="A geographic area, at country / state / metro / city granularity.",
    own_properties=(
        _p("iso_code", PropertyType.STRING, doc="ISO 3166-1 or -2 where one exists."),
        _p("level", PropertyType.STRING, values=("country", "state", "metro", "city")),
        _p("parent_region_id", PropertyType.STRING),
        _p(
            "centroid",
            PropertyType.POINT,
            doc="Neo4j point. Stored so a 'near' query does not need a second "
            "store; a point is one of the few non-scalar values Neo4j accepts.",
        ),
    ),
)

_EVENT = NodeSpec(
    entity_type=EntityType.EVENT,
    description="A dated occurrence: a launch, a funding round, an outage, a lawsuit.",
    own_properties=(
        _p(
            "event_type",
            PropertyType.STRING,
            values=("launch", "funding", "outage", "lawsuit", "conference", "layoff", "other"),
        ),
        _p("occurred_at", PropertyType.DATETIME),
        _p(
            "occurred_at_precision",
            PropertyType.STRING,
            values=("day", "month", "quarter", "year"),
            doc="Most extracted dates are month- or quarter-precise. Storing a "
            "quarter-precise date as a day and querying it as one manufactures "
            "confidence that was never in the source.",
        ),
        _p("region_id", PropertyType.STRING),
        _p("severity", PropertyType.FLOAT, doc="0-1, comparable within an event_type."),
    ),
)

NODE_SPECS: Final[Mapping[EntityType, NodeSpec]] = {
    spec.entity_type: spec
    for spec in (_COMPANY, _PRODUCT, _PERSON, _TOPIC, _TECHNOLOGY, _REGION, _EVENT)
}
"""The registry, keyed by `EntityType`.

Insertion order follows `EntityType` declaration order, and two things depend on
it: the fulltext index lists its labels in this order, and
`docker/local/neo4j/01-constraints.cypher` was written in it. A comprehension
over an explicit tuple keeps that order visible rather than implicit.
"""


def entity_labels() -> tuple[str, ...]:
    """Every entity label, in registry order. Excludes `Signal`, which is a stub."""
    return tuple(spec.label for spec in NODE_SPECS.values())


def node_spec(entity_type: EntityType) -> NodeSpec:
    """Return the spec for a label, raising on `UNKNOWN` and on anything absent.

    `EntityType` is a `TolerantStrEnum`, so `EntityType("Spaceship")` yields
    `UNKNOWN` rather than raising -- that tolerance is right for a *reader* and
    wrong for a writer. This is the boundary where it stops.
    """
    try:
        return NODE_SPECS[entity_type]
    except KeyError:
        raise GraphSchemaError(
            f"{entity_type!r} is not a writable node label. The seven labels are "
            f"{', '.join(entity_labels())}."
        ) from None


# --------------------------------------------------------------------------- #
# Write-time value validation
# --------------------------------------------------------------------------- #


def _type_matches(spec: PropertySpec, value: Any) -> bool:
    """Whether `value` is storable as `spec.type`.

    `bool` is checked before `int` throughout, because `bool` is a subclass of
    `int` in Python: without the guard `source_count=True` is a valid integer and
    lands in the graph as `1`.
    """
    match spec.type:
        case PropertyType.STRING:
            return isinstance(value, str)
        case PropertyType.BOOLEAN:
            return isinstance(value, bool)
        case PropertyType.INTEGER:
            return isinstance(value, int) and not isinstance(value, bool)
        case PropertyType.FLOAT:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        case PropertyType.DATETIME:
            return isinstance(value, datetime)
        case PropertyType.STRING_LIST:
            return isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)
        case PropertyType.FLOAT_LIST:
            return isinstance(value, (list, tuple)) and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
            )
        case PropertyType.POINT:
            # The driver accepts a `neo4j.spatial.Point` or a map with the right
            # keys. Checking the keys rather than importing the driver type keeps
            # `graph/` free of a driver import it does not otherwise need.
            return isinstance(value, Mapping) and (
                {"x", "y"} <= set(value) or {"latitude", "longitude"} <= set(value)
            )
    return False


def validate_property_map(
    specs: tuple[PropertySpec, ...],
    owner_name: str,
    properties: Mapping[str, Any],
) -> None:
    """Check a caller-supplied property map against a set of declarations.

    Four failures, each silent in Neo4j and expensive later:

    1. **Unknown property.** A typo becomes a new property on every node the
       writer touches. Nothing errors and nothing matches.
    2. **Wrong type.** Neo4j will happily store `"2024"` where a datetime is
       expected; the as-of predicate then compares a string to a datetime and
       returns nothing, for every query, forever.
    3. **Value outside a closed vocabulary.** `"GA"` is not `"ga"`.
    4. **A caller writing an ingest- or analytics-owned property.** Setting
       `source_count` directly discards every other writer's accumulation;
       setting `pagerank_score` fabricates an analytics result.

    Required properties are checked here too, so a `MERGE` never creates a node
    that resolution cannot see.

    Takes the specs rather than an `EntityType` because `graph/schema/edges.py`
    needs the identical seven checks over a set of specs it resolves from an
    `EdgeType`. `owner_name` appears in every message and is the only thing that
    tells the reader whether they are looking at a node or a relationship.
    """
    unknown = sorted(set(properties) - {p.name for p in specs})
    if unknown:
        raise GraphSchemaError(
            f"{owner_name} has no propert{'y' if len(unknown) == 1 else 'ies'} "
            f"{', '.join(repr(u) for u in unknown)}. Neo4j would create them "
            "silently rather than reject the write."
        )

    for spec in specs:
        if spec.name not in properties:
            if spec.required:
                raise GraphSchemaError(
                    f"{owner_name}.{spec.name} is required: {spec.description}"
                )
            continue

        if spec.owner is not PropertyOwner.CALLER:
            raise GraphSchemaError(
                f"{owner_name}.{spec.name} is owned by {spec.owner.value} and may "
                "not be supplied by a caller"
            )

        value = properties[spec.name]
        if value is None:
            # An explicit null is how a caller says "I have no value", and
            # `coalesce()` in the fragment keeps whatever is already there. It is
            # not, however, a way to satisfy a required property.
            if spec.required:
                raise GraphSchemaError(f"{owner_name}.{spec.name} is required and cannot be null")
            continue

        if not _type_matches(spec, value):
            raise GraphSchemaError(
                f"{owner_name}.{spec.name} is declared {spec.type.value} but got "
                f"{type(value).__name__}"
            )
        if spec.allowed_values is not None and value not in spec.allowed_values:
            raise GraphSchemaError(
                f"{owner_name}.{spec.name}={value!r} is outside its vocabulary "
                f"({', '.join(sorted(spec.allowed_values))})"
            )


def validate_node_properties(
    entity_type: EntityType,
    properties: Mapping[str, Any],
) -> None:
    """Check a caller-supplied property map against one label's declaration."""
    spec = node_spec(entity_type)
    validate_property_map(spec.properties, spec.label, properties)


# --------------------------------------------------------------------------- #
# Cypher generation
# --------------------------------------------------------------------------- #


def list_union_expression(target: str, source: str, cap: int = MAX_LIST_PROPERTY_LENGTH) -> str:
    """A deduplicating, capped union of two lists, in pure Cypher.

    `apoc.coll.toSet()` says this in one call and is what
    `docs/knowledge-graph.md` §7 sketches, but APOC is a *plugin*: it is loaded
    by `docker-compose.yml` and may not be present in a given deployment. A write
    path that fails when a plugin is missing fails at three in the morning, on
    the cluster that is hardest to reach. `reduce` is O(n²) over lists this
    module caps at 100 entries, which is nothing next to the round trip.

    Shared with `graph/schema/edges.py`, which needs the same union for
    `source_signal_ids`.
    """
    return (
        f"reduce(acc = [], x IN coalesce({target}, []) + coalesce({source}, []) | "
        f"CASE WHEN x IN acc THEN acc ELSE acc + x END)[..{cap}]"
    )


def _set_expression(spec: PropertySpec) -> str:
    """The right-hand side that writes one property inside the `MERGE` fragment."""
    if spec.accumulates:
        return list_union_expression(f"n.{spec.name}", f"row.{spec.name}", spec.max_length)
    if spec.required:
        # Validation guarantees presence, so a `coalesce` here would only hide a
        # writer bug behind a stale value.
        return f"row.{spec.name}"
    # Everything else: a row that says nothing about a property must not erase
    # it. This is the difference between an update and an overwrite, and getting
    # it wrong means a mention that knows only a company's name blanks its
    # ticker, its domain and its description.
    return f"coalesce(row.{spec.name}, n.{spec.name})"


@cache
def merge_cypher(entity_type: EntityType) -> str:
    """Return the idempotent, parameterised `MERGE` fragment for one label.

    Shape, and the reasoning behind each line:

    * `UNWIND $rows` -- one round trip per batch per label, not per node. The
      difference between 500 statements and one is roughly two orders of
      magnitude of wall time.
    * `MERGE (n:Label {id: row.id})` -- keyed on `id` alone, the only property
      under a uniqueness constraint. Merging on a fuller property map
      (`{id: …, canonical_name: …}`) creates a *second* node the moment a name
      changes, which is the most common way a Neo4j graph acquires duplicates.
      Without the constraint two concurrent writers would both create; the
      constraint is what makes `MERGE` a lock.
    * `WITH … AS replayed` -- captures whether this batch already touched this
      node *before* any `SET` runs. `SET` items apply in order, so computing the
      guard inline and then overwriting `last_batch_id` in the same clause would
      work today and break the first time someone reordered the list.
    * The counter is a per-batch *delta*, never a recomputation, and the delta is
      suppressed on replay. `backend/db/neo4j.py` warns that `SET n.c = n.c + 1`
      must not be routed through a managed transaction because a retry applies it
      twice; the `replayed` guard is what makes it safe to route it anyway, and
      it covers Kafka's at-least-once redelivery in the same stroke because
      `batch_id` is a content hash rather than a random id.
    * Analytics-owned properties are absent from the fragment on purpose. See
      `PropertyOwner.ANALYTICS`.

    Cached because the text must be byte-identical on every call: Neo4j caches a
    query plan keyed by query text, so rebuilding the string is wasted work and,
    if it ever varied, a plan-cache miss on every write.

    Parameters the caller must supply: `$rows` (a list of maps), `$batch_id`,
    `$schema_version`. No value is ever interpolated; only the label is, and it
    passes `validate_label()` on the way in.
    """
    spec = node_spec(entity_type)
    label = validate_label(spec.label)

    assignments = [
        f"n.{prop.name} = {_set_expression(prop)}"
        for prop in spec.caller_properties
        if prop.name != "id"  # part of the MERGE key; re-setting it is a no-op
    ]
    assignments += [
        "n.source_count = coalesce(n.source_count, 0) + "
        "CASE WHEN replayed THEN 0 ELSE coalesce(row.new_signal_count, 0) END",
        "n.first_seen = CASE WHEN n.first_seen IS NULL OR row.observed_at < n.first_seen "
        "THEN row.observed_at ELSE n.first_seen END",
        "n.last_seen = CASE WHEN n.last_seen IS NULL OR row.observed_at > n.last_seen "
        "THEN row.observed_at ELSE n.last_seen END",
        "n.updated_at = datetime()",
        "n.schema_version = $schema_version",
        "n.last_batch_id = $batch_id",
    ]
    body = ",\n    ".join(assignments)

    return (
        "UNWIND $rows AS row\n"
        f"MERGE (n:{label} {{id: row.id}})\n"
        "ON CREATE SET n.created_at = datetime(),\n"
        "              n.source_count = 0\n"
        "WITH n, row, (n.last_batch_id = $batch_id) AS replayed\n"
        f"SET {body}\n"
        "RETURN count(n) AS written"
    )
