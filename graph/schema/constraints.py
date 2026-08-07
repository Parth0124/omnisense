"""Uniqueness constraints and indexes -- the only schema Neo4j actually enforces.

Everything else in `graph/schema/` is a promise this repository keeps to itself.
These statements are different: the server enforces them, and two of them are
load-bearing for correctness rather than for speed.

**The uniqueness constraints are what make `MERGE` safe.** A `MERGE` on a
property with no uniqueness constraint is a read followed by a conditional
create, and two workers running it concurrently for the same entity both read
"absent" and both create. The result is two `(:Company {id: "ent_acme"})` nodes,
each holding half the edges, and no error anywhere. With the constraint, Neo4j
takes a lock on the indexed value and the second writer waits and then matches.
`docs/knowledge-graph.md` §7 states the rule as "`MERGE` only on a uniquely-
constrained property"; this file is the other half of that rule, and
`graph/ingest/writer.py` is unsafe without it.

**The fulltext index is what `GET /graph/search` runs on.** Not an optimisation:
`db.index.fulltext.queryNodes('entity_search', …)` fails outright if the index
does not exist, so the endpoint returns 500 rather than being slow.

**The temporal indexes are seek-vs-scan on the two hottest edge properties.**
`MENTIONS` outnumbers every other relationship type by orders of magnitude, and
the complaint query filters it by `observed_at`; `COMPETES_WITH.valid_from` is
touched by every as-of predicate the Competitor Agent issues.

**This file mirrors `docker/local/neo4j/01-constraints.cypher` exactly.**
Not approximately -- exactly, statement for statement. They are applied by
different paths (`make init-db` runs the bootstrap file, `graph/schema/migrator.py`
applies `versions/v001_initial.cypher`, and both must produce the same schema), so
any drift means a query that is index-backed on a developer's laptop and a full
scan in production, or a constraint that exists in one environment and not the
other. `tests/unit/graph/test_schema.py` parses all three sources and asserts
they are the same set; that test is the reason this is a generator rather than
three hand-maintained copies.

Two consequences of "exactly" worth stating plainly, because both look like
oversights and are not:

* Only `Company`, `Product` and `Topic` carry a `canonical_name` index. The other
  four labels are looked up by id or through the fulltext index today. Adding the
  missing four is a `v002` change -- an applied version is never edited in place.
* There is **no uniqueness constraint on `(:Signal {id})`**, even though
  `graph/ingest/writer.py` merges on it. That is a real gap, not a modelling
  choice: the `Signal` stub is an eighth label that postdates this file and needs
  an ADR before a version writes it (`docs/knowledge-graph.md`, open question 1).
  It is declared below as `PENDING_CONSTRAINTS` -- visible, named, and
  deliberately excluded from `statements()` so it cannot be applied by accident
  and cannot make this file drift from the bootstrap.

Layer note: **L1 library** -- `models/` and the standard library only.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from graph.schema.edges import SIGNAL_LABEL, edge_spec
from graph.schema.nodes import NODE_SPECS, entity_labels, validate_label
from models.enums import EdgeType

__all__ = [
    "FULLTEXT_INDEXES",
    "FULLTEXT_INDEX_NAME",
    "NODE_INDEXES",
    "PENDING_CONSTRAINTS",
    "RELATIONSHIP_INDEXES",
    "UNIQUENESS_CONSTRAINTS",
    "FulltextIndexSpec",
    "NodeIndexSpec",
    "RelationshipIndexSpec",
    "SchemaStatement",
    "UniquenessConstraintSpec",
    "all_specs",
    "statements",
]

FULLTEXT_INDEX_NAME: Final[str] = "entity_search"

FULLTEXT_PROPERTIES: Final[tuple[str, ...]] = ("canonical_name", "aliases", "description")
"""What `entity_search` indexes on each of the seven labels.

`aliases` is in the list for the reason the property exists: a user searching
"Big Blue" has to find IBM, and a fulltext index over a list property indexes
every element. `description` is there because an LLM-written line is often the
only place a company's actual business is stated in words a user would type.
"""


class SchemaStatement(abc.ABC):
    """Base for the four statement kinds. Each renders to one Cypher string.

    A small hierarchy rather than a bag of strings because a statement's *kind*
    carries information the string does not: a uniqueness constraint is a
    correctness guarantee that `graph/ingest/writer.py` depends on, while a range
    index is a performance choice. Both are schema commands, and Neo4j refuses to
    mix a schema command with a data write in one transaction -- so a later
    version that does both has to be able to tell them apart.

    Abstract rather than raising from a default: an instance of the base is not a
    statement, and the type checker should say so at the call site instead of the
    program discovering it at apply time, halfway through a migration.
    """

    name: str

    @abc.abstractmethod
    def to_cypher(self) -> str:
        """Render this statement. Always `IF NOT EXISTS`, always idempotent."""


@dataclass(frozen=True, slots=True)
class UniquenessConstraintSpec(SchemaStatement):
    """`REQUIRE n.<property> IS UNIQUE` on one label.

    Also creates a backing index, which is why there is no separate `company_id`
    index: constraining a property indexes it.
    """

    name: str
    label: str
    property: str

    def to_cypher(self) -> str:
        label = validate_label(self.label)
        return (
            f"CREATE CONSTRAINT {self.name} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{self.property} IS UNIQUE"
        )


@dataclass(frozen=True, slots=True)
class NodeIndexSpec(SchemaStatement):
    """A plain range index on one property of one label."""

    name: str
    label: str
    property: str

    def to_cypher(self) -> str:
        label = validate_label(self.label)
        return (
            f"CREATE INDEX {self.name} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{self.property})"
        )


@dataclass(frozen=True, slots=True)
class FulltextIndexSpec(SchemaStatement):
    """One fulltext index spanning several labels and several properties."""

    name: str
    labels: tuple[str, ...]
    properties: tuple[str, ...]

    def to_cypher(self) -> str:
        labels = "|".join(validate_label(label) for label in self.labels)
        props = ", ".join(f"n.{p}" for p in self.properties)
        # Rendered across three lines to match the bootstrap file byte for byte
        # where it can. The drift test normalises whitespace anyway, but a
        # generator whose output is diffable against the file it mirrors is worth
        # more than one whose output merely parses to the same thing.
        return (
            f"CREATE FULLTEXT INDEX {self.name} IF NOT EXISTS\n"
            f"  FOR (n:{labels})\n"
            f"  ON EACH [{props}]"
        )


@dataclass(frozen=True, slots=True)
class RelationshipIndexSpec(SchemaStatement):
    """A range index on one property of one relationship type.

    Declared undirected -- `FOR ()-[r:TYPE]-()` -- because that is how the reads
    traverse. `COMPETES_WITH` is stored once in canonical orientation and matched
    with `-[r:COMPETES_WITH]-`; a directed index would not be usable by that
    match, and the query would scan every relationship of the type.
    """

    name: str
    edge_type: EdgeType
    property: str

    def to_cypher(self) -> str:
        rel = validate_label(edge_spec(self.edge_type).type_name)
        return (
            f"CREATE INDEX {self.name} IF NOT EXISTS "
            f"FOR ()-[r:{rel}]-() ON (r.{self.property})"
        )


# --------------------------------------------------------------------------- #
# The declarations, in the order the bootstrap file applies them
# --------------------------------------------------------------------------- #

UNIQUENESS_CONSTRAINTS: Final[tuple[UniquenessConstraintSpec, ...]] = tuple(
    UniquenessConstraintSpec(
        name=f"{spec.label.lower()}_id",
        label=spec.label,
        property="id",
    )
    for spec in NODE_SPECS.values()
)
"""One per entity label. Generated from the registry so a label added to
`nodes.py` cannot be added without its constraint -- the failure mode that would
otherwise appear as duplicate nodes under concurrency, months later, with no
error to point at."""

NODE_INDEXES: Final[tuple[NodeIndexSpec, ...]] = tuple(
    NodeIndexSpec(
        name=f"{spec.label.lower()}_name",
        label=spec.label,
        property=name,
    )
    for spec in NODE_SPECS.values()
    for name in spec.indexed
)
"""Driven by `NodeSpec.indexed`, which is `("canonical_name",)` on exactly the
three labels the bootstrap file indexes. See the module docstring."""

FULLTEXT_INDEXES: Final[tuple[FulltextIndexSpec, ...]] = (
    FulltextIndexSpec(
        name=FULLTEXT_INDEX_NAME,
        labels=entity_labels(),
        properties=FULLTEXT_PROPERTIES,
    ),
)

RELATIONSHIP_INDEXES: Final[tuple[RelationshipIndexSpec, ...]] = (
    RelationshipIndexSpec(
        name="mentions_observed",
        edge_type=EdgeType.MENTIONS,
        property="observed_at",
    ),
    RelationshipIndexSpec(
        name="competes_validity",
        edge_type=EdgeType.COMPETES_WITH,
        property="valid_from",
    ),
)
"""Only two, and `docs/knowledge-graph.md` §5 says the other five relationship
types need the same `valid_from` index in `v002`. Left as-is deliberately: this
file mirrors the bootstrap, and improving on it here is what "drift" means."""

PENDING_CONSTRAINTS: Final[tuple[UniquenessConstraintSpec, ...]] = (
    UniquenessConstraintSpec(name="signal_id", label=SIGNAL_LABEL, property="id"),
)
"""Declared, named, and **not** applied. Read the module docstring.

`graph/ingest/writer.py` merges on `(:Signal {id})` with no uniqueness
constraint behind it, so two graph workers processing two signals that mention
the same third signal can each create a stub. It is the least damaging instance
of the problem -- a `Signal` stub carries no properties to lose and duplicates
converge once the constraint is added -- but it is real, and it is recorded here
rather than in a comment nobody greps for.
"""


def all_specs() -> tuple[SchemaStatement, ...]:
    """Every applied statement, in application order.

    Constraints first. An index on a property that is about to be constrained is
    wasted work -- the constraint creates its own backing index -- and, more
    importantly, applying constraints before any data-touching statement in a
    later version means the uniqueness guarantee is in place before anything can
    violate it.
    """
    return (
        *UNIQUENESS_CONSTRAINTS,
        *NODE_INDEXES,
        *FULLTEXT_INDEXES,
        *RELATIONSHIP_INDEXES,
    )


def statements() -> tuple[str, ...]:
    """The Cypher for every applied statement, in application order.

    Every one is `IF NOT EXISTS`, so the whole sequence is idempotent and a
    partially-applied run is safe to repeat. That is what `v001_initial.cypher`
    is made of, and what `scripts/init_databases.py` needs from the bootstrap
    file: re-running `make init-db` against a live graph must be a no-op, not an
    error and certainly not a rebuild.
    """
    return tuple(spec.to_cypher() for spec in all_specs())


def as_cypher_file(header: Sequence[str] = ()) -> str:
    """Render the statements as the body of a `.cypher` file.

    Used to generate `versions/v001_initial.cypher` and to diff it in a test. The
    file is checked in rather than generated at run time on purpose: a schema
    version is an immutable artefact, and one produced by code that can change is
    not immutable. The checksum the migrator records would follow the code
    instead of the schema, which defeats the point of recording it.
    """
    lines = [*header, ""] if header else []
    lines += [f"{statement};" for statement in statements()]
    return "\n".join(lines) + "\n"
