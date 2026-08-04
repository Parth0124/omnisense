"""One filter compiler, three backend dialects.

`retrieval.types.Filter` is the only place a caller expresses "which slice of the
corpus". This module turns that one object into an OpenSearch `bool.filter`
array, a Qdrant `models.Filter`, and a Cypher `WHERE` clause with parameters.

**Why one compiler rather than three query builders.** The three backends index
the same chunks and are fused on `chunk_id`. If OpenSearch reads a time window as
`[start, end]` and Qdrant reads it as `[start, end)`, the two backends answer the
same request over slightly different corpora -- and reciprocal rank fusion has no
way to notice. The symptom is not an error; it is a chunk that appears in the
keyword list and never in the vector list, scoring as though only one backend
found it (`retrieval/rerank/fusion.py` explains why that is expensive). Drift
between hand-written filters is invisible in every test that exercises one
backend at a time, which is most of them. So the semantics are decided exactly
once, in `compile_predicates()`, and the three emitters are mechanical
translations of that single list.

**Pushdown, never post-filtering.** `docs/retrieval.md` §7: asking Qdrant for the
100 nearest neighbours and then keeping the three that are in-date is not the
same as asking for the 100 nearest in-date neighbours. A dimension a backend
cannot express is a bug to be fixed in the index, not a case for filtering the
result afterwards -- so every emitter here is total over the predicate list and
raises rather than dropping a predicate it does not understand.

**Tenant is not optional.** It is a predicate on every compiled filter in every
dialect, including the single-tenant Phase 1 deployment where it is a constant.
A tenant filter that is applied "when there is more than one tenant" is applied
never, because the code that would have added it does not exist yet on the day
the second tenant arrives.

Layer note: L1 (`docs/architecture.md` §6.1). Imports `models/` and the Qdrant
model types; touches no service, no API and no worker.
"""

from __future__ import annotations

import enum
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from qdrant_client import models as qmodels

from models.base import utcnow
from retrieval.types import Filter
from retrieval.vector.collections import PayloadField

__all__ = [
    "CYPHER_PARAM_PREFIX",
    "CypherFilter",
    "FilterField",
    "OpenSearchFilter",
    "Operator",
    "Predicate",
    "as_of_for",
    "compile_cypher",
    "compile_opensearch",
    "compile_predicates",
    "compile_qdrant",
    "filter_fingerprint",
]


class FilterField(enum.StrEnum):
    """The document fields a `Filter` may constrain.

    The values are simultaneously the OpenSearch field names
    (`backend/db/opensearch.py` `SIGNAL_INDEX_MAPPINGS`), the Qdrant payload keys
    (`retrieval/vector/collections.py` `PayloadField`) and the Neo4j `Signal`
    node properties. That the three stores agree on spelling is what lets one
    predicate list compile to all three without a per-backend name table that
    would rot, so the agreement is checked rather than assumed: the Qdrant half
    at import, below, and the OpenSearch half in `tests/unit/retrieval/
    test_filters.py`.

    The split is not arbitrary. Importing `retrieval/vector/collections.py` costs
    nothing this module does not already pay -- the Qdrant model types are needed
    to emit a filter at all. Importing `backend/db/opensearch.py` would pull
    `opensearchpy` into every process that touches a filter, including the
    graph-only path in `retrieval/graph_retrieval/`, for a constant that is only
    ever read at build time. A test catches the same drift at the same commit
    without putting an HTTP client on the import graph of a Cypher query.
    """

    TENANT_ID = "tenant_id"
    PUBLISHED_AT = "published_at"
    PLATFORM = "platform"
    SOURCE = "source"
    LANGUAGE = "language"
    ENTITY_IDS = "entity_ids"
    CONFIDENCE = "confidence"


# A filter naming a payload key that was never written matches zero points and
# raises nothing -- indistinguishable from "no results for that query"
# (`retrieval/vector/collections.py` makes the same argument for `PayloadField`).
# Checking the containment here turns a rename of a payload key into an
# ImportError at process start instead of a query that silently returns nothing.
_MISSING_IN_QDRANT_PAYLOAD = {f.value for f in FilterField} - {f.value for f in PayloadField}
if _MISSING_IN_QDRANT_PAYLOAD:  # pragma: no cover - a build-time invariant
    raise ImportError(
        "filter fields have no Qdrant payload key and would match nothing: "
        f"{sorted(_MISSING_IN_QDRANT_PAYLOAD)}. Add them to "
        "retrieval/vector/collections.PayloadField and re-index."
    )


class Operator(enum.StrEnum):
    """The five comparisons every backend must be able to express.

    Deliberately small. Each member exists because some dimension of `Filter`
    needs it, and adding a sixth means proving all three backends can push it
    down -- which is the point of the constraint.
    """

    EQUALS = "equals"
    """Scalar equality. Only `tenant_id` uses it, and it always does."""

    ANY_OF = "any_of"
    """Scalar field is a member of a set. Platform, source, language."""

    CONTAINS_ANY = "contains_any"
    """List-valued field intersects a set. Entity ids.

    Distinct from `ANY_OF` because the *document* side is the list. OpenSearch
    and Qdrant happen to spell both the same way; Neo4j does not, and conflating
    them there produces a predicate that is always false.
    """

    GTE = "gte"
    """Inclusive lower bound. Window start, and `min_confidence`."""

    LT = "lt"
    """Exclusive upper bound. Window end, so windows tile without overlap."""


@dataclass(frozen=True, slots=True)
class Predicate:
    """One compiled constraint: field, comparison, value.

    The intermediate representation the three emitters share. Frozen and
    hashable-by-content so `filter_fingerprint()` can identify a filter without
    re-deriving it, and so a predicate list can be compared in a test rather than
    a rendered query string, which would test the formatting instead of the
    semantics.
    """

    field: FilterField
    op: Operator
    value: Any

    def __post_init__(self) -> None:
        if self.op in (Operator.ANY_OF, Operator.CONTAINS_ANY) and (
            not isinstance(self.value, tuple) or not self.value
        ):
            raise ValueError(
                f"{self.op} on {self.field} needs a non-empty tuple of values; "
                "an empty set is not 'match everything', it is a predicate no "
                "document satisfies, and compiling it silently empties the result"
            )

    def fingerprint_key(self) -> tuple[str, str, Any]:
        """A JSON-safe identity for hashing. Datetimes become ISO strings."""
        if isinstance(self.value, datetime):
            return (self.field.value, self.op.value, self.value.isoformat())
        if isinstance(self.value, tuple):
            return (self.field.value, self.op.value, [str(v) for v in self.value])
        return (self.field.value, self.op.value, self.value)


# --------------------------------------------------------------------------- #
# The single source of truth
# --------------------------------------------------------------------------- #


def compile_predicates(filters: Filter) -> tuple[Predicate, ...]:
    """Lower a `Filter` to the backend-independent predicate list.

    Every semantic decision lives here and nowhere else:

    * **Tenant first, always.** Emitted before anything else so that reading a
      logged query left to right shows the isolation boundary before it shows
      the business constraint, and so a truncated log line still proves it.
    * **The time window is half-open, `[after, before)`.** `published_after` is
      inclusive, `published_before` is exclusive. Closed-closed windows
      double-count the boundary instant, which turns "January" plus "February"
      into a corpus where one midnight-published article is counted twice and
      makes month-over-month deltas wrong by exactly the boundary.
    * **Empty sets mean "unconstrained", not "match nothing".** `Filter` uses an
      empty `frozenset` as its default, so treating it as an `IN ()` predicate
      would make the default filter select zero documents.
    * **`entity_ids` is `CONTAINS_ANY`, not all-of.** A chunk mentioning any
      requested entity is on topic; requiring all of them is a conjunctive query
      that returns nothing on a corpus of short social posts, which mention one
      entity each. `docs/retrieval.md` §7 keeps `all_of` as a future dimension;
      `Filter` does not carry it yet, and inventing it here would let the three
      backends disagree about which one it was.

    Raises:
        ValueError: if `tenant_id` is blank, or a datetime bound is naive.
    """
    if not filters.tenant_id or not filters.tenant_id.strip():
        raise ValueError(
            "Filter.tenant_id is blank. It is derived from the authenticated "
            "principal in backend/api/deps.py and is never caller-supplied or "
            "optional; a blank tenant would compile to a filter that reads every "
            "tenant's corpus."
        )

    predicates: list[Predicate] = [
        Predicate(FilterField.TENANT_ID, Operator.EQUALS, filters.tenant_id)
    ]

    if filters.published_after is not None:
        predicates.append(
            Predicate(
                FilterField.PUBLISHED_AT,
                Operator.GTE,
                _require_aware(filters.published_after, "published_after"),
            )
        )
    if filters.published_before is not None:
        predicates.append(
            Predicate(
                FilterField.PUBLISHED_AT,
                Operator.LT,
                _require_aware(filters.published_before, "published_before"),
            )
        )

    if filters.platforms:
        predicates.append(
            Predicate(FilterField.PLATFORM, Operator.ANY_OF, _sorted_values(filters.platforms))
        )
    if filters.sources:
        predicates.append(
            Predicate(FilterField.SOURCE, Operator.ANY_OF, _sorted_values(filters.sources))
        )
    if filters.languages:
        predicates.append(
            Predicate(FilterField.LANGUAGE, Operator.ANY_OF, _sorted_values(filters.languages))
        )
    if filters.entity_ids:
        predicates.append(
            Predicate(
                FilterField.ENTITY_IDS, Operator.CONTAINS_ANY, _sorted_values(filters.entity_ids)
            )
        )
    if filters.min_confidence is not None:
        predicates.append(
            Predicate(FilterField.CONFIDENCE, Operator.GTE, float(filters.min_confidence))
        )
    return tuple(predicates)


def as_of_for(filters: Filter) -> datetime:
    """The instant the knowledge graph should be read at, for this filter.

    The end of the requested window, or now when the window is open-ended. Not
    `utcnow()` unconditionally: an investigation scoped to Q1 must see the graph
    as it was believed to be in Q1. Reading today's graph for a Q1 question
    silently back-dates every edge learned since -- the report then asserts a
    competitive relationship "as of March" that was not known until August, which
    is the exact failure bitemporal edges exist to prevent
    (`docs/knowledge-graph.md` §5).
    """
    return filters.published_before or utcnow()


def filter_fingerprint(filters: Filter) -> str:
    """A short stable hash of the compiled predicates, for traces and caches.

    `docs/retrieval.md` §7 requires filter selectivity to be logged, so that "the
    model hallucinated" can be told apart from "the filter left three documents".
    That needs a stable identity for the filter: hashing the compiled predicates
    rather than the `Filter` object means two filters that differ only in set
    iteration order -- which `frozenset` gives no guarantees about -- hash the
    same, and a filter that changed meaning always hashes differently.
    """
    payload = json.dumps(
        [p.fingerprint_key() for p in compile_predicates(filters)],
        separators=(",", ":"),
        sort_keys=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


# --------------------------------------------------------------------------- #
# Target 1: OpenSearch
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OpenSearchFilter:
    """Clauses destined for the `filter` array of a `bool` query.

    The `filter` context and not `must`, because these are restrictive and never
    scoring (`docs/retrieval.md` §7). In `must` they would contribute to `_score`
    and a document would rank higher for matching more filters -- which reorders
    results by how well they match the *metadata* the caller already committed
    to, not by relevance. `filter` context is also cacheable by OpenSearch, and
    the tenant clause is the same on every query in the deployment.
    """

    clauses: tuple[Mapping[str, Any], ...] = ()

    def to_bool(self) -> dict[str, Any]:
        """The clauses as a standalone `bool` query, for use as a whole filter."""
        return {"bool": {"filter": [dict(c) for c in self.clauses]}}


def compile_opensearch(filters: Filter) -> OpenSearchFilter:
    """Compile to OpenSearch `bool.filter` clauses.

    Range bounds are emitted as ISO 8601 strings rather than epoch millis: the
    `published_at` mapping is `date`, which parses ISO natively, and an epoch
    number is silently interpreted as *milliseconds* -- a seconds-valued epoch
    lands in 1970 and quietly matches nothing.
    """
    clauses: list[Mapping[str, Any]] = []
    ranges: dict[str, dict[str, Any]] = {}

    for predicate in compile_predicates(filters):
        name = predicate.field.value
        match predicate.op:
            case Operator.EQUALS:
                clauses.append({"term": {name: predicate.value}})
            case Operator.ANY_OF | Operator.CONTAINS_ANY:
                # A `terms` clause over a keyword field and over a keyword array
                # field are the same query: it matches when any indexed value of
                # the field is in the list. That the array case is spelled
                # identically is an OpenSearch convenience, not a coincidence
                # worth relying on elsewhere -- Neo4j needs two forms.
                clauses.append({"terms": {name: list(predicate.value)}})
            case Operator.GTE | Operator.LT:
                ranges.setdefault(name, {})[predicate.op.value] = _os_scalar(predicate.value)
            case _:  # pragma: no cover - Operator is closed
                raise NotImplementedError(f"no OpenSearch form for operator {predicate.op}")

    # Both bounds of a window belong in one `range` clause. Two clauses on the
    # same field are equivalent here, but a single clause is what an operator
    # reading a slow-query log expects to see, and it keeps the emitted JSON
    # comparable between runs.
    clauses.extend({"range": {name: bounds}} for name, bounds in ranges.items())
    return OpenSearchFilter(clauses=tuple(clauses))


def _os_scalar(value: Any) -> Any:
    return value.isoformat() if isinstance(value, datetime) else value


# --------------------------------------------------------------------------- #
# Target 2: Qdrant
# --------------------------------------------------------------------------- #


def compile_qdrant(filters: Filter) -> qmodels.Filter:
    """Compile to a Qdrant payload filter, all conditions conjunctive.

    Every condition goes in `must`. Qdrant applies payload filtering *during*
    HNSW traversal rather than after it, which is precisely why pushdown here
    preserves recall instead of truncating it -- but only for fields with a
    payload index. `retrieval/vector/collections.PAYLOAD_INDEXES` covers every
    `FilterField`; without the index the filter is still correct and becomes a
    full scan, which shows up as latency rather than as an error.

    `DatetimeRange` and not `Range` for the time window: `Range` compares floats,
    so handing it a datetime is a type error and handing it an epoch number would
    only work against a numeric payload -- while `published_at` is indexed as
    `DATETIME` (`retrieval/vector/collections.py`).
    """
    conditions: list[qmodels.Condition] = []
    datetime_bounds: dict[str, dict[str, datetime]] = {}
    numeric_bounds: dict[str, dict[str, float]] = {}

    for predicate in compile_predicates(filters):
        name = predicate.field.value
        match predicate.op:
            case Operator.EQUALS:
                conditions.append(
                    qmodels.FieldCondition(
                        key=name, match=qmodels.MatchValue(value=predicate.value)
                    )
                )
            case Operator.ANY_OF | Operator.CONTAINS_ANY:
                # `MatchAny` against a list-valued payload key matches when the
                # intersection is non-empty, which is exactly `CONTAINS_ANY`.
                # `entity_ids` is written as a list even when it has one element
                # for this reason (`retrieval/vector/collections.py`).
                conditions.append(
                    qmodels.FieldCondition(
                        key=name, match=qmodels.MatchAny(any=list(predicate.value))
                    )
                )
            case Operator.GTE | Operator.LT:
                target = (
                    datetime_bounds if isinstance(predicate.value, datetime) else numeric_bounds
                )
                target.setdefault(name, {})[predicate.op.value] = predicate.value
            case _:  # pragma: no cover - Operator is closed
                raise NotImplementedError(f"no Qdrant form for operator {predicate.op}")

    conditions.extend(
        qmodels.FieldCondition(key=name, range=qmodels.DatetimeRange(**bounds))
        for name, bounds in datetime_bounds.items()
    )
    conditions.extend(
        qmodels.FieldCondition(key=name, range=qmodels.Range(**bounds))
        for name, bounds in numeric_bounds.items()
    )
    return qmodels.Filter(must=conditions)


# --------------------------------------------------------------------------- #
# Target 3: Neo4j / Cypher
# --------------------------------------------------------------------------- #

CYPHER_PARAM_PREFIX: Final[str] = "flt_"
"""Namespace for compiled filter parameters.

Traversal queries carry parameters of their own -- `as_of`, `fanout_cap`, the
frontier -- and a compiled filter is merged into that map. Without a prefix a
filter on `source` would collide with a traversal parameter named `source` and
one would silently win, changing the query's meaning with no error at all.
"""


@dataclass(frozen=True, slots=True)
class CypherFilter:
    """Cypher `WHERE` fragments plus the parameters they reference.

    Never an interpolated string. Entity names and query terms reach this layer
    from LLM output and connector payloads, and Cypher injection is exactly as
    real as SQL injection (`backend/db/neo4j.py` makes the same point). A
    parameterized query is also plan-cached by the server, which an interpolated
    one is not.
    """

    clauses: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def where(self, *, extra: Sequence[str] = ()) -> str:
        """The full `WHERE ...` line, or `""` when there is nothing to constrain.

        `extra` is for predicates the caller owns -- the edge as-of window, say --
        so that a query has one `WHERE` rather than a filter clause bolted onto a
        second one with an `AND` the caller had to remember to write.
        """
        parts = [*self.clauses, *extra]
        return f"WHERE {' AND '.join(parts)}" if parts else ""

    def merged_parameters(self, **extra: Any) -> dict[str, Any]:
        """Filter parameters plus the caller's, refusing a name collision."""
        clash = set(extra) & set(self.parameters)
        if clash:
            raise ValueError(f"parameter names collide with compiled filter: {sorted(clash)}")
        return {**self.parameters, **extra}


def compile_cypher(
    filters: Filter,
    *,
    alias: str = "s",
    entity_alias: str = "_fe",
    mention_type: str = "MENTIONS",
) -> CypherFilter:
    """Compile to `WHERE` fragments over a `(:Signal)` node bound as `alias`.

    Six of the seven dimensions are property predicates on the Signal reference
    node (`docs/knowledge-graph.md` §2). The seventh, `entity_ids`, is
    **structural**: the reference node holds no entity array, the mention is an
    edge, and `docs/retrieval.md` §7 says so explicitly. It compiles to an
    `EXISTS { }` subquery rather than to a post-filter, because a post-filter is
    the one thing pushdown exists to avoid -- and because an `EXISTS` subquery is
    evaluated by the planner against the `MENTIONS` index rather than by
    materialising the neighbourhood.

    A note on a real failure mode: `language` and `confidence` must be mirrored
    onto the Signal reference node by `graph/ingest/writer.py`. Where they are
    not, `s.language IN $langs` is null-valued, the predicate is false, and the
    graph backend returns *nothing* rather than returning the wrong thing. That
    is the better of the two failures -- an empty graph list is visible in
    `RetrievalDiagnostics.per_backend_counts`, whereas an unfiltered one is not --
    but it is a failure, and it is why the mirror is a writer requirement rather
    than an optimisation.
    """
    clauses: list[str] = []
    parameters: dict[str, Any] = {}

    for predicate in compile_predicates(filters):
        name = predicate.field.value
        param = f"{CYPHER_PARAM_PREFIX}{name}_{predicate.op.value}"
        match predicate.op:
            case Operator.EQUALS:
                clauses.append(f"{alias}.{name} = ${param}")
                parameters[param] = predicate.value
            case Operator.ANY_OF:
                clauses.append(f"{alias}.{name} IN ${param}")
                parameters[param] = [str(v) for v in predicate.value]
            case Operator.CONTAINS_ANY:
                # The Signal reference node carries no entity array; the mention
                # is the edge. `ANY(x IN $ids WHERE ...)` over a property that
                # does not exist would be null, hence false, hence a graph
                # backend that silently contributes nothing.
                clauses.append(
                    f"EXISTS {{ MATCH ({alias})-[:{mention_type}]->({entity_alias}) "
                    f"WHERE {entity_alias}.id IN ${param} }}"
                )
                parameters[param] = [str(v) for v in predicate.value]
            case Operator.GTE:
                clauses.append(f"{alias}.{name} >= ${param}")
                parameters[param] = predicate.value
            case Operator.LT:
                clauses.append(f"{alias}.{name} < ${param}")
                parameters[param] = predicate.value
            case _:  # pragma: no cover - Operator is closed
                raise NotImplementedError(f"no Cypher form for operator {predicate.op}")

    return CypherFilter(clauses=tuple(clauses), parameters=parameters)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _sorted_values(values: Iterable[Any]) -> tuple[str, ...]:
    """A `frozenset` in a stable order, as strings.

    Sorted because `frozenset` iteration order varies between processes: an
    unsorted `terms` list produces a different query JSON, a different filter
    fingerprint and a different OpenSearch request cache key on every restart,
    for a filter that never changed. `str()` because `Platform` and
    `SourceCategory` are `StrEnum`s that must reach the wire as plain strings.
    """
    return tuple(sorted(str(v) for v in values))


def _require_aware(value: datetime, label: str) -> datetime:
    """Reject a timezone-naive bound instead of assuming UTC.

    Assuming UTC can move the boundary by up to a day for a caller in another
    zone, silently including or excluding a day of Signals, and nothing
    downstream can detect that it happened. `retrieval/vector/collections.py`
    refuses naive datetimes on the write side for the same reason; refusing them
    on the read side too is what keeps the two sides comparable.
    """
    if value.tzinfo is None:
        raise ValueError(
            f"Filter.{label} is timezone-naive. Retrieval compares instants across "
            "three stores; assuming UTC here would shift the window boundary by the "
            "caller's offset with no way to notice."
        )
    return value
