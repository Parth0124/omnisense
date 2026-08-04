"""Unit tests for `retrieval/filters/metadata.py`.

The compiler's whole reason to exist is that three backends must select the *same
logical set* of chunks for one `Filter`. That property cannot be tested by
inspecting three query objects and agreeing they look right -- which is exactly
how the drift it guards against gets in, because a `range` with an inclusive
upper bound and a `<` in Cypher look equally reasonable side by side.

So the central test here interprets all three dialects against one in-memory
corpus and asserts the three result sets are identical *and* equal to an
independent reference predicate. The reference matters: three compilers that are
wrong in the same way would agree with each other perfectly.

The interpreters are small and deliberately literal -- a `terms` clause is
intersection, a Cypher `<` is exclusive -- so a compiler change that alters
semantics fails here rather than in an integration test nobody runs locally.

No network, no services, no containers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from qdrant_client import models as qmodels

from models.enums import Platform, SourceCategory
from retrieval.filters.metadata import (
    CypherFilter,
    FilterField,
    Operator,
    Predicate,
    as_of_for,
    compile_cypher,
    compile_opensearch,
    compile_predicates,
    compile_qdrant,
    filter_fingerprint,
)
from retrieval.types import Filter

pytestmark = pytest.mark.unit

TENANT = "tnt_main"
OTHER_TENANT = "tnt_other"


def at(day: int, hour: int = 12) -> datetime:
    """A UTC instant in January 2026, for readable window boundaries."""
    return datetime(2026, 1, day, hour, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# The corpus every dialect is interpreted against
# --------------------------------------------------------------------------- #


def chunk(
    chunk_id: str,
    *,
    tenant: str = TENANT,
    published: datetime | None = None,
    platform: Platform = Platform.REDDIT,
    source: SourceCategory = SourceCategory.SOCIAL,
    language: str = "en",
    entities: Sequence[str] = (),
    confidence: float = 0.8,
) -> dict[str, Any]:
    """One indexed chunk, with exactly the fields a `Filter` can constrain."""
    return {
        "chunk_id": chunk_id,
        "tenant_id": tenant,
        "published_at": published or at(15),
        "platform": platform.value,
        "source": source.value,
        "language": language,
        "entity_ids": list(entities),
        "confidence": confidence,
    }


CORPUS: tuple[dict[str, Any], ...] = (
    chunk("a:0", published=at(1), entities=["ent_acme"]),
    chunk("b:0", published=at(10), platform=Platform.X, entities=["ent_acme", "ent_globex"]),
    chunk(
        "c:0",
        published=at(20),
        platform=Platform.RSS,
        source=SourceCategory.NEWS,
        language="de",
        entities=["ent_globex"],
        confidence=0.95,
    ),
    chunk("d:0", published=at(31), platform=Platform.ARXIV, source=SourceCategory.RESEARCH),
    chunk("e:0", published=at(10), confidence=0.2, entities=["ent_initech"]),
    chunk("f:0", tenant=OTHER_TENANT, published=at(10), entities=["ent_acme"]),
    # Boundary probes: exactly on each bound of the windows used below.
    chunk("start:0", published=at(10, 0)),
    chunk("end:0", published=at(20, 0)),
    chunk("g:0", published=at(12), language="fr", entities=[]),
)


def reference_matches(doc: Mapping[str, Any], filters: Filter) -> bool:
    """The intended semantics, written once and independently of the compiler.

    Half-open `[after, before)`, empty sets unconstrained, `entity_ids` as
    intersection, tenant always. If this disagrees with a dialect, one of them is
    the bug -- and having a fourth opinion is what makes it possible to say which.
    """
    if doc["tenant_id"] != filters.tenant_id:
        return False
    if filters.published_after and doc["published_at"] < filters.published_after:
        return False
    if filters.published_before and doc["published_at"] >= filters.published_before:
        return False
    if filters.platforms and doc["platform"] not in {str(p) for p in filters.platforms}:
        return False
    if filters.sources and doc["source"] not in {str(s) for s in filters.sources}:
        return False
    if filters.languages and doc["language"] not in {str(x) for x in filters.languages}:
        return False
    if filters.entity_ids and not set(doc["entity_ids"]) & {str(e) for e in filters.entity_ids}:
        return False
    return not (filters.min_confidence is not None and doc["confidence"] < filters.min_confidence)


# --------------------------------------------------------------------------- #
# Three interpreters
# --------------------------------------------------------------------------- #


def opensearch_matches(doc: Mapping[str, Any], clauses: Sequence[Mapping[str, Any]]) -> bool:
    """Evaluate a `bool.filter` array. All clauses conjunctive, as OpenSearch does."""
    for clause in clauses:
        if "term" in clause:
            field, value = next(iter(clause["term"].items()))
            if doc[field] != value:
                return False
        elif "terms" in clause:
            field, values = next(iter(clause["terms"].items()))
            actual = doc[field]
            haystack = set(actual) if isinstance(actual, list) else {actual}
            if not haystack & set(values):
                return False
        elif "range" in clause:
            field, bounds = next(iter(clause["range"].items()))
            actual = doc[field]
            for operator, raw in bounds.items():
                bound = datetime.fromisoformat(raw) if isinstance(actual, datetime) else raw
                if operator == "gte" and not actual >= bound:
                    return False
                if operator == "lt" and not actual < bound:
                    return False
                if operator not in ("gte", "lt"):
                    raise AssertionError(f"interpreter has no rule for range op {operator!r}")
        else:
            raise AssertionError(f"interpreter has no rule for clause {clause!r}")
    return True


def field_conditions(filters: Filter) -> list[qmodels.FieldCondition]:
    """The compiled Qdrant conditions, narrowed to the only kind emitted.

    Narrowing rather than indexing blindly: `Filter.must` is typed as a union of
    every condition Qdrant supports, and a compiler that started emitting a
    `NestedCondition` should fail this assertion rather than be silently skipped
    by an interpreter that only understands field conditions.
    """
    conditions = compile_qdrant(filters).must or ()
    assert isinstance(conditions, list) or conditions == ()
    narrowed = [c for c in conditions if isinstance(c, qmodels.FieldCondition)]
    assert len(narrowed) == len(list(conditions)), "compiler emitted a non-field condition"
    return narrowed


def qdrant_matches(doc: Mapping[str, Any], compiled: qmodels.Filter) -> bool:
    """Evaluate a Qdrant payload filter. Everything lives in `must`."""
    assert not compiled.should, "conditions in `should` would be OR, not restrictive"
    assert not compiled.must_not, "the compiler emits no negations"
    for condition in compiled.must or ():
        assert isinstance(condition, qmodels.FieldCondition)
        value = doc[condition.key]
        match = condition.match
        if isinstance(match, qmodels.MatchValue):
            if value != match.value:
                return False
        elif isinstance(match, qmodels.MatchAny):
            haystack = set(value) if isinstance(value, list) else {value}
            if not haystack & set(match.any):
                return False
        elif match is not None:
            raise AssertionError(f"interpreter has no rule for match {match!r}")
        bounds = condition.range
        if bounds is not None:
            assert bounds.gt is None and bounds.lte is None, "compiler emits only gte/lt"
            if bounds.gte is not None and not value >= bounds.gte:
                return False
            if bounds.lt is not None and not value < bounds.lt:
                return False
    return True


_EQ = re.compile(r"^s\.(\w+) = \$(\w+)$")
_IN = re.compile(r"^s\.(\w+) IN \$(\w+)$")
_GTE = re.compile(r"^s\.(\w+) >= \$(\w+)$")
_LT = re.compile(r"^s\.(\w+) < \$(\w+)$")
_EXISTS = re.compile(r"^EXISTS \{ MATCH \(s\)-\[:MENTIONS\]->\(\w+\) WHERE \w+\.id IN \$(\w+) \}$")


def cypher_matches(doc: Mapping[str, Any], compiled: CypherFilter) -> bool:
    """Evaluate the emitted `WHERE` fragments against a document.

    A tiny interpreter rather than a Cypher parser: the compiler emits five
    shapes, and pinning them here means a change to the emitted text is a test
    failure rather than a silent change in what the graph backend selects. The
    `EXISTS` form is evaluated as "some mentioned entity is in the set", which is
    what the subquery means when the Signal node has no entity array.
    """
    params = compiled.parameters
    for clause in compiled.clauses:
        if found := _EQ.match(clause):
            field, param = found.groups()
            if doc[field] != params[param]:
                return False
        elif found := _IN.match(clause):
            field, param = found.groups()
            if doc[field] not in params[param]:
                return False
        elif found := _GTE.match(clause):
            field, param = found.groups()
            if not doc[field] >= params[param]:
                return False
        elif found := _LT.match(clause):
            field, param = found.groups()
            if not doc[field] < params[param]:
                return False
        elif found := _EXISTS.match(clause):
            (param,) = found.groups()
            if not set(doc["entity_ids"]) & set(params[param]):
                return False
        else:
            raise AssertionError(f"interpreter has no rule for clause {clause!r}")
    return True


def selected(filters: Filter) -> tuple[set[str], set[str], set[str], set[str]]:
    """The chunk ids each dialect selects, plus the reference set."""
    os_clauses = compile_opensearch(filters).clauses
    qdrant = compile_qdrant(filters)
    cypher = compile_cypher(filters)
    return (
        {d["chunk_id"] for d in CORPUS if opensearch_matches(d, os_clauses)},
        {d["chunk_id"] for d in CORPUS if qdrant_matches(d, qdrant)},
        {d["chunk_id"] for d in CORPUS if cypher_matches(d, cypher)},
        {d["chunk_id"] for d in CORPUS if reference_matches(d, filters)},
    )


# --------------------------------------------------------------------------- #
# The headline property
# --------------------------------------------------------------------------- #

FILTER_CASES: dict[str, Filter] = {
    "tenant only": Filter(tenant_id=TENANT),
    "other tenant": Filter(tenant_id=OTHER_TENANT),
    "window closed": Filter(
        tenant_id=TENANT, published_after=at(10, 0), published_before=at(20, 0)
    ),
    "window open end": Filter(tenant_id=TENANT, published_after=at(12)),
    "window open start": Filter(tenant_id=TENANT, published_before=at(12)),
    "platforms": Filter(tenant_id=TENANT, platforms=frozenset({Platform.X, Platform.RSS})),
    "sources": Filter(tenant_id=TENANT, sources=frozenset({SourceCategory.NEWS})),
    "languages": Filter(tenant_id=TENANT, languages=frozenset({"en", "fr"})),
    "one entity": Filter(tenant_id=TENANT, entity_ids=frozenset({"ent_acme"})),
    "two entities": Filter(tenant_id=TENANT, entity_ids=frozenset({"ent_acme", "ent_initech"})),
    "confidence": Filter(tenant_id=TENANT, min_confidence=0.5),
    "everything at once": Filter(
        tenant_id=TENANT,
        published_after=at(1),
        published_before=at(25),
        platforms=frozenset({Platform.REDDIT, Platform.X, Platform.RSS}),
        sources=frozenset({SourceCategory.SOCIAL, SourceCategory.NEWS}),
        languages=frozenset({"en", "de"}),
        entity_ids=frozenset({"ent_acme", "ent_globex"}),
        min_confidence=0.5,
    ),
    "selects nothing": Filter(tenant_id=TENANT, languages=frozenset({"is"})),
}


@pytest.mark.parametrize("name", list(FILTER_CASES))
def test_three_dialects_select_the_same_documents(name: str) -> None:
    """One filter, three backends, one corpus slice.

    The failure this prevents is not an error: it is a chunk that OpenSearch
    returns and Qdrant does not, scoring in fusion as though a single backend
    found it. Nothing raises, no count looks wrong, and the ranking is quietly
    worse for every query that touches the boundary.
    """
    from_opensearch, from_qdrant, from_cypher, expected = selected(FILTER_CASES[name])
    assert from_opensearch == expected, f"OpenSearch dialect drifted on {name!r}"
    assert from_qdrant == expected, f"Qdrant dialect drifted on {name!r}"
    assert from_cypher == expected, f"Cypher dialect drifted on {name!r}"


def test_the_corpus_actually_discriminates() -> None:
    """Guard on the guard: an all-or-nothing corpus would pass everything above."""
    sizes = {name: len(selected(f)[3]) for name, f in FILTER_CASES.items()}
    interesting = [n for n, size in sizes.items() if 0 < size < len(CORPUS)]
    assert len(interesting) >= len(FILTER_CASES) - 2, sizes


def test_window_boundaries_are_half_open_in_every_dialect() -> None:
    """`[after, before)`: the start instant is in, the end instant is out.

    A closed-closed window double-counts the boundary, so "January" plus
    "February" counts a midnight-published article twice and every
    month-over-month delta is wrong by exactly that article.
    """
    filters = Filter(tenant_id=TENANT, published_after=at(10, 0), published_before=at(20, 0))
    from_opensearch, from_qdrant, from_cypher, expected = selected(filters)
    assert "start:0" in expected and "end:0" not in expected
    assert from_opensearch == from_qdrant == from_cypher == expected


def test_every_dimension_reaches_every_dialect() -> None:
    """No dimension may be silently dropped by one emitter.

    A dropped predicate is the pushdown failure the module exists to prevent: the
    backend answers over a wider corpus and returns *more* results, which looks
    like better recall right up until the extra documents are cited.
    """
    filters = FILTER_CASES["everything at once"]
    expected_fields = {p.field.value for p in compile_predicates(filters)}
    assert expected_fields == {f.value for f in FilterField}

    os_text = repr([dict(c) for c in compile_opensearch(filters).clauses])
    qdrant_keys = {c.key for c in field_conditions(filters)}
    cypher = compile_cypher(filters)
    cypher_text = " ".join(cypher.clauses) + " " + " ".join(cypher.parameters)

    for name in expected_fields:
        assert name in os_text, f"OpenSearch dropped {name}"
        assert name in qdrant_keys, f"Qdrant dropped {name}"
        assert name in cypher_text, f"Cypher dropped {name}"


# --------------------------------------------------------------------------- #
# Tenant
# --------------------------------------------------------------------------- #


def test_tenant_is_applied_even_when_nothing_else_is() -> None:
    """The default `Filter` is empty of *business* constraints, never of tenant."""
    filters = Filter()
    assert filters.is_empty()

    predicates = compile_predicates(filters)
    assert predicates[0] == Predicate(FilterField.TENANT_ID, Operator.EQUALS, "default")
    assert compile_opensearch(filters).clauses == ({"term": {"tenant_id": "default"}},)
    assert len(field_conditions(filters)) == 1
    assert compile_cypher(filters).clauses == ("s.tenant_id = $flt_tenant_id_equals",)


def test_blank_tenant_is_refused_by_every_entry_point() -> None:
    """A blank tenant would compile to a filter that reads every tenant's corpus."""
    filters = Filter(tenant_id="   ")
    for compile in (compile_predicates, compile_opensearch, compile_qdrant, compile_cypher):
        with pytest.raises(ValueError, match="tenant_id"):
            compile(filters)


def test_tenant_partitions_the_corpus() -> None:
    """Two tenants, disjoint results, no leakage in any dialect."""
    mine = selected(Filter(tenant_id=TENANT))[3]
    theirs = selected(Filter(tenant_id=OTHER_TENANT))[3]
    assert mine and theirs
    assert not mine & theirs


# --------------------------------------------------------------------------- #
# Semantics that are easy to get quietly wrong
# --------------------------------------------------------------------------- #


def test_empty_sets_mean_unconstrained_not_empty() -> None:
    """An empty `frozenset` is the default; compiling it as `IN ()` selects nothing."""
    _, _, _, everything = selected(Filter(tenant_id=TENANT))
    assert everything == {d["chunk_id"] for d in CORPUS if d["tenant_id"] == TENANT}


def test_entity_ids_is_any_of_not_all_of() -> None:
    """A chunk mentioning *any* requested entity is on topic.

    All-of on a corpus of short social posts -- which mention one entity each --
    returns nothing, and an empty result reads as "no coverage" rather than as a
    query that could not have matched.
    """
    both = selected(Filter(tenant_id=TENANT, entity_ids=frozenset({"ent_acme", "ent_globex"})))[3]
    assert both == {"a:0", "b:0", "c:0"}


def test_naive_datetimes_are_refused() -> None:
    """Assuming UTC would move the boundary by the caller's offset, undetectably."""
    with pytest.raises(ValueError, match="timezone-naive"):
        compile_predicates(Filter(published_after=datetime(2026, 1, 1)))
    with pytest.raises(ValueError, match="timezone-naive"):
        compile_predicates(Filter(published_before=datetime(2026, 1, 1)))


def test_predicate_refuses_an_empty_value_set() -> None:
    """`IN ()` is not "match everything"; it is a predicate nothing satisfies."""
    with pytest.raises(ValueError, match="non-empty"):
        Predicate(FilterField.PLATFORM, Operator.ANY_OF, ())


def test_platform_and_source_reach_the_wire_as_plain_strings() -> None:
    """`StrEnum` members must serialize as their values, not as `Platform.X`."""
    filters = Filter(platforms=frozenset({Platform.X}), sources=frozenset({SourceCategory.NEWS}))
    terms = [c["terms"] for c in compile_opensearch(filters).clauses if "terms" in c]
    assert {"platform": ["x"]} in terms
    assert {"source": ["news"]} in terms
    # Qdrant's client serializes through pydantic, which would happily keep an
    # enum member and send `"Platform.X"` to a payload holding `"x"`.
    matches = [c.match for c in field_conditions(filters) if c.key == "platform"]
    assert [type(v) for v in matches[0].any] == [str]  # type: ignore[union-attr]


# --------------------------------------------------------------------------- #
# Dialect-specific shapes that carry a cost when wrong
# --------------------------------------------------------------------------- #


def test_opensearch_emits_one_range_clause_per_field() -> None:
    """Both bounds in one clause: what an operator reading a slow-query log expects."""
    filters = Filter(published_after=at(1), published_before=at(20))
    ranges = [c for c in compile_opensearch(filters).clauses if "range" in c]
    assert len(ranges) == 1
    assert set(ranges[0]["range"]["published_at"]) == {"gte", "lt"}


def test_opensearch_dates_are_iso_strings_not_epochs() -> None:
    """An epoch is read as *milliseconds*; a seconds-valued one lands in 1970."""
    filters = Filter(published_after=at(1))
    bounds = next(c for c in compile_opensearch(filters).clauses if "range" in c)
    assert bounds["range"]["published_at"]["gte"] == at(1).isoformat()


def test_qdrant_uses_datetime_range_for_time_and_range_for_confidence() -> None:
    """`Range` compares floats; handing it a datetime is a type error at query time."""
    filters = Filter(published_after=at(1), min_confidence=0.4)
    by_key = {c.key: c for c in field_conditions(filters)}
    assert isinstance(by_key["published_at"].range, qmodels.DatetimeRange)
    assert isinstance(by_key["confidence"].range, qmodels.Range)


def test_cypher_never_interpolates_a_value() -> None:
    """Cypher injection is as real as SQL injection, and entity names come from LLMs."""
    hostile = "x' OR 1=1 //"
    filters = Filter(tenant_id=hostile, languages=frozenset({hostile}))
    compiled = compile_cypher(filters)
    for clause in compiled.clauses:
        assert hostile not in clause
    assert hostile in compiled.parameters.values()


def test_cypher_entity_filter_is_structural_not_a_property_read() -> None:
    """The Signal reference node holds no entity array; the mention is the edge."""
    compiled = compile_cypher(Filter(entity_ids=frozenset({"ent_acme"})))
    entity_clause = next(c for c in compiled.clauses if "entity" in c or "MENTIONS" in c)
    assert entity_clause.startswith("EXISTS {")
    assert "s.entity_ids" not in entity_clause


def test_cypher_where_merges_caller_predicates_into_one_clause() -> None:
    compiled = compile_cypher(Filter())
    assert compiled.where(extra=("r.valid_to IS NULL",)) == (
        "WHERE s.tenant_id = $flt_tenant_id_equals AND r.valid_to IS NULL"
    )


def test_cypher_parameters_refuse_a_name_collision() -> None:
    """A silent collision does not raise -- one value wins and the query changes meaning."""
    compiled = compile_cypher(Filter())
    assert compiled.merged_parameters(trv_as_of=at(1))["flt_tenant_id_equals"] == "default"
    with pytest.raises(ValueError, match="collide"):
        compiled.merged_parameters(flt_tenant_id_equals="someone_else")


def test_empty_filter_produces_no_where_clause_beyond_tenant() -> None:
    assert compile_cypher(Filter()).where() == "WHERE s.tenant_id = $flt_tenant_id_equals"


# --------------------------------------------------------------------------- #
# Fingerprint and as-of
# --------------------------------------------------------------------------- #


def test_every_filter_field_exists_in_the_opensearch_mapping() -> None:
    """A filter on a field the index never mapped matches nothing, and says nothing.

    `retrieval/filters/metadata.py` makes the equivalent check against the Qdrant
    payload keys at import; OpenSearch is checked here instead so the compiler
    does not drag an HTTP client onto the import graph of a Cypher query. The
    failure it guards against is silent either way: a renamed field produces a
    `term` clause on a nonexistent field, which is not an error to OpenSearch --
    it is a query that matches zero documents, indistinguishable from a hard one.
    """
    from backend.db.opensearch import SIGNAL_INDEX_MAPPINGS

    mapped = set(SIGNAL_INDEX_MAPPINGS["properties"])
    missing = {f.value for f in FilterField} - mapped
    assert not missing, (
        f"filter fields absent from SIGNAL_INDEX_MAPPINGS: {sorted(missing)}. "
        "Add them to the mapping and reindex, or the filter silently selects nothing."
    )


def test_fingerprint_is_stable_across_set_iteration_order() -> None:
    """`frozenset` order varies between processes; the filter's identity must not."""
    one = Filter(platforms=frozenset({Platform.X, Platform.REDDIT, Platform.RSS}))
    two = Filter(platforms=frozenset({Platform.RSS, Platform.REDDIT, Platform.X}))
    assert filter_fingerprint(one) == filter_fingerprint(two)


def test_fingerprint_changes_when_meaning_changes() -> None:
    """Otherwise filter selectivity in a trace cannot be attributed to a filter."""
    base = Filter(tenant_id=TENANT, published_after=at(1))
    assert filter_fingerprint(base) != filter_fingerprint(
        Filter(tenant_id=TENANT, published_after=at(2))
    )
    assert filter_fingerprint(base) != filter_fingerprint(
        Filter(tenant_id=OTHER_TENANT, published_after=at(1))
    )


def test_as_of_follows_the_window_end() -> None:
    """A Q1 question reads the graph as it was believed to be at the end of Q1."""
    assert as_of_for(Filter(published_before=at(20))) == at(20)


def test_as_of_defaults_to_now_for_an_open_window() -> None:
    before = datetime.now(UTC)
    resolved = as_of_for(Filter())
    assert before <= resolved <= datetime.now(UTC)
