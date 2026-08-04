"""Query expansion from the graph, and the facts that ride along with the text.

"DDOG", "Datadog Inc" and "Datadog" are one entity to the knowledge graph and
three unrelated strings to BM25. An analyst who types the ticker gets none of the
coverage that names the company, and nothing about that failure is visible: the
query returns *results*, just not the right ones, and the count in
`RetrievalDiagnostics` looks healthy. Expansion is the only thing in the pipeline
that closes that gap, because it is the only component that knows the three
strings denote one thing.

This module implements `retrieval/hybrid.py`'s `GraphExpander` and does exactly
two jobs.

**`expand_query`** turns seed entity ids -- and, when asked, the entities the
query text itself names -- into canonical names and aliases for the lexical
backend to use as extra `should` clauses at boost 0.6 (`docs/retrieval.md` §4).
Boosted *below* the literal terms, which is the whole reason the boost is
specified: an alias match must never outrank a document that used the words the
analyst actually typed. Three properties matter here and each has a failure mode
behind it:

* **Bounded.** Terms are capped. An entity absorbed by an over-eager resolution
  pass can carry hundreds of aliases, and Lucene refuses a boolean query beyond
  `max_clause_count` (1024 by default) with an error that names the clause count
  and nothing about where the clauses came from.
* **Deterministic.** Canonical names first, then aliases, both in a stable order.
  Expansion changes ranking; if the term list reshuffles between runs, so does
  the ranking, and the evaluation harness cannot tell a regression from noise.
* **Non-fatal.** `HybridRetriever` already treats an expansion failure as a
  degraded run rather than a failed one. What it cannot do is notice expansion
  that returned *nothing*, so the empty case is logged here.

**`facts_for`** returns the graph relationships worth carrying beside the
retrieved passages (`docs/retrieval.md` §9). A fact is not a passage: it has no
quotable span, so it travels in its own list with its own validity interval and
supporting signal ids. Every fact is read as-of the end of the request window --
`docs/knowledge-graph.md` §5 -- so a report about Q1 cannot assert a relationship
that was not known until August. Facts with a closed `valid_to` are still
returned, because "Acme competed with Globex until March" is an answer; the
interval is on the fact so the consumer can say which it is.

Read-only, through the same `GraphReader` port as `retrieval/graph_retrieval/
traversal.py`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final

from backend.core.logging import get_logger
from models.enums import EdgeType
from retrieval.filters.metadata import as_of_for
from retrieval.graph_retrieval.traversal import GraphReader, default_reader
from retrieval.types import GraphFact, RetrievalRequest

__all__ = [
    "ENTITY_FULLTEXT_INDEX",
    "FACT_EDGE_TYPES",
    "GraphQueryExpander",
    "lucene_safe",
]

_log = get_logger(__name__)

ENTITY_FULLTEXT_INDEX: Final[str] = "entity_search"
"""The fulltext index declared in `docker/local/neo4j/01-constraints.cypher`.

It covers `canonical_name`, `aliases` and `description` across all seven labels,
which is what makes "resolve the words in the query to entities" one index call
rather than a scan. Named here rather than inlined so the coupling to that file
is greppable -- if the index is renamed, this is the line that has to change."""

FACT_EDGE_TYPES: Final[tuple[str, ...]] = (
    EdgeType.COMPETES_WITH.value,
    EdgeType.ACQUIRED.value,
    EdgeType.USES.value,
    EdgeType.COMPLAINS_ABOUT.value,
    EdgeType.LAUNCHED_BY.value,
)
"""Edge types rendered as facts.

`MENTIONS` is excluded: "this document mentions Acme" is not a fact about the
world, it is the retrieval that already happened, and emitting one per passage
would fill the fact budget in `docs/retrieval.md` §9 with restatements of the
passage list."""

_MIN_TERM_LENGTH: Final[int] = 2
"""Tickers are short -- "X", "DDOG" -- but a one-character expansion term matches
most of the corpus. Two is the compromise; it costs the platform entity `X` its
own ticker, which the canonical-name path still supplies."""

_MAX_QUERY_TOKENS: Final[int] = 12
"""Tokens fed to the fulltext index. A pasted paragraph would otherwise become a
200-clause Lucene query that scores every entity in the graph."""

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[\w][\w'&.-]*", re.UNICODE)


class GraphQueryExpander:
    """Alias expansion and graph facts. Satisfies the `GraphExpander` Protocol.

    Stateless beyond its configuration and its reader, so one instance per
    process serves concurrent requests.
    """

    def __init__(
        self,
        reader: GraphReader | None = None,
        *,
        max_terms: int = 24,
        max_aliases_per_entity: int = 8,
        resolve_query_text: bool = True,
        max_resolved_entities: int = 5,
        min_resolution_score: float = 0.0,
        max_facts: int = 25,
        min_fact_confidence: float = 0.0,
        max_supporting_signals: int = 5,
        include_historical_facts: bool = False,
    ) -> None:
        if max_terms < 1:
            raise ValueError(f"max_terms must be at least 1, got {max_terms}")
        if max_aliases_per_entity < 0:
            raise ValueError(
                f"max_aliases_per_entity must be non-negative, got {max_aliases_per_entity}"
            )
        if max_facts < 1:
            raise ValueError(f"max_facts must be at least 1, got {max_facts}")
        if max_supporting_signals < 0:
            # Reaches Cypher as a list slice, `[..$exp_supporting]`, where a
            # negative bound counts from the end instead of raising: -2 quietly
            # drops the last two supporting signals from every fact. A fact is
            # rendered with a citation handle, so that is lost provenance rather
            # than a smaller list.
            raise ValueError(
                f"max_supporting_signals must be non-negative, got {max_supporting_signals}"
            )
        self._reader = reader if reader is not None else default_reader()
        self._max_terms = max_terms
        self._max_aliases = max_aliases_per_entity
        self._resolve_query_text = resolve_query_text
        self._max_resolved = max_resolved_entities
        self._min_resolution_score = min_resolution_score
        self._max_facts = max_facts
        self._min_fact_confidence = min_fact_confidence
        self._max_supporting_signals = max_supporting_signals
        # Off by default so `facts_for` obeys the invariant every as-of query in
        # `docs/knowledge-graph.md` §11 must hold: never return an edge whose
        # interval excludes the query instant. The Competitor agent has a real
        # use for "who did Acme compete with before the merger", which is why the
        # switch exists at all -- but a closed interval reaching a report that
        # did not ask for history reads as a current relationship, and the reader
        # has no way to tell.
        self._include_historical_facts = include_historical_facts

    # ------------------------------------------------------------ expansion --

    async def expand_query(self, request: RetrievalRequest) -> Sequence[str]:
        """Canonical names and aliases to widen lexical matching.

        Two sources, in priority order. Seed ids come from the caller -- the
        Planner resolved them, or the user picked an entity -- and are trusted;
        fulltext resolution of the query text is a guess and is therefore both
        bounded and score-floored. Seeds first means that when the term budget
        runs out it is the guesses that are dropped.

        The returned terms exclude anything already present in the query text.
        Re-adding a literal term as a 0.6-boosted `should` clause changes the
        ranking of documents that matched it for no reason a reader of the query
        could reconstruct, and spends budget that an actual alias needed.
        """
        # Concurrent because the two lookups are independent and expansion sits
        # *before* fan-out: every millisecond here is on the critical path of a
        # query that has not started searching yet. Sequentially this step costs
        # the sum of two round trips to widen a query by a handful of terms.
        seeded, resolved = await asyncio.gather(
            self._entities_by_id(request), self._entities_from_text(request)
        )

        terms: list[str] = []
        for entity in [*seeded, *resolved]:
            terms.extend(self._terms_for(entity))

        expanded = _dedupe_terms(terms, query=request.query, limit=self._max_terms)
        if not expanded:
            # Not an error: a query naming no known entity is the normal case on
            # a cold graph. It is logged because "expansion contributed nothing"
            # and "expansion was never called" produce identical results and very
            # different fixes.
            _log.debug(
                "graph.expansion.empty",
                seeds=len(request.seed_entity_ids),
                query_chars=len(request.query),
            )
        return expanded

    def _terms_for(self, entity: Mapping[str, Any]) -> list[str]:
        """Canonical name first, then a bounded slice of the aliases.

        Aliases are truncated per entity rather than only globally, so one
        over-merged entity cannot consume the whole term budget and starve the
        other seeds -- which is precisely what happens after a bad resolution
        pass, when the term list would otherwise be 300 spellings of one company.
        """
        terms = [_clean(entity.get("canonical_name"))]
        aliases = entity.get("aliases") or ()
        if isinstance(aliases, str):  # a single-alias property written unwrapped
            aliases = [aliases]
        cleaned = sorted({_clean(a) for a in aliases if _clean(a)})
        terms.extend(cleaned[: self._max_aliases])
        return [t for t in terms if t]

    async def _entities_by_id(self, request: RetrievalRequest) -> list[Mapping[str, Any]]:
        """Names and aliases for the seed ids, in the order the caller gave them.

        Order is preserved deliberately: the caller's first seed is the subject
        of the question, and when the term cap bites it should be the last thing
        dropped. Cypher returns rows in whatever order the planner produces, so
        the re-ordering happens here rather than in an `ORDER BY` the planner is
        free to ignore.
        """
        seeds = [s.strip() for s in request.seed_entity_ids if s and s.strip()]
        if not seeds:
            return []
        rows = await self._reader(
            _ENTITIES_BY_ID_CYPHER,
            {"exp_ids": seeds, "exp_tenant": request.filters.tenant_id},
        )
        by_id = {_clean(r.get("id")): r for r in rows if _clean(r.get("id"))}
        found = [by_id[s] for s in seeds if s in by_id]
        if len(found) < len(seeds):
            # A seed that resolves to nothing is a stale entity id -- a merge
            # tombstone, or an id from a report written before an un-merge. It
            # costs recall silently, so it is named.
            _log.info(
                "graph.expansion.unresolved_seeds",
                requested=len(seeds),
                resolved=len(found),
                missing=sorted(set(seeds) - set(by_id)),
            )
        return found

    async def _entities_from_text(self, request: RetrievalRequest) -> list[Mapping[str, Any]]:
        """Entities the query text itself names, via the fulltext index.

        Disabled by setting `resolve_query_text=False`: it costs a round trip,
        and a caller that already resolved its entities upstream is paying for a
        guess it does not need.
        """
        if not self._resolve_query_text or self._max_resolved < 1:
            return []
        lucene = lucene_safe(request.query)
        if not lucene:
            return []
        rows = await self._reader(
            _ENTITY_FULLTEXT_CYPHER,
            {
                "exp_index": ENTITY_FULLTEXT_INDEX,
                "exp_query": lucene,
                "exp_tenant": request.filters.tenant_id,
                "exp_min_score": self._min_resolution_score,
                # Over-fetch: the index scores across every tenant and label, and
                # the tenant predicate is applied after `YIELD`. Asking for
                # exactly `max_resolved` would return fewer once another tenant's
                # entities are filtered out -- the classic post-filter recall
                # loss, here confined to a lookup that cannot push the predicate
                # into the Lucene query at all.
                "exp_limit": self._max_resolved * _FULLTEXT_OVERFETCH,
                "exp_keep": self._max_resolved,
            },
        )
        return [row for row in rows if _clean(row.get("id"))]

    # ---------------------------------------------------------------- facts --

    async def facts_for(
        self, request: RetrievalRequest, signal_ids: Sequence[str]
    ) -> Sequence[GraphFact]:
        """Relationships among the entities the retrieved signals mention.

        Anchored on the signals that survived reranking rather than on the whole
        neighbourhood, so the facts describe what the report is about to cite
        instead of everything two hops from the seed. Seeds are added to the
        anchor set because the entity the analyst asked about deserves its facts
        even in the case where no retrieved passage happened to mention it by a
        name the extractor recognised.
        """
        anchors = _unique([*signal_ids])
        seeds = _unique(list(request.seed_entity_ids))
        if not anchors and not seeds:
            return []

        as_of = as_of_for(request.filters)
        rows = await self._reader(
            _FACTS_CYPHER.format(
                valid_to_clause=(
                    ""
                    if self._include_historical_facts
                    else "\n  AND (r.valid_to IS NULL OR r.valid_to > $exp_as_of)"
                )
            ),
            {
                "exp_signal_ids": anchors,
                "exp_seed_ids": seeds,
                "exp_tenant": request.filters.tenant_id,
                "exp_edge_types": list(FACT_EDGE_TYPES),
                "exp_as_of": as_of,
                "exp_min_confidence": self._min_fact_confidence,
                "exp_supporting": self._max_supporting_signals,
                # Over-fetched for the same reason the dedupe below exists: an
                # undirected match returns an edge once per endpoint that is in
                # the anchor set, so a fact between two anchored entities arrives
                # twice and would otherwise halve the effective budget.
                "exp_limit": self._max_facts * 2,
            },
        )

        facts: dict[tuple[str, str, str, str], GraphFact] = {}
        for row in rows:
            fact = _to_fact(row, self._max_supporting_signals)
            if fact is None:
                continue
            # Keyed on the *stored* orientation, which the query returns via
            # startNode/endNode rather than by match direction. COMPETES_WITH is
            # stored once and matched undirected (`docs/knowledge-graph.md` §3),
            # so without a normalised key the same edge can appear as both "A
            # competes with B" and "B competes with A" -- which a reader takes
            # for two independent pieces of evidence. The query already applies
            # `WITH DISTINCT r`; this is the net for the case where two *edges*
            # duplicate one relationship, which replay of a batch with a drifted
            # `edge_key` produces.
            key = (
                fact.predicate,
                fact.subject_id,
                fact.object_id,
                fact.valid_from.isoformat() if fact.valid_from else "",
            )
            existing = facts.get(key)
            if existing is None or fact.confidence > existing.confidence:
                facts[key] = fact

        ordered = sorted(
            facts.values(),
            key=lambda f: (-f.confidence, f.predicate, f.subject_id, f.object_id),
        )
        return ordered[: self._max_facts]


# --------------------------------------------------------------------------- #
# Cypher
# --------------------------------------------------------------------------- #

_FULLTEXT_OVERFETCH: Final[int] = 4

# Label-less by necessity: an entity id does not carry its label, and the
# uniqueness constraints are per-label. Bounded by the number of seeds, which the
# caller controls, so the scan is small -- but it is a scan, and a global lookup
# index on `id` is the fix if seed counts ever grow.
_ENTITIES_BY_ID_CYPHER: Final[str] = """
MATCH (e)
WHERE e.id IN $exp_ids AND e.tenant_id = $exp_tenant
RETURN e.id                            AS id,
       coalesce(e.canonical_name, '')  AS canonical_name,
       coalesce(e.aliases, [])         AS aliases,
       labels(e)[0]                    AS label
"""

# `db.index.fulltext.queryNodes` takes the index name and the query as
# parameters, so nothing is interpolated -- but the *query* is Lucene syntax, and
# a stray quote or trailing `~` from user text is a syntax error that raises
# inside the expansion call. `lucene_safe()` is what prevents that; it is not
# about injection (parameters handle that) but about a parse failure taking down
# a step that is only ever an optimisation.
_ENTITY_FULLTEXT_CYPHER: Final[str] = """
CALL db.index.fulltext.queryNodes($exp_index, $exp_query, {limit: $exp_limit})
YIELD node, score
WHERE node.tenant_id = $exp_tenant AND score >= $exp_min_score
RETURN node.id                            AS id,
       coalesce(node.canonical_name, '')  AS canonical_name,
       coalesce(node.aliases, [])         AS aliases,
       labels(node)[0]                    AS label,
       score                              AS score
ORDER BY score DESC, id ASC
LIMIT $exp_keep
"""

# Anchors are built in two steps rather than one `WHERE ... OR ...`, because the
# signal-mention half can start from the `Signal` label while an id lookup cannot
# start from any label at all -- an entity id does not carry its label and the
# uniqueness constraints are per-label. Folding them into one predicate would
# force the planner to scan for both.
#
# `collect()` over zero rows still yields one row holding an empty list, which is
# what lets the seed half survive an empty `$exp_signal_ids` (and vice versa).
# An `OR` over two `MATCH`es would drop everything the moment one side was empty.
#
# `startNode(r)` / `endNode(r)` rather than the match direction, so an undirected
# match reports the orientation the edge was stored in. Half-open validity per
# `docs/knowledge-graph.md` §5: `valid_to` is exclusive and null means open.
_FACTS_CYPHER: Final[str] = """
MATCH (s:Signal)-[:MENTIONS]->(mentioned)
WHERE s.id IN $exp_signal_ids AND mentioned.tenant_id = $exp_tenant
WITH collect(DISTINCT mentioned) AS mentioned_entities
OPTIONAL MATCH (seed)
WHERE seed.id IN $exp_seed_ids AND seed.tenant_id = $exp_tenant
WITH mentioned_entities + collect(DISTINCT seed) AS anchors
UNWIND anchors AS anchor
WITH DISTINCT anchor
MATCH (anchor)-[r]-(other)
WHERE type(r) IN $exp_edge_types
  AND other.tenant_id = $exp_tenant
  AND r.valid_from <= $exp_as_of{valid_to_clause}
  AND coalesce(r.confidence, 0.0) >= $exp_min_confidence
WITH DISTINCT r
WITH r, startNode(r) AS subject, endNode(r) AS object
RETURN type(r)                              AS predicate,
       subject.id                           AS subject_id,
       coalesce(subject.canonical_name, '') AS subject_name,
       object.id                            AS object_id,
       coalesce(object.canonical_name, '')  AS object_name,
       r.valid_from                         AS valid_from,
       r.valid_to                           AS valid_to,
       coalesce(r.confidence, 0.0)          AS confidence,
       coalesce(r.source_signal_ids, [])[..$exp_supporting] AS supporting_signal_ids
ORDER BY confidence DESC, predicate ASC, subject_id ASC
LIMIT $exp_limit
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def lucene_safe(text: str) -> str:
    """A Lucene query string that cannot fail to parse, from arbitrary text.

    Every token is quoted rather than escaped character by character. Quoting is
    the smaller surface: inside a phrase only `"` and `\\` are special, so two
    replacements make any token safe, whereas the escape-table approach has to
    stay in step with Lucene's operator set forever and fails open when it drifts.

    Tokens are joined with `OR` because the caller wants entities matching *any*
    of the query's words -- an `AND` would resolve only entities whose name
    contains the whole question. Empty output means "nothing worth looking up",
    and the caller skips the round trip entirely.
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0)
        if len(token) < _MIN_TERM_LENGTH:
            continue
        tokens.append('"' + token.replace("\\", "\\\\").replace('"', '\\"') + '"')
        if len(tokens) >= _MAX_QUERY_TOKENS:
            break
    return " OR ".join(tokens)


def _dedupe_terms(terms: Iterable[str], *, query: str, limit: int) -> list[str]:
    """Case-insensitive dedupe, drop terms the query already has, cap the rest.

    First-seen order is kept, which is why the caller appends seed terms before
    resolved ones: the cap then falls on the guesses rather than on the entity
    the analyst named.
    """
    haystack = (query or "").casefold()
    seen: set[str] = set()
    kept: list[str] = []
    for term in terms:
        cleaned = _clean(term)
        if len(cleaned) < _MIN_TERM_LENGTH:
            continue
        folded = cleaned.casefold()
        if folded in seen or folded in haystack:
            continue
        seen.add(folded)
        kept.append(cleaned)
        if len(kept) >= limit:
            break
    return kept


def _to_fact(row: Mapping[str, Any], max_supporting: int) -> GraphFact | None:
    """One record -> a `GraphFact`, or `None` when it cannot be cited.

    A fact whose subject or object has no id cannot be resolved by a reader, and
    a fact whose predicate is empty says nothing. Both are dropped rather than
    rendered as a line with a blank in it: a graph fact appears in the report
    with a citation handle beside it (`docs/retrieval.md` §9), and a handle that
    resolves to nothing is worse than an absent fact.
    """
    subject_id = _clean(row.get("subject_id"))
    object_id = _clean(row.get("object_id"))
    predicate = _clean(row.get("predicate"))
    if not subject_id or not object_id or not predicate:
        return None

    supporting = row.get("supporting_signal_ids") or ()
    if isinstance(supporting, str):
        supporting = [supporting]
    return GraphFact(
        subject_id=subject_id,
        subject_name=_clean(row.get("subject_name")) or subject_id,
        predicate=predicate,
        object_id=object_id,
        object_name=_clean(row.get("object_name")) or object_id,
        valid_from=_as_datetime(row.get("valid_from")),
        valid_to=_as_datetime(row.get("valid_to")),
        confidence=_clamp_score(row.get("confidence")),
        supporting_signal_ids=tuple(
            _clean(s) for s in list(supporting)[:max_supporting] if _clean(s)
        ),
    )


def _as_datetime(value: Any) -> datetime | None:
    """A timezone-aware datetime from whatever the driver handed back.

    The Bolt driver returns `neo4j.time.DateTime` for a `datetime` property and
    `neo4j.time.Date` for a `date`; a graph written through a different path may
    hold an ISO string. All three have to become one type before
    `GraphFact.is_current` means anything, and a *naive* result would compare
    wrongly against the aware `as_of` used everywhere else -- Python raises on
    that comparison, which at least fails loudly, but only after the fact has
    already been rendered. Unparseable values become `None`, which reads as "open
    interval" and is the conservative answer for a validity bound nobody wrote.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    native = getattr(value, "to_native", None)
    if callable(native):
        try:
            converted = native()
        except (TypeError, ValueError):  # pragma: no cover - driver-side oddity
            return None
        if isinstance(converted, datetime):
            return converted if converted.tzinfo else converted.replace(tzinfo=UTC)
        return None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _clamp_score(value: Any) -> float:
    """A confidence in [0, 1]. `GraphFact.confidence` is a `Score`.

    Clamped rather than rejected: an extractor that emits 1.0000001 should not
    make the fact unciteable, and one that emits 4.0 should not let a single
    relationship dominate the confidence-ordered fact budget.
    """
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number))


def _unique(values: Sequence[str]) -> list[str]:
    """De-duplicated, order-preserving, blanks dropped."""
    seen: dict[str, None] = {}
    for value in values:
        cleaned = _clean(value)
        if cleaned:
            seen.setdefault(cleaned, None)
    return list(seen)


def _clean(value: Any) -> str:
    """Trimmed text with internal whitespace collapsed, or `""`.

    Collapsed because an alias arriving as "Acme  Corp\\n" and one arriving as
    "Acme Corp" are the same term to a human and two distinct `should` clauses to
    OpenSearch, and the duplicate spends term budget to change nothing.
    """
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())
