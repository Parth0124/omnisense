"""Turning a `RetrievalRequest` into an OpenSearch `bool` query, and back again.

`docs/retrieval.md` §4 specifies the shape: `should` clauses for the analysed
query, a phrase clause on `text.exact`, a boosted `title` clause, every compiled
filter in `filter`, an optional `gauss` decay on `published_at`, and expansion
terms at a lower boost than a literal match. This module is that specification,
plus the four things it does not say out loud -- each of which produces a
plausible-looking result list rather than an error when it is got wrong.

**Filters go in `filter`, and `filter` is not optional.** A `filter` clause is
restrictive and non-scoring: a document that fails it is *absent*, not demoted.
The alternative -- fetch the top 100 and drop the ones that fail -- silently
turns `k=100` into however many survive, and the symptom is "the last month of
data is unreachable", noticed weeks later by a human. Nothing in this module
post-filters, and the search body is exported so a test can assert that.

**`minimum_should_match` is set explicitly.** A `bool` query with `filter` and
`should` but no `must` defaults to `minimum_should_match: 0`, which means *every
document passing the filter matches*, scored 0 for the query. The keyword backend
would then answer a query that matches nothing with 100 arbitrary in-window
documents at rank 1..100 -- which fusion happily blends into the pool, and which
reads downstream as bad ranking rather than as no results.

**Recall clauses and boost clauses are separated.** Anything that decides
*whether* a document matches lives in an inner `bool`; anything that only decides
*how well* it scores hangs off the outer `should`. Keyword boosts from
`Signal.keywords` are boosts -- putting them where they can satisfy
`minimum_should_match` would let a document whose extracted keyword list happens
to contain the query term match on that alone.

**Clause counts are capped.** `indices.query.bool.max_clause_count` defaults to
1024, and a `match` clause expands to one term query per analysed token. A Signal
with 300 keywords plus a generous alias expansion reaches that ceiling, and the
cluster answers with a `too_many_clauses` failure for that one query -- which
`retrieval/hybrid.py` correctly degrades around, so the only trace is a
keyword-backend outage that nobody can reproduce.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from backend.core.logging import get_logger
from models.signal import Keyword
from retrieval.keyword.index import (
    EXACT_SUBFIELD,
    ChunkField,
    language_text_field,
    primary_subtag,
)
from retrieval.types import (
    Backend,
    Candidate,
    Filter,
    RetrievalRequest,
    split_chunk_id,
)

__all__ = [
    "MAX_RESULT_WINDOW",
    "QueryOptions",
    "build_filter_clauses",
    "build_search_body",
    "candidates_from_response",
]

_log = get_logger(__name__)

MAX_RESULT_WINDOW: Final[int] = 10_000
"""OpenSearch's default `index.max_result_window`.

A `size` above it is rejected by the cluster, not clamped. Checked here so the
error names the caller's `limit` rather than arriving as a shard failure from a
request nobody can see.
"""

TEXT_EXACT_FIELD: Final[str] = f"{ChunkField.TEXT.value}.{EXACT_SUBFIELD}"
TITLE_EXACT_FIELD: Final[str] = f"{ChunkField.TITLE.value}.{EXACT_SUBFIELD}"


@dataclass(frozen=True, slots=True)
class QueryOptions:
    """The scoring knobs, in one place so they are greppable and tunable.

    Every value here is an **untuned starting point** (`docs/retrieval.md` §3).
    They are gathered into a dataclass rather than scattered as default arguments
    because the evaluation harness in `retrieval/evaluation/` has to be able to
    sweep them, and a default argument buried in a query function is not
    something anyone sweeps.
    """

    title_boost: float = 2.0
    """`docs/retrieval.md` §4. A title match is a statement about the whole
    document; a body match is a statement about one sentence in it."""

    phrase_boost: float = 2.0
    """Boost on the phrase clauses over `text.exact` / `title.exact`.

    The unstemmed, unstopped analyzer is the reason this field exists: "connection
    reset by peer" as a phrase is a different query from those four words OR'd
    together, and the standard analyzer cannot express the difference.
    """

    language_boost: float = 1.2
    """Boost on the per-language sibling field, when a language is pinned.

    Deliberately modest. Stemmed matching adds recall (`laufen` matching `lief`)
    but also adds false positives, so it should nudge ranking rather than decide
    it.
    """

    expansion_boost: float = 0.6
    """`docs/retrieval.md` §4: "so an alias match never outranks a literal
    match". Below 1.0 is the whole point -- an alias is a graph inference about
    what the user meant, not something they typed."""

    keyword_boost: float = 1.5
    """Base boost for a term matching the document's extracted `keywords`.

    Multiplied by the `Keyword.weight` the extractor assigned, so a salient term
    contributes more than a marginal one.
    """

    recency_scale_days: int | None = None
    """Half-life scale of the `gauss` decay on `published_at`, or None for no decay.

    Off by default and switched on per request: `docs/retrieval.md` §4 attaches it
    to "trend-shaped questions", and applying it to "what did the CEO say in
    2019" would bury the only correct answer under last week's noise.
    """

    recency_decay: float = 0.5
    """Score multiplier at exactly `recency_scale_days` from now."""

    max_expansion_clauses: int = 32
    max_keyword_clauses: int = 25
    """Clause caps. See the module docstring on `max_clause_count`."""

    def __post_init__(self) -> None:
        if self.recency_scale_days is not None and self.recency_scale_days <= 0:
            raise ValueError(
                f"recency_scale_days must be positive, got {self.recency_scale_days}; "
                "pass None to disable the decay rather than 0, which would make "
                "every document older than 'now' score zero"
            )
        if not 0.0 < self.recency_decay < 1.0:
            raise ValueError(
                f"recency_decay must be in (0, 1), got {self.recency_decay}"
            )


DEFAULT_OPTIONS: Final[QueryOptions] = QueryOptions()


# --------------------------------------------------------------------------- #
# Filters -- restrictive, non-scoring, and pushed down
# --------------------------------------------------------------------------- #


def build_filter_clauses(filters: Filter) -> list[dict[str, Any]]:
    """Compile the metadata filter into OpenSearch `filter` clauses.

    Returned as a list so the caller drops it straight into `bool.filter`, where
    OpenSearch skips scoring entirely and can cache the bitset. Putting the same
    conditions in `must` would produce identical *results* and different
    *scores* -- every filtered document would contribute its constant-score match
    to the total, so a query with a five-platform filter would rank differently
    from the same query with a one-platform filter for reasons that have nothing
    to do with relevance.

    The tenant term is emitted unconditionally, including in single-tenant Phase
    1 where it is the constant `"default"`. `docs/retrieval.md` §7: "Tenant is not
    optional." A filter that is conditional is a filter that will one day be
    conditionally absent.
    """
    clauses: list[dict[str, Any]] = [
        {"term": {ChunkField.TENANT_ID.value: filters.tenant_id}}
    ]

    time_range = _time_range(filters)
    if time_range:
        clauses.append({"range": {ChunkField.PUBLISHED_AT.value: time_range}})

    if filters.platforms:
        clauses.append(
            {"terms": {ChunkField.PLATFORM.value: sorted(str(p) for p in filters.platforms)}}
        )
    if filters.sources:
        clauses.append(
            {"terms": {ChunkField.SOURCE.value: sorted(str(s) for s in filters.sources)}}
        )
    if filters.languages:
        # Normalised through the same function the indexer uses. A `terms` clause
        # is an exact keyword comparison: `pt` and `pt-BR` are simply different
        # strings, and the mismatch shows up as an empty result set, never as an
        # error.
        clauses.append(
            {
                "terms": {
                    ChunkField.LANGUAGE.value: sorted(
                        {primary_subtag(code) for code in filters.languages}
                    )
                }
            }
        )
    if filters.entity_ids:
        # `any_of` semantics (`docs/retrieval.md` §7): a chunk mentioning any of
        # the requested entities is in scope. `all_of` would be one `term` clause
        # per id; the `Filter` type carries no `all_of` set, so expressing it here
        # would be inventing a semantic the caller cannot ask for.
        clauses.append(
            {"terms": {ChunkField.ENTITY_IDS.value: sorted(filters.entity_ids)}}
        )
    if filters.min_confidence is not None:
        clauses.append(
            {"range": {ChunkField.CONFIDENCE.value: {"gte": filters.min_confidence}}}
        )
    return clauses


def _time_range(filters: Filter) -> dict[str, str]:
    """The `[after, before)` window, half-open, as OpenSearch range bounds.

    Half-open is not a detail. `docs/retrieval.md` §7 specifies `[start, end)`,
    and a closed upper bound makes a document published at exactly midnight
    appear in both the window that ends there and the window that starts there --
    so two adjacent trend buckets each count it, and the totals silently exceed
    the corpus.
    """
    bounds: dict[str, str] = {}
    if filters.published_after is not None:
        bounds["gte"] = _instant(filters.published_after, "published_after")
    if filters.published_before is not None:
        bounds["lt"] = _instant(filters.published_before, "published_before")
    return bounds


def _instant(value: datetime, label: str) -> str:
    """Serialise a bound, refusing a naive datetime.

    OpenSearch reads a datetime with no offset as UTC. That is a guess, and for a
    bound it is a guess that moves the edge of the window by up to a day --
    quietly including or excluding a day of evidence. The same refusal guards the
    write path in `retrieval/keyword/index.py`, for the same reason.
    """
    if value.tzinfo is None:
        raise ValueError(
            f"{label} is timezone-naive; OpenSearch would assume UTC and shift the "
            "window boundary by up to a day without reporting anything"
        )
    return value.isoformat()


# --------------------------------------------------------------------------- #
# The query
# --------------------------------------------------------------------------- #


def build_search_body(
    request: RetrievalRequest,
    *,
    limit: int,
    options: QueryOptions = DEFAULT_OPTIONS,
    expansions: Sequence[str] = (),
    keywords: Sequence[Keyword] = (),
    include_source: bool = False,
) -> dict[str, Any]:
    """Build the complete `_search` request body.

    Args:
        request: the retrieval request; its `filters` are pushed down here and
            nowhere else.
        limit: `size`. The caller's `k_keyword`, passed by `retrieval/hybrid.py`.
        options: scoring knobs; see `QueryOptions`.
        expansions: aliases and canonical names from
            `retrieval/graph_retrieval/expansion.py`. Entering as *phrase*
            clauses is deliberate -- "Datadog Inc" tokenised into an OR would
            match every document containing "Inc".
        keywords: salient terms (`Signal.keywords`) to boost. Scoring only; they
            never widen the result set.
        include_source: whether to fetch `_source`. Off by default: the passage
            text is re-read from PostgreSQL by the resolver
            (`retrieval/hybrid.py`), so returning it here would ship the whole
            chunk body over the wire on every query for data that is thrown away.

    Raises:
        ValueError: a non-positive `limit`, a `limit` past `max_result_window`,
            or a timezone-naive filter bound.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    if limit > MAX_RESULT_WINDOW:
        raise ValueError(
            f"limit {limit} exceeds index.max_result_window ({MAX_RESULT_WINDOW}); "
            "OpenSearch rejects the request rather than clamping it. Deep paging "
            "needs search_after, not a bigger size."
        )

    query_text = request.query.strip()
    matching = _matching_clauses(query_text, request.filters, options, expansions)
    boosting = _boost_clauses(keywords, options)

    inner: dict[str, Any] = {"filter": build_filter_clauses(request.filters)}
    if matching:
        # The inner bool is the recall decision, isolated so the boost clauses
        # below cannot satisfy it. `minimum_should_match: 1` is what stops a
        # `filter`-only bool from matching the entire corpus; see the module
        # docstring.
        inner["must"] = [{"bool": {"should": matching, "minimum_should_match": 1}}]
    else:
        # A filter-only browse: no query text, no expansions. Legitimate (a
        # graph-seeded "everything about this entity last week" request), but it
        # must be *explicit*. `match_all` says "every document passing the
        # filter, unranked" rather than leaving it to a defaulted
        # `minimum_should_match`.
        inner["must"] = [{"match_all": {}}]
    if boosting:
        inner["should"] = boosting

    query: dict[str, Any] = {"bool": inner}
    if options.recency_scale_days is not None:
        query = _with_recency_decay(query, options)

    return {
        "size": limit,
        "query": query,
        # Only the id and the score are used: the candidate carries a rank, and
        # the text is resolved from PostgreSQL.
        "_source": include_source,
        # Counting every match costs a full scan of the posting lists past the
        # top-k. Nothing in the retrieval path reads a total; `docs/retrieval.md`
        # §7 wants *result counts*, which is `len(candidates)`.
        "track_total_hits": False,
    }


def _matching_clauses(
    query_text: str,
    filters: Filter,
    options: QueryOptions,
    expansions: Sequence[str],
) -> list[dict[str, Any]]:
    """The clauses that decide whether a document matches at all."""
    clauses: list[dict[str, Any]] = []
    if query_text:
        clauses.append({"match": {ChunkField.TEXT.value: {"query": query_text}}})
        clauses.append(
            {
                "match": {
                    ChunkField.TITLE.value: {
                        "query": query_text,
                        "boost": options.title_boost,
                    }
                }
            }
        )
        clauses.append(
            {
                "match_phrase": {
                    TEXT_EXACT_FIELD: {
                        "query": query_text,
                        "boost": options.phrase_boost,
                    }
                }
            }
        )
        clauses.append(
            {
                "match_phrase": {
                    TITLE_EXACT_FIELD: {
                        "query": query_text,
                        "boost": options.phrase_boost,
                    }
                }
            }
        )
        clauses.extend(_language_clauses(query_text, filters, options))

    clauses.extend(_expansion_clauses(expansions, options))
    return clauses


def _language_clauses(
    query_text: str, filters: Filter, options: QueryOptions
) -> list[dict[str, Any]]:
    """Stemmed matching on the sibling field, only when a language is pinned.

    When the request does not constrain language, no sibling field is queried at
    all. Querying all of them would run the query text through fourteen
    morphologies -- an English query stemmed by the Turkish analyzer produces
    tokens that match Turkish documents for no reason -- at fourteen times the
    term lookups, to add recall in a language the caller never asked about. The
    language-neutral `text` field already covers every document, including the
    `und` ones that have no sibling field.
    """
    clauses: list[dict[str, Any]] = []
    # Deduped through `primary_subtag` first: a filter naming both `pt` and
    # `pt-BR` is one language, and emitting `text_pt` twice would double its
    # contribution to the score for no reason a reader of the query could see.
    for code in sorted({primary_subtag(code) for code in filters.languages}):
        field = language_text_field(code)
        if field is None:
            # A filtered language with no analyzer (or `und`). The filter clause
            # still restricts to it; there is simply no stemmed field to boost
            # on, which is a missing optimisation rather than a missing result.
            continue
        clauses.append(
            {"match": {field: {"query": query_text, "boost": options.language_boost}}}
        )
    return clauses


def _expansion_clauses(
    expansions: Sequence[str], options: QueryOptions
) -> list[dict[str, Any]]:
    """Alias and canonical-name clauses, deduped, capped and under-weighted.

    Phrase clauses on the unstemmed field rather than `match`: an alias is a
    proper noun ("Datadog Inc", "Elastic N.V."), and OR-ing its tokens turns the
    corporate suffix into a term that matches a large fraction of the news
    corpus.
    """
    terms = _distinct(expansions, options.max_expansion_clauses, kind="expansion")
    return [
        {
            "match_phrase": {
                TEXT_EXACT_FIELD: {"query": term, "boost": options.expansion_boost}
            }
        }
        for term in terms
    ]


def _boost_clauses(
    keywords: Sequence[Keyword], options: QueryOptions
) -> list[dict[str, Any]]:
    """Scoring-only clauses matching the document's extracted `keywords`.

    `term` rather than `match`, because the `keywords` field is a `keyword` type
    holding terms that have already been through extraction: re-analysing them
    would reintroduce exactly the noise that step removed
    (`backend/db/opensearch.py`).

    Highest-weighted first and then capped, so the cap drops the marginal terms
    rather than whichever ones the extractor happened to emit last.
    """
    if not keywords:
        return []
    ranked = sorted(keywords, key=lambda k: (-float(k.weight), k.term))
    clauses: list[dict[str, Any]] = []
    seen: set[str] = set()
    for keyword in ranked:
        term = keyword.term.strip().lower()
        if not term or term in seen:
            continue
        seen.add(term)
        if len(clauses) >= options.max_keyword_clauses:
            _log.debug(
                "opensearch.query.keyword_clauses_capped",
                cap=options.max_keyword_clauses,
                offered=len(keywords),
            )
            break
        clauses.append(
            {
                "term": {
                    ChunkField.KEYWORDS.value: {
                        "value": term,
                        "boost": options.keyword_boost * float(keyword.weight),
                    }
                }
            }
        )
    return clauses


def _distinct(values: Iterable[str], cap: int, *, kind: str) -> list[str]:
    """Trimmed, deduped, order-preserving, capped."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = value.strip()
        folded = term.lower()
        if not term or folded in seen:
            continue
        seen.add(folded)
        out.append(term)
        if len(out) >= cap:
            _log.debug("opensearch.query.clauses_capped", kind=kind, cap=cap)
            break
    return out


def _with_recency_decay(query: dict[str, Any], options: QueryOptions) -> dict[str, Any]:
    """Wrap the query in a `gauss` decay on `published_at`.

    A *scoring* function, never a filter. The distinction is the whole design:
    an old document that is the only real answer must be demoted, not deleted,
    and a range filter cannot express "prefer recent".

    Documents with no `published_at` are neutral rather than penalised -- a decay
    function scores a missing field as 1.0 -- which is right: a missing date is
    unknown recency, not old.

    `origin: "now"` is resolved per shard at query time, so a decayed query is
    not cacheable in the request cache. That is affordable because the decay is
    opt-in per request (`QueryOptions.recency_scale_days`), and it is worth
    saying out loud because "search got slower after we enabled recency" is
    otherwise a mystery.
    """
    return {
        "function_score": {
            "query": query,
            "functions": [
                {
                    "gauss": {
                        ChunkField.PUBLISHED_AT.value: {
                            "origin": "now",
                            "scale": f"{options.recency_scale_days}d",
                            "offset": "0d",
                            "decay": options.recency_decay,
                        }
                    }
                }
            ],
            "score_mode": "multiply",
            # Multiply rather than replace: `replace` would discard the BM25
            # score entirely and rank purely by date, which is a feed, not a
            # search result.
            "boost_mode": "multiply",
        }
    }


# --------------------------------------------------------------------------- #
# The response
# --------------------------------------------------------------------------- #


def candidates_from_response(response: Mapping[str, Any]) -> list[Candidate]:
    """Turn a `_search` response into ranked `Candidate`s.

    Ranks are **1-based and dense**: rank 1 is the top hit, and skipping a
    malformed hit closes the gap rather than leaving a hole. Fusion scores by
    `1 / (k + rank)` (`retrieval/rerank/fusion.py`), so a hole would hand the
    surviving candidates a rank -- and therefore a fused score -- that reflects
    a document that was discarded.

    An empty result is an empty list, not an error. "No lexical match" is the
    normal answer to a semantic question, and raising here would mark the whole
    keyword backend as failed in `RetrievalDiagnostics` -- turning a legitimately
    empty channel into a reported degradation and a lowered confidence on the
    final report.
    """
    hits = _hits(response)

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for hit in hits:
        chunk_id = hit.get("_id")
        if not isinstance(chunk_id, str):
            continue
        try:
            signal_id, _ = split_chunk_id(chunk_id)
        except ValueError:
            # A document whose `_id` is not a chunk id was written by something
            # that is not this indexer. Skipping keeps one bad document from
            # failing every query that matches it; logging keeps it from being
            # invisible, which is how it would stay bad.
            _log.warning("opensearch.hit.malformed_id", chunk_id=chunk_id)
            continue
        if chunk_id in seen:
            # Only reachable when the searched name is an alias spanning two
            # indices mid-reindex. Fusion accumulates an RRF term per occurrence,
            # so a duplicate here would let one backend manufacture agreement
            # with itself and outrank a chunk three backends actually agree on.
            continue
        seen.add(chunk_id)

        score = hit.get("_score")
        candidates.append(
            Candidate(
                chunk_id=chunk_id,
                backend=Backend.KEYWORD,
                rank=len(candidates) + 1,
                # `_score` is null when the request sorts by a field instead of
                # by relevance. Kept as 0.0 rather than dropped: raw scores are
                # diagnostics only, and fusion consumes the rank.
                raw_score=float(score) if isinstance(score, (int, float)) else 0.0,
                signal_id=signal_id,
            )
        )
    return candidates


def _hits(response: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    """The hit list, tolerating every shape a non-error response can take.

    `hits.hits` is absent from a response to a `count`-shaped request and from
    some error-but-200 shard responses. Defaulting to empty rather than raising
    keeps a malformed response a zero-result query -- which the diagnostics
    already show -- instead of an exception attributed to the backend being down.
    """
    hits = response.get("hits")
    if not isinstance(hits, Mapping):
        return ()
    inner = hits.get("hits")
    if not isinstance(inner, Sequence) or isinstance(inner, (str, bytes)):
        return ()
    return [hit for hit in inner if isinstance(hit, Mapping)]
