"""ANN search over the chunk collection, with the metadata filter pushed down.

This is the `SearchBackend` (`retrieval/hybrid.py`) that speaks Qdrant. It turns
a `RetrievalRequest` into one `query_points` call and its hits into `Candidate`s
-- chunk ids and ranks, never text. Resolving a chunk to a citable passage is a
separate batched step, because all three backends routinely return the same chunk
and fetching it three times is exactly the case the design hopes for.

**The filter is pushed into the ANN search, never applied to its output.**
Requesting the 100 nearest neighbours and then keeping the ones inside the date
window is *not* the same operation as requesting the 100 nearest neighbours
inside the date window. The first walks the HNSW graph over the whole collection
and hands back whatever survives -- on a corpus where last month is 2% of the
points, a one-month window turns k=100 into a couple of candidates -- while the
second keeps descending until it has 100 that match. Both return "results",
neither errors, and the difference is invisible until somebody notices that
recent data is unreachable. Qdrant evaluates the filter *inside* the graph
traversal, so pushing it down is both the correct and the cheap option; the only
cost is that every filtered field must carry a payload index, which
`retrieval/vector/collections.py` guarantees.

**This module never embeds anything.** The query vector arrives from the caller,
through the `QueryEmbedder` handed to the constructor. Choosing a provider here
would put an HTTP client and an API key behind a retrieval import, and would make
the layering claim in `docs/architecture.md` §6.1 false: `retrieval/` reads
`models/` and the graph, and takes everything else as an argument. It also keeps
the *identity* problem visible -- the (provider, model, dimensions) triple that
produced the collection must be the one producing the query vector, and the
dimension check below is the only part of that anyone gets for free.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Final

from qdrant_client import models

from backend.core.logging import get_logger
from retrieval.types import (
    Backend,
    Candidate,
    Filter,
    RetrievalRequest,
    chunk_id_for,
)
from retrieval.vector.collections import (
    CollectionSpec,
    PayloadField,
    signal_collection_spec,
)
from retrieval.vector.qdrant_client import VectorStore, search_params

__all__ = [
    "QueryEmbedder",
    "VectorBackend",
    "compile_filter",
]

_log = get_logger(__name__)

QueryEmbedder = Callable[[str], Awaitable[Sequence[float]]]
"""How the caller supplies the query vector.

A callable rather than a provider object so this module has no opinion about
where embeddings come from: `services/llm/embeddings.py` in production, a stored
vector in an evaluation replay, a fixed list in a test. Async because in
production it is a network call, and a synchronous one would block the event loop
for the whole of the fan-out it is supposed to run beside.
"""

_PAYLOAD_FIELDS: Final[tuple[str, ...]] = (
    PayloadField.SIGNAL_ID.value,
    PayloadField.CHUNK_INDEX.value,
)
"""The only payload keys a search reads back.

Enough to rebuild the `chunk_id`, and nothing else. Two reasons to narrow it
rather than pass `with_payload=True`: `entity_ids` can run to hundreds of ids per
chunk, so at k=100 the payload would dominate the response body; and
`docs/data-stores.md` §3.3 forbids reading anything out of Qdrant for display,
because the moment a rendered field lives only here the store stops being
rebuildable. Provenance for the citation comes from PostgreSQL, via the
`PassageResolver`.
"""


def compile_filter(filters: Filter) -> models.Filter:
    """Compile the request's metadata filter into a Qdrant filter.

    Every condition is a `must`: filters are restrictive, never scoring
    (`docs/retrieval.md` §7). A document that fails one is absent, not demoted --
    "published inside the window" is not a relevance hint that a strong cosine
    may outvote.

    The returned filter is never `None`, because `tenant_id` is always present.
    An unfiltered query against a multi-tenant collection returns another
    tenant's documents with no error anywhere, so the tenant condition is
    unconditional even in single-tenant Phase 1 where it is a constant.

    Raises:
        ValueError: no tenant, a timezone-naive bound, or an inverted window.
    """
    # `strip()` as well as truthiness, matching `retrieval/filters/metadata.py`: a
    # whitespace tenant is a configuration slip that compiles to a condition no point
    # satisfies, and an empty result set reads downstream as "nothing was published",
    # which is the one failure this layer must never fake.
    if not filters.tenant_id or not filters.tenant_id.strip():
        raise ValueError(
            "Filter.tenant_id is empty; a Qdrant query without a tenant condition "
            "returns other tenants' chunks and reports nothing wrong. The tenant "
            "is derived from the authenticated principal in backend/api/deps.py, "
            "never defaulted here."
        )

    must: list[models.Condition] = [
        models.FieldCondition(
            key=PayloadField.TENANT_ID.value,
            match=models.MatchValue(value=filters.tenant_id),
        )
    ]

    window = _time_condition(filters)
    if window is not None:
        must.append(window)

    # `MatchAny` for every set-valued dimension: Qdrant reads it as OR over the
    # values, resolved against a posting list rather than by comparing each
    # point. The enums go through `str()` so the query spells the value exactly
    # the way `ChunkPayload.to_payload()` wrote it -- a `Platform.REDDIT` repr
    # leaking into a match value would match zero points and raise nothing.
    if filters.platforms:
        must.append(
            models.FieldCondition(
                key=PayloadField.PLATFORM.value,
                match=models.MatchAny(any=sorted(str(p) for p in filters.platforms)),
            )
        )
    if filters.sources:
        must.append(
            models.FieldCondition(
                key=PayloadField.SOURCE.value,
                match=models.MatchAny(any=sorted(str(s) for s in filters.sources)),
            )
        )
    if filters.languages:
        must.append(
            models.FieldCondition(
                key=PayloadField.LANGUAGE.value,
                match=models.MatchAny(any=sorted(filters.languages)),
            )
        )
    if filters.entity_ids:
        # `entity_ids` is list-valued in the payload, and `MatchAny` against a
        # list field means "the lists intersect" -- the `any_of` semantics of
        # `docs/retrieval.md` §7. `all_of` would need one `FieldCondition` per
        # id; `retrieval.types.Filter` carries no `all_of`, so nothing here
        # pretends to offer it.
        must.append(
            models.FieldCondition(
                key=PayloadField.ENTITY_IDS.value,
                match=models.MatchAny(any=sorted(filters.entity_ids)),
            )
        )
    if filters.min_confidence is not None:
        must.append(
            models.FieldCondition(
                key=PayloadField.CONFIDENCE.value,
                range=models.Range(gte=filters.min_confidence),
            )
        )

    # Values are sorted and `must` is built in a fixed order, so the same logical
    # filter compiles to the same object every time. That is what lets a filter
    # fingerprint on a trace be compared across requests (`docs/retrieval.md` §7:
    # log the filter, so "the model hallucinated" can be told apart from "the
    # filter left three documents").
    return models.Filter(must=must)


def _condition_count(query_filter: models.Filter) -> int:
    """How many conditions the query carried, for the debug line.

    `Filter.must` is typed as one condition, a list of them, or nothing -- Qdrant
    accepts all three spellings. `compile_filter()` only ever emits the list form, but
    `len()` on the union is a type error and, more to the point, would raise at runtime
    on a filter that arrived by some other route. Selectivity is a diagnostic
    (`docs/retrieval.md` §7); it must never be the thing that breaks a query.
    """
    must = query_filter.must
    if must is None:
        return 0
    return len(must) if isinstance(must, list) else 1


def _time_condition(filters: Filter) -> models.FieldCondition | None:
    """The `published_at` window, as a half-open `[after, before)` interval.

    `gte` / `lt`, not `gte` / `lte`, per `docs/retrieval.md` §7. Under an
    inclusive upper bound a Signal published exactly at midnight belongs to two
    adjacent daily windows at once, which double-counts it in every bucketed
    trend and makes two windows that should partition the corpus overlap by one
    instant.

    Naive datetimes are refused rather than assumed to be UTC: guessing the zone
    moves the boundary by up to a day, and nothing downstream can detect that it
    happened -- the query simply covers a slightly different week than the one
    the report claims it does.
    """
    after, before = filters.published_after, filters.published_before
    for label, bound in (("published_after", after), ("published_before", before)):
        if bound is not None and bound.tzinfo is None:
            raise ValueError(
                f"Filter.{label} is timezone-naive; Qdrant range conditions "
                "compare instants, and assuming UTC here would silently shift "
                "the retrieval window."
            )
    if after is not None and before is not None and after >= before:
        # This compiles to a filter matching nothing, which reads downstream as
        # "no data for that period" -- far harder to debug than an error naming
        # both bounds.
        raise ValueError(
            f"empty retrieval window: published_after={after.isoformat()} is not "
            f"before published_before={before.isoformat()}"
        )
    if after is None and before is None:
        return None
    return models.FieldCondition(
        key=PayloadField.PUBLISHED_AT.value,
        range=models.DatetimeRange(gte=after, lt=before),
    )


class VectorBackend:
    """Dense retrieval over one Qdrant collection. Satisfies `SearchBackend`.

    Holds a client, a collection spec and the query embedder; no per-request
    state, so one instance per process serves concurrent queries.

    The spec is held rather than read from settings at call time so a
    re-embedding migration (`.env.example` §11) can point a searcher at the old
    collection while an indexer fills the new one -- code that consults global
    settings on every call cannot express two geometries at once.
    """

    backend: Backend = Backend.VECTOR

    def __init__(
        self,
        client: VectorStore,
        embed_query: QueryEmbedder,
        spec: CollectionSpec | None = None,
        *,
        hnsw_ef: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self._client = client
        self._embed_query = embed_query
        self._spec = spec or signal_collection_spec()
        self._hnsw_ef = hnsw_ef
        # A hard similarity cut-off, off by default and deliberately not wired to
        # `RETRIEVAL_MIN_SCORE`: that threshold belongs after fusion. Applied
        # here it drops candidates *before* RRF can see that two other backends
        # also found them, and cosine thresholds are corpus-dependent enough that
        # a value tuned on one collection quietly empties another.
        self._score_threshold = score_threshold

    @property
    def collection(self) -> str:
        return self._spec.name

    @property
    def spec(self) -> CollectionSpec:
        return self._spec

    async def search(
        self, request: RetrievalRequest, *, limit: int
    ) -> Sequence[Candidate]:
        """Embed the query, then ANN-search with the request's filters applied.

        Exceptions propagate. `HybridRetriever` catches them, records the backend
        as failed and continues on the remaining two (`docs/architecture.md`
        §7.3), so swallowing a Qdrant outage here would produce an empty
        candidate list indistinguishable from a hard query -- and the diagnostics
        would report a healthy run that happened to find nothing.
        """
        vector = await self._embed_query(request.query)
        return await self.search_with_vector(vector, request.filters, limit=limit)

    async def search_with_vector(
        self,
        vector: Sequence[float],
        filters: Filter | None = None,
        *,
        limit: int,
        hnsw_ef: int | None = None,
    ) -> list[Candidate]:
        """ANN search from a vector the caller already holds.

        The entry point for anything that must not pay for an embedding: the
        evaluation harness replaying stored query vectors, and more-like-this
        from a chunk's own vector.

        Raises:
            ValueError: `limit` below 1, or a vector whose length is not the
                collection's vector size.
        """
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self._assert_dimensions(vector)
        effective = filters or Filter()
        query_filter = compile_filter(effective)

        response = await self._client.query_points(
            collection_name=self._spec.name,
            query=list(vector),
            # The push-down. Everything above exists to make this one argument
            # correct; a caller that filtered the returned list instead would get
            # results that look the same and recall that has been amputated.
            query_filter=query_filter,
            search_params=search_params(
                hnsw_ef if hnsw_ef is not None else self._hnsw_ef
            ),
            limit=limit,
            with_payload=list(_PAYLOAD_FIELDS),
            # 1536 floats per hit for a vector nothing downstream reads: fusion
            # needs ranks, the reranker needs text. At k=100 this is the
            # difference between a ~600 KB response and a ~10 KB one.
            with_vectors=False,
            score_threshold=self._score_threshold,
        )

        candidates = self._to_candidates(response)
        _log.debug(
            "qdrant.search",
            collection=self._spec.name,
            limit=limit,
            returned=len(candidates),
            conditions=_condition_count(query_filter),
            tenant_id=effective.tenant_id,
        )
        return candidates

    # ------------------------------------------------------------ internals --

    def _to_candidates(self, response: Any) -> list[Candidate]:
        """Turn scored points into ranked candidates, best first.

        Rank is the *position in this list*, assigned after unusable points are
        skipped, so the ranks handed to RRF are dense and 1-based. Qdrant returns
        hits in descending score order; the score is kept only for diagnostics,
        because a cosine and a BM25 score cannot be compared or added.
        """
        # `query_points` wraps hits in a `QueryResponse`, while the deprecated
        # `search()` returned a bare list. Reading `.points` when it is there
        # keeps this working across that client change instead of failing with an
        # attribute error deep in a fan-out branch that swallows exceptions.
        points = getattr(response, "points", response) or []

        candidates: list[Candidate] = []
        seen: set[str] = set()
        skipped = 0
        for point in points:
            identified = _identify(point)
            if identified is None:
                # A point whose payload cannot name its chunk was written by
                # something other than this codebase, or by an older schema.
                # Skipping costs one result; raising would let one bad point
                # break every query that happens to match it.
                skipped += 1
                continue
            signal_id, chunk_id = identified
            if chunk_id in seen:
                # Two points for one chunk means the derived point id was
                # bypassed somewhere. Passing both on would hand RRF two terms
                # from a single backend -- Qdrant manufacturing agreement with
                # itself -- and the better-scoring one is already kept.
                skipped += 1
                continue
            seen.add(chunk_id)
            candidates.append(
                Candidate(
                    chunk_id=chunk_id,
                    backend=Backend.VECTOR,
                    rank=len(candidates) + 1,
                    raw_score=float(getattr(point, "score", 0.0) or 0.0),
                    signal_id=signal_id,
                )
            )

        if skipped:
            _log.warning(
                "qdrant.search.unusable_points",
                collection=self._spec.name,
                skipped=skipped,
                detail=(
                    "points missing signal_id/chunk_index, or duplicate chunk "
                    "ids; recall is quietly reduced until they are re-indexed"
                ),
            )
        return candidates

    def _assert_dimensions(self, vector: Sequence[float]) -> None:
        """Refuse a query vector the collection cannot compare against.

        Qdrant rejects a mismatched query too, but its error names neither the
        collection geometry nor the configuration that produced the vector -- and
        the mismatch always means one thing: the embedding model answering
        queries is not the one that built the index. Note what this cannot catch:
        a *different* model of the same dimensionality returns vectors of the
        right length and nonsense neighbours, which only the evaluation harness
        in `retrieval/evaluation/` will ever notice.
        """
        size = len(vector)
        if size != self._spec.vector_size:
            raise ValueError(
                f"query vector has {size} dimensions but collection "
                f"{self._spec.name!r} holds {self._spec.vector_size}. The query "
                "embedder is not the model this collection was built with: point "
                "EMBEDDING_PROVIDER/EMBEDDING_MODEL/EMBEDDING_DIMENSIONS back at "
                "it, or search the collection that model built."
            )


def _identify(point: Any) -> tuple[str, str] | None:
    """`(signal_id, chunk_id)` for a scored point, or `None` if it cannot be read.

    The point id is a uuid5 and therefore one-way: `retrieval/vector/indexer.py`
    derives it from the chunk id precisely so re-indexing upserts, and the price
    of that is that the chunk id has to come back out of the payload. A point
    written without `signal_id` cannot be joined to PostgreSQL at all, so it is
    unusable rather than merely unlabelled.
    """
    payload = getattr(point, "payload", None) or {}
    signal_id = payload.get(PayloadField.SIGNAL_ID.value)
    chunk_index = payload.get(PayloadField.CHUNK_INDEX.value)
    if not isinstance(signal_id, str) or not signal_id:
        return None
    # JSON round-trips small integers as ints, but a payload written through a
    # path that widened them to float would give `3.0` here; `chunk_id_for`
    # would then build "sig:3.0", which joins to nothing in OpenSearch.
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int | float):
        return None
    index = int(chunk_index)
    if index != chunk_index or index < 0:
        return None
    return signal_id, chunk_id_for(signal_id, index)
