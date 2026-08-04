"""Unit tests for `retrieval/vector/` -- the Qdrant side of hybrid retrieval.

Everything in this module is written against a fake client. `docs/testing-strategy.md`
fixes the unit suite as "no external services", and the three properties that matter
most here are properties of the *arguments we send*, not of Qdrant's ANN quality:

**Point ids must be derived and stable.** A `uuid4()` id turns every replay --
`scripts/reindex.py`, a reconciler pass, a Kafka partition rebalance -- into a second
copy of the corpus. Duplicated points score identically, so the top-k for every query
fills with the same passage repeated and it reads as a ranking bug rather than a write
bug. The stability assertions therefore pin the *literal* uuid rather than merely
comparing two calls in one process: comparing two calls only proves the function is
deterministic today, while a changed namespace constant would silently re-key the whole
collection and still pass.

**The filter must reach the client.** Asking for the 100 nearest neighbours and then
keeping the ones inside the date window is not the same operation as asking for the 100
nearest neighbours *inside* the window. Both return "results", neither errors, and the
difference is invisible until someone notices recent data is unreachable. So the fake
here does what Qdrant does -- it evaluates the filter during traversal and *then* takes
`limit` -- and refuses to run a condition it does not understand, because a fake that
ignored an unknown condition would let a broken filter pass every test in this file.

**A dimension mismatch must be refused before the request goes out.** By the time a
vector reaches this layer the embedding provider has already been billed for it. Qdrant
rejects the batch it is handed, so one bad vector fails 255 good ones and the retry
re-sends all 256; the local check turns that into one error naming the chunk and both
dimensions.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from qdrant_client import models as qmodels

from backend.core.config import VectorDistance, get_settings
from models.enums import Platform, SourceCategory
from retrieval.hybrid import SearchBackend
from retrieval.types import Backend, Filter, RetrievalRequest, chunk_id_for
from retrieval.vector.collections import (
    PAYLOAD_INDEXES,
    ChunkPayload,
    CollectionSpec,
    PayloadField,
    ensure_payload_indexes,
    ensure_signal_collection,
    signal_collection_spec,
)
from retrieval.vector.indexer import (
    POINT_ID_NAMESPACE,
    ChunkVector,
    VectorIndexer,
    point_id_for,
)
from retrieval.vector.qdrant_client import VectorStore
from retrieval.vector.search import VectorBackend, compile_filter

pytestmark = pytest.mark.unit

DIMENSIONS = 8
"""Vector width for the tests. Small on purpose: nothing here measures recall, and a
1536-float literal in every fixture would hide the assertions."""

SPEC = CollectionSpec(
    name="test_signals", vector_size=DIMENSIONS, distance=VectorDistance.COSINE
)

JAN = datetime(2026, 1, 15, tzinfo=UTC)
FEB = datetime(2026, 2, 15, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# A fake that behaves like Qdrant where it matters
# --------------------------------------------------------------------------- #


class FakePoint:
    """A scored point as `query_points` returns it."""

    def __init__(self, payload: dict[str, Any], score: float) -> None:
        self.id = payload.get(PayloadField.SIGNAL_ID.value, "?")
        self.payload = payload
        self.score = score
        self.version = 0


class FakeQueryResponse:
    """`query_points` wraps hits in an object carrying `.points`."""

    def __init__(self, points: Sequence[FakePoint]) -> None:
        self.points = list(points)


class FakeIndexInfo:
    """One entry of a collection's `payload_schema`."""

    def __init__(self, data_type: qmodels.PayloadSchemaType) -> None:
        self.data_type = data_type


class FakeCollectionInfo:
    def __init__(self, payload_schema: dict[str, FakeIndexInfo] | None = None) -> None:
        self.payload_schema = payload_schema or {}


class FakeQdrant:
    """The five methods of `VectorStore`, recording every call.

    `query_points` is not a stub returning a canned list: it evaluates the filter it
    was given against an in-memory corpus and only then truncates to `limit`, which is
    the order Qdrant uses and the order the pushdown argument depends on. A fake that
    returned the corpus unfiltered would make every assertion in `TestFilterPushdown`
    pass against a `VectorBackend` that never sent a filter at all.
    """

    def __init__(self, corpus: Sequence[FakePoint] = ()) -> None:
        self.corpus = list(corpus)  # Best-first, as an ANN index would return it.
        self.query_calls: list[dict[str, Any]] = []
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.index_calls: list[dict[str, Any]] = []
        self.collection_info = FakeCollectionInfo()

    async def query_points(
        self,
        collection_name: str,
        query: Any = None,
        *,
        query_filter: qmodels.Filter | None = None,
        search_params: qmodels.SearchParams | None = None,
        limit: int = 10,
        with_payload: Any = None,
        with_vectors: Any = None,
        **kwargs: Any,
    ) -> FakeQueryResponse:
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "query_filter": query_filter,
                "search_params": search_params,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
                **kwargs,
            }
        )
        matched = [p for p in self.corpus if matches(query_filter, p.payload)]
        return FakeQueryResponse(matched[:limit])

    async def upsert(
        self, collection_name: str, points: Sequence[Any], *, wait: bool = False, **kw: Any
    ) -> None:
        self.upsert_calls.append(
            {"collection_name": collection_name, "points": list(points), "wait": wait}
        )

    async def delete(
        self, collection_name: str, points_selector: Any, *, wait: bool = False, **kw: Any
    ) -> None:
        self.delete_calls.append(
            {
                "collection_name": collection_name,
                "points_selector": points_selector,
                "wait": wait,
            }
        )

    async def get_collection(self, collection_name: str) -> FakeCollectionInfo:
        return self.collection_info

    async def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Any = None,
        *,
        wait: bool = False,
        **kw: Any,
    ) -> None:
        self.index_calls.append(
            {
                "collection_name": collection_name,
                "field_name": field_name,
                "field_schema": field_schema,
            }
        )


def matches(query_filter: qmodels.Filter | None, payload: dict[str, Any]) -> bool:
    """Evaluate a compiled Qdrant filter against a payload dict.

    Raises on anything it cannot interpret. That is the point: a permissive fake would
    quietly treat a malformed condition as "matches everything", which is precisely the
    failure the pushdown tests exist to catch.
    """
    if query_filter is None:
        return True
    if query_filter.should or query_filter.must_not:
        raise AssertionError("filters here are conjunctive; see compile_filter()")
    return all(condition_matches(c, payload) for c in conditions(query_filter))


def condition_matches(condition: Any, payload: dict[str, Any]) -> bool:
    if not isinstance(condition, qmodels.FieldCondition):
        raise AssertionError(f"fake cannot evaluate condition {condition!r}")
    value = payload.get(condition.key)

    if condition.match is not None:
        if isinstance(condition.match, qmodels.MatchValue):
            return bool(value == condition.match.value)
        if isinstance(condition.match, qmodels.MatchAny):
            wanted = set(condition.match.any or ())
            # A list-valued payload key intersects the wanted set; a scalar is a
            # membership test. Qdrant spells both `MatchAny`, and `entity_ids` relies
            # on the list reading.
            if isinstance(value, list):
                return bool(wanted & set(value))
            return value in wanted
        raise AssertionError(f"fake cannot evaluate match {condition.match!r}")

    if isinstance(condition.range, qmodels.DatetimeRange):
        # A missing or null `published_at` fails every bound rather than passing them,
        # which is Qdrant's reading too: a point with no value for the key is not in
        # the window.
        if not isinstance(value, str):
            return False
        return within(datetime.fromisoformat(value), condition.range)

    if isinstance(condition.range, qmodels.Range):
        if not isinstance(value, int | float):
            return False
        return within(value, condition.range)

    raise AssertionError(f"fake cannot evaluate condition {condition!r}")


def conditions(query_filter: qmodels.Filter | None) -> list[Any]:
    """The `must` conditions as a plain list.

    `Filter.must` is typed as "one condition, or a list of eleven kinds of condition,
    or None", so every assertion that reads `.key` off it otherwise has to narrow a
    union that only ever holds `FieldCondition` here.
    """
    assert query_filter is not None, "a query without a filter has no tenant condition"
    must = query_filter.must
    if must is None:
        return []
    return list(must) if isinstance(must, list) else [must]


def within(value: Any, bounds: Any) -> bool:
    """All four bounds at once, each one skipped when it was not set."""
    return (
        (bounds.gte is None or value >= bounds.gte)
        and (bounds.gt is None or value > bounds.gt)
        and (bounds.lt is None or value < bounds.lt)
        and (bounds.lte is None or value <= bounds.lte)
    )


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def make_payload(
    signal_id: str = "sig_1",
    chunk_index: int = 0,
    *,
    published_at: datetime | None = JAN,
    platform: Platform = Platform.REDDIT,
    source: SourceCategory = SourceCategory.SOCIAL,
    language: str | None = "en",
    entity_ids: Sequence[str] = (),
    confidence: float = 0.8,
    tenant_id: str = "default",
) -> ChunkPayload:
    return ChunkPayload(
        signal_id=signal_id,
        chunk_index=chunk_index,
        tenant_id=tenant_id,
        platform=platform,
        source=source,
        published_at=published_at,
        language=language,
        entity_ids=tuple(entity_ids),
        confidence=confidence,
        pipeline_version=100_000,
    )


def make_chunk(
    signal_id: str = "sig_1", chunk_index: int = 0, *, width: int = DIMENSIONS
) -> ChunkVector:
    return ChunkVector(
        vector=[0.1] * width, payload=make_payload(signal_id, chunk_index)
    )


def make_point(payload: ChunkPayload, score: float) -> FakePoint:
    return FakePoint(payload.to_payload(), score)


def raw_point(score: float, **payload: Any) -> FakePoint:
    """A point with a hand-written payload, for the malformed cases.

    `tenant_id` is filled in because the fake evaluates the tenant condition like the
    real server does: a payload without it is dropped *before* `VectorBackend` ever
    sees the point, which would make a test about malformed points assert nothing.
    """
    return FakePoint({PayloadField.TENANT_ID.value: "default", **payload}, score)


async def embed_fixed(_query: str) -> list[float]:
    """A query embedder that costs nothing. The caller supplies the vector; this layer
    never constructs a provider (`retrieval/vector/search.py`)."""
    return [0.2] * DIMENSIONS


@pytest.fixture
def store() -> FakeQdrant:
    return FakeQdrant()


@pytest.fixture
def settings_reset() -> Any:
    """Clear the settings cache around a test that manipulates the environment."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Point identity
# --------------------------------------------------------------------------- #


class TestPointIdsAreDerivedAndStable:
    """The idempotency story: a re-run must upsert, never duplicate."""

    def test_id_is_pinned_to_a_literal(self) -> None:
        """The exact uuid, not just "the same twice in this process".

        A determinism check inside one process passes even when the namespace
        constant has been changed, and a changed namespace re-keys every point in
        the collection: the next indexing run inserts a full second copy of the
        corpus alongside the first and nothing reports an error.
        """
        assert point_id_for("sig_abc", 3) == "6c3ecee2-1fbf-560e-8ed9-e9bed32eeaba"
        assert str(POINT_ID_NAMESPACE) == "6979eead-d426-5311-873b-f61c66ebd53e"

    def test_id_is_a_uuid5_of_the_chunk_id(self) -> None:
        """Derived from the same join key OpenSearch uses as its `_id`."""
        import uuid

        expected = str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id_for("sig_abc", 3)))
        assert point_id_for("sig_abc", 3) == expected

    def test_chunk_index_and_signal_both_participate(self) -> None:
        assert point_id_for("sig_abc", 0) != point_id_for("sig_abc", 1)
        assert point_id_for("sig_abc", 3) != point_id_for("sig_def", 3)

    async def test_a_second_index_run_reuses_the_same_ids(self, store: FakeQdrant) -> None:
        """The property that makes a replay an upsert instead of a second corpus."""
        indexer = VectorIndexer(store, SPEC)
        chunks = [make_chunk("sig_1", i) for i in range(3)]

        await indexer.index_chunks(chunks)
        await indexer.index_chunks([make_chunk("sig_1", i) for i in range(3)])

        first = [p.id for p in store.upsert_calls[0]["points"]]
        second = [p.id for p in store.upsert_calls[1]["points"]]
        assert first == second, "a replay wrote different point ids and duplicated the corpus"

    async def test_point_id_matches_the_pure_function(self, store: FakeQdrant) -> None:
        """The id on the wire is the one `delete_chunks()` re-derives.

        If these ever disagree, deleting a chunk by id silently deletes nothing and
        the point stays searchable forever.
        """
        indexer = VectorIndexer(store, SPEC)
        await indexer.index_chunks([make_chunk("sig_7", 2)])
        (point,) = store.upsert_calls[0]["points"]
        assert point.id == point_id_for("sig_7", 2)


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #


class TestIndexChunks:
    async def test_batches_at_the_configured_size(self, store: FakeQdrant) -> None:
        """Five points at a batch size of two is 2 + 2 + 1, not one 5 MB request."""
        indexer = VectorIndexer(store, SPEC, batch_size=2)
        outcome = await indexer.index_chunks(make_chunk("sig_1", i) for i in range(5))

        assert [len(c["points"]) for c in store.upsert_calls] == [2, 2, 1]
        assert (outcome.points, outcome.batches) == (5, 3)
        assert outcome.collection == SPEC.name
        assert outcome.chunk_ids == tuple(chunk_id_for("sig_1", i) for i in range(5))

    async def test_no_request_when_there_is_nothing_to_write(self, store: FakeQdrant) -> None:
        outcome = await VectorIndexer(store, SPEC).index_chunks([])
        assert store.upsert_calls == []
        assert (outcome.points, outcome.batches) == (0, 0)

    async def test_wait_defaults_to_false(self, store: FakeQdrant) -> None:
        """Qdrant applies upserts asynchronously and nothing here reads its own write."""
        await VectorIndexer(store, SPEC).index_chunks([make_chunk()])
        assert store.upsert_calls[0]["wait"] is False

    async def test_the_payload_written_is_the_closed_schema(self, store: FakeQdrant) -> None:
        """Only `PayloadField` keys reach the wire -- Qdrant is a filter index, not a
        document store (`docs/data-stores.md` §3.3)."""
        await VectorIndexer(store, SPEC).index_chunks([make_chunk("sig_1", 4)])
        (point,) = store.upsert_calls[0]["points"]
        assert set(point.payload) == {f.value for f in PayloadField}
        assert point.payload[PayloadField.SIGNAL_ID.value] == "sig_1"
        assert point.payload[PayloadField.CHUNK_INDEX.value] == 4

    async def test_a_duplicate_chunk_in_one_call_is_refused(self, store: FakeQdrant) -> None:
        """Two vectors for one chunk would resolve to whichever landed last."""
        indexer = VectorIndexer(store, SPEC)
        with pytest.raises(ValueError, match="appears twice"):
            await indexer.index_chunks([make_chunk("sig_1", 0), make_chunk("sig_1", 0)])

    def test_a_zero_batch_size_is_refused(self, store: FakeQdrant) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            VectorIndexer(store, SPEC, batch_size=0)


class TestDimensionMismatchIsRefused:
    """The check that runs before the request, not after the provider was billed."""

    async def test_a_wrong_width_vector_never_reaches_the_client(
        self, store: FakeQdrant
    ) -> None:
        indexer = VectorIndexer(store, SPEC)
        wrong = ChunkVector(vector=[0.1] * (DIMENSIONS + 1), payload=make_payload())

        with pytest.raises(ValueError) as excinfo:
            await indexer.index_chunks([wrong])

        message = str(excinfo.value)
        assert str(DIMENSIONS + 1) in message and str(DIMENSIONS) in message, (
            f"the error must name both widths or the operator cannot tell which side "
            f"is wrong: {message}"
        )
        assert store.upsert_calls == [], "a mismatched batch was sent anyway"

    async def test_one_bad_vector_fails_the_call_before_the_good_ones_ship(
        self, store: FakeQdrant
    ) -> None:
        """Checked up front so the retry does not re-send 255 good points to fail again.

        A batch that has already been flushed stays flushed -- the write is an upsert,
        so re-running after fixing the embedder converges rather than duplicating.
        """
        indexer = VectorIndexer(store, SPEC, batch_size=2)
        chunks = [
            make_chunk("sig_1", 0),
            make_chunk("sig_1", 1),
            ChunkVector(vector=[0.1] * 3, payload=make_payload("sig_1", 2)),
        ]
        with pytest.raises(ValueError, match="dimensional vector"):
            await indexer.index_chunks(chunks)
        assert len(store.upsert_calls) == 1, "the bad point's batch must not be sent"

    async def test_a_wrong_width_query_vector_is_refused(self, store: FakeQdrant) -> None:
        backend = VectorBackend(store, embed_fixed, SPEC)
        with pytest.raises(ValueError) as excinfo:
            await backend.search_with_vector([0.1] * 3, Filter(), limit=5)

        assert "3 dimensions" in str(excinfo.value)
        assert store.query_calls == [], "a mismatched query was sent to Qdrant"

    async def test_an_embedder_of_the_wrong_model_is_caught_on_search(
        self, store: FakeQdrant
    ) -> None:
        """The realistic form: EMBEDDING_MODEL changed, the collection did not."""

        async def embed_wrong(_query: str) -> list[float]:
            return [0.2] * 1024

        backend = VectorBackend(store, embed_wrong, SPEC)
        with pytest.raises(ValueError, match="1024"):
            await backend.search(RetrievalRequest(query="anything"), limit=5)


class TestDeletion:
    """Erasure and canonical-election demotion are the same operation."""

    async def test_delete_signal_selects_by_payload_not_by_id_list(
        self, store: FakeQdrant
    ) -> None:
        """The caller does not know how many chunks the Signal had.

        A caller re-deriving ids from today's chunk count leaves the tail of a longer
        previous chunking behind, and those orphans stay searchable forever.
        """
        await VectorIndexer(store, SPEC).delete_signal("sig_1")

        (call,) = store.delete_calls
        selector = call["points_selector"]
        assert isinstance(selector, qmodels.FilterSelector)
        (condition,) = conditions(selector.filter)
        assert condition.key == PayloadField.SIGNAL_ID.value
        assert condition.match.value == "sig_1"

    async def test_delete_signal_can_trim_a_tail_after_a_re_chunk(
        self, store: FakeQdrant
    ) -> None:
        await VectorIndexer(store, SPEC).delete_signal("sig_1", from_chunk_index=5)

        (call,) = store.delete_calls
        sent = conditions(call["points_selector"].filter)
        keys = {c.key for c in sent}
        assert keys == {PayloadField.SIGNAL_ID.value, PayloadField.CHUNK_INDEX.value}
        (tail,) = [c for c in sent if c.key == PayloadField.CHUNK_INDEX.value]
        assert tail.range.gte == 5

    async def test_delete_signals_is_one_request_for_a_whole_cluster(
        self, store: FakeQdrant
    ) -> None:
        """A dedup run demotes 400 losers at once; 400 round trips is not the design."""
        await VectorIndexer(store, SPEC).delete_signals(["sig_1", "sig_2", "sig_1", ""])

        (call,) = store.delete_calls
        (condition,) = conditions(call["points_selector"].filter)
        assert sorted(condition.match.any) == ["sig_1", "sig_2"]

    async def test_deleting_nothing_sends_nothing(self, store: FakeQdrant) -> None:
        """An empty `MatchAny` compiles to a filter that matches the whole collection."""
        await VectorIndexer(store, SPEC).delete_signals([])
        await VectorIndexer(store, SPEC).delete_chunks([])
        assert store.delete_calls == []

    async def test_delete_chunks_uses_the_derived_point_ids(self, store: FakeQdrant) -> None:
        await VectorIndexer(store, SPEC).delete_chunks([chunk_id_for("sig_1", 2)])
        (call,) = store.delete_calls
        selector = call["points_selector"]
        assert isinstance(selector, qmodels.PointIdsList)
        assert selector.points == [point_id_for("sig_1", 2)]


# --------------------------------------------------------------------------- #
# Collection definition
# --------------------------------------------------------------------------- #


class TestCollectionSpec:
    def test_spec_reads_geometry_from_settings(
        self, monkeypatch: pytest.MonkeyPatch, settings_reset: None
    ) -> None:
        """Vector size from EMBEDDING_DIMENSIONS, metric from QDRANT_DISTANCE."""
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "1024")
        monkeypatch.setenv("QDRANT_DISTANCE", "dot")
        monkeypatch.setenv("QDRANT_COLLECTION", "somewhere_else")
        get_settings.cache_clear()

        spec = signal_collection_spec()
        assert spec.vector_size == 1024
        assert spec.distance is VectorDistance.DOT
        assert spec.name == "somewhere_else"
        assert spec.vectors_config().distance == qmodels.Distance.DOT

    def test_overrides_exist_for_a_re_embedding_migration(self) -> None:
        """The new collection is built alongside the live one, at a different width."""
        spec = signal_collection_spec(name="v2", vector_size=1024)
        assert (spec.name, spec.vector_size) == ("v2", 1024)

    def test_a_zero_width_collection_is_refused(self) -> None:
        with pytest.raises(ValueError, match="vector_size"):
            CollectionSpec(name="x", vector_size=0, distance=VectorDistance.COSINE)

    def test_hnsw_parameters_are_stated_not_inherited(self) -> None:
        """Pinned so a Qdrant upgrade cannot quietly change recall on a live index."""
        config = SPEC.hnsw_config()
        assert (config.m, config.ef_construct) == (16, 128)


class TestPayloadIndexes:
    def test_every_filterable_field_is_indexed(self) -> None:
        """An unindexed payload filter is correct and O(collection).

        It never errors, so the only symptom at 10M points is "search got slow", which
        reads like a slow disk and sends the investigation somewhere else entirely.
        """
        assert set(PAYLOAD_INDEXES) == set(PayloadField)

    def test_published_at_is_a_datetime_index_not_a_keyword(self) -> None:
        """A keyword index cannot answer a range, so the window query falls back to a
        scan while the collection info still reports the field as indexed."""
        assert (
            PAYLOAD_INDEXES[PayloadField.PUBLISHED_AT]
            == qmodels.PayloadSchemaType.DATETIME
        )
        assert PAYLOAD_INDEXES[PayloadField.CONFIDENCE] == qmodels.PayloadSchemaType.FLOAT
        assert PAYLOAD_INDEXES[PayloadField.ENTITY_IDS] == qmodels.PayloadSchemaType.KEYWORD

    async def test_missing_indexes_are_created(self, store: FakeQdrant) -> None:
        created = await ensure_payload_indexes(store, SPEC)
        assert set(created) == set(PayloadField)
        assert {c["field_name"] for c in store.index_calls} == {f.value for f in PayloadField}
        assert all(c["collection_name"] == SPEC.name for c in store.index_calls)

    async def test_existing_indexes_are_not_rebuilt(self, store: FakeQdrant) -> None:
        """Every replica calls this at boot; re-creating serialises them behind rebuilds."""
        store.collection_info = FakeCollectionInfo(
            {
                field.value: FakeIndexInfo(schema)
                for field, schema in PAYLOAD_INDEXES.items()
            }
        )
        created = await ensure_payload_indexes(store, SPEC)
        assert created == []
        assert store.index_calls == []

    async def test_a_wrongly_typed_index_is_reported_not_replaced(
        self, store: FakeQdrant
    ) -> None:
        """Dropping an index in place removes the one live queries are using.

        The right response is an operator rebuilding from PostgreSQL, not a boot path
        deoptimising every concurrent query.
        """
        store.collection_info = FakeCollectionInfo(
            {PayloadField.PUBLISHED_AT.value: FakeIndexInfo(qmodels.PayloadSchemaType.KEYWORD)}
        )
        created = await ensure_payload_indexes(store, SPEC)

        assert PayloadField.PUBLISHED_AT not in created
        assert PayloadField.PUBLISHED_AT.value not in {
            c["field_name"] for c in store.index_calls
        }

    async def test_ensure_signal_collection_checks_geometry_then_indexes(
        self, store: FakeQdrant, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Geometry stays in the L1k kernel; a second implementation would disagree."""
        import retrieval.vector.collections as mod

        seen: list[tuple[str, int, VectorDistance]] = []

        async def fake_ensure(
            name: str, *, vector_size: int, distance: VectorDistance
        ) -> bool:
            seen.append((name, vector_size, distance))
            return True

        monkeypatch.setattr(mod, "ensure_collection_geometry", fake_ensure)
        resolved = await ensure_signal_collection(store, SPEC)

        assert seen == [(SPEC.name, DIMENSIONS, VectorDistance.COSINE)]
        assert resolved is SPEC
        assert store.index_calls, "payload indexes were never created"


class TestChunkPayload:
    def test_entity_ids_is_always_a_list(self) -> None:
        """A bare string and a one-element list are different to `MatchAny`, and the
        difference only shows up as a filter that matches nothing."""
        payload = make_payload(entity_ids=("ent_1",)).to_payload()
        assert payload[PayloadField.ENTITY_IDS.value] == ["ent_1"]
        assert make_payload().to_payload()[PayloadField.ENTITY_IDS.value] == []

    def test_published_at_is_rfc3339(self) -> None:
        payload = make_payload(published_at=JAN).to_payload()
        assert payload[PayloadField.PUBLISHED_AT.value] == JAN.isoformat()

    def test_a_naive_published_at_is_refused(self) -> None:
        """Guessing the zone can move a Signal across a window boundary by a day."""
        naive = make_payload(published_at=datetime(2026, 1, 15))
        with pytest.raises(ValueError, match="timezone-naive"):
            naive.to_payload()

    def test_pipeline_version_is_the_ordinal_not_the_string(self) -> None:
        """Compared as text, '1.10.0' sorts below '1.9.0' and staleness inverts."""
        assert isinstance(make_payload().to_payload()[PayloadField.PIPELINE_VERSION.value], int)


# --------------------------------------------------------------------------- #
# Search: the pushdown
# --------------------------------------------------------------------------- #


class TestFilterPushdown:
    """The filter must reach Qdrant, because filtering afterwards silently loses recall."""

    def _corpus(self) -> list[FakePoint]:
        """Five high-scoring February chunks ahead of five January ones.

        Shaped so that "nearest 5, then keep January" and "nearest 5 January" give
        visibly different answers: the first yields nothing at all.
        """
        recent = [
            make_point(make_payload(f"sig_feb_{i}", 0, published_at=FEB), 0.99 - i * 0.01)
            for i in range(5)
        ]
        older = [
            make_point(make_payload(f"sig_jan_{i}", 0, published_at=JAN), 0.50 - i * 0.01)
            for i in range(5)
        ]
        return [*recent, *older]

    async def test_the_window_is_applied_before_the_limit(self) -> None:
        """The whole argument for pushdown, as an executable assertion.

        The fake filters during traversal and truncates afterwards, exactly as Qdrant
        does. If `VectorBackend` sent no filter -- or filtered the response instead --
        the five February chunks would fill `limit=5` and this returns zero January
        results while reporting a perfectly healthy query.
        """
        store = FakeQdrant(self._corpus())
        backend = VectorBackend(store, embed_fixed, SPEC)

        candidates = await backend.search_with_vector(
            [0.2] * DIMENSIONS,
            Filter(published_before=FEB - timedelta(days=1)),
            limit=5,
        )

        assert len(candidates) == 5, "the date window was not pushed into the search"
        assert all(c.signal_id.startswith("sig_jan_") for c in candidates)

    async def test_the_limit_reaches_the_client_unmodified(self) -> None:
        """No secret over-fetch. `k_vector` is what fusion was tuned against, and an
        over-fetch that is trimmed here would make the two numbers disagree."""
        store = FakeQdrant(self._corpus())
        await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=3
        )
        assert store.query_calls[0]["limit"] == 3

    async def test_every_filter_dimension_becomes_a_condition(self) -> None:
        store = FakeQdrant()
        backend = VectorBackend(store, embed_fixed, SPEC)
        filters = Filter(
            published_after=JAN,
            published_before=FEB,
            platforms=frozenset({Platform.REDDIT, Platform.X}),
            sources=frozenset({SourceCategory.NEWS}),
            languages=frozenset({"en"}),
            entity_ids=frozenset({"ent_1", "ent_2"}),
            min_confidence=0.4,
            tenant_id="acme",
        )

        await backend.search_with_vector([0.2] * DIMENSIONS, filters, limit=10)

        sent = store.query_calls[0]["query_filter"]
        assert {c.key for c in conditions(sent)} == {
            PayloadField.TENANT_ID.value,
            PayloadField.PUBLISHED_AT.value,
            PayloadField.PLATFORM.value,
            PayloadField.SOURCE.value,
            PayloadField.LANGUAGE.value,
            PayloadField.ENTITY_IDS.value,
            PayloadField.CONFIDENCE.value,
        }
        assert sent.must_not is None and sent.should is None, "filters are restrictive"

    async def test_the_tenant_condition_is_never_optional(self) -> None:
        """An unfiltered query on a shared collection returns another tenant's chunks
        and reports nothing wrong."""
        store = FakeQdrant()
        await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, None, limit=5
        )
        sent = store.query_calls[0]["query_filter"]
        assert sent is not None
        (tenant,) = [c for c in conditions(sent) if c.key == PayloadField.TENANT_ID.value]
        assert tenant.match.value == "default"

    async def test_tenant_isolation_is_observable_end_to_end(self) -> None:
        """Not just "a condition was sent" -- the other tenant's chunk must not come back."""
        store = FakeQdrant(
            [
                make_point(make_payload("sig_other", 0, tenant_id="other"), 0.99),
                make_point(make_payload("sig_mine", 0, tenant_id="acme"), 0.10),
            ]
        )
        candidates = await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(tenant_id="acme"), limit=10
        )
        assert [c.signal_id for c in candidates] == ["sig_mine"]

    async def test_entity_ids_intersect_rather_than_require_all(self) -> None:
        """A short social post mentions one entity; requiring all of them returns none."""
        store = FakeQdrant(
            [make_point(make_payload("sig_1", 0, entity_ids=("ent_1",)), 0.9)]
        )
        candidates = await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS,
            Filter(entity_ids=frozenset({"ent_1", "ent_2"})),
            limit=10,
        )
        assert [c.signal_id for c in candidates] == ["sig_1"]

    async def test_the_window_is_half_open(self) -> None:
        """`[after, before)`. Under an inclusive upper bound a midnight Signal belongs
        to two adjacent windows and every bucketed trend double-counts it."""
        boundary = make_payload("sig_boundary", 0, published_at=FEB)
        store = FakeQdrant([make_point(boundary, 0.9)])
        backend = VectorBackend(store, embed_fixed, SPEC)

        excluded = await backend.search_with_vector(
            [0.2] * DIMENSIONS, Filter(published_before=FEB), limit=10
        )
        included = await backend.search_with_vector(
            [0.2] * DIMENSIONS, Filter(published_after=FEB), limit=10
        )
        assert excluded == []
        assert [c.signal_id for c in included] == ["sig_boundary"]

    async def test_only_the_join_keys_are_read_back(self) -> None:
        """`entity_ids` can run to hundreds per chunk, and nothing here is for display:
        provenance for the citation comes from PostgreSQL."""
        store = FakeQdrant()
        await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=5
        )
        call = store.query_calls[0]
        assert set(call["with_payload"]) == {
            PayloadField.SIGNAL_ID.value,
            PayloadField.CHUNK_INDEX.value,
        }
        assert call["with_vectors"] is False, "1536 floats per hit that nothing reads"

    async def test_search_params_carry_the_hnsw_budget(self) -> None:
        store = FakeQdrant()
        await VectorBackend(store, embed_fixed, SPEC, hnsw_ef=256).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=5
        )
        params = store.query_calls[0]["search_params"]
        assert params.hnsw_ef == 256
        assert params.exact is False, "exact search is linear in collection size"

    async def test_a_zero_limit_is_refused(self) -> None:
        store = FakeQdrant()
        backend = VectorBackend(store, embed_fixed, SPEC)
        with pytest.raises(ValueError, match="limit"):
            await backend.search_with_vector([0.2] * DIMENSIONS, Filter(), limit=0)


class TestFilterCompilationGuards:
    @pytest.mark.parametrize("tenant", ["", "   "])
    def test_a_blank_tenant_is_refused(self, tenant: str) -> None:
        """Whitespace counts as blank, as it does in the shared compiler.

        A `"   "` tenant matches no point, so it does not leak -- it silently empties
        the corpus, which reads downstream as "nothing was published in that window".
        """
        with pytest.raises(ValueError, match="tenant"):
            compile_filter(Filter(tenant_id=tenant))

    def test_a_naive_bound_is_refused(self) -> None:
        """Assuming UTC shifts the boundary by the caller's offset, undetectably."""
        with pytest.raises(ValueError, match="timezone-naive"):
            compile_filter(Filter(published_after=datetime(2026, 1, 1)))

    def test_an_inverted_window_is_refused(self) -> None:
        """It compiles to a filter matching nothing, which reads downstream as "no data
        for that period" -- far harder to debug than an error naming both bounds."""
        with pytest.raises(ValueError, match="empty retrieval window"):
            compile_filter(Filter(published_after=FEB, published_before=JAN))

    def test_enum_values_are_spelled_the_way_the_payload_wrote_them(self) -> None:
        """A `Platform.REDDIT` repr leaking into a match value matches zero points and
        raises nothing."""
        compiled = compile_filter(Filter(platforms=frozenset({Platform.REDDIT})))
        (platform,) = [
            c for c in conditions(compiled) if c.key == PayloadField.PLATFORM.value
        ]
        assert platform.match.any == ["reddit"]

    def test_compilation_is_deterministic(self) -> None:
        """`frozenset` iteration order varies between processes; a filter fingerprint
        that changed on restart would be useless on a trace."""
        filters = Filter(
            platforms=frozenset({Platform.REDDIT, Platform.X, Platform.YOUTUBE}),
            entity_ids=frozenset({"e3", "e1", "e2"}),
        )
        assert compile_filter(filters) == compile_filter(filters)

    def test_it_agrees_with_the_shared_filter_compiler(self) -> None:
        """Drift guard against `retrieval/filters/metadata.compile_qdrant`.

        Two Qdrant filter compilers exist in this layer: this one, on the live search
        path, and the shared one that also emits the OpenSearch and Cypher dialects. If
        they disagree about a window bound, the keyword and vector backends answer the
        same request over slightly different corpora and reciprocal rank fusion cannot
        notice -- the symptom is a chunk that scores as though only one backend found
        it. Condition *order* is not compared, only the set of conditions, because the
        two build `must` in different orders and that has no effect on the query.
        """
        from retrieval.filters.metadata import compile_qdrant

        filters = Filter(
            published_after=JAN,
            published_before=FEB,
            platforms=frozenset({Platform.REDDIT}),
            sources=frozenset({SourceCategory.NEWS}),
            languages=frozenset({"en"}),
            entity_ids=frozenset({"ent_1", "ent_2"}),
            min_confidence=0.4,
            tenant_id="acme",
        )
        mine = sorted(c.model_dump_json() for c in conditions(compile_filter(filters)))
        shared = sorted(c.model_dump_json() for c in conditions(compile_qdrant(filters)))
        assert mine == shared


# --------------------------------------------------------------------------- #
# Search: candidates
# --------------------------------------------------------------------------- #


class TestCandidates:
    async def test_ranks_are_dense_and_one_based(self) -> None:
        """RRF consumes ranks. A zero rank divides differently and a gap in the
        sequence quietly demotes everything after it."""
        store = FakeQdrant(
            [make_point(make_payload(f"sig_{i}", 0), 0.9 - i * 0.1) for i in range(3)]
        )
        candidates = await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=10
        )
        assert [c.rank for c in candidates] == [1, 2, 3]
        assert [c.backend for c in candidates] == [Backend.VECTOR] * 3
        assert [c.signal_id for c in candidates] == ["sig_0", "sig_1", "sig_2"]

    async def test_the_chunk_id_is_rebuilt_from_the_payload(self) -> None:
        """The point id is a uuid5 and therefore one-way; the join key has to come back
        out of the payload or nothing can be resolved against PostgreSQL."""
        store = FakeQdrant([make_point(make_payload("sig_1", 7), 0.9)])
        (candidate,) = await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=10
        )
        assert candidate.chunk_id == chunk_id_for("sig_1", 7)

    async def test_the_raw_score_is_kept_for_diagnostics(self) -> None:
        store = FakeQdrant([make_point(make_payload("sig_1", 0), 0.83)])
        (candidate,) = await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=10
        )
        assert candidate.raw_score == pytest.approx(0.83)

    async def test_unusable_points_are_skipped_not_fatal(self) -> None:
        """A point written by an older schema costs one result; raising would let it
        break every query it happens to match."""
        store = FakeQdrant(
            [
                raw_point(0.99),
                raw_point(0.98, signal_id="sig_1"),
                raw_point(0.97, signal_id="sig_2", chunk_index=-1),
                raw_point(0.96, signal_id="", chunk_index=0),
                raw_point(0.95, signal_id="sig_3", chunk_index="4"),
                make_point(make_payload("sig_ok", 0), 0.5),
            ]
        )
        candidates = await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=10
        )
        assert [c.signal_id for c in candidates] == ["sig_ok"]
        assert [c.rank for c in candidates] == [1], "ranks must close over the gap"

    async def test_a_duplicate_chunk_is_returned_once(self) -> None:
        """Two points for one chunk would hand RRF two terms from one backend --
        Qdrant manufacturing agreement with itself."""
        payload = make_payload("sig_1", 0)
        store = FakeQdrant([make_point(payload, 0.9), make_point(payload, 0.4)])
        candidates = await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=10
        )
        assert len(candidates) == 1
        assert candidates[0].raw_score == pytest.approx(0.9), "the better hit is kept"

    async def test_a_float_chunk_index_still_joins(self) -> None:
        """A payload widened to `3.0` somewhere would build "sig:3.0", which joins to
        nothing in OpenSearch."""
        store = FakeQdrant([raw_point(0.9, signal_id="sig_1", chunk_index=3.0)])
        (candidate,) = await VectorBackend(store, embed_fixed, SPEC).search_with_vector(
            [0.2] * DIMENSIONS, Filter(), limit=10
        )
        assert candidate.chunk_id == chunk_id_for("sig_1", 3)


class TestBackendContract:
    def test_it_satisfies_the_search_backend_protocol(self, store: FakeQdrant) -> None:
        """`HybridRetriever` keys its fan-out on `backend`; a missing attribute would
        surface inside a gather that swallows exceptions."""
        backend = VectorBackend(store, embed_fixed, SPEC)
        assert isinstance(backend, SearchBackend)
        assert backend.backend is Backend.VECTOR

    def test_the_fake_satisfies_the_vector_store_protocol(self, store: FakeQdrant) -> None:
        """If the fake and the real client diverge, these tests stop meaning anything."""
        assert isinstance(store, VectorStore)

    async def test_search_uses_the_supplied_embedder(self) -> None:
        """The query vector arrives from the caller: this layer constructs no provider,
        holds no API key, and makes no network call."""
        seen: list[str] = []

        async def embed(query: str) -> list[float]:
            seen.append(query)
            return [0.5] * DIMENSIONS

        store = FakeQdrant()
        await VectorBackend(store, embed, SPEC).search(
            RetrievalRequest(query="datadog outage"), limit=7
        )
        assert seen == ["datadog outage"]
        assert store.query_calls[0]["query"] == [0.5] * DIMENSIONS

    async def test_a_client_failure_propagates(self) -> None:
        """`HybridRetriever` records the backend as failed and degrades to the other
        two (`docs/architecture.md` §7.3). Swallowing here would produce an empty list
        indistinguishable from a hard query, and diagnostics would report a healthy run.
        """

        class Broken(FakeQdrant):
            async def query_points(self, *args: Any, **kwargs: Any) -> Any:
                raise ConnectionError("qdrant is down")

        backend = VectorBackend(Broken(), embed_fixed, SPEC)
        with pytest.raises(ConnectionError):
            await backend.search(RetrievalRequest(query="anything"), limit=5)
