"""Fan-out into Qdrant and OpenSearch, and the stamps that make it recoverable.

Consumes `omnisense.signals.enriched` -- step 6 of `docs/data-stores.md` §5.1 --
and performs steps 6 and 8 for the two search stores. Neo4j is step 7 and belongs
to `workers/graph_worker.py`, in its own consumer group, so that a Neo4j outage
accrues lag on the graph topic while indexing keeps up (`docs/architecture.md`
§7.3).

Why the Signal is re-read from PostgreSQL
-----------------------------------------
`SignalEnrichedEvent` carries identity and version, never content
(`services/events/schemas.py`), so this worker's first act is a `SELECT` against
the commit point. That is not an accident of the event's shape -- it is the
reason for it. A consumer that indexed content taken off the topic would index
whatever was true when the message was produced, while the row may have been
reprocessed twice since; the vector and the row would disagree, and no
reconciler could detect it because reconcilers compare *against* PostgreSQL. A
row that has moved on is therefore indexed at its current version, which is
exactly right: the derived stores converge on PostgreSQL, never on the log.

The index-state columns are the whole recovery story
-----------------------------------------------------
`indexed_vector_at` and `indexed_keyword_at` are stamped **after** the
corresponding store acknowledges, one column per store, in separate statements.
A crash between the PostgreSQL commit and the Qdrant upsert therefore leaves the
column `NULL`, the sweep in `docs/data-stores.md` §6 finds it, the message is
re-driven, and the upsert -- idempotent by derived point id -- redoes the work
harmlessly. Stamping first would invert that: the row would claim to be indexed,
every reconciler looks for `NULL`, and the Signal would be permanently missing
from search with nothing able to notice. `workers/runtime/index_state.py` holds
the statement and the reasoning.

The two stores are stamped independently because they fail independently. An
OpenSearch write that succeeds while Qdrant is down must keep its stamp: forcing
both back to `NULL` because one failed would re-index the healthy store on every
redelivery, which is how a Qdrant outage becomes an OpenSearch write storm.

Only canonical members are indexed
-----------------------------------
`docs/signal-model.md` §4.3: a press release that appeared on six platforms is
one dedup cluster, and only its canonical member is retrievable. So a Signal that
is `duplicate` or `quarantined` is *removed* from both stores rather than
skipped -- it may have been canonical when it was last indexed, and a demotion
that only stops writing leaves the loser searchable forever. The removal is
followed by a stamp, which reads oddly until you notice what the alternative
does: leaving the columns `NULL` for a Signal that must never be indexed parks it
in the sweeper's backlog permanently, and the sweeper re-drives it forever. The
stamp means "the derived stores agree with PostgreSQL about this Signal", which
is true of a Signal correctly absent from both.

Where the vectors come from
----------------------------
`retrieval/vector/indexer.py` never embeds; it writes vectors it is handed. The
vectors computed during ingestion do not survive the process (stage 6 hands them
to a `VectorSink`, and the default composition wires none), and they deliberately
never travel on the bus -- a 1536-dimension vector per chunk would multiply the
size of the system's hottest message for no reader's benefit
(`services/signal_engine/embeddings.py`). Something therefore has to produce them
here, and that is `ChunkVectorSource`: a port whose production implementation
re-embeds the chunk texts. The cost is real, it is the cost stage 6 already
documents, and it is injectable precisely so a deployment that stages vectors
somewhere durable can hand them over instead of paying it.

A Signal whose `embeddings` list is empty is **not** embedded here. An empty list
means stage 6 degraded (rate limit, provider outage) or the body had nothing to
chunk. The first case belongs to `workers/embedding_worker.py`, which owns the
`partial` backlog, the attempt counter and the eventual quarantine; re-embedding
it inline would race that sweeper and bypass its accounting. So the keyword
document is written, `indexed_keyword_at` is stamped, `indexed_vector_at` stays
`NULL`, and the backlog query finds it -- the degraded path expressed entirely
through the columns that were built for it.

Layer note: `workers/` (L4).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Final, Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from models.lineage import pipeline_version_ordinal
from models.orm.mixins import DEFAULT_TENANT
from models.orm.signal import SignalRow
from models.signal import SignalView
from retrieval.chunking.splitter import Chunk, split_text, strategy_for
from retrieval.keyword.index import ChunkDocument
from retrieval.keyword.opensearch_client import KeywordIndexer
from retrieval.vector.collections import ChunkPayload
from retrieval.vector.indexer import ChunkVector, VectorIndexer
from services.events.consumer import ConsumedMessage
from services.events.schemas import SignalEnrichedEvent
from services.events.topics import TopicRole
from workers.runtime.base_worker import ConsumerWorker, run_worker
from workers.runtime.health import DependencyProbe
from workers.runtime.index_state import IndexState, stamp_index_state

__all__ = [
    "WORKER_NAME",
    "ChunkVectorSource",
    "EmbeddingChunkVectors",
    "IndexingWorker",
    "build_worker",
    "chunk_payloads",
    "chunks_for",
    "keyword_documents",
]

logger = get_logger(__name__)

WORKER_NAME: Final = "indexing"
"""Consumer-group suffix and metric label. Stable: renaming it rejoins the group
under a new name, which resets every committed offset to `auto_offset_reset` and
either replays or skips the whole retention window."""


class ChunkVectorSource(Protocol):
    """Where the worker obtains vectors for a Signal's chunks.

    Returns one vector per chunk, in order. Order is the contract: the caller
    zips the result against the chunk list, so a source that returned vectors in
    a different order would attach each vector to the wrong span. Nothing
    downstream could detect that -- search simply gets worse, for reasons nobody
    reports as a bug.
    """

    async def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class EmbeddingChunkVectors:
    """The production `ChunkVectorSource`: embed the chunk texts.

    This is the "second embedding call" that `services/signal_engine/embeddings.py`
    names as a known cost of not wiring a durable `VectorSink`. A class rather
    than a closure so the provider it wraps is visible in a composition root and
    replaceable there, and so the batch size is bounded here rather than being
    whatever a provider happens to accept.

    Batching is deliberately redundant with any batching inside the provider: the
    `EmbeddingProvider` protocol promises only "embed every text, in order", so a
    self-hosted or fake implementation is free to forward a 400-chunk paper as
    one request that then fails whole.
    """

    def __init__(self, provider: Any, *, batch_size: int | None = None) -> None:
        self._provider = provider
        self._batch_size = batch_size or get_settings().embedding.batch_size

    async def __call__(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[Sequence[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            produced = await self._provider.embed(batch)
            if len(produced) != len(batch):
                # A short result shifts every later vector onto the wrong chunk.
                # Raising costs one redelivery; accepting it silently mislabels
                # the corpus in a way only a human reading search results finds.
                raise ValueError(
                    f"embedding provider returned {len(produced)} vectors for "
                    f"{len(batch)} chunks; vectors are matched to chunks by "
                    "position, so a short result attaches each vector to the "
                    "wrong span"
                )
            vectors.extend(produced)
        return vectors


class IndexingWorker(ConsumerWorker):
    """Writes Qdrant and OpenSearch for one Signal, then stamps what it wrote.

    Every collaborator is injected. The worker constructs no client, so the unit
    suite drives the real sequence -- read row, chunk, index, stamp -- against
    in-memory SQLite and two in-memory stores, with the ordering production has.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        vector_indexer: VectorIndexer | None = None,
        keyword_indexer: KeywordIndexer | None = None,
        vector_source: ChunkVectorSource | None = None,
        tenant_id: str = DEFAULT_TENANT,
        settings: Settings | None = None,
        name: str = WORKER_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            topics=[TopicRole.SIGNALS],
            settings=settings,
            **kwargs,
        )
        self._session_factory = session_factory
        self._vector_indexer = vector_indexer
        self._keyword_indexer = keyword_indexer
        self._vector_source = vector_source
        self._tenant_id = tenant_id
        if vector_indexer is not None and vector_source is None:
            # A vector indexer with nowhere to get vectors writes nothing, and
            # `indexed_vector_at` would stay NULL for the whole corpus while the
            # worker reported healthy batches. Saying so at construction beats
            # discovering it as "vector search returns nothing" weeks later.
            logger.warning(
                "indexing.vector_source_missing",
                reason="a VectorIndexer was configured without a ChunkVectorSource",
                consequence="no vectors are written and indexed_vector_at stays NULL",
            )

    # -------------------------------------------------------------- health --

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        """PostgreSQL, plus whichever search store this replica actually writes.

        PostgreSQL is non-negotiable: without it there is no Signal to index, and
        `docs/architecture.md` §7.3 makes it a hard failure on this path. Qdrant
        and OpenSearch are probed only when configured -- a replica running
        keyword-only must not remove itself from rotation because a collection it
        never touches is down.
        """
        from backend.db.session import check_postgres

        probes: dict[str, DependencyProbe] = {"postgres": check_postgres}
        if self._vector_indexer is not None:
            from backend.db.qdrant import check_qdrant

            probes["qdrant"] = check_qdrant
        if self._keyword_indexer is not None:
            from backend.db.opensearch import check_opensearch

            probes["opensearch"] = check_opensearch
        return probes

    # ------------------------------------------------------------ handling --

    async def handle(self, message: ConsumedMessage) -> None:
        """Index one Signal. Raising sends the message to the DLQ after retries.

        Idempotent throughout: the OpenSearch `_id` is the chunk id, the Qdrant
        point id is derived from it, and the stamps are absolute assignments
        rather than increments. Processing the same message twice therefore
        yields one document, one point and the same two timestamps.
        """
        event = message.envelope.payload_as(SignalEnrichedEvent)
        view = await self._load(event.signal_id)

        if view is None:
            # The row is gone. Because the event is published *after* the commit
            # (`docs/data-stores.md` §5.1 step 5), this cannot be a race with the
            # writer -- it is an erasure that happened in between. Erasure owns
            # removing the derived copies, and re-driving would be indexing a
            # Signal that no longer exists.
            logger.info(
                "indexing.signal_missing",
                signal_id=event.signal_id,
                reason="no signals row; erased after the enriched event was published",
            )
            return

        if not _is_indexable(view):
            await self._withdraw(view)
            return

        chunks = chunks_for(view, settings=self._settings)
        if not chunks:
            # A media-only post or an empty body: nothing to write to either
            # store. Both columns are stamped because the derived stores *do*
            # agree with PostgreSQL -- they correctly hold nothing. Leaving them
            # NULL would park the Signal in the reconciler's backlog forever.
            logger.info(
                "indexing.nothing_to_index",
                signal_id=view.id,
                reason="the cleaned body produced no chunks",
            )
            await stamp_index_state(
                self._session_factory, view.id, IndexState.KEYWORD, IndexState.VECTOR
            )
            return

        await self._index_keyword(view, chunks)
        await self._index_vector(view, chunks)

    # -------------------------------------------------------------- the two --

    async def _index_keyword(self, view: SignalView, chunks: Sequence[Chunk]) -> None:
        """Write the chunk documents, then stamp. In that order, always."""
        if self._keyword_indexer is None:
            return
        outcome = await self._keyword_indexer.index_chunks(
            keyword_documents(view, chunks, tenant_id=self._tenant_id)
        )
        await stamp_index_state(self._session_factory, view.id, IndexState.KEYWORD)
        logger.info(
            "indexing.keyword.written",
            signal_id=view.id,
            index=outcome.index,
            documents=outcome.indexed,
            # A conflict is the external-version guard rejecting a write older
            # than what the index holds -- correct behaviour, and the reason it
            # is counted separately rather than folded into failures.
            conflicts=outcome.conflicts,
        )

    async def _index_vector(self, view: SignalView, chunks: Sequence[Chunk]) -> None:
        """Obtain the vectors, upsert them, then stamp. In that order.

        Skipped entirely when the Signal carries no `EmbeddingRef`s: that is the
        signature of a degraded stage 6, and `workers/embedding_worker.py` owns
        it. `indexed_vector_at` stays `NULL`, which is precisely how that sweeper
        finds the row.
        """
        if self._vector_indexer is None or self._vector_source is None:
            return
        if not view.embeddings:
            logger.info(
                "indexing.vector.deferred",
                signal_id=view.id,
                status=view.lineage.status.value,
                reason="the Signal carries no embedding refs; stage 6 degraded",
                owner="workers/embedding_worker.py",
            )
            return

        vectors = await self._vector_source([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                f"vector source returned {len(vectors)} vectors for {len(chunks)} "
                f"chunks of {view.id!r}; they are matched by position"
            )

        payloads = chunk_payloads(view, chunks, tenant_id=self._tenant_id)
        outcome = await self._vector_indexer.index_chunks(
            ChunkVector(vector=vector, payload=payload)
            for vector, payload in zip(vectors, payloads, strict=True)
        )
        await stamp_index_state(self._session_factory, view.id, IndexState.VECTOR)
        logger.info(
            "indexing.vector.written",
            signal_id=view.id,
            collection=outcome.collection,
            points=outcome.points,
            batches=outcome.batches,
        )

    async def _withdraw(self, view: SignalView) -> None:
        """Remove a non-retrievable Signal from both stores, then stamp both.

        By Signal, not by enumerating chunk ids. The chunk count is a property of
        the text *at the time it was chunked*, so a caller that re-derives ids
        from today's text leaves the tail of an older, longer chunking behind --
        orphan documents that stay searchable forever and point at character
        spans that no longer exist.
        """
        if self._keyword_indexer is not None:
            await self._keyword_indexer.delete_signal(view.id)
        if self._vector_indexer is not None:
            await self._vector_indexer.delete_signal(view.id)
        await stamp_index_state(
            self._session_factory, view.id, IndexState.KEYWORD, IndexState.VECTOR
        )
        logger.info(
            "indexing.withdrawn",
            signal_id=view.id,
            status=view.lineage.status.value,
            duplicate_of=view.lineage.duplicate_of,
            reason="only the canonical member of a dedup cluster is retrievable",
        )

    # ------------------------------------------------------------ internals --

    async def _load(self, signal_id: str) -> SignalView | None:
        """Re-read the Signal from the commit point. See the module docstring.

        The import is function-local because `services/signal_service.py` pulls
        in the whole query layer, and the projection is the only thing wanted
        from it here.
        """
        from services.signal_service import signal_view_from_row

        async with self._session_factory() as session:
            row = await session.get(SignalRow, signal_id)
            if row is None:
                return None
            return signal_view_from_row(row)


# --------------------------------------------------------------------------- #
# Projection helpers -- pure, so `scripts/reindex.py` can share them
# --------------------------------------------------------------------------- #


def _is_indexable(view: SignalView) -> bool:
    """Whether this Signal belongs in the search stores at all.

    Reads `SignalStatus.is_retrievable` rather than testing statuses by hand, so
    a status added to `models/enums.py` cannot silently become indexable here
    while being unretrievable everywhere else. The canonical check is separate
    because a row can carry a retrievable status and still be a cluster loser
    during the window between dedup electing a survivor and the status being
    written.
    """
    return view.lineage.status.is_retrievable and view.lineage.duplicate_of is None


def chunks_for(view: SignalView, *, settings: Settings | None = None) -> list[Chunk]:
    """Re-derive the chunk boundaries stage 6 used.

    Identical geometry to `services/signal_engine/embeddings.py`: the same
    strategy selection (`strategy_for(source)`), the same `max_chars`, the same
    overlap, and only `content.text` -- never `title + text`. Every one of those
    has to match, and the last is the subtle one: chunk spans index the exact
    string `services/evidence_service.py` re-reads to verify a quote, so
    prepending a title would shift every offset by its length and quietly
    invalidate the whole citation mechanism while producing chunk ids that still
    look right.

    Re-deriving rather than reading stored boundaries is deliberate: only the
    `EmbeddingRef`s survive on the Signal (model, dimensions, chunk index,
    collection, point id) and the spans do not. The derivation is pure, so the
    same text and the same settings reproduce the same ids on any machine, which
    is what makes a re-index an upsert instead of a second corpus.
    """
    resolved = (settings or get_settings()).embedding
    return split_text(
        view.content.text,
        strategy=strategy_for(view.source),
        max_chars=resolved.max_chars_per_chunk,
        overlap_chars=resolved.chunk_overlap_chars,
    )


def keyword_documents(
    view: SignalView,
    chunks: Sequence[Chunk],
    *,
    tenant_id: str = DEFAULT_TENANT,
) -> list[ChunkDocument]:
    """Project one Signal onto its OpenSearch documents.

    The title rides on every chunk rather than on the first one only. BM25 scores
    a document, not a Signal, so a title-bearing first chunk would outrank the
    chunk that actually answers the query for any title-adjacent term -- and the
    citation would then point at the introduction of an article whose relevant
    passage is on page three.

    `pipeline_version` is the integer ordinal, which becomes the external
    document version: compared as text, `'1.10.0' >= '1.9.0'` is False, so a
    string here would let a stale backfill overwrite newer enrichment the moment
    a version component reached 10.
    """
    version = pipeline_version_ordinal(view.lineage.pipeline_version)
    entity_ids = _resolved_entity_ids(view)
    published_at = _aware(view.timestamp)
    return [
        ChunkDocument(
            signal_id=view.id,
            chunk_index=chunk.index,
            text=chunk.text,
            char_start=chunk.char_start,
            char_end=chunk.char_end,
            tenant_id=tenant_id,
            title=view.content.title,
            source=view.source,
            platform=view.platform,
            url=view.url,
            author_handle=view.author.handle if view.author is not None else None,
            published_at=published_at,
            language=view.language.code,
            keywords=[keyword.term for keyword in view.keywords],
            topics=[topic.topic for topic in view.topics],
            entity_ids=entity_ids,
            sentiment_polarity=view.sentiment.polarity if view.sentiment is not None else None,
            engagement_score=view.engagement.score,
            confidence=view.confidence,
            pipeline_version=version,
        )
        for chunk in chunks
    ]


def chunk_payloads(
    view: SignalView,
    chunks: Sequence[Chunk],
    *,
    tenant_id: str = DEFAULT_TENANT,
) -> list[ChunkPayload]:
    """Project one Signal onto its Qdrant point payloads.

    Narrower than the OpenSearch document on purpose: a Qdrant payload exists to
    be *filtered on* and to decide staleness, never to be displayed. A citation
    is resolved from PostgreSQL by chunk id, so putting the text here would
    duplicate the corpus into a store that cannot serve it any better than the
    one that already holds it.
    """
    version = pipeline_version_ordinal(view.lineage.pipeline_version)
    entity_ids = _resolved_entity_ids(view)
    published_at = _aware(view.timestamp)
    return [
        ChunkPayload(
            signal_id=view.id,
            chunk_index=chunk.index,
            tenant_id=tenant_id,
            platform=view.platform,
            source=view.source,
            published_at=published_at,
            language=view.language.code,
            entity_ids=entity_ids,
            confidence=view.confidence,
            pipeline_version=version,
        )
        for chunk in chunks
    ]


def _resolved_entity_ids(view: SignalView) -> list[str]:
    """Distinct canonical entity ids, first-seen order, unresolved mentions dropped.

    Spelled out here rather than reusing `Signal.resolved_entity_ids()` because
    this is a `SignalView` -- the lenient read model -- and it deliberately does
    not carry the producer-side accessors. An unresolved mention has no id that
    any filter could match, so including it would put `None` in a payload list
    that `MatchAny` compares against.
    """
    seen: dict[str, None] = {}
    for mention in view.entities:
        if mention.resolved_id is not None:
            seen.setdefault(mention.resolved_id, None)
    return list(seen)


def _aware(value: datetime) -> datetime:
    """Force a timezone onto a timestamp read back from the database.

    SQLite -- and any PostgreSQL column that ended up without `timestamptz` --
    hands back a naive datetime. Both derived stores mis-handle one:
    `ChunkPayload.to_payload()` raises outright, and OpenSearch silently parses a
    naive value *as UTC*, which for a Signal timestamped in Asia/Tokyo moves it
    by up to a day and can push it across the boundary of a `[start, end)` window
    filter. Nothing downstream can detect that it happened.

    UTC is the right assumption specifically because every writer in this system
    stores UTC (`models/base.UtcDatetime` normalises on the way in); this restores
    information the storage layer dropped rather than guessing at it.
    """
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


# --------------------------------------------------------------------------- #
# Composition root
# --------------------------------------------------------------------------- #


def build_worker(*, settings: Settings | None = None, **kwargs: Any) -> IndexingWorker:
    """Resolve real clients from settings and assemble the worker.

    Separate from `IndexingWorker.__init__` because construction and *resolution*
    are different concerns: the constructor takes ports and is therefore testable
    with fakes, while this function opens sockets and is only runnable in a real
    deployment.
    """
    from backend.db.session import get_sessionmaker
    from retrieval.keyword.opensearch_client import get_keyword_store
    from retrieval.vector.qdrant_client import get_vector_store
    from services.llm.embeddings import OpenAICompatibleEmbeddingProvider

    resolved = settings or get_settings()
    return IndexingWorker(
        session_factory=get_sessionmaker(),
        # The narrowed accessors, not the raw singletons: `retrieval/` owns the
        # protocol each store is used through, and going via them keeps the
        # worker from depending on a client surface wider than it needs.
        vector_indexer=VectorIndexer(get_vector_store()),
        keyword_indexer=KeywordIndexer(get_keyword_store()),
        vector_source=EmbeddingChunkVectors(
            OpenAICompatibleEmbeddingProvider(settings=resolved.embedding)
        ),
        settings=resolved,
        **kwargs,
    )


def main() -> None:  # pragma: no cover -- process entrypoint
    """`python -m workers.indexing_worker`, per `docs/deployment.md` §3."""
    run_worker(build_worker())


if __name__ == "__main__":  # pragma: no cover
    main()
