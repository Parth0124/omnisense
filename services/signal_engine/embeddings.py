"""Stage 6 -- Embedding: chunk the body, embed the chunks, record the addresses.

Design Doc §6 stage 6, `docs/signal-model.md` §5.1. Degrades to `[]` on failure
and is retried later by `workers/embedding_worker.py`; a Signal with no vector is
still findable through BM25 and still countable in a trend, so losing embeddings
is a degradation, never a reason to drop an observation.

**The vector does not travel inside the Signal.** `Signal.embeddings` holds
`EmbeddingRef`s -- model, dimensions, chunk index, collection, point id -- and
nothing else. The arithmetic is decisive: a 1536-dimension `float32` vector is
6 KB raw and roughly 20-30 KB once JSON-serialized as decimal text, against a
Signal body that is typically 2-4 KB. A ten-chunk news article would therefore
ship ~250 KB of numbers that no reader of a Signal ever looks at, and it would
ship them *everywhere* at once:

- through Kafka, whose default `max.message.bytes` is 1 MB -- a long document
  would not merely be slow, it would be **rejected**, and stage 7's produce is
  the fatal one;
- into PostgreSQL, where the row TOASTs and every `SELECT *` on the hot table
  pays de-TOASTing for a column nobody filters or projects;
- into every API response, every Redis cache entry and every DLQ record, each of
  which is sized for a Signal and not for a matrix.

Meanwhile the one component that actually needs the numbers -- Qdrant -- is a
vector database that is about to store them anyway. So the vector goes to
Qdrant, the Signal carries the address, and a reader that wants the vector asks
the store that owns it. That is the same "two-tier rule" `docs/data-stores.md`
§1 applies to raw payloads.

**Point ids are derived, not assigned.** `point_id = uuid5(namespace,
"{signal_id}:{chunk_index}")` (`docs/data-stores.md` §5.2). Re-running this stage
after a rate-limit failure recomputes the identical id, so the Qdrant write is an
upsert that overwrites in place. A random id would instead accumulate a fresh
copy of every chunk on every retry -- duplicate hits in retrieval, duplicate
citations in a report, and a collection that grows without bound while looking
perfectly healthy.

**The vectors reach stage 7 out of band.** Because they must not be on the
Signal, they are handed to a `VectorSink` supplied at construction. When no sink
is given the vectors are computed, used to validate width, and dropped; the refs
still record where the points belong, and `workers/embedding_worker.py` re-derives
the same chunks and the same ids to fill Qdrant later. That is a real cost (a
second embedding call) and it is stated here rather than hidden, because a sink
that is silently absent in production is the kind of thing that presents as
"vector search returns nothing" three weeks after launch.

Chunking lives in `retrieval/chunking/splitter.py` and happens exactly once, here
(`docs/retrieval.md` §8): both indexers consume the ids this stage produced, and
two independent chunkers would derive different ids for the same Signal, leaving
hybrid fusion with nothing to join on.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

from backend.core.config import EmbeddingSettings, get_settings
from models.enums import StageName
from models.signal import SIGNAL_ID_NAMESPACE, EmbeddingRef, Signal
from retrieval.chunking.splitter import (
    Chunk,
    ChunkStrategy,
    chunk_id,
    split_text,
    strategy_for,
)
from services.llm.embeddings import EmbeddingDimensionMismatch, EmbeddingProvider
from services.signal_engine.pipeline import EnrichmentContext

__all__ = [
    "EMBEDDING_STAGE_VERSION",
    "ChunkVector",
    "EmbeddingStage",
    "InMemoryVectorSink",
    "VectorSink",
    "point_id_for",
]


EMBEDDING_STAGE_VERSION: Final = "1.0.0"
"""Semantic version of this stage, written into `lineage.stages[]`.

Bump it when the *chunking geometry or strategy selection* changes, not only
when this file changes: chunk ids are derived from the boundaries, so a change
here silently invalidates stored citations unless a reindex follows
(`docs/retrieval.md` §8).
"""


def point_id_for(signal_id: str, chunk_index: int) -> str:
    """Derive the Qdrant point id for one chunk. Pure, total and stable forever.

    Reuses `SIGNAL_ID_NAMESPACE` rather than minting a second never-rotated
    constant. The two name spaces cannot collide: Signal ids are derived from
    `"{platform}:{native_id}"` where the platform is a `Platform` member, while
    these are derived from `"{sig_...hex}:{index}"`, and no platform is named
    `sig_<32 hex digits>`. One constant means one thing to never rotate.

    The model id is deliberately *not* an input. Changing the embedding model
    changes the collection, not the point -- `docs/retrieval.md` §5 makes the
    model part of the index identity, so a re-embed writes the same logical
    points into a new collection and swaps an alias.
    """
    return str(uuid.uuid5(SIGNAL_ID_NAMESPACE, chunk_id(signal_id, chunk_index)))


@dataclass(frozen=True, slots=True)
class ChunkVector:
    """One embedded chunk, on its way to Qdrant and OpenSearch.

    Carries exactly what the Signal deliberately does not: the vector itself,
    and the chunk text and span that the indexers need but that would duplicate
    `content.text` if they rode along on the Signal. Frozen because two workers
    consume the same object and neither owns it.
    """

    chunk_id: str
    point_id: str
    collection: str
    chunk_index: int
    char_start: int
    char_end: int
    text: str
    vector: tuple[float, ...]


@runtime_checkable
class VectorSink(Protocol):
    """Where stage 6 hands vectors so stage 7 can write them.

    A port rather than a Qdrant client: stage 6 must be runnable with no
    datastore at all (that is what makes the unit suite offline), and the write
    ordering in `docs/data-stores.md` §5.1 puts the Qdrant upsert *after* the
    PostgreSQL commit -- so the component that embeds cannot also be the
    component that persists.
    """

    async def collect(self, signal_id: str, vectors: Sequence[ChunkVector]) -> None:
        """Accept every vector for one Signal. Called once per successful stage run.

        Must be idempotent per `signal_id`: a re-run replaces that Signal's
        vectors rather than adding to them, because the point ids are identical
        and Qdrant would upsert them onto each other anyway.
        """
        ...


@dataclass(slots=True)
class InMemoryVectorSink:
    """Process-local `VectorSink`, for tests and for a single-process pipeline.

    Deliberately not a queue: `collect` *replaces* a Signal's entry so that a
    reprocessed Signal cannot leave stale vectors behind for a chunk that no
    longer exists after a chunker change.
    """

    staged: dict[str, list[ChunkVector]] = field(default_factory=dict)

    async def collect(self, signal_id: str, vectors: Sequence[ChunkVector]) -> None:
        self.staged[signal_id] = list(vectors)

    def take(self, signal_id: str) -> list[ChunkVector]:
        """Remove and return one Signal's vectors. Empty list when there are none.

        Removal is the point: holding vectors after stage 7 has written them
        turns this sink into an unbounded per-worker memory leak, at ~6 KB per
        chunk.
        """
        return self.staged.pop(signal_id, [])


class EmbeddingStage:
    """Stage 6. Satisfies `services.signal_engine.pipeline.Stage`.

    Stateless across records -- one instance is shared by a worker and driven
    concurrently, so everything per-Signal lives on the context or in the sink,
    never on `self`.
    """

    name: StageName = StageName.EMBEDDING
    version: str = EMBEDDING_STAGE_VERSION

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        settings: EmbeddingSettings | None = None,
        collection: str | None = None,
        sink: VectorSink | None = None,
        strategy: ChunkStrategy | None = None,
    ) -> None:
        """Take the provider; never construct one.

        Injection is what makes the AI layer swappable (Design Doc §15) and what
        lets every test here pass a ten-line fake instead of a network. The same
        applies to `sink`: an argument, so the pipeline can be assembled for a
        dry run that embeds and discards.

        `strategy` overrides the per-source choice from `docs/retrieval.md` §8.
        It exists for `scripts/reindex.py`, which must re-chunk an existing
        corpus exactly as it was originally chunked, not as today's table says.
        """
        self._provider = provider
        resolved = settings if settings is not None else get_settings().embedding
        self._max_chars = resolved.max_chars_per_chunk
        self._overlap_chars = resolved.chunk_overlap_chars
        self._batch_size = resolved.batch_size
        if collection is None:
            collection = get_settings().qdrant.collection
        self._collection = collection
        self._sink = sink
        self._strategy = strategy

    @property
    def model_id(self) -> str | None:
        """The embedding model, recorded per Signal in `lineage.stages[]`.

        Stage 6 is one of the non-deterministic stages (`docs/signal-model.md`
        §5.1): reproducing a vector later is impossible without knowing which
        model produced it, and "which model was live in July" is not a question
        the deployment history can answer reliably.
        """
        return self._provider.model

    async def apply(self, ctx: EnrichmentContext) -> None:
        """Chunk, embed, and record one `EmbeddingRef` per chunk.

        Raises on provider failure. Deciding that an embedding failure is
        survivable is the pipeline's job, not this stage's -- a stage that
        swallowed its own exception would report success, `extraction_quality`
        would credit it in full, and the Signal would look complete while being
        unreachable by vector search forever.
        """
        signal = ctx.require_signal()

        if not signal.is_canonical:
            # `docs/signal-model.md` §4.3: only the canonical member of a dedup
            # cluster is embedded, so the same press release from six platforms
            # returns one hit instead of six. Skipping here is also the single
            # largest cost saving in the pipeline -- five sixths of the embedding
            # spend on a heavily syndicated story.
            signal.embeddings = []
            return

        chunks = self._chunks_of(signal)
        if not chunks:
            # A media-only post or a title-only excerpt. Not a failure: stage 6b
            # scores the missing body through `content_integrity`, and raising
            # here would mark a perfectly valid Signal `partial`.
            signal.embeddings = []
            return

        vectors = await self._embed(chunks)

        refs: list[EmbeddingRef] = []
        staged: list[ChunkVector] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            point_id = point_id_for(signal.id, chunk.index)
            refs.append(
                EmbeddingRef(
                    model=self._provider.model,
                    dimensions=self._provider.dimensions,
                    chunk_index=chunk.index,
                    collection=self._collection,
                    point_id=point_id,
                )
            )
            staged.append(
                ChunkVector(
                    chunk_id=chunk.id_for(signal.id),
                    point_id=point_id,
                    collection=self._collection,
                    chunk_index=chunk.index,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    text=chunk.text,
                    vector=tuple(vector),
                )
            )

        # Assigned once, after every batch has landed. A partial assignment
        # followed by a raise would leave refs pointing at points that were never
        # upserted -- retrieval would return a hit whose vector does not exist,
        # which reads as data corruption rather than as a failed stage.
        signal.embeddings = refs
        if self._sink is not None:
            await self._sink.collect(signal.id, staged)

    # ------------------------------------------------------------ internals --

    def _chunks_of(self, signal: Signal) -> list[Chunk]:
        """Split `content.text` at the geometry this deployment is configured for.

        Only `content.text` is chunked, never `title + text`. Spans must index
        the exact string `services/evidence_service.py` re-reads to verify a
        quote (`docs/retrieval.md` §8); a synthesized "title\\n\\ntext" would
        shift every offset by the title's length and quietly invalidate the whole
        citation mechanism. The title is indexed as its own field in OpenSearch
        instead, which is where it belongs.
        """
        strategy = self._strategy or strategy_for(signal.source)
        return split_text(
            signal.content.text,
            strategy=strategy,
            max_chars=self._max_chars,
            overlap_chars=self._overlap_chars,
        )

    async def _embed(self, chunks: Sequence[Chunk]) -> list[list[float]]:
        """Embed every chunk, in `EMBEDDING_BATCH_SIZE` groups, and check the widths.

        Batching here is deliberately redundant with the batching inside
        `OpenAICompatibleEmbeddingProvider`. The `EmbeddingProvider` protocol
        promises only "embed every text, in order" -- a self-hosted or fake
        provider is free to forward the list verbatim, and a 400-chunk paper
        would then be one request that fails whole.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = [chunk.text for chunk in chunks[start : start + self._batch_size]]
            produced = await self._provider.embed(batch)
            if len(produced) != len(batch):
                # Silently short results shift every later vector onto the wrong
                # chunk. Nothing downstream can detect that: search simply gets
                # worse, for reasons no user reports as a bug.
                raise ValueError(
                    f"the embedding provider returned {len(produced)} vectors for "
                    f"{len(batch)} chunks; vector-to-chunk alignment cannot be recovered"
                )
            vectors.extend(produced)

        expected = self._provider.dimensions
        for index, vector in enumerate(vectors):
            if len(vector) != expected:
                # The provider's own client checks this, but a provider is an
                # injected port and this stage is what writes `dimensions` into
                # the ref. Recording 1536 next to a 1024-wide vector is a lie
                # that only surfaces at the Qdrant upsert -- after the spend, and
                # after the pipeline has already called the stage successful.
                raise EmbeddingDimensionMismatch(
                    f"chunk {index} embedded to {len(vector)} dimensions but the "
                    f"provider declares {expected} (model {self._provider.model!r})",
                    details={"returned": len(vector), "expected": expected},
                )
        return vectors
