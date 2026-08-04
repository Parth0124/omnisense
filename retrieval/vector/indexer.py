"""Writing chunk vectors into Qdrant, and taking them back out again.

Two properties make this module correct, and both of them are about *re-runs*
rather than first runs. Re-runs are the normal case: `scripts/reindex.py` replays
the whole corpus, the reconciler in `docs/data-stores.md` §6 replays whatever
PostgreSQL says is unindexed, and a Kafka consumer replays its partition after
any rebalance.

**Point ids are derived, never assigned.** The id of a point is
`uuid5(namespace, "{signal_id}:{chunk_index}")`. Derivation is what turns "index
this batch again" into an upsert instead of a second copy of the corpus: with a
`uuid4()` id, replaying a partition doubles the collection, every duplicate
scores identically, and the top-k for every query fills up with the same passage
repeated -- which reads as a ranking bug, not a write bug. uuid5 specifically
because Qdrant point ids must be an unsigned integer or a UUID; `chunk_id` is
neither, and a hash truncated into an int would collide across a corpus this
size.

**Deletion is by payload filter, not by enumerating ids.** Erasure
(`docs/security-and-privacy.md`) and canonical-election demotion
(`docs/signal-model.md` §4.3) both mean "remove every chunk of this Signal", and
the caller does not know how many chunks there were -- the chunk count is a
property of the *text at the time it was chunked*. A caller that re-derives ids
from a current chunk count silently leaves the tail behind when a re-chunk
produced fewer chunks than before, and those orphans stay searchable forever.

This module never embeds anything. Vectors arrive from
`workers/embedding_worker.py`; the split matters because embedding is the
expensive, rate-limited, billed step and re-indexing must not become re-embedding
(`docs/data-stores.md` §3.3).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from qdrant_client import models

from backend.core.logging import get_logger
from retrieval.types import chunk_id_for, split_chunk_id
from retrieval.vector.collections import (
    ChunkPayload,
    CollectionSpec,
    PayloadField,
    signal_collection_spec,
)
from retrieval.vector.qdrant_client import VectorStore

__all__ = [
    "DEFAULT_UPSERT_BATCH_SIZE",
    "POINT_ID_NAMESPACE",
    "ChunkVector",
    "IndexOutcome",
    "VectorIndexer",
    "point_id_for",
]

POINT_ID_NAMESPACE: Final[uuid.UUID] = uuid.UUID("6979eead-d426-5311-873b-f61c66ebd53e")
"""Namespace for chunk point ids.

Literal rather than computed so it is greppable and can never move: this is
`uuid5(NAMESPACE_URL, "https://omnisense.dev/qdrant/signal-chunk")`, and changing
it -- or recomputing it from a different string -- re-keys every point in the
collection. The next indexing run would then insert a full second copy of the
corpus alongside the first rather than upserting it, and nothing would report an
error.
"""

DEFAULT_UPSERT_BATCH_SIZE: Final[int] = 256
"""Points per upsert request.

Sized by request body, not by taste: a 1536-dimensional float vector serialises
to roughly 20 KB of JSON, so 256 points is a ~5 MB request -- comfortably under
the usual proxy limits while still amortising the round trip. Larger batches stop
helping (the server writes them in segments anyway) and start turning one slow
request into a timeout that re-sends the whole batch.
"""

_log = get_logger(__name__)


def point_id_for(signal_id: str, chunk_index: int) -> str:
    """The Qdrant point id for a chunk. Pure, stable, and the whole idempotency story.

    Derived from `chunk_id` -- the same key OpenSearch uses as its `_id` and the
    key hybrid fusion joins on -- so the two derived stores address the same chunk
    by the same identity even though only one of them can spell it directly.
    """
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id_for(signal_id, chunk_index)))


@dataclass(frozen=True, slots=True)
class ChunkVector:
    """One embedded chunk, ready to write.

    Carries the payload as a `ChunkPayload` rather than a dict so a caller cannot
    quietly add a field that no index covers, or omit `tenant_id` -- an
    unfiltered tenant leak is a payload the writer forgot, not a query the reader
    got wrong.
    """

    vector: Sequence[float]
    payload: ChunkPayload

    @property
    def chunk_id(self) -> str:
        return self.payload.chunk_id

    @property
    def point_id(self) -> str:
        return point_id_for(self.payload.signal_id, self.payload.chunk_index)

    def to_point(self) -> models.PointStruct:
        return models.PointStruct(
            id=self.point_id,
            vector=list(self.vector),
            payload=self.payload.to_payload(),
        )


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    """What one `index_chunks()` call did. Returned for the reconciler's log.

    `points` and `batches` are separate because they diagnose different problems:
    a low point count is a producer that fell behind, a high batch count at the
    same point count is a batch size someone tuned down and forgot.
    """

    points: int = 0
    batches: int = 0
    collection: str = ""
    chunk_ids: Sequence[str] = field(default_factory=tuple)


class VectorIndexer:
    """Upserts and deletes chunk vectors in one collection.

    Holds a client and a spec, no other state, so one instance per process is
    fine and concurrent calls are independent.
    """

    def __init__(
        self,
        client: VectorStore,
        spec: CollectionSpec | None = None,
        *,
        batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self._client = client
        self._spec = spec or signal_collection_spec()
        self._batch_size = batch_size

    @property
    def collection(self) -> str:
        return self._spec.name

    @property
    def spec(self) -> CollectionSpec:
        return self._spec

    async def index_chunks(
        self, chunks: Iterable[ChunkVector], *, wait: bool = False
    ) -> IndexOutcome:
        """Upsert chunk vectors in batches. Idempotent by derived point id.

        Every vector's dimensionality is checked against the collection geometry
        *before* the first request goes out. Qdrant would reject a mismatched
        vector too, but it rejects the batch it is given -- so a single bad
        vector in a batch of 256 fails 255 good ones, and the retry re-sends all
        256 and fails again. Checking here turns that into one error naming the
        offending chunk.

        `wait=False` by default: Qdrant applies upserts asynchronously and
        `docs/data-stores.md` §3.3 states plainly that a just-written vector is
        not guaranteed to be immediately searchable. Anything needing
        read-your-write reads PostgreSQL, so blocking the indexer on the index
        flush buys nothing. Tests and `scripts/reindex.py` pass `wait=True`
        because they assert on what is searchable next.

        Raises:
            ValueError: a vector whose length is not the collection's vector
                size, or a duplicate chunk within one call.
        """
        pending: list[models.PointStruct] = []
        chunk_ids: list[str] = []
        seen: dict[str, None] = {}
        outcome_points = 0
        batches = 0

        for chunk in chunks:
            self._assert_dimensions(chunk)
            if chunk.chunk_id in seen:
                # Two entries for one chunk in a single call means the caller
                # chunked twice, or merged two batches wrongly. Qdrant would
                # apply both and keep whichever landed last -- a coin flip
                # between two different vectors for the same text, which is not
                # a thing to resolve silently.
                raise ValueError(
                    f"chunk {chunk.chunk_id!r} appears twice in one index_chunks() "
                    "call; the last write would win arbitrarily"
                )
            seen[chunk.chunk_id] = None
            pending.append(chunk.to_point())
            chunk_ids.append(chunk.chunk_id)

            if len(pending) >= self._batch_size:
                await self._flush(pending, wait=wait)
                outcome_points += len(pending)
                batches += 1
                pending = []

        if pending:
            await self._flush(pending, wait=wait)
            outcome_points += len(pending)
            batches += 1

        return IndexOutcome(
            points=outcome_points,
            batches=batches,
            collection=self._spec.name,
            chunk_ids=tuple(chunk_ids),
        )

    async def delete_signal(
        self, signal_id: str, *, from_chunk_index: int = 0, wait: bool = False
    ) -> None:
        """Delete a Signal's chunks. The erasure and demotion path.

        Used for two things that look different and are the same operation:

        - **Erasure.** A deletion request must remove the derived copies too, and
          Qdrant is a derived store with no other record of what it holds for a
          Signal.
        - **Demotion.** A Signal that loses a canonical election becomes
          `DUPLICATE` and is no longer retrievable (`models/enums.py`), so its
          vectors must leave the index while the row stays in PostgreSQL.

        `from_chunk_index` trims a *tail*: after a re-chunk that produced fewer
        chunks than the previous run, points 5..n of the old chunking are still
        in the collection, still matching queries, and pointing at character
        spans that no longer exist. Nothing else notices them -- resolution just
        drops the chunk ids it cannot find, so the symptom is a quietly shorter
        result list.

        Deletes are asynchronous in Qdrant regardless of `wait`; `wait=True` only
        guarantees the operation was accepted and applied to the WAL.
        """
        must: list[models.Condition] = [
            models.FieldCondition(
                key=PayloadField.SIGNAL_ID.value,
                match=models.MatchValue(value=signal_id),
            )
        ]
        if from_chunk_index > 0:
            must.append(
                models.FieldCondition(
                    key=PayloadField.CHUNK_INDEX.value,
                    range=models.Range(gte=from_chunk_index),
                )
            )

        await self._client.delete(
            collection_name=self._spec.name,
            points_selector=models.FilterSelector(filter=models.Filter(must=must)),
            wait=wait,
        )
        _log.info(
            "qdrant.delete_signal",
            collection=self._spec.name,
            signal_id=signal_id,
            from_chunk_index=from_chunk_index,
        )

    async def delete_signals(
        self, signal_ids: Sequence[str], *, wait: bool = False
    ) -> None:
        """Delete every chunk of several Signals in one request.

        One `MatchAny` rather than a loop of `delete_signal()` calls: a
        deduplication run demotes whole clusters at once, and n round trips for n
        losers turns a cluster of 400 near-duplicates into 400 requests.
        """
        ids = [s for s in dict.fromkeys(signal_ids) if s]
        if not ids:
            # An empty selector would compile to a filter matching everything.
            # Returning is the only safe reading of "delete nothing".
            return
        await self._client.delete(
            collection_name=self._spec.name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=PayloadField.SIGNAL_ID.value,
                            match=models.MatchAny(any=list(ids)),
                        )
                    ]
                )
            ),
            wait=wait,
        )
        _log.info(
            "qdrant.delete_signals",
            collection=self._spec.name,
            signal_count=len(ids),
        )

    async def delete_chunks(self, chunk_ids: Sequence[str], *, wait: bool = False) -> None:
        """Delete specific chunks by derived point id.

        The narrow case: a chunk known to be individually wrong. Whole-Signal
        removal must go through `delete_signal()`, which does not depend on the
        caller knowing the chunk count.
        """
        point_ids = [point_id_for(*split_chunk_id(cid)) for cid in chunk_ids]
        if not point_ids:
            return
        await self._client.delete(
            collection_name=self._spec.name,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=wait,
        )

    # ------------------------------------------------------------ internals --

    async def _flush(self, points: Sequence[models.PointStruct], *, wait: bool) -> None:
        await self._client.upsert(
            collection_name=self._spec.name, points=list(points), wait=wait
        )

    def _assert_dimensions(self, chunk: ChunkVector) -> None:
        """Refuse a vector the collection cannot hold.

        A dimension mismatch means the embedding model changed without the
        collection being rebuilt (`docs/signal-model.md` §9). The check is here
        rather than left to the server because by this point the provider has
        already been billed for the embedding, and the useful error names the
        chunk and both dimensions rather than reporting a rejected batch.
        """
        size = len(chunk.vector)
        if size != self._spec.vector_size:
            raise ValueError(
                f"chunk {chunk.chunk_id!r} has a {size}-dimensional vector but "
                f"collection {self._spec.name!r} holds {self._spec.vector_size} "
                "dimensions. The embedding model changed without the collection "
                "being rebuilt: point EMBEDDING_DIMENSIONS back at the model that "
                "produced this vector, or re-embed into a new collection with "
                "scripts/reindex.py."
            )
