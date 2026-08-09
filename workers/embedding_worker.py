"""Backfills embeddings for Signals the enrichment pipeline could not embed.

The pipeline's `EmbeddingStage` is *degradable* (`models/enums.py`
`FATAL_STAGES`): when the embedding provider is down, enrichment continues, the
Signal is stored, and it lands in PostgreSQL with no vector. That is the right
trade -- losing semantic search on a Signal is far better than losing the Signal
-- but it is only right if something comes back for it.

This worker is that something. It is a `PeriodicWorker` rather than a consumer
because the work is defined by a *predicate over the corpus* ("rows with no
vector"), not by an event. There is no message for "the embedding provider
recovered", and inventing a topic whose only producer is a timer adds a component
to provision and monitor in exchange for nothing.

**It is the same sweep pattern as `docs/data-stores.md` §6**, applied one step
earlier in the pipeline: `indexed_vector_at` says whether Qdrant holds the
vectors, and this worker exists for the case where there were no vectors to hold.

**Rate-limited on purpose.** After a provider outage every Signal from the outage
window is eligible at once, and embedding all of them as fast as the API allows
is how a recovered provider is immediately rate-limited again -- by us. The batch
cap turns a thundering herd into a steady drain.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from backend.core.logging import get_logger
from workers.runtime.base_worker import PeriodicWorker, run_worker
from workers.runtime.index_state import IndexState, stamp_index_state

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from backend.core.config import Settings
    from services.llm.embeddings import EmbeddingProvider
    from workers.runtime.health import DependencyProbe

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_INTERVAL_SECONDS",
    "EmbeddingBackfillWorker",
    "main",
]

logger = get_logger(__name__)

WORKER_NAME: Final = "embedding"

DEFAULT_INTERVAL_SECONDS: Final = 60.0
DEFAULT_BATCH_SIZE: Final = 50
"""Signals embedded per tick.

The rate limiter. Fifty per minute drains a backlog steadily without
re-saturating a provider that has just come back, and it bounds the memory one
tick holds -- fifty texts plus fifty vectors, not the whole outage window.
"""

MIN_AGE_SECONDS: Final = 120.0
"""How old a Signal must be before this worker touches it.

Two minutes. Without it the backfill races the enrichment pipeline: a Signal
stored thirty seconds ago may be mid-embedding right now, and embedding it again
concurrently means two writers producing two vectors for the same chunk -- with
whichever finishes last silently winning. The age floor makes the two paths
disjoint rather than merely usually-disjoint.
"""


class EmbeddingBackfillWorker(PeriodicWorker):
    """Finds Signals without vectors and embeds them."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        embeddings: EmbeddingProvider,
        vector_indexer: Any | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        settings: Settings | None = None,
        name: str = WORKER_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, interval_seconds=interval_seconds, settings=settings, **kwargs
        )
        self._session_factory = session_factory
        self._embeddings = embeddings
        self._indexer = vector_indexer
        self._batch_size = batch_size
        self.embedded = 0
        self.skipped = 0

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        """PostgreSQL and Qdrant. Both are required for this worker to do anything."""
        from backend.db.qdrant import check_qdrant
        from backend.db.session import check_postgres

        return {"postgres": check_postgres, "qdrant": check_qdrant}

    async def tick(self) -> None:
        """Embed one batch of un-vectored Signals.

        Re-entrant across replicas by the same mechanism as the scheduler:
        `SKIP LOCKED` on the select, so two replicas draw disjoint batches rather
        than one blocking on the other. Idempotent by derived point id, so even
        an overlap upserts rather than duplicating.
        """
        pending = await self._claim_pending()
        if not pending:
            return

        texts = [item["text"] for item in pending]
        try:
            vectors = await self._embeddings.embed_documents(texts)
        except Exception as error:  # noqa: BLE001 -- the tick loop keeps the worker alive
            # Raising would be equally correct -- the runtime logs and continues
            # either way -- but logging here lets the batch size and the provider
            # name into the record, which is what tells an operator whether this
            # is an outage or a payload problem.
            logger.error(
                "embedding.batch_failed",
                batch=len(texts),
                error=type(error).__name__,
            )
            raise

        if len(vectors) != len(pending):
            # A provider that returns a different count has silently reordered or
            # dropped something, and zipping them would attach vectors to the
            # wrong Signals -- a corruption no downstream check could detect,
            # because every vector would be individually valid.
            raise RuntimeError(
                f"embedding provider returned {len(vectors)} vectors for "
                f"{len(pending)} inputs; the correspondence is lost and attaching "
                "them positionally would give Signals each other's vectors"
            )

        for item, vector in zip(pending, vectors, strict=True):
            await self._store(item, vector)
            self.embedded += 1

        logger.info("embedding.backfilled", count=len(pending))

    # ------------------------------------------------------------ internals --

    async def _claim_pending(self) -> list[dict[str, Any]]:
        """Signals old enough to be safe and still missing a vector."""
        from sqlalchemy import select

        from models.orm.signal import SignalRow

        cutoff = datetime.now(UTC) - timedelta(seconds=MIN_AGE_SECONDS)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(SignalRow)
                    .where(
                        SignalRow.indexed_vector_at.is_(None),
                        SignalRow.created_at < cutoff,
                    )
                    .order_by(SignalRow.created_at)
                    .limit(self._batch_size)
                    .with_for_update(skip_locked=True)
                )
            ).scalars().all()

            claimed: list[dict[str, Any]] = []
            for row in rows:
                text = _text_of(row)
                if not text:
                    # Nothing to embed. Stamped anyway so the sweep stops
                    # returning it -- otherwise an empty Signal is re-selected
                    # every minute forever, and the backlog metric never reaches
                    # zero for a reason nobody can find.
                    self.skipped += 1
                    await stamp_index_state(
                        self._session_factory, row.id, IndexState.VECTOR
                    )
                    continue
                claimed.append({"id": row.id, "text": text})
            return claimed

    async def _store(self, item: Mapping[str, Any], vector: Sequence[float]) -> None:
        """Write the vector, then stamp. Never the other way round.

        Stamping first would let a crash in between leave a row claiming to be
        vectored with nothing in Qdrant -- and every reconciler looks for `NULL`,
        so the Signal would be permanently missing from semantic search with
        nothing able to notice.
        """
        if self._indexer is not None:
            await self._indexer.upsert(signal_id=item["id"], vector=list(vector))
        await stamp_index_state(self._session_factory, item["id"], IndexState.VECTOR)


def _text_of(row: Any) -> str:
    """The text to embed, from whichever field carries it."""
    for attribute in ("clean_text", "text", "content", "title"):
        value = getattr(row, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
        if value is not None and hasattr(value, "text"):
            nested = getattr(value, "text", None)
            if isinstance(nested, str) and nested.strip():
                return nested
    return ""


def main() -> None:  # pragma: no cover -- process entry point
    from backend.core.config import get_settings
    from backend.db.session import get_sessionmaker
    from services.llm.embeddings import OpenAICompatibleEmbeddingProvider

    run_worker(
        EmbeddingBackfillWorker(
            session_factory=get_sessionmaker(),
            embeddings=OpenAICompatibleEmbeddingProvider(
                settings=get_settings().embedding
            ),
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
