"""Stage 7 -- Store: the commit point of the whole ingestion path.

Everything before this stage is speculative work held in memory. This stage is
where a Signal starts existing, and `docs/data-stores.md` §5.1 fixes exactly how:

    4. INSERT ... ON CONFLICT (id) DO UPDATE   <- PostgreSQL, the commit point
    5. publish SignalEnrichedEvent             <- **after** the commit, never before
    6. Qdrant / OpenSearch upserts             <- their own workers, off that topic
    7. Neo4j MERGE                             <- its own consumer group
    8. stamp indexed_vector_at / indexed_keyword_at / graphed_at

Steps 6-8 are deliberately *not* performed here. Writing a vector inline would
put an embedding-provider timeout, an OpenSearch rolling restart and a Neo4j
compaction pause directly on the ingestion path -- a slow index would become the
reason ingestion stalls, and `docs/architecture.md` §7.3 is explicit that Qdrant,
OpenSearch and Neo4j degrade rather than halt. Publishing one event and letting
four independent consumer groups fan out is what keeps their lag *their* problem.

Why ordering and idempotency carry the whole weight
---------------------------------------------------
There is no distributed transaction across five stores, and adding one would
trade a manageable consistency problem for an unmanageable availability problem.
Three properties replace it, and this module is where two of them live:

**A single commit point.** The Signal exists exactly when the PostgreSQL
transaction commits, and never before. Publishing first would let an indexing
worker read a row from a transaction that then rolled back -- a vector that
cites nothing, invisible to `scripts/reindex.py`, and undetectable by any
reconciler because reconcilers compare *against* PostgreSQL.

**Idempotent writes.** `Signal.id` is derived from `(platform, native_id)`
(`docs/signal-model.md` §4.1), so a replayed Kafka message, a connector's overlap
re-fetch and a DLQ redrive all recompute the same id and converge on one row.
`ON CONFLICT (id) DO UPDATE` is what makes "converge" true rather than
"IntegrityError". That upsert bypasses SQLAlchemy's ORM unit of work, so
`updated_at` -- whose `onupdate=` is a *client-side* construct that emits SQL only
for ORM-issued UPDATEs, installing no trigger -- is set explicitly here. See
`models/orm/mixins.py`, which documents this exact trap; a stale `updated_at`
would silently disable the reconciler, whose backlog query is
`indexed_vector_at IS NULL AND updated_at < now() - interval '15 minutes'`.

**Index state written as NULL.** The three index-state columns are set to NULL on
every write, including a rewrite of a Signal that was previously indexed. That is
what makes a crash between the PostgreSQL commit and the Qdrant upsert self-heal:
the row says "PostgreSQL does not believe this is indexed anywhere", the
reconciler (`docs/data-stores.md` §6) finds it and republishes, and the indexing
worker -- idempotent by point id -- redoes the work harmlessly. Setting them to
NULL on a *re*-write matters just as much: reprocessing changes the text, so the
vector and the keyword document that already exist are stale, and a row still
claiming to be indexed would never be revisited.

Known gap: `Signal.media` has nowhere to go
-------------------------------------------
`models/orm/signal.py` has no media column and there is no `signal_media` table,
so the `MediaRef` list -- which carries the R2 `object_key` of each archived image
or video and the `transcript_ref` that makes a video searchable -- cannot be
persisted at the commit point today. Since PostgreSQL is the store everything else
is rebuilt from, an unreferenced media object is an orphan that the R2 orphan sweep
(`docs/data-stores.md` §6) will eventually propose deleting.

Dropping it silently is the one thing this stage must not do, so a Signal carrying
media is written with a `store.media_dropped` warning naming the count. The fix is
a `signal_media` table keyed by `signal_id`; that is a schema change and an
Alembic migration, neither of which belongs in a pipeline stage.

The upsert guard, and why it compares an integer
------------------------------------------------
`docs/data-stores.md` §5.2 specifies a guard so a slow reprocess running old code
cannot reinstate stale enrichment over newer output. It is written here as
`WHERE excluded.pipeline_version_ord >= signals.pipeline_version_ord` -- against
the **integer ordinal**, never against `pipeline_version` itself.

That distinction is the whole point. `pipeline_version` is a `VARCHAR`, so the
database orders it as *text*, and text ordering matches semantic-version ordering
only while every component stays one digit wide. `'1.10.0' >= '1.9.0'` is
**False** as text and **True** as versions, so a guard written against the string
would invert on the first bump past `.9`: silently rejecting the newer pipeline's
output while admitting a stale backfill -- precisely the corruption the guard
exists to prevent, and it would surface as a `store.superseded` warning rather
than an error.

`models.lineage.pipeline_version_ordinal` encodes the semver as
`major*1_000_000 + minor*1_000 + patch`, `_row_values` populates the column on
every write, and `pipeline_version` is retained for display only. The same
ordinal serves as the OpenSearch external document version §5.2 requires.

Layer note: `services/` (L2). Takes its session factory and its publisher as
constructor arguments and constructs neither, so the unit suite exercises the real
statement against in-memory SQLite with no broker anywhere in the process.
"""

from __future__ import annotations

from typing import Any, Final, Protocol, cast

from sqlalchemy import CursorResult, Table, func
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.dml import Insert

from backend.core.logging import get_logger
from models.enums import SignalStatus, StageName
from models.lineage import pipeline_version_ordinal
from models.orm.mixins import DEFAULT_TENANT
from models.orm.signal import SignalRow
from models.signal import Signal
from services.events.schemas import SignalEnrichedEvent
from services.signal_engine.pipeline import EnrichmentContext

__all__ = ["SignalPublisher", "StoreStage", "publish_enriched"]

logger = get_logger(__name__)

STORE_STAGE_VERSION: Final = "1.0.0"
"""Version of this stage's implementation, recorded in `lineage.stages[]`."""

_SIGNALS: Final[Table] = cast(Table, SignalRow.__table__)
"""The `signals` table, addressed as Core rather than as the ORM entity.

`ON CONFLICT` is a Core construct and the statement is built from column *names*,
which matters in one place: the `metadata` column is the `signal_metadata`
attribute in Python, because `metadata` is reserved on a Declarative class.
Building from the table keeps the column names in this module identical to the
ones in `models/orm/signal.py`. The cast is because `DeclarativeBase.__table__`
is typed as the wider `FromClause`.
"""

_NEVER_OVERWRITTEN: Final[frozenset[str]] = frozenset({"id", "created_at"})
"""Columns the `DO UPDATE` half must not touch.

`id` is the conflict target -- assigning it would be a no-op at best and a
primary-key rewrite at worst. `created_at` records when the Signal first entered
the corpus; overwriting it on every reprocess would erase the only record of when
an item was actually first seen, which is exactly the question asked when
auditing a trend's origin.
"""

_COMPUTED_ON_UPDATE: Final[frozenset[str]] = frozenset({"updated_at", "enrichment_attempts"})
"""Columns whose update value is computed, not copied from the incoming row."""


class SignalPublisher(Protocol):
    """How a committed Signal is announced on `omnisense.signals.enriched`.

    A callable protocol rather than a Kafka client, for the same reason
    `services/events/consumer.py` declares `DlqPublisher`: the unit suite must be
    able to substitute a ten-line fake without any part of `aiokafka` being
    constructed, and the stage has no business knowing about topics, envelopes or
    partition keys -- `services/events/producer.py` owns all three as the single
    write path onto the log.
    """

    async def __call__(
        self, event: SignalEnrichedEvent, *, tenant_id: str | None = None
    ) -> None: ...


async def publish_enriched(event: SignalEnrichedEvent, *, tenant_id: str | None = None) -> None:
    """The production `SignalPublisher`: publish through the process-wide producer.

    The import is function-local on purpose. `services/events/producer.py` owns a
    module-level singleton, and importing it at module scope would mean that
    merely importing this stage -- which the test suite does at collection time --
    drags the aiokafka client into the process.
    """
    from services.events.producer import publish

    await publish(event, tenant_id=tenant_id)


class StoreStage:
    """Persist the Signal to PostgreSQL, then announce it. Satisfies `Stage`.

    Fatal on failure (`FATAL_STAGES`), and correctly so: every other stage
    degrades to a documented empty value, but a Signal that was never committed
    does not exist, cannot be cited, and must go to the DLQ with its R2 key so a
    replay can produce it once the cause is fixed.

    Stateless per record -- one instance is shared by a worker and driven
    concurrently, so everything about one Signal lives on the stack.
    """

    name: StageName = StageName.STORE
    version: str = STORE_STAGE_VERSION

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        publisher: SignalPublisher,
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._publish = publisher
        # Phase 1 is single-tenant, but the column and the envelope field both
        # exist from day one (`models/orm/mixins.py`), and a Signal written with
        # the wrong tenant is not repairable by any reconciler -- it is simply
        # another tenant's row.
        self._tenant_id = tenant_id

    @property
    def model_id(self) -> str | None:
        """No model. Storing is deterministic and replays identically."""
        return None

    async def apply(self, ctx: EnrichmentContext) -> None:
        """Commit the Signal, then publish. Raises on any failure.

        Never catches its own exception: the pipeline classifies failures from
        `FATAL_STAGES`, and a stage that swallowed a database error would report
        success for a Signal that is nowhere.

        Driver exceptions propagate unwrapped rather than being folded into
        `DependencyUnavailableError`. The distinction they carry is the one the
        worker needs: an `IntegrityError` is poison and belongs in the DLQ
        immediately, while an `OperationalError` is a retry. Collapsing both into
        one retryable error turns a bad row into an infinite redelivery loop.
        """
        signal = ctx.require_signal()
        status = self._resolve_status(signal)

        if signal.media:
            # See the module docstring. There is no column and no table for these
            # yet, and the alternative to saying so out loud is an R2 object that
            # nothing references and the orphan sweep eventually deletes.
            logger.warning(
                "signal_engine.store.media_dropped",
                signal_id=signal.id,
                media_count=len(signal.media),
                reason="no signal_media table exists yet",
            )

        values = _row_values(signal, tenant_id=self._tenant_id)

        async with self._session_factory() as session:
            statement = _upsert(session, values)
            result = cast(CursorResult[Any], await session.execute(statement))
            await session.commit()
            # Zero rows means the `pipeline_version` guard rejected the update --
            # the insert half always affects one row when the id is new.
            applied = result.rowcount != 0

        # --- the commit point has passed; the Signal now exists --------------
        #
        # The publish is deliberately outside the session block. Holding a
        # pooled connection open across a broker round trip couples PostgreSQL
        # pool occupancy to Kafka latency, and under a broker slowdown that is
        # how a store stage exhausts the connection pool for every other query
        # in the process.
        if not applied:
            # The version guard rejected the update: a newer pipeline already
            # committed this Signal. Not an error -- that is the guard doing its
            # job -- but a backfill racing live traffic should be visible.
            logger.warning(
                "signal_engine.store.superseded",
                signal_id=signal.id,
                pipeline_version=signal.lineage.pipeline_version,
            )
        else:
            logger.info(
                "signal_engine.store.committed",
                signal_id=signal.id,
                platform=signal.platform.value,
                status=status.value,
                pipeline_version=signal.lineage.pipeline_version,
            )

        # Published either way. The event carries identity, not content, and a
        # consumer re-reads the committed row -- so announcing a Signal whose
        # write was superseded costs one redundant re-index of the *newer* row,
        # whereas staying silent risks leaving the derived stores waiting on an
        # announcement the other writer may itself have failed to send.
        await self._publish(
            SignalEnrichedEvent(
                signal_id=signal.id,
                platform=signal.platform,
                native_id=signal.lineage.native_id,
                status=status,
                pipeline_version=signal.lineage.pipeline_version,
                signal_schema_version=signal.lineage.schema_version,
                confidence=signal.confidence,
                failed_stages=signal.lineage.failed_stages(),
            ),
            tenant_id=self._tenant_id,
        )

    # ------------------------------------------------------------ internals --

    @staticmethod
    def _resolve_status(signal: Signal) -> SignalStatus:
        """Settle the lifecycle status being written, filling it in if nobody did.

        Store does not own `status`; dedup owns `duplicate`, the enrichment
        sweeper owns `quarantined`, and stage 6b may already have settled
        `enriched`/`partial`. Anything other than `raw` is therefore written
        through untouched.

        `raw` is the one value that cannot be correct here -- the pipeline has by
        definition just run -- and writing it would make the Signal permanently
        unretrievable (`SignalStatus.is_retrievable`) with nothing to flag it.
        So it is resolved from what `lineage` already knows: a non-canonical
        cluster member is `duplicate`, any failed degradable stage makes the
        Signal `partial` (`docs/signal-model.md` §5.2), otherwise `enriched`.

        The resolution is written back onto `lineage` before the row is built so
        the `status` column and the lineage JSON copy of it cannot disagree.
        """
        if signal.lineage.status is not SignalStatus.RAW:
            return signal.lineage.status

        if not signal.is_canonical:
            resolved = SignalStatus.DUPLICATE
        elif signal.lineage.failed_stages():
            resolved = SignalStatus.PARTIAL
        else:
            resolved = SignalStatus.ENRICHED

        signal.lineage.status = resolved
        return resolved


def _upsert(session: AsyncSession, values: dict[str, Any]) -> Insert:
    """Build the dialect's `INSERT ... ON CONFLICT (id) DO UPDATE`.

    `ON CONFLICT` is not in core SQLAlchemy -- it is spelled per dialect -- so the
    construct is chosen from the bound dialect. An unknown dialect raises rather
    than falling back to a plain `INSERT`: that fallback would work perfectly
    until the first replayed message, then fail with an `IntegrityError` that
    reads like a connector bug and would take the whole ingestion path down
    exactly when it is under redelivery pressure.
    """
    dialect = session.get_bind().dialect.name
    if dialect == "postgresql":
        return _on_conflict_do_update(postgresql_insert(_SIGNALS).values(**values), values)
    if dialect == "sqlite":
        # The unit suite (`tests/conftest.py`). SQLite has spoken
        # `ON CONFLICT ... DO UPDATE` since 3.24, so the real statement -- guard
        # clause included -- is exercised rather than a simplified stand-in.
        return _on_conflict_do_update(sqlite_insert(_SIGNALS).values(**values), values)
    raise NotImplementedError(
        f"no ON CONFLICT syntax implemented for SQLAlchemy dialect {dialect!r}; "
        "the Signal upsert must be idempotent, and a plain INSERT is not"
    )


def _on_conflict_do_update(statement: Any, values: dict[str, Any]) -> Insert:
    """Attach the shared `DO UPDATE` half to whichever dialect's `INSERT` this is.

    `statement` is `Any` because the two dialects' `Insert` classes are unrelated
    in the type system while being identical in the two methods used here. The
    alternative -- writing this clause out once per dialect -- is how the
    PostgreSQL and SQLite statements drift apart, and the drift would only ever
    show up in production, where only one of the two is exercised.
    """
    assignments: dict[str, Any] = {
        column: statement.excluded[column]
        for column in values
        if column not in _NEVER_OVERWRITTEN and column not in _COMPUTED_ON_UPDATE
    }
    # `onupdate=func.now()` on the mixin never fires for this statement -- it is a
    # client-side SQLAlchemy hook, not a trigger -- so the server clock is asked
    # for explicitly. Server-side rather than Python-side because workers run on
    # several hosts and a skewed clock makes the reconciler's staleness window
    # meaningless (`models/orm/mixins.py`).
    assignments["updated_at"] = func.now()
    # Counts enrichment passes, not failures. `partial` rows are re-driven by a
    # sweeper, and this is the counter that lets it give up and quarantine a
    # Signal instead of retrying a permanently poisoned record forever.
    assignments["enrichment_attempts"] = _SIGNALS.c.enrichment_attempts + 1

    upsert: Insert = statement.on_conflict_do_update(
        index_elements=[_SIGNALS.c.id],
        set_=assignments,
        # `docs/data-stores.md` §5.2: the only defence against a reindex backfill
        # racing a live update and reinstating stale enrichment.
        #
        # Compared on the ORDINAL, never on `pipeline_version` itself. That
        # column is a VARCHAR, so the database orders it as text, where
        # `'1.10.0' >= '1.9.0'` is False -- the guard would invert as soon as any
        # component reached 10, rejecting the newer pipeline and admitting the
        # stale backfill. See `models.lineage.pipeline_version_ordinal`.
        where=statement.excluded.pipeline_version_ord >= _SIGNALS.c.pipeline_version_ord,
    )
    return upsert


def _row_values(signal: Signal, *, tenant_id: str) -> dict[str, Any]:
    """Flatten a Signal into `signals` column values.

    Keyed by *column* name, not by ORM attribute name, because the statement is
    built against the Core table (see `_SIGNALS`). The two differ in one place:
    the `metadata` column is the `signal_metadata` attribute in Python, since
    `metadata` is reserved on a Declarative class.

    Nested models are dumped with `mode="json"` so that datetimes and enums
    become strings. The JSON columns are `JSONB` on PostgreSQL and plain `JSON`
    on SQLite, and neither can bind a `datetime`.
    """
    lineage = signal.lineage
    author = signal.author

    return {
        "id": signal.id,
        "native_id": lineage.native_id,
        "tenant_id": tenant_id,
        "source": signal.source,
        "platform": signal.platform,
        "url": signal.url,
        # `author_platform_id` is promoted out of the JSON blob because
        # per-author aggregates join on it and a JSON path lookup cannot use a
        # btree index. The rest of the author is read whole or not at all.
        "author_platform_id": author.platform_author_id if author else None,
        "author_handle": author.handle if author else None,
        "author_payload": author.model_dump(mode="json") if author else None,
        "timestamp": signal.timestamp,
        "fetched_at": lineage.fetched_at,
        "content_title": signal.content.title,
        "content_text": signal.content.text,
        "content_char_count": signal.content.char_count,
        "content_truncated": signal.content.truncated,
        "content_type": signal.content.content_type,
        # Lineage is the authority on where the immutable original lives;
        # `content.raw_ref` is the same address restated for readers that only
        # hold a `Content`. Lineage wins, and the fallback keeps a Signal built
        # by a connector that filled in only one of them addressable.
        "raw_object_key": lineage.raw_object_key or signal.content.raw_ref,
        "raw_sha256": lineage.raw_sha256 or signal.content.raw_sha256,
        "language_code": signal.language.code,
        "language_confidence": signal.language.confidence,
        "entities": [mention.model_dump(mode="json") for mention in signal.entities],
        "topics": [topic.model_dump(mode="json") for topic in signal.topics],
        "keywords": [keyword.model_dump(mode="json") for keyword in signal.keywords],
        "embeddings": [ref.model_dump(mode="json") for ref in signal.embeddings],
        "sentiment": signal.sentiment.model_dump(mode="json") if signal.sentiment else None,
        "engagement": signal.engagement.model_dump(mode="json"),
        # Promoted out of the JSON blob because it is an ORDER BY target.
        "engagement_score": signal.engagement.score,
        "confidence": signal.confidence,
        "metadata": signal.metadata,
        "lineage": lineage.model_dump(mode="json"),
        "status": lineage.status,
        "dedup_cluster_id": lineage.dedup_cluster_id,
        "duplicate_of": lineage.duplicate_of,
        "schema_version": lineage.schema_version,
        "pipeline_version": lineage.pipeline_version,
        "pipeline_version_ord": pipeline_version_ordinal(lineage.pipeline_version),
        "connector_slug": lineage.connector_slug,
        "sync_run_id": lineage.sync_run_id,
        # The three index-state columns are NULL on every write, which is what
        # makes a crash between here and the Qdrant upsert self-heal: the row
        # says "not indexed", the reconciler finds it and republishes, and the
        # indexing worker's upsert-by-point-id makes the redo a no-op if it had
        # in fact already happened. Stamping them is step 8, and belongs to the
        # worker that actually did the indexing -- claiming it here would make
        # PostgreSQL assert something no store has done yet, and the drift audit
        # would then be comparing a row against its own optimism.
        #
        # Non-canonical dedup members are stored with everything above intact --
        # they still contribute graph edges and trend volume
        # (`docs/signal-model.md` §4.3) -- and are kept out of the index backlog
        # by `status = 'duplicate'`, which is precisely why the backlog indexes
        # in `models/orm/signal.py` are on `(indexed_*_at, status)`. Faking a
        # timestamp to exclude them would instead assert they *are* indexed,
        # and the drift audit would chase a mismatch that does not exist.
        "indexed_vector_at": None,
        "indexed_keyword_at": None,
        "graphed_at": None,
        "updated_at": func.now(),
        "enrichment_attempts": 1,
    }
