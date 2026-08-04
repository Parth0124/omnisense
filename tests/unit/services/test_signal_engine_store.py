"""Unit tests for `services/signal_engine/store.py`: stage 7, the commit point.

Everything the ingestion path does before this stage is undone by a crash;
everything after it is repairable by a reconciler. That asymmetry makes a
regression here a corrupt corpus rather than a failed request, so the properties
asserted below are the protocol from `docs/data-stores.md` §5, not the
implementation:

1. **PostgreSQL commits before anything is announced.** Publishing first lets an
   indexing worker build derived state from a transaction that then rolled back,
   and no idempotency key repairs that. Proved twice -- once by ordering a trace
   taken from a real SQLAlchemy commit event, once by making the publish raise and
   showing the row survived intact.
2. **The upsert is genuinely idempotent.** The same Signal written twice is one
   row, not two and not an `IntegrityError`, and `updated_at` moves -- which the
   ORM's `onupdate=` does *not* do for a statement it did not issue
   (`models/orm/mixins.py`). A stale `updated_at` silently disables the
   reconciler, whose backlog query is a staleness window.
3. **Index state is written as NULL**, including over a previously stamped row,
   because that is the mechanism that makes a crash between the PostgreSQL commit
   and the Qdrant upsert self-heal.
4. **Non-canonical dedup members are persisted but not indexed**
   (`docs/signal-model.md` §4.3). They still carry their own row, so they still
   contribute graph edges and trend volume, and they stay out of the index
   backlog by status rather than by a faked timestamp.

The database is in-memory SQLite from `tests/conftest.py` and the publisher is a
fake, so nothing here opens a socket. SQLite is not a weaker substitute for the
statement under test: it has spoken `ON CONFLICT ... DO UPDATE` since 3.24, so the
real upsert -- guard clause included -- is what runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import event, func, select, update
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from models.entity import EntityMention
from models.enums import (
    EntityType,
    Platform,
    SentimentLabel,
    SignalStatus,
    StageName,
    StageStatus,
)
from models.lineage import Lineage, StageRecord
from models.orm.mixins import DEFAULT_TENANT
from models.orm.signal import SignalRow
from models.signal import (
    Author,
    Content,
    EmbeddingRef,
    Engagement,
    Keyword,
    Language,
    MediaRef,
    Sentiment,
    Signal,
    TopicScore,
    signal_id,
)
from services.events.schemas import EventType, SignalEnrichedEvent
from services.signal_engine import store as store_mod
from services.signal_engine.pipeline import (
    EnrichmentContext,
    SignalPipeline,
    Stage,
)
from services.signal_engine.store import StoreStage

pytestmark = pytest.mark.unit

TIMESTAMP = datetime(2026, 7, 28, 14, 2, 11, tzinfo=UTC)
FETCHED_AT = datetime(2026, 7, 28, 14, 29, 55, tzinfo=UTC)

LONG_AGO = datetime(2020, 1, 1, 0, 0, 0)
"""A naive past instant, used to push `updated_at` backwards before a rewrite.

Naive because SQLite has no timestamp type -- `DateTime(timezone=True)` compiles
to `DATETIME` and values come back without an offset (see `tests/unit/models/
test_orm.py`). Comparing "did it move?" against a value planted deep in the past
is what makes the assertion independent of `CURRENT_TIMESTAMP`'s one-second
resolution, which would otherwise make two writes inside the same second look
identical and the test flaky in exactly the wrong direction.
"""


# --------------------------------------------------------------------------- #
# Builders and fakes
# --------------------------------------------------------------------------- #


def make_signal(
    *,
    native_id: str = "t3_1abcde",
    text: str = "Our observability bill tripled after the renewal.",
    status: SignalStatus = SignalStatus.ENRICHED,
    pipeline_version: str = "1.0.0",
    stages: list[StageRecord] | None = None,
    **fields: Any,
) -> Signal:
    """A minimal but valid Signal. Overrides go straight to `Signal.create`."""
    lineage = Lineage(
        pipeline_version=pipeline_version,
        connector_slug="reddit",
        connector_version="0.1.0",
        sync_run_id="run_42",
        fetched_at=FETCHED_AT,
        native_id=native_id,
        status=status,
        stages=stages or [],
        **{k: fields.pop(k) for k in list(fields) if k in Lineage.model_fields},
    )
    return Signal.create(
        platform=Platform.REDDIT,
        native_id=native_id,
        timestamp=TIMESTAMP,
        content=fields.pop("content", None) or Content(text=text),
        lineage=lineage,
        **fields,
    )


def rich_signal() -> Signal:
    """A Signal with every optional field populated.

    Used by the mapping test: a column that silently never receives its value is
    invisible in a test built from a minimal fixture, and the loss only surfaces
    months later as a field that is empty for every row in the corpus.
    """
    return make_signal(
        url="https://reddit.com/r/devops/comments/1abcde",
        author=Author(
            platform_author_id="t2_9xyz",
            handle="ops_gremlin",
            display_name="Ops Gremlin",
            follower_count=1200,
            verified=True,
            account_age_days=900,
        ),
        language=Language(code="en", confidence=0.98, detector="lingua"),
        entities=[
            EntityMention(
                surface="Datadog",
                type=EntityType.COMPANY,
                start=4,
                end=11,
                resolved_id="ent_datadog",
                link_score=0.91,
            )
        ],
        topics=[TopicScore(topic="observability_cost", score=0.8)],
        keywords=[Keyword(term="renewal", weight=0.6)],
        embeddings=[
            EmbeddingRef(
                model="text-embedding-3-small",
                dimensions=1536,
                chunk_index=0,
                collection="omnisense_signals",
                point_id="a1b2c3",
            )
        ],
        sentiment=Sentiment(polarity=-0.7, label=SentimentLabel.NEGATIVE, confidence=0.83),
        engagement=Engagement(raw={"score": 412}, reach=0.7, endorsement=0.6, score=0.65),
        confidence=0.72,
        metadata={"reddit.subreddit": "devops"},
    )


class FakePublisher:
    """Records what was announced, and can fail on demand.

    Keeps an ordered trace shared with the commit listener, because "published
    after the commit" is a statement about order and a fake that only records
    final state cannot answer it.
    """

    def __init__(self, trace: list[str] | None = None) -> None:
        self.events: list[SignalEnrichedEvent] = []
        self.tenants: list[str | None] = []
        self.trace: list[str] = trace if trace is not None else []
        self.error: Exception | None = None

    async def __call__(self, event: SignalEnrichedEvent, *, tenant_id: str | None = None) -> None:
        self.trace.append("publish")
        self.events.append(event)
        self.tenants.append(tenant_id)
        if self.error is not None:
            raise self.error

    @property
    def only(self) -> SignalEnrichedEvent:
        assert len(self.events) == 1, f"expected exactly one event, got {len(self.events)}"
        return self.events[0]


class RecordingLogger:
    """Captures structlog-style calls, for the warnings that are the behaviour."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str, event: str, **fields: Any) -> None:
        self.calls.append((level, event, fields))

    def info(self, event: str, **fields: Any) -> None:
        self._record("info", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._record("warning", event, **fields)

    def events(self, level: str) -> list[str]:
        return [name for lvl, name, _ in self.calls if lvl == level]


@pytest.fixture
def session_factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """The factory the stage is constructed with.

    Configured exactly as `backend/db/session.py` configures the application's --
    a test factory that behaved differently would be testing something the
    application never does.
    """
    return async_sessionmaker(
        bind=orm_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest.fixture
def publisher() -> FakePublisher:
    return FakePublisher()


@pytest.fixture
def stage(
    session_factory: async_sessionmaker[AsyncSession], publisher: FakePublisher
) -> StoreStage:
    return StoreStage(session_factory, publisher)


async def store(stage: StoreStage, signal: Signal) -> None:
    """Run the stage over one Signal, as the pipeline would."""
    await stage.apply(EnrichmentContext(signal=signal))


async def load(session_factory: async_sessionmaker[AsyncSession], signal: Signal) -> SignalRow:
    """Read the row back through a fresh session."""
    async with session_factory() as session:
        row = await session.get(SignalRow, signal.id)
    assert row is not None, f"no row for {signal.id}"
    return row


async def count_rows(session_factory: async_sessionmaker[AsyncSession]) -> int:
    async with session_factory() as session:
        result = await session.execute(select(func.count()).select_from(SignalRow))
    return int(result.scalar_one())


async def age_row(
    session_factory: async_sessionmaker[AsyncSession], signal: Signal, **values: Any
) -> None:
    """Force column values behind the stage's back.

    Used to plant a stale `updated_at` or a stamped `indexed_vector_at` so the
    next write has something observable to change.
    """
    async with session_factory() as session:
        await session.execute(update(SignalRow).where(SignalRow.id == signal.id).values(**values))
        await session.commit()


# --------------------------------------------------------------------------- #
# The commit point
# --------------------------------------------------------------------------- #


class TestCommitBeforeAnnounce:
    """`docs/data-stores.md` §5.1 steps 4 and 5, in that order and no other."""

    async def test_the_event_is_published_only_after_the_commit(
        self,
        orm_engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """A consumer that sees the event must be able to read the row.

        Ordering is taken from SQLAlchemy's own commit event rather than from
        anything the stage reports about itself, because the failure being
        guarded against -- announcing a Signal whose transaction later rolls
        back -- is invisible in every value the stage returns.
        """
        trace: list[str] = []

        @event.listens_for(orm_engine.sync_engine, "commit")
        def _on_commit(connection: Any) -> None:
            trace.append("commit")

        publisher = FakePublisher(trace)
        stage = StoreStage(session_factory, publisher)

        await store(stage, make_signal())

        assert trace == ["commit", "publish"]

    async def test_a_publish_failure_leaves_the_committed_row_intact(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        publisher: FakePublisher,
    ) -> None:
        """A broker outage must not unwind the commit point.

        The row surviving a raising publisher is itself proof of the ordering:
        had the publish been attempted first, the exception would have left the
        session block before `commit()` and there would be no row at all. The
        Signal is then in PostgreSQL but unannounced, which is precisely the
        state the index reconciler (`docs/data-stores.md` §6) exists to repair --
        `indexed_vector_at IS NULL` is still true, so the sweep finds it.
        """
        publisher.error = RuntimeError("redpanda: leader not available")
        stage = StoreStage(session_factory, publisher)
        signal = make_signal()

        with pytest.raises(RuntimeError):
            await store(stage, signal)

        row = await load(session_factory, signal)
        assert row.content_text == signal.content.text
        assert row.status is SignalStatus.ENRICHED
        assert row.indexed_vector_at is None

    async def test_the_exception_is_not_swallowed(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        publisher: FakePublisher,
    ) -> None:
        """A stage never decides its own failure is survivable.

        `FATAL_STAGES` makes that the pipeline's call, and a stage that returned
        normally after a failed publish would report success for a Signal no
        derived store will ever hear about.
        """
        publisher.error = RuntimeError("broker down")
        stage = StoreStage(session_factory, publisher)

        with pytest.raises(RuntimeError, match="broker down"):
            await store(stage, make_signal())

    async def test_the_event_identifies_the_signal_without_carrying_it(
        self,
        stage: StoreStage,
        publisher: FakePublisher,
    ) -> None:
        """`SignalEnrichedEvent` is a reference: consumers re-read the commit point.

        Asserted field by field because these are the values a consumer uses to
        decide what work to skip -- `failed_stages` in particular tells the
        indexing worker not to wait for embeddings that were never produced.
        """
        signal = make_signal(
            status=SignalStatus.PARTIAL,
            confidence=0.42,
            stages=[
                StageRecord(
                    name=StageName.EMBEDDING,
                    version="1.0.0",
                    started_at=FETCHED_AT,
                    duration_ms=12,
                    status=StageStatus.FAILED,
                    error="RateLimitError",
                )
            ],
        )

        await store(stage, signal)

        published = publisher.only
        assert published.EVENT_TYPE is EventType.SIGNAL_ENRICHED
        assert published.signal_id == signal.id
        assert published.partition_key == signal.id
        assert published.platform is Platform.REDDIT
        assert published.native_id == "t3_1abcde"
        assert published.status is SignalStatus.PARTIAL
        assert published.pipeline_version == "1.0.0"
        assert published.signal_schema_version == signal.lineage.schema_version
        assert published.confidence == pytest.approx(0.42)
        assert published.failed_stages == [StageName.EMBEDDING]
        # The body is deliberately absent: a consumer that read content off the
        # topic could index a version the commit point never held.
        assert not hasattr(published, "content")
        assert publisher.tenants == [DEFAULT_TENANT]


# --------------------------------------------------------------------------- #
# Idempotency
# --------------------------------------------------------------------------- #


class TestUpsertIdempotency:
    """Reprocessing is an upsert, never an append (`docs/signal-model.md` §5.3)."""

    async def test_writing_the_same_signal_twice_leaves_one_row(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """The property every replay path depends on.

        At-least-once delivery, a connector's overlap re-fetch and a DLQ redrive
        all recompute the same derived id, so a second write must converge on the
        same row rather than raising `IntegrityError` or duplicating the corpus.
        """
        first = make_signal(text="original body")
        await store(stage, first)

        second = make_signal(text="reprocessed body")
        assert second.id == first.id, "identity must be derived, or nothing converges"
        await store(stage, second)

        assert await count_rows(session_factory) == 1
        row = await load(session_factory, second)
        assert row.content_text == "reprocessed body"
        assert row.content_char_count == len("reprocessed body")

    async def test_the_upsert_moves_updated_at(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """`ON CONFLICT DO UPDATE` bypasses `onupdate=`, so the stage sets it.

        `models/orm/mixins.py` documents this trap: `onupdate=` is a client-side
        SQLAlchemy hook that fires only for UPDATEs the ORM itself issues, and it
        installs no trigger. A stale `updated_at` would silently disable the
        index reconciler, which selects on `updated_at < now() - 15 minutes` --
        the backlog would simply never appear.
        """
        signal = make_signal()
        await store(stage, signal)
        await age_row(session_factory, signal, updated_at=LONG_AGO)

        await store(stage, make_signal(text="reprocessed"))

        row = await load(session_factory, signal)
        assert row.updated_at > LONG_AGO

    async def test_created_at_survives_a_rewrite(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """When the corpus first saw an item is not something reprocessing changes."""
        signal = make_signal()
        await store(stage, signal)
        await age_row(session_factory, signal, created_at=LONG_AGO)

        await store(stage, make_signal(text="reprocessed"))

        assert (await load(session_factory, signal)).created_at == LONG_AGO

    async def test_a_stale_pipeline_version_cannot_overwrite_newer_output(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """The guard from `docs/data-stores.md` §5.2.

        A reindex backfill running old code races live enrichment; without the
        `WHERE excluded.pipeline_version >= signals.pipeline_version` clause it
        would reinstate stale enrichment over newer output and nothing would ever
        notice. Versions are kept single-digit here deliberately -- the guard is a
        text comparison, and `store.py` documents where that stops being safe.
        """
        await store(stage, make_signal(text="from the new pipeline", pipeline_version="2.0.0"))

        await store(stage, make_signal(text="from the old backfill", pipeline_version="1.0.0"))

        row = await load(session_factory, make_signal())
        assert row.content_text == "from the new pipeline"
        assert row.pipeline_version == "2.0.0"

    async def test_a_superseded_write_is_still_announced(
        self,
        stage: StoreStage,
        publisher: FakePublisher,
    ) -> None:
        """Rejection by the guard is not a stage failure.

        The row is committed either way -- by the newer writer -- and the event
        carries identity rather than content, so the redundant announcement costs
        one re-read of the *current* row. Raising instead would send a Signal to
        the DLQ for the crime of already being stored correctly.
        """
        await store(stage, make_signal(pipeline_version="2.0.0"))

        await store(stage, make_signal(pipeline_version="1.0.0"))

        assert len(publisher.events) == 2

    async def test_enrichment_attempts_counts_passes(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """The counter behind the `partial` -> `quarantined` transition.

        A `partial` Signal is re-driven by a sweeper rather than blocking the
        ingest path (`docs/signal-model.md` §5.2), and without a count that grows
        the sweeper can never decide to stop retrying a permanently poisoned
        record.
        """
        signal = make_signal()
        await store(stage, signal)
        assert (await load(session_factory, signal)).enrichment_attempts == 1

        await store(stage, make_signal())
        assert (await load(session_factory, signal)).enrichment_attempts == 2


# --------------------------------------------------------------------------- #
# Index state
# --------------------------------------------------------------------------- #


class TestIndexStateIsWrittenNull:
    """The columns that make a crash between step 4 and step 6 self-heal."""

    async def test_index_state_is_null_on_first_write(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """PostgreSQL must claim only what has actually happened.

        Nothing has been indexed at the moment of the commit -- the workers have
        not even received the event -- so the row says so. That NULL is what the
        reconciler's backlog query selects on; stamping a timestamp here would
        make the row assert an indexing that no store performed, and the drift
        audit would then be comparing the row against its own optimism.
        """
        signal = make_signal()
        await store(stage, signal)

        row = await load(session_factory, signal)
        assert row.indexed_vector_at is None
        assert row.indexed_keyword_at is None
        assert row.graphed_at is None

    async def test_reprocessing_clears_a_previously_stamped_row(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Reprocessing changes the text, so the existing vector is stale.

        A row that kept its old stamps would never be revisited by the
        reconciler, leaving Qdrant and OpenSearch serving the previous pipeline's
        output forever while PostgreSQL held the corrected version.
        """
        signal = make_signal()
        await store(stage, signal)
        stamped = datetime(2026, 7, 29, 9, 0, 0)
        await age_row(
            session_factory,
            signal,
            indexed_vector_at=stamped,
            indexed_keyword_at=stamped,
            graphed_at=stamped,
        )

        await store(stage, make_signal(text="corrected body"))

        row = await load(session_factory, signal)
        assert row.indexed_vector_at is None
        assert row.indexed_keyword_at is None
        assert row.graphed_at is None

    async def test_the_row_appears_in_the_reconcilers_backlog_query(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """The sweep from `docs/data-stores.md` §6, written as the reconciler runs it.

        Asserting against the actual query rather than against the column is what
        makes this a test of the *mechanism*: the whole point of writing NULL is
        that this SELECT returns the Signal.
        """
        signal = make_signal()
        await store(stage, signal)

        async with session_factory() as session:
            backlog = (
                (
                    await session.execute(
                        select(SignalRow.id).where(
                            SignalRow.indexed_vector_at.is_(None),
                            SignalRow.status.in_([SignalStatus.ENRICHED, SignalStatus.PARTIAL]),
                        )
                    )
                )
                .scalars()
                .all()
            )

        assert list(backlog) == [signal.id]


# --------------------------------------------------------------------------- #
# Dedup cluster members
# --------------------------------------------------------------------------- #


def duplicate_of(canonical: Signal, *, native_id: str = "t3_copy") -> Signal:
    """A non-canonical member of the same dedup cluster."""
    return make_signal(
        native_id=native_id,
        text=canonical.content.text,
        status=SignalStatus.DUPLICATE,
        dedup_cluster_id="clu_1",
        duplicate_of=canonical.id,
    )


class TestDuplicateClusterMembers:
    """`docs/signal-model.md` §4.3: never deleted, never indexed."""

    async def test_a_duplicate_is_persisted_with_its_own_row(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Six copies of a press release are evidence of spread, not noise.

        Deleting five of them would destroy the trend signal and the corroboration
        term in `confidence`, so every member keeps its own id, platform,
        engagement and timestamp -- and its own row.
        """
        canonical = make_signal()
        await store(stage, canonical)
        member = duplicate_of(canonical)

        await store(stage, member)

        assert await count_rows(session_factory) == 2
        row = await load(session_factory, member)
        assert row.id == signal_id(Platform.REDDIT, "t3_copy")
        assert row.status is SignalStatus.DUPLICATE
        assert row.dedup_cluster_id == "clu_1"
        assert row.duplicate_of == canonical.id
        # The body is kept: the member still contributes graph edges and trend
        # volume, and both are computed from the stored row.
        assert row.content_text == canonical.content.text

    async def test_a_duplicate_is_not_in_the_index_backlog(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Only the canonical member is embedded and keyword-indexed.

        The exclusion is by `status`, not by a faked `indexed_vector_at`. That
        distinction is the test: stamping a timestamp would claim the duplicate
        *is* in Qdrant, which would make the drift audit chase a mismatch that
        does not exist -- while leaving the column NULL without a status filter
        would have the reconciler republish it on every sweep, forever. The
        backlog indexes in `models/orm/signal.py` are on `(indexed_*_at, status)`
        precisely because the sweep needs both.
        """
        canonical = make_signal()
        await store(stage, canonical)
        member = duplicate_of(canonical)
        await store(stage, member)

        member_row = await load(session_factory, member)
        assert member_row.indexed_vector_at is None
        assert member_row.indexed_keyword_at is None

        async with session_factory() as session:
            backlog = (
                (
                    await session.execute(
                        select(SignalRow.id).where(
                            SignalRow.indexed_vector_at.is_(None),
                            SignalRow.status.in_([SignalStatus.ENRICHED, SignalStatus.PARTIAL]),
                        )
                    )
                )
                .scalars()
                .all()
            )

        assert list(backlog) == [canonical.id]

    async def test_a_duplicate_is_still_announced(
        self,
        stage: StoreStage,
        publisher: FakePublisher,
    ) -> None:
        """The graph and trend workers consume the same topic as indexing.

        Suppressing the event would keep the member's `MENTIONS` edges out of
        Neo4j and its volume out of trend detection; the `status` on the event is
        how the indexing worker knows to skip it instead.
        """
        canonical = make_signal()
        await store(stage, canonical)
        member = duplicate_of(canonical)

        await store(stage, member)

        assert publisher.events[-1].signal_id == member.id
        assert publisher.events[-1].status is SignalStatus.DUPLICATE


# --------------------------------------------------------------------------- #
# Status resolution
# --------------------------------------------------------------------------- #


class TestStatusResolution:
    """Store does not own `status`, but it must not write `raw` either."""

    async def test_a_settled_status_is_written_through(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """`quarantined` is the enrichment sweeper's decision, not this stage's."""
        signal = make_signal(status=SignalStatus.QUARANTINED)

        await store(stage, signal)

        assert (await load(session_factory, signal)).status is SignalStatus.QUARANTINED

    async def test_raw_becomes_enriched_when_every_stage_succeeded(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """A full pipeline pass has just run, so `raw` cannot still be true.

        Writing it would leave the Signal permanently unretrievable
        (`SignalStatus.is_retrievable`) with nothing anywhere to flag it as
        wrong -- the corpus would simply be missing rows nobody could explain.
        """
        signal = make_signal(status=SignalStatus.RAW)

        await store(stage, signal)

        assert (await load(session_factory, signal)).status is SignalStatus.ENRICHED
        # The lineage JSON copy must agree with the column, or a rebuild from
        # `lineage` would resurrect the `raw` the column just corrected.
        assert signal.lineage.status is SignalStatus.ENRICHED

    async def test_raw_becomes_partial_when_a_degradable_stage_failed(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """`docs/signal-model.md` §5.2: a failed enrichment degrades, never discards."""
        signal = make_signal(
            status=SignalStatus.RAW,
            stages=[
                StageRecord(
                    name=StageName.SENTIMENT,
                    version="1.0.0",
                    started_at=FETCHED_AT,
                    duration_ms=5,
                    status=StageStatus.FAILED,
                    error="TimeoutError",
                )
            ],
        )

        await store(stage, signal)

        assert (await load(session_factory, signal)).status is SignalStatus.PARTIAL

    async def test_a_raw_non_canonical_member_becomes_duplicate(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Canonicity outranks stage outcomes when resolving an unset status.

        A member whose `duplicate_of` is set but whose status was never updated
        would otherwise be stored as `enriched` -- retrievable, and returning the
        sixth copy of a press release alongside the canonical one.
        """
        canonical = make_signal()
        member = make_signal(
            native_id="t3_copy",
            status=SignalStatus.RAW,
            dedup_cluster_id="clu_1",
            duplicate_of=canonical.id,
        )

        await store(stage, member)

        assert (await load(session_factory, member)).status is SignalStatus.DUPLICATE


# --------------------------------------------------------------------------- #
# Row mapping
# --------------------------------------------------------------------------- #


class TestRowMapping:
    """Every Signal field that has a column reaches it.

    PostgreSQL is the store every other one is rebuilt from, so a field that
    silently never lands here is not recoverable from anywhere else.
    """

    async def test_a_fully_populated_signal_round_trips(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        signal = rich_signal()

        await store(stage, signal)

        row = await load(session_factory, signal)
        assert row.id == signal.id
        assert row.native_id == "t3_1abcde"
        assert row.tenant_id == DEFAULT_TENANT
        assert row.platform is Platform.REDDIT
        assert row.source is signal.source
        assert row.url == signal.url
        # Author is flattened: the platform id is joined against for per-author
        # aggregates, and a JSON path lookup cannot use a btree index.
        assert row.author_platform_id == "t2_9xyz"
        assert row.author_handle == "ops_gremlin"
        assert row.author_payload is not None
        assert row.author_payload["follower_count"] == 1200
        assert row.content_text == signal.content.text
        assert row.content_char_count == signal.content.char_count
        assert row.language_code == "en"
        assert row.language_confidence == pytest.approx(0.98)
        assert row.entities[0]["resolved_id"] == "ent_datadog"
        assert row.topics[0]["topic"] == "observability_cost"
        assert row.keywords[0]["term"] == "renewal"
        # Embedding *refs*, never vectors -- this column is what makes an
        # embedding-model migration a query rather than a full rescan.
        assert row.embeddings[0]["point_id"] == "a1b2c3"
        assert row.sentiment is not None
        assert row.sentiment["label"] == SentimentLabel.NEGATIVE.value
        assert row.engagement["raw"] == {"score": 412}
        assert row.engagement_score == pytest.approx(0.65)
        assert row.confidence == pytest.approx(0.72)
        assert row.signal_metadata == {"reddit.subreddit": "devops"}
        assert row.connector_slug == "reddit"
        assert row.sync_run_id == "run_42"
        assert row.schema_version == signal.lineage.schema_version
        assert row.pipeline_version == "1.0.0"

    async def test_lineage_is_stored_whole_and_json_serializable(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """Provenance is what makes a claim in a report a claim with a receipt.

        Dumped in JSON mode because the column is `JSONB` on PostgreSQL and plain
        `JSON` on SQLite, and neither can bind a `datetime`.
        """
        signal = make_signal(
            stages=[
                StageRecord(
                    name=StageName.LANGUAGE,
                    version="1.0.0",
                    started_at=FETCHED_AT,
                    duration_ms=3,
                    status=StageStatus.OK,
                )
            ]
        )

        await store(stage, signal)

        row = await load(session_factory, signal)
        assert row.lineage["native_id"] == "t3_1abcde"
        assert row.lineage["connector_slug"] == "reddit"
        assert isinstance(row.lineage["fetched_at"], str)
        assert row.lineage["stages"][0]["name"] == StageName.LANGUAGE.value
        assert row.fetched_at is not None

    async def test_the_raw_object_key_falls_back_to_content_raw_ref(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """The archived original must stay addressable from the row.

        `Lineage.raw_object_key` and `Content.raw_ref` are the same address stated
        twice; a Signal that filled in only the second one would otherwise commit
        with no pointer to its own payload, and reprocessing after a cleaning bug
        would have nothing to reprocess from.
        """
        key = "raw/reddit/2026/07/28/" + "a" * 64 + ".json"
        signal = make_signal(content=Content(text="body", raw_ref=key))

        await store(stage, signal)

        assert (await load(session_factory, signal)).raw_object_key == key

    async def test_media_is_reported_rather_than_silently_dropped(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """There is no media column and no `signal_media` table yet.

        The Signal must still commit -- refusing every image-bearing post would
        be a far larger loss -- but the drop is announced, because an R2 media
        object that no row references is an orphan the R2 sweep will eventually
        propose deleting.
        """
        recorder = RecordingLogger()
        monkeypatch.setattr(store_mod, "logger", recorder)
        signal = make_signal(media=[MediaRef(kind="image", object_key="media/x/abc.png")])

        await store(stage, signal)

        assert "signal_engine.store.media_dropped" in recorder.events("warning")
        assert (await load(session_factory, signal)).content_text == signal.content.text


# --------------------------------------------------------------------------- #
# The stage contract
# --------------------------------------------------------------------------- #


class _NoDialectSession:
    """A session bound to a dialect this stage has no `ON CONFLICT` syntax for."""

    def get_bind(self) -> Any:
        return type("Bind", (), {"dialect": type("Dialect", (), {"name": "mysql"})()})()


class TestStageContract:
    """What the pipeline relies on when it drives this stage."""

    def test_it_satisfies_the_stage_protocol(self, stage: StoreStage) -> None:
        """Structural conformance, checked rather than assumed.

        `SignalPipeline` accepts anything shaped like a `Stage`, so a renamed
        attribute would not fail at wiring time -- it would fail on the first
        record, in production, at the commit point.
        """
        assert isinstance(stage, Stage)
        assert stage.name is StageName.STORE
        assert stage.model_id is None, "storing is deterministic; there is no model"

    async def test_a_missing_signal_raises_instead_of_writing_nothing(
        self,
        stage: StoreStage,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        """A stage order bug must be loud.

        Returning quietly would let the pipeline report a stored Signal that was
        never built, and the absence would only surface as a gap in a time series
        months later.
        """
        with pytest.raises(RuntimeError, match="no Signal on the context"):
            await stage.apply(EnrichmentContext())

        assert await count_rows(session_factory) == 0

    async def test_the_pipeline_treats_a_store_failure_as_fatal(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        publisher: FakePublisher,
    ) -> None:
        """Driven through the real `SignalPipeline`, not simulated.

        Store is in `FATAL_STAGES` because a Signal that was never committed does
        not exist. The pipeline -- not the stage -- makes that call, and the
        result must carry enough to build a replayable DLQ record.
        """
        publisher.error = RuntimeError("broker down")
        pipeline = SignalPipeline([StoreStage(session_factory, publisher)])
        ctx = EnrichmentContext(signal=make_signal())

        result = await pipeline.run(ctx)

        assert result.fatal_stage is StageName.STORE
        assert result.status is SignalStatus.QUARANTINED
        assert result.error == "RuntimeError"
        assert not result.succeeded
        # Only the exception class travels: a driver message can echo the
        # statement that caused it, and statements carry fetched content.
        assert result.outcomes[-1].error == "RuntimeError"

    def test_an_unsupported_dialect_refuses_to_guess(self) -> None:
        """A plain INSERT would pass every test until the first replayed message.

        It would then raise `IntegrityError` under redelivery pressure -- exactly
        when the ingestion path can least afford a new failure mode -- so the
        unsupported dialect is named at the point of use instead.
        """
        with pytest.raises(NotImplementedError, match="mysql"):
            store_mod._upsert(_NoDialectSession(), {"id": "sig_1"})  # type: ignore[arg-type]

    async def test_the_tenant_is_carried_onto_the_row_and_the_event(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        publisher: FakePublisher,
    ) -> None:
        """Phase 1 is single-tenant; the column and the envelope field are not.

        A row written under the wrong tenant is not repairable by any reconciler
        -- it is simply another tenant's data -- so the value is explicit at the
        write rather than defaulted somewhere downstream.
        """
        stage = StoreStage(session_factory, publisher, tenant_id="acme")
        signal = make_signal()

        await store(stage, signal)

        assert (await load(session_factory, signal)).tenant_id == "acme"
        assert publisher.tenants == ["acme"]
