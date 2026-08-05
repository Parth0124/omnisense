"""The eight-stage pipeline, assembled and driven over a real provider payload.

Every other suite under `tests/unit/services/` exercises one stage against a
context somebody handed it a `Signal` on. This one is the only place the
`Clean -> Normalize -> Language -> Entities -> Sentiment -> Embedding -> Scoring
-> Store` chain of Design Doc §6 is built the way `workers/enrichment_worker.py`
will build it and run from provider bytes to a committed row. That is what makes
it worth more than the sum of the stage suites: the failures it catches are the
ones that live *between* stages and are invisible from either side.

Three of those seams are asserted here and nowhere else.

**Stage 2 exists and produces a Signal.** Until it did, `ctx.signal` was never
assigned and every stage after Clean raised out of `require_signal()`. The
pipeline could not emit a Signal at all, and no single-stage test could notice,
because each of them starts by putting a Signal on the context by hand.

**Identity survives the rebuild.** `RawRecordEvent` carries the *address* of the
payload in R2 and its provenance, never the connector's mapped output
(`docs/data-stores.md` §5.1), so stage 2 maps the payload a second time. The
`native_id` it derives has to equal the one that already keyed the R2 object and
the Kafka partition, or one item exists twice across five stores.

**`pipeline_version` becomes real.** Connectors stamp `"0.0.0"` because they run
no enrichment. `services/signal_engine/store.py` makes that field the upsert
guard, so a Signal still carrying the zero version loses every race forever. The
value that reaches PostgreSQL must be the version of the pipeline that ran.

The payload is the newest-format entry of `tests/fixtures/payloads/rss_20_sample.xml`,
parsed by the same `feedparser` the RSS connector uses and wrapped in the same
`omnisense` envelope, so the field paths under test are the real ones rather than
a hand-written approximation of them. The LLM and the embedding provider are the
offline fakes from `services/llm/`; the database is in-memory SQLite from
`tests/conftest.py`; the publisher is four lines. Nothing here opens a socket and
nothing here needs a service running.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from time import struct_time
from typing import Any

import feedparser
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from connectors.exceptions import NormalizationError
from connectors.news.rss import ENVELOPE_KEY
from connectors.normalize.mapper import UNENRICHED_PIPELINE_VERSION
from models.enums import Platform, SignalStatus, SourceCategory, StageName, StageStatus
from models.orm.signal import SignalRow
from services.events.schemas import RawRecordEvent, SignalEnrichedEvent
from services.llm.embeddings import FakeEmbeddingProvider
from services.llm.provider import FakeLLMProvider
from services.signal_engine.cleaning import EMAIL_PLACEHOLDER, CleaningStage, RegexRedactor
from services.signal_engine.embeddings import EmbeddingStage, InMemoryVectorSink
from services.signal_engine.enrichment import InMemoryCohortBaseline, ScoringStage
from services.signal_engine.entities import EntityExtractionStage
from services.signal_engine.language import LangdetectDetector, LanguageStage
from services.signal_engine.normalize import (
    NormalizeStage,
    default_field_map_resolver,
)
from workers.enrichment_worker import build_pipeline as production_build_pipeline
from services.signal_engine.pipeline import (
    EnrichmentContext,
    PipelineResult,
    SignalPipeline,
)
from services.signal_engine.sentiment import SentimentStage
from services.signal_engine.store import StoreStage

pytestmark = pytest.mark.unit


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "payloads"
RSS_BYTES = (FIXTURES / "rss_20_sample.xml").read_bytes()
REDDIT_LISTING = json.loads((FIXTURES / "reddit_listing_new_page1.json").read_text())

FEED_URL = "https://news.example.com/feed.xml"
FULL_BODY_GUID = "tag:news.example.com,2026:post-4166"
"""The one entry in the RSS fixture carrying a whole article in `content:encoded`.

Chosen over the teaser entries because it is the only one that exercises the
`content.0.value` path, `truncated=False`, and a body long enough for the
language detector and the chunker to have something to say about.
"""

COLLECTION = "omnisense_signals_test"
"""Named explicitly rather than read from settings: the collection travels into
every `EmbeddingRef.collection` this test asserts on, and a config default that
changed under it would fail the assertion for a reason that has nothing to do
with the pipeline."""


# --------------------------------------------------------------------------- #
# Building the record the worker would receive
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    """Render feedparser's output as something `json.dumps` accepts.

    Mirrors `connectors.news.rss._jsonable`, which the connector applies before
    the runtime PUTs the payload to R2. The `struct_time` branch is the whole
    point: `published_parsed` is the first path the RSS field map tries, and a
    payload that could not be serialized is a payload that could never have come
    back out of the object store.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, struct_time):
        moment = dt.datetime(*value[:6], tzinfo=dt.UTC)
        return moment.isoformat().replace("+00:00", "Z")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def rss_entry_payload(guid: str = FULL_BODY_GUID) -> dict[str, Any]:
    """One entry from the RSS fixture, shaped as `RssConnector` emits it.

    Real bytes through real feedparser, plus the `omnisense` envelope the
    connector adds -- `native_id` and `author_id` in particular, which the field
    map addresses at `omnisense.native_id` and `omnisense.author_id` and which no
    feed carries on its own (RSS has bylines, not author ids).
    """
    parsed = feedparser.parse(RSS_BYTES)
    feed = parsed.get("feed") or {}
    for entry in parsed.entries:
        if entry.get("id") != guid:
            continue
        payload: dict[str, Any] = dict(_jsonable(dict(entry)))
        payload["enclosures"] = _jsonable(list(entry.get("enclosures") or ()))
        payload[ENVELOPE_KEY] = {
            "feed_url": FEED_URL,
            "feed_title": feed.get("title"),
            "feed_link": feed.get("link"),
            "bozo": bool(parsed.get("bozo")),
            # Host-scoped, as `_author_identity` builds it: "John Smith" writes
            # for more than one publication.
            "author_id": "news.example.com:dmitri@news.example.com",
            # Rule 1 of the identity ladder, derived by the connector in fetch()
            # so the Kafka reference and the Signal cannot disagree.
            "native_id": guid,
        }
        return payload
    raise AssertionError(f"no entry {guid!r} in the RSS fixture")


def archived_bytes(payload: dict[str, Any]) -> bytes:
    """The object the runtime PUT to R2.

    `docs/data-stores.md` §5.1 step 1 archives the *payload* as JSON at
    `raw/{platform}/{yyyy}/{mm}/{dd}/{payload_sha256}.json`, which is why the
    record that reaches the pipeline declares `application/json` even for a
    connector whose provider speaks XML. Stage 1 therefore takes its structured
    branch and leaves the body to the field map.
    """
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def raw_record_event(payload: dict[str, Any], **overrides: Any) -> RawRecordEvent:
    """The `omnisense.records.raw` message for this payload."""
    raw = archived_bytes(payload)
    fields: dict[str, Any] = {
        "platform": Platform.RSS,
        "native_id": payload[ENVELOPE_KEY]["native_id"],
        "connector_slug": "rss",
        "connector_version": "0.1.0",
        "sync_run_id": "run_01J8XN5Q2P",
        "fetched_at": dt.datetime(2026, 7, 27, 8, 0, tzinfo=dt.UTC),
        "raw_object_key": f"raw/rss/2026/07/27/{hashlib.sha256(raw).hexdigest()}.json",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "raw_content_type": "application/json",
        "source_url": FEED_URL,
        "request_fingerprint": "rss:news.example.com/feed.xml",
    }
    fields.update(overrides)
    return RawRecordEvent(**fields)


def context_for(payload: dict[str, Any], **overrides: Any) -> EnrichmentContext:
    """The context `workers/enrichment_worker.py` assembles per record."""
    raw = archived_bytes(payload)
    fields: dict[str, Any] = {
        "raw_bytes": raw,
        "content_type": "application/json",
        "payload": payload,
        "record": raw_record_event(payload),
    }
    fields.update(overrides)
    return EnrichmentContext(**fields)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakePublisher:
    """Records what stage 7 announced instead of producing to Redpanda."""

    def __init__(self) -> None:
        self.events: list[SignalEnrichedEvent] = []

    async def __call__(
        self, event: SignalEnrichedEvent, *, tenant_id: str | None = None
    ) -> None:
        self.events.append(event)


def scripted_llm() -> FakeLLMProvider:
    """Two scripted responses: one extraction, one sentiment verdict.

    Scripted rather than defaulted so the call count is pinned. Stages 4 and 5
    run once per ingested Signal each, and an accidental third call is a doubling
    of the model bill that no assertion on the output would reveal --
    `FakeLLMProvider` raises when the script is exhausted, which is the assertion.

    The surfaces are real substrings of the fixture body and the offsets are
    omitted on purpose: stage 4 locates and verifies every mention itself, so
    supplying offsets here would test the fixture rather than the locator.
    """
    return FakeLLMProvider(
        script=[
            {
                "mentions": [
                    {"surface": "Grafana", "type": "Product", "candidates": ["ent_grafana"]},
                    {"surface": "Loki", "type": "Product", "candidates": ["ent_loki"]},
                ],
                "topics": [
                    {"topic": "observability-tooling", "score": 0.91},
                    {"topic": "vendor-pricing", "score": 0.84},
                ],
            },
            {
                "polarity": 0.45,
                "label": "positive",
                "subjectivity": 0.40,
                "confidence": 0.86,
                "targets": [],
            },
        ]
    )


@pytest.fixture
def session_factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Configured exactly as `backend/db/session.py` configures the real one."""
    return async_sessionmaker(
        bind=orm_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


def build_pipeline(
    session_factory: async_sessionmaker[AsyncSession],
    publisher: FakePublisher,
    *,
    llm: FakeLLMProvider,
    embedder: FakeEmbeddingProvider,
    sink: InMemoryVectorSink,
    pipeline_version: str = "1.4.0",
) -> SignalPipeline:
    """The whole of Design Doc §6, in order, with every dependency injected.

    Delegates to `workers.enrichment_worker.build_pipeline` rather than listing
    the stages again. That indirection is the point: the *order* and the
    *dependencies* are what this suite asserts on, and asserting them against a
    local copy would prove only that the copy is self-consistent. Pointing at the
    shipping factory means a stage added, removed or reordered in production
    fails here -- which is what the assertions were always meant to catch.
    """
    return production_build_pipeline(
        llm=llm,
        embeddings=embedder,
        baseline=InMemoryCohortBaseline(),
        session_factory=session_factory,
        publisher=publisher,
        language_detector=LangdetectDetector(),
        collection=COLLECTION,
        vector_sink=sink,
        pipeline_version=pipeline_version,
    )


# --------------------------------------------------------------------------- #
# The full pass
# --------------------------------------------------------------------------- #


class TestFullPipeline:
    """Provider bytes in, a committed `enriched` Signal out."""

    @pytest.fixture
    def publisher(self) -> FakePublisher:
        return FakePublisher()

    @pytest.fixture
    def llm(self) -> FakeLLMProvider:
        return scripted_llm()

    @pytest.fixture
    def embedder(self) -> FakeEmbeddingProvider:
        return FakeEmbeddingProvider(dimensions=8)

    @pytest.fixture
    def sink(self) -> InMemoryVectorSink:
        return InMemoryVectorSink()

    @pytest.fixture
    async def result(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        publisher: FakePublisher,
        llm: FakeLLMProvider,
        embedder: FakeEmbeddingProvider,
        sink: InMemoryVectorSink,
    ) -> PipelineResult:
        pipeline = build_pipeline(
            session_factory,
            publisher,
            llm=llm,
            embedder=embedder,
            sink=sink,
        )
        return await pipeline.run(context_for(rss_entry_payload()))

    async def test_every_stage_succeeds_and_the_signal_is_enriched(
        self, result: PipelineResult
    ) -> None:
        """`enriched` means all eight ran, in order, with nothing degraded.

        Asserted on the outcome list rather than only on the final status,
        because `partial` and `enriched` differ by one failed stage and the
        status alone will not say which.
        """
        assert result.status is SignalStatus.ENRICHED
        assert result.fatal_stage is None
        assert result.failed_stages == []
        assert result.succeeded
        assert [outcome.name for outcome in result.outcomes] == [
            StageName.CLEAN,
            StageName.NORMALIZE,
            StageName.LANGUAGE,
            StageName.ENTITIES,
            StageName.SENTIMENT,
            StageName.EMBEDDING,
            StageName.SCORING,
            StageName.STORE,
        ]
        assert all(o.status is StageStatus.OK for o in result.outcomes)

    async def test_the_signal_carries_design_doc_fields_one_to_eight(
        self, result: PipelineResult
    ) -> None:
        """Stage 2's contract: fields 1-8 populated from the payload.

        `source` is checked because it is *derived* from `platform` rather than
        mapped -- `Signal` rejects a connector that declares them inconsistently,
        and this is the assertion that the derivation ran at all.
        """
        signal = result.signal
        assert signal is not None

        assert signal.id.startswith("sig_")
        assert signal.source is SourceCategory.NEWS
        assert signal.platform is Platform.RSS
        assert signal.url == "https://news.example.com/2026/07/27/self-hosted-migration"
        assert signal.author is not None
        assert signal.author.platform_author_id == "news.example.com:dmitri@news.example.com"
        assert signal.author.display_name == "Dmitri Sokolov"
        # Event time at the source, from `<pubDate>`, not ingestion time.
        assert signal.timestamp == dt.datetime(2026, 7, 27, 7, 5, tzinfo=dt.UTC)
        assert signal.content.title == "We moved forty services off hosted observability"
        assert signal.content.text.startswith("We moved forty services onto self-hosted")
        assert signal.content.char_count == len(signal.content.text)
        # `content:encoded` is the full article by definition, never a teaser.
        assert signal.content.truncated is False
        # The markup is gone: an unstripped `<article>` would reach the embedding.
        assert "<" not in signal.content.text
        assert signal.media == []

    async def test_fields_fifteen_and_seventeen_come_from_the_payload(
        self, result: PipelineResult
    ) -> None:
        """Engagement counters and the namespaced metadata overflow.

        RSS publishes no counters, so `raw` is legitimately empty and the axes
        stay `None` -- an honest "unknown" rather than a 0.0 that retrieval would
        read as "nobody engaged". Metadata keys must all be platform-namespaced;
        an un-namespaced one collides across connectors in one jsonb column.
        """
        signal = result.signal
        assert signal is not None

        assert signal.engagement.raw == {}
        assert signal.engagement.available_axes() == {}
        assert signal.metadata["rss.feed_url"] == FEED_URL
        assert signal.metadata["rss.guid"] == FULL_BODY_GUID
        assert signal.metadata["rss.categories"] == ["case study"]
        assert all(key.startswith("rss.") for key in signal.metadata)

    async def test_identity_matches_the_record_that_produced_it(
        self, result: PipelineResult
    ) -> None:
        """The rebuilt Signal is the same item the raw event named.

        `RawRecordEvent.partition_key` is `signal_id(platform, native_id)`,
        computed one topic earlier so a re-fetch lands behind its earlier copy.
        If stage 2's rebuild derived anything else, the enriched event would be
        published under a different key from the raw one and the ordering
        guarantee would be silently gone.
        """
        signal = result.signal
        assert signal is not None
        assert signal.lineage.native_id == FULL_BODY_GUID
        assert signal.id == raw_record_event(rss_entry_payload()).partition_key

    async def test_lineage_records_the_run_that_produced_it(
        self, result: PipelineResult
    ) -> None:
        """Provenance: who fetched it, when, from where, and at what version.

        `pipeline_version` is the load-bearing one. A connector stamps `"0.0.0"`
        because it has run no enrichment; leaving that on a fully enriched Signal
        makes `store.py`'s upsert guard reject every subsequent write.
        """
        signal = result.signal
        assert signal is not None
        lineage = signal.lineage

        assert lineage.pipeline_version == "1.4.0"
        assert lineage.pipeline_version != UNENRICHED_PIPELINE_VERSION
        assert lineage.connector_slug == "rss"
        assert lineage.connector_version == "0.1.0"
        assert lineage.sync_run_id == "run_01J8XN5Q2P"
        assert lineage.fetched_at == dt.datetime(2026, 7, 27, 8, 0, tzinfo=dt.UTC)
        assert lineage.request_fingerprint == "rss:news.example.com/feed.xml"

    async def test_the_archived_original_is_addressable_from_the_signal(
        self, result: PipelineResult
    ) -> None:
        """`raw_ref` and `raw_object_key` point at the object in R2.

        The connector could not fill either in -- it does not perform the PUT --
        so a Signal that reached storage without them would be unreprocessable:
        "a cleaning bug is repairable by reprocessing rather than re-fetching"
        depends entirely on this pointer existing.
        """
        signal = result.signal
        assert signal is not None
        event = raw_record_event(rss_entry_payload())

        assert signal.lineage.raw_object_key == event.raw_object_key
        assert signal.content.raw_ref == event.raw_object_key
        assert signal.lineage.raw_sha256 == event.raw_sha256
        assert signal.lineage.raw_bytes == event.raw_bytes
        assert signal.lineage.raw_content_type == "application/json"

    async def test_every_stage_from_normalize_onward_is_recorded_in_lineage(
        self, result: PipelineResult
    ) -> None:
        """`lineage.stages[]` is the audit trail a claim is traced back through.

        Clean is absent and that is not a gap: stage 1 runs before a Signal
        exists to attach a record to, and a Clean failure is fatal, so the raw
        record goes to the DLQ carrying the outcome list directly.
        """
        signal = result.signal
        assert signal is not None
        recorded = signal.lineage.latest_stages()

        assert StageName.CLEAN not in recorded
        assert set(recorded) == {
            StageName.NORMALIZE,
            StageName.LANGUAGE,
            StageName.ENTITIES,
            StageName.SENTIMENT,
            StageName.EMBEDDING,
            StageName.SCORING,
            StageName.STORE,
        }
        assert all(record.status is StageStatus.OK for record in recorded.values())
        assert all(record.error is None for record in recorded.values())
        # Stage 2 is deterministic, so it records no model -- the deliberate
        # asymmetry with stages 4-6, which cannot be replayed without one.
        assert recorded[StageName.NORMALIZE].model is None
        assert recorded[StageName.ENTITIES].model is not None
        assert recorded[StageName.EMBEDDING].model == "fake-embed-v1"

    async def test_the_enrichment_fields_are_populated_by_their_own_stages(
        self, result: PipelineResult
    ) -> None:
        """Fields 9-16, each proving its stage saw the body stage 2 built.

        The entity offsets are the sharp assertion: they index `content.text`,
        and if stage 2 had handed stage 4 a different string from the one that
        was stored, every citation of this Signal would highlight the wrong
        words with nothing raising anywhere.
        """
        signal = result.signal
        assert signal is not None

        assert signal.language.code == "en"
        assert signal.language.is_determinate

        assert [mention.surface for mention in signal.entities] == ["Grafana", "Loki"]
        for mention in signal.entities:
            assert signal.content.text[mention.start : mention.end] == mention.surface

        assert [topic.topic for topic in signal.topics] == [
            "observability-tooling",
            "vendor-pricing",
        ]
        assert signal.keywords

        assert signal.sentiment is not None
        assert signal.sentiment.polarity == pytest.approx(0.45)

        assert len(signal.embeddings) == 1
        assert signal.embeddings[0].collection == COLLECTION
        assert signal.embeddings[0].dimensions == 8

        # Stage 6b: every degradable stage succeeded, so extraction quality is full.
        assert signal.lineage.confidence_components is not None
        assert signal.lineage.confidence_components.extraction_quality == pytest.approx(1.0)
        assert 0.0 < signal.confidence <= 1.0

    async def test_the_signal_is_committed_and_then_announced(
        self,
        result: PipelineResult,
        session_factory: async_sessionmaker[AsyncSession],
        publisher: FakePublisher,
    ) -> None:
        """Stage 7 is the commit point: one row, then one event carrying its id.

        The event carries identity and version, never content -- a consumer
        re-reads the committed row, which is the only store whose contents are
        definitionally true.
        """
        signal = result.signal
        assert signal is not None

        async with session_factory() as session:
            count = (
                await session.execute(select(func.count()).select_from(SignalRow))
            ).scalar_one()
            row = await session.get(SignalRow, signal.id)

        assert count == 1
        assert row is not None
        assert row.content_text == signal.content.text
        assert row.pipeline_version == "1.4.0"
        assert row.status is SignalStatus.ENRICHED
        assert row.native_id == FULL_BODY_GUID
        # Written NULL on every write so a crash before the Qdrant upsert
        # self-heals through the reconciler rather than being believed indexed.
        assert row.indexed_vector_at is None

        assert len(publisher.events) == 1
        published = publisher.events[0]
        assert published.signal_id == signal.id
        assert published.status is SignalStatus.ENRICHED
        assert published.pipeline_version == "1.4.0"
        assert published.failed_stages == []

    async def test_the_chunk_text_reaches_the_vector_sink(
        self, result: PipelineResult, sink: InMemoryVectorSink
    ) -> None:
        """Stage 6 stages the chunk text alongside the vector.

        The spans have to index `content.text` exactly, because
        `services/evidence_service.py` re-reads that string to verify a quote.
        """
        signal = result.signal
        assert signal is not None
        staged = sink.take(signal.id)

        assert len(staged) == len(signal.embeddings)
        for vector in staged:
            assert signal.content.text[vector.char_start : vector.char_end] == vector.text
            assert len(vector.vector) == 8

    async def test_the_model_is_called_once_per_model_stage(
        self, result: PipelineResult, llm: FakeLLMProvider, embedder: FakeEmbeddingProvider
    ) -> None:
        """Two LLM calls and one embedding batch for one Signal.

        Pinned because cost regressions are invisible in output: a stage that
        retried internally, or a second prompt added for "context", produces
        identical Signals and doubles the ingestion bill.
        """
        assert len(llm.calls) == 2
        assert len(embedder.batches) == 1
        assert result.status is SignalStatus.ENRICHED


# --------------------------------------------------------------------------- #
# Stage 2 in isolation
# --------------------------------------------------------------------------- #


async def normalize(ctx: EnrichmentContext) -> EnrichmentContext:
    """Run stages 1 and 2 over a context, as the front of the pipeline would."""
    await CleaningStage().apply(ctx)
    await NormalizeStage().apply(ctx)
    return ctx


class TestNormalizeStage:
    """The stage's own contract, at the edges the full pass never reaches."""

    def test_it_declares_itself_as_a_deterministic_fatal_stage(self) -> None:
        """No model id: stage 2 replays identically, which is what makes
        reprocessing cheap. Naming a model would claim otherwise."""
        stage = NormalizeStage()
        assert stage.name is StageName.NORMALIZE
        assert stage.version
        assert stage.model_id is None

    async def test_a_missing_record_event_fails_loudly(self) -> None:
        """The payload alone cannot say which connector produced it.

        Without the event there is no slug, no sync run and no fetch time, and
        the failure has to name that rather than surface as a `KeyError` inside
        a field map.
        """
        ctx = context_for(rss_entry_payload(), record=None)
        with pytest.raises(NormalizationError, match="RawRecordEvent"):
            await normalize(ctx)

    async def test_an_unmapped_connector_names_the_slug(self) -> None:
        """A half-built Signal is worse than a DLQ record.

        An unmapped source that fell back to a generic map would produce a
        Signal with a plausible id and an empty body, which retrieval still
        returns and a report still quotes.
        """
        payload = rss_entry_payload()
        ctx = context_for(payload, record=raw_record_event(payload, connector_slug="mastodon"))
        with pytest.raises(NormalizationError, match="mastodon"):
            await normalize(ctx)

    async def test_a_rebuild_that_renames_the_record_is_refused(self) -> None:
        """Identity may not move between the connector and the pipeline.

        The event's `native_id` already keyed the R2 object and the Kafka
        partition. A rebuild deriving a different one would give the same item
        two identities across five stores, discovered later as duplicate rows
        nobody can reconcile.
        """
        payload = rss_entry_payload()
        ctx = context_for(
            payload,
            record=raw_record_event(payload, native_id="tag:news.example.com,2026:post-9999"),
        )
        with pytest.raises(NormalizationError, match="two identities"):
            await normalize(ctx)

    async def test_a_payload_the_worker_never_read_is_refused(self) -> None:
        """Distinguished from an unmapped slug: this is a worker that did not
        read R2, not code that was never written."""
        payload = rss_entry_payload()
        ctx = EnrichmentContext(
            content_type="application/json",
            payload={},
            record=raw_record_event(payload),
        )
        with pytest.raises(NormalizationError, match="never read onto the context"):
            await NormalizeStage().apply(ctx)

    async def test_failure_is_fatal_and_quarantines_the_record(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """Stage 2 is in `FATAL_STAGES`, and the pipeline -- not the stage --
        is what decides that. Nothing is stored and nothing is published."""
        publisher = FakePublisher()
        pipeline = SignalPipeline(
            [CleaningStage(), NormalizeStage(), StoreStage(session_factory, publisher)]
        )
        payload = rss_entry_payload()
        result = await pipeline.run(
            context_for(payload, record=raw_record_event(payload, connector_slug="mastodon"))
        )

        assert result.status is SignalStatus.QUARANTINED
        assert result.fatal_stage is StageName.NORMALIZE
        assert result.error == "NormalizationError"
        assert result.signal is None
        assert not result.succeeded
        assert publisher.events == []

    async def test_the_pipeline_version_of_the_run_replaces_the_zero_version(
        self,
    ) -> None:
        """`"0.0.0"` is what a connector stamps when it has run no enrichment.

        Stage 2 is where it becomes real. Left alone it would make every write
        lose `store.py`'s `pipeline_version` guard, and `docs/signal-model.md` §7
        makes the field the basis for deciding what needs reprocessing.
        """
        ctx = context_for(rss_entry_payload(), pipeline_version="2.10.3")
        await normalize(ctx)

        signal = ctx.require_signal()
        assert signal.lineage.pipeline_version == "2.10.3"

    async def test_a_structured_record_takes_its_body_from_the_field_map(self) -> None:
        """Stage 1 writes `""` for a JSON payload -- "cleaned, and there is no
        document body". Adopting that as the body would empty `content.text` for
        every JSON source in the system."""
        ctx = context_for(rss_entry_payload())
        await normalize(ctx)

        assert ctx.cleaned_text == ""
        assert ctx.require_signal().content.text.startswith("We moved forty services")

    async def test_a_cleaned_document_body_wins_over_the_mapped_one(self) -> None:
        """When stage 1 produced a body, stage 2 adopts it -- and the reason is
        redaction, not prose quality.

        The `Redactor` runs in stage 1 and nowhere else. A stage 2 that preferred
        the payload-mapped body would route un-redacted text into `content.text`
        and from there into the embedding, where PII is neither readable nor
        auditable nor deletable. `char_count` is re-derived with it, because a
        `Content` whose count belonged to the other body misleads every consumer
        that reads it.
        """
        payload = rss_entry_payload()
        article = (
            "<html><body><article><p>Reach the desk at desk@news.example.com "
            "for corrections to this migration report.</p></article></body></html>"
        )
        ctx = EnrichmentContext(
            raw_bytes=article.encode("utf-8"),
            content_type="text/html; charset=utf-8",
            payload=payload,
            record=raw_record_event(payload, raw_content_type="text/html"),
        )
        await CleaningStage(redactor=RegexRedactor()).apply(ctx)
        await NormalizeStage().apply(ctx)

        content = ctx.require_signal().content
        assert content.text == ctx.cleaned_text
        assert EMAIL_PLACEHOLDER in content.text
        assert "desk@news.example.com" not in content.text
        assert content.char_count == len(content.text)


class TestDefaultFieldMapResolver:
    """One resolver, four shipped connectors, and the payload-dependent choices."""

    def test_it_covers_every_shipped_connector(self) -> None:
        for slug, platform in (
            ("rss", Platform.RSS),
            ("news_api", Platform.NEWS_API),
            ("gdelt", Platform.GDELT),
            ("reddit", Platform.REDDIT),
        ):
            field_map = default_field_map_resolver(slug, {"kind": "t3"})
            assert field_map is not None, slug
            assert field_map.platform is platform

    def test_an_unknown_slug_resolves_to_none(self) -> None:
        """`None`, not an exception: the stage attaches the `native_id` to the
        error, and a DLQ record nobody can attribute is one nobody can replay."""
        assert default_field_map_resolver("mastodon", {}) is None

    def test_reddit_selects_its_map_from_the_payload_kind(self) -> None:
        """One slug, two maps. A comment has no title and its body is at
        `data.body`; a post's is at `data.selftext`. Mapping one as the other
        silently empties every comment in the corpus."""
        post = default_field_map_resolver("reddit", {"kind": "t3"})
        comment = default_field_map_resolver("reddit", {"kind": "t1"})

        assert post is not None and post.text is not None
        assert comment is not None and comment.text is not None
        assert post.text.paths == ("data.selftext",)
        assert comment.text.paths == ("data.body",)
        assert post.title is not None
        assert comment.title is None

    def test_an_unmappable_reddit_kind_resolves_to_none(self) -> None:
        """`kind` is a closed set; a new one is a payload shape nobody mapped."""
        assert default_field_map_resolver("reddit", {"kind": "t6"}) is None
        assert default_field_map_resolver("reddit", {}) is None

    def test_rss_selects_by_where_the_body_came_from(self) -> None:
        """Atom `<content>` is a whole article; a `<summary>` may be a teaser,
        and `truncated` caps `content_integrity` for the life of the Signal."""
        full = default_field_map_resolver("rss", rss_entry_payload(FULL_BODY_GUID))
        teaser = default_field_map_resolver(
            "rss", rss_entry_payload("tag:news.example.com,2026:post-4181")
        )

        assert full is not None and full.truncated is False
        assert teaser is not None and teaser.truncated is True

    async def test_it_maps_a_reddit_post_from_the_real_listing_fixture(self) -> None:
        """The resolver is not RSS-shaped: a second connector, a second payload
        layout, and the same stage builds a valid Signal from it."""
        child = REDDIT_LISTING["data"]["children"][0]
        event = RawRecordEvent(
            platform=Platform.REDDIT,
            native_id=child["data"]["name"],
            connector_slug="reddit",
            connector_version="0.1.0",
            sync_run_id="run_2",
        )
        ctx = EnrichmentContext(payload=child, record=event, content_type="application/json")
        await normalize(ctx)

        signal = ctx.require_signal()
        assert signal.platform is Platform.REDDIT
        assert signal.lineage.native_id == "t3_p1a"
        assert signal.content.text.startswith("Six weeks in.")
        assert signal.author is not None
        assert signal.author.platform_author_id == "t2_9xk3q"
        assert signal.engagement.raw["score"] == 412
        assert signal.metadata["reddit.subreddit"] == "selfhosted"


# --------------------------------------------------------------------------- #
# The construction-time guard
# --------------------------------------------------------------------------- #


class TestStageOrderGuard:
    """A pipeline that can never build a Signal must not be constructible."""

    def test_an_ingest_pipeline_without_normalize_is_rejected(self) -> None:
        """Every stage after Clean reads `ctx.signal`. Without stage 2 the run
        is five `RuntimeError`s that describe a consequence, none of which names
        the cause, and the record is only lost at Store."""
        with pytest.raises(ValueError, match="no 'normalize' stage"):
            SignalPipeline([CleaningStage(), LanguageStage(LangdetectDetector())])

    def test_the_message_names_the_stages_that_would_have_failed(self) -> None:
        with pytest.raises(ValueError) as caught:
            SignalPipeline(
                [
                    CleaningStage(),
                    LanguageStage(LangdetectDetector()),
                    ScoringStage(baseline=InMemoryCohortBaseline()),
                ]
            )
        message = str(caught.value)
        assert "language" in message
        assert "scoring" in message
        assert "require_signal" in message

    def test_clean_alone_is_still_a_valid_pipeline(self) -> None:
        """Stage 1 builds no Signal and needs none: a decode-and-clean pass over
        an archive is a legitimate thing to run."""
        SignalPipeline([CleaningStage()])

    def test_the_full_assembly_is_accepted(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        build_pipeline(
            session_factory,
            FakePublisher(),
            llm=scripted_llm(),
            embedder=FakeEmbeddingProvider(dimensions=8),
            sink=InMemoryVectorSink(),
        )

    def test_a_re_drive_fragment_starting_after_normalize_is_accepted(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        """The exception the guard deliberately makes.

        `workers/embedding_worker.py` re-runs stage 6 over a Signal read back
        from PostgreSQL, and the enrichment sweeper re-runs the degradable stages
        over a `partial` row. Both are handed a context that already carries a
        Signal, and rejecting them at construction would forbid the retry path
        `docs/signal-model.md` §5.2 requires. A fragment that genuinely is
        mis-wired is caught by `require_signal()` on its first record, which is
        the only place that knows whether a Signal was meant to be supplied.
        """
        SignalPipeline(
            [
                EmbeddingStage(FakeEmbeddingProvider(dimensions=8), collection=COLLECTION),
                ScoringStage(baseline=InMemoryCohortBaseline()),
                StoreStage(session_factory, FakePublisher()),
            ]
        )
