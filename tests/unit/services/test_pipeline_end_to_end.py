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

The payload is the first issue of `tests/fixtures/payloads/github_issues_page1.json`,
shaped exactly as `GitHubConnector.fetch()` emits it -- the provider object
verbatim under the same `omnisense` envelope -- so the field paths under test are
the real ones rather than a hand-written approximation of them. GitHub is the
right choice for this suite rather than merely an available one: it is the only
shipped connector that maps several payload shapes through a *table* of
`FieldMap`s, so it is the only one whose rebuild in stage 2 can pick the wrong map
and still produce a plausible Signal. The LLM and the embedding provider are the
offline fakes from `services/llm/`; the database is in-memory SQLite from
`tests/conftest.py`; the publisher is four lines. Nothing here opens a socket and
nothing here needs a service running.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from connectors.enterprise.github import ENVELOPE_KEY
from connectors.exceptions import NormalizationError
from connectors.normalize.mapper import UNENRICHED_PIPELINE_VERSION
from models.enums import Platform, SignalStatus, SourceCategory, StageName, StageStatus
from models.orm.signal import SignalRow
from services.events.schemas import RawRecordEvent, SignalEnrichedEvent
from services.llm.embeddings import FakeEmbeddingProvider
from services.llm.provider import FakeLLMProvider
from services.signal_engine.cleaning import EMAIL_PLACEHOLDER, CleaningStage, RegexRedactor
from services.signal_engine.embeddings import EmbeddingStage, InMemoryVectorSink
from services.signal_engine.enrichment import InMemoryCohortBaseline, ScoringStage
from services.signal_engine.language import LangdetectDetector, LanguageStage
from services.signal_engine.normalize import (
    NormalizeStage,
    default_field_map_resolver,
)
from services.signal_engine.pipeline import (
    EnrichmentContext,
    PipelineResult,
    SignalPipeline,
)
from services.signal_engine.store import StoreStage
from workers.enrichment_worker import build_pipeline as production_build_pipeline

pytestmark = pytest.mark.unit


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "payloads"
GITHUB_ISSUES = json.loads((FIXTURES / "github_issues_page1.json").read_text())

REPOSITORY = "omnisense/omnisense"
ISSUE_NODE_ID = "I_kwDOABCD1M6TqXyN"
"""The plain issue in the fixture -- the one that is not also a pull request.

Chosen over the PR entry because it is the longer body, which gives the language
detector and the chunker something to say, and because it leaves the PR available
as the second payload shape the same map has to handle.
"""

PR_NODE_ID = "PR_kwDOABCD1M6QmZ4A"
"""The pull request. GitHub's issues endpoint returns both, and a PR *is* an issue
in the data model -- distinguishable only by the `pull_request` key."""

COLLECTION = "omnisense_signals_test"
"""Named explicitly rather than read from settings: the collection travels into
every `EmbeddingRef.collection` this test asserts on, and a config default that
changed under it would fail the assertion for a reason that has nothing to do
with the pipeline."""


# --------------------------------------------------------------------------- #
# Building the record the worker would receive
# --------------------------------------------------------------------------- #


def github_issue_payload(node_id: str = ISSUE_NODE_ID) -> dict[str, Any]:
    """One issue from the fixture, shaped as `GitHubConnector.fetch()` emits it.

    The provider object verbatim, plus the `omnisense` envelope the connector
    adds. Two of the envelope's keys are load-bearing rather than decorative:
    `stream` is what the resolver reads to choose between three field maps, and
    the issue map addresses `is_pull_request` at `omnisense.is_pull_request`
    because GitHub's issues endpoint returns pull requests too and nothing in the
    object itself says so except the presence of a `pull_request` key -- a rule
    the connector applies once, in `fetch()`, so no consumer has to re-derive it.
    """
    for issue in GITHUB_ISSUES:
        if issue.get("node_id") != node_id:
            continue
        payload: dict[str, Any] = json.loads(json.dumps(issue))
        payload[ENVELOPE_KEY] = {
            "repository": REPOSITORY,
            "stream": "issues",
            "is_pull_request": "pull_request" in issue,
        }
        return payload
    raise AssertionError(f"no issue {node_id!r} in the GitHub fixture")


def envelope(stream: str | None) -> dict[str, Any]:
    """The smallest payload the resolver can classify: an envelope and a stream.

    Deliberately not a whole GitHub object. The resolver reads exactly one key,
    and handing it a full payload would let a test pass because the object
    happened to look like an issue rather than because the stream said so.
    """
    return {ENVELOPE_KEY: {"repository": REPOSITORY, "stream": stream}}


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
        "platform": Platform.GITHUB,
        "native_id": payload["node_id"],
        "connector_slug": "github",
        "connector_version": "0.1.0",
        "sync_run_id": "run_01J8XN5Q2P",
        "fetched_at": dt.datetime(2026, 7, 27, 8, 0, tzinfo=dt.UTC),
        "raw_object_key": f"raw/github/2026/07/27/{hashlib.sha256(raw).hexdigest()}.json",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "raw_content_type": "application/json",
        "source_url": f"https://api.github.com/repos/{REPOSITORY}/issues",
        "request_fingerprint": "github:GET /repos/{owner}/{repo}/issues?state=all",
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
        return await pipeline.run(context_for(github_issue_payload()))

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
        assert signal.source is SourceCategory.ENTERPRISE
        assert signal.platform is Platform.GITHUB
        # `html_url`, never `url`: the latter is the API endpoint, and a citation
        # that resolves to api.github.com answers a browser with JSON.
        assert signal.url == "https://github.com/omnisense/omnisense/issues/4166"
        # `user.node_id`, never `user.login`. Logins are renameable and the rename
        # rewrites every URL containing one, forking an author's history silently.
        assert signal.author is not None
        assert signal.author.platform_author_id == "MDQ6VXNlcjE0ODEyMzM="
        assert signal.author.handle == "dsokolov"
        # Event time at the source, from `created_at`, not `updated_at` -- an issue
        # stamped with the moment somebody added a label files a three-year-old bug
        # report as today's news.
        assert signal.timestamp == dt.datetime(2026, 7, 27, 7, 5, tzinfo=dt.UTC)
        assert signal.content.title == (
            "Scheduler drops queued jobs when the leader restarts mid-lease"
        )
        assert signal.content.text.startswith("We moved forty services onto the self-hosted")
        assert signal.content.char_count == len(signal.content.text)
        # The whole body, as GitHub stores it -- an issue is never a teaser.
        assert signal.content.truncated is False
        # Markdown reaches the embedding intact rather than through
        # `extract_readable`, which would strip nothing and could mangle the fenced
        # code blocks that are most of what a bug report is made of.
        assert "`reaper: 0 orphans`" in signal.content.text
        assert signal.media == []

    async def test_fields_fifteen_and_seventeen_come_from_the_payload(
        self, result: PipelineResult
    ) -> None:
        """Engagement counters and the namespaced metadata overflow.

        Every counter is GitHub's raw number. The *normalized* axes stay `None`
        because they are percentiles within a `(platform, content_type)` cohort and
        stage 2 holds one record -- an honest "not computed here" rather than a 0.0
        that retrieval would read as "nobody engaged". Metadata keys must all be
        platform-namespaced; an un-namespaced one collides across connectors in one
        jsonb column.
        """
        signal = result.signal
        assert signal is not None

        assert signal.engagement.raw == {
            "comments": 7,
            "reactions": 12,
            "reactions_plus_one": 9,
        }
        assert signal.engagement.available_axes() == {}
        assert signal.metadata["github.repository"] == REPOSITORY
        assert signal.metadata["github.stream"] == "issues"
        assert signal.metadata["github.number"] == 4166
        assert signal.metadata["github.state"] == "open"
        # `_label_names` flattens GitHub's label objects to their names; the raw
        # objects carry ids and colours no consumer of this column wants.
        assert signal.metadata["github.labels"] == ["bug", "area/scheduler"]
        # The plain issue, so `fetch()` recorded False rather than omitting the key.
        assert signal.metadata["github.is_pull_request"] is False
        assert all(key.startswith("github.") for key in signal.metadata)

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
        assert signal.lineage.native_id == ISSUE_NODE_ID
        assert signal.id == raw_record_event(github_issue_payload()).partition_key

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
        assert lineage.connector_slug == "github"
        assert lineage.connector_version == "0.1.0"
        assert lineage.sync_run_id == "run_01J8XN5Q2P"
        assert lineage.fetched_at == dt.datetime(2026, 7, 27, 8, 0, tzinfo=dt.UTC)
        assert lineage.request_fingerprint == (
            "github:GET /repos/{owner}/{repo}/issues?state=all"
        )

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
        event = raw_record_event(github_issue_payload())

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
        assert row.native_id == ISSUE_NODE_ID
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
        ctx = context_for(github_issue_payload(), record=None)
        with pytest.raises(NormalizationError, match="RawRecordEvent"):
            await normalize(ctx)

    async def test_an_unmapped_connector_names_the_slug(self) -> None:
        """A half-built Signal is worse than a DLQ record.

        An unmapped source that fell back to a generic map would produce a
        Signal with a plausible id and an empty body, which retrieval still
        returns and a report still quotes.
        """
        payload = github_issue_payload()
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
        payload = github_issue_payload()
        ctx = context_for(
            payload,
            record=raw_record_event(payload, native_id="I_kwDOABCD1M6ZZZZZZ"),
        )
        with pytest.raises(NormalizationError, match="two identities"):
            await normalize(ctx)

    async def test_a_payload_the_worker_never_read_is_refused(self) -> None:
        """Distinguished from an unmapped slug: this is a worker that did not
        read R2, not code that was never written."""
        payload = github_issue_payload()
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
        payload = github_issue_payload()
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
        ctx = context_for(github_issue_payload(), pipeline_version="2.10.3")
        await normalize(ctx)

        signal = ctx.require_signal()
        assert signal.lineage.pipeline_version == "2.10.3"

    async def test_a_structured_record_takes_its_body_from_the_field_map(self) -> None:
        """Stage 1 writes `""` for a JSON payload -- "cleaned, and there is no
        document body". Adopting that as the body would empty `content.text` for
        every JSON source in the system."""
        ctx = context_for(github_issue_payload())
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
        payload = github_issue_payload()
        article = (
            "<html><body><article><p>Reach the maintainers at desk@news.example.com "
            "for corrections to this report.</p></article></body></html>"
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
    """One resolver, one slug, three maps, and the choice between them.

    The table is short because a `FieldMap` earns its place only where a
    connector emits several payload shapes -- see `_shipped_selectors`. GitHub is
    the only one that does, and every other shipped slug resolving to `None` is
    the documented answer rather than a hole.
    """

    def test_it_covers_every_github_stream(self) -> None:
        """All three, and each declaring GitHub. A map that resolved but named
        another platform would build a Signal whose `source` is derived from the
        wrong category and file a repository issue under `NEWS`."""
        for stream in ("issues", "discussions", "releases"):
            field_map = default_field_map_resolver("github", envelope(stream))
            assert field_map is not None, stream
            assert field_map.platform is Platform.GITHUB

    def test_an_unknown_slug_resolves_to_none(self) -> None:
        """`None`, not an exception: the stage attaches the `native_id` to the
        error, and a DLQ record nobody can attribute is one nobody can replay."""
        assert default_field_map_resolver("mastodon", {}) is None

    def test_a_connector_that_does_not_map_by_table_resolves_to_none(self) -> None:
        """Jira, Slack and the rest build their `Signal` directly in `normalize()`.

        `None` here means "not normalized by field map", which is a different
        thing from "unknown connector" and produces the same outcome by design:
        the pipeline's rebuild path is only for connectors that have a map to
        rebuild from.
        """
        for slug in ("jira", "slack", "confluence", "notion", "arxiv"):
            assert default_field_map_resolver(slug, {}) is None, slug

    def test_the_stream_selects_the_map_and_the_streams_genuinely_differ(self) -> None:
        """One slug, three maps. An issue's body is at `body` and its timestamp at
        `created_at`; a discussion is GraphQL and uses `createdAt`; a release is
        dated by `published_at` because `created_at` is when the *tag* was cut,
        which for a release published from an old tag can be years earlier.

        Mapping one as another does not fail -- it produces a Signal with a
        plausible id, an empty body and a wrong date, which retrieval still
        returns and a report still quotes.
        """
        issues = default_field_map_resolver("github", envelope("issues"))
        discussions = default_field_map_resolver("github", envelope("discussions"))
        releases = default_field_map_resolver("github", envelope("releases"))

        assert issues is not None and issues.timestamp.paths == ("created_at",)
        assert discussions is not None and discussions.timestamp.paths == ("createdAt",)
        assert releases is not None and releases.timestamp.paths == ("published_at",)
        # A release has no author-facing title of its own; `name` falls back to
        # `tag_name`, which is the one map here with a two-path title.
        assert releases.title is not None
        assert releases.title.paths == ("name", "tag_name")

    def test_an_unmapped_stream_resolves_to_none(self) -> None:
        """`STREAMS` is a closed set. A stream this table has never seen is a
        payload shape nobody mapped, and the connector's own `normalize()` raises
        on it for the same reason."""
        assert default_field_map_resolver("github", envelope("commits")) is None
        assert default_field_map_resolver("github", envelope(None)) is None

    def test_a_payload_without_the_envelope_resolves_to_none(self) -> None:
        """The stream lives in the envelope, never in the GitHub object.

        A payload that arrived without one did not come through this connector's
        `fetch()`, and guessing the stream from the object's shape would
        misclassify an issue as a discussion -- they are structurally similar
        enough for the guess to look right.
        """
        assert default_field_map_resolver("github", {}) is None
        assert default_field_map_resolver("github", {ENVELOPE_KEY: "issues"}) is None

    async def test_it_maps_a_discussion_from_the_graphql_shape(self) -> None:
        """The resolver is not issue-shaped: a second stream, a camelCase payload
        layout from an entirely different API, and the same stage builds a valid
        Signal from it.

        This is the assertion that the stream key is doing real work. Were the
        resolver returning the issue map for everything, this payload would map to
        an empty body and a missing timestamp rather than to the discussion below.
        """
        payload = {
            "id": "D_kwDOABCD1M4APqZ3",
            "number": 812,
            "title": "RFC: drive the reaper off the lease epoch",
            "body": "Splitting this out of #4166 so the design has somewhere to live.",
            "url": "https://github.com/omnisense/omnisense/discussions/812",
            "createdAt": "2026-07-29T09:15:00Z",
            "updatedAt": "2026-07-29T13:40:00Z",
            "author": {
                "id": "MDQ6VXNlcjE0ODEyMzM=",
                "login": "dsokolov",
                "url": "https://github.com/dsokolov",
            },
            "category": {"name": "Design"},
            "isAnswered": False,
            "upvoteCount": 18,
            "comments": {"totalCount": 5},
            "reactions": {"totalCount": 9},
            ENVELOPE_KEY: {"repository": REPOSITORY, "stream": "discussions"},
        }
        event = RawRecordEvent(
            platform=Platform.GITHUB,
            native_id="D_kwDOABCD1M4APqZ3",
            connector_slug="github",
            connector_version="0.1.0",
            sync_run_id="run_2",
        )
        ctx = EnrichmentContext(payload=payload, record=event, content_type="application/json")
        await normalize(ctx)

        signal = ctx.require_signal()
        assert signal.platform is Platform.GITHUB
        assert signal.lineage.native_id == "D_kwDOABCD1M4APqZ3"
        assert signal.content.text.startswith("Splitting this out of")
        # `author.id` from GraphQL *is* the REST `node_id` -- the same author
        # identity as the issue above, reached by a different path.
        assert signal.author is not None
        assert signal.author.platform_author_id == "MDQ6VXNlcjE0ODEyMzM="
        assert signal.timestamp == dt.datetime(2026, 7, 29, 9, 15, tzinfo=dt.UTC)
        assert signal.engagement.raw["upvotes"] == 18
        assert signal.metadata["github.category"] == "Design"


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
