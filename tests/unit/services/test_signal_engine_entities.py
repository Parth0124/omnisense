"""Unit tests for enrichment stage 4 -- entities, topics and keywords.

This stage produces the only field in the system that is *silently* wrong when it
is wrong. A bad sentiment score looks bad. A bad embedding retrieves badly. A
character offset that is off by four highlights the wrong words in every citation
of that Signal, forever, and nothing raises, nothing alerts, and no downstream
consumer is in a position to notice -- `EntityMention` carries no evidence that
would let it. So the bulk of this file is one assertion made many ways:

    text[mention.start:mention.end] == mention.surface

for every mention this stage is capable of emitting, under every way a model is
known to get offsets wrong.

The three failure modes the suite is built around, in the order they cost:

- **A wrong offset that verifies as plausible.** Models count tokens, or UTF-16
  code units, and hand back an index that is in range and slices out *something*.
  `TestOffsetRepair` and `TestUnicodeOffsets` exist to prove that such a mention
  is repaired or dropped, and never emitted as reported.
- **A malformed response taking the Signal with it.** `docs/signal-model.md` §5.2
  makes stage 4 degradable: the fields go empty, the Signal is stored `partial`,
  and it stays retrievable and citable. `TestDegradation` proves the degradation
  happens where the design puts it -- in the pipeline, from `FATAL_STAGES` -- and
  that the stage itself does not swallow the failure and claim success, which
  would credit `extraction_quality` for work that did not happen.
- **A second LLM call.** Stage 4 runs once per ingested Signal, so an extra call
  here is not a rounding error on the model bill, it is a doubling of it.
  `TestCost` pins the call count.

Everything runs offline against `FakeLLMProvider`. No network, no key, no
datastore, no clock dependency.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from backend.core.config import LLMSettings
from models.entity import EntityMention
from models.enums import (
    EntityType,
    Platform,
    SignalStatus,
    StageName,
    StageStatus,
)
from models.lineage import Lineage
from models.signal import Content, Language, Signal
from services.llm.provider import FakeLLMProvider, LLMRateLimited, LLMSchemaError
from services.signal_engine.entities import (
    MAX_EXTRACTION_CHARS,
    EntityExtractionStage,
    ExtractedMention,
    candidate_ids_for,
    coerce_entity_type,
    locate_span,
    resolve_mentions,
)
from services.signal_engine.keywords import (
    STOPWORDS,
    TOPIC_VOCABULARY,
    extract_keywords,
    normalize_topic,
    select_topics,
)
from services.signal_engine.pipeline import EnrichmentContext, SignalPipeline, Stage

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

BODY = "We left Datadog after the renewal quote tripled. Grafana and Loki replaced it."
"""One body used by most tests. `Datadog` is at 8-15, `Grafana` at 48-55."""


def make_signal(text: str, *, title: str | None = None, language: str = "en") -> Signal:
    """A minimally valid Signal carrying `text`.

    Built through `Signal.create` rather than by hand because `Signal` enforces
    that `id` is derived from `(platform, native_id)`; a hand-assembled Signal
    would fail validation before a single offset was checked.
    """
    moment = datetime(2026, 7, 28, 14, 30, tzinfo=UTC)
    return Signal.create(
        platform=Platform.REDDIT,
        native_id="t3_offsets",
        timestamp=moment,
        content=Content(title=title, text=text),
        lineage=Lineage(
            pipeline_version="1.0.0",
            connector_slug="reddit",
            connector_version="0.1.0",
            sync_run_id="run_offsets",
            fetched_at=moment,
            native_id="t3_offsets",
        ),
        language=Language(code=language, confidence=0.99, detector="test"),
    )


def response(
    mentions: list[dict[str, Any]] | None = None,
    topics: list[dict[str, Any]] | None = None,
) -> str:
    """Serialize a scripted model answer.

    Scripted as a JSON *string* rather than as an already-built model instance so
    that `FakeLLMProvider` runs the real schema validation. A test that handed in
    a constructed `EntityExtraction` would bypass exactly the boundary where a
    malformed response is supposed to be caught.
    """
    return json.dumps({"mentions": mentions or [], "topics": topics or []})


def make_stage(script: list[Any], **kwargs: Any) -> tuple[EntityExtractionStage, FakeLLMProvider]:
    """A stage wired to a scripted fake, with settings pinned away from the env."""
    provider = FakeLLMProvider(script=script)
    stage = EntityExtractionStage(
        provider,
        settings=LLMSettings(),
        model=kwargs.pop("model", "fast-tier-model"),
        **kwargs,
    )
    return stage, provider


async def run_stage(stage: EntityExtractionStage, signal: Signal) -> Signal:
    """Drive one stage directly, outside the pipeline."""
    ctx = EnrichmentContext(signal=signal)
    await stage.apply(ctx)
    return signal


def assert_spans_are_honest(signal: Signal) -> None:
    """The invariant the whole stage exists to maintain.

    Asserted after every extraction in this file rather than only where offsets
    are the subject, because a repair path that fixes one mention while corrupting
    its neighbour would otherwise pass every targeted test.
    """
    text = signal.content.text
    for mention in signal.entities:
        assert 0 <= mention.start < mention.end <= len(text)
        assert text[mention.start : mention.end] == mention.surface


# --------------------------------------------------------------------------- #
# Offsets
# --------------------------------------------------------------------------- #


class TestOffsetRepair:
    """A model-reported offset is a hint. These tests hold it to that."""

    async def test_correct_offsets_are_used_as_reported(self) -> None:
        """The common case must be cheap and lossless: verified offsets pass through."""
        stage, _ = make_stage(
            [response([{"surface": "Datadog", "type": "Company", "start": 8, "end": 15}])]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert [(m.start, m.end) for m in signal.entities] == [(8, 15)]
        assert_spans_are_honest(signal)

    @pytest.mark.parametrize(
        ("start", "end"),
        [
            (4, 11),  # early by four -- the classic token-arithmetic slip
            (12, 19),  # late by four
            (0, 7),  # anchored at the start of the document
            (None, None),  # no hint at all
            (8, 40),  # right start, absurd end
            (900, 907),  # past the end of the text entirely
            (15, 8),  # inverted
            (-3, 4),  # negative
        ],
    )
    async def test_wrong_offsets_are_repaired_to_the_true_span(
        self, start: int | None, end: int | None
    ) -> None:
        """Every shape of wrong offset lands on the real span, or nowhere.

        Parametrized rather than written once because each of these takes a
        different branch -- verify, in-range mismatch, out-of-range, inverted --
        and a repair that handles four of them is a repair that emits a wrong
        offset for the fifth.
        """
        stage, _ = make_stage(
            [response([{"surface": "Datadog", "type": "Company", "start": start, "end": end}])]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert [(m.start, m.end) for m in signal.entities] == [(8, 15)]
        assert_spans_are_honest(signal)

    async def test_surface_absent_from_the_text_is_dropped(self) -> None:
        """A hallucinated mention is dropped rather than given a plausible span.

        This is the decision the whole module turns on. The alternative -- keep
        the mention and trust the offsets -- produces a citation that highlights
        text having nothing to do with the entity named, and there is no
        downstream check able to catch it.
        """
        stage, _ = make_stage(
            [
                response(
                    [
                        {"surface": "Splunk", "type": "Company", "start": 8, "end": 14},
                        {"surface": "Datadog", "type": "Company", "start": 8, "end": 15},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert [m.surface for m in signal.entities] == ["Datadog"]
        assert_spans_are_honest(signal)

    async def test_case_normalized_surface_relocates_and_reports_the_real_text(self) -> None:
        """A model that lower-cased the surface still yields the document's casing.

        `EntityMention.surface` is documented as "the literal text as it
        appeared". Emitting the model's `datadog` next to a span covering
        `Datadog` would make the two disagree, which is the same defect as a
        wrong offset in a different field.
        """
        stage, _ = make_stage(
            [response([{"surface": "datadog", "type": "Company", "start": 8, "end": 15}])]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert [m.surface for m in signal.entities] == ["Datadog"]
        assert_spans_are_honest(signal)

    async def test_surface_split_by_a_line_break_relocates(self) -> None:
        """Cleaned text keeps newlines; a model reading it collapses them.

        Without the whitespace-flexible pass this mention would be dropped, and
        multi-word company names in wrapped article bodies are common enough that
        the recall loss would be systematic rather than incidental.
        """
        text = "Reports say Elastic\nSearch pricing changed again."
        stage, _ = make_stage(
            [response([{"surface": "Elastic Search", "type": "Product", "start": 12, "end": 26}])]
        )
        signal = await run_stage(stage, make_signal(text))

        assert [m.surface for m in signal.entities] == ["Elastic\nSearch"]
        assert_spans_are_honest(signal)

    async def test_two_mentions_of_one_name_do_not_collapse_onto_one_span(self) -> None:
        """Repeated names must repair to distinct occurrences.

        A naive `text.find()` repair sends both mentions to the first occurrence.
        The Signal then claims two mentions, the graph writes two `MENTIONS`
        edges, and both citations highlight the same seven characters -- which
        reads as correct in the UI and is not.
        """
        text = "Datadog raised prices, so we left Datadog in June."
        stage, _ = make_stage(
            [
                response(
                    [
                        {"surface": "Datadog", "type": "Company", "start": 2, "end": 9},
                        {"surface": "Datadog", "type": "Company", "start": 36, "end": 43},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(text))

        assert [(m.start, m.end) for m in signal.entities] == [(0, 7), (34, 41)]
        assert_spans_are_honest(signal)

    async def test_a_name_reported_more_often_than_it_occurs_drops_the_extra(self) -> None:
        """Three mentions of a name that appears twice means one is invented.

        Emitting it anywhere would duplicate an existing span, so the surplus is
        dropped rather than stacked onto an occurrence that already has an owner.
        """
        text = "Datadog and Datadog again."
        stage, _ = make_stage(
            [
                response(
                    [
                        {"surface": "Datadog", "type": "Company", "start": 0, "end": 7},
                        {"surface": "Datadog", "type": "Company", "start": 12, "end": 19},
                        {"surface": "Datadog", "type": "Company", "start": 40, "end": 47},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(text))

        assert [(m.start, m.end) for m in signal.entities] == [(0, 7), (12, 19)]

    async def test_identical_mention_reported_twice_is_emitted_once(self) -> None:
        """Duplicate rows would double-count the entity in every aggregate."""
        stage, _ = make_stage(
            [
                response(
                    [
                        {"surface": "Datadog", "type": "Company", "start": 8, "end": 15},
                        {"surface": "Datadog", "type": "Company", "start": 8, "end": 15},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert len(signal.entities) == 1

    async def test_nested_mentions_of_different_names_both_survive(self) -> None:
        """"Apple" inside "Apple Vision Pro" is two real mentions, not a conflict.

        The de-duplication that stops repeated names from colliding must key on
        the name, not on span overlap, or every product-inside-company mention
        would be silently discarded.
        """
        text = "The Apple Vision Pro shipped late."
        stage, _ = make_stage(
            [
                response(
                    [
                        {"surface": "Apple Vision Pro", "type": "Product", "start": 4, "end": 20},
                        {"surface": "Apple", "type": "Company", "start": 4, "end": 9},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(text))

        assert [(m.surface, m.type) for m in signal.entities] == [
            ("Apple Vision Pro", EntityType.PRODUCT),
            ("Apple", EntityType.COMPANY),
        ]
        assert_spans_are_honest(signal)

    async def test_mentions_are_emitted_in_document_order(self) -> None:
        """Stable order, so reprocessing the same answer produces the same list.

        Outer spans lead the inner spans they contain, which is what a nested
        highlighter needs; `test_nested_mentions_of_different_names_both_survive`
        covers that half.
        """
        stage, _ = make_stage(
            [
                response(
                    [
                        {"surface": "Grafana", "type": "Product", "start": 48, "end": 55},
                        {"surface": "Datadog", "type": "Company", "start": 8, "end": 15},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert [m.start for m in signal.entities] == sorted(m.start for m in signal.entities)

    async def test_only_the_window_the_model_saw_is_searched(self) -> None:
        """Repair must not reach into text the model was never shown.

        The prefix sent to the model is bounded for cost. A mention repaired onto
        an occurrence beyond that boundary would be an offset the model could not
        have meant, invented by the repair rather than by the model -- which is
        the same wrong-span failure arriving through the fix.
        """
        text = "x" * 40 + " Datadog is fine."
        stage, _ = make_stage(
            [response([{"surface": "Datadog", "type": "Company", "start": 3, "end": 10}])],
            max_chars=20,
        )
        signal = await run_stage(stage, make_signal(text))

        assert signal.entities == []

    async def test_prefix_window_keeps_reported_offsets_valid(self) -> None:
        """Truncation is a prefix, so in-window offsets need no adjustment."""
        text = "Datadog costs more. " + "filler. " * 200
        stage, _ = make_stage(
            [response([{"surface": "Datadog", "type": "Company", "start": 0, "end": 7}])],
            max_chars=30,
        )
        signal = await run_stage(stage, make_signal(text))

        assert [(m.start, m.end) for m in signal.entities] == [(0, 7)]
        assert_spans_are_honest(signal)

    def test_locate_span_refuses_rather_than_guessing(self) -> None:
        """The locator's contract, exercised without the stage around it."""
        assert locate_span(BODY, "Datadog", reported_start=8, reported_end=15) == (8, 15)
        assert locate_span(BODY, "Datadog", reported_start=99, reported_end=106) == (8, 15)
        assert locate_span(BODY, "Splunk") is None
        assert locate_span(BODY, "   ") is None
        assert locate_span("", "Datadog") is None
        assert locate_span(BODY, "Datadog", claimed=[(8, 15)]) is None


class TestUnicodeOffsets:
    """Text that is not plain ASCII, which is where offset bugs actually live."""

    async def test_emoji_shift_utf16_style_offsets_and_are_repaired(self) -> None:
        """A model counting UTF-16 code units drifts by one per astral character.

        This is not hypothetical: it is what any JavaScript-side tokenizer
        reports, and it only manifests on documents containing an emoji, so it
        survives every test written against ASCII fixtures.
        """
        text = "🚀🚀 Datadog raised prices."
        assert text.index("Datadog") == 3  # Python counts code points
        utf16_start = len(text[: text.index("Datadog")].encode("utf-16-le")) // 2
        assert utf16_start == 5  # what the model would report

        stage, _ = make_stage(
            [
                response(
                    [
                        {
                            "surface": "Datadog",
                            "type": "Company",
                            "start": utf16_start,
                            "end": utf16_start + 7,
                        }
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(text))

        assert [(m.start, m.end) for m in signal.entities] == [(3, 10)]
        assert_spans_are_honest(signal)

    async def test_utf8_byte_offsets_over_cjk_are_repaired(self) -> None:
        """A model counting bytes drifts by two per CJK character."""
        text = "私たちはDatadogを使っています。"
        char_start = text.index("Datadog")
        byte_start = len(text[:char_start].encode("utf-8"))
        assert byte_start != char_start

        stage, _ = make_stage(
            [
                response(
                    [
                        {
                            "surface": "Datadog",
                            "type": "Company",
                            "start": byte_start,
                            "end": byte_start + 7,
                        }
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(text, language="ja"))

        assert [(m.start, m.end) for m in signal.entities] == [(char_start, char_start + 7)]
        assert_spans_are_honest(signal)

    async def test_cjk_surface_is_extracted_with_exact_offsets(self) -> None:
        """A non-Latin surface must round-trip through the span unchanged."""
        text = "楽天とアマゾンの価格競争が激しい。"
        start = text.index("アマゾン")
        stage, _ = make_stage(
            [
                response(
                    [
                        {"surface": "楽天", "type": "Company", "start": 0, "end": 2},
                        {"surface": "アマゾン", "type": "Company", "start": start, "end": start + 4},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(text, language="ja"))

        assert [m.surface for m in signal.entities] == ["楽天", "アマゾン"]
        assert_spans_are_honest(signal)

    async def test_astral_characters_between_mentions_do_not_corrupt_later_spans(self) -> None:
        """Repairing one mention must not disturb the next one's offsets."""
        text = "Grafana 🎯 and Loki 🔥 replaced Datadog."
        stage, _ = make_stage(
            [
                response(
                    [
                        {"surface": "Grafana", "type": "Product", "start": 0, "end": 7},
                        {"surface": "Loki", "type": "Product", "start": 99, "end": 103},
                        {"surface": "Datadog", "type": "Company", "start": None, "end": None},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(text))

        assert [m.surface for m in signal.entities] == ["Grafana", "Loki", "Datadog"]
        assert_spans_are_honest(signal)

    async def test_combining_marks_are_not_normalized_away(self) -> None:
        """NFC/NFD folding would change the length of the text and every offset."""
        text = "Zoë Baker joined Grafana Labs."
        stage, _ = make_stage(
            [response([{"surface": "Zoë Baker", "type": "Person", "start": 0, "end": 9}])]
        )
        signal = await run_stage(stage, make_signal(text))

        assert [m.surface for m in signal.entities] == ["Zoë Baker"]
        assert_spans_are_honest(signal)


# --------------------------------------------------------------------------- #
# Degradation
# --------------------------------------------------------------------------- #


class TestDegradation:
    """Stage 4 degrades to `[]`; where that happens is part of the contract."""

    @pytest.mark.parametrize(
        "malformed",
        [
            "not json at all",
            '{"mentions": "Datadog"}',
            '{"mentions": [[1, 2, 3]]}',
            "[]",
            '{"mentions": null}',
        ],
    )
    async def test_malformed_response_raises_so_the_pipeline_can_record_it(
        self, malformed: str
    ) -> None:
        """The stage raises; it does not swallow the failure and report success.

        `pipeline.py` is explicit that a stage "must NEVER catch its own exception
        and pretend success". A stage that returned quietly here would write
        `status = "ok"` into `lineage.stages[]`, claim the 0.35 of
        `extraction_quality` that `STAGE_QUALITY_WEIGHTS` assigns to entities, and
        leave a prompt regression indistinguishable from a document that mentions
        nobody. The Signal would look fully enriched and be empty.
        """
        stage, _ = make_stage([malformed])
        signal = make_signal(BODY)

        with pytest.raises(LLMSchemaError):
            await run_stage(stage, signal)

    async def test_pipeline_degrades_a_malformed_response_to_empty_lists(self) -> None:
        """End to end: the malformed answer costs the fields, not the Signal.

        This is `docs/signal-model.md` §5.2 in one assertion -- the Signal
        survives, stays retrievable, records why, and carries the empty values the
        stage contract documents.
        """
        stage, _ = make_stage(["not json at all"])
        signal = make_signal(BODY)
        result = await SignalPipeline([stage]).run(EnrichmentContext(signal=signal))

        assert result.signal is not None
        assert result.signal.entities == []
        assert result.signal.topics == []
        assert result.signal.keywords == []
        assert result.status is SignalStatus.PARTIAL
        assert result.succeeded  # `partial` is still a usable, citable Signal
        assert result.status.is_retrievable

        record = result.signal.lineage.latest_stages()[StageName.ENTITIES]
        assert record.status is StageStatus.FAILED
        assert record.error == "LLMSchemaError"
        assert record.model == "fast-tier-model"

    async def test_provider_outage_propagates_for_the_sweeper_to_retry(self) -> None:
        """A rate limit is retryable, so it must reach the pipeline unmodified.

        Converting it to an empty result here would mark the Signal `enriched`
        with no entities, and the `partial`-row sweeper that re-drives degraded
        Signals would never look at it again.
        """
        stage, _ = make_stage([LLMRateLimited(retry_after_seconds=30.0)])

        with pytest.raises(LLMRateLimited):
            await run_stage(stage, make_signal(BODY))

    async def test_one_junk_item_does_not_discard_the_valid_ones(self) -> None:
        """Per-item tolerance, which is a different decision from per-response.

        A response is either usable or not; an item within a usable response is
        just an item. Failing the batch because the model put a string where an
        offset belonged would throw away four good mentions to punish a fifth.
        """
        stage, _ = make_stage(
            [
                json.dumps(
                    {
                        "mentions": [
                            {"surface": "Datadog", "type": "Company", "start": "?", "end": None},
                            {"surface": 42, "type": "Company", "start": 0, "end": 2},
                            {"surface": "", "type": "Company", "start": 0, "end": 2},
                            {"surface": "Grafana", "type": "Product", "start": 48, "end": 55},
                        ],
                        "topics": [{"topic": "vendor-pricing", "score": "high"}],
                    }
                )
            ]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert [m.surface for m in signal.entities] == ["Datadog", "Grafana"]
        assert_spans_are_honest(signal)

    async def test_missing_keys_and_extra_keys_are_tolerated(self) -> None:
        """"Nothing to extract" is an answer, and `{}` is how a model gives it."""
        stage, _ = make_stage(['{"note": "nothing here", "mentions": []}'])
        signal = await run_stage(stage, make_signal(BODY))

        assert signal.entities == []
        assert signal.topics == []
        assert signal.keywords  # keywords are deterministic and unaffected


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #


class TestCost:
    """Stage 4 runs on every ingested Signal, so call count is a design property."""

    async def test_entities_topics_and_keywords_cost_exactly_one_call(self) -> None:
        """Keywords are deterministic, so they add no call and no output tokens.

        A second per-Signal call for keywords would roughly double the ingestion
        model bill for the least model-shaped part of the job.
        """
        stage, provider = make_stage(
            [
                response(
                    [{"surface": "Datadog", "type": "Company", "start": 8, "end": 15}],
                    [{"topic": "vendor-pricing", "score": 0.9}],
                )
            ]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert len(provider.calls) == 1
        assert signal.entities and signal.topics and signal.keywords

    async def test_empty_text_makes_no_call_at_all(self) -> None:
        """Media-only posts are routine; paying to be told so is not defensible."""
        stage, provider = make_stage([])  # an unscripted call would raise
        signal = await run_stage(stage, make_signal("   \n  "))

        assert provider.calls == []
        assert signal.entities == []
        assert signal.topics == []
        assert signal.keywords == []

    async def test_the_call_goes_to_the_fast_tier_with_a_bounded_output(self) -> None:
        """Tier choice here dominates the model bill; pin it."""
        stage, provider = make_stage([response()])
        await run_stage(stage, make_signal(BODY))

        call = provider.calls[0]
        assert call.kind == "structured"
        assert call.model == "fast-tier-model"
        assert call.max_tokens is not None and call.max_tokens <= 4096
        assert call.schema == "EntityExtraction"

    async def test_only_a_bounded_prefix_of_a_long_document_is_sent(self) -> None:
        """An unbounded body would put a whole article in the prompt of every call."""
        stage, provider = make_stage([response()])
        await run_stage(stage, make_signal("Datadog. " * 20_000))

        assert len(provider.calls[0].prompt) < MAX_EXTRACTION_CHARS + 2_000

    def test_model_id_defaults_to_the_configured_fast_model(self) -> None:
        """`model_id` is written to lineage; it must name the model actually used."""
        settings = LLMSettings()
        stage = EntityExtractionStage(FakeLLMProvider(), settings=settings)

        assert stage.model_id == settings.model_fast


# --------------------------------------------------------------------------- #
# Types, candidates and the resolution boundary
# --------------------------------------------------------------------------- #


class TestTypesAndCandidates:
    """Extraction proposes; `graph/resolution/` decides."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Company", EntityType.COMPANY),
            ("company", EntityType.COMPANY),
            ("ORG", EntityType.COMPANY),
            ("organisation", EntityType.COMPANY),
            ("GPE", EntityType.REGION),
            ("loc", EntityType.REGION),
            ("framework", EntityType.TECHNOLOGY),
            ("app", EntityType.PRODUCT),
            ("PER", EntityType.PERSON),
            ("something-nobody-defined", EntityType.UNKNOWN),
            ("", EntityType.UNKNOWN),
        ],
    )
    def test_common_ner_tagsets_map_onto_the_graph_labels(
        self, raw: str, expected: EntityType
    ) -> None:
        """Models answer in whatever tagset they were trained on.

        `EntityType` values are Neo4j labels, so an unmapped `ORG` would create a
        Signal whose mentions cannot join anything in the graph.
        """
        assert coerce_entity_type(raw) is expected

    async def test_an_unknown_type_keeps_its_span(self) -> None:
        """The span is the expensive, verified half; the label is the cheap half.

        Dropping a verified mention because its label was unfamiliar discards the
        part that took an LLM call to obtain, and resolution can type it later.
        """
        stage, _ = make_stage(
            [response([{"surface": "Datadog", "type": "Zebra", "start": 8, "end": 15}])]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert [(m.surface, m.type) for m in signal.entities] == [
            ("Datadog", EntityType.UNKNOWN)
        ]

    async def test_extraction_never_resolves(self) -> None:
        """`resolved_id` and `link_score` stay `None`. This is a layering rule.

        A resolution guessed inside an enrichment stage is indistinguishable
        downstream from one made by `graph/resolution/` with the alias table and
        the corpus in view -- and it would be wrong far more often.
        """
        stage, _ = make_stage(
            [
                response(
                    [
                        {
                            "surface": "Datadog",
                            "type": "Company",
                            "start": 8,
                            "end": 15,
                            "candidates": ["ent_datadog"],
                        }
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(BODY))

        mention = signal.entities[0]
        assert mention.resolved_id is None
        assert mention.link_score is None
        assert not mention.is_resolved

    def test_candidate_ids_lead_with_a_deterministic_blocking_key(self) -> None:
        """The derived key is first because it is the only reproducible one."""
        assert candidate_ids_for("Datadog")[0] == "ent_datadog"
        assert candidate_ids_for("Elastic Search")[0] == "ent_elastic_search"

    def test_model_proposed_candidate_ids_are_shape_checked(self) -> None:
        """A model asked for an id sometimes answers with a sentence.

        Unchecked, that string reaches `graph/resolution/` as a lookup key and
        into `Entity.merged_from` audit trails as if it named something.
        """
        hints = candidate_ids_for(
            "Datadog",
            ["ent_datadog_inc", "the company that makes Datadog", "", "ENT_DATADOG"],
        )

        assert hints == ["ent_datadog", "ent_datadog_inc"]

    def test_candidate_ids_survive_a_surface_with_no_ascii(self) -> None:
        """A blocking key must exist for non-Latin names too, or they never block."""
        assert candidate_ids_for("楽天") == ["ent_楽天"]


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #


class TestTopics:
    """The vocabulary is closed. Nothing at runtime may widen it."""

    async def test_topics_outside_the_vocabulary_are_dropped(self) -> None:
        """An invented slug would silently turn the closed set into an open one.

        Every cross-source topic aggregate depends on membership being finite and
        shared; one accepted invention makes those counts wrong with no field
        recording that it happened.
        """
        stage, _ = make_stage(
            [
                response(
                    topics=[
                        {"topic": "vendor-pricing", "score": 0.9},
                        {"topic": "pricing-pressure-in-the-observability-market", "score": 0.95},
                    ]
                )
            ]
        )
        signal = await run_stage(stage, make_signal(BODY))

        assert [t.topic for t in signal.topics] == ["vendor-pricing"]

    def test_aliases_and_spellings_normalize_onto_one_slug(self) -> None:
        """Recovering a proposal is not the same as widening the set.

        An alias is a curator's decision recorded in `TOPIC_VOCABULARY`; the
        model still cannot introduce a member.
        """
        assert normalize_topic("Vendor Pricing") == "vendor-pricing"
        assert normalize_topic("VENDOR_PRICING") == "vendor-pricing"
        assert normalize_topic("pricing") == "vendor-pricing"
        assert normalize_topic("  observability  ") == "observability-tooling"
        assert normalize_topic("pricing-pressure") is None

    def test_scores_are_clamped_deduplicated_and_ordered(self) -> None:
        """A model answering `1.4` must not raise inside a degradable stage.

        `TopicScore.score` is a `Score`; an unclamped value would fail validation
        and turn a cosmetic model error into a failed stage and a `partial`
        Signal.
        """
        topics = select_topics(
            [("vendor-pricing", 1.4), ("vendor-pricing", 0.3), ("churn", -2.0)]
        )

        assert [(t.topic, t.score) for t in topics] == [
            ("vendor-pricing", 1.0),
            ("customer-churn", 0.0),
        ]

    def test_the_vocabulary_has_no_ambiguous_aliases(self) -> None:
        """Two members claiming one alias would bind by source order.

        Enforced at import by `_build_index`; asserted here so the failure is a
        named test rather than a collection error in an unrelated suite.
        """
        seen: dict[str, str] = {}
        for definition in TOPIC_VOCABULARY:
            for alias in (definition.slug, *definition.aliases):
                assert seen.setdefault(normalize_topic(alias) or "", definition.slug) == (
                    definition.slug
                )


# --------------------------------------------------------------------------- #
# Keywords
# --------------------------------------------------------------------------- #


class TestKeywords:
    """Open vocabulary, deterministic, and consumed by BM25."""

    async def test_terms_actually_occur_in_the_document(self) -> None:
        """The property that makes keywords useful to a lexical index.

        A model asked for keywords abstracts -- "pricing concerns" for a document
        that says "renewal quote". A term absent from the document cannot help
        BM25 match the document, which is why this half of stage 4 does not use
        the model at all.
        """
        stage, _ = make_stage([response()])
        signal = await run_stage(stage, make_signal(BODY))

        folded = signal.content.text.casefold()
        assert signal.keywords
        for keyword in signal.keywords:
            assert keyword.term.casefold() in folded

    def test_weights_are_scores_and_the_top_term_anchors_at_one(self) -> None:
        """`Keyword.weight` is a `Score`; RAKE's raw values are unbounded."""
        keywords = extract_keywords(BODY)

        assert all(0.0 <= k.weight <= 1.0 for k in keywords)
        assert max(k.weight for k in keywords) == pytest.approx(1.0)

    def test_stopwords_never_become_keywords(self) -> None:
        assert all(k.term.casefold() not in STOPWORDS for k in extract_keywords(BODY))

    def test_extraction_is_deterministic(self) -> None:
        """Stage 4 is already non-deterministic through the model.

        Keeping keywords reproducible means an extraction-model swap changes
        `entities` alone, which is what makes an A/B of two models interpretable.
        """
        assert extract_keywords(BODY) == extract_keywords(BODY)

    def test_entity_surfaces_are_boosted(self) -> None:
        """The mentions are already in hand, so the salience signal is free."""
        text = "Grafana dashboards are fine. Teams discuss dashboards constantly."
        plain = {k.term: k.weight for k in extract_keywords(text)}
        boosted = {k.term: k.weight for k in extract_keywords(text, entity_surfaces=["Grafana"])}

        assert boosted["Grafana dashboards"] > plain["Grafana dashboards"]

    def test_punctuation_breaks_phrases(self) -> None:
        """Without it, "Datadog, Grafana" becomes a term occurring nowhere."""
        terms = {k.term for k in extract_keywords("We compared Datadog, Grafana and Loki.")}

        assert "Datadog Grafana" not in terms

    def test_empty_and_whitespace_text_yield_nothing(self) -> None:
        assert extract_keywords("") == []
        assert extract_keywords("   \n\t ") == []

    def test_cjk_text_is_bounded_rather_than_correct(self) -> None:
        """A documented gap, pinned so it cannot silently get worse.

        There is no segmenter for scripts without whitespace word boundaries, so
        terms come out clause-shaped. The contract that must hold regardless is
        that nothing unbounded reaches `Keyword.term`.
        """
        keywords = extract_keywords("私たちはDatadogを使っていますが価格が高すぎます。" * 5)

        assert all(len(k.term) <= 60 for k in keywords)
        assert all(0.0 <= k.weight <= 1.0 for k in keywords)


# --------------------------------------------------------------------------- #
# The stage contract
# --------------------------------------------------------------------------- #


class TestStageContract:
    """The stage must be substitutable wherever `Stage` is expected."""

    def test_satisfies_the_stage_protocol(self) -> None:
        stage = EntityExtractionStage(FakeLLMProvider(), settings=LLMSettings())

        assert isinstance(stage, Stage)
        assert stage.name is StageName.ENTITIES
        assert stage.version

    def test_the_stage_is_not_fatal(self) -> None:
        """Guards the classification stage 4 depends on for its degradation."""
        from models.enums import FATAL_STAGES

        assert StageName.ENTITIES not in FATAL_STAGES

    async def test_running_before_normalize_fails_loudly(self) -> None:
        """Reading a Signal that does not exist yet is a wiring bug, not a data bug."""
        stage, _ = make_stage([response()])

        with pytest.raises(RuntimeError, match="no Signal on the context"):
            await stage.apply(EnrichmentContext())

    async def test_a_successful_run_records_the_model_in_lineage(self) -> None:
        """Stage 4 is non-deterministic, so the model id is what makes it explainable."""
        stage, _ = make_stage([response()])
        signal = make_signal(BODY)
        result = await SignalPipeline([stage]).run(EnrichmentContext(signal=signal))

        record = result.signal.lineage.latest_stages()[StageName.ENTITIES] if result.signal else None
        assert record is not None
        assert record.status is StageStatus.OK
        assert record.model == "fast-tier-model"
        assert result.status is SignalStatus.ENRICHED


class TestResolveMentionsDirectly:
    """`resolve_mentions` without a provider, for the cases a script cannot express."""

    def test_the_cap_keeps_the_models_own_ordering(self) -> None:
        """Truncation must keep what the model ranked first, then sort by position.

        Sorting before truncating would silently substitute "earliest in the
        document" for "most important", which is a different answer wearing the
        same shape.
        """
        text = "Alpha Beta Gamma Delta"
        raw = [
            ExtractedMention(surface=name, type="Company")
            for name in ("Delta", "Gamma", "Alpha", "Beta")
        ]
        mentions = resolve_mentions(raw, text=text, limit=2)

        assert [m.surface for m in mentions] == ["Gamma", "Delta"]

    def test_every_returned_mention_slices_back_to_its_surface(self) -> None:
        """The single invariant, asserted over a deliberately hostile batch."""
        text = "Datadog 🚀 and datadog and Datadog\nLabs."
        raw = [
            ExtractedMention(surface="Datadog", type="Company", start=0, end=7),
            ExtractedMention(surface="Datadog", type="Company", start=99, end=106),
            ExtractedMention(surface="Datadog Labs", type="Company", start=1, end=13),
            ExtractedMention(surface="Splunk", type="Company", start=0, end=6),
            ExtractedMention(surface="", type="Company", start=0, end=1),
        ]
        mentions: list[EntityMention] = resolve_mentions(raw, text=text)

        assert mentions
        for mention in mentions:
            assert text[mention.start : mention.end] == mention.surface
        assert "Splunk" not in [m.surface for m in mentions]
