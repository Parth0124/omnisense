"""Unit tests for the canonical Signal and its nested types.

`docs/signal-model.md` §8 names two properties the rest of the system assumes and
which therefore deserve explicit assertions: reprocessing the same fixture twice
yields an identical `id`, and a Signal whose enrichment stages all fail still
round-trips with a reduced confidence. Both are covered below.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from models.entity import Entity, EntityMention
from models.enums import (
    EntityType,
    Platform,
    SentimentLabel,
    SignalStatus,
    SourceCategory,
    StageName,
    StageStatus,
)
from models.lineage import Lineage, StageRecord
from models.signal import (
    Author,
    Content,
    Engagement,
    Language,
    Sentiment,
    Signal,
    SignalView,
    signal_id,
)

pytestmark = pytest.mark.unit

VALID_SHA = "ebad8169cc3aeee5890e6632a636c33c28f220f38f92a45cfc7182bdb9cd967e"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def lineage() -> Lineage:
    return Lineage(
        pipeline_version="1.0.0",
        connector_slug="reddit",
        connector_version="0.1.0",
        sync_run_id="run_01J8XN5Q2P",
        fetched_at="2026-07-28T14:29:55Z",
        native_id="t3_1abcde",
        status=SignalStatus.ENRICHED,
        raw_sha256=VALID_SHA,
    )


@pytest.fixture
def signal(lineage: Lineage) -> Signal:
    return Signal.create(
        platform=Platform.REDDIT,
        native_id="t3_1abcde",
        timestamp="2026-07-28T14:02:11Z",
        content=Content(text="Our observability bill tripled after the renewal."),
        lineage=lineage,
    )


# --------------------------------------------------------------------------- #
# Identity (§4.1)
# --------------------------------------------------------------------------- #


class TestIdentity:
    def test_is_deterministic(self) -> None:
        """The property that makes every store idempotent."""
        assert signal_id(Platform.REDDIT, "t3_1abcde") == signal_id(Platform.REDDIT, "t3_1abcde")

    def test_is_stable_across_runs(self) -> None:
        """Pinned literal: a change here silently orphans every stored artifact."""
        assert signal_id(Platform.REDDIT, "t3_1abcde") == ("sig_af53359fd1835722956a66f1e051c33e")

    def test_is_platform_scoped(self) -> None:
        assert signal_id(Platform.X, "t3_1abcde") != signal_id(Platform.REDDIT, "t3_1abcde")

    def test_accepts_str_or_enum(self) -> None:
        assert signal_id("reddit", "abc") == signal_id(Platform.REDDIT, "abc")

    def test_rejects_empty_native_id(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            signal_id(Platform.REDDIT, "")

    def test_reprocessing_yields_identical_id(self, signal: Signal) -> None:
        """`docs/signal-model.md` §8, property 1."""
        again = Signal.model_validate(signal.model_dump(mode="json"))
        assert again.id == signal.id


# --------------------------------------------------------------------------- #
# Invariants enforced by the model
# --------------------------------------------------------------------------- #


class TestInvariants:
    def test_rejects_assigned_id(self, lineage: Lineage) -> None:
        with pytest.raises(ValidationError, match="derived, never assigned"):
            Signal(
                id="sig_deadbeef",
                source=SourceCategory.SOCIAL,
                platform=Platform.REDDIT,
                timestamp="2026-07-28T14:02:11Z",
                content=Content(text="x"),
                lineage=lineage,
            )

    def test_rejects_platform_source_mismatch(self, signal: Signal, lineage: Lineage) -> None:
        with pytest.raises(ValidationError, match="does not match platform"):
            Signal(
                id=signal.id,
                source=SourceCategory.NEWS,
                platform=Platform.REDDIT,
                timestamp="2026-07-28T14:02:11Z",
                content=Content(text="x"),
                lineage=lineage,
            )

    def test_create_infers_source(self, signal: Signal) -> None:
        assert signal.source is SourceCategory.SOCIAL

    def test_rejects_deep_metadata(self, lineage: Lineage) -> None:
        with pytest.raises(ValidationError, match="levels deep"):
            Signal.create(
                platform=Platform.REDDIT,
                native_id="t3_1abcde",
                timestamp="2026-07-28T14:02:11Z",
                content=Content(text="x"),
                lineage=lineage,
                metadata={"a": {"b": {"c": {"d": 1}}}},
            )

    def test_allows_metadata_at_the_limit(self, lineage: Lineage) -> None:
        sig = Signal.create(
            platform=Platform.REDDIT,
            native_id="t3_1abcde",
            timestamp="2026-07-28T14:02:11Z",
            content=Content(text="x"),
            lineage=lineage,
            metadata={"reddit.awards": {"gold": [1, 2]}},
        )
        assert sig.metadata["reddit.awards"] == {"gold": [1, 2]}

    def test_rejects_naive_timestamp(self, lineage: Lineage) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            Signal.create(
                platform=Platform.REDDIT,
                native_id="t3_1abcde",
                timestamp=datetime(2026, 7, 28, 14, 2, 11),
                content=Content(text="x"),
                lineage=lineage,
            )

    def test_normalizes_offset_to_utc(self, lineage: Lineage) -> None:
        sig = Signal.create(
            platform=Platform.REDDIT,
            native_id="t3_1abcde",
            timestamp="2026-07-28T16:02:11+02:00",
            content=Content(text="x"),
            lineage=lineage,
        )
        assert sig.timestamp.tzinfo is UTC
        assert sig.timestamp.hour == 14


# --------------------------------------------------------------------------- #
# Producer strict / consumer lenient (§7)
# --------------------------------------------------------------------------- #


class TestSchemaTolerance:
    @pytest.fixture
    def newer_payload(self, signal: Signal) -> dict:
        payload = signal.model_dump(mode="json")
        payload["viral_score"] = 0.9  # field from a newer pipeline_version
        payload["platform"] = "mastodon"  # platform this build has never heard of
        return payload

    def test_producer_rejects_unknown_field(self, newer_payload: dict) -> None:
        with pytest.raises(ValidationError):
            Signal.model_validate(newer_payload)

    def test_consumer_tolerates_unknown_field(self, newer_payload: dict) -> None:
        assert SignalView.model_validate(newer_payload).confidence == 0.0

    def test_consumer_degrades_unknown_enum(self, newer_payload: dict) -> None:
        assert SignalView.model_validate(newer_payload).platform is Platform.UNKNOWN

    def test_known_enum_still_resolves(self) -> None:
        assert Platform("REDDIT") is Platform.REDDIT

    def test_closed_enum_still_raises(self) -> None:
        """`SignalStatus` omits UNKNOWN deliberately -- it is pipeline-owned."""
        with pytest.raises(ValueError):
            SignalStatus("bogus")


# --------------------------------------------------------------------------- #
# Content, Language, Engagement
# --------------------------------------------------------------------------- #


class TestContent:
    def test_derives_char_count(self) -> None:
        assert Content(text="hello").char_count == 5

    def test_respects_explicit_char_count(self) -> None:
        """A truncated body reports the original length, not the excerpt's."""
        assert Content(text="hello", char_count=9000).char_count == 9000

    def test_rejects_malformed_sha256(self) -> None:
        with pytest.raises(ValidationError):
            Content(text="x", raw_sha256="tooshort")

    def test_accepts_valid_sha256(self) -> None:
        assert Content(text="x", raw_sha256=VALID_SHA).raw_sha256 == VALID_SHA

    def test_is_empty_ignores_whitespace(self) -> None:
        assert Content(text="   ").is_empty


class TestLanguage:
    def test_weak_detection_becomes_und(self) -> None:
        assert Language.detected("en", 0.42, "langdetect").code == "und"

    def test_strong_detection_is_kept(self) -> None:
        assert Language.detected("en", 0.99, "langdetect").code == "en"

    def test_und_is_not_filterable(self) -> None:
        assert not Language.detected("en", 0.42, "langdetect").is_determinate

    def test_floor_is_inclusive(self) -> None:
        assert Language.detected("en", 0.70, "langdetect").code == "en"


class TestEngagement:
    def test_weighted_mean_of_all_axes(self) -> None:
        eng = Engagement(reach=0.71, endorsement=0.88, amplification=0.34, discussion=0.92)
        assert eng.compute_score() == 0.729

    def test_renormalizes_over_available_axes(self) -> None:
        """A source lacking an axis must not be penalized for lacking it."""
        eng = Engagement(reach=0.5, discussion=1.0)
        assert eng.compute_score() == pytest.approx((0.5 * 0.30 + 1.0 * 0.20) / 0.50)

    def test_no_axes_is_none_not_zero(self) -> None:
        """0.0 would read as 'nobody engaged'; None reads as 'unknown'."""
        assert Engagement().compute_score() is None

    def test_raw_counters_are_preserved(self) -> None:
        eng = Engagement(raw={"score": 412, "num_comments": 137})
        assert eng.raw["score"] == 412


# --------------------------------------------------------------------------- #
# Lineage
# --------------------------------------------------------------------------- #


class TestLineage:
    def test_stage_failure_requires_error(self) -> None:
        with pytest.raises(ValidationError, match="recorded no error"):
            StageRecord(
                name=StageName.SENTIMENT,
                version="1.0.0",
                started_at="2026-07-28T14:30:03Z",
                duration_ms=1,
                status=StageStatus.FAILED,
            )

    def test_stage_success_forbids_error(self) -> None:
        with pytest.raises(ValidationError, match="recorded an error"):
            StageRecord(
                name=StageName.SENTIMENT,
                version="1.0.0",
                started_at="2026-07-28T14:30:03Z",
                duration_ms=1,
                status=StageStatus.OK,
                error="Timeout",
            )

    def test_extraction_quality_accumulates(self, lineage: Lineage) -> None:
        assert lineage.compute_extraction_quality() == 0.0
        for name in (
            StageName.LANGUAGE,
            StageName.ENTITIES,
            StageName.SENTIMENT,
            StageName.EMBEDDING,
        ):
            lineage.append_stage(
                StageRecord(
                    name=name,
                    version="1.0.0",
                    started_at="2026-07-28T14:30:02Z",
                    duration_ms=1,
                    status=StageStatus.OK,
                )
            )
        assert lineage.compute_extraction_quality() == 1.0

    def test_reprocessing_appends_and_newest_wins(self, lineage: Lineage) -> None:
        for name in (
            StageName.LANGUAGE,
            StageName.ENTITIES,
            StageName.SENTIMENT,
            StageName.EMBEDDING,
        ):
            lineage.append_stage(
                StageRecord(
                    name=name,
                    version="1.0.0",
                    started_at="2026-07-28T14:30:02Z",
                    duration_ms=1,
                    status=StageStatus.OK,
                )
            )
        lineage.append_stage(
            StageRecord(
                name=StageName.SENTIMENT,
                version="1.1.0",
                started_at="2026-07-29T09:00:00Z",
                duration_ms=1,
                status=StageStatus.FAILED,
                error="ProviderTimeout",
            )
        )
        assert len(lineage.stages) == 5, "history must be preserved"
        assert lineage.compute_extraction_quality() == 0.80
        assert lineage.failed_stages() == [StageName.SENTIMENT]

    def test_skipped_earns_full_credit(self, lineage: Lineage) -> None:
        """A stage disabled by configuration is not a quality failure."""
        for name in (
            StageName.LANGUAGE,
            StageName.ENTITIES,
            StageName.EMBEDDING,
        ):
            lineage.append_stage(
                StageRecord(
                    name=name,
                    version="1.0.0",
                    started_at="2026-07-28T14:30:02Z",
                    duration_ms=1,
                    status=StageStatus.OK,
                )
            )
        lineage.append_stage(
            StageRecord(
                name=StageName.SENTIMENT,
                version="1.0.0",
                started_at="2026-07-28T14:30:02Z",
                duration_ms=0,
                status=StageStatus.SKIPPED,
            )
        )
        assert lineage.compute_extraction_quality() == 1.0

    def test_duplicate_requires_pointer(self) -> None:
        with pytest.raises(ValidationError, match="unreachable"):
            Lineage(
                pipeline_version="1.0.0",
                connector_slug="reddit",
                connector_version="0.1.0",
                sync_run_id="r",
                fetched_at="2026-07-28T14:29:55Z",
                native_id="x",
                status=SignalStatus.DUPLICATE,
            )


# --------------------------------------------------------------------------- #
# Degradation (§5.2) and round-tripping
# --------------------------------------------------------------------------- #


class TestDegradation:
    def test_fully_degraded_signal_still_round_trips(self, lineage: Lineage) -> None:
        """`docs/signal-model.md` §8, property 2.

        Every degradable stage failed. The Signal must still be constructible,
        serializable and retrievable -- with a reduced confidence, not an error.
        """
        for name in (
            StageName.LANGUAGE,
            StageName.ENTITIES,
            StageName.SENTIMENT,
            StageName.EMBEDDING,
        ):
            lineage.append_stage(
                StageRecord(
                    name=name,
                    version="1.0.0",
                    started_at="2026-07-28T14:30:02Z",
                    duration_ms=1,
                    status=StageStatus.FAILED,
                    error="ProviderTimeout",
                )
            )
        lineage.status = SignalStatus.PARTIAL

        sig = Signal.create(
            platform=Platform.REDDIT,
            native_id="t3_1abcde",
            timestamp="2026-07-28T14:02:11Z",
            content=Content(text="body survived"),
            lineage=lineage,
            confidence=0.21,
        )

        assert lineage.compute_extraction_quality() == 0.0
        assert sig.is_retrievable, "a partial Signal is still citable"
        assert sig.language.code == "und"
        assert sig.entities == [] and sig.embeddings == [] and sig.sentiment is None

        again = Signal.model_validate(sig.model_dump(mode="json"))
        assert again.model_dump(mode="json") == sig.model_dump(mode="json")

    def test_duplicate_is_not_retrievable(self, signal: Signal) -> None:
        signal.lineage.dedup_cluster_id = "dc_1"
        signal.lineage.duplicate_of = "sig_other"
        signal.lineage.status = SignalStatus.DUPLICATE
        assert not signal.is_retrievable
        assert not signal.is_canonical

    def test_no_vectors_leak_into_serialization(self, signal: Signal) -> None:
        """EmbeddingRef carries point ids, never float arrays."""
        assert "vector" not in signal.model_dump_json()


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #


class TestEntities:
    def test_rejects_inverted_span(self) -> None:
        with pytest.raises(ValidationError, match="inverted"):
            EntityMention(surface="x", type=EntityType.COMPANY, start=11, end=4)

    def test_rejects_link_score_without_resolution(self) -> None:
        with pytest.raises(ValidationError, match="unresolved"):
            EntityMention(surface="x", type=EntityType.COMPANY, start=0, end=1, link_score=0.5)

    def test_backfills_resolver_id_into_candidates(self) -> None:
        """The candidate list is the audit trail; keep it complete."""
        m = EntityMention(
            surface="DD",
            type=EntityType.COMPANY,
            start=0,
            end=2,
            resolved_id="ent_datadog",
            link_score=0.6,
        )
        assert m.candidate_ids == ["ent_datadog"]

    def test_signal_collects_distinct_entity_ids_in_order(self, signal: Signal) -> None:
        signal.entities = [
            EntityMention(
                surface="Datadog",
                type=EntityType.COMPANY,
                start=0,
                end=7,
                resolved_id="ent_datadog",
                link_score=0.9,
            ),
            EntityMention(
                surface="Grafana",
                type=EntityType.PRODUCT,
                start=8,
                end=15,
                resolved_id="ent_grafana",
                link_score=0.9,
            ),
            EntityMention(
                surface="DD",
                type=EntityType.COMPANY,
                start=16,
                end=18,
                resolved_id="ent_datadog",
                link_score=0.7,
            ),
            EntityMention(surface="unknown", type=EntityType.COMPANY, start=19, end=26),
        ]
        assert signal.resolved_entity_ids() == ["ent_datadog", "ent_grafana"]

    def test_entity_rejects_inverted_temporal_range(self) -> None:
        with pytest.raises(ValidationError, match="precedes"):
            Entity(
                id="e",
                type=EntityType.COMPANY,
                canonical_name="X",
                first_seen="2026-01-01T00:00:00Z",
                last_seen="2024-01-01T00:00:00Z",
            )

    def test_entity_surfaces_include_aliases(self) -> None:
        ent = Entity(
            id="ent_datadog",
            type=EntityType.COMPANY,
            canonical_name="Datadog",
            aliases=["Datadog Inc", "DDOG"],
        )
        assert ent.all_surfaces() == {"Datadog", "Datadog Inc", "DDOG"}


# --------------------------------------------------------------------------- #
# Nested types the pipeline populates
# --------------------------------------------------------------------------- #


class TestNestedTypes:
    def test_polarity_bounds_are_enforced(self) -> None:
        with pytest.raises(ValidationError):
            Sentiment(polarity=-1.5, label=SentimentLabel.NEGATIVE)

    def test_mixed_is_distinct_from_neutral(self) -> None:
        assert SentimentLabel("mixed") is not SentimentLabel.NEUTRAL

    def test_author_requires_platform_id(self) -> None:
        with pytest.raises(ValidationError):
            Author(handle="ops_gremlin")

    def test_confidence_bounds_are_enforced(self, lineage: Lineage) -> None:
        with pytest.raises(ValidationError):
            Signal.create(
                platform=Platform.REDDIT,
                native_id="t3_1abcde",
                timestamp="2026-07-28T14:02:11Z",
                content=Content(text="x"),
                lineage=lineage,
                confidence=1.4,
            )
