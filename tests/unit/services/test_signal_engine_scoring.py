"""Unit tests for enrichment stages 6 (Embedding) and 6b (Scoring).

These two stages decide what a Signal *costs* and what it is *worth*, and both
fail quietly when they fail at all. The tests are aimed accordingly.

- **The geometric mean.** `docs/signal-model.md` §3.5 picks a weighted geometric
  mean over an arithmetic one for a single reason: a claim is trustworthy only
  if every component holds, so one dead component must sink the score rather
  than being averaged away by three healthy ones. That is the whole design, so
  it is asserted directly -- against a literal arithmetic mean of the same
  components, on the same inputs.
- **Point ids that are not stable.** A random point id turns every embedding
  retry into a duplicated Qdrant point: retrieval returns the same chunk twice,
  a report cites it twice as if it were two sources, and the collection grows
  without bound while every health check stays green.
- **Vectors leaking into the Signal.** `EmbeddingRef` exists so that ~20 KB of
  JSON floats per chunk never enters Kafka or PostgreSQL. Nothing raises if one
  does; the topic just starts rejecting long documents.
- **Spans that do not match their source.** `services/evidence_service.py`
  verifies quotes by re-reading `[char_start, char_end)`. A chunker that trims
  or synthesizes text breaks every citation it produces, and it breaks them
  later, in the Critic, against Signals that were correct when written.
- **Invented percentiles.** An engagement axis normalized against an empty
  cohort is a fabricated number wearing the costume of a measurement
  (`docs/signal-model.md` §9, open question 4).

Everything runs offline: the embedding provider is `FakeEmbeddingProvider`, the
cohort store is the in-memory port, and no settings are read from the
environment except in the one test that asserts the default collection.
"""

from __future__ import annotations

import itertools
import math
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, ClassVar

import pytest

from backend.core.config import EmbeddingSettings, get_settings
from models.enums import (
    Platform,
    SignalStatus,
    SourceCategory,
    StageName,
    StageStatus,
)
from models.lineage import ConfidenceComponents, Lineage, StageRecord
from models.signal import (
    SIGNAL_ID_NAMESPACE,
    Author,
    Content,
    EmbeddingRef,
    Engagement,
    Signal,
)
from retrieval.chunking.splitter import (
    Chunk,
    ChunkStrategy,
    chunk_id,
    split_text,
    strategy_for,
)
from services.llm.embeddings import EmbeddingDimensionMismatch, FakeEmbeddingProvider
from services.signal_engine.embeddings import (
    ChunkVector,
    EmbeddingStage,
    InMemoryVectorSink,
    point_id_for,
)
from services.signal_engine.enrichment import (
    CONFIDENCE_FLOOR,
    CONFIDENCE_WEIGHTS,
    MAX_CORROBORATING_SOURCES,
    MIN_COHORT_SAMPLES,
    ClusterCorroboration,
    CohortPercentile,
    ColdStartPolicy,
    InMemoryCohortBaseline,
    InMemoryCorroborationIndex,
    ScoringStage,
    compose_confidence,
    content_integrity_of,
    corroboration_of,
    source_credibility_of,
)
from services.signal_engine.pipeline import EnrichmentContext, SignalPipeline, Stage

pytestmark = pytest.mark.unit


COLLECTION = "test_signals"


# --------------------------------------------------------------------------- #
# Fixtures and fakes
# --------------------------------------------------------------------------- #


def make_signal(
    *,
    text: str = "A reasonably long body sentence. And a second one to chunk.",
    title: str | None = "A title",
    platform: Platform = Platform.REDDIT,
    native_id: str = "t3_1abcde",
    truncated: bool = False,
    author: Author | None = None,
    engagement: Engagement | None = None,
    metadata: dict[str, Any] | None = None,
    dedup_cluster_id: str | None = None,
    duplicate_of: str | None = None,
    content_type: str = "text/plain",
) -> Signal:
    """A minimal but valid Signal. Built through `create()` so the id is derived."""
    lineage = Lineage(
        pipeline_version="1.0.0",
        connector_slug="test",
        connector_version="0.1.0",
        sync_run_id="run_1",
        fetched_at=datetime(2026, 7, 28, 14, 29, 55, tzinfo=UTC),
        native_id=native_id,
        dedup_cluster_id=dedup_cluster_id,
        duplicate_of=duplicate_of,
    )
    return Signal.create(
        platform=platform,
        native_id=native_id,
        timestamp=datetime(2026, 7, 28, 14, 2, 11, tzinfo=UTC),
        content=Content(title=title, text=text, truncated=truncated, content_type=content_type),
        lineage=lineage,
        author=author,
        engagement=engagement or Engagement(),
        metadata=metadata or {},
    )


def record_stages(signal: Signal, *names: StageName, status: StageStatus = StageStatus.OK) -> None:
    """Append stage records so `compute_extraction_quality()` has something to read."""
    for name in names:
        signal.lineage.append_stage(
            StageRecord(
                name=name,
                version="1.0.0",
                started_at=datetime(2026, 7, 28, 14, 30, tzinfo=UTC),
                duration_ms=1,
                status=status,
                error="RuntimeError" if status is StageStatus.FAILED else None,
            )
        )


def embedding_settings(**overrides: Any) -> EmbeddingSettings:
    values: dict[str, Any] = {
        "model": "fake-embed-v1",
        "dimensions": 8,
        "batch_size": 2,
        "max_chars_per_chunk": 200,
        "chunk_overlap_chars": 20,
    }
    values.update(overrides)
    return EmbeddingSettings(_env_file=None, **values)


def embedding_stage(
    provider: Any = None,
    *,
    sink: InMemoryVectorSink | None = None,
    **setting_overrides: Any,
) -> EmbeddingStage:
    return EmbeddingStage(
        provider or FakeEmbeddingProvider(dimensions=8),
        settings=embedding_settings(**setting_overrides),
        collection=COLLECTION,
        sink=sink,
    )


class ExplodingEmbeddingProvider:
    """A provider that fails the way a rate-limited one does."""

    model = "fake-embed-v1"
    dimensions = 8

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        raise self._error

    async def aclose(self) -> None:  # pragma: no cover -- never closed in these tests
        return None


class WidthDriftingProvider:
    """Declares one width and returns another -- the model-swap failure."""

    model = "fake-embed-v1"
    dimensions = 8

    def __init__(self, actual: int) -> None:
        self._actual = actual

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * self._actual for _ in texts]

    async def aclose(self) -> None:  # pragma: no cover
        return None


class ShortCountProvider:
    """Returns fewer vectors than it was given texts."""

    model = "fake-embed-v1"
    dimensions = 8

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in texts][:-1]

    async def aclose(self) -> None:  # pragma: no cover
        return None


class ExplodingBaseline:
    """A cohort store that is down."""

    async def percentile(self, **_: Any) -> CohortPercentile | None:
        raise ConnectionError("cohort store unreachable")


def arithmetic_mean(components: ConfidenceComponents) -> float:
    """The mean `docs/signal-model.md` §3.5 explicitly rejects.

    Present in the test file, not in the implementation, precisely so the
    difference between the two can be asserted rather than described.
    """
    return sum(
        float(getattr(components, name)) * weight
        for name, weight in CONFIDENCE_WEIGHTS.items()
    )


def components(**overrides: float) -> ConfidenceComponents:
    values = {
        "source_credibility": 0.9,
        "extraction_quality": 0.9,
        "content_integrity": 0.9,
        "corroboration": 0.9,
    }
    values.update(overrides)
    return ConfidenceComponents(**values)


def mature_cohort(
    platform: Platform = Platform.REDDIT,
    content_type: str = "text/plain",
    *,
    axes: tuple[str, ...] = ("reach", "endorsement", "amplification", "discussion"),
    size: int = 200,
) -> InMemoryCohortBaseline:
    """A cohort of `size` evenly spread observations per axis, 0..size-1."""
    baseline = InMemoryCohortBaseline(window="30d")
    for axis in axes:
        baseline.extend(platform, content_type, axis, [float(value) for value in range(size)])
    return baseline


# --------------------------------------------------------------------------- #
# The confidence composite -- the reason this module exists
# --------------------------------------------------------------------------- #


class TestGeometricMeanIsNotAnArithmeticMean:
    """The single decision `docs/signal-model.md` §3.5 makes, asserted directly.

    If these pass with `compose_confidence` reimplemented as a weighted sum,
    the module has lost the property it was designed around.
    """

    def test_one_near_zero_component_drags_the_composite_far_below_the_mean(self) -> None:
        """A title-only scrap from a credible source must not score like an article.

        Everything is excellent except `content_integrity`, which is near zero.
        The arithmetic mean barely notices -- 0.2 of weight moving from 0.95 to
        0.02 costs it under 0.19. The geometric mean multiplies by a factor near
        zero and cannot recover.
        """
        crippled = components(
            source_credibility=0.95,
            extraction_quality=0.95,
            content_integrity=0.02,
            corroboration=0.95,
        )
        composite = compose_confidence(crippled)
        assert arithmetic_mean(crippled) > 0.75
        assert composite < 0.55
        assert arithmetic_mean(crippled) - composite > 0.20

    def test_two_signals_with_equal_arithmetic_means_are_ranked_by_their_weakest(self) -> None:
        """Same average, different shape -- and the shape is what matters.

        One Signal is uniformly mediocre; the other is excellent on three
        components and dead on the fourth. An arithmetic mean calls them equal
        and an agent picks either. The geometric mean prefers the one with no
        fatal hole, which is the ranking a human analyst would make.
        """
        crippled = components(
            source_credibility=1.0,
            extraction_quality=1.0,
            content_integrity=0.05,
            corroboration=1.0,
        )
        level = arithmetic_mean(crippled)
        balanced = components(
            source_credibility=level,
            extraction_quality=level,
            content_integrity=level,
            corroboration=level,
        )

        assert arithmetic_mean(balanced) == pytest.approx(arithmetic_mean(crippled))
        assert compose_confidence(balanced) - compose_confidence(crippled) > 0.20

    def test_a_dead_component_caps_the_composite_no_matter_the_rest(self) -> None:
        """Perfect on everything else still cannot beat `FLOOR ** weight`.

        This is the ceiling the geometric mean imposes and the arithmetic mean
        does not have at all: with three components at 1.0 a weighted sum floors
        out at 0.80, which reads as a strong Signal.
        """
        dead = components(
            source_credibility=1.0,
            extraction_quality=1.0,
            content_integrity=0.0,
            corroboration=1.0,
        )
        cap = CONFIDENCE_FLOOR ** CONFIDENCE_WEIGHTS["content_integrity"]
        assert compose_confidence(dead) == pytest.approx(cap, abs=1e-6)
        assert arithmetic_mean(dead) >= 0.80

    def test_the_floor_keeps_a_dead_component_from_zeroing_the_score(self) -> None:
        """`0.0` and `FLOOR` must compose identically.

        A true zero would produce `confidence == 0.0`, which is
        indistinguishable from "never scored" to both the API and a
        `WHERE confidence > 0` filter -- so a weak Signal would vanish from
        retrieval rather than rank last.
        """
        assert compose_confidence(components(corroboration=0.0)) > 0.0
        assert compose_confidence(components(corroboration=0.0)) == compose_confidence(
            components(corroboration=CONFIDENCE_FLOOR)
        )

    def test_all_components_perfect_scores_one(self) -> None:
        """The composite must reach its upper bound, or `Score` validation is wasted."""
        assert compose_confidence(components(**dict.fromkeys(CONFIDENCE_WEIGHTS, 1.0))) == 1.0

    @pytest.mark.parametrize("name", sorted(CONFIDENCE_WEIGHTS))
    def test_the_composite_is_monotonic_in_every_component(self, name: str) -> None:
        """Raising any component must raise the score.

        A non-monotonic scorer is unexplainable in the UI: "we extracted more
        entities and the confidence went down" has no honest answer.
        """
        low = compose_confidence(components(**{name: 0.4}))
        high = compose_confidence(components(**{name: 0.8}))
        assert high > low

    def test_weights_match_the_specification_and_sum_to_one(self) -> None:
        """Weights that do not sum to 1.0 push the composite outside [0, 1].

        `Score` would then reject the assignment, and the traceback would point
        at pydantic rather than at the arithmetic that caused it.
        """
        assert CONFIDENCE_WEIGHTS == {
            "source_credibility": 0.35,
            "extraction_quality": 0.25,
            "content_integrity": 0.20,
            "corroboration": 0.20,
        }
        assert sum(CONFIDENCE_WEIGHTS.values()) == pytest.approx(1.0)


class TestTheModuleDocstringWorkedExample:
    """Pins the five figures quoted in `enrichment.py`'s module docstring.

    The docstring justifies the geometric mean with a worked example, and a
    worked example is a second implementation of the formula written in prose.
    It had already drifted once -- quoting a composite from one set of inputs
    beside an arithmetic mean from a different set -- and nothing failed,
    because prose has no test. This class is that test: change a weight or the
    floor and it goes red, pointing at the paragraph that now lies.
    """

    CREDIBLE_PUBLISHER: ClassVar[dict[str, float]] = {
        "source_credibility": 0.80,
        "extraction_quality": 1.0,
        "corroboration": 1.0,
    }

    def test_an_intact_body_scores_the_same_under_either_mean(self) -> None:
        """The control: with nothing broken the choice of mean barely matters."""
        intact = components(**self.CREDIBLE_PUBLISHER, content_integrity=1.0)
        assert arithmetic_mean(intact) == pytest.approx(0.93)
        assert compose_confidence(intact) == pytest.approx(0.924872, abs=5e-7)

    def test_a_dead_body_separates_the_two_means(self) -> None:
        """The point of the paragraph: 0.73 still reads usable, 0.508014 does not.

        `content_integrity_of` returns 0.0 for a media-only post; the floor
        clamps it to `CONFIDENCE_FLOOR` inside the composite, which is why the
        arithmetic figure is taken from the raw 0.0 and the composite is not.
        """
        dead = components(**self.CREDIBLE_PUBLISHER, content_integrity=0.0)
        assert arithmetic_mean(dead) == pytest.approx(0.73)
        assert compose_confidence(dead) == pytest.approx(0.508014, abs=5e-7)

    def test_a_title_only_scrap_sits_between_the_two(self) -> None:
        """0.2 is what `content_integrity_of` actually returns for title-only."""
        scrap = components(**self.CREDIBLE_PUBLISHER, content_integrity=0.2)
        assert arithmetic_mean(scrap) == pytest.approx(0.77)
        assert compose_confidence(scrap) == pytest.approx(0.670328, abs=5e-7)


class TestConfidenceComponents:
    """Each input to the composite, on its own."""

    def test_content_integrity_follows_the_documented_three_cases(self) -> None:
        """1.0 full body, 0.5 truncated, 0.2 title-only (`docs/signal-model.md` §3.5)."""
        assert content_integrity_of(make_signal(text="Full body here.")) == 1.0
        assert content_integrity_of(make_signal(text="Excerpt...", truncated=True)) == 0.5
        assert content_integrity_of(make_signal(text="", title="Only a title")) == 0.2

    def test_a_signal_with_neither_text_nor_title_scores_zero(self) -> None:
        """A media-only post has nothing a quote can be verified against.

        The doc enumerates three cases; this is the fourth. Zero is floored to
        `CONFIDENCE_FLOOR` in the composite, so the Signal still exists and still
        counts toward trend volume -- it just cannot carry a claim.
        """
        assert content_integrity_of(make_signal(text="   ", title=None)) == 0.0

    def test_source_credibility_uses_the_platform_prior_when_there_is_no_author(self) -> None:
        """Most connectors cannot see author signals; absence must not penalize."""
        rss = make_signal(platform=Platform.RSS, native_id="guid-1", author=None)
        reddit = make_signal(author=None)
        assert source_credibility_of(rss) > source_credibility_of(reddit)

    def test_a_platform_override_beats_its_category_prior(self) -> None:
        """arXiv is research, but a preprint is not peer review.

        Without the override an unreviewed submission would carry the same
        weight as a published paper, which is exactly the claim a Strategy agent
        should not be allowed to make cheaply.
        """
        arxiv = make_signal(platform=Platform.ARXIV, native_id="2401.01234v2")
        semantic = make_signal(platform=Platform.SEMANTIC_SCHOLAR, native_id="corpus:42")
        assert source_credibility_of(arxiv) < source_credibility_of(semantic)

    def test_author_signals_raise_credibility_without_leaving_the_unit_interval(self) -> None:
        """The bonus is applied toward 1.0, so it can never overflow `Score`."""
        anonymous = make_signal(author=Author(platform_author_id="t2_x"))
        established = make_signal(
            author=Author(
                platform_author_id="t2_x",
                verified=True,
                follower_count=250_000,
                account_age_days=4_000,
            )
        )
        assert source_credibility_of(established) > source_credibility_of(anonymous)
        assert source_credibility_of(established) <= 1.0

    def test_follower_count_is_log_scaled(self) -> None:
        """The interesting gap is 100 vs 10,000, not 900,000 vs 910,000.

        Linear scaling would make every non-celebrity account look identical to
        a brand-new one, which is the opposite of what the component measures.
        """
        def credibility(followers: int) -> float:
            return source_credibility_of(
                make_signal(author=Author(platform_author_id="t2_x", follower_count=followers))
            )

        small_step = credibility(10_000) - credibility(100)
        large_step = credibility(910_000) - credibility(900_000)
        assert small_step > large_step * 10

    def test_corroboration_treats_a_lone_signal_as_one_source_not_zero(self) -> None:
        """A Signal is itself an observation.

        At enrichment time almost nothing is corroborated yet -- the copies have
        not been fetched. Scoring a lone Signal at zero would floor every
        newly-ingested item regardless of how good it is.
        """
        assert 0.0 < corroboration_of(None) < 0.5

    def test_corroboration_rises_with_independent_platforms_and_saturates(self) -> None:
        """Log-scaled, per `docs/signal-model.md` §3.5, capped at 1.0."""
        one = corroboration_of(ClusterCorroboration(members=1, independent_platforms=1))
        four = corroboration_of(ClusterCorroboration(members=9, independent_platforms=4))
        many = corroboration_of(
            ClusterCorroboration(members=50, independent_platforms=MAX_CORROBORATING_SOURCES * 3)
        )
        assert one < four < 1.0
        assert many == 1.0

    def test_corroboration_counts_platforms_not_cluster_members(self) -> None:
        """Six copies in one subreddit is one community, not six sources.

        `docs/signal-model.md` §4.3 dedups per platform when counting spread, so
        that a single platform cannot manufacture corroboration for its own
        claim.
        """
        one_platform = ClusterCorroboration(members=6, independent_platforms=1)
        four_platforms = ClusterCorroboration(members=6, independent_platforms=4)
        assert corroboration_of(one_platform) < corroboration_of(four_platforms)

    async def test_extraction_quality_comes_from_lineage_not_a_second_copy(self) -> None:
        """A failed stage must cost exactly its `STAGE_QUALITY_WEIGHTS` share.

        Recomputing the weighting inside the scoring stage would fork from
        `models/lineage.py` the first time a stage was added, and the two would
        disagree silently.
        """
        signal = make_signal()
        record_stages(signal, StageName.LANGUAGE, StageName.ENTITIES, StageName.SENTIMENT)
        record_stages(signal, StageName.EMBEDDING, status=StageStatus.FAILED)

        await ScoringStage(baseline=InMemoryCohortBaseline()).apply(
            EnrichmentContext(signal=signal)
        )

        assert signal.lineage.confidence_components is not None
        assert signal.lineage.confidence_components.extraction_quality == pytest.approx(0.70)


# --------------------------------------------------------------------------- #
# Engagement normalization
# --------------------------------------------------------------------------- #


class TestEngagementNormalization:
    """`docs/signal-model.md` §3.4: percentiles inside a cohort, never raw counters."""

    async def test_axes_are_percentiles_within_the_platform_cohort(self) -> None:
        """A value at the top of its cohort scores near 1.0, at the bottom near 0."""
        signal = make_signal(
            engagement=Engagement(
                raw={
                    "subreddit_subscribers": 199.0,
                    "score": 0.0,
                    "crossposts": 100.0,
                    "num_comments": 100.0,
                }
            )
        )
        await ScoringStage(baseline=mature_cohort()).apply(EnrichmentContext(signal=signal))

        assert signal.engagement.reach == pytest.approx(1.0)
        assert signal.engagement.endorsement == pytest.approx(0.005)
        assert signal.engagement.discussion == pytest.approx(0.505)

    async def test_a_cohort_is_scoped_to_its_own_platform(self) -> None:
        """A 400-point Reddit post is never scored against YouTube views.

        The whole reason the axes exist. If cohorts leaked across platforms,
        every YouTube video would outrank every Reddit post on `reach` by six
        orders of magnitude, and cross-platform ranking would just be a list of
        the highest-volume platform.
        """
        baseline = InMemoryCohortBaseline(window="30d")
        baseline.extend(Platform.YOUTUBE, "text/plain", "reach", [1e6] * 200)
        baseline.extend(Platform.REDDIT, "text/plain", "reach", [float(i) for i in range(200)])

        signal = make_signal(engagement=Engagement(raw={"subreddit_subscribers": 190.0}))
        await ScoringStage(baseline=baseline).apply(EnrichmentContext(signal=signal))

        assert signal.engagement.reach is not None
        assert signal.engagement.reach > 0.9

    async def test_the_score_renormalizes_over_the_axes_a_platform_supports(self) -> None:
        """An RSS item with no `endorsement` axis is not penalized for lacking one.

        Delegated to `Engagement.compute_score()`, which owns the
        renormalization; a second implementation here would fork the weights the
        moment `ENGAGEMENT_AXIS_WEIGHTS` changed.
        """
        baseline = InMemoryCohortBaseline(window="30d")
        baseline.extend(Platform.RSS, "text/plain", "amplification", [0.0] * 200)
        baseline.extend(Platform.RSS, "text/plain", "discussion", [0.0] * 200)

        signal = make_signal(
            platform=Platform.RSS,
            native_id="guid-1",
            engagement=Engagement(raw={"syndication_count": 5.0, "comments": 5.0}),
        )
        await ScoringStage(baseline=baseline).apply(EnrichmentContext(signal=signal))

        assert signal.engagement.reach is None
        assert signal.engagement.endorsement is None
        # Both surviving axes are at the top of their cohorts, so the
        # renormalized mean is 1.0 rather than the 0.4 an un-renormalized
        # weighting over four axes would produce.
        assert signal.engagement.score == pytest.approx(1.0)

    async def test_raw_counters_survive_verbatim(self) -> None:
        """`engagement.raw` is the platform's, and normalization must not touch it.

        Overwriting a counter with its percentile would be undetectable
        afterwards: an upvote ratio and a percentile are both plausible floats
        in [0, 1].
        """
        raw = {"score": 412, "num_comments": 137, "upvote_ratio": 0.94}
        signal = make_signal(engagement=Engagement(raw=dict(raw)))
        await ScoringStage(baseline=mature_cohort()).apply(EnrichmentContext(signal=signal))
        assert signal.engagement.raw == raw

    async def test_an_unmapped_platform_yields_no_axes_and_a_null_score(self) -> None:
        """`None` means unknown; `0.0` would mean nobody engaged.

        Retrieval reads a zero as a real measurement and ranks the Signal last
        forever, which is a permanent penalty for a connector we simply have not
        mapped yet.
        """
        signal = make_signal(
            platform=Platform.NOTION, native_id="page-1", engagement=Engagement(raw={"score": 9.0})
        )
        await ScoringStage(baseline=mature_cohort(Platform.NOTION)).apply(
            EnrichmentContext(signal=signal)
        )
        assert signal.engagement.available_axes() == {}
        assert signal.engagement.score is None

    async def test_a_null_counter_is_skipped_rather_than_read_as_zero(self) -> None:
        """`Engagement.raw` allows `None`, and a connector writes it for "not returned".

        Reading it as 0.0 would percentile-rank an unreported counter at the
        bottom of its cohort -- a measurement invented out of a missing field.
        """
        signal = make_signal(engagement=Engagement(raw={"score": None, "num_comments": 5.0}))
        await ScoringStage(baseline=mature_cohort()).apply(EnrichmentContext(signal=signal))
        assert signal.engagement.endorsement is None
        assert signal.engagement.discussion is not None

    async def test_the_cohort_key_can_be_refined_by_the_connector(self) -> None:
        """"Text post versus link post" is knowledge only a connector has.

        Nothing above `connectors/` may branch on platform shape
        (`models/signal.py`), so the connector publishes the discriminator into
        metadata and this stage reads it blind.
        """
        baseline = InMemoryCohortBaseline(window="30d")
        baseline.extend(Platform.REDDIT, "text_post", "endorsement", [0.0] * 200)

        signal = make_signal(
            engagement=Engagement(raw={"score": 5.0}),
            metadata={"cohort.content_type": "text_post"},
        )
        await ScoringStage(baseline=baseline).apply(EnrichmentContext(signal=signal))

        assert signal.engagement.endorsement == pytest.approx(1.0)
        assert signal.engagement.baseline_window is not None
        assert "text_post" in signal.engagement.baseline_window

    async def test_the_baseline_label_records_platform_cohort_and_window(self) -> None:
        """§3.4: a score from a cold cohort is not comparable to one from a mature one.

        Mirrored onto `lineage.engagement_baseline` so the provenance survives
        even if the engagement block is re-derived later.
        """
        signal = make_signal(engagement=Engagement(raw={"score": 5.0}))
        await ScoringStage(baseline=mature_cohort()).apply(EnrichmentContext(signal=signal))

        assert signal.engagement.baseline_window == "reddit:text/plain:30d"
        assert signal.lineage.engagement_baseline == "reddit:text/plain:30d"


class TestEngagementColdStart:
    """`docs/signal-model.md` §9, open question 4 -- documented, not assumed away."""

    async def test_an_empty_cohort_omits_the_axis_rather_than_inventing_one(self) -> None:
        """No baseline means no percentile. There is no honest default.

        A 0.5 here would be a fabricated number indistinguishable from a
        measured one, and it would bias every cross-platform comparison toward
        whichever platform is newest.
        """
        signal = make_signal(engagement=Engagement(raw={"score": 412.0}))
        await ScoringStage(baseline=InMemoryCohortBaseline()).apply(
            EnrichmentContext(signal=signal)
        )
        assert signal.engagement.endorsement is None
        assert signal.engagement.score is None

    async def test_a_cold_start_is_distinguishable_from_no_engagement(self) -> None:
        """"No baseline" and "nobody engaged" need different fixes.

        Both leave the axes empty, so without the label on the record they are
        the same row in the database.
        """
        cold = make_signal(engagement=Engagement(raw={"score": 412.0}))
        await ScoringStage(baseline=InMemoryCohortBaseline()).apply(EnrichmentContext(signal=cold))

        silent = make_signal(engagement=Engagement(raw={}))
        await ScoringStage(baseline=mature_cohort()).apply(EnrichmentContext(signal=silent))

        assert cold.engagement.baseline_window is not None
        assert cold.engagement.baseline_window.endswith(":coldstart")
        assert silent.engagement.baseline_window is None

    async def test_a_thin_cohort_is_omitted_under_the_default_policy(self) -> None:
        """Below `MIN_COHORT_SAMPLES` a percentile encodes sampling noise."""
        thin = mature_cohort(size=MIN_COHORT_SAMPLES - 1)
        signal = make_signal(engagement=Engagement(raw={"score": 5.0}))
        await ScoringStage(baseline=thin).apply(EnrichmentContext(signal=signal))
        assert signal.engagement.endorsement is None

    async def test_the_provisional_policy_keeps_the_axis_and_flags_it(self) -> None:
        """The other plausible answer from §9, available and clearly marked.

        A deployment that wants early Signals to rank at all can opt in; what it
        cannot do is get the number without the flag.
        """
        thin = mature_cohort(size=MIN_COHORT_SAMPLES - 1)
        signal = make_signal(engagement=Engagement(raw={"score": 5.0}))
        stage = ScoringStage(baseline=thin, cold_start=ColdStartPolicy.PROVISIONAL)
        await stage.apply(EnrichmentContext(signal=signal))

        assert signal.engagement.endorsement is not None
        assert signal.engagement.baseline_window is not None
        assert ":provisional" in signal.engagement.baseline_window


class TestScoringStageContract:
    """The stage's obligations to `pipeline.py`."""

    def test_it_satisfies_the_stage_protocol(self) -> None:
        """Structural conformance, checked rather than assumed.

        The pipeline accepts anything shaped like a `Stage`; a missing
        `model_id` would only surface when a record failed and lineage was
        written.
        """
        stage = ScoringStage(baseline=InMemoryCohortBaseline())
        assert isinstance(stage, Stage)
        assert stage.name is StageName.SCORING
        assert stage.version

    def test_it_reports_no_model_because_it_is_deterministic(self) -> None:
        """`docs/signal-model.md` §5.1 lists 6b as reproducible given its inputs.

        Recording a model id would imply a replay could drift, which would make
        the reprocessing guarantee in §5.3 unreadable.
        """
        assert ScoringStage(baseline=InMemoryCohortBaseline()).model_id is None

    async def test_it_writes_the_components_next_to_the_scalar(self) -> None:
        """§3.5: a low score has to be explainable, in the UI and to the Critic.

        A bare float leaves "why is this 0.31?" unanswerable, which is how a
        scoring system stops being trusted.
        """
        signal = make_signal()
        record_stages(signal, StageName.LANGUAGE, StageName.ENTITIES)
        await ScoringStage(baseline=mature_cohort()).apply(EnrichmentContext(signal=signal))

        stored = signal.lineage.confidence_components
        assert stored is not None
        assert signal.confidence == compose_confidence(stored)

    async def test_a_cohort_store_outage_degrades_rather_than_dropping_the_signal(self) -> None:
        """Stage 6b is degradable (`docs/signal-model.md` §5.2).

        Driven through the real pipeline rather than by calling `apply`, because
        the fatal-vs-degradable decision belongs to the pipeline and this test
        exists to prove the stage did not quietly promote itself.
        """
        signal = make_signal(engagement=Engagement(raw={"score": 412.0}))
        pipeline = SignalPipeline([ScoringStage(baseline=ExplodingBaseline())])
        result = await pipeline.run(EnrichmentContext(signal=signal))

        assert result.status is SignalStatus.PARTIAL
        assert result.succeeded
        assert result.failed_stages == [StageName.SCORING]
        assert signal.lineage.latest_stages()[StageName.SCORING].error == "ConnectionError"
        # The documented empty value: an unscored Signal, not a wrong score.
        assert signal.confidence == 0.0
        assert signal.lineage.confidence_components is None

    async def test_rescoring_replaces_rather_than_accumulates(self) -> None:
        """§5.3: reprocessing is an upsert. Confidence must not ratchet.

        `lineage.stages[]` is append-only, so a second run reads more records;
        the components must still reflect the latest state only.
        """
        signal = make_signal()
        stage = ScoringStage(baseline=mature_cohort())
        await stage.apply(EnrichmentContext(signal=signal))
        first = signal.confidence

        record_stages(signal, StageName.LANGUAGE, StageName.ENTITIES)
        await stage.apply(EnrichmentContext(signal=signal))

        assert signal.confidence > first
        assert signal.lineage.confidence_components is not None


# --------------------------------------------------------------------------- #
# Chunking -- stage 6's input
# --------------------------------------------------------------------------- #


class TestChunkSpansMatchTheirSource:
    """The invariant every citation depends on (`docs/retrieval.md` §8)."""

    @pytest.mark.parametrize(
        "strategy", [ChunkStrategy.WHOLE, ChunkStrategy.PARAGRAPH, ChunkStrategy.HEADING]
    )
    def test_every_chunk_is_an_exact_slice_of_the_original(self, strategy: ChunkStrategy) -> None:
        """`services/evidence_service.py` re-reads `[char_start, char_end)`.

        A chunker that trims, re-joins or normalizes what it emits breaks every
        quote it ever produced -- and breaks it later, in the Critic, against
        Signals that were correct when they were written.
        """
        text = (
            "# Heading one\n\nFirst paragraph with several words in it. Second sentence here.\n\n"
            "Another paragraph, longer than the first one, with more text to pack.\n\n"
            "## Heading two\n\nA third paragraph closes the document out."
        )
        for chunk in split_text(text, strategy=strategy, max_chars=80, overlap_chars=20):
            assert chunk.text == text[chunk.char_start : chunk.char_end]

    def test_chunks_are_ordered_and_contiguously_indexed(self) -> None:
        """Indices are assigned in document order and are the id's only variable part."""
        text = "\n\n".join(f"Paragraph number {index} of the document." for index in range(12))
        chunks = split_text(text, strategy=ChunkStrategy.PARAGRAPH, max_chars=100)
        assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
        assert [chunk.char_start for chunk in chunks] == sorted(
            chunk.char_start for chunk in chunks
        )

    def test_empty_and_whitespace_only_bodies_produce_no_chunks(self) -> None:
        """A media-only post is a normal outcome, not an error."""
        assert split_text("", strategy=ChunkStrategy.WHOLE, max_chars=100) == []
        assert split_text("  \n\n\t ", strategy=ChunkStrategy.PARAGRAPH, max_chars=100) == []

    def test_a_blank_chunk_cannot_be_constructed(self) -> None:
        """Blank chunks cost an embedding call and most providers reject them."""
        with pytest.raises(ValueError, match="blank"):
            Chunk(index=0, text="   ", char_start=0, char_end=3)


class TestChunkBoundaries:
    """§8's rules, each of which exists to prevent one citation failure."""

    def test_sentences_are_not_split_when_a_paragraph_overflows(self) -> None:
        """"We evaluated X and rejected it" cut after "evaluated" supports the opposite.

        The reranker cannot see the half that is missing, so the chunk is worse
        than useless -- it is confidently wrong.
        """
        sentences = [f"Sentence number {index} runs on for a while here." for index in range(10)]
        text = " ".join(sentences)
        chunks = split_text(text, strategy=ChunkStrategy.PARAGRAPH, max_chars=120)

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.text.rstrip().endswith(".")

    def test_an_abbreviation_is_not_a_sentence_boundary(self) -> None:
        """Splitting after "e.g." or "Dr." produces a fragment that cites nothing."""
        text = (
            "The team evaluated several vendors, e.g. Datadog and Grafana Labs, before "
            "deciding. Dr. Smith signed the renewal in March of the following year."
        )
        chunks = split_text(text, strategy=ChunkStrategy.PARAGRAPH, max_chars=90)
        for chunk in chunks:
            assert not chunk.text.startswith("Datadog")
            assert not chunk.text.startswith("Smith")

    def test_a_single_oversized_sentence_is_cut_only_past_twice_the_target(self) -> None:
        """§8 permits a hard cut in exactly one case, and this is it."""
        sentence = "word " * 60 + "end."
        assert len(sentence) < 2 * 200
        assert len(split_text(sentence, strategy=ChunkStrategy.PARAGRAPH, max_chars=200)) == 1

        wall = "x" * 5000
        pieces = split_text(wall, strategy=ChunkStrategy.PARAGRAPH, max_chars=1000)
        assert len(pieces) == 5
        assert all(piece.char_count <= 1000 for piece in pieces)

    def test_a_fragment_below_the_minimum_is_merged_into_its_predecessor(self) -> None:
        """A two-word chunk wins on BM25 term density and carries no context.

        It then outranks the passage that actually answers the question, and the
        citation it produces is unreadable.
        """
        text = "A" * 190 + ".\n\ntiny."
        chunks = split_text(text, strategy=ChunkStrategy.PARAGRAPH, max_chars=200)
        assert len(chunks) == 1
        assert chunks[0].text.endswith("tiny.")

    def test_overlap_widens_the_span_instead_of_prepending_text(self) -> None:
        """Overlap must not break the slice invariant.

        Prepending the previous chunk's tail as a string is the obvious
        implementation and it silently invalidates every span offset.
        """
        text = "\n\n".join(f"Paragraph {index} has a fair number of words." for index in range(8))
        plain = split_text(text, strategy=ChunkStrategy.PARAGRAPH, max_chars=100)
        overlapped = split_text(text, strategy=ChunkStrategy.PARAGRAPH, max_chars=100,
                                overlap_chars=30)

        assert len(plain) == len(overlapped)
        assert overlapped[1].char_start < plain[1].char_start
        for chunk in overlapped:
            assert chunk.text == text[chunk.char_start : chunk.char_end]

    def test_overlap_never_swallows_a_whole_predecessor(self) -> None:
        """Clamped to the previous chunk's own start, so overlap cannot chain."""
        text = "\n\n".join(f"Short {index}." for index in range(6))
        chunks = split_text(text, strategy=ChunkStrategy.PARAGRAPH, max_chars=40,
                            overlap_chars=39)
        for previous, current in itertools.pairwise(chunks):
            assert current.char_start >= previous.char_start

    def test_sections_are_never_packed_together(self) -> None:
        """Heading-aware chunking exists so a chunk belongs to one section.

        Packing the end of "Method" with the start of "Results" produces a
        passage that supports a claim neither section makes.
        """
        text = "# Alpha\n\nOne.\n\n# Beta\n\nTwo.\n\n# Gamma\n\nThree."
        chunks = split_text(text, strategy=ChunkStrategy.HEADING, max_chars=500)
        assert len(chunks) == 3
        assert [chunk.text.splitlines()[0] for chunk in chunks] == ["# Alpha", "# Beta", "# Gamma"]

    def test_overlap_must_be_smaller_than_the_chunk(self) -> None:
        """Otherwise every chunk restarts inside its predecessor and nothing advances."""
        with pytest.raises(ValueError, match="smaller than max_chars"):
            split_text("text", strategy=ChunkStrategy.PARAGRAPH, max_chars=100, overlap_chars=100)

    def test_transcript_chunking_says_what_is_missing(self) -> None:
        """Rule 4 of the build: a gap raises and names the blocker, never `pass`."""
        with pytest.raises(NotImplementedError, match="transcript"):
            split_text("speaker text", strategy=ChunkStrategy.SPEAKER_TURN, max_chars=100)


class TestChunkStrategySelection:
    """§8's source-class table."""

    def test_social_and_reviews_are_never_packed_with_each_other(self) -> None:
        """Two comments by different authors in one chunk misattributes a quote."""
        assert strategy_for(SourceCategory.SOCIAL) is ChunkStrategy.WHOLE
        assert strategy_for(SourceCategory.REVIEWS) is ChunkStrategy.WHOLE

    def test_papers_and_enterprise_documents_are_heading_aware(self) -> None:
        assert strategy_for(SourceCategory.RESEARCH) is ChunkStrategy.HEADING
        assert strategy_for(SourceCategory.ENTERPRISE) is ChunkStrategy.HEADING

    def test_news_and_unknown_sources_are_paragraph_packed(self) -> None:
        """`UNKNOWN` packs rather than emitting one huge chunk for a long body."""
        assert strategy_for(SourceCategory.NEWS) is ChunkStrategy.PARAGRAPH
        assert strategy_for(SourceCategory.UNKNOWN) is ChunkStrategy.PARAGRAPH

    def test_a_short_post_stays_one_chunk(self) -> None:
        """"One signal, one chunk" for anything that fits."""
        chunks = split_text(
            "Our observability bill tripled after the renewal.",
            strategy=ChunkStrategy.WHOLE,
            max_chars=2000,
        )
        assert len(chunks) == 1

    def test_an_oversized_post_degrades_to_packing_instead_of_failing(self) -> None:
        """A deliberate deviation from "never split", and the reason for it.

        A 40 KB self-post handed to a provider whole earns a 400 for exceeding
        the model's context, which fails the stage and loses *every* chunk of the
        Signal rather than splitting one.
        """
        text = "\n\n".join(f"Paragraph {index} of a very long self post." for index in range(60))
        chunks = split_text(text, strategy=ChunkStrategy.WHOLE, max_chars=200)
        assert len(chunks) > 1
        assert all(chunk.char_count <= 200 for chunk in chunks)


# --------------------------------------------------------------------------- #
# Stage 6 -- Embedding
# --------------------------------------------------------------------------- #


class TestEmbeddingRefsCarryAddressesNotVectors:
    """The reason `EmbeddingRef` exists at all."""

    def test_the_ref_model_cannot_hold_a_vector(self) -> None:
        """A 1536-float array is ~20 KB of JSON against a 2-4 KB Signal body.

        Carried on the Signal it would enter Kafka -- whose default
        `max.message.bytes` is 1 MB, so a long document would be *rejected*, not
        merely slow -- plus PostgreSQL, every API response and every DLQ record.
        Qdrant is about to store the numbers anyway.
        """
        assert set(EmbeddingRef.model_fields) == {
            "model",
            "dimensions",
            "chunk_index",
            "collection",
            "point_id",
        }

    async def test_no_vector_component_appears_in_the_serialized_signal(self) -> None:
        """The end-to-end version of the same claim, over real JSON."""
        signal = make_signal(text="Body text worth embedding once.")
        provider = FakeEmbeddingProvider(dimensions=8)
        await embedding_stage(provider).apply(EnrichmentContext(signal=signal))

        payload = signal.model_dump_json()
        vector = provider.vector_for(signal.content.text)
        assert signal.embeddings
        for component in vector:
            assert f"{component:.6f}"[:8] not in payload

    async def test_one_ref_per_chunk_with_the_provider_identity_recorded(self) -> None:
        """Model and width are part of the address: a collection's width is fixed."""
        signal = make_signal(text="\n\n".join(f"Paragraph {i} here." for i in range(6)))
        sink = InMemoryVectorSink()
        await embedding_stage(sink=sink, max_chars_per_chunk=100).apply(
            EnrichmentContext(signal=signal)
        )

        assert len(signal.embeddings) == len(sink.staged[signal.id])
        assert [ref.chunk_index for ref in signal.embeddings] == list(
            range(len(signal.embeddings))
        )
        assert {ref.model for ref in signal.embeddings} == {"fake-embed-v1"}
        assert {ref.dimensions for ref in signal.embeddings} == {8}
        assert {ref.collection for ref in signal.embeddings} == {COLLECTION}

    async def test_the_collection_defaults_to_the_configured_one(self) -> None:
        """`QDRANT_COLLECTION` is the index identity (`docs/retrieval.md` §5)."""
        signal = make_signal(text="Short body.")
        stage = EmbeddingStage(
            FakeEmbeddingProvider(dimensions=8), settings=embedding_settings()
        )
        await stage.apply(EnrichmentContext(signal=signal))
        assert signal.embeddings[0].collection == get_settings().qdrant.collection


class TestPointIdsAreDerived:
    """`docs/data-stores.md` §5.2: upsert by construction."""

    def test_the_point_id_is_uuid5_over_the_chunk_id(self) -> None:
        """Pure, total and identical on every machine, forever."""
        expected = uuid.uuid5(SIGNAL_ID_NAMESPACE, "sig_abc:3")
        assert point_id_for("sig_abc", 3) == str(expected)

    def test_re_running_the_stage_reproduces_the_same_point_ids(self) -> None:
        """A retry after a rate limit must overwrite, not duplicate.

        Random ids would accumulate a fresh copy of every chunk on every retry:
        duplicate hits in retrieval, a report citing one chunk twice as if it
        were two sources, and a collection that grows without bound while every
        health check stays green.
        """
        first = [point_id_for("sig_abc", index) for index in range(4)]
        second = [point_id_for("sig_abc", index) for index in range(4)]
        assert first == second
        assert len(set(first)) == 4

    def test_different_signals_and_chunks_never_collide(self) -> None:
        ids = {point_id_for(f"sig_{s}", index) for s in "ab" for index in range(3)}
        assert len(ids) == 6

    def test_the_chunk_id_is_the_documented_join_key(self) -> None:
        """`{signal_id}:{chunk_index}` is what hybrid fusion joins on (§8).

        OpenSearch uses it as `_id`; Qdrant hashes it into the point id. Two
        chunkers deriving different ids would leave fusion with nothing to join.
        """
        assert chunk_id("sig_abc", 0) == "sig_abc:0"

    async def test_reprocessing_replaces_the_refs_rather_than_appending(self) -> None:
        """§5.3: reprocessing is an upsert, never an append."""
        signal = make_signal(text="\n\n".join(f"Paragraph {i} here." for i in range(4)))
        stage = embedding_stage(max_chars_per_chunk=100)
        await stage.apply(EnrichmentContext(signal=signal))
        first = list(signal.embeddings)
        await stage.apply(EnrichmentContext(signal=signal))

        assert signal.embeddings == first


class TestEmbeddingStageBehaviour:
    """Cost, alignment and the handoff to stage 7."""

    def test_it_satisfies_the_stage_protocol_and_records_its_model(self) -> None:
        """Stage 6 is non-deterministic, so the model id must reach lineage (§5.1)."""
        stage = embedding_stage()
        assert isinstance(stage, Stage)
        assert stage.name is StageName.EMBEDDING
        assert stage.model_id == "fake-embed-v1"

    async def test_chunks_are_embedded_in_configured_batches(self) -> None:
        """One oversized call loses every chunk in it, not the last one.

        Batching is deliberately duplicated here and in the HTTP client: the
        `EmbeddingProvider` protocol promises only "embed every text, in order",
        so a self-hosted provider is free to forward a 400-chunk list verbatim.
        """
        signal = make_signal(text="\n\n".join(f"Paragraph {i} here." for i in range(5)))
        provider = FakeEmbeddingProvider(dimensions=8)
        await embedding_stage(provider, max_chars_per_chunk=100).apply(
            EnrichmentContext(signal=signal)
        )
        assert all(len(batch) <= 2 for batch in provider.batches)
        assert sum(len(batch) for batch in provider.batches) == len(signal.embeddings)

    async def test_the_sink_receives_the_vectors_and_the_citation_spans(self) -> None:
        """Everything the Signal deliberately does not carry, handed off out of band."""
        text = "\n\n".join(f"Paragraph {i} with a little more text." for i in range(4))
        signal = make_signal(text=text)
        sink = InMemoryVectorSink()
        await embedding_stage(sink=sink, max_chars_per_chunk=100).apply(
            EnrichmentContext(signal=signal)
        )

        staged = sink.staged[signal.id]
        assert all(isinstance(item, ChunkVector) for item in staged)
        assert [item.point_id for item in staged] == [
            ref.point_id for ref in signal.embeddings
        ]
        for item in staged:
            assert len(item.vector) == 8
            assert item.text == text[item.char_start : item.char_end]

    async def test_taking_from_the_sink_releases_the_vectors(self) -> None:
        """Holding them after stage 7 has written turns the sink into a leak."""
        signal = make_signal(text="Body text worth embedding.")
        sink = InMemoryVectorSink()
        await embedding_stage(sink=sink).apply(EnrichmentContext(signal=signal))

        assert sink.take(signal.id)
        assert sink.take(signal.id) == []

    async def test_a_non_canonical_duplicate_is_not_embedded(self) -> None:
        """§4.3: only the canonical member is embedded and indexed.

        Also the largest single cost saving in the pipeline -- five sixths of the
        embedding spend on a press release syndicated across six platforms.
        """
        signal = make_signal(
            text="A syndicated press release body.",
            dedup_cluster_id="dc_5f3b21",
            duplicate_of="sig_" + "0" * 32,
        )
        provider = FakeEmbeddingProvider(dimensions=8)
        await embedding_stage(provider).apply(EnrichmentContext(signal=signal))

        assert signal.embeddings == []
        assert provider.batches == []

    async def test_a_body_with_no_text_costs_nothing_and_does_not_fail(self) -> None:
        """A media-only post is valid. Raising would mark it `partial` for nothing."""
        signal = make_signal(text="", title="Just a headline")
        provider = FakeEmbeddingProvider(dimensions=8)
        await embedding_stage(provider).apply(EnrichmentContext(signal=signal))

        assert signal.embeddings == []
        assert provider.batches == []

    async def test_only_the_body_is_chunked_never_title_plus_body(self) -> None:
        """Spans must index `content.text` exactly.

        A synthesized "title\\n\\ntext" would shift every offset by the title's
        length and silently invalidate the citation mechanism.
        """
        signal = make_signal(text="Body only.", title="A long title that would shift offsets")
        sink = InMemoryVectorSink()
        await embedding_stage(sink=sink).apply(EnrichmentContext(signal=signal))

        staged = sink.staged[signal.id][0]
        assert staged.char_start == 0
        assert staged.text == signal.content.text

    async def test_a_width_mismatch_raises_instead_of_recording_a_lie(self) -> None:
        """1536 next to a 1024-wide vector only surfaces at the Qdrant upsert.

        That is after the spend and after the pipeline has called the stage
        successful, so the check happens where the ref is written.
        """
        signal = make_signal(text="Body text worth embedding.")
        with pytest.raises(EmbeddingDimensionMismatch):
            await embedding_stage(WidthDriftingProvider(actual=4)).apply(
                EnrichmentContext(signal=signal)
            )

    async def test_a_short_vector_list_raises_rather_than_misaligning(self) -> None:
        """Silently short results shift every later vector onto the wrong chunk.

        Nothing downstream can detect that: search simply gets worse, for
        reasons no user reports as a bug.
        """
        signal = make_signal(text="\n\n".join(f"Paragraph {i} here." for i in range(4)))
        with pytest.raises(ValueError, match="alignment"):
            await embedding_stage(ShortCountProvider(), max_chars_per_chunk=100).apply(
                EnrichmentContext(signal=signal)
            )

    async def test_a_provider_failure_propagates_and_the_pipeline_degrades(self) -> None:
        """The stage must not catch its own exception and report success.

        If it did, `extraction_quality` would credit it in full and the Signal
        would look complete while being unreachable by vector search forever.
        Degradation is the pipeline's decision, from `FATAL_STAGES`.
        """
        signal = make_signal(text="Body text worth embedding.")
        provider = ExplodingEmbeddingProvider(TimeoutError("provider timed out"))
        pipeline = SignalPipeline([embedding_stage(provider)])
        result = await pipeline.run(EnrichmentContext(signal=signal))

        assert provider.calls == 1
        assert result.status is SignalStatus.PARTIAL
        assert result.succeeded
        assert signal.embeddings == []
        assert signal.lineage.latest_stages()[StageName.EMBEDDING].error == "TimeoutError"
        assert signal.lineage.latest_stages()[StageName.EMBEDDING].model == "fake-embed-v1"

    async def test_no_refs_are_recorded_when_a_later_batch_fails(self) -> None:
        """Refs pointing at points that were never upserted read as corruption.

        Retrieval would return a hit whose vector does not exist, which looks
        like a broken store rather than a failed stage.
        """
        text = "\n\n".join(f"Paragraph {index} of the document body here." for index in range(12))
        signal = make_signal(text=text)
        provider = FailAfterFirstBatch()
        with pytest.raises(RuntimeError):
            await embedding_stage(provider, max_chars_per_chunk=100).apply(
                EnrichmentContext(signal=signal)
            )
        assert provider.calls == 2
        assert signal.embeddings == []


class FailAfterFirstBatch:
    """Succeeds once, then fails -- the rate-limit-mid-document case."""

    model = "fake-embed-v1"
    dimensions = 8

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("rate limited")
        return [[0.1] * 8 for _ in texts]

    async def aclose(self) -> None:  # pragma: no cover
        return None


class TestStageOrderIntegration:
    """Stage 6 and 6b together, as `pipeline.py` will run them."""

    async def test_scoring_reads_the_embedding_outcome_it_ran_after(self) -> None:
        """6b must run after 6, or `extraction_quality` misses 0.30 of its weight.

        The pipeline enforces the order; this asserts the coupling is real, so
        that a future reordering fails here rather than silently under-scoring
        every Signal in the corpus.
        """
        signal = make_signal(text="Body text worth embedding once.")
        record_stages(signal, StageName.LANGUAGE, StageName.ENTITIES, StageName.SENTIMENT)
        pipeline = SignalPipeline(
            [embedding_stage(), ScoringStage(baseline=mature_cohort())]
        )
        result = await pipeline.run(EnrichmentContext(signal=signal))

        assert result.status is SignalStatus.ENRICHED
        assert signal.lineage.confidence_components is not None
        assert signal.lineage.confidence_components.extraction_quality == 1.0
        assert signal.confidence > 0.0

    async def test_a_failed_embedding_lowers_confidence_without_dropping_the_signal(self) -> None:
        """§5.2 end to end: partial, retrievable, and visibly less trustworthy."""
        healthy = make_signal(text="Body text worth embedding once.")
        record_stages(healthy, StageName.LANGUAGE, StageName.ENTITIES, StageName.SENTIMENT)
        await SignalPipeline(
            [embedding_stage(), ScoringStage(baseline=mature_cohort())]
        ).run(EnrichmentContext(signal=healthy))

        degraded = make_signal(text="Body text worth embedding once.")
        record_stages(degraded, StageName.LANGUAGE, StageName.ENTITIES, StageName.SENTIMENT)
        broken = embedding_stage(ExplodingEmbeddingProvider(TimeoutError("nope")))
        result = await SignalPipeline(
            [broken, ScoringStage(baseline=mature_cohort())]
        ).run(EnrichmentContext(signal=degraded))

        assert result.status is SignalStatus.PARTIAL
        assert degraded.is_retrievable is False  # status is set by stage 7, not here
        assert degraded.confidence < healthy.confidence
        assert degraded.confidence > 0.0


class TestPorts:
    """The in-memory implementations the rest of the suite runs against."""

    async def test_the_in_memory_cohort_reports_the_empirical_percentile(self) -> None:
        """The same quantity §3.4 defines, so tests transfer to the real store."""
        baseline = InMemoryCohortBaseline(window="30d")
        baseline.extend(Platform.REDDIT, "text/plain", "endorsement", [1.0, 2.0, 3.0, 4.0])
        result = await baseline.percentile(
            platform=Platform.REDDIT, content_type="text/plain", axis="endorsement", value=3.0
        )
        assert result is not None
        assert result.value == pytest.approx(0.75)
        assert result.sample_size == 4
        assert result.is_provisional

    async def test_an_empty_cohort_returns_none_rather_than_a_default(self) -> None:
        """Implementations must not invent a value; see `ColdStartPolicy`."""
        baseline = InMemoryCohortBaseline()
        assert (
            await baseline.percentile(
                platform=Platform.REDDIT, content_type="text/plain", axis="reach", value=1.0
            )
            is None
        )

    def test_a_percentile_outside_the_unit_interval_is_rejected(self) -> None:
        """A percentile is a proportion; anything else is an implementation bug."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            CohortPercentile(value=1.5, sample_size=10, window="30d")

    async def test_the_corroboration_index_reports_unknown_clusters_as_unknown(self) -> None:
        """`None`, not a fabricated cluster of one, so the caller decides."""
        assert await InMemoryCorroborationIndex().lookup("dc_missing") is None

    def test_corroboration_saturation_is_the_documented_log_curve(self) -> None:
        """Asserted against the formula so a silent change to the shape fails here."""
        spread = ClusterCorroboration(members=4, independent_platforms=3)
        expected = math.log1p(3) / math.log1p(MAX_CORROBORATING_SOURCES)
        assert corroboration_of(spread) == pytest.approx(expected, abs=1e-6)
