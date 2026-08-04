"""Unit tests for `services/trend_service.py`.

The property that matters more than every other assertion in this file is the
counting rule from `docs/signal-model.md` §4.3: trend volume counts distinct
`(dedup_cluster_id, platform)` pairs, not Signals and not clusters. It is asserted
three ways -- over the pure bucketing function, over the detector, and end to end
through a real query against the in-memory database -- because each layer can
break it independently and only the last one is what production runs.

`test_six_reposts_on_one_platform_is_not_a_trend` is the named case: one outlet
publishing the same story six times must produce a volume of 1 and no burst. The
inverse, `test_six_platforms_carrying_one_story_is_a_trend`, is asserted in the
same breath, because a fix for the first that also collapses cross-platform
spread destroys exactly the evidence §4.3 says to keep.

The detector's own tests are written against *behaviour under contamination*
rather than against fixed z values: a real burst detector has to stay sensitive
in the days after a spike, has to abstain when it has no history, and must not be
willing to call a burst off a single mention. A fixed-threshold implementation
passes none of those.

No network, no broker, no container: the database is the in-memory SQLite engine
from `tests/conftest.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.exceptions import ValidationError
from models.enums import Platform, SignalStatus
from models.orm.signal import SignalRow
from services.trend_service import (
    BurstConfig,
    Mention,
    TrendDimension,
    TrendService,
    VolumePoint,
    acceleration,
    bucket_volume,
    build_series,
    detect_bursts,
    novelty,
    summarize,
    velocity,
)

pytestmark = pytest.mark.unit


WINDOW_START = datetime(2026, 6, 1, tzinfo=UTC)
DAY = timedelta(days=1)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def mention(
    *,
    signal_id: str,
    key: str = "observability-tooling",
    cluster_id: str = "dc_1",
    platform: Platform = Platform.RSS,
    day: int = 0,
    hour: int = 12,
) -> Mention:
    return Mention(
        signal_id=signal_id,
        key=key,
        cluster_id=cluster_id,
        platform=platform,
        at=WINDOW_START + timedelta(days=day, hours=hour),
    )


def point(volume: int, *, day: int = 0, platforms: int = 1) -> VolumePoint:
    """A `VolumePoint` for detector tests, where only volume and spread matter."""
    every = (
        Platform.RSS,
        Platform.REDDIT,
        Platform.X,
        Platform.YOUTUBE,
        Platform.LINKEDIN,
        Platform.GITHUB,
    )
    return VolumePoint(
        bucket_start=WINDOW_START + day * DAY,
        volume=volume,
        raw_signals=volume,
        clusters=volume,
        platforms=every[:platforms],
        signal_ids=(),
    )


def series_of(volumes: list[int], *, platforms: int = 2) -> tuple[VolumePoint, ...]:
    return tuple(point(v, day=i, platforms=platforms) for i, v in enumerate(volumes))


def signal_row(
    *,
    signal_id: str,
    platform: Platform,
    day: int,
    cluster_id: str | None = None,
    topics: list[str] | None = None,
    entities: list[dict[str, Any]] | None = None,
    status: SignalStatus = SignalStatus.ENRICHED,
    tenant_id: str = "default",
    hour: int = 12,
    duplicate_of: str | None = None,
) -> SignalRow:
    at = WINDOW_START + timedelta(days=day, hours=hour)
    return SignalRow(
        id=signal_id,
        tenant_id=tenant_id,
        native_id=f"native-{signal_id}",
        source="news",
        platform=platform,
        timestamp=at,
        fetched_at=at,
        connector_slug=platform.value,
        status=status,
        # `ck_signals_duplicate_points_elsewhere` requires a canonical member
        # whenever the status is `duplicate`, restating the invariant `Lineage`
        # enforces in Python.
        duplicate_of=duplicate_of
        or (f"canon_{cluster_id}" if status is SignalStatus.DUPLICATE else None),
        dedup_cluster_id=cluster_id,
        topics=[{"topic": t, "score": 0.9} for t in (topics or ["observability-tooling"])],
        entities=entities or [],
        lineage={},
    )


@pytest.fixture
def session_factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


async def insert(factory: async_sessionmaker[AsyncSession], rows: list[SignalRow]) -> None:
    async with factory() as session:
        session.add_all(rows)
        await session.commit()


# --------------------------------------------------------------------------- #
# §4.3 -- the counting rule
# --------------------------------------------------------------------------- #


def test_bucket_volume_counts_cluster_platform_pairs() -> None:
    """One cluster on one platform is one mention, however many Signals carry it."""
    six_reposts = [
        mention(signal_id=f"sig_{i}", cluster_id="dc_press_release", platform=Platform.RSS)
        for i in range(6)
    ]
    assert len(six_reposts) == 6
    assert bucket_volume(six_reposts) == 1


def test_bucket_volume_preserves_cross_platform_spread() -> None:
    """Six platforms carrying one story is six, because spread *is* the signal."""
    spread = [
        mention(signal_id=f"sig_{i}", cluster_id="dc_press_release", platform=platform)
        for i, platform in enumerate(
            [
                Platform.RSS,
                Platform.REDDIT,
                Platform.X,
                Platform.YOUTUBE,
                Platform.LINKEDIN,
                Platform.GITHUB,
            ]
        )
    ]
    assert bucket_volume(spread) == 6


def test_unclustered_signals_are_each_their_own_cluster() -> None:
    """Absent a cluster id, two independent posts must not collapse into one.

    The failure this guards is a shared sentinel (`None`, `""`) used as the
    cluster key, which would make the entire un-clustered corpus count as one.
    """
    independent = [
        mention(signal_id="sig_a", cluster_id="sig_a", platform=Platform.REDDIT),
        mention(signal_id="sig_b", cluster_id="sig_b", platform=Platform.REDDIT),
    ]
    assert bucket_volume(independent) == 2


def test_six_reposts_on_one_platform_is_not_a_trend() -> None:
    """THE case. One outlet re-posting one story six times is one observation.

    The series is silent for two weeks and then a single platform publishes the
    same clustered story six times in a day. A raw `COUNT(*)` sees a 0-to-6 jump
    against a zero baseline and calls a trend; the §4.3 rule sees a volume of 1.
    """
    reposts = [
        mention(
            signal_id=f"repost_{i}",
            cluster_id="dc_press_release",
            platform=Platform.RSS,
            day=14,
            hour=i,
        )
        for i in range(6)
    ]

    # Fourteen silent days precede the spike: the window is longer than the
    # mentions, and `build_series` materializes the empty buckets.
    series = build_series(
        reposts,
        key="observability-tooling",
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 15 * DAY,
        bucket=DAY,
    )

    spike = series.points[-1]
    assert spike.raw_signals == 6, "all six Signals must still be visible"
    assert spike.volume == 1, "but they are one cluster on one platform"
    assert spike.duplication_ratio == 6.0

    trend = summarize(series)
    assert trend.is_bursting is False
    assert trend.total_volume == 1
    assert [p.volume for p in series.points] == [0] * 14 + [1]


def test_six_platforms_carrying_one_story_is_a_trend() -> None:
    """The mirror case: the same cluster on six platforms must still burst.

    Asserted alongside the previous test on purpose. A "fix" that de-duplicates
    the cluster outright passes that test and fails this one, and it would delete
    exactly the cross-platform corroboration §4.3 exists to preserve.
    """
    platforms = [
        Platform.RSS,
        Platform.REDDIT,
        Platform.X,
        Platform.YOUTUBE,
        Platform.LINKEDIN,
        Platform.GITHUB,
    ]
    spread = [
        mention(
            signal_id=f"spread_{i}",
            cluster_id="dc_press_release",
            platform=platform,
            day=14,
        )
        for i, platform in enumerate(platforms)
    ]

    series = build_series(
        spread,
        key="observability-tooling",
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 15 * DAY,
        bucket=DAY,
    )
    assert series.points[-1].volume == 6
    assert series.points[-1].clusters == 1

    trend = summarize(series)
    assert trend.is_bursting is True
    assert trend.novelty == 1.0, "silent for the whole prior window"


# --------------------------------------------------------------------------- #
# Bucketing
# --------------------------------------------------------------------------- #


def test_empty_buckets_are_materialized() -> None:
    """A silent day is a zero, never a missing point.

    Omitting it would make every derivative below measure a longer interval than
    it claims, and would render a collapse as a flat line.
    """
    series = build_series(
        [mention(signal_id="a", day=0), mention(signal_id="b", cluster_id="dc_2", day=4)],
        key="k",
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 5 * DAY,
        bucket=DAY,
    )
    assert series.volumes == (1, 0, 0, 0, 1)


def test_mentions_outside_the_window_are_dropped_not_clamped() -> None:
    """Clamping would inflate the two buckets the detector is most sensitive to."""
    series = build_series(
        [
            mention(signal_id="before", cluster_id="dc_b", day=-3),
            mention(signal_id="inside", cluster_id="dc_i", day=1),
            mention(signal_id="after", cluster_id="dc_a", day=9),
        ],
        key="k",
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 5 * DAY,
        bucket=DAY,
    )
    assert series.volumes == (0, 1, 0, 0, 0)


def test_build_series_rejects_an_inverted_window() -> None:
    with pytest.raises(ValidationError):
        build_series(
            [],
            key="k",
            dimension=TrendDimension.TOPIC,
            window_start=WINDOW_START,
            window_end=WINDOW_START - DAY,
            bucket=DAY,
        )


def test_build_series_rejects_a_non_positive_bucket() -> None:
    with pytest.raises(ValidationError):
        build_series(
            [],
            key="k",
            dimension=TrendDimension.TOPIC,
            window_start=WINDOW_START,
            window_end=WINDOW_START + DAY,
            bucket=timedelta(0),
        )


# --------------------------------------------------------------------------- #
# Derivatives
# --------------------------------------------------------------------------- #


def test_velocity_is_the_backward_difference() -> None:
    assert velocity([1, 3, 6, 10]) == (2.0, 3.0, 4.0)


def test_acceleration_is_the_second_difference() -> None:
    assert acceleration([1, 3, 6, 10]) == (1.0, 1.0)


def test_acceleration_separates_still_growing_from_peaking() -> None:
    """Both series rise; only one is still accelerating. That sign is the story."""
    breaking = acceleration([1, 2, 4, 8])
    peaking = acceleration([1, 8, 12, 13])
    assert breaking[-1] > 0
    assert peaking[-1] < 0


def test_derivatives_of_a_degenerate_series_are_empty_not_an_error() -> None:
    assert velocity([]) == ()
    assert velocity([5]) == ()
    assert acceleration([5, 5]) == ()


# --------------------------------------------------------------------------- #
# Novelty
# --------------------------------------------------------------------------- #


def test_novelty_is_one_for_a_key_never_seen_before() -> None:
    assert novelty([0, 0, 0, 0, 9], 4) == 1.0
    assert novelty([9], 0) == 1.0


def test_novelty_is_low_for_a_recurring_spike() -> None:
    """A weekly spike is high velocity and low novelty -- `docs/glossary.md`."""
    weekly = [9, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 1, 9]
    assert novelty(weekly, 14) < 0.1


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #


def test_detector_abstains_without_enough_history() -> None:
    """Too little baseline is an abstention, never a burst."""
    verdicts = detect_bursts(series_of([0, 0, 40]))
    assert [v.determined for v in verdicts] == [False, False, False]
    assert not any(v.is_burst for v in verdicts)


def test_detector_fires_on_a_genuine_spike() -> None:
    verdicts = detect_bursts(series_of([2] * 14 + [40]))
    assert verdicts[-1].determined is True
    assert verdicts[-1].is_burst is True
    assert verdicts[-1].z_score > 3.0
    assert verdicts[-1].baseline_centre == 2.0


def test_detector_ignores_a_single_mention_against_silence() -> None:
    """0,0,...,0,1 is not a trend. Without the Poisson floor its z is unbounded."""
    verdicts = detect_bursts(series_of([0] * 14 + [1]))
    assert verdicts[-1].determined is True
    assert verdicts[-1].is_burst is False


def test_detector_scale_never_collapses_to_zero() -> None:
    """A perfectly flat baseline must not produce a division by zero or an inf."""
    verdicts = detect_bursts(series_of([7] * 14 + [7]))
    assert verdicts[-1].baseline_scale > 0.0
    assert verdicts[-1].z_score == 0.0


def test_detector_stays_sensitive_the_day_after_a_spike() -> None:
    """The reason the estimator is robust rather than mean/stdev.

    An escalation whose second day is larger than its first must still register.
    With a mean-and-standard-deviation baseline the first spike lifts the centre
    and inflates the spread, so the larger second day scores *lower* and a
    sustained escalation goes unreported from its second day onward.
    """
    escalating = series_of([2] * 14 + [40, 60])
    verdicts = detect_bursts(escalating)
    assert verdicts[-2].is_burst is True
    assert verdicts[-1].is_burst is True, "the larger second day must still fire"


def test_detector_respects_a_cross_platform_requirement() -> None:
    """`min_platforms` suppresses a single-platform spike when a caller asks it to."""
    single = tuple(point(v, day=i, platforms=1) for i, v in enumerate([2] * 14 + [40]))
    assert detect_bursts(single)[-1].is_burst is True
    assert detect_bursts(single, config=BurstConfig(min_platforms=2))[-1].is_burst is False


def test_burst_config_rejects_incoherent_tuning() -> None:
    with pytest.raises(ValidationError):
        BurstConfig(baseline_buckets=5, min_baseline_buckets=9)
    with pytest.raises(ValidationError):
        BurstConfig(min_volume=0)


def test_is_bursting_reports_the_latest_bucket_not_any_bucket() -> None:
    """A spike that already subsided is history, not a live trend."""
    subsided = build_series(
        [
            mention(signal_id=f"s{i}", cluster_id=f"dc_{i}", platform=Platform.REDDIT, day=14)
            for i in range(40)
        ],
        key="k",
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 30 * DAY,
        bucket=DAY,
    )
    trend = summarize(subsided)
    assert any(b.is_burst for b in trend.bursts)
    assert trend.is_bursting is False


# --------------------------------------------------------------------------- #
# The service, against the real query
# --------------------------------------------------------------------------- #


async def test_service_counts_six_reposts_as_one(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The §4.3 rule end to end, through the query production actually runs."""
    await insert(
        session_factory,
        [
            signal_row(
                signal_id=f"sig_repost_{i}",
                platform=Platform.RSS,
                day=14,
                hour=i,
                cluster_id="dc_press_release",
            )
            for i in range(6)
        ],
    )
    service = TrendService(session_factory)
    series = await service.volume_series(
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 15 * DAY,
        bucket=DAY,
    )
    assert len(series) == 1
    assert series[0].points[-1].raw_signals == 6
    assert series[0].points[-1].volume == 1

    trends = await service.detect(
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 15 * DAY,
    )
    assert trends[0].is_bursting is False


async def test_service_counts_cross_platform_spread(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    platforms = [
        Platform.RSS,
        Platform.REDDIT,
        Platform.X,
        Platform.YOUTUBE,
        Platform.LINKEDIN,
        Platform.GITHUB,
    ]
    await insert(
        session_factory,
        [
            signal_row(
                signal_id=f"sig_spread_{i}",
                platform=platform,
                day=14,
                cluster_id="dc_press_release",
            )
            for i, platform in enumerate(platforms)
        ],
    )
    service = TrendService(session_factory)
    trends = await service.detect(
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 15 * DAY,
    )
    assert trends[0].total_volume == 6
    assert trends[0].is_bursting is True
    assert len(trends[0].supporting_signal_ids) == 6


async def test_duplicate_signals_count_but_quarantined_do_not(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """`duplicate` is spread and counts; `quarantined` was withdrawn and does not."""
    await insert(
        session_factory,
        [
            signal_row(
                signal_id="sig_canonical",
                platform=Platform.RSS,
                day=1,
                cluster_id="dc_1",
            ),
            signal_row(
                signal_id="sig_dupe",
                platform=Platform.REDDIT,
                day=1,
                cluster_id="dc_1",
                status=SignalStatus.DUPLICATE,
            ),
            signal_row(
                signal_id="sig_quarantined",
                platform=Platform.X,
                day=1,
                cluster_id="dc_1",
                status=SignalStatus.QUARANTINED,
            ),
            signal_row(
                signal_id="sig_raw",
                platform=Platform.YOUTUBE,
                day=1,
                cluster_id="dc_1",
                status=SignalStatus.RAW,
            ),
        ],
    )
    service = TrendService(session_factory)
    series = await service.volume_series(
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 3 * DAY,
        bucket=DAY,
    )
    assert series[0].total_volume == 2


async def test_service_is_tenant_scoped(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await insert(
        session_factory,
        [
            signal_row(signal_id="sig_ours", platform=Platform.RSS, day=1, cluster_id="dc_a"),
            signal_row(
                signal_id="sig_theirs",
                platform=Platform.RSS,
                day=1,
                cluster_id="dc_b",
                tenant_id="other",
            ),
        ],
    )
    ours = TrendService(session_factory)
    theirs = TrendService(session_factory, tenant_id="other")
    window = {
        "dimension": TrendDimension.TOPIC,
        "window_start": WINDOW_START,
        "window_end": WINDOW_START + 3 * DAY,
        "bucket": DAY,
    }
    assert (await ours.volume_series(**window))[0].signal_ids == ("sig_ours",)
    assert (await theirs.volume_series(**window))[0].signal_ids == ("sig_theirs",)


async def test_service_filters_by_key_and_platform(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await insert(
        session_factory,
        [
            signal_row(
                signal_id="sig_a",
                platform=Platform.RSS,
                day=1,
                cluster_id="dc_a",
                topics=["vendor-pricing"],
            ),
            signal_row(
                signal_id="sig_b",
                platform=Platform.REDDIT,
                day=1,
                cluster_id="dc_b",
                topics=["vendor-pricing"],
            ),
            signal_row(
                signal_id="sig_c",
                platform=Platform.RSS,
                day=1,
                cluster_id="dc_c",
                topics=["something-else"],
            ),
        ],
    )
    service = TrendService(session_factory)
    window: dict[str, Any] = {
        "dimension": TrendDimension.TOPIC,
        "window_start": WINDOW_START,
        "window_end": WINDOW_START + 3 * DAY,
        "bucket": DAY,
    }

    by_key = await service.volume_series(keys=["vendor-pricing"], **window)
    assert [s.key for s in by_key] == ["vendor-pricing"]
    assert by_key[0].total_volume == 2

    by_platform = await service.volume_series(platforms=[Platform.RSS], **window)
    assert sorted(s.key for s in by_platform) == ["something-else", "vendor-pricing"]
    assert {s.key: s.total_volume for s in by_platform} == {
        "vendor-pricing": 1,
        "something-else": 1,
    }


async def test_entity_dimension_falls_back_to_the_surface_form(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An unresolved entity is disproportionately the new thing a trend is about."""
    await insert(
        session_factory,
        [
            signal_row(
                signal_id="sig_resolved",
                platform=Platform.RSS,
                day=1,
                cluster_id="dc_a",
                entities=[{"surface": "Datadog", "resolved_id": "ent_datadog"}],
            ),
            signal_row(
                signal_id="sig_unresolved",
                platform=Platform.REDDIT,
                day=1,
                cluster_id="dc_b",
                entities=[{"surface": "Grafana", "resolved_id": None}],
            ),
        ],
    )
    service = TrendService(session_factory)
    series = await service.volume_series(
        dimension=TrendDimension.ENTITY,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 3 * DAY,
        bucket=DAY,
    )
    assert sorted(s.key for s in series) == ["Grafana", "ent_datadog"]


async def test_one_signal_naming_an_entity_twice_counts_once(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A verbose post must not outweigh independent ones."""
    await insert(
        session_factory,
        [
            signal_row(
                signal_id="sig_verbose",
                platform=Platform.RSS,
                day=1,
                cluster_id="dc_a",
                entities=[
                    {"surface": "Datadog", "resolved_id": "ent_datadog"},
                    {"surface": "datadog", "resolved_id": "ent_datadog"},
                    {"surface": "DataDog", "resolved_id": "ent_datadog"},
                ],
            )
        ],
    )
    service = TrendService(session_factory)
    mentions = await service.mentions(
        dimension=TrendDimension.ENTITY,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 3 * DAY,
    )
    assert len(mentions) == 1


async def test_service_rejects_an_inverted_window(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    service = TrendService(session_factory)
    with pytest.raises(ValidationError):
        await service.mentions(
            dimension=TrendDimension.TOPIC,
            window_start=WINDOW_START,
            window_end=WINDOW_START,
        )


async def test_truncation_is_reported_rather_than_silent(
    session_factory: async_sessionmaker[AsyncSession],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A silently truncated read is a *wrong* trend, not a slow one."""
    await insert(
        session_factory,
        [
            signal_row(
                signal_id=f"sig_{i}",
                platform=Platform.RSS,
                day=1,
                hour=i,
                cluster_id=f"dc_{i}",
            )
            for i in range(5)
        ],
    )
    service = TrendService(session_factory, max_rows=2)
    capsys.readouterr()  # discard anything emitted while inserting
    mentions = await service.mentions(
        dimension=TrendDimension.TOPIC,
        window_start=WINDOW_START,
        window_end=WINDOW_START + 3 * DAY,
    )
    assert len(mentions) == 2
    # structlog renders to stdout rather than through the stdlib `logging`
    # handlers `caplog` installs, so the assertion reads the rendered line.
    assert "trend_service.window_truncated" in capsys.readouterr().out
