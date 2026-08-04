"""Trend detection: burst detection, velocity and novelty scoring.

A Trend in OmniSense is a *measurement of the past* (`docs/glossary.md`): volume
observed over a historical window, its first derivative (velocity), its second
(acceleration), and whether the most recent buckets are statistically abnormal
against what came before. Nothing here projects forward -- that is
`services/forecast_service.py`, and the two are kept apart deliberately because a
trend can only be wrong if the data is wrong, whereas a forecast can be wrong
with perfect data.

The correctness property of this module is *what gets counted*
------------------------------------------------------------
`docs/signal-model.md` §4.3 is unambiguous and it is the reason this module
exists as more than a `GROUP BY`:

    All six [near-duplicate Signals] count toward trend volume, with per-platform
    deduplication so one platform cannot inflate a cluster by itself.

So the counting unit is neither the Signal nor the dedup cluster. It is the
**distinct `(dedup_cluster_id, platform)` pair**. Three consequences, each of
which a naive `COUNT(*)` gets wrong:

- One outlet posting the same story six times contributes **1**, not 6. A
  scheduled re-post loop, an RSS feed that re-publishes on every edit, and a
  brigading account are all the same shape, and all three would otherwise
  manufacture a trend out of a single observation. This is the single most
  important behaviour in the module and it is asserted explicitly in
  `tests/unit/services/test_trend_service.py`.
- The same story appearing on six *different* platforms contributes **6**,
  because cross-platform spread is exactly the evidence a trend is made of.
  Collapsing the cluster to 1 would destroy the signal §4.3 says to preserve.
- `status = 'duplicate'` Signals are counted. They are excluded from *retrieval*
  because the canonical member answers instead, but they are the cluster's
  spread and dropping them here would silently re-introduce the first bug.

Why the arithmetic is separated from the query
----------------------------------------------
`bucket_volume`, `velocity`, `acceleration`, `detect_bursts` and `novelty` are
module-level functions over plain sequences, and `TrendService` is a thin layer
that turns rows into `Mention`s and calls them. That split is what lets the
counting property be tested exhaustively without a database, and it is what
allows the Layer 5 Trend agent to re-run the detector over a series it already
holds without a second round trip.

Why the detector is not a threshold
-----------------------------------
"More than N mentions in a day" is not a detector; it is a constant that is wrong
for every topic except the one it was tuned on. A topic averaging 400 mentions a
day and a topic averaging 2 need different alarms, and neither wants to be woken
by seasonality it has shown every week for a year. `detect_bursts` scores each
bucket against its own trailing baseline with a **robust z-score** -- median and
a MAD-derived scale rather than mean and standard deviation, because the mean of
a window containing yesterday's spike is dragged toward the spike and the
standard deviation inflates with it, which is precisely how a real burst hides a
following one.

The one place a constant survives is `BurstConfig.min_volume`, and it is a floor
rather than the decision: count data is sparse, a baseline of all zeros has zero
dispersion, and without a floor a series going 0,0,0,1 scores an unbounded z and
declares a trend from a single post.

Layer note: `services/` (L2). Takes its session factory as a constructor argument
and constructs none, so the unit suite runs the real query against in-memory
SQLite with nothing else in the process.
"""

from __future__ import annotations

import enum
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger
from models.enums import Platform, SignalStatus
from models.orm.mixins import DEFAULT_TENANT
from models.orm.signal import SignalRow

__all__ = [
    "COUNTED_STATUSES",
    "MAX_MENTION_ROWS",
    "BurstConfig",
    "BurstPoint",
    "Mention",
    "Trend",
    "TrendDimension",
    "TrendService",
    "VolumePoint",
    "VolumeSeries",
    "acceleration",
    "bucket_volume",
    "build_series",
    "detect_bursts",
    "novelty",
    "summarize",
    "velocity",
]

logger = get_logger(__name__)


COUNTED_STATUSES: Final[frozenset[SignalStatus]] = frozenset(
    {SignalStatus.ENRICHED, SignalStatus.PARTIAL, SignalStatus.DUPLICATE}
)
"""Signal states that contribute to trend volume (`docs/signal-model.md` §4.3, §5.4).

`DUPLICATE` is in the set on purpose. Those Signals are not *retrievable* -- the
canonical cluster member is returned instead -- but they are the record of the
story spreading, and excluding them would collapse a six-platform cluster to one
mention. `RAW` is excluded because enrichment has not run, so it carries no
topics or entities to be counted under; `QUARANTINED` is excluded because it was
withdrawn.
"""

MAX_MENTION_ROWS: Final[int] = 200_000
"""Hard cap on rows pulled into one detection pass.

The topic and entity predicates live in JSON columns, and neither PostgreSQL nor
SQLite can serve them from an index as the schema stands today, so filtering
happens in Python after a windowed read. That is honest for Phase 1 volumes and
catastrophic without a ceiling: an unbounded window over a full corpus would pull
the whole `signals` table into one process. The cap is enforced in SQL and a
truncated read is logged loudly, because a silently truncated trend is a *wrong*
trend rather than a slow one.
"""

_ROBUST_SCALE_FACTOR: Final[float] = 1.4826
"""MAD -> standard-deviation conversion for a normal distribution.

`1.4826 * MAD` estimates sigma with a breakdown point of 50%: half the baseline
buckets can be spikes before the estimate moves. The sample standard deviation's
breakdown point is zero, which is why it is not used here.
"""

_SCALE_EPSILON: Final[float] = 1e-9


class TrendDimension(enum.StrEnum):
    """What a series is keyed by.

    Not in `models/enums.py`: this never crosses a process boundary, is never
    serialized into a Kafka envelope or a Qdrant payload, and adding it to the
    shared vocabulary would grow the cross-service contract to describe one
    module's dispatch.
    """

    TOPIC = "topic"
    ENTITY = "entity"


# --------------------------------------------------------------------------- #
# The counting unit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Mention:
    """One Signal's contribution to one key's volume, reduced to what counting needs.

    Frozen and hashable so a bucket is a `set` operation rather than a manual
    de-duplication loop, which is the kind of loop that grows an off-by-one and
    quietly re-inflates a cluster.
    """

    signal_id: str
    key: str
    cluster_id: str
    platform: Platform
    at: datetime

    @property
    def counting_key(self) -> tuple[str, Platform]:
        """The pair volume is counted over: one cluster, once per platform.

        This property *is* `docs/signal-model.md` §4.3. Everything else in this
        module is arithmetic on top of it.
        """
        return (self.cluster_id, self.platform)


@dataclass(frozen=True, slots=True)
class VolumePoint:
    """One time bucket of one series."""

    bucket_start: datetime

    volume: int
    """Distinct `(cluster, platform)` pairs. The number every derivative uses."""

    raw_signals: int
    """Signals before de-duplication. Carried so callers can *see* the collapse:
    a bucket with `raw_signals=6, volume=1` is a repost loop, and a UI that shows
    only the counted volume gives an analyst no way to notice one."""

    clusters: int
    """Distinct clusters ignoring platform. `volume - clusters` is the spread."""

    platforms: tuple[Platform, ...]
    signal_ids: tuple[str, ...]

    @property
    def duplication_ratio(self) -> float:
        """`raw_signals / volume`, or 0.0 for an empty bucket.

        A ratio far above 1 means most of the traffic in this bucket was the same
        story repeated on the same platform -- the exact condition that makes a
        raw count lie.
        """
        if self.volume == 0:
            return 0.0
        return self.raw_signals / self.volume


@dataclass(frozen=True, slots=True)
class VolumeSeries:
    """A contiguous, evenly spaced volume history for one key.

    Contiguous matters: empty buckets are materialized with `volume=0` rather
    than omitted. A series that skips its quiet days makes every derivative below
    measure the wrong interval, and makes a topic that went silent look flat
    instead of collapsing.
    """

    key: str
    dimension: TrendDimension
    bucket: timedelta
    window_start: datetime
    window_end: datetime
    points: tuple[VolumePoint, ...]

    @property
    def volumes(self) -> tuple[int, ...]:
        return tuple(point.volume for point in self.points)

    @property
    def total_volume(self) -> int:
        return sum(point.volume for point in self.points)

    @property
    def platforms(self) -> tuple[Platform, ...]:
        seen: dict[Platform, None] = {}
        for point in self.points:
            for platform in point.platforms:
                seen.setdefault(platform, None)
        return tuple(seen)

    @property
    def signal_ids(self) -> tuple[str, ...]:
        return tuple(sid for point in self.points for sid in point.signal_ids)

    def velocity(self) -> tuple[float, ...]:
        return velocity(self.volumes)

    def acceleration(self) -> tuple[float, ...]:
        return acceleration(self.volumes)

    @property
    def latest_velocity(self) -> float:
        series = self.velocity()
        return series[-1] if series else 0.0

    @property
    def latest_acceleration(self) -> float:
        series = self.acceleration()
        return series[-1] if series else 0.0


# --------------------------------------------------------------------------- #
# Burst detection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BurstConfig:
    """Tuning for `detect_bursts`. Every field changes a documented failure mode."""

    baseline_buckets: int = 14
    """How far back the trailing baseline reaches. Two weeks of daily buckets is
    long enough to contain a weekly cycle -- shorter and every Monday is a burst."""

    min_baseline_buckets: int = 7
    """Below this the bucket is scored `undetermined` rather than compared.

    A z-score against three observations is arithmetic, not evidence: MAD over
    such a window is dominated by whichever single value happens to be there, and
    the first days of any new topic would all read as bursts."""

    z_threshold: float = 3.0
    """Robust z at which a bucket is called a burst."""

    min_volume: int = 3
    """Absolute floor, not the decision.

    Count series are sparse and a baseline of zeros has no dispersion at all, so
    the Poisson fallback in `_robust_scale` gives a scale of 1.0 and a pair of
    mentions against a silent history would otherwise be enough to declare a
    trend. Requiring a handful of independent cluster-platform pairs is what
    keeps "somebody mentioned it twice" out of a report."""

    min_platforms: int = 1
    """Distinct platforms required in the bursting bucket.

    Left at 1 by default: a genuine Reddit-only story is a real trend and a
    cross-platform requirement would suppress it. Raised to 2 by a caller that
    only wants corroborated spread, which is a policy choice and therefore the
    caller's."""

    def __post_init__(self) -> None:
        if self.baseline_buckets < 1:
            raise ValidationError("baseline_buckets must be at least 1")
        if not 1 <= self.min_baseline_buckets <= self.baseline_buckets:
            raise ValidationError(
                "min_baseline_buckets must be between 1 and baseline_buckets "
                f"({self.baseline_buckets}); got {self.min_baseline_buckets}"
            )
        if self.min_volume < 1:
            raise ValidationError("min_volume must be at least 1")
        if self.min_platforms < 1:
            raise ValidationError("min_platforms must be at least 1")


@dataclass(frozen=True, slots=True)
class BurstPoint:
    """The detector's verdict on one bucket, with the numbers behind it.

    The baseline is reported alongside the verdict because "z = 4.2" is not
    reviewable on its own -- `docs/frontend.md` §6 asks for "the baseline the
    velocity is measured against", and an analyst cannot sanity-check a claim
    whose comparison set is hidden.
    """

    bucket_start: datetime
    volume: int
    baseline_centre: float
    baseline_scale: float
    z_score: float
    is_burst: bool

    determined: bool
    """False when the trailing baseline was too short to score against. Distinct
    from `is_burst=False`, which is a decision; this one is an abstention."""


@dataclass(frozen=True, slots=True)
class Trend:
    """One key's measured behaviour over one window. The Layer 5 Trend agent's unit."""

    key: str
    dimension: TrendDimension
    window_start: datetime
    window_end: datetime
    bucket: timedelta
    total_volume: int
    raw_signals: int
    velocity: float
    acceleration: float
    novelty: float
    peak_z: float
    bursts: tuple[BurstPoint, ...]
    platforms: tuple[Platform, ...]
    supporting_signal_ids: tuple[str, ...]

    @property
    def is_bursting(self) -> bool:
        """Whether the *most recent* determined bucket is a burst.

        Deliberately not "any bucket in the window burst": a spike three weeks ago
        that has since subsided is history, and surfacing it as a live trend is
        how a dashboard tells a user to act on something that already ended.
        """
        for point in reversed(self.bursts):
            if point.determined:
                return point.is_burst
        return False

    @property
    def duplication_ratio(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return self.raw_signals / self.total_volume


# --------------------------------------------------------------------------- #
# Arithmetic -- pure functions over sequences, no database in sight
# --------------------------------------------------------------------------- #


def bucket_volume(mentions: Iterable[Mention]) -> int:
    """Volume of a set of mentions: distinct `(cluster, platform)` pairs.

    The whole §4.3 rule, in one line, so that there is exactly one place it can
    be got wrong.
    """
    return len({mention.counting_key for mention in mentions})


def velocity(volumes: Sequence[float | int]) -> tuple[float, ...]:
    """First discrete derivative, in mentions per bucket.

    Backward differences (`v[i] - v[i-1]`), not centred: a centred difference at
    the last bucket needs a bucket that has not happened yet, and the last bucket
    is the only one anybody is asking about. The returned series is one shorter
    than the input and aligns with `volumes[1:]`.
    """
    return tuple(float(volumes[i] - volumes[i - 1]) for i in range(1, len(volumes)))


def acceleration(volumes: Sequence[float | int]) -> tuple[float, ...]:
    """Second discrete derivative, in mentions per bucket squared.

    The sign is what distinguishes "growing" from "still growing but slowing",
    which is the difference between a story that is breaking and one that has
    peaked. Aligns with `volumes[2:]`.
    """
    return velocity(velocity(volumes))


def novelty(volumes: Sequence[int], at_index: int) -> float:
    """How new this key is at `at_index`, in `[0, 1]`.

    Defined as the share of prior buckets in which the key was completely silent.
    A key first seen at `at_index` scores 1.0; a key that spikes every Monday
    scores low however large the spike, which is exactly the distinction
    `docs/glossary.md` asks for -- "a topic can have high velocity and low novelty
    (a recurring spike) or the reverse".

    Deliberately not "1 - volume share": a topic whose history is one enormous
    day and thirty quiet ones would score as familiar under a share-based
    definition even though it has been silent all month.
    """
    if at_index <= 0:
        return 1.0
    history = volumes[:at_index]
    silent = sum(1 for value in history if value == 0)
    return silent / len(history)


def detect_bursts(
    points: Sequence[VolumePoint],
    *,
    config: BurstConfig | None = None,
) -> tuple[BurstPoint, ...]:
    """Score every bucket against its own trailing baseline.

    The estimator is a robust z-score: the baseline's *median* is the centre and
    `1.4826 * MAD` is the scale. Both choices are about the same failure -- a
    spike contaminating the baseline that follows it. With mean and standard
    deviation, one large bucket lifts the centre and inflates the spread, so the
    genuinely larger bucket a day later scores *lower*; a sustained escalation
    then reads as normal from its second day onward.

    Two fallbacks are load-bearing:

    - **Zero dispersion.** A flat or all-zero baseline gives MAD = 0 and an
      infinite z for any movement at all. Counts are Poisson-ish, so the scale
      falls back to `sqrt(max(centre, 1))` -- the standard deviation a Poisson
      process with that rate would have. A quiet topic then needs several
      independent mentions to clear the threshold rather than one.
    - **Short history.** Fewer than `min_baseline_buckets` prior buckets is an
      abstention (`determined=False`), never a burst. Scoring a topic against its
      own first three days is how every new topic becomes a trend.
    """
    config = config or BurstConfig()
    volumes = [point.volume for point in points]
    verdicts: list[BurstPoint] = []

    for index, point in enumerate(points):
        window = volumes[max(0, index - config.baseline_buckets) : index]
        if len(window) < config.min_baseline_buckets:
            verdicts.append(
                BurstPoint(
                    bucket_start=point.bucket_start,
                    volume=point.volume,
                    baseline_centre=statistics.fmean(window) if window else 0.0,
                    baseline_scale=0.0,
                    z_score=0.0,
                    is_burst=False,
                    determined=False,
                )
            )
            continue

        centre = float(statistics.median(window))
        scale = _robust_scale(window, centre)
        z_score = (point.volume - centre) / scale
        is_burst = (
            z_score >= config.z_threshold
            and point.volume >= config.min_volume
            and len(point.platforms) >= config.min_platforms
        )
        verdicts.append(
            BurstPoint(
                bucket_start=point.bucket_start,
                volume=point.volume,
                baseline_centre=centre,
                baseline_scale=scale,
                z_score=z_score,
                is_burst=is_burst,
                determined=True,
            )
        )

    return tuple(verdicts)


def _robust_scale(window: Sequence[int], centre: float) -> float:
    """MAD-derived sigma, with a Poisson floor so sparse counts stay finite."""
    mad = statistics.median([abs(value - centre) for value in window])
    scale = _ROBUST_SCALE_FACTOR * mad
    if scale > _SCALE_EPSILON:
        return scale
    # Every baseline bucket held (nearly) the same value. That is not evidence of
    # zero variance, it is evidence that the sample is too coarse to show any --
    # so fall back to the dispersion a Poisson process at this rate would have.
    return math.sqrt(max(centre, 1.0))


def build_series(
    mentions: Iterable[Mention],
    *,
    key: str,
    dimension: TrendDimension,
    window_start: datetime,
    window_end: datetime,
    bucket: timedelta,
) -> VolumeSeries:
    """Bucket mentions into a contiguous series, de-duplicating per platform.

    Buckets are aligned to `window_start`, not to the Unix epoch. The tradeoff is
    explicit: epoch alignment would let two overlapping queries share bucket
    boundaries, while window alignment makes one query's output depend only on
    its own arguments. The second matters more here because a Trend is reported
    with its window attached, and a caller that asks for "the last 30 days" and
    gets a first bucket starting before that window has to explain a partial
    bucket it did not ask for.

    Mentions outside `[window_start, window_end)` are dropped rather than clamped
    into the edge buckets, which would inflate exactly the two buckets a burst
    detector is most sensitive to.
    """
    if bucket <= timedelta(0):
        raise ValidationError("bucket must be a positive duration")
    if window_end <= window_start:
        raise ValidationError("window_end must be after window_start")

    span = window_end - window_start
    bucket_count = math.ceil(span / bucket)
    grouped: dict[int, list[Mention]] = defaultdict(list)

    for mention in mentions:
        at = _as_utc(mention.at)
        if at < window_start or at >= window_end:
            continue
        # `//` on timedeltas floors, and `at` is strictly inside the window, so
        # the index is always in `[0, bucket_count)`.
        grouped[int((at - window_start) // bucket)].append(mention)

    points: list[VolumePoint] = []
    for index in range(bucket_count):
        members = grouped.get(index, [])
        counted = {mention.counting_key for mention in members}
        points.append(
            VolumePoint(
                bucket_start=window_start + index * bucket,
                volume=len(counted),
                raw_signals=len(members),
                clusters=len({mention.cluster_id for mention in members}),
                platforms=tuple(dict.fromkeys(mention.platform for mention in members)),
                signal_ids=tuple(mention.signal_id for mention in members),
            )
        )

    return VolumeSeries(
        key=key,
        dimension=dimension,
        bucket=bucket,
        window_start=window_start,
        window_end=window_end,
        points=tuple(points),
    )


def summarize(series: VolumeSeries, *, config: BurstConfig | None = None) -> Trend:
    """Turn one series into a `Trend`: derivatives, novelty and burst verdicts."""
    bursts = detect_bursts(series.points, config=config)
    volumes = series.volumes

    determined = [point for point in bursts if point.determined]
    peak_z = max((point.z_score for point in determined), default=0.0)

    # Novelty is measured at the most recent burst if there is one, and at the
    # end of the window otherwise. Measuring it at the peak is what makes "this
    # spiked out of nowhere" and "this spikes every week" different numbers.
    novelty_index = len(volumes) - 1
    for index in range(len(bursts) - 1, -1, -1):
        if bursts[index].is_burst:
            novelty_index = index
            break

    return Trend(
        key=series.key,
        dimension=series.dimension,
        window_start=series.window_start,
        window_end=series.window_end,
        bucket=series.bucket,
        total_volume=series.total_volume,
        raw_signals=sum(point.raw_signals for point in series.points),
        velocity=series.latest_velocity,
        acceleration=series.latest_acceleration,
        novelty=novelty(volumes, max(novelty_index, 0)),
        peak_z=peak_z,
        bursts=bursts,
        platforms=series.platforms,
        supporting_signal_ids=series.signal_ids,
    )


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class TrendService:
    """Volume, velocity, acceleration and bursts over the `signals` table.

    Stateless per call. Takes a session factory rather than a session so that one
    instance can be shared by an API process and a worker, both of which drive it
    concurrently.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
        config: BurstConfig | None = None,
        max_rows: int = MAX_MENTION_ROWS,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id
        self._config = config or BurstConfig()
        self._max_rows = max_rows

    @property
    def config(self) -> BurstConfig:
        return self._config

    async def mentions(
        self,
        *,
        dimension: TrendDimension,
        window_start: datetime,
        window_end: datetime,
        keys: Sequence[str] | None = None,
        platforms: Sequence[Platform] | None = None,
    ) -> list[Mention]:
        """Read the window and expand each Signal into one `Mention` per key.

        One Signal mentioning three topics contributes to three series. That is
        not double counting: the question "how much was topic X discussed" is
        answered per topic, and a Signal covering three topics genuinely is an
        observation of each.

        The key predicate is applied in Python. `signals.topics` and
        `signals.entities` are JSON arrays and neither dialect can serve a
        containment predicate from an index as the schema stands, so pushing it
        into SQL would buy a sequential scan with a JSON function on top. The
        time, tenant and status predicates *are* pushed down, because those are
        indexed and they are what bounds the read.
        """
        window_start = _as_utc(window_start)
        window_end = _as_utc(window_end)
        if window_end <= window_start:
            raise ValidationError("window_end must be after window_start")

        wanted = {key.casefold() for key in keys} if keys else None

        statement = (
            select(
                SignalRow.id,
                SignalRow.platform,
                SignalRow.timestamp,
                SignalRow.dedup_cluster_id,
                SignalRow.topics,
                SignalRow.entities,
            )
            .where(
                SignalRow.tenant_id == self._tenant_id,
                SignalRow.timestamp >= window_start,
                SignalRow.timestamp < window_end,
                SignalRow.status.in_(tuple(COUNTED_STATUSES)),
            )
            .order_by(SignalRow.timestamp)
            .limit(self._max_rows + 1)
        )
        if platforms:
            statement = statement.where(SignalRow.platform.in_(tuple(platforms)))

        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()

        if len(rows) > self._max_rows:
            # Truncation makes every number below an undercount, and an
            # undercount nobody was told about is a wrong trend rather than a
            # slow one. Reported here; the caller narrows the window.
            logger.warning(
                "trend_service.window_truncated",
                max_rows=self._max_rows,
                window_start=window_start.isoformat(),
                window_end=window_end.isoformat(),
                dimension=dimension.value,
            )
            rows = rows[: self._max_rows]

        collected: list[Mention] = []
        for row in rows:
            signal_id = str(row.id)
            for key in _keys_for(dimension, row.topics, row.entities):
                if wanted is not None and key.casefold() not in wanted:
                    continue
                collected.append(
                    Mention(
                        signal_id=signal_id,
                        key=key,
                        # A Signal with no cluster is its own cluster of one.
                        # Falling back to the Signal id rather than to a shared
                        # sentinel is what stops every un-clustered Signal in the
                        # corpus from collapsing into a single counted mention.
                        cluster_id=row.dedup_cluster_id or signal_id,
                        platform=_as_platform(row.platform),
                        at=_as_utc(row.timestamp),
                    )
                )
        return collected

    async def volume_series(
        self,
        *,
        dimension: TrendDimension,
        window_start: datetime,
        window_end: datetime,
        bucket: timedelta,
        keys: Sequence[str] | None = None,
        platforms: Sequence[Platform] | None = None,
    ) -> list[VolumeSeries]:
        """One contiguous series per key, busiest first."""
        collected = await self.mentions(
            dimension=dimension,
            window_start=window_start,
            window_end=window_end,
            keys=keys,
            platforms=platforms,
        )
        by_key: dict[str, list[Mention]] = defaultdict(list)
        for mention in collected:
            by_key[mention.key].append(mention)

        series = [
            build_series(
                members,
                key=key,
                dimension=dimension,
                window_start=_as_utc(window_start),
                window_end=_as_utc(window_end),
                bucket=bucket,
            )
            for key, members in by_key.items()
        ]
        series.sort(key=lambda item: (-item.total_volume, item.key))
        return series

    async def detect(
        self,
        *,
        dimension: TrendDimension,
        window_start: datetime,
        window_end: datetime,
        bucket: timedelta = timedelta(days=1),
        keys: Sequence[str] | None = None,
        platforms: Sequence[Platform] | None = None,
        config: BurstConfig | None = None,
        bursting_only: bool = False,
    ) -> list[Trend]:
        """Measure every key in the window and rank it.

        Ranking is by peak robust z, then by volume, then by key. Volume alone
        would put a large, flat, permanently-busy topic above the small one that
        just tripled, and the second is what a trend endpoint exists to surface.
        """
        every = await self.volume_series(
            dimension=dimension,
            window_start=window_start,
            window_end=window_end,
            bucket=bucket,
            keys=keys,
            platforms=platforms,
        )
        trends = [summarize(series, config=config or self._config) for series in every]
        if bursting_only:
            trends = [trend for trend in trends if trend.is_bursting]
        trends.sort(key=lambda trend: (-trend.peak_z, -trend.total_volume, trend.key))
        return trends


# --------------------------------------------------------------------------- #
# Row -> key extraction
# --------------------------------------------------------------------------- #


def _keys_for(
    dimension: TrendDimension,
    topics: Sequence[Mapping[str, Any]] | None,
    entities: Sequence[Mapping[str, Any]] | None,
) -> list[str]:
    """Pull the series keys out of a Signal's JSON columns.

    Deduplicated within the row: a Signal naming the same entity four times is
    still one observation of it, and counting each surface form would let a
    single verbose post outweigh four independent ones.
    """
    if dimension is TrendDimension.TOPIC:
        source: Sequence[Mapping[str, Any]] = topics or ()
        field = "topic"
    else:
        source = entities or ()
        field = "resolved_id"

    seen: dict[str, None] = {}
    for item in source:
        if not isinstance(item, Mapping):
            continue
        value = item.get(field)
        if not value and dimension is TrendDimension.ENTITY:
            # An unresolved entity still has a surface form, and dropping it
            # would make trends blind to anything the linker has not seen yet --
            # which is disproportionately the new things a trend is about.
            value = item.get("surface")
        if isinstance(value, str) and value.strip():
            seen.setdefault(value.strip(), None)
    return list(seen)


def _as_platform(value: Platform | str) -> Platform:
    """Rows read back through `TolerantEnumType` are already `Platform`; be safe."""
    return value if isinstance(value, Platform) else Platform(value)


def _as_utc(value: datetime) -> datetime:
    """Normalize a database timestamp to aware UTC.

    PostgreSQL returns `timestamptz` as an aware datetime; SQLite has no such
    type and SQLAlchemy hands back a naive one even though the column is declared
    `DateTime(timezone=True)`. The value written was UTC either way, so attaching
    UTC is a restoration rather than a guess -- and without it every comparison
    in this module raises `TypeError` on the unit suite's database but not on the
    production one, which is the worst possible place to find that out.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
