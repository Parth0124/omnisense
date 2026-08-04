"""Quantitative tools: time series, aggregation, forecasting, description.

These four tools exist so that **no number in a report was written by a model**.
`docs/agent-system.md` §5.4 and §5.6 both state the rule from the failure side:
the Trend agent must not narrate noise as a trend, and the Forecast agent must
not produce point estimates. Enforcing that by asking the prompt nicely does not
survive contact with a model that is confident and wrong, so it is enforced by
the *shape of the arguments* instead.

That is the one design decision worth explaining here. Every tool below is
selected by a **series identity** -- dimension, key, window, bucket -- and never
by a list of values. An agent cannot hand `fit_forecast` the numbers it would
like forecast; it names a series, and the tool reads that series from the store
itself. The Critic can then check §5.6 mechanically: every figure in a forecast
has to match a recorded tool result, and there is no argument shape through which
a hallucinated history could have entered.

The second decision is that an over-wide request is **refused, not truncated**.
`timeseries` over a year at hourly buckets is 8,760 points; returning the first
180 of them would be a *wrong* series rather than a big one -- every derivative
below measures the wrong interval, and a topic that went quiet looks flat instead
of collapsing. `services/trend_service.py::MAX_MENTION_ROWS` makes the same call
for the same reason. Truncation is right for a ranked list, where the tail is the
part that mattered least; it is never right for a time series.

Both services are bound here and nowhere else: `services/trend_service.py` for
volume, velocity, acceleration and bursts, and `services/forecast_service.py` for
projections. Statsmodels, numpy and pandas stay behind that boundary -- an agent
never sees a fitted model, only its result and the caveats that came with it.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Final

from pydantic import Field, model_validator

from agents.tools.registry import BoundedResult, ProvenanceStr, ToolSpec
from backend.core.exceptions import ConfigurationError, ValidationError
from backend.core.logging import get_logger
from models.base import StrictModel, utcnow
from models.enums import Platform
from services.forecast_service import (
    ForecastMethod,
    ForecastService,
    observations_from_buckets,
)
from services.trend_service import (
    Mention,
    TrendDimension,
    TrendService,
    VolumeSeries,
    bucket_volume,
    summarize,
)

__all__ = [
    "MAX_GROUPS",
    "MAX_HORIZON",
    "MAX_POINTS",
    "MAX_SERIES",
    "AggregateInput",
    "AggregateResult",
    "AnalyticsToolset",
    "DescribeInput",
    "DescribeResult",
    "FitForecastInput",
    "ForecastResult",
    "GroupCount",
    "SeriesSelector",
    "TimeseriesInput",
    "TimeseriesResult",
]

logger = get_logger(__name__)

MAX_POINTS: Final = 180
"""Buckets one series may contain.

Half a year of daily buckets, or a month of four-hourly. Beyond this the request
is refused rather than shortened -- see the module docstring. It is also roughly
where a series stops being readable by the model consuming it: 180 integers is a
paragraph, 8,760 is a file.
"""

MAX_SERIES: Final = 5
MAX_GROUPS: Final = 25
MAX_HORIZON: Final = 12
"""Buckets a forecast may project.

Twelve, because `ForecastConfig.min_observations` is twelve: projecting further
than the history is long is extrapolation dressed as a forecast, and the interval
that would honestly accompany it is too wide to act on.
"""

MAX_WINDOW_DAYS: Final = 365
MAX_KEY_CHARS: Final = 120


# --------------------------------------------------------------------------- #
# Series identity -- the shape that keeps numbers out of arguments
# --------------------------------------------------------------------------- #


class SeriesSelector(StrictModel):
    """Which series a tool should read. Never the series itself.

    `window_days` / `bucket_hours` rather than two timestamps: an agent has no
    reliable clock, and a model that has to invent "now" invents a plausible
    date. The toolset's clock resolves the window, so two tools called in one
    node measure the same interval.
    """

    dimension: TrendDimension = TrendDimension.TOPIC
    window_days: int = Field(default=30, ge=1, le=MAX_WINDOW_DAYS)
    bucket_hours: int = Field(default=24, ge=1, le=168)
    platforms: list[Platform] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def _bounded_point_count(self) -> SeriesSelector:
        buckets = (self.window_days * 24) // self.bucket_hours
        if buckets > MAX_POINTS:
            raise ValueError(
                f"{self.window_days}d at {self.bucket_hours}h buckets is {buckets} points, "
                f"above the {MAX_POINTS} ceiling. Widen bucket_hours or shorten "
                "window_days -- a truncated time series measures the wrong interval."
            )
        return self

    @property
    def bucket(self) -> timedelta:
        return timedelta(hours=self.bucket_hours)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class TimeseriesInput(SeriesSelector):
    keys: list[str] = Field(min_length=1, max_length=MAX_SERIES)
    """Topics or entities to measure. Required: an unkeyed request would read the
    whole window for every key in the corpus, which is the shape of an accidental
    full-table scan."""


class SeriesPoint(StrictModel):
    bucket_start: datetime
    volume: int
    raw_signals: int
    """Signals before de-duplication.

    Carried so the collapse is visible: `raw_signals=6, volume=1` is a repost
    loop, and a series showing only counted volume gives the agent no way to
    notice one before calling it a trend.
    """

    clusters: int


class SeriesOut(StrictModel):
    key: ProvenanceStr
    """A topic or entity string extracted from ingested text, so third-party and
    scrubbed by its type -- a topic named "ignore previous instructions" is a
    thing a corpus can contain."""

    dimension: TrendDimension
    bucket_hours: int
    window_start: datetime
    window_end: datetime
    points: list[SeriesPoint] = Field(default_factory=list, max_length=MAX_POINTS)
    total_volume: int = 0

    # -- the statistical qualifiers ------------------------------------------
    latest_velocity: float = 0.0
    latest_acceleration: float = 0.0
    peak_z: float = 0.0
    novelty: float = 0.0
    is_bursting: bool = False
    """Whether the most recent determined bucket is a burst.

    `docs/agent-system.md` §5.4 requires every emitted Trend to carry a
    statistical qualifier "from the tool output, not from the model's prose".
    This is that qualifier, computed by `services/trend_service.py` under a
    documented robust-z rule, so "emerging" in a report is checkable against a
    number rather than being a word the model chose.
    """


class TimeseriesResult(BoundedResult):
    ITEMS_FIELD = "series"

    series: list[SeriesOut] = Field(default_factory=list)


class AggregateInput(SeriesSelector):
    keys: list[str] = Field(default_factory=list, max_length=50)
    group_by: str = Field(default="key", pattern="^(key|platform)$")
    limit: int = Field(default=10, ge=1, le=MAX_GROUPS)


class GroupCount(StrictModel):
    label: ProvenanceStr
    volume: int
    """Distinct `(cluster, platform)` pairs -- `docs/signal-model.md` §4.3.

    Computed by `services/trend_service.bucket_volume`, not re-derived here.
    Counting raw signals instead would let one story reposted forty times
    outweigh forty independent ones, and the two are indistinguishable in the
    output.
    """

    raw_signals: int
    share: float = 0.0


class AggregateResult(BoundedResult):
    ITEMS_FIELD = "groups"

    group_by: str
    window_start: datetime
    window_end: datetime
    groups: list[GroupCount] = Field(default_factory=list)
    total_volume: int = 0
    distinct_groups: int = 0


class FitForecastInput(SeriesSelector):
    key: str = Field(min_length=1, max_length=MAX_KEY_CHARS)
    horizon: int = Field(default=7, ge=1, le=MAX_HORIZON)
    interval_level: float = Field(default=0.8, gt=0.5, lt=1.0)
    method: ForecastMethod | None = Field(
        default=None,
        description="Leave unset to let the service choose from the history length. "
        "The model selects a method and writes caveats; it never produces numbers.",
    )


class ForecastPointOut(StrictModel):
    at: datetime
    mean: float
    lower: float
    upper: float


class ForecastResult(BoundedResult):
    """A projection and everything needed to judge it.

    `ITEMS_FIELD` is deliberately unset, so the registry never shrinks this
    result. Dropping the tail of a forecast silently shortens its horizon, and a
    12-bucket projection reported as a 9-bucket one is a different claim. It
    fits by construction instead: `MAX_HORIZON` points, each four floats.
    """

    key: ProvenanceStr
    method: str
    horizon: int
    bucket_hours: int
    interval_level: float
    points: list[ForecastPointOut] = Field(default_factory=list, max_length=MAX_HORIZON)
    assumptions: list[str] = Field(default_factory=list, max_length=12)
    caveats: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = 0.0
    observations_used: int = 0
    history_start: datetime | None = None
    history_end: datetime | None = None


class DescribeInput(SeriesSelector):
    key: str = Field(min_length=1, max_length=MAX_KEY_CHARS)


class DescribeResult(BoundedResult):
    """Descriptive statistics over one series. Fixed size, never shrunk."""

    key: ProvenanceStr
    observations: int = 0
    total: int = 0
    mean: float = 0.0
    median: float = 0.0
    stdev: float = 0.0
    minimum: int = 0
    maximum: int = 0
    p90: float = 0.0
    zero_buckets: int = 0
    duplication_ratio: float = 0.0
    """`raw_signals / volume` across the window.

    Far above 1 means most of the traffic was one story repeated on one platform
    -- the exact condition under which a raw count lies, and the check that keeps
    a repost loop from being reported as demand.
    """


# --------------------------------------------------------------------------- #
# The toolset
# --------------------------------------------------------------------------- #


class AnalyticsToolset:
    """Binds the trend and forecast services to four read-only tools.

    The clock is injected because every window is relative to "now": a test that
    cannot fix now has to either freeze the process clock or assert on ranges,
    and both make a bucket-boundary bug invisible.
    """

    def __init__(
        self,
        *,
        trends: TrendService | None = None,
        forecasts: ForecastService | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        if trends is None and forecasts is None:
            raise ConfigurationError(
                "AnalyticsToolset needs at least a TrendService; with neither service "
                "it registers no tools, which is a wiring mistake rather than a policy."
            )
        if forecasts is not None and trends is None:
            # `fit_forecast` reads its own history so the agent cannot supply
            # one. Without a trend service there is nowhere to read it from, and
            # the only remaining way to forecast would be to accept numbers from
            # the model -- which is precisely what §5.6 forbids.
            raise ConfigurationError(
                "A ForecastService without a TrendService cannot source history; "
                "fit_forecast would have to accept numbers from the model."
            )
        self._trends = trends
        self._forecasts = forecasts
        self._clock = clock

    # -------------------------------------------------------- registration --

    def specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = []
        if self._trends is not None:
            specs.extend(
                (
                    ToolSpec(
                        name="timeseries",
                        description=(
                            "Bucketed volume for named topics or entities, with velocity, "
                            "acceleration and a burst qualifier computed from the data."
                        ),
                        input_model=TimeseriesInput,
                        output_model=TimeseriesResult,
                        handler=self._timeseries,
                    ),
                    ToolSpec(
                        name="aggregate",
                        description=(
                            "Count de-duplicated mention volume over a window, grouped by "
                            "topic/entity or by platform."
                        ),
                        input_model=AggregateInput,
                        output_model=AggregateResult,
                        handler=self._aggregate,
                    ),
                    ToolSpec(
                        name="describe",
                        description=(
                            "Descriptive statistics for one series: mean, median, spread, "
                            "quiet buckets and duplication ratio."
                        ),
                        input_model=DescribeInput,
                        output_model=DescribeResult,
                        handler=self._describe,
                    ),
                )
            )
        if self._forecasts is not None:
            specs.append(
                ToolSpec(
                    name="fit_forecast",
                    description=(
                        "Project one series forward with a prediction interval. The tool "
                        "reads the history itself; it never accepts values. Returns the "
                        "method, its assumptions and its caveats."
                    ),
                    input_model=FitForecastInput,
                    output_model=ForecastResult,
                    handler=self._fit_forecast,
                )
            )
        return specs

    # ------------------------------------------------------------ handlers --

    async def _timeseries(self, args: TimeseriesInput) -> TimeseriesResult:
        series = await self._read_series(args, keys=args.keys)
        kept = series[:MAX_SERIES]
        return TimeseriesResult(
            series=[self._to_series_out(item) for item in kept],
            truncated=len(series) > len(kept),
            dropped=max(0, len(series) - len(kept)),
        )

    async def _aggregate(self, args: AggregateInput) -> AggregateResult:
        if self._trends is None:  # pragma: no cover - guarded by specs()
            raise ConfigurationError("aggregate invoked with no trend service bound")
        window_start, window_end = self._window(args)
        mentions = await self._trends.mentions(
            dimension=args.dimension,
            window_start=window_start,
            window_end=window_end,
            keys=args.keys or None,
            platforms=args.platforms or None,
        )

        grouped: dict[str, list[Mention]] = {}
        for mention in mentions:
            label = mention.key if args.group_by == "key" else str(mention.platform)
            grouped.setdefault(label, []).append(mention)

        counts = [
            GroupCount(
                label=label,
                # `bucket_volume` is the §4.3 rule; re-implementing the distinct
                # `(cluster, platform)` count here would be a second place for it
                # to drift out of agreement with every other volume in the system.
                volume=bucket_volume(members),
                raw_signals=len({member.signal_id for member in members}),
            )
            for label, members in grouped.items()
        ]
        counts.sort(key=lambda item: (-item.volume, item.label))
        total = sum(item.volume for item in counts)
        kept = counts[: min(args.limit, MAX_GROUPS)]
        for item in kept:
            item.share = round(item.volume / total, 4) if total else 0.0

        return AggregateResult(
            group_by=args.group_by,
            window_start=window_start,
            window_end=window_end,
            groups=kept,
            total_volume=total,
            distinct_groups=len(counts),
            truncated=len(counts) > len(kept),
            dropped=max(0, len(counts) - len(kept)),
        )

    async def _describe(self, args: DescribeInput) -> DescribeResult:
        series = await self._read_series(args, keys=[args.key])
        if not series:
            # An empty result is a legitimate answer -- the key was never
            # mentioned -- so it is reported as zeros rather than raised. The
            # `observations=0` is what tells the agent not to describe anything.
            return DescribeResult(key=args.key)

        first = series[0]
        volumes = [float(value) for value in first.volumes]
        raw = sum(point.raw_signals for point in first.points)
        total = int(sum(volumes))
        return DescribeResult(
            key=first.key,
            observations=len(volumes),
            total=total,
            mean=round(statistics.fmean(volumes), 4) if volumes else 0.0,
            median=round(statistics.median(volumes), 4) if volumes else 0.0,
            stdev=round(statistics.pstdev(volumes), 4) if len(volumes) > 1 else 0.0,
            minimum=int(min(volumes)) if volumes else 0,
            maximum=int(max(volumes)) if volumes else 0,
            p90=round(_percentile(volumes, 0.9), 4),
            zero_buckets=sum(1 for value in volumes if value == 0),
            duplication_ratio=round(raw / total, 4) if total else 0.0,
        )

    async def _fit_forecast(self, args: FitForecastInput) -> ForecastResult:
        if self._forecasts is None:  # pragma: no cover - guarded by specs()
            raise ConfigurationError("fit_forecast invoked with no forecast service bound")
        series = await self._read_series(args, keys=[args.key])
        if not series:
            # Raised rather than returned empty: `ForecastService` refuses a
            # history it cannot support, and a "forecast" of a series that does
            # not exist is the single most dangerous empty result this layer
            # could hand a report.
            raise ValidationError(
                f"no series for key {args.key!r} in the requested window; nothing to forecast"
            )

        first = series[0]
        history = observations_from_buckets(
            first.window_start, first.bucket, list(first.volumes)
        )
        projection = self._forecasts.forecast(
            history,
            horizon=min(args.horizon, MAX_HORIZON),
            interval_level=args.interval_level,
            method=args.method,
        )
        return ForecastResult(
            key=first.key,
            method=str(projection.method),
            horizon=projection.horizon,
            bucket_hours=args.bucket_hours,
            interval_level=projection.interval_level,
            points=[
                ForecastPointOut(at=point.at, mean=point.mean, lower=point.lower,
                                 upper=point.upper)
                for point in projection.points
            ],
            assumptions=list(projection.assumptions)[:12],
            caveats=list(projection.caveats)[:12],
            confidence=projection.confidence,
            observations_used=projection.observations_used,
            history_start=projection.history_start,
            history_end=projection.history_end,
        )

    # ----------------------------------------------------------- internals --

    def _window(self, args: SeriesSelector) -> tuple[datetime, datetime]:
        end = self._clock()
        return end - timedelta(days=args.window_days), end

    async def _read_series(
        self, args: SeriesSelector, *, keys: Sequence[str]
    ) -> list[VolumeSeries]:
        if self._trends is None:  # pragma: no cover - guarded by specs()
            raise ConfigurationError("a series read was attempted with no trend service")
        window_start, window_end = self._window(args)
        return await self._trends.volume_series(
            dimension=args.dimension,
            window_start=window_start,
            window_end=window_end,
            bucket=args.bucket,
            keys=list(keys),
            platforms=args.platforms or None,
        )

    def _to_series_out(self, series: VolumeSeries) -> SeriesOut:
        trend = summarize(series, config=self._trends.config if self._trends else None)
        return SeriesOut(
            key=series.key,
            dimension=series.dimension,
            bucket_hours=int(series.bucket.total_seconds() // 3600),
            window_start=series.window_start,
            window_end=series.window_end,
            points=[
                SeriesPoint(
                    bucket_start=point.bucket_start,
                    volume=point.volume,
                    raw_signals=point.raw_signals,
                    clusters=point.clusters,
                )
                for point in series.points[:MAX_POINTS]
            ],
            total_volume=series.total_volume,
            latest_velocity=round(series.latest_velocity, 4),
            latest_acceleration=round(series.latest_acceleration, 4),
            peak_z=round(trend.peak_z, 4),
            novelty=round(trend.novelty, 4),
            is_bursting=trend.is_bursting,
        )


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Nearest-rank rather than interpolated because these are counts: an
    interpolated 90th percentile of bucket volumes reports 4.5 mentions, and a
    fractional mention is not a thing anyone can check against the corpus.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered)) - 1))
    return ordered[index]
