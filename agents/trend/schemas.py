"""Trend agent input and output schemas.

Every number in a `DetectedTrend` must have come back from the `timeseries` or
`aggregate` tool. The schema cannot enforce that on its own -- a model can write
any float into a float field -- so the enforcement is split: the fields are
shaped so that a fabricated number is *visible* (a trend with no
`observation_count` and no `window` is obviously unsupported), and
`TrendAgent.execute` cross-checks the claimed direction against the series it
actually retrieved.

`docs/agent-system.md` §5.4 makes the reason explicit: a trend is the most
quotable thing an investigation produces -- "mentions of battery complaints rose
40% this quarter" travels straight into a slide -- and a fabricated percentage is
indistinguishable from a real one once it leaves the system.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel

__all__ = [
    "MAX_TRENDS",
    "MIN_OBSERVATIONS_FOR_TREND",
    "DetectedTrend",
    "TrendDirection",
    "TrendInput",
    "TrendOutput",
]

MAX_TRENDS: Final = 10
MIN_OBSERVATIONS_FOR_TREND: Final = 3
"""Below three points there is no trend, only a line between two numbers.

Stated as a constant and enforced in the validator because "mentions doubled"
computed from two observations is technically true, reads as a finding, and is
noise. Three is the minimum at which direction is distinguishable from a single
step change.
"""


class TrendDirection(enum.StrEnum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"
    """Neither rising nor falling but moving more than noise -- a real and
    frequently correct answer that a three-way enum forces into a wrong bucket."""


class DetectedTrend(StrictModel):
    """One trend, with the evidence that makes it checkable."""

    topic: str = Field(min_length=1, max_length=200)
    direction: TrendDirection
    change_pct: float | None = Field(
        default=None,
        description=(
            "Percentage change across the window. Null when the series cannot "
            "support one -- a zero baseline makes the percentage undefined, and "
            "reporting 'infinite growth' from one prior mention is the classic "
            "way this metric embarrasses a report."
        ),
    )
    window_start: datetime | None = None
    window_end: datetime | None = None
    observation_count: int = Field(
        default=0,
        ge=0,
        description="Points in the underlying series. The reader's check on the claim.",
    )
    confidence: Score = 0.0
    entity_ids: list[str] = Field(default_factory=list, max_length=16)
    signal_ids: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Evidence for this trend. A trend with none is unsupported.",
    )
    summary: str = Field(min_length=1, max_length=600)

    @model_validator(mode="after")
    def _check_support(self) -> DetectedTrend:
        """A trend claiming a direction must have the observations to justify it.

        `STABLE` and `VOLATILE` are exempt from the count floor for opposite
        reasons: stability over two points is at least an honest statement about
        what was seen, and volatility is itself a statement that the series is
        too noisy to call. It is `RISING` and `FALLING` that travel into slides
        as facts, and those are the two this gate exists for.
        """
        if self.direction in (TrendDirection.RISING, TrendDirection.FALLING):
            if self.observation_count < MIN_OBSERVATIONS_FOR_TREND:
                raise ValueError(
                    f"a {self.direction.value} trend needs at least "
                    f"{MIN_OBSERVATIONS_FOR_TREND} observations; "
                    f"{self.observation_count} is a line between two points"
                )
        if self.window_start and self.window_end and self.window_end <= self.window_start:
            raise ValueError("window_end must be after window_start")
        return self


class TrendInput(StrictModel):
    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    entity_ids: list[str] = Field(default_factory=list, max_length=32)
    topics: list[str] = Field(default_factory=list, max_length=16)
    evidence_count: int = 0
    window_days: int = Field(default=90, ge=1, le=730)


class TrendOutput(StrictModel):
    trends: list[DetectedTrend] = Field(default_factory=list, max_length=MAX_TRENDS)
    series_retrieved: int = Field(
        default=0, description="Time series actually fetched. Zero means nothing was measured."
    )
    notes: str | None = Field(default=None, max_length=1000)

    @property
    def has_measurable_trends(self) -> bool:
        return bool(self.trends) and self.series_retrieved > 0
