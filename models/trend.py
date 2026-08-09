"""The trend domain model: measured change, and what makes a measurement one.

A trend is the most quotable thing this system produces -- "complaints rose 40%
this quarter" travels into a slide and is never checked again. So this model is
shaped so that an unsupported trend is *unconstructable* rather than merely
discouraged.

**A direction requires observations.** Three of them, enforced. Two points are a
line, and "doubled" computed from two observations is technically true, reads as
a finding, and is noise.

**`change_pct` is optional and must be.** A series starting at zero has no
defined percentage change. "Up 400%" from one prior mention to five is
arithmetically correct and substantively meaningless -- the classic way this
metric embarrasses a report.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel, UtcDatetime

__all__ = [
    "MIN_OBSERVATIONS_FOR_DIRECTION",
    "STABLE_BAND_PCT",
    "Trend",
    "TrendDirection",
    "TrendPoint",
]

MIN_OBSERVATIONS_FOR_DIRECTION: Final = 3
STABLE_BAND_PCT: Final = 10.0
"""Below this absolute change, the series is stable rather than moving.

Written down so "stable" means the same thing in every report this system
produces -- which is what makes two reports comparable. Left to a model, the
threshold varies between calls.
"""


class TrendDirection(enum.StrEnum):
    RISING = "rising"
    FALLING = "falling"
    STABLE = "stable"
    VOLATILE = "volatile"
    """Moving more than noise, without direction.

    A real answer for a jumpy series, and the one a three-way enum forces into a
    wrong bucket. "Rising" sounds like a finding and "volatile" does not, so the
    pressure is one-directional -- which is exactly why the option must exist.
    """

    @property
    def is_directional(self) -> bool:
        return self in (TrendDirection.RISING, TrendDirection.FALLING)


class TrendPoint(StrictModel):
    """One observation in a series."""

    at: UtcDatetime
    value: float


class Trend(StrictModel):
    """One measured trend, with the evidence that makes it checkable."""

    topic: str = Field(min_length=1, max_length=300)
    direction: TrendDirection
    observation_count: int = Field(default=0, ge=0)
    change_pct: float | None = None
    window_start: UtcDatetime | None = None
    window_end: UtcDatetime | None = None
    confidence: Score = 0.0
    entity_ids: list[str] = Field(default_factory=list, max_length=20)
    signal_ids: list[str] = Field(default_factory=list, max_length=20)
    summary: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def _direction_is_supported(self) -> Trend:
        if (
            self.direction.is_directional
            and self.observation_count < MIN_OBSERVATIONS_FOR_DIRECTION
        ):
            raise ValueError(
                f"a {self.direction.value} trend needs at least "
                f"{MIN_OBSERVATIONS_FOR_DIRECTION} observations; "
                f"{self.observation_count} is a line between two points"
            )
        if (
            self.window_start is not None
            and self.window_end is not None
            and self.window_end <= self.window_start
        ):
            raise ValueError("window_end must be after window_start")
        return self

    @property
    def is_significant(self) -> bool:
        """Whether the movement is large enough to report as a change."""
        if not self.direction.is_directional or self.change_pct is None:
            return False
        return abs(self.change_pct) >= STABLE_BAND_PCT
