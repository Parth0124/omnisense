"""The forecast domain model: a projection that cannot be stated without its band.

A wrong statement about the past can be checked against the corpus. A wrong
statement about the future cannot be checked until it is too late to matter, and
it reads as authoritative precisely because it arrives with a confidence
interval. This model is therefore built so that the interval is not optional and
the refusal is a first-class value.

**`INSUFFICIENT_DATA` carries no points, enforced.** A label next to a number is
a label that gets ignored -- the number is what survives into the slide. So a
forecast that declines to project must be *empty*, not annotated.

**Caveats are required for anything that does project.** For the same reason the
Strategy model requires risks: without the failure condition, the headline is all
that survives.
"""

from __future__ import annotations

import enum
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel, UtcDatetime

__all__ = [
    "MAX_HORIZON_DAYS",
    "MIN_HISTORY_POINTS",
    "Forecast",
    "ForecastMethod",
    "ForecastPoint",
]

MAX_HORIZON_DAYS: Final = 180
MIN_HISTORY_POINTS: Final = 8
"""Observations required before a projection is attempted.

Not a statistical result -- a refusal threshold. Below it every available method
is fitting noise, and the interval it produces understates the uncertainty
because the fit itself is unstable.
"""


class ForecastMethod(enum.StrEnum):
    NAIVE = "naive"
    LINEAR = "linear"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SEASONAL = "seasonal"
    INSUFFICIENT_DATA = "insufficient_data"
    """Explicitly not a forecast. A value rather than an absence, so the refusal
    is visible in a report instead of showing up as a missing section."""

    @property
    def projects(self) -> bool:
        return self is not ForecastMethod.INSUFFICIENT_DATA


class ForecastPoint(StrictModel):
    """One projected value with its uncertainty band."""

    at: UtcDatetime
    lower: float
    value: float
    upper: float

    @model_validator(mode="after")
    def _band_contains_its_centre(self) -> ForecastPoint:
        if not (self.lower <= self.value <= self.upper):
            raise ValueError(
                f"interval [{self.lower}, {self.upper}] does not contain the estimate "
                f"{self.value}; a band that excludes its own centre is not an "
                "uncertainty interval"
            )
        return self

    @property
    def width(self) -> float:
        return self.upper - self.lower


class Forecast(StrictModel):
    """A projection for one subject, with everything needed to discount it."""

    subject: str = Field(min_length=1, max_length=300)
    method: ForecastMethod
    points: list[ForecastPoint] = Field(default_factory=list, max_length=120)
    history_points: int = Field(default=0, ge=0)
    horizon_days: int = Field(default=30, ge=1, le=MAX_HORIZON_DAYS)
    confidence: Score = 0.0
    caveats: list[str] = Field(default_factory=list, max_length=8)
    interpretation: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _projection_is_supported(self) -> Forecast:
        if not self.method.projects:
            if self.points:
                raise ValueError(
                    "an INSUFFICIENT_DATA forecast must carry no points; publishing "
                    "numbers beside that label is how the label gets ignored"
                )
            return self
        if self.history_points < MIN_HISTORY_POINTS:
            raise ValueError(
                f"{self.subject!r} was projected from {self.history_points} "
                f"observations; below {MIN_HISTORY_POINTS} the fit is to noise"
            )
        if not self.points:
            raise ValueError("a forecast must project at least one point")
        if not self.caveats:
            raise ValueError(
                f"{self.subject!r} has no caveats; a projection without its failure "
                "conditions is a prediction, not a forecast"
            )
        return self

    @property
    def is_actionable(self) -> bool:
        """Whether the band is tight enough relative to the movement to act on.

        A range spanning "up a third" to "down a fifth" does not support a
        decision, and presenting its midpoint as the forecast is how a wide band
        gets quietly dropped.
        """
        if not self.method.projects or not self.points:
            return False
        first, last = self.points[0], self.points[-1]
        movement = abs(last.value - first.value)
        return movement > 0 and last.width < movement
