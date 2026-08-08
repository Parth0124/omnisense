"""Forecast agent input and output schemas.

`docs/agent-system.md` §5.6 makes one structural demand of this agent: **every
number in its output must have come back from `fit_forecast`.** The model selects
a method and writes the caveats; it does not produce the numbers.

That is enforceable here in a way it is not for other agents, because a forecast
has a shape the schema can pin down. A `ForecastPoint` carries a lower bound, a
point estimate and an upper bound, and the validator refuses an interval that is
inverted or degenerate. A model inventing numbers produces well-ordered ones by
accident often enough that ordering alone is a weak check -- so the real control
is `ForecastAgent.execute`, which builds the points from the tool result and
never lets the model write them at all. These schemas make that possible by
separating what the model *does* choose (method, caveats, interpretation) from
what it must not (the series).

**Intervals are mandatory, and that is the point of the module.** A forecast
without an interval is a prediction; a forecast with one is a statement about
uncertainty. The first is what gets quoted and the second is what is true, and
making the interval non-optional is the only reliable way to stop the first from
being all that survives into a slide.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel

__all__ = [
    "MAX_FORECASTS",
    "MAX_HORIZON_DAYS",
    "MIN_HISTORY_POINTS",
    "ForecastMethod",
    "ForecastPoint",
    "ForecastInput",
    "ForecastOutput",
    "SeriesForecast",
]

MAX_FORECASTS: Final = 6
MAX_HORIZON_DAYS: Final = 180
"""Ceiling on how far ahead a forecast may reach.

Six months. Beyond that, a forecast built from mention volume is describing the
model's assumptions rather than the data -- the confidence interval widens past
usefulness, and a reader who sees a twelve-month number reads it as a
twelve-month claim regardless of how wide the band is.
"""

MIN_HISTORY_POINTS: Final = 8
"""Observations required before a forecast is attempted.

Eight is not a statistical result, it is a refusal threshold. Below it, every
method available here is fitting noise, and the interval it produces understates
the uncertainty because the fit itself is unstable.
"""


class ForecastMethod(enum.StrEnum):
    """How the projection was produced. Recorded because it bounds the claim."""

    NAIVE = "naive"
    """Last value carried forward. The honest default for a short, noisy series."""

    LINEAR = "linear"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SEASONAL = "seasonal"
    INSUFFICIENT_DATA = "insufficient_data"
    """Explicitly not a forecast. Carried so the absence is visible in the report
    rather than showing up as a missing section nobody notices."""


class ForecastPoint(StrictModel):
    """One projected value with its uncertainty band."""

    at: datetime
    lower: float
    value: float
    upper: float

    @model_validator(mode="after")
    def _check_interval(self) -> ForecastPoint:
        if not (self.lower <= self.value <= self.upper):
            raise ValueError(
                f"interval [{self.lower}, {self.upper}] does not contain the point "
                f"estimate {self.value}; a band that excludes its own centre is not "
                "an uncertainty interval"
            )
        return self

    @property
    def interval_width(self) -> float:
        return self.upper - self.lower


class SeriesForecast(StrictModel):
    """A forecast for one subject, with everything needed to discount it."""

    subject: str = Field(min_length=1, max_length=200)
    method: ForecastMethod
    points: list[ForecastPoint] = Field(default_factory=list, max_length=60)
    history_points: int = Field(default=0, ge=0)
    confidence: Score = 0.0
    caveats: list[str] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "What would make this projection wrong. Required for anything other "
            "than INSUFFICIENT_DATA -- a forecast presented without its failure "
            "conditions is a prediction, and this system does not make those."
        ),
    )
    interpretation: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def _check_support(self) -> SeriesForecast:
        if self.method is ForecastMethod.INSUFFICIENT_DATA:
            if self.points:
                raise ValueError(
                    "a forecast marked INSUFFICIENT_DATA must carry no points; "
                    "publishing numbers alongside that label is how the label "
                    "gets ignored"
                )
            return self

        if self.history_points < MIN_HISTORY_POINTS:
            raise ValueError(
                f"{self.subject!r} was forecast from {self.history_points} "
                f"observations; below {MIN_HISTORY_POINTS} every available method "
                "is fitting noise and the interval understates the uncertainty"
            )
        if not self.points:
            raise ValueError("a forecast must project at least one point")
        if not self.caveats:
            raise ValueError(
                f"{self.subject!r} has no caveats. A projection without its failure "
                "conditions is a prediction, and §5.6 does not permit one."
            )
        return self


class ForecastInput(StrictModel):
    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    subjects: list[str] = Field(default_factory=list, max_length=MAX_FORECASTS)
    horizon_days: int = Field(default=30, ge=1, le=MAX_HORIZON_DAYS)
    trend_count: int = 0


class ForecastOutput(StrictModel):
    forecasts: list[SeriesForecast] = Field(default_factory=list, max_length=MAX_FORECASTS)
    horizon_days: int = Field(default=30, ge=1, le=MAX_HORIZON_DAYS)
    notes: str | None = Field(default=None, max_length=1000)

    @property
    def has_projections(self) -> bool:
        return any(item.method is not ForecastMethod.INSUFFICIENT_DATA for item in self.forecasts)
