"""Strategy agent input and output schemas.

A recommendation is the only output of this system that asks someone to *do*
something, which changes what the schema has to guarantee. An insight that turns
out to be wrong costs credibility; a recommendation that turns out to be wrong
costs whatever the reader spent acting on it.

Three fields carry that weight, and none of them is optional:

`based_on_insight_ids` -- a recommendation must descend from a stated insight,
which in turn cites signals. That chain is what makes "why are you telling me
this" answerable all the way down to a document. A recommendation with no parent
insight is the model's opinion about the industry.

`assumptions` -- what must be true for this to be right. A recommendation without
them reads as unconditional, and the conditions are usually where it breaks.

`risks` -- what happens if it is wrong. Required for the same reason
`agents/forecast/schemas.py` requires caveats: the hedge is what stops the
headline from being the only thing that survives into a slide.
"""

from __future__ import annotations

import enum
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel

__all__ = [
    "MAX_RECOMMENDATIONS",
    "Horizon",
    "Recommendation",
    "StrategyInput",
    "StrategyOutput",
    "Urgency",
]

MAX_RECOMMENDATIONS: Final = 8
"""Ceiling on recommendations.

Not arbitrary: a list of twenty recommendations is a list nobody acts on, and
the model will produce twenty if allowed to, because generating a plausible
additional action is always easier than deciding the list is complete.
"""


class Urgency(enum.StrEnum):
    IMMEDIATE = "immediate"
    NEAR_TERM = "near_term"
    MONITOR = "monitor"
    """Explicitly a non-action. Preserved as a first-class option because "watch
    this and do nothing yet" is frequently the correct recommendation, and a
    vocabulary without it forces every observation into a call to act."""


class Horizon(enum.StrEnum):
    DAYS = "days"
    WEEKS = "weeks"
    QUARTERS = "quarters"


class Recommendation(StrictModel):
    """One proposed action, with its provenance and its failure conditions."""

    id: str = Field(min_length=1, max_length=40)
    action: str = Field(min_length=1, max_length=500)
    rationale: str = Field(min_length=1, max_length=1500)
    urgency: Urgency
    horizon: Horizon
    confidence: Score = 0.0
    based_on_insight_ids: list[str] = Field(
        min_length=1,
        max_length=12,
        description="The insights this descends from. Required -- see the module docstring.",
    )
    assumptions: list[str] = Field(min_length=1, max_length=6)
    risks: list[str] = Field(min_length=1, max_length=6)
    expected_signals: list[str] = Field(
        default_factory=list,
        max_length=6,
        description=(
            "What would show this was the right call. The difference between a "
            "recommendation and an opinion: one is checkable afterwards."
        ),
    )

    @model_validator(mode="after")
    def _immediate_actions_need_conviction(self) -> Recommendation:
        """An urgent recommendation held with low confidence is incoherent.

        Telling a reader to act immediately on something the system is 30% sure
        of transfers a risk it has not priced. Either the confidence is higher
        than stated -- in which case say so -- or the urgency is lower.
        """
        if self.urgency is Urgency.IMMEDIATE and self.confidence < 0.5:
            raise ValueError(
                f"an immediate action cannot be held at confidence {self.confidence}; "
                "either the conviction is higher or the urgency is lower. Asking "
                "someone to act now on a coin flip transfers an unpriced risk."
            )
        return self


class StrategyInput(StrictModel):
    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    insights: list[dict] = Field(default_factory=list, max_length=12)
    trends: list[dict] = Field(default_factory=list, max_length=10)
    forecasts: list[dict] = Field(default_factory=list, max_length=6)
    competitor_view: dict | None = None
    unanswered_questions: list[str] = Field(default_factory=list, max_length=8)


class StrategyOutput(StrictModel):
    recommendations: list[Recommendation] = Field(
        default_factory=list, max_length=MAX_RECOMMENDATIONS
    )
    summary: str | None = Field(default=None, max_length=2000)
    withheld_reason: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Why no recommendation was made. Set when the evidence does not "
            "support one -- an explicit refusal is a useful output, and an empty "
            "recommendations list with no explanation reads as a pipeline failure."
        ),
    )

    @model_validator(mode="after")
    def _silence_must_be_explained(self) -> StrategyOutput:
        if not self.recommendations and not self.withheld_reason:
            raise ValueError(
                "no recommendations and no reason given. Either recommend "
                "something or say why the evidence does not support one -- an "
                "unexplained empty list is indistinguishable from a crash."
            )
        return self
