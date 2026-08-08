"""Insight agent input and output schemas.

An insight is a claim about what the evidence *means*, which makes it the most
useful and the least verifiable thing this system produces. A retrieved passage
can be checked against its source; a trend can be checked against its series; an
insight is a synthesis, and nothing downstream can mechanically confirm it.

So the schema does the only thing a schema can do about that: it makes an insight
**structurally inseparable from its support**. `signal_ids` is required and
non-empty. There is no path to an `Insight` object that is not attached to the
evidence it came from, which means the Critic can check every one of them and a
report can cite every one of them -- and an insight the model produced from its
own prior knowledge about the industry has nowhere to attach itself and fails
validation instead of being published.

`docs/agent-system.md` §5.7.
"""

from __future__ import annotations

import enum
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel

__all__ = [
    "MAX_INSIGHTS",
    "MIN_SUPPORTING_SIGNALS",
    "Insight",
    "InsightInput",
    "InsightKind",
    "InsightOutput",
]

MAX_INSIGHTS: Final = 12
MIN_SUPPORTING_SIGNALS: Final = 1
"""Every insight cites at least one signal. Zero is not a degraded insight, it is
an unsupported assertion, and there is no report section where one belongs."""


class InsightKind(enum.StrEnum):
    """What sort of claim this is. Bounds how strongly it may be worded.

    The distinction the report renderer needs: an `observation` is a restatement
    of evidence and can be asserted, while a `hypothesis` is the model reasoning
    beyond what any passage says and must be hedged. Collapsing them is how a
    guess acquires the tone of a finding.
    """

    OBSERVATION = "observation"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    CAUSAL_HYPOTHESIS = "causal_hypothesis"
    """The most dangerous kind. Correlation in a mention corpus is mostly
    co-reporting -- two things written about together because one article covered
    both -- so a causal claim needs the strongest hedging and the most evidence."""

    IMPLICATION = "implication"


class Insight(StrictModel):
    """One claim, permanently attached to what supports it."""

    id: str = Field(min_length=1, max_length=40)
    kind: InsightKind
    statement: str = Field(min_length=1, max_length=800)
    reasoning: str = Field(
        min_length=1,
        max_length=1500,
        description="How the evidence leads here. Required: an insight whose "
        "derivation cannot be written down is one nobody can check.",
    )
    confidence: Score = 0.0
    signal_ids: list[str] = Field(min_length=MIN_SUPPORTING_SIGNALS, max_length=20)
    entity_ids: list[str] = Field(default_factory=list, max_length=16)
    sub_question_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="Which planned questions this answers. Drives coverage checking.",
    )
    contradicts: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Ids of insights this one conflicts with. Recorded rather than "
            "resolved: a corpus that genuinely disagrees with itself is a "
            "finding, and silently keeping whichever the model wrote last "
            "destroys it."
        ),
    )

    @model_validator(mode="after")
    def _causal_claims_need_more_support(self) -> Insight:
        """A causal hypothesis on a single source is not a hypothesis, it is a quote.

        Correlation in a mention corpus is mostly co-reporting: two things appear
        together because one article covered both. A causal claim resting on that
        single article is restating the article's framing as an analytical
        finding.
        """
        if self.kind is InsightKind.CAUSAL_HYPOTHESIS and len(self.signal_ids) < 2:
            raise ValueError(
                "a causal hypothesis needs at least two independent signals; on one "
                "source it is the source's own framing restated as analysis"
            )
        if self.kind is InsightKind.CAUSAL_HYPOTHESIS and self.confidence > 0.8:
            raise ValueError(
                f"confidence {self.confidence} is too high for a causal hypothesis "
                "drawn from a mention corpus, where co-occurrence is usually "
                "co-reporting rather than causation"
            )
        return self


class InsightInput(StrictModel):
    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    sub_questions: list[str] = Field(default_factory=list, max_length=8)
    sub_question_ids: list[str] = Field(default_factory=list, max_length=8)
    evidence_ids: list[str] = Field(default_factory=list, max_length=60)
    trends: list[dict] = Field(default_factory=list, max_length=10)
    forecasts: list[dict] = Field(default_factory=list, max_length=6)
    competitor_view: dict | None = None
    prior_critique: dict | None = Field(
        default=None,
        description="A Critic finding from an earlier pass, when this is a revision.",
    )


class InsightOutput(StrictModel):
    insights: list[Insight] = Field(default_factory=list, max_length=MAX_INSIGHTS)
    unanswered_sub_questions: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Questions the evidence could not answer. Stated explicitly because "
            "an omitted question looks identical to an answered one in a finished "
            "report, and the reader cannot tell which they are looking at."
        ),
    )
    notes: str | None = Field(default=None, max_length=1000)

    @property
    def has_contradictions(self) -> bool:
        return any(insight.contradicts for insight in self.insights)
