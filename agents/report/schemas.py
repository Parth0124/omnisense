"""Report agent input and output schemas.

The last node, and the one whose output a human actually reads -- which makes it
the last place a defect can be caught and the first place one becomes visible.

The schema is built around a single structural claim: **a rendered claim carries
its citations or it is not a claim.** `ReportClaim.citations` is required and
non-empty. There is no way to construct a claim without them, so a sentence that
reaches the document has, by construction, something behind it.

`gaps` is equally non-negotiable, for the reason `docs/architecture.md` §7.3
gives: this system is allowed to return a smaller, honestly-labelled answer
instead of failing. That promise is only kept if the smallness is *visible*. A
report that quietly omits the section it could not support looks exactly like a
report that had nothing to say there.
"""

from __future__ import annotations

import enum
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel

__all__ = [
    "MAX_CLAIMS",
    "MAX_SECTIONS",
    "ConfidenceBand",
    "ReportClaim",
    "ReportInput",
    "ReportOutput",
    "ReportSection",
    "SectionKind",
]

MAX_SECTIONS: Final = 10
MAX_CLAIMS: Final = 40


class SectionKind(enum.StrEnum):
    EXECUTIVE_SUMMARY = "executive_summary"
    FINDINGS = "findings"
    TRENDS = "trends"
    COMPETITIVE = "competitive"
    FORECAST = "forecast"
    RECOMMENDATIONS = "recommendations"
    GAPS = "gaps"
    """What the investigation could not establish. Mandatory -- see below."""

    METHODOLOGY = "methodology"


class ConfidenceBand(enum.StrEnum):
    """The badge the UI renders.

    A band rather than the raw float because a reader shown "0.63" will treat the
    second digit as meaningful, and it is not -- the underlying number is an
    aggregate of judgements, not a measurement. Three buckets say what can
    honestly be said.
    """

    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float) -> ConfidenceBand:
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.45:
            return cls.MODERATE
        return cls.LOW


class ReportClaim(StrictModel):
    """One assertion in the document, inseparable from its support."""

    id: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=1500)
    citations: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Signal ids. Required -- a claim without one is not a claim.",
    )
    insight_ids: list[str] = Field(default_factory=list, max_length=8)
    confidence: Score = 0.0
    hedged: bool = Field(
        default=False,
        description=(
            "Whether this is stated conditionally. Set for anything derived from "
            "a causal hypothesis or a degraded retrieval, so the renderer can "
            "mark it rather than relying on the prose to have hedged itself."
        ),
    )


class ReportSection(StrictModel):
    kind: SectionKind
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=20000)
    claims: list[ReportClaim] = Field(default_factory=list, max_length=MAX_CLAIMS)
    order: int = Field(default=0, ge=0)


class ReportInput(StrictModel):
    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    investigation_id: str
    insights: list[dict] = Field(default_factory=list, max_length=12)
    recommendations: list[dict] = Field(default_factory=list, max_length=8)
    trends: list[dict] = Field(default_factory=list, max_length=10)
    forecasts: list[dict] = Field(default_factory=list, max_length=6)
    competitor_view: dict | None = None
    critique: dict | None = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=60)
    unanswered_sub_questions: list[str] = Field(default_factory=list, max_length=8)
    degraded_backends: list[str] = Field(default_factory=list, max_length=4)
    collection_failures: list[str] = Field(default_factory=list, max_length=8)
    confidence: Score = 0.0


class ReportOutput(StrictModel):
    """The finished document."""

    title: str = Field(min_length=1, max_length=300)
    executive_summary: str = Field(min_length=1, max_length=4000)
    sections: list[ReportSection] = Field(min_length=1, max_length=MAX_SECTIONS)
    confidence: Score = 0.0
    confidence_band: ConfidenceBand = ConfidenceBand.LOW
    gaps: list[str] = Field(
        default_factory=list,
        max_length=12,
        description=(
            "What this investigation could not establish. Populated from the "
            "Critic's findings, the unanswered sub-questions and any degraded "
            "dependency -- not written freehand by the model, so it cannot be "
            "quietly omitted."
        ),
    )
    citation_count: int = 0

    @model_validator(mode="after")
    def _gaps_must_be_rendered(self) -> ReportOutput:
        """A report with gaps must carry the section that shows them.

        `docs/architecture.md` §7.3 permits a smaller, honestly-labelled answer
        instead of a failure. That promise is only kept if the smallness is
        visible -- and a quietly omitted section looks identical to a section
        that had nothing to say.
        """
        if self.gaps and not any(
            section.kind is SectionKind.GAPS for section in self.sections
        ):
            raise ValueError(
                f"{len(self.gaps)} gap(s) recorded but no GAPS section renders them. "
                "An unrendered gap is a gap the reader never learns about."
            )
        return self

    @property
    def all_claims(self) -> list[ReportClaim]:
        return [claim for section in self.sections for claim in section.claims]
