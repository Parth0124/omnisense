"""The report domain model: a versioned document that carries its own limits.

`services/report_service.py` persists these and `backend/schemas/report.py`
publishes them. This is the shape in between -- and the place two rules live that
neither of those layers should be free to reinterpret.

**A claim without a citation cannot be constructed.** Not "should not" --
`ReportClaim.citations` has `min_length=1`, so the failure is at construction
rather than at review. Every layer that builds a report gets the same refusal.

**A report with gaps must render them.** `docs/architecture.md` §7.3 permits a
smaller, honestly-labelled answer instead of a failure, and that permission is
only meaningful if the label survives to the reader. The validator makes an
unlabelled degraded report unrepresentable.
"""

from __future__ import annotations

import enum
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel, UtcDatetime, utcnow
from models.evidence import Citation

__all__ = [
    "MAX_SECTIONS",
    "ConfidenceBand",
    "Report",
    "ReportClaim",
    "ReportSection",
    "SectionKind",
]

MAX_SECTIONS: Final = 20


class SectionKind(enum.StrEnum):
    EXECUTIVE_SUMMARY = "executive_summary"
    FINDINGS = "findings"
    TRENDS = "trends"
    COMPETITIVE = "competitive"
    FORECAST = "forecast"
    RECOMMENDATIONS = "recommendations"
    GAPS = "gaps"
    METHODOLOGY = "methodology"


class ConfidenceBand(enum.StrEnum):
    """Three buckets, because the underlying float is not precise to two digits.

    It is an aggregate of judgements, not a measurement. Publishing 0.63 invites a
    reader to treat the second digit as meaningful; a band says only what can
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
    """One assertion. Inseparable from its citations by construction."""

    id: str
    text: str = Field(min_length=1, max_length=2000)
    citations: list[Citation] = Field(min_length=1)
    confidence: Score = 0.0
    hedged: bool = False

    @property
    def is_printable(self) -> bool:
        """Whether every citation survived verification.

        All, not any. A claim with one verified and one fabricated citation reads
        as doubly-sourced and is partly invented, which is worse than a claim with
        one honest source.
        """
        return all(citation.is_printable for citation in self.citations)

    @property
    def signal_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.signal_id for c in self.citations))


class ReportSection(StrictModel):
    kind: SectionKind
    heading: str = Field(min_length=1, max_length=300)
    body: str = Field(default="", max_length=50_000)
    ordinal: int = Field(default=0, ge=0)
    claims: list[ReportClaim] = Field(default_factory=list)
    confidence: Score = 0.0


class Report(StrictModel):
    """A finished, versioned report."""

    id: str
    investigation_id: str
    tenant_id: str
    title: str = Field(min_length=1, max_length=1000)
    executive_summary: str = Field(default="", max_length=8000)
    sections: list[ReportSection] = Field(default_factory=list, max_length=MAX_SECTIONS)
    gaps: list[str] = Field(default_factory=list, max_length=20)
    confidence: Score = 0.0
    version: int = Field(default=1, ge=1)
    superseded_by: str | None = None
    created_at: UtcDatetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _gaps_are_rendered(self) -> Report:
        """A report with recorded gaps must carry the section showing them.

        See the module docstring. An unrendered gap is a gap the reader never
        learns about, and the honesty guarantee quietly stops holding.
        """
        if self.gaps and not any(s.kind is SectionKind.GAPS for s in self.sections):
            raise ValueError(
                f"{len(self.gaps)} gap(s) recorded but no GAPS section renders them"
            )
        return self

    @property
    def confidence_band(self) -> ConfidenceBand:
        return ConfidenceBand.from_score(self.confidence)

    @property
    def is_current(self) -> bool:
        return self.superseded_by is None

    @property
    def all_claims(self) -> list[ReportClaim]:
        return [claim for section in self.sections for claim in section.claims]

    @property
    def unprintable_claims(self) -> list[ReportClaim]:
        """Claims whose citations did not all verify.

        Exposed rather than filtered here: dropping them silently would leave a
        section whose prose implies support it no longer has. The caller decides
        whether to drop the claim or the section.
        """
        return [claim for claim in self.all_claims if not claim.is_printable]

    @property
    def citation_count(self) -> int:
        return sum(len(claim.citations) for claim in self.all_claims)
