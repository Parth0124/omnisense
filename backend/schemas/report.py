"""Wire shapes for `/api/v1/reports` (`docs/api-reference.md` §4.4).

A report is the product's output, so these schemas are where its two central
promises become visible to a client rather than merely true internally.

**Every claim carries its citations.** `ClaimItem.citations` is required and
non-empty, mirroring `agents/report/schemas.py`. A client can therefore render a
source marker beside every sentence without checking whether one exists, and a
report that lost its citations in storage fails serialisation here rather than
rendering as authoritative prose.

**Gaps are part of the document, not metadata.** `gaps` is a top-level field, not
buried in a `meta` object a client may not read. `docs/architecture.md` §7.3
permits a smaller, honestly-labelled answer instead of a failure -- and that
promise only holds if the label is somewhere a renderer cannot miss.

**Confidence is a band and a number.** The number is for filtering and
comparison; the band is what a badge renders. Publishing only the float means
every client picks its own thresholds and two screens disagree about whether 0.63
is "moderate".
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Final

from pydantic import Field

from backend.schemas.common import ResponseModel

__all__ = [
    "MAX_BODY_CHARS",
    "CitationItem",
    "ClaimItem",
    "ConfidenceBand",
    "ReportDetail",
    "ReportFormatName",
    "ReportSectionItem",
    "ReportSummaryItem",
    "ReportStatusName",
]

MAX_BODY_CHARS: Final = 50_000
"""Cap on one section's body.

Fifty thousand characters is roughly twenty pages -- far beyond any real section
and low enough that a runaway generation cannot make a single response
multi-megabyte.
"""


class ReportStatusName(enum.StrEnum):
    PENDING = "pending"
    """The row exists so a client has an id to subscribe with; the body does not.

    Fetching a pending report is `409 report_not_ready`, never 404. A 404 would
    say the report will never exist, and the client would stop polling something
    that is thirty seconds away.
    """

    COMPLETE = "complete"
    FAILED = "failed"


class ReportFormatName(enum.StrEnum):
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"


class ConfidenceBand(enum.StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float) -> ConfidenceBand:
        """Thresholds live here so the whole product agrees on them."""
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.45:
            return cls.MODERATE
        return cls.LOW


class CitationItem(ResponseModel):
    """One source reference, resolvable to a signal a reader can open."""

    id: str
    signal_id: str
    quote: str = Field(
        description=(
            "The supporting span, verbatim from the source. Verified against the "
            "stored signal before it was written -- a paraphrase never reaches here."
        )
    )
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    relevance: float = 0.0


class ClaimItem(ResponseModel):
    """One assertion, inseparable from what supports it."""

    text: str
    citations: list[CitationItem] = Field(
        min_length=1,
        description="Non-empty by construction. A claim without a source is not published.",
    )
    hedged: bool = Field(
        default=False,
        description=(
            "Stated conditionally -- derived from a causal hypothesis or from "
            "degraded retrieval. Rendered as a hedge rather than relying on the "
            "prose to have hedged itself."
        ),
    )


class ReportSectionItem(ResponseModel):
    id: str
    ordinal: int
    heading: str
    body: str = Field(max_length=MAX_BODY_CHARS)
    confidence: float = 0.0
    citations: list[CitationItem] = Field(default_factory=list)


class ReportSummaryItem(ResponseModel):
    """A report without its body, for a list."""

    id: str
    investigation_id: str
    title: str
    summary: str | None = None
    status: ReportStatusName
    format: ReportFormatName
    confidence: float
    confidence_band: ConfidenceBand
    version: int
    is_current: bool = Field(
        description=(
            "False when a newer version supersedes this one. Reports are versioned "
            "rather than edited: a document people decide from must not change "
            "under them without a record."
        )
    )
    superseded_by: str | None = None
    created_at: datetime
    updated_at: datetime


class ReportDetail(ReportSummaryItem):
    """A report with its sections, citations and limitations."""

    sections: list[ReportSectionItem] = Field(default_factory=list)
    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "What the investigation could not establish. A top-level field rather "
            "than metadata, because §7.3's promise of an honestly-labelled smaller "
            "answer only holds if the label is somewhere a renderer cannot miss."
        ),
    )
    citation_count: int = 0
    uncited_sections: list[str] = Field(
        default_factory=list,
        description=(
            "Headings whose sections carry no citation. The cheapest integrity "
            "check a reader has, and the reason a mostly-uncited report should not "
            "render identically to a fully-cited one."
        ),
    )
    download_url: str | None = Field(
        default=None,
        description="Relative path to the rendered artefact, when one exists.",
    )
