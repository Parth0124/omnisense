"""ORM tables for reports, sections and citations.

Design Doc §2 asks for "evidence-backed reports with citations and confidence".
`citations` is the table that makes that claim checkable: every row ties one span
of one report to one `Signal`, with the exact quoted text and its character
offsets. Without it a report is prose; with it, every sentence can be walked back
through the Signal to the raw bytes in R2 and the moment they were fetched
(`models/lineage.py`).

Three tables rather than one JSON document because each is queried on its own
axis. Sections are fetched and re-rendered in order. Citations are counted per
report, resolved individually by the Critic (`docs/agent-system.md` §13), and --
critically -- looked up *by `signal_id`*, which is how an erasure request finds
every report affected by deleting a Signal. A JSON blob answers none of those
without a full scan.

The narrative text lives here rather than in R2 because it is queried by
attribute and mutated by the Critic loop, and R2 stores only immutable,
content-addressed objects (`docs/data-stores.md` §3.6). What goes to R2 is the
*rendered artifact* -- the Markdown or PDF a user downloads -- referenced by
`ReportRow.object_key`.
"""

from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from models.base import TolerantStrEnum
from models.orm.base import Base, TolerantEnumType
from models.orm.mixins import TenantMixin, TimestampMixin

__all__ = [
    "CitationRow",
    "ReportFormat",
    "ReportRow",
    "ReportSectionRow",
    "ReportStatus",
]


class ReportStatus(TolerantStrEnum):
    """Publication state of a report.

    `DRAFT` exists because `docs/api-reference.md` §4.1 allocates the report id
    when the investigation is created, so the row can exist before the Report
    agent has written anything -- that is the state `409 report_not_ready`
    describes. `SUPERSEDED` is what a row moves to when a re-run produces a newer
    version; earlier versions stay fetchable (§4.4 `version` parameter), so they
    are marked rather than deleted.

    Defined here rather than in `models/enums.py` because `models/report.py` is
    still a stub and this vocabulary has exactly one consumer today.
    """

    DRAFT = "draft"
    READY = "ready"
    SUPERSEDED = "superseded"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ReportFormat(TolerantStrEnum):
    """Serialization of the rendered artifact stored at `ReportRow.object_key`."""

    MARKDOWN = "markdown"
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    UNKNOWN = "unknown"


class ReportRow(Base, TimestampMixin, TenantMixin):
    """One version of one investigation's report."""

    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    investigation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False,
    )
    """`CASCADE`: the report is the investigation's output and quotes user
    content, so an erasure of the investigation must take it along. The reverse
    direction (`investigations.report_id`) is `SET NULL` for the opposite reason
    -- deleting one report must not erase the record that the run happened."""

    # -- content -----------------------------------------------------------
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    """Empty while the row is a `DRAFT` placeholder holding the eagerly allocated
    id. `Text` rather than `String(n)` because a generated title is a sentence,
    and truncating it mid-word is worse than storing a long one."""

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    """The executive summary. Nullable rather than defaulted: "not written yet"
    and "deliberately has no summary" need to stay distinguishable, because only
    the first is a reason to retry the Report agent."""

    format: Mapped[ReportFormat] = mapped_column(
        TolerantEnumType(ReportFormat, 16), nullable=False, default=ReportFormat.MARKDOWN
    )

    object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    """R2 key of the rendered artifact. `NULL` until it is rendered.

    Only the key lives here. The bytes never do -- PostgreSQL must not hold
    payload bodies (`docs/data-stores.md` §3.1), and the artifact is immutable and
    content-addressed, which is precisely what R2 is for.
    """

    # -- quality -----------------------------------------------------------
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    """Calibrated belief in `[0, 1]`, not a retrieval score
    (`docs/glossary.md`, "commonly confused pairs"). Rolled up from the section
    confidences by the Report agent and reviewed by the Critic."""

    status: Mapped[ReportStatus] = mapped_column(
        TolerantEnumType(ReportStatus, 32), nullable=False, default=ReportStatus.DRAFT
    )

    # -- versioning --------------------------------------------------------
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    """Monotonic per investigation. A re-run writes a *new* row rather than
    updating this one, so a link shared yesterday keeps resolving to the text
    that was actually read (`docs/api-reference.md` §4.4)."""

    superseded_by: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("reports.id", ondelete="SET NULL"),
        nullable=True,
    )
    """Forward pointer to the version that replaced this one.

    `SET NULL` rather than `CASCADE`: deleting a newer version must leave the
    older one intact -- cascading would delete backwards through the whole
    version chain from a single delete of the newest row.
    """

    __table_args__ = (
        # Versions are per investigation, and two rows claiming the same version
        # would make `?version=2` ambiguous.
        UniqueConstraint("investigation_id", "version"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="confidence_range"),
        CheckConstraint("version >= 1", name="version_positive"),
        # A version chain must move forward. Self-reference would make the
        # "latest version" walk in `services/report_service.py` loop forever.
        CheckConstraint(
            "superseded_by IS NULL OR superseded_by <> id", name="supersede_points_elsewhere"
        ),
        # The report list for a tenant, newest first.
        Index("ix_reports_tenant_created", "tenant_id", "created_at"),
        # Sweep for drafts whose investigation died before the Report agent ran;
        # they are what `409 report_not_ready` keeps returning forever otherwise.
        Index("ix_reports_status_created", "status", "created_at"),
    )


class ReportSectionRow(Base, TimestampMixin, TenantMixin):
    """One ordered section of a report's narrative."""

    __tablename__ = "report_sections"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    report_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    """Render order. Explicit rather than implied by insertion order or by `id`,
    because the Critic can force a rewrite that reorders sections without
    recreating the report."""

    heading: Mapped[str] = mapped_column(Text, nullable=False)

    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    """The section prose, with inline citation markers. The markers resolve
    against `citations.id`; the mapping is not duplicated into this text, so a
    dropped citation cannot leave a marker pointing at a row that never existed.
    """

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    __table_args__ = (
        # Two sections cannot occupy the same slot, and this also serves the
        # "sections of this report, in order" read that every render performs.
        UniqueConstraint("report_id", "ordinal"),
        CheckConstraint("ordinal >= 0", name="ordinal_non_negative"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="confidence_range"),
    )


class CitationRow(Base, TimestampMixin, TenantMixin):
    """One claim-to-Signal link: the row that makes a report auditable."""

    __tablename__ = "citations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    report_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("reports.id", ondelete="CASCADE"),
        nullable=False,
    )

    section_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("report_sections.id", ondelete="CASCADE"),
        nullable=True,
    )
    """Nullable because a citation can support the report as a whole -- the
    executive summary and the confidence rationale both cite evidence without
    belonging to a numbered section."""

    signal_id: Mapped[str] = mapped_column(String(64), nullable=False)
    """The cited Signal. Deliberately **not** a foreign key to `signals.id`.

    A deletion request has to be able to hard-delete a Signal
    (`docs/security-and-privacy.md`), and neither foreign-key behaviour is
    acceptable there: `CASCADE` would silently rewrite a report a user already
    read, and `RESTRICT` would make erasure impossible. So the reference is
    intentionally soft, and `services/evidence_service.py` treats an unresolvable
    `signal_id` as the `broken_citation` finding the Critic already knows how to
    report (`docs/agent-system.md` §13), rather than as data corruption.
    """

    quote: Mapped[str] = mapped_column(Text, nullable=False)
    """The exact quoted span, copied at citation time.

    Redundant with `signals.content_text` and stored anyway: verification compares
    this text against the Signal, so keeping only the offsets would make a
    re-crawl that shifted the text look like a match. It is also what survives
    when the Signal is erased -- the quote is the audit record of what was
    actually relied on.
    """

    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    """Half-open `[char_start, char_end)` offsets into the Signal's cleaned text,
    which is what the API renders as `char_range` (`docs/api-reference.md` §4.4).
    Offsets are into the *cleaned* text because that is what retrieval indexed;
    resolving them against the raw payload would land in the wrong place."""

    relevance: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    """Normalized retrieval score for the cited passage, in `[0, 1]`.

    Normalization is a storage precondition, not a suggestion: raw cross-encoder
    logits are unbounded and not comparable between rerankers, and the range check
    below is what stops one leaking in. This is a *score*, never a confidence, and
    must not be rendered as one (`docs/glossary.md`).
    """

    __table_args__ = (
        CheckConstraint("char_start >= 0", name="char_start_non_negative"),
        # Half-open and possibly empty is fine; inverted is not.
        CheckConstraint("char_end >= char_start", name="char_range_ordered"),
        CheckConstraint("relevance >= 0.0 AND relevance <= 1.0", name="relevance_range"),
        # Rendering a report: all of its citations, grouped by section. The
        # leading column also serves the plain "citations of this report" count.
        Index("ix_citations_report_section", "report_id", "section_id"),
        # The reverse lookup, and the reason this table is worth its indexes:
        # "which reports cite this Signal" is what an erasure request must answer
        # before it deletes one, and what "show me everything built on this
        # source" answers in the UI.
        Index("ix_citations_signal", "signal_id"),
        # PostgreSQL does not index a foreign key automatically, so without this
        # the ON DELETE CASCADE from `report_sections` degrades to a sequential
        # scan of every citation in the tenant.
        Index("ix_citations_section", "section_id"),
    )
