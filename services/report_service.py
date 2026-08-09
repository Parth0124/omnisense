"""Report persistence: assemble, version, and read back with citations intact.

The Report *agent* produces a structured document inside an investigation run.
This service is what makes that document durable, addressable and re-readable --
and it is the layer that enforces the one property the whole product rests on:
**a stored report's citations resolve.**

**Reports are versioned, never edited.** A revision writes a new row and points
the old one at it through `superseded_by`. The alternative -- updating in place --
means a report someone read on Tuesday says something different on Thursday with
no record that it changed, which for a document people make decisions from is
worse than having no report. Version 3 existing does not delete version 2; it
supersedes it, and the chain is walkable.

**Citations are rows, not JSON.** `citations` is its own table with a foreign key
to the report and a signal id, a quote and a character range. Storing them inside
the report body as JSON would make "which reports cite this signal" -- the query
you need when a source is retracted or a signal is erased -- a full table scan
over serialised blobs. As rows it is an index lookup.

**Assembly is transactional.** A report, its sections and its citations are
written in one transaction. A partial write would leave a report whose sections
exist and whose citations do not, which renders as a document making unsourced
claims -- indistinguishable, to a reader, from a report that was never
evidence-backed at all.

Layer note: **L2 service**. Imports `models/`, `backend/db`, the kernel. Called by
`backend/api/v1/reports.py` and `workers/report_worker.py`.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.core.logging import get_logger
from models.orm.mixins import DEFAULT_TENANT
from models.orm.report import (
    CitationRow,
    ReportFormat,
    ReportRow,
    ReportSectionRow,
    ReportStatus,
)

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

__all__ = [
    "MAX_CITATIONS_PER_REPORT",
    "MAX_SECTIONS",
    "CitationView",
    "ReportService",
    "ReportSummary",
    "ReportView",
    "SectionView",
]

logger = get_logger(__name__)

MAX_SECTIONS: Final = 20
MAX_CITATIONS_PER_REPORT: Final = 500
"""Ceiling on stored citations.

Not a storage concern -- it is a signal. A report with five hundred citations is
not better evidenced than one with fifty; it is one where something is emitting
citations mechanically, and the cap turns that into a visible failure instead of
a slowly growing table.
"""


# --------------------------------------------------------------------------- #
# Read shapes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CitationView:
    id: str
    signal_id: str
    quote: str
    char_start: int
    char_end: int
    relevance: float
    section_id: str | None = None


@dataclass(frozen=True, slots=True)
class SectionView:
    id: str
    ordinal: int
    heading: str
    body: str
    confidence: float
    citations: tuple[CitationView, ...] = ()


@dataclass(frozen=True, slots=True)
class ReportSummary:
    """A report without its body. What a list endpoint returns."""

    id: str
    investigation_id: str
    title: str
    summary: str | None
    status: ReportStatus
    format: ReportFormat
    confidence: float
    version: int
    superseded_by: str | None
    created_at: datetime
    updated_at: datetime

    @property
    def is_current(self) -> bool:
        """Whether this is the latest version. A superseded report is history."""
        return self.superseded_by is None

    @property
    def is_readable(self) -> bool:
        """Whether the body exists yet.

        Distinct from `is_current`: a report row is created when the
        investigation starts so a client has an id to subscribe with, and it is
        empty until the Report agent finishes. Fetching it before then is a
        `409 report_not_ready`, not a 404 -- the resource exists.
        """
        return self.status is ReportStatus.COMPLETE


@dataclass(frozen=True, slots=True)
class ReportView(ReportSummary):
    """A report with its sections and their citations."""

    sections: tuple[SectionView, ...] = ()
    object_key: str | None = None

    @property
    def citation_count(self) -> int:
        return sum(len(section.citations) for section in self.sections)

    @property
    def uncited_sections(self) -> tuple[str, ...]:
        """Headings whose sections make claims with nothing behind them.

        Exposed because it is the cheapest integrity check a reader has, and
        because a report where half the sections are uncited should not render
        identically to one where none are. `agents/report/agent.py` drops
        unsupported claims before storage, so a section arriving here uncited is
        usually narrative -- but "usually" is exactly why the caller should be
        able to see it rather than assume.
        """
        return tuple(
            section.heading for section in self.sections if not section.citations
        )


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class ReportService:
    """Writes and reads reports. Tenant-scoped at construction, never per call."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    # -------------------------------------------------------------- writes --

    async def create_placeholder(
        self, *, investigation_id: str, report_id: str | None = None
    ) -> ReportSummary:
        """Allocate a report id before the report exists.

        Called when an investigation is created, so `POST /investigations` can
        return a `links.report` a client can subscribe to immediately. Without
        this the client has no handle until the run is nearly over, and the
        streaming design loses most of its value.

        The row is `PENDING`, which is what makes `409 report_not_ready`
        expressible -- as opposed to a 404, which would tell a client the report
        will never exist.
        """
        from sqlalchemy import select

        resolved_id = report_id or f"rpt_{uuid.uuid4()}"
        async with self._session_factory() as session, session.begin():
            existing = (
                await session.execute(
                    select(ReportRow).where(ReportRow.id == resolved_id)
                )
            ).scalar_one_or_none()
            if existing is not None:
                # Idempotent: an investigation retried under the same id must not
                # fail here, and must not create a second report either.
                return _summary(existing)

            row = ReportRow(
                id=resolved_id,
                tenant_id=self._tenant_id,
                investigation_id=investigation_id,
                title="",
                status=ReportStatus.PENDING,
                format=ReportFormat.MARKDOWN,
                confidence=0.0,
                version=1,
            )
            session.add(row)
            await session.flush()
            return _summary(row)

    async def store(
        self,
        *,
        investigation_id: str,
        document: Mapping[str, Any],
        report_id: str | None = None,
        object_key: str | None = None,
    ) -> ReportView:
        """Persist a finished report: row, sections and citations, in one transaction.

        A partial write leaves sections without their citations, which renders as
        a document making unsourced claims -- and a reader cannot distinguish
        that from a report that was never evidence-backed. One transaction is
        what makes the stored artefact either whole or absent.

        Supersedes any existing current report for the investigation rather than
        updating it. See the module docstring: a document people decide from must
        not silently change under them.
        """
        from sqlalchemy import select

        sections = _sections_of(document)
        if len(sections) > MAX_SECTIONS:
            raise ValidationError(
                f"report has {len(sections)} sections; the maximum is {MAX_SECTIONS}"
            )
        total_citations = sum(len(s.get("claims") or []) for s in sections)
        if total_citations > MAX_CITATIONS_PER_REPORT:
            raise ValidationError(
                f"report carries {total_citations} citations, above the "
                f"{MAX_CITATIONS_PER_REPORT} ceiling. That many is not better "
                "evidence -- it means something is emitting citations mechanically."
            )

        async with self._session_factory() as session, session.begin():
            previous = (
                await session.execute(
                    select(ReportRow)
                    .where(
                        ReportRow.investigation_id == investigation_id,
                        ReportRow.tenant_id == self._tenant_id,
                        ReportRow.superseded_by.is_(None),
                    )
                    .order_by(ReportRow.version.desc())
                )
            ).scalars().first()

            # A pending placeholder is filled in rather than superseded: it has
            # no content to preserve, and superseding it would leave a permanent
            # empty version 1 in every investigation's history.
            reuse = previous if previous is not None and previous.status is ReportStatus.PENDING else None
            version = 1 if reuse is not None else (previous.version + 1 if previous else 1)
            resolved_id = report_id or (reuse.id if reuse else f"rpt_{uuid.uuid4()}")

            row = reuse
            if row is None:
                row = ReportRow(
                    id=resolved_id,
                    tenant_id=self._tenant_id,
                    investigation_id=investigation_id,
                    version=version,
                )
                session.add(row)

            row.title = str(document.get("title") or "Investigation report")[:1000]
            row.summary = _as_str_or_none(document.get("executive_summary"))
            row.status = ReportStatus.COMPLETE
            row.format = ReportFormat.MARKDOWN
            row.confidence = float(document.get("confidence") or 0.0)
            row.object_key = object_key
            await session.flush()

            if previous is not None and previous.id != row.id:
                previous.superseded_by = row.id

            await self._write_sections(session, row.id, sections)
            await session.flush()

        stored = await self.get(resolved_id)
        if stored is None:  # pragma: no cover -- written in the transaction above
            raise ConflictError("the report vanished immediately after being written")
        logger.info(
            "report.stored",
            report_id=stored.id,
            investigation_id=investigation_id,
            version=stored.version,
            sections=len(stored.sections),
            citations=stored.citation_count,
        )
        return stored

    async def _write_sections(
        self, session: AsyncSession, report_id: str, sections: Sequence[Mapping[str, Any]]
    ) -> None:
        """Replace this report's sections and citations.

        Delete-then-insert rather than upsert. A report version is written once
        and never partially amended, so reconciling row-by-row would be work
        spent to preserve state that has no meaning -- and a missed reconcile
        would leave a stale section from a previous store() attempt sitting in
        the middle of the document.
        """
        from sqlalchemy import delete

        await session.execute(
            delete(CitationRow).where(CitationRow.report_id == report_id)
        )
        await session.execute(
            delete(ReportSectionRow).where(ReportSectionRow.report_id == report_id)
        )

        for ordinal, section in enumerate(sections):
            section_id = f"sec_{uuid.uuid4()}"
            session.add(
                ReportSectionRow(
                    id=section_id,
                    tenant_id=self._tenant_id,
                    report_id=report_id,
                    ordinal=ordinal,
                    heading=str(section.get("title") or f"Section {ordinal + 1}")[:500],
                    body=str(section.get("body") or ""),
                    confidence=float(section.get("confidence") or 0.0),
                )
            )
            for claim in section.get("claims") or []:
                if not isinstance(claim, Mapping):
                    continue
                quote = str(claim.get("text") or "")
                for signal_id in claim.get("citations") or []:
                    if not isinstance(signal_id, str) or not signal_id:
                        continue
                    session.add(
                        CitationRow(
                            id=f"cit_{uuid.uuid4()}",
                            tenant_id=self._tenant_id,
                            report_id=report_id,
                            section_id=section_id,
                            signal_id=signal_id,
                            quote=quote[:4000],
                            char_start=int(claim.get("char_start") or 0),
                            char_end=int(claim.get("char_end") or len(quote)),
                            relevance=float(claim.get("confidence") or 0.0),
                        )
                    )

    # --------------------------------------------------------------- reads --

    async def get(self, report_id: str) -> ReportView | None:
        """One report with its sections and citations, or `None`."""
        from sqlalchemy import select

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(ReportRow).where(
                        ReportRow.id == report_id,
                        ReportRow.tenant_id == self._tenant_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                return None

            section_rows = (
                await session.execute(
                    select(ReportSectionRow)
                    .where(ReportSectionRow.report_id == report_id)
                    .order_by(ReportSectionRow.ordinal)
                )
            ).scalars().all()
            citation_rows = (
                await session.execute(
                    select(CitationRow).where(CitationRow.report_id == report_id)
                )
            ).scalars().all()

        by_section: dict[str | None, list[CitationView]] = {}
        for citation in citation_rows:
            by_section.setdefault(citation.section_id, []).append(
                CitationView(
                    id=citation.id,
                    signal_id=citation.signal_id,
                    quote=citation.quote,
                    char_start=citation.char_start,
                    char_end=citation.char_end,
                    relevance=citation.relevance,
                    section_id=citation.section_id,
                )
            )

        return ReportView(
            **_summary_fields(row),
            object_key=row.object_key,
            sections=tuple(
                SectionView(
                    id=section.id,
                    ordinal=section.ordinal,
                    heading=section.heading,
                    body=section.body,
                    confidence=section.confidence,
                    citations=tuple(by_section.get(section.id, ())),
                )
                for section in section_rows
            ),
        )

    async def require(self, report_id: str) -> ReportView:
        """`get`, but a missing report is a 404 and an unfinished one is a 409.

        The two are genuinely different and a caller acts differently on each:
        404 means stop, 409 means poll. Collapsing them into one status is how a
        client ends up either giving up on a report that was thirty seconds from
        existing, or retrying forever against one that never will.
        """
        report = await self.get(report_id)
        if report is None:
            raise NotFoundError.for_resource("report", report_id)
        if not report.is_readable:
            raise ConflictError(
                "the report is not ready yet; the investigation has not reached "
                "its reporting step",
                details={"status": report.status.value, "code": "report_not_ready"},
            )
        return report

    async def for_investigation(
        self, investigation_id: str, *, include_superseded: bool = False
    ) -> list[ReportSummary]:
        """Every report for an investigation, newest version first."""
        from sqlalchemy import select

        statement = select(ReportRow).where(
            ReportRow.investigation_id == investigation_id,
            ReportRow.tenant_id == self._tenant_id,
        )
        if not include_superseded:
            statement = statement.where(ReportRow.superseded_by.is_(None))

        async with self._session_factory() as session:
            rows = (
                await session.execute(statement.order_by(ReportRow.version.desc()))
            ).scalars().all()
        return [_summary(row) for row in rows]

    async def citing_signal(self, signal_id: str, *, limit: int = 100) -> list[str]:
        """Report ids that cite a signal. The query erasure needs.

        An index lookup because citations are rows. Stored as JSON inside the
        report body this would be a full scan over serialised blobs, and it is
        exactly the query you need under time pressure -- when a source has been
        retracted or a signal must be erased and someone has to know which
        published documents depended on it.
        """
        from sqlalchemy import select

        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(CitationRow.report_id)
                    .where(
                        CitationRow.signal_id == signal_id,
                        CitationRow.tenant_id == self._tenant_id,
                    )
                    .distinct()
                    .limit(limit)
                )
            ).scalars().all()
        return list(rows)


# --------------------------------------------------------------------------- #
# Projection helpers
# --------------------------------------------------------------------------- #


def _summary_fields(row: ReportRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "investigation_id": row.investigation_id,
        "title": row.title,
        "summary": row.summary,
        "status": row.status,
        "format": row.format,
        "confidence": float(row.confidence),
        "version": row.version,
        "superseded_by": row.superseded_by,
        "created_at": _as_utc(row.created_at),
        "updated_at": _as_utc(row.updated_at),
    }


def _summary(row: ReportRow) -> ReportSummary:
    return ReportSummary(**_summary_fields(row))


def _sections_of(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = document.get("sections")
    if not isinstance(raw, (list, tuple)):
        return []
    return [section for section in raw if isinstance(section, Mapping)]


def _as_str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _as_utc(value: datetime | None) -> datetime:
    """Normalise a stored timestamp to aware UTC.

    SQLite returns naive datetimes even for `DateTime(timezone=True)` columns, so
    a value read back in the test suite is naive where the same value from
    PostgreSQL is aware. Comparing the two silently succeeds under one backend
    and raises under the other.
    """
    if value is None:
        return datetime.now(UTC)
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
