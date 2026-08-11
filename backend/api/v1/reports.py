"""`/api/v1/reports` -- read finished reports (`docs/api-reference.md` §4.4).

Read-only. There is no `POST /reports` and there must not be one: a report is
*produced by an investigation*, and an endpoint that let a client write one would
create a document with the appearance of provenance and none of the substance --
no run, no evidence, no critique, and no way for a reader to tell the difference.

**`409 report_not_ready` is the interesting status.** A report row is created
when its investigation starts, so a client has an id to poll before the content
exists. Fetching it then is a 409, never a 404: a 404 says the report will never
exist and the client stops, when in fact it was thirty seconds away. This is the
single most important distinction in this module.

**Superseded versions stay readable.** Reports are versioned rather than edited,
so `GET /reports/{id}` on an old version returns it with `is_current=false` and a
pointer to its successor. Returning a redirect to the newest version instead
would mean a link someone saved silently resolves to a different document -- the
exact failure versioning exists to prevent.
"""

from __future__ import annotations

from typing import Annotated, Final

from fastapi import APIRouter, Depends, Query

from backend.api.deps import Principal, require_scopes, upstream
from backend.schemas.common import problem_responses
from backend.schemas.report import (
    CitationItem,
    ConfidenceBand,
    ReportDetail,
    ReportFormatName,
    ReportSectionItem,
    ReportStatusName,
    ReportSummaryItem,
)
from models.orm.report import ReportStatus
from services.report_service import ReportService, ReportSummary, ReportView

__all__ = ["get_report_service", "router"]

router = APIRouter(prefix="/reports", tags=["reports"])

ReaderPrincipal = Annotated[Principal, Depends(require_scopes("reports:read"))]


async def get_report_service(principal: ReaderPrincipal) -> ReportService:
    """The service, scoped to the caller's tenant at construction.

    Tenant bound here rather than passed per call, matching every other service
    in this API. A tenant threaded through call sites is a tenant one call site
    can forget.
    """
    from backend.db.session import get_sessionmaker

    return ReportService(get_sessionmaker(), tenant_id=principal.tenant_id)


ServiceDep = Annotated[ReportService, Depends(get_report_service)]


_STATUS_NAMES: Final[dict[ReportStatus, ReportStatusName]] = {
    ReportStatus.DRAFT: ReportStatusName.PENDING,
    ReportStatus.READY: ReportStatusName.COMPLETE,
    # A superseded report is finished and still fetchable by version; the client
    # asked for this id and this id has a body. "Pending" would tell it to keep
    # polling something that will never change again.
    ReportStatus.SUPERSEDED: ReportStatusName.COMPLETE,
    ReportStatus.FAILED: ReportStatusName.FAILED,
}
"""Storage vocabulary -> API vocabulary, stated rather than inferred.

These two enums deliberately differ: storage distinguishes `draft` from
`superseded`, and a client only needs to know whether there is a body to fetch.
The previous code bridged them by testing whether the stored *value* happened to
be spelled the same as an API name and defaulting to `pending` otherwise -- which
silently mapped `ready` to `pending`, so a finished report told every client to
keep polling forever. An explicit table cannot fail that way: a member added to
either enum raises a KeyError here instead of quietly becoming `pending`.
"""


def _summary_fields(record: ReportSummary) -> dict[str, object]:
    confidence = float(record.confidence)
    return {
        "id": record.id,
        "investigation_id": record.investigation_id,
        "title": record.title,
        "summary": record.summary,
        "status": _STATUS_NAMES.get(record.status, ReportStatusName.PENDING),
        "format": ReportFormatName(record.format.value)
        if record.format.value in set(ReportFormatName)
        else ReportFormatName.MARKDOWN,
        "confidence": confidence,
        "confidence_band": ConfidenceBand.from_score(confidence),
        "version": record.version,
        "is_current": record.is_current,
        "superseded_by": record.superseded_by,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _detail(report: ReportView) -> ReportDetail:
    """Project the stored report onto the wire shape.

    `gaps` is reconstructed from the section whose heading marks it rather than
    stored separately, because `report_sections` has no kind column -- the
    limitations arrive as an ordinary section written last by
    `agents/report/agent.py`. Reading it back out here keeps the promise that
    gaps are a top-level field a renderer cannot miss.
    """
    gaps: list[str] = []
    sections: list[ReportSectionItem] = []

    for section in report.sections:
        if section.heading.casefold().startswith("what this investigation could not"):
            gaps = [line.lstrip("- ").strip() for line in section.body.splitlines() if line.strip()]
            continue
        sections.append(
            ReportSectionItem(
                id=section.id,
                ordinal=section.ordinal,
                heading=section.heading,
                body=section.body,
                confidence=section.confidence,
                citations=[
                    CitationItem(
                        id=citation.id,
                        signal_id=citation.signal_id,
                        quote=citation.quote,
                        char_start=citation.char_start,
                        char_end=citation.char_end,
                        relevance=citation.relevance,
                    )
                    for citation in section.citations
                ],
            )
        )

    return ReportDetail(
        **_summary_fields(report),
        sections=sections,
        gaps=gaps,
        citation_count=report.citation_count,
        uncited_sections=list(report.uncited_sections),
        download_url=(f"/api/v1/reports/{report.id}/download" if report.object_key else None),
    )


@router.get(
    "/{report_id}",
    summary="One report with its sections, citations and gaps.",
    response_model=ReportDetail,
    responses=problem_responses(401, 403, 404, 409, 502),
)
async def get_report(
    report_id: str,
    principal: ReaderPrincipal,
    service: ServiceDep,
) -> ReportDetail:
    """Fetch a report.

    `409 report_not_ready` when the investigation has not reached its reporting
    step, `404` when no such report exists for this tenant. The distinction is
    actionable: 409 means poll, 404 means stop. `ReportService.require` makes it.
    """
    async with upstream("postgres"):
        report = await service.require(report_id)
    return _detail(report)


@router.get(
    "",
    summary="Reports for an investigation, newest version first.",
    response_model=list[ReportSummaryItem],
    responses=problem_responses(401, 403, 422, 502),
)
async def list_reports(
    principal: ReaderPrincipal,
    service: ServiceDep,
    investigation_id: Annotated[str, Query(min_length=1)],
    include_superseded: Annotated[
        bool,
        Query(
            description=(
                "Include earlier versions. Off by default: a client asking for "
                "'the report' wants the current one, and returning three versions "
                "invites it to render whichever comes first."
            )
        ),
    ] = False,
) -> list[ReportSummaryItem]:
    """List an investigation's reports.

    Requires `investigation_id` rather than listing every report in the tenant.
    A bare list would be a firehose with no useful ordering -- reports are read
    by way of the investigation that produced them, which is also the only
    context in which a version chain means anything.
    """
    async with upstream("postgres"):
        records = await service.for_investigation(
            investigation_id, include_superseded=include_superseded
        )
    return [ReportSummaryItem(**_summary_fields(record)) for record in records]


@router.get(
    "/{report_id}/citations",
    summary="Every citation in a report, flattened.",
    response_model=list[CitationItem],
    responses=problem_responses(401, 403, 404, 409, 502),
)
async def get_citations(
    report_id: str,
    principal: ReaderPrincipal,
    service: ServiceDep,
) -> list[CitationItem]:
    """The flat citation list, for verification tooling.

    Separate from the report body because the consumer is different: a reader
    wants citations beside the claims they support, while an auditor wants the
    whole set to resolve against the corpus in one pass. Forcing the second to
    walk the section tree is the kind of friction that makes verification not
    happen.
    """
    async with upstream("postgres"):
        report = await service.require(report_id)

    return [
        CitationItem(
            id=citation.id,
            signal_id=citation.signal_id,
            quote=citation.quote,
            char_start=citation.char_start,
            char_end=citation.char_end,
            relevance=citation.relevance,
        )
        for section in report.sections
        for citation in section.citations
    ]
