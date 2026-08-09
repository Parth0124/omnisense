"""`/api/v1/investigations` -- create, inspect, cancel (`docs/api-reference.md` §4.1-§4.4).

The endpoint the product is *for*. Everything else in the API supports it.

**Creation returns 202, never 200.** An investigation takes minutes; a synchronous
201 would mean holding an HTTP connection through a multi-agent run, and the first
proxy with a 60-second idle timeout would kill it mid-analysis. 202 with a
`links.stream` pointing at the SSE endpoint is the shape that survives real
infrastructure -- and it is what makes the execution timeline in the UI possible
at all.

**The report id is allocated eagerly.** A client subscribing to a report that does
not exist yet gets `409 report_not_ready`, which is a state it can poll. The
alternative -- allocating the id when the Report agent finishes -- means the
client has no handle to subscribe *with* until the run is nearly over, so the
whole point of streaming is lost.

**Cancellation is cooperative and says so.** `POST /cancel` marks the run
cancelled and returns immediately; the orchestrator notices at its next
checkpoint. Blocking until the graph actually stops would make cancellation take
as long as whatever step is currently running, which for a connector sync is
minutes -- and a cancel endpoint that hangs is one users press repeatedly.

**`include` omits rather than nulls.** `plan`, `steps` and `usage` are absent when
not requested, because `investigations.plan` is nullable precisely so that "not
planned yet" stays distinguishable from "planned to do nothing". Nulling an
unrequested plan throws that distinction away at the last hop.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Response, status

from backend.api.deps import (
    CursorPage,
    Principal,
    get_investigation_service,
    idempotency_key,
    pagination,
    require_scopes,
    trace_id,
    upstream,
)
from backend.core.exceptions import ConflictError, NotFoundError
from backend.schemas.common import PageInfo, problem_responses
from backend.schemas.investigation import (
    CreateInvestigationRequest,
    InvestigationCounts,
    InvestigationCreated,
    InvestigationDetail,
    InvestigationError,
    InvestigationLinks,
    InvestigationProgress,
    InvestigationUsage,
    StepItem,
    StepsPage,
)
from models.enums import InvestigationStatus
from services.investigation_service import InvestigationRecord, InvestigationService

__all__ = ["router"]

router = APIRouter(prefix="/investigations", tags=["investigations"])

ReaderPrincipal = Annotated[Principal, Depends(require_scopes("investigations:read"))]
WriterPrincipal = Annotated[Principal, Depends(require_scopes("investigations:write"))]
ServiceDep = Annotated[InvestigationService, Depends(get_investigation_service)]


MAX_STEPS_INLINE = 200
"""Steps returned inline with `?include=steps`.

A cap rather than the whole sub-collection, because a run that hit the Critic
loop several times can accumulate hundreds of steps and this is a *detail* view,
not the timeline. The timeline is the SSE stream, which is designed for it.
"""


def _step_item(step: Any) -> StepItem:
    """Project a `StepRecord` onto the wire shape.

    `completed_at` is derived from `started_at + duration_ms` rather than read:
    `investigation_steps` stores a duration, not an end timestamp, and computing
    it here keeps the arithmetic in one place instead of in every client.
    `tool_calls` and `evidence_count` stay null because the columns do not exist
    -- null says "not recorded", where 0 would claim the step made no tool calls.
    """
    from datetime import timedelta

    started = getattr(step, "started_at", None)
    duration = getattr(step, "duration_ms", None)
    completed = (
        started + timedelta(milliseconds=duration)
        if started is not None and isinstance(duration, int)
        else None
    )
    return StepItem(
        id=step.id,
        seq=step.sequence,
        agent=step.agent,
        title=f"{step.agent.value} step {step.sequence}",
        state=step.status,
        started_at=started,
        completed_at=completed,
        duration_ms=duration,
    )


def _links(record: InvestigationRecord) -> InvestigationLinks:
    """Relative paths, never absolute URLs.

    The API is reached through a proxy, a tunnel and a browser origin this
    process cannot see. `Host` is forgeable and `X-Forwarded-*` is only as
    trustworthy as the last hop, so building an absolute URL here eventually
    hands a client a link to an internal hostname.
    """
    return InvestigationLinks(
        self=f"/api/v1/investigations/{record.id}",
        stream=f"/api/v1/investigations/{record.id}/stream",
        report=f"/api/v1/reports/{record.report_id}" if record.report_id else None,
    )


def _detail(
    record: InvestigationRecord,
    *,
    trace: str,
    include: set[str],
    steps: StepsPage | None = None,
) -> InvestigationDetail:
    """Project a record onto the §4.2 body, omitting what was not requested."""
    error = (
        InvestigationError(code=record.error, message=record.error)
        if record.error
        else None
    )
    plan_steps = (record.plan or {}).get("steps") if isinstance(record.plan, dict) else None
    return InvestigationDetail(
        id=record.id,
        state=record.status,
        query=record.query,
        # `depth` is deliberately null on a read: `investigations` has no depth
        # column, so the preset chosen at creation is not recoverable. Echoing
        # "standard" would report a budget the run may never have been given.
        depth=None,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        progress=InvestigationProgress(
            steps_completed=record.step_count,
            steps_total_estimate=len(plan_steps) if isinstance(plan_steps, list) else None,
        ),
        counts=InvestigationCounts(),
        report_id=record.report_id,
        trace_id=trace,
        error=error,
        links=_links(record),
        plan=record.plan if "plan" in include else None,
        steps=steps if "steps" in include else None,
        usage=(
            InvestigationUsage(
                input_tokens=record.token_input,
                output_tokens=record.token_output,
            )
            if "usage" in include
            else None
        ),
    )


@router.post(
    "",
    summary="Start an investigation. Returns 202 with a stream link.",
    response_model=InvestigationCreated,
    status_code=status.HTTP_202_ACCEPTED,
    responses=problem_responses(400, 401, 403, 409, 422, 429, 502),
)
async def create_investigation(
    payload: CreateInvestigationRequest,
    principal: WriterPrincipal,
    service: ServiceDep,
    trace: Annotated[str, Depends(trace_id)],
    idempotency: Annotated[Any, Depends(idempotency_key)] = None,
) -> InvestigationCreated:
    """Queue an investigation.

    202 rather than 201: the resource exists but the work has not been done, and
    a client that reads 201 as "finished" will fetch the report immediately and
    get a 409. The distinction is the whole reason §4.1 specifies 202.

    `Idempotency-Key` is honoured through the dependency. Without it, a client
    retrying a request whose response was lost to a network blip starts a second
    multi-minute investigation -- and pays for both.
    """
    async with upstream("postgres"):
        record = await service.create(payload.query, created_by=principal.subject)

    return InvestigationCreated(
        id=record.id,
        state=record.status,
        query=record.query,
        depth=payload.depth,
        created_at=record.created_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        report_id=record.report_id,
        trace_id=trace,
        links=_links(record),
    )


@router.get(
    "",
    summary="List investigations for this tenant.",
    response_model=list[InvestigationCreated],
    responses=problem_responses(401, 403, 422, 502),
)
async def list_investigations(
    principal: ReaderPrincipal,
    service: ServiceDep,
    trace: Annotated[str, Depends(trace_id)],
    page: Annotated[CursorPage, Depends(pagination)],
    state: Annotated[
        list[InvestigationStatus] | None,
        Query(description="Repeatable. OR within: `?state=running&state=queued`."),
    ] = None,
) -> list[InvestigationCreated]:
    """Recent investigations, newest first.

    Scoped to the caller's tenant by the service, which was constructed with
    `tenant_id=principal.tenant_id` -- not filtered here. A filter a handler
    applies is a filter the next handler can forget.
    """
    async with upstream("postgres"):
        records = await service.list_investigations(statuses=state, limit=page.limit)

    return [
        InvestigationCreated(
            id=record.id,
            state=record.status,
            query=record.query,
            depth=None,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            report_id=record.report_id,
            trace_id=trace,
            links=_links(record),
        )
        for record in records
    ]


@router.get(
    "/{investigation_id}",
    summary="One investigation, with optional plan, steps and usage.",
    response_model=InvestigationDetail,
    responses=problem_responses(401, 403, 404, 502),
)
async def get_investigation(
    investigation_id: str,
    principal: ReaderPrincipal,
    service: ServiceDep,
    trace: Annotated[str, Depends(trace_id)],
    include: Annotated[
        list[str] | None,
        Query(description="Repeatable: `plan`, `steps`, `usage`. Omitted when unrequested."),
    ] = None,
) -> InvestigationDetail:
    """Fetch one investigation.

    A failed run still returns 200. §4.2 is explicit about this: the *request*
    succeeded and the *investigation* did not, and conflating the two makes a
    client retry an HTTP call that worked perfectly.

    Another tenant's id is a 404 rather than a 403 -- a 403 confirms the id
    exists, turning this into an existence oracle.
    """
    requested = {item.strip().casefold() for item in (include or [])}

    async with upstream("postgres"):
        record = await service.get(investigation_id)

    if record is None:
        raise NotFoundError.for_resource("investigation", investigation_id)

    steps: StepsPage | None = None
    if "steps" in requested:
        async with upstream("postgres"):
            step_records = await service.steps(investigation_id, limit=MAX_STEPS_INLINE)
        steps = StepsPage(
            items=[_step_item(step) for step in step_records],
            # `next_cursor=None` because the whole sub-collection is returned
            # inline up to the cap. `PageInfo.of` derives `has_more` from the
            # cursor, so the two cannot disagree -- which is exactly the
            # invariant §3.4 states and the reason for the factory.
            page=PageInfo.of(limit=MAX_STEPS_INLINE, next_cursor=None),
        )

    return _detail(record, trace=trace, include=requested, steps=steps)


@router.post(
    "/{investigation_id}/cancel",
    summary="Request cancellation. Cooperative -- returns immediately.",
    response_model=InvestigationDetail,
    responses=problem_responses(401, 403, 404, 409, 502),
)
async def cancel_investigation(
    investigation_id: str,
    principal: WriterPrincipal,
    service: ServiceDep,
    trace: Annotated[str, Depends(trace_id)],
    reason: Annotated[str | None, Query(max_length=500)] = None,
) -> InvestigationDetail:
    """Mark an investigation cancelled.

    Returns as soon as the state is written; the orchestrator observes it at its
    next checkpoint. Blocking until the graph actually stops would make this take
    as long as the currently-running step -- minutes, for a connector sync -- and
    a cancel endpoint that hangs is one users press repeatedly, each press
    queueing another request against a run that is already stopping.

    Cancelling an already-terminal run is a 409 rather than a silent success: a
    client that believes it cancelled a completed investigation will report to
    its user that the work was stopped, when in fact the report exists.
    """
    async with upstream("postgres"):
        record = await service.get(investigation_id)
        if record is None:
            raise NotFoundError.for_resource("investigation", investigation_id)
        if record.is_terminal:
            raise ConflictError(
                f"investigation is already {record.status.value}; there is nothing "
                "left to cancel",
                details={"state": record.status.value},
            )
        cancelled = await service.cancel(investigation_id, reason=reason)

    return _detail(cancelled, trace=trace, include=set())
