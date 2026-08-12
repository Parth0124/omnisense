"""`/api/v1/projects` -- the projects a question can be asked about.

A project groups the sources a product's work is recorded in, so that "what
happened this week" is one query over three repositories and a Slack channel
rather than four queries stitched together.

**There is no delete.** Deactivating is the whole vocabulary. Deleting a project
would either orphan its sources or cascade into their artifacts, and "we stopped
working on this" is a different fact from "this never happened" -- the second one
destroys the history that every citation in the system resolves against.

**Attach and detach are `PUT` and `DELETE` on the membership, not on the source.**
`DELETE /projects/{slug}/sources/{id}` removes the *membership*; the source and
its artifacts survive. Modelling it as deleting the source would make the
destructive reading the obvious one.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from backend.api.deps import Principal, require_scopes
from backend.schemas.common import problem_responses
from backend.schemas.project import (
    ProjectCreateRequest,
    ProjectDetail,
    ProjectSourceItem,
    ProjectSummary,
)
from services.project_service import ProjectService, build_project_service

__all__ = ["get_project_service", "router"]

router = APIRouter(prefix="/projects", tags=["projects"])

ReaderPrincipal = Annotated[Principal, Depends(require_scopes("projects:read"))]
WriterPrincipal = Annotated[Principal, Depends(require_scopes("projects:write"))]


async def get_project_service(principal: ReaderPrincipal) -> ProjectService:
    """One service per request, scoped to the caller's tenant.

    Built from the principal rather than from settings so the tenant can never be
    defaulted by accident -- a service constructed with the wrong tenant reads
    another customer's projects and returns a perfectly well-formed answer.

    **Every route below also declares its principal explicitly**, even the reads
    that need nothing else from it. That looks redundant and is not: this
    dependency is the kind of thing a test overrides, and when it is the only
    place authentication is required, overriding it silently removes the check.
    That is not hypothetical -- the first version of this router did exactly
    that, and a test asking for the project list with no token got a 200.
    """
    return build_project_service(tenant_id=principal.tenant_id)


ServiceDep = Annotated[ProjectService, Depends(get_project_service)]


def _summary(project: object) -> dict[str, object]:
    return {
        "id": project.id,  # type: ignore[attr-defined]
        "slug": project.slug,  # type: ignore[attr-defined]
        "name": project.name,  # type: ignore[attr-defined]
        "description": project.description,  # type: ignore[attr-defined]
        "is_active": project.is_active,  # type: ignore[attr-defined]
        "created_at": project.created_at,  # type: ignore[attr-defined]
    }


@router.get(
    "",
    summary="Every project, newest configuration first.",
    response_model=list[ProjectSummary],
    responses=problem_responses(401, 403),
)
async def list_projects(
    principal: ReaderPrincipal,
    service: ServiceDep,
    include_inactive: Annotated[
        bool,
        Query(description="Include paused projects. They keep their history either way."),
    ] = False,
) -> list[ProjectSummary]:
    projects = await service.list_projects(include_inactive=include_inactive)
    return [ProjectSummary(**_summary(project)) for project in projects]


@router.post(
    "",
    summary="Create a project.",
    response_model=ProjectSummary,
    status_code=status.HTTP_201_CREATED,
    responses=problem_responses(401, 403, 409, 422),
)
async def create_project(
    request: ProjectCreateRequest,
    principal: WriterPrincipal,
    service: ServiceDep,
) -> ProjectSummary:
    """Create one. The slug is derived from the name when not supplied.

    A duplicate slug is a 409 rather than a silent adoption of the existing
    project: `omnisense init` run twice with the same name almost always means
    the second run was a mistake, and quietly attaching new repositories to an
    existing project is the harder mistake to notice.
    """
    project = await service.create(
        name=request.name,
        slug=request.slug,
        description=request.description,
    )
    return ProjectSummary(**_summary(project))


@router.get(
    "/{slug}",
    summary="One project and the sources it owns.",
    response_model=ProjectDetail,
    responses=problem_responses(401, 403, 404),
)
async def get_project(slug: str, principal: ReaderPrincipal, service: ServiceDep) -> ProjectDetail:
    project = await service.get(slug)
    sources = await service.sources(slug)
    return ProjectDetail(
        **_summary(project),
        sources=[ProjectSourceItem(**source.model_dump()) for source in sources],
        artifact_count=sum(source.artifact_count for source in sources),
    )


@router.put(
    "/{slug}/sources/{source_id}",
    summary="Put a source in this project, moving it if it was in another.",
    response_model=ProjectSourceItem,
    responses=problem_responses(401, 403, 404),
)
async def attach_source(
    slug: str,
    source_id: str,
    principal: WriterPrincipal,
    service: ServiceDep,
) -> ProjectSourceItem:
    """`PUT` rather than `POST`: attaching a source already attached here is a
    no-op with the same result, which is exactly what idempotent means."""
    source = await service.attach_source(slug=slug, source_id=source_id)
    return ProjectSourceItem(**source.model_dump())


@router.delete(
    "/{slug}/sources/{source_id}",
    summary="Remove a source from this project. The source and its history remain.",
    status_code=status.HTTP_204_NO_CONTENT,
    responses=problem_responses(401, 403, 404),
)
async def detach_source(
    slug: str,
    source_id: str,
    principal: WriterPrincipal,
    service: ServiceDep,
) -> None:
    await service.detach_source(source_id=source_id)


@router.post(
    "/{slug}/deactivate",
    summary="Pause a project. It stops syncing and keeps everything it has.",
    response_model=ProjectSummary,
    responses=problem_responses(401, 403, 404),
)
async def deactivate_project(
    slug: str, principal: WriterPrincipal, service: ServiceDep
) -> ProjectSummary:
    project = await service.set_active(slug=slug, is_active=False)
    return ProjectSummary(**_summary(project))


@router.post(
    "/{slug}/activate",
    summary="Resume a paused project.",
    response_model=ProjectSummary,
    responses=problem_responses(401, 403, 404),
)
async def activate_project(
    slug: str, principal: WriterPrincipal, service: ServiceDep
) -> ProjectSummary:
    project = await service.set_active(slug=slug, is_active=True)
    return ProjectSummary(**_summary(project))
