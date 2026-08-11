"""Projects, and attaching sources to them.

The only writer of `projects` and the only thing that sets `sources.project_id`.
Both the CLI and `/api/v1/projects` go through here, so the rules about what a
project is -- a unique slug, a source belonging to at most one -- are enforced in
one place rather than twice with a drift between them.

**Why the errors here are specific.** This service is driven almost entirely by a
person typing at a prompt during `omnisense init`, and the failures are the
ordinary ones: a slug already taken, a repository already attached elsewhere, a
project that does not exist. Each of those has an obvious next action, and a
generic 409 does not carry it. `docs/api-reference.md` maps these onto HTTP
status codes; the messages are written to be read by a human in a terminal.

Layer note: **L2 service** -- imports `models/` and the kernel; imported by
`backend/api/` and the CLI. It does not import `agents/` or `retrieval/`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.core.logging import get_logger
from models.orm.artifact import ArtifactRow, SourceRow
from models.orm.mixins import DEFAULT_TENANT
from models.orm.project import ProjectRow
from models.project import (
    MAX_SLUG_LENGTH,
    Project,
    ProjectSource,
    normalize_slug,
    project_id,
)

__all__ = ["ProjectService", "build_project_service"]

_log = get_logger(__name__)


class ProjectService:
    """Create projects, list them, and decide which project a source belongs to."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    # ------------------------------------------------------------- writing --

    async def create(
        self,
        *,
        name: str,
        slug: str | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Project:
        """Create a project, deriving the slug from the name when none is given.

        The uniqueness check is a real insert that may fail, not a `SELECT` first.
        Checking then inserting is a race: two `omnisense init` runs in different
        terminals both see the slug free and both proceed, and the loser gets an
        `IntegrityError` from three frames deeper with no useful message. Letting
        the constraint decide means there is exactly one answer and it is the
        database's.
        """
        resolved_slug = slug or normalize_slug(name)
        if not resolved_slug:
            raise ValidationError(
                f"cannot derive a project slug from {name!r}; supply one explicitly",
                details={"name": name},
            )
        if len(resolved_slug) > MAX_SLUG_LENGTH:
            raise ValidationError(
                f"slug {resolved_slug!r} is longer than {MAX_SLUG_LENGTH} characters"
            )

        # Constructed before the insert so an invalid slug is rejected by the
        # model's own validator, with its message, rather than by a database
        # constraint that can only say "check failed".
        project = Project(
            id=project_id(self._tenant_id, resolved_slug),
            tenant_id=self._tenant_id,
            slug=resolved_slug,
            name=name,
            description=description,
            metadata=metadata or {},
        )

        async with self._session_factory() as session:
            session.add(
                ProjectRow(
                    id=project.id,
                    tenant_id=project.tenant_id,
                    slug=project.slug,
                    name=project.name,
                    description=project.description,
                    is_active=project.is_active,
                    project_metadata=project.metadata,
                )
            )
            try:
                await session.commit()
            except IntegrityError as error:
                await session.rollback()
                raise ConflictError(
                    f"a project with slug {resolved_slug!r} already exists. "
                    f"Use it with --project {resolved_slug}, or choose another slug.",
                    details={"slug": resolved_slug},
                ) from error

        _log.info("project.created", project_id=project.id, slug=project.slug)
        return project

    async def attach_source(self, *, slug: str, source_id: str) -> ProjectSource:
        """Put a source in a project, or move it from another one.

        Moving is allowed and is not a special case: a repository reassigned to a
        different product is an ordinary thing that happens, and refusing it would
        mean deleting and re-adding the source, which would take its artifacts
        with it. The previous owner is logged so the move is visible afterwards.
        """
        async with self._session_factory() as session:
            project = await self._require_project(session, slug)
            source = await session.get(SourceRow, source_id)
            if source is None:
                raise NotFoundError.for_resource("source", source_id)
            if source.tenant_id != self._tenant_id:
                # Reported as absent rather than forbidden: confirming a source
                # exists in another tenant is itself a disclosure.
                raise NotFoundError.for_resource("source", source_id)

            previous = source.project_id
            source.project_id = project.id
            await session.commit()
            # `TimestampMixin.updated_at` carries `onupdate=`, so the commit
            # expires it and the next attribute read triggers a lazy refresh.
            # Under async SQLAlchemy that is not a query, it is a
            # `MissingGreenlet` -- raised from the attribute access rather than
            # from anything that looks like IO. Refreshing explicitly costs one
            # SELECT on a write path and makes the row safe to read.
            await session.refresh(source)

            if previous and previous != project.id:
                _log.info(
                    "project.source_moved",
                    source_id=source_id,
                    from_project=previous,
                    to_project=project.id,
                )
            return _as_project_source(source)

    async def detach_source(self, *, source_id: str) -> None:
        """Remove a source from its project without deleting either.

        The source keeps its artifacts; they simply stop answering project-scoped
        questions. That is the honest outcome of "this repository is not part of
        that product any more" -- the work still happened.
        """
        async with self._session_factory() as session:
            source = await session.get(SourceRow, source_id)
            if source is None or source.tenant_id != self._tenant_id:
                raise NotFoundError.for_resource("source", source_id)
            source.project_id = None
            await session.commit()

    async def set_active(self, *, slug: str, is_active: bool) -> Project:
        """Pause or resume a project. Never deletes anything.

        There is deliberately no `delete`. Removing a project would either orphan
        its sources or cascade into their history, and "we stopped working on
        this" is a different fact from "this never happened".
        """
        async with self._session_factory() as session:
            row = await self._require_project(session, slug)
            row.is_active = is_active
            await session.commit()
            await session.refresh(row)  # see `attach_source` for why
            return _as_project(row)

    # ------------------------------------------------------------- reading --

    async def get(self, slug: str) -> Project:
        async with self._session_factory() as session:
            return _as_project(await self._require_project(session, slug))

    async def list(self, *, include_inactive: bool = False) -> list[Project]:
        async with self._session_factory() as session:
            statement = select(ProjectRow).where(ProjectRow.tenant_id == self._tenant_id)
            if not include_inactive:
                statement = statement.where(ProjectRow.is_active.is_(True))
            rows = (await session.execute(statement.order_by(ProjectRow.slug))).scalars().all()
            return [_as_project(row) for row in rows]

    async def sources(self, slug: str) -> list[ProjectSource]:
        """Every source in a project, with how many artifacts each holds.

        The count comes from one grouped subquery rather than a query per source.
        A project with a dozen repositories would otherwise issue thirteen
        round trips to render one screen, and that is the shape of loop that only
        shows up as slowness once somebody has real data.
        """
        async with self._session_factory() as session:
            project = await self._require_project(session, slug)

            counts = (
                select(
                    ArtifactRow.source_id.label("source_id"),
                    func.count().label("total"),
                )
                .group_by(ArtifactRow.source_id)
                .subquery()
            )
            rows = (
                await session.execute(
                    select(SourceRow, func.coalesce(counts.c.total, 0))
                    .outerjoin(counts, counts.c.source_id == SourceRow.id)
                    .where(SourceRow.project_id == project.id)
                    .order_by(SourceRow.name)
                )
            ).all()
            return [_as_project_source(source, count) for source, count in rows]

    async def unassigned_sources(self) -> list[ProjectSource]:
        """Sources belonging to no project.

        Worth its own method because it is the thing `omnisense init` shows next:
        a connector has discovered repositories and nobody has said what they are
        part of. Without it they are invisible -- present, collecting artifacts,
        and absent from every project-scoped answer.
        """
        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(SourceRow)
                        .where(
                            SourceRow.tenant_id == self._tenant_id,
                            SourceRow.project_id.is_(None),
                        )
                        .order_by(SourceRow.name)
                    )
                )
                .scalars()
                .all()
            )
            return [_as_project_source(row) for row in rows]

    async def resolve_source_ids(self, slug: str) -> list[str]:
        """The source ids a project owns, for callers that query artifacts directly.

        This is what makes "everything in this project" one query: the caller gets
        the ids and filters artifacts by them, rather than joining through
        `sources` on a table that will be the largest in the system.
        """
        async with self._session_factory() as session:
            project = await self._require_project(session, slug)
            return list(
                (
                    await session.execute(
                        select(SourceRow.id).where(SourceRow.project_id == project.id)
                    )
                )
                .scalars()
                .all()
            )

    # ------------------------------------------------------------ internal --

    async def _require_project(self, session: AsyncSession, slug: str) -> ProjectRow:
        row = (
            await session.execute(
                select(ProjectRow).where(
                    ProjectRow.tenant_id == self._tenant_id,
                    ProjectRow.slug == slug,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            raise NotFoundError.for_resource("project", slug)
        return row


def _as_project(row: ProjectRow) -> Project:
    return Project(
        id=row.id,
        tenant_id=row.tenant_id,
        slug=row.slug,
        name=row.name,
        description=row.description,
        is_active=row.is_active,
        created_at=row.created_at,
        updated_at=row.updated_at,
        metadata=dict(row.project_metadata or {}),
    )


def _as_project_source(row: SourceRow, artifact_count: int = 0) -> ProjectSource:
    return ProjectSource(
        source_id=row.id,
        project_id=row.project_id,
        platform=row.platform.value,
        name=row.name,
        display_name=row.display_name,
        url=row.url,
        is_active=row.is_active,
        artifact_count=artifact_count,
    )


def build_project_service(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    *,
    tenant_id: str = DEFAULT_TENANT,
) -> ProjectService:
    """Construct the service against the process-wide session factory."""
    if session_factory is None:
        from backend.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
    return ProjectService(session_factory, tenant_id=tenant_id)


def project_sources_summary(sources: Sequence[ProjectSource]) -> str:
    """A one-line summary for the CLI: `3 sources, 2,970 artifacts`."""
    total = sum(source.artifact_count for source in sources)
    return f"{len(sources)} source{'s' if len(sources) != 1 else ''}, {total:,} artifacts"
