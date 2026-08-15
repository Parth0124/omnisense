"""Find everything you work on, then wait to be told what to actually read.

The premise this exists to serve: you should not have to tell the system where to
look. You ask what happened, and it already knows about every repository you
commit to, every channel you are in, every board you are on.

Which is only true if it goes and finds them -- and finding them is the easy
half. A GitHub token with normal affiliations sees the tutorial you forked in
2021, the repository somebody added you to for one review, and four archived
services nobody has touched since. Ingesting all of it costs API budget, fills
the database with noise, and makes "what happened last week" longer and worse.

So discovery **proposes and stops**. Everything it finds lands as `PENDING`,
nothing is read from a pending source, and a person decides. That is the same
shape as `identity_service.suggest` and `feature_service.decide`, for the same
reason each time: the system is allowed to be confident about what exists and
never about what matters.

Rejections are permanent
------------------------
An excluded source is kept, not deleted. Discovery runs again next week, finds
the same forked tutorial, and must not offer it again -- somebody who rejects the
same five repositories every week stops reading the list, and then the one that
mattered goes past unread too.

Layer note: **L2 service.**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.exceptions import NotFoundError
from backend.core.logging import get_logger
from connectors.exceptions import ConnectorError
from connectors.github.reader import GitHubReader, parse_time
from models.artifact import WatchStatus, source_id
from models.enums import Platform
from models.orm.artifact import SourceRow

__all__ = [
    "Candidate",
    "DiscoveryReport",
    "DiscoveryService",
    "build_discovery_service",
]

_log = get_logger(__name__)

DEFAULT_TENANT = "local"

QUIET_AFTER_DAYS = 365
"""Beyond this, a repository is proposed but flagged as dormant.

Not filtered out -- a year-old repository is exactly the thing somebody might
want a briefing about when they come back to it. Flagged, so a review queue of
sixty can be skimmed by the one column that separates "still moving" from
"finished in 2024".
"""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One thing discovery found, and enough to decide about it in a second."""

    source_id: str
    platform: Platform
    name: str
    status: WatchStatus

    private: bool = False
    archived: bool = False
    last_activity: datetime | None = None
    description: str | None = None

    @property
    def is_dormant(self) -> bool:
        if self.last_activity is None:
            return True
        last = self.last_activity
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        return (datetime.now(UTC) - last).days > QUIET_AFTER_DAYS


@dataclass(frozen=True, slots=True)
class DiscoveryReport:
    found: int = 0
    new: int = 0

    still_pending: int = 0
    """Seen before and *still* waiting on somebody.

    Counted apart from `already_decided`, which it was originally folded into.
    The merged number read "109 already decided" on a run where 107 of them had
    never been looked at -- an encouraging sentence describing an untouched
    backlog, which is worse than no sentence.
    """

    already_decided: int = 0
    previously_excluded: int = 0
    """Found again and left alone. Counted so the number is visible rather than
    looking like discovery quietly missed them."""
    error: str | None = None


class DiscoveryService:
    """Finds sources. Never reads from one nobody has approved."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        reader: GitHubReader,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._reader = reader
        self._tenant_id = tenant_id

    # ------------------------------------------------------------ finding --

    async def discover_github(self, *, include_forks: bool = False) -> DiscoveryReport:
        """Walk everything the token can see and record what is new.

        Writes only `PENDING` rows for things not seen before. A source that is
        already `INCLUDED` or `EXCLUDED` is left exactly as it is -- discovery
        has no business revising a decision somebody made.
        """
        found = new = pending = decided = excluded = 0
        failure: str | None = None

        # The `try` sits *inside* the transaction, not around it. Wrapped around,
        # a rate limit raised on the sixtieth repository escapes `session.begin()`
        # first -- which rolls back -- and only then reaches the handler, so the
        # report cheerfully announced "kept 59" over an empty table. Catching it
        # within the block lets the commit run on the way out with everything read
        # so far still in it.
        async with self._session_factory() as session, session.begin():
            try:
                async for payload in self._reader.viewer_repositories(include_forks=include_forks):
                    node_id = payload.get("node_id")
                    name = payload.get("full_name")
                    if not node_id or not name:
                        continue

                    found += 1
                    identifier = source_id(Platform.GITHUB, str(node_id))
                    existing = await session.get(SourceRow, identifier)

                    if existing is not None:
                        if existing.watch_status is WatchStatus.EXCLUDED:
                            excluded += 1
                        elif existing.watch_status is WatchStatus.PENDING:
                            pending += 1
                        else:
                            decided += 1
                        # The name is refreshed even for a decided source: a
                        # renamed repository keeps its node id, and leaving the
                        # old name would make the review queue and every later
                        # report refer to something that no longer exists.
                        existing.name = str(name)
                        continue

                    session.add(
                        SourceRow(
                            id=identifier,
                            tenant_id=self._tenant_id,
                            platform=Platform.GITHUB,
                            external_id=str(node_id),
                            name=str(name),
                            url=payload.get("html_url"),
                            default_branch=payload.get("default_branch"),
                            watch_status=WatchStatus.PENDING,
                            source_metadata={
                                "private": bool(payload.get("private")),
                                "archived": bool(payload.get("archived")),
                                "pushed_at": payload.get("pushed_at"),
                                "description": (payload.get("description") or "")[:500],
                            },
                        )
                    )
                    new += 1
            except ConnectorError as error:
                # Reported rather than raised: discovery that found sixty
                # repositories and then hit the rate limit has still done sixty
                # repositories of useful work, and throwing that away would mean
                # starting over against the same wall.
                failure = str(error)
                _log.warning("discovery.failed", error=failure)

        if failure is None:
            _log.info("discovery.completed", found=found, new=new)
        return DiscoveryReport(
            found=found,
            new=new,
            still_pending=pending,
            already_decided=decided,
            previously_excluded=excluded,
            error=failure,
        )

    # ----------------------------------------------------------- reviewing --

    async def candidates(self, status: WatchStatus | None = None) -> list[Candidate]:
        """Sources, newest activity first. Filtered by status when one is given."""
        async with self._session_factory() as session:
            statement = select(SourceRow).where(SourceRow.tenant_id == self._tenant_id)
            if status is not None:
                statement = statement.where(SourceRow.watch_status == status)
            rows = list((await session.execute(statement)).scalars().all())

        candidates = [
            Candidate(
                source_id=row.id,
                platform=row.platform,
                name=row.name,
                status=row.watch_status,
                private=bool((row.source_metadata or {}).get("private")),
                archived=bool((row.source_metadata or {}).get("archived")),
                last_activity=parse_time((row.source_metadata or {}).get("pushed_at")),
                description=(row.source_metadata or {}).get("description") or None,
            )
            for row in rows
        ]
        # Newest first, and undated last rather than first: `None` sorting to the
        # top would put the least informative rows where the eye lands.
        return sorted(
            candidates,
            key=lambda c: (
                c.last_activity is not None,
                c.last_activity or datetime.min.replace(tzinfo=UTC),
            ),
            reverse=True,
        )

    async def decide(self, *, source: str, include: bool) -> Candidate:
        """Include or exclude one source. Accepts a name or an id prefix."""
        async with self._session_factory() as session, session.begin():
            row = await self._resolve(session, source)
            row.watch_status = WatchStatus.INCLUDED if include else WatchStatus.EXCLUDED
            resolved = Candidate(
                source_id=row.id,
                platform=row.platform,
                name=row.name,
                status=row.watch_status,
            )

        _log.info("discovery.decided", source_id=resolved.source_id, include=include)
        return resolved

    async def decide_all_pending(self, *, include: bool) -> int:
        """Settle everything still waiting. The bulk escape hatch.

        Exists because a first discovery on a real account can surface sixty
        repositories, and requiring sixty commands to say "yes, all of them" is
        how somebody abandons the review and goes back to adding repositories by
        hand.
        """
        async with self._session_factory() as session, session.begin():
            rows = list(
                (
                    await session.execute(
                        select(SourceRow).where(
                            SourceRow.tenant_id == self._tenant_id,
                            SourceRow.watch_status == WatchStatus.PENDING,
                        )
                    )
                )
                .scalars()
                .all()
            )
            for row in rows:
                row.watch_status = WatchStatus.INCLUDED if include else WatchStatus.EXCLUDED

        _log.info("discovery.bulk_decided", count=len(rows), include=include)
        return len(rows)

    async def included_source_ids(self) -> list[str]:
        """What sync is allowed to read. The only caller that matters."""
        async with self._session_factory() as session:
            return list(
                (
                    await session.execute(
                        select(SourceRow.id).where(
                            SourceRow.tenant_id == self._tenant_id,
                            SourceRow.watch_status == WatchStatus.INCLUDED,
                        )
                    )
                )
                .scalars()
                .all()
            )

    async def _resolve(self, session: AsyncSession, reference: str) -> SourceRow:
        """A source from its full name, or from an id prefix.

        Names first, because `owner/repo` is what a person reads off the review
        queue and typing an id when a name is on screen is friction with no
        purpose.
        """
        cleaned = reference.strip()
        if not cleaned:
            raise NotFoundError.for_resource("source", reference)

        by_name = (
            await session.execute(
                select(SourceRow).where(
                    SourceRow.tenant_id == self._tenant_id, SourceRow.name == cleaned
                )
            )
        ).scalar_one_or_none()
        if by_name is not None:
            return by_name

        matches = list(
            (
                await session.execute(
                    select(SourceRow)
                    .where(
                        SourceRow.tenant_id == self._tenant_id,
                        SourceRow.id.startswith(cleaned),
                    )
                    .limit(2)
                )
            )
            .scalars()
            .all()
        )
        if len(matches) == 1:
            return matches[0]
        raise NotFoundError.for_resource("source", reference)


def build_discovery_service(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    *,
    token: str | None = None,
    tenant_id: str = DEFAULT_TENANT,
) -> DiscoveryService:
    if session_factory is None:
        from backend.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
    return DiscoveryService(session_factory, reader=GitHubReader(token=token), tenant_id=tenant_id)
