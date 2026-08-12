"""Write artifacts and the people they reference, idempotently.

Every write here is an upsert on the platform's own identity, which is what makes
a sync safe to run twice, safe to interrupt, and safe to resume. That property is
worth more than it sounds: a backfill that exhausts the rate limit halfway is the
*normal* outcome on a large repository, and it has to be restartable without
either duplicating what it already read or requiring a transaction that stays
open for an hour.

**People are written before the artifacts that reference them.** A commit's
`actor_id` is a foreign key, and inserting the commit first fails on it. That
ordering is not incidental -- it is why this module takes artifacts and people
together rather than offering two independent methods a caller could get the
wrong way round.

**A person is upserted, never merely inserted.** The same author appears on
hundreds of commits in one batch and on thousands across syncs; inserting would
conflict on all but the first, and skipping when present would leave a display
name that has since changed frozen at whatever it was on first sight.

Layer note: **L2 service** -- imports `models/` and the kernel. It does not know
where artifacts come from, which is what lets Slack and arXiv reuse it unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.logging import get_logger
from models.artifact import Artifact, Person
from models.orm.artifact import ArtifactRow, PersonRow

__all__ = ["ArtifactStore", "WriteResult", "build_artifact_store"]

_log = get_logger(__name__)

BATCH_SIZE = 500
"""Rows per statement.

Not a tuning knob so much as a ceiling. PostgreSQL's protocol caps a statement at
65535 bound parameters, and an artifact binds roughly eighteen -- so a batch much
above three thousand fails outright, on a large repository, in the middle of a
backfill. Five hundred leaves room for the row to grow columns.
"""


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What a write actually did. Returned so a sync can report honestly."""

    artifacts_written: int = 0
    people_written: int = 0

    def __add__(self, other: WriteResult) -> WriteResult:
        return WriteResult(
            self.artifacts_written + other.artifacts_written,
            self.people_written + other.people_written,
        )


def _chunk(
    items: Sequence[dict[str, Any]], size: int = BATCH_SIZE
) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class ArtifactStore:
    """Upserts artifacts and people. The only writer of either table."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def write(
        self, artifacts: Sequence[Artifact], people: Sequence[Person] = ()
    ) -> WriteResult:
        """Write a batch. People first, then artifacts, in one transaction.

        One transaction because the two are only correct together: committing the
        people and then failing on the artifacts leaves rows nothing references,
        and committing artifacts whose people failed is impossible anyway -- the
        foreign key would refuse it.
        """
        if not artifacts and not people:
            return WriteResult()

        async with self._session_factory() as session:
            written_people = await self._write_people(session, people)
            written_artifacts = await self._write_artifacts(session, artifacts)
            await session.commit()

        return WriteResult(artifacts_written=written_artifacts, people_written=written_people)

    async def _write_people(self, session: AsyncSession, people: Sequence[Person]) -> int:
        if not people:
            return 0

        # Deduplicated in memory first. The same author appears on most commits in
        # a batch, and PostgreSQL refuses an `ON CONFLICT` statement that touches
        # the same row twice -- "cannot affect row a second time" -- so a batch
        # with one author and a hundred commits would fail outright.
        unique = {person.id: person for person in people}
        rows = [
            {
                "id": person.id,
                "tenant_id": person.tenant_id,
                "platform": person.platform,
                "external_id": person.external_id,
                "handle": person.handle,
                "display_name": person.display_name,
                "email": person.email,
                "avatar_url": person.avatar_url,
                "is_bot": person.is_bot,
                "person_metadata": dict(person.metadata),
            }
            for person in unique.values()
        ]

        for batch in _chunk(rows):
            statement = _upsert(session, PersonRow, batch)
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=[PersonRow.id],
                    # Refreshed rather than left alone: a display name or avatar
                    # that changed since the first sighting should follow, and a
                    # `DO NOTHING` would freeze both at whatever they were the
                    # first time this person appeared.
                    set_={
                        "handle": statement.excluded.handle,
                        "display_name": statement.excluded.display_name,
                        "avatar_url": statement.excluded.avatar_url,
                        "is_bot": statement.excluded.is_bot,
                    },
                )
            )
        return len(rows)

    async def _write_artifacts(self, session: AsyncSession, artifacts: Sequence[Artifact]) -> int:
        if not artifacts:
            return 0

        unique = {artifact.id: artifact for artifact in artifacts}
        rows = [
            {
                "id": artifact.id,
                "tenant_id": artifact.tenant_id,
                "kind": artifact.kind,
                "source_id": artifact.source_id,
                "actor_id": artifact.actor_id,
                "platform": artifact.platform,
                "native_id": artifact.native_id,
                "url": artifact.url,
                "title": artifact.title,
                "body": artifact.body,
                "occurred_at": artifact.occurred_at,
                "updated_at_source": artifact.updated_at_source,
                "state": artifact.state,
                "outcome": artifact.outcome,
                "links": [link.model_dump(mode="json") for link in artifact.links],
                "details": artifact.details.model_dump(mode="json") if artifact.details else None,
                "provenance": artifact.provenance.model_dump(mode="json"),
                "artifact_metadata": dict(artifact.metadata),
            }
            for artifact in unique.values()
        ]

        for batch in _chunk(rows):
            statement = _upsert(session, ArtifactRow, batch)
            await session.execute(
                statement.on_conflict_do_update(
                    index_elements=[ArtifactRow.id],
                    # Everything mutable is refreshed, because most of these
                    # genuinely change: a pull request opens, gets reviewed,
                    # gains commits and merges, and every sync sees a different
                    # state for the same row.
                    #
                    # `occurred_at` is *not* in this list. When a thing happened
                    # does not change, and letting a later read move it would let
                    # one bad payload silently relocate history.
                    set_={
                        "title": statement.excluded.title,
                        "body": statement.excluded.body,
                        "url": statement.excluded.url,
                        "updated_at_source": statement.excluded.updated_at_source,
                        "state": statement.excluded.state,
                        "outcome": statement.excluded.outcome,
                        "links": statement.excluded.links,
                        "details": statement.excluded.details,
                        "provenance": statement.excluded.provenance,
                        "artifact_metadata": statement.excluded.artifact_metadata,
                        "actor_id": statement.excluded.actor_id,
                    },
                )
            )
        return len(rows)


def _upsert(session: AsyncSession, model: type[Any], rows: Sequence[dict[str, Any]]) -> Any:
    """The dialect's `INSERT ... ON CONFLICT`.

    PostgreSQL and SQLite spell it identically at the API but expose it from
    different modules, and the generic `insert()` has no `on_conflict_do_update`
    at all. Chosen from the live connection rather than from configuration so the
    unit suite -- which runs on SQLite by design -- exercises this same code
    rather than a second path that only looks like it.
    """
    dialect = session.bind.dialect.name if session.bind is not None else "postgresql"
    insert = sqlite_insert if dialect == "sqlite" else pg_insert
    return insert(model).values(list(rows))


def build_artifact_store(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> ArtifactStore:
    if session_factory is None:
        from backend.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
    return ArtifactStore(session_factory)
