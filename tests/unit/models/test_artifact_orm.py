"""The three artifact tables: round-trip, referential integrity, and the JSON edges.

The domain tests next door check the model. These check that it survives the
database -- which is a different question, and the one where the failures are
expensive: a model that validates and then loses its `details` on the way through
`JSONVariant` produces rows that look right in a test and are empty in production.

Run against SQLite from `tests/conftest.py`, so they are fast and need nothing
running. `models/orm/base.py` was written for that: `JSONVariant` compiles to
portable JSON, so the mapping under test is the real one rather than a stand-in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from models.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactLink,
    ArtifactOutcome,
    ArtifactProvenance,
    ArtifactState,
    CIRunDetails,
    CommitDetails,
    JobResult,
    LinkRelation,
    artifact_id,
    person_id,
    source_id,
)
from models.enums import Platform
from models.orm.artifact import ArtifactRow, PersonRow, SourceRow

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 11, 9, 0, tzinfo=UTC)

REPO_EXTERNAL = "R_kgDOABCD1M"
USER_EXTERNAL = "MDQ6VXNlcjE0ODEyMzM="


async def seed_refs(session: AsyncSession) -> tuple[str, str]:
    """A source and a person for artifacts to point at."""
    src = source_id(Platform.GITHUB, REPO_EXTERNAL)
    per = person_id(Platform.GITHUB, USER_EXTERNAL)
    session.add(
        SourceRow(
            id=src,
            platform=Platform.GITHUB,
            external_id=REPO_EXTERNAL,
            name="omnisense/api",
            default_branch="main",
        )
    )
    session.add(
        PersonRow(
            id=per,
            platform=Platform.GITHUB,
            external_id=USER_EXTERNAL,
            handle="dsokolov",
        )
    )
    await session.commit()
    return src, per


def row_from(model: Artifact) -> ArtifactRow:
    """Map the domain object onto the row, the way a store will.

    Written out rather than hidden behind a helper because the mapping is the
    thing under test: a field dropped here is a field silently lost on write.
    """
    return ArtifactRow(
        id=model.id,
        kind=model.kind,
        source_id=model.source_id,
        actor_id=model.actor_id,
        platform=model.platform,
        native_id=model.native_id,
        url=model.url,
        title=model.title,
        body=model.body,
        occurred_at=model.occurred_at,
        updated_at_source=model.updated_at_source,
        state=model.state,
        outcome=model.outcome,
        links=[link.model_dump(mode="json") for link in model.links],
        details=model.details.model_dump(mode="json") if model.details else None,
        provenance=model.provenance.model_dump(mode="json"),
        artifact_metadata=dict(model.metadata),
    )


class TestRoundTrip:
    async def test_a_ci_run_survives_the_database_intact(self, db_session: AsyncSession) -> None:
        """The full path: model -> row -> database -> row -> model.

        A CI run because it is the kind with the most structure -- nested jobs,
        an outcome, and links -- so it exercises every JSON column at once.
        """
        src, _ = await seed_refs(db_session)
        model = Artifact(
            id=artifact_id(Platform.GITHUB, "WFR_1"),
            kind=ArtifactKind.CI_RUN,
            source_id=src,
            actor_id=None,
            platform=Platform.GITHUB,
            native_id="WFR_1",
            title="Run tests",
            occurred_at=NOW,
            state=ArtifactState.COMPLETED,
            outcome=ArtifactOutcome.FAILURE,
            details=CIRunDetails(
                workflow_name="test.yml",
                head_sha="a3f9c1",
                duration_seconds=420.0,
                jobs=[
                    JobResult(name="unit", outcome=ArtifactOutcome.SUCCESS),
                    JobResult(name="integration", outcome=ArtifactOutcome.FAILURE),
                ],
            ),
            links=[ArtifactLink(relation=LinkRelation.TESTS, target_native_id="a3f9c1")],
            provenance=ArtifactProvenance(connector_slug="github", fetched_at=NOW),
            metadata={"github.repository": "omnisense/api"},
        )
        db_session.add(row_from(model))
        await db_session.commit()

        stored = await db_session.get(ArtifactRow, model.id)
        assert stored is not None

        restored = Artifact(
            id=stored.id,
            kind=stored.kind,
            source_id=stored.source_id,
            actor_id=stored.actor_id,
            platform=stored.platform,
            native_id=stored.native_id,
            title=stored.title,
            occurred_at=stored.occurred_at,
            state=stored.state,
            outcome=stored.outcome,
            links=[ArtifactLink.model_validate(link) for link in stored.links],
            details=stored.details,
            provenance=ArtifactProvenance.model_validate(stored.provenance),
            metadata=stored.artifact_metadata,
        )
        assert restored.details is not None
        assert restored.details.failed_jobs == ["integration"]  # type: ignore[union-attr]
        assert restored.failed
        assert restored.links[0].relation is LinkRelation.TESTS
        assert restored.metadata["github.repository"] == "omnisense/api"

    async def test_an_artifact_with_no_details_stores_null(self, db_session: AsyncSession) -> None:
        """Papers, messages and agent runs have no detail class. `None` must reach
        the column as NULL rather than as an empty object -- `{}` would parse back
        as a details payload with no `kind` and fail the discriminator."""
        src, _ = await seed_refs(db_session)
        model = Artifact(
            id=artifact_id(Platform.ARXIV, "2408.01234v1"),
            kind=ArtifactKind.PAPER,
            source_id=src,
            platform=Platform.ARXIV,
            native_id="2408.01234v1",
            title="A paper",
            occurred_at=NOW,
            provenance=ArtifactProvenance(connector_slug="arxiv", fetched_at=NOW),
        )
        db_session.add(row_from(model))
        await db_session.commit()

        stored = await db_session.get(ArtifactRow, model.id)
        assert stored is not None
        assert stored.details is None


class TestIdentityInTheDatabase:
    async def test_the_same_object_twice_is_rejected_not_duplicated(
        self, db_session: AsyncSession
    ) -> None:
        """`(platform, native_id)` is unique, which is what makes a re-sync safe:
        the second write conflicts and becomes an upsert rather than a second row
        for the same commit."""
        src, _ = await seed_refs(db_session)
        for _ in range(2):
            db_session.add(
                ArtifactRow(
                    id=artifact_id(Platform.GITHUB, "dup"),
                    kind=ArtifactKind.COMMIT,
                    source_id=src,
                    platform=Platform.GITHUB,
                    native_id="dup",
                    occurred_at=NOW,
                )
            )
        with pytest.raises(IntegrityError):
            await db_session.commit()

    async def test_two_platforms_may_share_a_native_id(self, db_session: AsyncSession) -> None:
        """Native ids are only unique within a platform. Jira `1699` and Slack
        `1699` are different things and both must be storable."""
        src, _ = await seed_refs(db_session)
        for platform in (Platform.JIRA, Platform.SLACK):
            db_session.add(
                ArtifactRow(
                    id=artifact_id(platform, "1699"),
                    kind=ArtifactKind.MESSAGE,
                    source_id=src,
                    platform=platform,
                    native_id="1699",
                    occurred_at=NOW,
                )
            )
        await db_session.commit()
        rows = (await db_session.execute(select(ArtifactRow))).scalars().all()
        assert len(rows) == 2


class TestReferentialIntegrity:
    async def test_an_artifact_cannot_point_at_a_source_that_does_not_exist(
        self, db_session: AsyncSession
    ) -> None:
        """An artifact whose origin is unknown cannot be cited, and a citation is
        what every claim in this system rests on."""
        db_session.add(
            ArtifactRow(
                id=artifact_id(Platform.GITHUB, "orphan"),
                kind=ArtifactKind.COMMIT,
                source_id="src_does_not_exist",
                platform=Platform.GITHUB,
                native_id="orphan",
                occurred_at=NOW,
            )
        )
        with pytest.raises(IntegrityError):
            await db_session.commit()

    async def test_an_artifact_may_have_no_actor(self, db_session: AsyncSession) -> None:
        """Common and meaningful: a CI run is triggered by a machine, and a deleted
        account leaves its commits behind."""
        src, _ = await seed_refs(db_session)
        db_session.add(
            ArtifactRow(
                id=artifact_id(Platform.GITHUB, "machine"),
                kind=ArtifactKind.CI_RUN,
                source_id=src,
                actor_id=None,
                platform=Platform.GITHUB,
                native_id="machine",
                occurred_at=NOW,
            )
        )
        await db_session.commit()
        stored = await db_session.get(ArtifactRow, artifact_id(Platform.GITHUB, "machine"))
        assert stored is not None
        assert stored.actor_id is None

    async def test_a_renamed_source_keeps_every_artifact(self, db_session: AsyncSession) -> None:
        """The entire reason sources are a separate table.

        Renaming a repository updates one row. If the name lived on each artifact
        this would be a rewrite of every row, and the ones missed would point at a
        repository that no longer exists under that name.
        """
        src, _ = await seed_refs(db_session)
        db_session.add(
            ArtifactRow(
                id=artifact_id(Platform.GITHUB, "c1"),
                kind=ArtifactKind.COMMIT,
                source_id=src,
                platform=Platform.GITHUB,
                native_id="c1",
                occurred_at=NOW,
            )
        )
        await db_session.commit()

        source = await db_session.get(SourceRow, src)
        assert source is not None
        source.name = "omnisense/backend"
        await db_session.commit()

        artifact_row = await db_session.get(ArtifactRow, artifact_id(Platform.GITHUB, "c1"))
        assert artifact_row is not None
        assert artifact_row.source_id == src

    async def test_a_source_with_artifacts_cannot_be_deleted(
        self, db_session: AsyncSession
    ) -> None:
        """`RESTRICT`, not `CASCADE`. Removing a source is a decision about what
        happens to its history, and it should have to be made explicitly rather
        than taken silently as a side effect."""
        src, _ = await seed_refs(db_session)
        db_session.add(
            ArtifactRow(
                id=artifact_id(Platform.GITHUB, "keep"),
                kind=ArtifactKind.COMMIT,
                source_id=src,
                platform=Platform.GITHUB,
                native_id="keep",
                occurred_at=NOW,
            )
        )
        await db_session.commit()

        source = await db_session.get(SourceRow, src)
        assert source is not None
        await db_session.delete(source)
        with pytest.raises(IntegrityError):
            await db_session.commit()


class TestTheQueriesTheIndexesExistFor:
    """If these read awkwardly, the columns are wrong -- so they are worth writing."""

    async def test_a_time_window_over_every_kind_is_one_query(
        self, db_session: AsyncSession
    ) -> None:
        """`catch-up`, in its simplest form. The reason there is one table."""
        src, per = await seed_refs(db_session)
        kinds = [
            ArtifactKind.COMMIT,
            ArtifactKind.PULL_REQUEST,
            ArtifactKind.CI_RUN,
            ArtifactKind.MESSAGE,
            ArtifactKind.PAPER,
        ]
        for index, kind in enumerate(kinds):
            db_session.add(
                ArtifactRow(
                    id=artifact_id(Platform.GITHUB, f"w{index}"),
                    kind=kind,
                    source_id=src,
                    actor_id=per,
                    platform=Platform.GITHUB,
                    native_id=f"w{index}",
                    occurred_at=NOW - timedelta(days=index),
                )
            )
        # Older than the window, and must not appear.
        db_session.add(
            ArtifactRow(
                id=artifact_id(Platform.GITHUB, "old"),
                kind=ArtifactKind.COMMIT,
                source_id=src,
                platform=Platform.GITHUB,
                native_id="old",
                occurred_at=NOW - timedelta(days=60),
            )
        )
        await db_session.commit()

        rows = (
            (
                await db_session.execute(
                    select(ArtifactRow)
                    .where(
                        ArtifactRow.source_id == src,
                        ArtifactRow.occurred_at >= NOW - timedelta(days=7),
                    )
                    .order_by(ArtifactRow.occurred_at.desc())
                )
            )
            .scalars()
            .all()
        )
        assert len(rows) == len(kinds)
        assert {row.kind for row in rows} == set(kinds)
        assert rows[0].occurred_at >= rows[-1].occurred_at

    async def test_what_failed_is_one_predicate_across_kinds(
        self, db_session: AsyncSession
    ) -> None:
        """A failed CI run and a rejected review answer the same question, and
        `outcome` is what makes that a single `WHERE`."""
        src, _ = await seed_refs(db_session)
        rows = [
            (ArtifactKind.CI_RUN, ArtifactOutcome.FAILURE),
            (ArtifactKind.CI_RUN, ArtifactOutcome.SUCCESS),
            (ArtifactKind.REVIEW, ArtifactOutcome.CHANGES_REQUESTED),
            (ArtifactKind.COMMIT, None),
        ]
        for index, (kind, outcome) in enumerate(rows):
            db_session.add(
                ArtifactRow(
                    id=artifact_id(Platform.GITHUB, f"f{index}"),
                    kind=kind,
                    source_id=src,
                    platform=Platform.GITHUB,
                    native_id=f"f{index}",
                    occurred_at=NOW,
                    outcome=outcome,
                )
            )
        await db_session.commit()

        failed = (
            (
                await db_session.execute(
                    select(ArtifactRow).where(
                        ArtifactRow.outcome.in_(
                            [ArtifactOutcome.FAILURE, ArtifactOutcome.TIMED_OUT]
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(failed) == 1
        assert failed[0].kind is ArtifactKind.CI_RUN

    async def test_what_is_still_open_ignores_finished_work(self, db_session: AsyncSession) -> None:
        src, _ = await seed_refs(db_session)
        for index, state in enumerate(
            [ArtifactState.OPEN, ArtifactState.MERGED, ArtifactState.DRAFT]
        ):
            db_session.add(
                ArtifactRow(
                    id=artifact_id(Platform.GITHUB, f"s{index}"),
                    kind=ArtifactKind.PULL_REQUEST,
                    source_id=src,
                    platform=Platform.GITHUB,
                    native_id=f"s{index}",
                    occurred_at=NOW,
                    state=state,
                )
            )
        await db_session.commit()

        open_rows = (
            (
                await db_session.execute(
                    select(ArtifactRow).where(
                        ArtifactRow.state.in_([ArtifactState.OPEN, ArtifactState.DRAFT])
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(open_rows) == 2

    async def test_everything_one_person_did_spans_kinds(self, db_session: AsyncSession) -> None:
        src, per = await seed_refs(db_session)
        for index, kind in enumerate([ArtifactKind.COMMIT, ArtifactKind.REVIEW]):
            db_session.add(
                ArtifactRow(
                    id=artifact_id(Platform.GITHUB, f"p{index}"),
                    kind=kind,
                    source_id=src,
                    actor_id=per,
                    platform=Platform.GITHUB,
                    native_id=f"p{index}",
                    occurred_at=NOW,
                )
            )
        db_session.add(
            ArtifactRow(
                id=artifact_id(Platform.GITHUB, "nobody"),
                kind=ArtifactKind.CI_RUN,
                source_id=src,
                actor_id=None,
                platform=Platform.GITHUB,
                native_id="nobody",
                occurred_at=NOW,
            )
        )
        await db_session.commit()

        mine = (
            (await db_session.execute(select(ArtifactRow).where(ArtifactRow.actor_id == per)))
            .scalars()
            .all()
        )
        assert len(mine) == 2


class TestDefaults:
    async def test_json_columns_default_to_empty_rather_than_null(
        self, db_session: AsyncSession
    ) -> None:
        """`links`, `provenance` and metadata are non-nullable with empty defaults,
        so a reader can index them without a null check on every access."""
        src, _ = await seed_refs(db_session)
        db_session.add(
            ArtifactRow(
                id=artifact_id(Platform.GITHUB, "bare"),
                kind=ArtifactKind.COMMIT,
                source_id=src,
                platform=Platform.GITHUB,
                native_id="bare",
                occurred_at=NOW,
            )
        )
        await db_session.commit()

        stored = await db_session.get(ArtifactRow, artifact_id(Platform.GITHUB, "bare"))
        assert stored is not None
        assert stored.links == []
        assert stored.provenance == {}
        assert stored.artifact_metadata == {}
        assert stored.state is ArtifactState.COMPLETED
        assert stored.tenant_id == "default"

    async def test_a_commits_file_list_survives_nesting(self, db_session: AsyncSession) -> None:
        """`details` holds a list of objects. JSON round-tripping a nested
        structure is where a column type that looked fine starts losing data."""
        src, _ = await seed_refs(db_session)
        details = CommitDetails(
            sha="a3f9c1",
            additions=47,
            deletions=12,
            files=[{"path": "a.py", "additions": 40}, {"path": "b.py", "deletions": 12}],  # type: ignore[list-item]
        )
        db_session.add(
            ArtifactRow(
                id=artifact_id(Platform.GITHUB, "files"),
                kind=ArtifactKind.COMMIT,
                source_id=src,
                platform=Platform.GITHUB,
                native_id="files",
                occurred_at=NOW,
                details=details.model_dump(mode="json"),
            )
        )
        await db_session.commit()

        stored = await db_session.get(ArtifactRow, artifact_id(Platform.GITHUB, "files"))
        assert stored is not None
        restored = CommitDetails.model_validate(stored.details)
        assert [f.path for f in restored.files] == ["a.py", "b.py"]
        assert restored.files[0].additions == 40
