"""What counts as blocked, and what the answer is allowed to claim.

Two of the four rules work by *absence* -- "no review exists", "nothing has
happened". Absence is the most dangerous kind of evidence in this system, because
it looks identical whether the data was collected and empty or never collected at
all. Most of this file is about that distinction being preserved rather than
flattened into a confident sentence.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.exceptions import NotFoundError
from models.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactOutcome,
    ArtifactProvenance,
    ArtifactState,
    artifact_id,
    source_id,
)
from models.enums import Platform
from models.feature import FeatureState
from models.orm.artifact import SourceRow
from models.orm.feature import FeatureRow
from models.orm.project import ProjectRow
from services.artifact_store import ArtifactStore
from services.blocking_service import (
    REVIEW_OVERDUE_AFTER,
    STALE_AFTER,
    BlockingService,
)
from services.feature_service import FeatureService

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
SRC = source_id(Platform.GITHUB, "R_1")
PROJECT_ID = "prj_test"


@pytest.fixture
def factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest.fixture
async def project(factory) -> str:
    async with factory() as session:
        session.add(ProjectRow(id=PROJECT_ID, tenant_id="local", slug="proj", name="Proj"))
        await session.flush()
        session.add(
            SourceRow(
                id=SRC,
                tenant_id="local",
                project_id=PROJECT_ID,
                platform=Platform.GITHUB,
                external_id="R_1",
                name="me/repo",
            )
        )
        await session.commit()
    return "proj"


@pytest.fixture
def features(factory) -> FeatureService:
    return FeatureService(factory, tenant_id="local")


@pytest.fixture
def service(factory) -> BlockingService:
    return BlockingService(factory, tenant_id="local", now=NOW)


async def store(factory, *artifacts: Artifact) -> None:
    await ArtifactStore(factory).write(list(artifacts))


def make(
    native: str,
    kind: ArtifactKind,
    *,
    days_ago: int = 0,
    title: str = "upload work",
    state: ArtifactState | None = None,
    outcome: ArtifactOutcome | None = None,
) -> Artifact:
    return Artifact(
        id=artifact_id(Platform.GITHUB, native),
        tenant_id="local",
        kind=kind,
        source_id=SRC,
        platform=Platform.GITHUB,
        native_id=native,
        title=title,
        state=state,
        outcome=outcome,
        occurred_at=NOW - timedelta(days=days_ago),
        provenance=ArtifactProvenance(connector_slug="github", fetched_at=NOW),
    )


async def a_feature(features: FeatureService, project: str, name: str = "upload") -> str:
    identifier = await features.add_feature(project=project, name=name)
    await features.sort(project)
    return identifier


class TestCiFailures:
    async def test_a_failed_run_with_nothing_green_after_it_blocks(
        self, factory, features, service, project
    ) -> None:
        await store(
            factory, make("r1", ArtifactKind.CI_RUN, days_ago=1, outcome=ArtifactOutcome.FAILURE)
        )
        await a_feature(features, project)

        report = await service.blocking(project, "upload")

        assert [b.kind for b in report.blockers] == ["ci_failing"]

    async def test_a_later_success_clears_it(self, factory, features, service, project) -> None:
        """The "after it" is the whole rule. A feature fixed on Tuesday must not
        still be blocked on Friday, or the list never clears and nobody trusts
        it."""
        await store(
            factory,
            make("r1", ArtifactKind.CI_RUN, days_ago=3, outcome=ArtifactOutcome.FAILURE),
            make("r2", ArtifactKind.CI_RUN, days_ago=1, outcome=ArtifactOutcome.SUCCESS),
        )
        await a_feature(features, project)

        report = await service.blocking(project, "upload")

        assert not [b for b in report.blockers if b.kind == "ci_failing"]

    async def test_a_ci_failure_is_certain_and_needs_no_caveat(
        self, factory, features, service, project
    ) -> None:
        """The only one of the four rules resting on data that exists rather than
        data that does not."""
        await store(
            factory, make("r1", ArtifactKind.CI_RUN, days_ago=1, outcome=ArtifactOutcome.FAILURE)
        )
        await a_feature(features, project)

        blocker = (await service.blocking(project, "upload")).blockers[0]
        assert blocker.confidence == 1.0
        assert blocker.caveat is None


class TestUnreviewedPullRequests:
    async def test_an_old_open_pull_request_with_no_review_blocks(
        self, factory, features, service, project
    ) -> None:
        await store(
            factory,
            make(
                "pr1",
                ArtifactKind.PULL_REQUEST,
                days_ago=REVIEW_OVERDUE_AFTER.days + 1,
                state=ArtifactState.OPEN,
            ),
        )
        await a_feature(features, project)

        assert "unreviewed" in [
            b.kind for b in (await service.blocking(project, "upload")).blockers
        ]

    async def test_a_fresh_pull_request_is_not_yet_a_blocker(
        self, factory, features, service, project
    ) -> None:
        """A working day plus the weekend. Shorter and every pull request opened
        on a Friday is a blocker by Monday."""
        await store(
            factory, make("pr1", ArtifactKind.PULL_REQUEST, days_ago=1, state=ArtifactState.OPEN)
        )
        await a_feature(features, project)

        assert "unreviewed" not in [
            b.kind for b in (await service.blocking(project, "upload")).blockers
        ]

    async def test_a_merged_pull_request_is_not_a_blocker(
        self, factory, features, service, project
    ) -> None:
        await store(
            factory,
            make(
                "pr1",
                ArtifactKind.PULL_REQUEST,
                days_ago=30,
                state=ArtifactState.MERGED,
            ),
        )
        await a_feature(features, project)

        assert "unreviewed" not in [
            b.kind for b in (await service.blocking(project, "upload")).blockers
        ]

    async def test_it_says_that_absence_might_mean_unfetched(
        self, factory, features, service, project
    ) -> None:
        """The most important assertion in this file.

        "No review exists" means one thing when reviews are being collected and
        nothing at all when they are not. The reviews stream has never completed a
        live pass, so the blocker must carry that rather than stating it flatly.
        """
        await store(
            factory,
            make(
                "pr1",
                ArtifactKind.PULL_REQUEST,
                days_ago=30,
                state=ArtifactState.OPEN,
            ),
        )
        await a_feature(features, project)

        blocker = next(
            b
            for b in (await service.blocking(project, "upload")).blockers
            if b.kind == "unreviewed"
        )
        assert blocker.confidence < 1.0
        assert blocker.caveat is not None
        assert "unfetched" in blocker.caveat


class TestStalledWork:
    async def test_untouched_unfinished_work_blocks(
        self, factory, features, service, project
    ) -> None:
        await store(factory, make("c1", ArtifactKind.COMMIT, days_ago=STALE_AFTER.days + 5))
        await a_feature(features, project)

        assert "stalled" in [b.kind for b in (await service.blocking(project, "upload")).blockers]

    async def test_recent_work_is_not_stalled(self, factory, features, service, project) -> None:
        await store(factory, make("c1", ArtifactKind.COMMIT, days_ago=1))
        await a_feature(features, project)

        assert "stalled" not in [
            b.kind for b in (await service.blocking(project, "upload")).blockers
        ]

    async def test_a_finished_feature_is_never_stalled(
        self, factory, features, service, project
    ) -> None:
        """Done is done. Reporting a shipped feature as stalled is the fastest way
        to teach somebody to ignore the list."""
        await store(factory, make("c1", ArtifactKind.COMMIT, days_ago=400))
        identifier = await a_feature(features, project)
        async with factory() as session:
            row = await session.get(FeatureRow, identifier)
            row.state = FeatureState.DONE
            await session.commit()

        assert not (await service.blocking(project, "upload")).blockers

    async def test_a_dropped_feature_is_never_stalled(
        self, factory, features, service, project
    ) -> None:
        """The mute. A deliberate pause is indistinguishable from abandonment in
        the data, so there has to be a way to say which it was."""
        await store(factory, make("c1", ArtifactKind.COMMIT, days_ago=400))
        identifier = await a_feature(features, project)
        async with factory() as session:
            row = await session.get(FeatureRow, identifier)
            row.state = FeatureState.DROPPED
            await session.commit()

        assert not (await service.blocking(project, "upload")).blockers

    async def test_it_admits_a_pause_looks_the_same(
        self, factory, features, service, project
    ) -> None:
        await store(factory, make("c1", ArtifactKind.COMMIT, days_ago=400))
        await a_feature(features, project)

        blocker = next(
            b for b in (await service.blocking(project, "upload")).blockers if b.kind == "stalled"
        )
        assert blocker.confidence < 1.0
        assert "deliberate pause" in (blocker.caveat or "")

    async def test_a_feature_with_no_artifacts_is_not_stalled(
        self, features, service, project
    ) -> None:
        """Nothing has happened because nothing was ever attached, which is a
        different problem and not a blocker."""
        await features.add_feature(project=project, name="upload")

        assert not (await service.blocking(project, "upload")).blockers


class TestTargets:
    async def test_a_version_checks_every_feature_in_it(
        self, factory, features, service, project
    ) -> None:
        await store(
            factory,
            make(
                "r1",
                ArtifactKind.CI_RUN,
                days_ago=1,
                title="upload",
                outcome=ArtifactOutcome.FAILURE,
            ),
            make(
                "r2",
                ArtifactKind.CI_RUN,
                days_ago=1,
                title="deploy",
                outcome=ArtifactOutcome.FAILURE,
            ),
        )
        await features.add_version(project=project, name="v1")
        await features.add_feature(project=project, name="upload", version="v1")
        await features.add_feature(project=project, name="deploy", version="v1")
        await features.sort(project)

        report = await service.blocking(project, "v1")

        assert report.features_checked == 2
        assert len(report.blockers) == 2

    async def test_a_version_name_wins_over_a_feature_of_the_same_name(
        self, factory, features, service, project
    ) -> None:
        """`blocking v1.1` is the common question; the zoomed-in one is rarer and
        can be reached by its own name."""
        await features.add_version(project=project, name="shared")
        await features.add_feature(project=project, name="shared")

        report = await service.blocking(project, "shared")

        assert report.features_checked == 0

    async def test_matching_ignores_case(self, features, service, project) -> None:
        await features.add_version(project=project, name="v1")
        assert (await service.blocking(project, "V1")).features_checked == 0

    async def test_an_unknown_target_is_reported(self, service, project) -> None:
        with pytest.raises(NotFoundError):
            await service.blocking(project, "nothing-like-this")

    async def test_an_unknown_project_is_reported(self, service) -> None:
        with pytest.raises(NotFoundError):
            await service.blocking("ghost", "v1")


class TestRanking:
    async def test_a_broken_build_outranks_a_stalled_feature(
        self, factory, features, service, project
    ) -> None:
        """Four blockers with no ordering is a list somebody reads top to bottom
        and acts on at random."""
        await store(
            factory,
            make("c1", ArtifactKind.COMMIT, days_ago=400, title="upload"),
            make(
                "r1",
                ArtifactKind.CI_RUN,
                days_ago=400,
                title="upload",
                outcome=ArtifactOutcome.FAILURE,
            ),
        )
        await a_feature(features, project)

        kinds = [b.kind for b in (await service.blocking(project, "upload")).blockers]
        assert kinds.index("ci_failing") < kinds.index("stalled")

    async def test_a_clear_feature_reports_clear(self, factory, features, service, project) -> None:
        await store(
            factory,
            make("r1", ArtifactKind.CI_RUN, days_ago=1, outcome=ArtifactOutcome.SUCCESS),
        )
        await a_feature(features, project)

        assert (await service.blocking(project, "upload")).is_clear


class TestExclusions:
    async def test_an_artifact_a_person_rejected_cannot_block(
        self, factory, features, service, project
    ) -> None:
        """The correction has to reach every reader of the membership, not just
        the listing screen."""
        await store(
            factory, make("r1", ArtifactKind.CI_RUN, days_ago=1, outcome=ArtifactOutcome.FAILURE)
        )
        identifier = await a_feature(features, project)
        await features.decide(
            feature=identifier, artifact=artifact_id(Platform.GITHUB, "r1"), belongs=False
        )

        assert not [
            b
            for b in (await service.blocking(project, "upload")).blockers
            if b.kind == "ci_failing"
        ]
