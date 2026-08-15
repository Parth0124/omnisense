"""Sorting artifacts into features, and letting a person overrule it.

The important tests here are the ones about *corrections surviving*. A guess that
comes back after being rejected is worse than no guessing at all: the first time
somebody re-rejects it they are annoyed, the second time they stop correcting
anything, and from then on the feature is quietly whatever the matcher decided.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.exceptions import ConflictError, NotFoundError
from models.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactProvenance,
    CommitDetails,
    PullRequestDetails,
    artifact_id,
    source_id,
)
from models.enums import Platform
from models.feature import (
    DEFAULT_MEMBERSHIP_CONFIDENCE,
    FeatureLink,
    MembershipMethod,
    Version,
    VersionState,
    feature_id,
    version_id,
)
from models.orm.artifact import SourceRow
from models.orm.project import ProjectRow
from services.artifact_store import ArtifactStore
from services.feature_service import MIN_TERM_LENGTH, FeatureService

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
def service(factory) -> FeatureService:
    return FeatureService(factory, tenant_id="local")


async def store(factory, *artifacts: Artifact) -> None:
    await ArtifactStore(factory).write(list(artifacts))


def commit(native: str, title: str, *, days_ago: int = 0, **overrides) -> Artifact:
    fields = {
        "id": artifact_id(Platform.GITHUB, native),
        "tenant_id": "local",
        "kind": ArtifactKind.COMMIT,
        "source_id": SRC,
        "platform": Platform.GITHUB,
        "native_id": native,
        "title": title,
        "occurred_at": NOW - timedelta(days=days_ago),
        "details": CommitDetails(sha=native),
        "provenance": ArtifactProvenance(connector_slug="github", fetched_at=NOW),
    }
    fields.update(overrides)
    return Artifact(**fields)


def pull(native: str, title: str, *, head_ref: str) -> Artifact:
    return Artifact(
        id=artifact_id(Platform.GITHUB, native),
        tenant_id="local",
        kind=ArtifactKind.PULL_REQUEST,
        source_id=SRC,
        platform=Platform.GITHUB,
        native_id=native,
        title=title,
        occurred_at=NOW,
        details=PullRequestDetails(number=1, head_ref=head_ref),
        provenance=ArtifactProvenance(connector_slug="github", fetched_at=NOW),
    )


class TestTheGuardOnConfidence:
    def test_a_decision_carries_full_confidence(self) -> None:
        with pytest.raises(ValueError, match="decision"):
            FeatureLink(
                feature_id="f",
                artifact_id="a",
                method=MembershipMethod.CONFIRMED,
                confidence=0.5,
            )

    def test_an_inference_may_not_claim_certainty(self) -> None:
        with pytest.raises(ValueError, match="reserved for"):
            FeatureLink(
                feature_id="f", artifact_id="a", method=MembershipMethod.BRANCH, confidence=1.0
            )

    def test_a_branch_outranks_a_title_outranks_a_path(self) -> None:
        """A branch name is *chosen* -- `feature/image-upload` is somebody
        declaring intent. A title mention may be incidental, and a shared path
        usually is."""
        assert (
            DEFAULT_MEMBERSHIP_CONFIDENCE[MembershipMethod.BRANCH]
            > DEFAULT_MEMBERSHIP_CONFIDENCE[MembershipMethod.TITLE]
            > DEFAULT_MEMBERSHIP_CONFIDENCE[MembershipMethod.PATH]
        )

    def test_a_shipped_version_needs_a_date(self) -> None:
        """Otherwise it cannot be placed in a timeline, which is most of what
        anybody wants a shipped version for."""
        with pytest.raises(ValueError, match="shipped_at"):
            Version(id="v", project_id="p", name="v1", state=VersionState.SHIPPED, shipped_at=None)


class TestDeclaring:
    async def test_a_feature_can_be_attached_to_a_version(self, service, project) -> None:
        await service.add_version(project=project, name="v1")
        await service.add_feature(project=project, name="image upload", version="v1")

        features = await service.features(project)
        assert features[0].version_name == "v1"

    async def test_a_feature_without_a_version_is_allowed(self, service, project) -> None:
        """Work routinely starts before anybody decides which release it lands in,
        and forcing the choice then means inventing a version to hold it."""
        await service.add_feature(project=project, name="image upload")

        assert (await service.features(project))[0].version_name is None

    async def test_the_same_name_twice_is_refused(self, service, project) -> None:
        await service.add_feature(project=project, name="image upload")
        with pytest.raises(ConflictError):
            await service.add_feature(project=project, name="image upload")

    async def test_an_unknown_version_is_reported(self, service, project) -> None:
        with pytest.raises(NotFoundError):
            await service.add_feature(project=project, name="x", version="v9")

    async def test_ids_are_derived_so_names_are_stable(self) -> None:
        assert feature_id("local", "p", "Image Upload") == feature_id("local", "p", "image upload")
        assert version_id("local", "p", "v1") != version_id("local", "p", "v2")


class TestSorting:
    async def test_a_title_mention_attaches_the_artifact(self, factory, service, project) -> None:
        await store(factory, commit("c1", "implemented cloudinary service for image upload"))
        await service.add_feature(project=project, name="image upload")

        report = await service.sort(project)

        assert report.linked == 1

    async def test_a_keyword_catches_what_the_name_misses(self, factory, service, project) -> None:
        """The reason `--keyword` exists. A feature called "image upload" does not
        match "implemented cloudinary service", and that commit *is* the feature."""
        await store(factory, commit("c1", "implemented cloudinary service"))
        await service.add_feature(project=project, name="image upload", keywords=["cloudinary"])

        assert (await service.sort(project)).linked == 1

    async def test_a_branch_name_beats_a_title_mention(self, factory, service, project) -> None:
        await store(factory, pull("pr1", "upload fixes", head_ref="feature/upload"))
        await service.add_feature(project=project, name="upload")

        await service.sort(project)

        members = await service.members(feature_id("local", PROJECT_ID, "upload"))
        assert members[0][1].method is MembershipMethod.BRANCH

    async def test_one_artifact_can_belong_to_two_features(self, factory, service, project) -> None:
        """A commit adding an upload route to the deploy pipeline honestly belongs
        to both, and a single-owner key would force a wrong answer for one."""
        await store(factory, commit("c1", "deploy the upload service"))
        await service.add_feature(project=project, name="upload")
        await service.add_feature(project=project, name="deploy")

        assert (await service.sort(project)).linked == 2

    async def test_sorting_twice_attaches_nothing_new(self, factory, service, project) -> None:
        """Safe to run after every sync, which is the only way it stays current."""
        await store(factory, commit("c1", "image upload work"))
        await service.add_feature(project=project, name="image upload")

        await service.sort(project)
        second = await service.sort(project)

        assert second.linked == 0
        assert second.already_linked == 1

    async def test_short_terms_are_dropped(self, factory, service, project) -> None:
        """A feature keyworded `ui` would claim every commit containing "build",
        "guide" or "requirements"."""
        await store(factory, commit("c1", "rebuild the guide"))
        await service.add_feature(project=project, name="ui")

        assert (await service.sort(project)).linked == 0

    async def test_a_feature_matching_nothing_reports_zero_rather_than_failing(
        self, factory, service, project
    ) -> None:
        await store(factory, commit("c1", "something unrelated"))
        await service.add_feature(project=project, name="image upload")

        report = await service.sort(project)
        assert report.linked == 0
        assert report.scanned == 1

    async def test_a_project_with_no_features_does_nothing(self, factory, service, project) -> None:
        await store(factory, commit("c1", "anything"))
        assert (await service.sort(project)).scanned == 0

    async def test_matching_ignores_case(self, factory, service, project) -> None:
        await store(factory, commit("c1", "Image Upload done"))
        await service.add_feature(project=project, name="image upload")

        assert (await service.sort(project)).linked == 1

    async def test_evidence_records_what_actually_matched(self, factory, service, project) -> None:
        """A person can judge "title: cloudinary" instantly and cannot judge
        "0.55"."""
        await store(factory, commit("c1", "implemented cloudinary service"))
        await service.add_feature(project=project, name="image upload", keywords=["cloudinary"])
        await service.sort(project)

        members = await service.members(feature_id("local", PROJECT_ID, "image upload"))
        assert members[0][1].evidence == "title: cloudinary"


class TestCorrectionsSurvive:
    """The tests that decide whether correcting the system is worth doing twice."""

    async def test_a_rejection_is_not_undone_by_the_next_sort(
        self, factory, service, project
    ) -> None:
        """The whole reason `EXCLUDED` is stored rather than the row deleted."""
        await store(factory, commit("c1", "image upload work"))
        feature = await service.add_feature(project=project, name="image upload")
        await service.sort(project)

        await service.decide(
            feature=feature, artifact=artifact_id(Platform.GITHUB, "c1"), belongs=False
        )
        report = await service.sort(project)

        assert report.linked == 0
        assert report.protected == 1
        assert await service.members(feature) == []

    async def test_a_confirmation_is_not_downgraded_by_the_next_sort(
        self, factory, service, project
    ) -> None:
        await store(factory, commit("c1", "image upload work"))
        feature = await service.add_feature(project=project, name="image upload")
        await service.sort(project)
        await service.decide(
            feature=feature, artifact=artifact_id(Platform.GITHUB, "c1"), belongs=True
        )

        await service.sort(project)

        members = await service.members(feature)
        assert members[0][1].method is MembershipMethod.CONFIRMED
        assert members[0][1].confidence == 1.0

    async def test_an_artifact_the_matcher_missed_can_be_added_by_hand(
        self, factory, service, project
    ) -> None:
        """The matcher will always miss things; a feature nobody can correct
        upward is only half usable."""
        await store(factory, commit("c1", "totally unrelated wording"))
        feature = await service.add_feature(project=project, name="image upload")
        await service.sort(project)

        await service.decide(
            feature=feature, artifact=artifact_id(Platform.GITHUB, "c1"), belongs=True
        )

        assert len(await service.members(feature)) == 1

    async def test_a_rejection_can_be_reversed(self, factory, service, project) -> None:
        await store(factory, commit("c1", "image upload work"))
        feature = await service.add_feature(project=project, name="image upload")
        await service.sort(project)
        artifact = artifact_id(Platform.GITHUB, "c1")

        await service.decide(feature=feature, artifact=artifact, belongs=False)
        await service.decide(feature=feature, artifact=artifact, belongs=True)

        assert len(await service.members(feature)) == 1

    async def test_excluded_artifacts_are_not_counted(self, factory, service, project) -> None:
        """A count that includes things somebody said do not belong is a count
        nobody can act on."""
        await store(factory, commit("c1", "image upload"), commit("c2", "image upload again"))
        feature = await service.add_feature(project=project, name="image upload")
        await service.sort(project)
        await service.decide(
            feature=feature, artifact=artifact_id(Platform.GITHUB, "c1"), belongs=False
        )

        assert (await service.features(project))[0].artifact_count == 1


class TestReading:
    async def test_the_guessed_count_is_reported_separately(
        self, factory, service, project
    ) -> None:
        """ "12 artifacts" and "12 artifacts, 12 guessed" are the same number
        meaning very different things."""
        await store(factory, commit("c1", "image upload"), commit("c2", "image upload two"))
        feature = await service.add_feature(project=project, name="image upload")
        await service.sort(project)
        await service.decide(
            feature=feature, artifact=artifact_id(Platform.GITHUB, "c1"), belongs=True
        )

        summary = (await service.features(project))[0]
        assert summary.artifact_count == 2
        assert summary.guessed_count == 1

    async def test_a_version_counts_each_artifact_once(self, factory, service, project) -> None:
        """An artifact in two features of one version is one piece of work, and
        counting it twice would make a release look bigger than it is."""
        await store(factory, commit("c1", "deploy the upload service"))
        await service.add_version(project=project, name="v1")
        await service.add_feature(project=project, name="upload", version="v1")
        await service.add_feature(project=project, name="deploy", version="v1")
        await service.sort(project)

        assert (await service.versions(project))[0].artifact_count == 1

    async def test_an_unknown_project_is_reported(self, service) -> None:
        with pytest.raises(NotFoundError):
            await service.features("ghost")


class TestResolvingIds:
    async def test_a_prefix_finds_the_artifact(self, factory, service, project) -> None:
        await store(factory, commit("c1", "x"))
        full = artifact_id(Platform.GITHUB, "c1")

        assert await service.resolve_artifact(full[:12]) == full

    async def test_an_ambiguous_prefix_refuses_rather_than_picking(
        self, factory, service, project
    ) -> None:
        """Two matches is exactly when guessing is silent and wrong."""
        await store(factory, commit("c1", "x"), commit("c2", "y"))

        with pytest.raises(ConflictError, match="more than one"):
            await service.resolve_artifact("art_")

    async def test_an_unknown_prefix_is_reported(self, service, project) -> None:
        with pytest.raises(NotFoundError):
            await service.resolve_artifact("art_nothing")

    async def test_an_empty_prefix_does_not_match_everything(
        self, factory, service, project
    ) -> None:
        await store(factory, commit("c1", "x"))
        with pytest.raises(NotFoundError):
            await service.resolve_artifact("   ")


class TestTermLength:
    def test_the_floor_is_where_the_docstring_says(self) -> None:
        assert MIN_TERM_LENGTH == 3
