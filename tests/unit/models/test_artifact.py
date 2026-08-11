"""The artifact model: identity, the shared frame, and the per-kind detail.

Two things are worth stating about what these tests are for.

**The shared columns are the risky part, not the detail classes.** `Signal` was a
perfectly good model that turned out to fit only the thing it was designed for,
and the way that went wrong was not a bad field -- it was a *frame* that assumed
every observation has an author, a sentiment and an engagement count. So the
tests that matter most here are the ones that put a CI run, a paper and a Slack
message through the same columns and check none of them needs a special case.

**Identity is the other one.** Ids are derived, not assigned, and every re-sync
depends on that derivation being stable. A change to it does not fail: it
silently re-ingests the entire corpus alongside itself.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from models.artifact import (
    ARTIFACT_ID_PREFIX,
    Artifact,
    ArtifactKind,
    ArtifactLink,
    ArtifactOutcome,
    ArtifactProvenance,
    ArtifactState,
    CIRunDetails,
    CommitDetails,
    IssueDetails,
    JobResult,
    LinkRelation,
    Person,
    PullRequestDetails,
    ReviewDetails,
    Source,
    artifact_id,
    person_id,
    source_id,
)
from models.enums import Platform

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 11, 9, 0, tzinfo=UTC)


def provenance() -> ArtifactProvenance:
    return ArtifactProvenance(connector_slug="github", fetched_at=NOW)


def artifact(**overrides: object) -> Artifact:
    fields: dict[str, object] = {
        "id": artifact_id(Platform.GITHUB, "node-1"),
        "kind": ArtifactKind.COMMIT,
        "source_id": source_id(Platform.GITHUB, "R_1"),
        "platform": Platform.GITHUB,
        "native_id": "node-1",
        "occurred_at": NOW,
        "provenance": provenance(),
    }
    fields.update(overrides)
    return Artifact(**fields)  # type: ignore[arg-type]


class TestIdentity:
    """Derived, deterministic, and keyed on things that do not change."""

    def test_the_same_inputs_always_give_the_same_id(self) -> None:
        """The property that makes a re-sync an upsert instead of a duplicate."""
        assert artifact_id(Platform.GITHUB, "abc") == artifact_id(Platform.GITHUB, "abc")
        assert artifact_id(Platform.GITHUB, "abc") == artifact_id("github", "abc")

    def test_different_platforms_never_collide(self) -> None:
        """Native ids are only unique within a platform. A Slack `1699…` and a Jira
        `1699…` are different things and must not share a row."""
        assert artifact_id(Platform.GITHUB, "1") != artifact_id(Platform.SLACK, "1")

    def test_the_three_id_spaces_are_distinct(self) -> None:
        """An artifact, a source and a person built from the same string are three
        different rows. Sharing a derivation would make `source_id` a valid
        `actor_id`, and the foreign keys would happily accept it."""
        ids = {
            artifact_id(Platform.GITHUB, "x"),
            source_id(Platform.GITHUB, "x"),
            person_id(Platform.GITHUB, "x"),
        }
        assert len(ids) == 3

    def test_ids_are_prefixed_so_a_bare_id_is_recognisable(self) -> None:
        assert artifact_id(Platform.GITHUB, "x").startswith(ARTIFACT_ID_PREFIX)

    def test_an_empty_native_id_is_refused(self) -> None:
        """Deriving identity from nothing produces a single id that every
        unidentifiable object collides on -- one row that silently absorbs them."""
        for derive in (artifact_id, source_id, person_id):
            with pytest.raises(ValueError, match="non-empty"):
                derive(Platform.GITHUB, "")

    def test_the_derivation_is_pinned(self) -> None:
        """A literal, not a recomputation.

        This is the one test that fails if someone changes the namespace uuid or
        the joining format. That change does not break anything visibly -- it
        silently re-ingests the whole corpus under new ids, alongside the old
        rows, and every existing citation stops resolving.
        """
        assert artifact_id(Platform.GITHUB, "I_kwDOABCD1M6TqXyN") == (
            "art_2fecfa686c9d5bbc815203958d3100ab"
        )


class TestTheSharedFrameHoldsAcrossKinds:
    """A CI run, a paper and a message through the same columns, no special cases.

    This is the check `Signal` failed. Each of these is constructed with only the
    shared fields plus its own detail -- if any of them needed a column the others
    do not have, or had to abuse one, the frame is wrong and it is far cheaper to
    learn that here than after four connectors depend on it.
    """

    def test_a_ci_run_needs_no_author_and_no_body(self) -> None:
        run = artifact(
            kind=ArtifactKind.CI_RUN,
            actor_id=None,
            title="Run tests",
            body=None,
            state=ArtifactState.COMPLETED,
            outcome=ArtifactOutcome.FAILURE,
            details=CIRunDetails(workflow_name="test.yml", duration_seconds=420.0),
        )
        assert run.actor_id is None
        assert run.failed

    def test_a_paper_is_an_artifact_with_no_detail_class(self) -> None:
        """The world's output rather than yours. Nothing about the frame objects."""
        paper = artifact(
            kind=ArtifactKind.PAPER,
            platform=Platform.ARXIV,
            title="Certified Split Windows for Parallel Lexing",
            body="We present...",
            state=ArtifactState.COMPLETED,
            details=None,
        )
        assert paper.is_finished
        assert not paper.failed

    def test_a_message_is_an_artifact(self) -> None:
        message = artifact(
            kind=ArtifactKind.MESSAGE,
            platform=Platform.SLACK,
            title=None,
            body="the reaper starts too early",
            state=ArtifactState.COMPLETED,
            details=None,
        )
        assert message.title is None
        assert message.is_finished

    def test_an_agent_run_is_an_artifact(self) -> None:
        """The record that makes an autonomous action auditable rather than merely
        something that happened."""
        run = artifact(
            kind=ArtifactKind.AGENT_RUN,
            title="fix the flaky lease test",
            state=ArtifactState.RUNNING,
            outcome=None,
            links=[ArtifactLink(relation=LinkRelation.PRODUCED, target_native_id="PR_123")],
            metadata={"agent.name": "claude_code", "agent.cost_usd": 0.42},
        )
        assert not run.is_finished
        assert run.links[0].relation is LinkRelation.PRODUCED


class TestStateAndOutcome:
    """The pair that lets one query span every kind."""

    def test_finished_is_not_the_same_as_succeeded(self) -> None:
        """A CI run that finished and a CI run that passed are different claims,
        and only one of them is good news."""
        failed = artifact(
            kind=ArtifactKind.CI_RUN,
            state=ArtifactState.COMPLETED,
            outcome=ArtifactOutcome.FAILURE,
        )
        assert failed.is_finished
        assert failed.failed

    def test_in_flight_work_has_no_outcome(self) -> None:
        running = artifact(kind=ArtifactKind.CI_RUN, state=ArtifactState.RUNNING)
        assert running.outcome is None
        assert not running.is_finished
        assert not running.failed

    def test_merged_and_closed_both_count_as_finished(self) -> None:
        for state in (ArtifactState.MERGED, ArtifactState.CLOSED):
            assert artifact(kind=ArtifactKind.PULL_REQUEST, state=state).is_finished

    def test_an_unknown_state_from_a_platform_degrades_rather_than_raising(self) -> None:
        """`ArtifactState` is tolerant on purpose: platforms add vocabulary, and a
        new one must not break ingestion of every other row in the same page."""
        assert ArtifactState("something_new_github_invented") is ArtifactState.UNKNOWN
        assert ArtifactOutcome("brand_new_conclusion") is ArtifactOutcome.UNKNOWN

    def test_kind_is_strict_because_it_decides_what_is_read(self) -> None:
        """The deliberate asymmetry with state and outcome.

        `kind` selects rows and selects which detail shape is parsed. Degrading an
        unrecognised kind to UNKNOWN would drop rows from every filtered query
        instead of failing where the bad value entered.
        """
        with pytest.raises(ValueError):
            ArtifactKind("not_a_kind")


class TestDetailsAreTypedNotABag:
    """The whole point of the split: detail differs per kind and is still checked."""

    def test_each_kind_parses_back_into_its_own_class(self) -> None:
        pairs = [
            (ArtifactKind.COMMIT, CommitDetails(sha="a3f9c1")),
            (ArtifactKind.PULL_REQUEST, PullRequestDetails(number=4181)),
            (ArtifactKind.REVIEW, ReviewDetails(pull_request_number=4181)),
            (ArtifactKind.CI_RUN, CIRunDetails(workflow_name="test.yml")),
            (ArtifactKind.ISSUE, IssueDetails(number=4166)),
        ]
        for kind, details in pairs:
            restored = Artifact.model_validate(artifact(kind=kind, details=details).model_dump())
            assert type(restored.details) is type(details), kind

    def test_the_failing_job_is_recoverable_from_a_run(self) -> None:
        """ "The build failed" is not actionable; "the integration job failed" is."""
        details = CIRunDetails(
            jobs=[
                JobResult(name="unit", outcome=ArtifactOutcome.SUCCESS),
                JobResult(name="lint", outcome=ArtifactOutcome.SKIPPED),
                JobResult(name="integration", outcome=ArtifactOutcome.FAILURE),
            ]
        )
        assert details.failed_jobs == ["integration"]

    def test_a_commits_author_and_committer_can_differ(self) -> None:
        """They genuinely do -- a rebase or a squash-merge rewrites the committer
        while keeping the author. Collapsing them credits the wrong person."""
        commit = artifact(
            kind=ArtifactKind.COMMIT,
            actor_id=person_id(Platform.GITHUB, "author"),
            details=CommitDetails(sha="a3", committer_id=person_id(Platform.GITHUB, "merger")),
        )
        assert commit.actor_id != commit.details.committer_id  # type: ignore[union-attr]

    def test_a_merge_commit_is_visible_from_its_parents(self) -> None:
        details = CommitDetails(sha="m1", parent_shas=["a", "b"], is_merge=True)
        assert len(details.parent_shas) == 2

    def test_details_of_the_wrong_kind_are_rejected(self) -> None:
        """The discriminator earns its keep here: a PR's detail on a commit row
        would otherwise be stored happily and fail much later, on read."""
        with pytest.raises(ValueError):
            Artifact.model_validate(
                {
                    **artifact().model_dump(),
                    "kind": ArtifactKind.COMMIT.value,
                    "details": {"kind": "pull_request", "number": 1},
                }
            )

    def test_an_unknown_detail_field_is_refused(self) -> None:
        """`StrictModel` forbids extras, so a field a connector invents fails at
        the boundary rather than vanishing silently into a dump."""
        with pytest.raises(ValueError):
            CommitDetails(sha="a", invented_field="x")  # type: ignore[call-arg]


class TestLinks:
    """Relationships between artifacts, and why they hold native ids."""

    def test_a_link_survives_its_target_not_existing_yet(self) -> None:
        """Ingestion order is not guaranteed -- a PR routinely references an issue
        that has not been fetched. Holding the target's *native* id means the link
        is recorded now and resolved whenever the target arrives, rather than
        being dropped for pointing at nothing."""
        pr = artifact(
            kind=ArtifactKind.PULL_REQUEST,
            links=[
                ArtifactLink(
                    relation=LinkRelation.CLOSES,
                    target_native_id="I_kwDOABCD1M6TqXyN",
                    target_kind=ArtifactKind.ISSUE,
                )
            ],
        )
        assert pr.links[0].target_native_id == "I_kwDOABCD1M6TqXyN"

    def test_an_unknown_relation_degrades(self) -> None:
        assert LinkRelation("invented_relation") is LinkRelation.UNKNOWN


class TestSourceAndPerson:
    """The two referenced tables, and the rename problem they exist for."""

    def test_a_source_is_identified_by_the_platforms_id_not_its_name(self) -> None:
        """A renamed repository keeps its id, so every artifact follows for free.
        Identity keyed on the name would fork the history at the rename."""
        before = Source(
            id=source_id(Platform.GITHUB, "R_kgDOABCD1M"),
            platform=Platform.GITHUB,
            external_id="R_kgDOABCD1M",
            name="omnisense/api",
        )
        after = Source(
            id=source_id(Platform.GITHUB, "R_kgDOABCD1M"),
            platform=Platform.GITHUB,
            external_id="R_kgDOABCD1M",
            name="omnisense/backend",
        )
        assert before.id == after.id

    def test_a_person_is_identified_by_the_platforms_id_not_the_handle(self) -> None:
        """GitHub logins are renameable and the rename rewrites every URL that
        contained one. Keyed on a handle, a person forks in two the first time
        somebody rebrands -- and their history splits with them."""
        assert person_id(Platform.GITHUB, "MDQ6VXNlcjE=") == person_id(
            Platform.GITHUB, "MDQ6VXNlcjE="
        )

    def test_the_same_human_on_two_platforms_is_two_rows(self) -> None:
        """Deliberate. Deciding two accounts are one human is a cross-source
        inference with a confidence attached, and it belongs in the graph where a
        guess can be stored as a guess. A foreign key here would make it
        indistinguishable from a fact."""
        assert person_id(Platform.GITHUB, "dmitri") != person_id(Platform.SLACK, "dmitri")

    def test_default_branch_is_absent_for_things_that_are_not_repositories(self) -> None:
        channel = Source(
            id=source_id(Platform.SLACK, "C123"),
            platform=Platform.SLACK,
            external_id="C123",
            name="#eng-scheduler",
        )
        assert channel.default_branch is None

    def test_a_bot_is_marked_as_one(self) -> None:
        """Otherwise "who has been most active this week" answers `dependabot`."""
        bot = Person(
            id=person_id(Platform.GITHUB, "BOT_1"),
            platform=Platform.GITHUB,
            external_id="BOT_1",
            handle="dependabot[bot]",
            is_bot=True,
        )
        assert bot.is_bot


class TestRoundTrip:
    def test_a_full_artifact_survives_dump_and_reload(self) -> None:
        original = artifact(
            kind=ArtifactKind.CI_RUN,
            title="Run tests",
            state=ArtifactState.COMPLETED,
            outcome=ArtifactOutcome.FAILURE,
            details=CIRunDetails(
                workflow_name="test.yml",
                jobs=[JobResult(name="integration", outcome=ArtifactOutcome.FAILURE)],
            ),
            links=[ArtifactLink(relation=LinkRelation.TESTS, target_native_id="a3f9c1")],
            metadata={"github.repository": "omnisense/api"},
        )
        restored = Artifact.model_validate(original.model_dump())
        assert restored == original
        assert restored.details.failed_jobs == ["integration"]  # type: ignore[union-attr]

    def test_occurred_at_keeps_its_timezone(self) -> None:
        """Event time is compared across sources in every window query. A naive
        datetime would be wrong by the server's offset, silently, and only for
        deployments outside UTC."""
        restored = Artifact.model_validate(artifact().model_dump())
        assert restored.occurred_at.tzinfo is not None
        assert restored.occurred_at == NOW
