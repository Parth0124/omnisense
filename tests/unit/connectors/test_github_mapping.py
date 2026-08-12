"""GitHub payloads to artifacts.

A mapping bug is silent, which is why this file is long. A commit stored with the
wrong timestamp does not fail -- it appears in the wrong week, forever, and the
only way anyone finds out is by noticing that a report is wrong and not believing
it. So the tests here are weighted towards the fields that fail quietly:
timestamps, identity, and the state/outcome pair.

The payloads are trimmed versions of real GitHub responses, keeping the fields
these functions read and the shapes that caused trouble -- a null author, a
pending review, a run that has not finished.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from connectors.github.mapping import (
    map_commit,
    map_person,
    map_pull_request,
    map_review,
    map_workflow_run,
)
from models.artifact import ArtifactKind, ArtifactOutcome, ArtifactState, LinkRelation

pytestmark = pytest.mark.unit

SRC = "src_test"


def commit_payload(**overrides):
    payload = {
        "sha": "9c4dfdaebe0e6b2aabc566eb81f6f10eb5cd6ea1",
        "node_id": "C_kwDOASN_zNoAKDlj",
        "html_url": "https://github.com/pallets/click/commit/9c4dfda",
        "commit": {
            "message": "Sort help args\n\nSo the output is reproducible.",
            "author": {"name": "Rowlando13", "date": "2026-08-09T17:51:56Z"},
            "committer": {"name": "GitHub", "date": "2026-08-09T18:02:11Z"},
        },
        "author": {"login": "Rowlando13", "node_id": "U_author", "type": "User"},
        "committer": {"login": "web-flow", "node_id": "U_webflow", "type": "User"},
        "parents": [{"sha": "aaa"}],
    }
    payload.update(overrides)
    return payload


class TestUnfetchedSizes:
    """ "Changed nothing" and "we never asked" must not look the same.

    GitHub's *list* endpoints omit `stats`, `files`, `additions` and the rest --
    they come only from the single-item endpoints, at one request each. Defaulted
    to `0`, every one of the 1,639 pull requests read from `pallets/click` claimed
    to be an empty diff, and no later query could tell that from the truth.
    """

    def test_a_commit_from_the_list_endpoint_reports_unknown_not_zero(self) -> None:
        artifact = map_commit(commit_payload(), source_id="src_1")

        assert artifact is not None
        assert artifact.details.additions is None
        assert artifact.details.deletions is None
        assert artifact.details.changed_files is None

    def test_a_commit_fetched_individually_reports_its_real_sizes(self) -> None:
        artifact = map_commit(
            commit_payload(stats={"additions": 12, "deletions": 3, "total": 15}),
            source_id="src_1",
        )

        assert artifact is not None
        assert artifact.details.additions == 12
        assert artifact.details.deletions == 3

    def test_a_genuinely_empty_diff_is_still_zero(self) -> None:
        """The other half of the distinction: when GitHub *does* answer, a zero is
        a real zero and must survive."""
        artifact = map_commit(
            commit_payload(stats={"additions": 0, "deletions": 0, "total": 0}),
            source_id="src_1",
        )

        assert artifact is not None
        assert artifact.details.additions == 0

    def test_a_pull_request_from_the_list_endpoint_reports_unknown_too(self) -> None:
        artifact = map_pull_request(pull_payload(), source_id="src_1")

        assert artifact is not None
        assert artifact.details.additions is None
        assert artifact.details.changed_files is None
        assert artifact.details.commit_count is None
        assert artifact.details.has_conflicts is None


class TestCommits:
    def test_the_authored_date_wins_over_the_committed_one(self) -> None:
        """They differ whenever somebody rebases, and the authored date is when the
        work was done. Using the committed date would move a year of history into
        the afternoon somebody tidied the branch."""
        artifact = map_commit(commit_payload(), source_id=SRC)

        assert artifact is not None
        assert artifact.occurred_at == datetime(2026, 8, 9, 17, 51, 56, tzinfo=UTC)
        assert artifact.updated_at_source == datetime(2026, 8, 9, 18, 2, 11, tzinfo=UTC)

    def test_the_subject_and_body_are_split(self) -> None:
        """A commit message's first line is its title everywhere else in the world;
        storing the whole thing as a title makes every list unreadable."""
        artifact = map_commit(commit_payload(), source_id=SRC)
        assert artifact is not None
        assert artifact.title == "Sort help args"
        assert artifact.body == "So the output is reproducible."

    def test_two_parents_is_a_merge(self) -> None:
        """The API has no flag for it; parent count is the definition."""
        merge = map_commit(commit_payload(parents=[{"sha": "a"}, {"sha": "b"}]), source_id=SRC)
        assert merge is not None
        assert merge.details.is_merge

    def test_a_committer_who_is_the_author_is_not_recorded_twice(self) -> None:
        """Storing the same person in both places implies a handoff that did not
        happen -- which is exactly what the field exists to show when it did."""
        same = {"login": "a", "node_id": "U_same", "type": "User"}
        artifact = map_commit(commit_payload(author=same, committer=same), source_id=SRC)
        assert artifact is not None
        assert artifact.details.committer_id is None

    def test_a_deleted_author_leaves_the_commit_intact(self) -> None:
        """GitHub answers `null` for a deleted account. The work still happened, so
        the commit is stored with no actor rather than dropped or given a
        placeholder person who does not exist."""
        artifact = map_commit(commit_payload(author=None), source_id=SRC)
        assert artifact is not None
        assert artifact.actor_id is None

    def test_a_commit_with_no_node_id_is_refused(self) -> None:
        """Identity is not optional. Without it a re-sync cannot recognise the same
        commit and would store it again on every run."""
        assert map_commit(commit_payload(node_id=None), source_id=SRC) is None


def pull_payload(**overrides):
    payload = {
        "node_id": "PR_node",
        "number": 3757,
        "title": "Don't break option names at hyphens",
        "body": "Fixes wrapping.",
        "html_url": "https://github.com/pallets/click/pull/3757",
        "created_at": "2026-08-01T10:00:00Z",
        "updated_at": "2026-08-11T09:00:00Z",
        "closed_at": None,
        "merged_at": None,
        "draft": False,
        "mergeable": None,
        "user": {"login": "someone", "node_id": "U_1", "type": "User"},
        "base": {"ref": "main"},
        "head": {"ref": "fix/wrapping"},
        "labels": [{"name": "bug"}],
    }
    payload.update(overrides)
    return payload


class TestPullRequests:
    def test_merged_is_a_state_not_an_outcome(self) -> None:
        """A merged pull request reached a different resting place from a closed
        one, and conflating them loses the only distinction anyone cares about
        when asking what shipped."""
        merged = map_pull_request(
            pull_payload(merged_at="2026-08-10T12:00:00Z", closed_at="2026-08-10T12:00:00Z"),
            source_id=SRC,
        )
        assert merged is not None
        assert merged.state is ArtifactState.MERGED
        assert merged.outcome is None

    def test_closed_without_merging_is_distinguishable(self) -> None:
        closed = map_pull_request(pull_payload(closed_at="2026-08-10T12:00:00Z"), source_id=SRC)
        assert closed is not None
        assert closed.state is ArtifactState.CLOSED

    def test_a_draft_is_its_own_state(self) -> None:
        draft = map_pull_request(pull_payload(draft=True), source_id=SRC)
        assert draft is not None
        assert draft.state is ArtifactState.DRAFT

    def test_unknown_mergeability_is_not_the_same_as_mergeable(self) -> None:
        """GitHub returns `null` until it has computed it. Coercing that to False
        would claim there are no conflicts when nobody has looked."""
        assert map_pull_request(pull_payload(), source_id=SRC).details.has_conflicts is None
        assert (
            map_pull_request(pull_payload(mergeable=True), source_id=SRC).details.has_conflicts
            is False
        )
        assert (
            map_pull_request(pull_payload(mergeable=False), source_id=SRC).details.has_conflicts
            is True
        )

    def test_occurred_at_is_when_it_opened_not_when_it_changed(self) -> None:
        """`updated_at` moves for a comment or a label. If it drove `occurred_at`,
        every touched pull request would jump to the top of "what happened this
        week" regardless of when the work was done."""
        artifact = map_pull_request(pull_payload(), source_id=SRC)
        assert artifact is not None
        assert artifact.occurred_at == datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
        assert artifact.updated_at_source == datetime(2026, 8, 11, 9, 0, tzinfo=UTC)

    def test_the_merge_commit_is_linked(self) -> None:
        artifact = map_pull_request(
            pull_payload(merge_commit_sha="abc123", merged_at="2026-08-10T12:00:00Z"),
            source_id=SRC,
        )
        assert artifact is not None
        assert artifact.links[0].relation is LinkRelation.PART_OF
        assert artifact.links[0].target_native_id == "abc123"


class TestReviews:
    def test_the_verdict_becomes_the_outcome(self) -> None:
        for state, expected in [
            ("APPROVED", ArtifactOutcome.APPROVED),
            ("CHANGES_REQUESTED", ArtifactOutcome.CHANGES_REQUESTED),
            ("COMMENTED", ArtifactOutcome.COMMENTED),
            ("DISMISSED", ArtifactOutcome.DISMISSED),
        ]:
            artifact = map_review(
                {
                    "node_id": f"RV_{state}",
                    "state": state,
                    "submitted_at": "2026-08-10T12:00:00Z",
                    "user": {"login": "r", "node_id": "U_r", "type": "User"},
                },
                source_id=SRC,
                pull_number=1,
                pull_node_id="PR_1",
            )
            assert artifact is not None
            assert artifact.outcome is expected

    def test_a_pending_review_is_not_an_event_yet(self) -> None:
        """Drafted, visible only to its author, not submitted. Storing it would
        show a verdict nobody has given."""
        assert (
            map_review(
                {"node_id": "RV_1", "state": "PENDING", "submitted_at": None},
                source_id=SRC,
                pull_number=1,
                pull_node_id="PR_1",
            )
            is None
        )

    def test_a_review_has_no_title_of_its_own(self) -> None:
        """The thing being reviewed has the title. Inventing "Review of #4181"
        would make a list of titles read as though somebody wrote them."""
        artifact = map_review(
            {"node_id": "RV_1", "state": "APPROVED", "submitted_at": "2026-08-10T12:00:00Z"},
            source_id=SRC,
            pull_number=4181,
            pull_node_id="PR_1",
        )
        assert artifact is not None
        assert artifact.title is None
        assert artifact.links[0].relation is LinkRelation.REVIEWS


def run_payload(**overrides):
    payload = {
        "id": 123,
        "node_id": "WFR_node",
        "name": "pre-commit",
        "status": "completed",
        "conclusion": "success",
        "run_number": 42,
        "run_attempt": 1,
        "event": "pull_request",
        "head_sha": "b3516cb",
        "head_branch": "main",
        "run_started_at": "2026-08-11T10:00:00Z",
        "updated_at": "2026-08-11T10:00:16Z",
        "actor": {"login": "someone", "node_id": "U_1", "type": "User"},
    }
    payload.update(overrides)
    return payload


class TestWorkflowRuns:
    def test_the_title_distinguishes_runs_of_the_same_workflow(self) -> None:
        """`name` is the workflow's name and is identical on every run it ever
        produces. `display_title` is the triggering commit's message, which is what
        GitHub shows in the Actions list and the only thing that tells four runs of
        one pipeline apart -- four rows all reading "Full-Stack CI/CD" is a list
        nobody can use."""
        artifact = map_workflow_run(
            run_payload(name="Full-Stack CI/CD", display_title="Delete DEPLOYMENT.md"),
            source_id=SRC,
        )

        assert artifact is not None
        assert artifact.title == "Delete DEPLOYMENT.md"
        # Not lost, just moved somewhere it is not the row's identity.
        assert artifact.details.workflow_name == "Full-Stack CI/CD"

    def test_the_workflow_name_is_the_fallback(self) -> None:
        """Older runs and some event types carry no `display_title`, and a titleless
        row is worse than a repeated one."""
        artifact = map_workflow_run(run_payload(name="pre-commit"), source_id=SRC)

        assert artifact is not None
        assert artifact.title == "pre-commit"

    def test_finished_and_succeeded_are_separate_facts(self) -> None:
        artifact = map_workflow_run(run_payload(conclusion="failure"), source_id=SRC)
        assert artifact is not None
        assert artifact.state is ArtifactState.COMPLETED
        assert artifact.outcome is ArtifactOutcome.FAILURE
        assert artifact.failed

    def test_a_running_job_has_no_outcome_and_no_duration(self) -> None:
        """A duration computed from `updated_at` while it is still going measures
        how long it has been running, not how long it took -- and would be stored
        as though it were final."""
        artifact = map_workflow_run(
            run_payload(status="in_progress", conclusion=None), source_id=SRC
        )
        assert artifact is not None
        assert artifact.state is ArtifactState.RUNNING
        assert artifact.outcome is None
        assert artifact.details.duration_seconds is None
        assert artifact.details.completed_at is None

    def test_a_skipped_run_is_not_a_failure(self) -> None:
        """Treating anything non-success as failure would report every skipped
        job as broken, and "what is failing" would be useless."""
        artifact = map_workflow_run(run_payload(conclusion="skipped"), source_id=SRC)
        assert artifact is not None
        assert artifact.outcome is ArtifactOutcome.SKIPPED
        assert not artifact.failed

    def test_the_failing_job_is_recoverable(self) -> None:
        """ "The build failed" is not actionable. "The integration job failed" is."""
        artifact = map_workflow_run(
            run_payload(conclusion="failure"),
            source_id=SRC,
            jobs=[
                {"name": "unit", "conclusion": "success"},
                {"name": "integration", "conclusion": "failure"},
            ],
        )
        assert artifact is not None
        assert artifact.details.failed_jobs == ["integration"]

    def test_it_is_dated_by_when_it_started_running(self) -> None:
        """A run queued behind a concurrency group waits minutes. Dating it by
        creation reports it as slower than it was."""
        artifact = map_workflow_run(run_payload(created_at="2026-08-11T09:00:00Z"), source_id=SRC)
        assert artifact is not None
        assert artifact.occurred_at == datetime(2026, 8, 11, 10, 0, tzinfo=UTC)

    def test_a_scheduled_run_has_no_actor(self) -> None:
        artifact = map_workflow_run(run_payload(actor=None), source_id=SRC)
        assert artifact is not None
        assert artifact.actor_id is None

    def test_it_links_to_the_commit_it_tested(self) -> None:
        artifact = map_workflow_run(run_payload(), source_id=SRC)
        assert artifact is not None
        assert artifact.links[0].relation is LinkRelation.TESTS
        assert artifact.links[0].target_native_id == "b3516cb"


class TestPeople:
    def test_a_bot_is_marked_rather_than_dropped(self) -> None:
        """dependabot opens pull requests that get merged. Hiding them makes
        "what happened this week" wrong; marking them lets a reader decide."""
        bot = map_person({"login": "dependabot[bot]", "node_id": "U_bot", "type": "Bot"})
        assert bot is not None
        assert bot.is_bot

    def test_identity_is_the_node_id_not_the_login(self) -> None:
        """Logins are renameable, and the rename rewrites every URL containing
        one. Keyed on a handle, a person forks in two the first time somebody
        rebrands, and their history splits with them."""
        before = map_person({"login": "old-name", "node_id": "U_stable", "type": "User"})
        after = map_person({"login": "new-name", "node_id": "U_stable", "type": "User"})
        assert before is not None and after is not None
        assert before.id == after.id

    def test_a_missing_user_maps_to_nothing(self) -> None:
        assert map_person(None) is None
        assert map_person({}) is None


class TestEveryKindAgreesWithItsDetails:
    def test_the_kind_matches_the_detail_class(self) -> None:
        """The validator on `Artifact` catches a mismatch, so this asserts each
        mapper puts its artifact in the right family at all."""
        pairs = [
            (map_commit(commit_payload(), source_id=SRC), ArtifactKind.COMMIT),
            (map_pull_request(pull_payload(), source_id=SRC), ArtifactKind.PULL_REQUEST),
            (map_workflow_run(run_payload(), source_id=SRC), ArtifactKind.CI_RUN),
        ]
        for artifact, kind in pairs:
            assert artifact is not None
            assert artifact.kind is kind
            assert artifact.details is not None
            assert artifact.details.kind is kind
