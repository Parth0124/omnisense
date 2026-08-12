"""Turn GitHub payloads into `Artifact`s. Pure functions, no network, no database.

Separate from `reader.py` so every mapping below can be checked against a
recorded payload with nothing running -- which matters more here than usual,
because a mapping bug is silent. A commit stored with the wrong timestamp does
not fail; it appears in the wrong week, forever, and nothing ever says so.

Two rules run through all of it
-------------------------------
**Event time, never ingestion time.** A repository synced today is full of things
that happened over three years, and `occurred_at` is what every window query
reads. Using the moment we fetched would file the entire history as "today", and
the first `catch-up` would report three years of work as this morning's.

**Identity is the platform's, never ours.** Every artifact and person is keyed on
a `node_id`. Numbers repeat across repositories, URLs change when a repository is
renamed, and logins change whenever their owner feels like it -- and each of
those, used as a key, silently forks one thing's history into two.

What is deliberately dropped
----------------------------
Bot accounts are marked rather than skipped: `dependabot` genuinely does open
pull requests that genuinely do get merged, and hiding them would make "what
happened this week" wrong. Marking them lets a reader decide, which is the
difference between a filter and a lie.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from connectors.github.reader import parse_time
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
    Person,
    PullRequestDetails,
    ReviewDetails,
    artifact_id,
    person_id,
)
from models.enums import Platform

__all__ = [
    "CONNECTOR_SLUG",
    "map_commit",
    "map_person",
    "map_pull_request",
    "map_review",
    "map_workflow_run",
]

CONNECTOR_SLUG = "github"
CONNECTOR_VERSION = "0.1.0"

_REVIEW_OUTCOMES: Mapping[str, ArtifactOutcome] = {
    "APPROVED": ArtifactOutcome.APPROVED,
    "CHANGES_REQUESTED": ArtifactOutcome.CHANGES_REQUESTED,
    "COMMENTED": ArtifactOutcome.COMMENTED,
    "DISMISSED": ArtifactOutcome.DISMISSED,
}

_RUN_OUTCOMES: Mapping[str, ArtifactOutcome] = {
    "success": ArtifactOutcome.SUCCESS,
    "failure": ArtifactOutcome.FAILURE,
    "cancelled": ArtifactOutcome.CANCELLED,
    "timed_out": ArtifactOutcome.TIMED_OUT,
    "skipped": ArtifactOutcome.SKIPPED,
    # GitHub's own vocabulary, mapped to ours rather than carried through: a
    # `neutral` or `stale` conclusion is not a success, and treating anything
    # non-`success` as failure would report a skipped job as broken.
    "action_required": ArtifactOutcome.FAILURE,
    "startup_failure": ArtifactOutcome.FAILURE,
    "neutral": ArtifactOutcome.UNKNOWN,
    "stale": ArtifactOutcome.UNKNOWN,
}

_RUN_STATES: Mapping[str, ArtifactState] = {
    "queued": ArtifactState.QUEUED,
    "requested": ArtifactState.QUEUED,
    "waiting": ArtifactState.QUEUED,
    "pending": ArtifactState.QUEUED,
    "in_progress": ArtifactState.RUNNING,
    "completed": ArtifactState.COMPLETED,
}


def _int_or_none(value: Any) -> int | None:
    """An integer, or `None` when the field was not in the payload at all.

    The distinction the whole size-field design turns on: `0` says "changed
    nothing", `None` says "we did not ask". GitHub's list endpoints omit these
    entirely, so conflating the two would fill the table with confident zeroes.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _time(value: Any) -> datetime | None:
    """Aliased rather than reimplemented -- the reader already owns this."""
    return parse_time(value)


def _provenance(fetched_at: datetime | None = None) -> ArtifactProvenance:
    return ArtifactProvenance(
        connector_slug=CONNECTOR_SLUG,
        connector_version=CONNECTOR_VERSION,
        fetched_at=fetched_at or datetime.now(UTC),
    )


def map_person(payload: Mapping[str, Any] | None) -> Person | None:
    """A GitHub user account. `None` when the account is gone.

    GitHub answers with `null` for a deleted account and with a `ghost` user for
    content it once owned. Both are real states: the work happened, the person is
    no longer reachable, and inventing a placeholder person would put an author on
    a commit that has none.
    """
    if not payload or not isinstance(payload, Mapping):
        return None
    node_id = payload.get("node_id")
    if not node_id:
        return None
    login = payload.get("login")
    return Person(
        id=person_id(Platform.GITHUB, str(node_id)),
        tenant_id="local",
        platform=Platform.GITHUB,
        external_id=str(node_id),
        handle=login,
        display_name=payload.get("name") or login,
        avatar_url=payload.get("avatar_url"),
        # Marked, not filtered. dependabot opens pull requests that get merged,
        # and hiding them makes "what happened this week" wrong.
        is_bot=payload.get("type") == "Bot" or bool(login and login.endswith("[bot]")),
    )


def map_commit(
    payload: Mapping[str, Any], *, source_id: str, fetched_at: datetime | None = None
) -> Artifact | None:
    """One commit.

    `commit.author.date`, not `commit.committer.date`. They differ whenever
    somebody rebases, and the authored date is when the work was done -- which is
    what "what happened this week" means. A rebase would otherwise move a year of
    history into the afternoon somebody tidied the branch.
    """
    node_id = payload.get("node_id")
    if not node_id:
        return None

    commit = payload.get("commit") or {}
    message = str(commit.get("message") or "")
    subject, _, body = message.partition("\n")

    authored = _time((commit.get("author") or {}).get("date"))
    committed = _time((commit.get("committer") or {}).get("date"))
    if authored is None:
        return None

    author = map_person(payload.get("author"))
    committer = map_person(payload.get("committer"))
    parents = [p.get("sha") for p in payload.get("parents") or [] if p.get("sha")]
    stats = payload.get("stats") or {}
    files = payload.get("files") or []

    return Artifact(
        id=artifact_id(Platform.GITHUB, str(node_id)),
        tenant_id="local",
        kind=ArtifactKind.COMMIT,
        source_id=source_id,
        actor_id=author.id if author else None,
        platform=Platform.GITHUB,
        native_id=str(node_id),
        url=payload.get("html_url"),
        title=subject[:1000] or None,
        body=body.strip() or None,
        occurred_at=authored,
        updated_at_source=committed,
        state=ArtifactState.COMPLETED,
        details=CommitDetails(
            sha=str(payload.get("sha") or ""),
            parent_shas=[str(p) for p in parents],
            committer_id=committer.id
            if committer and committer.id != (author.id if author else None)
            else None,
            committed_at=committed,
            # `None` for a commit read from the *list* endpoint, which carries
            # neither `stats` nor `files`. Filling them in costs one request per
            # commit -- a hundred extra per page -- so they arrive only when a
            # caller has fetched the commit individually.
            #
            # Absent rather than zero, deliberately: zero is a claim that the
            # commit changed nothing, and no later query could tell that apart
            # from "we never asked".
            additions=_int_or_none(stats.get("additions")),
            deletions=_int_or_none(stats.get("deletions")),
            changed_files=_int_or_none(stats.get("total")) if not files else len(files),
            # Two or more parents is the definition of a merge; the API has no
            # flag for it.
            is_merge=len(parents) > 1,
            verified=bool((payload.get("commit") or {}).get("verification", {}).get("verified"))
            if (payload.get("commit") or {}).get("verification")
            else None,
        ),
        provenance=_provenance(fetched_at),
        metadata={"github.sha": payload.get("sha")},
    )


def map_pull_request(
    payload: Mapping[str, Any], *, source_id: str, fetched_at: datetime | None = None
) -> Artifact | None:
    """One pull request, with its lifecycle in `state`.

    `merged` is a *state*, not an outcome. A merged pull request did not "end
    successfully" in the way a CI run does -- it reached a different resting
    place from `closed`, and conflating the two loses the only distinction anyone
    cares about when asking what shipped.
    """
    node_id = payload.get("node_id")
    created = _time(payload.get("created_at"))
    if not node_id or created is None:
        return None

    merged_at = _time(payload.get("merged_at"))
    closed_at = _time(payload.get("closed_at"))

    if merged_at:
        state = ArtifactState.MERGED
    elif closed_at:
        state = ArtifactState.CLOSED
    elif payload.get("draft"):
        state = ArtifactState.DRAFT
    else:
        state = ArtifactState.OPEN

    author = map_person(payload.get("user"))
    base = (payload.get("base") or {}).get("ref")
    head = (payload.get("head") or {}).get("ref")

    links: list[ArtifactLink] = []
    merge_sha = payload.get("merge_commit_sha")
    if merge_sha:
        links.append(
            ArtifactLink(
                relation=LinkRelation.PART_OF,
                target_native_id=str(merge_sha),
                target_kind=ArtifactKind.COMMIT,
            )
        )

    return Artifact(
        id=artifact_id(Platform.GITHUB, str(node_id)),
        tenant_id="local",
        kind=ArtifactKind.PULL_REQUEST,
        source_id=source_id,
        actor_id=author.id if author else None,
        platform=Platform.GITHUB,
        native_id=str(node_id),
        url=payload.get("html_url"),
        title=payload.get("title"),
        body=payload.get("body"),
        occurred_at=created,
        updated_at_source=_time(payload.get("updated_at")),
        state=state,
        links=links,
        details=PullRequestDetails(
            number=int(payload.get("number") or 0),
            base_ref=base,
            head_ref=head,
            draft=bool(payload.get("draft")),
            merged_at=merged_at,
            closed_at=closed_at,
            merge_commit_sha=merge_sha,
            # `mergeable` is null until GitHub has computed it, which is a third
            # state and not the same as "no conflicts". Carried through as None
            # rather than coerced to False, which would claim it is mergeable.
            has_conflicts=(payload.get("mergeable") is False)
            if payload.get("mergeable") is not None
            else None,
            # Absent from the list endpoint, same as the commit fields above --
            # and `None` rather than `0` for the same reason.
            additions=_int_or_none(payload.get("additions")),
            deletions=_int_or_none(payload.get("deletions")),
            changed_files=_int_or_none(payload.get("changed_files")),
            commit_count=_int_or_none(payload.get("commits")),
            requested_reviewer_ids=[
                person.id
                for person in (map_person(r) for r in payload.get("requested_reviewers") or [])
                if person
            ],
            labels=[
                str(label.get("name")) for label in payload.get("labels") or [] if label.get("name")
            ],
        ),
        provenance=_provenance(fetched_at),
        metadata={"github.number": payload.get("number")},
    )


def map_review(
    payload: Mapping[str, Any],
    *,
    source_id: str,
    pull_number: int,
    pull_node_id: str,
    fetched_at: datetime | None = None,
) -> Artifact | None:
    """One review. Its verdict is the artifact's `outcome`.

    A review has no title -- the thing being reviewed has one -- so `title` is
    `None` rather than a manufactured "Review of #4181". Inventing one would make
    a list of titles read as though someone wrote them.
    """
    node_id = payload.get("node_id")
    submitted = _time(payload.get("submitted_at"))
    if not node_id or submitted is None:
        # A review with no `submitted_at` is a *pending* review: drafted, visible
        # only to its author, not yet an event. Storing it would show a verdict
        # nobody has given.
        return None

    reviewer = map_person(payload.get("user"))
    return Artifact(
        id=artifact_id(Platform.GITHUB, str(node_id)),
        tenant_id="local",
        kind=ArtifactKind.REVIEW,
        source_id=source_id,
        actor_id=reviewer.id if reviewer else None,
        platform=Platform.GITHUB,
        native_id=str(node_id),
        url=payload.get("html_url"),
        title=None,
        body=payload.get("body") or None,
        occurred_at=submitted,
        state=ArtifactState.COMPLETED,
        outcome=_REVIEW_OUTCOMES.get(str(payload.get("state") or "").upper()),
        links=[
            ArtifactLink(
                relation=LinkRelation.REVIEWS,
                target_native_id=pull_node_id,
                target_kind=ArtifactKind.PULL_REQUEST,
            )
        ],
        details=ReviewDetails(
            pull_request_number=pull_number,
            submitted_at=submitted,
            commit_sha=payload.get("commit_id"),
        ),
        provenance=_provenance(fetched_at),
    )


def map_workflow_run(
    payload: Mapping[str, Any],
    *,
    source_id: str,
    jobs: Sequence[Mapping[str, Any]] | None = None,
    fetched_at: datetime | None = None,
) -> Artifact | None:
    """One Actions run.

    `run_started_at` when present, `created_at` otherwise: a run queued behind a
    concurrency group can wait minutes, and dating it by creation reports it as
    slower than it was.
    """
    node_id = payload.get("node_id")
    started = _time(payload.get("run_started_at")) or _time(payload.get("created_at"))
    if not node_id or started is None:
        return None

    completed = _time(payload.get("updated_at"))
    status = str(payload.get("status") or "")
    conclusion = str(payload.get("conclusion") or "")

    state = _RUN_STATES.get(status, ArtifactState.UNKNOWN)
    outcome = _RUN_OUTCOMES.get(conclusion) if conclusion else None

    duration = None
    if completed and state is ArtifactState.COMPLETED:
        duration = max(0.0, (completed - started).total_seconds())

    # The person whose push triggered it, when there is one. A scheduled run has
    # no actor at all, which is why the column is nullable.
    actor = map_person(payload.get("actor"))

    return Artifact(
        id=artifact_id(Platform.GITHUB, str(node_id)),
        tenant_id="local",
        kind=ArtifactKind.CI_RUN,
        source_id=source_id,
        actor_id=actor.id if actor else None,
        platform=Platform.GITHUB,
        native_id=str(node_id),
        url=payload.get("html_url"),
        # `display_title` first, which is the *run's* title -- the message of the
        # commit that triggered it, and what GitHub itself shows in the Actions
        # list. `name` is the workflow's name, identical on every run it ever
        # produces: four runs of one workflow stored four rows all reading
        # "Full-Stack CI/CD - Deploy Frontend & Backend", with nothing to tell
        # them apart. The workflow name is not lost -- it is in `details`.
        title=payload.get("display_title") or payload.get("name"),
        body=None,
        occurred_at=started,
        updated_at_source=completed,
        state=state,
        outcome=outcome,
        links=[
            ArtifactLink(
                relation=LinkRelation.TESTS,
                target_native_id=str(payload["head_sha"]),
                target_kind=ArtifactKind.COMMIT,
            )
        ]
        if payload.get("head_sha")
        else [],
        details=CIRunDetails(
            workflow_name=payload.get("name"),
            run_number=int(payload.get("run_number") or 0) or None,
            run_attempt=int(payload.get("run_attempt") or 1),
            event=payload.get("event"),
            head_sha=payload.get("head_sha"),
            head_branch=payload.get("head_branch"),
            started_at=started,
            completed_at=completed if state is ArtifactState.COMPLETED else None,
            duration_seconds=duration,
            jobs=[
                JobResult(
                    name=str(job.get("name") or ""),
                    outcome=_RUN_OUTCOMES.get(str(job.get("conclusion") or "")),
                    started_at=_time(job.get("started_at")),
                    completed_at=_time(job.get("completed_at")),
                )
                for job in jobs or []
            ],
            logs_url=payload.get("logs_url"),
        ),
        provenance=_provenance(fetched_at),
        metadata={"github.workflow_id": payload.get("workflow_id")},
    )
