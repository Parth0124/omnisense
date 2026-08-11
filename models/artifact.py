"""What happened in a project: one shape for commits, PRs, CI runs, papers, messages.

`models/signal.py` describes an *observation about the world* -- a post with an
author, a sentiment and an engagement count. That shape was built for social media
and news, and it fits a GitHub issue well enough by accident. It fits a CI run not
at all: a run has no author, no sentiment and no body, and everything you would
actually ask about it -- did it pass, how long did it take, which job broke -- has
nowhere to live except an untyped metadata blob.

An `Artifact` is the developer-platform counterpart: **one thing that happened, or
exists, somewhere a project's work is recorded.** A commit is an artifact. So is a
pull request, a review, a CI run, an issue, a deployment, a Slack message, a
meeting, a design doc, an arXiv paper, and a task handed to a coding agent.

Why one type rather than one per kind
-------------------------------------
Because every question worth asking crosses kinds. "What happened this week" is a
time window over *all* of them, interleaved -- the tests broke, the discussion
followed, the PR fixed it, the paper informed it. Six types means six queries
stitched together, and every new feature has to remember all six; miss one and it
silently returns an incomplete answer, which is the worst kind of wrong.

What every kind genuinely shares is the frame: something happened, somewhere, at a
time, usually because of someone, and it is now in some state. That frame is the
columns below. What differs is the detail, and detail lives in `details` -- typed
per kind, not a loose bag.

`state` and `outcome`, and why both
------------------------------------
These are the pair that makes one type work across every kind, so they are worth
stating plainly. `state` is where a thing is in its life; `outcome` is how it
ended, and is absent while it is still going:

    commit        completed             --
    pull_request  open                  --            (merged -> state, not outcome)
    review        completed             changes_requested
    ci_run        running -> completed  -- -> failure
    issue         closed                not_planned
    paper         completed             --

That is what turns "show me everything that failed this week" into one indexed
query rather than six special cases.

Three tables, not one
---------------------
`Source` (where an artifact came from) and `Person` (who did it) are separate
rows referenced by id, rather than names repeated on every artifact. The reason is
not disk -- a repo name is fifteen bytes and the saving is meaningless. It is that
**both get renamed.** GitHub repositories are renamed and transferred between
organisations, and usernames change; the existing GitHub connector already keys
authors on `node_id` for exactly this reason. With the name written on every row,
one rename means rewriting two hundred thousand rows or living with stale ones.
With a reference it is a single update, and every artifact follows.

Layer note: **L0 `models/`** -- imports only from `models/` and the standard
library. Everything above may import it; it imports nothing above.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from models.base import StrictModel, TolerantStrEnum
from models.enums import Platform

__all__ = [
    "ARTIFACT_ID_PREFIX",
    "Artifact",
    "ArtifactDetails",
    "ArtifactKind",
    "ArtifactLink",
    "ArtifactOutcome",
    "ArtifactProvenance",
    "ArtifactState",
    "CIRunDetails",
    "CommitDetails",
    "DeploymentDetails",
    "FileChange",
    "IssueDetails",
    "JobResult",
    "LinkRelation",
    "Person",
    "PullRequestDetails",
    "ReviewDetails",
    "Source",
    "artifact_id",
    "person_id",
    "source_id",
]

ARTIFACT_ID_PREFIX = "art_"
SOURCE_ID_PREFIX = "src_"
PERSON_ID_PREFIX = "per_"

_ARTIFACT_NAMESPACE = uuid.UUID("4f1c9a2e-7b3d-4c5a-9e8f-2d6b1a0c3e7f")
"""Fixed namespace for deterministic ids. Never regenerate it.

A new namespace changes every id this system has ever derived, which does not
fail loudly -- it silently re-ingests the entire corpus as new rows alongside the
old ones.
"""


class ArtifactKind(enum.StrEnum):
    """What an artifact *is*. The discriminator.

    A plain `StrEnum`, deliberately not a `TolerantStrEnum`. Tolerance exists so a
    reader survives a value written by a newer producer, and it is right for
    `state` and `outcome` below, which come from platforms that add vocabulary
    without asking. It is wrong here: `kind` decides which rows a query returns
    and which `details` shape is parsed, so an unrecognised kind degrading to
    `UNKNOWN` would silently drop rows from every filtered read rather than
    failing where the bad value entered.
    """

    # Where the work happens.
    COMMIT = "commit"
    PULL_REQUEST = "pull_request"
    REVIEW = "review"
    CI_RUN = "ci_run"
    ISSUE = "issue"
    DEPLOYMENT = "deployment"

    # Where the talking happens. Not yet emitted by any connector; the columns
    # below were checked against them before being fixed. See the module docstring.
    MESSAGE = "message"
    MEETING = "meeting"
    DOCUMENT = "document"

    # Where the reading happens -- the world's output rather than yours.
    PAPER = "paper"
    MODEL = "model"

    # What the system did on your behalf. The record that makes an autonomous
    # action auditable rather than something that merely happened.
    AGENT_RUN = "agent_run"


class ArtifactState(TolerantStrEnum):
    """Where a thing is in its life. Tolerant: platforms add states.

    `COMPLETED` is the resting state for anything that has no lifecycle at all --
    a commit, a paper, a Slack message. Those are final the instant they exist,
    and inventing a per-kind spelling (`published`, `posted`) would buy nothing
    and cost every cross-kind query a longer `IN` list.
    """

    QUEUED = "queued"
    RUNNING = "running"
    OPEN = "open"
    DRAFT = "draft"
    CLOSED = "closed"
    MERGED = "merged"
    COMPLETED = "completed"
    UNKNOWN = "unknown"


class ArtifactOutcome(TolerantStrEnum):
    """How a thing ended. `None` while it is still going, or when it cannot end badly.

    Tolerant for the same reason as `ArtifactState`: GitHub has added conclusions
    to Actions more than once, and a new one must not break ingestion of every
    other run in the same page.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    COMMENTED = "commented"
    DISMISSED = "dismissed"
    NOT_PLANNED = "not_planned"
    UNKNOWN = "unknown"


class LinkRelation(TolerantStrEnum):
    """How one artifact relates to another, named from the subject's side.

    Directional and read subject-first: a review `REVIEWS` a pull request, a CI
    run `TESTS` a commit. Naming them from the subject means the edge reads the
    same way it was written, and there is exactly one place to look for it.
    """

    PART_OF = "part_of"
    REVIEWS = "reviews"
    TESTS = "tests"
    CLOSES = "closes"
    DEPLOYS = "deploys"
    PRODUCED = "produced"
    REFERENCES = "references"
    UNKNOWN = "unknown"


def _derive(prefix: str, *parts: str) -> str:
    joined = ":".join(parts)
    return prefix + uuid.uuid5(_ARTIFACT_NAMESPACE, joined).hex


def artifact_id(platform: Platform | str, native_id: str) -> str:
    """Deterministic artifact id from the platform and its own identifier.

    Pure and total, exactly like `signal_id`: the same inputs give the same id on
    any machine, forever. That is what lets a re-sync be an upsert rather than a
    duplicate, and what lets the graph reference an artifact it has not seen yet.

    `native_id` must be the platform's *stable* identifier -- a GitHub `node_id`,
    an arXiv id with its version, a Slack `channel:ts`. Not a URL and not a
    sequence number: GitHub renames rewrite every URL, and issue numbers repeat
    across repositories.
    """
    if not native_id:
        raise ValueError("native_id must be non-empty; identity cannot be derived")
    value = platform.value if isinstance(platform, Platform) else str(platform)
    return _derive(ARTIFACT_ID_PREFIX, value, native_id)


def source_id(platform: Platform | str, external_id: str) -> str:
    """Deterministic source id. `external_id` is the platform's own handle for the
    container -- a repository `node_id`, a Slack channel id, an arXiv category."""
    if not external_id:
        raise ValueError("external_id must be non-empty; identity cannot be derived")
    value = platform.value if isinstance(platform, Platform) else str(platform)
    return _derive(SOURCE_ID_PREFIX, value, external_id)


def person_id(platform: Platform | str, external_id: str) -> str:
    """Deterministic person id, keyed on the platform's immutable user id.

    Never on a username. GitHub logins are renameable and the rename rewrites
    every URL that contained one, so a person keyed on a handle silently forks
    into two the first time somebody rebrands -- and their history splits with
    them.
    """
    if not external_id:
        raise ValueError("external_id must be non-empty; identity cannot be derived")
    value = platform.value if isinstance(platform, Platform) else str(platform)
    return _derive(PERSON_ID_PREFIX, value, external_id)


# --------------------------------------------------------------------------- #
# The two referenced tables
# --------------------------------------------------------------------------- #


class Source(StrictModel):
    """Where artifacts come from: a repository, a channel, a space, a feed.

    Deliberately not called `Repository`. A Slack channel, a Notion space and an
    arXiv category are the same idea -- a container of activity a project cares
    about -- and naming it for the first one would mean renaming the table at
    step 10, along with every reference to it.
    """

    id: str
    tenant_id: str = "default"
    platform: Platform
    external_id: str = Field(
        description="The platform's own immutable id for this container. A GitHub "
        "repository node_id, a Slack channel id. Not the name: names are renamed."
    )
    name: str = Field(description="Canonical name, e.g. 'omnisense/api' or '#eng-scheduler'.")
    display_name: str | None = None
    url: str | None = None
    default_branch: str | None = Field(
        default=None, description="Repositories only; `None` everywhere else."
    )
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Person(StrictModel):
    """One human (or bot) on one platform.

    Per *platform*, not per human. The same person on GitHub and Slack is two
    rows here, and joining them into one identity is a cross-source inference
    with a confidence attached -- which belongs in the graph, not in a foreign
    key. Asserting it here would make a guess indistinguishable from a fact.
    """

    id: str
    tenant_id: str = "default"
    platform: Platform
    external_id: str = Field(
        description="The platform's immutable user id -- GitHub node_id, Slack member id."
    )
    handle: str | None = None
    display_name: str | None = None
    email: str | None = None
    avatar_url: str | None = None
    is_bot: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Per-kind detail
# --------------------------------------------------------------------------- #


class FileChange(StrictModel):
    """One file touched by a commit or a pull request."""

    path: str
    status: str | None = Field(default=None, description="added | modified | removed | renamed")
    additions: int = 0
    deletions: int = 0
    previous_path: str | None = Field(
        default=None, description="Set on a rename, so history can be followed across it."
    )


class JobResult(StrictModel):
    """One job inside a CI run.

    Present because "the build failed" is not actionable and "the `integration`
    job failed" is. The failing job is the first thing anyone asks for and the
    only part of a run most people ever read.
    """

    name: str
    outcome: ArtifactOutcome | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class CommitDetails(StrictModel):
    """A commit.

    `author` and `committer` are separate because they genuinely differ -- a
    rebase, a cherry-pick or a squash-merge rewrites the committer while keeping
    the author. Collapsing them credits the wrong person for the work, which is
    exactly the question "who changed this" is asked to answer.
    """

    kind: Literal[ArtifactKind.COMMIT] = ArtifactKind.COMMIT
    sha: str
    parent_shas: list[str] = Field(
        default_factory=list, description="Two or more means this is a merge commit."
    )
    committer_id: str | None = Field(
        default=None, description="Person id, when it differs from the artifact's actor."
    )
    committed_at: datetime | None = None
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    files: list[FileChange] = Field(default_factory=list)
    is_merge: bool = False
    verified: bool | None = Field(default=None, description="Signature verified by the platform.")


class PullRequestDetails(StrictModel):
    """A pull request.

    `has_conflicts` is here rather than derived on read because "which of my PRs
    have merge conflicts" is a question asked directly, and a conflict is a
    property of the PR right now rather than an event with its own identity --
    fix the branch and it disappears with nothing to record.
    """

    kind: Literal[ArtifactKind.PULL_REQUEST] = ArtifactKind.PULL_REQUEST
    number: int
    base_ref: str | None = None
    head_ref: str | None = None
    draft: bool = False
    merged_at: datetime | None = None
    closed_at: datetime | None = None
    merge_commit_sha: str | None = None
    has_conflicts: bool | None = Field(
        default=None, description="`None` means the platform has not computed mergeability yet."
    )
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    commit_count: int = 0
    requested_reviewer_ids: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


class ReviewDetails(StrictModel):
    """A code review. Its verdict lives in the artifact's `outcome`."""

    kind: Literal[ArtifactKind.REVIEW] = ArtifactKind.REVIEW
    pull_request_number: int | None = None
    submitted_at: datetime | None = None
    comment_count: int = 0
    commit_sha: str | None = Field(
        default=None,
        description="Which revision was reviewed -- a review of an outdated one is weaker evidence.",
    )


class CIRunDetails(StrictModel):
    """A CI or Actions run."""

    kind: Literal[ArtifactKind.CI_RUN] = ArtifactKind.CI_RUN
    workflow_name: str | None = None
    run_number: int | None = None
    run_attempt: int = 1
    event: str | None = Field(default=None, description="push | pull_request | schedule | ...")
    head_sha: str | None = None
    head_branch: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_seconds: float | None = None
    jobs: list[JobResult] = Field(default_factory=list)
    logs_url: str | None = None

    @property
    def failed_jobs(self) -> list[str]:
        """Names of the jobs that did not succeed. The actionable part of a failure."""
        return [
            job.name
            for job in self.jobs
            if job.outcome not in (None, ArtifactOutcome.SUCCESS, ArtifactOutcome.SKIPPED)
        ]


class IssueDetails(StrictModel):
    """An issue or ticket. Closing *reason* is the artifact's `outcome`."""

    kind: Literal[ArtifactKind.ISSUE] = ArtifactKind.ISSUE
    number: int | None = None
    closed_at: datetime | None = None
    comment_count: int = 0
    labels: list[str] = Field(default_factory=list)
    assignee_ids: list[str] = Field(default_factory=list)


class DeploymentDetails(StrictModel):
    """A deployment to an environment."""

    kind: Literal[ArtifactKind.DEPLOYMENT] = ArtifactKind.DEPLOYMENT
    environment: str
    ref: str | None = None
    sha: str | None = None
    deployed_at: datetime | None = None
    is_production: bool = False


ArtifactDetails = Annotated[
    CommitDetails
    | PullRequestDetails
    | ReviewDetails
    | CIRunDetails
    | IssueDetails
    | DeploymentDetails,
    Field(discriminator="kind"),
]
"""The kind-specific part, discriminated on `kind` so it stays type-checked.

Only the six kinds a connector can currently produce appear here. The rest --
message, meeting, document, paper, model, agent_run -- are valid `ArtifactKind`
values whose `details` is `None` until something emits them. That asymmetry is
deliberate: the shared columns were designed against those kinds and verified
against a real arXiv response, but writing detail classes for payloads no code
produces yet would be guessing at fields nobody can check.
"""


# --------------------------------------------------------------------------- #
# The artifact
# --------------------------------------------------------------------------- #


class ArtifactLink(StrictModel):
    """A pointer from this artifact to another.

    Holds the *target's* native id rather than its artifact id, because the two
    are often ingested out of order -- a PR references an issue that has not been
    fetched yet. Storing the native id means the link survives that, and
    resolution to a real row happens whenever the target arrives.
    """

    relation: LinkRelation
    target_native_id: str
    target_kind: ArtifactKind | None = None
    url: str | None = None


class ArtifactProvenance(StrictModel):
    """How this artifact came to exist. The receipt.

    A lean cousin of `models/lineage.Lineage`, which is documented as "how a
    Signal came to exist" and carries the eight-stage enrichment vocabulary --
    stage records, a `SignalStatus`, dedup cluster ids. Artifacts do not travel
    that pipeline, and importing a type whose fields can never be set would make
    every reader wonder which ones matter.
    """

    connector_slug: str
    connector_version: str = "0.0.0"
    sync_run_id: str | None = None
    fetched_at: datetime
    raw_object_key: str | None = Field(
        default=None, description="Where the untouched payload was archived, for reprocessing."
    )
    raw_sha256: str | None = None
    request_fingerprint: str | None = None


class Artifact(StrictModel):
    """One thing that happened in a project.

    Field order below is the order the questions are asked in: what is it, where
    did it come from, who did it, when, what state is it in, what does it point
    at, and finally the kind-specific detail.
    """

    id: str
    tenant_id: str = "default"

    kind: ArtifactKind
    source_id: str = Field(description="`Source.id`. Never a repository name.")
    actor_id: str | None = Field(
        default=None,
        description="`Person.id`. `None` is meaningful and common -- a CI run has no "
        "human author, and a deleted GitHub account leaves an artifact with no actor.",
    )

    platform: Platform
    native_id: str
    url: str | None = None

    title: str | None = Field(
        default=None,
        description="A commit's subject line, a PR's title, a workflow's name. `None` "
        "for kinds that have none, such as a review.",
    )
    body: str | None = None

    occurred_at: datetime = Field(
        description="When it happened *at the source*. Never ingestion time -- a repo "
        "synced today is full of things that happened over three years."
    )
    updated_at_source: datetime | None = Field(
        default=None, description="Last modification at the source; drives incremental sync."
    )

    state: ArtifactState = ArtifactState.COMPLETED
    outcome: ArtifactOutcome | None = None

    links: list[ArtifactLink] = Field(default_factory=list)
    details: ArtifactDetails | None = None

    provenance: ArtifactProvenance
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-namespaced overflow, e.g. `github.milestone`. Namespaced "
        "because one jsonb column serves every connector and an unprefixed key collides.",
    )

    @model_validator(mode="after")
    def _details_match_kind(self) -> Artifact:
        """`details` must describe the same kind the artifact declares.

        The discriminated union checks that `details` is *internally* consistent
        -- that a payload tagged `pull_request` parses as `PullRequestDetails`.
        It cannot check that the payload agrees with the row it is attached to,
        so `kind=commit` with pull-request details validated cleanly, stored
        cleanly, and would have failed much later on read, in whichever feature
        happened to touch that row first.

        Kinds with no detail class -- message, paper, agent_run -- are unaffected:
        `None` is always acceptable.
        """
        if self.details is not None and self.details.kind is not self.kind:
            raise ValueError(
                f"artifact kind is {self.kind.value!r} but details describe "
                f"{self.details.kind.value!r}; they must agree"
            )
        return self

    @property
    def is_finished(self) -> bool:
        """Whether this has stopped moving. Everything else is still in flight."""
        return self.state in (
            ArtifactState.CLOSED,
            ArtifactState.MERGED,
            ArtifactState.COMPLETED,
        )

    @property
    def failed(self) -> bool:
        """Ended badly. The predicate behind "what is broken".

        Reads `outcome` rather than `state`, because a CI run that finished and a
        CI run that passed are different claims and only one of them is good news.
        """
        return self.outcome in (
            ArtifactOutcome.FAILURE,
            ArtifactOutcome.TIMED_OUT,
            ArtifactOutcome.CANCELLED,
        )
