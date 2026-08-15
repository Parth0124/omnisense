"""Storage for `models/artifact.py`: three tables, and the indexes that matter.

`sources`, `people` and `artifacts`. The split exists because sources and people
get *renamed* -- see the domain module's docstring -- so their names are written
once and referenced by id.

**Why one `artifacts` table and not one per kind.** Every question the product
answers crosses kinds: "what happened this week" is a time window over commits,
pull requests, CI runs and messages interleaved. Six tables makes that a six-way
union that changes shape whenever a kind is added, and reads *more* pages than
the single-table version because it sorts and merges six result sets instead of
walking one index. Splitting would optimise for table size, which is not a problem
Postgres has at this scale, at the cost of the query the product is built on.

If it ever does become one, Postgres partitions a single table by range or by list
without a single query changing -- so the escape hatch stays open and costs
nothing to leave unused.

**On the indexes.** Every one below exists for a query someone will actually run,
and each is prefixed by `tenant_id` because every read is tenant-scoped and a
composite index is only usable from its leading column. An index that is never the
best plan is not free: it is paid for on every insert, and this table takes the
highest write volume in the system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from models.artifact import ArtifactKind, ArtifactOutcome, ArtifactState, WatchStatus
from models.enums import Platform
from models.orm.base import Base, JSONVariant, TolerantEnumType
from models.orm.mixins import TenantMixin, TimestampMixin, tenant_scoped_index

__all__ = ["ArtifactRow", "PersonRow", "SourceRow"]


class SourceRow(TenantMixin, TimestampMixin, Base):
    """A repository, channel, space or feed. Written once, referenced by every artifact."""

    __tablename__ = "sources"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    # Nullable, and that is a real state rather than a gap: a source is
    # discovered by a connector and assigned to a project by a person, and those
    # do not happen in the same instant. An unassigned source still collects
    # artifacts; they simply do not answer any project-scoped question yet.
    #
    # `SET NULL` on delete, not `CASCADE`: removing a project must not delete the
    # repositories it grouped, still less their history. It un-groups them.
    project_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="SET NULL"), nullable=True
    )

    platform: Mapped[Platform] = mapped_column(
        TolerantEnumType(Platform), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_branch: Mapped[str | None] = mapped_column(String(256), nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    watch_status: Mapped[WatchStatus] = mapped_column(
        TolerantEnumType(WatchStatus),
        nullable=False,
        default=WatchStatus.INCLUDED,
        server_default=WatchStatus.INCLUDED.value,
        index=True,
    )
    """Whether this source is synced, and whether we are still asking.

    `server_default` because the column arrives on a table that already holds
    rows, and every one of those was added by hand -- so the safe backfill is
    `INCLUDED`, not `PENDING`. Defaulting them to pending would silently stop
    syncing repositories somebody had already chosen, and the symptom would be a
    project that quietly went stale rather than an error.

    Indexed because the two hot reads are both filters on it: "what do I sync"
    and "what is waiting for me to decide".
    """

    source_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, default=dict
    )

    __table_args__ = (
        # The platform's own id is the identity, not the name. A repository that
        # is renamed keeps this row and this id; only `name` changes, and every
        # artifact pointing here follows for free.
        UniqueConstraint("platform", "external_id", name="uq_sources_platform_external_id"),
        tenant_scoped_index("sources", "platform", "name"),
        # "Every source in this project" -- walked on every project-scoped read,
        # which is most of them.
        tenant_scoped_index("sources", "project_id"),
    )


class PersonRow(TenantMixin, TimestampMixin, Base):
    """One account on one platform.

    Not one human. The same person on GitHub and Slack is two rows, and deciding
    they are the same human is a cross-source inference carrying a confidence --
    which belongs in the graph, where a guess can be stored *as* a guess. A
    foreign key here would make it indistinguishable from a fact.
    """

    __tablename__ = "people"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    platform: Mapped[Platform] = mapped_column(
        TolerantEnumType(Platform), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(String(256), nullable=False)
    handle: Mapped[str | None] = mapped_column(String(256), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    email: Mapped[str | None] = mapped_column(String(512), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_bot: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    person_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, default=dict
    )

    __table_args__ = (
        UniqueConstraint("platform", "external_id", name="uq_people_platform_external_id"),
        tenant_scoped_index("people", "platform", "handle"),
    )


class ArtifactRow(TenantMixin, TimestampMixin, Base):
    """One thing that happened, whatever kind of thing it was."""

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    kind: Mapped[ArtifactKind] = mapped_column(
        TolerantEnumType(ArtifactKind), nullable=False, index=True
    )

    # `ondelete="RESTRICT"`: deleting a source out from under its artifacts would
    # orphan them, and an artifact whose origin is unknown cannot be cited, which
    # is the one thing every claim in this system depends on. Removing a source
    # means deciding what happens to its history, explicitly.
    source_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("sources.id", ondelete="RESTRICT"), nullable=False
    )
    # Nullable and `SET NULL`: plenty of artifacts have no actor at all -- a CI
    # run is triggered by a machine -- and a deleted account must not take the
    # commits it authored with it.
    actor_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("people.id", ondelete="SET NULL"), nullable=True
    )

    platform: Mapped[Platform] = mapped_column(TolerantEnumType(Platform), nullable=False)
    native_id: Mapped[str] = mapped_column(String(512), nullable=False)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)

    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    body: Mapped[str | None] = mapped_column(Text, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at_source: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    state: Mapped[ArtifactState] = mapped_column(
        TolerantEnumType(ArtifactState), nullable=False, default=ArtifactState.COMPLETED
    )
    outcome: Mapped[ArtifactOutcome | None] = mapped_column(
        TolerantEnumType(ArtifactOutcome), nullable=True
    )

    links: Mapped[list[dict[str, Any]]] = mapped_column(JSONVariant, nullable=False, default=list)
    details: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    artifact_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, default=dict
    )

    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __table_args__ = (
        # Identity. A re-sync of the same object is an upsert onto this, which is
        # what makes a crashed backfill safe to simply run again.
        UniqueConstraint("platform", "native_id", name="uq_artifacts_platform_native_id"),
        # "What happened in this source lately" -- the timeline read, and the one
        # `catch-up` is built on. Descending, because every caller wants newest
        # first and a backwards scan of a forward index is measurably slower.
        Index(
            "ix_artifacts_tenant_source_time",
            "tenant_id",
            "source_id",
            occurred_at.desc(),
        ),
        # "Show me just the pull requests" -- the same window narrowed by kind.
        Index(
            "ix_artifacts_tenant_kind_time",
            "tenant_id",
            "kind",
            occurred_at.desc(),
        ),
        # "What is still open" and "what failed". Partial rather than full: a
        # finished artifact is never the answer to either question, and the vast
        # majority of rows are finished, so indexing them costs write throughput
        # to store pointers no query follows.
        Index(
            "ix_artifacts_tenant_open",
            "tenant_id",
            "state",
            postgresql_where=state.in_(
                (
                    ArtifactState.OPEN,
                    ArtifactState.DRAFT,
                    ArtifactState.RUNNING,
                    ArtifactState.QUEUED,
                )
            ),
        ),
        Index(
            "ix_artifacts_tenant_failed",
            "tenant_id",
            "outcome",
            occurred_at.desc(),
            postgresql_where=outcome.in_(
                (ArtifactOutcome.FAILURE, ArtifactOutcome.TIMED_OUT, ArtifactOutcome.CANCELLED)
            ),
        ),
        # "Everything this person did", across kinds and sources.
        Index("ix_artifacts_tenant_actor_time", "tenant_id", "actor_id", occurred_at.desc()),
        # Incremental sync reads this to find what changed since the last run.
        Index("ix_artifacts_source_updated", "source_id", "updated_at_source"),
    )
