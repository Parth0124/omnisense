"""ORM tables for investigations and their step history.

An investigation is a long-running, resumable user question (Design Doc §10).
LangGraph owns the *execution* state -- its checkpoints live in the separate
`checkpoints` schema so `pg_dump --schema=omnisense` captures application data
without orchestration scratch space (`models/orm/base.py`). What lives here is
the part a user, an auditor or a bill needs after the graph has finished: the
question asked, the plan chosen, the ordered record of which agent ran when, and
what it cost.

`investigation_steps` is a separate table rather than a JSON array on the parent
for three reasons. Steps are appended one at a time while the investigation is
running and the SSE timeline (`docs/api-reference.md` §5) reads them as they
land; the step sub-collection is paginated independently (`steps_limit`,
`steps_cursor` in §4.2); and per-agent latency analytics group across
investigations, which a JSON array cannot index.

Deliberately **not** soft-deletable. An investigation is derived: its query text
came from a user and its findings are reproducible from the Signals and the
recorded plan. When a user asks for an investigation to be deleted, the row must
actually go -- a tombstone still holding `query` would defeat that, for exactly
the reason `signals` is not soft-deletable either.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from models.base import TolerantStrEnum
from models.enums import AgentName, InvestigationStatus
from models.orm.base import Base, JSONVariant, TolerantEnumType
from models.orm.mixins import TenantMixin, TimestampMixin

__all__ = [
    "InvestigationRow",
    "InvestigationStepRow",
    "StepStatus",
]


class StepStatus(TolerantStrEnum):
    """Lifecycle of one agent execution inside an investigation.

    Distinct from `StageStatus` in `models/enums.py`, which is the outcome of an
    *enrichment* stage and has no running state because a stage is observed only
    after it finishes. A step is streamed live -- `docs/api-reference.md` §4.2
    shows a step in `"state": "running"` -- so the vocabulary needs the states a
    stage never has.

    Defined here rather than in `models/enums.py` because there is no domain-level
    `InvestigationStep` model yet. It should move once a second module needs it.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class InvestigationRow(Base, TimestampMixin, TenantMixin):
    """One user question and the aggregate state of the graph run answering it."""

    __tablename__ = "investigations"

    # -- identity ----------------------------------------------------------
    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    query: Mapped[str] = mapped_column(Text, nullable=False)
    """The user's question, verbatim. `Text` and not `String(n)`: truncating the
    question would silently change what was asked, and the whole point of storing
    it is that a report can be defended against it later."""

    created_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    """Principal that submitted the investigation. Nullable because Phase 1 has no
    real authentication (`docs/security-and-privacy.md` §3) and a scheduled or
    script-driven run has no user behind it."""

    # -- execution ---------------------------------------------------------
    status: Mapped[InvestigationStatus] = mapped_column(
        TolerantEnumType(InvestigationStatus, 32),
        nullable=False,
        default=InvestigationStatus.QUEUED,
    )

    plan: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)
    """The Planner's decomposition: sub-questions, agent sequence, budget.

    Nullable because it does not exist until the Planner runs -- a `queued`
    investigation has no plan, and defaulting to `{}` would make "not planned yet"
    indistinguishable from "planned to do nothing".
    """

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Terminal failure message. Carries the `code` of the raised `OmniSenseError`
    rather than a stack trace -- handlers and dashboards must never have to
    pattern-match on message text (`docs/coding-standards.md` §2.7)."""

    # -- accounting --------------------------------------------------------
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    """Denormalized count of `investigation_steps`. Maintained by the runtime so
    the list endpoint can render progress without a correlated subquery per row,
    and so the `max_steps` budget check is a single-row read."""

    token_input: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_output: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    cost_usd: Mapped[float] = mapped_column(
        Numeric(14, 6, asdecimal=False), nullable=False, default=0.0
    )
    """Rolled up from `agent_runs`. `NUMERIC` rather than `FLOAT` because this
    column is summed for billing: floating-point `SUM` over many rows is
    order-dependent, and PostgreSQL is free to reorder it under a parallel plan,
    so the same query can return two different totals. `asdecimal=False` keeps the
    Python type `float` on both dialects -- SQLite has no native decimal and
    would otherwise emit a lossy-conversion warning on every read in the unit
    suite, while the storage and the server-side `SUM` stay exact.
    """

    # -- output ------------------------------------------------------------
    report_id: Mapped[str | None] = mapped_column(
        String(64),
        # `use_alter=True` is load-bearing, not decoration. `reports` points back
        # at `investigations`, so the two tables form a foreign-key cycle that
        # `create_all` cannot order; `use_alter` emits this constraint as a
        # separate `ALTER TABLE` afterwards and breaks the cycle. SQLite reports
        # `supports_alter=False`, so the constraint is skipped there entirely --
        # which is correct, because SQLite does not enforce foreign keys unless
        # `PRAGMA foreign_keys=ON` anyway.
        ForeignKey("reports.id", ondelete="SET NULL", use_alter=True),
        nullable=True,
    )
    """The report this investigation produced, if it got that far.

    `SET NULL` rather than `CASCADE`: deleting a report must not delete the record
    that the investigation ran, what it cost and what it concluded. Nullable
    because `docs/api-reference.md` §4.1 allocates the report *id* eagerly at
    creation while the row itself appears later -- until it does, this stays
    `NULL` and `GET /reports/{id}` answers `409 report_not_ready`.
    """

    __table_args__ = (
        CheckConstraint("step_count >= 0", name="step_count_non_negative"),
        CheckConstraint("token_input >= 0", name="token_input_non_negative"),
        CheckConstraint("token_output >= 0", name="token_output_non_negative"),
        CheckConstraint("cost_usd >= 0", name="cost_non_negative"),
        # Time cannot run backwards, and an investigation cannot finish without
        # having started. Stated as an ordering invariant rather than as a list of
        # terminal statuses on purpose: `InvestigationStatus` is a tolerant enum
        # that may gain members, and a CHECK naming them would reintroduce exactly
        # the migration-per-member problem `TolerantEnumType` exists to avoid.
        CheckConstraint(
            "completed_at IS NULL OR (started_at IS NOT NULL AND completed_at >= started_at)",
            name="completed_after_started",
        ),
        # The list endpoint: one tenant's investigations, optionally filtered by
        # state, newest first.
        Index("ix_investigations_tenant_status_created", "tenant_id", "status", "created_at"),
        # The reaper: runs still in a non-terminal state past
        # INVESTIGATION_TIMEOUT_SECONDS. A partial index on the non-terminal
        # states would be tighter but would have to name them, so it would need a
        # migration every time the enum grows.
        Index("ix_investigations_status_started", "status", "started_at"),
    )


class InvestigationStepRow(Base, TimestampMixin, TenantMixin):
    """One agent execution within an investigation, in graph order."""

    __tablename__ = "investigation_steps"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    investigation_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False,
    )
    """`CASCADE`: a step has no meaning without its investigation, and a deletion
    request for the investigation must take the step payloads with it -- `input`
    and `output` below can quote the user's question and retrieved content."""

    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    """Monotonic position within the investigation. The Critic loop can re-run an
    agent (`docs/agent-system.md` §13), so `agent` repeats within an
    investigation and only this column gives the timeline a stable order."""

    agent: Mapped[AgentName] = mapped_column(TolerantEnumType(AgentName, 32), nullable=False)

    status: Mapped[StepStatus] = mapped_column(
        TolerantEnumType(StepStatus, 32), nullable=False, default=StepStatus.PENDING
    )

    input: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)
    """What the node was handed. Kept because reproducing a step means replaying
    it with the same input, not with whatever the current state would produce."""

    output: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)
    """The node's validated result. `NULL` while the step is running and after a
    failure -- an unvalidated partial output is worse than none, because the
    Critic would then be judging something the schema rejected."""

    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    """Wall-clock duration. Stored rather than derived from a `completed_at`
    because it is the field every dashboard groups on (`docs/coding-standards.md`
    §2.8 fixes `duration_ms` as the standard log field name), and because a step
    that never reported completion has a duration of "unknown", not "now minus
    started_at"."""

    token_input: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    token_output: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    """A step may issue several LLM calls; these are the sum over the
    `agent_runs` rows it produced, so the parent investigation's totals are a sum
    of steps rather than a join across the whole run table."""

    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # Two steps cannot occupy the same slot in one investigation's timeline.
        # This also serves the "steps for this investigation, in order" read that
        # both §4.2 and the SSE replay use, so no separate index is needed.
        UniqueConstraint("investigation_id", "sequence"),
        CheckConstraint("sequence >= 0", name="sequence_non_negative"),
        CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="duration_non_negative"),
        CheckConstraint("token_input >= 0", name="token_input_non_negative"),
        CheckConstraint("token_output >= 0", name="token_output_non_negative"),
        # Per-agent latency and failure rates across every investigation -- the
        # numbers behind "the Retriever is the slow one" in
        # `docs/observability.md`.
        Index("ix_investigation_steps_agent_started", "agent", "started_at"),
    )
