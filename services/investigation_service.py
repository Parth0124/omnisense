"""Create, resume, cancel and inspect long-running investigations.

This module owns the *lifecycle and persistence* of an investigation and nothing
else. The orchestration -- which agent runs next, how the Critic loop decides to
go round again -- is Layer 5 (`agents/`, `docs/agent-system.md`). The split is
deliberate: LangGraph already owns execution state in its own `checkpoints`
schema (`models/orm/investigation.py`), and a service that also tried to own it
would give the same run two sources of truth that drift the first time a worker
dies between the graph checkpoint and the row.

What lives here is the part that outlives the graph: the question asked, the
plan, the ordered step history, what it cost, and -- above all -- the state
machine.

The state machine is the correctness property
---------------------------------------------
`InvestigationStatus.is_terminal` already names the four states from which no
further transition is possible, and this module refuses every one of them. That
refusal is not bookkeeping. A `cancel` that lands on an investigation which
already `completed` would move a finished run back to `cancelled`, and every
downstream reading of that row -- the billing rollup, the report's provenance,
the SSE stream's terminal event, the reaper's "is this still alive" sweep --
would then describe a run that did not happen that way. Worse, it is a *silent*
corruption: nothing raises, the row simply says something untrue.

So there are two guards, in this order:

1. `is_terminal` on the current state. Terminal means terminal, including a
   transition to the same terminal state -- cancelling an already-cancelled
   investigation is a `409`, not a no-op, because the caller believed it was
   stopping something that was still running and deserves to be told otherwise.
2. `ALLOWED_TRANSITIONS`, an explicit successor table. Being non-terminal is not
   a licence to move anywhere: `queued -> completed` would produce a report from
   an investigation that never retrieved anything.

Both are enforced with a **compare-and-swap `UPDATE`** rather than a read
followed by a write. Two workers can hold the same investigation -- the reaper
timing it out while the graph completes it is the ordinary case, not a race worth
ignoring -- and a read-then-write lets both pass their guard against the same
observed state and lets the second overwrite the first. The `WHERE status = :seen`
clause makes the loser's `rowcount` zero, and a zero is reported as a conflict
rather than swallowed.

Why the returned type is not the ORM row
----------------------------------------
Every method returns a frozen `InvestigationRecord` or `StepRecord`. Handing back
a `InvestigationRow` would hand back an object whose session has closed, so the
first attribute the caller touched that happened not to be loaded would raise
`DetachedInstanceError` somewhere far from here -- and would also let a caller
mutate the row and believe it had saved something. The records also normalize
timestamps to aware UTC, which SQLite otherwise returns naive (see `_as_utc`).

Layer note: `services/` (L2). Takes its session factory as a constructor argument
and constructs none, so the unit suite runs the real SQL against in-memory SQLite
with nothing else running.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.config import get_settings
from backend.core.exceptions import ConflictError, NotFoundError, ValidationError
from backend.core.logging import get_logger
from models.base import utcnow
from models.enums import AgentName, InvestigationStatus
from models.orm.investigation import InvestigationRow, InvestigationStepRow, StepStatus
from models.orm.mixins import DEFAULT_TENANT

__all__ = [
    "ALLOWED_STEP_TRANSITIONS",
    "ALLOWED_TRANSITIONS",
    "MAX_QUERY_LENGTH",
    "InvestigationRecord",
    "InvestigationService",
    "StepRecord",
    "TERMINAL_STEP_STATUSES",
]

logger = get_logger(__name__)


MAX_QUERY_LENGTH: Final[int] = 2000
"""Longest accepted question, from `docs/api-reference.md` §4.1 ("1-2000 chars").

Enforced here as well as at the schema boundary because the column is `Text` and
will accept anything: a runaway caller pasting a document into `query` would be
stored verbatim, echoed into every prompt built from it, and billed for.
"""


ALLOWED_TRANSITIONS: Final[Mapping[InvestigationStatus, frozenset[InvestigationStatus]]] = {
    InvestigationStatus.QUEUED: frozenset(
        {
            InvestigationStatus.PLANNING,
            InvestigationStatus.RUNNING,
            InvestigationStatus.CANCELLED,
            InvestigationStatus.FAILED,
        }
    ),
    # Planning cannot finish the investigation. A plan is not an answer, and a
    # `planning -> completed` edge is how an orchestrator bug ships a report
    # whose evidence section is empty because no agent ever ran.
    InvestigationStatus.PLANNING: frozenset(
        {
            InvestigationStatus.RUNNING,
            InvestigationStatus.CANCELLED,
            InvestigationStatus.FAILED,
        }
    ),
    InvestigationStatus.RUNNING: frozenset(
        {
            InvestigationStatus.REFLECTING,
            InvestigationStatus.COMPLETED,
            InvestigationStatus.COMPLETED_WITH_FINDINGS,
            InvestigationStatus.CANCELLED,
            InvestigationStatus.FAILED,
        }
    ),
    # `reflecting -> running` is the Critic loop (`docs/agent-system.md` §13):
    # the Critic asks for more evidence and the graph goes round again. It is the
    # only backward edge in the machine and the reason the table is explicit
    # rather than a linear rank comparison.
    InvestigationStatus.REFLECTING: frozenset(
        {
            InvestigationStatus.RUNNING,
            InvestigationStatus.COMPLETED,
            InvestigationStatus.COMPLETED_WITH_FINDINGS,
            InvestigationStatus.CANCELLED,
            InvestigationStatus.FAILED,
        }
    ),
    InvestigationStatus.COMPLETED: frozenset(),
    InvestigationStatus.COMPLETED_WITH_FINDINGS: frozenset(),
    InvestigationStatus.FAILED: frozenset(),
    InvestigationStatus.CANCELLED: frozenset(),
    # `UNKNOWN` is what `TolerantStrEnum` yields for a status written by a newer
    # deployment than this one (`models/base.py`). Refusing to move it is the
    # only safe answer: the real state might have been terminal, and a rolling
    # deploy that let an old pod "resume" a state it cannot name would resurrect
    # finished investigations. An operator repairs it in SQL, deliberately.
    InvestigationStatus.UNKNOWN: frozenset(),
}
"""Which states may follow which. Terminal states map to the empty set.

Read with `ALLOWED_TRANSITIONS[current]`, never with `.get(...)` and a permissive
default -- a status missing from this table is a bug, and defaulting it to
"anything goes" would hide the bug behind a corrupted row.
"""


ALLOWED_STEP_TRANSITIONS: Final[Mapping[StepStatus, frozenset[StepStatus]]] = {
    StepStatus.PENDING: frozenset(
        {StepStatus.RUNNING, StepStatus.SKIPPED, StepStatus.CANCELLED, StepStatus.FAILED}
    ),
    StepStatus.RUNNING: frozenset(
        {StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.CANCELLED}
    ),
    StepStatus.COMPLETED: frozenset(),
    StepStatus.FAILED: frozenset(),
    StepStatus.SKIPPED: frozenset(),
    StepStatus.CANCELLED: frozenset(),
    StepStatus.UNKNOWN: frozenset(),
}

TERMINAL_STEP_STATUSES: Final[frozenset[StepStatus]] = frozenset(
    status for status, successors in ALLOWED_STEP_TRANSITIONS.items() if not successors
)
"""Derived rather than restated, so the two cannot disagree.

`StepStatus` has no `is_terminal` of its own -- it lives in `models/orm/` next to
the table rather than in the shared vocabulary -- so the successor table is the
single definition and this reads it.
"""


# --------------------------------------------------------------------------- #
# Detached views
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class InvestigationRecord:
    """One investigation, detached from the session that read it."""

    id: str
    tenant_id: str
    query: str
    status: InvestigationStatus
    created_by: str | None
    plan: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None
    step_count: int
    token_input: int
    token_output: int
    cost_usd: float
    report_id: str | None

    @property
    def is_terminal(self) -> bool:
        """Whether the run is over. Delegates to the enum rather than restating it."""
        return self.status.is_terminal

    @property
    def total_tokens(self) -> int:
        return self.token_input + self.token_output

    @classmethod
    def from_row(cls, row: InvestigationRow) -> InvestigationRecord:
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            query=row.query,
            status=row.status,
            created_by=row.created_by,
            plan=row.plan,
            created_at=_as_utc(row.created_at),
            updated_at=_as_utc(row.updated_at),
            started_at=_as_utc_or_none(row.started_at),
            completed_at=_as_utc_or_none(row.completed_at),
            error=row.error,
            step_count=row.step_count,
            token_input=row.token_input,
            token_output=row.token_output,
            cost_usd=float(row.cost_usd),
            report_id=row.report_id,
        )


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One agent execution inside an investigation, detached from its session."""

    id: str
    investigation_id: str
    sequence: int
    agent: AgentName
    status: StepStatus
    input: dict[str, Any] | None
    output: dict[str, Any] | None
    started_at: datetime | None
    duration_ms: int | None
    token_input: int
    token_output: int
    error: str | None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STEP_STATUSES

    @classmethod
    def from_row(cls, row: InvestigationStepRow) -> StepRecord:
        return cls(
            id=row.id,
            investigation_id=row.investigation_id,
            sequence=row.sequence,
            agent=row.agent,
            status=row.status,
            input=row.input,
            output=row.output,
            started_at=_as_utc_or_none(row.started_at),
            duration_ms=row.duration_ms,
            token_input=row.token_input,
            token_output=row.token_output,
            error=row.error,
        )


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


class InvestigationService:
    """Persistence and lifecycle for investigations and their steps.

    Holds a session factory rather than a session: each method below is one short
    transaction, and a service holding an open session would pin a pooled
    connection for as long as anything referenced the service.

    Stateless per call, so one instance is shared by the API process and the
    orchestration worker, both of which drive it concurrently. That sharing is
    exactly why every mutation is a compare-and-swap.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        tenant_id: str = DEFAULT_TENANT,
    ) -> None:
        self._session_factory = session_factory
        self._tenant_id = tenant_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    # -- creation ---------------------------------------------------------- #

    async def create(
        self,
        query: str,
        *,
        created_by: str | None = None,
        investigation_id: str | None = None,
    ) -> InvestigationRecord:
        """Enqueue a new investigation in `QUEUED`.

        `plan` is deliberately not a parameter. The column is nullable precisely
        so that "not planned yet" stays distinguishable from "planned to do
        nothing" (`models/orm/investigation.py`), and accepting a plan at creation
        would let a caller skip the Planner while the status still said `queued`.
        `record_plan` writes it when the Planner has actually run.

        `report_id` is likewise not set here. `docs/api-reference.md` §4.1 hands
        the client a report id eagerly, but the column is a foreign key to
        `reports.id`; writing an id for a row that does not exist yet would fail
        on PostgreSQL and succeed on SQLite, which is the worst possible pair of
        outcomes. `services/report_service.py` creates the `DRAFT` row and links
        it back in one transaction.
        """
        cleaned = query.strip()
        if not cleaned:
            raise ValidationError(
                "an investigation needs a question; `query` was empty or whitespace"
            )
        if len(cleaned) > MAX_QUERY_LENGTH:
            raise ValidationError(
                f"query is {len(cleaned)} characters, above the maximum of "
                f"{MAX_QUERY_LENGTH} (docs/api-reference.md §4.1)",
                details={"length": len(cleaned), "maximum": MAX_QUERY_LENGTH},
            )

        row = InvestigationRow(
            id=investigation_id or _new_id(),
            tenant_id=self._tenant_id,
            query=cleaned,
            created_by=created_by,
            status=InvestigationStatus.QUEUED,
            step_count=0,
            token_input=0,
            token_output=0,
            cost_usd=0.0,
        )
        async with self._session_factory() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return InvestigationRecord.from_row(row)

    # -- reads ------------------------------------------------------------- #

    async def get(self, investigation_id: str) -> InvestigationRecord | None:
        """Fetch one investigation, or `None` if this tenant has no such row.

        `None` rather than a raise: the API turns it into a `404`, the reaper
        treats it as "already deleted, nothing to do", and only the caller knows
        which. `require` is the raising variant for callers that have already
        decided.
        """
        async with self._session_factory() as session:
            row = await self._load(session, investigation_id)
            return None if row is None else InvestigationRecord.from_row(row)

    async def require(self, investigation_id: str) -> InvestigationRecord:
        """`get`, but a missing row is a `NotFoundError`."""
        record = await self.get(investigation_id)
        if record is None:
            raise NotFoundError.for_resource("investigation", investigation_id)
        return record

    async def list_investigations(
        self,
        *,
        statuses: Sequence[InvestigationStatus] | None = None,
        created_by: str | None = None,
        limit: int = 50,
    ) -> list[InvestigationRecord]:
        """Newest first, optionally filtered by state.

        Ordered by `(created_at DESC, id DESC)` rather than by `created_at` alone.
        `created_at` is a server clock reading and thousands of rows can share a
        second, so a non-total order lets tied rows swap places between two
        identical queries -- the same defect keyset pagination exists to avoid in
        `services/signal_service.py`.
        """
        if limit < 1:
            raise ValidationError("limit must be at least 1")

        statement = (
            select(InvestigationRow)
            .where(InvestigationRow.tenant_id == self._tenant_id)
            .order_by(InvestigationRow.created_at.desc(), InvestigationRow.id.desc())
            .limit(limit)
        )
        if statuses:
            statement = statement.where(InvestigationRow.status.in_(tuple(statuses)))
        if created_by is not None:
            statement = statement.where(InvestigationRow.created_by == created_by)

        async with self._session_factory() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [InvestigationRecord.from_row(row) for row in rows]

    async def find_timed_out(
        self, *, now: datetime | None = None, timeout: timedelta | None = None
    ) -> list[InvestigationRecord]:
        """Non-terminal runs whose `started_at` is older than the timeout.

        The reaper's query, and the reason `ix_investigations_status_started`
        exists. `started_at IS NOT NULL` is part of the predicate rather than an
        oversight: a `queued` investigation has not started, so it cannot have
        overrun, and sweeping it up would time out work that is merely waiting for
        a free worker.

        The timeout defaults to `INVESTIGATION_TIMEOUT_SECONDS`; it is a parameter
        so that a caller sweeping a backlog can widen it without an env var.
        """
        settings = get_settings()
        window = timeout or timedelta(seconds=settings.agents.timeout_seconds)
        cutoff = (now or utcnow()) - window
        live = tuple(status for status in InvestigationStatus if not status.is_terminal)

        statement = (
            select(InvestigationRow)
            .where(
                InvestigationRow.tenant_id == self._tenant_id,
                InvestigationRow.status.in_(live),
                InvestigationRow.started_at.is_not(None),
                InvestigationRow.started_at < cutoff,
            )
            .order_by(InvestigationRow.started_at)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [InvestigationRecord.from_row(row) for row in rows]

    # -- lifecycle --------------------------------------------------------- #

    async def transition(
        self,
        investigation_id: str,
        target: InvestigationStatus,
        *,
        error: str | None = None,
        expected: InvestigationStatus | None = None,
    ) -> InvestigationRecord:
        """Move an investigation to `target`, or refuse and say why.

        The general form; `start`, `resume`, `cancel`, `complete` and `fail` are
        named wrappers over it, because a caller writing
        `transition(id, CANCELLED)` by hand has to remember to clear `error` and
        stamp `completed_at`, and the wrappers are where that is remembered once.

        `expected` makes the compare-and-swap the caller's to control: pass it
        when you read a state, decided something on the strength of it, and need
        the write to fail if it moved underneath you. Left `None`, the swap is
        still performed against whatever this call read, which closes the window
        between this method's own read and its write but not the caller's.
        """
        async with self._session_factory() as session:
            row = await self._load(session, investigation_id)
            if row is None:
                raise NotFoundError.for_resource("investigation", investigation_id)

            current = row.status
            if expected is not None and current is not expected:
                raise ConflictError(
                    f"investigation {investigation_id!r} is {current.value!r}, not the "
                    f"expected {expected.value!r}; it changed under the caller.",
                    details={
                        "investigation_id": investigation_id,
                        "status": current.value,
                        "expected": expected.value,
                    },
                )

            self._require_transition(investigation_id, current, target)
            values = self._transition_values(row, target, error=error)

            # Compare-and-swap. The `status == current` clause is what makes two
            # concurrent writers safe: the second one matches zero rows because
            # the first already moved the status, and it is told so rather than
            # overwriting a decision it never saw.
            result = await session.execute(
                update(InvestigationRow)
                .where(
                    InvestigationRow.id == investigation_id,
                    InvestigationRow.tenant_id == self._tenant_id,
                    InvestigationRow.status == current,
                )
                .values(**values)
            )
            if result.rowcount == 0:
                await session.rollback()
                raise ConflictError(
                    f"investigation {investigation_id!r} moved out of "
                    f"{current.value!r} while this transition was being applied; "
                    "another worker won. Re-read the state and decide again.",
                    details={"investigation_id": investigation_id, "from": current.value},
                )
            await session.commit()

            refreshed = await self._load(session, investigation_id)
            if refreshed is None:  # pragma: no cover -- deleted mid-transaction
                raise NotFoundError.for_resource("investigation", investigation_id)
            logger.info(
                "investigation.transition",
                investigation_id=investigation_id,
                from_status=current.value,
                to_status=target.value,
            )
            return InvestigationRecord.from_row(refreshed)

    async def start(self, investigation_id: str) -> InvestigationRecord:
        """`queued -> planning`. Stamps `started_at` on the first entry only."""
        return await self.transition(investigation_id, InvestigationStatus.PLANNING)

    async def resume(self, investigation_id: str) -> InvestigationRecord:
        """Put a live investigation back into `RUNNING` after an interruption.

        Resuming is what happens when a worker died mid-run: LangGraph replays
        from its checkpoint and this row has to agree. That is why
        `running -> running` is permitted (see `_require_transition`) -- the
        crashed run left the status at `running`, and refusing the very case
        resume exists for would make the method useless.

        A terminal investigation is refused. There is no checkpoint to replay
        from after a run finished, and "resuming" one would move a completed
        investigation back to `running` and re-open a report someone has read.
        """
        return await self.transition(investigation_id, InvestigationStatus.RUNNING)

    async def reflect(self, investigation_id: str) -> InvestigationRecord:
        """`running -> reflecting`: hand the run to the Critic."""
        return await self.transition(investigation_id, InvestigationStatus.REFLECTING)

    async def cancel(self, investigation_id: str, *, reason: str | None = None) -> InvestigationRecord:
        """Stop a live investigation. A terminal one is a `409`, not a no-op.

        The reason is stored in `error` because that column is the row's single
        "why did this end" slot, and a cancelled run with an empty explanation is
        indistinguishable from one cancelled by a stray click.
        """
        return await self.transition(
            investigation_id,
            InvestigationStatus.CANCELLED,
            error=reason or "cancelled by request",
        )

    async def complete(
        self, investigation_id: str, *, with_findings: bool = False
    ) -> InvestigationRecord:
        """Finish successfully.

        `with_findings=True` selects `COMPLETED_WITH_FINDINGS`, which is what the
        Critic loop hitting `MAX_CRITIC_REVISIONS` produces: the report ships with
        its unresolved findings surfaced rather than being withheld or silently
        presented as clean (`docs/agent-system.md` §13). Two states rather than a
        boolean column because every consumer -- the list filter, the SSE terminal
        event, the badge in the UI -- switches on the status and would otherwise
        have to know to read a second field.
        """
        target = (
            InvestigationStatus.COMPLETED_WITH_FINDINGS
            if with_findings
            else InvestigationStatus.COMPLETED
        )
        return await self.transition(investigation_id, target)

    async def fail(self, investigation_id: str, *, error: str) -> InvestigationRecord:
        """Finish unsuccessfully, recording why.

        `error` is required rather than optional: a failed investigation with no
        explanation is unactionable, and the column is documented to carry the
        `code` of the raised `OmniSenseError` rather than a stack trace so that
        handlers never pattern-match on message text
        (`docs/coding-standards.md` §2.7).
        """
        if not error.strip():
            raise ValidationError("a failed investigation must record why it failed")
        return await self.transition(investigation_id, InvestigationStatus.FAILED, error=error)

    async def record_plan(
        self, investigation_id: str, plan: Mapping[str, Any]
    ) -> InvestigationRecord:
        """Attach the Planner's decomposition.

        Refused once the investigation is terminal. The plan is what the run is
        defended against afterwards -- "why did it retrieve that" is answered from
        it -- so rewriting it after the fact would make the record describe a run
        that did not happen.
        """
        async with self._session_factory() as session:
            row = await self._load(session, investigation_id)
            if row is None:
                raise NotFoundError.for_resource("investigation", investigation_id)
            if row.status.is_terminal:
                raise ConflictError(
                    f"investigation {investigation_id!r} is {row.status.value!r} and "
                    "terminal; its plan is part of the record of what ran and cannot "
                    "be rewritten afterwards.",
                    details={"investigation_id": investigation_id, "status": row.status.value},
                )
            row.plan = dict(plan)
            await session.commit()
            await session.refresh(row)
            return InvestigationRecord.from_row(row)

    async def attach_report(self, investigation_id: str, report_id: str) -> InvestigationRecord:
        """Point the investigation at the report it produced.

        Overwriting an existing pointer is allowed and is the *revision* path:
        `services/report_service.py` supersedes version *n* with version *n+1* and
        moves this pointer to the new row, so that `GET /investigations/{id}`
        resolves to the current version while the old one stays fetchable by
        `?version=` (`docs/api-reference.md` §4.4).
        """
        async with self._session_factory() as session:
            row = await self._load(session, investigation_id)
            if row is None:
                raise NotFoundError.for_resource("investigation", investigation_id)
            row.report_id = report_id
            await session.commit()
            await session.refresh(row)
            return InvestigationRecord.from_row(row)

    # -- steps ------------------------------------------------------------- #

    async def append_step(
        self,
        investigation_id: str,
        agent: AgentName,
        *,
        step_input: Mapping[str, Any] | None = None,
        step_id: str | None = None,
        max_steps: int | None = None,
    ) -> StepRecord:
        """Append the next step of the timeline, enforcing the step budget.

        The sequence is `step_count`, and `step_count` is incremented in the same
        transaction with `step_count = step_count + 1` computed **in SQL**. Reading
        the value into Python and writing back `value + 1` would let two workers
        appending concurrently both read *n* and both write *n+1*, which the
        `UNIQUE (investigation_id, sequence)` constraint would then reject -- so
        the defect would surface as a spurious integrity error rather than as the
        lost step it actually is.

        A terminal investigation cannot grow new steps. Its step history is the
        record of what ran, and appending to it afterwards would make the timeline
        disagree with the outcome.
        """
        settings = get_settings()
        budget = settings.agents.max_steps if max_steps is None else max_steps
        if budget < 1:
            raise ValidationError("max_steps must be at least 1")

        async with self._session_factory() as session:
            row = await self._load(session, investigation_id)
            if row is None:
                raise NotFoundError.for_resource("investigation", investigation_id)
            if row.status.is_terminal:
                raise ConflictError(
                    f"investigation {investigation_id!r} is {row.status.value!r} and "
                    "terminal; no further steps can be appended to a finished run.",
                    details={"investigation_id": investigation_id, "status": row.status.value},
                )
            if row.step_count >= budget:
                raise ConflictError(
                    f"investigation {investigation_id!r} has reached its step budget "
                    f"of {budget} (INVESTIGATION_MAX_STEPS). The orchestrator must "
                    "stop rather than loop; an unbounded Critic cycle is what this "
                    "budget exists to break.",
                    details={
                        "investigation_id": investigation_id,
                        "step_count": row.step_count,
                        "max_steps": budget,
                    },
                )

            sequence = row.step_count
            step = InvestigationStepRow(
                id=step_id or _new_id(),
                tenant_id=self._tenant_id,
                investigation_id=investigation_id,
                sequence=sequence,
                agent=agent,
                status=StepStatus.PENDING,
                input=dict(step_input) if step_input is not None else None,
                token_input=0,
                token_output=0,
            )
            session.add(step)
            await session.execute(
                update(InvestigationRow)
                .where(
                    InvestigationRow.id == investigation_id,
                    InvestigationRow.tenant_id == self._tenant_id,
                )
                .values(step_count=InvestigationRow.step_count + 1)
            )
            await session.commit()
            await session.refresh(step)
            return StepRecord.from_row(step)

    async def start_step(self, step_id: str) -> StepRecord:
        """`pending -> running`, stamping `started_at`."""
        return await self._transition_step(
            step_id, StepStatus.RUNNING, values={"started_at": utcnow()}
        )

    async def complete_step(
        self,
        step_id: str,
        *,
        output: Mapping[str, Any] | None = None,
        duration_ms: int | None = None,
        token_input: int = 0,
        token_output: int = 0,
        cost_usd: float = 0.0,
    ) -> StepRecord:
        """`running -> completed`, rolling usage up to the investigation.

        The rollup happens here rather than in a periodic job because the parent's
        totals are read by the budget check on the *next* step, and a total that
        lags by a sweep interval is a budget that is not enforced.

        `output` is written only on success. A failed step keeps `NULL`, which the
        column documents as deliberate: an unvalidated partial output is worse
        than none, because the Critic would then be judging something the schema
        rejected.
        """
        if token_input < 0 or token_output < 0 or cost_usd < 0:
            raise ValidationError("usage counters cannot be negative")
        if duration_ms is not None and duration_ms < 0:
            raise ValidationError("duration_ms cannot be negative")

        return await self._transition_step(
            step_id,
            StepStatus.COMPLETED,
            values={
                "output": dict(output) if output is not None else None,
                "duration_ms": duration_ms,
                "token_input": token_input,
                "token_output": token_output,
            },
            usage=(token_input, token_output, cost_usd),
        )

    async def fail_step(
        self,
        step_id: str,
        *,
        error: str,
        duration_ms: int | None = None,
        token_input: int = 0,
        token_output: int = 0,
        cost_usd: float = 0.0,
    ) -> StepRecord:
        """`pending|running -> failed`.

        Usage is still rolled up. Tokens spent on a step that failed were spent,
        and a budget that only counted successes would be trivially exceeded by a
        loop that fails.
        """
        if not error.strip():
            raise ValidationError("a failed step must record why it failed")
        return await self._transition_step(
            step_id,
            StepStatus.FAILED,
            values={"error": error, "duration_ms": duration_ms,
                    "token_input": token_input, "token_output": token_output},
            usage=(token_input, token_output, cost_usd),
        )

    async def skip_step(self, step_id: str, *, reason: str | None = None) -> StepRecord:
        """`pending -> skipped`: the plan named this agent, the graph did not run it."""
        return await self._transition_step(
            step_id, StepStatus.SKIPPED, values={"error": reason}
        )

    async def steps(
        self,
        investigation_id: str,
        *,
        after_sequence: int | None = None,
        limit: int = 50,
    ) -> list[StepRecord]:
        """The timeline in graph order, paginated by sequence.

        Keyset on `sequence` rather than `LIMIT/OFFSET`, and for once the reason is
        not concurrent deletion: steps are appended while the client is paging, so
        an offset walk over a growing collection is stable at the head and skips at
        the tail. `sequence` is unique per investigation and monotonic, so it is
        already the cursor -- no encoding required.
        """
        if limit < 1:
            raise ValidationError("limit must be at least 1")

        statement = (
            select(InvestigationStepRow)
            .where(
                InvestigationStepRow.tenant_id == self._tenant_id,
                InvestigationStepRow.investigation_id == investigation_id,
            )
            .order_by(InvestigationStepRow.sequence)
            .limit(limit)
        )
        if after_sequence is not None:
            statement = statement.where(InvestigationStepRow.sequence > after_sequence)

        async with self._session_factory() as session:
            rows = (await session.execute(statement)).scalars().all()
        return [StepRecord.from_row(row) for row in rows]

    async def usage(self, investigation_id: str) -> tuple[int, int]:
        """`(input, output)` tokens summed from the steps.

        The authoritative reconciliation for the denormalized counters on the
        parent: if this disagrees with `InvestigationRecord.token_input`, a rollup
        was lost, and it is better to be able to prove that than to trust a
        counter nothing checks.
        """
        statement = select(
            func.coalesce(func.sum(InvestigationStepRow.token_input), 0),
            func.coalesce(func.sum(InvestigationStepRow.token_output), 0),
        ).where(
            InvestigationStepRow.tenant_id == self._tenant_id,
            InvestigationStepRow.investigation_id == investigation_id,
        )
        async with self._session_factory() as session:
            row = (await session.execute(statement)).one()
        return int(row[0]), int(row[1])

    # -- internals --------------------------------------------------------- #

    async def _load(self, session: AsyncSession, investigation_id: str) -> InvestigationRow | None:
        """Tenant-scoped fetch. The tenant predicate is never optional.

        A lookup by primary key alone would resolve another tenant's row, and the
        method above it would then happily transition it. `DEFAULT_TENANT` makes
        that harmless today and will not once Phase 7 lands
        (`models/orm/mixins.py`), which is exactly why the filter is written now.
        """
        statement = select(InvestigationRow).where(
            InvestigationRow.id == investigation_id,
            InvestigationRow.tenant_id == self._tenant_id,
        )
        return (await session.execute(statement)).scalars().one_or_none()

    @staticmethod
    def _require_transition(
        investigation_id: str,
        current: InvestigationStatus,
        target: InvestigationStatus,
    ) -> None:
        """The state machine, in one place.

        Terminality is checked first and separately so the error says *why*. "A
        completed investigation cannot be cancelled" is actionable; "cancelled is
        not a successor of completed" reads like a table lookup failed.
        """
        if current.is_terminal:
            raise ConflictError(
                f"investigation {investigation_id!r} is already {current.value!r}, "
                f"which is terminal; it cannot move to {target.value!r}. A terminal "
                "state is the record of how the run ended, and rewriting it would "
                "make every downstream reading of this row -- billing, the report's "
                "provenance, the stream's terminal event -- describe a run that did "
                "not happen that way.",
                details={
                    "investigation_id": investigation_id,
                    "status": current.value,
                    "attempted": target.value,
                    "terminal": True,
                },
            )
        # A non-terminal state may be re-entered. That is not laxity: `resume`
        # exists for a worker that died mid-run, which left the status at
        # `running`, so refusing `running -> running` would refuse the only case
        # the method is for. Re-entering a *terminal* state is still refused
        # above, which is where the property that matters lives.
        if target is current:
            return
        allowed = ALLOWED_TRANSITIONS[current]
        if target not in allowed:
            raise ConflictError(
                f"investigation {investigation_id!r} cannot move from "
                f"{current.value!r} to {target.value!r}; the permitted successors "
                f"are {sorted(status.value for status in allowed)}.",
                details={
                    "investigation_id": investigation_id,
                    "status": current.value,
                    "attempted": target.value,
                    "allowed": sorted(status.value for status in allowed),
                },
            )

    @staticmethod
    def _transition_values(
        row: InvestigationRow,
        target: InvestigationStatus,
        *,
        error: str | None,
    ) -> dict[str, Any]:
        """The columns a transition touches beyond `status`.

        Two rules, both forced by the table rather than chosen:

        `started_at` is stamped once. Re-stamping it on a resume would make the
        run's wall-clock duration reset every time a worker restarted, and that
        duration is the only evidence a timeout sweep has.

        `completed_at` is stamped **only if the run actually started**. The
        `completed_after_started` CHECK constraint forbids "finished but never
        started", and a `queued -> cancelled` transition is exactly that: nothing
        ever ran. Writing `started_at = now` alongside it to satisfy the
        constraint would be this service inventing a start that did not occur, so
        the honest encoding leaves `completed_at` NULL and lets the status change
        plus `updated_at` carry the moment.
        """
        values: dict[str, Any] = {"status": target}
        now = utcnow()

        if target in (InvestigationStatus.PLANNING, InvestigationStatus.RUNNING):
            if row.started_at is None:
                values["started_at"] = now

        if target.is_terminal:
            if row.started_at is not None:
                values["completed_at"] = now
            values["error"] = error
        elif error is not None:
            values["error"] = error

        return values

    async def _transition_step(
        self,
        step_id: str,
        target: StepStatus,
        *,
        values: Mapping[str, Any] | None = None,
        usage: tuple[int, int, float] | None = None,
    ) -> StepRecord:
        """Move one step, and roll its usage into the parent in the same transaction.

        Same transaction is the point. Two statements in two transactions can
        leave a step marked `completed` whose tokens were never counted, and the
        step budget is then enforced against a total that is quietly low.
        """
        async with self._session_factory() as session:
            statement = select(InvestigationStepRow).where(
                InvestigationStepRow.id == step_id,
                InvestigationStepRow.tenant_id == self._tenant_id,
            )
            step = (await session.execute(statement)).scalars().one_or_none()
            if step is None:
                raise NotFoundError.for_resource("investigation step", step_id)

            current = step.status
            if current in TERMINAL_STEP_STATUSES:
                raise ConflictError(
                    f"step {step_id!r} is already {current.value!r}, which is "
                    f"terminal; it cannot move to {target.value!r}.",
                    details={"step_id": step_id, "status": current.value,
                             "attempted": target.value},
                )
            if target not in ALLOWED_STEP_TRANSITIONS[current] and target is not current:
                raise ConflictError(
                    f"step {step_id!r} cannot move from {current.value!r} to "
                    f"{target.value!r}.",
                    details={"step_id": step_id, "status": current.value,
                             "attempted": target.value},
                )

            step.status = target
            for column, value in (values or {}).items():
                if value is not None:
                    setattr(step, column, value)

            if usage is not None:
                token_input, token_output, cost_usd = usage
                await session.execute(
                    update(InvestigationRow)
                    .where(
                        InvestigationRow.id == step.investigation_id,
                        InvestigationRow.tenant_id == self._tenant_id,
                    )
                    .values(
                        token_input=InvestigationRow.token_input + token_input,
                        token_output=InvestigationRow.token_output + token_output,
                        cost_usd=InvestigationRow.cost_usd + cost_usd,
                    )
                )

            await session.commit()
            await session.refresh(step)
            return StepRecord.from_row(step)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _new_id() -> str:
    """A UUID string, per `docs/api-reference.md` §3 ("investigation, report, step
    ... ids are UUID strings"). Dashed rather than hex so a client that does
    validate one against a UUID parser -- against the advice to treat ids as
    opaque -- still gets an answer it can parse."""
    return str(uuid.uuid4())


def _as_utc(value: datetime) -> datetime:
    """Normalize a database timestamp to aware UTC.

    PostgreSQL returns `timestamptz` as an aware datetime; SQLite has no such
    type and hands back a naive one even though the column is declared
    `DateTime(timezone=True)`. The value written was UTC either way, so attaching
    UTC restores rather than guesses -- and without it every comparison against
    `utcnow()` raises `TypeError` on the unit suite's database and on no other,
    which is the worst possible place to discover it.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _as_utc_or_none(value: datetime | None) -> datetime | None:
    """`_as_utc`, passing `None` through.

    Separate from `_as_utc` rather than folded into it so the nullable columns
    (`started_at`, `completed_at`) and the non-nullable ones (`created_at`) keep
    different static types -- a single permissive helper would make every
    timestamp on `InvestigationRecord` optional and push a `None` check onto every
    caller of a field that can never be `None`.
    """
    return None if value is None else _as_utc(value)
