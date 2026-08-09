"""The investigation domain model: one question, its run, its accounting.

`models/orm/investigation.py` is how a run is *stored*; `agents/state.py` is how
it is *executed*; `backend/schemas/investigation.py` is how it is *published*.
This is the shape those three agree on -- the vocabulary a service can accept and
return without importing a database row into a graph node or a wire model into a
worker.

**Why it is not just the ORM row.** A `InvestigationRow` is bound to a session:
read it, close the session, touch an unloaded attribute and you get a
`DetachedInstanceError` somewhere far from the read. Every service in this
codebase therefore already detaches into a plain object before returning. This is
that object, defined once instead of per service.

**Status transitions are validated here, not in the database.** A CHECK
constraint could enforce the enum but not the *graph* -- that `completed` may not
become `running`, that `cancelled` is terminal. Those are domain rules, and
putting them in one place means a worker resuming a run and an API cancelling one
cannot disagree about what is legal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final

from pydantic import Field, model_validator

from models.base import LenientModel, Score, StrictModel, UtcDatetime, utcnow
from models.enums import AgentName, InvestigationStatus

__all__ = [
    "TERMINAL_STATUSES",
    "VALID_TRANSITIONS",
    "Investigation",
    "InvestigationStep",
    "TokenUsage",
    "can_transition",
]

TERMINAL_STATUSES: Final[frozenset[InvestigationStatus]] = frozenset(
    {
        InvestigationStatus.COMPLETED,
        InvestigationStatus.COMPLETED_WITH_FINDINGS,
        InvestigationStatus.FAILED,
        InvestigationStatus.CANCELLED,
    }
)
"""States from which nothing may move.

`COMPLETED_WITH_FINDINGS` is terminal and *successful*: the run finished and the
report names what it could not establish. Treating it as a failure -- which a
naive `status == COMPLETED` check does -- discards every honestly-degraded run,
which are the majority of real ones.
"""

VALID_TRANSITIONS: Final[dict[InvestigationStatus, frozenset[InvestigationStatus]]] = {
    InvestigationStatus.QUEUED: frozenset(
        {InvestigationStatus.PLANNING, InvestigationStatus.CANCELLED, InvestigationStatus.FAILED}
    ),
    InvestigationStatus.PLANNING: frozenset(
        {InvestigationStatus.RUNNING, InvestigationStatus.CANCELLED, InvestigationStatus.FAILED}
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
    # Reflecting goes back to running: the Critic loop sends the graph round
    # again (`docs/agent-system.md` §13). Without this edge a revision would be
    # an illegal transition, and the guard would fire on the system's own
    # designed behaviour.
    InvestigationStatus.REFLECTING: frozenset(
        {
            InvestigationStatus.RUNNING,
            InvestigationStatus.COMPLETED,
            InvestigationStatus.COMPLETED_WITH_FINDINGS,
            InvestigationStatus.CANCELLED,
            InvestigationStatus.FAILED,
        }
    ),
}
"""The legal state graph. Terminal states are absent, which is how they are terminal."""


def can_transition(current: InvestigationStatus, target: InvestigationStatus) -> bool:
    """Whether a state change is legal.

    A no-op transition (`running -> running`) is allowed: at-least-once delivery
    means a worker will re-apply the same transition after a redelivery, and
    rejecting it would turn ordinary redelivery into an error.
    """
    if current is target:
        return True
    return target in VALID_TRANSITIONS.get(current, frozenset())


class TokenUsage(LenientModel):
    """Token and cost accounting for a run.

    `LenientModel` because it is written by every node and read by cost
    reporting: a field added mid-deploy must not break a run resuming from an
    older checkpoint.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of input tokens served from cache.

        Worth surfacing: prompt caching is the difference between a viable and an
        unviable cost model for a multi-agent run, and a rate that quietly drops
        to zero -- because a prompt stopped being byte-stable -- looks like
        nothing except a larger bill.
        """
        return self.cached_tokens / self.input_tokens if self.input_tokens else 0.0


class InvestigationStep(StrictModel):
    """One agent execution inside a run."""

    id: str
    investigation_id: str
    sequence: int = Field(ge=0)
    agent: AgentName
    started_at: UtcDatetime | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None

    @property
    def completed_at(self) -> datetime | None:
        """Derived, not stored.

        `investigation_steps` records a duration, not an end timestamp. Computing
        it here keeps the arithmetic in one place rather than in every consumer.
        """
        from datetime import timedelta

        if self.started_at is None or self.duration_ms is None:
            return None
        return self.started_at + timedelta(milliseconds=self.duration_ms)

    @property
    def succeeded(self) -> bool:
        return self.error is None


class Investigation(StrictModel):
    """One investigation, detached from any session."""

    id: str
    tenant_id: str
    query: str = Field(min_length=1)
    status: InvestigationStatus = InvestigationStatus.QUEUED
    objective: str | None = None
    created_by: str | None = None
    created_at: UtcDatetime = Field(default_factory=utcnow)
    started_at: UtcDatetime | None = None
    completed_at: UtcDatetime | None = None
    error: str | None = None
    step_count: int = Field(default=0, ge=0)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    report_id: str | None = None
    confidence: Score = 0.0

    @model_validator(mode="after")
    def _terminal_runs_have_an_end(self) -> Investigation:
        """A finished run must say when it finished.

        Without it, "how long do investigations take" is unanswerable for exactly
        the runs that ended badly -- and those are the ones worth measuring.
        """
        if self.status in TERMINAL_STATUSES and self.completed_at is None:
            raise ValueError(
                f"status is {self.status.value} but completed_at is unset; a "
                "terminal run must record when it ended"
            )
        return self

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def succeeded(self) -> bool:
        """Whether the run produced an answer.

        `COMPLETED_WITH_FINDINGS` counts. It means the report was produced and it
        names its own gaps, which is the system working as designed -- not a
        failure.
        """
        return self.status in (
            InvestigationStatus.COMPLETED,
            InvestigationStatus.COMPLETED_WITH_FINDINGS,
        )

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()
