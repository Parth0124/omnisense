"""ORM tables for agent runs, traces and token accounting.

`agent_runs` is one row per **LLM call**, not per agent and not per investigation.
That granularity is the point: a single Retriever step can issue a dozen calls at
three different model tiers, and "what did this investigation cost" is only
answerable if each one is recorded with the model it used and the tokens it
burned. It is also the reproducibility record -- `prompt_id`, `prompt_version` and
`prompt_hash` together say exactly which text produced the output, which
`prompts/README.md` rule 2 requires of every run and which
`docs/agent-system.md` §12 extends to the shared fragments, since the hash covers
those too.

`traces` is the OpenTelemetry correlation table. Spans are exported to a real
tracing backend at runtime; what is kept here is the subset needed to answer
"which trace was this investigation" long after the backend's retention window
has closed, so that a report generated months ago can still be tied to the calls
that produced it.

Both tables are append-only in practice. Neither is soft-deletable: there is
nothing to tombstone, because nothing references them. Cost history instead
survives investigation deletion by way of a nullable foreign key -- see
`AgentRunRow.investigation_id`.
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
from models.enums import AgentName
from models.orm.base import Base, JSONVariant, TolerantEnumType
from models.orm.mixins import TenantMixin, TimestampMixin

__all__ = [
    "AgentRunRow",
    "RunStatus",
    "TraceRow",
]


class RunStatus(TolerantStrEnum):
    """Outcome of a single model call.

    A finer vocabulary than `StepStatus` because the distinctions are the ones
    accounting cares about: a `TIMEOUT` has already been billed for its input
    tokens, a `RATE_LIMITED` call cost nothing at all, and an
    `OUTPUT_SCHEMA_ERROR` cost full price for output the agent had to discard.
    Collapsing those into "failed" makes cost-per-useful-token unrecoverable.

    Defined here rather than in `models/enums.py` for the same reason as the other
    row-status enums: it currently has exactly one consumer.
    """

    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    OUTPUT_SCHEMA_ERROR = "output_schema_error"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class AgentRunRow(Base, TimestampMixin, TenantMixin):
    """One LLM call: what was asked, of which model, at what cost."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    investigation_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("investigations.id", ondelete="SET NULL"),
        nullable=True,
    )
    """`SET NULL`, not `CASCADE`, and nullable for two independent reasons.

    Nullable: not every call belongs to an investigation. The eval harness
    (`agents/evaluation/harness.py`) and one-off scripts spend real money and must
    appear in the same ledger.

    `SET NULL`: money that was spent stays spent. Cascading would let deleting an
    investigation quietly rewrite last month's cost total, and this row holds no
    user content -- only identifiers, counters and a model name -- so keeping it
    after an erasure request is not a privacy problem.
    """

    agent: Mapped[AgentName] = mapped_column(TolerantEnumType(AgentName, 32), nullable=False)

    # -- what was called ---------------------------------------------------
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    """The concrete model id actually used, e.g. `claude-sonnet-5`.

    Recorded, never assumed. Code selects a *tier* through
    `services/llm/router.py` and the tier-to-model mapping is configuration
    (`docs/coding-standards.md` §2.9), so the id in force on the day of the run is
    the only thing that makes the run reproducible after that config changes.
    """

    prompt_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(16), nullable=False)
    """Version directory the template came from, e.g. `v1`. Prompts are immutable
    once used -- a change means a new version (`prompts/README.md` rule 1) -- so
    this pair is a stable coordinate rather than a moving target."""

    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    """SHA-256 of the *composed* prompt: the agent template plus every shared
    fragment rendered into it.

    Kept alongside `prompt_version` rather than instead of it because the two
    catch different mistakes. The version says which file was intended; the hash
    proves what was actually sent, so editing `prompts/shared/citation_rules.md`
    changes the hash of every agent that includes it even though no version moved
    (`docs/agent-system.md` §12). A mismatch between them is the signal that rule
    1 was broken.
    """

    # -- accounting --------------------------------------------------------
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    cached_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    """Prompt-cache hits, counted separately because they are billed at a
    different rate. Folding them into `input_tokens` would make the ledger
    disagree with the invoice and would hide whether caching is working at all --
    which is the whole reason the system framing is factored into shared
    fragments."""

    cost_usd: Mapped[float] = mapped_column(
        Numeric(14, 6, asdecimal=False), nullable=False, default=0.0
    )
    """Cost of this call, computed at write time from the price list in force.

    Stored rather than derived on read: prices change, and a report's cost must
    not silently move when they do. `NUMERIC` because this column is summed for
    billing -- floating-point `SUM` is order-dependent and PostgreSQL may reorder
    it under a parallel plan. `asdecimal=False` keeps the Python type `float` on
    both dialects, avoiding SQLite's lossy-decimal warning while leaving storage
    and server-side aggregation exact.
    """

    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    status: Mapped[RunStatus] = mapped_column(
        TolerantEnumType(RunStatus, 32), nullable=False, default=RunStatus.OK
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    """When the call was issued. Not the same as `created_at`, which is when the
    row was written -- the row is written after the response returns, so for a
    30-second call the two differ by the whole latency. Every cost and latency
    query buckets on this one."""

    __table_args__ = (
        CheckConstraint("input_tokens >= 0", name="input_tokens_non_negative"),
        CheckConstraint("output_tokens >= 0", name="output_tokens_non_negative"),
        CheckConstraint("cached_tokens >= 0", name="cached_tokens_non_negative"),
        CheckConstraint("cost_usd >= 0", name="cost_non_negative"),
        CheckConstraint("latency_ms >= 0", name="latency_non_negative"),
        # "What did this investigation cost, call by call" -- the `usage` block of
        # `GET /investigations/{id}` and the budget check that enforces
        # `max_tokens`.
        Index("ix_agent_runs_investigation_started", "investigation_id", "started_at"),
        # Spend per tenant per day, which is also the quota enforcement read.
        Index("ix_agent_runs_tenant_started", "tenant_id", "started_at"),
        # Cost and latency broken down by model -- the evidence behind moving a
        # step from one tier to another.
        Index("ix_agent_runs_model_started", "model", "started_at"),
        # "Every call made with exactly this prompt text", which is how a
        # regression is traced back to a prompt edit.
        Index("ix_agent_runs_prompt_hash", "prompt_hash"),
    )


class TraceRow(Base, TimestampMixin, TenantMixin):
    """One OpenTelemetry span, retained for correlation after export.

    Not a replacement for a tracing backend: there are no events, links or
    resource attributes here, and nothing queries this table for a flame graph.
    It exists so that `trace_id` -- the field every log line already carries
    (`docs/observability.md` §1) -- remains resolvable to a name and a parent once
    the backend has aged the trace out.
    """

    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    trace_id: Mapped[str] = mapped_column(String(32), nullable=False)
    """W3C trace id: 16 bytes, lowercase hex, 32 characters."""

    span_id: Mapped[str] = mapped_column(String(16), nullable=False)
    """W3C span id: 8 bytes, lowercase hex, 16 characters."""

    parent_span_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    """`NULL` for the root span of a trace."""

    name: Mapped[str] = mapped_column(String(256), nullable=False)
    """Span name, e.g. `agent.retriever.hybrid_search`. Dotted and static, for the
    same reason log event names are (`docs/coding-standards.md` §2.8): it is a
    grouping key, so interpolating an id into it destroys the grouping."""

    attributes: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    """Span attributes as exported. JSON because the key set is per-span and open;
    anything that needs to be filtered on across spans belongs in a column, not in
    here, since a JSON path lookup cannot use a btree index."""

    agent_run_id: Mapped[str | None] = mapped_column(
        String(64),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=True,
    )
    """The LLM call this span covers, when it covers one.

    Nullable because most spans are not model calls -- a retrieval fan-out or a
    tool invocation produces spans with no run behind them. `CASCADE` because a
    span describing a deleted call is unresolvable noise, unlike the cost record
    above, which is deliberately preserved.
    """

    __table_args__ = (
        # A span id is unique within its trace. Re-ingesting an exported batch
        # must be idempotent, and without this an at-least-once exporter
        # duplicates every span it retries. The leading column also serves
        # "every span of this trace", which is the only read this table has.
        UniqueConstraint("trace_id", "span_id"),
        # Fixed-width hex identifiers from the W3C trace-context spec. Worth
        # checking because a truncated or dashed id silently fails to join
        # against the ids in the log stream, and that failure looks like missing
        # data rather than malformed data.
        CheckConstraint("length(trace_id) = 32", name="trace_id_length"),
        CheckConstraint("length(span_id) = 16", name="span_id_length"),
        CheckConstraint(
            "parent_span_id IS NULL OR length(parent_span_id) = 16",
            name="parent_span_id_length",
        ),
        # A span cannot be its own parent; the tree walk would not terminate.
        CheckConstraint(
            "parent_span_id IS NULL OR parent_span_id <> span_id",
            name="parent_is_not_self",
        ),
        # Reconstructing the tree: the children of a given span.
        Index("ix_traces_parent_span", "parent_span_id"),
        # "Spans for this call", and the index PostgreSQL needs for the CASCADE
        # above to be a lookup rather than a sequential scan.
        Index("ix_traces_agent_run", "agent_run_id"),
    )
