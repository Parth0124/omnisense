"""`InvestigationState`: the shared, checkpointed state every agent reads and writes.

Two rules govern this file, and both exist because of how LangGraph actually
behaves rather than because of taste.

**Small.** The state is serialised to PostgreSQL after *every* node. Bulk payloads
-- raw passages, media, intermediate frames -- live in Qdrant, R2 or the Redis
scratchpad and appear here only as references. A checkpoint should be kilobytes.
A state carrying full passage text turns every node transition into a multi-
megabyte write, and a 30-minute investigation into a Postgres bandwidth problem.

**Reducer-annotated.** The `{Trend, Competitor, Forecast}` fan-out writes
concurrently. Any key more than one concurrent node writes carries an append
reducer (`Annotated[list[X], operator.add]`); any scalar must be written by
exactly one node. Get this wrong and the failure is a *lost write* under
concurrency -- the branch that finishes second overwrites the first, intermittently
and only under load, which is the worst class of bug this system could have.

Payload types come from `models/` so the graph state and the API schemas in
`backend/schemas/` share one definition instead of drifting into two.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any, TypedDict

from models.base import LenientModel, Score, StrictModel, UtcDatetime, utcnow
from models.enums import AgentName, InvestigationStatus
from pydantic import Field

__all__ = [
    "AgentError",
    "CollectionResult",
    "EvidenceRef",
    "GraphContext",
    "InvestigationState",
    "PlanStep",
    "PromptRef",
    "SubQuestion",
    "TokenLedger",
    "merge_ledgers",
    "new_state",
]


# --------------------------------------------------------------------------- #
# Payload types carried in the state
# --------------------------------------------------------------------------- #


class PlanStep(StrictModel):
    """One step of the Planner's decomposition.

    `requires_fresh_data` is read by the router to decide whether the run visits
    the Collector at all. It is a plan-level decision rather than a Collector-level
    one because dispatching a connector sync costs minutes and a quota; deciding
    *after* entering the node would mean the run pays that cost to discover it did
    not need it.
    """

    id: str
    description: str
    agent: AgentName
    requires_fresh_data: bool = False
    depends_on: Sequence[str] = ()
    rationale: str | None = None


class SubQuestion(StrictModel):
    """A decomposed question the evidence must answer.

    Carried separately from `plan` because the Critic checks coverage against
    these: a report that answers four of six sub-questions is incomplete in a way
    that a step list cannot express.
    """

    id: str
    question: str
    answered: bool = False
    evidence_ids: Sequence[str] = ()


class EvidenceRef(StrictModel):
    """A *reference* to retrieved evidence, never the passage text.

    This is the single most important size decision in the state. A run gathers
    hundreds of passages; inlining them would make each checkpoint megabytes and
    each resume a full re-read. The text is fetched on demand from the store via
    `services/evidence_service.py`, which also verifies the quote still matches --
    so a stale reference fails loudly instead of silently citing text that has
    since been erased.
    """

    signal_id: str
    chunk_id: str | None = None
    quote: str | None = Field(
        default=None,
        description="A short verbatim span, only when a downstream agent has "
        "already committed to citing it. Bounded deliberately -- this is the one "
        "place text is allowed into the state, and only a sentence of it.",
        max_length=500,
    )
    char_start: int | None = None
    char_end: int | None = None
    relevance: Score = 0.0
    retrieved_by: AgentName = AgentName.RETRIEVER


class CollectionResult(StrictModel):
    """What one connector sync contributed to this run."""

    connector_slug: str
    run_id: str
    requested_at: UtcDatetime = Field(default_factory=utcnow)
    emitted: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class GraphContext(StrictModel):
    """Graph neighbourhood assembled for this run.

    Entity ids and fact references only. The full subgraph stays in Neo4j.
    """

    seed_entity_ids: Sequence[str] = ()
    expanded_entity_ids: Sequence[str] = ()
    fact_count: int = 0
    community_ids: Sequence[str] = ()
    as_of: UtcDatetime | None = None
    truncated: bool = False
    """Whether the fanout cap bit. A truncated neighbourhood is a weaker basis for
    a claim, and the Critic needs to know rather than infer it."""


class AgentError(StrictModel):
    """A non-fatal failure inside one node.

    Appended rather than raised: `docs/architecture.md` §7.3 requires an
    investigation to return a smaller, honestly-labelled answer instead of
    failing, so a Forecast that could not converge records itself here and the run
    continues. The Report renders these as a "gaps" section, which is what keeps
    the degradation visible instead of silent.
    """

    agent: AgentName
    error_type: str
    message: str
    occurred_at: UtcDatetime = Field(default_factory=utcnow)
    recoverable: bool = True


class PromptRef(StrictModel):
    """Which prompt text produced a node's output.

    `sha256` rather than just a version string: prompts are files, and a file can
    be edited without its version being bumped. The hash is what makes "this
    claim was produced by exactly this prompt" checkable a year later
    (`prompts/README.md` rule 2).
    """

    agent: AgentName
    version: str
    sha256: str


class TokenLedger(LenientModel):
    """Token and cost accounting, merge-added across every node.

    `LenientModel` because it is written by every node and read by cost reporting
    -- a field added mid-deploy must not break a running investigation resuming
    from an older checkpoint.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def plus(self, other: TokenLedger) -> TokenLedger:
        return TokenLedger(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            calls=self.calls + other.calls,
            cost_usd=self.cost_usd + other.cost_usd,
        )


def merge_ledgers(left: TokenLedger, right: TokenLedger) -> TokenLedger:
    """Reducer for `tokens_spent`.

    Addition rather than last-write, because the fan-out nodes each spend tokens
    concurrently and a last-write reducer would silently discard the spend of
    every branch but one -- under-reporting cost precisely when the run is most
    expensive.
    """
    return left.plus(right)


def _merge_prompt_versions(
    left: dict[str, PromptRef], right: dict[str, PromptRef]
) -> dict[str, PromptRef]:
    """Reducer for `prompt_versions`: union, newest wins per agent."""
    return {**left, **right}


# --------------------------------------------------------------------------- #
# The state
# --------------------------------------------------------------------------- #


class InvestigationState(TypedDict, total=False):
    """The graph state. One `TypedDict`, checkpointed after every node.

    Reducers are the annotation, not a convention: LangGraph reads them off the
    type. A key written by concurrent nodes without one is a lost update, so the
    per-field comments below record *who writes* each key -- that is the check a
    reviewer needs to make when adding a node.
    """

    # -- set once at entry ---------------------------------------------------
    investigation_id: str
    tenant_id: str
    query: str
    deadline_at: datetime
    scratchpad_key: str
    trace_id: str

    # -- Planner (single writer, last-write) ---------------------------------
    objective: str
    plan: list[PlanStep]
    sub_questions: list[SubQuestion]

    # -- router --------------------------------------------------------------
    cursor: int
    status: InvestigationStatus

    # -- appended by concurrent writers: reducer REQUIRED --------------------
    collection_results: Annotated[list[CollectionResult], operator.add]
    evidence: Annotated[list[EvidenceRef], operator.add]
    insights: Annotated[list[dict[str, Any]], operator.add]
    recommendations: Annotated[list[dict[str, Any]], operator.add]
    critique_history: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[AgentError], operator.add]

    # -- single-writer scalars ------------------------------------------------
    graph_context: GraphContext
    critique: dict[str, Any] | None
    revision_count: int
    report: dict[str, Any] | None
    confidence: float

    # -- accounting, written by every node ------------------------------------
    step_count: Annotated[int, operator.add]
    tokens_spent: Annotated[TokenLedger, merge_ledgers]
    prompt_versions: Annotated[dict[str, PromptRef], _merge_prompt_versions]


def new_state(
    *,
    investigation_id: str,
    tenant_id: str,
    query: str,
    deadline_at: datetime,
    trace_id: str,
    scratchpad_key: str | None = None,
) -> InvestigationState:
    """Build the entry-point state.

    Every reducer-bearing key is initialised to an empty value here rather than
    left absent. LangGraph applies a reducer only to a key that already exists,
    so an absent `evidence` makes the first concurrent write *replace* instead of
    append -- which loses the other branch's evidence exactly once, on the first
    fan-out, and never reproduces afterwards.
    """
    return InvestigationState(
        investigation_id=investigation_id,
        tenant_id=tenant_id,
        query=query,
        deadline_at=deadline_at,
        trace_id=trace_id,
        scratchpad_key=scratchpad_key or f"os:scratch:{investigation_id}",
        objective="",
        plan=[],
        sub_questions=[],
        cursor=0,
        status=InvestigationStatus.QUEUED,
        collection_results=[],
        evidence=[],
        insights=[],
        recommendations=[],
        critique_history=[],
        errors=[],
        graph_context=GraphContext(),
        critique=None,
        revision_count=0,
        report=None,
        confidence=0.0,
        step_count=0,
        tokens_spent=TokenLedger(),
        prompt_versions={},
    )
