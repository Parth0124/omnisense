"""Every conditional edge and every termination predicate in the investigation graph.

`docs/agent-system.md` §2 draws the topology and §6 the guards. This module is
the executable version of both, kept apart from `agents/graph.py` for one
reason: routing is where a long-running system stops or fails to stop, and a
predicate that lives in a lambda inside a wiring function cannot be tested
without building the whole graph. Everything here is a pure function of the
state, so "does this run terminate?" is answerable by a unit test rather than by
running an investigation.

**The guard runs first, always.** `check_guards()` is the first statement of
every edge function, before any edge-specific predicate. Ordering it the other
way is how a run that has already exhausted its step budget takes one more
"cheap" branch, and then another, because each individual edge looked
reasonable.

**Halting is not failing.** Step, deadline and budget exhaustion all route to the
Report, not to `END`. §6 is explicit that the run ships what it has and declares
itself partial -- an investigation that spent thirty minutes gathering evidence
and then returned nothing because the thirty-first minute arrived has wasted
everything it did, and told the user less than a truncated answer would have.

**The Critic loop has four independent brakes** (§13), and this module owns
three of them: the hard cap on `revision_count`, the monotonic-progress check
(a revision that resolves nothing ends the loop), and the artifact-hash check (a
re-run that produced byte-identical output will produce an identical critique).
The fourth is the global guard, which applies inside the loop like anywhere
else. Any one of them terminates the loop on its own; together they make an
infinite rewrite cycle structurally impossible rather than improbable.

One contract gap is worth naming here rather than in a commit message.
`models/enums.py` has no `truncated`, `timed_out` or `budget_exhausted` member,
though §6 names all three as terminal states. They are mapped onto
`COMPLETED_WITH_FINDINGS`, which is behaviourally right -- the report ships with
its limitations surfaced -- and the specific reason is preserved verbatim as the
`error_type` of an `AgentError` in `errors[]`, so the distinction survives for
the report and the evaluation harness. Adding the members is an enum change
outside this module's remit.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from langgraph.graph import END

from agents.state import AgentError, InvestigationState, PlanStep, TokenLedger
from backend.core.config import AgentSettings, get_settings
from models.base import utcnow
from models.enums import AgentName, InvestigationStatus

__all__ = [
    "ANALYSIS_BRANCHES",
    "FIXED_TAIL_STEPS",
    "NODE_AGENT",
    "Halt",
    "HaltReason",
    "NodeName",
    "check_guards",
    "dispatch_analysis",
    "estimated_steps_remaining",
    "halt_error",
    "phase_status",
    "route_after_collector",
    "route_after_critic",
    "route_after_final_critic",
    "route_after_graph_expansion",
    "route_after_insight",
    "route_after_planner",
    "route_after_report",
    "route_after_retriever",
    "route_after_strategy",
    "terminal_status",
]


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #


class NodeName(enum.StrEnum):
    """The graph's nodes. `StrEnum` so a name is usable directly as a LangGraph id.

    `CRITIC_FINAL` is a second node bound to the same Critic agent rather than a
    re-entry into `CRITIC`, because LangGraph node ids are unique and because the
    final pass has different rules: §13 makes it annotate-only, and giving it its
    own id is what makes "the final pass cannot re-open the loop" a property of
    the graph instead of a flag someone must remember to check.
    """

    PLANNER = "planner"
    COLLECTOR = "collector"
    RETRIEVER = "retriever"
    GRAPH_EXPANSION = "graph_expansion"
    INSIGHT = "insight"
    STRATEGY = "strategy"
    CRITIC = "critic"
    REPORT = "report"
    CRITIC_FINAL = "critic_final"


ANALYSIS_BRANCHES: Final[dict[AgentName, NodeName]] = {}
"""The fan-out: agents that may run concurrently when the plan names them.

**Empty right now, and deliberately kept rather than deleted.** The three market
agents that used to live here (Trend, Competitor, Forecast) went with the pivot,
so the graph is currently linear -- Planner through Report with no branch.

It stays because the developer platform gets its own fan-out -- research,
delegation and diagnosis are independent once a plan names them -- and rebuilding
guard-aware concurrent dispatch and the join would be a fortnight of re-deriving
decisions that are already made and tested here.

**What survived the pivot and what did not.** `dispatch_analysis` still selects
only the branches a plan named, still consults the guard before fanning out, and
still emits them in this table's canonical order so the downstream prompt-cache
prefix is stable. What went with the market agents is *dependency
serialisation*: `depends_on` was honoured by each branch's own outbound edge
(`route_after_trend` sent the run to Forecast when the plan declared Forecast
depended on it), and those edges were deleted with their nodes. Today every
planned branch is dispatched in one superstep regardless of `depends_on`. With an
empty table that is unobservable, but the first branch added here inherits it, so
whoever adds the second branch owns re-deriving the serialisation rather than
assuming it is still in place.

An empty table means `dispatch_analysis` routes straight to Insight, which is the
correct behaviour for a linear graph rather than a special case.
"""

NODE_AGENT: Final[dict[NodeName, AgentName]] = {
    NodeName.PLANNER: AgentName.PLANNER,
    NodeName.COLLECTOR: AgentName.COLLECTOR,
    NodeName.RETRIEVER: AgentName.RETRIEVER,
    # Graph Expansion is a node, not an agent (§2): it is backed by
    # `retrieval/graph_retrieval/`, makes no LLM call, and has no `AgentName`
    # member. Attributing its errors to the Retriever would blame the wrong
    # failure domain, which is the one thing the node exists to keep separate.
    NodeName.GRAPH_EXPANSION: AgentName.UNKNOWN,
    NodeName.INSIGHT: AgentName.INSIGHT,
    NodeName.STRATEGY: AgentName.STRATEGY,
    NodeName.CRITIC: AgentName.CRITIC,
    NodeName.REPORT: AgentName.REPORT,
    NodeName.CRITIC_FINAL: AgentName.CRITIC,
}

FIXED_TAIL_STEPS: Final = 5
"""Nodes every run must still execute after the plan's own steps.

Graph Expansion, Insight, Strategy, Critic, Report -- the final Critic pass is
excluded because it is exempt from the step guard (a run must always be able to
annotate the report it just shipped). Used to reject a plan that cannot reach
the Report inside `INVESTIGATION_MAX_STEPS` (§5.1) *before* its first step is
executed, rather than discovering it eleven steps in.
"""

_REENTRY_TARGETS: Final[dict[str, NodeName]] = {
    str(AgentName.RETRIEVER): NodeName.RETRIEVER,
    str(AgentName.INSIGHT): NodeName.INSIGHT,
    str(AgentName.STRATEGY): NodeName.STRATEGY,
    # A finding aimed at the Report is aimed at the wrong node: the Report only
    # renders what the state already holds, so re-running it reproduces the
    # defect. Route it to the reasoning step that produced the claim.
    str(AgentName.REPORT): NodeName.INSIGHT,
}

TERMINAL_NODES: Final[frozenset[NodeName]] = frozenset({NodeName.REPORT, NodeName.CRITIC_FINAL})
"""Nodes the guards do not stop.

A run that hit its ceiling still has to produce and annotate the partial report
-- otherwise the guard converts a truncated answer into no answer, which §6
explicitly rejects. Their spend is bounded (two nodes) and is charged against a
small documented overdraft in `agents/graph.py`.
"""


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


class HaltReason(enum.StrEnum):
    """Why a run stopped early. Recorded verbatim as an `AgentError.error_type`."""

    MAX_STEPS = "max_steps_exceeded"
    DEADLINE = "deadline_exceeded"
    TOKEN_BUDGET = "token_budget_exhausted"
    PLAN_TOO_LARGE = "plan_exceeds_step_budget"
    BLOCKING_ERROR = "blocking_error"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Halt:
    """A fired guard: why, where the run goes, and what it ends up as."""

    reason: HaltReason
    detail: str
    route: str
    """`NodeName.REPORT` for "ship what we have", `END` for "there is nothing to ship"."""

    status: InvestigationStatus


def check_guards(
    state: InvestigationState,
    *,
    settings: AgentSettings | None = None,
    now: datetime | None = None,
) -> Halt | None:
    """The global termination guard. First predicate to fire wins.

    Evaluated before every edge-specific predicate, in this order:

    1. **Already terminal / cancelled.** Nothing to do, and re-entering the graph
       after a cancel would restart work the user asked to stop.
    2. **Blocking error.** A node declared its failure unsurvivable (an empty
       plan, say). Continuing produces a confident report about nothing, so this
       is the one guard that routes to `END` with `failed`.
    3. **Step ceiling**, 4. **deadline**, 5. **token budget** -- the three
       resource limits, all of which ship a partial report.

    `now` is injectable because a deadline test that sleeps is a slow test that
    still only proves the clock advances.
    """
    resolved = settings if settings is not None else get_settings().agents
    moment = now if now is not None else utcnow()

    status = _status_of(state)
    if status == InvestigationStatus.CANCELLED:
        return Halt(
            reason=HaltReason.CANCELLED,
            detail="the investigation was cancelled",
            route=END,
            status=InvestigationStatus.CANCELLED,
        )
    if status.is_terminal:
        return Halt(
            reason=HaltReason.CANCELLED,
            detail=f"the investigation is already terminal ({status})",
            route=END,
            status=status,
        )

    blocking = _first_blocking_error(state)
    if blocking is not None:
        return Halt(
            reason=HaltReason.BLOCKING_ERROR,
            detail=f"{blocking.agent} failed unrecoverably: {blocking.error_type}",
            route=END,
            status=InvestigationStatus.FAILED,
        )

    steps = state.get("step_count", 0)
    if steps >= resolved.max_steps:
        return Halt(
            reason=HaltReason.MAX_STEPS,
            detail=f"{steps} steps executed, ceiling is {resolved.max_steps}",
            route=NodeName.REPORT,
            status=InvestigationStatus.COMPLETED_WITH_FINDINGS,
        )

    deadline = state.get("deadline_at")
    if deadline is not None and moment >= deadline:
        return Halt(
            reason=HaltReason.DEADLINE,
            detail=f"deadline {deadline.isoformat()} passed",
            route=NodeName.REPORT,
            status=InvestigationStatus.COMPLETED_WITH_FINDINGS,
        )

    spent = state.get("tokens_spent") or TokenLedger()
    if spent.total_tokens >= resolved.token_budget_per_investigation:
        return Halt(
            reason=HaltReason.TOKEN_BUDGET,
            detail=(
                f"{spent.total_tokens} tokens spent of {resolved.token_budget_per_investigation}"
            ),
            route=NodeName.REPORT,
            status=InvestigationStatus.COMPLETED_WITH_FINDINGS,
        )

    return None


def halt_error(halt: Halt, *, agent: AgentName = AgentName.UNKNOWN) -> AgentError:
    """The record a fired guard leaves in `errors[]`.

    This is the only place the *specific* reason survives, since the status enum
    collapses three distinct halts into `completed_with_findings`. The Report
    renders `errors[]` as its gaps section, so recording it here is what makes a
    truncated report say why it is short.
    """
    return AgentError(
        agent=agent,
        error_type=str(halt.reason),
        message=f"run halted: {halt.detail}",
        recoverable=halt.status != InvestigationStatus.FAILED,
    )


def _status_of(state: InvestigationState) -> InvestigationStatus:
    """Read `status` back as an enum.

    A checkpoint round-trips through JSON, so what comes back is a plain string,
    and `"cancelled".is_terminal` is an `AttributeError` that would surface as a
    node crash on resume rather than as a routing decision. `TolerantStrEnum`
    answers an unknown value with `UNKNOWN`, which is non-terminal -- the right
    default, because a status this build does not recognise is not permission to
    stop.
    """
    return InvestigationStatus(state.get("status") or InvestigationStatus.QUEUED)


def _first_blocking_error(state: InvestigationState) -> AgentError | None:
    """The first recorded error that the run cannot continue past."""
    for error in state.get("errors", []):
        if not error.recoverable:
            return error
    return None


# --------------------------------------------------------------------------- #
# Plan inspection
# --------------------------------------------------------------------------- #


def _plan(state: InvestigationState) -> Sequence[PlanStep]:
    return state.get("plan", [])


def estimated_steps_remaining(state: InvestigationState) -> int:
    """How many node executions the current plan still implies.

    Coarse by design -- it counts plan steps plus the fixed tail rather than
    modelling the Critic loop -- because its only job is to catch the Planner
    failure mode §5.1 names: a 30-step decomposition that exhausts
    `INVESTIGATION_MAX_STEPS` before it ever reaches the Report. Under-counting
    the loop is safe here; the loop has its own cap.
    """
    return len(_plan(state)) + FIXED_TAIL_STEPS


def _plan_exceeds_budget(state: InvestigationState, settings: AgentSettings) -> bool:
    executed = state.get("step_count", 0)
    return executed + estimated_steps_remaining(state) > settings.max_steps


def _needs_fresh_data(state: InvestigationState) -> bool:
    """Whether any plan step asked for a connector sync.

    A plan-level decision, decided here rather than inside the Collector,
    because dispatching a sync costs minutes and a quota: entering the node to
    find out would mean paying that cost to discover it was unnecessary
    (`agents/state.py`, `PlanStep.requires_fresh_data`).
    """
    return any(step.requires_fresh_data for step in _plan(state))


def _planned_branches(state: InvestigationState) -> list[NodeName]:
    """Fan-out branches named by the plan, in the graph's canonical order.

    Canonical order, not plan order: the dispatch list is part of the prompt
    cache prefix for the nodes downstream, and an order that varies with the
    plan's phrasing would invalidate it for no behavioural gain.
    """
    named = {step.agent for step in _plan(state)}
    return [node for agent, node in ANALYSIS_BRANCHES.items() if agent in named]



# --------------------------------------------------------------------------- #
# Edges
# --------------------------------------------------------------------------- #


def route_after_planner(state: InvestigationState, *, settings: AgentSettings | None = None) -> str:
    """`Planner -> Collector` when the plan needs fresh data, else `-> Retriever`.

    The extra predicate here is §5.1's plan-validity check: a plan whose steps
    cannot fit in the remaining step budget is rejected *now*, when the run can
    still say so in a report, rather than after eleven of its thirty steps have
    been paid for.
    """
    resolved = settings if settings is not None else get_settings().agents
    halt = check_guards(state, settings=resolved)
    if halt is not None:
        return halt.route
    if _plan_exceeds_budget(state, resolved):
        return NodeName.REPORT
    return NodeName.COLLECTOR if _needs_fresh_data(state) else NodeName.RETRIEVER


def route_after_collector(
    state: InvestigationState, *, settings: AgentSettings | None = None
) -> str:
    """`Collector -> Retriever`, always -- but the guard still gets first refusal.

    Static in §2's table, conditional here, because a static LangGraph edge
    cannot observe a deadline that expired during a connector sync. That is the
    single most likely place for one to expire.
    """
    halt = check_guards(state, settings=settings)
    return halt.route if halt is not None else NodeName.RETRIEVER


def route_after_retriever(
    state: InvestigationState, *, settings: AgentSettings | None = None
) -> str:
    """`Retriever -> Graph Expansion`, always, subject to the guard."""
    halt = check_guards(state, settings=settings)
    return halt.route if halt is not None else NodeName.GRAPH_EXPANSION


def dispatch_analysis(
    state: InvestigationState, *, settings: AgentSettings | None = None
) -> list[str]:
    """The fan-out: only the branches the plan named.

    Returns a *list*, which is how LangGraph expresses "start all of these in
    the same superstep". Two cases are worth their comments:

    - **no branches named** -- the plan is retrieval-only, so the run goes
      straight to Insight. Dispatching all three "just in case" would spend three
      model calls to produce three empty slices.
    """
    halt = check_guards(state, settings=settings)
    if halt is not None:
        return [halt.route]

    branches = _planned_branches(state)
    return [str(node) for node in branches] or [str(NodeName.INSIGHT)]



def route_after_graph_expansion(
    state: InvestigationState, *, settings: AgentSettings | None = None
) -> list[str]:
    """Alias of `dispatch_analysis`, named for the edge it wires."""
    return dispatch_analysis(state, settings=settings)


def route_after_insight(state: InvestigationState, *, settings: AgentSettings | None = None) -> str:
    """`Insight -> Strategy`, subject to the guard."""
    halt = check_guards(state, settings=settings)
    return halt.route if halt is not None else NodeName.STRATEGY


def route_after_strategy(
    state: InvestigationState, *, settings: AgentSettings | None = None
) -> str:
    """`Strategy -> Critic`, subject to the guard."""
    halt = check_guards(state, settings=settings)
    return halt.route if halt is not None else NodeName.CRITIC


def route_after_critic(state: InvestigationState, *, settings: AgentSettings | None = None) -> str:
    """The revision loop, and the four ways out of it.

    Order is deliberate: guard, then accept, then reject, then the three
    revision brakes, then the re-entry dispatch. Any reordering that puts the
    re-entry dispatch before a brake makes the brake advisory.

    `reject` exits to the Report rather than re-entering. A reject says the
    artifact is unsound, not that a named stage can repair it; §2's edge table
    has no `reject` edge, and re-running a stage against a verdict that named no
    target burns the remaining budget to arrive at the same place.
    """
    resolved = settings if settings is not None else get_settings().agents
    halt = check_guards(state, settings=resolved)
    if halt is not None:
        return halt.route

    critique = state.get("critique") or {}
    verdict = str(critique.get("verdict", "accept")).lower()
    if verdict != "revise":
        return NodeName.REPORT

    if state.get("revision_count", 0) >= resolved.max_critic_revisions:
        return NodeName.REPORT
    if _artifact_unchanged(state):
        return NodeName.REPORT
    if _progress_stalled(state):
        return NodeName.REPORT

    return _reentry_target(critique)


def route_after_report(state: InvestigationState, *, settings: AgentSettings | None = None) -> str:
    """`Report -> Critic (final pass)`, always.

    Not guarded. The final pass is what catches citation drift introduced during
    synthesis (§5.10), and a report shipped without it is a report whose
    citations nobody checked -- which is worse than a late one, and cheaper than
    the run that produced it.
    """
    return NodeName.CRITIC_FINAL


def route_after_final_critic(
    state: InvestigationState, *, settings: AgentSettings | None = None
) -> str:
    """`Critic (final pass) -> END`, unconditionally.

    A function rather than a static edge purely so that the annotate-only rule
    of §13 has somewhere to be stated and tested. If this ever returns a node,
    `Report -> Critic -> Report` becomes a second unbounded cycle, and unlike the
    revision loop it would have no counter to stop it.
    """
    return END


# --------------------------------------------------------------------------- #
# Critic-loop brakes
# --------------------------------------------------------------------------- #


def _critique_history(state: InvestigationState) -> Sequence[dict[str, Any]]:
    return state.get("critique_history", [])


def _artifact_unchanged(state: InvestigationState) -> bool:
    """Brake 3 (§13): the re-run produced a byte-identical artifact.

    Re-critiquing identical bytes yields an identical verdict, so the loop would
    oscillate until the step ceiling caught it. Reads `artifact_sha256` off the
    critique payload; a Critic that does not report one simply loses this brake,
    and the other three still hold.
    """
    history = _critique_history(state)
    if len(history) < 2:
        return False
    latest, previous = history[-1], history[-2]
    current_hash = latest.get("artifact_sha256")
    return bool(current_hash) and current_hash == previous.get("artifact_sha256")


def _unresolved_count(critique: dict[str, Any]) -> int:
    findings = critique.get("findings") or []
    return sum(1 for finding in findings if not _is_resolved(finding))


def _is_resolved(finding: Any) -> bool:
    return bool(finding.get("resolved", False)) if isinstance(finding, dict) else False


def _progress_stalled(state: InvestigationState) -> bool:
    """Brake 2 (§13): a revision that did not resolve at least one finding.

    Strictly monotonic, so a revision that trades one finding for another
    counts as no progress. That is intentional: the alternative is a loop that
    swaps defects at a fixed cost per cycle until the budget runs out, and the
    run ends with the same number of problems it started with, having paid for
    two extra passes to prove it.
    """
    history = _critique_history(state)
    if len(history) < 2:
        return False
    return _unresolved_count(history[-1]) >= _unresolved_count(history[-2])


def _reentry_target(critique: dict[str, Any]) -> NodeName:
    """Which stage a `revise` verdict re-enters, from the finding's `target_stage`.

    The critique's own `target_stage` wins; otherwise the first finding that
    names one does. An unlabelled critique routes to Insight, not to the
    Retriever: re-retrieval is the most expensive re-entry in the graph, and an
    unlabelled finding is not evidence that evidence is missing.
    """
    stage = critique.get("target_stage")
    if stage is None:
        for finding in critique.get("findings") or []:
            if isinstance(finding, dict) and finding.get("target_stage"):
                stage = finding["target_stage"]
                break
    return _REENTRY_TARGETS.get(str(stage).lower(), NodeName.INSIGHT)


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def phase_status(node: NodeName) -> InvestigationStatus:
    """The status a node stamps while it runs.

    Coarse on purpose -- the UI timeline reads per-step rows from
    `models/orm/run.py`, not this field. `status` exists so the API can answer
    "what is this run doing" in one column without joining.
    """
    if node is NodeName.PLANNER:
        return InvestigationStatus.PLANNING
    if node in (NodeName.CRITIC, NodeName.CRITIC_FINAL):
        return InvestigationStatus.REFLECTING
    return InvestigationStatus.RUNNING


def terminal_status(
    state: InvestigationState, *, settings: AgentSettings | None = None
) -> InvestigationStatus:
    """What the run ends as, evaluated once at the final Critic pass.

    `completed` requires an `accept` verdict *and* no unresolved findings and no
    fired guard. Everything short of that is `completed_with_findings`, which is
    the honest label: §13 requires the report to ship with its unresolved
    findings surfaced rather than presented as clean, and a status of `completed`
    on a report with an "unverified claims" section is precisely the
    presentation that rule forbids.
    """
    resolved = settings if settings is not None else get_settings().agents

    if _first_blocking_error(state) is not None:
        return InvestigationStatus.FAILED
    status = _status_of(state)
    if status in (InvestigationStatus.CANCELLED, InvestigationStatus.FAILED):
        return status
    if check_guards(state, settings=resolved) is not None:
        return InvestigationStatus.COMPLETED_WITH_FINDINGS
    if state.get("report") is None:
        # No artifact was produced at all. Not a failure -- the errors already
        # explain why -- but calling it `completed` would be a lie the API would
        # repeat.
        return InvestigationStatus.COMPLETED_WITH_FINDINGS

    critique = state.get("critique") or {}
    verdict = str(critique.get("verdict", "")).lower()
    if verdict == "accept" and _unresolved_count(critique) == 0:
        return InvestigationStatus.COMPLETED
    return InvestigationStatus.COMPLETED_WITH_FINDINGS
