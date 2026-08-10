"""Unit tests for `agents/router.py` -- every edge and every way a run stops.

These are the tests that decide whether a long-running system halts, so they are
written against the failure rather than against the happy path:

- **a guard that loses to an edge predicate.** The order in `docs/agent-system.md`
  §6 is guard-first; the way that regresses is someone adding a cheap-looking
  branch above it, after which an exhausted run takes one more step per edge.
- **a critic loop with one working brake.** §13 lists four. A test that only
  exercises the revision cap passes while the other three are broken, and the
  broken ones are the ones that fire when the Critic misbehaves rather than when
  it works.
- **a halt that fails the run.** Truncation, timeout and budget exhaustion all
  ship a partial report; routing any of them to `END` would turn thirty minutes
  of gathered evidence into nothing, which is precisely what §6 forbids.
- **a status that lies.** `completed` on a report carrying unresolved findings is
  the presentation §13 exists to prevent, and nothing downstream can detect it.

Everything is a pure function of the state, so there is no graph, no provider and
no clock dependency here: `check_guards()` takes `now` and every predicate takes
`settings`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import pytest
from langgraph.graph import END

from agents.router import (
    ANALYSIS_BRANCHES,
    HaltReason,
    NodeName,
    check_guards,
    dispatch_analysis,
    halt_error,
    phase_status,
    route_after_collector,
    route_after_critic,
    route_after_final_critic,
    route_after_insight,
    route_after_planner,
    route_after_report,
    route_after_retriever,
    route_after_strategy,
    terminal_status,
)
from agents.state import AgentError, InvestigationState, PlanStep, TokenLedger, new_state
from backend.core.config import AgentSettings
from models.base import utcnow
from models.enums import AgentName, InvestigationStatus

pytestmark = pytest.mark.unit


@contextmanager
def registered_branch(agent: AgentName, node: NodeName) -> Iterator[None]:
    """Temporarily put a branch in `ANALYSIS_BRANCHES`.

    The table is empty since the pivot and the dispatch machinery around it is
    not -- planned-only selection, dependency serialisation and the deadline
    guard are all still live code with no data to exercise them. Registering a
    branch for the duration of a test is what keeps that code covered until the
    developer platform's own branches land, rather than letting it rot untested
    and be rediscovered as broken the day something is added.

    Any existing agent will do as the stand-in: `dispatch_analysis` reads the
    plan's agent names against this table and never calls the node, so the
    borrowed pairing costs nothing and no node executes.
    """
    ANALYSIS_BRANCHES[agent] = node
    try:
        yield
    finally:
        del ANALYSIS_BRANCHES[agent]


SETTINGS = AgentSettings(
    INVESTIGATION_MAX_STEPS=20,
    INVESTIGATION_TIMEOUT_SECONDS=1800,
    MAX_CRITIC_REVISIONS=2,
    INVESTIGATION_TOKEN_BUDGET=10_000,
)
"""Deliberately small limits. The defaults (50 steps, a million tokens) would
make every guard test either slow to set up or indistinguishable from the
unguarded case."""


def make_state(**overrides: Any) -> InvestigationState:
    """An entry-point state with the fields under test overridden."""
    state = new_state(
        investigation_id="inv-1",
        tenant_id="tenant-1",
        query="how is our category shifting?",
        deadline_at=utcnow() + timedelta(minutes=30),
        trace_id="trace-1",
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def plan(*specs: tuple[str, AgentName, bool]) -> list[PlanStep]:
    return [
        PlanStep(id=step_id, description=step_id, agent=agent, requires_fresh_data=fresh)
        for step_id, agent, fresh in specs
    ]


def critique(
    verdict: str,
    *,
    findings: list[dict[str, Any]] | None = None,
    target_stage: str | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"verdict": verdict, "findings": findings or []}
    if target_stage is not None:
        payload["target_stage"] = target_stage
    if artifact_sha256 is not None:
        payload["artifact_sha256"] = artifact_sha256
    return payload


# --------------------------------------------------------------------------- #
# Topology: the edges of §2's table
# --------------------------------------------------------------------------- #


def test_planner_routes_to_collector_only_when_a_step_needs_fresh_data() -> None:
    fresh = make_state(plan=plan(("s1", AgentName.RETRIEVER, True)))
    stale = make_state(plan=plan(("s1", AgentName.RETRIEVER, False)))

    assert route_after_planner(fresh, settings=SETTINGS) == NodeName.COLLECTOR
    assert route_after_planner(stale, settings=SETTINGS) == NodeName.RETRIEVER


def test_planner_rejects_a_plan_that_cannot_reach_the_report() -> None:
    """§5.1's over-decomposition failure, caught before the first step is paid for."""
    oversized = make_state(
        plan=plan(*[(f"s{i}", AgentName.RETRIEVER, False) for i in range(SETTINGS.max_steps)])
    )
    assert route_after_planner(oversized, settings=SETTINGS) == NodeName.REPORT


def test_static_edges_still_consult_the_guard() -> None:
    ok = make_state()
    assert route_after_collector(ok, settings=SETTINGS) == NodeName.RETRIEVER
    assert route_after_retriever(ok, settings=SETTINGS) == NodeName.GRAPH_EXPANSION
    assert route_after_insight(ok, settings=SETTINGS) == NodeName.STRATEGY
    assert route_after_strategy(ok, settings=SETTINGS) == NodeName.CRITIC

    expired = make_state(deadline_at=utcnow() - timedelta(seconds=1))
    for edge in (
        route_after_collector,
        route_after_retriever,
        route_after_insight,
        route_after_strategy,
    ):
        assert edge(expired, settings=SETTINGS) == NodeName.REPORT


def test_dispatch_goes_straight_to_insight_while_no_branches_are_registered() -> None:
    """`ANALYSIS_BRANCHES` is empty today, so every plan joins at Insight.

    Kept as a conditional edge rather than collapsed to a static one. The fan-out
    machinery -- planned-only dispatch, dependency serialisation, the deadline
    guard -- is tested below against a temporarily registered branch, because the
    developer platform's own branches (research, delegation, diagnosis) plug into
    exactly this table and rebuilding tested dispatch from scratch is expensive in
    a way that keeping forty lines is not.
    """
    for planned in (AgentName.RETRIEVER, AgentName.INSIGHT, AgentName.STRATEGY):
        state = make_state(plan=plan(("s1", planned, False)))
        assert dispatch_analysis(state, settings=SETTINGS) == [str(NodeName.INSIGHT)]


def test_only_planned_branches_are_dispatched() -> None:
    """A branch the plan never asked for costs a model call and produces a
    section of the report nobody requested."""
    with registered_branch(AgentName.STRATEGY, NodeName.STRATEGY):
        planned = make_state(plan=plan(("s1", AgentName.STRATEGY, False)))
        assert dispatch_analysis(planned, settings=SETTINGS) == [str(NodeName.STRATEGY)]

        unplanned = make_state(plan=plan(("s1", AgentName.RETRIEVER, False)))
        assert dispatch_analysis(unplanned, settings=SETTINGS) == [str(NodeName.INSIGHT)]


def test_branches_dispatch_in_canonical_order_not_plan_order() -> None:
    """The dispatch list is part of the prompt-cache prefix for the nodes
    downstream, so an order that varied with the plan's phrasing would invalidate
    that cache on every run for no behavioural gain."""
    with registered_branch(AgentName.STRATEGY, NodeName.STRATEGY), registered_branch(
        AgentName.CRITIC, NodeName.CRITIC
    ):
        forwards = plan(("a1", AgentName.STRATEGY, False), ("b1", AgentName.CRITIC, False))
        backwards = plan(("b1", AgentName.CRITIC, False), ("a1", AgentName.STRATEGY, False))
        expected = [str(NodeName.STRATEGY), str(NodeName.CRITIC)]

        assert dispatch_analysis(make_state(plan=forwards), settings=SETTINGS) == expected
        assert dispatch_analysis(make_state(plan=backwards), settings=SETTINGS) == expected


def test_depends_on_no_longer_serialises_a_branch() -> None:
    """Regression pin on a capability the pivot removed.

    `depends_on` used to be honoured by each branch's own outbound edge, and those
    edges were deleted with the market agents. Every planned branch now dispatches
    in one superstep regardless. This is asserted rather than left implicit
    because it is invisible while the table is empty and inherited silently by the
    first branch added -- a dependent branch would run against state its
    predecessor has not written, which does not fail, it answers from nothing.
    """
    with registered_branch(AgentName.STRATEGY, NodeName.STRATEGY), registered_branch(
        AgentName.CRITIC, NodeName.CRITIC
    ):
        steps = [
            PlanStep(id="a1", description="first", agent=AgentName.STRATEGY),
            PlanStep(
                id="b1", description="second", agent=AgentName.CRITIC, depends_on=("a1",)
            ),
        ]
        assert dispatch_analysis(make_state(plan=steps), settings=SETTINGS) == [
            str(NodeName.STRATEGY),
            str(NodeName.CRITIC),
        ]


def test_an_expired_deadline_skips_dispatch_entirely() -> None:
    """The guard is consulted on the fan-out edge too. A run past its deadline
    that still fanned out would spend its remaining budget on branches whose
    output the Report node has no time left to read."""
    with registered_branch(AgentName.STRATEGY, NodeName.STRATEGY):
        expired = make_state(
            plan=plan(("s1", AgentName.STRATEGY, False)),
            deadline_at=utcnow() - timedelta(seconds=1),
        )
        assert dispatch_analysis(expired, settings=SETTINGS) == [str(NodeName.REPORT)]


def test_report_always_hands_off_to_the_final_critic() -> None:
    assert route_after_report(make_state(), settings=SETTINGS) == NodeName.CRITIC_FINAL


def test_the_final_critic_pass_can_never_re_open_the_loop() -> None:
    """§13: annotate-only. A second destination here is an unbounded cycle."""
    for state in (
        make_state(),
        make_state(critique=critique("revise", findings=[{"target_stage": "insight"}])),
        make_state(status=InvestigationStatus.RUNNING, revision_count=0),
    ):
        assert route_after_final_critic(state, settings=SETTINGS) == END


# --------------------------------------------------------------------------- #
# Global guards (§6)
# --------------------------------------------------------------------------- #


def test_step_ceiling_halts_the_run_towards_the_report() -> None:
    state = make_state(step_count=SETTINGS.max_steps)
    halt = check_guards(state, settings=SETTINGS)

    assert halt is not None
    assert halt.reason is HaltReason.MAX_STEPS
    assert halt.route == NodeName.REPORT
    assert halt.status is InvestigationStatus.COMPLETED_WITH_FINDINGS


def test_deadline_halts_the_run_towards_the_report() -> None:
    deadline = utcnow() + timedelta(minutes=30)
    state = make_state(deadline_at=deadline)

    assert check_guards(state, settings=SETTINGS, now=deadline - timedelta(seconds=1)) is None

    halt = check_guards(state, settings=SETTINGS, now=deadline)
    assert halt is not None
    assert halt.reason is HaltReason.DEADLINE
    assert halt.route == NodeName.REPORT


def test_token_budget_halts_the_run_towards_the_report() -> None:
    state = make_state(
        tokens_spent=TokenLedger(input_tokens=9_000, output_tokens=1_000, calls=4),
    )
    halt = check_guards(state, settings=SETTINGS)

    assert halt is not None
    assert halt.reason is HaltReason.TOKEN_BUDGET
    assert halt.route == NodeName.REPORT


def test_cancellation_and_blocking_errors_end_the_run_instead_of_reporting() -> None:
    """The two halts that must *not* spend a model call writing a report."""
    cancelled = check_guards(make_state(status=InvestigationStatus.CANCELLED), settings=SETTINGS)
    assert cancelled is not None
    assert cancelled.route == END
    assert cancelled.status is InvestigationStatus.CANCELLED

    blocked = check_guards(
        make_state(
            errors=[
                AgentError(
                    agent=AgentName.PLANNER,
                    error_type="schema_violation",
                    message="no plan",
                    recoverable=False,
                )
            ]
        ),
        settings=SETTINGS,
    )
    assert blocked is not None
    assert blocked.reason is HaltReason.BLOCKING_ERROR
    assert blocked.route == END
    assert blocked.status is InvestigationStatus.FAILED


def test_a_recoverable_branch_failure_does_not_halt_the_run() -> None:
    """§6: a single branch failure is recorded and the run continues."""
    state = make_state(
        errors=[
            AgentError(
                agent=AgentName.STRATEGY,
                error_type="provider_error",
                message="did not converge",
                recoverable=True,
            )
        ]
    )
    assert check_guards(state, settings=SETTINGS) is None


def test_the_guard_wins_against_every_edge_predicate() -> None:
    """Guard first, always. Each of these states would otherwise route elsewhere."""
    exhausted = {"step_count": SETTINGS.max_steps}

    assert (
        route_after_planner(
            make_state(plan=plan(("s1", AgentName.RETRIEVER, True)), **exhausted),
            settings=SETTINGS,
        )
        == NodeName.REPORT
    )
    with registered_branch(AgentName.STRATEGY, NodeName.STRATEGY):
        assert dispatch_analysis(
            make_state(plan=plan(("t1", AgentName.STRATEGY, False)), **exhausted),
            settings=SETTINGS,
        ) == [str(NodeName.REPORT)]
    assert (
        route_after_critic(make_state(critique=critique("revise"), **exhausted), settings=SETTINGS)
        == NodeName.REPORT
    )


def test_a_status_string_from_a_checkpoint_does_not_crash_the_guard() -> None:
    """A resumed run's state has been through JSON: `status` comes back a `str`."""
    state = make_state()
    state["status"] = "cancelled"  # type: ignore[typeddict-item]
    halt = check_guards(state, settings=SETTINGS)

    assert halt is not None
    assert halt.status is InvestigationStatus.CANCELLED


def test_halt_error_preserves_the_reason_the_status_enum_cannot_express() -> None:
    halt = check_guards(make_state(step_count=SETTINGS.max_steps), settings=SETTINGS)
    assert halt is not None

    recorded = halt_error(halt, agent=AgentName.REPORT)
    assert recorded.error_type == str(HaltReason.MAX_STEPS)
    assert recorded.recoverable is True


# --------------------------------------------------------------------------- #
# The Critic loop (§13)
# --------------------------------------------------------------------------- #


def test_accept_leaves_the_loop_for_the_report() -> None:
    state = make_state(critique=critique("accept"))
    assert route_after_critic(state, settings=SETTINGS) == NodeName.REPORT


def test_reject_ships_rather_than_re_entering() -> None:
    state = make_state(critique=critique("reject", findings=[{"target_stage": "insight"}]))
    assert route_after_critic(state, settings=SETTINGS) == NodeName.REPORT


@pytest.mark.parametrize(
    ("target_stage", "expected"),
    [
        ("retriever", NodeName.RETRIEVER),
        ("insight", NodeName.INSIGHT),
        ("strategy", NodeName.STRATEGY),
        ("report", NodeName.INSIGHT),
        (None, NodeName.INSIGHT),
        ("nonsense", NodeName.INSIGHT),
    ],
)
def test_revise_re_enters_at_the_stage_the_finding_names(
    target_stage: str | None, expected: NodeName
) -> None:
    findings = [{"code": "unsupported_claim", "target_stage": target_stage}]
    state = make_state(critique=critique("revise", findings=findings), revision_count=0)
    assert route_after_critic(state, settings=SETTINGS) == expected


def test_brake_one_the_revision_cap_forces_the_report() -> None:
    state = make_state(
        critique=critique("revise", target_stage="insight"),
        revision_count=SETTINGS.max_critic_revisions,
    )
    assert route_after_critic(state, settings=SETTINGS) == NodeName.REPORT


def test_brake_two_a_revision_that_resolves_nothing_ends_the_loop() -> None:
    """Strictly monotonic: trading one finding for another is not progress."""
    first = critique("revise", findings=[{"target_stage": "insight"}])
    second = critique("revise", findings=[{"target_stage": "strategy"}])
    state = make_state(critique=second, critique_history=[first, second], revision_count=1)

    assert route_after_critic(state, settings=SETTINGS) == NodeName.REPORT


def test_brake_two_allows_a_revision_that_resolved_a_finding() -> None:
    first = critique("revise", findings=[{"target_stage": "insight"}, {"target_stage": "insight"}])
    second = critique("revise", findings=[{"target_stage": "insight"}])
    state = make_state(critique=second, critique_history=[first, second], revision_count=1)

    assert route_after_critic(state, settings=SETTINGS) == NodeName.INSIGHT


def test_brake_three_an_unchanged_artifact_is_not_re_critiqued() -> None:
    first = critique("revise", findings=[{"target_stage": "insight"}], artifact_sha256="abc")
    second = critique(
        "revise",
        findings=[{"target_stage": "insight"}],
        artifact_sha256="abc",
    )
    state = make_state(critique=second, critique_history=[first, second], revision_count=1)

    assert route_after_critic(state, settings=SETTINGS) == NodeName.REPORT


def test_an_always_revising_critic_cannot_loop_forever() -> None:
    """The loop's termination, proved without a graph.

    Drives the edge with a Critic that always says `revise` and a revision
    counter that only ever advances, and asserts the run reaches the Report
    within the cap. If any brake stopped advancing the state, this would not
    terminate -- which is the property being asserted.
    """
    state = make_state(critique=critique("revise", target_stage="insight"))
    visited: list[str] = []

    for _ in range(SETTINGS.max_critic_revisions + 5):
        target = route_after_critic(state, settings=SETTINGS)
        visited.append(str(target))
        if target == NodeName.REPORT:
            break
        state["revision_count"] = state.get("revision_count", 0) + 1
    else:  # pragma: no cover -- reaching this is the failure the test exists for
        pytest.fail(f"critic loop did not terminate: {visited}")

    assert visited.count(str(NodeName.REPORT)) == 1
    assert len(visited) == SETTINGS.max_critic_revisions + 1


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


def test_phase_status_distinguishes_planning_and_reflection() -> None:
    assert phase_status(NodeName.PLANNER) is InvestigationStatus.PLANNING
    assert phase_status(NodeName.CRITIC) is InvestigationStatus.REFLECTING
    assert phase_status(NodeName.CRITIC_FINAL) is InvestigationStatus.REFLECTING
    assert phase_status(NodeName.RETRIEVER) is InvestigationStatus.RUNNING


def test_a_clean_accepted_run_completes() -> None:
    state = make_state(report={"title": "x"}, critique=critique("accept"), step_count=9)
    assert terminal_status(state, settings=SETTINGS) is InvestigationStatus.COMPLETED


def test_unresolved_findings_downgrade_the_terminal_status() -> None:
    state = make_state(
        report={"title": "x"},
        critique=critique("accept", findings=[{"code": "overconfident"}]),
        step_count=9,
    )
    assert terminal_status(state, settings=SETTINGS) is InvestigationStatus.COMPLETED_WITH_FINDINGS


def test_a_capped_revision_loop_ships_with_findings_rather_than_failing() -> None:
    state = make_state(
        report={"title": "x"},
        critique=critique("revise", findings=[{"code": "unsupported_claim"}]),
        revision_count=SETTINGS.max_critic_revisions,
        step_count=12,
    )
    assert terminal_status(state, settings=SETTINGS) is InvestigationStatus.COMPLETED_WITH_FINDINGS


def test_a_report_that_was_never_produced_is_not_reported_as_completed() -> None:
    state = make_state(critique=critique("accept"), step_count=9)
    assert terminal_status(state, settings=SETTINGS) is InvestigationStatus.COMPLETED_WITH_FINDINGS


def test_a_blocking_error_ends_as_failed() -> None:
    state = make_state(
        report={"title": "x"},
        critique=critique("accept"),
        errors=[
            AgentError(
                agent=AgentName.PLANNER,
                error_type="schema_violation",
                message="no plan",
                recoverable=False,
            )
        ],
    )
    assert terminal_status(state, settings=SETTINGS) is InvestigationStatus.FAILED
