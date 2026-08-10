"""Unit tests for `agents/graph.py` -- the topology, the join, and the brakes.

Driven end to end with fake agents. They are real `BaseAgent` subclasses over
`FakeLLMProvider`, not stand-in callables, because half of what is being asserted
lives in the base class: that a node returns a delta instead of raising, that its
spend lands in `tokens_spent`, and that its `PromptRef` lands in
`prompt_versions`. A test double that skipped `BaseAgent` would prove the wiring
and nothing about what flows through it.

What these tests are actually defending:

- **the topology drifting from §2.** Asserted as an execution *order*, not as a
  set of edges, because the order is what a reader of the design document would
  recognise -- and because LangGraph's drawn graph shows extra incoming edges for
  a deferred node, so an exhaustive edge assertion would be asserting an
  implementation detail of the drawing code.
- **a fan-out branch that dies taking the join with it.** Two ways to die are
  covered, and the second is the one nobody writes a test for: not raising, but
  never returning.
- **a critic loop that does not terminate.** Proved with a Critic that *always*
  says `revise` and never increments the counter that would stop it, which is
  exactly the shape a prompt regression takes.
- **a guard that stops the run but also stops the report.** Each guard must halt
  the work and still ship, because §6 asks for a partial answer rather than none.

No network, no services, no Docker: the provider is `FakeLLMProvider`, the
checkpointer is `InMemorySaver`, and the tool registry denies everything (none of
these agents is given a tool).
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import pytest
from langgraph.graph import END, START
from pydantic import BaseModel, Field

from agents.base import BaseAgent, CollectingTraceSink, StaticPromptSource
from agents.checkpointer import memory_checkpointer, thread_config
from agents.errors import NO_RETRY_POLICY, PermanentAgentError
from agents.graph import (
    CONCURRENT_NODES,
    MIN_NODE_TIMEOUT_SECONDS,
    REQUIRED_NODES,
    build_investigation_graph,
    unavailable_branches,
    wrap_node,
)
from agents.router import NodeName
from agents.state import InvestigationState, PlanStep, TokenLedger, new_state
from backend.core.config import AgentSettings
from models.base import utcnow
from models.enums import AgentName, InvestigationStatus
from services.llm.provider import FakeLLMProvider
from services.llm.router import ModelTier

pytestmark = pytest.mark.unit


SETTINGS = AgentSettings(
    INVESTIGATION_MAX_STEPS=30,
    INVESTIGATION_TIMEOUT_SECONDS=1800,
    MAX_CRITIC_REVISIONS=2,
    INVESTIGATION_TOKEN_BUDGET=1_000_000,
)

SPEC_ORDER = [
    NodeName.PLANNER,
    NodeName.RETRIEVER,
    NodeName.GRAPH_EXPANSION,
    NodeName.INSIGHT,
    NodeName.STRATEGY,
    NodeName.CRITIC,
    NodeName.REPORT,
    NodeName.CRITIC_FINAL,
]
"""The §2 spine, with the fan-out removed -- branch order between concurrent
nodes is not deterministic and asserting it would be asserting the scheduler."""


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class ScriptedRequest(BaseModel):
    """What a fake agent reads out of the state."""

    query: str
    evidence_count: int = 0
    insight_count: int = 0


class ScriptedOutput(BaseModel):
    """What a fake agent produces. The delta is scripted per test."""

    delta: dict[str, Any] = Field(default_factory=dict)


class DenyAllRegistry:
    """A tool registry that grants nothing.

    None of these agents declares a tool, so every grant would be unused -- and a
    permissive fake would hide a `use_tool` call that the real allowlist denies.
    """

    def is_allowed(self, agent: AgentName, name: str) -> bool:
        return False

    async def invoke(
        self,
        *,
        agent: AgentName,
        tool: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> Any:  # pragma: no cover -- unreachable while `is_allowed` is False
        raise AssertionError("no fake agent in this suite is allowed a tool")


class ScriptedAgent(BaseAgent[ScriptedRequest, ScriptedOutput], abstract=True):
    """A `BaseAgent` whose output is scripted and whose calls are recorded.

    `retry_policy = NO_RETRY_POLICY` so a test that scripts a failure spends no
    wall-clock on backoff: the retry behaviour itself belongs to the error-policy
    tests, and paying for it in every graph test buys nothing.

    `script` supplies a *different* delta per invocation, with the last one
    repeating. Only the Critic needs it, and it needs it for a specific reason:
    a Critic that returns the identical critique every pass is stopped by §13's
    monotonic-progress brake, so a test written against a constant fake proves
    that brake and never reaches the revision cap it claims to be testing.
    """

    output_model = ScriptedOutput
    tools = frozenset()
    retry_policy = NO_RETRY_POLICY

    def __init__(
        self,
        journal: list[str],
        *,
        delta: dict[str, Any] | None = None,
        script: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
        hang: bool = False,
    ) -> None:
        super().__init__(
            FakeLLMProvider(default=ScriptedOutput()),
            DenyAllRegistry(),
            prompts=StaticPromptSource(f"system prompt for {self.name}"),
            trace=CollectingTraceSink(),
        )
        self.journal = journal
        self.script = list(script) if script else [delta or {}]
        self.error = error
        self.hang = hang
        self.pause_seconds = 0.0
        """Real work, briefly. Only for the tests that assert on a node's time slice;
        everything else runs at zero cost."""
        self.seen: list[ScriptedRequest] = []
        self.calls = 0

    @property
    def delta(self) -> dict[str, Any]:
        """This invocation's delta: the next scripted one, or the last one again."""
        return self.script[min(self.calls, len(self.script) - 1)]

    def build_input(self, state: InvestigationState) -> ScriptedRequest:
        return ScriptedRequest(
            query=state.get("query", ""),
            evidence_count=len(state.get("evidence", [])),
            insight_count=len(state.get("insights", [])),
        )

    async def execute(self, request: ScriptedRequest, ctx: Any) -> ScriptedOutput:
        self.journal.append(str(self.name))
        self.seen.append(request)
        delta = self.delta
        self.calls += 1
        if self.hang:
            await asyncio.sleep(3600)
        if self.pause_seconds:
            await asyncio.sleep(self.pause_seconds)
        if self.error is not None:
            raise self.error
        # A real model call, so the ledger and the prompt ref are exercised
        # rather than assumed.
        prompt = self.render_prompt(ctx)
        await self.call_model(ctx, prompt=request.query, schema=ScriptedOutput, system=prompt.text)
        return ScriptedOutput(delta=delta)

    def to_delta(self, output: ScriptedOutput, state: InvestigationState) -> dict[str, Any]:
        return dict(output.delta)


def agent_for(
    node: NodeName,
    journal: list[str],
    *,
    delta: dict[str, Any] | None = None,
    script: list[dict[str, Any]] | None = None,
    error: Exception | None = None,
    hang: bool = False,
) -> ScriptedAgent:
    """Build a fake agent bound to one node's identity and tier."""
    agent_name = {
        NodeName.PLANNER: AgentName.PLANNER,
        NodeName.COLLECTOR: AgentName.COLLECTOR,
        NodeName.RETRIEVER: AgentName.RETRIEVER,
        NodeName.GRAPH_EXPANSION: AgentName.RETRIEVER,
        NodeName.INSIGHT: AgentName.INSIGHT,
        NodeName.STRATEGY: AgentName.STRATEGY,
        NodeName.CRITIC: AgentName.CRITIC,
        NodeName.REPORT: AgentName.REPORT,
        NodeName.CRITIC_FINAL: AgentName.CRITIC,
    }[node]
    # A subclass per node, so `__init_subclass__` runs and the declared contract
    # (name, tier, output model, allowlist) is exercised for every one of them.
    cls = type(
        f"{node}Agent",
        (ScriptedAgent,),
        {"name": agent_name, "tier": ModelTier.FAST, "__module__": __name__},
    )
    return cls(  # type: ignore[no-any-return]
        journal, delta=delta, script=script, error=error, hang=hang
    )


def build(
    journal: list[str],
    *,
    plan: list[PlanStep] | None = None,
    overrides: Mapping[NodeName, ScriptedAgent] | None = None,
    critique: dict[str, Any] | None = None,
    settings: AgentSettings = SETTINGS,
    checkpointer: Any | None = None,
) -> Any:
    """Compile a graph of fake agents, with per-node overrides where a test needs one."""
    verdict = critique or {"verdict": "accept", "findings": []}
    deltas: dict[NodeName, dict[str, Any]] = {
        NodeName.PLANNER: {
            "objective": "understand the shift",
            "plan": plan if plan is not None else [],
        },
        NodeName.INSIGHT: {"insights": [{"statement": "because"}]},
        NodeName.STRATEGY: {"recommendations": [{"action": "do the thing"}]},
        NodeName.CRITIC: {"critique": verdict},
        NodeName.REPORT: {"report": {"title": "findings"}},
        NodeName.CRITIC_FINAL: {"confidence": 0.61},
    }
    nodes: dict[NodeName, Any] = {
        node: agent_for(node, journal, delta=deltas.get(node)) for node in REQUIRED_NODES
    }
    nodes.update(overrides or {})
    return build_investigation_graph(nodes, settings=settings, checkpointer=checkpointer)


def entry_state(**overrides: Any) -> InvestigationState:
    state = new_state(
        investigation_id="inv-1",
        tenant_id="tenant-1",
        query="how is our category shifting?",
        deadline_at=utcnow() + timedelta(minutes=30),
        trace_id="trace-1",
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def analysis_plan(*agents: AgentName) -> list[PlanStep]:
    return [
        PlanStep(id=f"s{i}", description=str(agent), agent=agent) for i, agent in enumerate(agents)
    ]


def relentless_critic(
    journal: list[str],
    *,
    target_stage: str,
    passes: int = 6,
) -> ScriptedAgent:
    """A Critic that says `revise` forever while still resolving a finding each pass.

    The distinction matters. §13 gives the loop four independent brakes, and a
    fake that returns the identical critique every time is stopped by the second
    (monotonic progress) on its very first revision -- so a test built on one
    would report the loop as bounded while the *cap* was broken. Shrinking the
    finding list keeps brakes 2 and 3 satisfied, which leaves brake 1 as the only
    thing that can end the loop, which is the thing being asserted.
    """
    return agent_for(
        NodeName.CRITIC,
        journal,
        script=[
            {
                "critique": {
                    "verdict": "revise",
                    "findings": [
                        {"code": "unsupported_claim", "target_stage": target_stage}
                        for _ in range(remaining)
                    ],
                }
            }
            for remaining in range(passes, 0, -1)
        ],
    )


# --------------------------------------------------------------------------- #
# Topology
# --------------------------------------------------------------------------- #


def test_every_node_of_the_spec_is_present_and_none_else() -> None:
    graph = build([]).get_graph()
    assert set(graph.nodes) == {START, END, *(str(node) for node in REQUIRED_NODES)}


def test_the_spec_edges_all_exist() -> None:
    """Every transition §2's table names, present in the compiled graph."""
    drawn = {(edge.source, edge.target) for edge in build([]).get_graph().edges}
    expected = {
        (START, str(NodeName.PLANNER)),
        (str(NodeName.PLANNER), str(NodeName.COLLECTOR)),
        (str(NodeName.PLANNER), str(NodeName.RETRIEVER)),
        (str(NodeName.COLLECTOR), str(NodeName.RETRIEVER)),
        (str(NodeName.RETRIEVER), str(NodeName.GRAPH_EXPANSION)),
        (str(NodeName.GRAPH_EXPANSION), str(NodeName.INSIGHT)),
        (str(NodeName.INSIGHT), str(NodeName.STRATEGY)),
        (str(NodeName.STRATEGY), str(NodeName.CRITIC)),
        (str(NodeName.CRITIC), str(NodeName.REPORT)),
        (str(NodeName.CRITIC), str(NodeName.RETRIEVER)),
        (str(NodeName.CRITIC), str(NodeName.INSIGHT)),
        (str(NodeName.CRITIC), str(NodeName.STRATEGY)),
        (str(NodeName.REPORT), str(NodeName.CRITIC_FINAL)),
        (str(NodeName.CRITIC_FINAL), END),
    }
    assert expected <= drawn


def test_a_missing_node_is_refused_rather_than_stubbed() -> None:
    journal: list[str] = []
    nodes = {node: agent_for(node, journal) for node in REQUIRED_NODES if node != NodeName.CRITIC}

    with pytest.raises(ValueError, match="critic"):
        build_investigation_graph(nodes, settings=SETTINGS)


async def test_the_happy_path_visits_the_spec_order() -> None:
    journal: list[str] = []
    graph = build(journal, plan=analysis_plan(AgentName.INSIGHT, AgentName.STRATEGY))

    final = await graph.ainvoke(entry_state())

    assert journal == [
        str(AgentName.PLANNER),
        str(AgentName.RETRIEVER),  # graph expansion is bound to the retriever identity
        str(AgentName.RETRIEVER),
        str(AgentName.INSIGHT),
        str(AgentName.STRATEGY),
        str(AgentName.CRITIC),
        str(AgentName.REPORT),
        str(AgentName.CRITIC),
    ]
    assert final["status"] is InvestigationStatus.COMPLETED
    assert final["report"] == {"title": "findings"}


async def test_the_collector_runs_only_when_the_plan_asks_for_fresh_data() -> None:
    stale: list[str] = []
    await build(stale, plan=analysis_plan(AgentName.INSIGHT)).ainvoke(entry_state())
    assert str(AgentName.COLLECTOR) not in stale

    fresh: list[str] = []
    plan = [
        PlanStep(id="s0", description="sync", agent=AgentName.COLLECTOR, requires_fresh_data=True)
    ]
    await build(fresh, plan=plan).ainvoke(entry_state())
    assert str(AgentName.COLLECTOR) in fresh


async def test_the_graph_is_linear_while_no_branches_are_registered() -> None:
    """Graph Expansion joins straight to Insight, whatever the plan names.

    The fan-out tests that used to live here -- planned-only dispatch, the join
    waiting for every branch, a dead branch neither hanging nor failing the join
    -- went with the three market branches they were written against, because
    `ANALYSIS_BRANCHES` and `CONCURRENT_NODES` are both empty and `NodeName` has
    no member to stand one up under. The machinery in `agents/graph.py` that
    those tests covered is still there and is now dormant and unexercised;
    whoever registers the developer platform's first branch owns re-deriving that
    coverage rather than assuming it survived.
    """
    journal: list[str] = []
    graph = build(journal, plan=analysis_plan(AgentName.INSIGHT, AgentName.STRATEGY))

    final = await graph.ainvoke(entry_state())

    assert unavailable_branches(final) == []
    assert journal.count(str(AgentName.INSIGHT)) == 1
    assert final["status"] is InvestigationStatus.COMPLETED


async def test_a_fan_out_branch_never_writes_a_scalar_the_others_also_write() -> None:
    """The lost-update hazard, asserted as an absence.

    Concurrent nodes must leave `status` alone; the wrapper is what guarantees
    it, and the only way to see the guarantee is to check that a branch's delta
    does not carry the key.
    """
    journal: list[str] = []
    for node in CONCURRENT_NODES:
        wrapped = wrap_node(node, agent_for(node, journal), settings=SETTINGS)
        delta = await wrapped(entry_state())
        assert "status" not in delta
        assert "revision_count" not in delta


# --------------------------------------------------------------------------- #
# The Critic loop
# --------------------------------------------------------------------------- #


async def test_a_critic_that_always_revises_still_terminates() -> None:
    """The loop's termination, end to end, with the worst-case Critic.

    The fake never increments `revision_count` -- the graph does. That is the
    point: if termination depended on the Critic cooperating, a prompt regression
    would produce exactly this agent and the run would rewrite until it ran out
    of budget.

    It also keeps the other three brakes off the table (it makes progress, its
    artifacts differ, and the limits are far away), so the cap is the only thing
    left that can stop it. A test that let two brakes fire at once would pass
    with either one broken.
    """
    journal: list[str] = []
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT),
        overrides={NodeName.CRITIC: relentless_critic(journal, target_stage="insight")},
    )

    final = await asyncio.wait_for(graph.ainvoke(entry_state()), timeout=30)

    # `MAX_CRITIC_REVISIONS = 2` means two rewrites are performed, not two
    # `revise` verdicts observed: the third verdict is what the cap refuses.
    assert final["revision_count"] == SETTINGS.max_critic_revisions
    assert journal.count(str(AgentName.INSIGHT)) == SETTINGS.max_critic_revisions + 1
    # One critique per pass, plus the annotate-only final pass.
    assert journal.count(str(AgentName.CRITIC)) == SETTINGS.max_critic_revisions + 2
    assert final["report"] == {"title": "findings"}
    assert final["status"] is InvestigationStatus.COMPLETED_WITH_FINDINGS


async def test_a_critic_that_repeats_itself_stops_before_the_cap() -> None:
    """§13 brake 2: a revision that resolves nothing ends the loop immediately.

    The complement of the test above. Here the Critic returns the identical
    critique every pass, so the loop stops one revision in -- well short of
    `MAX_CRITIC_REVISIONS` -- rather than paying for the remaining rewrites to
    arrive at the same findings.
    """
    journal: list[str] = []
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT),
        critique={
            "verdict": "revise",
            "target_stage": "insight",
            "findings": [{"code": "unsupported_claim", "target_stage": "insight"}],
        },
    )

    final = await asyncio.wait_for(graph.ainvoke(entry_state()), timeout=30)

    assert final["revision_count"] == 1 < SETTINGS.max_critic_revisions
    assert journal.count(str(AgentName.INSIGHT)) == 2
    assert final["report"] == {"title": "findings"}
    assert final["status"] is InvestigationStatus.COMPLETED_WITH_FINDINGS


async def test_a_critic_that_stops_producing_critiques_cannot_re_enter_forever() -> None:
    """The counter advances even when the Critic writes nothing.

    A Critic that fails -- or returns a delta without a critique -- leaves the
    previous `revise` standing in the state, so the re-entry edge keeps firing on
    it. If the counter only moved when a *new* critique was written, the loop
    would run until the step ceiling caught it: brake 4 covering for brake 1, at
    the cost of the entire step budget.
    """
    journal: list[str] = []
    first_pass = {"critique": {"verdict": "revise", "findings": [{"target_stage": "insight"}]}}
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT),
        overrides={
            # One critique, then silence: the state keeps re-offering the same
            # `revise` verdict to the router on every subsequent pass.
            NodeName.CRITIC: agent_for(NodeName.CRITIC, journal, script=[first_pass, {}]),
        },
    )

    final = await asyncio.wait_for(graph.ainvoke(entry_state()), timeout=30)

    assert final["revision_count"] == SETTINGS.max_critic_revisions
    assert journal.count(str(AgentName.CRITIC)) == SETTINGS.max_critic_revisions + 2
    assert final["step_count"] < SETTINGS.max_steps  # brake 1 stopped it, not brake 4
    assert final["report"] == {"title": "findings"}


async def test_the_final_pass_cannot_replace_the_report_or_touch_the_counter() -> None:
    """§13's annotate-only rule, enforced structurally rather than by prompt."""
    journal: list[str] = []
    rogue = agent_for(
        NodeName.CRITIC_FINAL,
        journal,
        delta={"report": {"title": "rewritten"}, "revision_count": 99, "confidence": 0.2},
    )
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT),
        overrides={NodeName.CRITIC_FINAL: rogue},
    )

    final = await graph.ainvoke(entry_state())

    assert final["report"] == {"title": "findings"}
    assert final["revision_count"] == 0
    assert final["confidence"] == 0.2  # annotation is still allowed


async def test_the_re_entry_target_comes_from_the_finding() -> None:
    journal: list[str] = []
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT),
        overrides={NodeName.CRITIC: relentless_critic(journal, target_stage="strategy")},
    )

    await asyncio.wait_for(graph.ainvoke(entry_state()), timeout=30)

    # Strategy re-runs on every revision; Insight does not, because no finding
    # named it.
    assert journal.count(str(AgentName.STRATEGY)) == SETTINGS.max_critic_revisions + 1
    assert journal.count(str(AgentName.INSIGHT)) == 1


# --------------------------------------------------------------------------- #
# Global guards
# --------------------------------------------------------------------------- #


async def test_the_step_ceiling_halts_the_run_and_still_ships_a_report() -> None:
    journal: list[str] = []
    settings = AgentSettings(
        INVESTIGATION_MAX_STEPS=8,
        INVESTIGATION_TIMEOUT_SECONDS=1800,
        MAX_CRITIC_REVISIONS=2,
        INVESTIGATION_TOKEN_BUDGET=1_000_000,
    )
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT, AgentName.STRATEGY),
        critique={"verdict": "revise", "findings": [{"target_stage": "insight"}]},
        settings=settings,
    )

    final = await asyncio.wait_for(graph.ainvoke(entry_state()), timeout=30)

    assert final["step_count"] >= settings.max_steps
    assert journal.count(str(AgentName.CRITIC)) == 2  # the loop pass, then the final pass
    assert final["report"] == {"title": "findings"}
    assert final["status"] is InvestigationStatus.COMPLETED_WITH_FINDINGS
    assert "max_steps_exceeded" in {error.error_type for error in final["errors"]}


async def test_an_expired_deadline_halts_before_the_first_agent_call() -> None:
    journal: list[str] = []
    graph = build(journal, plan=analysis_plan(AgentName.INSIGHT))

    final = await graph.ainvoke(entry_state(deadline_at=utcnow() - timedelta(seconds=1)))

    assert journal == [str(AgentName.REPORT), str(AgentName.CRITIC)]
    assert final["report"] == {"title": "findings"}
    assert final["status"] is InvestigationStatus.COMPLETED_WITH_FINDINGS
    assert "deadline_exceeded" in {error.error_type for error in final["errors"]}


async def test_the_report_gets_real_wall_clock_after_the_deadline_that_halted_it() -> None:
    """The exit path must outlive the deadline that routed the run to it.

    §6 ships a partial report when the clock runs out, so the Report is by
    definition running after `deadline_at` has passed. Clamped to the remaining
    time it would get `MIN_NODE_TIMEOUT_SECONDS` and time out -- the guard would
    exempt it from being skipped and the timeout would kill it anyway, which is
    two protections for the exit path and no working one. This Report takes
    longer than that floor and must still finish.
    """
    journal: list[str] = []
    slow_report = agent_for(NodeName.REPORT, journal, delta={"report": {"title": "findings"}})
    slow_report.pause_seconds = MIN_NODE_TIMEOUT_SECONDS + 0.2
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT),
        overrides={NodeName.REPORT: slow_report},
    )

    final = await asyncio.wait_for(
        graph.ainvoke(entry_state(deadline_at=utcnow() - timedelta(seconds=1))), timeout=30
    )

    assert final["report"] == {"title": "findings"}
    assert not [error for error in final["errors"] if error.error_type == "timeout"]


async def test_an_exhausted_token_budget_halts_but_the_exit_path_still_runs() -> None:
    """The overdraft: the Report is allowed to spend after the budget is gone."""
    journal: list[str] = []
    settings = AgentSettings(
        INVESTIGATION_MAX_STEPS=30,
        INVESTIGATION_TIMEOUT_SECONDS=1800,
        MAX_CRITIC_REVISIONS=2,
        INVESTIGATION_TOKEN_BUDGET=1_000,
    )
    graph = build(journal, plan=analysis_plan(AgentName.INSIGHT), settings=settings)

    final = await graph.ainvoke(
        entry_state(tokens_spent=TokenLedger(input_tokens=900, output_tokens=100, calls=3))
    )

    assert journal == [str(AgentName.REPORT), str(AgentName.CRITIC)]
    assert final["report"] == {"title": "findings"}
    assert "token_budget_exhausted" in {error.error_type for error in final["errors"]}


async def test_a_cancelled_run_writes_nothing_further() -> None:
    journal: list[str] = []
    graph = build(journal, plan=analysis_plan(AgentName.INSIGHT))

    final = await graph.ainvoke(entry_state(status=InvestigationStatus.CANCELLED))

    assert journal == []
    assert final["report"] is None
    assert final["status"] is InvestigationStatus.CANCELLED


async def test_a_blocking_failure_ends_the_run_as_failed() -> None:
    """A Planner with no plan is not a degraded run; it is a failed one."""
    journal: list[str] = []
    graph = build(
        journal,
        overrides={
            NodeName.PLANNER: agent_for(
                NodeName.PLANNER,
                journal,
                error=PermanentAgentError(
                    "no plan could be produced", agent=AgentName.PLANNER, blocking=True
                ),
            )
        },
    )

    final = await graph.ainvoke(entry_state())

    assert journal == [str(AgentName.PLANNER)]
    assert final["report"] is None
    assert final["status"] is InvestigationStatus.FAILED


async def test_the_halt_reason_is_recorded_once_not_once_per_skipped_node() -> None:
    journal: list[str] = []
    graph = build(journal, plan=analysis_plan(AgentName.INSIGHT, AgentName.STRATEGY))

    final = await graph.ainvoke(entry_state(deadline_at=utcnow() - timedelta(seconds=1)))

    halts = [error for error in final["errors"] if error.error_type == "deadline_exceeded"]
    assert len(halts) == 1


# --------------------------------------------------------------------------- #
# Accounting and checkpointing
# --------------------------------------------------------------------------- #


async def test_every_node_contributes_its_spend_and_its_prompt_version() -> None:
    journal: list[str] = []
    graph = build(journal, plan=analysis_plan(AgentName.INSIGHT, AgentName.STRATEGY))

    final = await graph.ainvoke(entry_state())

    ledger = final["tokens_spent"]
    assert ledger.calls == len(journal)
    assert ledger.total_tokens > 0
    # One entry per distinct agent that ran; Graph Expansion shares the
    # Retriever's identity in these fakes, and the final Critic pass overwrites
    # the loop Critic's entry rather than adding a second one.
    assert {str(AgentName.PLANNER), str(AgentName.CRITIC), str(AgentName.REPORT)} <= set(
        final["prompt_versions"]
    )
    assert final["step_count"] == len(journal)


async def test_state_is_checkpointed_after_every_node() -> None:
    """Resumability, in the only form a unit test can assert: one write per node."""
    journal: list[str] = []
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT),
        checkpointer=memory_checkpointer(),
    )
    config = thread_config("inv-1")

    await graph.ainvoke(entry_state(), config=config)

    snapshot = await graph.aget_state(config)
    assert snapshot.values["report"] == {"title": "findings"}
    assert snapshot.next == ()

    history = [checkpoint async for checkpoint in graph.aget_state_history(config)]
    assert len(history) > len(journal)


async def test_a_resumed_run_continues_from_the_checkpoint() -> None:
    """§7: resume is the same thread with a `None` input.

    The first attempt is interrupted before the Critic; the second resumes and
    finishes without re-running anything that already completed.
    """
    journal: list[str] = []
    graph = build(
        journal,
        plan=analysis_plan(AgentName.INSIGHT),
        checkpointer=memory_checkpointer(),
    )
    graph.checkpointer = graph.checkpointer  # explicit: the saver is the one above
    config = thread_config("inv-resume")

    interrupted = build_investigation_graph(
        {
            node: agent_for(node, journal, delta={"report": {"title": "findings"}})
            if node is NodeName.REPORT
            else agent_for(node, journal)
            for node in REQUIRED_NODES
        },
        settings=SETTINGS,
        checkpointer=graph.checkpointer,
    )
    await interrupted.ainvoke(entry_state(), config=config, interrupt_before=["critic"])
    before = list(journal)
    assert str(AgentName.CRITIC) not in before

    await interrupted.ainvoke(None, config=config)
    resumed = journal[len(before) :]

    assert str(AgentName.CRITIC) in resumed
    assert str(AgentName.PLANNER) not in resumed  # completed work is not repeated
