"""The investigation `StateGraph`: `docs/agent-system.md` §2, wired.

    START -> Planner -> (Collector if the plan needs fresh data, else Retriever)
          -> Retriever -> Graph Expansion -> {Trend, Competitor, Forecast}
          -> join -> Insight -> Strategy -> Critic
          -> Report (on accept, or once the revision cap is reached)
          -> Critic final pass (annotate only) -> END

Every edge decision lives in `agents/router.py`; this module contributes the
wiring and the node *wrapper*, which is where four graph-level invariants are
enforced that no individual agent can be trusted with.

**A halted run stops spending, but still ships.** The wrapper consults the global
guard before invoking an agent. If it has fired, the node returns without calling
a model -- but `Report` and the final `Critic` pass are exempt, because §6 wants
a partial report rather than none, and a guard that suppressed the report would
turn "we ran out of time" into "we have nothing for you". Their spend is bounded
by two nodes and charged against the explicit overdraft `BaseAgent` grants the
exit path (`EXIT_PATH_OVERDRAFT`).

**No node can hang the join.** Every node runs inside a timeout derived from
`deadline_at`, and a node that exceeds it becomes an `AgentError` in the state
like any other failure. Without that, one branch stuck on an await holds the
`{Trend, Competitor, Forecast}` join open forever -- and unlike a crash, nothing
would ever report it. Together with `BaseAgent.__call__`, which converts failures
into deltas instead of raising, this is what makes "a dead branch does not hang
the join" structural.

**The revision counter is the graph's, not the Critic's.** The wrapper increments
`revision_count` whenever a critique comes back `revise`. Termination must not
depend on an agent choosing to increment a counter that ends its own loop; a
Critic with a prompt regression would otherwise cycle until the step ceiling
caught it, having spent the entire budget on rewrites.

**`Insight` is a deferred node.** The join must wait for every dispatched branch,
including the case where the plan serialises Forecast behind Trend and one path
is a superstep longer than the others. LangGraph's `defer=True` holds the node
until nothing else is pending, which is precisely the barrier §2's join calls
for; without it, an uneven fan-out schedules Insight twice and the second run
overwrites the first's reasoning.

The graph is built from *supplied* node callables rather than constructing agents
itself. Composition belongs to `services/investigation_service.py`, which owns
the provider, the tool registry and the tenant scoping -- and a graph that
constructed its own agents could not be tested with fakes, which is the only way
this file is testable at all.
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Final, Protocol, get_args, get_origin, get_type_hints

from langgraph.graph import END, START, StateGraph

from agents.errors import AgentTimeoutError, to_state_error
from agents.router import (
    NODE_AGENT,
    TERMINAL_NODES,
    Halt,
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
    route_after_trend,
    terminal_status,
)
from agents.state import AgentError, InvestigationState, TokenLedger
from backend.core.config import AgentSettings, get_settings
from models.base import utcnow
from models.enums import InvestigationStatus

__all__ = [
    "APPEND_REDUCED_KEYS",
    "CONCURRENT_NODES",
    "EXIT_PATH_GRACE_FRACTION",
    "MIN_NODE_TIMEOUT_SECONDS",
    "REQUIRED_NODES",
    "NodeCallable",
    "build_investigation_graph",
    "initial_status_delta",
    "unavailable_branches",
    "wrap_node",
]


class NodeCallable(Protocol):
    """What a node is, from the graph's point of view.

    `BaseAgent` instances satisfy it via `__call__`; Graph Expansion is a plain
    async function, because §2 is explicit that it is a node without being an
    agent -- it makes no model call and has no prompt to version.

    A `Protocol` rather than the obvious `Callable[[InvestigationState], ...]`
    alias, and the parameter name is load-bearing. LangGraph's own node protocol
    declares `__call__(self, state)`, so it may invoke a node with `state=` as a
    keyword; a `Callable[...]` alias declares its parameters positional-only and
    therefore does *not* satisfy that protocol. Spelled as an alias, every
    `add_node` call in this module is a type error -- which, since the runtime
    accepts it either way, would be silenced with an ignore comment and the real
    constraint (name your node's parameter `state`) would go unrecorded.
    """

    def __call__(self, state: InvestigationState) -> Awaitable[dict[str, Any]]: ...


def _append_reduced_keys() -> frozenset[str]:
    """State keys whose reducer appends a list, read off `agents/state.py` itself.

    Derived rather than hand-listed, because `_merged()` has to replay those
    reducers to build the view the guards read, and a hand-listed set is exactly
    what goes stale when a new fan-out key is added. The stale case is silent and
    one-sided: the new key alone would be replaced instead of appended in that
    view, so a guard reading it would see one node's increment where the run's
    whole history should be.

    `step_count` also reduces with `operator.add` and is deliberately excluded --
    it is an `int`, not a list, and `_merged()` sums it separately.
    """
    keys: set[str] = set()
    for field, annotation in get_type_hints(InvestigationState, include_extras=True).items():
        args = get_args(annotation)
        if len(args) > 1 and operator.add in args[1:] and get_origin(args[0]) is list:
            keys.add(field)
    return frozenset(keys)


APPEND_REDUCED_KEYS: Final[frozenset[str]] = _append_reduced_keys()
"""The state's append-reduced list keys. See `_append_reduced_keys`."""

REQUIRED_NODES: Final[tuple[NodeName, ...]] = tuple(NodeName)
"""Every node the topology names. All of them must be supplied.

No defaults and no placeholders: a graph that quietly substituted a no-op for a
missing Critic would run, produce a report, and have skipped the only step that
checks whether the report is true.
"""

CONCURRENT_NODES: Final[frozenset[NodeName]] = frozenset(
    {NodeName.TREND, NodeName.COMPETITOR, NodeName.FORECAST}
)
"""The fan-out branches -- the only nodes that can execute concurrently.

They are excluded from every scalar write the wrapper makes (`status`,
`revision_count`). `agents/state.py` is unambiguous: a scalar written by
concurrent nodes is a lost update that appears only under load, which is the
worst class of bug this system could have.
"""

MIN_NODE_TIMEOUT_SECONDS: Final = 1.0
"""Floor under the per-node timeout.

A node handed 0.01 seconds cannot finish, so the run would burn its remaining
nodes on timeouts and report nothing but timeouts. One second is enough for a
node whose work is already cached or trivially short, and the deadline guard
stops the run at the next edge regardless.
"""

EXIT_PATH_GRACE_FRACTION: Final = 0.1
"""Share of `INVESTIGATION_TIMEOUT_SECONDS` the exit path keeps past the deadline.

The wall-clock twin of `agents/base.py`'s `EXIT_PATH_OVERDRAFT`, and needed for
the same reason: §6 ships a partial report when the deadline passes, so the two
nodes that produce it are running *after* the moment their remaining time went
negative. Ten percent of a thirty-minute run is three minutes across two nodes --
enough to write and check a report, small enough that a wedged Report cannot
double the run it belongs to.
"""


# --------------------------------------------------------------------------- #
# The node wrapper
# --------------------------------------------------------------------------- #


def wrap_node(
    name: NodeName,
    node: NodeCallable,
    *,
    settings: AgentSettings | None = None,
) -> NodeCallable:
    """Wrap one node callable with the graph-level invariants.

    Exported because the invariants are worth testing on a single node, without
    building and driving a whole graph to reach them.
    """
    resolved = settings if settings is not None else get_settings().agents
    agent = NODE_AGENT[name]

    async def run(state: InvestigationState) -> dict[str, Any]:
        halt = check_guards(state, settings=resolved)
        if halt is not None and not _is_exempt(name, halt):
            return _halt_delta(name, halt, state)

        slice_seconds = _node_timeout(name, state, resolved)
        try:
            async with asyncio.timeout(slice_seconds):
                delta = dict(await node(state))
        except TimeoutError as exc:
            # Not re-raised: a branch that timed out must still complete its
            # superstep, or the join it feeds waits on a task that will never
            # arrive. The error is what Insight and the Report read to know the
            # branch is missing rather than empty.
            delta = {
                "errors": [
                    to_state_error(
                        AgentTimeoutError(
                            f"{name} exceeded its {slice_seconds:.0f}s slice.",
                            agent=agent,
                            cause=exc,
                        ),
                        agent=agent,
                    )
                ],
                "step_count": 1,
            }
        except Exception as exc:  # a node that is not a BaseAgent can still raise
            delta = {"errors": [to_state_error(exc, agent=agent)], "step_count": 1}

        # `setdefault`, not `+= 1`: a `BaseAgent` already counted itself, and
        # counting it twice would halve the effective step ceiling. Graph
        # Expansion is not an agent and counts nothing, and a node the guard
        # cannot see is a node that can loop without moving the ceiling.
        delta.setdefault("step_count", 1)

        if halt is not None:
            # The exit path ran *despite* a fired guard. Nothing else will record
            # why: `_halt_delta` only fires for a node the guard skipped, and on a
            # halt that ships a report every remaining node is exempt -- so
            # without this the run would truncate itself and the report's gaps
            # section would have nothing to say about it. Merged rather than
            # assigned, because the node may have recorded failures of its own.
            delta = _record_halt(name, halt, delta, state)

        if name is NodeName.CRITIC:
            delta = _enforce_revision_progress(delta, state)
        elif name is NodeName.CRITIC_FINAL:
            delta = _annotate_only(delta)

        return _stamp_status(name, delta, state, resolved)

    run.__name__ = f"node_{name}"
    return run


def _is_exempt(name: NodeName, halt: Halt) -> bool:
    """Whether this node still runs despite a fired guard.

    Only the exit path, and only for a halt that ships a report. A cancellation
    or a blocking error routes to `END`, and exempting the Report from *those*
    would spend a model call writing up a run the user asked to stop -- or one
    whose plan never existed.
    """
    return name in TERMINAL_NODES and halt.route != END


def _halt_delta(name: NodeName, halt: Halt, state: InvestigationState) -> dict[str, Any]:
    """What a skipped node returns once a guard has fired.

    No `step_count`: the node did no work, and counting a skip would make the
    step ceiling appear to be the cause of skips it was itself the cause of. The
    halt is recorded once -- keyed on `error_type` -- because the run traverses
    several skipped nodes on its way to the Report and a gaps section listing
    "deadline exceeded" five times reads as five separate problems.
    """
    delta: dict[str, Any] = {}
    if not _already_recorded(state, str(halt.reason)):
        delta["errors"] = [halt_error(halt, agent=NODE_AGENT[name])]
    if name not in CONCURRENT_NODES and halt.status.is_terminal and halt.route == END:
        # Only when the run is actually ending here. A halt that routes to the
        # Report is not terminal yet -- the final Critic pass decides that, and
        # stamping it now would make `is_terminal` true while two nodes still
        # have to run.
        delta["status"] = halt.status
    return delta


def _record_halt(
    name: NodeName,
    halt: Halt,
    delta: dict[str, Any],
    state: InvestigationState,
) -> dict[str, Any]:
    """Append the halt record to a node that ran anyway, if nothing recorded it yet.

    Keyed on `error_type` for the same reason `_halt_delta` is: the Report and
    the final Critic pass both traverse a fired guard, and a gaps section that
    said "deadline exceeded" once per exempt node would read as several distinct
    problems rather than one.
    """
    if _already_recorded(state, str(halt.reason)):
        return delta
    recorded = halt_error(halt, agent=NODE_AGENT[name])
    return {**delta, "errors": [*delta.get("errors", []), recorded]}


def _already_recorded(state: InvestigationState, error_type: str) -> bool:
    return any(error.error_type == error_type for error in state.get("errors", []))


def _node_timeout(name: NodeName, state: InvestigationState, settings: AgentSettings) -> float:
    """How long this node gets: whatever is left of the investigation, capped.

    Derived from `deadline_at` rather than a fixed per-node constant so that a
    node cannot outlive the run that owns it. A fixed constant large enough for
    the Report is also large enough for a wedged Collector to blow straight
    through the deadline.

    The exit path is the documented exception, and it has to be. A deadline halt
    routes to the Report *because* the deadline passed, so by the time the Report
    runs the remaining time is negative and the clamp would hand it the one-second
    floor -- a slice no long-form synthesis finishes in. The run would then halt,
    exempt the Report from the guard exactly as §6 requires, and kill it on the
    clock anyway, ending with nothing to show. It would also make the token
    overdraft `BaseAgent` grants the same two nodes unspendable, which is how you
    end up with two mechanisms protecting the exit path and neither working.
    """
    ceiling = float(settings.timeout_seconds)
    deadline = state.get("deadline_at")
    if deadline is not None:
        remaining = (deadline - utcnow()).total_seconds()
        if name in TERMINAL_NODES:
            remaining = max(remaining, _exit_path_grace(settings))
        ceiling = min(ceiling, remaining)
    return max(MIN_NODE_TIMEOUT_SECONDS, ceiling)


def _exit_path_grace(settings: AgentSettings) -> float:
    """Wall-clock the Report and the final Critic pass keep after the deadline.

    A fraction of the run's own timeout rather than a constant, so a two-minute
    interactive query and a thirty-minute deep investigation are each granted an
    exit proportional to what they were budgeted -- and so the grace can never be
    a large multiple of the run it is attached to. Bounded and charged to two
    nodes, which is what makes it a grace rather than a second deadline.
    """
    return max(MIN_NODE_TIMEOUT_SECONDS, settings.timeout_seconds * EXIT_PATH_GRACE_FRACTION)


def _enforce_revision_progress(delta: dict[str, Any], state: InvestigationState) -> dict[str, Any]:
    """Advance the revision counter once per revision the router actually dispatched.

    This is the structural half of §13's first brake. `agents/router.py` caps the
    loop at `MAX_CRITIC_REVISIONS`, but a cap on a counter nobody increments is
    not a cap; the counter has to be the graph's responsibility because the graph
    is what the loop is made of.

    It counts on *entry*, from the critique already in the state, rather than on
    the verdict this pass just produced. Two things follow, and both are the
    reason for the indirection:

    - `MAX_CRITIC_REVISIONS = 2` permits two rewrites. Counting the outgoing
      verdict would spend the second count on a verdict that was never acted on,
      so the cap would bite one rewrite early -- an off-by-one nobody would see
      except as a Critic loop that gives up sooner than its configuration says.
    - A Critic that *fails* still advances the counter. Otherwise a node erroring
      on every pass leaves the previous `revise` critique standing in the state,
      the router re-enters on it forever, and only the step ceiling ever stops the
      run -- which is brake 4 catching what brake 1 was supposed to.

    The critique is also mirrored into `critique_history` when the Critic wrote
    one without the other, since the monotonic-progress and artifact-hash brakes
    read history rather than the latest critique alone.
    """
    updated = dict(delta)

    critique = delta.get("critique")
    if critique is not None and "critique_history" not in updated:
        updated["critique_history"] = [critique]

    if _verdict_of(state.get("critique")) == "revise":
        floor = state.get("revision_count", 0) + 1
        updated["revision_count"] = max(int(updated.get("revision_count", 0)), floor)
    return updated


def _verdict_of(critique: Any) -> str:
    """The verdict of a critique payload, tolerant of a missing or malformed one.

    Lowercased and defaulted to the empty string rather than to `accept`: an
    unreadable critique must not be read as permission to leave the loop, and it
    must not be read as an instruction to re-enter it either.
    """
    if not isinstance(critique, dict):
        return ""
    return str(critique.get("verdict", "")).lower()


def _annotate_only(delta: dict[str, Any]) -> dict[str, Any]:
    """Strip anything the final Critic pass is not allowed to change (§13).

    The pass may lower confidence and add findings; it may not touch the
    revision counter or replace the report. Enforced here rather than trusted to
    the prompt, because `Report -> Critic -> Report` would be a second unbounded
    cycle and, unlike the revision loop, it has no counter to stop it.
    """
    return {key: value for key, value in delta.items() if key not in ("revision_count", "report")}


def _stamp_status(
    name: NodeName,
    delta: dict[str, Any],
    state: InvestigationState,
    settings: AgentSettings,
) -> dict[str, Any]:
    """Write `status`, but only from a node that cannot be running concurrently.

    `status` is a single-writer scalar. The fan-out branches are therefore
    silent about it -- three branches stamping `running` in one superstep is the
    lost-update race `agents/state.py` warns about, and it would be invisible
    because all three write the same value today and different values the moment
    someone adds a phase.
    """
    if name in CONCURRENT_NODES or "status" in delta:
        return delta

    after = _merged(state, delta)
    if name is NodeName.CRITIC_FINAL:
        return {**delta, "status": terminal_status(after, settings=settings)}

    halt = check_guards(after, settings=settings)
    if halt is not None and halt.route == END:
        # This node's own delta ended the run -- a blocking error, typically, and
        # the Planner's is the one that matters: no plan means the edge routes
        # straight to `END` and no later node exists to stamp the outcome. Left to
        # `phase_status` the run would come to rest reporting `planning`, which
        # the API would serve as an investigation still in progress forever.
        # Only for `END`: a halt that ships a report is not terminal yet, and
        # `is_terminal` must not be true while two nodes still have to run.
        return {**delta, "status": halt.status}
    return {**delta, "status": phase_status(name)}


def _merged(state: InvestigationState, delta: dict[str, Any]) -> InvestigationState:
    """A read-only view of the state as it will be *after* this delta lands.

    Needed because both `check_guards()` and `terminal_status()` have to see the
    error this node just recorded and the step it just spent, and LangGraph does
    not apply the reducers until the node has returned. The reducer-bearing keys
    are therefore merged here the way their reducers would merge them -- anything
    looser, a plain `{**state, **delta}`, would let a node's one-step increment
    overwrite the run's total step count, and the guard that reads it would then
    see a run that had executed one step.
    """
    merged: dict[str, Any] = {**state}
    for key, value in delta.items():
        # Read the accumulator out of `merged`, not out of `state`: a `TypedDict`
        # indexed by a variable key answers `object`, which is not iterable, and
        # every key is visited at most once so the two are the same value anyway.
        if key in APPEND_REDUCED_KEYS:
            existing: list[Any] = merged.get(key) or []
            merged[key] = [*existing, *value]
        elif key == "step_count":
            merged[key] = int(merged.get("step_count", 0)) + int(value)
        elif key == "tokens_spent":
            ledger: TokenLedger = merged.get("tokens_spent") or TokenLedger()
            merged[key] = ledger.plus(value)
        else:
            merged[key] = value
    return merged  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def build_investigation_graph(
    nodes: Mapping[NodeName | str, NodeCallable],
    *,
    checkpointer: Any | None = None,
    settings: AgentSettings | None = None,
    wrap: bool = True,
) -> Any:
    """Compile the investigation graph from the supplied node callables.

    `wrap=False` exists for tests that drive raw node behaviour; production must
    leave it on, since the guard, the node timeout and the revision counter all
    live in the wrapper.

    `checkpointer=None` compiles a graph with no durability -- fine for a unit
    test, and what `AGENT_CHECKPOINT_ENABLED=false` produces. `agents/
    checkpointer.py` is where that choice is made; taking the saver as an
    argument here keeps this function free of any database import, which is what
    lets the topology be tested in a process that has no Postgres driver
    installed.
    """
    resolved = settings if settings is not None else get_settings().agents
    supplied = {NodeName(str(key)): value for key, value in nodes.items()}
    missing = [node for node in REQUIRED_NODES if node not in supplied]
    if missing:
        raise ValueError(
            "cannot build the investigation graph: no callable supplied for "
            f"{', '.join(str(node) for node in missing)}. Every node in "
            "docs/agent-system.md §2 must be provided -- a substituted no-op would "
            "produce a run that silently skipped a stage."
        )

    def bind(predicate: Callable[..., Any]) -> Callable[[InvestigationState], Any]:
        """Freeze this graph's settings into an edge predicate.

        LangGraph calls an edge function with the state alone, so a predicate
        that read `get_settings()` internally would ignore whatever limits this
        graph was built with -- and a test that lowered `MAX_CRITIC_REVISIONS` to
        one would silently keep testing the default of two.
        """

        def bound(state: InvestigationState) -> Any:
            return predicate(state, settings=resolved)

        bound.__name__ = predicate.__name__
        return bound

    builder: StateGraph[InvestigationState, Any, Any, Any] = StateGraph(InvestigationState)
    for node in REQUIRED_NODES:
        builder.add_node(
            str(node),
            wrap_node(node, supplied[node], settings=resolved) if wrap else supplied[node],
            # The join. See the module docstring: an uneven fan-out (Forecast
            # serialised behind Trend) would otherwise schedule Insight twice.
            defer=node is NodeName.INSIGHT,
        )

    builder.add_edge(START, str(NodeName.PLANNER))

    # Every transition below is a conditional edge, even the ones §2's table
    # calls static, so that `agents/router.py` remains the only place a
    # transition is decided. A static edge here would be a second, invisible
    # routing rule that no router test could ever cover -- and the "static"
    # edges are exactly the ones that need to observe a deadline that expired
    # during a long connector sync.
    builder.add_conditional_edges(
        str(NodeName.PLANNER),
        bind(route_after_planner),
        [str(NodeName.COLLECTOR), str(NodeName.RETRIEVER), str(NodeName.REPORT), END],
    )
    builder.add_conditional_edges(
        str(NodeName.COLLECTOR),
        bind(route_after_collector),
        [str(NodeName.RETRIEVER), str(NodeName.REPORT), END],
    )
    builder.add_conditional_edges(
        str(NodeName.RETRIEVER),
        bind(route_after_retriever),
        [str(NodeName.GRAPH_EXPANSION), str(NodeName.REPORT), END],
    )
    builder.add_conditional_edges(
        str(NodeName.GRAPH_EXPANSION),
        bind(dispatch_analysis),
        [
            str(NodeName.TREND),
            str(NodeName.COMPETITOR),
            str(NodeName.FORECAST),
            str(NodeName.INSIGHT),
            str(NodeName.REPORT),
            END,
        ],
    )
    # The only conditional edge inside the fan-out, and it has exactly two
    # destinations: Forecast when the plan serialised it behind Trend, otherwise
    # the join. Competitor and Forecast take unconditional edges to the join --
    # a guard-aware conditional on a concurrent branch could send two branches
    # to two different nodes in the same superstep and split one join into two.
    builder.add_conditional_edges(
        str(NodeName.TREND),
        bind(route_after_trend),
        [str(NodeName.FORECAST), str(NodeName.INSIGHT)],
    )
    builder.add_edge(str(NodeName.COMPETITOR), str(NodeName.INSIGHT))
    builder.add_edge(str(NodeName.FORECAST), str(NodeName.INSIGHT))

    builder.add_conditional_edges(
        str(NodeName.INSIGHT),
        bind(route_after_insight),
        [str(NodeName.STRATEGY), str(NodeName.REPORT), END],
    )
    builder.add_conditional_edges(
        str(NodeName.STRATEGY),
        bind(route_after_strategy),
        [str(NodeName.CRITIC), str(NodeName.REPORT), END],
    )
    builder.add_conditional_edges(
        str(NodeName.CRITIC),
        bind(route_after_critic),
        [
            str(NodeName.RETRIEVER),
            str(NodeName.INSIGHT),
            str(NodeName.STRATEGY),
            str(NodeName.REPORT),
            END,
        ],
    )
    builder.add_conditional_edges(
        str(NodeName.REPORT), bind(route_after_report), [str(NodeName.CRITIC_FINAL)]
    )
    # One destination, and it is `END`. The final pass annotates; if this ever
    # gained a second destination the graph would contain a cycle with no
    # counter on it.
    builder.add_conditional_edges(str(NodeName.CRITIC_FINAL), bind(route_after_final_critic), [END])

    return builder.compile(checkpointer=checkpointer)


def initial_status_delta() -> dict[str, Any]:
    """The status a run carries the moment it enters the graph.

    Separate from `new_state()` because `agents/state.py` builds the entry state
    for the *service* layer, which may enqueue a run long before a worker picks
    it up: a run that reported `planning` while sitting in a queue would make
    every latency measurement of the Planner wrong.
    """
    return {"status": InvestigationStatus.RUNNING}


def unavailable_branches(state: InvestigationState) -> list[AgentError]:
    """The fan-out branches that failed, for Insight and the Report to read.

    §6: "the join records the branch as unavailable and Insight proceeds with
    fewer inputs, at lower confidence". This is how a downstream node tells
    "there is no competitor view because nothing was found" from "there is no
    competitor view because the branch died" -- two facts that look identical in
    the state and mean opposite things in a report.
    """
    branch_agents = {NODE_AGENT[node] for node in CONCURRENT_NODES}
    return [error for error in state.get("errors", []) if error.agent in branch_agents]
