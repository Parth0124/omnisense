"""Planner input and output schemas.

The Planner is the only agent whose output the *router* reads structurally rather
than passing along: `agents/router.py` branches on `plan[].agent` and
`requires_fresh_data`, and `check_guards` counts remaining steps from the plan's
length. That makes these schemas load-bearing in a way most agent output is not
-- a malformed plan does not produce a bad answer, it produces a run that visits
the wrong nodes or never terminates.

So the constraints here are tighter than they look. Every one of them exists
because the unconstrained version has a specific failure:

* `max_length` on `steps` -- the model will happily emit a forty-step plan for a
  three-sentence question, and every step is a node visit against the run's step
  ceiling. The ceiling is then hit mid-analysis, and the run reports a partial
  answer for a question that needed four steps.
* `min_length=1` on `steps` -- an empty plan routes straight to the Report with
  no evidence, and the Report writes a fluent document about nothing.
* Unique `id` -- `depends_on` refers to steps by id, and duplicates make the
  dependency graph ambiguous in a way that silently drops a step.
* `depends_on` must reference declared ids -- a dangling reference is a step
  that waits forever, which presents as a hung run rather than a bad plan.

`StrictModel` (`extra="forbid"`) because this is *producer* validation: the model
is the producer, and a field it invented is a hallucinated instruction we should
refuse rather than ignore.
"""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

from models.base import StrictModel
from models.enums import AgentName

__all__ = [
    "MAX_PLAN_STEPS",
    "MAX_SUB_QUESTIONS",
    "PlannedStep",
    "PlannerInput",
    "PlannerOutput",
    "PlannedSubQuestion",
]

MAX_PLAN_STEPS = 12
"""Ceiling on plan length.

Twelve is the number of analysis nodes plus a margin: a plan longer than the
graph has nodes is describing work the graph cannot route. Beyond that it is
also a budget statement -- every step is a node visit, and the step ceiling in
`agents/router.py` is what stops a run costing unboundedly.
"""

MAX_SUB_QUESTIONS = 8
"""Ceiling on decomposition.

The Critic checks coverage against these, and coverage of eight sub-questions is
already a demanding bar for one investigation. Twenty would guarantee the Critic
reports the report as incomplete, every time, which trains whoever reads it to
ignore the finding.
"""


class PlannedStep(StrictModel):
    """One step the model proposes.

    Deliberately not `agents.state.PlanStep`. That type is the *state's* shape and
    carries fields the model must not set; this is the model's shape, and
    `PlannerAgent.to_delta` converts one into the other. Letting the model write
    the state type directly would mean any field added to the state for internal
    bookkeeping instantly becomes something a prompt can set.
    """

    id: str = Field(
        min_length=1,
        max_length=40,
        description="Short stable identifier, referenced by depends_on.",
    )
    description: str = Field(min_length=1, max_length=500)
    agent: AgentName = Field(description="Which agent performs this step.")
    requires_fresh_data: bool = Field(
        default=False,
        description=(
            "Whether this step needs a connector sync. Read by the router to "
            "decide whether the run enters the Collector at all -- a sync costs "
            "minutes and a quota, so the decision is made here rather than after "
            "the run has already paid for it."
        ),
    )
    depends_on: list[str] = Field(default_factory=list, max_length=MAX_PLAN_STEPS)
    rationale: str | None = Field(default=None, max_length=500)

    @field_validator("agent")
    @classmethod
    def _reject_unroutable_agents(cls, value: AgentName) -> AgentName:
        """`AgentName` is tolerant; a plan is not the place for that tolerance.

        `AgentName("plannner")` decodes to `UNKNOWN` rather than raising, which is
        right when *reading* a checkpoint written by a newer build. Here it would
        put a step in the plan that the router cannot dispatch, so the run would
        either skip it silently or halt -- both of which look like a routing bug
        rather than a typo the model made.
        """
        if value is AgentName.UNKNOWN:
            raise ValueError(
                "a plan step must name a routable agent; UNKNOWN means the name "
                "was not recognised, and the router has nowhere to send it"
            )
        if value is AgentName.PLANNER:
            raise ValueError(
                "the Planner cannot schedule itself; a self-referential plan is "
                "how a run loops without ever producing evidence"
            )
        return value


class PlannedSubQuestion(StrictModel):
    """A question the evidence must answer for the report to be complete."""

    id: str = Field(min_length=1, max_length=40)
    question: str = Field(min_length=1, max_length=400)


class PlannerInput(StrictModel):
    """The projection of the state the Planner needs.

    Small on purpose. The Planner runs first, so almost nothing exists yet -- and
    passing the whole state would let a future field silently start influencing
    the plan without anyone deciding it should.
    """

    query: str = Field(min_length=1)
    tenant_id: str
    available_connectors: list[str] = Field(default_factory=list, max_length=64)
    known_entities: list[str] = Field(default_factory=list, max_length=32)
    seconds_remaining: float | None = None
    """Wall clock left. A plan that cannot finish in the time available is worse
    than a shorter plan, because it burns the budget before reaching the Report."""


class PlannerOutput(StrictModel):
    """The plan, as the model produces it."""

    objective: str = Field(
        min_length=1,
        max_length=1000,
        description="One sentence restating what this investigation must establish.",
    )
    steps: list[PlannedStep] = Field(min_length=1, max_length=MAX_PLAN_STEPS)
    sub_questions: list[PlannedSubQuestion] = Field(
        default_factory=list, max_length=MAX_SUB_QUESTIONS
    )
    reasoning: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _check_step_graph(self) -> PlannerOutput:
        """Reject a plan whose dependency graph cannot be executed.

        Three failures, each of which presents as something other than what it is:

        *Duplicate ids* make `depends_on` ambiguous, so a step is dropped and the
        run produces a partial answer with no error.

        *Dangling dependencies* make a step wait for something that will never
        complete -- a hung run, diagnosed as a deadlock in the graph engine.

        *Cycles* are the same failure with a more confusing trace.

        Checked here rather than in the router because the router's job is to
        dispatch, and a router that has to validate is a router that has to
        decide what to do with an invalid plan mid-run. Failing at parse time
        means the Planner retries -- which is the recoverable path.
        """
        ids = [step.id for step in self.steps]
        duplicates = sorted({identifier for identifier in ids if ids.count(identifier) > 1})
        if duplicates:
            raise ValueError(f"duplicate step ids {duplicates}; depends_on would be ambiguous")

        declared = set(ids)
        for step in self.steps:
            dangling = sorted(set(step.depends_on) - declared)
            if dangling:
                raise ValueError(
                    f"step {step.id!r} depends on undeclared step(s) {dangling}; "
                    "it would wait forever"
                )
            if step.id in step.depends_on:
                raise ValueError(f"step {step.id!r} depends on itself")

        self._reject_cycles()

        question_ids = [question.id for question in self.sub_questions]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("duplicate sub-question ids; coverage would be miscounted")
        return self

    def _reject_cycles(self) -> None:
        """Depth-first cycle detection over `depends_on`."""
        edges = {step.id: list(step.depends_on) for step in self.steps}
        visiting: set[str] = set()
        done: set[str] = set()

        def walk(node: str, trail: list[str]) -> None:
            if node in done:
                return
            if node in visiting:
                cycle = " -> ".join([*trail, node])
                raise ValueError(f"plan contains a dependency cycle: {cycle}")
            visiting.add(node)
            for dependency in edges.get(node, ()):
                walk(dependency, [*trail, node])
            visiting.discard(node)
            done.add(node)

        for step_id in edges:
            walk(step_id, [])

    @property
    def requires_collection(self) -> bool:
        """Whether any step needs fresh data. Read by the router."""
        return any(step.requires_fresh_data for step in self.steps)
