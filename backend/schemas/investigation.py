"""Wire DTOs for `POST /investigations` and `GET /investigations/{id}`.

`docs/api-reference.md` §4.1 and §4.2 are the contract. Two places where this
module deliberately does **not** match the document are called out here rather
than buried, because both are real divergences a client will notice and neither
is fixable at the schema layer.

**1. The state vocabulary is the persisted one, not the document's.**
§4.2 draws a state machine over `queued -> planning -> collecting -> retrieving ->
analyzing -> reflecting -> reporting -> completed`, plus `timed_out`. The column
that holds this value is `investigations.status`, typed
`TolerantEnumType(InvestigationStatus)` (`models/orm/investigation.py`), and
`models/enums.py::InvestigationStatus` has a coarser vocabulary: `queued`,
`planning`, `running`, `reflecting`, `completed`, `completed_with_findings`,
`failed`, `cancelled`. There is nowhere to store `collecting` as distinct from
`retrieving`, and `timed_out` is recorded as `failed` with a reason.

Inventing a mapping here would be worse than the divergence. Reporting
`retrieving` for a row that says `running` would make the API assert something
the database does not know, and `GET` would contradict the SSE stream, the
orchestrator's own checkpoint and the reaper. So the wire carries the stored
value. §1's versioning policy explicitly permits this direction -- *"Adding a new
enum member to a response field: yes, clients must tolerate unknown members"* --
and `completed_with_findings` is a state clients genuinely need
(`docs/agent-system.md` §13: the report ships with unresolved Critic findings
surfaced rather than withheld). Closing the gap properly means finer statuses in
`models/enums.py` and a migration, not a translation table in a DTO.

**2. Several documented counters have no column and are reported as `null`.**
`counts.signals_considered`, `counts.evidence`, `usage.tool_calls`, and a step's
`evidence_count` and `tool_calls` are all in §4.2's example. None of them exists:
there is no `evidence` table in `models/orm/`, `investigation_steps` has no
tool-call column, and nothing records how many Signals a retrieval pass looked at.

`null` and not `0`. This repository already takes that position where it matters
-- `Engagement.compute_score` returns `None` rather than `0.0` because "nobody
engaged" and "we did not measure" are different claims, and a dashboard cannot
tell them apart once the second has been rendered as the first. A `0` here would
show every investigation as having considered no evidence.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from backend.schemas.common import Page, RequestModel, ResponseModel
from models.enums import AgentName, InvestigationStatus
from models.orm.investigation import StepStatus

__all__ = [
    "MAX_METADATA_KEYS",
    "MAX_METADATA_VALUE_CHARS",
    "MAX_QUERY_CHARS",
    "CreateInvestigationRequest",
    "InvestigationBudget",
    "InvestigationCounts",
    "InvestigationCreated",
    "InvestigationDepth",
    "InvestigationDetail",
    "InvestigationError",
    "InvestigationLinks",
    "InvestigationProgress",
    "InvestigationScope",
    "InvestigationUsage",
    "StepItem",
    "StepsPage",
    "TimeWindow",
]

MAX_QUERY_CHARS = 2000
"""§4.1: "1-2000 chars".

Restated from `services/investigation_service.py`, which enforces the same bound
at the service boundary. Both are needed: this one gives the caller a `422`
naming the field, and the service's protects the `Text` column from a caller that
never came through HTTP.
"""

MAX_METADATA_KEYS = 16
MAX_METADATA_VALUE_CHARS = 256
"""§4.1: "<=16 keys, string values <=256 chars; echoed back verbatim".

Bounded because it *is* echoed back verbatim, which makes it the one field a
caller controls end to end. Unbounded, it is free storage attached to every
investigation and a reflection surface in every response built from one.
"""


class InvestigationDepth(enum.StrEnum):
    """Budget preset from §4.1. Not persisted -- see `CreateInvestigationRequest`."""

    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class TimeWindow(RequestModel):
    """Half-open bound on signal timestamps: `[from, to)`.

    `from` is a Python keyword, so the field is `from_` with an alias. The
    half-open convention is inherited from `services/signal_service.py`, where it
    is load-bearing: consecutive daily windows tile without overlap, so a Signal
    sitting exactly on midnight is neither double-counted nor missed, and trend
    volume is computed from those counts.
    """

    from_: datetime | None = Field(default=None, alias="from")
    to: datetime | None = None

    @field_validator("from_", "to")
    @classmethod
    def _must_be_aware(cls, value: datetime | None) -> datetime | None:
        """§3.2: "Naive datetimes are rejected with 422".

        Enforced rather than defaulted because the column being compared against
        is `TIMESTAMP WITH TIME ZONE`: a naive bound is compared against whatever
        timezone the driver assumed, which silently shifts the window by hours and
        produces a plausible, wrong answer.
        """
        if value is not None and value.tzinfo is None:
            raise ValueError("must be timezone-aware, e.g. 2026-05-01T00:00:00Z")
        return value

    @model_validator(mode="after")
    def _ordered(self) -> TimeWindow:
        if self.from_ is not None and self.to is not None and self.from_ > self.to:
            raise ValueError("`from` is later than `to`; the window is empty")
        return self


class InvestigationScope(RequestModel):
    """What the investigation is allowed to look at (§4.1).

    Every list defaults to empty, and empty means *unrestricted* rather than
    *nothing*: "empty means every enabled connector", "empty means all". That
    reading is stated in §4.1 for `platforms` and `languages`, and it is the only
    coherent one -- a default that matched nothing would make the simplest
    possible request return no evidence.
    """

    platforms: list[str] = Field(default_factory=list, max_length=64)
    entities: list[str] = Field(default_factory=list, max_length=64)
    languages: list[str] = Field(default_factory=list, max_length=32)
    time_window: TimeWindow = Field(default_factory=TimeWindow)


class InvestigationBudget(RequestModel):
    """Hard ceilings on spend for one run (§4.1). `None` means the deployment default."""

    max_tokens: int | None = Field(default=None, ge=1000)
    max_tool_calls: int | None = Field(default=None, ge=1)


class CreateInvestigationRequest(RequestModel):
    """The `POST /investigations` body.

    Most of this is accepted and validated but not yet *persisted*, and the
    reason is structural rather than an omission: `investigations` has columns for
    the question, the plan, the status and the accounting, and none for scope,
    depth, budget, callback or metadata (`models/orm/investigation.py`). Those
    belong to the orchestrator's run configuration, and the orchestrator
    (`agents/graph.py` and the worker that drives it) is what will consume them.

    Validating them here anyway is the point of the endpoint being a contract:
    a caller that sends `from` after `to`, a naive datetime, seventeen metadata
    keys or `max_steps=0` gets the documented `422` today, and the day the
    orchestrator reads these fields nothing about the request changes.
    """

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    objective: str | None = Field(default=None, max_length=2000)
    depth: InvestigationDepth = InvestigationDepth.STANDARD
    scope: InvestigationScope = Field(default_factory=InvestigationScope)
    refresh_connectors: bool = False
    max_steps: int | None = Field(default=None, ge=1, le=200)
    budget: InvestigationBudget = Field(default_factory=InvestigationBudget)
    callback_url: str | None = Field(default=None, max_length=2048)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def _query_is_not_blank(cls, value: str) -> str:
        """A whitespace-only question is an empty question.

        `RequestModel` strips strings, but relying on the strip alone couples this
        rule to Pydantic's field-processing order. Checking the stripped value
        explicitly makes the outcome independent of it.
        """
        if not value.strip():
            raise ValueError("an investigation needs a question")
        return value

    @field_validator("callback_url")
    @classmethod
    def _https_only(cls, value: str | None) -> str | None:
        """§4.1: "HTTPS only".

        A webhook carries the investigation's outcome to a URL the caller chose,
        so plain HTTP would put that outcome on the wire in clear text to a host
        this system does not authenticate. Rejected at the edge because the field
        is stored and used later, by a process with no caller left to ask.
        """
        if value is None:
            return None
        if not value.startswith("https://"):
            raise ValueError("callback_url must be an https:// URL")
        return value

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > MAX_METADATA_KEYS:
            raise ValueError(f"metadata accepts at most {MAX_METADATA_KEYS} keys")
        for key, item in value.items():
            if len(item) > MAX_METADATA_VALUE_CHARS:
                raise ValueError(
                    f"metadata[{key!r}] is longer than {MAX_METADATA_VALUE_CHARS} characters"
                )
        return value


class InvestigationLinks(ResponseModel):
    """Where to go next. §4.1 returns these alongside the id.

    Relative paths, not absolute URLs. The API is reached through a proxy, a
    tunnel and a browser origin that this process cannot see -- `Host` is
    forgeable and `X-Forwarded-*` is only as trustworthy as the last hop -- so
    building an absolute URL here would eventually hand a client a link to the
    wrong scheme or to an internal hostname.
    """

    self: str
    stream: str
    report: str | None = None


class InvestigationCreated(ResponseModel):
    """The `202` body of §4.1."""

    id: str
    state: InvestigationStatus
    query: str
    depth: InvestigationDepth
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    report_id: str | None = Field(
        default=None,
        description="Allocated eagerly so a client can subscribe before the report "
        "exists. Fetching it before the investigation reaches reporting returns "
        "409 report_not_ready.",
    )
    trace_id: str
    links: InvestigationLinks


class StepItem(ResponseModel):
    """One row of the step sub-collection (§4.2).

    `tool_calls` and `evidence_count` are `null` for every step, always:
    `investigation_steps` has neither column, and there is no evidence table to
    count from. See this module's docstring for why they are `null` and not `0`.
    """

    id: str
    seq: int
    agent: AgentName
    title: str
    state: StepStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    tool_calls: int | None = None
    evidence_count: int | None = None


type StepsPage = Page[StepItem]
"""The paginated step sub-collection. Cursor-based like every other collection."""


class InvestigationProgress(ResponseModel):
    """How far along the run is (§4.2).

    `steps_total_estimate` comes from the Planner's decomposition when one has
    been recorded, and is `null` before that. It is named an *estimate* in the
    contract for a good reason: the Critic loop can send the graph round again
    (`docs/agent-system.md` §13), so the plan's step count is a lower bound, not a
    denominator. A UI that renders it as one will show progress going backwards.
    """

    steps_completed: int
    steps_total_estimate: int | None = None


class InvestigationCounts(ResponseModel):
    """Volume of what the run has produced (§4.2). See the module docstring."""

    signals_considered: int | None = None
    evidence: int | None = None
    citations: int | None = None


class InvestigationUsage(ResponseModel):
    """Token and tool accounting (§4.2).

    Token counts are real: they are rolled up into `investigations.token_input`
    and `token_output` inside the same transaction that completes each step
    (`services/investigation_service.py`), precisely so the budget check on the
    next step reads a total that is not lagging a sweep interval behind.
    """

    input_tokens: int
    output_tokens: int
    tool_calls: int | None = None


class InvestigationError(ResponseModel):
    """Why a terminal run ended badly (§4.2).

    Present only when the run failed or was cancelled. §4.2 is explicit that this
    is still a `200`: the request succeeded, the investigation did not, and
    conflating the two would make a client retry an HTTP call that worked.

    `code` and `message` carry the same string today. `investigations.error` is
    documented to hold the `code` of the raised `OmniSenseError` rather than a
    stack trace, so that handlers never pattern-match on message text
    (`docs/coding-standards.md` §2.7) -- there is one column and it holds the
    code. Splitting them needs a second column, not a guess at where the code
    ends.
    """

    code: str
    message: str
    step_id: str | None = None


class InvestigationDetail(ResponseModel):
    """The `200` body of §4.2.

    `plan`, `steps` and `usage` are omitted entirely rather than nulled when the
    caller did not ask for them via `include`. Omission is what makes "you did not
    request this" distinguishable from "this is empty", which matters most for
    `plan`: `investigations.plan` is nullable precisely so that *not planned yet*
    stays distinguishable from *planned to do nothing*
    (`models/orm/investigation.py`), and a response that nulled an unrequested
    plan would throw that distinction away at the last hop.
    """

    id: str
    state: InvestigationStatus
    query: str
    depth: InvestigationDepth | None = Field(
        default=None,
        description="Null on a read: `investigations` has no depth column, so the "
        "preset chosen at creation is not recoverable. Echoing `standard` would "
        "report a budget the run may never have been given.",
    )
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    progress: InvestigationProgress
    counts: InvestigationCounts
    report_id: str | None = None
    trace_id: str
    error: InvestigationError | None = None
    links: InvestigationLinks

    plan: dict[str, Any] | None = None
    steps: StepsPage | None = None
    usage: InvestigationUsage | None = None
