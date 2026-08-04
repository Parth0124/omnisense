"""Agent error taxonomy and the retry policy the node runtime applies to it.

`docs/agent-system.md` §6 states the rule this module exists to make executable:
transient failures (rate limit, timeout, 5xx) retry with backoff *inside the
node*; permanent failures (schema violation, denied tool, unknown tool) append to
`InvestigationState.errors` and let the router decide whether the branch was
required. A single fan-out branch failing never fails the join.

The taxonomy is therefore not a catalogue of what can go wrong -- it is exactly
the set of distinctions that change behaviour:

- **transient vs permanent** decides whether the node retries at all. Retrying a
  schema violation spends the run's budget re-asking a question the model has
  already shown it cannot answer in that shape.
- **blocking vs degradable** decides what the *router* does with a failure that
  survived retry. A Forecast that could not converge costs the report one
  section; a Planner that produced no plan leaves the graph with nothing to
  execute, and pretending otherwise produces a confident report about nothing.
- **retry-after** decides *when*. A provider that tells us how long to wait is
  giving us the only number in the system that is not a guess, so an exponential
  backoff that ignores it will keep failing until it accidentally exceeds it.

Two deliberate omissions. There is no `AgentError` exception class here: that
name is already a *record* in `agents/state.py` -- the thing these exceptions get
converted into by `to_state_error()` -- and two meanings for one name in one
package is how a `raise AgentError(...)` ends up constructing a Pydantic model.
And there is no catch-all retry on `Exception`: an unclassified exception is a
bug, and retrying a bug three times only makes the traceback arrive later.

Backoff is `tenacity` rather than a hand-rolled loop because `connectors/` and
`services/` already pace against it, and because `AsyncRetrying` takes an
injectable `sleep`, which is what lets a test assert on the delay sequence
without spending it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from tenacity import AsyncRetrying, RetryCallState, retry_if_exception, stop_after_attempt
from tenacity.wait import wait_base

from agents.state import AgentError
from backend.core.exceptions import ExternalServiceError, OmniSenseError, RateLimitedError
from models.enums import AgentName
from services.llm.provider import LLMError, LLMRateLimited, LLMSchemaError, LLMTimeout
from services.llm.router import TokenBudgetExceeded

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "NO_RETRY_POLICY",
    "AgentExecutionError",
    "AgentTimeoutError",
    "ErrorClass",
    "PermanentAgentError",
    "RetryPolicy",
    "StructuredOutputError",
    "ToolExecutionError",
    "ToolNotAllowedError",
    "TransientAgentError",
    "UnsafeToolOutputError",
    "classify",
    "is_blocking",
    "is_transient",
    "run_with_retry",
    "to_state_error",
]

# --------------------------------------------------------------------------- #
# Classification vocabulary
# --------------------------------------------------------------------------- #


class ErrorClass:
    """The closed vocabulary of `error_type` strings written into the state.

    A namespace of constants rather than `type(exc).__name__`: these strings
    reach the report's "gaps" section, metric labels and the evaluation
    harness's regression comparisons, and a label that changes when a class is
    renamed silently splits a time series in two.
    """

    RATE_LIMITED: Final = "rate_limited"
    TIMEOUT: Final = "timeout"
    PROVIDER_ERROR: Final = "provider_error"
    SCHEMA_VIOLATION: Final = "schema_violation"
    TOOL_DENIED: Final = "tool_denied"
    TOOL_FAILED: Final = "tool_failed"
    UNSAFE_TOOL_OUTPUT: Final = "unsafe_tool_output"
    BUDGET_EXHAUSTED: Final = "budget_exhausted"
    DEPENDENCY_MISSING: Final = "dependency_missing"
    UNEXPECTED: Final = "unexpected"


# --------------------------------------------------------------------------- #
# The exceptions
# --------------------------------------------------------------------------- #


class AgentExecutionError(OmniSenseError):
    """Base for a failure raised *inside* an agent node.

    Carries the two flags the runtime branches on. `transient` is read by the
    retry policy; `blocking` is read by the router once the retries are gone --
    and defaults to `False` because the design's stated preference
    (`docs/architecture.md` §7.3) is a smaller, honestly-labelled answer over a
    failed run.
    """

    status_code = 500
    code = "agent_execution_error"
    default_message = "An agent step failed."

    transient: bool = False
    error_type: str = ErrorClass.UNEXPECTED

    def __init__(
        self,
        message: str | None = None,
        *,
        agent: AgentName = AgentName.UNKNOWN,
        blocking: bool = False,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, details=details, cause=cause)
        self.agent = agent
        self.blocking = blocking
        self.details.setdefault("agent", str(agent))


class TransientAgentError(AgentExecutionError):
    """Something upstream was busy or slow. The identical call may work shortly."""

    status_code = 503
    code = "agent_transient_error"
    default_message = "A transient failure interrupted an agent step."

    transient = True
    error_type = ErrorClass.PROVIDER_ERROR


class AgentTimeoutError(TransientAgentError):
    """The node exceeded its wall-clock slice.

    A class of its own because it is the only failure mode that can *hang* a
    fan-out join. `agents/graph.py` bounds every node by a timeout derived from
    `deadline_at`, so a branch stuck on an await ends as this error and the join
    proceeds with one fewer input. Without it the join waits on a coroutine that
    is never going to return, and a 30-minute investigation becomes an infinite
    one.
    """

    status_code = 504
    code = "agent_timeout"
    default_message = "An agent step exceeded its time slice."

    error_type = ErrorClass.TIMEOUT


class PermanentAgentError(AgentExecutionError):
    """Repeating this call cannot help. Record it and route around it."""

    code = "agent_permanent_error"
    default_message = "An agent step failed permanently."

    transient = False


class StructuredOutputError(PermanentAgentError):
    """The model's output did not satisfy the agent's `schemas.py` contract.

    Separated from provider failures for the same reason `LLMSchemaError` is
    separated in `services/llm/provider.py`: a sustained rate of these is a
    prompt or schema regression, and folding it into the provider-reliability
    metric hides it behind an unrelated dashboard.
    """

    status_code = 502
    code = "agent_output_invalid"
    default_message = "An agent produced output that failed schema validation."

    error_type = ErrorClass.SCHEMA_VIOLATION


class ToolNotAllowedError(PermanentAgentError):
    """An agent asked for a tool outside its allowlist.

    `docs/agent-system.md` §9 makes allowlisting deny-by-default, and this is
    the denial. Loud rather than a silent no-op on purpose: a silent no-op turns
    "the Critic tried to retrieve new evidence" -- the exact move that would
    make the Critic a second author instead of a reviewer -- into a missing
    result that nobody ever looks at.
    """

    status_code = 403
    code = "agent_tool_denied"
    default_message = "That tool is not in this agent's allowlist."

    error_type = ErrorClass.TOOL_DENIED


class ToolExecutionError(AgentExecutionError):
    """A permitted tool raised.

    Transience is a *parameter* here rather than a subclass, because the same
    tool fails both ways: `hybrid_search` against a saturated OpenSearch is
    worth retrying, and `hybrid_search` with a malformed filter is not. The
    tool's own exception decides, via `classify()`.
    """

    status_code = 502
    code = "agent_tool_failed"
    default_message = "A tool invoked by an agent failed."

    error_type = ErrorClass.TOOL_FAILED

    def __init__(
        self,
        message: str | None = None,
        *,
        tool: str,
        agent: AgentName = AgentName.UNKNOWN,
        transient: bool = False,
        blocking: bool = False,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message, agent=agent, blocking=blocking, details=details, cause=cause)
        self.tool = tool
        self.transient = transient
        self.details.setdefault("tool", tool)


class UnsafeToolOutputError(PermanentAgentError):
    """Tool output failed the content/instruction separation check.

    Third-party text reaching an agent's context is a prompt-injection surface
    (`docs/security-and-privacy.md`); `agents/base.py` fences that text as data
    and this is what a fence violation raises. Permanent by construction -- the
    same bytes fail the same check forever, and retrying is only a slower way of
    not reading the passage.
    """

    status_code = 422
    code = "agent_unsafe_tool_output"
    default_message = "Tool output could not be safely fenced as data."

    error_type = ErrorClass.UNSAFE_TOOL_OUTPUT


# --------------------------------------------------------------------------- #
# Mapping arbitrary exceptions onto the taxonomy
# --------------------------------------------------------------------------- #

_TRANSIENT_STATUSES: Final = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
"""Upstream statuses that mean "later", not "no".

500 is included for the reason `services/llm/router.py` includes it: providers
return it for transient capacity problems as often as for real bugs, and
treating it as permanent fails a run that a second attempt would have answered.
"""


def is_transient(exc: BaseException) -> bool:
    """Whether retrying the identical call is worth the tokens.

    Order matters. `LLMSchemaError` and `TokenBudgetExceeded` are tested before
    the broader branches that would otherwise claim them: the first *is* an
    `LLMError`, and retrying it re-asks a question already shown to be
    unanswerable in that shape; the second means the budget is gone, and a retry
    is precisely what must not happen.
    """
    if isinstance(exc, TokenBudgetExceeded | LLMSchemaError):
        return False
    if isinstance(exc, AgentExecutionError):
        return exc.transient
    if isinstance(exc, LLMRateLimited | LLMTimeout | RateLimitedError | TimeoutError):
        return True
    if isinstance(exc, LLMError):
        return exc.provider_status in _TRANSIENT_STATUSES
    return isinstance(exc, ExternalServiceError)


def classify(exc: BaseException) -> str:
    """The `error_type` string this exception is recorded under.

    Deliberately total: an exception nothing recognises still gets a label
    (`unexpected`) rather than escaping, because a node that dies on an
    unclassified exception takes the whole graph run with it, and the state at
    that point is worth more than the traceback.
    """
    if isinstance(exc, TokenBudgetExceeded):
        return ErrorClass.BUDGET_EXHAUSTED
    if isinstance(exc, AgentExecutionError):
        return exc.error_type
    if isinstance(exc, LLMSchemaError):
        return ErrorClass.SCHEMA_VIOLATION
    if isinstance(exc, LLMRateLimited | RateLimitedError):
        return ErrorClass.RATE_LIMITED
    if isinstance(exc, LLMTimeout | TimeoutError):
        return ErrorClass.TIMEOUT
    if isinstance(exc, NotImplementedError):
        # A dependency this layer is built against does not exist yet. Labelled
        # distinctly so a half-built system's gaps are countable rather than
        # indistinguishable from provider flakiness.
        return ErrorClass.DEPENDENCY_MISSING
    if isinstance(exc, LLMError | ExternalServiceError):
        return ErrorClass.PROVIDER_ERROR
    return ErrorClass.UNEXPECTED


def is_blocking(exc: BaseException) -> bool:
    """Whether this failure leaves the graph unable to make honest progress.

    Only an `AgentExecutionError` can declare itself blocking, and only the
    agent knows: the Planner's failure means there is no plan to execute, while
    the identical provider error inside Forecast costs the report one section.
    The default is non-blocking, so a run degrades unless someone deliberately
    said otherwise.
    """
    return isinstance(exc, AgentExecutionError) and exc.blocking


def to_state_error(exc: BaseException, *, agent: AgentName) -> AgentError:
    """Convert a caught exception into the record the state carries.

    Provider free-text is not copied through -- `LLMError` documents why -- so
    the message here is the exception's rendered string, which for this taxonomy
    is a fixed default or a message we wrote. `recoverable` is `is_blocking()`
    inverted, and is what `agents/router.py` reads to choose between "continue
    with less" and "stop".
    """
    return AgentError(
        agent=agent,
        error_type=classify(exc),
        message=str(exc) or type(exc).__name__,
        recoverable=not is_blocking(exc),
    )


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


class _RetryAfterAwareWait(wait_base):
    """Exponential backoff that yields to an upstream `Retry-After`.

    The provider's own number wins whenever it exists. Anything else is a guess
    competing with a fact, and a guess that lands *inside* the stated window
    burns an attempt to be told the same thing again -- which is how a run
    exhausts its retries during a rate-limit window without ever having waited
    it out.
    """

    def __init__(self, policy: RetryPolicy) -> None:
        self._policy = policy

    def __call__(self, retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        exc = outcome.exception() if outcome is not None else None
        hinted = getattr(exc, "retry_after_seconds", None)
        if isinstance(hinted, int | float) and not isinstance(hinted, bool) and hinted >= 0:
            return min(float(hinted), self._policy.max_delay_seconds)
        return self._policy.backoff_for(retry_state.attempt_number)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times, how far apart, and for which failures.

    Frozen and passed in rather than read from config inside the loop: the
    Collector wants patience against a rate-limited API and the Critic wants
    none against a schema violation, so this is a per-agent decision that one
    global setting cannot express.

    `max_attempts` counts the *first* call, so `3` means one attempt and two
    retries. The off-by-one here is the difference between a documented policy
    and an accidental one.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 20.0
    retry_on: Callable[[BaseException], bool] = is_transient

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1: a policy that never calls is a bug.")

    def backoff_for(self, attempt_number: int) -> float:
        """Delay before the attempt following `attempt_number` (1-based).

        `2.0 **`, not `2 **`: an integer base raised to a non-literal integer
        exponent is `Any` to a type checker (a negative exponent would make it a
        float), and an `Any` here would silently propagate through every delay
        this policy reports.
        """
        exponent = max(0, attempt_number - 1)
        return min(self.base_delay_seconds * (2.0**exponent), self.max_delay_seconds)

    def delays(self) -> Sequence[float]:
        """The full delay sequence, ignoring any `Retry-After` hint.

        Exposed so an operator (and a test) can read a policy's actual patience
        off the object instead of re-deriving the formula, which is the sort of
        duplication that makes a test agree with a bug.
        """
        return [self.backoff_for(n) for n in range(1, self.max_attempts)]


DEFAULT_RETRY_POLICY: Final = RetryPolicy()
"""Three attempts, half a second apart doubling to twenty.

Sized against the node, not the call: a node holds a checkpoint boundary open
while it retries, so a policy generous enough to ride out a multi-minute outage
would stall the whole investigation behind one step. Riding out an outage is the
scheduler's job (`docs/agent-system.md` §7), which resumes from the checkpoint.
"""

NO_RETRY_POLICY: Final = RetryPolicy(max_attempts=1)
"""One attempt. For nodes whose work is neither idempotent nor cheap to repeat."""


@dataclass(slots=True)
class _Attempts:
    """Mutable counter threaded through one `run_with_retry` call."""

    count: int = 0
    errors: list[BaseException] = field(default_factory=list)


async def run_with_retry[ResultT](
    call: Callable[[], Awaitable[ResultT]],
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    on_retry: Callable[[BaseException, int], None] | None = None,
) -> ResultT:
    """Invoke `call`, retrying only what `policy.retry_on` accepts.

    `reraise=True` so the caller sees the provider's own exception rather than
    tenacity's `RetryError` wrapper: `agents/base.py` classifies on the
    exception's type, and the wrapper erases exactly the information it
    classifies on.

    `sleep` is injectable so a test can assert the backoff sequence without
    waiting it out. Patching `asyncio.sleep` globally instead would silence
    every other sleeper in the process, including the ones under test.
    """
    attempts = _Attempts()

    def _should_retry(exc: BaseException) -> bool:
        attempts.count += 1
        attempts.errors.append(exc)
        retryable = policy.retry_on(exc)
        if retryable and on_retry is not None:
            on_retry(exc, attempts.count)
        return retryable

    async for attempt in AsyncRetrying(
        retry=retry_if_exception(_should_retry),
        wait=_RetryAfterAwareWait(policy),
        stop=stop_after_attempt(policy.max_attempts),
        sleep=sleep,
        reraise=True,
    ):
        with attempt:
            return await call()

    # Unreachable: `stop_after_attempt` either yields a successful attempt or
    # reraises. Explicit rather than an implicit `None` return, which would
    # otherwise surface three frames away as an inexplicable `NoneType`.
    raise AssertionError("unreachable: AsyncRetrying exits by returning or reraising")
