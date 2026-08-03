"""Retry pacing for the runtime, and the circuit that stops it retrying forever.

A connector never sleeps and never retries (`docs/connector-spec.md` §1). It
raises `TransientError` and stops. This module is the other half of that
arrangement: the policy the *runtime* applies around a page, kept here so it is
uniform across every source and observable in one place, rather than
reimplemented eight times inside eight `fetch()` loops with eight subtly
different ceilings.

§5.2 fixes the numbers -- full jitter, base 1s, cap 60s, five attempts per page,
a five-minute budget per page, `Retry-After` overriding jitter entirely, and
anything over 900s converted to a `QuotaError`. Three of those deserve their
reasoning written down.

**Full jitter, not equal jitter.** Equal jitter (`half + uniform(0, half)`) keeps
half the delay deterministic, so every worker that failed in the same second
retries inside the same narrow window -- the herd is thinned, not dispersed, and
the provider sees a spike at every doubling. Full jitter spreads N workers
uniformly across the whole interval, so the expected instantaneous rate falls as
the window grows, which is the entire point of backing off. It also costs one
extra retry on average versus equal jitter, and that is the trade being made
deliberately: one extra request per worker beats a synchronized volley.

**`Retry-After` overrides the jitter entirely, not as an upper bound.** The
provider is stating when it will serve us again. Retrying earlier -- which any
`min()` with a jittered value would do -- is a request we know will be refused,
and refused requests are what an abuse heuristic counts.

**Over 900s it becomes a `QuotaError`.** Fifteen minutes of `asyncio.sleep`
inside a page is a worker held out of the pool for fifteen minutes to serve one
account. `QuotaError` is a partial success: the cursor commits, the run is
rescheduled at `reset_at`, and the worker moves on
(`connectors/exceptions.py`).

Not built on `tenacity` despite §5.2's parenthetical, and that is a considered
divergence rather than an omission. Two of the rules above are not expressible
in tenacity's model: the wait depends on a *response header carried by the
exception*, which `RetryCallState` has no channel for, and the >900s rule
changes the exception's class mid-retry, which `stop`/`wait` cannot do. Wrapping
it would mean smuggling both through a mutable closure -- more machinery than
the loop it replaces. `tenacity_wait()` is provided for callers already inside a
tenacity retry that only want the jitter curve.
"""

from __future__ import annotations

import asyncio
import enum
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from connectors.exceptions import CircuitOpenError, ConnectorError, QuotaError, TransientError

__all__ = [
    "DEFAULT_POLICY",
    "BackoffPolicy",
    "CircuitBreaker",
    "CircuitState",
    "RetryController",
    "full_jitter",
    "next_delay",
    "retry_after_of",
    "retry_page",
    "tenacity_wait",
]

_SHARED_RNG = random.Random()
"""Module-level default so callers need not thread one through.

Every function here takes an optional `rng` precisely so tests can inject a
deterministic one: asserting on a real `random.uniform` is a flaky test by
construction, and a jitter test that occasionally passes is worse than none.
"""


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """The numbers from `docs/connector-spec.md` §5.2, in one place."""

    base_seconds: float = 1.0
    cap_seconds: float = 60.0
    max_attempts: int = 5
    """Attempts per page, including the first. Five means four retries."""

    total_budget_seconds: float = 300.0
    """Wall-clock ceiling per page. Whichever of this and `max_attempts` trips
    first ends the retrying -- the budget is what actually binds when
    `Retry-After` is driving, since two 200s waits exhaust it in two attempts."""

    quota_threshold_seconds: float = 900.0
    """A `Retry-After` above this becomes a `QuotaError` instead of a sleep."""

    def ceiling_for(self, attempt: int) -> float:
        """`min(cap, base * 2 ** attempt)`, with `attempt` zero-based.

        Clamped at the cap *before* jitter, not after, because clamping after
        would leave the distribution's mass at the ceiling instead of spread
        under it -- which is equal jitter's failure mode by another route.
        """
        if attempt < 0:
            raise ValueError(f"attempt must be >= 0, got {attempt}")
        # Guard the shift itself: 2 ** 1024 is a float overflow, and an attempt
        # counter is the kind of thing a caller passes a loop variable to.
        if attempt > 32:
            return self.cap_seconds
        return min(self.cap_seconds, self.base_seconds * (2.0**attempt))


DEFAULT_POLICY = BackoffPolicy()


def full_jitter(
    attempt: int, *, policy: BackoffPolicy = DEFAULT_POLICY, rng: random.Random | None = None
) -> float:
    """`uniform(0, min(cap, base * 2 ** attempt))`. Zero-based `attempt`."""
    return (rng or _SHARED_RNG).uniform(0.0, policy.ceiling_for(attempt))


def retry_after_of(error: BaseException) -> float | None:
    """Dig the provider's `Retry-After` out of a connector error, if it carried one.

    Checks the attribute first (`QuotaError` declares it) and then `details`,
    because a `TransientError` raised for a 429-within-cap has nowhere else to
    put it -- `ConnectorError.__init__` takes `details`, not `retry_after_seconds`.
    Looking in only one of the two places is how the header gets parsed
    correctly, attached correctly, and then ignored.
    """
    value = getattr(error, "retry_after_seconds", None)
    if value is None:
        details = getattr(error, "details", None)
        if isinstance(details, dict):
            value = details.get("retry_after_seconds")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def next_delay(
    attempt: int,
    *,
    retry_after: float | None = None,
    policy: BackoffPolicy = DEFAULT_POLICY,
    rng: random.Random | None = None,
    connector: str | None = None,
    account_id: str | None = None,
) -> float:
    """Seconds to wait before attempt `attempt + 1`.

    Raises `QuotaError` when the provider asked for longer than the policy is
    willing to hold a worker for.
    """
    if retry_after is None:
        return full_jitter(attempt, policy=policy, rng=rng)

    if retry_after > policy.quota_threshold_seconds:
        raise QuotaError(
            f"provider asked for {retry_after:.0f}s, above the "
            f"{policy.quota_threshold_seconds:.0f}s hold threshold",
            reset_at=time.time() + retry_after,
            retry_after_seconds=retry_after,
            connector=connector,
            account_id=account_id,
        )
    # Exactly as asked: no jitter, no minimum, no cap. See the module docstring.
    return max(0.0, retry_after)


class RetryController:
    """Attempt and budget accounting for one page.

    One controller per page, not per run: §5.2 scopes both limits to a page, and
    a run-scoped controller would let a long backfill's early hiccups exhaust the
    budget for a page an hour later.
    """

    def __init__(
        self,
        *,
        policy: BackoffPolicy = DEFAULT_POLICY,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        connector: str | None = None,
        account_id: str | None = None,
    ) -> None:
        self._policy = policy
        self._rng = rng
        # Monotonic, unlike the limiter's wall clock: this measures a duration
        # inside one process, and an NTP step mid-page must not appear to consume
        # or refund the budget.
        self._clock = clock
        self._sleep = sleep
        self._connector = connector
        self._account_id = account_id
        self._started = clock()
        self.attempts = 0
        """Attempts already made, including the one that just failed."""

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    @property
    def remaining_budget(self) -> float:
        return self._policy.total_budget_seconds - self.elapsed

    async def sleep_before_retry(self, error: ConnectorError) -> float:
        """Wait before the next attempt, or re-raise when there is not one.

        Re-raises the *original* error on exhaustion rather than a wrapper, so
        the run records why the provider failed rather than "retries exhausted",
        which is never the interesting half.
        """
        self.attempts += 1

        # Computed before the attempt counter is consulted, because the >900s
        # conversion is about the *provider's* instruction and not about how
        # many tries we have left. Checking attempts first would let a "come
        # back in an hour" arriving on the fifth attempt surface as a failed run
        # rather than as a run rescheduled at `reset_at` -- the same information,
        # classified in the one way that loses the work already done.
        delay = next_delay(
            self.attempts - 1,
            retry_after=retry_after_of(error),
            policy=self._policy,
            rng=self._rng,
            connector=self._connector,
            account_id=self._account_id,
        )

        if self.attempts >= self._policy.max_attempts:
            raise error

        # Checked before sleeping, not after: sleeping past the budget and then
        # noticing wastes exactly the time the budget exists to bound.
        if delay > self.remaining_budget:
            raise error

        await self._sleep(delay)
        return delay


class CircuitState(enum.StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class _Circuit:
    consecutive_failures: int = 0
    opens_until: float | None = None
    probing: bool = False
    reason: str = ""


class CircuitBreaker:
    """Five consecutive failures per account opens for ten minutes (§5.2).

    "Failure" means a failed **operation**, not a failed attempt. A page that
    burns all five attempts counts once. Counting attempts would open the circuit
    on the first genuinely flaky page every time, which inverts the design: the
    retry loop absorbs blips, and the breaker exists for the case where retrying
    is pointless because the integration itself is broken.

    Deliberately in-process. A shared breaker would need Redis and consensus
    about what a failure is; a local one costs at most `replicas x 5` doomed
    calls before every worker has independently stopped, which is a price worth
    paying to keep this module free of a datastore -- and a breaker that fails
    closed because Redis is down is a self-inflicted outage.

    Keyed by whatever the caller considers an account: §5.2 says per account, so
    `f"{slug}:{account_id}"`. A per-connector key would let one revoked token
    stop every tenant on that source.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        open_seconds: float = 600.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        # Wall clock, because `opens_until` is reported to operators and
        # travels in `CircuitOpenError.details` alongside `QuotaError.reset_at`.
        # A monotonic value there is a number nobody can act on.
        self._clock = clock
        self._circuits: dict[str, _Circuit] = {}

    def state(self, key: str) -> CircuitState:
        circuit = self._circuits.get(key)
        if circuit is None or circuit.opens_until is None:
            return CircuitState.CLOSED
        if self._clock() < circuit.opens_until:
            return CircuitState.OPEN
        return CircuitState.HALF_OPEN

    def consecutive_failures(self, key: str) -> int:
        """Failures in a row for this key, as the breaker currently counts them.

        Surfaced so `GET /health` can show an account approaching the threshold
        rather than only reporting it once the integration has already gone
        dark. Reads zero while a half-open probe is in flight: the probe is
        judged on its own result, not on the outage that preceded it.
        """
        circuit = self._circuits.get(key)
        return 0 if circuit is None else circuit.consecutive_failures

    def check(self, key: str) -> None:
        """Raise `CircuitOpenError` while open; let exactly one probe through after.

        The half-open probe is what closes the circuit again. Without it the
        breaker would need a separate healthcheck to ever recover, and the
        integration would stay dark until someone noticed.
        """
        circuit = self._circuits.get(key)
        if circuit is None or circuit.opens_until is None:
            return
        now = self._clock()
        if now < circuit.opens_until:
            raise CircuitOpenError(
                f"circuit open for {key!r} {circuit.reason}",
                opens_until=circuit.opens_until,
                account_id=key,
                details={"retry_after_seconds": circuit.opens_until - now},
            )
        # Entering half-open. The count is cleared so that the evidence gathered
        # *before* the timer cannot combine with evidence gathered after: what
        # matters now is whether the one probe works.
        circuit.probing = True
        circuit.consecutive_failures = 0

    def record_success(self, key: str) -> None:
        """Reset. Consecutive means consecutive: one success clears the count."""
        self._circuits.pop(key, None)

    def record_failure(self, key: str) -> float | None:
        """Count one failed operation. Returns `opens_until` if this opened it."""
        circuit = self._circuits.setdefault(key, _Circuit())
        circuit.consecutive_failures += 1

        # A failed probe re-opens on its own, without waiting for another five.
        # This is the branch the cleared counter above makes necessary: without
        # it, the one call we allowed through to test the water would be followed
        # by four more into a source we just watched fail.
        if circuit.probing:
            circuit.probing = False
            circuit.reason = "after a failed recovery probe"
        elif circuit.consecutive_failures >= self._failure_threshold:
            circuit.reason = f"after {circuit.consecutive_failures} consecutive failures"
        else:
            return None

        circuit.opens_until = self._clock() + self._open_seconds
        return circuit.opens_until


async def retry_page[T](
    operation: Callable[[], Awaitable[T]],
    *,
    policy: BackoffPolicy = DEFAULT_POLICY,
    rng: random.Random | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    breaker: CircuitBreaker | None = None,
    breaker_key: str | None = None,
    connector: str | None = None,
    account_id: str | None = None,
) -> T:
    """Run one page's fetch under the §5.2 policy.

    Only `TransientError` is retried; that is the taxonomy's whole purpose
    (`connectors/exceptions.py`). `AuthError` and `PermanentError` are re-raised
    untouched, and a `QuotaError` -- whether raised by the provider or
    manufactured here from a long `Retry-After` -- ends the page as a partial
    success.

    Anything that is *not* a `ConnectorError` propagates without touching the
    breaker. A `KeyError` in a mapper is our defect, not the provider being
    unhealthy, and letting it trip the breaker would take the integration down
    for ten minutes over a typo while hiding the traceback behind a circuit
    message.
    """

    def record_failure() -> None:
        if breaker is not None and breaker_key is not None:
            breaker.record_failure(breaker_key)

    def record_success() -> None:
        if breaker is not None and breaker_key is not None:
            breaker.record_success(breaker_key)

    if breaker is not None and breaker_key is not None:
        breaker.check(breaker_key)

    controller = RetryController(
        policy=policy,
        rng=rng,
        clock=clock,
        sleep=sleep,
        connector=connector,
        account_id=account_id,
    )

    while True:
        try:
            result = await operation()
        except TransientError as exc:
            try:
                await controller.sleep_before_retry(exc)
            except ConnectorError:
                # Attempts or budget exhausted, or the wait converted to a
                # quota. Either way this page is over.
                record_failure()
                raise
            continue
        except ConnectorError:
            record_failure()
            raise
        else:
            record_success()
            return result


@dataclass(frozen=True, slots=True)
class _TenacityFullJitterWait:
    """`tenacity`-compatible wait, duck-typed so tenacity stays optional.

    Only the jitter curve: a caller using this gets none of the `Retry-After`
    handling, which is the reason the rest of this module does not go through
    tenacity at all.
    """

    policy: BackoffPolicy = DEFAULT_POLICY
    rng: random.Random | None = None

    def __call__(self, retry_state: Any) -> float:
        # tenacity counts attempts from 1; `ceiling_for` counts from 0, so the
        # first retry after one failure waits under `base`, not under `2 * base`.
        attempt = max(0, int(getattr(retry_state, "attempt_number", 1)) - 1)
        return full_jitter(attempt, policy=self.policy, rng=self.rng)


def tenacity_wait(
    policy: BackoffPolicy = DEFAULT_POLICY, rng: random.Random | None = None
) -> _TenacityFullJitterWait:
    """A `wait=` callable for `tenacity.AsyncRetrying`, for callers already using it."""
    return _TenacityFullJitterWait(policy=policy, rng=rng)
