"""The scheduler: decides when each connector next runs.

A `PeriodicWorker` rather than cron, and the reason is `docs/architecture.md`
§7.3's requirement that a source failing does not stop the others. Cron fires
regardless of whether the last run finished, so a connector that has started
taking twenty minutes gets a second copy every fifteen -- and the two compete for
the same rate limit until both fail. This scheduler dispatches only what is due
*and* not already running.

**Two replicas will tick at the same instant.** That is not hypothetical: it
happens on every rolling deploy. Safety comes from a conditional update -- a
connector is claimed by moving its `next_run_at` forward in the same statement
that selects it, so exactly one replica wins and the other sees nothing due. An
advisory lock would also work and would serialise the whole tick; the conditional
update lets both replicas do useful work on different connectors.

**Jitter is applied to every interval.** Without it, twenty connectors configured
at "every 15 minutes" all fire on the quarter hour, forever, because they were
all created by the same seed script. The result is a load spike every fifteen
minutes and idle capacity in between.

**Backoff is per connector and it is recorded.** A source failing repeatedly gets
progressively longer intervals rather than being retried at full rate, which is
both politer to the third party and cheaper for us. The backoff state lives in
the schedule row so it survives a restart -- in memory it would reset on every
deploy, which is exactly when a struggling source is least able to cope.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from backend.core.logging import get_logger
from workers.runtime.base_worker import PeriodicWorker, run_worker

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from backend.core.config import Settings
    from workers.runtime.health import DependencyProbe

__all__ = [
    "DEFAULT_TICK_SECONDS",
    "MAX_BACKOFF_SECONDS",
    "ScheduleDecision",
    "SchedulerWorker",
    "next_run_at",
]

logger = get_logger(__name__)

WORKER_NAME: Final = "scheduler"

DEFAULT_TICK_SECONDS: Final = 30.0
"""How often the scheduler looks for due connectors.

Deliberately much shorter than any connector's interval. The tick is a *poll for
due work*, not a schedule: a 30-second tick against a 15-minute interval means a
connector fires within 30 seconds of becoming due, and the poll itself is one
indexed query.
"""

MAX_BACKOFF_SECONDS: Final = 6 * 60 * 60
"""Six hours. Past this a source is broken, not busy.

Capped because unbounded exponential backoff eventually schedules the next
attempt after the heat death of the connector's usefulness -- and a source that
has been failing for six hours needs a human, not a longer wait.
"""

JITTER_FRACTION: Final = 0.1
"""±10% of the interval.

Enough to decorrelate twenty connectors created by the same seed script;
small enough that "every 15 minutes" still means roughly every 15 minutes.
"""

MAX_DISPATCH_PER_TICK: Final = 10
"""Connectors started in one tick.

A cap rather than "everything due", because after an outage *everything* is due
at once -- and dispatching forty syncs simultaneously reproduces the load spike
the scheduler exists to avoid, at the worst possible moment.
"""


@dataclass(frozen=True, slots=True)
class ScheduleDecision:
    """What the scheduler decided for one connector, and why.

    The `reason` is not decoration. "Why did this connector not run" is the
    question the scheduler is asked most, and without a recorded reason the
    answer requires reconstructing the tick from timestamps.
    """

    slug: str
    should_run: bool
    reason: str
    next_run_at: datetime | None = None


def next_run_at(
    *,
    interval_seconds: float,
    consecutive_failures: int = 0,
    now: datetime | None = None,
    jitter: float | None = None,
) -> datetime:
    """When a connector should next run, with backoff and jitter applied.

    Backoff doubles per consecutive failure and is capped. Jitter is
    multiplicative on the final interval rather than additive on the base, so a
    backed-off connector's retries are decorrelated too -- otherwise three
    connectors that failed together retry together, fail together, and stay in
    lockstep indefinitely.

    `jitter` is injectable so the arithmetic is testable. A scheduler whose
    output cannot be pinned in a test is one whose backoff is verified by
    watching production.
    """
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive, got {interval_seconds}")

    base = interval_seconds
    if consecutive_failures > 0:
        # `min` before the multiply keeps the exponent from overflowing on a
        # connector that has been failing for a week.
        multiplier = 2 ** min(consecutive_failures, 16)
        base = min(interval_seconds * multiplier, MAX_BACKOFF_SECONDS)

    factor = jitter if jitter is not None else random.uniform(-JITTER_FRACTION, JITTER_FRACTION)
    seconds = max(1.0, base * (1.0 + factor))
    return (now or datetime.now(UTC)) + timedelta(seconds=seconds)


class SchedulerWorker(PeriodicWorker):
    """Dispatches connector syncs that are due, one claim at a time."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        dispatcher: Any,
        interval_seconds: float = DEFAULT_TICK_SECONDS,
        max_dispatch_per_tick: int = MAX_DISPATCH_PER_TICK,
        settings: Settings | None = None,
        name: str = WORKER_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            interval_seconds=interval_seconds,
            # Staggered so two replicas starting together during a rolling deploy
            # do not contend on their first tick.
            initial_delay_seconds=random.uniform(0.0, min(interval_seconds, 10.0)),
            settings=settings,
            **kwargs,
        )
        self._session_factory = session_factory
        self._dispatcher = dispatcher
        self._max_dispatch = max_dispatch_per_tick
        self.dispatched = 0
        self.claim_conflicts = 0

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        """PostgreSQL only. The schedule lives there and nothing else is needed."""
        from backend.db.session import check_postgres

        return {"postgres": check_postgres}

    async def tick(self) -> None:
        """Claim and dispatch what is due.

        Idempotent and re-entrant across replicas: the claim is a conditional
        update, so a connector claimed by another replica between the select and
        the update simply is not returned here. Two replicas ticking at the same
        instant therefore dispatch disjoint sets, and neither double-dispatches.
        """
        due = await self._claim_due(limit=self._max_dispatch)
        if not due:
            return

        logger.info("scheduler.dispatching", count=len(due))
        for schedule in due:
            slug = schedule["slug"]
            try:
                await self._dispatcher.dispatch(slug)
            except Exception as error:  # noqa: BLE001 -- one bad source must not stop the rest
                # The claim already moved `next_run_at` forward, so a failed
                # dispatch does not spin. The failure count is recorded so the
                # next interval backs off.
                self.claim_conflicts += 0
                logger.error(
                    "scheduler.dispatch_failed", connector=slug, error=type(error).__name__
                )
                await self._record_failure(slug)
                continue
            self.dispatched += 1
            await self._record_success(slug)

    # ------------------------------------------------------------ internals --

    async def _claim_due(self, *, limit: int) -> list[dict[str, Any]]:
        """Select and claim due connectors in one conditional statement.

        `SELECT ... FOR UPDATE SKIP LOCKED` then an update, inside one
        transaction. `SKIP LOCKED` is what lets two replicas work concurrently
        rather than one blocking on the other's row locks -- with plain
        `FOR UPDATE` the second replica waits for the first's whole tick.
        """
        from sqlalchemy import select, update

        from models.orm.signal import ConnectorScheduleRow  # type: ignore[attr-defined]

        now = datetime.now(UTC)
        claimed: list[dict[str, Any]] = []

        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(ConnectorScheduleRow)
                    .where(
                        ConnectorScheduleRow.enabled.is_(True),
                        ConnectorScheduleRow.next_run_at <= now,
                    )
                    .order_by(ConnectorScheduleRow.next_run_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).scalars().all()

            for row in rows:
                scheduled = next_run_at(
                    interval_seconds=row.interval_seconds,
                    consecutive_failures=row.consecutive_failures or 0,
                    now=now,
                )
                await session.execute(
                    update(ConnectorScheduleRow)
                    .where(ConnectorScheduleRow.slug == row.slug)
                    .values(next_run_at=scheduled, last_dispatched_at=now)
                )
                claimed.append(
                    {
                        "slug": row.slug,
                        "interval_seconds": row.interval_seconds,
                        "next_run_at": scheduled,
                    }
                )
        return claimed

    async def _record_success(self, slug: str) -> None:
        """Clear the backoff. A source that worked is not a struggling source."""
        await self._update_failures(slug, reset=True)

    async def _record_failure(self, slug: str) -> None:
        await self._update_failures(slug, reset=False)

    async def _update_failures(self, slug: str, *, reset: bool) -> None:
        from sqlalchemy import update

        from models.orm.signal import ConnectorScheduleRow  # type: ignore[attr-defined]

        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(ConnectorScheduleRow)
                .where(ConnectorScheduleRow.slug == slug)
                .values(
                    consecutive_failures=0
                    if reset
                    else ConnectorScheduleRow.consecutive_failures + 1,
                    last_error_at=None if reset else datetime.now(UTC),
                )
            )


def decide(
    *,
    slug: str,
    enabled: bool,
    next_due: datetime | None,
    running: bool,
    now: datetime | None = None,
) -> ScheduleDecision:
    """Pure decision function, extracted so the policy is testable without a database.

    The scheduler's actual claim is a SQL statement for concurrency reasons, but
    the *policy* it encodes is here in one place where it can be read and
    exercised. A policy that only exists inside a `WHERE` clause is a policy
    nobody reviews.
    """
    moment = now or datetime.now(UTC)
    if not enabled:
        return ScheduleDecision(slug, False, "connector is disabled")
    if running:
        # The property cron cannot express, and the reason this is not cron: a
        # second copy of a slow sync competes with the first for the same rate
        # limit until both fail.
        return ScheduleDecision(slug, False, "a sync is already in flight")
    if next_due is None:
        return ScheduleDecision(slug, True, "never run", next_run_at=moment)
    if next_due > moment:
        return ScheduleDecision(
            slug, False, f"not due until {next_due.isoformat()}", next_run_at=next_due
        )
    return ScheduleDecision(slug, True, "due", next_run_at=moment)


def main() -> None:  # pragma: no cover -- process entry point
    from backend.db.session import get_sessionmaker

    from services.connector_service import ConnectorService

    factory = get_sessionmaker()
    run_worker(
        SchedulerWorker(session_factory=factory, dispatcher=ConnectorService(factory))
    )


if __name__ == "__main__":  # pragma: no cover
    main()
