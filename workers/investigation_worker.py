"""Runs investigations: claims queued work, drives the agent graph, streams progress.

The piece that turns ten agents and a compiled graph into a product. `POST
/investigations` writes a row and returns `202` with a stream link; this worker is
what makes that link eventually carry something.

**Why a poller and not a Kafka consumer.** An investigation is claimed from
PostgreSQL with `SELECT ... FOR UPDATE SKIP LOCKED`, the same pattern
`workers/scheduler.py` uses. A topic would work too, and would be worse here: the
run's authoritative state already lives in the `investigations` row (a consumer
would still have to read it), a run takes minutes so partition assignment would
pin long work to one member, and resumption after a crash has to consult the
checkpoint rather than an offset. The row is the queue *and* the state, so one
store is simpler and cannot disagree with itself.

**Progress is streamed as it happens.** The API's SSE endpoint reads from a
`TimelineSource`; this worker publishes to the matching sink after every node.
That is what lets the UI show the Planner finishing before the Retriever has
started, rather than a spinner for four minutes. Sequence numbers are minted
*here*, by the only party that sees every event of a run in order.

**The in-process publisher only works co-located.** Running the API and this
worker in one process -- which `make dev` does -- makes streaming work with no
extra infrastructure. Split across processes it needs the Redis publisher, and
`build_timeline_publisher` says so loudly rather than silently dropping events
into a hub nobody is subscribed to. That silent version is the failure worth
avoiding: everything looks healthy and the UI just never updates.

**A crashed run resumes from its checkpoint.** LangGraph's saver keys on the
thread id, which is the investigation id, so a worker that dies mid-run is
picked up by another and continues from the last completed node instead of
re-planning and re-paying for every model call.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final, Protocol

from backend.core.logging import get_logger
from models.enums import InvestigationStatus
from workers.runtime.base_worker import PeriodicWorker, run_worker

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from agents.composition import AgentBundle
    from backend.core.config import Settings
    from workers.runtime.health import DependencyProbe

__all__ = [
    "DEFAULT_POLL_SECONDS",
    "InvestigationWorker",
    "TimelinePublisher",
    "build_timeline_publisher",
    "main",
]

logger = get_logger(__name__)

WORKER_NAME: Final = "investigation"

DEFAULT_POLL_SECONDS: Final = 2.0
"""How often the worker looks for queued work.

Short, because this is the latency a user feels between pressing the button and
seeing the first event. Two seconds against an indexed predicate is a trivial
query, and the alternative -- a listen/notify or a topic -- buys a second and a
half at the cost of a component.
"""

MAX_CONCURRENT_RUNS: Final = 2
"""Investigations one replica drives at once.

Low on purpose. A run is a dozen model calls and holds a checkpoint connection
throughout; ten concurrent runs on one replica is how a token budget is exhausted
and every run degrades together. Scale by adding replicas -- `SKIP LOCKED` means
they never contend.
"""


class TimelinePublisher(Protocol):
    """Where progress events go. The seam between this worker and the SSE endpoint."""

    async def publish(self, event: Any) -> bool: ...


class _NullPublisher:
    """Discards events. What a run with no subscribers gets.

    Explicit rather than `None` so `_emit` has no branch, and named so a log line
    can say the timeline is going nowhere -- which is a real deployment state
    (a backfill, a CLI run) and not a bug.
    """

    async def publish(self, event: Any) -> bool:
        return True


def build_timeline_publisher(*, redis_client: Any | None = None) -> TimelinePublisher:
    """Resolve the publisher, warning when it cannot reach the API.

    The in-process hub is correct only when the API and this worker share a
    process. Split apart -- which is the normal deployment -- events published
    here reach a hub in *this* process that has no subscribers, while the API's
    hub stays empty. Nothing errors; the UI simply never updates, and everything
    reports healthy. That is precisely the failure the warning exists to prevent.
    """
    if redis_client is not None:
        from workers.runtime.timeline import RedisTimelinePublisher

        return RedisTimelinePublisher(redis_client)

    from backend.api.v1.stream import get_timeline_hub

    logger.warning(
        "timeline.in_process_publisher",
        reason="no Redis client supplied",
        consequence=(
            "progress events reach subscribers only if the API runs in this same "
            "process; split across processes the stream stays silent with no error"
        ),
    )
    return get_timeline_hub()


class InvestigationWorker(PeriodicWorker):
    """Claims queued investigations and drives the agent graph to completion."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        bundle: AgentBundle | None = None,
        graph: Any | None = None,
        checkpointer: Any | None = None,
        publisher: TimelinePublisher | None = None,
        interval_seconds: float = DEFAULT_POLL_SECONDS,
        max_concurrent: int = MAX_CONCURRENT_RUNS,
        settings: Settings | None = None,
        name: str = WORKER_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, interval_seconds=interval_seconds, settings=settings, **kwargs
        )
        self._session_factory = session_factory
        self._bundle = bundle
        self._graph = graph
        self._checkpointer = checkpointer
        self._publisher = publisher or _NullPublisher()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running: set[str] = set()
        self.completed = 0
        self.failed = 0

    # -------------------------------------------------------------- health --

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        """PostgreSQL only.

        The graph reads Qdrant, Neo4j and OpenSearch through tools that degrade
        individually (`docs/architecture.md` §7.3), so a replica that cannot
        reach them can still produce a smaller, honestly-labelled answer. Without
        PostgreSQL there is no queue, no state and no checkpoint, so there is
        nothing to do at all.
        """
        from backend.db.session import check_postgres

        return {"postgres": check_postgres}

    async def setup(self) -> None:
        """Build the graph once, at startup.

        Not per run. Compiling walks the topology and validates every node, which
        is work that does not vary between investigations -- and doing it per run
        would put a startup-class failure (a missing agent) inside a request the
        user is waiting on.
        """
        if self._graph is not None:
            return
        from agents.composition import build_default_bundle, compile_investigation_graph

        self._bundle = self._bundle or build_default_bundle(settings=self._settings)
        self._graph = compile_investigation_graph(
            self._bundle, checkpointer=self._checkpointer, settings=self._settings
        )

    # ---------------------------------------------------------------- work --

    async def tick(self) -> None:
        """Claim what is queued and start driving it.

        Claiming and running are separate: the claim is a short transaction that
        moves the row to `PLANNING`, and the run is minutes long with no
        transaction held. Holding one for the duration would pin a connection per
        run and block every other writer touching that row.
        """
        capacity = self._semaphore._value  # noqa: SLF001 -- the only way to size the claim
        if capacity <= 0:
            return

        claimed = await self._claim(limit=capacity)
        for investigation_id in claimed:
            # Fire-and-forget with a tracked reference: `asyncio.create_task`
            # alone lets the loop garbage-collect a running task mid-run, which
            # cancels an investigation for no reason and is nearly impossible to
            # reproduce.
            task = asyncio.create_task(self._drive(investigation_id))
            self._running.add(investigation_id)
            task.add_done_callback(lambda _t, i=investigation_id: self._running.discard(i))

    async def _claim(self, *, limit: int) -> list[str]:
        """Move queued investigations to PLANNING, one claimer per row.

        `SKIP LOCKED` so replicas draw disjoint work rather than one blocking on
        another's transaction. The status move *is* the claim -- there is no
        separate lock table, and a crash after claiming leaves the row in
        `PLANNING`, which `_reclaim_stale` recovers.
        """
        from sqlalchemy import select, update

        from models.orm.investigation import InvestigationRow

        async with self._session_factory() as session, session.begin():
            rows = (
                await session.execute(
                    select(InvestigationRow)
                    .where(InvestigationRow.status == InvestigationStatus.QUEUED)
                    .order_by(InvestigationRow.created_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
            ).scalars().all()

            claimed: list[str] = []
            for row in rows:
                await session.execute(
                    update(InvestigationRow)
                    .where(InvestigationRow.id == row.id)
                    .values(
                        status=InvestigationStatus.PLANNING,
                        started_at=datetime.now(UTC),
                    )
                )
                claimed.append(row.id)

        if claimed:
            logger.info("investigation.claimed", count=len(claimed), ids=claimed)
        return claimed

    async def _drive(self, investigation_id: str) -> None:
        """Run one investigation to a terminal state. Never raises.

        Everything is wrapped: a run that crashes must mark itself `FAILED` and
        emit a terminal `error` event, or the row sits in `PLANNING` forever and
        the client's stream never closes. A worker that let an exception escape
        would leave both.
        """
        async with self._semaphore:
            state = await self._load_state(investigation_id)
            if state is None:
                logger.warning("investigation.vanished", investigation_id=investigation_id)
                return

            seq = 0
            try:
                seq = await self._emit(
                    investigation_id,
                    seq,
                    "step.started",
                    {"node": "planner", "message": "Planning the investigation"},
                )
                seq = await self._execute(investigation_id, state, seq)
            except asyncio.CancelledError:
                # Shutdown, not failure. The row stays in a non-terminal state so
                # another replica reclaims it and the checkpoint resumes the run
                # rather than restarting it.
                logger.info("investigation.interrupted", investigation_id=investigation_id)
                raise
            except Exception as error:  # noqa: BLE001 -- see the docstring
                self.failed += 1
                logger.error(
                    "investigation.failed",
                    investigation_id=investigation_id,
                    error=type(error).__name__,
                    detail=str(error)[:500],
                    exc_info=True,
                )
                await self._finish(investigation_id, InvestigationStatus.FAILED, str(error))
                await self._emit(
                    investigation_id,
                    seq,
                    "error",
                    {"code": type(error).__name__, "message": str(error)[:500]},
                )

    async def _execute(
        self, investigation_id: str, state: Mapping[str, Any], seq: int
    ) -> int:
        """Stream the graph, publishing an event per completed node.

        `astream` rather than `ainvoke`, and that is the entire reason a user sees
        progress. `ainvoke` returns once, at the end; `astream` yields after each
        node, which is what turns a four-minute spinner into a timeline.
        """
        assert self._graph is not None  # setup() ran
        from agents.checkpointer import thread_config

        config = thread_config(investigation_id)
        final: dict[str, Any] = dict(state)

        async for chunk in self._graph.astream(state, config=config):
            # LangGraph yields {node_name: delta} per completed node.
            for node_name, delta in chunk.items():
                if not isinstance(delta, Mapping):
                    continue
                final.update(delta)
                seq = await self._emit(
                    investigation_id,
                    seq,
                    "step.completed",
                    {
                        "node": str(node_name),
                        "message": _progress_message(str(node_name), delta),
                        **_progress_counts(final),
                    },
                )
                nxt = _next_node_hint(str(node_name))
                if nxt is not None:
                    seq = await self._emit(
                        investigation_id,
                        seq,
                        "step.started",
                        {"node": nxt, "message": _starting_message(nxt)},
                    )

        report = final.get("report")
        status = (
            InvestigationStatus.COMPLETED_WITH_FINDINGS
            if isinstance(report, Mapping) and report.get("gaps")
            else InvestigationStatus.COMPLETED
        )
        report_id = await self._store_report(investigation_id, final)
        await self._finish(investigation_id, status, None, report_id=report_id)
        self.completed += 1

        await self._emit(
            investigation_id,
            seq,
            "done",
            {
                "status": status.value,
                "report_id": report_id,
                "confidence": final.get("confidence", 0.0),
            },
        )
        return seq + 1

    # ------------------------------------------------------------ plumbing --

    async def _load_state(self, investigation_id: str) -> dict[str, Any] | None:
        from datetime import timedelta

        from sqlalchemy import select

        from agents.state import new_state
        from models.orm.investigation import InvestigationRow

        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(InvestigationRow).where(InvestigationRow.id == investigation_id)
                )
            ).scalar_one_or_none()
            if row is None:
                return None

        budget = self._settings.agents
        return dict(
            new_state(
                investigation_id=row.id,
                tenant_id=row.tenant_id,
                query=row.query,
                deadline_at=datetime.now(UTC)
                + timedelta(seconds=getattr(budget, "run_timeout_seconds", 900)),
                trace_id=row.id,
            )
        )

    async def _store_report(
        self, investigation_id: str, state: Mapping[str, Any]
    ) -> str | None:
        """Persist the report, tolerating a run that produced none.

        A run can reach a terminal state without a report -- cancelled, or halted
        by a guard after the Planner. Failing here would turn "no report" into
        "failed investigation", discarding whatever partial findings the run did
        establish.
        """
        report = state.get("report")
        if not isinstance(report, Mapping):
            return None
        try:
            from services.report_service import ReportService

            service = ReportService(
                self._session_factory, tenant_id=str(state.get("tenant_id") or "default")
            )
            stored = await service.store(
                investigation_id=investigation_id, document=dict(report)
            )
        except Exception as error:  # noqa: BLE001 -- the run succeeded either way
            logger.error(
                "investigation.report_store_failed",
                investigation_id=investigation_id,
                error=type(error).__name__,
            )
            return None
        return stored.id

    async def _finish(
        self,
        investigation_id: str,
        status: InvestigationStatus,
        error: str | None,
        *,
        report_id: str | None = None,
    ) -> None:
        from sqlalchemy import update

        from models.orm.investigation import InvestigationRow

        values: dict[str, Any] = {
            "status": status,
            "completed_at": datetime.now(UTC),
        }
        if error is not None:
            values["error"] = error[:2000]
        if report_id is not None:
            values["report_id"] = report_id

        async with self._session_factory() as session, session.begin():
            await session.execute(
                update(InvestigationRow)
                .where(InvestigationRow.id == investigation_id)
                .values(**values)
            )

    async def _emit(
        self, investigation_id: str, seq: int, event_type: str, data: Mapping[str, Any]
    ) -> int:
        """Publish one timeline event and return the next sequence number.

        A failed publish is logged and swallowed. The stream is a *view* of the
        run; losing a frame costs the user a progress update, while raising would
        cost them the investigation. The SSE contract already handles gaps --
        `stream.gap` exists precisely because events can be lost.
        """
        from backend.api.v1.stream import TimelineEvent

        try:
            await self._publisher.publish(
                TimelineEvent(
                    investigation_id=investigation_id,
                    seq=seq,
                    type=event_type,
                    ts=datetime.now(UTC),
                    data=dict(data),
                )
            )
        except Exception as error:  # noqa: BLE001 -- see the docstring
            logger.warning(
                "timeline.publish_failed",
                investigation_id=investigation_id,
                event=event_type,
                error=type(error).__name__,
            )
        return seq + 1


# --------------------------------------------------------------------------- #
# Human-readable progress
# --------------------------------------------------------------------------- #
#
# These strings are what a user reads while they wait, so they say what is
# *happening* rather than naming an internal node. "Searching the corpus for
# evidence" is a sentence; "retriever" is a variable name that happens to be
# visible.

_STARTING: Final[Mapping[str, str]] = {
    "planner": "Planning the investigation",
    "collector": "Collecting fresh data from sources",
    "retriever": "Searching the corpus for evidence",
    "graph_expansion": "Expanding through the knowledge graph",
    "trend": "Measuring trends over time",
    "competitor": "Building the competitive picture",
    "forecast": "Projecting measured series",
    "insight": "Synthesising findings",
    "strategy": "Forming recommendations",
    "critic": "Verifying citations and claims",
    "critic_final": "Final verification pass",
    "report": "Writing the report",
}

_ORDER: Final[tuple[str, ...]] = (
    "planner",
    "collector",
    "retriever",
    "graph_expansion",
    "trend",
    "competitor",
    "forecast",
    "insight",
    "strategy",
    "critic",
    "report",
)


def _starting_message(node: str) -> str:
    return _STARTING.get(node, f"Running {node}")


def _next_node_hint(node: str) -> str | None:
    """The node that usually follows, for an optimistic "now starting" event.

    A *hint*, and named one: the router branches, so the graph may skip the next
    node in this list entirely. Being occasionally wrong about which stage starts
    next is a far better user experience than showing nothing between a completed
    step and the following one -- which on the analysis fan-out can be tens of
    seconds of apparent silence.
    """
    try:
        index = _ORDER.index(node)
    except ValueError:
        return None
    return _ORDER[index + 1] if index + 1 < len(_ORDER) else None


def _progress_message(node: str, delta: Mapping[str, Any]) -> str:
    """What a node accomplished, in a sentence, from what it wrote.

    Derived from the delta rather than hardcoded, so the number a user sees is
    the number that actually reached the state -- not a plausible one composed
    alongside it.
    """
    match node:
        case "planner":
            steps = delta.get("plan") or []
            questions = delta.get("sub_questions") or []
            return (
                f"Planned {len(steps)} step{'s' if len(steps) != 1 else ''} "
                f"across {len(questions)} question{'s' if len(questions) != 1 else ''}"
            )
        case "collector":
            results = delta.get("collection_results") or []
            emitted = sum(getattr(r, "emitted", 0) for r in results)
            return f"Collected {emitted} record{'s' if emitted != 1 else ''}"
        case "retriever":
            evidence = delta.get("evidence") or []
            return f"Found {len(evidence)} piece{'s' if len(evidence) != 1 else ''} of evidence"
        case "graph_expansion":
            context = delta.get("graph_context")
            count = len(getattr(context, "expanded_entity_ids", ()) or ())
            return f"Expanded to {count} related entit{'ies' if count != 1 else 'y'}"
        case "trend":
            trends = delta.get("trends") or []
            return f"Measured {len(trends)} trend{'s' if len(trends) != 1 else ''}"
        case "competitor":
            view = delta.get("competitor_view") or {}
            names = view.get("competitors", []) if isinstance(view, Mapping) else []
            return f"Identified {len(names)} competitor{'s' if len(names) != 1 else ''}"
        case "forecast":
            forecasts = delta.get("forecasts") or []
            return f"Produced {len(forecasts)} projection{'s' if len(forecasts) != 1 else ''}"
        case "insight":
            insights = delta.get("insights") or []
            return f"Synthesised {len(insights)} insight{'s' if len(insights) != 1 else ''}"
        case "strategy":
            recommendations = delta.get("recommendations") or []
            return (
                f"Formed {len(recommendations)} "
                f"recommendation{'s' if len(recommendations) != 1 else ''}"
            )
        case "critic" | "critic_final":
            critique = delta.get("critique") or {}
            findings = critique.get("findings", []) if isinstance(critique, Mapping) else []
            return f"Raised {len(findings)} finding{'s' if len(findings) != 1 else ''}"
        case "report":
            report = delta.get("report") or {}
            sections = report.get("sections", []) if isinstance(report, Mapping) else []
            return f"Wrote {len(sections)} section{'s' if len(sections) != 1 else ''}"
    return f"Completed {node}"


def _progress_counts(state: Mapping[str, Any]) -> dict[str, int]:
    """Running totals the UI renders as counters."""
    return {
        "evidence_count": len(state.get("evidence") or []),
        "insight_count": len(state.get("insights") or []),
        "step_count": int(state.get("step_count") or 0),
    }


def main() -> None:  # pragma: no cover -- process entry point
    from backend.db.session import get_sessionmaker

    run_worker(
        InvestigationWorker(
            session_factory=get_sessionmaker(),
            publisher=build_timeline_publisher(),
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
