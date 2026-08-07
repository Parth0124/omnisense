"""Process lifecycle for every worker: start, turn, drain, exit.

`services/events/consumer.py` owns one *batch*: fetch, pause, handle, commit,
resume, DLQ. This module owns everything around it -- the loop that drives
batches, the signal handling that ends the loop, the health surface that reports
on it, and the metrics collector that makes lag visible before it becomes an
incident. The split is deliberate: the consumer runtime is a `services/` concern
with no opinion about processes, and this is an L4 process entrypoint that has no
opinion about Kafka semantics.

The drain is the whole point
-----------------------------
`docs/deployment.md` §3.2 fixes the contract: on `SIGTERM`, flip readiness false,
stop polling, finish the in-flight message, commit offsets, close the consumer.
Every clause is load-bearing and the ordering is not negotiable.

*Readiness first.* A worker that keeps advertising itself while draining keeps
receiving partition assignments during a rolling deploy -- the orchestrator hands
it work seconds before killing it.

*Finish, do not cancel.* Cancelling mid-handler is survivable (handlers are
idempotent by contract, ADR-0007) but it is survivable at a cost: the message is
redelivered, its side effects are re-applied, and every re-application is a
chance for a non-idempotent bug nobody has found yet to show itself. A drain that
finishes work costs a second; a drain that abandons it costs a redelivery per
in-flight message per replica per deploy.

*Commit before exit.* An offset is a claim that work was done. Exiting with the
work done and the claim unmade means the whole batch is replayed on restart --
correct, but pure waste, and under a rolling restart of twenty replicas it is
twenty batches of waste per deploy.

This is why `run()` does **not** call `EventConsumer.run()`. That method owns its
own loop and closes its client as soon as `stop()` is called, which would shut the
socket while a batch was still in flight -- the commit at the end of that batch
would then fail against a closed client, and the batch would be replayed despite
having been handled. Driving `run_once()` from here keeps "stop after the current
batch" and "close the client" as two separate events in the right order.

A bounded drain, not an unbounded one
--------------------------------------
`terminationGracePeriodSeconds` is finite; when it expires the orchestrator sends
`SIGKILL`, which is the one outcome with no ordering guarantees at all. So the
drain has its own, shorter deadline: if the in-flight batch has not finished
within `drain_timeout_seconds` the loop task is cancelled, which unwinds through
`EventConsumer.run_once()`'s `except BaseException` -- rewinding the fetch
position so nothing is silently skipped -- and the process exits under its own
control. Losing the race deliberately is strictly better than losing it to
`SIGKILL`.

Two shapes of worker
---------------------
`ConsumerWorker` drives a Kafka consumer group. `PeriodicWorker` drives a timer:
the enrichment sweeper (`workers/embedding_worker.py`), the scheduler and the
report and forecast workers are not event-driven, and forcing them through a
consumer would mean inventing a topic whose only producer is a clock. They share
this module's `BaseWorker` because the lifecycle, the health surface, the signal
handling and the metrics are identical -- only the definition of "one iteration"
differs.

Layer note: `workers/` (L4). May import `services/`, `agents/`, `graph/`,
`retrieval/`, `models/` and the L1k kernel; never `backend/api/`.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import signal
import time
from collections.abc import Mapping, Sequence
from typing import Final

from aiokafka.errors import KafkaError

from backend.core.config import Settings, get_settings
from backend.core.logging import configure_logging, correlation_scope, get_logger
from services.events.consumer import (
    BatchOutcome,
    ConsumedMessage,
    DlqPublisher,
    EventConsumer,
    KafkaConsumerLike,
)
from services.events.topics import TopicRole, topic_name
from workers.runtime.health import (
    DependencyProbe,
    HealthServer,
    HealthState,
    WorkerMetrics,
)

__all__ = [
    "BaseWorker",
    "ConsumerWorker",
    "PeriodicWorker",
    "run_worker",
]

logger = get_logger(__name__)

_DRAIN_TIMEOUT_SECONDS: Final = 45.0
"""How long the in-flight batch has to finish once shutdown is requested.

Below the 60s grace `docs/deployment.md` §3.2 gives an ingestion worker, so the
process wins the race against `SIGKILL` and gets to close its client and its
connection pools. The margin is what teardown runs in.
"""

_COLLECTOR_INTERVAL_SECONDS: Final = 15.0
"""How often gauges are refreshed.

`docs/observability.md` §3.2 forbids sampling a gauge inside a handler -- doing
so makes the value a function of traffic rather than of the thing being measured.
Fifteen seconds is well inside a typical Prometheus scrape interval, so a scrape
never reads a value more than one interval stale.
"""

_BROKER_RETRY_SECONDS: Final = 5.0
"""Pause after a broker-level failure before polling again.

Without it, an unreachable broker turns the loop into a hot loop that emits
thousands of identical log lines a second and does nothing else.
"""

_SHUTDOWN_SIGNALS: Final[tuple[signal.Signals, ...]] = (signal.SIGTERM, signal.SIGINT)
"""`SIGTERM` from the orchestrator, `SIGINT` from a developer's Ctrl-C.

Both mean the same thing here. Handling `SIGINT` matters more than it looks: the
default handler raises `KeyboardInterrupt` from wherever the loop happens to be,
which aborts the in-flight batch and skips the commit -- so a developer testing
locally would see redelivery behaviour that production never exhibits, and would
draw conclusions from it.
"""


class BaseWorker(abc.ABC):
    """Lifecycle, health, metrics and signal handling. Subclass to define work.

    A subclass supplies `_work()` -- the loop -- and optionally `setup()`,
    `teardown()`, `readiness_probes()` and `collect_gauges()`. Everything else,
    including the guarantee that a shutdown drains rather than drops, is here so
    that no worker can implement it slightly differently.
    """

    def __init__(
        self,
        *,
        name: str,
        settings: Settings | None = None,
        metrics: WorkerMetrics | None = None,
        health_state: HealthState | None = None,
        health_server: HealthServer | None = None,
        serve_health: bool = True,
        health_port: int | None = None,
        install_signal_handlers: bool = True,
        drain_timeout_seconds: float = _DRAIN_TIMEOUT_SECONDS,
        collector_interval_seconds: float = _COLLECTOR_INTERVAL_SECONDS,
    ) -> None:
        self.name = name
        self._settings = settings or get_settings()
        self.metrics = metrics or WorkerMetrics(name)
        self.health = health_state or HealthState(worker=name)
        self._serve_health = serve_health
        self._health_port = health_port
        self._health_server = health_server
        self._install_signals = install_signal_handlers
        self._drain_timeout = drain_timeout_seconds
        self._collector_interval = collector_interval_seconds

        self._shutdown = asyncio.Event()
        self._shutdown_reason: str | None = None
        self._installed_signals: list[signal.Signals] = []
        self._collector: asyncio.Task[None] | None = None

    # --------------------------------------------------------------- hooks --

    async def setup(self) -> None:  # noqa: B027 -- an optional hook, not a contract
        """Open whatever the work needs. Called once, before the loop starts.

        Raising here aborts startup, which is the correct response to a
        misconfiguration: a worker that starts, joins a consumer group and then
        fails every message is worse than one that never joins, because the first
        one takes partitions away from healthy replicas.
        """

    async def teardown(self) -> None:  # noqa: B027 -- an optional hook, not a contract
        """Release what `setup()` opened. Runs even when the loop raised.

        Must not raise. It runs in a `finally` while another exception may be
        propagating, and a failure here would replace the interesting error with
        an uninteresting one.
        """

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        """Dependencies `/readyz` should check. Empty by default.

        Deliberately opt-in per worker rather than "probe everything": a graph
        worker's readiness has nothing to do with OpenSearch, and a worker that
        reports itself unready because of a store it never writes to removes
        itself from rotation for someone else's outage.
        """
        return {}

    async def collect_gauges(self) -> None:  # noqa: B027 -- an optional hook
        """Refresh gauge-shaped metrics. Called on a timer, never per message."""

    @abc.abstractmethod
    async def _work(self) -> None:
        """Turn until `self.shutting_down`. Implemented by the two shapes below."""

    # ----------------------------------------------------------- lifecycle --

    @property
    def shutting_down(self) -> bool:
        return self._shutdown.is_set()

    @property
    def shutdown_reason(self) -> str | None:
        return self._shutdown_reason

    def request_shutdown(self, reason: str) -> None:
        """Ask the loop to stop after its current iteration. Safe from a signal.

        Synchronous and allocation-free on purpose: this is called from an
        asyncio signal callback, where an `await` is not available and an
        exception would be reported against the signal handler rather than
        against anything a reader could act on.

        Readiness flips here, not when the loop actually stops. `docs/deployment.md`
        §3.2 requires the worker to stop advertising itself *before* it starts
        draining, or the orchestrator will keep routing work to a process that is
        seconds from exiting.
        """
        if self._shutdown.is_set():
            # A second SIGTERM. Ignored rather than escalated to an immediate
            # exit: an impatient operator sending it twice must not be the thing
            # that abandons an in-flight batch, and the drain is already bounded.
            logger.info("worker.shutdown.repeat_ignored", worker=self.name, reason=reason)
            return
        self._shutdown_reason = reason
        self.health.ready = False
        self.health.draining = True
        self._shutdown.set()
        logger.info("worker.shutdown.requested", worker=self.name, reason=reason)

    async def run(self) -> None:
        """Start, work, drain, stop. The entry point every worker module calls.

        Returns normally on a clean drain and re-raises whatever the loop raised
        otherwise -- the process exit code is how an orchestrator distinguishes
        "asked to stop" from "fell over", and swallowing the exception here would
        make a crash-looping worker look like a healthy one being rescheduled.
        """
        configure_logging(self._settings)
        self._install_signal_handlers()
        started = time.monotonic()
        logger.info("worker.starting", worker=self.name)

        try:
            await self.setup()
            await self._start_health_server()
            self._collector = asyncio.create_task(
                self._collect_loop(), name=f"{self.name}-collector"
            )
            await self._run_until_drained()
        finally:
            self.health.ready = False
            self.health.started = False
            await self._cancel_collector()
            await self._stop_health_server()
            await self._safe_teardown()
            self._remove_signal_handlers()
            logger.info(
                "worker.stopped",
                worker=self.name,
                reason=self._shutdown_reason or "loop_exited",
                uptime_seconds=round(time.monotonic() - started, 3),
                handled=self.health.messages_handled,
                dlq_routed=self.health.messages_dlq_routed,
            )

    async def _run_until_drained(self) -> None:
        """Run `_work()` and, on a shutdown request, give it a bounded drain.

        The loop runs as a task rather than being awaited directly so that the
        shutdown request -- which arrives on a *different* task, from a signal
        callback -- can be observed while the loop is inside an await. Awaiting
        `_work()` directly would mean the timeout could only be applied after it
        returned, which is exactly when it is no longer needed.
        """
        self.health.started = True
        work = asyncio.create_task(self._work(), name=f"{self.name}-loop")
        stop = asyncio.create_task(self._shutdown.wait(), name=f"{self.name}-shutdown")

        try:
            await asyncio.wait({work, stop}, return_when=asyncio.FIRST_COMPLETED)
            if work.done():
                # The loop exited on its own -- either it noticed the shutdown
                # flag between iterations, or it raised. Either way there is
                # nothing left to drain.
                await work
                return

            # Shutdown requested with the loop still turning. It checks the flag
            # between iterations, so the wait below is exactly the time the
            # in-flight batch needs to finish and commit.
            logger.info(
                "worker.draining",
                worker=self.name,
                reason=self._shutdown_reason,
                timeout_seconds=self._drain_timeout,
            )
            await asyncio.wait({work}, timeout=self._drain_timeout)
            if not work.done():
                # The drain deadline beat the handler. Cancelling unwinds through
                # `EventConsumer.run_once()`, which rewinds the fetch position
                # and commits nothing -- so the batch is redelivered rather than
                # silently skipped. Strictly better than being SIGKILLed here,
                # where the client would be closed by the kernel mid-commit.
                logger.warning(
                    "worker.drain_timeout",
                    worker=self.name,
                    timeout_seconds=self._drain_timeout,
                    outcome="cancelled_in_flight_batch",
                )
                work.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await work
                return
            await work
        finally:
            stop.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop

    # -------------------------------------------------------------- signals --

    def _install_signal_handlers(self) -> None:
        """Route `SIGTERM`/`SIGINT` into `request_shutdown()`.

        `loop.add_signal_handler` rather than `signal.signal`: the C-level
        handler runs on whatever thread the OS picks and can interrupt a syscall
        mid-write, while the asyncio variant wakes the loop through its self-pipe
        and runs the callback as a normal loop callback. That difference is what
        makes "finish the in-flight batch" implementable at all.

        Not available on every platform -- Windows has no `add_signal_handler` --
        so failure is logged and the worker runs without it. A worker that
        refused to start because it could not register a shutdown handler would
        be unusable in exactly the environment where a developer is trying to
        reproduce something.
        """
        if not self._install_signals:
            return
        loop = asyncio.get_running_loop()
        for sig in _SHUTDOWN_SIGNALS:
            try:
                loop.add_signal_handler(sig, self.request_shutdown, f"signal:{sig.name}")
            except (NotImplementedError, RuntimeError, ValueError):
                logger.warning(
                    "worker.signal_handler_unavailable", worker=self.name, signal=sig.name
                )
                continue
            self._installed_signals.append(sig)

    def _remove_signal_handlers(self) -> None:
        """Restore the previous disposition. Only for handlers we installed.

        Load-bearing in a test suite and in any process that runs a worker inside
        a larger program: an asyncio signal handler outlives the loop that
        registered it, so a leaked one would fire into a closed loop the next
        time a signal arrived.
        """
        if not self._installed_signals:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover -- only if run() is driven oddly
            self._installed_signals.clear()
            return
        for sig in self._installed_signals:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.remove_signal_handler(sig)
        self._installed_signals.clear()

    # --------------------------------------------------------------- health --

    async def _start_health_server(self) -> None:
        if not self._serve_health:
            return
        if self._health_server is None:
            self._health_server = HealthServer(
                self.health,
                self.metrics,
                probes=self.readiness_probes(),
                port=self._health_port,
                settings=self._settings,
            )
        await self._health_server.start()

    async def _stop_health_server(self) -> None:
        if self._health_server is not None:
            await self._health_server.stop()

    async def _safe_teardown(self) -> None:
        """Run `teardown()`, converting a failure into a log line.

        See `teardown()`'s contract. This is the enforcement: a subclass that
        raises there would otherwise mask the exception that caused the shutdown.
        """
        try:
            await self.teardown()
        except Exception as err:  # noqa: BLE001 -- must not mask the original failure
            logger.error("worker.teardown_failed", worker=self.name, error=type(err).__name__)

    async def _collect_loop(self) -> None:
        """Refresh gauges on a timer until shutdown.

        Failures are logged and the loop continues. A collector that died on one
        unreachable dependency would take every gauge with it, including the ones
        that would have explained the outage.
        """
        while not self._shutdown.is_set():
            try:
                self.metrics.observe_state(self.health)
                await self.collect_gauges()
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 -- metrics must never stop the worker
                logger.warning(
                    "worker.collector_failed", worker=self.name, error=type(err).__name__
                )
            await self._sleep_or_stop(self._collector_interval)

    async def _cancel_collector(self) -> None:
        collector, self._collector = self._collector, None
        if collector is None:
            return
        collector.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await collector

    # ------------------------------------------------------------ internals --

    async def _sleep_or_stop(self, seconds: float) -> None:
        """Sleep, but wake immediately if shutdown was requested.

        A plain `asyncio.sleep` on a retry or tick path would make a graceful
        shutdown wait out the full interval, which for the scheduler's hourly
        tick is the difference between a two-second restart and a `SIGKILL`.
        """
        if seconds <= 0:
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._shutdown.wait(), timeout=seconds)


class ConsumerWorker(BaseWorker):
    """A worker whose iteration is one Kafka fetch/handle/commit cycle.

    The subclass supplies `topics` and `handle()`. Everything about at-least-once
    delivery -- commit after the handler, bounded retries, DLQ on poison, block
    rather than drop when the DLQ itself is unreachable -- belongs to
    `services/events/consumer.py` and is not re-implemented here.
    """

    def __init__(
        self,
        *,
        name: str,
        topics: Sequence[TopicRole | str],
        group_id: str | None = None,
        consumer: EventConsumer | None = None,
        kafka_consumer: KafkaConsumerLike | None = None,
        dlq_publisher: DlqPublisher | None = None,
        max_attempts: int = 3,
        batch_size: int | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, **kwargs)  # type: ignore[arg-type]
        self._topics = tuple(topics)
        # Per-worker consumer group by default. Sharing one group across worker
        # families would make them compete for the same partitions: the indexing
        # worker would consume a message the graph worker then never sees, and
        # the symptom -- "some signals are in Neo4j and some are in Qdrant, at
        # random" -- looks nothing like its cause.
        self._group_id = group_id or f"{self._settings.kafka.consumer_group}.{name}"
        self._consumer = consumer or EventConsumer(
            topics=self._topics,
            handler=self._dispatch,
            group_id=self._group_id,
            consumer=kafka_consumer,
            dlq_publisher=dlq_publisher,
            max_attempts=max_attempts,
            batch_size=batch_size,
            settings=self._settings,
        )

    @property
    def consumer(self) -> EventConsumer:
        return self._consumer

    @property
    def group_id(self) -> str:
        return self._group_id

    @abc.abstractmethod
    async def handle(self, message: ConsumedMessage) -> None:
        """Process one message. Raise to report failure.

        **Must be idempotent.** At-least-once delivery guarantees redelivery
        after any crash, rebalance or uncommitted batch, and every derived store
        in this system is written with a stable derived id precisely so that a
        redelivery upserts instead of duplicating (`docs/data-stores.md` §5.2).
        A handler that appends, increments or inserts without a key is a handler
        that produces a different result the second time.

        Raising is the only correct way to fail. Swallowing an exception advances
        the offset past work that was never done, which is indistinguishable from
        success in every metric there is.
        """

    async def _dispatch(self, message: ConsumedMessage) -> None:
        """Bind the correlation id and time the handler.

        The correlation id comes from the *envelope*, never minted here.
        `docs/observability.md` §1 makes it the join key across logs, metrics,
        traces and events; a worker that minted its own would break the chain at
        precisely the boundary where an investigation crosses from the API into
        the pipeline.
        """
        started = time.perf_counter()
        with correlation_scope(message.correlation_id):
            try:
                await self.handle(message)
            finally:
                self.metrics.handler_duration.labels(worker=self.name).observe(
                    time.perf_counter() - started
                )

    async def _work(self) -> None:
        """Drive `run_once()` until shutdown, then close the client.

        Not `EventConsumer.run()`. See the module docstring: that method closes
        its client the moment `stop()` is called, which on a real (non-injected)
        consumer would shut the socket out from under a batch that is still
        trying to commit.
        """
        await self._consumer.start()
        self.health.ready = True
        self._refresh_assignment()
        logger.info(
            "worker.consuming",
            worker=self.name,
            group=self._group_id,
            topics=[
                topic_name(t, settings=self._settings) if isinstance(t, TopicRole) else t
                for t in self._topics
            ],
        )

        try:
            while not self.shutting_down:
                await self._one_batch()
        finally:
            # After the loop, never during it. The in-flight batch has finished
            # and committed by the time control reaches here.
            await self._consumer.stop()

    async def _one_batch(self) -> None:
        """One cycle, with broker failures absorbed rather than propagated.

        A worker that exited on a transient broker error would be restarted by
        the orchestrator into exactly the same condition, having lost its group
        membership and triggered a rebalance for every other member of the group.
        """
        started = time.perf_counter()
        try:
            outcome = await self._consumer.run_once()
        except (KafkaError, OSError) as err:
            self.health.last_error = type(err).__name__
            logger.warning(
                "worker.broker_unavailable",
                worker=self.name,
                group=self._group_id,
                error=type(err).__name__,
                retry_in_seconds=_BROKER_RETRY_SECONDS,
            )
            # Heartbeat anyway: the loop *is* turning, and a worker that cannot
            # reach its broker for two minutes must not also be restarted for
            # appearing dead. Liveness answers "is this loop alive", and readiness
            # is the flag that reflects "can it do useful work".
            self.health.heartbeat()
            await self._sleep_or_stop(_BROKER_RETRY_SECONDS)
            return

        self.health.heartbeat()
        self._record(outcome, time.perf_counter() - started)

    def _record(self, outcome: BatchOutcome, elapsed: float) -> None:
        """Fold one batch's outcome into the health state and the registry."""
        self.metrics.batch_duration.labels(worker=self.name).observe(elapsed)
        self.metrics.batches.labels(
            worker=self.name, state="idle" if outcome.is_empty else "active"
        ).inc()
        if outcome.is_empty:
            return

        # `BatchOutcome.handled` counts every message whose offset advanced,
        # which includes the ones parked in the DLQ. Subtracting keeps `handled`
        # meaning "actually processed" in the metric, where an operator reads it
        # as a throughput number.
        processed = max(0, outcome.handled - outcome.dlq_routed)
        if processed:
            self.metrics.messages.labels(worker=self.name, outcome="handled").inc(processed)
        if outcome.dlq_routed:
            self.metrics.messages.labels(worker=self.name, outcome="dlq").inc(outcome.dlq_routed)
        if outcome.blocked:
            self.metrics.messages.labels(worker=self.name, outcome="blocked").inc(outcome.blocked)

        self.health.messages_handled += processed
        self.health.messages_dlq_routed += outcome.dlq_routed
        self._refresh_assignment()

        logger.debug(
            "worker.batch",
            worker=self.name,
            fetched=outcome.fetched,
            handled=processed,
            dlq_routed=outcome.dlq_routed,
            blocked=outcome.blocked,
            duration_ms=round(elapsed * 1000, 2),
        )

    def _refresh_assignment(self) -> None:
        """Record how many partitions this member holds, for `/readyz`.

        A consumer with zero partitions is joined but idle -- which is normal
        with more replicas than partitions and pathological otherwise -- so it is
        reported rather than treated as an error. Failures are swallowed because
        this is telemetry: `assignment()` touches client state that does not exist
        before the first successful join.
        """
        with contextlib.suppress(Exception):
            client = self._consumer_client()
            if client is not None:
                self.health.assigned_partitions = len(client.assignment())

    def _consumer_client(self) -> KafkaConsumerLike | None:
        return getattr(self._consumer, "_consumer", None)


class PeriodicWorker(BaseWorker):
    """A worker whose iteration is a tick rather than a message.

    The sweepers and the scheduler are driven by a clock, not by an event, and
    modelling them as consumers would mean inventing a topic whose only producer
    is a timer -- one more thing to provision, monitor and reason about, in
    exchange for nothing.

    The tick contract mirrors the handler contract: `tick()` raises to report
    failure, the runtime logs it and keeps the loop alive, and the next tick is
    expected to converge. A sweeper that stopped on its first failure would be a
    sweeper that stops permanently the first time PostgreSQL restarts.
    """

    def __init__(
        self,
        *,
        name: str,
        interval_seconds: float,
        initial_delay_seconds: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(name=name, **kwargs)  # type: ignore[arg-type]
        if interval_seconds <= 0:
            raise ValueError(
                f"interval_seconds must be positive, got {interval_seconds}; a "
                "zero interval is a hot loop, not a schedule"
            )
        self.interval_seconds = interval_seconds
        self._initial_delay = initial_delay_seconds
        self.ticks = 0
        self.tick_failures = 0

    @abc.abstractmethod
    async def tick(self) -> None:
        """Do one pass of work. Raise to report failure.

        **Must be idempotent and re-entrant across replicas.** Two replicas of a
        sweeper will tick at the same instant sooner or later; whatever makes
        that safe -- an advisory lock, a conditional update, a derived id -- is
        the tick's own responsibility.
        """

    async def _work(self) -> None:
        self.health.ready = True
        logger.info(
            "worker.ticking",
            worker=self.name,
            interval_seconds=self.interval_seconds,
        )
        # An initial delay staggers replicas that all started at the same instant
        # during a rolling deploy. Without it, every replica's first tick lands
        # on the same second and contends for the same advisory lock.
        await self._sleep_or_stop(self._initial_delay)

        while not self.shutting_down:
            started = time.perf_counter()
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as err:  # noqa: BLE001 -- one bad tick must not end the loop
                self.tick_failures += 1
                self.health.last_error = type(err).__name__
                logger.error(
                    "worker.tick_failed",
                    worker=self.name,
                    error=type(err).__name__,
                    consecutive_context=self.tick_failures,
                    exc_info=True,
                )
            else:
                self.ticks += 1
            finally:
                elapsed = time.perf_counter() - started
                self.metrics.batch_duration.labels(worker=self.name).observe(elapsed)
                self.metrics.batches.labels(worker=self.name, state="active").inc()
                self.health.heartbeat()

            await self._sleep_or_stop(self.interval_seconds)


def run_worker(worker: BaseWorker) -> None:
    """Run one worker to completion. What every `__main__` block calls.

    `asyncio.run` rather than a hand-managed loop: it installs a fresh loop,
    cancels lingering tasks and closes the async generators on the way out, all
    of which a worker that opens driver connections actually needs. The signal
    handlers are installed *inside* the loop by `run()` rather than here, because
    `loop.add_signal_handler` requires a running loop.
    """
    asyncio.run(worker.run())
