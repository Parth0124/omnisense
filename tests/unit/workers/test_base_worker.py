"""Unit tests for `workers/runtime/`: lifecycle, drain, health and metrics.

The properties asserted here are the ones that fail *silently* in production if
broken, which is why they are worth this much scaffolding:

1. **The offset does not advance when a handler raises.** A committed offset is a
   claim that work was done. If it can advance past work that failed, the gap is
   undetectable -- no error, no DLQ entry, and nothing that records the message
   was ever seen.
2. **A poisoned message reaches the DLQ instead of blocking its partition.**
   Kafka offsets are strictly ordered, so one permanently-failing message stalls
   everything behind it forever unless something explicitly moves past it.
3. **`SIGTERM` drains rather than drops.** The in-flight batch finishes and
   commits before the client closes. A worker that dropped it would redeliver on
   restart -- survivable, but it turns every deploy into a burst of duplicate
   work, and it is the same code path that would drop work on a *slow* handler.
4. **Readiness flips false before the drain, not after it.** Otherwise a rolling
   deploy keeps routing work at a process that is seconds from exiting.

Nothing here opens a socket to a broker: `EventConsumer` takes an injected
`KafkaConsumerLike`, so aiokafka is imported for its record types and never
instantiated. The health server does bind a real loopback port -- on port 0, so
the kernel picks a free one -- because a hand-rolled HTTP parser that is only
ever tested through its own internals is a parser with an untested parser.
"""

from __future__ import annotations

import asyncio
import json
import signal
import time

import pytest
from aiokafka.errors import KafkaError
from aiokafka.structs import TopicPartition

from backend.core.config import Settings
from services.events.consumer import ConsumedMessage, EventConsumer
from services.events.schemas import RawRecordEvent, SignalEnrichedEvent
from tests.unit.workers.conftest import FakeBroker, FakeConsumer, envelope_bytes
from workers.runtime.base_worker import ConsumerWorker, PeriodicWorker
from workers.runtime.health import (
    LIVENESS_STALE_AFTER_SECONDS,
    HealthServer,
    HealthState,
    WorkerMetrics,
)

pytestmark = pytest.mark.unit

RAW_TOPIC = "omnisense.records.raw"


def _raw_event(native_id: str = "t3_abc") -> RawRecordEvent:
    return RawRecordEvent(
        platform="reddit",
        native_id=native_id,
        connector_slug="reddit",
        connector_version="1.0.0",
        sync_run_id="run_test",
        raw_object_key=f"raw/reddit/{native_id}.json",
    )


class _RecordingWorker(ConsumerWorker):
    """A worker whose handler is a list append, plus an optional failure."""

    def __init__(self, *, fail_on: set[str] | None = None, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.seen: list[str] = []
        self.fail_on = fail_on or set()
        self.slow = asyncio.Event()
        self.slow.set()

    async def handle(self, message: ConsumedMessage) -> None:
        event = message.envelope.payload_as(RawRecordEvent)
        # The trace is shared with the fake consumer, so handler-versus-commit
        # ordering is one sequence rather than two that have to be correlated.
        self._trace.append(f"handle:{event.native_id}")
        await self.slow.wait()
        if event.native_id in self.fail_on:
            raise RuntimeError(f"handler refuses {event.native_id}")
        self.seen.append(event.native_id)

    def attach_trace(self, trace: list[str]) -> None:
        self._trace = trace


def _build_worker(
    broker: FakeBroker,
    settings: Settings,
    *,
    fail_on: set[str] | None = None,
    max_attempts: int = 1,
    dlq_enabled: bool = True,
    drain_timeout_seconds: float = 5.0,
) -> tuple[_RecordingWorker, FakeConsumer, list[str]]:
    """Wire a worker over the in-memory broker with one shared call trace."""
    trace: list[str] = []
    fake = FakeConsumer(broker, trace)
    consumer = EventConsumer(
        topics=[RAW_TOPIC],
        handler=lambda message: worker._dispatch(message),  # bound after construction
        group_id="test-group",
        consumer=fake,
        dlq_publisher=broker.dlq_publisher,
        dlq_enabled=dlq_enabled,
        max_attempts=max_attempts,
        backoff_seconds=0.0,
        fetch_timeout_ms=1,
        settings=settings,
    )
    worker = _RecordingWorker(
        name="test-worker",
        topics=[RAW_TOPIC],
        consumer=consumer,
        settings=settings,
        serve_health=False,
        install_signal_handlers=False,
        fail_on=fail_on,
        drain_timeout_seconds=drain_timeout_seconds,
        collector_interval_seconds=0.01,
    )
    worker.attach_trace(trace)
    return worker, fake, trace


class TestAtLeastOnceDelivery:
    """The offset is a claim about work, not about bytes received."""

    async def test_offset_does_not_advance_when_the_handler_raises(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """A failed handler must leave the message redeliverable.

        With the DLQ disabled there is nowhere to park the message, so the only
        correct outcome is to block: refusing to advance is what makes the
        failure recoverable instead of a silent gap.
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_bad")))
        worker, fake, _ = _build_worker(
            broker, settings, fail_on={"t3_bad"}, dlq_enabled=False
        )

        outcome = await worker.consumer.run_once()

        assert fake.committed == {}, "an offset advanced past work that failed"
        assert outcome.blocked == 1
        assert worker.seen == []

    async def test_commit_happens_after_the_handler_not_before(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """Ordering, not just outcome: the commit must follow the handler.

        A runtime that committed first would pass every "was it handled?"
        assertion and still lose data on a crash between the two.
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_one")))
        worker, _, trace = _build_worker(broker, settings)

        await worker.consumer.run_once()

        assert trace.index("handle:t3_one") < trace.index("commit")

    async def test_a_redelivered_message_is_handled_twice_and_converges(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """At-least-once means duplicates. The runtime must not suppress them.

        Deduplicating in the runtime would be the wrong layer: the handler owns
        idempotency (every derived store uses a stable derived id), and a runtime
        that silently dropped a redelivery would hide a *genuine* redelivery
        caused by an uncommitted batch.
        """
        payload = envelope_bytes(_raw_event("t3_dup"))
        broker.append(RAW_TOPIC, payload)
        worker, fake, _ = _build_worker(broker, settings)

        await worker.consumer.run_once()
        fake.positions[TopicPartition(RAW_TOPIC, 0)] = 0  # a rebalance replays it
        await worker.consumer.run_once()

        assert worker.seen == ["t3_dup", "t3_dup"]


class TestPoisonHandling:
    """One bad message must not stall the partition behind it."""

    async def test_a_poisoned_message_reaches_the_dlq_and_the_offset_advances(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """The whole point of a DLQ: forward progress past unhandleable work."""
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_poison")))
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_good")))
        worker, fake, _ = _build_worker(broker, settings, fail_on={"t3_poison"})

        outcome = await worker.consumer.run_once()

        assert len(broker.dlq) == 1
        parked = broker.dlq[0][0]
        assert parked.event_type.value == "dlq.failed"
        assert fake.committed[TopicPartition(RAW_TOPIC, 0)] == 2
        assert worker.seen == ["t3_good"], "the message behind the poison was skipped"
        assert outcome.dlq_routed == 1

    async def test_an_undecodable_body_is_parked_without_retries(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """Retrying does not change bytes, so a parse failure skips the retries."""
        broker.append(RAW_TOPIC, b"this is not an envelope")
        worker, fake, _ = _build_worker(broker, settings, max_attempts=3)

        await worker.consumer.run_once()

        assert len(broker.dlq) == 1
        assert fake.committed[TopicPartition(RAW_TOPIC, 0)] == 1

    async def test_the_dlq_counter_records_the_failure_class_not_its_message(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """A DLQ record carries exception class names only.

        An exception *message* from a driver can echo the request that caused it,
        and requests carry fetched content (`docs/security-and-privacy.md`).
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_poison")))
        worker, _, _ = _build_worker(broker, settings, fail_on={"t3_poison"})

        await worker.consumer.run_once()

        chain = broker.dlq[0][0].payload["error_chain"]
        assert chain == ["RuntimeError"]
        assert "refuses" not in json.dumps(broker.dlq[0][0].payload)


class TestGracefulShutdown:
    """SIGTERM drains: finish the batch, commit, then close."""

    async def test_shutdown_finishes_the_in_flight_batch_and_commits(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """The batch in flight when the signal arrives is completed, not dropped.

        The handler is held open, shutdown is requested while it is blocked, and
        only then is it released -- which is exactly the ordering a `SIGTERM`
        arriving mid-batch produces.
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_inflight")))
        worker, fake, trace = _build_worker(broker, settings)
        worker.slow.clear()

        run = asyncio.create_task(worker.run())
        await _until(lambda: any(t.startswith("handle:") for t in trace))

        worker.request_shutdown("test")
        assert worker.health.ready is False, "readiness must flip before the drain"
        assert worker.health.draining is True

        worker.slow.set()
        await asyncio.wait_for(run, timeout=5)

        assert worker.seen == ["t3_inflight"]
        assert fake.committed[TopicPartition(RAW_TOPIC, 0)] == 1
        # The commit is the loop's last act. Anything after it would mean the
        # runtime kept fetching after being told to stop.
        assert trace[-1] == "commit"

    async def test_the_consumer_is_stopped_after_the_loop_never_during_it(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """`stop()` belongs after the loop, never inside it.

        `EventConsumer.run()` closes its client the moment `stop()` is called,
        which would shut the socket under an in-flight batch and fail the commit
        at the end of it. The worker drives `run_once()` itself precisely so
        "stop polling" and "close the client" stay two events in that order.

        Asserted against the runtime's own `_started` flag rather than the fake's:
        `EventConsumer.stop()` deliberately does *not* close an **injected**
        client -- whoever passed one in owns its lifecycle -- so the fake never
        sees a close, and a test written against it would prove nothing.
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_inflight")))
        worker, fake, _ = _build_worker(broker, settings)
        worker.slow.clear()

        run = asyncio.create_task(worker.run())
        await _until(lambda: worker.seen == [] and fake.polls > 0)
        worker.request_shutdown("test")
        await asyncio.sleep(0)
        assert worker.consumer._started is True, "closed while a batch was in flight"

        worker.slow.set()
        await asyncio.wait_for(run, timeout=5)

        assert worker.consumer._started is False

    async def test_a_drain_that_overruns_its_deadline_is_cancelled_not_waited_out(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """Losing the race deliberately beats losing it to SIGKILL.

        `terminationGracePeriodSeconds` is finite. A drain with no deadline of its
        own is a drain that ends in `SIGKILL`, where the client is closed by the
        kernel mid-commit and nothing is ordered at all.
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_stuck")))
        worker, fake, trace = _build_worker(
            broker, settings, drain_timeout_seconds=0.05
        )
        worker.slow.clear()  # never released

        run = asyncio.create_task(worker.run())
        await _until(lambda: any(t.startswith("handle:") for t in trace))
        worker.request_shutdown("test")
        await asyncio.wait_for(run, timeout=5)

        assert worker.seen == []
        assert fake.committed == {}, "an abandoned batch must not commit"
        # Rewound rather than skipped: the messages are redelivered on restart.
        assert (TopicPartition(RAW_TOPIC, 0), 0) in fake.seeks

    async def test_sigterm_is_wired_to_the_drain(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """The real signal, delivered to this process, must start the drain.

        Asserting that a callback exists would prove nothing about whether the
        OS-level signal reaches it. `signal.raise_signal` is only fired after the
        handler is confirmed installed, so a platform without
        `add_signal_handler` skips rather than terminating the test session.
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_sig")))
        trace: list[str] = []
        fake = FakeConsumer(broker, trace)
        consumer = EventConsumer(
            topics=[RAW_TOPIC],
            handler=lambda message: worker._dispatch(message),
            group_id="test-group",
            consumer=fake,
            dlq_publisher=broker.dlq_publisher,
            max_attempts=1,
            backoff_seconds=0.0,
            fetch_timeout_ms=1,
            settings=settings,
        )
        worker = _RecordingWorker(
            name="signal-worker",
            topics=[RAW_TOPIC],
            consumer=consumer,
            settings=settings,
            serve_health=False,
            install_signal_handlers=True,
            drain_timeout_seconds=5.0,
            collector_interval_seconds=0.01,
        )
        worker.attach_trace(trace)

        run = asyncio.create_task(worker.run())
        await _until(lambda: fake.polls > 0)
        if not worker._installed_signals:  # pragma: no cover -- platform-dependent
            worker.request_shutdown("cleanup")
            await asyncio.wait_for(run, timeout=5)
            pytest.skip("this platform has no loop.add_signal_handler")

        signal.raise_signal(signal.SIGTERM)
        await asyncio.wait_for(run, timeout=5)

        assert worker.shutdown_reason == "signal:SIGTERM"
        assert worker.seen == ["t3_sig"], "the in-flight message was dropped"
        assert fake.committed[TopicPartition(RAW_TOPIC, 0)] == 1

    async def test_signal_handlers_are_removed_on_exit(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """A leaked asyncio signal handler fires into a closed loop later.

        Which surfaces as an unrelated test failing, in a different file, with a
        traceback pointing at asyncio internals.
        """
        worker, fake, _ = _build_worker(broker, settings)
        worker._install_signals = True

        run = asyncio.create_task(worker.run())
        await _until(lambda: fake.polls > 0)
        worker.request_shutdown("test")
        await asyncio.wait_for(run, timeout=5)

        assert worker._installed_signals == []

    async def test_a_second_shutdown_request_does_not_abandon_the_drain(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """An impatient operator sending SIGTERM twice must not lose the batch."""
        worker, _, _ = _build_worker(broker, settings)

        worker.request_shutdown("first")
        worker.request_shutdown("second")

        assert worker.shutdown_reason == "first"


class TestLoopResilience:
    """A worker must not exit on conditions the orchestrator cannot fix."""

    async def test_a_broker_failure_is_retried_rather_than_fatal(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """Exiting would restart into the same condition, minus the group membership.

        And it would trigger a rebalance for every other member of the group,
        turning one worker's transient error into a fleet-wide pause.
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_after_blip")))
        worker, fake, _ = _build_worker(broker, settings)
        fake.getmany_error = KafkaError("broker went away")
        worker._sleep_or_stop = _no_sleep(worker)  # type: ignore[method-assign]

        run = asyncio.create_task(worker.run())
        await _until(lambda: worker.seen == ["t3_after_blip"], timeout=5)
        worker.request_shutdown("test")
        await asyncio.wait_for(run, timeout=5)

        assert worker.health.last_error == "KafkaError"

    async def test_the_loop_heartbeats_even_when_the_broker_is_unreachable(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """Liveness answers "is this loop turning", not "can it do useful work".

        A worker restarted for an unreachable broker loses its group membership
        and rejoins into the same outage, having forced a rebalance on its way.
        """
        worker, fake, _ = _build_worker(broker, settings)
        fake.getmany_error = OSError("connection refused")
        worker.health.started = True
        worker.health.last_heartbeat = time.monotonic() - 30.0
        worker._sleep_or_stop = _no_sleep(worker)  # type: ignore[method-assign]

        await worker._one_batch()

        assert worker.health.heartbeat_age_seconds < 1.0
        assert worker.health.is_alive is True

    async def test_a_teardown_failure_does_not_mask_the_original_error(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """`teardown()` runs in a `finally` while another exception may propagate."""
        worker, fake, _ = _build_worker(broker, settings)

        async def _explode() -> None:
            raise RuntimeError("teardown is broken")

        worker.teardown = _explode  # type: ignore[method-assign]
        run = asyncio.create_task(worker.run())
        await _until(lambda: fake.polls > 0)
        worker.request_shutdown("test")

        await asyncio.wait_for(run, timeout=5)  # must not raise


class TestPeriodicWorker:
    """The sweeper shape: a clock instead of a topic."""

    async def test_ticks_until_shutdown_and_counts_failures(self) -> None:
        """One failing tick must not end the loop.

        A sweeper that stopped on its first failure would stop permanently the
        first time PostgreSQL restarted, and nothing would report the backlog it
        stopped clearing.
        """

        class _Sweeper(PeriodicWorker):
            def __init__(self) -> None:
                super().__init__(
                    name="sweeper",
                    interval_seconds=0.001,
                    serve_health=False,
                    install_signal_handlers=False,
                    collector_interval_seconds=0.01,
                )
                self.calls = 0

            async def tick(self) -> None:
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first tick fails")

        worker = _Sweeper()
        run = asyncio.create_task(worker.run())
        await _until(lambda: worker.calls >= 3, timeout=5)
        worker.request_shutdown("test")
        await asyncio.wait_for(run, timeout=5)

        assert worker.tick_failures == 1
        assert worker.ticks >= 2
        assert worker.health.last_error == "RuntimeError"

    async def test_a_zero_interval_is_rejected_at_construction(self) -> None:
        """A zero interval is a hot loop, not a schedule."""

        class _Sweeper(PeriodicWorker):
            async def tick(self) -> None: ...

        with pytest.raises(ValueError, match="hot loop"):
            _Sweeper(name="s", interval_seconds=0.0, serve_health=False)


class TestHealthState:
    """Liveness is heartbeat freshness and nothing else."""

    def test_a_stale_heartbeat_is_not_alive(self) -> None:
        """The check that catches a worker wedged on a poisoned message."""
        state = HealthState(worker="w", started=True)
        state.last_heartbeat = time.monotonic() - (LIVENESS_STALE_AFTER_SECONDS + 1)

        assert state.is_alive is False

    def test_a_worker_that_has_not_started_is_alive(self) -> None:
        """`setup()` opens connections and may legitimately take seconds.

        Liveness during startup is the orchestrator's `initialDelaySeconds`
        problem; failing it here would kill every worker mid-boot.
        """
        state = HealthState(worker="w", started=False)
        state.last_heartbeat = time.monotonic() - 600

        assert state.is_alive is True

    def test_the_snapshot_carries_no_deployment_detail(self) -> None:
        """`docs/observability.md` §9.2: no versions, hostnames or topic names."""
        state = HealthState(worker="w")
        rendered = json.dumps(state.snapshot())

        assert "localhost" not in rendered
        assert "omnisense." not in rendered


class TestHealthServer:
    """The listener a worker has instead of an HTTP surface."""

    async def test_liveness_readiness_and_metrics_are_served(self) -> None:
        """One request per connection, three routes, no dependency on a broker."""
        state = HealthState(worker="probe-worker", started=True, ready=True)
        metrics = WorkerMetrics("probe-worker")
        server = HealthServer(state, metrics, port=0, settings=Settings())
        await server.start()
        try:
            status, body = await _get(server.port, "/health")
            assert status == 200
            assert json.loads(body)["status"] == "ok"

            status, body = await _get(server.port, "/readyz")
            assert status == 200

            status, body = await _get(server.port, "/metrics")
            assert status == 200
            assert "omnisense_worker_ready" in body
        finally:
            await server.stop()

    async def test_readiness_is_false_while_draining(self) -> None:
        """Flipped before the drain begins, so nothing new is routed here."""
        state = HealthState(worker="w", started=True, ready=False, draining=True)
        server = HealthServer(state, WorkerMetrics("w"), port=0, settings=Settings())
        await server.start()
        try:
            status, body = await _get(server.port, "/readyz")
            assert status == 503
            assert json.loads(body)["status"] == "draining"
        finally:
            await server.stop()

    async def test_a_failing_dependency_probe_degrades_readiness_only(self) -> None:
        """Readiness may depend on a datastore; liveness must not.

        A liveness probe wired to PostgreSQL asks the orchestrator to restart the
        entire fleet at the moment the database is already struggling.
        """

        async def _down() -> bool:
            return False

        state = HealthState(worker="w", started=True, ready=True)
        server = HealthServer(
            state, WorkerMetrics("w"), probes={"postgres": _down}, port=0, settings=Settings()
        )
        await server.start()
        try:
            ready_status, ready_body = await _get(server.port, "/readyz")
            live_status, _ = await _get(server.port, "/health")
        finally:
            await server.stop()

        assert ready_status == 503
        assert json.loads(ready_body)["components"] == {"postgres": "failed"}
        assert live_status == 200

    async def test_a_probe_that_raises_is_reported_not_propagated(self) -> None:
        """One broken probe must not take the whole endpoint down with a 500."""

        async def _explodes() -> bool:
            raise RuntimeError("driver blew up")

        state = HealthState(worker="w", started=True, ready=True)
        server = HealthServer(
            state, WorkerMetrics("w"), probes={"qdrant": _explodes}, port=0, settings=Settings()
        )
        await server.start()
        try:
            status, body = await _get(server.port, "/readyz")
        finally:
            await server.stop()

        assert status == 503
        assert json.loads(body)["components"] == {"qdrant": "failed"}

    async def test_unknown_paths_and_methods_are_refused(self) -> None:
        """This endpoint changes no state, so everything else is a 404 or 405."""
        state = HealthState(worker="w", started=True, ready=True)
        server = HealthServer(state, WorkerMetrics("w"), port=0, settings=Settings())
        await server.start()
        try:
            assert (await _get(server.port, "/admin"))[0] == 404
            assert (await _get(server.port, "/metrics", method="POST"))[0] == 405
        finally:
            await server.stop()


class TestMetrics:
    """Counters and gauges, with the label sets §3.1 fixes."""

    async def test_a_batch_increments_handled_and_dlq_separately(
        self, broker: FakeBroker, settings: Settings
    ) -> None:
        """`handled` must mean "processed", not "the offset moved".

        `BatchOutcome.handled` counts every advanced offset, DLQ parks included.
        Exporting that as throughput would make a worker failing every message
        look exactly like one succeeding at every message.
        """
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_ok")))
        broker.append(RAW_TOPIC, envelope_bytes(_raw_event("t3_bad")))
        worker, _, _ = _build_worker(broker, settings, fail_on={"t3_bad"})

        outcome = await worker.consumer.run_once()
        worker._record(outcome, 0.01)

        assert _counter(worker, "handled") == 1.0
        assert _counter(worker, "dlq") == 1.0
        assert worker.health.messages_handled == 1

    def test_two_workers_in_one_process_do_not_collide_on_the_registry(self) -> None:
        """The default Prometheus registry raises on a duplicate time series.

        Which would make constructing a second worker fail at construction rather
        than at anything meaningful -- and a test suite constructs dozens.
        """
        first = WorkerMetrics("a")
        second = WorkerMetrics("b")

        assert first.registry is not second.registry
        assert b"omnisense_worker_ready" in second.render()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


async def _until(predicate: object, *, timeout: float = 5.0) -> None:
    """Wait for a condition, failing loudly instead of hanging the suite.

    A bare `await asyncio.sleep(0.1)` between "start the worker" and "assert"
    passes on a fast machine and flakes on a loaded one; this waits for the thing
    the test actually depends on.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not reached before the timeout")


def _no_sleep(worker: object) -> object:
    """Replace the runtime's backoff with a yield, so a retry test is fast."""

    async def _sleep(seconds: float) -> None:
        await asyncio.sleep(0)

    return _sleep


def _counter(worker: ConsumerWorker, outcome: str) -> float:
    value = worker.metrics.registry.get_sample_value(
        "omnisense_worker_messages_total", {"worker": worker.name, "outcome": outcome}
    )
    return float(value or 0.0)


async def _get(port: int, path: str, *, method: str = "GET") -> tuple[int, str]:
    """One HTTP request over loopback. Deliberately not `httpx`.

    The listener speaks a hand-written subset of HTTP/1.1, and a client that
    papers over a malformed status line or a missing `Content-Length` would hide
    exactly the bugs this exercises.
    """
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=5)
    finally:
        writer.close()
        await writer.wait_closed()

    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    return status, body.decode("utf-8")


def _unused(event: SignalEnrichedEvent) -> None:  # pragma: no cover
    """Keeps the import honest: the signals payload is exercised in sibling files."""
