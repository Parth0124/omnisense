"""Liveness, readiness and metrics for a process that serves no HTTP traffic.

A worker is invisible to every orchestrator convention the API gets for free.
Kubernetes decides an API pod is healthy by asking it a question; a worker has
nobody to ask, so the supervisor falls back to "the process exists" -- and a
process blocked forever on a poisoned message, or wedged inside a driver that
never times out, satisfies that check perfectly while doing nothing at all.
`docs/observability.md` §9.3 names that exact failure: *alive to the process
supervisor and dead to the system*. This module is the check that catches it.

Three surfaces, one tiny listener on `PROMETHEUS_PORT`:

`/health` -- **liveness**. True when the consumer loop has heartbeated within
`LIVENESS_STALE_AFTER_SECONDS`. Deliberately *not* dependent on any datastore: a
liveness probe that fails when PostgreSQL is down asks the orchestrator to
restart every worker in the fleet at the moment the database is already
struggling, which converts a recoverable outage into a restart storm. Liveness
answers one question only -- "is this loop still turning?"

`/readyz` -- **readiness**. Consumer group joined, partitions assigned, and the
downstream stores this worker actually writes to reachable. Readiness *may*
depend on dependencies, because failing it removes the worker from rotation
without killing it, which is the correct response to "I cannot do useful work
right now". It also flips to false the instant a `SIGTERM` arrives, before the
drain begins (`docs/deployment.md` §3.2), so nothing new is routed at a process
that is on its way out.

`/metrics` -- Prometheus text exposition. The API exposes this on its own port;
a worker has no other port, which is the whole reason this listener exists.

Why a hand-rolled listener rather than uvicorn
----------------------------------------------
Because the alternative is worse in a specific way. Mounting FastAPI inside every
worker means an ASGI server with its own signal handling and its own shutdown
semantics running alongside the consumer loop's -- two things racing to interpret
the same `SIGTERM`, in a process whose entire correctness argument rests on
draining in a particular order. `asyncio.start_server` plus about a hundred lines
of HTTP/1.1 has no opinion about signals, adds no dependency, and cannot lose a
race it does not enter. The parser below is deliberately minimal and refuses
anything it does not recognise: this endpoint is for a scrape agent on a
cluster-internal port, and `docs/observability.md` §9.2 forbids leaking anything
identifying from it.

Layer note: `workers/` (L4). Imports the L1k kernel and `services/`, never
`backend/api/`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger

__all__ = [
    "LIVENESS_STALE_AFTER_SECONDS",
    "DependencyProbe",
    "HealthServer",
    "HealthState",
    "WorkerMetrics",
]

logger = get_logger(__name__)

LIVENESS_STALE_AFTER_SECONDS: Final = 60.0
"""How long the loop may go without a heartbeat before it is declared dead.

Sixty seconds, from `docs/observability.md` §9.3, and it has to sit above two
things at once: the consumer's own idle fetch timeout (1s, so an idle worker
heartbeats constantly) and the worst case of one batch of handlers. A handler
that legitimately takes longer than a minute is a design problem -- the broker's
`max_poll_interval_ms` (300s) would eventually evict the member anyway -- but a
*shorter* liveness window would restart healthy workers under load, which is the
failure mode that teaches an on-call engineer to delete the probe.
"""

_READ_LIMIT_BYTES: Final = 8_192
"""Ceiling on one request's header block. Anything larger is refused unread.

Without a ceiling, a single connection that never sends a blank line grows this
process's heap until the OOM killer resolves it -- on a listener whose entire
purpose is to report that the process is healthy.
"""

_READ_TIMEOUT_SECONDS: Final = 5.0
"""Per-connection deadline for reading the request line and headers.

A scrape that opens a socket and stalls would otherwise hold a task forever.
Prometheus itself times out in ten seconds; this is deliberately tighter.
"""

_PROBE_TIMEOUT_SECONDS: Final = 2.0
"""Hard deadline across all readiness probes together.

`docs/observability.md` §9.2: "a readiness probe that hangs is worse than one
that fails". The probes run concurrently, so this bounds the whole endpoint
rather than one dependency at a time.
"""

_READINESS_CACHE_SECONDS: Final = 5.0
"""How long a readiness verdict is reused.

Kubernetes probes every replica every few seconds, and a probe that opens a
connection to five datastores per call becomes its own load problem. Five
seconds is the same window `docs/observability.md` §9.2 fixes for the API.
"""


DependencyProbe = Callable[[], Awaitable[bool]]
"""A named readiness check. Returns a bool; never raises, never blocks for long.

A bool rather than an exception for the same reason `backend/db/session.py`'s
`check_postgres()` returns one: readiness aggregates several dependencies, and
one being down must not prevent reporting on the others.
"""


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HealthState:
    """What the probes read. Owned by the worker, mutated only from its loop.

    Deliberately a plain mutable object rather than a set of async accessors:
    every write happens on the worker's own event loop and every read happens on
    the same loop from a connection handler, so there is no concurrency to
    protect against, and a lock here would only add a way to deadlock the probe
    that exists to diagnose a deadlock.

    Time is `time.monotonic()`, never wall clock. A worker that heartbeats at
    12:00:00 and is then hit by an NTP step correction backwards would, on wall
    clock, appear not to have heartbeated for however long the correction was --
    and would be restarted for it.
    """

    worker: str
    stale_after_seconds: float = LIVENESS_STALE_AFTER_SECONDS

    started: bool = False
    """The loop has begun. False during `setup()`, which may take a while."""

    ready: bool = False
    """Consumer group joined and dependencies reachable. Flips false on SIGTERM."""

    draining: bool = False
    """A shutdown was requested and in-flight work is being finished."""

    last_heartbeat: float = field(default_factory=time.monotonic)
    """Monotonic timestamp of the last completed loop iteration."""

    assigned_partitions: int = 0
    """Partitions this member holds. Zero on a consumer worker means not joined."""

    messages_handled: int = 0
    messages_dlq_routed: int = 0

    last_error: str | None = None
    """Exception *class name* of the last loop-level failure, never a message.

    A driver error can echo the request that caused it, and requests carry
    fetched content (`docs/security-and-privacy.md`). This value is served on an
    HTTP endpoint, so it is the one field here that could leak.
    """

    def heartbeat(self) -> None:
        """Record that the loop completed an iteration. Called every cycle.

        Called on an *idle* cycle too. That is the point: an idle worker whose
        broker has no records still has to prove it is turning, and a heartbeat
        gated on "did work" would declare a correctly-idle worker dead the first
        quiet minute.
        """
        self.last_heartbeat = time.monotonic()

    @property
    def heartbeat_age_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.last_heartbeat)

    @property
    def is_alive(self) -> bool:
        """Liveness: the loop turned recently enough.

        A process that has not yet started its loop is alive -- `setup()` opens
        connections and may legitimately take seconds -- which is why `started`
        gates this. Liveness during startup is the orchestrator's
        `initialDelaySeconds` problem, not ours.
        """
        if not self.started:
            return True
        return self.heartbeat_age_seconds <= self.stale_after_seconds

    def snapshot(self) -> dict[str, object]:
        """The `/readyz` body. Booleans and counters only.

        No hostnames, no versions, no connection strings, no topic names --
        `docs/observability.md` §9.2 requires this endpoint to leak nothing about
        the deployment.
        """
        return {
            "worker": self.worker,
            "started": self.started,
            "ready": self.ready,
            "draining": self.draining,
            "alive": self.is_alive,
            "heartbeat_age_seconds": round(self.heartbeat_age_seconds, 3),
            "assigned_partitions": self.assigned_partitions,
            "messages_handled": self.messages_handled,
            "messages_dlq_routed": self.messages_dlq_routed,
            "last_error": self.last_error,
        }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class WorkerMetrics:
    """The Prometheus families a worker publishes, bound to one registry.

    A dedicated `CollectorRegistry` rather than `prometheus_client`'s global
    default, for one blunt reason: the default registry raises `ValueError` on a
    duplicate time series, so a second worker constructed in the same process --
    which is exactly what a test suite does, dozens of times -- would fail at
    construction rather than at anything meaningful. One registry per worker also
    means a scrape of worker A cannot report worker B's counters.

    Names follow `docs/observability.md` §3: `omnisense_<subsystem>_<name>_<unit>`,
    counters end in `_total`. `worker` is a label rather than part of the metric
    name so a dashboard can sum across the family; it is a small closed set (one
    per module in `workers/`), so cardinality is bounded by the repository rather
    than by traffic.
    """

    def __init__(self, worker: str, *, registry: CollectorRegistry | None = None) -> None:
        self.worker = worker
        self.registry = registry if registry is not None else CollectorRegistry()

        self.messages = Counter(
            "omnisense_worker_messages_total",
            "Messages a worker took a terminal decision about.",
            labelnames=("worker", "outcome"),
            registry=self.registry,
        )
        """`outcome` is `handled`, `dlq` or `blocked`.

        Three outcomes, not two, because `blocked` is the one that needs an
        alert: it means the offset did not advance and the partition is stalled
        behind a message that could be neither handled nor parked. A binary
        success/failure counter hides that inside "failure" alongside the benign
        case of a message that reached the DLQ exactly as designed.
        """

        self.batches = Counter(
            "omnisense_worker_batches_total",
            "Fetch/handle/commit cycles completed, including empty polls.",
            labelnames=("worker", "state"),
            registry=self.registry,
        )

        self.batch_duration = Histogram(
            "omnisense_worker_batch_duration_seconds",
            "Wall time of one fetch/handle/commit cycle.",
            labelnames=("worker",),
            registry=self.registry,
        )

        self.handler_duration = Histogram(
            "omnisense_worker_handler_duration_seconds",
            "Wall time of one unit of work inside a cycle.",
            labelnames=("worker",),
            registry=self.registry,
        )

        self.heartbeat_age = Gauge(
            "omnisense_worker_heartbeat_age_seconds",
            "Seconds since the consumer loop last completed an iteration.",
            labelnames=("worker",),
            registry=self.registry,
        )
        """The metric form of the liveness probe.

        Exported as well as probed because the probe answers "restart or not"
        while the gauge answers "was it creeping upward for an hour first" --
        and only the second one prevents the next incident.
        """

        self.ready = Gauge(
            "omnisense_worker_ready",
            "1 when the worker is ready to take work, 0 otherwise.",
            labelnames=("worker",),
            registry=self.registry,
        )

        self.consumer_lag = Gauge(
            "omnisense_kafka_consumer_lag_records",
            "Records between the committed offset and the log end.",
            labelnames=("topic", "partition", "group"),
            registry=self.registry,
        )
        """`docs/observability.md` §3.1. Refreshed by the periodic collector in
        `workers/runtime/base_worker.py`, never sampled inside a handler (§3.2)."""

        self.dlq_messages = Counter(
            "omnisense_dlq_messages_total",
            "Messages routed to the dead-letter topic.",
            labelnames=("source_topic", "reason"),
            registry=self.registry,
        )
        """`reason` is an exception *class name*, a bounded set in practice, and
        it is what makes the DLQ triageable -- you group by it."""

    def observe_state(self, state: HealthState) -> None:
        """Copy the gauge-shaped parts of `HealthState` into the registry.

        Called by the periodic collector *and* by the `/metrics` handler. The
        collector is what §3.2 requires -- a gauge refreshed only on scrape has a
        value that depends on how often Prometheus asks -- and the handler call
        exists so a scrape that arrives between collector ticks does not report
        an age that is one whole interval stale.
        """
        self.heartbeat_age.labels(worker=self.worker).set(state.heartbeat_age_seconds)
        self.ready.labels(worker=self.worker).set(1.0 if state.ready else 0.0)

    def render(self) -> bytes:
        """Prometheus text exposition for this worker's registry."""
        return generate_latest(self.registry)


# --------------------------------------------------------------------------- #
# The listener
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Response:
    status: int
    reason: str
    body: bytes
    content_type: str = "application/json"

    def to_bytes(self) -> bytes:
        return b"".join(
            (
                f"HTTP/1.1 {self.status} {self.reason}\r\n".encode(),
                f"Content-Type: {self.content_type}\r\n".encode(),
                f"Content-Length: {len(self.body)}\r\n".encode(),
                b"Connection: close\r\n",
                # Nothing here is cacheable: a cached readiness answer is how a
                # drained worker keeps receiving traffic after it stopped
                # accepting it.
                b"Cache-Control: no-store\r\n\r\n",
                self.body,
            )
        )


class HealthServer:
    """A small HTTP/1.1 listener serving `/health`, `/readyz` and `/metrics`.

    One connection, one request, connection closed -- no keep-alive, no
    pipelining, no request bodies. A scrape agent handles that fine, and the
    absence of state is what keeps this from being a second thing that can wedge
    inside a process whose job is to report that it has not wedged.
    """

    def __init__(
        self,
        state: HealthState,
        metrics: WorkerMetrics,
        *,
        probes: Mapping[str, DependencyProbe] | None = None,
        host: str = "0.0.0.0",  # noqa: S104 -- a container port, published by the orchestrator
        port: int | None = None,
        settings: Settings | None = None,
        readiness_cache_seconds: float = _READINESS_CACHE_SECONDS,
    ) -> None:
        self._state = state
        self._metrics = metrics
        self._probes = dict(probes or {})
        self._settings = settings or get_settings()
        self._host = host
        self._port = port if port is not None else self._settings.observability.prometheus_port
        self._server: asyncio.Server | None = None
        self._cache_seconds = readiness_cache_seconds
        self._cached: tuple[float, bool, dict[str, str]] | None = None

    # ------------------------------------------------------------ lifecycle --

    @property
    def port(self) -> int:
        """The port actually bound. Differs from the requested one when it was 0.

        Port 0 asks the kernel for a free port, which is how the test suite runs
        several of these concurrently without colliding on 9090.
        """
        if self._server is None:
            return self._port
        for sock in self._server.sockets or ():
            bound = sock.getsockname()
            if isinstance(bound, tuple) and len(bound) >= 2:
                return int(bound[1])
        return self._port

    async def start(self) -> None:
        """Bind and serve. Idempotent.

        A bind failure is raised, not swallowed. It almost always means a second
        worker was started in the same network namespace on the same port, and a
        worker whose metrics silently go nowhere is a worker nobody notices
        falling behind.
        """
        if self._server is not None:
            return
        self._server = await asyncio.start_server(self._handle, host=self._host, port=self._port)
        logger.info("worker.health.listening", worker=self._state.worker, port=self.port)

    async def stop(self) -> None:
        """Close the listener and wait for in-flight connections. Idempotent."""
        server, self._server = self._server, None
        if server is None:
            return
        server.close()
        await server.wait_closed()

    # -------------------------------------------------------------- requests --

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Serve exactly one request, then close. Never propagates an exception.

        A raise here would be swallowed by the task the stream server spawns and
        resurface as an "exception was never retrieved" warning minutes later, in
        a different part of the log, with no request context. Answering 500 keeps
        the failure attributable.
        """
        response: _Response | None
        try:
            response = await self._respond(reader)
        except (TimeoutError, ConnectionResetError, asyncio.IncompleteReadError):
            # The scraper hung up or stalled. Nothing to report and nobody to
            # report it to; logging this above debug would make a rolling
            # Prometheus restart look like an incident.
            response = None
        except Exception as err:  # noqa: BLE001 -- the probe must never crash the worker
            logger.warning(
                "worker.health.request_failed",
                worker=self._state.worker,
                error=type(err).__name__,
            )
            response = _Response(500, "Internal Server Error", b'{"error":"probe_failed"}')

        try:
            if response is not None:
                writer.write(response.to_bytes())
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()
            # `wait_closed()` can itself raise if the peer already vanished, and
            # this runs on the shutdown path where a raise would be attributed to
            # the drain rather than to a socket nobody cares about.
            with contextlib.suppress(ConnectionResetError, BrokenPipeError):
                await writer.wait_closed()

    async def _respond(self, reader: asyncio.StreamReader) -> _Response:
        """Parse the request line and route it."""
        async with asyncio.timeout(_READ_TIMEOUT_SECONDS):
            request_line = await reader.readline()
            # Drain the header block so the client's write cannot block on a full
            # receive buffer while this side is writing the response.
            consumed = len(request_line)
            while consumed < _READ_LIMIT_BYTES:
                header = await reader.readline()
                consumed += len(header)
                if header in (b"\r\n", b"\n", b""):
                    break

        parts = request_line.decode("latin-1").split()
        if len(parts) < 2:
            return _Response(400, "Bad Request", b'{"error":"malformed_request"}')
        method, target = parts[0], parts[1]
        path = target.split("?", 1)[0]

        if method not in ("GET", "HEAD"):
            # This endpoint has no state to change. Refusing everything else is
            # free and removes a whole class of "what happens if" questions.
            return _Response(405, "Method Not Allowed", b'{"error":"method_not_allowed"}')

        match path:
            case "/health" | "/healthz" | "/livez":
                return self._liveness()
            case "/readyz" | "/ready":
                return await self._readiness()
            case "/metrics":
                self._metrics.observe_state(self._state)
                return _Response(200, "OK", self._metrics.render(), CONTENT_TYPE_LATEST)
            case _:
                return _Response(404, "Not Found", b'{"error":"not_found"}')

    def _liveness(self) -> _Response:
        """`/health`. Heartbeat freshness only -- no dependency is consulted."""
        alive = self._state.is_alive
        body = json.dumps(
            {
                "status": "ok" if alive else "stale",
                "heartbeat_age_seconds": round(self._state.heartbeat_age_seconds, 3),
            }
        ).encode("utf-8")
        return _Response(200, "OK", body) if alive else _Response(503, "Service Unavailable", body)

    async def _readiness(self) -> _Response:
        """`/readyz`. Group membership plus the injected dependency probes."""
        if self._state.draining or not self._state.ready:
            # Short-circuit before touching a dependency. A draining worker is
            # not ready by definition, and probing five datastores to confirm
            # that would add latency to every shutdown.
            status = "draining" if self._state.draining else "not_ready"
            body = json.dumps({"status": status, **self._state.snapshot()}).encode("utf-8")
            return _Response(503, "Service Unavailable", body)

        ok, components = await self._probe_dependencies()
        body = json.dumps(
            {
                "status": "ok" if ok else "degraded",
                "components": components,
                **self._state.snapshot(),
            }
        ).encode("utf-8")
        return _Response(200, "OK", body) if ok else _Response(503, "Service Unavailable", body)

    async def _probe_dependencies(self) -> tuple[bool, dict[str, str]]:
        """Run every probe concurrently under one deadline, with a short cache."""
        now = time.monotonic()
        if self._cached is not None and now - self._cached[0] < self._cache_seconds:
            return self._cached[1], self._cached[2]

        names = list(self._probes)
        components: dict[str, str] = {}
        if names:
            results: list[object]
            try:
                async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
                    results = list(
                        await asyncio.gather(
                            *(self._probes[name]() for name in names), return_exceptions=True
                        )
                    )
            except TimeoutError:
                # A hung dependency is a failed dependency for readiness
                # purposes. Which one hung is unknowable once the group deadline
                # fires, so every probe is reported as `timeout` rather than one
                # being blamed arbitrarily.
                results = ["timeout"] * len(names)
            for name, result in zip(names, results, strict=True):
                if isinstance(result, BaseException):
                    components[name] = "failed"
                elif result == "timeout":
                    components[name] = "timeout"
                else:
                    components[name] = "ok" if result else "failed"

        ok = all(value == "ok" for value in components.values())
        self._cached = (now, ok, components)
        return ok, components
