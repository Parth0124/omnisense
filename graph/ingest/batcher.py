"""Write buffering with back-pressure, flushed on size **or** age.

One graph update per transaction is the obvious implementation and the wrong one:
a transaction costs a round trip, a lock acquisition and a commit, and the
enrichment stream produces entity and edge updates by the thousand. Buffering
them into one transaction per few hundred rows is roughly two orders of magnitude
of throughput. That much is uncontroversial. The two decisions worth writing down
are what happens when the buffer fills, and what happens when it stops filling.

**Age, not just size.** A size-only flush is correct exactly while traffic is
heavy, which is the case nobody has trouble with. The moment the stream goes
quiet -- overnight, between connector runs, or because the upstream just died --
whatever is in the buffer sits there. Not for a while: indefinitely, until the
next row happens to push the count over the threshold. Those rows are already
acknowledged from Kafka's point of view if the consumer commits on receipt, and
they are invisible in the graph. The bug reproduces only at the end of a busy
period, which is the hardest time to notice something is *missing*. So the
buffer also flushes when its oldest row reaches `max_age_seconds`, and the timer
starts when the buffer goes from empty to non-empty -- measured from the *oldest*
row, so a slow trickle cannot keep pushing the deadline out.

**Back-pressure, not dropping.** When the writer is slower than the producer,
something has to give. Dropping rows is not an option -- these are the only copy
of the entity updates for signals that PostgreSQL has already accepted -- and
neither is unbounded growth, which converts a slow database into an out-of-memory
kill and turns a recoverable stall into a restart loop that replays from the last
committed offset. So the buffer has a hard capacity, `submit()` *waits* when it
is reached, and the wait propagates back through the Kafka consumer as the
absence of a `poll()`. `docs/knowledge-graph.md` §7 asks for the consumer to be
paused above two batches; `on_pressure` is that signal, fired with hysteresis so
a buffer hovering at the threshold does not pause and resume on every row.

**Failure keeps the rows.** A flush that fails leaves its rows at the head of the
buffer and retries with exponential backoff and jitter. Rows are never discarded
on the floor: without a `on_dead_letter` hook the batcher retries indefinitely
and lets back-pressure stop the world, which is the correct default for a store
whose contents are supposed to be reconstructible but whose reconstruction is a
manual operation. With the hook wired to `KAFKA_TOPIC_DLQ`, a batch that has
failed `max_attempts` times is handed over and dropped from the buffer, so one
poisonous batch cannot stall the stream forever.

Layer note: **L1 library** -- `models/` and the standard library only. The Neo4j
seam is `GraphWriter`, whose runner the caller supplies.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import Final

import structlog

from graph.ingest.writer import (
    EdgeWrite,
    GraphBatch,
    GraphWriter,
    NodeWrite,
    SignalStub,
    WriteOutcome,
)

__all__ = [
    "DEFAULT_MAX_AGE_SECONDS",
    "DEFAULT_MAX_BATCH_ROWS",
    "BatcherClosedError",
    "GraphWrite",
    "WriteBatcher",
]

_log = structlog.get_logger(__name__)

GraphWrite = NodeWrite | SignalStub | EdgeWrite
"""Anything the batcher accepts. Partitioned into a `GraphBatch` at flush time."""

DEFAULT_MAX_BATCH_ROWS: Final[int] = 500
DEFAULT_MAX_AGE_SECONDS: Final[float] = 2.0
"""`docs/knowledge-graph.md` §7: 500 rows or 2 seconds, whichever comes first.

Distinct from `INGESTION_BATCH_SIZE`, which governs how many records a *connector*
emits per batch. Conflating the two knobs would tie the graph's transaction size
to a rate limit on the other side of the pipeline.
"""

_DEFAULT_CAPACITY_BATCHES: Final[int] = 2
_DEFAULT_MAX_ATTEMPTS: Final[int] = 3
_BACKOFF_BASE_SECONDS: Final[float] = 0.5
_BACKOFF_CAP_SECONDS: Final[float] = 30.0


class BatcherClosedError(RuntimeError):
    """`submit()` was called on a closed batcher, or a waiter was released by close.

    Raised rather than silently discarding, because the row is the caller's
    responsibility until the batcher accepts it: a consumer that treats a closed
    batcher as "delivered" commits an offset for data that was never written.
    """


class WriteBatcher:
    """Buffers graph writes and applies them in fixed-size transactions.

    Not thread-safe and not meant to be. It is driven by one asyncio consumer
    loop; the internal lock exists to keep the flusher task and an explicit
    `flush()` from running a batch twice, not to guard against threads.
    """

    def __init__(
        self,
        writer: GraphWriter,
        *,
        max_batch_rows: int = DEFAULT_MAX_BATCH_ROWS,
        max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS,
        capacity_batches: int = _DEFAULT_CAPACITY_BATCHES,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        on_pressure: Callable[[bool], None] | None = None,
        on_dead_letter: Callable[[GraphBatch, Exception], Awaitable[None]] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_batch_rows < 1:
            raise ValueError("max_batch_rows must be at least 1")
        if max_age_seconds <= 0:
            raise ValueError(
                "max_age_seconds must be positive; a zero age flushes on every "
                "row, which is the per-row transaction this class exists to avoid"
            )
        if capacity_batches < 1:
            raise ValueError(
                "capacity_batches must be at least 1, or submit() blocks before a "
                "single batch can be assembled and the batcher deadlocks"
            )

        self._writer = writer
        self._max_batch_rows = max_batch_rows
        self._max_age_seconds = max_age_seconds
        self._capacity = max_batch_rows * capacity_batches
        self._max_attempts = max_attempts
        self._on_pressure = on_pressure
        self._on_dead_letter = on_dead_letter
        self._sleep = sleep

        self._buffer: list[GraphWrite] = []
        self._deadline: float | None = None
        self._closing = False
        self._closed = False
        self._pressured = False

        # Set whenever the flusher should re-evaluate: a row arrived, the size
        # threshold was crossed, or the batcher is closing. Cleared by the
        # flusher immediately before it waits, so a set that arrives during the
        # wait is not lost.
        self._wake = asyncio.Event()
        # Notified when rows leave the buffer, releasing submitters that are
        # blocked on capacity.
        self._drained = asyncio.Condition()
        self._flush_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------ lifecycle --

    async def __aenter__(self) -> WriteBatcher:
        self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def start(self) -> None:
        """Start the age-flush task. Idempotent.

        Separate from construction because creating a task requires a running
        event loop, and a batcher is often built during synchronous wiring. It is
        also what makes "was this batcher ever started" answerable: a batcher
        that is submitted to but never started buffers until it hits capacity and
        then blocks forever, which without this method would look like a hung
        consumer with no explanation.
        """
        if self._task is not None:
            return
        if self._closed:
            raise BatcherClosedError("this batcher has been closed and cannot restart")
        self._task = asyncio.get_running_loop().create_task(
            self._run(), name="graph-write-batcher"
        )

    async def aclose(self) -> None:
        """Drain the buffer, stop the flusher, and release blocked submitters.

        Draining on close is the point. Rows in the buffer at shutdown were
        accepted from the consumer and are not in the graph; discarding them
        loses exactly the writes that a clean shutdown is supposed to make safe.
        A close that cannot drain -- because the database is down -- surfaces the
        write error rather than swallowing it, so the caller can decide whether
        to abort the shutdown or accept the loss knowingly.
        """
        if self._closed:
            return
        self._closing = True
        self._wake.set()

        task, self._task = self._task, None
        if task is not None:
            await task

        try:
            while self._buffer:
                await self.flush()
        finally:
            self._closed = True
            # Anyone still waiting on capacity has to learn that no drain is
            # coming; leaving them blocked turns a shutdown into a hang.
            async with self._drained:
                self._drained.notify_all()
            self._set_pressure(False)

    # --------------------------------------------------------------- submit --

    async def submit(self, item: GraphWrite) -> None:
        """Buffer one write, waiting while the buffer is at capacity.

        The wait *is* the back-pressure. An `asyncio` consumer awaiting this call
        is not calling `poll()`, so the broker stops handing it records and the
        lag becomes visible in consumer-group metrics rather than in the
        process's resident memory. `on_pressure(True)` fires first for consumers
        that pause explicitly.
        """
        if self._closing or self._closed:
            raise BatcherClosedError("submit() after close; the row was not written")

        if len(self._buffer) >= self._capacity:
            self._set_pressure(True)
            async with self._drained:
                # A `while` and not an `if`: several submitters can be released
                # by one flush, and the first one to wake can refill the buffer
                # before the others run.
                while len(self._buffer) >= self._capacity and not self._closing:
                    await self._drained.wait()
            if self._closing or self._closed:
                raise BatcherClosedError("batcher closed while waiting for capacity")

        was_empty = not self._buffer
        self._buffer.append(item)
        if was_empty:
            # The age clock starts with the oldest row, not with the newest, so a
            # steady trickle cannot keep postponing the flush forever.
            self._deadline = self._now() + self._max_age_seconds
        if len(self._buffer) >= self._max_batch_rows or was_empty:
            self._wake.set()

    async def submit_many(self, items: Sequence[GraphWrite]) -> None:
        """Submit several writes, respecting capacity between each.

        Deliberately not an atomic bulk insert: admitting a whole sequence past
        the capacity check would let one large call overshoot the buffer bound by
        an arbitrary amount, which is the unbounded growth the bound exists to
        prevent.
        """
        for item in items:
            await self.submit(item)

    # ---------------------------------------------------------------- flush --

    async def flush(self) -> WriteOutcome | None:
        """Apply up to one batch. Returns None when there was nothing to write.

        One batch, not the whole buffer. A caller that has just submitted five
        thousand rows would otherwise get a single enormous transaction, which
        holds locks for the duration and is the thing most likely to deadlock
        against the other graph workers.
        """
        async with self._flush_lock:
            if not self._buffer:
                return None
            take = self._buffer[: self._max_batch_rows]
            batch = GraphBatch.of(take)
            outcome = await self._apply_with_retry(batch, len(take))
            # Only now are the rows gone. Removing them before the write means a
            # failure loses them, which is the one outcome this class must never
            # produce.
            del self._buffer[: len(take)]
            self._deadline = self._now() + self._max_age_seconds if self._buffer else None

        async with self._drained:
            self._drained.notify_all()
        if len(self._buffer) <= self._capacity // 2:
            # Hysteresis. Clearing the pause the instant the buffer drops below
            # capacity makes a saturated pipeline pause and resume on alternate
            # rows, and each resume costs a consumer-group round trip.
            self._set_pressure(False)

        return outcome

    async def _apply_with_retry(self, batch: GraphBatch, rows: int) -> WriteOutcome | None:
        """Write one batch, retrying transient failures with jittered backoff.

        Jitter, not plain exponential backoff: several graph workers deadlocking
        against each other retry in lockstep without it, collide again, and the
        contention that caused the first failure is reproduced exactly on every
        attempt. Randomising the wait is what breaks the convoy.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._writer.apply(batch)
            except Exception as exc:
                if attempt < self._max_attempts:
                    delay = min(
                        _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)),
                        _BACKOFF_CAP_SECONDS,
                    )
                    # Jitter, not cryptography: `random` is exactly right here.
                    delay *= 0.5 + random.random()
                    _log.warning(
                        "graph.batcher.flush_retry",
                        attempt=attempt,
                        max_attempts=self._max_attempts,
                        rows=rows,
                        delay_seconds=round(delay, 3),
                        error=str(exc),
                    )
                    await self._sleep(delay)
                    continue

                if self._on_dead_letter is None:
                    # No dead-letter sink configured, so there is nowhere safe to
                    # put these rows. Keeping them buffered and continuing to
                    # retry is the only option that does not lose them; the
                    # buffer fills, back-pressure engages, and the stall is
                    # visible. Silently dropping would be the one unrecoverable
                    # choice.
                    _log.error(
                        "graph.batcher.flush_failed",
                        attempts=attempt,
                        rows=rows,
                        error=str(exc),
                        note="no dead-letter sink; rows stay buffered and retry",
                    )
                    raise

                _log.error(
                    "graph.batcher.dead_lettered",
                    attempts=attempt,
                    rows=rows,
                    error=str(exc),
                )
                await self._on_dead_letter(batch, exc)
                return None

    # -------------------------------------------------------- the age timer --

    async def _run(self) -> None:
        """Flush on age, and on size when `submit` signals it.

        Structured as one loop with an interruptible wait rather than a periodic
        tick. A tick coarser than `max_age_seconds` misses the deadline; a finer
        one wakes the event loop constantly on an idle system for nothing.
        """
        while not self._closing:
            if not self._buffer:
                await self._wake.wait()
                self._wake.clear()
                continue

            remaining = 0.0 if self._deadline is None else self._deadline - self._now()
            if remaining <= 0 or len(self._buffer) >= self._max_batch_rows:
                try:
                    await self.flush()
                except Exception:
                    # `_apply_with_retry` has already exhausted its attempts and
                    # logged. The rows are still buffered, so re-raising here
                    # would kill the flusher task and leave them unflushed
                    # forever with no timer to try again. Waiting out one age
                    # window before the next attempt keeps the retry going
                    # without spinning.
                    await self._sleep(self._max_age_seconds)
                continue

            self._wake.clear()
            # A timeout here is the quiet-period case this timer exists for: the
            # deadline passed with no new rows, and the next iteration sees
            # `remaining <= 0` and flushes. Being woken early means the size
            # threshold was crossed, which the same next iteration handles.
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=remaining)

    def _now(self) -> float:
        """The event loop's clock, not `time.monotonic`.

        They are the same clock in CPython today, but `asyncio.wait_for` schedules
        against the loop's, and computing a deadline on one clock while waiting on
        another is the kind of thing that works until someone installs a different
        event loop policy.
        """
        return asyncio.get_running_loop().time()

    def _set_pressure(self, pressured: bool) -> None:
        """Fire `on_pressure` only on an edge, never on every row."""
        if pressured == self._pressured:
            return
        self._pressured = pressured
        _log.info("graph.batcher.pressure", paused=pressured, buffered=len(self._buffer))
        if self._on_pressure is not None:
            self._on_pressure(pressured)

    # ------------------------------------------------------------ inspection --

    @property
    def pending_rows(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_pressured(self) -> bool:
        """Whether back-pressure is currently signalled to the consumer."""
        return self._pressured

    @property
    def is_closed(self) -> bool:
        return self._closed
