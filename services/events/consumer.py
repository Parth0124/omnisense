"""The consumer-group runtime: at-least-once delivery, with backpressure and a DLQ.

Every worker in `workers/` drives one of these. It owns the four behaviours that
make at-least-once actually safe, none of which the broker provides on its own.

**Commit after the handler, never before.** `enable_auto_commit=False` is not a
tuning choice. With auto-commit, offsets advance on a timer that knows nothing
about whether the handler succeeded, so a worker that dies mid-batch can have
already committed past messages it never processed -- and there is no error, no
DLQ entry, and no way to discover the gap later. Manual commit makes the offset a
statement about work completed rather than about bytes received. It is the
consumer-side twin of the producer's cursor-commit-after-ack rule
(`docs/connector-spec.md` §4.1) and it fails in the same direction: on a crash,
messages are redelivered, and consumers are idempotent by contract (ADR-0007).

**Bounded fetch plus partition pause.** `docs/architecture.md` §7.2 forbids an
unbounded in-memory queue anywhere on the ingestion path: pressure is absorbed by
the log and expressed as consumer lag. `max_records` bounds one fetch to
`INGESTION_BATCH_SIZE`; pausing the batch's partitions while it is in flight is
what stops aiokafka's *prefetcher* from filling memory with the next batches
while a slow handler works through this one. Without the pause, a worker whose
handler slows to a crawl grows its heap instead of growing its lag -- which turns
a visible, alertable condition into an OOM kill.

**A poison message goes to the DLQ, it does not stop the partition.** Kafka
offsets are strictly ordered, so one message that can never succeed blocks every
message behind it, forever. After bounded retries the message is published to
`omnisense.dlq` with its original bytes intact and the offset advances. Two
failures are deliberately *not* treated that way: a body that cannot be parsed at
all skips the retries (retrying does not change bytes), and a failure to publish
*to* the DLQ blocks -- refusing to advance is correct there, because advancing
would drop the message entirely.

**Correlation id propagates from the envelope.** The runtime binds
`envelope.correlation_id` around the handler, so every log line the handler
emits -- and every event it publishes in turn -- joins the chain that started
with a user's request (`docs/observability.md` §1). A worker that minted a fresh
id per message would break the chain at exactly the boundary you need it.

Layer note: `services/` (L2). Imported by `workers/runtime/base_worker.py`, which
adds process lifecycle, signal handling and metrics on top.
"""

from __future__ import annotations

import asyncio
import enum
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
from aiokafka.structs import ConsumerRecord, TopicPartition

from backend.core.config import Settings, get_settings
from backend.core.logging import (
    UNBOUND_CORRELATION_ID,
    correlation_scope,
    get_logger,
)
from services.events.schemas import (
    HEADER_CORRELATION_ID,
    DlqEvent,
    EventDecodeError,
    EventEnvelope,
    EventType,
    header_value,
)
from services.events.topics import (
    TopicRole,
    encode_key,
    role_for_topic,
    topic_name,
)

__all__ = [
    "BatchOutcome",
    "ConsumedMessage",
    "DlqPublisher",
    "EventConsumer",
    "Handler",
    "KafkaConsumerLike",
]

logger = get_logger(__name__)

_DEFAULT_MAX_ATTEMPTS: Final = 3
"""Deliveries attempted before a message is declared poison.

Three, not five, because these retries are *in-process and immediate*: they exist
to absorb a blipped connection or a momentarily locked row, not to wait out an
outage. A dependency that is genuinely down fails every attempt in a few hundred
milliseconds and the message reaches the DLQ, where a human decides. Retrying
harder here just holds the partition paused for longer while lag builds behind it.
"""

_DEFAULT_BACKOFF_SECONDS: Final = 0.5
_MAX_BACKOFF_SECONDS: Final = 5.0
"""Ceiling on the in-batch retry sleep.

Bounded well below `max_poll_interval_ms` (300s): a batch that takes longer than
that is presumed dead by the group coordinator, the partitions are reassigned,
and the commit this runtime is about to make would be rejected as coming from a
stale generation -- after the work was done.
"""

_FETCH_TIMEOUT_MS: Final = 1_000
"""How long one `getmany` waits for records.

Short so that `stop()` is observed promptly: the loop checks for shutdown between
batches, so this interval is the worst-case delay on a graceful shutdown.
"""

_BROKER_RETRY_SECONDS: Final = 5.0
"""Pause after a broker-level failure in the fetch loop.

Without it, an unreachable broker turns the run loop into a hot loop that emits
thousands of identical log lines a second and does nothing else.
"""


class _Disposition(enum.Enum):
    """What to do with the offset after handling one message."""

    ADVANCE = "advance"
    """Handled, or safely parked in the DLQ. The offset may move past it."""

    BLOCK = "block"
    """Not handled and not parked. The offset must not move; redeliver later."""


@dataclass(frozen=True, slots=True)
class ConsumedMessage:
    """One message, decoded, as a handler sees it.

    Carries the broker coordinates and the raw bytes alongside the envelope
    because a handler that re-publishes, traces or archives needs them, and
    because passing the whole `ConsumerRecord` would leak the client library into
    every handler signature in `workers/`.
    """

    topic: str
    partition: int
    offset: int
    key: str | None
    envelope: EventEnvelope
    raw: bytes

    @property
    def correlation_id(self) -> str:
        return self.envelope.correlation_id


Handler = Callable[[ConsumedMessage], Awaitable[None]]
"""What a worker supplies. Raising is how it reports failure.

Same contract as `services/signal_engine/pipeline.py`'s stages: the handler
raises, and the *runtime* -- not the handler -- decides whether that means retry,
DLQ or block. A handler that swallowed its own exception would silently advance
the offset past work that was never done, which is indistinguishable from success
in every metric there is.
"""


class DlqPublisher(Protocol):
    """How a failed message is parked. Injectable so tests need no broker."""

    async def __call__(self, envelope: EventEnvelope, *, key: bytes | None) -> None: ...


class KafkaConsumerLike(Protocol):
    """The subset of `AIOKafkaConsumer` this runtime uses.

    Declared as a protocol so the unit suite can substitute an in-memory broker
    without patching the library: `aiokafka` must never open a socket in the unit
    suite, and the reliable way to guarantee that is for the test to hand in an
    object that has none.
    """

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]]: ...

    async def commit(self, offsets: dict[TopicPartition, int] | None = None) -> None: ...

    def pause(self, *partitions: TopicPartition) -> None: ...

    def resume(self, *partitions: TopicPartition) -> None: ...

    def seek(self, partition: TopicPartition, offset: int) -> None: ...

    def assignment(self) -> set[TopicPartition]: ...


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """What one fetch/handle/commit cycle did.

    Returned rather than only logged so that `workers/runtime/base_worker.py` can
    turn it into metrics, and so the run loop can tell an idle poll from a
    productive one without inspecting log output.
    """

    fetched: int = 0
    handled: int = 0
    dlq_routed: int = 0
    blocked: int = 0
    committed: dict[TopicPartition, int] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return self.fetched == 0


class EventConsumer:
    """Drives one consumer group over one or more topics.

    Stateless with respect to messages: everything per-message lives on the stack,
    so a future version that runs several batches concurrently does not have to
    unpick shared state. What is on `self` is configuration and the client.
    """

    def __init__(
        self,
        *,
        topics: Sequence[TopicRole | str],
        handler: Handler,
        group_id: str | None = None,
        consumer: KafkaConsumerLike | None = None,
        dlq_publisher: DlqPublisher | None = None,
        dlq_enabled: bool = True,
        batch_size: int | None = None,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        backoff_seconds: float = _DEFAULT_BACKOFF_SECONDS,
        fetch_timeout_ms: int = _FETCH_TIMEOUT_MS,
        settings: Settings | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1; 0 would never call the handler")

        self._settings = settings or get_settings()
        self._topics = tuple(
            topic_name(t, settings=self._settings) if isinstance(t, TopicRole) else t
            for t in topics
        )
        if not self._topics:
            raise ValueError("a consumer must subscribe to at least one topic")

        self._handler = handler
        self._group_id = group_id or self._settings.kafka.consumer_group
        self._consumer = consumer
        self._owns_consumer = consumer is None
        self._dlq_publisher = dlq_publisher
        self._dlq_enabled = dlq_enabled
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds
        self._fetch_timeout_ms = fetch_timeout_ms
        # The smaller of the two bounds wins. `INGESTION_BATCH_SIZE` is the
        # documented backpressure knob (`docs/architecture.md` §7.2) and
        # `KAFKA_MAX_POLL_RECORDS` is the client-level ceiling; honouring only one
        # of them would leave the other silently ineffective.
        self._batch_size = min(
            batch_size or self._settings.connectors.ingestion_batch_size,
            self._settings.kafka.max_poll_records,
        )
        self._stopping = asyncio.Event()
        self._started = False

    # ------------------------------------------------------------ lifecycle --

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def topics(self) -> tuple[str, ...]:
        return self._topics

    async def start(self) -> None:
        """Build the client if one was not injected, and join the group. Idempotent."""
        if self._started:
            return
        if self._consumer is None:
            self._consumer = self._build_consumer()
        await self._consumer.start()
        self._started = True
        logger.info(
            "events.consumer.started",
            topics=list(self._topics),
            group=self._group_id,
            batch_size=self._batch_size,
        )

    def _build_consumer(self) -> KafkaConsumerLike:
        """Construct the real client from settings.

        Imported lazily-in-spirit: nothing here runs at module import, so the unit
        suite can import this module without a broker. `client_transport_kwargs`
        is shared with the producer so a SASL misconfiguration cannot differ
        between the two halves of one deployment.
        """
        from services.events.producer import client_transport_kwargs

        kafka = self._settings.kafka
        client = AIOKafkaConsumer(
            *self._topics,
            group_id=self._group_id,
            # The single most important line in this file. See the module docstring.
            enable_auto_commit=False,
            auto_offset_reset=kafka.auto_offset_reset.value,
            max_poll_records=self._batch_size,
            client_id=self._settings.observability.otel_service_name,
            **client_transport_kwargs(self._settings),
        )
        # `aiokafka` ships no type information (`pyproject.toml` lists it under
        # `ignore_missing_imports`), so the constructor is `Any`. Narrowing here
        # rather than returning `Any` is what keeps the protocol meaningful: if
        # the client ever stops satisfying it, this is the line that fails.
        narrowed: KafkaConsumerLike = client
        return narrowed

    async def stop(self) -> None:
        """Ask the loop to finish its current batch, then close the client.

        Never interrupts a batch mid-flight. A batch abandoned after its handlers
        ran but before its commit is redelivered in full on restart -- correct,
        but wasteful; a batch abandoned mid-handler is the case idempotency has to
        cover, and the fewer times that happens the better.

        Terminal for this instance: `run()` will not resume afterwards. A worker
        that wants to reconnect builds a new consumer, which is also what gets the
        group membership and partition assignment re-established cleanly.

        An injected client is not closed -- whoever passed one in owns its
        lifecycle, and closing someone else's connection on shutdown is how a
        shared client disappears out from under a second consumer.
        """
        self._stopping.set()
        if self._consumer is not None and self._started and self._owns_consumer:
            await self._consumer.stop()
        self._started = False

    async def run(self) -> None:
        """Consume until `stop()` is called. The worker entry point.

        Broker-level failures are logged and retried rather than propagated: a
        worker that exited on a transient broker error would be restarted by the
        orchestrator into exactly the same condition, having lost its group
        membership and triggered a rebalance for every other member.
        """
        await self.start()
        try:
            while not self._stopping.is_set():
                try:
                    await self.run_once()
                except (KafkaError, OSError) as err:
                    logger.warning(
                        "events.consumer.broker_unavailable",
                        group=self._group_id,
                        error=type(err).__name__,
                        retry_in_seconds=_BROKER_RETRY_SECONDS,
                    )
                    await self._sleep_or_stop(_BROKER_RETRY_SECONDS)
        finally:
            await self.stop()

    # ---------------------------------------------------------- the cycle --

    async def run_once(self) -> BatchOutcome:
        """Fetch one bounded batch, handle it, commit what succeeded.

        Split out from `run()` because it is the unit of behaviour worth testing:
        pause, handle, commit, resume in that order, with the offset moving only
        past work that actually completed.
        """
        consumer = self._require_consumer()
        batch = await consumer.getmany(
            timeout_ms=self._fetch_timeout_ms, max_records=self._batch_size
        )
        if not batch:
            return BatchOutcome()

        partitions = tuple(batch)
        # Backpressure. Held until the commit lands, so the prefetcher cannot run
        # ahead of the handler (`docs/architecture.md` §7.2).
        consumer.pause(*partitions)

        fetched = sum(len(records) for records in batch.values())
        handled = 0
        dlq_routed = 0
        blocked = 0
        commits: dict[TopicPartition, int] = {}

        try:
            for partition, records in batch.items():
                next_offset: int | None = None
                for index, record in enumerate(records):
                    disposition, routed = await self._handle_record(record)
                    if disposition is _Disposition.ADVANCE:
                        next_offset = record.offset + 1
                        handled += 1
                        dlq_routed += int(routed)
                        continue

                    # Blocked. Rewind so this message is redelivered in *this*
                    # process: not committing is enough to survive a restart, but
                    # the consumer's own position has already moved past these
                    # records, so without a seek the message would not be seen
                    # again until a rebalance -- which may be never.
                    #
                    # Everything behind it in this partition is left unhandled on
                    # purpose. Skipping ahead would break the per-Signal ordering
                    # that the partition key exists to provide
                    # (`services/events/topics.py`).
                    blocked += len(records) - index
                    consumer.seek(partition, record.offset)
                    break

                if next_offset is not None:
                    commits[partition] = next_offset

            if commits:
                await self._commit(consumer, commits, batch)
        except BaseException:
            # Anything unexpected -- including cancellation on shutdown -- leaves
            # the consumer's fetch position advanced past records that were never
            # committed. Those messages would then be skipped for the life of this
            # process: not lost (the offset is uncommitted, so a restart replays
            # them) but invisible until one happens, which on a long-lived worker
            # may be days. Rewinding to the last known-good offset makes the
            # failure cost a redelivery instead of a delay nobody can see.
            self._rewind(consumer, batch, commits)
            raise
        finally:
            consumer.resume(*partitions)

        return BatchOutcome(
            fetched=fetched,
            handled=handled,
            dlq_routed=dlq_routed,
            blocked=blocked,
            committed=dict(commits),
        )

    async def _commit(
        self,
        consumer: KafkaConsumerLike,
        commits: dict[TopicPartition, int],
        batch: dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]],
    ) -> None:
        """Commit the batch's offsets, rewinding everything if the commit fails.

        A failed commit means the broker does not know this work was done, so the
        only consistent thing to do is arrange for it to be done again: the
        partitions are rewound to the batch's first offset and the messages are
        redelivered. Handlers are idempotent by contract (ADR-0007), so
        reprocessing is a cost rather than a correctness problem -- whereas
        leaving the position advanced past uncommitted work would produce a gap
        that no reconciler could detect.
        """
        try:
            await consumer.commit(commits)
        except (KafkaError, OSError) as err:
            logger.warning(
                "events.consumer.commit_failed",
                group=self._group_id,
                error=type(err).__name__,
                partitions=[f"{tp.topic}[{tp.partition}]" for tp in commits],
            )
            commits.clear()
            self._rewind(consumer, batch, commits)

    @staticmethod
    def _rewind(
        consumer: KafkaConsumerLike,
        batch: dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]],
        commits: dict[TopicPartition, int],
    ) -> None:
        """Move each partition's fetch position back to its last committed offset.

        Falls back to the batch's first offset for a partition that committed
        nothing, which is the same value the broker would hand back on a
        rebalance. Idempotent, so it is safe on a path that may already have
        sought (the blocked-message case does).
        """
        for partition, records in batch.items():
            if not records:
                continue
            consumer.seek(partition, commits.get(partition, records[0].offset))

    # ------------------------------------------------------- one message --

    async def _handle_record(
        self, record: ConsumerRecord[bytes, bytes]
    ) -> tuple[_Disposition, bool]:
        """Decode, deliver and -- on failure -- park one message.

        Returns the offset disposition and whether the message was routed to the
        DLQ, so the caller can count both without inspecting exceptions.
        """
        raw = record.value or b""

        try:
            envelope = EventEnvelope.from_bytes(raw)
        except EventDecodeError as err:
            # No retries: the bytes will not change. The correlation id is read
            # from the headers instead, which is exactly why the envelope
            # duplicates its routing fields there.
            with correlation_scope(header_value(record.headers, HEADER_CORRELATION_ID)):
                logger.error(
                    "events.consumer.undecodable",
                    topic=record.topic,
                    partition=record.partition,
                    offset=record.offset,
                    error=type(err).__name__,
                )
                return await self._park(record, None, err, attempts=1), True

        correlation = (
            None if envelope.correlation_id == UNBOUND_CORRELATION_ID else envelope.correlation_id
        )
        with correlation_scope(correlation):
            if not envelope.is_readable():
                # A newer envelope version. Refusing is the point: applying old
                # semantics to a newer body writes plausible-looking wrong data
                # into five stores (`docs/signal-model.md` §7).
                unreadable = EventDecodeError(
                    f"envelope schema_version={envelope.schema_version} is not "
                    "readable by this build"
                )
                logger.error(
                    "events.consumer.unreadable_schema_version",
                    topic=record.topic,
                    offset=record.offset,
                    schema_version=envelope.schema_version,
                    event_type=envelope.event_type.value,
                )
                return await self._park(record, envelope, unreadable, attempts=1), True

            message = ConsumedMessage(
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                key=_decode_key(record.key),
                envelope=envelope,
                raw=raw,
            )
            failure = await self._deliver(message)
            if failure is None:
                return _Disposition.ADVANCE, False

            logger.error(
                "events.consumer.handler_exhausted",
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                event_type=envelope.event_type.value,
                event_id=envelope.event_id,
                attempts=self._max_attempts,
                error=type(failure).__name__,
            )
            return await self._park(record, envelope, failure, attempts=self._max_attempts), True

    async def _deliver(self, message: ConsumedMessage) -> Exception | None:
        """Call the handler, retrying a bounded number of times.

        Returns the final exception rather than raising it: the caller needs to
        decide between DLQ and block, and that decision reads better as a value
        than as a second layer of `try`.

        `except Exception` deliberately does not catch `asyncio.CancelledError`
        (a `BaseException` since 3.8) -- a shutdown must cancel the handler, not
        be mistaken for a handler failure and retried.
        """
        last_error: Exception | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                await self._handler(message)
            except Exception as err:  # noqa: BLE001 -- classification is this method's job
                last_error = err
                logger.warning(
                    "events.consumer.handler_failed",
                    topic=message.topic,
                    partition=message.partition,
                    offset=message.offset,
                    attempt=attempt,
                    max_attempts=self._max_attempts,
                    error=type(err).__name__,
                )
                if attempt < self._max_attempts:
                    await self._sleep_or_stop(self._retry_delay(attempt))
                continue
            return None
        return last_error

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with full jitter, capped.

        Jittered because a broker-wide blip fails every partition of every replica
        at once; unjittered backoff would then synchronize every retry in the
        fleet onto the same instant and re-create the thundering herd that killed
        the dependency in the first place. Same strategy as
        `connectors/ratelimit/backoff.py` (`docs/connector-spec.md` §5.2).
        """
        ceiling = min(_MAX_BACKOFF_SECONDS, self._backoff_seconds * (2 ** (attempt - 1)))
        return random.uniform(0.0, ceiling)  # noqa: S311 -- jitter, not cryptography

    async def _park(
        self,
        record: ConsumerRecord[bytes, bytes],
        envelope: EventEnvelope | None,
        error: BaseException,
        *,
        attempts: int,
    ) -> _Disposition:
        """Publish a failed message to the DLQ. `ADVANCE` only if that succeeded.

        Blocking when the DLQ publish fails is the deliberate half of this. The
        alternative -- advance anyway -- means the message exists nowhere: not
        handled, not parked, and past the committed offset. That is the one
        outcome this whole design is built to prevent, and it is worth stalling a
        partition to avoid.
        """
        if not self._dlq_enabled:
            logger.error(
                "events.consumer.dlq_disabled",
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                outcome="blocked",
            )
            return _Disposition.BLOCK

        if role_for_topic(record.topic, settings=self._settings) is TopicRole.DLQ:
            # A message that fails while being replayed *from* the DLQ must not be
            # written back to it: that is an infinite loop that grows the topic
            # every cycle. `workers/dlq.py` owns the retry decision for these; the
            # runtime's job is to not block triage of everything behind it.
            logger.error(
                "events.consumer.dlq_message_failed",
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                outcome="skipped",
                error=type(error).__name__,
            )
            return _Disposition.ADVANCE

        dlq_event = DlqEvent.from_failure(
            topic=record.topic,
            body=record.value or b"",
            error=error,
            consumer_group=self._group_id,
            attempts=attempts,
            partition=record.partition,
            offset=record.offset,
            key=_decode_key(record.key),
            envelope=envelope,
        )
        dlq_envelope = EventEnvelope.wrap(
            dlq_event,
            producer=self._settings.observability.otel_service_name,
            # The original chain, not a new one: a DLQ record whose correlation id
            # did not match the ingestion that produced it would be untraceable
            # back to the request that caused it.
            correlation_id=envelope.correlation_id if envelope is not None else None,
            tenant_id=envelope.tenant_id if envelope is not None else None,
        )

        try:
            await self._publish_dlq(dlq_envelope, key=encode_key(dlq_event.partition_key))
        except Exception as err:  # noqa: BLE001 -- any failure here must block
            logger.error(
                "events.consumer.dlq_publish_failed",
                topic=record.topic,
                partition=record.partition,
                offset=record.offset,
                outcome="blocked",
                error=type(err).__name__,
            )
            return _Disposition.BLOCK

        logger.warning(
            "events.consumer.routed_to_dlq",
            topic=record.topic,
            partition=record.partition,
            offset=record.offset,
            event_type=(envelope.event_type if envelope else EventType.UNKNOWN).value,
            attempts=attempts,
            error_chain=dlq_event.error_chain,
        )
        return _Disposition.ADVANCE

    async def _publish_dlq(self, envelope: EventEnvelope, *, key: bytes | None) -> None:
        """Send to the DLQ through the injected publisher, or the real producer.

        The import is function-local so that constructing an `EventConsumer` with
        an injected publisher never touches `services/events/producer.py` and its
        module-level singleton -- which is what lets a unit test run without ever
        instantiating an aiokafka client.
        """
        if self._dlq_publisher is not None:
            await self._dlq_publisher(envelope, key=key)
            return

        from services.events.producer import publish_envelope

        await publish_envelope(envelope, role=TopicRole.DLQ, key=key)

    # ------------------------------------------------------------ internals --

    def _require_consumer(self) -> KafkaConsumerLike:
        if self._consumer is None:
            raise RuntimeError(
                "EventConsumer.start() has not been called; there is no client to "
                "fetch from. Use `await consumer.start()` or `await consumer.run()`."
            )
        return self._consumer

    async def _sleep_or_stop(self, seconds: float) -> None:
        """Sleep, but wake immediately if shutdown was requested.

        A plain `asyncio.sleep` inside the retry path would make a graceful
        shutdown wait out the full backoff, which under a failing dependency is
        the difference between a two-second rolling restart and one that trips the
        orchestrator's termination grace period and gets SIGKILLed mid-batch.
        """
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(self._stopping.wait(), timeout=seconds)
        except TimeoutError:
            return


def _decode_key(key: bytes | None) -> str | None:
    """Message keys are UTF-8 by construction here, but never trusted to be.

    A key written by another producer -- a mirroring tool, an operator's
    `rpk produce` -- may be arbitrary bytes, and this runs on the path that
    handles messages already known to be broken.
    """
    if key is None:
        return None
    try:
        return key.decode("utf-8")
    except UnicodeDecodeError:
        return None
