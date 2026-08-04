"""The event producer: one durable write path onto the log.

Follows the client shape established by `backend/db/session.py` -- a lazily
created module-level singleton, no I/O at import, a `check_kafka()` that never
raises for `/readyz`, and a `dispose_producer()` wired into lifespan shutdown.
Importing this module must never open a socket: the unit suite imports it to
assert exactly that.

Why `acks="all"` and `enable_idempotence=True` are not tunable
--------------------------------------------------------------
`docs/connector-spec.md` §4.1 rule 1 is the rule the whole ingestion path rests
on: **commit the cursor after the ack, never before.** A cursor committed before
the producer acknowledges silently loses records, and the reverse only duplicates
them, which dedup absorbs. That rule is sound only if an ack is a *durability*
statement. It is not, by default:

- `acks=1` (Kafka's default, and aiokafka's) means the partition leader wrote the
  message to its own log. If that broker dies before the followers replicate, the
  message is gone -- and the connector has already committed a cursor past it, so
  nothing will ever fetch those records again. The loss is permanent and silent:
  no error, no DLQ entry, just a gap in a time series that nobody can explain
  months later. `acks="all"` makes the ack mean "every in-sync replica has it",
  which is the thing the cursor rule assumes. It costs one replication round trip
  per batch.
- Without `enable_idempotence`, the producer's own internal retry -- which fires
  on exactly the transient network errors that are most common -- can write the
  same batch twice when an ack is lost in flight rather than the write failing.
  At-least-once is the accepted delivery contract (ADR-0007), but a *broker-side*
  duplicate is worse than a redelivery: it is invisible to consumer offsets, so
  no consumer-side idempotency key is ever consulted for it. Idempotence gives
  the producer a producer id and a per-partition sequence number, and the broker
  drops the duplicate.

Together they make `send_and_wait()` returning mean "durably written, exactly
once, on every in-sync replica" -- which is precisely the guarantee the caller
needs before it commits a cursor. Neither is exposed as a setting, because a
deployment that turned either off would be silently violating a rule written
three documents away.

Failure semantics: a produce failure raises. `docs/architecture.md` §7.3 lists
Redpanda as *halting* the ingestion path, not degrading it -- there is no
correct local buffer for a message that must be durable, and an in-memory queue
would be an unbounded one (§7.2 forbids exactly that). The caller stops fetching
and does not commit its cursor, so the page replays intact on the next run.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
from aiokafka.structs import RecordMetadata

from backend.core.config import KafkaSecurityProtocol, Settings, get_settings
from backend.core.exceptions import DependencyUnavailableError
from backend.core.logging import get_correlation_id, get_logger
from backend.db import HEALTH_PROBE_TIMEOUT_SECONDS
from services.events.schemas import EventEnvelope, EventPayload
from services.events.topics import TopicRole, encode_key, role_for_event, topic_name

__all__ = [
    "check_kafka",
    "client_transport_kwargs",
    "dispose_producer",
    "get_producer",
    "publish",
    "publish_envelope",
]

logger = get_logger(__name__)

_PRODUCE_TIMEOUT_SECONDS: Final = 30.0
"""Ceiling on one `send_and_wait`.

aiokafka bounds a single *request* with `request_timeout_ms`, but a produce that
is being retried internally can exceed that many times over while the broker is
in a leader election. A connector waiting indefinitely on one message holds a
rate-limit slot and a worker slot for as long as the outage lasts, so the wait is
bounded here and the run fails cleanly instead.
"""

_REQUEST_TIMEOUT_MS: Final = 20_000
_LINGER_MS: Final = 10
"""Ten milliseconds of batching.

Zero (the default) sends one request per message, which at ingestion rates is a
request per record and enough syscall overhead to become the bottleneck before
the broker does. Ten milliseconds is below the latency anyone notices on an
ingestion path and is enough to coalesce a page of records into one request.
"""

_MAX_REQUEST_SIZE: Final = 1_048_576
"""1 MiB, matching the broker's default `max.message.bytes`.

Configured explicitly so that an oversized message fails in *this* process with a
clear error, rather than being accepted locally and rejected by the broker after
the batch has been assembled.
"""

_producer: AIOKafkaProducer | None = None
_producer_lock: asyncio.Lock | None = None


def _lock() -> asyncio.Lock:
    """The lock guarding producer creation, created lazily.

    `backend/db/session.py` needs no lock because `create_async_engine()` is
    synchronous -- there is no await point between the `is None` check and the
    assignment, so no other task can interleave. Here `start()` *is* awaited, so
    two coroutines calling `get_producer()` concurrently would both see `None`,
    both build a producer, both start it, and one would be overwritten while
    still holding an open connection. That is a leaked socket per race, which is
    the kind of thing that only manifests as file-descriptor exhaustion in
    production.

    Created lazily rather than at module scope because an `asyncio.Lock` binds to
    the first event loop that uses it and raises if it later sees another; the
    test suite runs each test in a fresh loop, and `dispose_producer()` clears
    this alongside the client.
    """
    global _producer_lock
    if _producer_lock is None:
        _producer_lock = asyncio.Lock()
    return _producer_lock


def client_transport_kwargs(settings: Settings | None = None) -> dict[str, Any]:
    """Bootstrap and authentication arguments shared by producer and consumer.

    Deliberately one function used by both halves. A deployment where the producer
    speaks SASL_SSL and the consumer speaks PLAINTEXT is a configuration that
    half-works -- ingestion succeeds and enrichment silently never connects --
    and the only reliable way to prevent it is to leave no second place to
    configure.
    """
    kafka = (settings or get_settings()).kafka
    kwargs: dict[str, Any] = {
        "bootstrap_servers": kafka.bootstrap_servers,
        "security_protocol": kafka.security_protocol.value,
    }
    if kafka.security_protocol in (
        KafkaSecurityProtocol.SASL_PLAINTEXT,
        KafkaSecurityProtocol.SASL_SSL,
    ):
        if kafka.sasl_username is None or kafka.sasl_password is None:
            raise DependencyUnavailableError(
                f"KAFKA_SECURITY_PROTOCOL is {kafka.security_protocol.value} but "
                "KAFKA_SASL_USERNAME/KAFKA_SASL_PASSWORD are unset; the client "
                "would fall back to an anonymous connection and fail at handshake."
            )
        kwargs |= {
            "sasl_mechanism": "PLAIN",
            "sasl_plain_username": kafka.sasl_username,
            "sasl_plain_password": kafka.sasl_password.get_secret_value(),
        }
    return kwargs


async def get_producer() -> AIOKafkaProducer:
    """Return the process-wide producer, creating and starting it on first use.

    Async because both halves of "create" are: `AIOKafkaProducer.__init__` needs a
    running event loop, and `start()` opens the bootstrap connection. That is also
    why this cannot be a plain property -- import-time construction would make
    collecting the test suite require a broker.
    """
    global _producer
    if _producer is not None:
        return _producer

    async with _lock():
        # Re-checked inside the lock: another task may have finished building one
        # while this task waited.
        if _producer is not None:
            return _producer

        settings = get_settings()
        producer = AIOKafkaProducer(
            client_id=settings.observability.otel_service_name,
            # See the module docstring. Neither of these is a tunable.
            acks="all",
            enable_idempotence=True,
            compression_type="gzip",
            linger_ms=_LINGER_MS,
            request_timeout_ms=_REQUEST_TIMEOUT_MS,
            max_request_size=_MAX_REQUEST_SIZE,
            **client_transport_kwargs(settings),
        )
        try:
            await producer.start()
        except (KafkaError, OSError, TimeoutError) as err:
            # Leave the singleton unset so the next call retries rather than
            # handing out a producer that never connected.
            await _quietly_stop(producer)
            raise DependencyUnavailableError.for_store("Redpanda", cause=err) from err
        _producer = producer
    return _producer


async def publish(
    payload: EventPayload,
    *,
    producer_name: str | None = None,
    correlation_id: str | None = None,
    tenant_id: str | None = None,
    client: AIOKafkaProducer | None = None,
) -> RecordMetadata:
    """Publish one typed payload. Returns only once the broker has acknowledged.

    The topic and the message key both come from the payload -- the event type
    routes it (`services/events/topics.py`) and the payload declares its own
    partition key -- so a caller cannot put a signal event on the graph topic or
    forget the key that preserves per-Signal ordering.

    `correlation_id` defaults to the ambient one, which is what threads a user's
    question through ingestion, enrichment and indexing as a single searchable
    chain (`docs/observability.md` §1). Passing it explicitly is for the case
    where the ambient chain is the *wrong* one -- a replayed DLQ message belongs
    to its original chain, not to the replay tool's.
    """
    envelope = EventEnvelope.wrap(
        payload,
        producer=producer_name or get_settings().observability.otel_service_name,
        correlation_id=correlation_id or get_correlation_id(),
        tenant_id=tenant_id,
    )
    return await publish_envelope(
        envelope,
        role=role_for_event(envelope.event_type),
        key=encode_key(payload.partition_key),
        client=client,
    )


async def publish_envelope(
    envelope: EventEnvelope,
    *,
    role: TopicRole,
    key: bytes | None,
    client: AIOKafkaProducer | None = None,
) -> RecordMetadata:
    """Publish a pre-built envelope to a topic. The single write path.

    Used directly by the consumer runtime, which routes a failed message to the
    DLQ with a key taken from the *original* message rather than from the DLQ
    payload. `client` is injectable so tests -- and any caller managing its own
    producer lifecycle -- need not touch the singleton.

    Any broker-side failure becomes `DependencyUnavailableError` (503, retryable),
    because the correct response is always the same: stop, do not commit, retry
    later. Callers that need to distinguish causes have the `cause` attribute.
    """
    kafka = client or await get_producer()
    topic = topic_name(role)
    try:
        async with asyncio.timeout(_PRODUCE_TIMEOUT_SECONDS):
            metadata: RecordMetadata = await kafka.send_and_wait(
                topic,
                value=envelope.to_bytes(),
                key=key,
                headers=envelope.to_headers(),
            )
    except (KafkaError, OSError, TimeoutError) as err:
        # No application-level retry loop. aiokafka already retries internally
        # under an idempotent producer id, so a second attempt here would only
        # duplicate that logic with worse information -- and the caller's own
        # recovery (do not commit the cursor, replay the page) is both cheaper
        # and already required for the crash case.
        logger.warning(
            "events.publish.failed",
            topic=topic,
            event_type=envelope.event_type.value,
            event_id=envelope.event_id,
            error=type(err).__name__,
        )
        raise DependencyUnavailableError.for_store("Redpanda", cause=err) from err

    logger.debug(
        "events.published",
        topic=metadata.topic,
        partition=metadata.partition,
        offset=metadata.offset,
        event_type=envelope.event_type.value,
        event_id=envelope.event_id,
    )
    return metadata


async def check_kafka() -> bool:
    """Probe the broker for `/readyz`. Never raises.

    Returns a bool rather than raising because readiness aggregates several
    dependencies and one being down must not prevent reporting on the others
    (`docs/observability.md`). Bounded by the shared probe budget from
    `backend/db/__init__.py`: a blackholed broker accepts the TCP connect and
    then never answers, and an unbounded metadata fetch would hold `/readyz` open
    past the liveness deadline.

    Fetches metadata rather than merely checking that a producer object exists --
    a producer that started before the broker went away still looks alive.
    """
    try:
        async with asyncio.timeout(HEALTH_PROBE_TIMEOUT_SECONDS):
            producer = await get_producer()
            await producer.client.fetch_all_metadata()
    except Exception:
        return False
    return True


async def dispose_producer() -> None:
    """Flush and close the producer, resetting module state. Idempotent.

    `stop()` flushes buffered batches before closing, which matters because
    `linger_ms` means there is almost always something buffered. Skipping this on
    shutdown drops whatever was in flight -- and those are messages whose cursors
    may already have been committed.
    """
    global _producer, _producer_lock
    if _producer is not None:
        await _quietly_stop(_producer)
    _producer = None
    _producer_lock = None


async def _quietly_stop(producer: AIOKafkaProducer) -> None:
    """Close a producer, swallowing failures.

    Shutdown is the one path where raising helps nobody: the process is going
    away, and an exception here would mask whatever error caused the shutdown in
    the first place.
    """
    try:
        await producer.stop()
    except Exception as err:  # noqa: BLE001 -- shutdown must not raise
        logger.warning("events.producer.stop_failed", error=type(err).__name__)
