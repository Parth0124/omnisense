"""Unit tests for `services/events/`: topics, envelope, producer and consumer.

`aiokafka` never opens a socket here. The consumer runtime is driven against
`FakeBroker` below -- an in-memory partitioned log that models the four
behaviours the runtime actually depends on: bounded fetch, per-partition
positions, pause/resume, and manual offset commit. Everything else about Kafka is
irrelevant to the code under test, and modelling it would only make the fake
wrong in new ways.

The four properties worth this much scaffolding, all of which fail silently in
production if broken:

1. **The commit happens after the handler.** An offset is a claim that work was
   done. If it can advance first, a crash produces a gap that no reconciler can
   detect, because nothing records that the message was ever seen.
2. **A failing handler does not advance the offset.** The counterpart: work that
   did not happen must be redeliverable.
3. **A poison message reaches the DLQ instead of blocking its partition.** Kafka
   offsets are strictly ordered, so one permanently-failing message stalls
   everything behind it forever unless something explicitly moves past it.
4. **The envelope round-trips leniently.** Producers and consumers deploy
   independently (`docs/signal-model.md` §7); a field added by a newer producer
   must be invisible to an older reader rather than fatal to it.
"""

from __future__ import annotations

import base64
import contextlib
from datetime import UTC, datetime
from typing import Any

import pytest
from aiokafka.errors import KafkaError
from aiokafka.structs import ConsumerRecord, RecordMetadata, TopicPartition

import services.events.producer as producer_module
from backend.core.config import Settings
from backend.core.exceptions import DependencyUnavailableError
from backend.core.logging import correlation_scope, get_correlation_id
from models.enums import EdgeType, EntityType, Platform, SignalStatus, StageName
from models.signal import signal_id
from services.events.consumer import ConsumedMessage, EventConsumer
from services.events.producer import (
    check_kafka,
    dispose_producer,
    get_producer,
    publish,
)
from services.events.schemas import (
    EVENT_SCHEMA_VERSION,
    HEADER_CORRELATION_ID,
    HEADER_EVENT_TYPE,
    DlqEvent,
    EventDecodeError,
    EventEnvelope,
    EventType,
    GraphEdgeRef,
    GraphNodeRef,
    GraphUpdateEvent,
    RawRecordEvent,
    SignalEnrichedEvent,
    exception_chain,
    header_value,
)
from services.events.topics import (
    DEFAULT_PARTITIONS,
    PRODUCTION_REPLICATION_FACTOR,
    RETENTION,
    SINGLE_NODE_REPLICATION_FACTOR,
    TopicRole,
    all_topic_specs,
    encode_key,
    partition_key_for_item,
    partition_key_for_signal,
    role_for_event,
    role_for_topic,
    topic_name,
    topic_spec,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fixtures and fakes
# --------------------------------------------------------------------------- #

FIXED_TIME = datetime(2026, 7, 31, 9, 15, tzinfo=UTC)


def _settings(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    """Build a `Settings` from explicit values, ignoring any local `.env`.

    Injected rather than monkeypatched into the process-wide singleton: several
    of these tests want a *different* configuration than the rest of the suite,
    and mutating a cached global would leak that into whatever ran next.

    `Settings` is the only reader of the environment (`backend/core/config.py`
    §2.9 -- the one module allowed to touch `os.environ` at all), so the values
    have to be visible there for the duration of the constructor call. This
    helper used to snapshot `os.environ`, `update()` it and restore in a
    `finally`, which was the last direct environment access left in the repo.

    `monkeypatch.setenv` instead, for two reasons. The standard is the real one:
    one hand-rolled exception makes "config.py is the only place that reads the
    environment" unenforceable by grep. The narrower one is that the `update()`
    sat *outside* the `try`, so the restore did not actually cover it -- an
    interruption landing between the mutation and the `try` would have leaked
    `OMNISENSE_ENV=prod` into every later test in the session. Teardown here is
    owed by the fixture protocol rather than by control flow reaching a
    `finally`, so there is no window to get wrong.

    Note that the value stays set for the rest of the *calling test*, where the
    old helper unset it on return. Nothing depends on the difference: no test
    calls `get_settings.cache_clear()`, so the process-wide singleton is built
    once, before any of this, and every call site here passes its `Settings`
    explicitly.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def _raw_event(native_id: str = "t3_abc123") -> RawRecordEvent:
    return RawRecordEvent(
        platform=Platform.REDDIT,
        native_id=native_id,
        connector_slug="reddit",
        connector_version="0.1.0",
        sync_run_id="run_1",
        fetched_at=FIXED_TIME,
        raw_object_key="raw/reddit/2026/07/31/abc.json",
        raw_sha256="a" * 64,
        raw_bytes=812,
    )


def _enriched_event(native_id: str = "t3_abc123") -> SignalEnrichedEvent:
    return SignalEnrichedEvent(
        signal_id=signal_id(Platform.REDDIT, native_id),
        platform=Platform.REDDIT,
        native_id=native_id,
        status=SignalStatus.PARTIAL,
        pipeline_version="1.0.0",
        confidence=0.62,
        stored_at=FIXED_TIME,
        failed_stages=[StageName.SENTIMENT],
    )


def _record(
    *,
    topic: str = "omnisense.records.raw",
    partition: int = 0,
    offset: int = 0,
    value: bytes = b"{}",
    key: bytes | None = None,
    headers: tuple[tuple[str, bytes], ...] = (),
) -> ConsumerRecord[bytes, bytes]:
    """One `ConsumerRecord`, spelled once. The real dataclass, not a stand-in."""
    return ConsumerRecord(
        topic=topic,
        partition=partition,
        offset=offset,
        timestamp=0,
        timestamp_type=0,
        key=key,
        value=value,
        checksum=None,
        serialized_key_size=len(key or b""),
        serialized_value_size=len(value),
        headers=headers,
    )


class FakeBroker:
    """An in-memory partitioned log.

    Models exactly what the runtime observes: records live at monotonically
    increasing offsets within a partition, a consumer has a position per
    partition, and a paused partition serves nothing.
    """

    def __init__(self) -> None:
        self.partitions: dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]] = {}
        self.produced: list[tuple[str, bytes | None, bytes]] = []

    def append(self, topic: str, partition: int, value: bytes, key: bytes | None = None) -> None:
        tp = TopicPartition(topic, partition)
        log = self.partitions.setdefault(tp, [])
        envelope_headers: tuple[tuple[str, bytes], ...] = ()
        # An unparseable body still carries headers on a real broker; leaving them
        # empty here is the harsher case, and the one worth testing.
        with contextlib.suppress(EventDecodeError):
            envelope_headers = tuple(EventEnvelope.from_bytes(value).to_headers())
        log.append(
            _record(
                topic=topic,
                partition=partition,
                offset=len(log),
                value=value,
                key=key,
                headers=envelope_headers,
            )
        )


class FakeConsumer:
    """`KafkaConsumerLike` over a `FakeBroker`, recording an ordered call trace.

    The trace is the point of the whole fake: assertions like "commit happened
    after the handler" are statements about *order*, and order is the one thing a
    return-value-only fake cannot express.
    """

    def __init__(self, broker: FakeBroker, trace: list[str] | None = None) -> None:
        self.broker = broker
        self.trace: list[str] = trace if trace is not None else []
        self.positions: dict[TopicPartition, int] = dict.fromkeys(broker.partitions, 0)
        self.committed: dict[TopicPartition, int] = {}
        self.paused: set[TopicPartition] = set()
        self.pause_history: list[tuple[str, tuple[TopicPartition, ...]]] = []
        self.seeks: list[tuple[TopicPartition, int]] = []
        self.started = False
        self.stopped = False
        self.commit_error: Exception | None = None
        self.max_records_seen: list[int | None] = []

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]]:
        self.max_records_seen.append(max_records)
        batch: dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]] = {}
        budget = max_records if max_records is not None else 10_000
        for tp, log in self.broker.partitions.items():
            if tp in self.paused or budget <= 0:
                continue
            start = self.positions.get(tp, 0)
            slice_ = log[start : start + budget]
            if not slice_:
                continue
            batch[tp] = slice_
            # A real consumer's position advances on fetch, not on commit. That
            # is precisely why the runtime has to `seek` to get a redelivery.
            self.positions[tp] = start + len(slice_)
            budget -= len(slice_)
        return batch

    async def commit(self, offsets: dict[TopicPartition, int] | None = None) -> None:
        self.trace.append("commit")
        if self.commit_error is not None:
            raise self.commit_error
        self.committed.update(offsets or {})

    def pause(self, *partitions: TopicPartition) -> None:
        self.trace.append("pause")
        self.pause_history.append(("pause", partitions))
        self.paused.update(partitions)

    def resume(self, *partitions: TopicPartition) -> None:
        self.trace.append("resume")
        self.pause_history.append(("resume", partitions))
        self.paused.difference_update(partitions)

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seeks.append((partition, offset))
        self.positions[partition] = offset

    def assignment(self) -> set[TopicPartition]:
        return set(self.broker.partitions)


class FakeKafkaProducer:
    """Enough of `AIOKafkaProducer` for the publish path. No sockets."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.start_error: Exception | None = None
        self.send_error: Exception | None = None
        self.sent: list[dict[str, Any]] = []

    async def start(self) -> None:
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def send_and_wait(
        self,
        topic: str,
        value: bytes | None = None,
        key: bytes | None = None,
        partition: int | None = None,
        timestamp_ms: int | None = None,
        headers: list[tuple[str, bytes]] | None = None,
    ) -> RecordMetadata:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append({"topic": topic, "value": value, "key": key, "headers": headers})
        return RecordMetadata(
            topic=topic,
            partition=0,
            topic_partition=TopicPartition(topic, 0),
            offset=len(self.sent) - 1,
            timestamp=0,
            timestamp_type=0,
            log_start_offset=0,
        )


@pytest.fixture
async def no_producer_singleton() -> Any:
    """Guarantee the module-level producer is unset before and after a test.

    Without this a test that builds one leaks it into the next, and the next test
    silently exercises a started fake instead of the code path it meant to.
    """
    await dispose_producer()
    yield
    await dispose_producer()


# --------------------------------------------------------------------------- #
# Topics
# --------------------------------------------------------------------------- #


class TestTopicPolicy:
    """The decisions in `topics.py` that are permanent once messages exist."""

    def test_names_come_from_settings_not_from_literals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment that prefixes its topics must not need a code change."""
        renamed = _settings(monkeypatch, KAFKA_TOPIC_SIGNALS="staging.omnisense.signals.enriched")
        assert topic_name(TopicRole.SIGNALS, settings=renamed) == (
            "staging.omnisense.signals.enriched"
        )
        assert role_for_topic("staging.omnisense.signals.enriched", settings=renamed) is (
            TopicRole.SIGNALS
        )

    def test_unknown_topic_resolves_to_none_rather_than_raising(self) -> None:
        """A mirrored or pattern-subscribed topic must not crash the runtime."""
        assert role_for_topic("someone.elses.topic") is None

    def test_every_role_has_partitions_retention_and_a_spec(self) -> None:
        """A role added without a policy would be auto-created with broker defaults.

        Broker defaults mean one partition and infinite retention -- exactly the
        configuration this module exists to prevent.
        """
        specs = {spec.role for spec in all_topic_specs()}
        assert specs == set(TopicRole)
        for role in TopicRole:
            assert role in DEFAULT_PARTITIONS
            assert role in RETENTION

    def test_raw_and_enriched_events_share_a_partition_key(self) -> None:
        """The whole ordering argument rests on this.

        A re-fetch inside the connector overlap window must queue behind its
        earlier copy, which only happens if both topics key the same item the
        same way.
        """
        raw = _raw_event()
        enriched = _enriched_event()
        assert raw.partition_key == enriched.partition_key
        assert partition_key_for_item(Platform.REDDIT, "t3_abc123") == partition_key_for_signal(
            enriched.signal_id
        )

    def test_different_signals_get_different_keys(self) -> None:
        """Keying must spread across partitions, or parallelism is theatre."""
        assert _raw_event("t3_a").partition_key != _raw_event("t3_b").partition_key

    def test_graph_updates_are_keyed_by_signal_not_entity(self) -> None:
        """One event carries several entities; only the Signal is a single key."""
        event = GraphUpdateEvent(
            signal_id="sig_deadbeef",
            nodes=[
                GraphNodeRef(entity_id="ent_a", entity_type=EntityType.COMPANY),
                GraphNodeRef(entity_id="ent_b", entity_type=EntityType.PRODUCT),
            ],
        )
        assert event.partition_key == "sig_deadbeef"

    def test_empty_partition_key_is_refused(self) -> None:
        """An empty key routes round-robin and silently drops the ordering guarantee."""
        with pytest.raises(ValueError, match="ordering"):
            partition_key_for_signal("")

    def test_dlq_is_single_partition_and_retained_longest(self) -> None:
        """A DLQ record waits on a human; expiring one before triage loses data."""
        assert DEFAULT_PARTITIONS[TopicRole.DLQ] == 1
        assert RETENTION[TopicRole.DLQ] == max(RETENTION.values())

    def test_event_types_route_to_their_topics(self) -> None:
        assert role_for_event(EventType.RECORD_RAW) is TopicRole.RAW_RECORDS
        assert role_for_event(EventType.SIGNAL_ENRICHED) is TopicRole.SIGNALS
        assert role_for_event(EventType.GRAPH_UPDATE) is TopicRole.GRAPH_UPDATES
        assert role_for_event(EventType.DLQ_FAILED) is TopicRole.DLQ

    def test_unroutable_event_type_raises_rather_than_defaulting(self) -> None:
        """A message from a newer producer parses as UNKNOWN.

        Publishing that to a default topic would put a message nobody understands
        somewhere it accrues lag forever; the consumer DLQs it instead.
        """
        with pytest.raises(ValueError, match="no topic"):
            role_for_event(EventType.UNKNOWN)

    def test_replication_factor_follows_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RF=3 fails on single-node Redpanda; RF=1 in production loses a partition.

        Both are real, and they are opposite, so the value cannot be a constant.
        """
        local_settings = _settings(monkeypatch, OMNISENSE_ENV="local")
        local = topic_spec(TopicRole.SIGNALS, settings=local_settings)
        assert local.replication_factor == SINGLE_NODE_REPLICATION_FACTOR

        prod = topic_spec(
            TopicRole.SIGNALS,
            settings=_settings(
                monkeypatch,
                OMNISENSE_ENV="prod",
                LOG_FORMAT="json",
                SECRET_KEY="not-the-placeholder",
                CREDENTIAL_ENCRYPTION_KEY="A" * 43 + "=",
                CORS_ALLOWED_ORIGINS="https://app.omnisense.test",
            ),
        )
        assert prod.replication_factor == PRODUCTION_REPLICATION_FACTOR

    def test_broker_config_is_stringly_typed_for_the_admin_api(self) -> None:
        spec = topic_spec(TopicRole.RAW_RECORDS)
        config = spec.broker_config()
        assert config["retention.ms"] == str(int(spec.retention.total_seconds() * 1000))
        assert config["cleanup.policy"] == "delete"

    def test_encode_key_passes_none_through(self) -> None:
        """`None` is a real choice -- round-robin, no ordering -- not an accident."""
        assert encode_key(None) is None
        assert encode_key("sig_1") == b"sig_1"


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #


class TestEnvelopeRoundTrip:
    """`docs/signal-model.md` §7: consumers validate leniently."""

    def test_round_trips_through_bytes(self) -> None:
        envelope = EventEnvelope.wrap(
            _enriched_event(), producer="omnisense-enrichment", correlation_id="abc123"
        )
        restored = EventEnvelope.from_bytes(envelope.to_bytes())

        assert restored.event_id == envelope.event_id
        assert restored.event_type is EventType.SIGNAL_ENRICHED
        assert restored.correlation_id == "abc123"
        assert restored.producer == "omnisense-enrichment"
        assert restored.payload_as(SignalEnrichedEvent).signal_id == (_enriched_event().signal_id)

    def test_a_field_added_by_a_newer_producer_is_ignored(self) -> None:
        """The rolling-deploy case: old reader, new writer, same partition."""
        envelope = EventEnvelope.wrap(_enriched_event(), producer="omnisense-enrichment")
        body = envelope.model_dump(mode="json")
        body["tracing_baggage"] = {"experiment": "v2"}
        body["payload"]["novelty_score"] = 0.9

        restored = EventEnvelope.model_validate(body)
        assert not hasattr(restored, "tracing_baggage")
        payload = restored.payload_as(SignalEnrichedEvent)
        assert payload.pipeline_version == "1.0.0"
        assert not hasattr(payload, "novelty_score")

    def test_an_unknown_event_type_parses_as_unknown_rather_than_raising(self) -> None:
        """A message must be parseable in order to be routed to the DLQ at all."""
        envelope = EventEnvelope.model_validate(
            {"event_type": "signal.retracted", "producer": "omnisense-future", "payload": {}}
        )
        assert envelope.event_type is EventType.UNKNOWN

    def test_a_newer_schema_version_is_not_readable(self) -> None:
        """Leniency stops at `schema_version`; guessing would corrupt derived stores."""
        current = EventEnvelope.wrap(_enriched_event(), producer="p")
        assert current.is_readable()

        future = EventEnvelope.model_validate(
            {
                "event_type": "signal.enriched",
                "schema_version": EVENT_SCHEMA_VERSION + 1,
                "producer": "omnisense-future",
                "payload": {},
            }
        )
        assert not future.is_readable()

    def test_payload_as_refuses_a_mismatched_event_type(self) -> None:
        """The header is the authority on what the body is."""
        envelope = EventEnvelope.wrap(_raw_event(), producer="omnisense-ingestion")
        with pytest.raises(EventDecodeError, match="refusing to reinterpret"):
            envelope.payload_as(SignalEnrichedEvent)

    def test_wrap_derives_event_type_from_the_payload_class(self) -> None:
        """A caller that could restate the type is a caller that could mis-state it."""
        assert EventEnvelope.wrap(_raw_event(), producer="p").event_type is EventType.RECORD_RAW
        assert (
            EventEnvelope.wrap(_enriched_event(), producer="p").event_type
            is EventType.SIGNAL_ENRICHED
        )

    def test_garbage_bytes_raise_one_typed_error(self) -> None:
        """The consumer's DLQ path catches one exception, not five library types."""
        with pytest.raises(EventDecodeError):
            EventEnvelope.from_bytes(b"<html>502 Bad Gateway</html>")

    def test_headers_carry_the_correlation_id_for_an_unparseable_body(self) -> None:
        """This is why the header duplicates the envelope: it survives a bad body."""
        headers = EventEnvelope.wrap(
            _raw_event(), producer="omnisense-ingestion", correlation_id="chain-9"
        ).to_headers()
        assert header_value(headers, HEADER_CORRELATION_ID) == "chain-9"
        assert header_value(headers, HEADER_EVENT_TYPE) == "record.raw"
        assert header_value(headers, "os-absent") is None
        assert header_value(None, HEADER_CORRELATION_ID) is None

    def test_graph_edge_merge_key_includes_valid_from(self) -> None:
        """`docs/data-stores.md` §5.2: the same claim at two times is two facts."""
        january = GraphEdgeRef(
            from_id="ent_a",
            edge_type=EdgeType.ACQUIRED,
            to_id="ent_b",
            valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        )
        june = january.model_copy(update={"valid_from": datetime(2026, 6, 1, tzinfo=UTC)})
        assert january.merge_key() != june.merge_key()


class TestDlqRecord:
    """A DLQ record has to be replayable, and must not leak fetched content."""

    def test_body_round_trips_byte_exactly(self) -> None:
        """Most of what lands here failed *because* it could not be parsed."""
        original = b"\x00\xff not utf-8, not json"
        event = DlqEvent.from_failure(
            topic="omnisense.records.raw",
            body=original,
            error=ValueError("boom"),
            consumer_group="omnisense",
            attempts=3,
            partition=2,
            offset=17,
        )
        restored = DlqEvent.model_validate_json(event.model_dump_json())
        assert restored.body() == original
        assert restored.original_partition == 2
        assert restored.original_offset == 17

    def test_error_chain_records_class_names_never_messages(self) -> None:
        """Provider messages echo requests, and requests carry fetched content."""
        try:
            try:
                raise KeyError("subreddit_name_that_should_not_be_logged")
            except KeyError as inner:
                raise RuntimeError("outer detail") from inner
        except RuntimeError as err:
            event = DlqEvent.from_failure(
                topic="omnisense.records.raw",
                body=b"{}",
                error=err,
                consumer_group="omnisense",
                attempts=1,
            )

        assert event.error_chain == ["RuntimeError", "KeyError"]
        serialized = event.model_dump_json()
        assert "subreddit_name_that_should_not_be_logged" not in serialized
        assert "outer detail" not in serialized

    def test_exception_chain_terminates_on_a_cycle(self) -> None:
        """Hand-constructed exception graphs can loop; the walk must not."""
        first = ValueError("a")
        second = TypeError("b")
        first.__cause__ = second
        second.__cause__ = first
        assert exception_chain(first) == ["ValueError", "TypeError"]

    def test_corrupt_base64_raises_rather_than_replaying_nothing(self) -> None:
        """A replay tool handed `b""` would report success having done nothing."""
        event = DlqEvent(
            original_topic="omnisense.records.raw",
            consumer_group="omnisense",
            attempts=1,
            body_b64="not base64 at all!!",
        )
        with pytest.raises(EventDecodeError, match="unreadable base64"):
            event.body()

    def test_carries_the_original_identity_when_the_envelope_parsed(self) -> None:
        envelope = EventEnvelope.wrap(_raw_event(), producer="omnisense-ingestion")
        event = DlqEvent.from_failure(
            topic="omnisense.records.raw",
            body=envelope.to_bytes(),
            error=RuntimeError("handler blew up"),
            consumer_group="omnisense",
            attempts=3,
            envelope=envelope,
        )
        assert event.original_event_id == envelope.event_id
        assert event.original_event_type is EventType.RECORD_RAW
        assert base64.b64decode(event.body_b64) == envelope.to_bytes()


# --------------------------------------------------------------------------- #
# Producer
# --------------------------------------------------------------------------- #


class TestProducerContract:
    """Durability guarantees the cursor-commit rule depends on."""

    def test_importing_the_module_creates_no_client(self) -> None:
        """Import-time I/O would make collecting the test suite need a broker."""
        assert producer_module._producer is None

    async def test_acks_all_and_idempotence_are_not_optional(
        self, monkeypatch: pytest.MonkeyPatch, no_producer_singleton: None
    ) -> None:
        """`docs/connector-spec.md` §4.1: commit after ack is only sound if an ack
        means durable. `acks=1` would make it a promise about one disk."""
        built: list[FakeKafkaProducer] = []

        def factory(**kwargs: Any) -> FakeKafkaProducer:
            built.append(FakeKafkaProducer(**kwargs))
            return built[-1]

        monkeypatch.setattr(producer_module, "AIOKafkaProducer", factory)
        await get_producer()

        assert built[0].kwargs["acks"] == "all"
        assert built[0].kwargs["enable_idempotence"] is True

    async def test_producer_is_a_started_singleton(
        self, monkeypatch: pytest.MonkeyPatch, no_producer_singleton: None
    ) -> None:
        """Connection pools are per process; a second one is a leaked socket."""
        built: list[FakeKafkaProducer] = []

        def factory(**kwargs: Any) -> FakeKafkaProducer:
            built.append(FakeKafkaProducer(**kwargs))
            return built[-1]

        monkeypatch.setattr(producer_module, "AIOKafkaProducer", factory)
        first = await get_producer()
        second = await get_producer()

        assert first is second
        assert len(built) == 1
        assert built[0].started

        await dispose_producer()
        assert built[0].stopped
        assert producer_module._producer is None

    async def test_a_failed_start_does_not_cache_a_dead_producer(
        self, monkeypatch: pytest.MonkeyPatch, no_producer_singleton: None
    ) -> None:
        """Caching one would make the outage permanent for the process lifetime."""

        def factory(**kwargs: Any) -> FakeKafkaProducer:
            fake = FakeKafkaProducer(**kwargs)
            fake.start_error = KafkaError("no brokers available")
            return fake

        monkeypatch.setattr(producer_module, "AIOKafkaProducer", factory)
        with pytest.raises(DependencyUnavailableError):
            await get_producer()
        assert producer_module._producer is None

    async def test_publish_routes_by_event_type_and_keys_by_signal(self) -> None:
        """Topic and key both come from the payload, so neither can be forgotten."""
        client = FakeKafkaProducer()
        payload = _enriched_event()

        await publish(payload, client=client, correlation_id="chain-1")

        sent = client.sent[0]
        assert sent["topic"] == topic_name(TopicRole.SIGNALS)
        assert sent["key"] == payload.signal_id.encode("utf-8")
        assert header_value(sent["headers"], HEADER_CORRELATION_ID) == "chain-1"
        assert EventEnvelope.from_bytes(sent["value"]).event_type is EventType.SIGNAL_ENRICHED

    async def test_publish_defaults_to_the_ambient_correlation_id(self) -> None:
        """One join key across logs, metrics, traces and events."""
        client = FakeKafkaProducer()
        with correlation_scope("ambient-7"):
            await publish(_raw_event(), client=client)
        assert EventEnvelope.from_bytes(client.sent[0]["value"]).correlation_id == "ambient-7"

    async def test_broker_failure_becomes_a_retryable_dependency_error(self) -> None:
        """The caller's correct response is always: stop, do not commit, retry."""
        client = FakeKafkaProducer()
        client.send_error = KafkaError("leader not available")

        with pytest.raises(DependencyUnavailableError) as raised:
            await publish(_raw_event(), client=client)
        assert raised.value.status_code == 503

    async def test_check_kafka_reports_false_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch, no_producer_singleton: None
    ) -> None:
        """`/readyz` aggregates probes; one raising would hide the others."""

        def factory(**kwargs: Any) -> FakeKafkaProducer:
            fake = FakeKafkaProducer(**kwargs)
            fake.start_error = OSError("connection refused")
            return fake

        monkeypatch.setattr(producer_module, "AIOKafkaProducer", factory)
        assert await check_kafka() is False


# --------------------------------------------------------------------------- #
# Consumer
# --------------------------------------------------------------------------- #


def _build_consumer(
    broker: FakeBroker,
    *,
    handler: Any,
    trace: list[str] | None = None,
    dlq_sink: list[EventEnvelope] | None = None,
    dlq_error: Exception | None = None,
    dlq_enabled: bool = True,
    max_attempts: int = 1,
    settings: Settings | None = None,
    batch_size: int | None = None,
    topics: tuple[str, ...] = ("omnisense.records.raw",),
) -> tuple[EventConsumer, FakeConsumer]:
    """Wire an `EventConsumer` onto the fake broker. No sockets, no settings edits."""
    fake = FakeConsumer(broker, trace=trace)

    async def dlq_publisher(envelope: EventEnvelope, *, key: bytes | None) -> None:
        if dlq_error is not None:
            raise dlq_error
        if dlq_sink is not None:
            dlq_sink.append(envelope)

    consumer = EventConsumer(
        topics=topics,
        handler=handler,
        consumer=fake,
        dlq_publisher=dlq_publisher,
        dlq_enabled=dlq_enabled,
        max_attempts=max_attempts,
        backoff_seconds=0.0,
        fetch_timeout_ms=1,
        batch_size=batch_size,
        settings=settings,
    )
    return consumer, fake


class TestConsumerDelivery:
    """At-least-once: the offset is a claim about work, not about bytes."""

    async def test_commit_happens_after_the_handler(self) -> None:
        """If the offset could move first, a crash would leave an undetectable gap."""
        broker = FakeBroker()
        for index in range(2):
            broker.append(
                "omnisense.records.raw",
                0,
                EventEnvelope.wrap(_raw_event(f"t3_{index}"), producer="p").to_bytes(),
            )

        trace: list[str] = []

        async def handler(message: ConsumedMessage) -> None:
            trace.append(f"handle:{message.offset}")

        consumer, fake = _build_consumer(broker, handler=handler, trace=trace)
        outcome = await consumer.run_once()

        assert trace == ["pause", "handle:0", "handle:1", "commit", "resume"]
        assert outcome.handled == 2
        assert fake.committed == {TopicPartition("omnisense.records.raw", 0): 2}

    async def test_partitions_are_paused_while_the_batch_is_in_flight(self) -> None:
        """Backpressure (`docs/architecture.md` §7.2): lag, not heap growth."""
        broker = FakeBroker()
        broker.append(
            "omnisense.records.raw", 0, EventEnvelope.wrap(_raw_event(), producer="p").to_bytes()
        )

        paused_during_handler: list[bool] = []
        fake_holder: dict[str, FakeConsumer] = {}

        async def handler(message: ConsumedMessage) -> None:
            paused_during_handler.append(
                TopicPartition(message.topic, message.partition) in fake_holder["fake"].paused
            )

        consumer, fake = _build_consumer(broker, handler=handler)
        fake_holder["fake"] = fake
        await consumer.run_once()

        assert paused_during_handler == [True]
        assert fake.paused == set()
        assert [call for call, _ in fake.pause_history] == ["pause", "resume"]

    async def test_a_failing_handler_does_not_advance_the_offset(self) -> None:
        """The counterpart of commit-after-handler: undone work stays redeliverable."""
        broker = FakeBroker()
        broker.append(
            "omnisense.records.raw", 0, EventEnvelope.wrap(_raw_event(), producer="p").to_bytes()
        )

        dependency_is_down = True
        replay: list[int] = []

        async def handler(message: ConsumedMessage) -> None:
            if dependency_is_down:
                raise RuntimeError("postgres is down")
            replay.append(message.offset)

        consumer, fake = _build_consumer(broker, handler=handler, dlq_enabled=False)
        outcome = await consumer.run_once()

        assert fake.committed == {}
        assert outcome.committed == {}
        assert outcome.blocked == 1
        # Rewound, so the message is redelivered in this process rather than
        # waiting for a rebalance that may never come.
        assert fake.seeks == [(TopicPartition("omnisense.records.raw", 0), 0)]

        dependency_is_down = False
        await consumer.run_once()
        assert replay == [0]
        assert fake.committed == {TopicPartition("omnisense.records.raw", 0): 1}

    async def test_messages_behind_a_blocked_one_are_not_handled(self) -> None:
        """Skipping ahead would break the per-Signal ordering the key exists for."""
        broker = FakeBroker()
        for index in range(3):
            broker.append(
                "omnisense.records.raw",
                0,
                EventEnvelope.wrap(_raw_event(f"t3_{index}"), producer="p").to_bytes(),
            )

        seen: list[int] = []

        async def handler(message: ConsumedMessage) -> None:
            seen.append(message.offset)
            if message.offset == 1:
                raise RuntimeError("transient")

        consumer, fake = _build_consumer(broker, handler=handler, dlq_enabled=False)
        outcome = await consumer.run_once()

        assert seen == [0, 1]
        assert fake.committed == {TopicPartition("omnisense.records.raw", 0): 1}
        assert outcome.blocked == 2

    async def test_the_handler_is_retried_before_the_message_is_declared_poison(self) -> None:
        """In-batch retries absorb a blip; they are not a substitute for the DLQ."""
        broker = FakeBroker()
        broker.append(
            "omnisense.records.raw", 0, EventEnvelope.wrap(_raw_event(), producer="p").to_bytes()
        )

        attempts: list[int] = []

        async def handler(message: ConsumedMessage) -> None:
            attempts.append(message.offset)
            if len(attempts) < 3:
                raise ConnectionError("blip")

        consumer, fake = _build_consumer(broker, handler=handler, max_attempts=3)
        outcome = await consumer.run_once()

        assert len(attempts) == 3
        assert outcome.handled == 1
        assert outcome.dlq_routed == 0
        assert fake.committed == {TopicPartition("omnisense.records.raw", 0): 1}


class TestConsumerPoisonHandling:
    """One permanently-failing message must not stall a partition forever."""

    async def test_a_poisoned_message_reaches_the_dlq_and_the_partition_advances(
        self,
    ) -> None:
        broker = FakeBroker()
        broker.append(
            "omnisense.records.raw",
            0,
            EventEnvelope.wrap(_raw_event("t3_poison"), producer="p").to_bytes(),
        )
        broker.append(
            "omnisense.records.raw",
            0,
            EventEnvelope.wrap(_raw_event("t3_healthy"), producer="p").to_bytes(),
        )

        dlq: list[EventEnvelope] = []
        handled: list[str] = []

        async def handler(message: ConsumedMessage) -> None:
            native_id = message.envelope.payload_as(RawRecordEvent).native_id
            if native_id == "t3_poison":
                raise ValueError("this record will never parse")
            handled.append(native_id)

        consumer, fake = _build_consumer(broker, handler=handler, dlq_sink=dlq)
        outcome = await consumer.run_once()

        # The message behind the poison one was still delivered.
        assert handled == ["t3_healthy"]
        assert outcome.dlq_routed == 1
        assert fake.committed == {TopicPartition("omnisense.records.raw", 0): 2}

        parked = dlq[0].payload_as(DlqEvent)
        assert parked.original_topic == "omnisense.records.raw"
        assert parked.original_offset == 0
        assert parked.error_chain == ["ValueError"]
        assert EventEnvelope.from_bytes(parked.body()).payload_as(RawRecordEvent).native_id == (
            "t3_poison"
        )

    async def test_an_undecodable_body_is_parked_without_retrying(self) -> None:
        """Retrying does not change bytes; it only holds the partition longer."""
        broker = FakeBroker()
        broker.append("omnisense.records.raw", 0, b"<html>502 Bad Gateway</html>")

        calls: list[str] = []
        dlq: list[EventEnvelope] = []

        async def handler(message: ConsumedMessage) -> None:
            calls.append("called")

        consumer, fake = _build_consumer(broker, handler=handler, dlq_sink=dlq, max_attempts=5)
        outcome = await consumer.run_once()

        assert calls == []
        assert outcome.dlq_routed == 1
        assert dlq[0].payload_as(DlqEvent).attempts == 1
        assert fake.committed == {TopicPartition("omnisense.records.raw", 0): 1}

    async def test_an_unreadable_schema_version_is_parked_not_reinterpreted(self) -> None:
        """Applying v1 semantics to a v2 body writes wrong data into five stores."""
        broker = FakeBroker()
        future = EventEnvelope.wrap(_raw_event(), producer="omnisense-future").model_dump(
            mode="json"
        )
        future["schema_version"] = EVENT_SCHEMA_VERSION + 1
        broker.append(
            "omnisense.records.raw",
            0,
            EventEnvelope.model_validate(future).model_dump_json().encode("utf-8"),
        )

        calls: list[str] = []
        dlq: list[EventEnvelope] = []

        async def handler(message: ConsumedMessage) -> None:
            calls.append("called")

        consumer, fake = _build_consumer(broker, handler=handler, dlq_sink=dlq)
        await consumer.run_once()

        assert calls == []
        assert len(dlq) == 1
        assert fake.committed == {TopicPartition("omnisense.records.raw", 0): 1}

    async def test_a_dlq_publish_failure_blocks_rather_than_dropping(self) -> None:
        """Advancing anyway would leave the message nowhere at all."""
        broker = FakeBroker()
        broker.append(
            "omnisense.records.raw", 0, EventEnvelope.wrap(_raw_event(), producer="p").to_bytes()
        )

        async def handler(message: ConsumedMessage) -> None:
            raise RuntimeError("handler failed")

        consumer, fake = _build_consumer(
            broker, handler=handler, dlq_error=KafkaError("dlq topic unavailable")
        )
        outcome = await consumer.run_once()

        assert fake.committed == {}
        assert outcome.blocked == 1
        assert fake.seeks == [(TopicPartition("omnisense.records.raw", 0), 0)]

    async def test_a_failure_on_the_dlq_topic_is_not_written_back_to_the_dlq(self) -> None:
        """Re-parking a DLQ message is an infinite loop that grows the topic."""
        broker = FakeBroker()
        dlq_topic = topic_name(TopicRole.DLQ)
        broker.append(
            dlq_topic,
            0,
            EventEnvelope.wrap(
                DlqEvent(
                    original_topic="omnisense.records.raw",
                    consumer_group="omnisense",
                    attempts=1,
                    body_b64="",
                ),
                producer="p",
            ).to_bytes(),
        )

        dlq: list[EventEnvelope] = []

        async def handler(message: ConsumedMessage) -> None:
            raise RuntimeError("replay failed again")

        consumer, fake = _build_consumer(broker, handler=handler, dlq_sink=dlq, topics=(dlq_topic,))
        await consumer.run_once()

        assert dlq == []
        # Advanced anyway: a stuck DLQ message must not block triage of the rest.
        assert fake.committed == {TopicPartition(dlq_topic, 0): 1}


class TestConsumerRuntime:
    """Backpressure bounds, correlation propagation and commit-failure recovery."""

    async def test_the_batch_is_bounded_by_the_smaller_of_the_two_limits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`INGESTION_BATCH_SIZE` and `KAFKA_MAX_POLL_RECORDS` must both hold."""
        broker = FakeBroker()
        for index in range(5):
            broker.append(
                "omnisense.records.raw",
                0,
                EventEnvelope.wrap(_raw_event(f"t3_{index}"), producer="p").to_bytes(),
            )

        handled: list[int] = []

        async def handler(message: ConsumedMessage) -> None:
            handled.append(message.offset)

        consumer, fake = _build_consumer(
            broker,
            handler=handler,
            settings=_settings(monkeypatch, INGESTION_BATCH_SIZE="3", KAFKA_MAX_POLL_RECORDS="2"),
        )
        assert consumer.batch_size == 2

        await consumer.run_once()
        assert handled == [0, 1]
        assert fake.max_records_seen == [2]

    async def test_the_envelope_correlation_id_is_bound_around_the_handler(self) -> None:
        """A worker that minted its own id would break the chain at the boundary."""
        broker = FakeBroker()
        broker.append(
            "omnisense.records.raw",
            0,
            EventEnvelope.wrap(_raw_event(), producer="p", correlation_id="chain-42").to_bytes(),
        )

        observed: list[str] = []

        async def handler(message: ConsumedMessage) -> None:
            observed.append(get_correlation_id())

        consumer, _fake = _build_consumer(broker, handler=handler)
        await consumer.run_once()

        assert observed == ["chain-42"]
        # Restored afterwards, so the next message is not logged under this chain.
        assert get_correlation_id() == "-"

    async def test_a_failed_commit_rewinds_the_whole_batch(self) -> None:
        """Uncommitted work with an advanced position is a gap nothing can detect."""
        broker = FakeBroker()
        for index in range(2):
            broker.append(
                "omnisense.records.raw",
                0,
                EventEnvelope.wrap(_raw_event(f"t3_{index}"), producer="p").to_bytes(),
            )

        handled: list[int] = []

        async def handler(message: ConsumedMessage) -> None:
            handled.append(message.offset)

        consumer, fake = _build_consumer(broker, handler=handler)
        fake.commit_error = KafkaError("coordinator not available")
        outcome = await consumer.run_once()

        assert handled == [0, 1]
        assert fake.committed == {}
        assert outcome.committed == {}
        assert fake.seeks == [(TopicPartition("omnisense.records.raw", 0), 0)]

    async def test_an_unexpected_error_rewinds_instead_of_skipping(self) -> None:
        """An advanced position with no commit hides messages until a rebalance.

        They are not lost -- a restart replays them -- but on a long-lived worker
        "until a restart" can be days, and nothing reports the delay.
        """
        broker = FakeBroker()
        for index in range(3):
            broker.append(
                "omnisense.records.raw",
                0,
                EventEnvelope.wrap(_raw_event(f"t3_{index}"), producer="p").to_bytes(),
            )

        async def handler(message: ConsumedMessage) -> None:
            if message.offset == 1:
                raise BaseException("SIGTERM-shaped, not an ordinary failure")

        consumer, fake = _build_consumer(broker, handler=handler)
        with pytest.raises(BaseException, match="SIGTERM-shaped"):
            await consumer.run_once()

        # Rewound to the batch start, and the partition was still resumed.
        assert fake.seeks == [(TopicPartition("omnisense.records.raw", 0), 0)]
        assert fake.committed == {}
        assert fake.paused == set()

    async def test_an_empty_fetch_neither_commits_nor_pauses(self) -> None:
        """An idle topic must not produce work, log noise or offset churn."""

        async def handler(message: ConsumedMessage) -> None:
            raise AssertionError("handler must not run on an empty batch")

        consumer, fake = _build_consumer(FakeBroker(), handler=handler)
        outcome = await consumer.run_once()

        assert outcome.is_empty
        assert fake.pause_history == []
        assert fake.trace == []

    async def test_run_without_start_fails_loudly(self) -> None:
        """A wiring bug should not surface as `AttributeError` on `None`."""

        async def handler(message: ConsumedMessage) -> None:
            return None

        consumer = EventConsumer(topics=(TopicRole.RAW_RECORDS,), handler=handler)
        with pytest.raises(RuntimeError, match=r"start\(\) has not been called"):
            await consumer.run_once()

    async def test_a_topic_role_is_resolved_to_its_configured_name(self) -> None:
        """Workers subscribe by role; the name is a deployment detail."""

        async def handler(message: ConsumedMessage) -> None:
            return None

        consumer = EventConsumer(topics=(TopicRole.SIGNALS,), handler=handler)
        assert consumer.topics == (topic_name(TopicRole.SIGNALS),)

    async def test_zero_attempts_is_rejected_at_construction(self) -> None:
        """`max_attempts=0` would never call the handler and park everything."""

        async def handler(message: ConsumedMessage) -> None:
            return None

        with pytest.raises(ValueError, match="at least 1"):
            EventConsumer(topics=(TopicRole.SIGNALS,), handler=handler, max_attempts=0)

    async def test_stop_ends_the_run_loop_after_the_current_batch(self) -> None:
        """Graceful shutdown never abandons a batch between handler and commit."""
        broker = FakeBroker()
        broker.append(
            "omnisense.records.raw", 0, EventEnvelope.wrap(_raw_event(), producer="p").to_bytes()
        )

        consumer_holder: dict[str, EventConsumer] = {}
        handled: list[int] = []

        async def handler(message: ConsumedMessage) -> None:
            handled.append(message.offset)
            await consumer_holder["consumer"].stop()

        consumer, fake = _build_consumer(broker, handler=handler)
        consumer_holder["consumer"] = consumer
        await consumer.run()

        assert handled == [0]
        assert fake.committed == {TopicPartition("omnisense.records.raw", 0): 1}
