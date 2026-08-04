"""The event envelope: what actually travels on the log, and how it evolves.

Every message on every OmniSense topic is one `EventEnvelope`: a small, stable
header wrapped around an opaque `payload`. The header is the part that must never
change shape, because it is the part a consumer reads *before* it knows whether
it can understand the body -- routing, correlation and version negotiation all
happen off the header alone.

Three decisions are encoded here.

**Consumers validate leniently, producers validate strictly**
(`docs/signal-model.md` §7). Every model in this module derives from
`LenientModel` (`extra="ignore"`) because producers and consumers deploy
independently: during a rolling deploy an old worker and a new worker read the
same partition, and a field the new producer added must be invisible to the old
reader rather than fatal to it. `Platform`, `SignalStatus` and friends are
`TolerantStrEnum`s for the same reason -- a new connector is a new `Platform`
member, and adding one must not require every consumer to redeploy first.

**Leniency stops at `schema_version`.** Additive change *within* a version is
tolerated silently; a version the reader does not know is refused loudly, via
`is_readable()`, and the message goes to the DLQ. That asymmetry is the whole
point of the field. A reader that leniently ignored a `schema_version` bump would
apply v1 semantics to a v2 body -- narrowed types, repurposed fields -- and write
plausible-looking wrong data into five stores. Refusing is recoverable; a DLQ can
be replayed once the reader is upgraded.

**The payload is a reference, not a copy.** `docs/data-stores.md` §5.1 puts the
raw bytes in R2 and the Signal in PostgreSQL, and publishes only the address on
the bus. Two reasons: a full article body per message turns the broker into a
second copy of the data store it is meant to coordinate, and -- more importantly
-- a consumer that read Signal *content* off the topic could index a version of a
Signal that the commit point never held. Consumers re-read from PostgreSQL, which
is the only store whose contents are definitionally true.

Layer note: this module imports `models/` (L0) and nothing else in the repository,
so it is safe to import from anywhere that speaks events.
"""

from __future__ import annotations

import base64
import binascii
import uuid
from typing import Any, ClassVar, Final, Self, TypeVar

from pydantic import Field, ValidationError

from models.base import LenientModel, Score, Sha256Hex, TolerantStrEnum, UtcDatetime, utcnow
from models.enums import (
    EdgeType,
    EntityType,
    Platform,
    SignalStatus,
    StageName,
)
from models.signal import signal_id

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "HEADER_CORRELATION_ID",
    "HEADER_EVENT_ID",
    "HEADER_EVENT_TYPE",
    "HEADER_PRODUCER",
    "HEADER_SCHEMA_VERSION",
    "HEADER_TENANT_ID",
    "MIN_READABLE_SCHEMA_VERSION",
    "DlqEvent",
    "EventDecodeError",
    "EventEnvelope",
    "EventPayload",
    "EventType",
    "GraphEdgeRef",
    "GraphNodeRef",
    "GraphUpdateEvent",
    "RawRecordEvent",
    "SignalEnrichedEvent",
    "exception_chain",
    "header_value",
]


EVENT_SCHEMA_VERSION: Final = 1
"""The envelope version this build produces.

Distinct from `models.lineage.CURRENT_SCHEMA_VERSION`, which versions the *Signal*.
They move independently: the envelope can gain a routing field without any change
to the Signal, and the Signal's identity derivation can change without altering
the shape of the header that carries it.
"""

MIN_READABLE_SCHEMA_VERSION: Final = 1
"""Oldest envelope version this build still understands.

Raised only when an old version is genuinely unreadable, which means the DLQ has
been drained of it first. Keeping a floor separate from the ceiling is what makes
the dual-read window in `docs/signal-model.md` §7 expressible: during a bump both
versions are readable, and the floor rises only after the backfill completes.
"""

_MAX_CHAIN_DEPTH: Final = 8
"""Ceiling on how far `exception_chain` walks. Bounds pathological nesting."""

UNBOUND_CORRELATION_ID: Final = "-"
"""What `correlation_id` reads as when a producer had no chain to attach.

Same sentinel as `backend/core/logging.py`, duplicated as a literal rather than
imported so that `models/` remains this module's only repository dependency --
`services/events/schemas.py` is imported by the connector-side publisher, and a
kernel import here would drag `backend/` into that path.
"""


class EventType(TolerantStrEnum):
    """What happened. Drives topic routing and payload interpretation.

    Dotted `noun.verb` values so a new event on an existing topic reads naturally
    (`signal.updated` alongside `signal.enriched`) and so the value is legible in
    a log line without a lookup table.

    `UNKNOWN` is load-bearing: a newer producer may emit an event type this build
    has never heard of, and the reader must be able to *parse* that message in
    order to route it to the DLQ with a correlation id attached. Raising during
    parse would leave a message that can be neither handled nor explained.
    """

    RECORD_RAW = "record.raw"
    SIGNAL_ENRICHED = "signal.enriched"
    GRAPH_UPDATE = "graph.update"
    DLQ_FAILED = "dlq.failed"
    UNKNOWN = "unknown"


class EventDecodeError(ValueError):
    """A message could not be turned into a usable envelope or payload.

    Raised rather than returned so that the consumer's DLQ path is the same code
    for "the bytes are not JSON", "the envelope is missing `event_type`" and "the
    payload does not match the event type it claims". All three are poison in the
    same way: no amount of retrying changes the bytes.
    """


# --------------------------------------------------------------------------- #
# Payload bodies
# --------------------------------------------------------------------------- #


class EventPayload(LenientModel):
    """Base for the typed bodies carried inside an envelope.

    Each subclass declares the single `EventType` it belongs to, which is what
    lets `EventEnvelope.wrap()` derive the type instead of asking the caller to
    restate it. A caller that has to pass both is a caller that can eventually
    pass a mismatched pair, and a mismatched pair routes a signal event onto the
    graph topic where nothing will ever read it.
    """

    EVENT_TYPE: ClassVar[EventType]

    @property
    def partition_key(self) -> str | None:
        """The value this event is ordered by. See `services/events/topics.py`.

        Every concrete payload overrides this. The base raises rather than
        returning `None` because "no key" is a real, distinct choice -- it means
        round-robin placement and no ordering guarantee at all -- and defaulting
        to it silently would hand a new event type the weakest possible semantics
        without anybody having decided that.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not declare a partition key; add a "
            "`partition_key` property, or return None explicitly if this event "
            "genuinely needs no ordering guarantee."
        )


class RawRecordEvent(EventPayload):
    """One fetched provider payload has been archived and is ready to enrich.

    Published by `services/connector_service.py` as step 2 of
    `docs/data-stores.md` §5.1, *after* the R2 PUT and *before* the enrichment
    pipeline runs. The bytes stay in R2: this event carries the address
    (`raw_object_key`, `raw_sha256`) and the provenance needed to rebuild
    `models.lineage.Lineage`, so a consumer can fetch exactly the bytes that were
    fetched, not a re-fetch of whatever the provider serves today.

    Field names deliberately mirror `Lineage` so the enrichment worker copies
    rather than translates. A translation table between two nearly-identical
    field sets is where `fetched_at` quietly becomes ingestion time.
    """

    EVENT_TYPE: ClassVar[EventType] = EventType.RECORD_RAW

    platform: Platform
    native_id: str
    connector_slug: str
    connector_version: str
    sync_run_id: str
    fetched_at: UtcDatetime = Field(default_factory=utcnow)

    raw_object_key: str | None = Field(
        default=None,
        description="R2 key of the immutable original. None only when the R2 PUT "
        "was deferred (`docs/architecture.md` §7.3 lets ingestion continue "
        "without R2); the consumer then has provenance without a payload and must "
        "re-fetch or defer rather than silently enrich nothing.",
    )
    raw_sha256: Sha256Hex | None = None
    raw_bytes: int | None = Field(default=None, ge=0)
    raw_content_type: str = "application/json"

    source_url: str | None = None
    request_fingerprint: str | None = None

    @property
    def partition_key(self) -> str:
        """The Signal id this record will become.

        Derived here rather than left to the enrichment worker so that the raw
        event and the enriched event for the same item carry the *same* key. That
        is what makes a re-fetch inside the connector's overlap window
        (`docs/connector-spec.md` §4.1 rule 3) land behind its earlier copy
        instead of racing it on another partition.
        """
        return signal_id(self.platform, self.native_id)


class SignalEnrichedEvent(EventPayload):
    """A Signal has been committed to PostgreSQL and may now be derived from.

    Published as step 5 of `docs/data-stores.md` §5.1 -- **after** the commit,
    never before, or the embedding, indexing and graph workers would build
    derived state from a transaction that then rolled back.

    Carries identity and version, not content. A consumer re-reads the Signal
    from PostgreSQL, which is the commit point; putting the body here would let a
    slow consumer index a stale copy while believing it was current, and no
    amount of idempotency keys repairs that.
    """

    EVENT_TYPE: ClassVar[EventType] = EventType.SIGNAL_ENRICHED

    signal_id: str
    platform: Platform
    native_id: str
    status: SignalStatus
    pipeline_version: str
    signal_schema_version: int = Field(
        default=1,
        ge=1,
        description="`lineage.schema_version` of the committed Signal. Separate "
        "from the envelope's own `schema_version`: a consumer can understand the "
        "envelope perfectly and still not understand the Signal inside it.",
    )
    confidence: Score = 0.0
    stored_at: UtcDatetime = Field(default_factory=utcnow)
    failed_stages: list[StageName] = Field(
        default_factory=list,
        description="Degradable stages that failed (`docs/signal-model.md` §5.2). "
        "Present so a consumer can skip work it knows is missing -- an indexing "
        "worker need not wait on embeddings for a Signal whose embedding stage "
        "failed -- without reading the whole lineage back out of PostgreSQL.",
    )

    @property
    def partition_key(self) -> str:
        """`signal_id`. The same key `RawRecordEvent` derived, one topic earlier."""
        return self.signal_id


class GraphNodeRef(LenientModel):
    """An entity to `MERGE` into Neo4j."""

    entity_id: str
    entity_type: EntityType
    canonical_name: str | None = None


class GraphEdgeRef(LenientModel):
    """A relationship to `MERGE` into Neo4j.

    `valid_from` is part of the identity, not decoration: `docs/data-stores.md`
    §5.2 fixes the edge idempotency key as `(from, type, to, valid_from)` so that
    "Acme acquired Foo" asserted in January and again in June are two facts, not
    one overwritten fact. `merge_key()` is that tuple, spelled once.
    """

    from_id: str
    edge_type: EdgeType
    to_id: str
    valid_from: UtcDatetime
    confidence: Score = 1.0

    def merge_key(self) -> tuple[str, str, str, str]:
        """The tuple Neo4j `MERGE`s on. Stable across redeliveries by construction."""
        return (self.from_id, self.edge_type.value, self.to_id, self.valid_from.isoformat())


class GraphUpdateEvent(EventPayload):
    """Entities and relationships extracted from one Signal, ready for the graph.

    A separate topic from `signals.enriched` because the graph writer is an
    independent consumer group with its own lag budget: `docs/architecture.md`
    §7.3 says Neo4j being down makes `omnisense.graph.updates` accumulate lag
    while indexing continues. Folding graph writes into the signal topic would
    couple the two and turn a Neo4j outage into an indexing outage.
    """

    EVENT_TYPE: ClassVar[EventType] = EventType.GRAPH_UPDATE

    signal_id: str
    observed_at: UtcDatetime = Field(default_factory=utcnow)
    nodes: list[GraphNodeRef] = Field(default_factory=list)
    edges: list[GraphEdgeRef] = Field(default_factory=list)

    @property
    def partition_key(self) -> str:
        """`signal_id`, not entity id.

        Keying by entity would be the obvious choice and is wrong here: one event
        carries several entities, so it could only be keyed by one of them and the
        rest would be ordered by an unrelated key. Keying by Signal keeps one
        Signal's graph writes ordered, which is the guarantee
        `graph/ingest/batcher.py` actually depends on. Cross-entity ordering is
        not offered by any partitioned log, and `MERGE` does not need it.
        """
        return self.signal_id


class DlqEvent(EventPayload):
    """A message that could not be handled, preserved verbatim for replay.

    `docs/connector-spec.md` §6 requires the DLQ to carry enough to replay a fixed
    handler against a historical failure without re-hitting the provider, so the
    original bytes travel base64-encoded in `body_b64` -- byte-exact, not a
    re-serialization of a parsed object, because much of what lands here failed
    precisely because it could not be parsed.

    **Only exception class names are recorded, never messages.** Same rule as
    `services/signal_engine/pipeline.py`: a provider or driver error message can
    echo the request that caused it, and requests carry fetched content
    (`docs/security-and-privacy.md`). `error_chain` is the chain of class names,
    outermost first, which is what makes a DLQ triageable -- you group by it --
    whereas a free-text message is what makes a DLQ a place where fetched content
    quietly accumulates outside its own retention policy.
    """

    EVENT_TYPE: ClassVar[EventType] = EventType.DLQ_FAILED

    original_topic: str
    original_partition: int | None = None
    original_offset: int | None = None
    original_key: str | None = None
    original_event_id: str | None = None
    original_event_type: EventType = EventType.UNKNOWN

    consumer_group: str
    attempts: int = Field(ge=1, description="Deliveries attempted before giving up.")
    error_chain: list[str] = Field(
        default_factory=list,
        description="Exception class names, outermost first. Never messages.",
    )
    failed_at: UtcDatetime = Field(default_factory=utcnow)

    body_b64: str = Field(
        default="",
        description="Base64 of the original message bytes. Deliberately a string "
        "rather than `bytes`: `LenientModel` sets `ser_json_bytes='base64'` "
        "without the matching `val_json_bytes`, so a `bytes` field serializes to "
        "base64 text and then re-validates that *text* as the raw value -- a "
        "silent, lossy round trip. An explicit string cannot round-trip wrongly.",
    )

    @property
    def partition_key(self) -> str | None:
        """The original message's key, so one poisoned Signal's failures stay together."""
        return self.original_key

    def body(self) -> bytes:
        """The original message bytes.

        Raises `EventDecodeError` on corrupt base64 rather than returning empty
        bytes: a replay tool that silently received `b""` would report success
        having replayed nothing at all.
        """
        try:
            return base64.b64decode(self.body_b64.encode("ascii"), validate=True)
        except (binascii.Error, UnicodeEncodeError, ValueError) as err:
            raise EventDecodeError(
                f"DLQ record for {self.original_topic}"
                f"[{self.original_partition}]@{self.original_offset} carries "
                "unreadable base64; the original bytes cannot be recovered from "
                "this record."
            ) from err

    @classmethod
    def from_failure(
        cls,
        *,
        topic: str,
        body: bytes,
        error: BaseException,
        consumer_group: str,
        attempts: int,
        partition: int | None = None,
        offset: int | None = None,
        key: str | None = None,
        envelope: EventEnvelope | None = None,
    ) -> Self:
        """Build a DLQ record from a failed delivery.

        `envelope` is optional on purpose: the failure may be that the body could
        not be parsed into one. The identifying fields then stay `None` and the
        raw bytes carry the whole story, which is exactly the case a DLQ exists
        for.
        """
        return cls(
            original_topic=topic,
            original_partition=partition,
            original_offset=offset,
            original_key=key,
            original_event_id=envelope.event_id if envelope is not None else None,
            original_event_type=(
                envelope.event_type if envelope is not None else EventType.UNKNOWN
            ),
            consumer_group=consumer_group,
            attempts=attempts,
            error_chain=exception_chain(error),
            body_b64=base64.b64encode(body).decode("ascii"),
        )


# --------------------------------------------------------------------------- #
# The envelope
# --------------------------------------------------------------------------- #

PayloadT = TypeVar("PayloadT", bound=EventPayload)


HEADER_EVENT_ID: Final = "os-event-id"
HEADER_EVENT_TYPE: Final = "os-event-type"
HEADER_SCHEMA_VERSION: Final = "os-schema-version"
HEADER_CORRELATION_ID: Final = "os-correlation-id"
HEADER_PRODUCER: Final = "os-producer"
HEADER_TENANT_ID: Final = "os-tenant-id"


class EventEnvelope(LenientModel):
    """The header every OmniSense message carries.

    `payload` is an untyped mapping here and typed at the point of use via
    `payload_as()`. That is deliberate: a consumer must be able to read the
    header of a message whose body it cannot interpret -- otherwise a single
    unknown event type on a shared topic stalls the partition instead of going
    to the DLQ with a correlation id and a type attached.
    """

    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    event_type: EventType
    schema_version: int = Field(default=EVENT_SCHEMA_VERSION, ge=1)
    occurred_at: UtcDatetime = Field(
        default_factory=utcnow,
        description="When the fact happened, not when it was published. A retried "
        "produce keeps the original value, so `occurred_at` stays usable as an "
        "ordering hint across a broker outage.",
    )
    correlation_id: str = Field(
        default=UNBOUND_CORRELATION_ID,
        description="The single join key across logs, metrics, traces and events "
        "(`docs/observability.md` §1). Defaulted rather than required so a "
        "message from a producer that forgot it is still readable -- the "
        "consumer mints one and logs that it did.",
    )
    tenant_id: str | None = Field(
        default=None,
        description="Present from the start even though Phase 1 is single-tenant. "
        "Adding a tenant discriminator to a log that already holds retained "
        "messages leaves every historical message ambiguous forever.",
    )
    producer: str = Field(
        description="Service name that emitted this, e.g. 'omnisense-ingestion'. "
        "The first question asked of a malformed message is who wrote it."
    )
    payload: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------- factories --

    @classmethod
    def wrap(
        cls,
        payload: EventPayload,
        *,
        producer: str,
        correlation_id: str | None = None,
        tenant_id: str | None = None,
    ) -> Self:
        """Build an envelope around a typed payload.

        `event_type` comes from the payload class, never from the caller, so the
        header and the body cannot disagree about what this message is.
        """
        return cls(
            event_type=type(payload).EVENT_TYPE,
            correlation_id=correlation_id or UNBOUND_CORRELATION_ID,
            tenant_id=tenant_id,
            producer=producer,
            payload=payload.model_dump(mode="json"),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> Self:
        """Parse a message body. Raises `EventDecodeError` on anything unusable.

        Wrapping Pydantic's `ValidationError` in a local type is what lets the
        consumer's DLQ path catch one thing instead of enumerating every library
        exception that "these bytes are not a message" can present as.
        """
        try:
            return cls.model_validate_json(data)
        except ValidationError as err:
            raise EventDecodeError(f"message body is not a valid event envelope: {err}") from err

    def to_bytes(self) -> bytes:
        """Serialize for the wire. JSON, UTF-8.

        JSON rather than Avro or Protobuf: the schema-registry machinery those
        need buys compactness and enforced compatibility, and this system already
        gets compatibility from lenient readers plus `schema_version`. It is not
        worth a second piece of infrastructure at Phase 1 volumes, and a message
        an operator can read with `rpk topic consume` is worth a great deal during
        an incident.
        """
        return self.model_dump_json().encode("utf-8")

    # ------------------------------------------------------------- accessors --

    def is_readable(self) -> bool:
        """Whether this build understands the envelope version.

        `False` means DLQ, not crash and not best-effort. See the module
        docstring: applying old semantics to a newer body is the failure mode that
        silently corrupts derived stores.
        """
        return MIN_READABLE_SCHEMA_VERSION <= self.schema_version <= EVENT_SCHEMA_VERSION

    def payload_as(self, model: type[PayloadT]) -> PayloadT:
        """Validate the payload as `model`, checking it matches the declared type.

        The type check is not redundant with validation. Lenient models ignore
        unknown fields and default the missing ones, so a `GraphUpdateEvent` body
        would validate *successfully* as a `SignalEnrichedEvent` were the required
        fields to overlap -- and a consumer would then act on a Signal it invented.
        The header is the authority on what the body is.
        """
        if self.event_type is not model.EVENT_TYPE:
            raise EventDecodeError(
                f"envelope declares event_type={self.event_type.value!r} but was "
                f"read as {model.__name__} (expects {model.EVENT_TYPE.value!r}); "
                "refusing to reinterpret the body"
            )
        try:
            return model.model_validate(self.payload)
        except ValidationError as err:
            raise EventDecodeError(f"payload does not satisfy {model.__name__}: {err}") from err

    def to_headers(self) -> list[tuple[str, bytes]]:
        """Kafka headers duplicating the routing fields of the envelope.

        The duplication is the point. A body that fails to parse still has to be
        logged against a correlation id and routed by type, and headers are
        readable without touching the body at all. Keys are prefixed `os-` so they
        cannot collide with headers a broker, a proxy or a mirroring tool adds.
        """
        headers = [
            (HEADER_EVENT_ID, self.event_id.encode("utf-8")),
            (HEADER_EVENT_TYPE, self.event_type.value.encode("utf-8")),
            (HEADER_SCHEMA_VERSION, str(self.schema_version).encode("utf-8")),
            (HEADER_CORRELATION_ID, self.correlation_id.encode("utf-8")),
            (HEADER_PRODUCER, self.producer.encode("utf-8")),
        ]
        if self.tenant_id is not None:
            headers.append((HEADER_TENANT_ID, self.tenant_id.encode("utf-8")))
        return headers


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def header_value(
    headers: list[tuple[str, bytes]] | tuple[tuple[str, bytes], ...] | None,
    name: str,
) -> str | None:
    """Read one header as text, tolerating absence and undecodable bytes.

    Never raises. This runs on the path that handles messages already known to be
    broken, and a `UnicodeDecodeError` while trying to find out *why* a message is
    broken would replace a DLQ record with a crash loop.
    """
    if not headers:
        return None
    for key, value in headers:
        if key == name:
            try:
                return value.decode("utf-8")
            except (UnicodeDecodeError, AttributeError):
                return None
    return None


def exception_chain(error: BaseException, *, max_depth: int = _MAX_CHAIN_DEPTH) -> list[str]:
    """Class names of an exception and its causes, outermost first.

    Class names only -- see `DlqEvent`. Follows `__cause__` (an explicit
    `raise ... from ...`) in preference to `__context__` (an incidental exception
    raised while handling another), because the explicit chain is the one an
    author meant to communicate. Cycles are possible in hand-constructed exception
    graphs, so identity is tracked and the walk is depth-bounded.
    """
    names: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and len(names) < max_depth and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return names
