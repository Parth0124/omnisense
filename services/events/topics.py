"""The topic inventory: names, partition keys, partition counts and retention.

ADR-0007 makes the ingestion path a replayable log rather than a queue, and this
module is where the three decisions that are effectively permanent once messages
exist are written down: what a topic is called, how a message is keyed, and how
long the log keeps it.

The partition key
-----------------
**Every message is keyed by `Signal.id`.** Kafka guarantees ordering only within
a partition, and the key is what chooses the partition, so the key *is* the
ordering guarantee. Keying by Signal id gives exactly the guarantee the pipeline
needs and no more:

- The enrichment -> index/graph fan-out is safe to run with many consumers,
  because every event about one Signal -- the raw record, the enriched
  notification, the graph update -- lands on one partition and is therefore
  handled in order by one consumer. A re-fetch inside the connector's overlap
  window (`docs/connector-spec.md` §4.1 rule 3) queues behind its earlier copy
  rather than racing it, which is what makes at-least-once delivery converge
  instead of oscillating between two versions of the same row
  (`docs/data-stores.md` §5.2).
- Nothing weaker would do. Ordering across *different* Signals is explicitly not
  guaranteed (`docs/data-stores.md` §5.3), so there is no reason to pay for it.

The alternatives were considered and rejected, and the ADR left the choice open:

- **Key by connector or platform.** Reddit and RSS are the Phase 1 sources; RSS
  alone can outproduce everything else during a backfill. One key per source
  means one partition per source, so the busiest connector pins a single
  partition and a single consumer while the rest idle -- consumer parallelism
  collapses to "number of active sources", and adding replicas does nothing.
- **Key by entity.** An event mentions several entities, so it can be keyed by at
  most one of them and the others get ordered by an unrelated key. It also makes
  ordering depend on entity *resolution*, which is a model output that changes
  between pipeline versions -- the partition a message lands on would move when
  the extractor was retrained.
- **Key by tenant.** Correct for isolation, catastrophic for skew: a single large
  tenant is one partition. Tenant lives in the envelope
  (`services/events/schemas.py`) where filtering can use it without constraining
  placement.

Partition count is part of the same decision and is harder to change: increasing
it rehashes every key, so ordering is broken *across the change* for keys that
move. The counts below are therefore sized for the parallelism Phase 2 will want,
not for today's volume -- over-provisioning partitions costs a little broker
memory, under-provisioning costs a migration nobody can perform online.

Retention
---------
ADR-0007 left the retention window open, noting that long retention on
`omnisense.records.raw` duplicates the R2 archive. It is decided here: **R2 is the
archive, the log is the coordination bus.** R2 keeps raw payloads for
`R2_RAW_RETENTION_DAYS` (400 days, matching the Signal retention window), and
`scripts/reindex.py` reprocesses from PostgreSQL and R2 rather than from Kafka.
The log therefore only needs to cover the window in which *offset-reset replay*
is the right tool -- a bad deploy noticed over a weekend, a consumer that fell
over on Friday -- which is days, not months. The DLQ is the exception: it retains
far longer because a message there is waiting on a human, and a DLQ that expires
before triage is silent data loss.

Layer note: this module reads configuration, so it belongs to `services/` and may
not be imported by `connectors/` (`docs/architecture.md` §6.2 rule 2). A connector
never names a topic; `services/connector_service.py` does the producing.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from backend.core.config import Settings, get_settings
from models.enums import Platform
from models.signal import signal_id
from services.events.schemas import EventType

__all__ = [
    "DEFAULT_PARTITIONS",
    "PRODUCTION_REPLICATION_FACTOR",
    "RETENTION",
    "SINGLE_NODE_REPLICATION_FACTOR",
    "TopicRole",
    "TopicSpec",
    "all_topic_specs",
    "encode_key",
    "partition_key_for_item",
    "partition_key_for_signal",
    "role_for_event",
    "role_for_topic",
    "topic_name",
    "topic_spec",
]


class TopicRole(enum.StrEnum):
    """What a topic is *for*, independent of what it is called.

    Names are configurable (`KAFKA_TOPIC_*`) because a shared production cluster
    usually wants an environment prefix. Code therefore refers to roles and
    resolves the name once, here. The alternative -- passing
    `settings.kafka.topic_signals` through every call site -- puts a settings read
    in a dozen modules and makes "which topics does this service touch?"
    ungreppable.

    No `UNKNOWN` member: this is a closed set owned by this repository. A topic
    nobody declared is a configuration error, not version skew.
    """

    RAW_RECORDS = "raw_records"
    SIGNALS = "signals"
    GRAPH_UPDATES = "graph_updates"
    DLQ = "dlq"


DEFAULT_PARTITIONS: Final[dict[TopicRole, int]] = {
    # Sized for consumer parallelism, not for volume: a consumer group can never
    # have more useful members than the topic has partitions. Six covers the
    # Phase 2 fan-out (embedding, graph, indexing workers scaled to two replicas
    # each) and divides evenly by 1, 2, 3 and 6, so scaling a group up or down
    # keeps partitions balanced across its members.
    TopicRole.RAW_RECORDS: 6,
    TopicRole.SIGNALS: 6,
    # Graph writes are batched by `graph/ingest/batcher.py` into single
    # transactions, and Neo4j is the least parallel of the derived stores --
    # more partitions here would buy contention on the same nodes rather than
    # throughput.
    TopicRole.GRAPH_UPDATES: 3,
    # One partition. The DLQ is read by a human and by `workers/dlq.py`, both of
    # which want a single ordered stream; volume is by definition tiny, and a
    # single partition makes "read everything that failed today" a linear scan
    # instead of a merge across six cursors.
    TopicRole.DLQ: 1,
}
"""Partition counts. Increasing one later rehashes keys -- see the module docstring."""


RETENTION: Final[dict[TopicRole, timedelta]] = {
    # Long enough to reset an offset and replay a weekend's ingestion after a bad
    # deploy. Beyond that the R2 archive plus `scripts/reindex.py` is the replay
    # path, and it is the better one: it reprocesses what was *stored*, not what
    # happened to still be on the bus.
    TopicRole.RAW_RECORDS: timedelta(days=7),
    TopicRole.SIGNALS: timedelta(days=7),
    # Graph updates are cheap to regenerate from PostgreSQL, so the window only
    # has to outlast a Neo4j outage plus the backlog it leaves behind.
    TopicRole.GRAPH_UPDATES: timedelta(days=3),
    # A DLQ record is waiting on a person. Expiring one before triage is silent
    # data loss and exactly the failure ADR-0007 warns about -- "a DLQ that a
    # human must actually triage or it silently becomes a data-loss bucket".
    TopicRole.DLQ: timedelta(days=30),
}
"""Retention per role. See the module docstring for why the log is short-lived."""


SINGLE_NODE_REPLICATION_FACTOR: Final = 1
"""What a one-broker Redpanda can host. Requesting more fails topic creation."""

PRODUCTION_REPLICATION_FACTOR: Final = 3
"""Tolerates one broker loss with `min.insync.replicas=2` still satisfiable.

Load-bearing together with the producer's `acks=all`: an ack means "written to
every in-sync replica", which is only a durability statement if there is more
than one replica. An RF=1 production topic makes `acks=all` a promise about a
single disk, and the cursor-commit-after-ack rule in `docs/connector-spec.md`
§4.1 then loses records on any broker failure.
"""


@dataclass(frozen=True, slots=True)
class TopicSpec:
    """Everything needed to create one topic, and to explain why it looks that way.

    Frozen because a spec is a description of the cluster's intended state, not a
    builder. Code that wants a variation asks for a different role.
    """

    role: TopicRole
    name: str
    partitions: int
    replication_factor: int
    retention: timedelta
    key_meaning: str
    cleanup_policy: str = "delete"

    @property
    def retention_ms(self) -> int:
        """`retention.ms` as the broker wants it: whole milliseconds."""
        return int(self.retention.total_seconds() * 1000)

    def broker_config(self) -> dict[str, str]:
        """Per-topic config for an admin client or `rpk topic create -c ...`.

        Returned as strings because that is the only type the Kafka protocol has
        for topic configuration; converting at the boundary keeps the rest of this
        module in real types.
        """
        return {
            "retention.ms": str(self.retention_ms),
            "cleanup.policy": self.cleanup_policy,
        }


# --------------------------------------------------------------------------- #
# Name resolution
# --------------------------------------------------------------------------- #


def topic_name(role: TopicRole, *, settings: Settings | None = None) -> str:
    """Resolve a role to its configured topic name.

    `settings` is injectable so a test can exercise a renamed deployment without
    mutating the process-wide singleton; production always takes the default.
    """
    kafka = (settings or get_settings()).kafka
    match role:
        case TopicRole.RAW_RECORDS:
            return kafka.topic_raw_records
        case TopicRole.SIGNALS:
            return kafka.topic_signals
        case TopicRole.GRAPH_UPDATES:
            return kafka.topic_graph_updates
        case TopicRole.DLQ:
            return kafka.topic_dlq


def role_for_topic(name: str, *, settings: Settings | None = None) -> TopicRole | None:
    """Reverse of `topic_name`. `None` for a topic this build does not own.

    Used by the consumer runtime to describe a message by role in logs and DLQ
    records. `None` rather than a raise: a consumer subscribed to a topic by
    pattern, or one reading a mirrored topic, must not fall over because the name
    is unfamiliar.
    """
    resolved = settings or get_settings()
    for role in TopicRole:
        if topic_name(role, settings=resolved) == name:
            return role
    return None


_EVENT_ROLE: Final[dict[EventType, TopicRole]] = {
    EventType.RECORD_RAW: TopicRole.RAW_RECORDS,
    EventType.SIGNAL_ENRICHED: TopicRole.SIGNALS,
    EventType.GRAPH_UPDATE: TopicRole.GRAPH_UPDATES,
    EventType.DLQ_FAILED: TopicRole.DLQ,
}
"""Which topic each event type belongs on.

Routing lives here rather than on the payload classes so that `schemas.py` stays
free of configuration -- it is the one module in this package importable from
anywhere, including code that must not read settings.
"""


def role_for_event(event_type: EventType) -> TopicRole:
    """Route an event type to its topic role.

    Raises on an unroutable type instead of guessing. `EventType` is tolerant, so
    a message from a newer producer parses as `UNKNOWN`; publishing that to some
    default topic would put a message no consumer understands onto a topic where
    it silently accrues lag. The consumer sends it to the DLQ instead, which is
    the one destination that is always correct for "I do not know what this is".
    """
    role = _EVENT_ROLE.get(event_type)
    if role is None:
        raise ValueError(
            f"event type {event_type.value!r} has no topic; add it to "
            "services/events/topics.py::_EVENT_ROLE, or route it to the DLQ"
        )
    return role


# --------------------------------------------------------------------------- #
# Specs
# --------------------------------------------------------------------------- #

_KEY_MEANING: Final[dict[TopicRole, str]] = {
    TopicRole.RAW_RECORDS: "Signal.id derived from (platform, native_id)",
    TopicRole.SIGNALS: "Signal.id",
    TopicRole.GRAPH_UPDATES: "Signal.id of the Signal the entities came from",
    TopicRole.DLQ: "the failed message's original key, when it had one",
}


def topic_spec(role: TopicRole, *, settings: Settings | None = None) -> TopicSpec:
    """Full provisioning spec for one topic.

    The replication factor follows the environment because the two failure modes
    are opposite and both are silent-ish: asking a single-node local Redpanda for
    RF=3 fails topic creation outright (loud, but blocks every developer), while
    shipping RF=1 to production produces a cluster that works perfectly until one
    broker dies and takes a partition's only copy with it.
    """
    resolved = settings or get_settings()
    return TopicSpec(
        role=role,
        name=topic_name(role, settings=resolved),
        partitions=DEFAULT_PARTITIONS[role],
        replication_factor=(
            PRODUCTION_REPLICATION_FACTOR
            if resolved.app.environment.is_production_like
            else SINGLE_NODE_REPLICATION_FACTOR
        ),
        retention=RETENTION[role],
        key_meaning=_KEY_MEANING[role],
    )


def all_topic_specs(*, settings: Settings | None = None) -> tuple[TopicSpec, ...]:
    """Every topic this deployment owns, for provisioning and for `make init-topics`.

    Derived from `TopicRole` rather than listed, so a role added tomorrow is
    provisioned without anyone remembering to edit a second place -- the failure
    mode of a hand-maintained list is that the new topic is auto-created by the
    broker with default partitions and infinite retention, which is precisely the
    configuration this module exists to prevent.
    """
    resolved = settings or get_settings()
    return tuple(topic_spec(role, settings=resolved) for role in TopicRole)


# --------------------------------------------------------------------------- #
# Partition keys
# --------------------------------------------------------------------------- #


def encode_key(key: str | None) -> bytes | None:
    """Encode a partition key for the wire.

    `None` passes through as `None`, which means round-robin placement and no
    ordering guarantee. That is only ever correct for an event about nothing in
    particular; everything in this system is about a Signal.
    """
    return None if key is None else key.encode("utf-8")


def partition_key_for_signal(signal: str) -> bytes:
    """Key a message by an existing `Signal.id`."""
    if not signal:
        raise ValueError("partition key is empty; a Signal id is required to preserve ordering")
    return signal.encode("utf-8")


def partition_key_for_item(platform: Platform | str, native_id: str) -> bytes:
    """Key a message by the Signal id an item *will* have, before one exists.

    The connector publishes a raw record before any Signal has been built, but the
    id is a pure function of `(platform, native_id)` (`docs/signal-model.md` §4.1),
    so it can be derived early. Deriving it -- rather than keying the raw topic by
    something else -- is what puts the raw record and the enriched Signal on the
    same key, and therefore keeps a re-fetch ordered behind its original.
    """
    return partition_key_for_signal(signal_id(platform, native_id))
