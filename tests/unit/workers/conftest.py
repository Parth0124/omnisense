"""Fakes shared by the worker tests: an in-memory broker and its consumer.

Every test in this package runs with **no broker, no database container and no
network**, which `docs/testing-strategy.md` fixes as the unit suite's contract.
The interesting properties of a worker are all statements about *ordering* --
"the commit happened after the handler", "the offset did not advance", "the drain
finished the batch before closing the client" -- and ordering is the one thing a
return-value-only mock cannot express. So the fakes below record an ordered call
trace, and the assertions read off that trace.

`tests/unit/services/test_events.py` has a broker fake of its own. This is not
that one moved: that fake exists to exercise `EventConsumer` in isolation and
models the broker only as far as that runtime observes it. This one adds what
worker-level tests need and that one has no reason to carry -- a controllable
`getmany` that can block until released (so a drain has something in flight to
drain), a settable assignment, and a trace shared with the handler so
handler-versus-commit ordering is a single sequence rather than two.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from aiokafka.structs import ConsumerRecord, TopicPartition
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.config import Settings
from services.events.schemas import (
    EventDecodeError,
    EventEnvelope,
    EventPayload,
)

__all__ = [
    "FakeBroker",
    "FakeConsumer",
    "FakeKeywordStore",
    "FakeVectorStore",
    "envelope_bytes",
    "record",
]


def record(
    *,
    topic: str,
    partition: int = 0,
    offset: int = 0,
    value: bytes = b"{}",
    key: bytes | None = None,
    headers: tuple[tuple[str, bytes], ...] = (),
) -> ConsumerRecord[bytes, bytes]:
    """One `ConsumerRecord`. The real dataclass, never a stand-in.

    Using aiokafka's own type means a field it renames in a future release breaks
    these tests rather than letting them keep passing against a shape the runtime
    no longer receives.
    """
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


def envelope_bytes(payload: EventPayload, *, correlation_id: str = "corr-test") -> bytes:
    """Wire bytes for one payload, wrapped exactly as a producer would wrap it."""
    return EventEnvelope.wrap(
        payload, producer="omnisense-test", correlation_id=correlation_id
    ).to_bytes()


class FakeBroker:
    """An in-memory partitioned log.

    Models what the runtime actually observes: records live at monotonically
    increasing offsets within a partition, a consumer holds a position per
    partition, and a paused partition serves nothing. Everything else about Kafka
    is irrelevant here, and modelling it would only make the fake wrong in new
    ways.
    """

    def __init__(self) -> None:
        self.partitions: dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]] = {}
        self.dlq: list[tuple[EventEnvelope, bytes | None]] = []

    def append(
        self, topic: str, value: bytes, *, partition: int = 0, key: bytes | None = None
    ) -> TopicPartition:
        tp = TopicPartition(topic, partition)
        log = self.partitions.setdefault(tp, [])
        headers: tuple[tuple[str, bytes], ...] = ()
        # An unparseable body still carries headers on a real broker. Leaving
        # them empty when the body cannot be parsed is the harsher case, and the
        # one worth exercising.
        try:
            headers = tuple(EventEnvelope.from_bytes(value).to_headers())
        except EventDecodeError:
            headers = ()
        log.append(
            record(topic=topic, partition=partition, offset=len(log), value=value, key=key,
                   headers=headers)
        )
        return tp

    async def dlq_publisher(self, envelope: EventEnvelope, *, key: bytes | None) -> None:
        """A `DlqPublisher` that parks into a list instead of onto a topic."""
        self.dlq.append((envelope, key))


class FakeConsumer:
    """`KafkaConsumerLike` over a `FakeBroker`, recording an ordered call trace.

    `gate` is what makes drain tests possible: when set, `getmany` waits on it
    before returning, so a test can hold a batch in flight, request a shutdown,
    and then release it -- which is precisely the sequence a `SIGTERM` arriving
    mid-batch produces in production.
    """

    def __init__(self, broker: FakeBroker, trace: list[str] | None = None) -> None:
        self.broker = broker
        self.trace: list[str] = trace if trace is not None else []
        self.positions: dict[TopicPartition, int] = dict.fromkeys(broker.partitions, 0)
        self.committed: dict[TopicPartition, int] = {}
        self.paused: set[TopicPartition] = set()
        self.seeks: list[tuple[TopicPartition, int]] = []
        self.started = False
        self.stopped = False
        self.commit_error: Exception | None = None
        self.getmany_error: Exception | None = None
        self.gate: asyncio.Event | None = None
        self.polls = 0

    async def start(self) -> None:
        self.trace.append("start")
        self.started = True

    async def stop(self) -> None:
        self.trace.append("stop")
        self.stopped = True

    async def getmany(
        self,
        *partitions: TopicPartition,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]]:
        self.polls += 1
        if self.getmany_error is not None:
            error, self.getmany_error = self.getmany_error, None
            raise error
        if self.gate is not None:
            await self.gate.wait()

        batch: dict[TopicPartition, list[ConsumerRecord[bytes, bytes]]] = {}
        budget = max_records if max_records is not None else 10_000
        for tp, log in self.broker.partitions.items():
            if tp in self.paused or budget <= 0:
                continue
            start = self.positions.get(tp, 0)
            window = log[start : start + budget]
            if not window:
                continue
            batch[tp] = window
            # A real consumer's position advances on *fetch*, not on commit --
            # which is exactly why the runtime has to `seek` to get a redelivery.
            self.positions[tp] = start + len(window)
            budget -= len(window)
        if batch:
            self.trace.append("fetch")
        else:
            # An empty poll would otherwise spin the loop at full speed and starve
            # every other task in the test's event loop.
            await asyncio.sleep(0)
        return batch

    async def commit(self, offsets: dict[TopicPartition, int] | None = None) -> None:
        self.trace.append("commit")
        if self.commit_error is not None:
            raise self.commit_error
        self.committed.update(offsets or {})

    def pause(self, *partitions: TopicPartition) -> None:
        self.paused.update(partitions)

    def resume(self, *partitions: TopicPartition) -> None:
        self.paused.difference_update(partitions)

    def seek(self, partition: TopicPartition, offset: int) -> None:
        self.seeks.append((partition, offset))
        self.positions[partition] = offset

    def assignment(self) -> set[TopicPartition]:
        return set(self.broker.partitions)


@pytest.fixture
def settings() -> Settings:
    """Settings built from defaults, ignoring any developer `.env` on disk.

    Constructed rather than taken from `get_settings()` because that returns a
    process-wide singleton whose values depend on whatever `.env` the machine
    running the suite happens to have. A worker test that asserted on a topic
    name would then pass or fail depending on the developer's local file.
    """
    return Settings()


@pytest.fixture
def broker() -> FakeBroker:
    return FakeBroker()


def drain_trace(trace: Sequence[str], *marks: str) -> list[int]:
    """Positions of `marks` within a call trace, for ordering assertions.

    Returns indices rather than booleans so a failure message shows *where* each
    event landed -- "commit at 1, handle at 3" says what went wrong; "ordering
    assertion failed" does not.
    """
    return [trace.index(mark) for mark in marks]


def as_any(value: object) -> Any:
    """Escape hatch for handing a fake where a protocol is annotated.

    The fakes here satisfy the protocols structurally but are not registered as
    implementations of them, and `mypy --strict` is run over this repository.
    """
    return value


# --------------------------------------------------------------------------- #
# Derived-store fakes
# --------------------------------------------------------------------------- #
#
# These model the *identity* semantics of each store and nothing else, because
# identity is the only property the worker tests assert on: "the same message
# twice yields one point" is a statement about the id, not about HNSW. They are
# deliberately keyed dicts rather than append-only lists -- a list-backed fake
# makes a worker that duplicates on redelivery pass every assertion, which is
# precisely the bug the idempotency tests exist to catch.


class FakeVectorStore:
    """`retrieval.vector.qdrant_client.VectorStore`, keyed by point id.

    `upsert` overwrites by id the way Qdrant does, and `delete` evaluates the
    filter selector it is handed against stored payloads rather than assuming it
    means "everything for this signal". A fake that deleted unconditionally would
    let a worker pass while sending a filter that matches the whole collection.
    """

    def __init__(self) -> None:
        self.points: dict[str, Any] = {}
        self.upsert_calls: list[dict[str, Any]] = []
        self.delete_calls: list[Any] = []
        self.upsert_error: Exception | None = None

    async def upsert(
        self, collection_name: str, points: Sequence[Any], *, wait: bool = False, **_: Any
    ) -> None:
        self.upsert_calls.append({"collection": collection_name, "count": len(points)})
        if self.upsert_error is not None:
            raise self.upsert_error
        for point in points:
            self.points[str(point.id)] = point

    async def delete(
        self, collection_name: str, points_selector: Any, *, wait: bool = False, **_: Any
    ) -> None:
        self.delete_calls.append(points_selector)
        wanted = _selector_signal_ids(points_selector)
        if wanted is None:
            return
        for point_id, point in list(self.points.items()):
            if (point.payload or {}).get("signal_id") in wanted:
                self.points.pop(point_id, None)

    async def query_points(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("the worker tests never search; this call is a bug")

    async def get_collection(self, collection_name: str) -> Any:  # pragma: no cover
        raise AssertionError("the worker never inspects collection geometry")

    async def create_payload_index(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("payload indexes are provisioned by scripts/, not by a worker")

    def signal_ids(self) -> set[str]:
        """Which Signals currently have at least one point."""
        return {str((point.payload or {}).get("signal_id")) for point in self.points.values()}


def _selector_signal_ids(selector: Any) -> set[str] | None:
    """Pull the `signal_id` values out of a Qdrant filter selector.

    Returns `None` for a selector this fake does not model, which is reported as
    "deleted nothing" rather than as "deleted everything" -- a fake that guessed
    the other way would make an over-broad delete invisible.
    """
    query_filter = getattr(selector, "filter", None)
    if query_filter is None:
        return None
    wanted: set[str] = set()
    for condition in getattr(query_filter, "must", None) or ():
        if getattr(condition, "key", None) != "signal_id":
            continue
        match = getattr(condition, "match", None)
        value = getattr(match, "value", None)
        if value is not None:
            wanted.add(str(value))
        for any_value in getattr(match, "any", None) or ():
            wanted.add(str(any_value))
    return wanted or None


class FakeKeywordStore:
    """`retrieval.keyword.opensearch_client.KeywordStore`, keyed by `_id`.

    Honours the external-version guard, because the worker's idempotency claim
    depends on it: a redelivery re-sends the same `pipeline_version`, which under
    `external_gte` is accepted as an overwrite and under `external` would be a
    409. Getting that wrong in the fake would hide the difference.
    """

    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, int] = {}
        self.bulk_calls: list[list[dict[str, Any]]] = []
        self.delete_by_query_calls: list[dict[str, Any]] = []
        self.bulk_error: Exception | None = None

    async def bulk(self, **kwargs: Any) -> dict[str, Any]:
        body = list(kwargs["body"])
        self.bulk_calls.append(body)
        if self.bulk_error is not None:
            raise self.bulk_error

        items: list[dict[str, Any]] = []
        position = 0
        while position < len(body):
            action = body[position]
            operation, meta = next(iter(action.items()))
            chunk_id = meta["_id"]
            if operation == "delete":
                position += 1
                existed = self.documents.pop(chunk_id, None) is not None
                self.versions.pop(chunk_id, None)
                items.append({"delete": {"_id": chunk_id, "status": 200 if existed else 404}})
                continue
            source = body[position + 1]
            position += 2
            items.append(self._index(chunk_id, meta, source))
        return {"took": 1, "errors": any(_status(item) >= 300 for item in items), "items": items}

    def _index(
        self, chunk_id: str, meta: Mapping[str, Any], source: Mapping[str, Any]
    ) -> dict[str, Any]:
        version = int(meta["version"])
        stored = self.versions.get(chunk_id)
        if stored is not None:
            newer = version > stored if meta["version_type"] == "external" else version >= stored
            if not newer:
                return {
                    "index": {
                        "_id": chunk_id,
                        "status": 409,
                        "error": {"type": "version_conflict_engine_exception", "reason": ""},
                    }
                }
        self.documents[chunk_id] = dict(source)
        self.versions[chunk_id] = version
        return {"index": {"_id": chunk_id, "status": 201 if stored is None else 200}}

    async def delete_by_query(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_by_query_calls.append(kwargs)
        term = kwargs["body"]["query"]["term"]
        field, value = next(iter(term.items()))
        doomed = [
            chunk_id
            for chunk_id, source in self.documents.items()
            if source.get(field) == value
        ]
        for chunk_id in doomed:
            self.documents.pop(chunk_id, None)
            self.versions.pop(chunk_id, None)
        return {"deleted": len(doomed), "version_conflicts": 0, "failures": []}

    async def search(self, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("the worker tests never search; this call is a bug")

    def signal_ids(self) -> set[str]:
        return {str(source.get("signal_id")) for source in self.documents.values()}


def _status(item: Mapping[str, Any]) -> int:
    for value in item.values():
        if isinstance(value, Mapping):
            return int(value.get("status", 500))
    return 500


@pytest.fixture
def session_factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """A factory over the shared in-memory database, configured as production is.

    A *factory*, not a session: every worker in `workers/` takes one, because a
    worker opens a session per unit of work rather than holding one for the life
    of the process -- a long-lived session pins a pooled connection and turns a
    slow handler into pool exhaustion for everything else in the process.
    """
    return async_sessionmaker(
        bind=orm_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
