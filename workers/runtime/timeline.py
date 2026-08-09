"""Cross-process timeline transport: worker publishes, API replicas fan out.

`backend/api/v1/stream.py`'s `InProcessTimelineHub` is correct and complete for
one process. It is also the wrong thing in every real deployment, because the
worker that produces events and the API that serves the SSE stream are separate
processes -- often separate machines, and always more than one API replica.

The failure that produces is the reason this module exists, and it is quiet in
the worst way: the worker publishes into a hub in *its* process, that hub has no
subscribers, the API's hub stays empty, nothing errors, every health check passes
and the user watches a spinner forever.

**Redis pub/sub, not Streams.** A subscriber that is not connected misses the
message, which sounds like the wrong trade until you look at what the API already
does: `InProcessTimelineHub` keeps a bounded replay buffer per investigation and
the SSE contract has `stream.gap` for exactly this. Durability is not needed
because the *state* is durable -- the investigation row and the checkpoint are in
PostgreSQL, and a client that missed events refetches
`GET /investigations/{id}`. Streams would add consumer groups, trimming and
acknowledgement to guarantee delivery of something that is already recoverable.

**Every API replica receives every event.** Pub/sub fans out, which is what is
wanted: a browser is connected to one replica and the worker does not know which.
A queue would deliver each event to exactly one replica, so a client attached to
any other would see nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from backend.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from backend.api.v1.stream import TimelineEvent

__all__ = [
    "TIMELINE_CHANNEL_PREFIX",
    "RedisTimelinePublisher",
    "TimelineRelay",
    "channel_for",
    "decode_event",
    "encode_event",
]

logger = get_logger(__name__)

TIMELINE_CHANNEL_PREFIX: Final = "os:timeline:"
"""One channel per investigation, not one shared channel.

A single channel would make every API replica decode and discard events for
every concurrent investigation -- and with a hundred running, that is a hundred
times the deserialisation for the one stream a given replica actually serves.
Per-investigation channels mean a replica subscribes only to what it is watching.
"""

MAX_EVENT_BYTES: Final = 64 * 1024
"""Ceiling on one encoded event.

The timeline carries progress, not payloads: counts, node names, short messages.
An event approaching this size means something put a result set in `data`, which
would then be fanned out to every replica and every subscriber. Refusing is
better than making a slow stream out of an oversized frame.
"""


def channel_for(investigation_id: str) -> str:
    return f"{TIMELINE_CHANNEL_PREFIX}{investigation_id}"


def encode_event(event: TimelineEvent) -> str:
    """Serialise an event for the wire.

    `ts` becomes an ISO-8601 string with its offset intact. Dropping the offset
    -- which a naive `str(datetime)` on a naive value would -- makes every
    consumer guess the timezone, and `TimelineEvent` refuses naive timestamps on
    reconstruction, so the round trip would fail at the far end for a reason
    invisible at this one.
    """
    return json.dumps(
        {
            "investigation_id": event.investigation_id,
            "seq": event.seq,
            "type": event.type,
            "ts": event.ts.isoformat(),
            "data": dict(event.data),
            "request_id": event.request_id,
        },
        separators=(",", ":"),
        default=str,
    )


def decode_event(raw: str | bytes) -> TimelineEvent | None:
    """Rebuild an event, returning `None` for anything malformed.

    `None` rather than raising: a relay that died on one bad frame would stop
    delivering every *subsequent* event for that investigation, turning a single
    encoding bug into a stalled stream. Dropping the frame costs one progress
    update and leaves a gap the SSE layer already knows how to express.
    """
    from backend.api.v1.stream import TimelineEvent

    try:
        payload = json.loads(raw)
        timestamp = datetime.fromisoformat(str(payload["ts"]))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return TimelineEvent(
            investigation_id=str(payload["investigation_id"]),
            seq=int(payload["seq"]),
            type=str(payload["type"]),
            ts=timestamp,
            data=dict(payload.get("data") or {}),
            request_id=payload.get("request_id"),
        )
    except Exception as error:  # noqa: BLE001 -- see the docstring
        logger.warning("timeline.undecodable_event", error=type(error).__name__)
        return None


class RedisTimelinePublisher:
    """Publishes timeline events to the channel for their investigation."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def publish(self, event: TimelineEvent) -> bool:
        """Fan an event out. Returns whether it was sent.

        `False` rather than raising on an oversized event or an unreachable
        Redis. The caller (`workers/investigation_worker.py`) treats a failed
        publish as a lost progress frame and continues the run -- which is right:
        the stream is a view of the work, and losing a view must not lose the
        work.
        """
        encoded = encode_event(event)
        if len(encoded.encode("utf-8")) > MAX_EVENT_BYTES:
            logger.warning(
                "timeline.event_too_large",
                investigation_id=event.investigation_id,
                type=event.type,
                size=len(encoded),
                consequence="dropped; the timeline carries progress, not payloads",
            )
            return False
        try:
            await self._client.publish(channel_for(event.investigation_id), encoded)
        except Exception as error:  # noqa: BLE001 -- see the docstring
            logger.warning("timeline.publish_failed", error=type(error).__name__)
            return False
        return True


class TimelineRelay:
    """Subscribes to Redis on the API side and feeds the in-process hub.

    The complement of the publisher, and the reason the SSE endpoint needs no
    changes at all: it keeps reading from `InProcessTimelineHub`, which now
    receives events from wherever they were produced. The hub's replay buffer,
    backpressure policy and gap accounting all keep working, because from its
    point of view a relayed event is indistinguishable from a local one.

    Subscribed per investigation, on demand, rather than to everything: a replica
    serving three streams should not decode the events of three hundred runs.
    """

    def __init__(self, client: Any, hub: Any) -> None:
        self._client = client
        self._hub = hub
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def follow(self, investigation_id: str) -> None:
        """Start relaying one investigation, if not already.

        Idempotent. Two browsers watching the same investigation on the same
        replica must produce one subscription, not two -- otherwise every event
        is published into the hub twice and the hub rejects the duplicate `seq`,
        logging a warning per frame for a situation that is entirely normal.
        """
        if investigation_id in self._tasks:
            return
        self._tasks[investigation_id] = asyncio.create_task(self._pump(investigation_id))

    async def unfollow(self, investigation_id: str) -> None:
        task = self._tasks.pop(investigation_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _pump(self, investigation_id: str) -> None:
        """Read the channel and republish locally until cancelled."""
        channel = channel_for(investigation_id)
        pubsub = self._client.pubsub()
        try:
            await pubsub.subscribe(channel)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                event = decode_event(message["data"])
                if event is not None:
                    await self._hub.publish(event)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 -- one stream, not the process
            logger.warning(
                "timeline.relay_failed",
                investigation_id=investigation_id,
                error=type(error).__name__,
            )
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(channel)
                await pubsub.aclose()

    async def aclose(self) -> None:
        """Stop every relay. Called on API shutdown."""
        for investigation_id in list(self._tasks):
            await self.unfollow(investigation_id)
