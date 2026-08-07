"""`GET /api/v1/investigations/{id}/stream` -- the execution timeline as SSE.

An investigation runs for minutes to half an hour (`docs/architecture.md` §3).
The `202` from `POST /investigations` is therefore a promise, and this endpoint is
where the promise is kept: it is the live view of the agent graph executing, the
one surface where a user can see *why* an answer is taking thirty minutes rather
than watching a spinner and reloading (`docs/frontend.md` §4.1).

SSE rather than WebSockets because the traffic is one-way. A socket that only
ever sends server-to-client buys a second protocol, a second set of proxy
problems and a manual heartbeat, in exchange for a capability -- client-to-server
frames -- the contract explicitly does not have (`docs/api-reference.md` §5,
"Because the stream is one-way, cancellation is not expressible over it").

Five properties decide whether this survives contact with production. Each is
implemented below and each is named here because the code that implements it is
easy to mistake for boilerplate and delete.

**1. Heartbeat.** An SSE connection carrying no bytes is indistinguishable, to
everything between the browser and this process, from a connection that has hung.
nginx's `proxy_read_timeout` is 60s, an AWS ALB idles out at 60s, Cloudflare at
100s, and a corporate forward proxy is frequently 30s. A planning step that
thinks for 90 seconds would therefore be killed mid-investigation and the user
would see a reconnect storm rather than a plan. `HEARTBEAT_INTERVAL_SECONDS` is
15s -- half of the *shortest* of those timeouts, so a single dropped heartbeat
still does not cross the threshold. The frame is an SSE comment (`: ping`), which
is legal in the grammar, ignored by every conforming client, and invisible to the
`EventSource` `onmessage` handler.

**2. Client disconnect.** A browser tab closing must stop the producer.
`EventSourceResponse` watches the ASGI receive channel for `http.disconnect` and
cancels the task group, which throws into `_timeline_frames` at its `await` and
runs its `finally`. That `finally` -- plus `client_close_handler_callable`, which
fires earlier and does not depend on cancellation semantics -- unregisters the
subscription. Without it, a page-refresh loop leaks one bounded queue per refresh
and pins them all for the lifetime of the investigation. This handler deliberately
holds **no database session**: everything it needs comes from the in-memory
timeline source, so an abandoned stream cannot hold a pooled connection either.

**3. Backpressure.** One slow consumer -- a laptop that went to sleep with the tab
open, a phone on a train -- must not be able to grow memory in the API process.
Each subscriber gets a bounded queue (`SUBSCRIBER_QUEUE_EVENTS`). When it is full
the **oldest undelivered event is dropped**, not the newest, and not the producer
blocked:

- Blocking the producer would let the slowest reader in the world set the pace of
  the orchestrator, and the orchestrator is shared by every other subscriber.
- Dropping the *newest* would eventually drop `done`, and a client that never
  receives a terminal event waits forever on a run that finished. Terminal events
  are by definition the newest, so drop-oldest can never lose one.
- Coalescing was rejected: the six event types in `docs/api-reference.md` §5 are
  already minimal (ids and a snippet), and the only type that arrives in bulk,
  `evidence.found`, carries the evidence id the UI needs to hydrate. Merging ten
  of those into a count would discard exactly the payload that made them worth
  sending.

A drop is *reported*, never silent: the affected range is surfaced as a
`stream.gap` event (see below) and is separately visible as a jump in `seq`, which
§5 already defines as meaning "dropped, never reordered".

**4. Resumption.** `Last-Event-ID` is honoured: the value is the `seq` of the last
event the client rendered, and replay resumes at `seq + 1`. Two deliberate
choices around it:

- An **unparseable** `Last-Event-ID` replays the full retained history instead of
  returning `400`. A browser's `EventSource` reconnects automatically with the
  same id, so a `400` becomes a hot reconnect loop that the client cannot escape.
  Duplicates are safe -- every event carries `seq`, so a client can discard what
  it has already seen -- while gaps are not, and this endpoint prefers duplicates
  to gaps every time.
- A resume point **older than the retained buffer** cannot be honoured. Rather
  than silently starting late, the connection opens with a `stream.gap` naming the
  exact range that will never arrive, so the UI can fall back to
  `GET /api/v1/investigations/{id}` (`docs/frontend.md` §3, "polling is the
  fallback, used on reconnect exhaustion and on late attach").

**5. Encoding.** Every event payload originates in third-party content: a Reddit
comment quoted in `evidence.found`, an exception message from a vendor API in
`error`, a tool argument copied out of a fetched page. SSE frames are newline
delimited, so a raw `\\n` in a snippet ends the frame early and the remainder of
that snippet is parsed as the *next* frame -- and a snippet containing
`data: {"seq": 1}` then injects a forged event into a stream the client trusts.
Nothing here is ever interpolated into a frame: `encode_event_data` runs the whole
payload through `json.dumps`, which escapes every C0 control character including
CR and LF, and the `event:` and `id:` lines are built from an enum and an integer
that cannot contain a newline in the first place.

What this module does **not** yet provide
-----------------------------------------
`docs/architecture.md` §3.1 row 6 requires that events be "derived from checkpoint
deltas, not from in-process callbacks", so that any replica can serve any run.
`InProcessTimelineHub` does not satisfy that: it is memory in one process, so a
client load-balanced onto a replica that is not running the orchestration sees an
idle stream, and a restart loses the replay buffer.

Closing that gap needs a durable, ordered, per-investigation event log, and none
exists yet. `models/orm/` has no `investigation_events` table (`investigation_steps`
records steps but has no row for a tool call or an admitted piece of evidence, so
two of the six event types cannot be reconstructed from it), and
`backend/db/redis.py` exposes no Redis Stream helpers. `TimelineSource` is the
seam for that work: implement `subscribe`/`publish` against `XADD`/`XRANGE` or
against a new table, register it in place of the hub, and nothing in this file's
framing, heartbeat, disconnect or backpressure logic changes.

Layer note: `backend/api/` (L4). It imports `backend/core/` and nothing from
`services/` or `agents/`; producers reach *in* via `get_timeline_hub().publish()`
rather than this module reaching out, which is what keeps the orchestrator free of
an HTTP-layer import.
"""

from __future__ import annotations

import asyncio
import enum
import json
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Any, Final, Protocol

from fastapi import APIRouter, Depends, Header, Query, Request
from pydantic import BaseModel
from sse_starlette import EventSourceResponse, ServerSentEvent

from backend.core.exceptions import RateLimitedError
from backend.core.logging import UNBOUND_CORRELATION_ID, get_correlation_id, get_logger

__all__ = [
    "HEARTBEAT_INTERVAL_SECONDS",
    "MAX_SUBSCRIBERS_PER_INVESTIGATION",
    "REPLAY_BUFFER_EVENTS",
    "SUBSCRIBER_QUEUE_EVENTS",
    "TERMINAL_EVENT_TYPES",
    "InProcessTimelineHub",
    "SeqGap",
    "StreamAccess",
    "TimelineEvent",
    "TimelineEventType",
    "TimelineSource",
    "TimelineSubscription",
    "encode_event_data",
    "get_heartbeat_interval",
    "get_timeline_hub",
    "require_stream_access",
    "router",
]

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Tunables
#
# Module constants rather than settings because `backend/core/config.py` has no
# stream section and this module may not read the environment itself. When one is
# added, these become its defaults; the names are chosen to survive the move.
# --------------------------------------------------------------------------- #

HEARTBEAT_INTERVAL_SECONDS: Final[float] = 15.0
"""Seconds between `: ping` comment frames on an otherwise silent stream.

Half of the shortest idle timeout we expect to sit behind (nginx 60s, ALB 60s,
corporate forward proxies commonly 30s), so losing one heartbeat still does not
cross the threshold. `docs/api-reference.md` §5 fixes the value at 15s.
"""

SUBSCRIBER_QUEUE_EVENTS: Final[int] = 256
"""Undelivered events held per connection before the oldest is dropped.

Sized to absorb a burst -- a Retriever fan-out admitting a few hundred pieces of
evidence in a second -- without letting a sleeping laptop pin megabytes. The
ceiling is what makes the memory cost of a connection knowable: this many events,
each a few hundred bytes.
"""

REPLAY_BUFFER_EVENTS: Final[int] = 2048
"""Events retained per investigation for late attach and `Last-Event-ID` replay.

Bounded because it is process memory. `docs/api-reference.md` §5 promises the
investigation's lifetime plus an hour, which only a durable source can honour;
see the module docstring. When the ring has wrapped past a client's resume point,
the connection opens with a `stream.gap` rather than pretending it was complete.
"""

HISTORY_RETENTION_SECONDS: Final[float] = 3600.0
"""How long a subscriber-less channel is kept after its last event.

Matches the "lifetime plus one hour" of the contract, so a user who reloads the
report page an hour later still gets the timeline. Enforced by a lazy sweep on
publish rather than a background task, because a background task would need to be
started and stopped in `backend/main.py`'s lifespan and would keep the whole
history graph alive in a process that had no subscribers at all.
"""

PRUNE_INTERVAL_SECONDS: Final[float] = 60.0
"""Minimum spacing between retention sweeps, so `publish` stays O(1) amortized."""

MAX_SUBSCRIBERS_PER_INVESTIGATION: Final[int] = 32
"""Concurrent connections accepted for one investigation.

`docs/api-reference.md` §3.6 caps concurrent streams *per principal*, which needs
the authenticated identity and therefore belongs in `backend/core/ratelimit.py`
(still a stub). This is the cruder, complementary guard that protects *memory*:
without it a reconnect loop with a broken backoff allocates one queue per attempt
and the process dies of a client-side bug.
"""

SEND_TIMEOUT_SECONDS: Final[float] = 30.0
"""Deadline for writing one frame to one client.

A client that stops reading -- TCP window at zero, never closing the socket --
would otherwise block the response task forever, holding its subscription and its
place in every fan-out. Two heartbeat intervals is long enough that a merely slow
network is not mistaken for a dead one. The client can reconnect with
`Last-Event-ID` and lose nothing that is still in the ring.
"""

RECONNECT_ADVICE_MS: Final[int] = 3000
"""`retry:` hint sent once at open, in milliseconds.

Stated explicitly rather than left to the client's default so the backoff is a
server-side decision: raising it during an incident is how a thundering herd of
reconnects is slowed down without shipping a frontend build.
"""

MAX_REQUEST_ID_CHARS: Final[int] = 128
"""Ceiling on a client-supplied `X-Request-ID` before it is echoed into frames.

The value is JSON-encoded so it cannot break framing, but it is copied into every
event on the connection; unbounded, a 10 MB header becomes a 10 MB-per-event
stream. Truncating keeps the correlation useful and the amplification at zero.
"""


# --------------------------------------------------------------------------- #
# Event vocabulary
# --------------------------------------------------------------------------- #


class TimelineEventType(enum.StrEnum):
    """The six event names of `docs/api-reference.md` §5, plus one extension.

    A plain `StrEnum` rather than the repository's `TolerantStrEnum`: this
    vocabulary is *produced* here, never parsed from an external payload, so
    there is no unknown member to tolerate. Producers on a newer deployment may
    publish a name this process does not know -- `TimelineEvent.type` is typed
    `str` for exactly that reason, and such an event is forwarded verbatim, which
    is what makes "adding a new SSE event type" the additive change §1 says it is.
    """

    STEP_STARTED = "step.started"
    TOOL_CALLED = "tool.called"
    EVIDENCE_FOUND = "evidence.found"
    STEP_COMPLETED = "step.completed"
    ERROR = "error"
    DONE = "done"

    STREAM_GAP = "stream.gap"
    """OmniSense extension: events in a stated `seq` range will never arrive here.

    Emitted when a resume point has fallen out of the replay buffer, when
    backpressure dropped events for this connection, or when an event could not be
    encoded. It is not part of the investigation's timeline, so it is sent
    **without an `id:` line** -- a client's `Last-Event-ID` must not advance past
    events it never received, or the gap would be permanent on the next reconnect.

    Additive per §1 ("clients must ignore unknown event names"): a client that
    drops it is exactly as informed as before, since the gap is also visible as a
    jump in `seq`.
    """


TERMINAL_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {TimelineEventType.ERROR, TimelineEventType.DONE}
)
"""After one of these the server closes and the client must not reconnect (§5)."""


RESERVED_PAYLOAD_KEYS: Final[frozenset[str]] = frozenset(
    {"investigation_id", "seq", "ts", "request_id"}
)
"""Envelope keys a producer may not supply in `data`.

Shadowing `seq` would let an agent's output rewrite the ordering the client
depends on. Rejected at construction, in the producing process, rather than
silently overwritten here -- a producer that thinks it is setting `seq` has a bug
worth failing loudly.
"""


@dataclass(frozen=True, slots=True)
class TimelineEvent:
    """One event on one investigation's timeline.

    Frozen because it is fanned out to every subscriber of the investigation: a
    mutable event would let one connection's rendering path mutate what the others
    are about to render.

    `seq` is assigned by the **producer**, not by this process. The orchestrator is
    the only party that sees every event of a run in order, and a sequence number
    minted here would restart at zero on every API replica and every deploy.
    """

    investigation_id: str
    seq: int
    type: str
    ts: datetime
    data: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise ValueError(f"seq must be non-negative, got {self.seq}")
        if not self.investigation_id:
            raise ValueError("investigation_id is required")
        if not self.type or any(ch in str(self.type) for ch in "\r\n"):
            # An event name with a newline in it would terminate the `event:`
            # line and turn the remainder into a forged field. sse-starlette
            # strips them defensively; refusing here means a producer finds out
            # at publish time instead of shipping a stream nobody can parse.
            raise ValueError(f"invalid event type: {self.type!r}")
        if self.ts.tzinfo is None or self.ts.tzinfo.utcoffset(self.ts) is None:
            raise ValueError("ts must be timezone-aware (docs/api-reference.md §3.2)")
        shadowed = RESERVED_PAYLOAD_KEYS & set(self.data)
        if shadowed:
            raise ValueError(f"data may not shadow envelope keys: {sorted(shadowed)}")

    @property
    def is_terminal(self) -> bool:
        return self.type in TERMINAL_EVENT_TYPES

    def payload(self, *, request_id: str | None = None) -> dict[str, Any]:
        """The `data` object for the wire: shared envelope plus event fields.

        `request_id` falls back to the connection's correlation id when the
        producer did not carry one. Every frame therefore names *something* an
        operator can grep: with a producer id it points at the orchestrator's logs
        for that run, and without one it points at this connection, which is the
        difference between "the stream stopped" being diagnosable and not.
        """
        return {
            "investigation_id": self.investigation_id,
            "seq": self.seq,
            "ts": _iso_utc(self.ts),
            "request_id": self.request_id or request_id,
            **dict(self.data),
        }


@dataclass(frozen=True, slots=True)
class SeqGap:
    """A `seq` range this connection will not deliver, and why."""

    from_seq: int
    to_seq: int
    reason: str

    @property
    def missing(self) -> int:
        return max(0, self.to_seq - self.from_seq + 1)


# --------------------------------------------------------------------------- #
# Encoding -- the only place a payload becomes bytes
# --------------------------------------------------------------------------- #


def _iso_utc(moment: datetime) -> str:
    """RFC 3339 in UTC with a `Z` offset, matching every example in §5."""
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    """Coerce the few non-JSON types an agent payload legitimately contains.

    Deliberately narrow. A permissive `default=str` would serialise *anything*,
    including an ORM row or an exception object, and quietly publish whatever its
    `__str__` happened to contain -- a connection string, a row of user data -- to
    a browser. Everything not named here raises `TypeError`, which the frame
    renderer turns into a reported gap rather than a leak.
    """
    if isinstance(value, datetime):
        return _iso_utc(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        # Sorted so the same evidence set does not produce a different frame on
        # every run, which would make stream fixtures unreproducible.
        return sorted(value, key=repr)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"{type(value).__name__} is not serialisable into a timeline event")


def encode_event_data(event: TimelineEvent, *, request_id: str | None = None) -> str:
    """Render one event's payload as a single-line JSON string.

    The single line is the correctness property, not a formatting preference.
    `json.dumps` escapes every C0 control character -- `\\n`, `\\r`, `\\u0000` --
    so no value drawn from ingested content can terminate the `data:` line early.
    A Reddit comment reading `nice\\ndata: {"seq":9999}` therefore arrives as one
    string in one field instead of as a forged second event.

    `allow_nan=False` is the other half. Python's `json` emits bare `NaN` and
    `Infinity` by default, which are not JSON; a browser's `JSON.parse` throws on
    them, and the client cannot tell a poisoned frame from a truncated one. A NaN
    confidence score from a broken model raises here instead, and is reported as a
    gap.

    `ensure_ascii=False` keeps non-Latin content readable and roughly six times
    smaller on the wire. It is safe: SSE frames are UTF-8 and are split only on CR
    and LF, so U+2028/U+2029 -- which do break naive JavaScript string literals --
    are inert inside a JSON string that the client parses rather than evaluates.
    """
    return json.dumps(
        event.payload(request_id=request_id),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        default=_json_default,
    )


def _event_frame(event: TimelineEvent, *, request_id: str | None) -> ServerSentEvent:
    """Frame one timeline event. `id:` is the `seq`, which is what resumes it."""
    return ServerSentEvent(
        data=encode_event_data(event, request_id=request_id),
        event=str(event.type),
        id=str(event.seq),
    )


def _gap_frame(*, investigation_id: str, request_id: str, gap: SeqGap) -> ServerSentEvent:
    """Frame a `stream.gap`. Carries no `id:` -- see `TimelineEventType.STREAM_GAP`."""
    body = {
        "investigation_id": investigation_id,
        "ts": _iso_utc(datetime.now(UTC)),
        "request_id": request_id,
        "from_seq": gap.from_seq,
        "to_seq": gap.to_seq,
        "missing": gap.missing,
        "reason": gap.reason,
    }
    return ServerSentEvent(
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
        event=str(TimelineEventType.STREAM_GAP),
    )


def _internal_error_frame(*, investigation_id: str, request_id: str, seq: int) -> ServerSentEvent:
    """Terminal `error` frame synthesised when a real terminal event would not encode.

    Says nothing about the cause. Whatever made the original payload unencodable
    is, by definition, something we did not anticipate, and its repr may contain
    anything at all. `request_id` is the handle for the log line that does have
    the detail (`backend/api/errors.py` uses the same rule).
    """
    body = {
        "investigation_id": investigation_id,
        "seq": seq,
        "ts": _iso_utc(datetime.now(UTC)),
        "request_id": request_id,
        "code": "internal_error",
        "message": "The investigation ended but its final event could not be encoded.",
        "retryable": False,
    }
    return ServerSentEvent(
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
        event=str(TimelineEventType.ERROR),
        id=str(seq),
    )


# --------------------------------------------------------------------------- #
# Transport: the source a connection subscribes to
# --------------------------------------------------------------------------- #


class TimelineSubscription(Protocol):
    """One connection's view of one investigation's timeline."""

    investigation_id: str
    replay: tuple[TimelineEvent, ...]
    """Retained events at or after the resume point, snapshotted at subscribe."""

    resume_gap: SeqGap | None
    """Set when the requested `Last-Event-ID` predates what is still retained."""

    async def get(self) -> TimelineEvent:
        """Await the next live event. Blocks indefinitely; cancellation is normal."""
        ...

    def take_gap(self) -> SeqGap | None:
        """Return and clear any range dropped from this connection's queue."""
        ...

    def close(self) -> None:
        """Unregister. Idempotent -- both the disconnect hook and the generator's
        `finally` call it, and which arrives first depends on the server."""
        ...


class TimelineSource(Protocol):
    """Where timeline events come from. Implemented by `InProcessTimelineHub`.

    Both methods are `async` even though the in-process implementation awaits
    nothing, because the implementation that matters in production will not: a
    Redis Streams or table-backed source has to do I/O, and a synchronous
    signature here would force every call site to change when it lands.
    """

    async def subscribe(
        self, investigation_id: str, *, after_seq: int | None = None
    ) -> TimelineSubscription: ...

    async def publish(self, event: TimelineEvent) -> bool: ...


class _Subscription:
    """A bounded mailbox for one connection.

    Not an `asyncio.Queue` subclass: the overflow policy is the whole point of the
    class, and inheriting `put` would leave a caller one attribute access away
    from the unbounded default.
    """

    __slots__ = (
        "_closed",
        "_gap_from",
        "_gap_to",
        "_hub",
        "_queue",
        "dropped",
        "investigation_id",
        "replay",
        "resume_gap",
    )

    def __init__(
        self,
        hub: InProcessTimelineHub,
        investigation_id: str,
        *,
        maxsize: int,
        replay: tuple[TimelineEvent, ...],
        resume_gap: SeqGap | None,
    ) -> None:
        self._hub = hub
        self._queue: asyncio.Queue[TimelineEvent] = asyncio.Queue(maxsize=maxsize)
        self._closed = False
        self._gap_from: int | None = None
        self._gap_to: int | None = None
        self.dropped = 0
        self.investigation_id = investigation_id
        self.replay = replay
        self.resume_gap = resume_gap

    # -- producer side ----------------------------------------------------- #

    def offer(self, event: TimelineEvent) -> None:
        """Enqueue for this connection, evicting the oldest event if full.

        Never blocks and never raises: it runs inside `publish`, on the
        orchestrator's task. A `put` that awaited here would make the slowest
        subscriber the pacemaker for the investigation itself, and an exception
        would abort a fan-out part-way, delivering the event to some subscribers
        and not others.
        """
        if self._closed:
            return
        try:
            self._queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        # Full: evict the head. There is no `await` between the get and the put,
        # so no other task can observe the queue below capacity, and the newest
        # event -- which may be `done` -- is always the one that survives.
        try:
            evicted = self._queue.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover -- single-threaded, cannot happen
            evicted = None
        if evicted is not None:
            self._record_drop(evicted.seq)
        self._queue.put_nowait(event)

    def _record_drop(self, seq: int) -> None:
        self.dropped += 1
        self._gap_from = seq if self._gap_from is None else min(self._gap_from, seq)
        self._gap_to = seq if self._gap_to is None else max(self._gap_to, seq)

    # -- consumer side ----------------------------------------------------- #

    async def get(self) -> TimelineEvent:
        return await self._queue.get()

    def take_gap(self) -> SeqGap | None:
        if self._gap_from is None or self._gap_to is None:
            return None
        gap = SeqGap(self._gap_from, self._gap_to, reason="slow_consumer")
        self._gap_from = None
        self._gap_to = None
        return gap

    @property
    def pending(self) -> int:
        """Undelivered events. Used by tests and by the disconnect log line."""
        return self._queue.qsize()

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub.unsubscribe(self)


@dataclass(slots=True)
class _Channel:
    """Retained history and live subscribers for one investigation."""

    history: deque[TimelineEvent]
    subscribers: set[_Subscription] = field(default_factory=set)
    last_seq: int = -1
    last_activity: float = field(default_factory=time.monotonic)
    terminal: bool = False


class InProcessTimelineHub:
    """Fan-out of investigation events to the connections in *this* process.

    Correct only while the orchestrator and the connection share a process. See
    the module docstring for what that costs and what replaces it.

    Everything here is single-threaded by construction: it runs on the event loop
    and no method awaits between reading and mutating its state, which is why a
    `put_nowait`/`get_nowait` pair can be treated as atomic and why `subscribe`
    can snapshot history and register a queue with no window in between. A future
    implementation that does await must take a lock; this one must not, because a
    lock would let a subscriber's cancellation interleave a fan-out.
    """

    def __init__(
        self,
        *,
        queue_events: int = SUBSCRIBER_QUEUE_EVENTS,
        replay_events: int = REPLAY_BUFFER_EVENTS,
        retention_seconds: float = HISTORY_RETENTION_SECONDS,
        max_subscribers: int = MAX_SUBSCRIBERS_PER_INVESTIGATION,
    ) -> None:
        if queue_events < 1 or replay_events < 1:
            raise ValueError("queue_events and replay_events must be positive")
        self._channels: dict[str, _Channel] = {}
        self._queue_events = queue_events
        self._replay_events = replay_events
        self._retention_seconds = retention_seconds
        self._max_subscribers = max_subscribers
        self._last_prune = time.monotonic()

    # -- producers --------------------------------------------------------- #

    async def publish(self, event: TimelineEvent) -> bool:
        """Record an event and hand it to every current subscriber.

        Returns `False` for an event that does not advance the sequence. A
        re-delivered message (Kafka at-least-once, a worker retry) or a producer
        that restarted its counter would otherwise reorder a client's timeline,
        and §5 promises that gaps mean loss but never that order can go backwards.
        Rejecting is the conservative half of that promise; the loud log line is
        how the other half gets noticed.
        """
        channel = self._channel(event.investigation_id)
        if event.seq <= channel.last_seq:
            logger.warning(
                "stream.event_out_of_order",
                investigation_id=event.investigation_id,
                event_type=str(event.type),
                seq=event.seq,
                last_seq=channel.last_seq,
            )
            return False

        channel.history.append(event)
        channel.last_seq = event.seq
        channel.last_activity = time.monotonic()
        channel.terminal = channel.terminal or event.is_terminal

        for subscription in channel.subscribers:
            subscription.offer(event)

        self._prune()
        return True

    # -- consumers --------------------------------------------------------- #

    async def subscribe(
        self, investigation_id: str, *, after_seq: int | None = None
    ) -> TimelineSubscription:
        """Register a connection and snapshot the history it should replay.

        `after_seq=None` means "no resume point", and replays the **whole**
        retained history rather than starting live. That is the late-attach
        contract of §5 -- a client that connects after completion gets the full
        timeline, then `done`, then a close -- and it is also what lets the UI
        render a run it joined halfway through without a second round trip.

        Registration and snapshot happen with no `await` between them, so an event
        published concurrently is either in the snapshot or in the queue, never in
        both and never in neither. The consumer still de-duplicates on `seq`,
        because a future source doing real I/O here cannot make that guarantee.
        """
        channel = self._channel(investigation_id)
        if len(channel.subscribers) >= self._max_subscribers:
            # 429 rather than a silent queue: the caller is looping, and a
            # reconnect storm is only survivable if the server says stop.
            raise RateLimitedError(
                "Too many concurrent streams for this investigation.",
                retry_after_seconds=5,
                details={"investigation_id": investigation_id, "limit": self._max_subscribers},
            )

        if after_seq is None:
            replay = tuple(channel.history)
            resume_gap = None
        else:
            replay = tuple(e for e in channel.history if e.seq > after_seq)
            resume_gap = self._resume_gap(channel, after_seq)

        subscription = _Subscription(
            self,
            investigation_id,
            maxsize=self._queue_events,
            replay=replay,
            resume_gap=resume_gap,
        )
        channel.subscribers.add(subscription)
        channel.last_activity = time.monotonic()
        return subscription

    def unsubscribe(self, subscription: _Subscription) -> None:
        """Drop a connection's mailbox. Called by `_Subscription.close` only."""
        channel = self._channels.get(subscription.investigation_id)
        if channel is None:
            return
        channel.subscribers.discard(subscription)
        channel.last_activity = time.monotonic()

    def subscriber_count(self, investigation_id: str) -> int:
        """Live connections for one investigation. Observability and tests."""
        channel = self._channels.get(investigation_id)
        return len(channel.subscribers) if channel else 0

    def channel_count(self) -> int:
        """Investigations with retained history or subscribers."""
        return len(self._channels)

    # -- internals --------------------------------------------------------- #

    def _channel(self, investigation_id: str) -> _Channel:
        channel = self._channels.get(investigation_id)
        if channel is None:
            # A channel is created on subscribe as well as on publish: a client
            # legitimately attaches before the orchestrator emits anything, and
            # refusing to create one there would drop the first events of every
            # fast run.
            channel = _Channel(history=deque(maxlen=self._replay_events))
            self._channels[investigation_id] = channel
        return channel

    def _resume_gap(self, channel: _Channel, after_seq: int) -> SeqGap | None:
        """Whether the ring has already discarded what this client asked to resume."""
        if not channel.history:
            return None
        oldest = channel.history[0].seq
        if after_seq + 1 >= oldest:
            return None
        return SeqGap(after_seq + 1, oldest - 1, reason="replay_buffer_evicted")

    def _prune(self) -> None:
        """Forget channels nobody is watching and nothing is writing to.

        Without this the hub is a memory leak with a slow fuse: every
        investigation the process ever streamed keeps its ring buffer, and the
        process is long-lived. Terminal channels are kept for the full retention
        window anyway, because a user who opens the report an hour later still
        expects the timeline that produced it.
        """
        now = time.monotonic()
        if now - self._last_prune < PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        stale = [
            investigation_id
            for investigation_id, channel in self._channels.items()
            if not channel.subscribers and now - channel.last_activity > self._retention_seconds
        ]
        for investigation_id in stale:
            del self._channels[investigation_id]
        if stale:
            logger.debug("stream.channels_pruned", count=len(stale))


_hub: InProcessTimelineHub | None = None


def get_timeline_hub() -> InProcessTimelineHub:
    """The process-wide hub. Lazily created, following `backend/db/session.py`.

    Also the publish side: an orchestrator running in this process calls
    `await get_timeline_hub().publish(event)`. It is a FastAPI dependency as well,
    so a test substitutes its own hub with `app.dependency_overrides` instead of
    mutating module state that the next test would inherit.
    """
    global _hub
    if _hub is None:
        _hub = InProcessTimelineHub()
    return _hub


def get_heartbeat_interval() -> float:
    """Seconds between heartbeats, as a dependency so a test can shorten it.

    A test that waited 15 real seconds to observe a `: ping` would be the slowest
    test in the suite, and one that asserted nothing about the interval would not
    be testing the property that matters.
    """
    return HEARTBEAT_INTERVAL_SECONDS


# --------------------------------------------------------------------------- #
# Access control seam
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StreamAccess:
    """The authorized subject of one stream connection.

    Carries the investigation id the guard *resolved*, which the handler uses in
    preference to the raw path parameter: whatever normalization or tenant
    scoping the guard applied would otherwise be bypassed by reading the URL again.
    """

    investigation_id: str
    tenant_id: str
    principal_id: str


async def require_stream_access(
    investigation_id: str, request: Request
) -> StreamAccess:  # pragma: no cover -- replaced by `deps.py` when it lands
    """Authenticate, authorize `investigations:read`, and scope to the tenant.

    Fails closed. `docs/api-reference.md` §3.1 puts credential verification,
    scope checks and tenancy in `backend/api/deps.py`, and that module is still a
    docstring stub: it exposes no principal dependency, no scope check and no
    session dependency to resolve the investigation's tenant with. Guessing an API
    for it here would produce a stream that *looks* authorized and is not, which
    is the single worst outcome available -- this endpoint emits the full contents
    of an investigation, including quoted source material.

    Replace by overriding this dependency once `deps.py` provides
    `current_principal` and a scope guard.
    """
    raise NotImplementedError(
        "backend/api/deps.py provides no principal, scope or tenancy dependency yet, "
        "so the SSE stream cannot verify `investigations:read` for "
        f"investigation {investigation_id!r} on {request.url.path}. "
        "Implement deps.py and override `require_stream_access`."
    )


# --------------------------------------------------------------------------- #
# The endpoint
# --------------------------------------------------------------------------- #

router = APIRouter(tags=["investigations"])


def _resolve_request_id(request: Request) -> str:
    """The correlation id every frame on this connection carries.

    Resolution order is deliberate.

    `RequestIdMiddleware` (`backend/middleware/request_id.py`) wins, because it
    owns the policy for what a client-supplied id may look like -- it accepts only
    a UUID or ULID and mints a replacement otherwise, precisely so an
    arbitrary-length client string never reaches a log field. Reading the bound
    value here rather than re-parsing the header is what makes the id on these
    frames the *same* id as on every log line the request produced.

    Falling back to the raw header covers an application assembled without that
    middleware -- a focused test app, a future embedding of this router -- where
    honouring §3.7 is still better than minting an id the caller has never seen.
    It is truncated because on that path nothing has validated it, and it is
    copied into every event on a connection that can last half an hour.
    """
    bound = get_correlation_id()
    if bound and bound != UNBOUND_CORRELATION_ID:
        return bound
    supplied = request.headers.get("x-request-id")
    if supplied:
        return supplied[:MAX_REQUEST_ID_CHARS]
    return uuid.uuid4().hex


def _parse_last_event_id(raw: str | None) -> int | None:
    """`Last-Event-ID` to a resume point, tolerating anything a client sends.

    Returns `None` -- meaning "replay everything retained" -- for absent, blank or
    unparseable values. Never raises: a `400` here is answered by an `EventSource`
    reconnecting with the same bad id forever, which turns a malformed header into
    a self-inflicted denial of service. See the module docstring: duplicates are
    recoverable by the client, gaps are not.
    """
    if raw is None:
        return None
    candidate = raw.strip()
    if not candidate:
        return None
    try:
        parsed = int(candidate)
    except ValueError:
        logger.info("stream.unparseable_last_event_id", length=len(candidate))
        return None
    if parsed < 0:
        return None
    return parsed


@router.get(
    "/investigations/{investigation_id}/stream",
    summary="Server-sent execution timeline for a running investigation.",
    response_class=EventSourceResponse,
)
async def stream_investigation(
    investigation_id: str,
    request: Request,
    access: Annotated[StreamAccess, Depends(require_stream_access)],
    source: Annotated[TimelineSource, Depends(get_timeline_hub)],
    heartbeat_seconds: Annotated[float, Depends(get_heartbeat_interval)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    last_event_id_param: Annotated[str | None, Query(alias="last_event_id")] = None,
) -> EventSourceResponse:
    """Open the timeline stream.

    The query-parameter form of the resume point exists alongside the header
    because a browser's `EventSource` sets `Last-Event-ID` only on the reconnects
    *it* initiates. An application that reloads the page, or one that reconnects
    itself after exhausting backoff, knows the last `seq` it rendered but has no
    way to put it in a header -- `EventSource` accepts no headers at all. The
    header wins when both are present, since only the browser sends it.
    """
    request_id = _resolve_request_id(request)
    resume_hint = last_event_id if last_event_id is not None else last_event_id_param
    after_seq = _parse_last_event_id(resume_hint)

    # Subscribing before the response is constructed means a `429` from the
    # subscriber cap is a normal problem+json error, not a `200` that opens a
    # stream and then dies -- §3.6 requires the refusal to happen "before the
    # stream opens".
    subscription = await source.subscribe(access.investigation_id, after_seq=after_seq)

    logger.info(
        "stream.opened",
        investigation_id=access.investigation_id,
        tenant_id=access.tenant_id,
        principal_id=access.principal_id,
        request_id=request_id,
        resume_from=after_seq,
        replay_events=len(subscription.replay),
    )

    async def on_client_close(message: Mapping[str, Any]) -> None:
        """Stop producing the moment the socket goes away.

        Runs on `http.disconnect`, before the task-group cancellation that would
        eventually reach the generator's `finally`. Both paths close the same
        subscription and `close()` is idempotent; having both means the mailbox is
        released even on a server whose cancellation never reaches the generator
        -- which is precisely the case where a leaked subscription would be
        invisible and permanent.
        """
        logger.info(
            "stream.client_disconnected",
            investigation_id=access.investigation_id,
            request_id=request_id,
        )
        subscription.close()

    return EventSourceResponse(
        _timeline_frames(subscription, request_id=request_id),
        # `ping` is annotated `int | None` upstream but its setter accepts and
        # its sleep uses a float; a sub-second heartbeat is what makes this
        # testable in milliseconds instead of in quarter-minutes. Never pass 0 --
        # sse-starlette 3.4.6 documents 0 as "disabled" but implements it as
        # `sleep(0)` inside `while self.active`, which is a busy loop that
        # saturates the event loop with ping frames.
        ping=heartbeat_seconds,  # type: ignore[arg-type]
        # Exactly `: ping`, per §5. The library's default appends a timestamp,
        # which is a payload nobody reads on a frame sent every 15 seconds to
        # every open connection.
        ping_message_factory=lambda: ServerSentEvent(comment="ping"),
        send_timeout=SEND_TIMEOUT_SECONDS,
        client_close_handler_callable=on_client_close,
        headers={
            # `no-cache`, not the library's `no-store`: §5 fixes the value, and a
            # revalidating intermediary is fine on a response that never ends.
            "Cache-Control": "no-cache",
            "X-Request-ID": request_id,
        },
    )


async def _timeline_frames(
    subscription: TimelineSubscription, *, request_id: str
) -> AsyncIterator[ServerSentEvent]:
    """Yield frames for one connection until a terminal event or a disconnect.

    Structured as replay-then-live with `seq` de-duplication across the boundary,
    so a client sees each event once and in order however the source snapshotted
    it.

    The `finally` is the whole disconnect story: whether this generator ends by
    returning on `done`, by `GeneratorExit` when the consumer stops iterating, or
    by `CancelledError` when the task group is torn down on `http.disconnect`, the
    subscription is released. Nothing awaits in there, so the release cannot
    itself be interrupted by the cancellation that triggered it.
    """
    investigation_id = subscription.investigation_id
    last_seq = -1
    try:
        # An immediate first byte. Proxies buffer a response until something
        # arrives, and a browser's `onopen` does not fire until the headers are
        # flushed; without this, a stream that stays quiet for a minute is
        # indistinguishable from a connection that never established. The `retry:`
        # hint rides along on the same frame.
        yield ServerSentEvent(comment=f"open {investigation_id}", retry=RECONNECT_ADVICE_MS)

        if subscription.resume_gap is not None:
            logger.warning(
                "stream.resume_gap",
                investigation_id=investigation_id,
                request_id=request_id,
                missing=subscription.resume_gap.missing,
            )
            yield _gap_frame(
                investigation_id=investigation_id,
                request_id=request_id,
                gap=subscription.resume_gap,
            )

        for event in subscription.replay:
            yield _render(event, request_id=request_id, investigation_id=investigation_id)
            last_seq = max(last_seq, event.seq)
            if event.is_terminal:
                # Late attach: history already contains the terminal event, so the
                # contract's "replay the full history, then done, then close" is
                # satisfied by returning here rather than waiting for a live event
                # that will never come.
                return

        while True:
            event = await subscription.get()

            # Drops are reported before the event that survived them, so the
            # client learns about the hole in the same order the hole occurred.
            gap = subscription.take_gap()
            if gap is not None:
                logger.warning(
                    "stream.backpressure_drop",
                    investigation_id=investigation_id,
                    request_id=request_id,
                    missing=gap.missing,
                )
                yield _gap_frame(investigation_id=investigation_id, request_id=request_id, gap=gap)

            if event.seq <= last_seq:
                continue  # already delivered from the replay snapshot

            yield _render(event, request_id=request_id, investigation_id=investigation_id)
            last_seq = event.seq
            if event.is_terminal:
                return
    finally:
        subscription.close()


def _render(event: TimelineEvent, *, request_id: str, investigation_id: str) -> ServerSentEvent:
    """Encode one event, degrading to a reported gap when it cannot be encoded.

    An unencodable payload -- a NaN score, an object no `default` handles -- must
    not kill the connection, and must not be skipped silently either. Skipping
    silently leaves the client with a `seq` gap it cannot distinguish from a
    dropped event; killing the connection sends the client into a reconnect that
    replays the *same* poison event from the ring buffer and fails identically,
    forever.

    So: a non-terminal poison event becomes a `stream.gap` and the stream
    continues. A *terminal* one becomes a synthetic `error`, because a client that
    never receives a terminal event waits on a run that has already finished.
    """
    try:
        return _event_frame(event, request_id=request_id)
    except (TypeError, ValueError) as exc:
        logger.error(
            "stream.unencodable_event",
            investigation_id=investigation_id,
            request_id=request_id,
            event_type=str(event.type),
            seq=event.seq,
            reason=type(exc).__name__,
        )
        if event.is_terminal:
            return _internal_error_frame(
                investigation_id=investigation_id, request_id=request_id, seq=event.seq
            )
        return _gap_frame(
            investigation_id=investigation_id,
            request_id=request_id,
            gap=SeqGap(event.seq, event.seq, reason="unencodable_event"),
        )
