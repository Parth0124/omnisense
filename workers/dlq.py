"""The dead-letter queue: inspection, replay, and the discipline around both.

A message reaches the DLQ when `services/events/consumer.py` has exhausted its
retries. That is not a transient failure -- transient failures are what the
retries were for -- so a message here represents a bug, a schema mismatch, or a
dependency that was down long enough to burn the whole retry budget.

**The DLQ is not a queue that drains itself, and that is deliberate.** Automatic
replay is the obvious feature and the wrong one: a poison message that crashes a
handler will crash it again, and an auto-replaying DLQ turns one bad message into
an infinite loop that looks like healthy throughput. Replay here is explicit,
operator-initiated, and bounded.

**Replay preserves the original envelope.** The correlation id, the original
timestamp and the payload go back onto the source topic unchanged. Minting a new
correlation id would break the join across logs, metrics and traces at exactly
the moment somebody is trying to follow one message through the system
(`docs/observability.md` §1) -- which is the only reason anyone is looking at the
DLQ in the first place.

**Replay count is carried and capped.** A message that has already been replayed
three times and failed three times is not going to succeed on the fourth. The
counter lives in the DLQ envelope's headers so it survives the round trip, and
`MAX_REPLAY_ATTEMPTS` is what stops a well-meaning operator from building the
infinite loop by hand.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Final

from backend.core.logging import get_logger
from services.events.topics import TopicRole

if TYPE_CHECKING:  # pragma: no cover
    from services.events.consumer import ConsumedMessage
    from services.events.producer import EventProducer

__all__ = [
    "MAX_REPLAY_ATTEMPTS",
    "REPLAY_COUNT_HEADER",
    "DlqEntry",
    "DlqInspector",
    "DlqReplayer",
    "FailureClass",
    "ReplayOutcome",
    "classify_failure",
]

logger = get_logger(__name__)

MAX_REPLAY_ATTEMPTS: Final = 3
"""How many times one message may be replayed before it is refused.

The counter that prevents an operator from hand-building the infinite loop that
auto-replay would have built automatically. A message that has failed its retry
budget and then three full replays needs a code change, not a fourth attempt.
"""

REPLAY_COUNT_HEADER: Final = "x-omnisense-replay-count"
"""Header carrying the replay count across the round trip.

In a header rather than the payload because the payload is the *original*
message, byte-for-byte. Mutating it to add bookkeeping would mean the thing
replayed is not the thing that failed, and the reproduction would differ from the
incident in a way nobody could see.
"""

DEFAULT_INSPECT_LIMIT: Final = 100


class FailureClass(enum.StrEnum):
    """Why a message ended up here. Decides whether replay could ever work.

    The classification exists because "replay everything" is almost always wrong
    and "replay nothing" is almost always wasteful. A dependency outage that has
    since been fixed is exactly what replay is for; a schema mismatch will fail
    identically forever.
    """

    DEPENDENCY = "dependency"
    """A datastore or external service was unreachable. Replay after it recovers."""

    VALIDATION = "validation"
    """The payload did not match its schema. Replay will fail identically."""

    POISON = "poison"
    """The handler raised on content it cannot process. Needs a code change."""

    TIMEOUT = "timeout"
    """The handler exceeded its budget. Replay may work if the cause was load."""

    UNKNOWN = "unknown"
    """Unclassified. Treated as non-replayable by default -- see `is_replayable`."""

    @property
    def is_replayable(self) -> bool:
        """Whether replaying could plausibly succeed.

        `UNKNOWN` is deliberately excluded. An unrecognised failure replayed in
        bulk is the fastest way to re-run whatever caused the outage, and the
        cost of being wrong in this direction is a manual review rather than a
        repeat incident.
        """
        return self in (FailureClass.DEPENDENCY, FailureClass.TIMEOUT)


_DEPENDENCY_MARKERS: Final[tuple[str, ...]] = (
    "connection",
    "unavailable",
    "refused",
    "unreachable",
    "no route",
    "temporarily",
    "serviceunavailable",
    "operationalerror",
)
_VALIDATION_MARKERS: Final[tuple[str, ...]] = (
    "validationerror",
    "validation error",
    "field required",
    "extra inputs",
    "unable to parse",
    "invalid",
)
_TIMEOUT_MARKERS: Final[tuple[str, ...]] = ("timeout", "timed out", "deadline")


def classify_failure(error_type: str | None, error_message: str | None) -> FailureClass:
    """Bucket a recorded failure by its type name and message.

    Text matching, which is coarse -- the alternative is importing every driver's
    exception hierarchy into a module that exists to be run from a shell, and the
    DLQ record only carries strings anyway because it crossed a topic to get here.

    Ordered most-specific-first: a `ConnectionTimeout` is a dependency problem
    rather than a load problem, and classifying it as a timeout would suggest
    retrying under lighter load when the fix is to restore the dependency.
    """
    haystack = f"{error_type or ''} {error_message or ''}".casefold()
    if not haystack.strip():
        return FailureClass.UNKNOWN
    if any(marker in haystack for marker in _DEPENDENCY_MARKERS):
        return FailureClass.DEPENDENCY
    if any(marker in haystack for marker in _VALIDATION_MARKERS):
        return FailureClass.VALIDATION
    if any(marker in haystack for marker in _TIMEOUT_MARKERS):
        return FailureClass.TIMEOUT
    return FailureClass.POISON


@dataclass(frozen=True, slots=True)
class DlqEntry:
    """One dead-lettered message, as an operator sees it."""

    message_id: str
    original_topic: str
    correlation_id: str | None
    error_type: str | None
    error_message: str | None
    failed_at: datetime
    attempts: int = 0
    replay_count: int = 0
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failure_class(self) -> FailureClass:
        return classify_failure(self.error_type, self.error_message)

    @property
    def is_replayable(self) -> bool:
        """Replayable *and* not already exhausted."""
        return self.failure_class.is_replayable and self.replay_count < MAX_REPLAY_ATTEMPTS

    @property
    def refusal_reason(self) -> str | None:
        """Why this entry will not be replayed, or `None` if it will be.

        Returned as a sentence rather than a flag because it is printed to an
        operator who is deciding what to do next, and "not replayable" without a
        reason invites the assumption that the tool is broken.
        """
        if self.replay_count >= MAX_REPLAY_ATTEMPTS:
            return (
                f"already replayed {self.replay_count} times; this needs a code "
                "change, not another attempt"
            )
        if not self.failure_class.is_replayable:
            return (
                f"{self.failure_class.value} failures fail identically on replay; "
                "fix the cause first"
            )
        return None


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    """What one replay pass did. Every number here is actionable."""

    considered: int
    replayed: int
    refused: int
    failed: int
    refusals: tuple[str, ...] = ()

    @property
    def had_effect(self) -> bool:
        return self.replayed > 0


class DlqInspector:
    """Reads the DLQ without consuming it.

    Reading without committing is the whole requirement. An inspector that
    consumed would remove messages from the queue as a side effect of somebody
    looking at them, which is the last thing anyone wants during an incident.
    """

    def __init__(self, reader: Any) -> None:
        self._reader = reader

    async def recent(
        self,
        *,
        limit: int = DEFAULT_INSPECT_LIMIT,
        since: datetime | None = None,
        failure_class: FailureClass | None = None,
    ) -> list[DlqEntry]:
        """The most recent entries, newest first, optionally filtered."""
        raw = await self._reader.peek(limit=limit, since=since)
        entries = [_entry_from(item) for item in raw]
        entries = [entry for entry in entries if entry is not None]
        if failure_class is not None:
            entries = [entry for entry in entries if entry.failure_class is failure_class]
        return sorted(entries, key=lambda entry: entry.failed_at, reverse=True)[:limit]

    async def summarize(
        self, *, window: timedelta = timedelta(hours=24)
    ) -> dict[str, int]:
        """Counts by failure class over a window.

        The first thing to look at during an incident: fifty dependency failures
        and two poison messages is a different situation from fifty poison
        messages, and the raw list does not make that visible at a glance.
        """
        since = datetime.now(UTC) - window
        entries = await self.recent(limit=1000, since=since)
        counts: dict[str, int] = {member.value: 0 for member in FailureClass}
        for entry in entries:
            counts[entry.failure_class.value] += 1
        counts["total"] = len(entries)
        counts["replayable"] = sum(1 for entry in entries if entry.is_replayable)
        return counts


class DlqReplayer:
    """Puts messages back on their original topic, deliberately and boundedly."""

    def __init__(self, producer: EventProducer, *, dry_run: bool = False) -> None:
        self._producer = producer
        self._dry_run = dry_run

    async def replay(self, entries: Iterable[DlqEntry]) -> ReplayOutcome:
        """Replay every replayable entry, refusing the rest with a reason.

        Refusals are counted and reported rather than silently skipped. An
        operator who replayed a hundred messages and saw "replayed: 12" with no
        explanation for the other 88 will assume the tool failed and run it
        again.

        `dry_run` exists because the first thing anyone should do with a bulk
        replay is find out what it would do. A replay tool without one gets
        tested in production, once.
        """
        considered = replayed = refused = failed = 0
        refusals: list[str] = []

        for entry in entries:
            considered += 1
            reason = entry.refusal_reason
            if reason is not None:
                refused += 1
                refusals.append(f"{entry.message_id}: {reason}")
                continue

            if self._dry_run:
                replayed += 1
                logger.info(
                    "dlq.replay.dry_run",
                    message_id=entry.message_id,
                    topic=entry.original_topic,
                )
                continue

            try:
                await self._producer.publish(
                    topic=entry.original_topic,
                    payload=entry.payload,
                    # The original correlation id, not a new one. Breaking the
                    # chain here breaks it at exactly the moment somebody is
                    # following one message through the system.
                    correlation_id=entry.correlation_id,
                    headers={REPLAY_COUNT_HEADER: str(entry.replay_count + 1)},
                )
            except Exception as error:  # noqa: BLE001 -- one bad send must not stop the batch
                failed += 1
                logger.error(
                    "dlq.replay.failed",
                    message_id=entry.message_id,
                    error=type(error).__name__,
                )
                continue
            replayed += 1

        outcome = ReplayOutcome(
            considered=considered,
            replayed=replayed,
            refused=refused,
            failed=failed,
            refusals=tuple(refusals[:50]),
        )
        logger.info(
            "dlq.replay.complete",
            considered=considered,
            replayed=replayed,
            refused=refused,
            failed=failed,
            dry_run=self._dry_run,
        )
        return outcome


def _entry_from(raw: Any) -> DlqEntry | None:
    """Build an entry from a stored DLQ record, tolerating a partial one.

    Returns `None` rather than raising on a record it cannot read. The DLQ is
    where malformed things go; a malformed *DLQ record* must not take down the
    tool an operator is using to look at the other ninety-nine.
    """
    payload = raw if isinstance(raw, Mapping) else getattr(raw, "__dict__", {})
    if not isinstance(payload, Mapping):
        return None

    message_id = payload.get("message_id") or payload.get("id")
    topic = payload.get("original_topic") or payload.get("topic")
    if not isinstance(message_id, str) or not isinstance(topic, str):
        logger.warning("dlq.unreadable_record", keys=sorted(str(k) for k in payload)[:10])
        return None

    failed_at = payload.get("failed_at")
    if not isinstance(failed_at, datetime):
        failed_at = datetime.now(UTC)

    return DlqEntry(
        message_id=message_id,
        original_topic=topic,
        correlation_id=_str_or_none(payload.get("correlation_id")),
        error_type=_str_or_none(payload.get("error_type")),
        error_message=_str_or_none(payload.get("error_message")),
        failed_at=failed_at,
        attempts=_int_or(payload.get("attempts"), 0),
        replay_count=_int_or(payload.get("replay_count"), 0),
        payload=payload.get("payload") if isinstance(payload.get("payload"), Mapping) else {},
    )


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int_or(value: Any, default: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def dlq_topic_role() -> TopicRole:
    """The role the DLQ lives under. Named so callers need not import the enum."""
    return TopicRole.DLQ
