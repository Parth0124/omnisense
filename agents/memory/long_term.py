"""Cross-investigation memory: what the system already established, and when.

`agents/memory/short_term.py` holds one run's working context and
`agents/memory/scratchpad.py` holds one run's bulk intermediates. Both die with
the run. This is the third kind: facts that survive it.

The motivating case is concrete. A user asks about Acme's battery strategy on
Monday and about Acme's supply chain on Thursday. The Thursday run re-retrieves
the same corpus, re-extracts the same entities and re-derives the same background
-- paying full cost to rediscover what Monday already established. Long-term
memory is what lets Thursday start from Monday's conclusions.

**Every memory carries its provenance and its age, and both are load-bearing.**
A remembered fact is *weaker* evidence than a freshly retrieved one, because the
world moved and nobody checked. So a `Memory` is never returned bare: it comes
with the investigation that produced it, the signals that supported it then, and
`established_at`. An agent that treats a six-month-old memory as current is
making a claim about today from evidence about last winter, and the timestamp is
the only thing that lets it not.

**Memories decay rather than expire.** A hard TTL is wrong in both directions:
"Acme was founded in 2011" does not become false after ninety days, and "Acme's
CEO is X" can be false in a week. So `relevance()` combines the stored confidence
with an age penalty scaled by the memory's own `volatility`, and the caller
filters on the result. Nothing is deleted for being old -- it is *scored down*,
which keeps it available to a query that explicitly wants history.

**Nothing is written here that was not cited.** `supporting_signal_ids` is
required and non-empty, for the same reason `agents/insight/schemas.py` requires
it: a memory with no provenance is a belief, and a system that accumulates
beliefs across runs will confidently repeat its own earlier guesses back to
itself. That is the specific failure this module is built to prevent -- and it is
the failure mode that makes naive agent memory dangerous rather than merely
useless.

Layer note: **L3 agents.** Takes a store port; the vector search that finds
related memories is injected rather than imported, so this is testable with no
Qdrant.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Protocol

from backend.core.logging import get_logger
from models.enums import AgentName

__all__ = [
    "DEFAULT_HALF_LIFE_DAYS",
    "MIN_RELEVANCE",
    "InMemoryLongTermStore",
    "LongTermMemory",
    "Memory",
    "MemoryKind",
    "MemoryStore",
    "Volatility",
]

logger = get_logger(__name__)

MIN_RELEVANCE: Final = 0.25
"""Below this, a memory is not worth the tokens it costs to include.

A recalled fact enters a context window and displaces something else. One scored
below a quarter is old, weakly-evidenced, or both -- and including it trades a
retrieved passage for a stale claim, which is the wrong direction.
"""

DEFAULT_HALF_LIFE_DAYS: Final = 90.0


class Volatility(enum.StrEnum):
    """How fast a class of fact goes stale. Drives the decay half-life.

    Per-memory rather than global because the range is enormous. A founding year
    is fixed forever; a current price is wrong within a week. One decay curve for
    both either discards durable facts or trusts perishable ones, and there is no
    middle setting that is right for either.
    """

    STABLE = "stable"
    """Founding years, headquarters, historical events. Effectively permanent."""

    SLOW = "slow"
    """Market positioning, product lines, competitive structure. Months."""

    FAST = "fast"
    """Sentiment, pricing, personnel, active incidents. Weeks at most."""

    @property
    def half_life_days(self) -> float:
        match self:
            case Volatility.STABLE:
                return 3650.0
            case Volatility.SLOW:
                return DEFAULT_HALF_LIFE_DAYS
            case Volatility.FAST:
                return 14.0
        return DEFAULT_HALF_LIFE_DAYS


class MemoryKind(enum.StrEnum):
    """What sort of thing is remembered. Decides how it may be reused."""

    ENTITY_FACT = "entity_fact"
    """Something established about a named entity. Reusable as background."""

    RELATIONSHIP = "relationship"
    """A link between entities -- competes with, acquired, supplies."""

    FINDING = "finding"
    """A conclusion an investigation reached. Reusable, and always as a prior
    rather than as evidence: it was somebody else's synthesis of other data."""

    USER_CONTEXT = "user_context"
    """What this tenant keeps asking about. Shapes retrieval, never a claim.

    Deliberately separate from the others because it must never enter a report.
    "This tenant asks about batteries a lot" is a useful retrieval hint and a
    meaningless statement about the world.
    """


@dataclass(frozen=True, slots=True)
class Memory:
    """One remembered fact, with everything needed to discount it."""

    id: str
    tenant_id: str
    kind: MemoryKind
    statement: str
    supporting_signal_ids: tuple[str, ...]
    established_at: datetime
    confidence: float = 0.5
    volatility: Volatility = Volatility.SLOW
    entity_ids: tuple[str, ...] = ()
    investigation_id: str | None = None
    established_by: AgentName = AgentName.INSIGHT
    superseded_by: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.supporting_signal_ids:
            raise ValueError(
                f"memory {self.id!r} cites no signal. A memory without provenance is "
                "a belief, and a system that accumulates beliefs across runs will "
                "repeat its own earlier guesses back to itself as established fact."
            )
        if self.established_at.tzinfo is None:
            raise ValueError("established_at must be timezone-aware")

    def age_days(self, *, now: datetime | None = None) -> float:
        moment = now or datetime.now(UTC)
        return max(0.0, (moment - self.established_at).total_seconds() / 86_400.0)

    def relevance(self, *, now: datetime | None = None) -> float:
        """Confidence discounted by age, on this memory's own decay curve.

        Exponential half-life rather than a linear ramp or a cliff. A linear
        decay reaches zero at an arbitrary point and is equally wrong on both
        sides of it; a cliff makes a fact fully trusted the day before it expires
        and worthless the day after. Halving is the shape that matches how
        confidence in an unchecked fact actually behaves.

        A superseded memory scores zero unconditionally. It is not stale -- it is
        *known wrong*, having been replaced by something newer, and age has
        nothing to do with it.
        """
        if self.superseded_by is not None:
            return 0.0
        decay = 0.5 ** (self.age_days(now=now) / self.volatility.half_life_days)
        return max(0.0, min(1.0, self.confidence * decay))

    def is_usable(self, *, now: datetime | None = None, floor: float = MIN_RELEVANCE) -> bool:
        return self.relevance(now=now) >= floor

    def as_context_line(self, *, now: datetime | None = None) -> str:
        """Render for a prompt, with the age stated.

        The age is in the string on purpose. A model handed "Acme's CEO is X"
        will use it as current; handed "Acme's CEO is X (established 214 days
        ago, confidence 0.31)" it has what it needs to hedge -- and the hedge is
        the difference between recalling and asserting.
        """
        days = int(self.age_days(now=now))
        return (
            f"{self.statement} "
            f"(established {days}d ago, relevance {self.relevance(now=now):.2f}, "
            f"{len(self.supporting_signal_ids)} source"
            f"{'s' if len(self.supporting_signal_ids) != 1 else ''})"
        )


class MemoryStore(Protocol):
    """Persistence port. Deliberately small enough to fake in a test."""

    async def put(self, memory: Memory) -> None: ...

    async def get(self, tenant_id: str, memory_id: str) -> Memory | None: ...

    async def search(
        self,
        tenant_id: str,
        *,
        entity_ids: Sequence[str] = (),
        kinds: Sequence[MemoryKind] = (),
        limit: int = 20,
    ) -> Sequence[Memory]: ...

    async def supersede(self, tenant_id: str, memory_id: str, by: str) -> None: ...


class InMemoryLongTermStore:
    """A dict-backed store. Tests and single-process development.

    Filtering is a linear scan, which is correct at the scale this is for and
    wrong at any other -- named here so nobody deploys it and then wonders why
    recall got slow.
    """

    def __init__(self) -> None:
        self._memories: dict[tuple[str, str], Memory] = {}

    async def put(self, memory: Memory) -> None:
        self._memories[(memory.tenant_id, memory.id)] = memory

    async def get(self, tenant_id: str, memory_id: str) -> Memory | None:
        return self._memories.get((tenant_id, memory_id))

    async def search(
        self,
        tenant_id: str,
        *,
        entity_ids: Sequence[str] = (),
        kinds: Sequence[MemoryKind] = (),
        limit: int = 20,
    ) -> Sequence[Memory]:
        wanted_entities = set(entity_ids)
        wanted_kinds = set(kinds)
        matches = [
            memory
            for (owner, _), memory in self._memories.items()
            if owner == tenant_id
            and (not wanted_kinds or memory.kind in wanted_kinds)
            and (not wanted_entities or wanted_entities & set(memory.entity_ids))
        ]
        return sorted(matches, key=lambda m: -m.relevance())[:limit]

    async def supersede(self, tenant_id: str, memory_id: str, by: str) -> None:
        existing = self._memories.get((tenant_id, memory_id))
        if existing is not None:
            self._memories[(tenant_id, memory_id)] = Memory(
                **{**_as_dict(existing), "superseded_by": by}
            )


@dataclass(slots=True)
class LongTermMemory:
    """Recall and record facts across investigations."""

    store: MemoryStore
    tenant_id: str
    relevance_floor: float = MIN_RELEVANCE

    async def recall(
        self,
        *,
        entity_ids: Sequence[str] = (),
        kinds: Sequence[MemoryKind] = (),
        limit: int = 10,
        now: datetime | None = None,
    ) -> list[Memory]:
        """Memories worth spending context on, most relevant first.

        Filtered by *relevance*, not by age. A stable fact from two years ago
        outranks a volatile one from last month, which is the correct ordering
        and the whole reason volatility is per-memory.

        `USER_CONTEXT` is excluded unless asked for by name. It shapes retrieval
        and must never reach a report -- "this tenant asks about batteries" is a
        useful hint and a meaningless claim about the world, and the two are one
        careless prompt away from each other.
        """
        requested = list(kinds) or [
            MemoryKind.ENTITY_FACT,
            MemoryKind.RELATIONSHIP,
            MemoryKind.FINDING,
        ]
        found = await self.store.search(
            self.tenant_id, entity_ids=entity_ids, kinds=requested, limit=limit * 3
        )
        usable = [
            memory
            for memory in found
            if memory.is_usable(now=now, floor=self.relevance_floor)
        ]
        usable.sort(key=lambda memory: (-memory.relevance(now=now), memory.id))
        return usable[:limit]

    async def remember(
        self,
        *,
        memory_id: str,
        kind: MemoryKind,
        statement: str,
        supporting_signal_ids: Sequence[str],
        confidence: float,
        volatility: Volatility = Volatility.SLOW,
        entity_ids: Sequence[str] = (),
        investigation_id: str | None = None,
        established_by: AgentName = AgentName.INSIGHT,
        now: datetime | None = None,
    ) -> Memory | None:
        """Record a fact. Returns `None` when it was refused.

        Refused rather than raised, because writing a memory is never the point
        of a run: an investigation that produced a good report and failed to
        record a memory has still succeeded, and raising here would fail it.

        Two refusals, both deliberate:

        *No provenance* -- enforced by `Memory` itself. See its docstring.

        *Low confidence* -- a fact the run itself was unsure of should not become
        a prior that a later run treats as background. Uncertainty compounds
        silently across runs: remembered at 0.3, recalled as "established", cited
        as support for the next conclusion, which is then remembered in turn.
        """
        if confidence < 0.5:
            logger.debug(
                "memory.not_recorded",
                reason="confidence below 0.5; a fact the run was unsure of must not "
                "become a later run's background",
                confidence=confidence,
            )
            return None

        try:
            memory = Memory(
                id=memory_id,
                tenant_id=self.tenant_id,
                kind=kind,
                statement=statement.strip(),
                supporting_signal_ids=tuple(dict.fromkeys(supporting_signal_ids)),
                established_at=now or datetime.now(UTC),
                confidence=float(confidence),
                volatility=volatility,
                entity_ids=tuple(dict.fromkeys(entity_ids)),
                investigation_id=investigation_id,
                established_by=established_by,
            )
        except ValueError as error:
            logger.warning("memory.rejected", reason=str(error))
            return None

        try:
            await self.store.put(memory)
        except Exception as error:  # noqa: BLE001 -- a failed write must not fail a run
            logger.warning("memory.write_failed", error=type(error).__name__)
            return None
        return memory

    async def supersede(self, memory_id: str, *, by: str) -> None:
        """Mark a memory replaced by a newer one.

        Superseding rather than deleting. The old memory scores zero and stops
        being recalled, but it remains readable -- which matters when a report
        cited it and somebody later asks what the system believed at the time.
        Deleting would make that question unanswerable.
        """
        try:
            await self.store.supersede(self.tenant_id, memory_id, by)
        except Exception as error:  # noqa: BLE001
            logger.warning("memory.supersede_failed", error=type(error).__name__)

    async def as_context(
        self, *, entity_ids: Sequence[str] = (), limit: int = 10, now: datetime | None = None
    ) -> str:
        """Recalled memories rendered for a prompt, or an empty string.

        Empty rather than a "no prior knowledge" line. A sentence saying nothing
        is known still occupies context and still invites the model to comment on
        the absence; nothing at all is the honest rendering of nothing.
        """
        memories = await self.recall(entity_ids=entity_ids, limit=limit, now=now)
        if not memories:
            return ""
        lines = [
            "Previously established by earlier investigations (weaker than freshly "
            "retrieved evidence -- age and relevance are stated, use them):"
        ]
        lines.extend(f"- {memory.as_context_line(now=now)}" for memory in memories)
        return "\n".join(lines)


def _as_dict(memory: Memory) -> dict[str, Any]:
    """Field dict for reconstructing a frozen `Memory` with one value changed."""
    return {
        "id": memory.id,
        "tenant_id": memory.tenant_id,
        "kind": memory.kind,
        "statement": memory.statement,
        "supporting_signal_ids": memory.supporting_signal_ids,
        "established_at": memory.established_at,
        "confidence": memory.confidence,
        "volatility": memory.volatility,
        "entity_ids": memory.entity_ids,
        "investigation_id": memory.investigation_id,
        "established_by": memory.established_by,
        "superseded_by": memory.superseded_by,
        "metadata": memory.metadata,
    }
