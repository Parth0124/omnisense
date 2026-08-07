"""Community summaries: the "global" half of GraphRAG.

Hybrid retrieval answers *local* questions well. "What are people saying about
Acme's battery life" is a query with lexical and semantic handles, and vector
plus keyword plus a two-hop expansion finds the passages. It answers *global*
questions badly: "what is happening in the battery supply chain" has no single
passage that contains the answer, because the answer is a property of forty
documents taken together. Retrieving the top ten passages for that query returns
ten documents about batteries and no synthesis.

The GraphRAG answer is to precompute the synthesis. `graph/analytics/communities.py`
finds clusters of densely-connected entities; this module turns each cluster into
a short written summary, stores it, and makes it retrievable. A global query then
matches *summaries* rather than passages, and the model receives "there is a
cluster here about lithium sourcing, comprising these nine companies, evidenced
by these signals" -- which is the shape of the answer the question wanted.

**Summaries are evidence-bearing or they are not written.** Every summary carries
the entity ids it covers and the signal ids that evidenced those entities'
edges. Without that, a community summary is an LLM paragraph about a list of
company names -- fluent, unfalsifiable, and impossible to cite. Design Doc §2
requires every claim to carry a citation, and a summary is a claim about a group.

**Small communities are not summarised, and that is a quality decision.**
`graph/analytics/communities.py` already drops clusters below
`min_community_size`; this module additionally refuses to summarise a community
whose conductance says it is not really a cluster. Asking a model to find the
theme in a group that has no theme reliably produces one anyway, phrased
confidently. The refusal is cheaper than the retraction.

**Nothing here calls a model directly.** `CommunitySummarizer` takes a
`SummaryWriter` port, because `retrieval/` is an L1 library and may not import
`services/llm/`. The production implementation is a thin adapter over
`services/llm/provider.py`; the tests pass a function that returns a fixed
string, which is what makes the batching, the caching and the refusal logic
testable without a token of spend.

Layer note: **L1 library**, using the one declared exception in
`docs/architecture.md` §6.1 -- `retrieval/` may read `graph/`. It reads
`graph.analytics.communities` for the detection result and never writes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

import structlog

from graph.analytics.communities import Community, CommunityResult

__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_MAX_ENTITIES_IN_PROMPT",
    "MAX_SUMMARY_CHARS",
    "MIN_SUMMARISABLE_CONDUCTANCE",
    "CommunityMember",
    "CommunitySummarizer",
    "CommunitySummary",
    "SummaryRequest",
    "SummaryWriter",
    "render_community_prompt",
    "select_representatives",
]

_log = structlog.get_logger(__name__)

MAX_SUMMARY_CHARS: Final[int] = 1200
"""Ceiling on a stored summary.

A summary exists to be *retrieved into a context window alongside passages*. One
that runs to three thousand characters displaces the passages it was meant to
frame, and the model then answers from the summary alone -- which is a paraphrase
of a paraphrase, two steps from any quotable source.
"""

DEFAULT_MAX_ENTITIES_IN_PROMPT: Final[int] = 25
"""How many members are described to the model.

A sixty-entity community does not need sixty descriptions to be summarised; it
needs the ones that hold it together. `select_representatives` picks them by
centrality, and the cap is what keeps the prompt cost of a community roughly
constant instead of quadratic in cluster size.
"""

MIN_SUMMARISABLE_CONDUCTANCE: Final[float] = 0.6
"""Above this fraction of edge weight leaving the community, do not summarise.

Conductance 0.6 means most of the cluster's connections point *outside* it. There
is no theme to find, and a model asked to find one will produce a fluent sentence
about "diverse market participants" that a reader will mistake for a finding.
"""

DEFAULT_MAX_CONCURRENCY: Final[int] = 4


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CommunityMember:
    """What is known about one entity in a community, for the prompt.

    Deliberately thin. The summariser gets a name, a type, a one-line description
    and an importance score -- not the entity's full property set, and not the
    passages that mention it. Feeding passages in would make this a second
    retrieval pipeline with a different chunking strategy and no reranking, and
    the summary would start disagreeing with what direct retrieval returns for
    the same entities.
    """

    entity_id: str
    name: str
    entity_type: str = "Unknown"
    description: str | None = None
    importance: float = 0.0
    """Usually `pagerank_score`. Decides who is described when the cap bites."""

    signal_ids: Sequence[str] = ()
    """Evidence for this entity, carried through so the summary can cite."""


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    """One community, resolved to the members the summariser will describe."""

    community: Community
    members: tuple[CommunityMember, ...]
    representatives: tuple[CommunityMember, ...]

    @property
    def community_id(self) -> str:
        return self.community.community_id


@runtime_checkable
class SummaryWriter(Protocol):
    """Turns a rendered prompt into a paragraph. The whole LLM seam.

    Takes a string and returns a string. Nothing about messages, models,
    temperature or token budgets crosses this boundary, because `retrieval/` is
    an L1 library that may not import `services/llm/` -- and because a port this
    narrow is satisfied by a lambda in a test.
    """

    async def __call__(self, prompt: str) -> str: ...


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CommunitySummary:
    """A written summary of one community, with everything needed to cite it."""

    community_id: str
    title: str
    summary: str
    entity_ids: tuple[str, ...]
    entity_names: tuple[str, ...]
    signal_ids: tuple[str, ...]
    size: int
    conductance: float
    generated_at: datetime | None = None
    skipped_reason: str | None = None
    """Set when no summary was written. `summary` is empty in that case.

    Present rather than absent, because "we chose not to summarise this cluster"
    is information the caller needs: a community that appears in the graph but
    has no summary would otherwise look like a pipeline failure, and somebody
    would go looking for the bug.
    """

    @property
    def is_written(self) -> bool:
        return self.skipped_reason is None and bool(self.summary)

    def as_retrievable_text(self) -> str:
        """The form indexed for global-question retrieval.

        Title and body concatenated, entity names appended. The names matter: a
        user asking "what is happening with lithium suppliers" will not match a
        summary that discusses the topic without naming it, and the member names
        are the strongest lexical handle a community has.
        """
        if not self.is_written:
            return ""
        return f"{self.title}\n\n{self.summary}\n\nEntities: {', '.join(self.entity_names)}"


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


def select_representatives(
    members: Sequence[CommunityMember], limit: int = DEFAULT_MAX_ENTITIES_IN_PROMPT
) -> tuple[CommunityMember, ...]:
    """The most important members, most important first, deterministically.

    Sorted by importance descending and then by `entity_id` ascending. The second
    key is not decoration: PageRank values tie constantly on small graphs, and an
    unstable sort would send a different subset of a sixty-entity community to the
    model on every run, producing a different summary each time for a cluster
    that did not change.
    """
    ordered = sorted(members, key=lambda member: (-member.importance, member.entity_id))
    return tuple(ordered[:limit])


def render_community_prompt(request: SummaryRequest) -> str:
    """Build the summarisation prompt for one community.

    Three properties this prompt has on purpose:

    *It names the entities and nothing else.* No passage text, no scraped
    content. That means there is no untrusted third-party text in this prompt at
    all, so no injection fence is needed -- entity names and LLM-written
    descriptions have already been through the extraction boundary. If passages
    are ever added here, they must be fenced with `UntrustedText` first.

    *It states the size honestly.* A model told about 25 of 60 members and not
    told about the other 35 will write "this cluster of 25 companies", and the
    number will be wrong in a report.

    *It asks for a title and a body separately.* A single blob has to be split by
    heuristic afterwards, and the heuristic fails on the summaries that begin
    with a sentence rather than a heading -- which is most of them.
    """
    lines = [
        "You are summarising a cluster of related entities from a market "
        "intelligence knowledge graph.",
        "",
        f"The cluster contains {request.community.size} entities.",
    ]
    hidden = request.community.size - len(request.representatives)
    if hidden > 0:
        lines.append(
            f"The {len(request.representatives)} most central are listed below; "
            f"{hidden} less-connected members are omitted."
        )
    lines.append("")
    lines.append("Entities:")
    for member in request.representatives:
        description = f" -- {member.description}" if member.description else ""
        lines.append(f"- {member.name} ({member.entity_type}){description}")
    lines.extend(
        [
            "",
            "Write:",
            "TITLE: a short noun phrase naming what connects these entities.",
            "SUMMARY: two to four sentences on what this cluster is and why it "
            "holds together. State only what the entity list supports. If the "
            "entities have no clear common theme, say so plainly rather than "
            "inventing one.",
        ]
    )
    return "\n".join(lines)


def _parse_summary(raw: str) -> tuple[str, str]:
    """Split a `TITLE:` / `SUMMARY:` response, tolerating a model that ignores it.

    Falls back to "first line is the title" rather than raising. A malformed
    response is a prompt-adherence problem, not a pipeline failure, and losing a
    whole batch of summaries because one model call omitted a label would be a
    disproportionate response to a recoverable formatting slip.
    """
    title = ""
    body_lines: list[str] = []
    for line in raw.strip().splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("TITLE:"):
            title = stripped[6:].strip()
        elif upper.startswith("SUMMARY:"):
            body_lines.append(stripped[8:].strip())
        else:
            body_lines.append(stripped)

    body = "\n".join(line for line in body_lines if line).strip()
    if not title:
        parts = body.split("\n", 1)
        title = parts[0][:120].strip()
        body = parts[1].strip() if len(parts) > 1 else body
    return title, body[:MAX_SUMMARY_CHARS]


# --------------------------------------------------------------------------- #
# The summariser
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class CommunitySummarizer:
    """Turns a `CommunityResult` into retrievable summaries.

    Holds a cache keyed by `community_id`, which is why
    `graph/analytics/communities.community_id_for` is content-addressed: an
    unchanged cluster keeps its id across a recomputation and its summary is
    reused, while a cluster that gained or lost a member gets a new id and is
    re-summarised. A sequential id would invalidate every summary on every run,
    or worse, serve last week's summary for a cluster that now contains different
    companies.
    """

    writer: SummaryWriter
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_entities_in_prompt: int = DEFAULT_MAX_ENTITIES_IN_PROMPT
    min_conductance_to_skip: float = MIN_SUMMARISABLE_CONDUCTANCE
    _cache: dict[str, CommunitySummary] = field(default_factory=dict, repr=False)

    async def summarize_all(
        self,
        result: CommunityResult,
        members_by_entity: Mapping[str, CommunityMember],
        *,
        now: datetime | None = None,
    ) -> list[CommunitySummary]:
        """Summarise every community, concurrently and bounded.

        Bounded because the alternative is issuing one model call per community
        simultaneously: a graph with two hundred communities would open two
        hundred concurrent requests, get rate-limited, and retry into a longer
        wall-clock time than the bounded version would have taken.

        A community whose summary call fails is returned with a `skipped_reason`
        rather than dropped or raised. One provider hiccup must not discard the
        hundred and ninety-nine summaries that succeeded.
        """
        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

        async def one(community: Community) -> CommunitySummary:
            async with semaphore:
                return await self.summarize(community, members_by_entity, now=now)

        return list(await asyncio.gather(*(one(c) for c in result.communities)))

    async def summarize(
        self,
        community: Community,
        members_by_entity: Mapping[str, CommunityMember],
        *,
        now: datetime | None = None,
    ) -> CommunitySummary:
        """Summarise one community, or explain why it was not summarised."""
        cached = self._cache.get(community.community_id)
        if cached is not None:
            return cached

        members = tuple(
            members_by_entity[entity_id]
            for entity_id in community.members
            if entity_id in members_by_entity
        )

        if not members:
            # Every member id is unknown to the caller's metadata map. That is a
            # wiring bug, not a weak cluster, and saying so distinguishes it from
            # the conductance skip below -- which otherwise look identical in a
            # log.
            return self._skip(
                community, members, "no member metadata was supplied for this community"
            )

        if community.conductance > self.min_conductance_to_skip:
            return self._skip(
                community,
                members,
                f"conductance {community.conductance:.2f} exceeds "
                f"{self.min_conductance_to_skip}: most of this group's connections "
                "point outside it, so there is no theme to summarise",
            )

        request = SummaryRequest(
            community=community,
            members=members,
            representatives=select_representatives(members, self.max_entities_in_prompt),
        )

        try:
            raw = await self.writer(render_community_prompt(request))
        except Exception as error:  # noqa: BLE001 -- one failure must not lose the batch
            _log.warning(
                "graphrag.community.summary_failed",
                community_id=community.community_id,
                error=type(error).__name__,
            )
            return self._skip(community, members, f"summary call failed: {error}")

        title, body = _parse_summary(raw)
        if not body:
            return self._skip(community, members, "the summariser returned nothing")

        summary = CommunitySummary(
            community_id=community.community_id,
            title=title,
            summary=body,
            entity_ids=tuple(member.entity_id for member in members),
            entity_names=tuple(member.name for member in members),
            signal_ids=_dedupe(
                signal_id for member in members for signal_id in member.signal_ids
            ),
            size=community.size,
            conductance=community.conductance,
            generated_at=now,
        )
        self._cache[community.community_id] = summary
        return summary

    def _skip(
        self,
        community: Community,
        members: Sequence[CommunityMember],
        reason: str,
    ) -> CommunitySummary:
        """Record a deliberate non-summary. Not cached.

        Deliberately not cached: a skip is usually caused by something outside
        the community -- missing metadata, a provider outage -- and caching it
        would mean the cluster is never summarised for the lifetime of the
        process, long after the cause was fixed.
        """
        return CommunitySummary(
            community_id=community.community_id,
            title="",
            summary="",
            entity_ids=tuple(member.entity_id for member in members),
            entity_names=tuple(member.name for member in members),
            signal_ids=(),
            size=community.size,
            conductance=community.conductance,
            skipped_reason=reason,
        )

    def prime(self, summaries: Iterable[CommunitySummary]) -> None:
        """Load previously-generated summaries into the cache.

        How a worker avoids re-summarising the entire graph on restart. Only
        written summaries are admitted -- priming with a skip would make the skip
        permanent, which is the failure `_skip` avoids by not caching.
        """
        for summary in summaries:
            if summary.is_written:
                self._cache[summary.community_id] = summary


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    """Order-preserving dedupe. Signal ids repeat across a community's members."""
    seen: dict[str, None] = {}
    for value in values:
        seen.setdefault(value, None)
    return tuple(seen)


def members_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    id_key: str = "id",
    name_key: str = "name",
    type_key: str = "type",
    description_key: str = "description",
    importance_key: str = "pagerank_score",
) -> dict[str, CommunityMember]:
    """Build the metadata map from `services/graph_service.py` entity rows.

    Keyed by entity id because that is how `Community.members` refers to them,
    and building the index here rather than at each call site means one place
    knows that the join key is `id` and not `entity_id`.
    """
    members: dict[str, CommunityMember] = {}
    for row in rows:
        entity_id = row.get(id_key)
        if not isinstance(entity_id, str) or not entity_id:
            continue
        importance = row.get(importance_key)
        members[entity_id] = CommunityMember(
            entity_id=entity_id,
            name=str(row.get(name_key) or entity_id),
            entity_type=str(row.get(type_key) or "Unknown"),
            description=row.get(description_key)
            if isinstance(row.get(description_key), str)
            else None,
            importance=(
                float(importance)
                if isinstance(importance, (int, float)) and not isinstance(importance, bool)
                else 0.0
            ),
            signal_ids=tuple(
                item for item in (row.get("signal_ids") or ()) if isinstance(item, str)
            ),
        )
    return members
