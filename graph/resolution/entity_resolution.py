"""Clustering and merging: turn scored pairs into canonical entities, reversibly.

This is the module that changes the graph. Blocking proposes, the matcher scores,
and here the decision is taken and made durable. Two properties matter more than
accuracy, and both are structural rather than statistical -- a resolver can be
tuned into accuracy later, but it cannot be retrofitted into either of these.

**1. Merges are reversible.**

The absorbed entity is *never deleted*. It survives as a snapshot inside a
`MergeRecord` and as a `SAME_AS` edge pointing at the survivor
(`models/entity.py`, `docs/knowledge-graph.md` §6). Two things depend on that:

- Historical references stay resolvable. A report written in March cites
  `ent_acme_analytics`; if resolution absorbs that node in April, the citation
  must still lead somewhere. The `SAME_AS` edge is the redirect.
- A wrong merge is corrected by `unmerge()`, not by re-ingesting the corpus.
  Re-ingesting is not actually a repair: extraction is non-deterministic, months
  of raw payloads may be past their retention window, and the merge would
  frequently be recreated on the way back in. Hand-editing the graph is worse --
  the next ingest silently undoes it. The only durable correction is a recorded
  constraint, which is what `unmerge()` emits.

Reversal restores from **full pre-merge snapshots**, not from deltas. A merge
unions alias lists, takes a min and a max of timestamps and sums counters; the
union and the min/max are not invertible. Subtracting one alias list from
another cannot know which of two identical aliases each side contributed, so a
delta-based un-merge quietly loses aliases every round trip. Snapshots cost bytes
in an audit node; deltas cost correctness.

**2. Resolution is deterministic.**

Two workers consuming the same partition of `KAFKA_TOPIC_GRAPH_UPDATES` must
build the same graph. If they do not, the difference is permanent -- each writes
its own canonical nodes, and the divergence compounds with every later pass that
resolves against them. Determinism here means: identical inputs produce
byte-identical output, *including which id survives*, and independent of the
order the inputs arrived in. Every ordering decision in this module is therefore
made against a total order over ids rather than against iteration order:

- Input records are sorted by id before anything reads them.
- Candidate pairs come back sorted from `BlockingIndex.candidate_pairs()`.
- Agglomerative merge steps break linkage ties on the sorted member ids.
- Survivor election ends in a lexicographic id comparison, which cannot tie.
- Timestamps are injected (`decided_at`), never read from the clock inside the
  algorithm. A resolver that stamps `utcnow()` on its own audit records produces
  different records for identical inputs and cannot be replayed or diffed.

Clustering is **constrained connected components** (§6). Pairwise decisions are
not transitive: A~B at 0.95 and B~C at 0.95 says nothing about A~C, and plain
connected components will happily chain a hundred entities into one node through
a path of individually-defensible links. Components are therefore re-derived by
average-linkage agglomerative clustering with a floor, so a component only holds
together if its members agree on average -- not merely in a chain.

Everything here is pure and synchronous: records in, plan out. No Neo4j session,
no Kafka producer, no clock. `graph/ingest/writer.py` performs the writes;
`resolve()` only decides. That split is what lets the whole resolution path be
tested with no services running, and it is why this module returns
`SameAsEdge`/`MergeRecord` value objects rather than executing Cypher.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final

from graph.resolution.blocking import (
    DEFAULT_MAX_BLOCK_SIZE,
    BlockingIndex,
    BlockingStats,
    ResolutionRecord,
    pair_key,
)
from graph.resolution.matcher import (
    MatchDecision,
    MatchScore,
    MatchScorer,
)
from models.base import utcnow
from models.entity import Entity
from models.enums import EdgeType, EntityType

__all__ = [
    "DEFAULT_DECIDED_BY",
    "MAX_REFINEMENT_COMPONENT",
    "MIN_WITHIN_CLUSTER_LINKAGE",
    "MergeRecord",
    "ResolutionResult",
    "ResolvedCluster",
    "ReviewItem",
    "SameAsEdge",
    "UnmergeResult",
    "elect_survivor",
    "refine_component",
    "resolve",
    "resolve_async",
    "unmerge",
]


MIN_WITHIN_CLUSTER_LINKAGE: Final = 0.60
"""Average linkage below which a provisional component is split (§6).

Well under `REVIEW_THRESHOLD` on purpose. This number does not decide whether a
*pair* matches -- the matcher already did that, and every link inside a component
scored at least the auto-merge threshold. It decides whether a *chain* of such
links describes one thing. Setting it near the merge threshold would shatter
legitimate clusters whose members are related through a hub (three writings of a
company name that all match the fourth strongly and each other weakly); setting
it to zero restores plain connected components and with them the chaining bug
this exists to prevent.
"""

MAX_REFINEMENT_COMPONENT: Final = 64
"""Component size above which refinement degrades to connected components.

Refinement is O(k^2) in scores and O(k^3) in linkage recomputation. A component
of 64 is already ~2,000 comparisons; one of 10,000 -- which happens when a
popular generic name pulls half the corpus into one component -- would stall the
worker indefinitely. Oversized components are kept whole and reported in
`ResolutionResult.unrefined_components` so the compromise is visible; they are
the components most likely to contain a bad chain, and a run that produces one
deserves a human look rather than a silent best effort.
"""

DEFAULT_DECIDED_BY: Final = "graph.resolution/v1"
"""Author stamped on merge records and `SAME_AS` edges.

Versioned, because a merge is only explicable against the weights and thresholds
that produced it. When `MATCH_WEIGHTS` changes, this string changes, and a
reviewer looking at an old merge can tell it was decided under different rules.
"""


# --------------------------------------------------------------------------- #
# Value objects -- the plan handed to graph/ingest/writer.py
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SameAsEdge:
    """`(:Entity)-[:SAME_AS]->(:Entity)`: the absorbed id's forwarding address.

    Directed loser -> survivor, which is the direction a lookup travels: given a
    stale id from a citation, follow `SAME_AS` outward to find the node that
    holds the data now. Storing it the other way would make that lookup a scan of
    every merge ever performed.
    """

    from_id: str
    to_id: str
    merge_id: str
    confidence: float
    observed_at: datetime
    score: Mapping[str, object]
    decided_by: str
    edge_type: EdgeType = EdgeType.SAME_AS

    @property
    def edge_key(self) -> str:
        """Deterministic `MERGE` key for the writer (`docs/knowledge-graph.md` §3).

        Derived from the endpoints and the merge that created the edge, so
        replaying the same resolution batch -- which at-least-once Kafka delivery
        guarantees will happen -- upserts the same edge instead of creating a
        second one.
        """
        material = f"{self.edge_type.value}|{self.from_id}|{self.to_id}|{self.merge_id}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def as_dict(self) -> dict[str, object]:
        """Row shape for the writer's `UNWIND $rows` edge upsert."""
        return {
            "edge_key": self.edge_key,
            "type": self.edge_type.value,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "merge_id": self.merge_id,
            "confidence": round(self.confidence, 6),
            "observed_at": self.observed_at,
            "valid_from": self.observed_at,
            "valid_to": None,
            "extractor": self.decided_by,
            "score": dict(self.score),
        }


@dataclass(frozen=True, slots=True)
class MergeRecord:
    """The audit node for one merge, and the only thing `unmerge()` needs.

    Self-sufficient by design: it carries the *complete* pre-merge state of every
    member, so a reversal months later does not depend on the graph still holding
    anything, on the matcher still scoring the pair the same way, or on the raw
    signals still existing. An audit record that needs the rest of the system to
    be unchanged in order to be usable is not an audit record.
    """

    id: str
    canonical_id: str
    absorbed_ids: tuple[str, ...]
    snapshots: tuple[ResolutionRecord, ...]
    scores: tuple[MatchScore, ...]
    confidence: float
    decided_by: str
    decided_at: datetime

    @property
    def member_ids(self) -> tuple[str, ...]:
        """Canonical id first, then absorbed ids in sorted order."""
        return (self.canonical_id, *self.absorbed_ids)

    def snapshot_for(self, record_id: str) -> ResolutionRecord:
        """The pre-merge record for `record_id`.

        Raises `KeyError` rather than returning `None`: a merge record that
        cannot produce a snapshot for one of its own members is corrupt, and
        continuing would write a half-restored entity into the graph.
        """
        for snapshot in self.snapshots:
            if snapshot.id == record_id:
                return snapshot
        raise KeyError(f"merge {self.id} holds no snapshot for {record_id!r}")

    def as_dict(self) -> dict[str, object]:
        """Row shape for the `(:_MergeRecord)` audit node (§6, step 4)."""
        return {
            "id": self.id,
            "canonical_id": self.canonical_id,
            "absorbed_ids": list(self.absorbed_ids),
            "confidence": round(self.confidence, 6),
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "scores": [score.as_dict() for score in self.scores],
            "snapshots": [_snapshot_row(snapshot) for snapshot in self.snapshots],
        }


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """A pair the resolver refused to decide, queued for a human (§6).

    Carries the whole `MatchScore` rather than a score and a message, because the
    reviewer's first question is always "which feature drove this" and the second
    is "what was missing". Both are answerable only from the breakdown.
    """

    left_id: str
    right_id: str
    score: MatchScore
    reason: str

    @property
    def pair(self) -> tuple[str, str]:
        return pair_key(self.left_id, self.right_id)


@dataclass(frozen=True, slots=True)
class ResolvedCluster:
    """One canonical entity and the records that folded into it."""

    survivor_id: str
    member_ids: tuple[str, ...]
    canonical: ResolutionRecord
    weakest_link: float | None
    merge: MergeRecord | None

    @property
    def is_merge(self) -> bool:
        """Whether this cluster absorbed anything. Singletons are not merges."""
        return len(self.member_ids) > 1

    def to_entity(self) -> Entity:
        """The canonical record as a graph `Entity`, confidence included."""
        entity = self.canonical.to_entity()
        if self.weakest_link is not None:
            entity.resolution_confidence = min(1.0, max(0.0, self.weakest_link))
        return entity


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """Everything one resolution pass decided. Inputs are never mutated.

    A plan, not an effect. `graph/ingest/writer.py` applies it in the order the
    fields are declared: canonical nodes, then `SAME_AS` edges, then audit
    records -- nodes before edges, as §7 requires, or the edge `MATCH` misses.
    """

    clusters: tuple[ResolvedCluster, ...]
    merges: tuple[MergeRecord, ...]
    same_as_edges: tuple[SameAsEdge, ...]
    review_items: tuple[ReviewItem, ...]
    blocking: BlockingStats
    comparisons: int
    unrefined_components: tuple[tuple[str, ...], ...] = ()

    def canonical_id_for(self, record_id: str) -> str | None:
        """Which surviving entity `record_id` now resolves to, or `None`.

        The in-memory equivalent of following a `SAME_AS` edge, so a caller
        holding pre-resolution ids (an `EntityMention.resolved_id`, a citation in
        a draft report) can rewrite them without a graph round trip.
        """
        for cluster in self.clusters:
            if record_id in cluster.member_ids:
                return cluster.survivor_id
        return None

    def entities(self) -> tuple[Entity, ...]:
        """Every surviving entity, ordered by id."""
        return tuple(cluster.to_entity() for cluster in self.clusters)

    @property
    def merged_ids(self) -> frozenset[str]:
        """Ids absorbed by this pass. Empty when nothing merged."""
        return frozenset(rid for merge in self.merges for rid in merge.absorbed_ids)


@dataclass(frozen=True, slots=True)
class UnmergeResult:
    """The plan that reverses a merge, in whole or in part.

    `must_not_link` is the load-bearing field. Restoring the nodes without
    recording the constraint fixes the graph until the next resolution pass,
    which sees the same records, computes the same scores and merges them again
    -- the correction survives about as long as it takes the worker to catch up.
    `docs/knowledge-graph.md` §6 is explicit that the constraint *is* the
    correction; the node restoration is the visible part of it.
    """

    merge_id: str
    restored: tuple[ResolutionRecord, ...]
    canonical: ResolutionRecord | None
    retracted_edges: tuple[tuple[str, str], ...]
    must_not_link: tuple[tuple[str, str], ...]
    decided_by: str
    decided_at: datetime


def _snapshot_row(record: ResolutionRecord) -> dict[str, object]:
    """Serialize a record for the audit node. Sorted throughout.

    Sets and maps are written in sorted order so that two audit records for the
    same merge are byte-identical, which is what makes them comparable across
    workers and diffable in review.
    """
    return {
        "id": record.id,
        "type": record.type.value,
        "name": record.name,
        "aliases": list(record.aliases),
        "identifiers": dict(sorted(record.identifiers.items())),
        "context": sorted(record.context),
        "first_seen": record.first_seen,
        "last_seen": record.last_seen,
        "mention_count": record.mention_count,
        "merged_from": list(record.merged_from),
        "has_embedding": record.embedding is not None,
    }


# --------------------------------------------------------------------------- #
# Survivor election
# --------------------------------------------------------------------------- #


def elect_survivor(records: Sequence[ResolutionRecord]) -> ResolutionRecord:
    """Pick the record whose id survives a merge. Total, and stable.

    Ordered by **(earliest `first_seen`, then highest mention count, then
    lexicographically smallest id)**, and the order of those three is the whole
    point:

    1. **Earliest `first_seen` first.** The oldest id is the one the rest of the
       system has had the longest to reference -- in stored reports, in
       `EntityMention.resolved_id` on months of signals, in URLs somebody
       bookmarked. Electing it means those references keep pointing at a live
       node and the `SAME_AS` redirects that get created are the *fewest
       possible*. Every other criterion optimizes something internal to
       resolution; this one optimizes for everything outside it.

    2. **Highest mention count second.** It is a real quality signal -- the
       better-evidenced record usually carries the better name and aliases -- but
       it is *volatile*: it increases every time a signal arrives, so two workers
       resolving the same pair moments apart can see different counts and elect
       different survivors. A volatile criterion must never outrank a stable one,
       or determinism holds only within a single instant.

    3. **Lexicographically smallest id last.** Carries no meaning at all, which
       is exactly why it belongs at the bottom: it exists solely to guarantee the
       order is total. Ids are unique, so this can never tie, and no pair of
       inputs can reach the end of this function undecided.

    A record with no `first_seen` sorts *after* every record that has one. An
    absent timestamp is unknown, not infinitely old, and treating it as old would
    let an unstamped mention displace an established node with real history.

    **This deliberately differs from `docs/knowledge-graph.md` §6**, which orders
    the first two the other way round (highest `source_count`, tie-broken by
    earliest `first_seen`). That ordering makes the survivor depend on a counter
    that changes under concurrent ingestion, which the doc does not consider. The
    divergence is intentional and should be reconciled in the doc.
    """
    if not records:
        raise ValueError("cannot elect a survivor from an empty cluster")
    return min(records, key=_survivor_sort_key)


def _survivor_sort_key(record: ResolutionRecord) -> tuple[int, float, int, str]:
    """Sort key implementing the election order. Smaller wins."""
    if record.first_seen is None:
        seen_rank, seen_value = 1, 0.0
    else:
        seen_rank, seen_value = 0, record.first_seen.timestamp()
    return (seen_rank, seen_value, -record.mention_count, record.id)


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #


class _UnionFind:
    """Disjoint sets over record ids, with union by size and path compression.

    Deliberately not exposed: connected components are only the *provisional*
    clustering here, and a caller reaching for them directly would skip the
    linkage refinement that stops A-B-C-...-Z chaining into one entity.
    """

    __slots__ = ("_parent", "_size")

    def __init__(self, ids: Iterable[str]) -> None:
        self._parent = {rid: rid for rid in ids}
        self._size = dict.fromkeys(self._parent, 1)

    def find(self, rid: str) -> str:
        root = rid
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[rid] != root:
            self._parent[rid], rid = root, self._parent[rid]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        # Attach the smaller tree under the larger; on a tie attach the
        # lexicographically larger root under the smaller one so the structure
        # -- and therefore nothing observable -- depends on insertion order.
        left_size, right_size = self._size[left_root], self._size[right_root]
        if left_size < right_size or (left_size == right_size and left_root > right_root):
            left_root, right_root = right_root, left_root
        self._parent[right_root] = left_root
        self._size[left_root] += self._size[right_root]

    def components(self) -> tuple[tuple[str, ...], ...]:
        """Components, each sorted by id, ordered by their smallest member."""
        grouped: dict[str, list[str]] = {}
        for rid in self._parent:
            grouped.setdefault(self.find(rid), []).append(rid)
        return tuple(sorted(tuple(sorted(members)) for members in grouped.values()))


def refine_component(
    members: Sequence[str],
    similarity: Mapping[tuple[str, str], float],
    floor: float = MIN_WITHIN_CLUSTER_LINKAGE,
) -> tuple[tuple[str, ...], ...]:
    """Split one component by average-linkage agglomerative clustering.

    Starts from singletons and repeatedly joins the two sub-clusters with the
    highest average linkage, stopping when the best remaining linkage falls below
    `floor`. That reproduces the rule in `docs/knowledge-graph.md` §6 -- a
    component holds together only while its parts agree *on average* -- and it is
    what breaks the classic transitivity failure: A~B 0.95, B~C 0.95, A~C 0.10
    yields `{A, B}` and `{C}` rather than one three-member entity, because after
    A and B join, their average linkage to C is 0.525.

    Determinism comes from the tie-break, not from luck. Ties on the linkage
    value are common (a component of identical names produces identical scores),
    and `max()` over a dict would resolve them by iteration order. Candidates are
    therefore ranked by `(-linkage, sorted member ids)`, which is a total order
    over pairs of sub-clusters and independent of how the component was built.
    """
    clusters: list[tuple[str, ...]] = [(member,) for member in sorted(members)]
    if len(clusters) < 2:
        return tuple(clusters)

    while len(clusters) > 1:
        best_key: tuple[float, tuple[str, ...]] | None = None
        best_pair: tuple[int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                linkage = _average_linkage(clusters[i], clusters[j], similarity)
                if linkage < floor:
                    continue
                # Rounded before comparison so that two mathematically equal
                # linkages that differ in the last float bit -- summation order
                # over a set of scores is not associative -- do not resolve the
                # tie differently on different machines.
                key = (-round(linkage, 9), tuple(sorted(clusters[i] + clusters[j])))
                if best_key is None or key < best_key:
                    best_key, best_pair = key, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        joined = tuple(sorted(clusters[i] + clusters[j]))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(joined)
        clusters.sort()

    return tuple(sorted(clusters))


def _average_linkage(
    left: Sequence[str],
    right: Sequence[str],
    similarity: Mapping[tuple[str, str], float],
) -> float:
    """Mean similarity across the cross product of two sub-clusters.

    Average rather than single linkage, because single linkage is precisely the
    chaining behaviour being prevented -- it would keep any pair connected
    through one strong link no matter how badly the rest disagreed. Average
    rather than complete linkage, because complete linkage lets one outlier
    veto an otherwise coherent cluster, and outliers are guaranteed here: every
    component contains at least one record whose name is a typo.

    Pairs absent from `similarity` count as 0.0. Absent means blocking never
    proposed them *and* refinement did not compute them, which happens only for
    components too large to refine -- where the missing evidence genuinely is
    evidence of nothing.
    """
    total = 0.0
    for a in left:
        for b in right:
            total += similarity.get(pair_key(a, b), 0.0)
    return total / (len(left) * len(right))


# --------------------------------------------------------------------------- #
# Merging
# --------------------------------------------------------------------------- #


def _merge_into(
    survivor: ResolutionRecord, absorbed: Sequence[ResolutionRecord]
) -> ResolutionRecord:
    """Fold `absorbed` into `survivor`, producing a new record.

    Field by field, and why each rule is what it is:

    `name`
        The survivor's, unchanged. The absorbed names are not lost -- they become
        aliases, which is what keeps the absorbed spelling searchable and what
        lets the next pass block on it.
    `aliases`
        Union of every surface from every member, minus the surviving canonical
        name, sorted. Sorted rather than "survivor's first, then the rest":
        alias order is not information, and any order derived from input order
        would make the merged record differ between two workers that received
        the same records in different sequences.
    `identifiers`
        Union, with the survivor winning a collision. Conflicting strong
        identifiers cannot normally reach here -- the matcher treats them as a
        hard non-match -- but a `must_link` override from a human review can
        force the pair through, and in that case the human asserted identity, not
        that the ticker on the absorbed record was right.
    `embedding`
        The survivor's, or the first available by id order. Not averaged:
        averaging two vectors from different models is meaningless, and averaging
        two from the same model produces a vector that no longer corresponds to
        any text -- it drifts a little further from reality with every merge.
        Recomputation is the writer's job (`docs/knowledge-graph.md`, open
        question 7).
    `first_seen` / `last_seen`
        Min and max over the members that have one. The merged entity was
        observed across the union of the members' windows.
    `mention_count`
        Sum. `source_count` is documented as "distinct signals that evidenced
        it", and the members evidenced it separately.
    `merged_from`
        Transitive: the absorbed ids plus everything they had already absorbed,
        so a chain of merges leaves one flat, complete list. Without the
        transitive part, un-merging a two-step merge would restore a node whose
        own absorbed ids had silently vanished from the graph's redirect table.
    `type`
        The survivor's, unless it is `UNKNOWN` and a member carries a real label.
        A degraded mention that resolves into a typed entity should adopt the
        type, not impose its own ignorance on it.
    """
    members = [survivor, *absorbed]

    surviving_surface = survivor.name.strip()
    alias_pool = {
        surface
        for member in members
        for surface in (member.name, *member.aliases)
        if surface.strip() and surface.strip() != surviving_surface
    }

    identifiers: dict[str, str] = {}
    for member in reversed(members):  # survivor applied last, so it wins
        identifiers.update(member.identifiers)

    embedding = survivor.embedding
    if embedding is None:
        for member in sorted(absorbed, key=lambda record: record.id):
            if member.embedding is not None:
                embedding = member.embedding
                break

    first_seen_values = [m.first_seen for m in members if m.first_seen is not None]
    last_seen_values = [m.last_seen for m in members if m.last_seen is not None]

    entity_type = survivor.type
    if entity_type is EntityType.UNKNOWN:
        for member in sorted(absorbed, key=lambda record: record.id):
            if member.type is not EntityType.UNKNOWN:
                entity_type = member.type
                break

    absorbed_ids = {member.id for member in absorbed}
    inherited = {rid for member in members for rid in member.merged_from}
    merged_from = sorted((absorbed_ids | inherited) - {survivor.id})

    return replace(
        survivor,
        type=entity_type,
        aliases=tuple(sorted(alias_pool)),
        identifiers=identifiers,
        embedding=embedding,
        context=frozenset().union(*(m.context for m in members)) if members else frozenset(),
        first_seen=min(first_seen_values) if first_seen_values else None,
        last_seen=max(last_seen_values) if last_seen_values else None,
        mention_count=sum(m.mention_count for m in members),
        merged_from=tuple(merged_from),
    )


def _merge_id(canonical_id: str, member_ids: Sequence[str], decided_at: datetime) -> str:
    """A content-addressed id for one merge decision.

    Deterministic so that replaying a batch -- which at-least-once delivery
    guarantees -- upserts the same `(:_MergeRecord)` instead of accumulating one
    audit node per replay. `decided_at` is part of the material because the same
    cluster legitimately merges again after an un-merge and a re-decision, and
    those two decisions must be distinguishable in the audit trail.
    """
    material = "|".join([canonical_id, *sorted(member_ids), decided_at.isoformat()])
    return f"mrg_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


# --------------------------------------------------------------------------- #
# The pass
# --------------------------------------------------------------------------- #


def resolve(
    records: Iterable[ResolutionRecord],
    *,
    scorer: MatchScorer | None = None,
    index: BlockingIndex | None = None,
    must_link: Iterable[tuple[str, str]] = (),
    must_not_link: Iterable[tuple[str, str]] = (),
    decided_at: datetime | None = None,
    decided_by: str = DEFAULT_DECIDED_BY,
    min_within_cluster_linkage: float = MIN_WITHIN_CLUSTER_LINKAGE,
    max_block_size: int | None = DEFAULT_MAX_BLOCK_SIZE,
) -> ResolutionResult:
    """Resolve a batch of records into canonical entities. Pure; nothing is written.

    The pass, in order:

    1. Sort by id and reject duplicates. Sorting first is what makes every later
       step order-independent; two records sharing an id is a caller bug that
       would otherwise surface as a silently dropped record.
    2. Block (`BlockingIndex`) to get candidate pairs.
    3. Score each pair, honouring `must_link` / `must_not_link` overrides.
    4. Union-find over accepted pairs for provisional components.
    5. Refine each component by average linkage, computing the intra-component
       pairs blocking never proposed -- components are small, so exactness is
       affordable there even though it is not globally.
    6. Elect a survivor per cluster, merge, and emit `SAME_AS` edges plus an
       audit record.

    `decided_at` defaults to `utcnow()`, which is the only non-deterministic
    input to this function and is deliberately a parameter: a caller that needs
    reproducible output (a test, a replay, a differential run against a second
    worker) passes a fixed timestamp and gets byte-identical results.

    Blocking is the only recall risk in the pass, so `ResolutionResult.blocking`
    is returned alongside the decisions rather than logged and forgotten.
    """
    ordered = sorted(records, key=lambda record: record.id)
    by_id: dict[str, ResolutionRecord] = {}
    for record in ordered:
        if record.id in by_id:
            raise ValueError(
                f"duplicate record id {record.id!r} in one resolution batch; "
                "ids must be unique or the survivor election is ambiguous"
            )
        by_id[record.id] = record

    decided = decided_at or utcnow()
    scorer = scorer or MatchScorer()
    forbidden = frozenset(pair_key(a, b) for a, b in must_not_link)
    required = frozenset(pair_key(a, b) for a, b in must_link) - forbidden

    if index is None:
        index = BlockingIndex(max_block_size=max_block_size)
        index.add_all(ordered)

    similarity: dict[tuple[str, str], float] = {}
    accepted: list[tuple[str, str]] = []
    review: list[ReviewItem] = []
    scores_by_pair: dict[tuple[str, str], MatchScore] = {}

    for left_id, right_id in index.candidate_pairs():
        pair = pair_key(left_id, right_id)
        if pair[0] not in by_id or pair[1] not in by_id:
            # The index outlives the batch in the streaming case: it may hold
            # entities that are not being re-resolved right now. Their pairs are
            # somebody else's decision.
            continue
        if pair in forbidden:
            similarity[pair] = 0.0
            continue
        score = scorer.score(
            by_id[pair[0]],
            by_id[pair[1]],
            shared_keys=index.shared_keys(pair[0], pair[1]),
        )
        scores_by_pair[pair] = score
        similarity[pair] = score.combined
        if pair in required or score.decision is MatchDecision.MERGE:
            accepted.append(pair)
        elif score.decision is MatchDecision.REVIEW:
            review.append(
                ReviewItem(pair[0], pair[1], score, score.applied_rule or "score_band")
            )

    # A must_link override applies even when blocking never proposed the pair --
    # a human who says two entities are the same has information the index does
    # not, and requiring them to also fix the blocking keys first would make the
    # override useless in exactly the cases it exists for.
    for pair in sorted(required):
        if pair in similarity or pair[0] not in by_id or pair[1] not in by_id:
            continue
        similarity[pair] = 1.0
        accepted.append(pair)

    union = _UnionFind(by_id)
    for left_id, right_id in accepted:
        union.union(left_id, right_id)

    clusters: list[ResolvedCluster] = []
    merges: list[MergeRecord] = []
    edges: list[SameAsEdge] = []
    unrefined: list[tuple[str, ...]] = []

    for component in union.components():
        if len(component) > MAX_REFINEMENT_COMPONENT:
            unrefined.append(component)
            refined: tuple[tuple[str, ...], ...] = (component,)
        else:
            _score_missing_pairs(component, by_id, forbidden, scorer, similarity)
            refined = refine_component(component, similarity, min_within_cluster_linkage)
            # Refinement is a check on the *automatic* path. A `must_link` is a
            # human assertion of identity, and letting an average-linkage
            # calculation overrule it would mean a reviewer's decision silently
            # fails to stick -- they would adjudicate the same pair every pass.
            refined = _apply_required_links(refined, required)

        split = len(refined) > 1
        for cluster_ids in refined:
            cluster, merge, cluster_edges = _build_cluster(
                cluster_ids,
                by_id,
                similarity,
                scores_by_pair,
                decided_at=decided,
                decided_by=decided_by,
            )
            clusters.append(cluster)
            if merge is not None:
                merges.append(merge)
                edges.extend(cluster_edges)
        if split:
            review.extend(
                _split_review_items(component, refined, scores_by_pair)
            )

    clusters.sort(key=lambda cluster: cluster.survivor_id)
    review.sort(key=lambda item: (item.pair, item.reason))

    return ResolutionResult(
        clusters=tuple(clusters),
        merges=tuple(sorted(merges, key=lambda merge: merge.canonical_id)),
        same_as_edges=tuple(sorted(edges, key=lambda edge: (edge.to_id, edge.from_id))),
        review_items=tuple(_dedupe_reviews(review)),
        blocking=index.stats(),
        comparisons=scorer.comparisons,
        unrefined_components=tuple(unrefined),
    )


def _apply_required_links(
    refined: Sequence[Sequence[str]],
    required: frozenset[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    """Re-join refined sub-clusters that a `must_link` override spans.

    Runs after refinement rather than before it because a required pair must
    survive the linkage floor, not merely enter it: joining the pair up front
    still leaves the agglomerative pass free to split the result apart again on
    the strength of the members' other scores.
    """
    if not required:
        return tuple(tuple(cluster) for cluster in refined)

    owner = {rid: i for i, cluster in enumerate(refined) for rid in cluster}
    union = _UnionFind(str(i) for i in range(len(refined)))
    for left_id, right_id in sorted(required):
        left_cluster, right_cluster = owner.get(left_id), owner.get(right_id)
        if left_cluster is None or right_cluster is None:
            continue
        union.union(str(left_cluster), str(right_cluster))

    grouped: dict[str, list[str]] = {}
    for index_str in (str(i) for i in range(len(refined))):
        grouped.setdefault(union.find(index_str), []).extend(refined[int(index_str)])
    return tuple(sorted(tuple(sorted(members)) for members in grouped.values()))


def _score_missing_pairs(
    component: Sequence[str],
    by_id: Mapping[str, ResolutionRecord],
    forbidden: frozenset[tuple[str, str]],
    scorer: MatchScorer,
    similarity: dict[tuple[str, str], float],
) -> None:
    """Fill in intra-component pairs blocking never proposed.

    Refinement asks "do these members agree on average", and a pair with no
    recorded similarity counts as 0.0 in that average. For a pair blocking never
    looked at, 0.0 is a guess, not a measurement -- and it is the guess that
    splits clusters most aggressively, because two spellings that fail to share
    any blocking key are exactly the ones a third record is holding together.
    Components are bounded by `MAX_REFINEMENT_COMPONENT`, so measuring them
    properly costs at most a few thousand comparisons.
    """
    for i, left_id in enumerate(component):
        for right_id in component[i + 1 :]:
            pair = pair_key(left_id, right_id)
            if pair in similarity:
                continue
            if pair in forbidden:
                similarity[pair] = 0.0
                continue
            similarity[pair] = scorer.score(by_id[left_id], by_id[right_id]).combined


def _build_cluster(
    cluster_ids: Sequence[str],
    by_id: Mapping[str, ResolutionRecord],
    similarity: Mapping[tuple[str, str], float],
    scores_by_pair: Mapping[tuple[str, str], MatchScore],
    *,
    decided_at: datetime,
    decided_by: str,
) -> tuple[ResolvedCluster, MergeRecord | None, tuple[SameAsEdge, ...]]:
    """Elect, merge and record one cluster. Singletons short-circuit.

    The cluster's confidence is its **weakest internal link**, not the mean. A
    cluster is a claim that all of its members are one thing, and that claim is
    only as good as its least convincing pair -- averaging lets two excellent
    links hide a marginal third, which is the merge most likely to be wrong and
    the one a reviewer most needs to see ranked low.
    """
    members = [by_id[rid] for rid in cluster_ids]
    survivor = elect_survivor(members)
    absorbed = sorted(
        (record for record in members if record.id != survivor.id),
        key=lambda record: record.id,
    )

    if not absorbed:
        return (
            ResolvedCluster(survivor.id, (survivor.id,), survivor, None, None),
            None,
            (),
        )

    internal = [
        similarity.get(pair_key(a, b), 0.0)
        for i, a in enumerate(cluster_ids)
        for b in cluster_ids[i + 1 :]
    ]
    weakest = min(internal) if internal else 0.0
    merged = _merge_into(survivor, absorbed)
    member_ids = tuple(sorted(cluster_ids))
    merge_id = _merge_id(survivor.id, member_ids, decided_at)

    cluster_scores = tuple(
        scores_by_pair[pair_key(a, b)]
        for i, a in enumerate(cluster_ids)
        for b in cluster_ids[i + 1 :]
        if pair_key(a, b) in scores_by_pair
    )

    merge = MergeRecord(
        id=merge_id,
        canonical_id=survivor.id,
        absorbed_ids=tuple(record.id for record in absorbed),
        snapshots=tuple(sorted(members, key=lambda record: record.id)),
        scores=cluster_scores,
        confidence=weakest,
        decided_by=decided_by,
        decided_at=decided_at,
    )

    edges = tuple(
        SameAsEdge(
            from_id=record.id,
            to_id=survivor.id,
            merge_id=merge_id,
            confidence=similarity.get(pair_key(record.id, survivor.id), weakest),
            observed_at=decided_at,
            score=(
                scores_by_pair[pair_key(record.id, survivor.id)].as_dict()
                if pair_key(record.id, survivor.id) in scores_by_pair
                else {"combined": weakest, "note": "linked transitively within cluster"}
            ),
            decided_by=decided_by,
        )
        for record in absorbed
    )

    return (
        ResolvedCluster(survivor.id, member_ids, merged, weakest, merge),
        merge,
        edges,
    )


def _split_review_items(
    component: Sequence[str],
    refined: Sequence[Sequence[str]],
    scores_by_pair: Mapping[tuple[str, str], MatchScore],
) -> list[ReviewItem]:
    """Queue pairs that scored a merge but were split apart by refinement.

    These are the most interesting pairs in the whole pass: the matcher was
    confident and the cluster structure disagreed. Dropping them silently would
    hide every case where the linkage floor is set wrong, and hiding that is how
    a floor stays wrong for a year.
    """
    assignment = {rid: i for i, cluster in enumerate(refined) for rid in cluster}
    items: list[ReviewItem] = []
    for i, left_id in enumerate(component):
        for right_id in component[i + 1 :]:
            pair = pair_key(left_id, right_id)
            score = scores_by_pair.get(pair)
            if score is None or score.decision is not MatchDecision.MERGE:
                continue
            if assignment.get(left_id) != assignment.get(right_id):
                items.append(ReviewItem(pair[0], pair[1], score, "split_by_linkage"))
    return items


def _dedupe_reviews(items: Sequence[ReviewItem]) -> list[ReviewItem]:
    """One review item per pair, sorted. `split_by_linkage` wins a collision.

    A pair can arrive here twice -- once from the score band, once from the
    refinement split. Two rows for one decision means a reviewer adjudicates the
    same pair twice and the second adjudication silently overwrites the first.
    """
    best: dict[tuple[str, str], ReviewItem] = {}
    for item in sorted(items, key=lambda item: (item.pair, item.reason)):
        existing = best.get(item.pair)
        if existing is None or item.reason == "split_by_linkage":
            best[item.pair] = item
    return [best[pair] for pair in sorted(best)]


async def resolve_async(
    records: Iterable[ResolutionRecord],
    **kwargs: object,
) -> ResolutionResult:
    """Await-friendly `resolve()`, run on a worker thread.

    `resolve()` is deliberately synchronous -- it performs no I/O, so `async def`
    would promise a suspension point that never happens. It is also genuinely
    CPU-bound: a batch of a few thousand records is millions of `rapidfuzz`
    calls, and calling it directly from `workers/graph_worker.py` would block the
    event loop for the whole pass, stalling the Kafka heartbeat and triggering a
    consumer-group rebalance mid-batch. Offloading to a thread is what prevents
    that; `rapidfuzz` releases the GIL for its scoring calls, so the thread does
    real work rather than fighting for the interpreter.
    """
    return await asyncio.to_thread(lambda: resolve(records, **kwargs))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Reversal
# --------------------------------------------------------------------------- #


def unmerge(
    merge: MergeRecord,
    *,
    separate: Iterable[str] | None = None,
    decided_at: datetime | None = None,
    decided_by: str = DEFAULT_DECIDED_BY,
) -> UnmergeResult:
    """Reverse a merge -- entirely, or for named members only.

    Restores from the snapshots in `merge`, so the result is exact rather than
    reconstructed: aliases the survivor already had are not confused with
    aliases it absorbed, and counters return to their pre-merge values instead of
    to a subtraction that assumed nothing else had changed them.

    `separate` names the members to detach; omitting it detaches all of them. The
    partial form matters because a cluster of four is usually wrong about *one*
    member, and forcing a reviewer to explode the whole cluster and re-adjudicate
    every pair is how correction queues stop being used.

    The canonical id is **not** re-elected for the members that remain. Election
    exists to minimise dangling references, and after a merge the canonical id is
    the one everything now points at; re-electing on the way back would invalidate
    exactly the references the original election protected. The remaining members
    are re-folded into the same survivor.

    Returns `must_not_link` pairs between everything detached and everything
    retained. Without them the next `resolve()` sees the same records, computes
    the same scores and rebuilds the merge -- `docs/knowledge-graph.md` §6 is
    explicit that the constraint is the durable half of the correction. Pairs
    *among* the detached members are deliberately absent: a reviewer who says
    "B and C are not A" has said nothing about whether B is C.

    Raises `ValueError` when asked to detach the canonical id (that is a
    re-election, not a reversal) or an id the merge never absorbed.
    """
    decided = decided_at or utcnow()
    absorbed_ids = set(merge.absorbed_ids)
    targets = set(absorbed_ids) if separate is None else set(separate)

    if merge.canonical_id in targets:
        raise ValueError(
            f"cannot detach the canonical id {merge.canonical_id!r} from merge "
            f"{merge.id!r}: separating the survivor is a re-election, which "
            "resolve() performs, not an un-merge"
        )
    unknown = targets - absorbed_ids
    if unknown:
        raise ValueError(
            f"merge {merge.id!r} never absorbed {sorted(unknown)!r}; "
            f"it absorbed {sorted(absorbed_ids)!r}"
        )
    if not targets:
        raise ValueError(f"nothing to detach from merge {merge.id!r}")

    restored = tuple(merge.snapshot_for(rid) for rid in sorted(targets))

    retained_ids = sorted(absorbed_ids - targets)
    survivor_snapshot = merge.snapshot_for(merge.canonical_id)
    if retained_ids:
        canonical = _merge_into(
            survivor_snapshot,
            [merge.snapshot_for(rid) for rid in retained_ids],
        )
    else:
        canonical = survivor_snapshot

    retained_all = [merge.canonical_id, *retained_ids]
    must_not_link = tuple(
        sorted(
            {
                pair_key(target, retained)
                for target in targets
                for retained in retained_all
            }
        )
    )

    return UnmergeResult(
        merge_id=merge.id,
        restored=restored,
        canonical=canonical,
        retracted_edges=tuple(sorted((rid, merge.canonical_id) for rid in targets)),
        must_not_link=must_not_link,
        decided_by=decided_by,
        decided_at=decided,
    )
