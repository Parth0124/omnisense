"""The vocabulary of hybrid retrieval: what is asked for, what comes back, and why.

Every type here is shaped by one requirement from Design Doc §2: reports must be
**evidence-backed with citations and confidence**. That is not a rendering
concern bolted on at the end -- it constrains what retrieval is allowed to return.
A passage that cannot be cited is a bug, not a result, so `Passage` carries the
`signal_id` and character offsets needed to resolve a quote back to stored
content, and `RetrievalResult` carries the provenance of *how* each passage was
found.

The other shaping requirement is that three backends with incomparable score
scales are fused (`docs/retrieval.md` §3). BM25 scores are unbounded and
corpus-dependent, cosine similarities sit in [-1, 1], and graph proximity is a
hop count. Nothing here stores a "score" as though those were one quantity:
`Candidate` keeps each backend's *rank* alongside its raw score, because
reciprocal rank fusion needs ranks and needs no calibration between backends.

`retrieval/` is layer L1 (`docs/architecture.md` §6.1): it imports `models/`, and
reads from `graph/` for traversal, and nothing else.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from models.base import Score, utcnow
from models.enums import Platform, SourceCategory

__all__ = [
    "Backend",
    "Candidate",
    "EvidencePack",
    "GraphFact",
    "Passage",
    "RetrievalDiagnostics",
    "RetrievalRequest",
    "RetrievalResult",
    "chunk_id_for",
    "split_chunk_id",
]


def chunk_id_for(signal_id: str, chunk_index: int) -> str:
    """The join key across OpenSearch, Qdrant and PostgreSQL.

    `{signal_id}:{chunk_index}`. Deterministic so a re-index upserts rather than
    duplicating, and readable so a chunk id in a log line identifies its Signal
    without a lookup. `docs/data-stores.md` §3.5 makes this the OpenSearch `_id`.
    """
    if chunk_index < 0:
        raise ValueError(f"chunk_index must be non-negative, got {chunk_index}")
    return f"{signal_id}:{chunk_index}"


def split_chunk_id(chunk_id: str) -> tuple[str, int]:
    """Inverse of `chunk_id_for`. Raises on a malformed id rather than guessing."""
    signal_id, _, index = chunk_id.rpartition(":")
    if not signal_id or not index.isdigit():
        raise ValueError(f"malformed chunk_id {chunk_id!r}; expected '<signal_id>:<int>'")
    return signal_id, int(index)


class Backend(enum.StrEnum):
    """Which retrieval backend produced a candidate.

    Retained all the way through fusion and into the result. Without it, a
    recall regression is undiagnosable: "results got worse" is not actionable,
    "the graph backend stopped contributing" is.
    """

    KEYWORD = "keyword"
    VECTOR = "vector"
    GRAPH = "graph"


@dataclass(frozen=True, slots=True)
class Filter:
    """A metadata constraint, pushed down into every backend.

    Pushed down rather than applied afterwards, because post-filtering a
    fixed-size ANN result destroys recall: asking Qdrant for 100 neighbours and
    then keeping the 3 that are in-date is not the same as asking for the 100
    nearest in-date neighbours, and the difference is invisible until someone
    notices the last month of data is unreachable.
    """

    published_after: datetime | None = None
    published_before: datetime | None = None
    platforms: frozenset[Platform] = frozenset()
    sources: frozenset[SourceCategory] = frozenset()
    languages: frozenset[str] = frozenset()
    entity_ids: frozenset[str] = frozenset()
    min_confidence: float | None = None
    tenant_id: str = "default"

    def is_empty(self) -> bool:
        return not any(
            (
                self.published_after,
                self.published_before,
                self.platforms,
                self.sources,
                self.languages,
                self.entity_ids,
                self.min_confidence is not None,
            )
        )


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """One retrieval call.

    `seed_entity_ids` is what makes this *hybrid* rather than three searches
    stapled together: the graph backend expands from known entities while the
    lexical and vector backends work from the query text, and fusion reconciles
    them.
    """

    query: str
    filters: Filter = field(default_factory=Filter)
    seed_entity_ids: Sequence[str] = ()

    k_keyword: int = 100
    k_vector: int = 100
    k_graph: int = 50
    graph_depth: int = 2
    graph_fanout_cap: int = 25

    rerank: bool = True
    rerank_depth: int = 50
    k_final: int = 12
    pool_max: int = 150
    token_budget: int = 24_000

    backends: frozenset[Backend] = frozenset(
        {Backend.KEYWORD, Backend.VECTOR, Backend.GRAPH}
    )
    """Which backends to consult.

    Explicit so a degraded run is a *request-level* decision rather than a
    silent catch: `docs/architecture.md` §7.3 says a Qdrant outage degrades to
    keyword-only retrieval, and the caller that made that choice is the one that
    should report reduced confidence.
    """

    def with_backends(self, *backends: Backend) -> RetrievalRequest:
        """A copy restricted to the given backends, for degraded operation."""
        return replace_dataclass(self, backends=frozenset(backends))


@dataclass(frozen=True, slots=True)
class Candidate:
    """One chunk as returned by one backend, before fusion.

    Keeps `rank` *and* `raw_score`. Rank is what reciprocal rank fusion consumes
    and is comparable across backends; raw score is kept only for diagnostics,
    because a BM25 score of 14.2 and a cosine of 0.83 cannot be compared and
    must never be added together.
    """

    chunk_id: str
    backend: Backend
    rank: int
    raw_score: float
    signal_id: str = ""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank is 1-based; got {self.rank}")
        if not self.signal_id:
            object.__setattr__(self, "signal_id", split_chunk_id(self.chunk_id)[0])


@dataclass(frozen=True, slots=True)
class Passage:
    """A retrieved chunk, resolved to text and everything needed to cite it.

    `char_start` / `char_end` index into the *cleaned* `Signal.content.text`, the
    same coordinate space `EntityMention` offsets use. Without them a citation
    can name a Signal but not point at the sentence, which is the difference
    between "this report cites sources" and "this claim is checkable".
    """

    chunk_id: str
    signal_id: str
    text: str
    char_start: int
    char_end: int

    # -- provenance, carried for the citation and for confidence -------------
    platform: Platform = Platform.UNKNOWN
    source: SourceCategory = SourceCategory.UNKNOWN
    url: str | None = None
    title: str | None = None
    published_at: datetime | None = None
    author_handle: str | None = None
    signal_confidence: Score = 0.0

    # -- how it was found ----------------------------------------------------
    fused_score: float = 0.0
    rerank_score: float | None = None
    found_by: frozenset[Backend] = frozenset()
    ranks: Mapping[str, int] = field(default_factory=dict)

    # -- near-duplicate collapse --------------------------------------------
    duplicate_of_count: int = 0
    """How many near-identical passages collapsed into this one.

    Not discarded information: `docs/retrieval.md` §3 keeps it so a report can
    say "reported by 6 sources", which is corroboration evidence rather than
    noise. Collapsing to a single passage and forgetting the siblings would
    throw away the strongest signal a press release carries.
    """

    collapsed_signal_ids: Sequence[str] = ()

    @property
    def final_score(self) -> float:
        """Rerank score when the cross-encoder ran, otherwise the fused score."""
        return self.rerank_score if self.rerank_score is not None else self.fused_score

    @property
    def is_citable(self) -> bool:
        """Whether a citation built from this passage can be verified.

        A passage with no text or an empty span cannot be quote-checked by
        `services/evidence_service.py`, and an uncheckable citation is worse than
        none -- it looks like evidence.
        """
        return bool(self.text.strip()) and self.char_end > self.char_start


@dataclass(frozen=True, slots=True)
class GraphFact:
    """A relationship from the knowledge graph, carried alongside passages.

    GraphRAG merges graph neighbours with retrieved text
    (`docs/retrieval.md` §8). A fact is not a passage: it has no quotable span,
    so it is kept in its own list rather than being rendered into prose and
    mixed in. Blurring the two is how a graph inference ends up cited as though
    a human had written it.
    """

    subject_id: str
    subject_name: str
    predicate: str
    object_id: str
    object_name: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: Score = 0.0
    supporting_signal_ids: Sequence[str] = ()

    @property
    def is_current(self) -> bool:
        """Whether the fact holds now, per its temporal validity interval."""
        return self.valid_to is None


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    """Per-run measurements. Cheap to collect, and the only way to tune anything.

    `docs/retrieval.md` §3 states plainly that the defaults are starting points
    that have never been measured. Without per-backend counts and latencies,
    tuning them is guesswork -- and a backend silently returning zero results
    looks exactly like a hard query.
    """

    per_backend_counts: Mapping[str, int] = field(default_factory=dict)
    per_backend_latency_ms: Mapping[str, float] = field(default_factory=dict)
    backends_failed: Sequence[str] = ()
    fused_pool_size: int = 0
    after_dedupe: int = 0
    reranked: int = 0
    total_latency_ms: float = 0.0
    token_budget_used: int = 0
    truncated_for_budget: bool = False

    @property
    def degraded(self) -> bool:
        """Whether any backend failed, which must lower reported confidence."""
        return bool(self.backends_failed)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """What a retrieval call returns."""

    request: RetrievalRequest
    passages: Sequence[Passage] = ()
    graph_facts: Sequence[GraphFact] = ()
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)
    retrieved_at: datetime = field(default_factory=utcnow)

    def __len__(self) -> int:
        return len(self.passages)

    @property
    def signal_ids(self) -> list[str]:
        """Distinct Signals represented, in result order."""
        seen: dict[str, None] = {}
        for passage in self.passages:
            seen.setdefault(passage.signal_id, None)
        return list(seen)

    def citable_passages(self) -> list[Passage]:
        """Only the passages a verifiable citation can be built from."""
        return [p for p in self.passages if p.is_citable]


@dataclass(frozen=True, slots=True)
class EvidencePack:
    """Passages and facts assembled to fit a model context window.

    Built by `retrieval/graphrag/context_builder.py`. Separate from
    `RetrievalResult` because budgeting is a *consumer* concern: the same result
    is packed differently for a Critic that needs full text than for a Planner
    that needs titles, and baking a budget into retrieval would force a re-query
    to change it.
    """

    passages: Sequence[Passage] = ()
    graph_facts: Sequence[GraphFact] = ()
    token_count: int = 0
    token_budget: int = 24_000
    dropped_passages: int = 0
    dropped_facts: int = 0

    @property
    def within_budget(self) -> bool:
        return self.token_count <= self.token_budget

    @property
    def is_complete(self) -> bool:
        """Whether everything retrieved fitted.

        Surfaced to the agent because an answer built from a truncated pack is
        weaker than one built from a whole pack, and only the pack knows.
        """
        return self.dropped_passages == 0 and self.dropped_facts == 0


def replace_dataclass(instance: Any, **changes: Any) -> Any:
    """`dataclasses.replace` for slotted frozen dataclasses.

    Wrapped so callers do not need the import, and so the intent -- "a modified
    copy, never a mutation" -- reads at the call site.
    """
    import dataclasses

    return dataclasses.replace(instance, **changes)
