"""Wire shapes for `/api/v1/graph/*` (`docs/api-reference.md` §3.5).

These are deliberately *not* `services/graph_service.py`'s dataclasses. That
module returns `Entity`, `Competitor` and `GraphFactRecord` -- domain objects
shaped by what the graph knows. These are shaped by what a client needs, and the
difference is not ceremony:

* `normalized_name`, `merged_from` and `schema_version` are internal. They exist
  so resolution can un-merge and so a backfill can find nodes written by an older
  schema. Publishing them makes them part of the contract, and the next time
  resolution changes its normalisation someone's dashboard breaks.
* `analytics_are_stale` is computed from two timestamps a client should not have
  to reason about. It is published as one boolean, because "is this ranking
  current" is the actual question.
* Enums are serialised as their string values with a documented closed set, so a
  client can switch on them.

The response models inherit `ResponseModel`, which forbids extra fields on the
way out. That is what stops an internal field from leaking into a payload by
being added to a domain dataclass -- the failure mode where a database column
becomes a public API by accident, discovered when it is removed.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Final

from pydantic import Field

from backend.schemas.common import RequestModel, ResponseModel

__all__ = [
    "MAX_GRAPH_QUERY_CHARS",
    "MAX_SEED_IDS",
    "CompetitorItem",
    "CompetitorsResponse",
    "EntityDetail",
    "EntityHit",
    "EntitySearchResponse",
    "GraphEdge",
    "GraphNode",
    "GraphPathItem",
    "GraphPathsResponse",
    "OwnershipChainResponse",
    "RelationshipBasis",
    "SignalMentionItem",
    "SubgraphRequest",
    "SubgraphResponse",
]

MAX_GRAPH_QUERY_CHARS: Final = 200
"""Cap on a fulltext query string.

A Lucene query is parsed before it is matched, and a pathological one -- deeply
nested boolean groups, a wildcard on a two-character prefix -- costs the *server*
time proportional to something the client controls. 200 characters is far more
than any entity name and short enough that no such query fits.
"""

MAX_SEED_IDS: Final = 50
"""Cap on the seed set for a subgraph request.

Each seed expands to a neighbourhood, so the cost is multiplicative rather than
additive. Fifty seeds at depth 2 is already the largest graph a canvas can
usefully draw; beyond that the picture is a hairball and the query is expensive.
"""


class RelationshipBasis(enum.StrEnum):
    """How a relationship was established.

    Published because the distinction changes what a claim built on the edge is
    worth. `stated` means a document said it; `inferred` means the system
    concluded it from co-occurrence. A UI that renders both identically lets a
    statistical hunch be read as a sourced fact.
    """

    STATED = "stated"
    INFERRED = "inferred"
    DERIVED = "derived"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #


class EntityHit(ResponseModel):
    """One search result. Lean: a list of these is rendered as rows."""

    id: str
    name: str
    type: str = Field(description="Entity label: Company, Product, Person, Topic, ...")
    description: str | None = None
    aliases: list[str] = Field(default_factory=list, max_length=5)
    source_count: int = Field(
        default=0,
        description=(
            "Distinct signals that evidenced this entity. The cheapest "
            "anti-hallucination signal a client has: an entity with one source "
            "is a lead, not a fact."
        ),
    )
    score: float | None = Field(
        default=None, description="Fulltext relevance. Comparable within one response only."
    )


class EntitySearchResponse(ResponseModel):
    query: str
    results: list[EntityHit]
    total: int = Field(description="Results returned, not results available.")


class EntityDetail(ResponseModel):
    """One entity in full, for a detail panel."""

    id: str
    name: str
    type: str
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float | None = Field(
        default=None,
        description="0-1 that this entity is real and correctly resolved. Null means unscored.",
    )
    source_count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    pagerank_score: float | None = Field(
        default=None, description="Batch-computed importance. Null until first computed."
    )
    community_id: str | None = None
    analytics_are_stale: bool = Field(
        description=(
            "True when the ranking properties predate this entity's most recent "
            "observation, or were never computed. A stale pagerank_score is not "
            "wrong in any way a client can detect on its own."
        )
    )


# --------------------------------------------------------------------------- #
# Relationships
# --------------------------------------------------------------------------- #


class CompetitorItem(ResponseModel):
    id: str
    name: str
    type: str
    strength: float | None = Field(
        default=None,
        description=(
            "0-1 rivalry strength. Null means the relationship was never scored, "
            "which is not the same as 0.0 (assessed and negligible)."
        ),
    )
    basis: RelationshipBasis = RelationshipBasis.UNKNOWN
    market: str | None = None
    confidence: float | None = None
    evidence_count: int = 0
    valid_from: datetime | None = None
    valid_to: datetime | None = Field(
        default=None, description="Null means the relationship is currently held to be true."
    )
    citations: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Signal ids evidencing this edge. Truncated; not the full set.",
    )


class CompetitorsResponse(ResponseModel):
    subject: str
    as_of: datetime = Field(
        description=(
            "The instant the graph was read at. Echoed back because it defaults "
            "to now on the server, and a client that omitted it cannot otherwise "
            "reproduce the result."
        )
    )
    results: list[CompetitorItem]
    total: int


class OwnershipChainResponse(ResponseModel):
    """Who ultimately owns a company. `chain` is empty when nothing acquired it."""

    company_id: str
    as_of: datetime
    chain: list[str] = Field(
        default_factory=list, description="Entity ids, ultimate owner first."
    )
    names: list[str] = Field(default_factory=list)
    hops: int = 0
    is_independent: bool = Field(
        description="True when no closed acquisition reaches this company."
    )


class SignalMentionItem(ResponseModel):
    signal_id: str
    salience: float | None = None
    sentiment: float | None = None
    mention_text: str | None = None
    observed_at: datetime | None = None


# --------------------------------------------------------------------------- #
# Subgraph -- what the canvas renders
# --------------------------------------------------------------------------- #


class GraphNode(ResponseModel):
    id: str
    name: str
    type: str


class GraphEdge(ResponseModel):
    source: str
    target: str
    predicate: str
    confidence: float | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    supporting_signal_ids: list[str] = Field(default_factory=list, max_length=5)


class SubgraphRequest(RequestModel):
    entity_ids: list[str] = Field(min_length=1, max_length=MAX_SEED_IDS)
    depth: int = Field(default=1, ge=1, le=3)
    limit: int = Field(default=100, ge=1, le=500)


class SubgraphResponse(ResponseModel):
    """Nodes and edges, deduplicated, ready to lay out.

    Nodes are returned separately from edges rather than nested inside them,
    because an entity connected by six edges appears once here and six times in
    the nested form -- and a client that renders the nested form draws six
    overlapping copies of the same node.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = Field(
        description=(
            "True when the edge limit was reached. The graph shown is a subset, "
            "and a client must not present it as the complete neighbourhood."
        )
    )


class GraphPathItem(ResponseModel):
    entity_ids: list[str]
    entity_names: list[str]
    predicates: list[str]
    hops: int
    confidence: float = Field(
        description=(
            "Product of the per-edge confidences. Falls off quickly with length, "
            "which is correct: a four-hop connection through three uncertain "
            "edges is a weak claim however plausible each hop looks alone."
        )
    )


class GraphPathsResponse(ResponseModel):
    source_id: str
    target_id: str
    paths: list[GraphPathItem]
    connected: bool
