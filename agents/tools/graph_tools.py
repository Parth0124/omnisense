"""Knowledge-graph reads, wrapped as tools: entities, neighbours, paths, subgraphs.

Four **read** tools and no write tool, which is a rule rather than an omission.
`docs/security-and-privacy.md` §8.2: graph and index writes are performed by
workers consuming validated events (`workers/graph_worker.py`), never by an agent
tool call, so no agent has a write path to Neo4j. An agent that could write to the
graph could be talked into writing a fact by a passage it was asked to read, and
that fact would then be indistinguishable from one an extraction pipeline
produced -- laundering an injection into durable ground truth.

Everything here goes through `services/graph_service.py` rather than through
`graph/` directly, for the reason `docs/architecture.md` §6.2 gives: the service
layer owns sessions, tenant scoping and query budgets. That module is a stub
today, so `load_graph_service()` raises `NotImplementedError` with a message
naming what is missing, and the port it will satisfy is declared here as
`GraphReader`. Declaring the port now is what lets the tools, their schemas and
their tests be real code instead of waiting.

Two shapes below -- `EntityRef` and `GraphPath` -- are defined in this module
even though they are graph vocabulary. `graph/schema/nodes.py` is where they
belong and is a stub; `services/evidence_service.py::Citation` took the same
route for the same reason, and a shape defined here is at least defined
somewhere and checkable.

Bounding matters more here than anywhere else in the tool layer. Graph
traversal is the one capability whose cost is *super-linear* in its arguments: a
depth-3 expansion over a well-connected entity is not twice a depth-2 expansion,
it is a hundred times one. Depth, fanout and result count are all capped, and
whether a cap bit travels back in the result -- `GraphContext.truncated` exists
in the state for exactly this, because a truncated neighbourhood is a weaker
basis for a claim and the Critic has to be told rather than left to infer it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

from pydantic import Field

from agents.tools.registry import BoundedResult, ProvenanceStr, ToolSpec
from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger
from models.base import StrictModel
from models.enums import EdgeType, EntityType

__all__ = [
    "MAX_DEPTH",
    "MAX_EDGES",
    "MAX_ENTITIES",
    "MAX_HOPS",
    "MAX_PATHS",
    "EntityMatch",
    "EntityMatches",
    "EntityRef",
    "FindPathsInput",
    "GraphEdge",
    "GraphFactRecord",
    "GraphPath",
    "GraphPathOut",
    "GraphReader",
    "GraphToolset",
    "NeighboursInput",
    "Neighbourhood",
    "PathHop",
    "PathSet",
    "SearchEntitiesInput",
    "SubgraphInput",
    "SubgraphResult",
    "load_graph_service",
]

logger = get_logger(__name__)

MAX_ENTITIES: Final = 25
MAX_EDGES: Final = 50
MAX_PATHS: Final = 10
MAX_DEPTH: Final = 3
"""Hops a neighbourhood expansion may take.

Three, because the fourth hop in a market-intelligence graph reaches everything:
`Company -MENTIONS-> Topic -MENTIONED_BY-> Company` already connects most of a
category, and a fourth hop returns the category rather than the neighbourhood.
The cost is exponential and the information content collapses at the same rate.
"""

MAX_HOPS: Final = 4
"""Path length for `find_paths`.

One more than `MAX_DEPTH` because a *path* between two named endpoints is
anchored at both ends -- the search is constrained in a way an open expansion is
not, so the same hop count costs far less and says far more.
"""

MAX_SUPPORTING_IDS: Final = 5
"""Supporting signal ids carried per fact.

Enough to check corroboration, not enough for the list to become the payload. An
edge supported by 400 Signals is a fact about the corpus, not 400 facts.
"""


# --------------------------------------------------------------------------- #
# Domain shapes the graph layer will produce
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EntityRef:
    """One resolved knowledge-graph entity.

    Belongs in `graph/schema/nodes.py` or `models/entity.py` once either exists;
    both are stubs today. Defined here rather than nowhere, so the resolution
    path is real code -- the same call `services/evidence_service.py` made for
    `Citation`.
    """

    entity_id: str
    name: str
    entity_type: EntityType = EntityType.UNKNOWN
    aliases: Sequence[str] = ()
    confidence: float = 0.0
    mention_count: int = 0


@dataclass(frozen=True, slots=True)
class GraphPath:
    """One traversal between two entities.

    Node ids and edge predicates in alternating order, kept as parallel
    sequences rather than a list of triples because a path is read end-to-end and
    a triple list forces a reader to reconstruct the chain.
    """

    entity_ids: Sequence[str]
    entity_names: Sequence[str] = ()
    predicates: Sequence[EdgeType] = ()
    confidence: float = 0.0

    @property
    def hops(self) -> int:
        return max(0, len(self.entity_ids) - 1)


@dataclass(frozen=True, slots=True)
class GraphFactRecord:
    """One relationship as the graph service returns it.

    Mirrors `retrieval/types.py::GraphFact` field for field. Duplicated rather
    than imported so that a graph service does not have to depend on the
    retrieval package to answer a question that has nothing to do with
    retrieval; the two converge when `models/` grows the shared shape.
    """

    subject_id: str
    subject_name: str
    predicate: EdgeType
    object_id: str
    object_name: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 0.0
    supporting_signal_ids: Sequence[str] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Port
# --------------------------------------------------------------------------- #


@runtime_checkable
class GraphReader(Protocol):
    """The read surface `services/graph_service.py` must expose to agents.

    Tenant scoping is a *parameter* on every method rather than state on the
    reader, because one service instance serves every concurrent investigation in
    a worker: a tenant remembered on `self` is a cross-tenant leak waiting for
    two runs to overlap. The toolset supplies the value from its own binding, so
    it still never reaches a tool's input schema.
    """

    async def search_entities(
        self,
        query: str,
        *,
        tenant_id: str,
        entity_types: Sequence[EntityType] = (),
        limit: int = MAX_ENTITIES,
    ) -> Sequence[EntityRef]: ...

    async def neighbours(
        self,
        entity_id: str,
        *,
        tenant_id: str,
        edge_types: Sequence[EdgeType] = (),
        depth: int = 1,
        limit: int = MAX_EDGES,
        as_of: datetime | None = None,
    ) -> Sequence[GraphFactRecord]: ...

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        tenant_id: str,
        max_hops: int = 3,
        edge_types: Sequence[EdgeType] = (),
        limit: int = MAX_PATHS,
    ) -> Sequence[GraphPath]: ...

    async def subgraph(
        self,
        entity_ids: Sequence[str],
        *,
        tenant_id: str,
        depth: int = 1,
        limit: int = MAX_EDGES,
    ) -> Sequence[GraphFactRecord]: ...


def load_graph_service(**kwargs: object) -> GraphReader:
    """Construct the real graph service, or say precisely what is missing.

    `NotImplementedError` rather than a stub reader that returns nothing: an
    empty neighbourhood is a *meaningful* answer -- it says this entity has no
    recorded relationships -- so a stub would make "the graph is not built yet"
    indistinguishable from "these companies are unrelated", and the Competitor
    agent would report the second.
    """
    import services.graph_service as graph_service

    service_cls = getattr(graph_service, "GraphService", None)
    if service_cls is None:
        raise NotImplementedError(
            "services/graph_service.py does not define GraphService yet, so the graph "
            "tools cannot be bound. Implement it against "
            "agents.tools.graph_tools.GraphReader (search_entities, neighbours, "
            "find_paths, subgraph); agents must not reach graph/ or Neo4j directly."
        )
    reader = service_cls(**kwargs)
    if not isinstance(reader, GraphReader):
        raise NotImplementedError(
            "services.graph_service.GraphService does not satisfy "
            "agents.tools.graph_tools.GraphReader; the graph tools need all four "
            "read methods before they can be registered."
        )
    return reader


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class SearchEntitiesInput(StrictModel):
    query: str = Field(min_length=1, max_length=200)
    entity_types: list[EntityType] = Field(default_factory=list, max_length=8)
    limit: int = Field(default=10, ge=1, le=MAX_ENTITIES)


class EntityMatch(StrictModel):
    """A resolved entity. Names and aliases are third-party strings.

    `name` and `aliases` are `ProvenanceStr` because an entity name is extracted
    from ingested text: a product literally called "Ignore Previous Instructions"
    is a legal company name, and an entity name is one of the few third-party
    strings that has to travel outside a fence because the agent passes it back
    as a tool argument.
    """

    entity_id: str
    name: ProvenanceStr
    entity_type: EntityType = EntityType.UNKNOWN
    aliases: list[ProvenanceStr] = Field(default_factory=list, max_length=8)
    confidence: float = 0.0
    mention_count: int = 0


class EntityMatches(BoundedResult):
    ITEMS_FIELD = "entities"

    query: str
    entities: list[EntityMatch] = Field(default_factory=list)


class NeighboursInput(StrictModel):
    entity_id: str = Field(min_length=1, max_length=128)
    edge_types: list[EdgeType] = Field(default_factory=list, max_length=8)
    depth: int = Field(default=1, ge=1, le=MAX_DEPTH)
    limit: int = Field(default=25, ge=1, le=MAX_EDGES)
    as_of: datetime | None = Field(
        default=None,
        description="Point in time the neighbourhood is asserted at. Omit for now. "
        "A competitive claim about last quarter must not be answered with today's "
        "edges (`docs/knowledge-graph.md` temporal validity).",
    )


class GraphEdge(StrictModel):
    """One relationship, with its temporal validity and its support."""

    subject_id: str
    subject_name: ProvenanceStr
    predicate: EdgeType
    object_id: str
    object_name: ProvenanceStr
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    is_current: bool = True
    confidence: float = 0.0
    supporting_signal_ids: list[str] = Field(default_factory=list, max_length=MAX_SUPPORTING_IDS)


class Neighbourhood(BoundedResult):
    ITEMS_FIELD = "edges"

    entity_id: str
    depth: int
    edges: list[GraphEdge] = Field(default_factory=list)
    fanout_capped: bool = False
    """Whether the traversal hit its cap rather than running out of graph.

    The same distinction `GraphContext.truncated` carries into the state: an
    entity with three neighbours and an entity whose first 50 of 4,000 neighbours
    were returned support very different claims, and nothing downstream can tell
    them apart from the edge list alone.
    """


class FindPathsInput(StrictModel):
    source_id: str = Field(min_length=1, max_length=128)
    target_id: str = Field(min_length=1, max_length=128)
    max_hops: int = Field(default=3, ge=1, le=MAX_HOPS)
    edge_types: list[EdgeType] = Field(default_factory=list, max_length=8)
    limit: int = Field(default=5, ge=1, le=MAX_PATHS)


class PathHop(StrictModel):
    entity_id: str
    entity_name: ProvenanceStr = ""
    predicate_to_next: EdgeType | None = None


class GraphPathOut(StrictModel):
    hops: list[PathHop] = Field(default_factory=list, max_length=MAX_HOPS + 1)
    length: int = 0
    confidence: float = 0.0


class PathSet(BoundedResult):
    ITEMS_FIELD = "paths"

    source_id: str
    target_id: str
    paths: list[GraphPathOut] = Field(default_factory=list)


class SubgraphInput(StrictModel):
    entity_ids: list[str] = Field(min_length=1, max_length=10)
    depth: int = Field(default=1, ge=1, le=2)
    limit: int = Field(default=25, ge=1, le=MAX_EDGES)


class SubgraphResult(BoundedResult):
    ITEMS_FIELD = "edges"

    seed_entity_ids: list[str] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    fanout_capped: bool = False


# --------------------------------------------------------------------------- #
# The toolset
# --------------------------------------------------------------------------- #


class GraphToolset:
    """Binds a `GraphReader` to the four graph tools.

    The reader is required. A toolset that could be constructed without one
    would register tools that cannot work, and the failure would surface as an
    empty graph rather than as a wiring error -- see `load_graph_service()` for
    why an empty graph is the wrong lie to tell.
    """

    def __init__(self, *, reader: GraphReader, tenant_id: str) -> None:
        if not tenant_id:
            raise ConfigurationError("GraphToolset requires an explicit tenant_id")
        self._reader = reader
        self._tenant_id = tenant_id

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="search_entities",
                description=(
                    "Resolve names in a question to knowledge-graph entities. Returns "
                    "entity ids, types and aliases -- no free text."
                ),
                input_model=SearchEntitiesInput,
                output_model=EntityMatches,
                handler=self._search_entities,
            ),
            ToolSpec(
                name="neighbours",
                description=(
                    "List relationships around one entity, optionally as of a point in "
                    "time. Read-only: no agent can write to the graph."
                ),
                input_model=NeighboursInput,
                output_model=Neighbourhood,
                handler=self._neighbours,
            ),
            ToolSpec(
                name="find_paths",
                description=(
                    "Find bounded paths between two entities -- how a competitor, "
                    "product or topic connects to another."
                ),
                input_model=FindPathsInput,
                output_model=PathSet,
                handler=self._find_paths,
            ),
            ToolSpec(
                name="subgraph",
                description=(
                    "Expand a small set of seed entities into the relationships that "
                    "connect them."
                ),
                input_model=SubgraphInput,
                output_model=SubgraphResult,
                handler=self._subgraph,
            ),
        ]

    # ------------------------------------------------------------ handlers --

    async def _search_entities(self, args: SearchEntitiesInput) -> EntityMatches:
        limit = min(args.limit, MAX_ENTITIES)
        found = await self._reader.search_entities(
            args.query,
            tenant_id=self._tenant_id,
            entity_types=tuple(args.entity_types),
            limit=limit,
        )
        kept = list(found)[:limit]
        return EntityMatches(
            query=args.query,
            entities=[
                EntityMatch(
                    entity_id=ref.entity_id,
                    name=ref.name,
                    entity_type=ref.entity_type,
                    aliases=list(ref.aliases)[:8],
                    confidence=ref.confidence,
                    mention_count=ref.mention_count,
                )
                for ref in kept
            ],
            truncated=len(found) > len(kept),
            dropped=max(0, len(found) - len(kept)),
        )

    async def _neighbours(self, args: NeighboursInput) -> Neighbourhood:
        limit = min(args.limit, MAX_EDGES)
        facts = await self._reader.neighbours(
            args.entity_id,
            tenant_id=self._tenant_id,
            edge_types=tuple(args.edge_types),
            depth=min(args.depth, MAX_DEPTH),
            limit=limit,
            as_of=args.as_of,
        )
        kept = list(facts)[:limit]
        return Neighbourhood(
            entity_id=args.entity_id,
            depth=min(args.depth, MAX_DEPTH),
            edges=[_to_edge(fact) for fact in kept],
            # `>=` rather than `>`: a reader that returns exactly `limit` rows
            # cannot say whether the next row existed, and reporting an
            # exhausted budget as a complete neighbourhood is the failure this
            # flag is for.
            fanout_capped=len(facts) >= limit,
            truncated=len(facts) > len(kept),
            dropped=max(0, len(facts) - len(kept)),
        )

    async def _find_paths(self, args: FindPathsInput) -> PathSet:
        limit = min(args.limit, MAX_PATHS)
        paths = await self._reader.find_paths(
            args.source_id,
            args.target_id,
            tenant_id=self._tenant_id,
            max_hops=min(args.max_hops, MAX_HOPS),
            edge_types=tuple(args.edge_types),
            limit=limit,
        )
        kept = list(paths)[:limit]
        return PathSet(
            source_id=args.source_id,
            target_id=args.target_id,
            paths=[_to_path(path) for path in kept],
            truncated=len(paths) > len(kept),
            dropped=max(0, len(paths) - len(kept)),
        )

    async def _subgraph(self, args: SubgraphInput) -> SubgraphResult:
        limit = min(args.limit, MAX_EDGES)
        seeds = list(dict.fromkeys(args.entity_ids))
        facts = await self._reader.subgraph(
            seeds, tenant_id=self._tenant_id, depth=args.depth, limit=limit
        )
        kept = list(facts)[:limit]
        return SubgraphResult(
            seed_entity_ids=seeds,
            edges=[_to_edge(fact) for fact in kept],
            fanout_capped=len(facts) >= limit,
            truncated=len(facts) > len(kept),
            dropped=max(0, len(facts) - len(kept)),
        )


def _to_edge(fact: GraphFactRecord) -> GraphEdge:
    return GraphEdge(
        subject_id=fact.subject_id,
        subject_name=fact.subject_name,
        predicate=fact.predicate,
        object_id=fact.object_id,
        object_name=fact.object_name,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
        # A closed validity interval means the fact *was* true and is not now.
        # Rendered as a field rather than left for the agent to derive from the
        # dates, because "Datadog acquired X" read without its `valid_to` is how
        # a three-year-old divestiture becomes a present-tense claim.
        is_current=fact.valid_to is None,
        confidence=fact.confidence,
        supporting_signal_ids=list(fact.supporting_signal_ids)[:MAX_SUPPORTING_IDS],
    )


def _to_path(path: GraphPath) -> GraphPathOut:
    names = list(path.entity_names)
    predicates = list(path.predicates)
    hops = [
        PathHop(
            entity_id=entity_id,
            entity_name=names[index] if index < len(names) else "",
            predicate_to_next=predicates[index] if index < len(predicates) else None,
        )
        for index, entity_id in enumerate(path.entity_ids)
    ]
    return GraphPathOut(hops=hops, length=path.hops, confidence=path.confidence)
