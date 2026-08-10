"""The knowledge graph's only public face.

`backend/api/v1/graph.py`, the Competitor Agent and the trend service all want
graph facts. None of them should know that a fact costs a Cypher traversal, that
`COMPETES_WITH` is stored in one canonical orientation, or that `EntityType` is a
tolerant enum whose `UNKNOWN` member must never reach a query. This module knows
those things so nothing above it has to.

Its real job is **translation in three directions**, and each one is load-bearing.

**Errors become HTTP-shaped.** `graph/` is an L1 library and may not import
`backend/core/exceptions.py`, so it raises its own types: `GraphSchemaError` for
a bad argument, `GraphQueryError` for a statement the server rejected,
`GraphUnavailableError` for a graph it could not reach. Those are three different
HTTP responses -- 422, 502, 503 -- and getting them wrong has real consequences:
a 500 for a malformed filter sends an engineer to look at the database, and a 503
for a syntax error makes a client retry a query that will never succeed.

**Rows become models.** The client returns dicts. Callers get typed objects with
`float | None` where the graph has a null, because a service that hands out raw
rows makes every caller re-implement the same six `.get()` calls with slightly
different defaults.

**Absence becomes a decision.** `graph/client.py` returns `None` for an empty
result because at that layer "not in the graph" is an ordinary answer. Here it is
not: `get_entity` on an id the caller believes exists is a 404, while
`competitors_of` returning nothing is an empty list and a perfectly good answer.
Only this layer knows which is which.

**On degradation.** `docs/architecture.md` §7.3 makes Neo4j optional: graph
expansion is skipped, the report notes reduced context, and the request still
answers. That is expressed here as an explicit `allow_degraded` argument rather
than a blanket try/except, because the two call sites genuinely differ.
`GET /graph/search` *is* the graph -- degrading it means returning an empty list
that looks like "no such company", which is a wrong answer wearing the clothes of
a right one. Competitor expansion inside an investigation is one input among
several, and losing it should cost confidence, not the whole run.

Layer note: **L2 service** -- may import `graph/`, `models/` and the kernel;
imported by `backend/api/` and `agents/`. It does not import `retrieval/`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from backend.core.exceptions import (
    DependencyUnavailableError,
    ExternalServiceError,
    NotFoundError,
    ValidationError,
)
from backend.core.logging import get_logger
from graph.client import GraphClient, GraphQueryError, GraphUnavailableError
from graph.queries import cypher as templates
from graph.schema.nodes import GraphSchemaError
from models.enums import EdgeType, EntityType

__all__ = [
    "MAX_SEARCH_LIMIT",
    "Entity",
    "EntityRef",
    "GraphFactRecord",
    "GraphPath",
    "GraphService",
    "NeighbourSignal",
    "SignalMention",
    "TopicActivity",
    "build_graph_service",
]

_log = get_logger(__name__)

MAX_SEARCH_LIMIT: Final[int] = templates.MAX_LIMIT
"""Re-exported so the API layer can bound a query string without importing `graph/`."""


# --------------------------------------------------------------------------- #
# Return types
# --------------------------------------------------------------------------- #


def _as_float(value: Any) -> float | None:
    """Coerce a graph number, preserving `None`.

    `None` is preserved rather than defaulted to 0.0, and the distinction is not
    pedantry: a `strength` of 0.0 means "we assessed this and it is negligible",
    while `None` means "nobody assessed it". A UI that renders both as an empty
    bar is fine; one that renders 0.0 as "no competition" is asserting something
    the graph never said.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _as_str_list(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _as_datetime(value: Any) -> datetime | None:
    return value if isinstance(value, datetime) else None


@dataclass(frozen=True, slots=True)
class Entity:
    """One resolved entity, as the API and the agents see it."""

    id: str
    name: str
    type: EntityType
    description: str | None = None
    aliases: tuple[str, ...] = ()
    merged_from: tuple[str, ...] = ()
    normalized_name: str | None = None
    confidence: float | None = None
    source_count: int = 0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    pagerank_score: float | None = None
    community_id: str | None = None
    computed_at: datetime | None = None
    score: float | None = None
    """Fulltext relevance, present only on search results."""

    @property
    def analytics_are_stale(self) -> bool:
        """Whether the ranking properties predate the last time this node changed.

        Worth exposing rather than hiding, because a `pagerank_score` computed
        before an entity absorbed forty new mentions is not wrong in a way any
        consumer can detect -- it is a plausible number describing a graph that no
        longer exists.
        """
        if self.computed_at is None:
            return True
        if self.last_seen is None:
            return False
        return self.last_seen > self.computed_at

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Entity:
        raw_type = row.get("type")
        return cls(
            id=str(row.get("id", "")),
            name=str(row.get("name") or ""),
            # `EntityType` is tolerant on the way *in*, which is right here: this
            # is a reader, and a label written by a newer producer should degrade
            # to UNKNOWN rather than crash a search. The strictness lives on the
            # write and filter paths, where a typo must not silently match
            # nothing.
            type=EntityType(raw_type) if isinstance(raw_type, str) else EntityType.UNKNOWN,
            description=row.get("description") if isinstance(row.get("description"), str) else None,
            aliases=_as_str_list(row.get("aliases")),
            merged_from=_as_str_list(row.get("merged_from")),
            normalized_name=row.get("normalized_name")
            if isinstance(row.get("normalized_name"), str)
            else None,
            confidence=_as_float(row.get("confidence")),
            source_count=_as_int(row.get("source_count")),
            first_seen=_as_datetime(row.get("first_seen")),
            last_seen=_as_datetime(row.get("last_seen")),
            pagerank_score=_as_float(row.get("pagerank_score")),
            community_id=row.get("community_id")
            if isinstance(row.get("community_id"), str)
            else None,
            computed_at=_as_datetime(row.get("computed_at")),
            score=_as_float(row.get("score")),
        )








@dataclass(frozen=True, slots=True)
class NeighbourSignal:
    signal_id: str
    via_entity_id: str
    via_entity_name: str | None
    via_entity_type: EntityType
    salience: float | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> NeighbourSignal:
        raw_type = row.get("via_entity_type")
        return cls(
            signal_id=str(row.get("signal_id", "")),
            via_entity_id=str(row.get("via_entity_id", "")),
            via_entity_name=row.get("via_entity_name")
            if isinstance(row.get("via_entity_name"), str)
            else None,
            via_entity_type=(
                EntityType(raw_type) if isinstance(raw_type, str) else EntityType.UNKNOWN
            ),
            salience=_as_float(row.get("salience")),
        )


@dataclass(frozen=True, slots=True)
class SignalMention:
    signal_id: str
    salience: float | None
    sentiment: float | None
    mention_text: str | None
    observed_at: datetime | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SignalMention:
        return cls(
            signal_id=str(row.get("signal_id", "")),
            salience=_as_float(row.get("salience")),
            sentiment=_as_float(row.get("sentiment")),
            mention_text=row.get("mention_text")
            if isinstance(row.get("mention_text"), str)
            else None,
            observed_at=_as_datetime(row.get("observed_at")),
        )


@dataclass(frozen=True, slots=True)
class TopicActivity:
    topic_id: str
    topic: str
    mentions: int
    avg_sentiment: float | None
    last_mentioned_at: datetime | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> TopicActivity:
        return cls(
            topic_id=str(row.get("topic_id", "")),
            topic=str(row.get("topic") or ""),
            mentions=_as_int(row.get("mentions")),
            avg_sentiment=_as_float(row.get("avg_sentiment")),
            last_mentioned_at=_as_datetime(row.get("last_mentioned_at")),
        )


# --------------------------------------------------------------------------- #
# The agent-facing port shapes
# --------------------------------------------------------------------------- #
#
# These three mirror `agents/tools/graph_tools.py` field for field, and they are
# *duplicated rather than imported* for one reason: `agents/` is L3 and
# `services/` is L2, so importing them would invert the dependency and make this
# service unusable from anything that is not the agent runtime.
#
# `graph_tools.py` makes exactly the same call from the other side, duplicating
# `GraphFact` out of `retrieval/types.py` so a graph service does not have to
# depend on the retrieval package to answer a question that has nothing to do
# with retrieval. The two converge when `models/` grows the shared shape; until
# then the duplication is the layering, and `tests/unit/services/
# test_graph_service.py` asserts the field names still match so a drift is
# caught by a test rather than by an `AttributeError` inside an agent.


@dataclass(frozen=True, slots=True)
class EntityRef:
    """One resolved entity, in the lean shape agents receive."""

    entity_id: str
    name: str
    entity_type: EntityType = EntityType.UNKNOWN
    aliases: Sequence[str] = ()
    confidence: float = 0.0
    mention_count: int = 0


@dataclass(frozen=True, slots=True)
class GraphFactRecord:
    """One relationship, as subject-predicate-object with its evidence."""

    subject_id: str
    subject_name: str
    predicate: EdgeType
    object_id: str
    object_name: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    confidence: float = 0.0
    supporting_signal_ids: Sequence[str] = ()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GraphFactRecord:
        raw_predicate = row.get("predicate")
        return cls(
            subject_id=str(row.get("subject_id", "")),
            subject_name=str(row.get("subject_name") or ""),
            # Tolerant on read: an edge type written by a newer producer must
            # degrade rather than crash an agent mid-investigation. The strict
            # rejection lives on the write path in `graph/schema/edges.py`.
            predicate=(
                EdgeType(raw_predicate) if isinstance(raw_predicate, str) else EdgeType.UNKNOWN
            ),
            object_id=str(row.get("object_id", "")),
            object_name=str(row.get("object_name") or ""),
            valid_from=_as_datetime(row.get("valid_from")),
            valid_to=_as_datetime(row.get("valid_to")),
            confidence=_as_float(row.get("confidence")) or 0.0,
            supporting_signal_ids=_as_str_list(row.get("supporting_signal_ids")),
        )


@dataclass(frozen=True, slots=True)
class GraphPath:
    """One traversal between two entities, read end to end."""

    entity_ids: Sequence[str]
    entity_names: Sequence[str] = ()
    predicates: Sequence[EdgeType] = ()
    confidence: float = 0.0

    @property
    def hops(self) -> int:
        return max(0, len(self.entity_ids) - 1)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GraphPath:
        raw_predicates = row.get("predicates")
        predicates = (
            tuple(EdgeType(p) if isinstance(p, str) else EdgeType.UNKNOWN for p in raw_predicates)
            if isinstance(raw_predicates, (list, tuple))
            else ()
        )
        return cls(
            entity_ids=_as_str_list(row.get("entity_ids")),
            entity_names=_as_str_list(row.get("entity_names")),
            predicates=predicates,
            confidence=_as_float(row.get("confidence")) or 0.0,
        )


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GraphService:
    """Typed, error-translated reads over the knowledge graph.

    Holds a `GraphClient` and nothing else. Stateless, so one instance per
    process serves concurrent requests.

    Every method takes `tenant_id` explicitly rather than reading it from a
    context variable. Multi-tenancy is Phase 7 and a context variable would work
    today -- and would be the single most dangerous piece of implicit state in
    the system, because the failure mode of a stale one is serving one customer's
    graph to another. An explicit argument makes the wiring visible in every call
    site and reviewable in one grep.
    """

    client: GraphClient
    degraded_returns_empty: bool = True

    # ------------------------------------------------------------- reads --

    async def search(
        self,
        *,
        tenant_id: str,
        query: str,
        entity_types: Sequence[EntityType] | None = None,
        limit: int = templates.DEFAULT_LIMIT,
    ) -> list[Entity]:
        """Fulltext entity search, rich shape. Backs `GET /api/v1/graph/search`.

        `allow_degraded` is deliberately absent: this endpoint *is* the graph, so
        an empty list returned because Neo4j is unreachable is indistinguishable
        from "no such company". A 503 is the honest answer.
        """
        rows = await self._fetch(
            lambda: templates.entity_search(
                tenant_id=tenant_id, query=query, entity_types=entity_types, limit=limit
            ),
            operation="entity_search",
            allow_degraded=False,
        )
        return [Entity.from_row(row) for row in rows]

    async def get_entity(self, *, tenant_id: str, entity_id: str) -> Entity:
        """One entity, or a 404.

        The place where `graph/client.py`'s `None` becomes a decision. That layer
        cannot make it -- "no rows" is ambiguous there -- and here it is not:
        somebody asked for this id.
        """
        rows = await self._fetch(
            lambda: templates.entity_by_id(tenant_id=tenant_id, entity_id=entity_id),
            operation="entity_by_id",
            allow_degraded=False,
        )
        if not rows:
            raise NotFoundError.for_resource("entity", entity_id)
        return Entity.from_row(rows[0])




    async def neighbourhood(
        self,
        *,
        tenant_id: str,
        seed_ids: Sequence[str],
        start: datetime,
        end: datetime,
        max_level: int = 2,
        limit: int = templates.DEFAULT_LIMIT,
        allow_degraded: bool = True,
    ) -> list[NeighbourSignal]:
        """Signals reachable from seed entities. Degrades by default.

        The one read whose default is `allow_degraded=True`, because it has a
        well-defined weaker answer: this is GraphRAG expansion, one retrieval
        backend among three, and losing it costs recall rather than correctness.
        Vector and keyword hits still answer the question.
        """
        rows = await self._fetch(
            lambda: templates.neighbourhood_signals(
                tenant_id=tenant_id,
                seed_ids=seed_ids,
                start=start,
                end=end,
                max_level=max_level,
                limit=limit,
            ),
            operation="neighbourhood",
            allow_degraded=allow_degraded,
        )
        return [NeighbourSignal.from_row(row) for row in rows]

    async def signals_for_entity(
        self,
        *,
        tenant_id: str,
        entity_id: str,
        min_salience: float = 0.0,
        since: datetime | None = None,
        limit: int = templates.DEFAULT_LIMIT,
    ) -> list[SignalMention]:
        """The citation path: signals evidencing an entity, most salient first.

        Never degrades. A citation list that is empty because the graph was
        briefly unreachable produces a report claim with no support, which is
        precisely the failure `services/evidence_service.py` exists to prevent.
        """
        rows = await self._fetch(
            lambda: templates.signals_mentioning(
                tenant_id=tenant_id,
                entity_id=entity_id,
                min_salience=min_salience,
                since=since,
                limit=limit,
            ),
            operation="signals_for_entity",
            allow_degraded=False,
        )
        return [SignalMention.from_row(row) for row in rows]

    async def topic_activity(
        self,
        *,
        tenant_id: str,
        since: datetime,
        until: datetime | None = None,
        min_salience: float = 0.3,
        limit: int = templates.DEFAULT_LIMIT,
        allow_degraded: bool = True,
    ) -> list[TopicActivity]:
        """Topic mention volume over a window -- the graph's trend input."""
        rows = await self._fetch(
            lambda: templates.topic_activity(
                tenant_id=tenant_id,
                since=since,
                until=until,
                min_salience=min_salience,
                limit=limit,
            ),
            operation="topic_activity",
            allow_degraded=allow_degraded,
        )
        return [TopicActivity.from_row(row) for row in rows]

    # --------------------------------------------- the agent-facing port --
    #
    # The four methods below satisfy `agents.tools.graph_tools.GraphReader`.
    # Their argument order differs from the rest of this class -- first argument
    # positional, `tenant_id` keyword-only -- because that Protocol fixes it, and
    # matching it exactly is what lets `load_graph_service()` bind this service
    # without an adapter in between.
    #
    # Tenant scoping is a parameter on every one of them rather than state on
    # `self`, because one service instance serves every concurrent investigation
    # in a worker. A tenant remembered on the instance is a cross-tenant leak
    # waiting for two runs to overlap.

    async def search_entities(
        self,
        query: str,
        *,
        tenant_id: str,
        entity_types: Sequence[EntityType] = (),
        limit: int = 10,
    ) -> list[EntityRef]:
        """`GraphReader.search_entities`: the lean shape agents receive.

        Separate from `search()` rather than one method serving both, because the
        two callers want genuinely different payloads. An agent gets a name, a
        type and a mention count -- enough to decide whether to look closer, and
        small enough that ten of them do not crowd a context window. The API's
        `search()` returns descriptions, aliases and relevance scores because a
        human is about to read them.

        Collapsing the two would mean sending an LLM a `description` and a
        `pagerank_score` on every entity of every search, which is tokens spent
        on fields no prompt references.
        """
        entities = await self.search(
            tenant_id=tenant_id,
            query=query,
            entity_types=list(entity_types) or None,
            limit=limit,
        )
        return [
            EntityRef(
                entity_id=entity.id,
                name=entity.name,
                entity_type=entity.type,
                aliases=entity.aliases,
                confidence=entity.confidence or 0.0,
                mention_count=entity.source_count,
            )
            for entity in entities
        ]

    async def neighbours(
        self,
        entity_id: str,
        *,
        tenant_id: str,
        edge_types: Sequence[Any] = (),
        depth: int = 1,
        limit: int = 50,
        as_of: datetime | None = None,
    ) -> list[GraphFactRecord]:
        """Relationships around one entity, as facts an agent can cite.

        Never degrades. An empty neighbourhood is a *meaningful* answer -- "this
        entity has no recorded relationships" -- so returning `[]` because Neo4j
        was unreachable would make an outage indistinguishable from a finding,
        and the Competitor agent would report the finding.
        """
        rows = await self._fetch(
            lambda: templates.entity_neighbours(
                tenant_id=tenant_id,
                entity_id=entity_id,
                edge_types=tuple(edge_types) or templates.TRAVERSABLE_EDGE_TYPES,
                depth=depth,
                as_of=as_of,
                limit=limit,
            ),
            operation="neighbours",
            allow_degraded=False,
        )
        return [GraphFactRecord.from_row(row) for row in rows]

    async def find_paths(
        self,
        source_id: str,
        target_id: str,
        *,
        tenant_id: str,
        max_hops: int = 3,
        edge_types: Sequence[Any] = (),
        limit: int = 5,
    ) -> list[GraphPath]:
        """Shortest connections between two entities."""
        rows = await self._fetch(
            lambda: templates.paths_between(
                tenant_id=tenant_id,
                source_id=source_id,
                target_id=target_id,
                max_hops=max_hops,
                edge_types=tuple(edge_types) or templates.TRAVERSABLE_EDGE_TYPES,
                limit=limit,
            ),
            operation="find_paths",
            allow_degraded=False,
        )
        return [GraphPath.from_row(row) for row in rows]

    async def subgraph(
        self,
        entity_ids: Sequence[str],
        *,
        tenant_id: str,
        depth: int = 1,
        limit: int = 50,
    ) -> list[GraphFactRecord]:
        """The induced edge set around a group of entities. Backs the graph canvas."""
        rows = await self._fetch(
            lambda: templates.subgraph_edges(
                tenant_id=tenant_id, entity_ids=entity_ids, depth=depth, limit=limit
            ),
            operation="subgraph",
            allow_degraded=False,
        )
        return [GraphFactRecord.from_row(row) for row in rows]

    # ------------------------------------------------------- translation --

    async def _fetch(
        self,
        build: Any,
        *,
        operation: str,
        allow_degraded: bool,
    ) -> list[dict[str, Any]]:
        """Build, run, and map every failure onto a kernel exception.

        The query is built inside the try block on purpose. `GraphSchemaError` is
        raised at *build* time -- an out-of-range limit, an empty tenant, an
        `UNKNOWN` entity type -- and building outside would let it escape as a raw
        `ValueError` and become a 500. It is a 422: the caller sent something
        invalid, and the message says which.
        """
        try:
            query = build()
        except GraphSchemaError as error:
            raise ValidationError(str(error)) from error

        try:
            return await self.client.fetch(query)
        except GraphUnavailableError as error:
            if allow_degraded:
                _log.warning("graph.degraded", operation=operation, error=str(error))
                return []
            raise DependencyUnavailableError.for_store("neo4j", error) from error
        except GraphQueryError as error:
            # 502, not 503, and not 500. The graph answered -- it rejected the
            # statement. Retrying will not help, so a 503 (which clients retry)
            # would turn one bug into a loop; a 500 would send an engineer to
            # look at infrastructure that is working perfectly.
            _log.error("graph.query_rejected", operation=operation, error=str(error))
            raise ExternalServiceError(
                f"the knowledge graph rejected the {operation} query",
                details={"operation": operation},
            ) from error


# --------------------------------------------------------------------------- #
# Composition helper
# --------------------------------------------------------------------------- #


def build_graph_service(
    *,
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
) -> GraphService:
    """Wire a `GraphService` against the process-wide Neo4j driver.

    The composition root, and the only place `backend.db.neo4j` and `graph/` meet.
    Imported inside the function so that importing this module does not pull in
    the `neo4j` driver -- a test that only needs the return types should not need
    the package installed, which is the same reasoning as `backend/main.py`'s
    lazy disposer imports.
    """
    from backend.db.neo4j import read_session
    from graph.client import (
        DEFAULT_MAX_ATTEMPTS,
        DEFAULT_QUERY_TIMEOUT_SECONDS,
        read_runner_from_session_factory,
    )

    client = GraphClient(
        read_runner_from_session_factory(read_session),
        timeout_seconds=timeout_seconds or DEFAULT_QUERY_TIMEOUT_SECONDS,
        max_attempts=max_attempts or DEFAULT_MAX_ATTEMPTS,
    )
    return GraphService(client)
