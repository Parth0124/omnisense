"""`/api/v1/graph/*` -- entity search and relationship reads.

Thin, and that is the design rather than an accident. Every handler here does the
same four things: take the caller's tenant from the `Principal`, call one method
on `services/graph_service.py`, map the domain object onto a wire model, and
return. There is no query building, no Cypher, no error classification -- those
live one layer down, where they can be tested without an HTTP client.

Two decisions worth stating.

**Tenant comes from the token, never from the request.** There is no
`tenant_id` query parameter anywhere in this module, and there must never be one.
`Principal.tenant_id` is derived from a signed token, so the only way to read
another tenant's graph is to hold a token for it. A tenant parameter -- however
carefully validated -- turns an authorization boundary into a filter, and the
first handler that forgets to cross-check it leaks the whole graph.

**Search and reads are separated by failure semantics, not by resource.**
`GET /graph/search` returns 503 when Neo4j is unreachable; the neighbourhood read
behind the canvas returns what it can. That asymmetry is deliberate and it lives
in `services/graph_service.py` as an `allow_degraded` argument. It is repeated
here only in the response descriptions, so a client integrator can see which
endpoints can thin out under load and which will simply fail.

`docs/api-reference.md` §3.5.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from backend.api.deps import Principal, require_scopes
from backend.core.exceptions import ValidationError
from backend.schemas.common import problem_responses
from backend.schemas.graph import (
    MAX_GRAPH_QUERY_CHARS,
    CompetitorItem,
    CompetitorsResponse,
    EntityDetail,
    EntityHit,
    EntitySearchResponse,
    GraphEdge,
    GraphNode,
    GraphPathItem,
    GraphPathsResponse,
    OwnershipChainResponse,
    RelationshipBasis,
    SignalMentionItem,
    SubgraphRequest,
    TopicActivityItem,
    SubgraphResponse,
)
from models.enums import EntityType
from services.graph_service import (
    Competitor,
    Entity,
    GraphFactRecord,
    GraphService,
    build_graph_service,
)

__all__ = ["get_graph_service", "router"]

router = APIRouter(prefix="/graph", tags=["graph"])


async def get_graph_service() -> GraphService:
    """The service, wired to the process-wide Neo4j driver.

    A dependency rather than a module-level singleton so a test can override it
    with `app.dependency_overrides[get_graph_service]` and exercise every route
    against a fake reader with no database. Construction is cheap -- the driver
    underneath is the lazily-created singleton, and this only wraps it.
    """
    return build_graph_service()


GraphDep = Annotated[GraphService, Depends(get_graph_service)]
ReaderPrincipal = Annotated[Principal, Depends(require_scopes("graph:read"))]


def _parse_entity_types(raw: list[str] | None) -> list[EntityType] | None:
    """Turn repeated `?type=` values into enum members, rejecting unknowns.

    Rejecting rather than degrading, and this is the one place in the read path
    where that is right. `EntityType` is a `TolerantStrEnum`, so
    `EntityType("Compnay")` yields `UNKNOWN` instead of raising -- correct when
    *reading* a label a newer producer wrote, and wrong here: a typo would become
    a filter that matches nothing, and the caller would receive an empty list and
    conclude there are no companies.
    """
    if not raw:
        return None
    known = {member.value.casefold(): member for member in EntityType if member is not EntityType.UNKNOWN}
    resolved: list[EntityType] = []
    unknown: list[str] = []
    for value in raw:
        member = known.get(value.strip().casefold())
        if member is None:
            unknown.append(value)
        elif member not in resolved:
            resolved.append(member)
    if unknown:
        raise ValidationError(
            f"unknown entity type(s) {sorted(unknown)}",
            details={"allowed": sorted(known)},
        )
    return resolved or None


def _entity_hit(entity: Entity) -> EntityHit:
    return EntityHit(
        id=entity.id,
        name=entity.name,
        type=entity.type.value,
        description=entity.description,
        aliases=list(entity.aliases[:5]),
        source_count=entity.source_count,
        score=entity.score,
    )


def _competitor_item(competitor: Competitor) -> CompetitorItem:
    return CompetitorItem(
        id=competitor.id,
        name=competitor.name,
        type=competitor.type.value,
        strength=competitor.strength,
        # `RelationshipBasis` is a plain `StrEnum`, so an unrecognised basis
        # would raise here and turn one odd edge into a 500 for the whole
        # response. Mapped explicitly to UNKNOWN instead.
        basis=(
            RelationshipBasis(competitor.basis)
            if competitor.basis in set(RelationshipBasis)
            else RelationshipBasis.UNKNOWN
        ),
        market=competitor.market,
        confidence=competitor.confidence,
        evidence_count=competitor.evidence_count,
        valid_from=competitor.valid_from,
        valid_to=competitor.valid_to,
        citations=list(competitor.citations[:5]),
    )


@router.get(
    "/search",
    summary="Fulltext entity search across name, aliases and description.",
    response_model=EntitySearchResponse,
    responses=problem_responses(400, 401, 403, 422, 503),
)
async def search_entities(
    principal: ReaderPrincipal,
    service: GraphDep,
    q: Annotated[
        str,
        Query(
            min_length=1,
            max_length=MAX_GRAPH_QUERY_CHARS,
            description="Search terms. Matches canonical name, aliases and description.",
        ),
    ],
    type: Annotated[  # noqa: A002 -- the query parameter is named `type` in §3.5
        list[str] | None,
        Query(description="Repeatable entity-type filter, e.g. `?type=Company&type=Product`."),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> EntitySearchResponse:
    """Search the entity index.

    Returns 503 rather than an empty list when the graph is unreachable. This
    endpoint *is* the graph, so an empty result caused by an outage is
    indistinguishable from "no such company" -- a wrong answer wearing the
    clothes of a right one.
    """
    results = await service.search(
        tenant_id=principal.tenant_id,
        query=q,
        entity_types=_parse_entity_types(type),
        limit=limit,
    )
    hits = [_entity_hit(entity) for entity in results]
    return EntitySearchResponse(query=q, results=hits, total=len(hits))


@router.get(
    "/entities/{entity_id}",
    summary="One entity in full.",
    response_model=EntityDetail,
    responses=problem_responses(401, 403, 404, 503),
)
async def get_entity(
    entity_id: str,
    principal: ReaderPrincipal,
    service: GraphDep,
) -> EntityDetail:
    entity = await service.get_entity(tenant_id=principal.tenant_id, entity_id=entity_id)
    return EntityDetail(
        id=entity.id,
        name=entity.name,
        type=entity.type.value,
        description=entity.description,
        aliases=list(entity.aliases),
        confidence=entity.confidence,
        source_count=entity.source_count,
        first_seen=entity.first_seen,
        last_seen=entity.last_seen,
        pagerank_score=entity.pagerank_score,
        community_id=entity.community_id,
        analytics_are_stale=entity.analytics_are_stale,
    )


@router.get(
    "/entities/{name}/competitors",
    summary="Rivals of a company or product, valid at an instant.",
    response_model=CompetitorsResponse,
    responses=problem_responses(401, 403, 422, 503),
)
async def get_competitors(
    name: str,
    principal: ReaderPrincipal,
    service: GraphDep,
    as_of: Annotated[
        datetime | None,
        Query(
            description=(
                "Read the graph as it was believed at this instant. Defaults to "
                "now. Must carry a timezone offset."
            )
        ),
    ] = None,
    min_confidence: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> CompetitorsResponse:
    """Competitors by name or alias.

    Matched by name rather than by id because that is how a caller arrives here:
    from a search box or a report, holding a string. Aliases are matched too, so
    "Big Blue" finds IBM's rivals.
    """
    resolved_as_of = as_of or datetime.now(UTC)
    if as_of is not None and as_of.tzinfo is None:
        # A naive datetime would be compared against UTC values in Neo4j and be
        # silently wrong by the server's offset. Rejecting is the only honest
        # option: guessing UTC would be right in one deployment and wrong in the
        # next.
        raise ValidationError("as_of must include a timezone offset, e.g. 2026-08-06T00:00:00Z")

    results = await service.competitors(
        tenant_id=principal.tenant_id,
        name=name,
        as_of=resolved_as_of,
        min_confidence=min_confidence,
        limit=limit,
    )
    items = [_competitor_item(competitor) for competitor in results]
    return CompetitorsResponse(
        subject=name, as_of=resolved_as_of, results=items, total=len(items)
    )


@router.get(
    "/entities/{entity_id}/signals",
    summary="Signals evidencing an entity, most salient first.",
    response_model=list[SignalMentionItem],
    responses=problem_responses(401, 403, 503),
)
async def get_entity_signals(
    entity_id: str,
    principal: ReaderPrincipal,
    service: GraphDep,
    min_salience: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> list[SignalMentionItem]:
    """The citation path.

    Never degrades. A citation list that came back empty because the graph was
    briefly unreachable produces a claim with no visible support, which is
    exactly the failure `services/evidence_service.py` exists to prevent.
    """
    mentions = await service.signals_for_entity(
        tenant_id=principal.tenant_id,
        entity_id=entity_id,
        min_salience=min_salience,
        limit=limit,
    )
    return [
        SignalMentionItem(
            signal_id=mention.signal_id,
            salience=mention.salience,
            sentiment=mention.sentiment,
            mention_text=mention.mention_text,
            observed_at=mention.observed_at,
        )
        for mention in mentions
    ]


@router.get(
    "/companies/{company_id}/ownership",
    summary="Who ultimately owns a company, following closed acquisitions.",
    response_model=OwnershipChainResponse,
    responses=problem_responses(401, 403, 422, 503),
)
async def get_ownership(
    company_id: str,
    principal: ReaderPrincipal,
    service: GraphDep,
    as_of: Annotated[datetime | None, Query()] = None,
) -> OwnershipChainResponse:
    """Follows only *closed* acquisitions.

    Rumoured and announced deals are in the graph, because a rumour is
    intelligence worth having -- but an ownership chain built through one is a
    statement of fact about a transaction that may never happen.
    """
    resolved_as_of = as_of or datetime.now(UTC)
    chain = await service.ownership_chain(
        tenant_id=principal.tenant_id, company_id=company_id, as_of=resolved_as_of
    )
    if chain is None:
        return OwnershipChainResponse(
            company_id=company_id, as_of=resolved_as_of, is_independent=True
        )
    return OwnershipChainResponse(
        company_id=company_id,
        as_of=resolved_as_of,
        chain=list(chain.chain_ids),
        names=list(chain.names),
        hops=chain.hops,
        is_independent=False,
    )


@router.get(
    "/paths",
    summary="Shortest connections between two entities.",
    response_model=GraphPathsResponse,
    responses=problem_responses(401, 403, 422, 503),
)
async def get_paths(
    principal: ReaderPrincipal,
    service: GraphDep,
    source_id: Annotated[str, Query(min_length=1)],
    target_id: Annotated[str, Query(min_length=1)],
    max_hops: Annotated[int, Query(ge=1, le=4)] = 3,
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> GraphPathsResponse:
    """"How are these two connected?"

    Capped at four hops. Beyond that an entity graph connects everything to
    everything, and the path describes the graph's density rather than a fact
    about the endpoints.
    """
    paths = await service.find_paths(
        source_id,
        target_id,
        tenant_id=principal.tenant_id,
        max_hops=max_hops,
        limit=limit,
    )
    items = [
        GraphPathItem(
            entity_ids=list(path.entity_ids),
            entity_names=list(path.entity_names),
            predicates=[predicate.value for predicate in path.predicates],
            hops=path.hops,
            confidence=path.confidence,
        )
        for path in paths
    ]
    return GraphPathsResponse(
        source_id=source_id, target_id=target_id, paths=items, connected=bool(items)
    )


@router.get(
    "/topics/activity",
    summary="Topic mention volume over a window. Backs the trends view.",
    response_model=list[TopicActivityItem],
    responses=problem_responses(401, 403, 422, 503),
)
async def get_topic_activity(
    principal: ReaderPrincipal,
    service: GraphDep,
    window_days: Annotated[int, Query(ge=1, le=365)] = 30,
    min_salience: Annotated[float, Query(ge=0.0, le=1.0)] = 0.3,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
) -> list[TopicActivityItem]:
    """Which topics are being talked about, and how much.

    Degrades to an empty list when the graph is unreachable, unlike
    `/graph/search`. The distinction is that this is a *summary* view -- an empty
    trends page during a Neo4j outage is a thin page, while an empty search
    result is a wrong answer to a specific question.

    The window is computed here rather than in the query text so two calls a
    second apart read the same window; server-clock arithmetic inside Cypher
    would make a paginated result drop or repeat rows across pages.
    """
    from datetime import timedelta

    until = datetime.now(UTC)
    rows = await service.topic_activity(
        tenant_id=principal.tenant_id,
        since=until - timedelta(days=window_days),
        until=until,
        min_salience=min_salience,
        limit=limit,
        allow_degraded=True,
    )
    return [
        TopicActivityItem(
            topic_id=row.topic_id,
            topic=row.topic,
            mentions=row.mentions,
            avg_sentiment=row.avg_sentiment,
            last_mentioned_at=row.last_mentioned_at,
        )
        for row in rows
    ]


@router.post(
    "/subgraph",
    summary="The induced edge set around a group of entities.",
    response_model=SubgraphResponse,
    responses=problem_responses(401, 403, 422, 503),
)
async def get_subgraph(
    payload: SubgraphRequest,
    principal: ReaderPrincipal,
    service: GraphDep,
) -> SubgraphResponse:
    """Backs the graph canvas.

    A POST because the seed set is a list that can run to fifty ids -- a GET
    would put them in a query string, where proxies truncate at lengths that vary
    by deployment and the failure appears as a partially-drawn graph.

    Nodes are deduplicated and returned separately from edges, because an entity
    joined by six edges appears once in a node list and six times inside nested
    edge objects; a client rendering the nested form draws six overlapping copies
    of it.
    """
    facts = await service.subgraph(
        payload.entity_ids,
        tenant_id=principal.tenant_id,
        depth=payload.depth,
        limit=payload.limit,
    )
    return _to_subgraph(facts, limit=payload.limit)


def _to_subgraph(facts: list[GraphFactRecord], *, limit: int) -> SubgraphResponse:
    """Collapse an edge list into deduplicated nodes plus edges.

    `truncated` is derived from whether the service returned exactly `limit`
    edges. That is a heuristic -- a neighbourhood with precisely `limit` edges
    reports itself truncated when it is complete -- and it errs in the direction
    that matters: telling a client its picture might be partial when it is whole
    costs a caveat, while the reverse presents a subset as the full
    neighbourhood and invites a conclusion drawn from missing data.
    """
    nodes: dict[str, GraphNode] = {}
    edges: list[GraphEdge] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for fact in facts:
        for entity_id, name in (
            (fact.subject_id, fact.subject_name),
            (fact.object_id, fact.object_name),
        ):
            if entity_id and entity_id not in nodes:
                nodes[entity_id] = GraphNode(id=entity_id, name=name or entity_id, type="Unknown")

        key = (fact.subject_id, fact.predicate.value, fact.object_id)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append(
            GraphEdge(
                source=fact.subject_id,
                target=fact.object_id,
                predicate=fact.predicate.value,
                confidence=fact.confidence or None,
                valid_from=fact.valid_from,
                valid_to=fact.valid_to,
                supporting_signal_ids=list(fact.supporting_signal_ids[:5]),
            )
        )

    return SubgraphResponse(
        nodes=list(nodes.values()),
        edges=edges,
        truncated=len(facts) >= limit,
    )
