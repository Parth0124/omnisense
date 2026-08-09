"""`/api/v1/signals` -- the corpus, read (`docs/api-reference.md` §4.7).

Thin over `services/signal_service.py`, which already owns the filter algebra, the
cursor and the tenant scoping. What this module adds is the HTTP shape and one
translation the service deliberately does not do: mapping `SignalView` onto a
strict response model so an internal field cannot become public by being added to
a domain model.

**The AND/OR asymmetry is the thing to get right here.** §4.7 specifies that
repeated `platform`, `source`, `topic` and `language` are OR *within* a parameter
and AND *across* parameters, while repeated `entity_id` is an AND -- "signals
mentioning **all** listed entities". Getting `entity_id` backwards turns "posts
about both Acme and Globex" into "posts about either", which is the difference
between a competitor comparison and a noise pile. The asymmetry lives in
`SignalQuery`; this module's job is to hand it the right sets and not to
second-guess it.

**Cursor pagination, never offsets.** The corpus is written continuously, so an
offset shifts under the reader: rows inserted between page one and page two push
everything down and the client sees a duplicate or misses a row, silently. The
cursor embeds the filter fingerprint so that changing a filter mid-pagination is
*rejected* rather than resumed into a different result set.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from backend.api.deps import CursorPage, Principal, pagination, require_scopes, upstream
from backend.core.exceptions import NotFoundError, ValidationError
from backend.schemas.common import problem_responses
from backend.schemas.signal import (
    MAX_SEARCH_CHARS,
    MAX_TEXT_CHARS,
    AuthorRef,
    EngagementCounts,
    EntityMentionItem,
    SentimentBand,
    SentimentSummary,
    SignalDetail,
    SignalItem,
    SignalPageResponse,
    TopicScoreItem,
)
from models.enums import Platform, SentimentLabel, SourceCategory
from services.signal_service import SignalQuery, SignalService, SignalSort, SortOrder

__all__ = ["get_signal_service_dep", "router"]

router = APIRouter(prefix="/signals", tags=["signals"])

ReaderPrincipal = Annotated[Principal, Depends(require_scopes("signals:read"))]


async def get_signal_service_dep(
    principal: ReaderPrincipal,
) -> SignalService:
    """The service, built per request against the process session factory.

    A separate dependency from `deps.get_signal_service` so a test can override
    exactly this router's service without affecting any other. Construction is
    cheap -- the factory underneath is the process-wide lazy singleton.
    """
    from backend.db.session import get_sessionmaker

    return SignalService(get_sessionmaker())


ServiceDep = Annotated[SignalService, Depends(get_signal_service_dep)]


# --------------------------------------------------------------------------- #
# Enum parsing
# --------------------------------------------------------------------------- #


def _parse_enum_list(
    values: list[str] | None, enum_cls: Any, name: str
) -> frozenset[Any]:
    """Turn repeated query values into enum members, rejecting unknowns.

    Rejecting, not degrading. `Platform` and `SourceCategory` are
    `TolerantStrEnum`s, so `Platform("redit")` yields `UNKNOWN` rather than
    raising -- correct when *reading* a row written by a newer producer, and
    wrong here. A typo would become a filter matching only rows whose platform
    this build does not recognise, which is none of them, and the caller would
    receive an empty page and conclude the corpus has no Reddit content.
    """
    if not values:
        return frozenset()
    known = {
        member.value.casefold(): member
        for member in enum_cls
        if getattr(member, "value", None) != "unknown"
    }
    resolved: set[Any] = set()
    unknown: list[str] = []
    for raw in values:
        member = known.get(raw.strip().casefold())
        if member is None:
            unknown.append(raw)
        else:
            resolved.add(member)
    if unknown:
        raise ValidationError(
            f"unknown {name}: {sorted(unknown)}",
            details={"parameter": name, "allowed": sorted(known)},
        )
    return frozenset(resolved)


def _band(score: float) -> SentimentBand:
    """Bucket a sentiment score for the badge.

    Thresholds live here rather than in the UI so two screens in the same product
    cannot disagree about whether -0.15 is negative.
    """
    if score >= 0.25:
        return SentimentBand.POSITIVE
    if score <= -0.25:
        return SentimentBand.NEGATIVE
    return SentimentBand.NEUTRAL


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def _author(view: Any) -> AuthorRef | None:
    author = getattr(view, "author", None)
    if author is None:
        return None
    return AuthorRef(
        handle=getattr(author, "handle", None),
        display_name=getattr(author, "display_name", None),
        is_verified=bool(getattr(author, "is_verified", False)),
    )


def _sentiment(view: Any) -> SentimentSummary | None:
    sentiment = getattr(view, "sentiment", None)
    if sentiment is None:
        return None
    score = getattr(sentiment, "score", None)
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return None
    label = getattr(sentiment, "label", None)
    band = (
        SentimentBand.MIXED
        if label is SentimentLabel.MIXED
        else _band(float(score))
    )
    return SentimentSummary(
        score=float(score), band=band, confidence=getattr(sentiment, "confidence", None)
    )


def _engagement(view: Any) -> EngagementCounts:
    engagement = getattr(view, "engagement", None)
    if engagement is None:
        return EngagementCounts()
    return EngagementCounts(
        reach=int(getattr(engagement, "reach", 0) or 0),
        endorsement=int(getattr(engagement, "endorsement", 0) or 0),
        amplification=int(getattr(engagement, "amplification", 0) or 0),
        discussion=int(getattr(engagement, "discussion", 0) or 0),
        score=getattr(engagement, "score", None),
    )


def _to_item(view: Any, *, full_text: bool = False) -> dict[str, Any]:
    """Project a `SignalView` onto the wire shape's field dict."""
    content = getattr(view, "content", None)
    text = getattr(content, "text", "") or ""
    truncated = not full_text and len(text) > MAX_TEXT_CHARS

    language = getattr(view, "language", None)
    return {
        "id": view.id,
        "platform": getattr(view.platform, "value", str(view.platform)),
        "source": getattr(view.source, "value", str(view.source)),
        "url": getattr(view, "url", None),
        "timestamp": view.timestamp,
        "title": getattr(content, "title", None),
        "text": text[:MAX_TEXT_CHARS] if truncated else text,
        "text_truncated": truncated,
        "language": getattr(language, "code", None) if language else None,
        "author": _author(view),
        "sentiment": _sentiment(view),
        "engagement": _engagement(view),
        "entities": [
            EntityMentionItem(
                entity_id=getattr(mention, "entity_id", None),
                name=getattr(mention, "name", "") or "",
                type=getattr(getattr(mention, "type", None), "value", "unknown"),
                salience=getattr(mention, "salience", None),
            )
            for mention in (getattr(view, "entities", None) or [])[:20]
        ],
        "topics": [
            TopicScoreItem(
                topic=getattr(topic, "topic", "") or "",
                score=float(getattr(topic, "score", 0.0) or 0.0),
            )
            for topic in (getattr(view, "topics", None) or [])[:10]
        ],
        "confidence": float(getattr(view, "confidence", 0.0) or 0.0),
        "is_canonical": bool(getattr(view, "is_canonical", True)),
        "duplicate_of": getattr(view, "duplicate_of", None),
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@router.get(
    "",
    summary="List signals with filters and cursor pagination.",
    response_model=SignalPageResponse,
    responses=problem_responses(400, 401, 403, 422, 502),
)
async def list_signals(
    principal: ReaderPrincipal,
    service: ServiceDep,
    page: Annotated[CursorPage, Depends(pagination)],
    platform: Annotated[
        list[str] | None,
        Query(description="Repeatable. OR within, AND against other parameters."),
    ] = None,
    source: Annotated[list[str] | None, Query(description="Repeatable. OR within.")] = None,
    topic: Annotated[list[str] | None, Query(description="Repeatable. OR within.")] = None,
    language: Annotated[list[str] | None, Query(description="Repeatable. OR within.")] = None,
    entity_id: Annotated[
        list[str] | None,
        Query(
            description=(
                "Repeatable, and an **AND**: signals mentioning *all* listed "
                "entities. Deliberately different from the other repeatable "
                "filters -- see §4.7."
            )
        ),
    ] = None,
    sentiment: Annotated[str | None, Query()] = None,
    min_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    published_after: Annotated[datetime | None, Query()] = None,
    published_before: Annotated[datetime | None, Query()] = None,
    sort: Annotated[SignalSort, Query()] = SignalSort.TIMESTAMP,
    order: Annotated[SortOrder, Query()] = SortOrder.DESC,
) -> SignalPageResponse:
    """List signals for the caller's tenant.

    Naive datetimes are rejected rather than assumed UTC. `signals.timestamp` is
    stored with an offset, so a naive bound would compare against whatever
    timezone the driver assumed and shift the window silently -- correctly in a
    UTC deployment and wrongly in every other one, which is the worst way for a
    bug like this to behave.
    """
    for bound, name in ((published_after, "published_after"), (published_before, "published_before")):
        if bound is not None and bound.tzinfo is None:
            raise ValidationError(f"{name} must include a timezone offset")

    query = SignalQuery(
        platforms=_parse_enum_list(platform, Platform, "platform"),
        sources=_parse_enum_list(source, SourceCategory, "source"),
        entity_ids=frozenset(entity_id or ()),
        topics=frozenset(topic or ()),
        languages=frozenset(language or ()),
        sentiment=(
            _parse_enum_list([sentiment], SentimentLabel, "sentiment").pop()
            if sentiment
            else None
        ),
        min_confidence=min_confidence,
        published_after=published_after,
        published_before=published_before,
        sort=sort,
        order=order,
        tenant_id=principal.tenant_id,
    )

    async with upstream("postgres"):
        result = await service.list_signals(query, limit=page.limit, cursor=page.cursor)

    return SignalPageResponse(
        items=[SignalItem(**_to_item(view)) for view in result.items],
        limit=result.limit,
        next_cursor=result.next_cursor,
        has_more=result.has_more,
    )


@router.get(
    "/search",
    summary="Full-text search across the corpus.",
    response_model=SignalPageResponse,
    responses=problem_responses(400, 401, 403, 422, 502, 503),
)
async def search_signals(
    principal: ReaderPrincipal,
    service: ServiceDep,
    q: Annotated[str, Query(min_length=1, max_length=MAX_SEARCH_CHARS)],
    platform: Annotated[list[str] | None, Query()] = None,
    published_after: Annotated[datetime | None, Query()] = None,
    published_before: Annotated[datetime | None, Query()] = None,
) -> SignalPageResponse:
    """Keyword search.

    Distinct from `/api/v1/graph/search`, which searches *entities*. This searches
    signal text, and the two answer different questions -- "which companies do we
    know about" versus "what has been said".
    """
    query = SignalQuery(
        platforms=_parse_enum_list(platform, Platform, "platform"),
        published_after=published_after,
        published_before=published_before,
        tenant_id=principal.tenant_id,
    )
    async with upstream("opensearch"):
        result = await service.search_signals(q, query)

    return SignalPageResponse(
        items=[SignalItem(**_to_item(view)) for view in result.items],
        limit=result.limit,
        next_cursor=result.next_cursor,
        has_more=result.has_more,
    )


@router.get(
    "/{signal_id}",
    summary="One signal in full.",
    response_model=SignalDetail,
    responses=problem_responses(401, 403, 404, 502),
)
async def get_signal(
    signal_id: str,
    principal: ReaderPrincipal,
    service: ServiceDep,
) -> SignalDetail:
    """Fetch one signal, with its complete text.

    A signal belonging to another tenant is a 404, not a 403. A 403 would confirm
    the id exists, turning this endpoint into an existence oracle for other
    tenants' data -- which is a disclosure even though no content is returned.
    """
    async with upstream("postgres"):
        view = await service.get_signal(signal_id, tenant_id=principal.tenant_id)

    if view is None:
        raise NotFoundError.for_resource("signal", signal_id)

    fields = _to_item(view, full_text=True)
    return SignalDetail(
        **fields,
        keywords=[
            getattr(keyword, "term", "") or ""
            for keyword in (getattr(view, "keywords", None) or [])[:30]
        ],
        media_count=len(getattr(view, "media", None) or []),
        metadata=dict(getattr(view, "metadata", None) or {}),
    )
