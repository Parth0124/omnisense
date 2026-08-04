"""Signal query, filtering and pagination -- the read side of the corpus.

`docs/api-reference.md` §4.7 (`GET /signals`) is the contract this module
implements, and two of its clauses decide the whole design.

**"Offset pagination is not offered -- result sets change under concurrent
ingestion"** (§3.4). That is not a performance preference, it is a correctness
requirement. `LIMIT n OFFSET k` asks the database to *count past* k rows of a
result set that is being written to while the client pages through it. Two
failures follow, and neither raises:

- A Signal committed with a timestamp *newer* than the reader's position shifts
  every later row down by one under a `timestamp DESC` sort, so page 2 re-serves
  the last row of page 1. The client sees a duplicate.
- A Signal committed *older* than the reader's position -- a backfill, a
  connector's overlap re-fetch, a DLQ redrive, all routine here -- shifts rows up,
  and the row that was about to be returned is stepped over. The client never
  sees it, and nothing anywhere records that it was skipped.

Ingestion into `signals` is continuous by construction, so both are the normal
case rather than a race worth ignoring. Keyset (cursor) pagination replaces "skip
k rows" with "resume strictly after this exact key", which is stable under
concurrent writes and is a range scan rather than an O(n) count. The cursor
carries `(sort key, id)` -- the id being the tiebreak that makes the order a
*total* one, since thousands of Signals can share a timestamp to the second and a
non-total order lets tied rows swap places between pages with no concurrent
writer involved at all.

**Non-retrievable statuses are excluded by default.** `docs/signal-model.md` §5.4
makes only `enriched` and `partial` retrievable. `duplicate` rows exist so they
can still contribute graph edges and trend volume (§4.3), and `quarantined` rows
exist so a poisoned record stays inspectable -- but both are still rows in this
table, so a query that forgets to filter them returns six copies of one press
release and offers a quarantined record as evidence. `SignalStatus.is_retrievable`
is the single definition and this module defers to it rather than restating it.

What this module deliberately does not do
-----------------------------------------
`q` (free text) is not answered here. §4.7 says `q` "runs hybrid retrieval when
present", which means OpenSearch and Qdrant -- neither of which is a PostgreSQL
predicate, and both of which are still stubs. `search_signals` says so instead of
quietly degrading to `content_text LIKE '%q%'`, which would return
plausible-looking rows with none of the recall, none of the ranking, and a
`relevance_score` that meant nothing.

Field projection (`include=summary,content,entities,lineage`) is left to
`backend/schemas/signal.py`. Dropping `content.text` from a `SignalView` to save
bytes would make "this Signal has no body" indistinguishable from "you did not
ask for the body", and an empty `Content.text` is a legitimate value for a
media-only post.

Layer note: `services/` (L2). Takes a session factory as a constructor argument
and constructs none, so the unit suite runs the real SQL against in-memory SQLite
with nothing running.
"""

from __future__ import annotations

import base64
import binascii
import enum
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, cast

from sqlalchemy import (
    ColumnElement,
    Select,
    and_,
    func,
    literal,
    or_,
    select,
)
from sqlalchemy import cast as sql_cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.exceptions import OmniSenseError, ValidationError
from backend.core.logging import get_logger
from models.enums import Platform, SentimentLabel, SignalStatus, SourceCategory
from models.lineage import Lineage
from models.orm.mixins import DEFAULT_TENANT
from models.orm.signal import SignalRow
from models.signal import Content, Language, SignalView

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "MAX_PAGE_SIZE",
    "MalformedCursorError",
    "PageCursor",
    "SignalPage",
    "SignalQuery",
    "SignalService",
    "SignalSort",
    "SortOrder",
    "signal_view_from_row",
]

logger = get_logger(__name__)

DEFAULT_PAGE_SIZE: Final = 50
MAX_PAGE_SIZE: Final = 200
"""Page bounds from `docs/api-reference.md` §3.4.

Constants rather than settings: they are part of a published HTTP contract, and a
deployment that quietly raised the ceiling would change what a documented `422`
means for every client. §3.4 is explicit that a `limit` above the maximum is
*rejected*, never silently clamped -- clamping makes a client believe it has
reached the end of a collection when it has only reached the end of a page.
"""

_CURSOR_VERSION: Final = 1
"""Bumped when the cursor payload shape changes.

A cursor is an opaque token a client holds across requests, so a deploy that
changed the payload without a version would decode an old token into the wrong
fields and page from a position nobody asked for. With the version, an old token
is rejected as malformed and the client restarts from page one -- a visible
failure instead of a silent misread.
"""


class MalformedCursorError(OmniSenseError):
    """The cursor is unparseable, stale, or was issued for a different query.

    `400 malformed_request` per `docs/api-reference.md` §3.4 rather than `422`:
    the cursor is opaque, so a client cannot be told which of its fields is
    wrong, and the only available recovery is to page from the start.
    """

    status_code = 400
    code = "malformed_request"
    default_message = "The pagination cursor could not be used for this query."


class SignalSort(enum.StrEnum):
    """Sort key for `GET /signals` (`docs/api-reference.md` §4.7)."""

    TIMESTAMP = "timestamp"
    ENGAGEMENT = "engagement"
    CONFIDENCE = "confidence"
    RELEVANCE = "relevance"
    """Only meaningful alongside `q`, and therefore only through hybrid
    retrieval. A relevance order does not exist in PostgreSQL, and inventing one
    here would return a timestamp order under a relevance label."""


class SortOrder(enum.StrEnum):
    ASC = "asc"
    DESC = "desc"


@dataclass(frozen=True, slots=True)
class SignalQuery:
    """Everything `GET /signals` can constrain, as one immutable value.

    Frozen and fingerprinted by content on purpose: the cursor embeds a digest of
    this object, which is what lets a filter mutated mid-pagination be *rejected*
    rather than silently resumed from a key belonging to a different result set.

    The AND/OR asymmetry is taken verbatim from §4.7 and is not an accident.
    Repeated `platform`, `source_category`, `topic` and `language` are **OR
    within, AND across parameters**, while `entity_id` is documented as "Signals
    mentioning **all** listed entities" -- an AND. Getting that backwards turns
    "posts about both Acme and Globex" into "posts about either", which is the
    difference between a competitor comparison and a noise pile.
    """

    platforms: frozenset[Platform] = frozenset()
    sources: frozenset[SourceCategory] = frozenset()
    entity_ids: frozenset[str] = frozenset()
    topics: frozenset[str] = frozenset()
    languages: frozenset[str] = frozenset()
    sentiment: SentimentLabel | None = None
    min_confidence: float | None = None
    published_after: datetime | None = None
    published_before: datetime | None = None
    has_media: bool | None = None

    statuses: frozenset[SignalStatus] = field(
        default_factory=lambda: frozenset(s for s in SignalStatus if s.is_retrievable)
    )
    """Which lifecycle states may be returned. Retrievable-only by default.

    A parameter rather than a hardcoded set because two callers legitimately need
    the others: the dedup inspector wants the `duplicate` members of a cluster,
    and DLQ triage wants `quarantined`. Those callers then state their intent in
    the query, where it is visible, instead of every other caller having to
    remember to exclude them.
    """

    sort: SignalSort = SignalSort.TIMESTAMP
    order: SortOrder = SortOrder.DESC
    tenant_id: str = DEFAULT_TENANT

    def __post_init__(self) -> None:
        """Reject an incoherent query here, not with an empty result set.

        A `from` after `to` matches nothing, and "no results" is exactly what a
        genuinely empty window also looks like. §4.7 makes it a `422`, which is
        the only answer that tells the caller its query was wrong rather than
        that the corpus was.
        """
        if (
            self.published_after is not None
            and self.published_before is not None
            and self.published_after > self.published_before
        ):
            raise ValidationError(
                f"published_after ({self.published_after.isoformat()}) is later than "
                f"published_before ({self.published_before.isoformat()}); the window "
                "is empty"
            )
        if self.min_confidence is not None and not 0.0 <= self.min_confidence <= 1.0:
            raise ValidationError(
                f"min_confidence must be within [0, 1]; got {self.min_confidence}"
            )
        for bound, name in (
            (self.published_after, "published_after"),
            (self.published_before, "published_before"),
        ):
            if bound is not None and bound.tzinfo is None:
                # `docs/api-reference.md` §3.2 rejects naive datetimes. It matters
                # more here than at the HTTP edge: `signals.timestamp` is stored
                # with an offset, so a naive bound compares against whatever
                # timezone the driver assumed and silently shifts the window.
                raise ValidationError(f"{name} must be timezone-aware")
        if not self.statuses:
            raise ValidationError(
                "statuses is empty, which can never match a row; omit it to get the "
                "retrievable statuses"
            )

    def fingerprint(self) -> str:
        """A stable digest of everything that shapes the result set.

        Embedded in the cursor so that changing a filter mid-pagination is
        rejected (`docs/api-reference.md` §3.4) instead of resuming from a key
        that belongs to a different ordering. Without it, adding `platform=rss`
        to page 2 resumes at page 1's last timestamp inside a *different* result
        set, returning a page that is neither the second page of the old query
        nor the first of the new one -- with nothing for the client to notice.

        Sets are sorted before hashing: `frozenset` iteration order varies with
        insertion history and with `PYTHONHASHSEED`, so hashing it directly would
        make a cursor issued by one process unusable by another.
        """
        payload = {
            "platforms": sorted(p.value for p in self.platforms),
            "sources": sorted(s.value for s in self.sources),
            "entity_ids": sorted(self.entity_ids),
            "topics": sorted(self.topics),
            "languages": sorted(self.languages),
            "sentiment": self.sentiment.value if self.sentiment else None,
            "min_confidence": self.min_confidence,
            "published_after": _iso(self.published_after),
            "published_before": _iso(self.published_before),
            "has_media": self.has_media,
            "statuses": sorted(s.value for s in self.statuses),
            "sort": self.sort.value,
            "order": self.order.value,
            "tenant_id": self.tenant_id,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class PageCursor:
    """The exact position a page resumes from: one sort key plus one id.

    Not an offset, and not a timestamp alone. The id is what makes the ordering
    total -- Reddit comments and RSS items routinely share a `timestamp` to the
    second, and under a non-total order the database is free to return tied rows
    in a different sequence on each query. A page boundary falling inside a tie
    group would then drop or repeat members of that group without any concurrent
    write at all.
    """

    sort: SignalSort
    order: SortOrder
    key: datetime | float
    signal_id: str
    fingerprint: str

    def encode(self) -> str:
        """Serialize to the opaque base64 token clients round-trip.

        Base64url without padding, because the token travels in a query string.
        Deliberately neither signed nor encrypted: it names a public sort
        position, carries no tenant and no identity, and a forged cursor can only
        produce a page of rows the same query would already have returned -- the
        tenant filter comes from the query, never from the cursor.
        """
        payload = {
            "v": _CURSOR_VERSION,
            "s": self.sort.value,
            "o": self.order.value,
            "k": _iso(self.key) if isinstance(self.key, datetime) else self.key,
            "i": self.signal_id,
            "f": self.fingerprint,
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @classmethod
    def decode(cls, token: str, *, query: SignalQuery) -> PageCursor:
        """Parse a token and prove it belongs to `query`.

        Every failure raises the same `MalformedCursorError`, on purpose: a
        client cannot act differently on "not base64" than on "issued for a
        different filter set", and enumerating the reasons would leak the payload
        shape of a token documented as opaque.
        """
        try:
            padded = token + "=" * (-len(token) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded))
        except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
            raise MalformedCursorError("the cursor is not a valid token") from exc

        if not isinstance(payload, dict) or payload.get("v") != _CURSOR_VERSION:
            raise MalformedCursorError("the cursor was issued by an older version")
        if payload.get("f") != query.fingerprint():
            raise MalformedCursorError(
                "the cursor was issued for a different filter or sort; repeat the "
                "original parameters or page from the start"
            )
        try:
            sort = SignalSort(payload["s"])
            order = SortOrder(payload["o"])
            signal_id = str(payload["i"])
            key: datetime | float = (
                datetime.fromisoformat(str(payload["k"]))
                if sort is SignalSort.TIMESTAMP
                else float(payload["k"])
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedCursorError("the cursor payload is not usable") from exc
        if not signal_id:
            raise MalformedCursorError("the cursor carries no tiebreak id")
        return cls(
            sort=sort,
            order=order,
            key=key,
            signal_id=signal_id,
            fingerprint=str(payload["f"]),
        )


@dataclass(frozen=True, slots=True)
class SignalPage:
    """One page, shaped like the `page` envelope in `docs/api-reference.md` §3.4.

    No total count. §3.4 declines to return one and this respects that: `COUNT(*)`
    over a filtered slice of a continuously written table is both unbounded work
    and immediately stale, and a number that is already wrong by the time it
    renders is worse than an absent one, because the UI will do arithmetic on it.
    """

    items: Sequence[SignalView]
    limit: int
    next_cursor: str | None = None
    has_more: bool = False

    def __len__(self) -> int:
        return len(self.items)


class SignalService:
    """Read facade over the `signals` table.

    Holds a session factory rather than a session: a page is one short
    transaction, and a service holding an open session would keep a pooled
    connection alive for as long as anything referenced the service.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list_signals(
        self,
        query: SignalQuery | None = None,
        *,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
    ) -> SignalPage:
        """Return one keyset page of Signals matching `query`.

        Fetches `limit + 1` rows and returns at most `limit`. That extra row is
        how `has_more` is answered without a second query and without a count: if
        it came back, another page exists. Asking `COUNT(*)` instead would scan
        the entire filtered set to answer a boolean.
        """
        query = query or SignalQuery()
        limit = _validate_limit(limit)
        position = PageCursor.decode(cursor, query=query) if cursor else None

        async with self._session_factory() as session:
            # The dialect is read from the bound session because one of the
            # filters (JSON array containment) has no portable spelling, exactly
            # as `services/signal_engine/store.py` reads it for `ON CONFLICT`.
            dialect = session.get_bind().dialect.name
            statement = _build_statement(query, position=position, limit=limit + 1,
                                         dialect=dialect)
            rows = list((await session.execute(statement)).scalars().all())

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        next_cursor: str | None = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = PageCursor(
                sort=query.sort,
                order=query.order,
                key=_sort_key_of(last, query.sort),
                signal_id=last.id,
                fingerprint=query.fingerprint(),
            ).encode()

        return SignalPage(
            items=[signal_view_from_row(row) for row in page_rows],
            limit=limit,
            next_cursor=next_cursor,
            has_more=has_more,
        )

    async def get_signal(
        self, signal_id: str, *, tenant_id: str = DEFAULT_TENANT
    ) -> SignalView | None:
        """Fetch one Signal by id, or `None` if this tenant has no such row.

        Deliberately **not** status-filtered. Citation resolution
        (`services/evidence_service.py`) must be able to read a `duplicate` or
        `quarantined` Signal: a report written last week may cite one, and
        answering "no such Signal" for a row that plainly exists would raise a
        `broken_citation` for a Signal that was merely reclassified.

        `None` rather than a raise, because the caller decides what an absent
        Signal means -- a `404` for the API, a `broken_citation` finding for the
        Critic -- and those are different responses to the same fact.
        """
        statement = select(SignalRow).where(
            SignalRow.id == signal_id, SignalRow.tenant_id == tenant_id
        )
        async with self._session_factory() as session:
            row = (await session.execute(statement)).scalar_one_or_none()
        return signal_view_from_row(row) if row is not None else None

    async def get_signals(
        self, signal_ids: Iterable[str], *, tenant_id: str = DEFAULT_TENANT
    ) -> list[SignalView]:
        """Batch form of `get_signal`, preserving the caller's order.

        Order is preserved because the caller's order is usually a *ranking*:
        retrieval hands over fused, reranked ids, and re-sorting them by id or by
        whatever the database happened to return would discard the ranking the
        entire retrieval stack exists to produce.

        Missing ids are omitted rather than yielding `None` placeholders. A
        Signal can legitimately be gone -- an erasure request hard-deletes the row
        (`docs/security-and-privacy.md`) -- and the caller that cares, the
        citation resolver, compares what came back against what it asked for.
        """
        wanted = list(dict.fromkeys(signal_ids))
        if not wanted:
            return []
        statement = select(SignalRow).where(
            SignalRow.id.in_(wanted), SignalRow.tenant_id == tenant_id
        )
        async with self._session_factory() as session:
            rows = list((await session.execute(statement)).scalars().all())
        by_id = {row.id: row for row in rows}
        return [signal_view_from_row(by_id[i]) for i in wanted if i in by_id]

    async def search_signals(self, q: str, query: SignalQuery | None = None) -> SignalPage:
        """The `q` path of `GET /signals`: hybrid retrieval, then hydration.

        Not implemented, and not faked. §4.7 defines `q` as running hybrid
        retrieval -- BM25 from OpenSearch and ANN from Qdrant, fused by
        `retrieval/hybrid.py`. Both backend clients are still stubs, so there is
        nothing to fan out to.

        The tempting substitute, `content_text LIKE '%q%'`, is precisely why this
        raises instead. It returns rows, so it looks like it works; it has no
        stemming, no term weighting, no semantic recall and no ranking, and the
        `relevance_score` §4.7 promises would be a fabrication. A report built on
        it would cite real Signals in support of claims that retrieval never
        found evidence for.
        """
        raise NotImplementedError(
            "free-text search over Signals needs hybrid retrieval: "
            "retrieval/keyword/opensearch_client.py (BM25) and "
            "retrieval/vector/qdrant_client.py (ANN) are both stubs, so "
            "retrieval/hybrid.py has no backends to fan out to. Filter-only "
            "queries work today through list_signals()."
        )


# --------------------------------------------------------------------------- #
# Statement construction
# --------------------------------------------------------------------------- #


def _build_statement(
    query: SignalQuery, *, position: PageCursor | None, limit: int, dialect: str
) -> Select[tuple[SignalRow]]:
    """Filters, then the keyset predicate, then the total ordering."""
    statement = select(SignalRow).where(*_filters(query, dialect=dialect))
    if position is not None:
        statement = statement.where(_keyset_predicate(query, position))
    return statement.order_by(*_order_by(query)).limit(limit)


def _filters(query: SignalQuery, *, dialect: str) -> list[ColumnElement[bool]]:
    """Every documented `GET /signals` filter, as SQL predicates.

    Tenant first, deliberately: `models/orm/mixins.py` puts `tenant_id` at the
    head of every composite index for exactly this shape of query, and a filter
    list that grows a new leading clause later is how that leading column
    silently stops being used.
    """
    clauses: list[ColumnElement[bool]] = [SignalRow.tenant_id == query.tenant_id]

    if query.has_media is not None:
        raise NotImplementedError(
            "has_media cannot be answered: MediaRef is not persisted. "
            "models/orm/signal.py has no media column and there is no "
            "`signal_media` table, so services/signal_engine/store.py logs "
            "`store.media_dropped` and drops the list at the commit point. "
            "Filtering on media needs that table and an Alembic migration first."
        )

    clauses.append(SignalRow.status.in_(sorted(query.statuses, key=lambda s: s.value)))
    if query.platforms:
        clauses.append(SignalRow.platform.in_(sorted(query.platforms, key=lambda p: p.value)))
    if query.sources:
        clauses.append(SignalRow.source.in_(sorted(query.sources, key=lambda s: s.value)))
    if query.languages:
        clauses.append(SignalRow.language_code.in_(sorted(query.languages)))
    if query.min_confidence is not None:
        clauses.append(SignalRow.confidence >= query.min_confidence)
    if query.published_after is not None:
        # Inclusive lower bound, exclusive upper bound (`docs/api-reference.md`
        # §4.7). Half-open windows tile without overlap, so consecutive daily
        # queries neither double-count a Signal sitting exactly on midnight nor
        # miss one -- which matters because trend volume is counted from these.
        clauses.append(SignalRow.timestamp >= query.published_after)
    if query.published_before is not None:
        clauses.append(SignalRow.timestamp < query.published_before)
    if query.sentiment is not None:
        clauses.append(_sentiment_label_is(query.sentiment))
    for entity_id in sorted(query.entity_ids):
        # One clause per id, so they AND: §4.7, "mentioning **all** listed entities".
        clauses.append(
            _json_objects_contain(SignalRow.entities, "resolved_id", entity_id, dialect=dialect)
        )
    if query.topics:
        clauses.append(
            or_(
                *(
                    _json_objects_contain(SignalRow.topics, "topic", topic, dialect=dialect)
                    for topic in sorted(query.topics)
                )
            )
        )
    return clauses


def _sentiment_label_is(label: SentimentLabel) -> ColumnElement[bool]:
    """`sentiment->>'label' = ?`, spelled portably.

    SQLAlchemy's generic JSON index access compiles to `JSON_EXTRACT` on SQLite
    and to `->>` on PostgreSQL's JSONB, so one expression is correct on both
    without a dialect branch -- unlike array containment below, which has no
    portable spelling at all.

    `sentiment` is nullable, because stage 5 degrades to *no* sentiment rather
    than to neutral (`docs/signal-model.md` §5.2). A NULL column yields NULL
    here, which is equal to nothing, and that is the wanted behaviour: a Signal
    whose sentiment stage failed must not be returned as `neutral`, because "we
    did not measure" and "we measured no polarity" are different claims.
    """
    return cast(
        ColumnElement[bool], SignalRow.sentiment["label"].as_string() == label.value
    )


def _json_objects_contain(
    column: Any, key: str, value: str, *, dialect: str
) -> ColumnElement[bool]:
    """Whether a JSON array of objects holds one whose `key` equals `value`.

    Dialect-dispatched for the same reason `services/signal_engine/store.py`
    dispatches `ON CONFLICT`: there is no portable spelling, and the two real
    ones are genuinely different operations rather than syntax variants.

    - PostgreSQL: JSONB containment `@>`, which a GIN index on the column can
      answer directly. This is the production path and the only one that stays
      sub-linear as the corpus grows.
    - SQLite: a correlated `EXISTS` over `json_each`, a per-row scan of a short
      array. Adequate for the unit suite, and -- what matters -- the same
      *predicate*, so the filter semantics under test are the production ones.

    An unknown dialect raises rather than falling back. The only available
    fallback would be a `LIKE` over the serialized JSON, which matches an entity
    id appearing anywhere in the blob -- inside `candidate_ids`, inside a surface
    string -- and would return Signals mentioning an entity nobody resolved.
    """
    if dialect == "postgresql":
        probe = json.dumps([{key: value}], separators=(",", ":"))
        containment = column.op("@>", is_comparison=True)(sql_cast(literal(probe), JSONB))
        return cast(ColumnElement[bool], containment)
    if dialect == "sqlite":
        elements = func.json_each(column).table_valued("value")
        inner = (
            select(literal(1))
            .select_from(elements)
            .where(func.json_extract(elements.c.value, f"$.{key}") == value)
        )
        return cast(ColumnElement[bool], inner.exists())
    raise NotImplementedError(
        f"no JSON array containment predicate for SQLAlchemy dialect {dialect!r}; "
        "PostgreSQL uses JSONB `@>` and SQLite uses json_each(), and a LIKE over "
        "the serialized array would match ids that were never resolved"
    )


def _sort_column(sort: SignalSort) -> ColumnElement[Any]:
    """The column a sort orders on, normalized so the key is never NULL.

    `engagement_score` is nullable -- a platform exposing no counters gets no
    score -- and NULL breaks keyset pagination in a way no ordering clause
    repairs: `key < :last_key` evaluates to NULL, not true, for every unscored
    row, so the first page boundary that lands on one stops the scan dead and the
    rest of the corpus becomes unreachable. Coalescing to `-1.0` puts unscored
    Signals below every real score (scores are in `[0, 1]`) and keeps the
    comparison total.
    """
    if sort is SignalSort.TIMESTAMP:
        return cast(ColumnElement[Any], SignalRow.timestamp)
    if sort is SignalSort.CONFIDENCE:
        return cast(ColumnElement[Any], SignalRow.confidence)
    if sort is SignalSort.ENGAGEMENT:
        return func.coalesce(SignalRow.engagement_score, -1.0)
    raise ValidationError(
        "sort=relevance is only defined for a free-text query; it is produced by "
        "retrieval ranking, not by a SQL ordering. Use sort=timestamp, engagement "
        "or confidence for a filter-only query."
    )


def _sort_key_of(row: SignalRow, sort: SignalSort) -> datetime | float:
    """The cursor key for a row, matching `_sort_column` exactly.

    The two must agree. A cursor built from a raw `engagement_score` while the
    ordering used `coalesce(..., -1.0)` would resume at `None`, and every
    comparison against it evaluates to NULL.
    """
    if sort is SignalSort.TIMESTAMP:
        return _as_utc(row.timestamp)
    if sort is SignalSort.CONFIDENCE:
        return row.confidence
    if sort is SignalSort.ENGAGEMENT:
        return row.engagement_score if row.engagement_score is not None else -1.0
    raise ValidationError("sort=relevance has no SQL sort key")


def _order_by(query: SignalQuery) -> list[ColumnElement[Any]]:
    """Sort key then id, both in the same direction.

    Both, and in the same direction: the tiebreak has to *extend* the ordering,
    not cut across it. `ORDER BY timestamp DESC, id ASC` paired with the keyset
    predicate below would skip every tied row whose id sorts before the cursor's.
    """
    column = _sort_column(query.sort)
    if query.order is SortOrder.DESC:
        return [column.desc(), SignalRow.id.desc()]
    return [column.asc(), SignalRow.id.asc()]


def _keyset_predicate(query: SignalQuery, position: PageCursor) -> ColumnElement[bool]:
    """"Strictly after this (key, id)", in the query's own sort direction.

    Written as the expanded `(key < k) OR (key = k AND id < i)` rather than the
    row-value form `(key, id) < (k, i)`. The row-value form is tighter and
    PostgreSQL can drive an index scan straight from it, but it requires SQLite
    3.15+ at runtime and the version bundled with a given Python build is not
    something this module can assert. The expanded form is exactly equivalent and
    parses everywhere; the cost is that PostgreSQL may use the composite index
    less efficiently, which is a planner concern rather than a correctness one.
    """
    column = _sort_column(query.sort)
    if query.order is SortOrder.DESC:
        return or_(
            column < position.key,
            and_(column == position.key, SignalRow.id < position.signal_id),
        )
    return or_(
        column > position.key,
        and_(column == position.key, SignalRow.id > position.signal_id),
    )


def _validate_limit(limit: int) -> int:
    """Bound the page size, rejecting rather than clamping (§3.4)."""
    if limit < 1:
        raise ValidationError(f"limit must be at least 1; got {limit}")
    if limit > MAX_PAGE_SIZE:
        raise ValidationError(
            f"limit {limit} exceeds the maximum of {MAX_PAGE_SIZE}; page with a "
            "cursor instead of a larger limit"
        )
    return limit


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _as_utc(value: datetime) -> datetime:
    """Restore the UTC offset the database dropped, if it dropped one.

    PostgreSQL stores these columns as `TIMESTAMP WITH TIME ZONE` and hands back
    aware datetimes, so in production this is a no-op. SQLite has no timezone
    type at all: SQLAlchemy's dialect writes the wall-clock fields and reads them
    back **naive**, which is what the unit suite sees.

    Attaching UTC is a restoration rather than a guess. Every datetime that
    reaches this table passed through `UtcDatetime` (`models/base.py`), which
    rejects naive values outright, so the stored wall clock is UTC by
    construction. Without this the read path fails two ways at once: `SignalView`
    and `Lineage` reject the naive value they just read back, and any comparison
    between one of these and an aware datetime raises `TypeError` at whatever
    call site happens to make it.
    """
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Row -> view projection
# --------------------------------------------------------------------------- #


def signal_view_from_row(row: SignalRow) -> SignalView:
    """Project a `signals` row onto the read model consumers share.

    `SignalView` rather than `Signal` because this is the *consuming* side.
    `models/signal.py` makes the view lenient precisely so a row written by a
    newer `pipeline_version` stays readable during a rolling deploy, whereas
    `Signal`'s identity and source/platform validators would reject it and take
    down the reader rather than the writer that caused the problem.

    `media` is always empty: `MediaRef` has no column and there is no
    `signal_media` table, so `services/signal_engine/store.py` logs
    `store.media_dropped` and discards the list at the commit point. Returning an
    empty list is honest -- the alternative would be inventing refs from nothing.
    """
    return SignalView(
        id=row.id,
        source=row.source,
        platform=row.platform,
        url=row.url,
        author=row.author_payload,
        timestamp=_as_utc(row.timestamp),
        content=Content(
            title=row.content_title,
            text=row.content_text,
            char_count=row.content_char_count,
            truncated=row.content_truncated,
            content_type=row.content_type,
            raw_ref=row.raw_object_key,
            raw_sha256=row.raw_sha256,
        ),
        media=[],
        language=Language(code=row.language_code, confidence=row.language_confidence),
        entities=row.entities or [],
        topics=row.topics or [],
        keywords=row.keywords or [],
        embeddings=row.embeddings or [],
        sentiment=row.sentiment,
        engagement=row.engagement or {},
        confidence=row.confidence,
        metadata=row.signal_metadata or {},
        lineage=_lineage_from_row(row),
    )


def _lineage_from_row(row: SignalRow) -> Lineage:
    """Rebuild `Lineage`, with the promoted columns as the authority.

    The JSON blob supplies what has no column of its own -- `stages[]`,
    `connector_version`, the raw-payload sizes, `confidence_components` -- and the
    columns overwrite everything else. Validating the blob alone would be simpler
    and wrong twice over: a row written by a migration or a repair script may
    carry an empty blob, and a blob that disagrees with the columns would hand
    the caller provenance contradicting the row it just selected. A row returned
    by `status IN ('enriched', 'partial')` must not render as `quarantined`.

    `connector_version` has no column and no model default, so a blob without it
    yields `"unknown"` -- an explicit admission that the version was not
    recorded, which a reader needs in order to tell it apart from one that was.
    """
    blob = dict(row.lineage or {})
    blob.update(
        {
            "schema_version": row.schema_version,
            "pipeline_version": row.pipeline_version,
            "connector_slug": row.connector_slug,
            "connector_version": blob.get("connector_version") or "unknown",
            "sync_run_id": row.sync_run_id or blob.get("sync_run_id") or "",
            "fetched_at": _as_utc(row.fetched_at),
            "native_id": row.native_id,
            "status": row.status,
            "dedup_cluster_id": row.dedup_cluster_id,
            "duplicate_of": row.duplicate_of,
            "raw_object_key": row.raw_object_key,
            "raw_sha256": row.raw_sha256,
        }
    )
    return Lineage.model_validate(blob)
