"""Unit tests for `services/signal_service.py`: the read side of the corpus.

The properties asserted here are the ones whose violation is *silent*. A filter
that returns the wrong rows is caught the first time a human looks at a
dashboard; a paginator that drops one row in a thousand under concurrent
ingestion is caught by nobody, because the client has no way to know the row
existed.

Four groups, in order of how much a regression would cost:

1. **Keyset pagination is stable under concurrent writes.** A full traversal
   returns every matching Signal exactly once even when new Signals are
   committed between pages -- newer than the reader, older than the reader, and
   tied with the reader on the sort key. The offset-based equivalent is run
   alongside one of these and shown to repeat a row, so the property under test
   is demonstrably not vacuous.
2. **Non-retrievable statuses do not leak.** `duplicate` and `quarantined` rows
   exist in the table (`docs/signal-model.md` §4.3, §5.4) and must not appear in
   a default query, or one press release is returned six times and a quarantined
   record is offered as evidence.
3. **Every documented filter means what `docs/api-reference.md` §4.7 says** --
   including the AND/OR asymmetry between `entity_id` and `topic`, and the
   half-open time window.
4. **A cursor cannot be used against a different query.** §3.4 requires that a
   changed filter invalidate the cursor; resuming anyway pages inside a result
   set the key does not belong to.

The database is the in-memory SQLite from `tests/conftest.py`. Rows are inserted
through the ORM rather than through the pipeline, so these tests exercise the
query layer and nothing else -- no broker, no provider, no container.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.exceptions import ValidationError
from models.enums import Platform, SentimentLabel, SignalStatus, SourceCategory
from models.orm.signal import SignalRow
from services.signal_service import (
    MAX_PAGE_SIZE,
    MalformedCursorError,
    PageCursor,
    SignalQuery,
    SignalService,
    SignalSort,
    SortOrder,
    signal_view_from_row,
)

pytestmark = pytest.mark.unit

BASE_TIME = datetime(2026, 7, 1, tzinfo=UTC)


@pytest.fixture
async def session_factory(orm_engine: AsyncEngine) -> AsyncIterator[
    async_sessionmaker[AsyncSession]
]:
    """A factory configured exactly as `backend/db/session.py` configures its own.

    The service takes a factory rather than a session because a page is one short
    transaction; handing it a live session would test a different lifecycle from
    the one production uses.
    """
    yield async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest.fixture
def service(session_factory: async_sessionmaker[AsyncSession]) -> SignalService:
    return SignalService(session_factory)


def make_row(
    index: int,
    *,
    timestamp: datetime | None = None,
    status: SignalStatus = SignalStatus.ENRICHED,
    platform: Platform = Platform.REDDIT,
    source: SourceCategory = SourceCategory.SOCIAL,
    **overrides: Any,
) -> SignalRow:
    """One `signals` row with defaults that satisfy every check constraint.

    Ids are zero-padded so that lexical id order matches insertion order, which
    is what makes the tiebreak assertions readable: `sig_010` sorts after
    `sig_009` rather than before it.
    """
    values: dict[str, Any] = {
        "id": f"sig_{index:04d}",
        "native_id": f"native-{index}",
        "source": source,
        "platform": platform,
        "url": f"https://example.test/{index}",
        "timestamp": timestamp if timestamp is not None else BASE_TIME + timedelta(hours=index),
        "fetched_at": BASE_TIME + timedelta(days=1),
        "content_title": f"title {index}",
        "content_text": f"body of signal {index}",
        "content_char_count": len(f"body of signal {index}"),
        "content_type": "text/plain",
        "language_code": "en",
        "language_confidence": 0.99,
        "entities": [],
        "topics": [],
        "keywords": [],
        "embeddings": [],
        "engagement": {},
        "confidence": 0.5,
        "signal_metadata": {},
        "lineage": {"connector_version": "1.2.3", "stages": []},
        "status": status,
        "schema_version": 1,
        "pipeline_version": "1.0.0",
        "connector_slug": "reddit",
        "sync_run_id": "run-1",
    }
    values.update(overrides)
    return SignalRow(**values)


async def insert(
    factory: async_sessionmaker[AsyncSession], *rows: SignalRow
) -> None:
    async with factory() as session:
        session.add_all(rows)
        await session.commit()


async def drain(
    service: SignalService,
    query: SignalQuery | None = None,
    *,
    limit: int,
    between_pages: Callable[[], Any] | None = None,
) -> list[str]:
    """Page to exhaustion, returning ids in the order they were served.

    `between_pages` runs after each page, which is where a test commits a new
    Signal to simulate ingestion landing mid-traversal. Bounded by a page counter
    so a paginator that never terminates fails the test instead of hanging the
    suite.
    """
    seen: list[str] = []
    cursor: str | None = None
    for _ in range(50):
        page = await service.list_signals(query, limit=limit, cursor=cursor)
        seen.extend(item.id for item in page.items)
        if between_pages is not None:
            await between_pages()
        if not page.has_more:
            return seen
        assert page.next_cursor is not None
        cursor = page.next_cursor
    raise AssertionError("pagination did not terminate within 50 pages")


# --------------------------------------------------------------------------- #
# 1. Keyset pagination under concurrent ingestion
# --------------------------------------------------------------------------- #


async def test_full_traversal_returns_every_row_exactly_once(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """The baseline: paging a static table is a partition of the result set."""
    await insert(session_factory, *(make_row(i) for i in range(7)))

    seen = await drain(service, limit=3)

    assert seen == [f"sig_{i:04d}" for i in reversed(range(7))]
    assert len(seen) == len(set(seen))


async def test_signal_inserted_mid_pagination_neither_skips_nor_repeats(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """The property the whole cursor design exists for.

    Six Signals, paged three at a time under the default `timestamp DESC`. After
    the first page a *newer* Signal is committed -- exactly what a connector poll
    does while a dashboard is paging. Under `OFFSET 3` the new row occupies slot
    0, every later row shifts down one, and page 2 re-serves the last row of page
    1 while the sixth row is never reached.

    Under a keyset cursor the reader's position is a value, not a count, so the
    late arrival changes nothing about what follows it.
    """
    await insert(session_factory, *(make_row(i) for i in range(6)))

    first = await service.list_signals(limit=3)
    assert [item.id for item in first.items] == ["sig_0005", "sig_0004", "sig_0003"]

    # Ingestion lands between pages, newer than anything the reader has seen.
    await insert(
        session_factory,
        make_row(99, timestamp=BASE_TIME + timedelta(days=5)),
    )

    rest: list[str] = []
    cursor = first.next_cursor
    assert cursor is not None
    while cursor is not None:
        page = await service.list_signals(limit=3, cursor=cursor)
        rest.extend(item.id for item in page.items)
        cursor = page.next_cursor if page.has_more else None

    served = [item.id for item in first.items] + rest
    # No repeat: the late arrival did not shift any already-served row back into
    # view. No skip: every original Signal was served.
    assert len(served) == len(set(served))
    assert {f"sig_{i:04d}" for i in range(6)} <= set(served)
    # The late arrival is not retroactively inserted into a page already served.
    assert "sig_0099" not in served


async def test_offset_pagination_would_have_repeated_a_row(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The failure the cursor avoids, demonstrated rather than asserted by prose.

    Without this, `test_signal_inserted_mid_pagination_neither_skips_nor_repeats`
    proves only that *some* implementation is consistent -- it would pass against
    a table nothing ever writes to. Running the offset query against the same
    interleaving shows the interleaving is genuinely hostile.
    """
    await insert(session_factory, *(make_row(i) for i in range(6)))

    async def offset_page(offset: int) -> list[str]:
        statement = (
            select(SignalRow.id)
            .order_by(SignalRow.timestamp.desc(), SignalRow.id.desc())
            .limit(3)
            .offset(offset)
        )
        async with session_factory() as session:
            return list((await session.execute(statement)).scalars().all())

    page_one = await offset_page(0)
    await insert(session_factory, make_row(99, timestamp=BASE_TIME + timedelta(days=5)))
    page_two = await offset_page(3)

    overlap = set(page_one) & set(page_two)
    assert overlap, "expected the offset scan to repeat a row after a late insert"
    # And the row pushed off the end is never served by either page.
    assert "sig_0000" not in page_one + page_two


async def test_older_signal_inserted_mid_pagination_is_served_exactly_once(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A backfill landing behind the reader must appear once, and skip nothing.

    This is the direction that loses data under offset pagination: an older row
    shifts everything after it *up* by one, so the row that was about to be
    returned is stepped over entirely. Keyset resumption serves the backfilled
    Signal in its correct position and leaves every other row where it was.
    """
    await insert(session_factory, *(make_row(i) for i in range(6)))
    inserted: list[str] = []

    async def backfill() -> None:
        if inserted:
            return
        # Sits between sig_0001 and sig_0002 in time, i.e. behind the reader
        # after the first page but ahead of the pages still to come.
        await insert(
            session_factory,
            make_row(50, timestamp=BASE_TIME + timedelta(hours=1, minutes=30)),
        )
        inserted.append("sig_0050")

    seen = await drain(service, limit=3, between_pages=backfill)

    assert len(seen) == len(set(seen))
    assert {f"sig_{i:04d}" for i in range(6)} <= set(seen)
    assert seen.count("sig_0050") == 1
    # Served in its correct place in the descending order -- immediately before
    # sig_0001, which it post-dates by thirty minutes -- rather than appended or
    # dropped.
    assert seen.index("sig_0050") == seen.index("sig_0001") - 1


async def test_tied_timestamps_paginate_totally_via_the_id_tiebreak(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Ten Signals sharing one timestamp still page without loss.

    Reddit comments on a burst thread genuinely share a timestamp to the second.
    Keyed on `timestamp` alone the cursor cannot express "after this one of the
    ten", so a page boundary inside the tie group either repeats the whole group
    or skips the rest of it. The id tiebreak makes the order total.
    """
    tied = BASE_TIME + timedelta(hours=3)
    await insert(session_factory, *(make_row(i, timestamp=tied) for i in range(10)))

    seen = await drain(service, limit=3)

    assert seen == [f"sig_{i:04d}" for i in reversed(range(10))]


async def test_ascending_order_paginates_forwards(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """`order=asc` flips both the ordering and the keyset comparison together.

    Flipping only one produces an empty second page (predicate excludes
    everything) or an infinite one (predicate excludes nothing).
    """
    await insert(session_factory, *(make_row(i) for i in range(5)))

    seen = await drain(service, SignalQuery(order=SortOrder.ASC), limit=2)

    assert seen == [f"sig_{i:04d}" for i in range(5)]


async def test_engagement_sort_pages_past_signals_with_no_score(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A NULL `engagement_score` must not terminate the scan.

    `engagement_score` is nullable because a platform with no counters gets no
    score. Keyed on the raw column, the cursor key becomes NULL the moment a page
    ends on an unscored Signal, every `key < NULL` is NULL rather than true, and
    the remainder of the corpus becomes unreachable -- silently, with `has_more`
    reporting false. The coalesce in `_sort_column` is what this asserts.
    """
    await insert(
        session_factory,
        make_row(0, engagement_score=0.9),
        make_row(1, engagement_score=0.4),
        make_row(2, engagement_score=None),
        make_row(3, engagement_score=None),
        make_row(4, engagement_score=0.1),
    )

    seen = await drain(service, SignalQuery(sort=SignalSort.ENGAGEMENT), limit=2)

    assert len(seen) == 5
    assert seen[:3] == ["sig_0000", "sig_0001", "sig_0004"]
    assert set(seen[3:]) == {"sig_0002", "sig_0003"}


async def test_has_more_is_false_on_an_exactly_full_final_page(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Four rows at limit 2 is two pages, not three-with-an-empty-one.

    The `limit + 1` probe is what makes this true. A `len(page) == limit`
    heuristic would claim a third page exists and hand out a cursor that returns
    nothing, which reads to a client as an error rather than as the end.
    """
    await insert(session_factory, *(make_row(i) for i in range(4)))

    first = await service.list_signals(limit=2)
    assert first.has_more is True
    second = await service.list_signals(limit=2, cursor=first.next_cursor)

    assert [item.id for item in second.items] == ["sig_0001", "sig_0000"]
    assert second.has_more is False
    assert second.next_cursor is None


# --------------------------------------------------------------------------- #
# 2. Status visibility
# --------------------------------------------------------------------------- #


async def test_duplicate_and_quarantined_are_excluded_by_default(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Only `enriched` and `partial` are retrievable (`docs/signal-model.md` §5.4)."""
    await insert(
        session_factory,
        make_row(0, status=SignalStatus.ENRICHED),
        make_row(1, status=SignalStatus.PARTIAL),
        make_row(2, status=SignalStatus.DUPLICATE, duplicate_of="sig_0000",
                 dedup_cluster_id="clu_1"),
        make_row(3, status=SignalStatus.QUARANTINED),
        make_row(4, status=SignalStatus.RAW),
    )

    page = await service.list_signals(limit=10)

    assert {item.id for item in page.items} == {"sig_0000", "sig_0001"}
    assert all(item.lineage.status.is_retrievable for item in page.items)


async def test_statuses_can_be_widened_for_dedup_inspection(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """The exclusion is a default, not a rule.

    Dedup inspection needs the non-canonical members of a cluster; DLQ triage
    needs quarantined rows. Both state that intent in the query, where a reviewer
    can see it.
    """
    await insert(
        session_factory,
        make_row(0, status=SignalStatus.ENRICHED),
        make_row(1, status=SignalStatus.DUPLICATE, duplicate_of="sig_0000",
                 dedup_cluster_id="clu_1"),
    )

    page = await service.list_signals(
        SignalQuery(statuses=frozenset({SignalStatus.DUPLICATE})), limit=10
    )

    assert [item.id for item in page.items] == ["sig_0001"]


async def test_empty_status_set_is_rejected_rather_than_matching_nothing(
) -> None:
    """`statuses=frozenset()` can never match, so it is a caller bug, not a filter."""
    with pytest.raises(ValidationError):
        SignalQuery(statuses=frozenset())


# --------------------------------------------------------------------------- #
# 3. Documented filters
# --------------------------------------------------------------------------- #


async def test_platform_source_and_language_filters_or_within_and_across(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """`docs/api-reference.md` §4.7: OR within a parameter, AND across parameters."""
    await insert(
        session_factory,
        make_row(0, platform=Platform.REDDIT, source=SourceCategory.SOCIAL,
                 language_code="en"),
        make_row(1, platform=Platform.RSS, source=SourceCategory.NEWS, language_code="en"),
        make_row(2, platform=Platform.RSS, source=SourceCategory.NEWS, language_code="fr"),
    )

    both_platforms = await service.list_signals(
        SignalQuery(platforms=frozenset({Platform.REDDIT, Platform.RSS})), limit=10
    )
    assert {i.id for i in both_platforms.items} == {"sig_0000", "sig_0001", "sig_0002"}

    intersected = await service.list_signals(
        SignalQuery(
            platforms=frozenset({Platform.REDDIT, Platform.RSS}),
            languages=frozenset({"en"}),
        ),
        limit=10,
    )
    assert {i.id for i in intersected.items} == {"sig_0000", "sig_0001"}

    by_source = await service.list_signals(
        SignalQuery(sources=frozenset({SourceCategory.NEWS})), limit=10
    )
    assert {i.id for i in by_source.items} == {"sig_0001", "sig_0002"}


async def test_time_window_is_half_open(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """`from` inclusive, `to` exclusive, so consecutive windows tile exactly.

    A Signal sitting exactly on a boundary must belong to one window and one
    only, or every trend series double-counts it at the seam.
    """
    await insert(
        session_factory,
        make_row(0, timestamp=BASE_TIME),
        make_row(1, timestamp=BASE_TIME + timedelta(hours=1)),
        make_row(2, timestamp=BASE_TIME + timedelta(hours=2)),
    )
    boundary = BASE_TIME + timedelta(hours=1)

    lower = await service.list_signals(
        SignalQuery(published_before=boundary), limit=10
    )
    upper = await service.list_signals(SignalQuery(published_after=boundary), limit=10)

    assert {i.id for i in lower.items} == {"sig_0000"}
    assert {i.id for i in upper.items} == {"sig_0001", "sig_0002"}


async def test_inverted_time_window_is_rejected_at_construction() -> None:
    """An empty window and an empty corpus look identical in the response."""
    with pytest.raises(ValidationError):
        SignalQuery(
            published_after=BASE_TIME + timedelta(days=1), published_before=BASE_TIME
        )


async def test_naive_time_bounds_are_rejected() -> None:
    """`signals.timestamp` carries an offset; a naive bound shifts the window."""
    with pytest.raises(ValidationError):
        SignalQuery(published_after=datetime(2026, 7, 1))


async def test_min_confidence_filters_and_is_range_checked(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    await insert(
        session_factory,
        make_row(0, confidence=0.2),
        make_row(1, confidence=0.8),
    )

    page = await service.list_signals(SignalQuery(min_confidence=0.5), limit=10)
    assert [i.id for i in page.items] == ["sig_0001"]

    with pytest.raises(ValidationError):
        SignalQuery(min_confidence=1.4)


async def test_sentiment_filter_matches_the_label_and_never_a_missing_one(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A Signal whose sentiment stage failed must not be returned as neutral.

    `sentiment` is NULL when stage 5 degraded (`docs/signal-model.md` §5.2).
    "We did not measure" and "we measured no polarity" are different claims, and
    a filter that conflated them would put unanalysed Signals into a neutral
    bucket that a report then describes as measured.
    """
    await insert(
        session_factory,
        make_row(0, sentiment={"polarity": -0.7, "label": "negative"}),
        make_row(1, sentiment={"polarity": 0.0, "label": "neutral"}),
        make_row(2, sentiment=None),
    )

    negative = await service.list_signals(
        SignalQuery(sentiment=SentimentLabel.NEGATIVE), limit=10
    )
    neutral = await service.list_signals(
        SignalQuery(sentiment=SentimentLabel.NEUTRAL), limit=10
    )

    assert [i.id for i in negative.items] == ["sig_0000"]
    assert [i.id for i in neutral.items] == ["sig_0001"]


async def test_entity_ids_are_conjunctive_and_topics_are_disjunctive(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """§4.7 spells these differently and the difference is load-bearing.

    `entity_id` is documented as "Signals mentioning **all** listed entities" --
    the query behind a competitor comparison. `topic` follows the general "OR
    within a parameter" rule. Swapping them turns "posts about both Acme and
    Globex" into "posts about either", which is a different question with a much
    larger answer.
    """

    def mention(entity_id: str) -> dict[str, Any]:
        return {
            "surface": entity_id,
            "type": "product",
            "start": 0,
            "end": 3,
            "candidate_ids": [entity_id],
            "resolved_id": entity_id,
        }

    await insert(
        session_factory,
        make_row(0, entities=[mention("ent_acme")],
                 topics=[{"topic": "install failures", "score": 0.8}]),
        make_row(1, entities=[mention("ent_acme"), mention("ent_globex")],
                 topics=[{"topic": "pricing", "score": 0.6}]),
        make_row(2, entities=[mention("ent_globex")], topics=[]),
    )

    both = await service.list_signals(
        SignalQuery(entity_ids=frozenset({"ent_acme", "ent_globex"})), limit=10
    )
    assert [i.id for i in both.items] == ["sig_0001"]

    either_topic = await service.list_signals(
        SignalQuery(topics=frozenset({"install failures", "pricing"})), limit=10
    )
    assert {i.id for i in either_topic.items} == {"sig_0000", "sig_0001"}


async def test_unresolved_candidate_ids_do_not_satisfy_an_entity_filter(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """The predicate reads `resolved_id`, not the whole mention blob.

    A `LIKE '%ent_acme%'` over the serialized array -- the shortcut this module
    refuses -- would match a mention that merely *considered* the entity and
    resolved elsewhere, or resolved to nothing at all. That turns "signals about
    Acme" into "signals where an extractor once guessed Acme".
    """
    await insert(
        session_factory,
        make_row(
            0,
            entities=[
                {
                    "surface": "acme",
                    "type": "product",
                    "start": 0,
                    "end": 4,
                    "candidate_ids": ["ent_acme", "ent_other"],
                    "resolved_id": None,
                }
            ],
        ),
    )

    page = await service.list_signals(
        SignalQuery(entity_ids=frozenset({"ent_acme"})), limit=10
    )

    assert page.items == []


async def test_has_media_says_it_cannot_be_answered(service: SignalService) -> None:
    """No media column, no `signal_media` table -- so no honest predicate.

    Returning everything, or nothing, would both be answers to a question the
    database cannot answer. The message names the missing table so the next
    person does not go looking for a bug in the filter.
    """
    with pytest.raises(NotImplementedError, match="signal_media"):
        await service.list_signals(SignalQuery(has_media=True), limit=10)


async def test_free_text_search_names_the_missing_backends(service: SignalService) -> None:
    """`q` is hybrid retrieval, and a LIKE would be a fabricated relevance score."""
    with pytest.raises(NotImplementedError, match="opensearch_client"):
        await service.search_signals("install failure")


async def test_relevance_sort_is_rejected_without_a_query(service: SignalService) -> None:
    """A relevance order does not exist in SQL; silently substituting one lies."""
    with pytest.raises(ValidationError, match="relevance"):
        await service.list_signals(SignalQuery(sort=SignalSort.RELEVANCE), limit=10)


async def test_tenant_scoping_hides_another_tenants_rows(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Phase 1 is single-tenant, but the filter is not optional.

    A cross-tenant read is unrepairable after the fact: the data has already been
    shown. `docs/api-reference.md` §3.1 makes it a `403`/`404`, never a leak.
    """
    await insert(
        session_factory,
        make_row(0, tenant_id="default"),
        make_row(1, tenant_id="other"),
    )

    page = await service.list_signals(limit=10)
    assert [i.id for i in page.items] == ["sig_0000"]
    assert await service.get_signal("sig_0001") is None
    assert await service.get_signal("sig_0001", tenant_id="other") is not None


# --------------------------------------------------------------------------- #
# 4. Cursor integrity and limits
# --------------------------------------------------------------------------- #


async def test_cursor_is_rejected_when_the_filters_changed(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """§3.4: changing a filter invalidates the cursor.

    Resuming anyway would apply a key taken from one result set inside a
    different one, returning a page that is neither the second page of the old
    query nor the first of the new one -- and nothing about the response would
    say so.
    """
    await insert(session_factory, *(make_row(i) for i in range(4)))
    first = await service.list_signals(limit=2)
    assert first.next_cursor is not None

    with pytest.raises(MalformedCursorError):
        await service.list_signals(
            SignalQuery(platforms=frozenset({Platform.RSS})),
            limit=2,
            cursor=first.next_cursor,
        )


async def test_cursor_is_rejected_when_the_sort_changed(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A timestamp key is meaningless against a confidence ordering."""
    await insert(session_factory, *(make_row(i) for i in range(4)))
    first = await service.list_signals(limit=2)

    with pytest.raises(MalformedCursorError):
        await service.list_signals(
            SignalQuery(sort=SignalSort.CONFIDENCE), limit=2, cursor=first.next_cursor
        )


@pytest.mark.parametrize(
    "token",
    [
        "not-base64-at-all!!",
        "aGVsbG8",  # valid base64, but the payload is not JSON
        "eyJ2Ijo5OTk5fQ",  # {"v":9999} -- a payload from a future cursor version
        "eyJ2IjoxLCJzIjoidGltZXN0YW1wIiwibyI6ImRlc2MiLCJrIjoieCIsImkiOiJhIiwiZiI6ImIifQ",
    ],
)
async def test_unusable_cursor_tokens_raise_malformed(
    service: SignalService, token: str
) -> None:
    """Every failure mode gets the same opaque error, per §3.4.

    A client cannot act differently on "not base64" than on "issued for another
    filter set"; both mean page from the start. Distinguishing them in the
    response would also describe the payload of a token documented as opaque.
    """
    with pytest.raises(MalformedCursorError):
        await service.list_signals(limit=2, cursor=token)


async def test_cursor_round_trips_its_position() -> None:
    """Encode/decode is lossless for the key, the id and the fingerprint."""
    query = SignalQuery()
    cursor = PageCursor(
        sort=SignalSort.TIMESTAMP,
        order=SortOrder.DESC,
        key=BASE_TIME,
        signal_id="sig_0001",
        fingerprint=query.fingerprint(),
    )

    restored = PageCursor.decode(cursor.encode(), query=query)

    assert restored == cursor


async def test_fingerprint_is_stable_across_set_ordering() -> None:
    """Two equal queries built in different orders must share one cursor space.

    `frozenset` iteration order varies with insertion history, so hashing it
    directly would make a cursor issued by one process unusable by another --
    which under a load balancer is every second request.
    """
    a = SignalQuery(platforms=frozenset({Platform.REDDIT, Platform.RSS}))
    b = SignalQuery(platforms=frozenset({Platform.RSS, Platform.REDDIT}))

    assert a.fingerprint() == b.fingerprint()


@pytest.mark.parametrize("limit", [0, -1, MAX_PAGE_SIZE + 1])
async def test_out_of_range_limits_are_rejected_not_clamped(
    service: SignalService, limit: int
) -> None:
    """§3.4 rejects an oversized limit.

    Clamping would let a client that asked for 500 and got 200 believe it had
    seen the whole collection.
    """
    with pytest.raises(ValidationError):
        await service.list_signals(limit=limit)


# --------------------------------------------------------------------------- #
# 5. Row -> view projection
# --------------------------------------------------------------------------- #


async def test_get_signals_preserves_caller_order_and_drops_missing(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """The caller's order is a ranking; re-sorting it discards retrieval's work."""
    await insert(session_factory, *(make_row(i) for i in range(3)))

    views = await service.get_signals(["sig_0002", "sig_0000", "sig_9999", "sig_0002"])

    assert [v.id for v in views] == ["sig_0002", "sig_0000"]


async def test_get_signal_reads_a_quarantined_row(
    service: SignalService, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Citation resolution must reach a Signal that retrieval would not return.

    A report written last week may cite a Signal that has since been
    quarantined or marked duplicate. Answering "no such Signal" for a row that
    plainly exists would report a `broken_citation` for something that was merely
    reclassified.
    """
    await insert(session_factory, make_row(0, status=SignalStatus.QUARANTINED))

    view = await service.get_signal("sig_0000")

    assert view is not None
    assert view.lineage.status is SignalStatus.QUARANTINED


async def test_promoted_columns_win_over_a_stale_lineage_blob(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The columns are what the filters read, so they decide what the row *is*.

    A row selected by `status IN ('enriched', 'partial')` that rendered as
    `quarantined` would contradict the query that returned it, and a reader has
    no way to tell which half to believe. The blob keeps only what has no column:
    `stages[]`, `connector_version`, the raw-payload sizes.
    """
    row = make_row(
        0,
        status=SignalStatus.ENRICHED,
        pipeline_version="2.0.0",
        lineage={
            "status": "quarantined",
            "pipeline_version": "0.0.1",
            "native_id": "stale-native-id",
            "connector_version": "1.2.3",
            "stages": [],
        },
    )
    await insert(session_factory, row)

    async with session_factory() as session:
        stored = (await session.execute(select(SignalRow))).scalar_one()
    view = signal_view_from_row(stored)

    assert view.lineage.status is SignalStatus.ENRICHED
    assert view.lineage.pipeline_version == "2.0.0"
    assert view.lineage.native_id == "native-0"
    # Fields with no column of their own still come from the blob.
    assert view.lineage.connector_version == "1.2.3"


async def test_view_is_readable_when_the_lineage_blob_is_empty(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A repair script can write columns without a blob; reads must survive it.

    `connector_version` has no column, so it resolves to `"unknown"` -- an
    explicit admission rather than a fabricated version string, which is the
    difference a reader needs when auditing which code produced a Signal.
    """
    await insert(session_factory, make_row(0, lineage={}))

    async with session_factory() as session:
        stored = (await session.execute(select(SignalRow))).scalar_one()
    view = signal_view_from_row(stored)

    assert view.lineage.connector_version == "unknown"
    assert view.lineage.native_id == "native-0"
    assert view.timestamp.tzinfo is not None


async def test_view_carries_content_and_enrichment_from_columns(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The projection is not lossy for the fields a citation depends on."""
    await insert(
        session_factory,
        make_row(
            0,
            content_title="Install failures since 4.2",
            content_text="every clean install has failed",
            content_char_count=30,
            raw_object_key="raw/2026/07/01/abc.json",
            author_payload={"platform_author_id": "u_1", "handle": "u/example"},
            engagement={"score": 0.79, "raw": {"score": 148}},
            engagement_score=0.79,
        ),
    )

    async with session_factory() as session:
        stored = (await session.execute(select(SignalRow))).scalar_one()
    view = signal_view_from_row(stored)

    assert view.content.text == "every clean install has failed"
    assert view.content.raw_ref == "raw/2026/07/01/abc.json"
    assert view.author is not None and view.author.handle == "u/example"
    assert view.engagement.score == pytest.approx(0.79)
    # Media is never persisted (`services/signal_engine/store.py`), so the view
    # says so with an empty list rather than inventing refs.
    assert view.media == []
