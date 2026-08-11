"""Unit tests for the `BaseConnector` template method.

The template is where the six-stage contract is actually enforced, so these
tests target the behaviours a connector author could otherwise get wrong without
noticing:

- a malformed record costs one record, not the page;
- dropping and failing-to-map are counted separately;
- duplicates are suppressed but near-duplicates are *not* dropped here;
- the overlap window is applied on resume;
- an unreadable cursor triggers a bounded re-sync rather than misinterpretation;
- `aclose()` runs even when the consumer abandons the generator part-way.

Everything runs against in-memory fakes. No Redis, no network -- which is the
whole point of `connectors/` being forbidden from importing `backend/`
(`docs/architecture.md` §6.2 rule 2).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import pytest

from connectors.base import BaseConnector
from connectors.exceptions import NormalizationError
from connectors.protocol import (
    Credentials,
    Cursor,
    FetchPage,
    RateLimitHint,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
    SyncMode,
)
from models.enums import AuthType, Platform, SourceCategory
from models.lineage import Lineage
from models.signal import Content, Signal

pytestmark = pytest.mark.unit

T0 = datetime(2026, 7, 28, 12, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeLimiter:
    """Records acquisitions instead of talking to Redis."""

    def __init__(self) -> None:
        self.acquired: list[list[str]] = []
        self.observed: list[RateLimitHint] = []

    async def acquire(
        self, keys: Sequence[str], *, timeout_seconds: float | None = None
    ) -> None:
        self.acquired.append(list(keys))

    async def observe(self, keys: Sequence[str], hint: RateLimitHint) -> None:
        self.observed.append(hint)


class FakeDedup:
    """In-memory seen-set. TTL is recorded but not enforced."""

    def __init__(self) -> None:
        self.keys: dict[str, int] = {}

    async def seen(self, key: str) -> bool:
        return key in self.keys

    async def mark(self, key: str, ttl_seconds: int) -> None:
        self.keys[key] = ttl_seconds


class DemoConnector(BaseConnector):
    """A connector that returns whatever the test hands it."""

    slug = "demo"
    platform = Platform.RSS
    category = SourceCategory.NEWS
    auth_type = AuthType.NONE
    rate_limit = RateLimitPolicy(requests_per_minute=60)
    overlap_seconds = 300

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        self.pages: list[FetchPage] = []
        self.authenticated = 0
        self.closed = 0
        self.seen_cursor: Cursor | None = None
        self.bad_ids: set[str] = set()
        self.drop_ids: set[str] = set()

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        self.authenticated += 1

    async def aclose(self) -> None:
        self.closed += 1

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        self.seen_cursor = cursor
        for page in self.pages:
            await self.acquire_slot("https://feed.example.com/rss")
            yield page

    async def normalize(self, record: RawRecord) -> Signal | None:
        if record.native_id in self.bad_ids:
            raise NormalizationError("unmappable payload", native_id=record.native_id)
        if record.native_id in self.drop_ids:
            return None
        return Signal.create(
            platform=self.platform,
            native_id=record.native_id,
            timestamp=record.payload.get("published", T0),
            content=Content(text=str(record.payload.get("text", ""))),
            lineage=Lineage(
                pipeline_version="1.0.0",
                connector_slug=self.slug,
                connector_version=self.version,
                sync_run_id=self.ctx.run_id,
                fetched_at=record.fetched_at,
                native_id=record.native_id,
            ),
        )


def make_ctx(**overrides: Any) -> SyncContext:
    defaults: dict[str, Any] = {
        "connector_slug": "demo",
        "account_id": "acct_1",
        "run_id": "run_1",
        "limiter": FakeLimiter(),
        "dedup": FakeDedup(),
    }
    defaults.update(overrides)
    return SyncContext(**defaults)


def record(native_id: str, text: str | None = None, **payload: Any) -> RawRecord:
    """Build a raw record.

    `text` defaults to something unique per `native_id`, which is not laziness:
    layer 2 of the dedup design suppresses records with identical cleaned bodies
    regardless of their id, so a shared default body would make every multi-record
    fixture silently collapse to one and the tests would assert on the wrong
    thing. Tests that *want* that behaviour pass the same `text` explicitly.
    """
    return RawRecord(
        native_id=native_id,
        payload={"text": f"body of {native_id}" if text is None else text,
                 "published": T0, **payload},
        fetched_at=T0,
    )


def page(*records: RawRecord, cursor: Cursor | None = None, **headers: str) -> FetchPage:
    return FetchPage(
        records=list(records),
        cursor=cursor or Cursor(watermark=T0),
        raw_headers=headers,
    )


async def drain(connector: DemoConnector, cursor: Cursor | None = None) -> list[Any]:
    return [batch async for batch in connector.run(cursor)]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


class TestContractShape:
    def test_run_is_final(self) -> None:
        """The stage order is the contract, not a default."""
        assert getattr(BaseConnector.run, "__final__", False)

    def test_only_four_methods_are_abstract(self) -> None:
        """Everything else must have a usable default, or every connector
        re-implements rate limiting and gets it subtly wrong."""
        assert BaseConnector.__abstractmethods__ == frozenset(
            {"from_config", "authenticate", "fetch", "normalize"}
        )

    def test_connector_package_imports_no_backend_or_services(self) -> None:
        """The rule that keeps connectors testable without Redis."""
        import pathlib

        offenders: list[str] = []
        for path in pathlib.Path("connectors").rglob("*.py"):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith(("from backend", "import backend")) or stripped.startswith(
                    ("from services", "import services")
                ):
                    offenders.append(f"{path}:{lineno}")
        assert not offenders, f"connectors/ must not import backend/ or services/: {offenders}"


class TestLifecycle:
    async def test_authenticates_once_and_closes(self) -> None:
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [page(record("a"))]
        await drain(c)
        assert (c.authenticated, c.closed) == (1, 1)

    async def test_closes_even_when_consumer_abandons_the_generator(self) -> None:
        """The leak path: a consumer that stops early must still release the client."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [page(record("a")), page(record("b")), page(record("c"))]

        agen = c.run()
        await agen.__anext__()
        await agen.aclose()

        assert c.closed == 1, "aclose() must run on the generator-close path"

    async def test_acquires_rate_limit_tokens_per_page(self) -> None:
        ctx = make_ctx()
        c = DemoConnector.from_config(ctx, Credentials(account_id="acct_1"))
        c.pages = [page(record("a")), page(record("b"))]
        await drain(c)

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert len(limiter.acquired) == 2
        # connector-wide, per-account, and per-host politeness
        assert limiter.acquired[0] == [
            "os:rl:demo",
            "os:rl:demo:acct_1",
            "os:rl:host:feed.example.com",
        ]

    async def test_backfill_uses_a_separate_bucket(self) -> None:
        """A historical crawl must not crowd out live sync."""
        ctx = make_ctx(mode=SyncMode.BACKFILL)
        c = DemoConnector.from_config(ctx, Credentials(account_id="acct_1"))
        c.pages = [page(record("a"))]
        await drain(c)

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert "os:rl:demo:backfill" in limiter.acquired[0]

    async def test_missing_limiter_fails_open(self) -> None:
        """architecture.md §7.3: outbound limiting degrades open when Redis is down.

        Failing closed here would halt ingestion every time the cache blipped.
        """
        c = DemoConnector.from_config(
            make_ctx(limiter=None), Credentials(account_id="acct_1")
        )
        c.pages = [page(record("a"))]
        batches = await drain(c)
        assert len(batches) == 1


class TestNormalizeAndDedup:
    async def test_emits_one_batch_per_page(self) -> None:
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [page(record("a"), record("b")), page(record("c"))]
        batches = await drain(c)
        assert [len(b) for b in batches] == [2, 1]

    async def test_malformed_record_costs_one_record_not_the_page(self) -> None:
        """Aborting the page would let one bad item block every good item behind
        it permanently -- the cursor would never advance past it."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.bad_ids = {"b"}
        c.pages = [page(record("a"), record("b"), record("c"))]

        batches = await drain(c)
        assert len(batches[0]) == 2
        assert batches[0].stats["dlq"] == 1
        assert batches[0].stats["emitted"] == 2

    async def test_dropped_and_dlq_are_counted_separately(self) -> None:
        """Dropping is expected; failing to map is a defect. Conflating them
        buries real mapping bugs in a counter nobody reads."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.drop_ids = {"a"}
        c.bad_ids = {"b"}
        c.pages = [page(record("a"), record("b"), record("c"))]

        stats = (await drain(c))[0].stats
        assert stats["dropped"] == 1
        assert stats["dlq"] == 1
        assert stats["emitted"] == 1

    async def test_suppresses_a_repeated_identity(self) -> None:
        ctx = make_ctx()
        c = DemoConnector.from_config(ctx, Credentials(account_id="acct_1"))
        c.pages = [page(record("a")), page(record("a"))]

        batches = await drain(c)
        assert len(batches[0]) == 1
        assert len(batches[1]) == 0
        assert batches[1].stats["duplicates"] == 1

    async def test_suppresses_an_exact_repost_under_a_new_id(self) -> None:
        """Same text, different native id: layer 2 of the dedup design."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [page(record("a", text="identical body")), page(record("b", text="identical body"))]

        batches = await drain(c)
        assert len(batches[0]) == 1
        assert batches[1].stats["duplicates"] == 1

    async def test_empty_body_is_not_treated_as_a_duplicate(self) -> None:
        """Otherwise the first empty-bodied item would suppress every later one."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [page(record("a", text=""), record("b", text=""))]
        assert len((await drain(c))[0]) == 2

    async def test_no_dedup_store_means_no_suppression(self) -> None:
        c = DemoConnector.from_config(make_ctx(dedup=None), Credentials(account_id="acct_1"))
        c.pages = [page(record("a")), page(record("a"))]
        batches = await drain(c)
        assert len(batches[0]) == 1 and len(batches[1]) == 1


class TestCursorHandling:
    async def test_applies_the_overlap_window_on_resume(self) -> None:
        """Provider indexes lag their own timestamps; resuming exactly at the
        watermark silently drops late-written records."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [page(record("a"))]

        await drain(c, Cursor(watermark=T0))
        assert c.seen_cursor is not None
        assert c.seen_cursor.watermark == T0 - timedelta(seconds=300)

    async def test_first_run_starts_empty(self) -> None:
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [page(record("a"))]
        await drain(c, None)
        assert c.seen_cursor is not None and c.seen_cursor.is_empty

    async def test_unreadable_cursor_triggers_a_bounded_resync(self) -> None:
        """An unknown shape must never be reinterpreted as if it were current."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [page(record("a"))]

        await drain(c, Cursor(version=999, watermark=T0, checkpoint={"weird": 1}))
        assert c.seen_cursor is not None and c.seen_cursor.is_empty

    def test_watermark_never_moves_backwards(self) -> None:
        cursor = Cursor(watermark=T0)
        assert cursor.advanced_to(watermark=T0 - timedelta(hours=1)).watermark == T0
        assert cursor.advanced_to(watermark=T0 + timedelta(hours=1)).watermark == T0 + timedelta(
            hours=1
        )

    async def test_batch_carries_the_pages_cursor(self) -> None:
        """"These records are durable" and "resume here" must not drift apart."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        later = Cursor(watermark=T0 + timedelta(minutes=5), page_token="p2")
        c.pages = [page(record("a"), cursor=later)]

        assert (await drain(c))[0].cursor.page_token == "p2"


class TestWatermarkNeverRegresses:
    """Regression tests for the compounding rewind bug.

    `_effective_start` deliberately hands `fetch()` a cursor rewound by
    `overlap_seconds`. The obvious connector implementation carries that rewound
    cursor straight back out on any path with no newer record to report -- a 304,
    an exhausted page budget -- and `Cursor.advanced_to` only clamps upward, so it
    cannot repair it.

    The damage compounds: each poll rewinds from the previously-rewound value, so
    an idle feed walks its watermark backwards indefinitely and eventually
    re-fetches its entire history every cycle. Measured before the fix: four idle
    polls moved a watermark back twenty minutes.
    """

    async def test_idle_poll_does_not_rewind(self) -> None:
        """The steady state for a conditional GET, and where this bit hardest."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        # A page with no newer record: the connector reports back what it was given.
        c.pages = [FetchPage(records=[], cursor=Cursor(watermark=T0 - timedelta(seconds=300)))]

        batches = await drain(c, Cursor(watermark=T0))
        assert batches[0].cursor.watermark == T0

    async def test_repeated_idle_polls_do_not_compound(self) -> None:
        """The property that actually matters: no drift over many cycles."""
        stored = T0
        for _ in range(5):
            c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
            c.pages = [
                FetchPage(records=[], cursor=Cursor(watermark=stored - timedelta(seconds=300)))
            ]
            stored = (await drain(c, Cursor(watermark=stored)))[0].cursor.watermark
        assert stored == T0, f"watermark drifted to {stored}"

    async def test_none_watermark_is_raised_to_the_floor(self) -> None:
        """Regressing from a real timestamp to "no watermark" forces a full re-sync."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [FetchPage(records=[], cursor=Cursor(watermark=None, page_token="p2"))]

        batch = (await drain(c, Cursor(watermark=T0)))[0]
        assert batch.cursor.watermark == T0
        assert batch.cursor.page_token == "p2", "the guard must not disturb pagination"

    async def test_genuine_forward_progress_is_preserved(self) -> None:
        """The guard must clamp, not pin."""
        ahead = T0 + timedelta(hours=1)
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [FetchPage(records=[record("a")], cursor=Cursor(watermark=ahead))]

        assert (await drain(c, Cursor(watermark=T0)))[0].cursor.watermark == ahead

    async def test_first_run_has_no_floor(self) -> None:
        """With nothing stored there is nothing to protect, and pinning would be wrong."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [FetchPage(records=[], cursor=Cursor(watermark=None))]
        assert (await drain(c, None))[0].cursor.watermark is None

    async def test_unreadable_cursor_has_no_floor(self) -> None:
        """An unreadable cursor is discarded, so its watermark cannot bind the run."""
        c = DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))
        c.pages = [FetchPage(records=[], cursor=Cursor(watermark=None))]
        batch = (await drain(c, Cursor(version=999, watermark=T0)))[0]
        assert batch.cursor.watermark is None


class TestBudgets:
    async def test_stops_at_max_pages(self) -> None:
        c = DemoConnector.from_config(make_ctx(max_pages=2), Credentials(account_id="acct_1"))
        c.pages = [page(record(str(i))) for i in range(5)]
        assert len(await drain(c)) == 2

    async def test_stops_at_max_records(self) -> None:
        c = DemoConnector.from_config(make_ctx(max_records=2), Credentials(account_id="acct_1"))
        c.pages = [page(record("a")), page(record("b")), page(record("c"))]
        batches = await drain(c)
        assert sum(len(b) for b in batches) == 2


class TestRateLimitHeaderParsing:
    @pytest.fixture
    def connector(self) -> DemoConnector:
        return DemoConnector.from_config(make_ctx(), Credentials(account_id="acct_1"))

    def test_parses_standard_headers(self, connector: DemoConnector) -> None:
        hint = connector.parse_rate_limit(
            {"X-RateLimit-Remaining": "3", "X-RateLimit-Limit": "60"}
        )
        assert hint is not None and hint.remaining == 3 and hint.limit == 60

    def test_parses_retry_after_as_seconds(self, connector: DemoConnector) -> None:
        hint = connector.parse_rate_limit({"Retry-After": "120"})
        assert hint is not None and hint.retry_after_seconds == 120.0

    def test_parses_retry_after_as_http_date(self, connector: DemoConnector) -> None:
        """The form that silently becomes None if you only wrote int()."""
        future = utcnow_plus(60)
        hint = connector.parse_rate_limit({"Retry-After": future})
        assert hint is not None and hint.retry_after_seconds is not None
        assert 0 < hint.retry_after_seconds <= 61

    def test_headers_are_case_insensitive(self, connector: DemoConnector) -> None:
        assert connector.parse_rate_limit({"retry-after": "5"}) is not None

    def test_returns_none_when_nothing_useful_is_present(
        self, connector: DemoConnector
    ) -> None:
        assert connector.parse_rate_limit({"Content-Type": "application/json"}) is None

    def test_garbage_values_do_not_raise(self, connector: DemoConnector) -> None:
        """A provider sending nonsense must not fail the run."""
        assert connector.parse_rate_limit({"Retry-After": "soon"}) is None

    async def test_provider_hint_is_fed_back_into_the_bucket(self) -> None:
        """Provider truth beats local estimate."""
        ctx = make_ctx()
        c = DemoConnector.from_config(ctx, Credentials(account_id="acct_1"))
        c.pages = [page(record("a"), **{"X-RateLimit-Remaining": "3"})]
        await drain(c)

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert limiter.observed and limiter.observed[0].remaining == 3


class TestSafetyDeclarations:
    def test_tos_review_defaults_to_false_but_is_declarable(self) -> None:
        """A boolean in the class declaration is checkable; a comment is not."""
        assert DemoConnector.requires_tos_review is False

        class Scraper(DemoConnector):
            slug = "scraper"
            requires_tos_review = True

        assert Scraper.requires_tos_review is True

    def test_credentials_never_render_secrets(self) -> None:
        """A ConnectorError carrying these would otherwise log them."""
        creds = Credentials(account_id="acct_1", secrets={"client_secret": "hunter2"})
        assert "hunter2" not in repr(creds)
        assert "redacted" in repr(creds)

    def test_require_names_the_missing_key(self) -> None:
        creds = Credentials(account_id="acct_1", secrets={})
        with pytest.raises(KeyError, match="client_secret"):
            creds.require("client_secret")


def utcnow_plus(seconds: int) -> str:
    """An RFC 9110 HTTP-date `seconds` in the future."""
    from email.utils import format_datetime

    return format_datetime(datetime.now(UTC) + timedelta(seconds=seconds), usegmt=True)
