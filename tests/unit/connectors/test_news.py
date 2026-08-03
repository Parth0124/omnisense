"""Unit tests for the two Phase 1 news-API connectors.

Both sources answer with articles that carry a URL and no id, so both derive
identity by rule 2 of `docs/signal-model.md` §4.1 -- `sha256` of the canonicalized
URL. §7 records that changing how `Signal.id` is derived is "not migratable in
place", which is why the golden ids below are pinned as literals: a diff that
changes one of them is a full re-ingest, and it should be impossible to make by
accident.

What each concern here defends:

- **identity** -- one article spelled two ways (tracking parameters, a default
  port, a fragment) must produce one Signal, and the same URL seen by both
  connectors must produce the same `native_id` under two different platforms;
- **truncation** -- `Content.truncated` caps the `content_integrity` component of
  confidence (§3.5), so it must follow the *record*, not the connector;
- **the wider GDELT overlap** -- 900 seconds rather than 300 must reach the
  outbound request, because the value only matters where it turns into a query
  parameter (`docs/connector-spec.md` §4.1 rule 3);
- **budgets** -- GDELT answers in hundreds, so `max_records` has to narrow the
  request rather than discard the response;
- **the error taxonomy** -- the family of the exception is what the runtime acts
  on (§6), so a 429 that means "wait a minute" and one that means "come back
  tomorrow" must not be the same class.

Everything runs against recorded fixtures under `tests/fixtures/payloads/`
through `respx`. No provider is ever contacted: an unmatched request raises
inside `respx` rather than escaping to the network, so a missing mock fails the
test instead of hitting newsapi.org.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import respx

from connectors import registry
from connectors.base import BaseConnector
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.news.gdelt import GdeltConnector
from connectors.news.news_api import NewsApiConnector
from connectors.protocol import (
    Credentials,
    Cursor,
    EmittedBatch,
    RateLimitHint,
    SyncContext,
)
from models.enums import MediaKind, Platform, SourceCategory
from models.signal import Signal

pytestmark = pytest.mark.unit

PAYLOADS = Path(__file__).resolve().parents[2] / "fixtures" / "payloads"

NEWS_API_URL = "https://newsapi.org/v2/everything"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

T0 = datetime(2026, 7, 28, 14, 0, 0, tzinfo=UTC)
NEWEST_ARTICLE = datetime(2026, 7, 28, 14, 2, 11, tzinfo=UTC)
NEWEST_SEENDATE = datetime(2026, 7, 28, 14, 45, 0, tzinfo=UTC)

# The identity of the same article under each platform. Pinned rather than
# recomputed: recomputing would assert that the code agrees with itself, which it
# does even when `canonicalize_url` has been changed underneath it.
NEWS_API_GOLDEN_ID = "sig_b824b07c7caf5039b1f567fe9c9bedb8"
GDELT_GOLDEN_ID = "sig_315db73d35365d77b920468b2246d083"
SHARED_NATIVE_ID = "a7f5ba81426914a5237aa7506e428e74b74b147e5f820c9e0c0ad177579d2d1e"


def load(name: str) -> dict[str, Any]:
    """One recorded provider response."""
    payload: dict[str, Any] = json.loads((PAYLOADS / name).read_text())
    return payload


NEWS_PAGE_1 = load("news_api_everything_page1.json")
NEWS_PAGE_2 = load("news_api_everything_page2.json")
NEWS_RATE_LIMITED = load("news_api_error_rate_limited.json")
GDELT_ARTICLES = load("gdelt_doc_artlist.json")
GDELT_SATURATED = load("gdelt_doc_artlist_saturated.json")


# --------------------------------------------------------------------------- #
# Fakes -- the two ports a connector is given, in memory
# --------------------------------------------------------------------------- #


class FakeLimiter:
    """Records acquisitions instead of talking to Redis."""

    def __init__(self) -> None:
        self.acquired: list[list[str]] = []
        self.observed: list[RateLimitHint] = []

    async def acquire(self, keys: Sequence[str], *, timeout_seconds: float | None = None) -> None:
        self.acquired.append(list(keys))

    async def observe(self, keys: Sequence[str], hint: RateLimitHint) -> None:
        self.observed.append(hint)


class FakeDedup:
    """In-memory seen-set. Shared between two runs to exercise suppression."""

    def __init__(self) -> None:
        self.keys: dict[str, int] = {}

    async def seen(self, key: str) -> bool:
        return key in self.keys

    async def mark(self, key: str, ttl_seconds: int) -> None:
        self.keys[key] = ttl_seconds


def news_ctx(**overrides: Any) -> SyncContext:
    params: dict[str, Any] = {"query": "observability"}
    params.update(overrides.pop("params", {}))
    defaults: dict[str, Any] = {
        "connector_slug": "news_api",
        "account_id": "acct_1",
        "run_id": "run_1",
        "params": params,
        "limiter": FakeLimiter(),
        "dedup": FakeDedup(),
    }
    defaults.update(overrides)
    return SyncContext(**defaults)


def gdelt_ctx(**overrides: Any) -> SyncContext:
    params: dict[str, Any] = {"enabled": True, "query": "observability"}
    params.update(overrides.pop("params", {}))
    defaults: dict[str, Any] = {
        "connector_slug": "gdelt",
        "account_id": "acct_1",
        "run_id": "run_1",
        "params": params,
        "limiter": FakeLimiter(),
        "dedup": FakeDedup(),
    }
    defaults.update(overrides)
    return SyncContext(**defaults)


def news_connector(**overrides: Any) -> NewsApiConnector:
    return NewsApiConnector.from_config(
        news_ctx(**overrides), Credentials(account_id="acct_1", secrets={"api_key": "k-secret"})
    )


def gdelt_connector(**overrides: Any) -> GdeltConnector:
    return GdeltConnector.from_config(gdelt_ctx(**overrides), Credentials(account_id="acct_1"))


async def drain(connector: BaseConnector, cursor: Cursor | None = None) -> list[EmittedBatch]:
    return [batch async for batch in connector.run(cursor)]


def signals(batches: Sequence[EmittedBatch]) -> list[Signal]:
    return [signal for batch in batches for _, signal in batch.records]


def json_response(
    payload: Any, *, status: int = 200, headers: Mapping[str, str] | None = None
) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=dict(headers or {}))


# --------------------------------------------------------------------------- #
# Declarations
# --------------------------------------------------------------------------- #


class TestDeclarations:
    """Gate 1 of `connectors/registry.py`, applied to the real classes.

    The scheduler reads the `ClassVar` block before instantiating anything, so a
    platform that disagrees with its category is not caught by any amount of
    fetch testing -- it is caught here or in production after the quota is spent.
    """

    @pytest.mark.parametrize("connector", [NewsApiConnector, GdeltConnector])
    def test_declaration_passes_the_registry_gate(self, connector: type[BaseConnector]) -> None:
        """Importing `connectors` registers it, and registration *is* the gate.

        `connectors/__init__.py` calls `registry.register()` on every shipped
        connector at import time, and registration is what runs the declaration
        validation. A successful lookup here therefore means the gate passed --
        and a bad declaration would have raised during import, so this module
        would never have loaded to run the assertion at all.
        """
        import connectors  # noqa: F401  -- the import under test

        assert registry.get(connector.slug) is connector

    @pytest.mark.parametrize("connector", [NewsApiConnector, GdeltConnector])
    def test_neither_connector_needs_a_tos_review(self, connector: type[BaseConnector]) -> None:
        """Both are official, documented APIs; the flag is for scrapers (§9)."""
        assert connector.requires_tos_review is False

    def test_platforms_and_categories(self) -> None:
        assert (NewsApiConnector.platform, NewsApiConnector.category) == (
            Platform.NEWS_API,
            SourceCategory.NEWS,
        )
        assert (GdeltConnector.platform, GdeltConnector.category) == (
            Platform.GDELT,
            SourceCategory.NEWS,
        )


# --------------------------------------------------------------------------- #
# NewsAPI
# --------------------------------------------------------------------------- #


class TestNewsApiConfiguration:
    def test_a_search_with_no_terms_is_refused_before_any_io(self) -> None:
        """`/v2/everything` answers a term-less search with an authenticated 400,
        which spends a request against a daily cap to learn what the config knew."""
        with pytest.raises(ConnectorConfigurationError, match="query"):
            NewsApiConnector.from_config(
                news_ctx(params={"query": ""}), Credentials(account_id="acct_1")
            )

    def test_domains_alone_are_enough(self) -> None:
        connector = NewsApiConnector.from_config(
            news_ctx(params={"query": "", "domains": ["a.example.com", "b.example.com"]}),
            Credentials(account_id="acct_1"),
        )
        assert connector is not None

    async def test_a_missing_key_is_an_auth_error_not_a_crash(self) -> None:
        """`AuthError` is what flags the account `needs_reauth`; a bare KeyError
        out of an auth flow names neither the connector nor the account."""
        connector = NewsApiConnector.from_config(
            news_ctx(), Credentials(account_id="acct_1", secrets={})
        )
        with pytest.raises(AuthError, match="api_key"):
            await connector.authenticate()

    async def test_the_key_travels_in_a_header_never_the_query(self) -> None:
        """URLs are logged by every proxy in the path and land in `Referer`."""
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            await drain(news_connector())

        request = route.calls[0].request
        assert request.headers["X-Api-Key"] == "k-secret"
        assert "apiKey" not in request.url.params
        assert "k-secret" not in str(request.url)


class TestNewsApiIdentity:
    """Rule 2: `native_id = sha256(canonicalize_url(url))`."""

    async def test_identity_is_derived_from_the_canonical_url(self) -> None:
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            emitted = signals(await drain(news_connector()))

        article = emitted[-1]
        assert article.id == NEWS_API_GOLDEN_ID
        assert article.lineage.native_id == SHARED_NATIVE_ID
        # The stored permalink is canonical too, so `Signal.url` and `Signal.id`
        # cannot disagree about which page this is.
        assert article.url == "https://www.example-news.com/2026/07/28/observability-bill"

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.example-news.com/2026/07/28/observability-bill",
            "https://www.example-news.com/2026/07/28/observability-bill?utm_source=x&fbclid=y",
            "https://www.example-news.com:443/2026/07/28/observability-bill#comments",
            "https://www.example-news.com/2026/07/28/subdir/../observability-bill",
        ],
    )
    async def test_tracking_variants_collapse_to_one_signal(self, url: str) -> None:
        """Failing to collapse two spellings *forks* one article into two Signals,
        and every downstream store would then hold both."""
        payload = {**NEWS_PAGE_1, "articles": [{**NEWS_PAGE_1["articles"][0], "url": url}]}
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(payload))
            emitted = signals(await drain(news_connector()))

        assert [s.id for s in emitted] == [NEWS_API_GOLDEN_ID]

    async def test_an_article_with_no_url_is_dlq_and_never_rule_three(self) -> None:
        """`url` is required precisely so identity cannot silently fall to rule 3,
        which for this provider would hash a truncated body."""
        payload = {
            **NEWS_PAGE_1,
            "articles": [{**NEWS_PAGE_1["articles"][0], "url": None}],
        }
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(payload))
            batches = await drain(news_connector())

        assert batches[0].stats["dlq"] == 1
        assert batches[0].stats["emitted"] == 0


class TestNewsApiTruncation:
    """`Content.truncated` caps `content_integrity` (§3.5), so it must be per record."""

    async def test_marker_flags_truncation_and_records_the_missing_length(self) -> None:
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            emitted = {s.content.title: s for s in signals(await drain(news_connector()))}

        excerpt = emitted["Our observability bill tripled, so we moved forty services off it"]
        assert excerpt.content.truncated is True
        assert excerpt.metadata["news_api.truncated_chars"] == 2317

    async def test_the_marker_is_stripped_out_of_the_body(self) -> None:
        """It is provider bookkeeping. Left in, it would be embedded, and it would
        change the layer-2 content hash whenever the remaining count changed."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            emitted = signals(await drain(news_connector()))

        assert all("chars]" not in signal.content.text for signal in emitted)

    async def test_a_whole_article_is_not_flagged(self) -> None:
        """A static per-connector flag would understate every full article."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            emitted = {s.content.title: s for s in signals(await drain(news_connector()))}

        full = emitted["Port authority publishes quarterly freight volumes"]
        assert full.content.truncated is False
        assert "news_api.truncated_chars" not in full.metadata

    async def test_a_description_only_article_is_flagged_with_no_count(self) -> None:
        """No `content` at all means the body *is* the summary: an excerpt by
        construction, with nothing to report about what is missing."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            emitted = {s.content.title: s for s in signals(await drain(news_connector()))}

        summary = emitted["Grid operator delays interconnector decision"]
        assert summary.content.truncated is True
        assert "news_api.truncated_chars" not in summary.metadata
        assert summary.content.text.startswith("The decision on the cross-border")


class TestNewsApiNormalizeGolden:
    @pytest.fixture
    async def golden(self) -> Signal:
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            emitted = signals(await drain(news_connector()))
        return emitted[-1]

    def test_canonical_fields(self, golden: Signal) -> None:
        assert golden.platform is Platform.NEWS_API
        assert golden.source is SourceCategory.NEWS
        assert golden.timestamp == NEWEST_ARTICLE
        assert golden.timestamp.tzinfo is not None
        assert golden.content.char_count == len(golden.content.text)

    def test_the_byline_is_metadata_not_an_author(self, golden: Signal) -> None:
        """A display string with no identifier behind it. Keying an author's
        history on one merges every "Staff" in the corpus into one node."""
        assert golden.author is None
        assert golden.metadata["news_api.byline"] == "Jamie Nguyen"
        assert golden.metadata["news_api.source_name"] == "The Verge"

    def test_the_lead_image_survives_as_media(self, golden: Signal) -> None:
        assert [(m.kind, m.source_url) for m in golden.media] == [
            (MediaKind.IMAGE, "https://cdn.example-news.com/2026/07/observability.jpg")
        ]

    def test_lineage_is_complete_enough_to_replay(self, golden: Signal) -> None:
        lineage = golden.lineage
        assert lineage.connector_slug == "news_api"
        assert lineage.connector_version == NewsApiConnector.version
        assert lineage.sync_run_id == "run_1"
        assert lineage.request_fingerprint is not None
        # The digest is over the bytes that will be archived, and `Content` and
        # `Lineage` must agree about them or the R2 key addresses one and the
        # Signal cites the other.
        assert lineage.raw_sha256 == golden.content.raw_sha256
        assert lineage.raw_object_key is None, "the connector does not perform the PUT"

    def test_enrichment_fields_are_left_for_the_pipeline(self, golden: Signal) -> None:
        """A connector that filled these would be doing enrichment inside ingest."""
        assert (golden.entities, golden.topics, golden.keywords, golden.embeddings) == (
            [],
            [],
            [],
            [],
        )
        assert golden.sentiment is None
        assert golden.language.code == "und"


class TestNewsApiPagination:
    async def test_pages_a_window_and_stops_on_a_short_page(self) -> None:
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(
                side_effect=[json_response(NEWS_PAGE_1), json_response(NEWS_PAGE_2)]
            )
            batches = await drain(news_connector(params={"page_size": 3}))

        assert route.call_count == 2
        assert [call.request.url.params["page"] for call in route.calls] == ["1", "2"]
        assert [call.request.url.params["pageSize"] for call in route.calls] == ["3", "3"]

        totals = batches[-1].stats
        assert totals["emitted"] == 3
        assert totals["dropped"] == 1, "the [Removed] tombstone"
        assert totals["dlq"] == 1, "the article with no publishedAt"

    async def test_the_watermark_is_pinned_until_the_window_is_exhausted(self) -> None:
        """The provider sorts newest-first, so an intermediate page that advanced
        the watermark would skip everything the pager had not reached yet."""
        previous = T0 - timedelta(days=1)
        with respx.mock:
            respx.get(NEWS_API_URL).mock(
                side_effect=[json_response(NEWS_PAGE_1), json_response(NEWS_PAGE_2)]
            )
            batches = await drain(
                news_connector(params={"page_size": 3}), Cursor(watermark=previous)
            )

        assert batches[0].cursor.watermark == previous, "pinned, not advanced"
        assert batches[0].cursor.page_token == "2"
        assert batches[-1].cursor.watermark == NEWEST_ARTICLE
        assert batches[-1].cursor.page_token is None

    async def test_a_page_token_resumes_against_its_own_stored_window(self) -> None:
        """Page 4 of last hour's search addresses different articles than page 4
        of this one, so the window has to travel with the token."""
        cursor = Cursor(
            watermark=T0,
            page_token="2",
            checkpoint={
                "window": {"start": "2026-07-28T00:00:00+00:00", "end": "2026-07-28T23:59:59+00:00"}
            },
        )
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_2))
            await drain(news_connector(), cursor)

        params = route.calls[0].request.url.params
        assert params["page"] == "2"
        assert params["from"] == "2026-07-28T00:00:00"
        assert params["to"] == "2026-07-28T23:59:59"

    async def test_an_unreadable_window_repages_from_the_top(self) -> None:
        """§4.1 rule 4: a token the connector cannot use falls back to the
        watermark and re-pages rather than failing the run."""
        cursor = Cursor(watermark=T0, page_token="4", checkpoint={"window": "not a mapping"})
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_2))
            await drain(news_connector(), cursor)

        params = route.calls[0].request.url.params
        assert params["page"] == "1"
        # The default 300-second overlap, applied to the watermark by the base.
        assert params["from"] == "2026-07-28T13:55:00"

    async def test_max_records_narrows_the_page_before_the_request(self) -> None:
        """Every page costs quota, so a budget must shrink the request rather than
        discard the half of the response nobody wanted."""
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            await drain(news_connector(max_records=2))

        assert route.calls[0].request.url.params["pageSize"] == "2"

    async def test_max_pages_stops_the_pager(self) -> None:
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(
                side_effect=[json_response(NEWS_PAGE_1), json_response(NEWS_PAGE_1)]
            )
            batches = await drain(news_connector(max_pages=1, params={"page_size": 3}))

        assert route.call_count == 1
        assert len(batches) == 1

    async def test_a_response_with_no_total_stops_the_pager(self) -> None:
        """`totalResults` is always present in the documented shape, so a full page
        without one came from something that is not this API -- and a pager that
        ignored that would request page 2, 3, 4 ... forever against a proxy
        returning one constant body."""
        headless = {"status": "ok", "articles": NEWS_PAGE_1["articles"]}
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(return_value=json_response(headless))
            batches = await drain(news_connector(params={"page_size": 3}))

        assert route.call_count == 1
        assert len(signals(batches)) == 3

    async def test_the_plan_ceiling_closes_the_window_instead_of_failing(self) -> None:
        """A 426 means the plan will not page deeper. Failing there would fail
        every incremental run whose window outgrew the cap -- eventually all of
        them -- and leaving the watermark behind would re-fetch page 1 forever."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(
                side_effect=[
                    json_response(NEWS_PAGE_1),
                    httpx.Response(426, json={"status": "error", "code": "maximumResultsReached"}),
                ]
            )
            batches = await drain(news_connector(params={"page_size": 3}))

        assert len(batches) == 2
        assert len(batches[-1].records) == 0
        assert batches[-1].cursor.watermark == NEWEST_ARTICLE
        assert batches[-1].cursor.page_token is None


class TestNewsApiCursor:
    def test_cursor_round_trips_through_json(self) -> None:
        """The runtime persists the cursor as JSON, so anything the connector puts
        in a checkpoint has to survive the trip unchanged."""
        original = Cursor(
            watermark=NEWEST_ARTICLE,
            page_token="3",
            checkpoint={
                "window": {"start": "2026-07-28T00:00:00+00:00", "end": "2026-07-28T23:59:59+00:00"}
            },
        )
        encoded = json.dumps(
            {
                "version": original.version,
                "watermark": original.watermark.isoformat() if original.watermark else None,
                "page_token": original.page_token,
                "checkpoint": original.checkpoint,
            }
        )
        decoded = json.loads(encoded)
        restored = Cursor(
            version=decoded["version"],
            watermark=datetime.fromisoformat(decoded["watermark"]),
            page_token=decoded["page_token"],
            checkpoint=decoded["checkpoint"],
        )
        assert restored == original

    async def test_a_resumed_run_never_commits_a_watermark_backwards(self) -> None:
        """The base hands `fetch()` a cursor already shifted back by the overlap.
        A quiet window returns only records older than the stored watermark, and a
        connector that echoed the shifted value would commit backwards -- which
        §4.1 rule 2 says the runtime rejects."""
        stored = NEWEST_ARTICLE + timedelta(hours=6)
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            batches = await drain(news_connector(), Cursor(watermark=stored))

        assert batches[-1].cursor.watermark == stored

    async def test_a_budget_stop_mid_window_resumes_instead_of_closing(self) -> None:
        """Running out of budget is not the same as finishing the window.

        The provider pages newest-first, so the pages this run never reached hold
        articles *older* than everything it emitted. A cursor that released the
        watermark here would skip all of them permanently -- the exact failure the
        oldest-first rule in `connectors/base.py` exists to prevent.
        """
        previous = T0 - timedelta(days=1)
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            batches = await drain(
                news_connector(max_records=3, params={"page_size": 3}),
                Cursor(watermark=previous),
            )

        assert route.call_count == 1
        assert batches[-1].cursor.watermark == previous, "pinned, not released"
        assert batches[-1].cursor.page_token == "2"
        assert batches[-1].cursor.checkpoint["window"]["start"] is not None

    async def test_a_zero_budget_run_commits_nothing(self) -> None:
        """Yielding a cursor would close a window that was never opened."""
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            batches = await drain(news_connector(max_records=0))

        assert route.call_count == 0
        assert batches == []

    async def test_feeding_the_same_page_twice_emits_n_not_2n(self) -> None:
        dedup = FakeDedup()
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            first = await drain(news_connector(dedup=dedup))
            second = await drain(news_connector(dedup=dedup))

        assert len(signals(first)) == 3
        assert signals(second) == []
        assert second[-1].stats["duplicates"] == 3


class TestNewsApiFailures:
    """§6: the runtime decides what to do purely from the class it caught."""

    @pytest.mark.parametrize("status", [401, 403])
    async def test_a_rejected_key_halts_the_run(self, status: int) -> None:
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=httpx.Response(status))
            with pytest.raises(AuthError):
                await drain(news_connector())

    async def test_a_short_retry_after_is_transient(self) -> None:
        """Inside the cap the runtime backs off and keeps the worker."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(
                return_value=httpx.Response(429, headers={"Retry-After": "120"})
            )
            with pytest.raises(TransientError):
                await drain(news_connector())

    async def test_a_long_retry_after_becomes_a_quota_error(self) -> None:
        """Beyond fifteen minutes, holding a worker is more expensive than
        checkpointing and coming back (§5.2)."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(
                return_value=httpx.Response(429, headers={"Retry-After": "3600"})
            )
            with pytest.raises(QuotaError) as caught:
                await drain(news_connector())

        assert caught.value.retry_after_seconds == 3600.0

    async def test_an_error_body_is_classified_by_its_code(self) -> None:
        """This provider has been observed answering 200 with an exhausted key."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_RATE_LIMITED))
            with pytest.raises(QuotaError):
                await drain(news_connector())

    async def test_a_5xx_is_transient(self) -> None:
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=httpx.Response(503))
            with pytest.raises(TransientError):
                await drain(news_connector())

    async def test_a_timeout_is_transient_and_is_not_retried_here(self) -> None:
        """Backoff belongs to the runtime: a connector that retried privately would
        make the shared limiter's accounting wrong."""
        with respx.mock:
            route = respx.get(NEWS_API_URL).mock(side_effect=httpx.ConnectTimeout("slow"))
            with pytest.raises(TransientError):
                await drain(news_connector())

        assert route.call_count == 1

    async def test_a_400_is_permanent(self) -> None:
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=httpx.Response(400))
            with pytest.raises(PermanentError):
                await drain(news_connector())

    async def test_a_missing_articles_array_is_a_shape_change_not_a_quiet_day(self) -> None:
        """An empty result set is `"articles": []`; defaulting to that here would
        turn a breaking provider change into a run that succeeds with no records."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response({"status": "ok"}))
            with pytest.raises(PermanentError, match="articles"):
                await drain(news_connector())

    async def test_rate_limit_headers_reach_the_bucket(self) -> None:
        """Provider truth beats local estimate (§5.2)."""
        ctx = news_ctx()
        connector = NewsApiConnector.from_config(
            ctx, Credentials(account_id="acct_1", secrets={"api_key": "k"})
        )
        with respx.mock:
            respx.get(NEWS_API_URL).mock(
                return_value=json_response(NEWS_PAGE_1, headers={"X-RateLimit-Remaining": "3"})
            )
            await drain(connector)

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert limiter.observed and limiter.observed[0].remaining == 3

    async def test_only_rate_limit_headers_leave_the_connector(self) -> None:
        """A page's headers travel with it into the runtime, and a whole header map
        carries cookies and echoed authorization -- which §1 forbids logging."""
        connector = news_connector()
        await connector.authenticate()
        with respx.mock:
            respx.get(NEWS_API_URL).mock(
                return_value=json_response(
                    NEWS_PAGE_1,
                    headers={"Set-Cookie": "session=abc", "X-RateLimit-Remaining": "9"},
                )
            )
            pages = [page async for page in connector.fetch(Cursor())]
        await connector.aclose()

        assert dict(pages[0].raw_headers) == {"x-ratelimit-remaining": "9"}


# --------------------------------------------------------------------------- #
# GDELT
# --------------------------------------------------------------------------- #


class TestGdeltEnablement:
    """§9.5 ships GDELT off by default; `.env.example` carries `GDELT_ENABLED=false`."""

    def test_it_refuses_to_build_when_the_flag_is_absent(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="disabled"):
            GdeltConnector.from_config(
                SyncContext(
                    connector_slug="gdelt",
                    account_id="acct_1",
                    run_id="run_1",
                    params={"query": "observability"},
                ),
                Credentials(account_id="acct_1"),
            )

    def test_the_string_false_does_not_enable_it(self) -> None:
        """`GDELT_ENABLED` is a string long before it is a boolean, and
        `bool("false")` is True -- which would invert the whole gate."""
        with pytest.raises(ConnectorConfigurationError):
            GdeltConnector.from_config(
                gdelt_ctx(params={"enabled": "false"}), Credentials(account_id="acct_1")
            )

    def test_the_string_true_does_enable_it(self) -> None:
        assert GdeltConnector.from_config(
            gdelt_ctx(params={"enabled": "true"}), Credentials(account_id="acct_1")
        )

    def test_an_empty_query_is_refused(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="query"):
            GdeltConnector.from_config(
                gdelt_ctx(params={"query": ""}), Credentials(account_id="acct_1")
            )


class TestGdeltOverlap:
    """§4.1 rule 3: eventually-consistent providers resume further back."""

    def test_the_overlap_is_fifteen_minutes_not_five(self) -> None:
        assert GdeltConnector.overlap_seconds == 900
        assert BaseConnector.overlap_seconds == 300, "the default this deliberately overrides"

    async def test_the_wider_overlap_reaches_the_outbound_request(self) -> None:
        """The value only matters where it becomes a query parameter. GDELT's
        index lags its own `seendate`, so a five-minute overlap would query a
        range GDELT had not finished writing and drop those records for good."""
        with respx.mock:
            route = respx.get(GDELT_URL).mock(return_value=json_response({"articles": []}))
            await drain(gdelt_connector(), Cursor(watermark=T0))

        start = route.calls[0].request.url.params["startdatetime"]
        assert start == (T0 - timedelta(seconds=900)).strftime("%Y%m%d%H%M%S")
        assert start != (T0 - timedelta(seconds=300)).strftime("%Y%m%d%H%M%S")

    async def test_a_quiet_resume_does_not_walk_the_watermark_backwards(self) -> None:
        """The base hands `fetch()` a cursor already shifted back 900 seconds. On a
        quiet query -- nothing new in the window -- echoing that value would commit
        a watermark fifteen minutes older than the stored one, which §4.1 rule 2
        says the runtime rejects."""
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=json_response({"articles": []}))
            batches = await drain(gdelt_connector(), Cursor(watermark=T0))

        assert batches[-1].cursor.watermark == T0


class TestGdeltIdentity:
    async def test_identity_is_the_canonical_url(self) -> None:
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=json_response(GDELT_ARTICLES))
            emitted = signals(await drain(gdelt_connector()))

        first = emitted[0]
        assert first.id == GDELT_GOLDEN_ID
        assert first.lineage.native_id == SHARED_NATIVE_ID
        assert first.url == "https://www.example-news.com/2026/07/28/observability-bill"

    async def test_the_same_article_has_one_native_id_across_both_connectors(self) -> None:
        """Rule 2 is platform-independent, which is what lets cross-platform dedup
        recognise a syndicated story -- while `Signal.id` stays per platform, so
        the two observations are still counted as two (`docs/signal-model.md` §4.3
        keeps spread visible instead of deleting it)."""
        with respx.mock:
            respx.get(NEWS_API_URL).mock(return_value=json_response(NEWS_PAGE_1))
            respx.get(GDELT_URL).mock(return_value=json_response(GDELT_ARTICLES))
            from_news = signals(await drain(news_connector()))[-1]
            from_gdelt = signals(await drain(gdelt_connector()))[0]

        assert from_news.lineage.native_id == from_gdelt.lineage.native_id
        assert from_news.id != from_gdelt.id


class TestGdeltNormalize:
    @pytest.fixture
    async def emitted(self) -> list[Signal]:
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=json_response(GDELT_ARTICLES))
            return signals(await drain(gdelt_connector()))

    def test_records_are_title_only_and_say_so(self, emitted: list[Signal]) -> None:
        """GDELT indexes articles; it does not serve them. `truncated` asserts the
        weaker claim ("not the full body"); the empty text is what marks the
        title-only tier of `content_integrity`."""
        first = emitted[0]
        assert first.content.title
        assert first.content.text == ""
        assert first.content.truncated is True

    def test_the_provider_language_label_is_metadata_not_a_detection(
        self, emitted: list[Signal]
    ) -> None:
        """`Signal.language` is a detector result with a confidence behind it.
        Copying "Spanish" into it would fabricate one."""
        spanish = [s for s in emitted if s.metadata.get("gdelt.language") == "Spanish"]
        assert spanish and spanish[0].language.code == "und"

    def test_metadata_is_namespaced(self, emitted: list[Signal]) -> None:
        """Un-namespaced keys collide across connectors in one jsonb column and
        one OpenSearch mapping."""
        assert all(key.startswith("gdelt.") for s in emitted for key in s.metadata)

    def test_the_social_card_image_survives_as_media(self, emitted: list[Signal]) -> None:
        assert emitted[0].media[0].kind is MediaKind.IMAGE

    async def test_an_untitled_article_is_dropped_not_sent_to_the_dlq(self) -> None:
        """A title-only source with no title carries no observation. The payload is
        well-formed, so it is a drop -- and drops are counted separately from
        mapping failures precisely so real mapping bugs stay visible."""
        assert any(not a["title"] for a in GDELT_ARTICLES["articles"]), "fixture guard"
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=json_response(GDELT_ARTICLES))
            stats = (await drain(gdelt_connector()))[-1].stats

        assert (stats["fetched"], stats["emitted"], stats["dropped"], stats["dlq"]) == (
            4,
            3,
            1,
            0,
        )

    async def test_drop_and_dlq_are_counted_separately(self) -> None:
        broken = {
            "articles": [
                *GDELT_ARTICLES["articles"],
                {"url": "https://d.example.com/x", "title": "Dated wrongly", "seendate": "soon"},
            ]
        }
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=json_response(broken))
            batches = await drain(gdelt_connector())

        stats = batches[-1].stats
        assert (stats["emitted"], stats["dropped"], stats["dlq"]) == (3, 1, 1)


class TestGdeltPagination:
    async def test_it_walks_forward_from_the_newest_seendate(self) -> None:
        """There is no page token: `sort=DateAsc` plus a moving `startdatetime`
        *is* the pager, which is also why oldest-first is honest here."""
        with respx.mock:
            route = respx.get(GDELT_URL).mock(
                side_effect=[
                    json_response(GDELT_ARTICLES),
                    json_response({"articles": []}),
                ]
            )
            batches = await drain(
                gdelt_connector(params={"max_records_per_request": 4}),
                Cursor(watermark=T0),
            )

        assert route.call_count == 2
        assert route.calls[0].request.url.params["sort"] == "DateAsc"
        assert route.calls[1].request.url.params["startdatetime"] == "20260728144500"
        assert batches[0].cursor.watermark == NEWEST_SEENDATE

    async def test_the_watermark_advances_on_every_page(self) -> None:
        """Ascending order means everything older than the page's newest record has
        already been yielded, so committing it mid-run loses nothing."""
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=json_response(GDELT_ARTICLES))
            batches = await drain(gdelt_connector(), Cursor(watermark=T0))

        assert batches[0].cursor.watermark == NEWEST_SEENDATE
        assert batches[0].cursor.page_token is None

    async def test_a_saturated_second_steps_forward_and_records_the_gap(self) -> None:
        """GDELT stamps whole 15-minute batches with one `seendate`, so a second
        can hold more articles than `maxrecords` and the pager cannot advance.
        Stepping past it is the only way to make progress; the loss is written
        into the cursor because a silent gap in a news index is indistinguishable
        from a quiet hour."""
        with respx.mock:
            route = respx.get(GDELT_URL).mock(
                side_effect=[
                    json_response(GDELT_SATURATED),
                    json_response({"articles": []}),
                ]
            )
            batches = await drain(
                gdelt_connector(params={"max_records_per_request": 3}),
                Cursor(watermark=T0 + timedelta(seconds=900)),
            )

        assert route.calls[0].request.url.params["startdatetime"] == "20260728140000"
        assert route.calls[1].request.url.params["startdatetime"] == "20260728140001"
        assert batches[-1].cursor.checkpoint["saturated_seconds"] == ["2026-07-28T14:00:00+00:00"]

    async def test_a_full_page_of_undated_articles_stops_the_pager(self) -> None:
        """`startdatetime` is the whole pager, so a full page from which no date can
        be read leaves it with nowhere to move. Stopping repeats the window on the
        next run; not stopping requests it forever inside this one."""
        undated = {
            "articles": [{**article, "seendate": "soon"} for article in GDELT_SATURATED["articles"]]
        }
        with respx.mock:
            route = respx.get(GDELT_URL).mock(return_value=json_response(undated))
            batches = await drain(gdelt_connector(params={"max_records_per_request": 3}))

        assert route.call_count == 1
        assert batches[-1].stats["dlq"] == 3

    async def test_max_records_narrows_maxrecords_before_the_request(self) -> None:
        """The point of the whole exercise: this source answers in hundreds, and
        `BaseConnector.run()` can only stop the loop *after* a 250-record page has
        been fetched, normalized and hashed."""
        with respx.mock:
            route = respx.get(GDELT_URL).mock(return_value=json_response(GDELT_ARTICLES))
            await drain(gdelt_connector(max_records=2))

        assert route.calls[0].request.url.params["maxrecords"] == "2"

    async def test_max_pages_is_respected_strictly(self) -> None:
        with respx.mock:
            route = respx.get(GDELT_URL).mock(
                side_effect=[json_response(GDELT_ARTICLES), json_response(GDELT_ARTICLES)]
            )
            batches = await drain(
                gdelt_connector(max_pages=1, params={"max_records_per_request": 4}),
                Cursor(watermark=T0),
            )

        assert route.call_count == 1
        assert len(batches) == 1

    async def test_the_default_request_never_asks_for_more_than_the_api_allows(self) -> None:
        with respx.mock:
            route = respx.get(GDELT_URL).mock(return_value=json_response({"articles": []}))
            await drain(gdelt_connector(params={"max_records_per_request": 9999}))

        assert route.calls[0].request.url.params["maxrecords"] == "250"

    async def test_feeding_the_same_window_twice_emits_n_not_2n(self) -> None:
        dedup = FakeDedup()
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=json_response(GDELT_ARTICLES))
            first = await drain(gdelt_connector(dedup=dedup))
            second = await drain(gdelt_connector(dedup=dedup))

        assert len(signals(first)) == 3
        assert signals(second) == []


class TestGdeltFailures:
    async def test_an_empty_body_is_zero_records_not_a_failure(self) -> None:
        """The DOC API answers 200 with nothing at all for an empty window."""
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=httpx.Response(200, content=b""))
            batches = await drain(gdelt_connector())

        assert len(batches) == 1
        assert len(batches[0].records) == 0

    async def test_a_plain_text_body_is_permanent(self) -> None:
        """GDELT reports query syntax errors as text with status 200. Retrying a
        bad query is a bad query."""
        with respx.mock:
            respx.get(GDELT_URL).mock(
                return_value=httpx.Response(200, text="Your query was too short.")
            )
            with pytest.raises(PermanentError) as caught:
                await drain(gdelt_connector())

        assert "Your query" not in str(caught.value), "§1 forbids logging the body"

    async def test_a_5xx_is_transient(self) -> None:
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=httpx.Response(502))
            with pytest.raises(TransientError):
                await drain(gdelt_connector())

    async def test_a_throttle_is_transient_inside_the_cap(self) -> None:
        with respx.mock:
            respx.get(GDELT_URL).mock(
                return_value=httpx.Response(429, headers={"Retry-After": "30"})
            )
            with pytest.raises(TransientError):
                await drain(gdelt_connector())

    async def test_a_long_wait_becomes_a_quota_error(self) -> None:
        with respx.mock:
            respx.get(GDELT_URL).mock(
                return_value=httpx.Response(429, headers={"Retry-After": "1800"})
            )
            with pytest.raises(QuotaError):
                await drain(gdelt_connector())

    async def test_it_identifies_itself(self) -> None:
        """An unattributed client is the first thing an undocumented rate limiter
        throttles, and the only way a free service can reach us before blocking."""
        with respx.mock:
            route = respx.get(GDELT_URL).mock(return_value=json_response({"articles": []}))
            await drain(gdelt_connector())

        assert route.calls[0].request.headers["User-Agent"] == "omnisense/0.1"


class TestRateLimitBuckets:
    """§5.1: every outbound call acquires the connector and the account bucket."""

    async def test_news_api_acquires_before_every_request(self) -> None:
        ctx = news_ctx(params={"page_size": 3})
        connector = NewsApiConnector.from_config(
            ctx, Credentials(account_id="acct_1", secrets={"api_key": "k"})
        )
        with respx.mock:
            respx.get(NEWS_API_URL).mock(
                side_effect=[json_response(NEWS_PAGE_1), json_response(NEWS_PAGE_2)]
            )
            await drain(connector, Cursor(watermark=T0))

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert len(limiter.acquired) == 2, "one acquisition per outbound call"
        assert limiter.acquired[0][:2] == ["os:rl:news_api", "os:rl:news_api:acct_1"]

    async def test_gdelt_acquires_a_host_bucket_too(self) -> None:
        ctx = gdelt_ctx()
        connector = GdeltConnector.from_config(ctx, Credentials(account_id="acct_1"))
        with respx.mock:
            respx.get(GDELT_URL).mock(return_value=json_response({"articles": []}))
            await drain(connector)

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert "os:rl:host:api.gdeltproject.org" in limiter.acquired[0]


class TestClientLifecycle:
    async def test_the_client_is_released_even_when_the_consumer_walks_away(self) -> None:
        """The leak path: an abandoned generator with an open HTTP client."""
        connector = news_connector(params={"page_size": 3})
        with respx.mock:
            respx.get(NEWS_API_URL).mock(
                side_effect=[json_response(NEWS_PAGE_1), json_response(NEWS_PAGE_2)]
            )
            # cast: `run()` is declared as an AsyncIterator but is a generator,
            # and `aclose()` is the whole point of this test.
            agen = cast(AsyncGenerator[EmittedBatch, None], connector.run())
            await agen.__anext__()
            await agen.aclose()

        assert connector._client is None

    async def test_authenticate_is_idempotent(self) -> None:
        connector = news_connector()
        await connector.authenticate()
        first = connector._client
        await connector.authenticate()
        try:
            assert connector._client is first
        finally:
            await connector.aclose()
