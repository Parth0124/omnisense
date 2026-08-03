"""Unit tests for `connectors/news/rss.py`.

RSS is the reference connector, so these tests are as much a specification of
the *contract* as of this implementation. They target the behaviours that are
invisible when they are wrong:

- **oldest-first ordering** -- a newest-first pager that dies mid-run commits a
  watermark past records it never emitted, and nothing ever fetches them again;
- **conditional GET** -- the ETag has to survive a round-trip through
  `Cursor.checkpoint` and JSON, or every poll is a full download;
- **304 is not an error** -- the single most common response a healthy feed
  reader gets;
- **identity** -- guid when there is one, sha256 of the canonicalized link when
  there is not, and the id on the Kafka reference must equal the id on the
  Signal it points at;
- **one bad record costs one record** -- an undated entry goes to the DLQ while
  the rest of the feed still emits;
- **one dead feed costs one feed** -- unless every feed is dead, which is a fact
  about us and must fail loudly.

Everything runs against recorded fixtures under `tests/fixtures/payloads/` with
`respx`. No network, no Redis, no datastore: `docs/architecture.md` §6.2 rule 2
is what makes that possible, and `TestContract::test_imports_no_backend` is what
keeps it true.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from connectors.base import BaseConnector
from connectors.exceptions import (
    ConnectorConfigurationError,
    NormalizationError,
    PermanentError,
    TransientError,
)
from connectors.news.rss import (
    ENVELOPE_KEY,
    EXCERPT_CHAR_THRESHOLD,
    RssConnector,
)
from connectors.normalize.html import canonicalize_url
from connectors.protocol import (
    Credentials,
    Cursor,
    EmittedBatch,
    RateLimitHint,
    RawRecord,
    SyncContext,
)
from models.enums import AuthType, MediaKind, Platform, SourceCategory
from models.signal import signal_id

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "payloads"
RSS_BYTES = (FIXTURES / "rss_20_sample.xml").read_bytes()
ATOM_BYTES = (FIXTURES / "rss_atom_sample.xml").read_bytes()

NEWS_FEED = "https://news.example.com/feed.xml"
ENG_FEED = "https://eng.example.org/atom.xml"

RSS_ETAG = 'W/"3f8a"'
RSS_LAST_MODIFIED = "Fri, 31 Jul 2026 09:15:00 GMT"

# Event times in the RSS 2.0 fixture, oldest-first. The undated "corrections"
# item deliberately has no entry here -- that is the point of it.
RSS_TIMES = [
    datetime(2026, 7, 27, 7, 5, tzinfo=UTC),
    datetime(2026, 7, 29, 16, 40, tzinfo=UTC),
    datetime(2026, 7, 31, 9, 12, tzinfo=UTC),
]
RSS_GUID_NEWEST = "tag:news.example.com,2026:post-4181"
RSS_GUID_OLDEST = "tag:news.example.com,2026:post-4166"
RSS_LINK_NO_GUID = "https://news.example.com/2026/07/29/loki-34"


# --------------------------------------------------------------------------- #
# Fakes and helpers
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


def make_ctx(feeds: Any = NEWS_FEED, **overrides: Any) -> SyncContext:
    params = dict(overrides.pop("params", {}))
    params.setdefault("feeds", feeds)
    defaults: dict[str, Any] = {
        "connector_slug": "rss",
        "account_id": "acct_1",
        "run_id": "run_1",
        "params": params,
        "limiter": FakeLimiter(),
        "dedup": FakeDedup(),
        "user_agent": "omnisense/0.1 (+https://example.test/bot)",
    }
    defaults.update(overrides)
    return SyncContext(**defaults)


def build(ctx: SyncContext | None = None, **secrets: str) -> RssConnector:
    return RssConnector.from_config(
        ctx or make_ctx(), Credentials(account_id="acct_1", secrets=secrets)
    )


def feed_response(
    body: bytes = RSS_BYTES, *, etag: str | None = RSS_ETAG, **headers: str
) -> httpx.Response:
    merged = {"Content-Type": "application/rss+xml; charset=utf-8", **headers}
    if etag:
        merged.setdefault("ETag", etag)
    return httpx.Response(200, content=body, headers=merged)


async def drain(
    connector: RssConnector, cursor: Cursor | None = None
) -> list[EmittedBatch]:
    return [batch async for batch in connector.run(cursor)]


def signals(batches: Sequence[EmittedBatch]) -> list[Any]:
    return [signal for batch in batches for _, signal in batch.records]


def totals(batches: Sequence[EmittedBatch]) -> dict[str, int]:
    """`run()` accumulates stats, so the last batch carries the run total."""
    return dict(batches[-1].stats)


# --------------------------------------------------------------------------- #
# Declaration
# --------------------------------------------------------------------------- #


class TestContract:
    """What the scheduler and the registry read before anything is built."""

    def test_declaration_matches_the_catalogue(self) -> None:
        """`docs/connector-spec.md` §9.5 and §11.2 step 1, checkable rather than
        merely written down."""
        assert RssConnector.slug == "rss"
        assert RssConnector.platform is Platform.RSS
        assert RssConnector.category is SourceCategory.NEWS
        assert RssConnector.auth_type is AuthType.NONE
        assert RssConnector.requires_tos_review is False
        assert RssConnector.supports_incremental is True

    def test_backfill_is_not_supported(self) -> None:
        """A feed is a trailing window. Declaring backfill would have the
        scheduler plan a historical crawl that can only re-fetch the same
        fifty items forever."""
        assert RssConnector.supports_backfill is False

    def test_is_concrete(self) -> None:
        """A class with an unimplemented abstract method registers fine and
        then raises `TypeError` on the first scheduled run."""
        assert not RssConnector.__abstractmethods__

    def test_does_not_override_the_template(self) -> None:
        """The six-stage order is not a connector's decision."""
        assert RssConnector.run is BaseConnector.run

    def test_keeps_the_default_dedup_keys(self) -> None:
        """§11.2 step 7: layer 2 is what collapses one wire story appearing in a
        dozen feeds under a dozen different GUIDs, and it only works if the
        content hash is the shared default."""
        assert RssConnector.dedup_keys is BaseConnector.dedup_keys

    def test_imports_no_backend(self) -> None:
        """The rule that lets this whole file run with nothing installed."""
        source = Path("connectors/news/rss.py").read_text()
        offenders = [
            line
            for line in source.splitlines()
            if line.strip().startswith(("from backend", "import backend", "from services", "import services"))
        ]
        assert not offenders


class TestConfiguration:
    """`from_config` must perform no I/O and must reject a dangerous feed list."""

    def test_accepts_a_comma_separated_string(self) -> None:
        """`.env.example` seeds `RSS_FEED_URLS` as one line; a settings layer
        that has not split it yet is the common case, not a misuse."""
        connector = build(make_ctx(f"{NEWS_FEED}, {ENG_FEED}"))
        assert connector._feeds == (NEWS_FEED, ENG_FEED)

    def test_accepts_a_list(self) -> None:
        assert build(make_ctx([NEWS_FEED, ENG_FEED]))._feeds == (NEWS_FEED, ENG_FEED)

    def test_accepts_the_spec_and_env_spellings_of_the_param(self) -> None:
        """Configured-but-spelled-differently must not look like a healthy
        connector with nothing to say."""
        for key in ("feeds", "feed_urls", "rss_feed_urls"):
            ctx = SyncContext(
                connector_slug="rss",
                account_id="a",
                run_id="r",
                params={key: NEWS_FEED},
            )
            assert build(ctx)._feeds == (NEWS_FEED,)

    def test_collapses_two_spellings_of_one_feed(self) -> None:
        """Otherwise one server is polled twice per run for one document."""
        connector = build(make_ctx([NEWS_FEED, NEWS_FEED + "?utm_source=x"]))
        assert connector._feeds == (NEWS_FEED,)

    def test_requests_the_configured_spelling_not_the_canonical_one(self) -> None:
        """The canonical form keys the checkpoint; the operator's own URL is
        what goes on the wire, so they can reproduce a request with curl."""
        configured = "https://news.example.com/feed.xml?b=2&a=1"
        assert build(make_ctx(configured))._feeds == (configured,)

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "data:text/xml,<rss/>",
            "ftp://news.example.com/feed.xml",
        ],
    )
    def test_rejects_non_http_schemes(self, url: str) -> None:
        """A feed list is operator input; a `file://` entry turns it into a
        local-file read."""
        with pytest.raises(ConnectorConfigurationError, match="scheme"):
            build(make_ctx(url))

    def test_rejects_credentials_embedded_in_a_feed_url(self) -> None:
        """`BaseConnector.host_rate_limit_key()` builds its Redis key from the
        netloc, which includes userinfo -- so a password in the URL becomes a
        bucket name and leaks into every metric derived from it."""
        with pytest.raises(ConnectorConfigurationError, match="credentials"):
            build(make_ctx("https://user:hunter2@news.example.com/feed.xml"))

    def test_rejects_an_empty_feed_list(self) -> None:
        with pytest.raises(ConnectorConfigurationError):
            build(make_ctx([]))

    def test_rejects_a_missing_feed_param(self) -> None:
        ctx = SyncContext(connector_slug="rss", account_id="a", run_id="r")
        with pytest.raises(ConnectorConfigurationError, match="RSS_FEED_URLS"):
            build(ctx)

    def test_refuses_to_silently_truncate_an_oversized_feed_list(self) -> None:
        """Polling the first N of N+1 configured feeds is invisible data loss:
        the operator has no way to learn which one stopped."""
        ctx = make_ctx([NEWS_FEED, ENG_FEED], params={"max_feeds": 1})
        with pytest.raises(ConnectorConfigurationError, match="ceiling"):
            build(ctx)

    def test_construction_opens_no_socket(self) -> None:
        """`from_config` must not perform I/O: the scheduler builds connectors
        merely to read their declaration."""
        with respx.mock(assert_all_called=False):
            connector = build()
            assert connector._client is None

    async def test_basic_auth_is_not_sent_to_every_host(self) -> None:
        """An account with one private feed and forty-nine public ones must not
        broadcast its password to the forty-nine."""
        with pytest.raises(ConnectorConfigurationError, match="basic_auth_hosts"):
            RssConnector.from_config(
                make_ctx([NEWS_FEED, ENG_FEED]),
                Credentials(account_id="a", secrets={"username": "u", "password": "p"}),
            )

    def test_a_half_configured_credential_fails_at_config_time(self) -> None:
        """A password with no username would otherwise surface as a `KeyError`
        from inside the fetch loop, filed as a provider fault rather than as the
        operator error it is."""
        with pytest.raises(ConnectorConfigurationError, match="username"):
            RssConnector.from_config(
                make_ctx(), Credentials(account_id="a", secrets={"password": "p"})
            )

    async def test_basic_auth_is_sent_to_a_declared_host(self) -> None:
        connector = RssConnector.from_config(
            make_ctx([NEWS_FEED, ENG_FEED]),
            Credentials(
                account_id="a",
                secrets={"username": "u", "password": "p"},
                extra={"basic_auth_hosts": ["news.example.com"]},
            ),
        )
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            await drain(connector)
            sent = {
                call.request.url.host: call.request.headers.get("authorization")
                for call in respx.calls
            }

        assert sent["news.example.com"] is not None
        assert sent["eng.example.org"] is None


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


class TestOrdering:
    """`BaseConnector.fetch` requires oldest-first, and it is not stylistic."""

    async def test_emits_oldest_first_from_a_newest_first_feed(self) -> None:
        """The fixture is in document order, newest first, like every real
        feed. A watermark that moved forward over unemitted records could never
        move back to collect them."""
        connector = build()
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(connector)

        assert [signal.timestamp for signal in signals(batches)] == RSS_TIMES

    async def test_watermark_is_the_newest_emitted_event_time(self) -> None:
        connector = build()
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(connector)

        assert batches[-1].cursor.watermark == RSS_TIMES[-1]

    async def test_undated_entries_lead_and_never_move_the_watermark(self) -> None:
        """They have no position on the timeline, so they are emitted before
        anything that does and the watermark is computed without them."""
        connector = build()
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(connector)

        page = batches[0]
        assert page.records[0][0].native_id.startswith("tag:news.example.com")
        # The undated record is first in the page but never becomes a Signal.
        raw_ids = [record.native_id for record, _ in page.records]
        assert "tag:news.example.com,2026:page-corrections" not in raw_ids


class TestConditionalGet:
    """The politeness win, and the state that makes it work."""

    async def test_first_poll_sends_no_validators(self) -> None:
        connector = build()
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            await drain(connector)
            request = respx.calls[0].request

        assert "if-none-match" not in request.headers
        assert "if-modified-since" not in request.headers

    async def test_etag_and_last_modified_round_trip_through_the_cursor(self) -> None:
        """Stored on poll one, sent on poll two. This is the whole mechanism."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(
                return_value=feed_response(**{"Last-Modified": RSS_LAST_MODIFIED})
            )
            first = await drain(build())

        state = first[-1].cursor.checkpoint["feeds"][canonicalize_url(NEWS_FEED)]
        assert state["etag"] == RSS_ETAG
        assert state["last_modified"] == RSS_LAST_MODIFIED

        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(304))
            await drain(build(), first[-1].cursor)
            headers = respx.calls[0].request.headers

        assert headers["if-none-match"] == RSS_ETAG
        assert headers["if-modified-since"] == RSS_LAST_MODIFIED

    async def test_304_is_an_empty_page_and_not_an_error(self) -> None:
        """The most common response a healthy feed reader gets."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            first = await drain(build())

        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(304))
            second = await drain(build(), first[-1].cursor)

        assert len(second) == 1
        assert len(second[0]) == 0
        assert totals(second)["dlq"] == 0

    async def test_304_does_not_lose_the_feed_state(self) -> None:
        """RFC 9110 lets a 304 omit validators that have not changed; dropping
        them there would turn every later poll into a full download."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(
                return_value=feed_response(**{"Last-Modified": RSS_LAST_MODIFIED})
            )
            first = await drain(build())

        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(304))
            second = await drain(build(), first[-1].cursor)

        before = first[-1].cursor.checkpoint["feeds"][canonicalize_url(NEWS_FEED)]
        after = second[-1].cursor.checkpoint["feeds"][canonicalize_url(NEWS_FEED)]
        assert after == before

    async def test_304_adopts_a_refreshed_etag(self) -> None:
        """A 304 may carry a new validator; ignoring it re-downloads once."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            first = await drain(build())

        with respx.mock:
            respx.get(NEWS_FEED).mock(
                return_value=httpx.Response(304, headers={"ETag": 'W/"4b21"'})
            )
            second = await drain(build(), first[-1].cursor)

        assert second[-1].cursor.checkpoint["feeds"][canonicalize_url(NEWS_FEED)][
            "etag"
        ] == 'W/"4b21"'

    async def test_a_200_without_an_etag_drops_the_stale_one(self) -> None:
        """A server that stopped issuing ETags must not keep receiving a stale
        `If-None-Match` forever."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            first = await drain(build())

        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response(etag=None))
            second = await drain(build(), first[-1].cursor)

        assert "etag" not in second[-1].cursor.checkpoint["feeds"][canonicalize_url(NEWS_FEED)]

    async def test_checkpoint_survives_json(self) -> None:
        """The runtime persists it as JSON in Postgres; a `datetime` or a
        `struct_time` in there is a run that cannot commit its cursor."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build())

        checkpoint = batches[-1].cursor.checkpoint
        assert json.loads(json.dumps(checkpoint)) == checkpoint

    async def test_a_corrupt_checkpoint_degrades_to_a_full_poll(self) -> None:
        """One wasted poll is cheaper than every poll raising until a human
        edits the row back."""
        corrupt = Cursor(checkpoint={"feeds": {NEWS_FEED: ["not", "a", "mapping"]}})
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build(), corrupt)

        assert len(signals(batches)) == 3


class TestIncremental:
    """Per-feed high-water marks, and why they are per-feed."""

    async def test_a_second_poll_re_offers_only_the_overlap_window(self) -> None:
        """Not the whole feed. Everything older than `newest_seen - overlap` is
        dropped in `fetch()`, which saves a normalize and two Redis round-trips
        for the ~90% of a feed that is unchanged on every poll."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            first = await drain(build())

        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response(etag=None))
            second = await drain(build(), first[-1].cursor)

        # Three of four entries fall below the cutoff. What is left is the
        # newest item, which sits inside the overlap window, and the undated
        # one, which has no position on the timeline and goes to the DLQ.
        assert second[0].stats["fetched"] == 2
        assert [s.timestamp for s in signals(second)] == [RSS_TIMES[-1]]
        assert totals(second)["dlq"] == 1

    async def test_the_re_offered_window_is_then_suppressed_by_dedup(self) -> None:
        """Overlap plus dedup is how a late-indexed post is caught without
        emitting it twice (`docs/connector-spec.md` §4.1 rule 3)."""
        ctx = make_ctx()
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            first = await drain(RssConnector.from_config(ctx, Credentials(account_id="a")))

        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response(etag=None))
            second = await drain(
                RssConnector.from_config(ctx, Credentials(account_id="a")),
                first[-1].cursor,
            )

        assert len(signals(second)) == 0
        assert second[0].stats["duplicates"] == 1

    async def test_the_overlap_window_is_applied_per_feed(self) -> None:
        """Provider indexes lag their own timestamps. Resuming exactly at the
        stored mark silently drops a post whose CMS indexed it a minute late."""
        newest = RSS_TIMES[-1]
        state = {
            "feeds": {
                canonicalize_url(NEWS_FEED): {"newest_seen": newest.isoformat().replace("+00:00", "Z")}
            }
        }
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build(), Cursor(checkpoint=state))

        # The newest entry sits exactly on the stored mark and is re-emitted
        # because the cutoff is `mark - overlap_seconds`.
        assert [s.timestamp for s in signals(batches)] == [newest]

    async def test_a_slow_feed_is_not_filtered_by_a_fast_one(self) -> None:
        """The run-wide watermark belongs to the newest feed. Filtering a
        monthly newsletter against a wire service's timestamps would stop it
        emitting anything, permanently."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            batches = await drain(build(make_ctx([NEWS_FEED, ENG_FEED])))

        # The Atom feed is entirely older than the RSS feed's newest item.
        assert len(batches[1].records) == 3
        assert batches[-1].cursor.watermark == RSS_TIMES[-1]

    async def test_resume_after_a_budget_stop_continues_with_no_gap(self) -> None:
        """§10's resume test: stop after page one, restart, get page two."""
        ctx = make_ctx([NEWS_FEED, ENG_FEED], max_pages=1)
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            first = await drain(RssConnector.from_config(ctx, Credentials(account_id="a")))

        assert len(first) == 1
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(304))
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            second = await drain(build(make_ctx([NEWS_FEED, ENG_FEED])), first[-1].cursor)
            conditional = respx.calls[0].request.headers["if-none-match"]

        assert len(signals(second)) == 3
        assert conditional == RSS_ETAG


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


class TestIdentity:
    """`docs/signal-model.md` §4.1, which §7 calls not migratable in place."""

    @pytest.fixture
    async def emitted(self) -> list[Any]:
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            return signals(await drain(build()))

    async def test_guid_is_used_verbatim(self, emitted: list[Any]) -> None:
        """Rule 1. Verbatim rather than hashed so a DLQ record can be pasted
        into the publisher's own tooling."""
        assert emitted[-1].lineage.native_id == RSS_GUID_NEWEST
        assert emitted[0].lineage.native_id == RSS_GUID_OLDEST

    async def test_link_hash_is_used_when_there_is_no_guid(
        self, emitted: list[Any]
    ) -> None:
        """Rule 2, over the *canonicalized* link."""
        import hashlib

        expected = hashlib.sha256(
            canonicalize_url(RSS_LINK_NO_GUID).encode("utf-8")
        ).hexdigest()
        assert emitted[1].lineage.native_id == expected

    async def test_id_is_derived_from_platform_and_native_id(
        self, emitted: list[Any]
    ) -> None:
        """Ids are derived, never assigned: that invariant is what makes all
        five stores idempotent without coordination."""
        for signal in emitted:
            assert signal.id == signal_id(Platform.RSS, signal.lineage.native_id)

    async def test_the_kafka_reference_and_the_signal_agree(self) -> None:
        """`RawRecord.native_id` becomes the reference published to
        `omnisense.records.raw` (§2.6). If the connector derived identity twice
        the message and the Signal it points at could name different items."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build())

        for record, signal in batches[0].records:
            assert record.native_id == signal.lineage.native_id

    async def test_rule_three_is_reachable_and_is_declared(self) -> None:
        """`docs/signal-model.md` §4.1 requires a connector that can reach rule 3
        to say so in its module docstring, because rule-3 identity depends on
        cleaned text and a change to the cleaner forks it. This is the entry
        that gets there: no guid, no link, nothing but a date and a body."""
        body = b"""<?xml version="1.0"?>
        <rss version="2.0"><channel>
          <title>Notices</title><link>https://notices.example.com/</link>
          <description>Standing notices.</description>
          <item>
            <title>Scheduled maintenance</title>
            <pubDate>Wed, 29 Jul 2026 16:40:00 +0000</pubDate>
            <description>The reporting warehouse is unavailable between 0200 and
            0400 UTC on Saturday while the partition rebuild runs.</description>
          </item>
        </channel></rss>"""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response(body, etag=None))
            produced = signals(await drain(build()))

        assert len(produced) == 1
        native_id = produced[0].lineage.native_id
        # A bare sha256 -- neither a guid nor a URL hash, because there was
        # neither a guid nor a URL.
        assert len(native_id) == 64 and produced[0].url is None
        assert produced[0].id == signal_id(Platform.RSS, native_id)

    async def test_tracking_parameters_do_not_fork_identity(
        self, emitted: list[Any]
    ) -> None:
        """The newest item's link carries `utm_*`; a publisher that changes its
        campaign tag between polls must not mint a second Signal."""
        assert emitted[-1].url == "https://news.example.com/2026/07/31/grafana-renewals"
        assert "utm_" not in (emitted[-1].url or "")


# --------------------------------------------------------------------------- #
# Normalize
# --------------------------------------------------------------------------- #


class TestNormalize:
    """One malformed record costs one record, and the field mapping is honest."""

    @pytest.fixture
    async def rss_batch(self) -> EmittedBatch:
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            return (await drain(build()))[0]

    async def test_an_undated_entry_goes_to_the_dlq_and_the_rest_still_emit(
        self, rss_batch: EmittedBatch
    ) -> None:
        """Aborting the page would let one bad item block every good item
        behind it permanently -- the cursor would never advance past it."""
        assert rss_batch.stats["dlq"] == 1
        assert rss_batch.stats["emitted"] == 3
        assert rss_batch.stats["fetched"] == 4

    async def test_the_dlq_record_can_be_attributed(self) -> None:
        """A DLQ record nobody can attribute to an item is one nobody can
        replay."""
        connector = build()
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            await connector.authenticate()
            pages = [page async for page in connector.fetch(Cursor())]
            undated = next(
                record
                for record in pages[0].records
                if record.native_id.endswith("page-corrections")
            )
            with pytest.raises(NormalizationError) as caught:
                await connector.normalize(undated)
            await connector.aclose()

        assert caught.value.native_id == "tag:news.example.com,2026:page-corrections"
        assert "timestamp" in caught.value.message

    async def test_timestamps_are_utc_aware_event_times(
        self, rss_batch: EmittedBatch
    ) -> None:
        """Not ingestion time: trend and forecast agents key off this field
        exclusively, and ingestion time lives in `lineage.fetched_at`."""
        for _, signal in rss_batch.records:
            assert signal.timestamp.tzinfo is not None
            assert signal.timestamp.utcoffset() == timedelta(0)

    async def test_content_is_cleaned_of_markup(self, rss_batch: EmittedBatch) -> None:
        signal = rss_batch.records[0][1]
        assert "<p>" not in signal.content.text
        assert "self-hosted Grafana" in signal.content.text
        assert signal.content.char_count == len(signal.content.text)

    async def test_a_full_body_is_not_marked_truncated(
        self, rss_batch: EmittedBatch
    ) -> None:
        """`content:encoded` is the whole article; flagging it would cap its
        `content_integrity` for the life of the Signal."""
        full = rss_batch.records[0][1]
        assert full.content.truncated is False
        assert full.content.char_count > EXCERPT_CHAR_THRESHOLD

    async def test_a_short_summary_is_marked_truncated(
        self, rss_batch: EmittedBatch
    ) -> None:
        """A teaser trusted like a body is how a report cites half a sentence."""
        teaser = rss_batch.records[1][1]
        assert teaser.content.truncated is True

    async def test_author_identity_is_scoped_to_the_feed_host(
        self, rss_batch: EmittedBatch
    ) -> None:
        """RSS has no author ids, only bylines. "John Smith" writes for more
        than one publication, and merging two authors is unrecoverable where
        forking one is merely annoying."""
        author = rss_batch.records[0][1].author
        assert author is not None
        assert author.platform_author_id == "news.example.com:dmitri@news.example.com"
        assert author.display_name == "Dmitri Sokolov"

    async def test_an_entry_with_no_byline_has_no_author(
        self, rss_batch: EmittedBatch
    ) -> None:
        """Better than an `Author` keyed on nothing."""
        assert rss_batch.records[1][1].author is None

    async def test_enclosures_become_media(self, rss_batch: EmittedBatch) -> None:
        """`enclosures` is a computed member of feedparser's dict subclass and
        is lost by `dict(entry)`; this is the test that catches that."""
        media = rss_batch.records[2][1].media
        assert [(ref.kind, ref.mime_type) for ref in media] == [
            (MediaKind.IMAGE, "image/png")
        ]

    async def test_metadata_is_namespaced_and_shallow(
        self, rss_batch: EmittedBatch
    ) -> None:
        """Un-namespaced keys collide across connectors in one jsonb column and
        one OpenSearch mapping."""
        metadata = rss_batch.records[2][1].metadata
        assert all(key.startswith("rss.") for key in metadata)
        assert metadata["rss.feed_url"] == NEWS_FEED
        assert metadata["rss.feed_title"] == "Ledger Observability Weekly"
        assert metadata["rss.categories"] == ["observability", "pricing"]

    async def test_engagement_is_empty_rather_than_zero(
        self, rss_batch: EmittedBatch
    ) -> None:
        """A feed reports no counters. A zero would read as "nobody engaged"."""
        engagement = rss_batch.records[0][1].engagement
        assert engagement.raw == {}
        assert engagement.compute_score() is None

    async def test_lineage_records_the_acquisition(
        self, rss_batch: EmittedBatch
    ) -> None:
        """Provenance is what turns an assertion into a claim with a receipt."""
        _, signal = rss_batch.records[0]
        assert signal.lineage.connector_slug == "rss"
        assert signal.lineage.connector_version == RssConnector.version
        assert signal.lineage.sync_run_id == "run_1"
        assert signal.lineage.request_fingerprint
        assert signal.lineage.raw_sha256 is not None
        assert signal.lineage.raw_bytes == len(RSS_BYTES)

    async def test_pipeline_version_claims_no_enrichment(
        self, rss_batch: EmittedBatch
    ) -> None:
        """A plausible-looking version here would claim an enrichment that
        never happened, and §7 uses that field to decide what needs
        reprocessing."""
        assert rss_batch.records[0][1].lineage.pipeline_version == "0.0.0"

    async def test_enrichment_fields_are_left_empty(
        self, rss_batch: EmittedBatch
    ) -> None:
        """Filling them would be doing enrichment inside the ingest path."""
        _, signal = rss_batch.records[0]
        assert signal.entities == [] and signal.topics == [] and signal.keywords == []
        assert signal.embeddings == [] and signal.sentiment is None


class TestAtom:
    """Atom 1.0 is a different shape for the same job, mapped by the same map."""

    @pytest.fixture
    async def atom_batch(self) -> EmittedBatch:
        with respx.mock:
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            return (await drain(build(make_ctx(ENG_FEED))))[0]

    async def test_parses_and_orders_oldest_first(
        self, atom_batch: EmittedBatch
    ) -> None:
        assert [signal.timestamp for _, signal in atom_batch.records] == [
            datetime(2026, 7, 26, 8, 0, tzinfo=UTC),
            datetime(2026, 7, 28, 11, 0, tzinfo=UTC),
            datetime(2026, 7, 30, 18, 20, tzinfo=UTC),
        ]

    async def test_published_outranks_updated(self, atom_batch: EmittedBatch) -> None:
        """`Signal.timestamp` is when the observation happened, not when the
        publisher last touched its CMS. The newest entry carries both."""
        newest = atom_batch.records[-1][1]
        assert newest.timestamp == datetime(2026, 7, 30, 18, 20, tzinfo=UTC)

    async def test_falls_back_to_updated_when_there_is_no_published(
        self, atom_batch: EmittedBatch
    ) -> None:
        """Atom makes `<updated>` mandatory and `<published>` optional, so the
        fallback is the common case, not the edge one."""
        assert atom_batch.records[1][1].timestamp == datetime(2026, 7, 28, 11, tzinfo=UTC)

    async def test_content_is_preferred_over_summary(
        self, atom_batch: EmittedBatch
    ) -> None:
        newest = atom_batch.records[-1][1]
        assert "Head-based sampling makes its decision" in newest.content.text
        assert newest.content.truncated is False

    async def test_a_summary_only_entry_is_truncated(
        self, atom_batch: EmittedBatch
    ) -> None:
        excerpt = atom_batch.records[1][1]
        assert excerpt.content.truncated is True
        assert "<p>" not in excerpt.content.text

    async def test_atom_ids_are_used_verbatim(self, atom_batch: EmittedBatch) -> None:
        assert atom_batch.records[-1][1].lineage.native_id.startswith("urn:uuid:")


# --------------------------------------------------------------------------- #
# Failure isolation
# --------------------------------------------------------------------------- #


class TestFeedFailureIsolation:
    """One dead feed out of fifty is a DLQ record, not a failed sync."""

    async def test_a_dead_feed_does_not_stop_the_others(self) -> None:
        """Aborting would mean one unreachable host stops the other
        forty-nine from syncing on every poll until a human notices."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(500))
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            batches = await drain(build(make_ctx([NEWS_FEED, ENG_FEED])))

        assert totals(batches)["dlq"] == 1
        assert totals(batches)["emitted"] == 3

    @pytest.mark.parametrize("status", [400, 404, 410, 429, 500, 503])
    async def test_every_http_failure_is_isolated_the_same_way(
        self, status: int
    ) -> None:
        """A uniform rule is one an operator can reason about; a table of
        per-status behaviours is one they have to look up."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(status))
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            batches = await drain(build(make_ctx([NEWS_FEED, ENG_FEED])))

        assert totals(batches)["dlq"] == 1
        assert totals(batches)["emitted"] == 3

    async def test_a_401_does_not_flag_an_account_that_has_no_credentials(self) -> None:
        """`auth_type` is NONE, so there is nothing to re-authenticate. Raising
        `AuthError` would halt the run and mark the account `needs_reauth` over
        one private URL in a list of public ones."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(401))
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            batches = await drain(build(make_ctx([NEWS_FEED, ENG_FEED])))

        assert totals(batches)["emitted"] == 3

    async def test_a_network_error_is_isolated(self) -> None:
        with respx.mock:
            respx.get(NEWS_FEED).mock(side_effect=httpx.ConnectError("dns"))
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            batches = await drain(build(make_ctx([NEWS_FEED, ENG_FEED])))

        assert totals(batches)["dlq"] == 1
        assert totals(batches)["emitted"] == 3

    async def test_a_body_that_is_not_a_feed_is_isolated(self) -> None:
        """A captive portal or an HTML error page served with a 200."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(
                return_value=httpx.Response(
                    200, content=b"<html><body>Account suspended</body></html>",
                    headers={"Content-Type": "text/html"},
                )
            )
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            batches = await drain(build(make_ctx([NEWS_FEED, ENG_FEED])))

        assert totals(batches)["dlq"] == 1
        assert totals(batches)["emitted"] == 3

    async def test_a_failed_feed_keeps_its_cursor_state(self) -> None:
        """A feed that recovers next poll resumes where it left off instead of
        re-emitting its whole window."""
        feeds = [NEWS_FEED, ENG_FEED]
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            first = await drain(build(make_ctx(feeds)))

        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(503))
            respx.get(ENG_FEED).mock(return_value=httpx.Response(304))
            second = await drain(build(make_ctx(feeds)), first[-1].cursor)

        key = canonicalize_url(NEWS_FEED)
        assert second[-1].cursor.checkpoint["feeds"][key] == first[-1].cursor.checkpoint["feeds"][key]

    async def test_the_dlq_marker_names_the_feed(self) -> None:
        """The connector holds no logger the runtime reads and no store it can
        write to, so a record is its only channel for "this feed is broken"."""
        connector = build(make_ctx([NEWS_FEED, ENG_FEED]))
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(404))
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            await connector.authenticate()
            pages = [page async for page in connector.fetch(Cursor())]
            marker = pages[0].records[0]
            with pytest.raises(NormalizationError) as caught:
                await connector.normalize(marker)
            await connector.aclose()

        assert caught.value.details["feed_url"] == NEWS_FEED
        assert caught.value.details["cause_class"] == "PermanentError"
        assert marker.payload[ENVELOPE_KEY]["error_class"] == "PermanentError"

    async def test_total_failure_is_raised_not_swallowed(self) -> None:
        """Every feed dead at once is a fact about us -- DNS, egress, a blocked
        User-Agent -- and a successful run with zero records would hide it."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(503))
            respx.get(ENG_FEED).mock(side_effect=httpx.ConnectError("dns"))
            with pytest.raises(TransientError):
                await drain(build(make_ctx([NEWS_FEED, ENG_FEED])))

    async def test_total_failure_keeps_the_error_class(self) -> None:
        """`PermanentError` means a human must fix the configuration;
        retrying a 404 forever is not a policy."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(404))
            with pytest.raises(PermanentError):
                await drain(build())

    async def test_a_dead_feed_still_yields_its_page_before_the_raise(self) -> None:
        """The DLQ record is the record of the failure; losing it because the
        run later failed would leave nothing to triage."""
        batches: list[EmittedBatch] = []
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(503))
            with pytest.raises(TransientError):
                async for batch in build().run():
                    batches.append(batch)

        assert len(batches) == 1 and batches[0].stats["dlq"] == 1


# --------------------------------------------------------------------------- #
# Rate limiting and dedup
# --------------------------------------------------------------------------- #


class TestRateLimiting:
    """Every feed is a different origin, so the host bucket is the real limit."""

    async def test_every_request_takes_a_per_host_bucket(self) -> None:
        """A connector-wide limit would either throttle a thousand hosts as one
        or let sixty requests a minute land on one hobbyist's VPS."""
        ctx = make_ctx([NEWS_FEED, ENG_FEED])
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            await drain(RssConnector.from_config(ctx, Credentials(account_id="acct_1")))

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert limiter.acquired == [
            ["os:rl:rss", "os:rl:rss:acct_1", "os:rl:host:news.example.com"],
            ["os:rl:rss", "os:rl:rss:acct_1", "os:rl:host:eng.example.org"],
        ]

    async def test_a_conditional_request_still_costs_a_token(self) -> None:
        """A limiter that only counted the expensive requests would let a
        thousand unchanged feeds hammer one host for free."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            first = await drain(build())

        ctx = make_ctx()
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=httpx.Response(304))
            await drain(
                RssConnector.from_config(ctx, Credentials(account_id="acct_1")),
                first[-1].cursor,
            )

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert len(limiter.acquired) == 1

    async def test_a_429_retry_after_reaches_the_limiter(self) -> None:
        """§10's mandatory 429 test. The connector cannot honour a `Retry-After`
        itself -- it may not sleep -- so the instruction is routed to the shared
        bucket instead of being dropped, which is how an integration earns a
        ban."""
        ctx = make_ctx([NEWS_FEED, ENG_FEED])
        with respx.mock:
            respx.get(NEWS_FEED).mock(
                return_value=httpx.Response(429, headers={"Retry-After": "120"})
            )
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            await drain(RssConnector.from_config(ctx, Credentials(account_id="acct_1")))

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert [hint.retry_after_seconds for hint in limiter.observed] == [120.0]

    async def test_a_missing_limiter_fails_open(self) -> None:
        """architecture.md §7.3: outbound limiting degrades open when Redis is
        down, because failing closed would halt ingestion on every cache blip."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build(make_ctx(limiter=None)))

        assert len(signals(batches)) == 3


class TestDedup:
    """§10: feeding the same page twice emits N Signals, not 2N."""

    async def test_the_same_feed_twice_emits_once(self) -> None:
        ctx = make_ctx()
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            first = await drain(RssConnector.from_config(ctx, Credentials(account_id="a")))
            # Same context, so the same seen-set, and no cursor -- which is the
            # only way to isolate dedup from the incremental cutoff.
            second = await drain(RssConnector.from_config(ctx, Credentials(account_id="a")))

        assert len(signals(first)) == 3
        assert len(signals(second)) == 0
        assert totals(second)["duplicates"] == 3

    async def test_syndication_across_two_feeds_collapses_to_one(self) -> None:
        """Layer 2 is why RSS is worth doing carefully: the same wire story
        appears in a dozen feeds under a dozen different GUIDs, and only the
        canonicalized content hash collapses them."""
        ctx = make_ctx([NEWS_FEED, "https://mirror.example.net/feed.xml"])
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            respx.get("https://mirror.example.net/feed.xml").mock(
                return_value=feed_response(
                    RSS_BYTES.replace(b"tag:news.example.com,2026:post", b"tag:mirror,2026:post"),
                    etag=None,
                )
            )
            batches = await drain(RssConnector.from_config(ctx, Credentials(account_id="a")))

        assert len(batches[0].records) == 3
        # Different GUIDs, identical bodies: layer 1 misses, layer 2 catches.
        assert len(batches[1].records) == 0
        assert totals(batches)["duplicates"] == 3


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


class TestLifecycle:
    async def test_authenticate_is_idempotent(self) -> None:
        """The runtime calls it once at start and at most once more after a
        401; a second call must not orphan the first connection pool."""
        connector = build()
        await connector.authenticate()
        client = connector._client
        await connector.authenticate()
        assert connector._client is client
        await connector.aclose()

    async def test_the_client_is_released_when_a_consumer_gives_up(self) -> None:
        """The leak path: a consumer that stops iterating early."""
        connector = build(make_ctx([NEWS_FEED, ENG_FEED]))
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            runner = connector.run()
            await runner.__anext__()
            await runner.aclose()

        assert connector._client is None

    async def test_an_identifying_user_agent_is_sent(self) -> None:
        """It is a condition of use for most publishers and the first thing an
        operator blocks when it is absent."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            await drain(build())
            agent = respx.calls[0].request.headers["user-agent"]

        assert agent.startswith("omnisense/")

    async def test_a_run_produces_signals_the_strict_model_accepts(self) -> None:
        """The end-to-end assertion: bytes on the wire to canonical Signals,
        through the real `BaseConnector.run()` template, with every model
        validator live."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            respx.get(ENG_FEED).mock(return_value=feed_response(ATOM_BYTES, etag=None))
            batches = await drain(build(make_ctx([NEWS_FEED, ENG_FEED])))

        produced = signals(batches)
        assert len(produced) == 6
        for signal in produced:
            assert signal.source is SourceCategory.NEWS
            assert signal.platform is Platform.RSS
            assert signal.is_canonical
            # `status` is RAW until the enrichment pipeline has run, so a
            # freshly connected Signal is deliberately *not* retrievable yet
            # (`docs/signal-model.md` §5.4).
            assert signal.is_retrievable is False
            assert signal.content.text
            # Round-trips through the wire format the runtime will use.
            assert json.loads(signal.model_dump_json())["id"] == signal.id

    async def test_the_raw_payload_is_json_serializable(self) -> None:
        """The runtime PUTs `payload` to R2 as JSON. feedparser hands back
        `struct_time` objects, which no encoder accepts."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build())

        for record, _ in batches[0].records:
            assert json.loads(json.dumps(record.payload))["title"]

    async def test_raw_bytes_are_the_feed_document(self) -> None:
        """RSS's unit of retrieval is the document, not the item. The R2 key is
        content-addressed, so N entries from one poll reference one object."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build())

        records = [record for record, _ in batches[0].records]
        assert all(record.raw_bytes == RSS_BYTES for record in records)
        assert {record.content_type for record in records} == {"application/rss+xml"}

    async def test_health_check_opens_no_socket(self) -> None:
        """There is no provider session to validate, and probing a publisher on
        every `/health` poll is exactly the impoliteness this connector exists
        to avoid."""
        with respx.mock(assert_all_called=False):
            report = await build().health_check()

        assert report.healthy is True
        assert len(respx.calls) == 0


class TestPayloadHygiene:
    """What the connector computed must be distinguishable from what the
    publisher said."""

    async def test_everything_derived_lives_under_one_key(self) -> None:
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build())

        record: RawRecord = batches[0].records[0][0]
        envelope = record.payload[ENVELOPE_KEY]
        assert envelope["feed_url"] == NEWS_FEED
        assert envelope["native_id"] == record.native_id
        assert envelope["bozo"] is False

    async def test_the_feed_url_travels_with_every_record(self) -> None:
        """An entry does not name the feed it came from, so without this a DLQ
        record is unattributable."""
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(build())

        for record, signal in batches[0].records:
            assert record.source_url == NEWS_FEED
            assert signal.metadata["rss.feed_url"] == NEWS_FEED

    async def test_the_request_fingerprint_carries_no_credentials(self) -> None:
        """It makes a fetch reproducible; it must not make it re-authenticable."""
        connector = RssConnector.from_config(
            make_ctx(),
            Credentials(account_id="a", secrets={"username": "u", "password": "hunter2"}),
        )
        with respx.mock:
            respx.get(NEWS_FEED).mock(return_value=feed_response())
            batches = await drain(connector)

        for record, _ in batches[0].records:
            assert record.request_fingerprint is not None
            assert "hunter2" not in record.request_fingerprint
