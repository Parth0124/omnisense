"""Unit tests for `connectors/social/reddit.py`.

Reddit is the first connector that has to reconcile a real provider with the
`BaseConnector` contract, and every test here targets a place where the two
disagree and the disagreement is silent:

- `created_utc` is epoch *seconds*, and read as a naive local datetime it shifts
  every trend by the worker's UTC offset without anything raising;
- `author_fullname` and `author` are both present on every payload, and keying on
  the second forks an author's history the first time they rename;
- Reddit pages newest-first while the contract requires oldest-first, so the
  ordering and the watermark are asserted across batches rather than within one;
- a descent cut short by the page budget must not move the watermark, because the
  records it never reached would then sit below it forever;
- `X-Ratelimit-Remaining` is a float string and `X-Ratelimit-Reset` is a delta,
  both of which the inherited header parser silently reads as nothing;
- and a 401 must surface as `AuthError` on the first response, because a client
  that loops on rejected credentials earns an application-level ban rather than an
  account-level one.

Everything runs against `respx` and a recorded listing under
`tests/fixtures/payloads/`. Nothing here contacts reddit.com: an unmocked request
raises inside respx, so a route that goes missing fails the test rather than
reaching the network.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from connectors import registry
from connectors.auth.token_store import InMemoryTokenStore
from connectors.base import BaseConnector
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    NormalizationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.protocol import (
    Credentials,
    Cursor,
    EmittedBatch,
    FetchPage,
    RateLimitHint,
    RawRecord,
    SyncContext,
)
from connectors.social.reddit import (
    API_BASE,
    TOKEN_URL,
    RedditConnector,
)
from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "payloads"
LISTING_URL = f"{API_BASE}/r/selfhosted/new"
COMMENTS_URL = f"{API_BASE}/r/selfhosted/comments"

USER_AGENT = "omnisense:reddit-connector:0.1.0 (by /u/omnisense_bot)"

PAGE1 = "reddit_listing_new_page1.json"
PAGE2 = "reddit_listing_new_page2.json"
COMMENTS = "reddit_listing_comments.json"
T_NOW = datetime(2026, 7, 28, 13, 0, 0, tzinfo=UTC)

#: Event times in the fixture, newest first. Duplicated here rather than read back
#: out of the payload so that a fixture edited by accident fails a test instead of
#: quietly moving the expectations with it.
P1A = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)
P1B = datetime(2026, 7, 28, 11, 50, tzinfo=UTC)
P2A = datetime(2026, 7, 28, 11, 10, tzinfo=UTC)
P2B = datetime(2026, 7, 28, 11, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fakes and helpers
# --------------------------------------------------------------------------- #


class FakeLimiter:
    """Records acquisitions and observations instead of talking to Redis."""

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


def clock() -> datetime:
    """A fixed clock.

    Injected rather than patched so `reset_at` arithmetic can be asserted against
    a literal instead of against a window around `now()`.
    """
    return T_NOW


def load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


def child(name: str, fullname: str) -> dict[str, Any]:
    """One listing child from a fixture, by fullname."""
    for entry in load(name)["data"]["children"]:
        if entry["data"]["name"] == fullname:
            return entry
    raise AssertionError(f"{fullname} is not in {name}")


def record(entry: dict[str, Any]) -> RawRecord:
    """Wrap a child the way `fetch()` does.

    `native_id` is set from the fullname deliberately: that agreement between the
    fetch and normalize stages is itself part of the contract, and a helper that
    derived it differently would hide a break in it.
    """
    return RawRecord(
        native_id=entry["data"]["name"],
        payload=entry,
        fetched_at=T_NOW,
        content_type="application/json",
    )


def raw(name: str, fullname: str) -> RawRecord:
    """The record `fetch()` would build for one fixture child."""
    return record(child(name, fullname))


def make_ctx(**overrides: Any) -> SyncContext:
    defaults: dict[str, Any] = {
        "connector_slug": "reddit",
        "account_id": "acct_1",
        "run_id": "run_1",
        "params": {"subreddit": "selfhosted"},
        "limiter": FakeLimiter(),
        "dedup": FakeDedup(),
        "user_agent": USER_AGENT,
    }
    defaults.update(overrides)
    return SyncContext(**defaults)


def make_credentials(**overrides: Any) -> Credentials:
    secrets = {"client_id": "client-abc", "client_secret": "s3cret"}
    secrets.update(overrides.pop("secrets", {}))
    return Credentials(account_id="acct_1", secrets=secrets, **overrides)


def build(
    ctx: SyncContext | None = None, credentials: Credentials | None = None
) -> RedditConnector:
    return RedditConnector(
        ctx or make_ctx(),
        credentials or make_credentials(),
        token_store=InMemoryTokenStore(),
        now=clock,
    )


def mock_token(access_token: str = "token-1") -> respx.Route:
    return respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200, json={"access_token": access_token, "token_type": "bearer", "expires_in": 3600}
        )
    )


def listing_response(name: str, **headers: str) -> httpx.Response:
    """A recorded listing, with Reddit's rate-limit headers in their real shape."""
    return httpx.Response(
        200,
        json=load(name),
        headers={
            "X-Ratelimit-Used": "42.0",
            "X-Ratelimit-Remaining": "58.0",
            "X-Ratelimit-Reset": "240",
            "Set-Cookie": "session_tracker=abc; Path=/",
            "Content-Type": "application/json",
            **headers,
        },
    )


def mock_listing(*responses: httpx.Response, url: str = LISTING_URL) -> respx.Route:
    """One route answering successive requests, which is how `after` is exercised."""
    return respx.get(url).mock(side_effect=list(responses))


async def drain(
    connector: RedditConnector, cursor: Cursor | None = None
) -> list[EmittedBatch]:
    return [batch async for batch in connector.run(cursor)]


def signals(batches: Sequence[EmittedBatch]) -> list[Signal]:
    return [signal for batch in batches for _, signal in batch.records]


# --------------------------------------------------------------------------- #
# Declaration
# --------------------------------------------------------------------------- #


class TestDeclaration:
    """The ClassVar block the scheduler reads before anything is instantiated."""

    def test_declares_reddit_social_oauth2(self) -> None:
        assert RedditConnector.slug == "reddit"
        assert RedditConnector.platform is Platform.REDDIT
        assert RedditConnector.category is SourceCategory.SOCIAL
        assert RedditConnector.auth_type is AuthType.OAUTH2

    def test_passes_the_registry_declaration_gate(self) -> None:
        """The gate that turns a platform/category disagreement into an import
        error instead of four thousand rejected records after the quota is spent.

        `connectors/__init__.py` registers every shipped connector at import, and
        registration is what runs the declaration validation -- so a successful
        lookup here means the gate passed. A bad declaration would have raised
        during import and this module would never have loaded.
        """
        import connectors  # noqa: F401  -- the import under test

        assert registry.get("reddit") is RedditConnector

    def test_is_enablable_without_a_tos_review(self) -> None:
        """Reddit has an official API for this use case, unlike the four
        connectors §9 ships with the flag set."""
        assert RedditConnector.requires_tos_review is False

    def test_declares_no_backfill(self) -> None:
        """Reddit truncates listings at roughly a thousand items, so a backfill
        mode would differ only in which cursor row it wrote."""
        assert RedditConnector.supports_backfill is False
        assert RedditConnector.supports_incremental is True


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class TestUserAgent:
    """Reddit throttles unidentified clients far harder than identified ones, and
    the throttle arrives as a 429 that looks exactly like a real quota wall. Every
    rejection below is one an operator would otherwise debug as a provider fault."""

    def test_the_process_default_is_refused(self) -> None:
        """`SyncContext.user_agent` defaults to `omnisense/0.1` -- the value an
        operator gets by not deciding."""
        ctx = make_ctx(user_agent="omnisense/0.1")
        with pytest.raises(ConnectorConfigurationError, match="default"):
            build(ctx)

    def test_a_library_default_is_refused(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="library"):
            build(make_ctx(user_agent="python-requests/2.32.3"))

    def test_an_agent_with_no_contact_is_refused(self) -> None:
        """Identifying the app but not the operator leaves an admin with nothing
        to do but block the client."""
        with pytest.raises(ConnectorConfigurationError, match="contact"):
            build(make_ctx(user_agent="omnisense-collector/2.4.1"))

    def test_an_empty_agent_is_refused(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="user_agent"):
            build(make_ctx(user_agent="   "))

    def test_a_url_or_an_email_counts_as_a_contact(self) -> None:
        """The documented shape is `(by /u/name)`, but refusing an equally
        reachable operator over spelling is pedantry."""
        build(make_ctx(user_agent="omnisense:collector:0.1 (+https://omnisense.dev)"))
        build(make_ctx(user_agent="omnisense:collector:0.1 (ops@omnisense.dev)"))

    def test_the_credential_record_outranks_the_process_default(self) -> None:
        """The agent names the app registration the client id belongs to, and the
        two are rotated together."""
        connector = build(
            make_ctx(user_agent="omnisense/0.1"),
            make_credentials(extra={"user_agent": USER_AGENT}),
        )
        assert connector._user_agent == USER_AGENT


class TestConfiguration:
    def test_a_missing_subreddit_is_a_configuration_error(self) -> None:
        with pytest.raises(ConnectorConfigurationError, match="subreddit"):
            build(make_ctx(params={}))

    def test_a_subreddit_that_is_not_a_name_is_refused(self) -> None:
        """The value is interpolated into the request path, so anything else is a
        request for a resource nobody configured."""
        with pytest.raises(ConnectorConfigurationError, match="valid subreddit"):
            build(make_ctx(params={"subreddit": "selfhosted/../../api/v1/me"}))

    def test_several_subreddits_become_one_multireddit_path(self) -> None:
        """One listing is one cursor: fanning out would give a run several
        watermarks under a single params_hash, and the runtime persists one."""
        connector = build(make_ctx(params={"subreddits": ["selfhosted", "homelab"]}))
        assert connector._listing_url() == f"{API_BASE}/r/selfhosted+homelab/new"

    def test_a_ranked_listing_is_refused(self) -> None:
        """`hot` paged with `after` walks popularity, not time, so the watermark
        would advance past records the ranker had not promoted yet -- and the run
        would succeed while losing them."""
        with pytest.raises(ConnectorConfigurationError, match="chronological"):
            build(make_ctx(params={"subreddit": "selfhosted", "listing": "hot"}))

    def test_a_page_size_above_the_providers_ceiling_is_refused(self) -> None:
        """Reddit clamps it silently, which would leave the pagination arithmetic
        resting on a number the provider never agreed to."""
        with pytest.raises(ConnectorConfigurationError, match="between 1 and 100"):
            build(make_ctx(params={"subreddit": "selfhosted", "limit": 500}))

    def test_a_missing_secret_is_a_configuration_error_not_a_key_error(self) -> None:
        """A bare KeyError carries no error class, so the runtime would file a
        misconfigured account as an unhandled crash."""
        with pytest.raises(ConnectorConfigurationError, match="client_secret"):
            build(credentials=Credentials(account_id="acct_1", secrets={"client_id": "c"}))

    @respx.mock
    async def test_from_config_performs_no_io(self) -> None:
        """The scheduler builds connectors to inspect them; construction that
        opened a socket would make that impossible to do cheaply."""
        token = mock_token()
        listing = mock_listing(listing_response(PAGE1))
        RedditConnector.from_config(make_ctx(), make_credentials())
        assert (token.call_count, listing.call_count) == (0, 0)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


class TestAuthentication:
    @respx.mock
    async def test_uses_the_client_credentials_grant_with_basic_auth(self) -> None:
        """Reddit rejects the body form of client authentication, and reports it
        as `invalid_client` -- indistinguishable from a wrong secret."""
        token = mock_token()
        connector = build()
        await connector.authenticate()
        await connector.aclose()

        request = token.calls[0].request
        assert b"grant_type=client_credentials" in request.content
        assert request.headers["Authorization"].startswith("Basic ")
        assert request.headers["User-Agent"] == USER_AGENT

    @respx.mock
    async def test_authenticate_is_idempotent(self) -> None:
        """The runtime calls it at the start of a run and again after a 401; a
        second mint per call would double the load on the most rate-limited
        endpoint Reddit has."""
        token = mock_token()
        connector = build()
        await connector.authenticate()
        await connector.authenticate()
        await connector.aclose()
        assert token.call_count == 1

    @respx.mock
    async def test_the_bearer_token_and_agent_ride_on_every_listing_call(self) -> None:
        mock_token(access_token="token-xyz")
        listing = mock_listing(listing_response(PAGE2))
        await drain(build())

        request = listing.calls[0].request
        assert request.headers["Authorization"] == "Bearer token-xyz"
        assert request.headers["User-Agent"] == USER_AGENT

    @respx.mock
    async def test_a_401_surfaces_as_auth_error_and_is_not_retried(self) -> None:
        """Terminal on the first recurrence: a client that loops on rejected
        credentials earns an application-level ban, not an account-level one."""
        mock_token()
        listing = respx.get(LISTING_URL).mock(
            return_value=httpx.Response(401, json={"message": "Unauthorized", "error": 401})
        )
        with pytest.raises(AuthError) as caught:
            await drain(build())

        assert listing.call_count == 1, "a 401 must not be retried inside the connector"
        assert caught.value.retryable is False
        assert caught.value.account_id == "acct_1", "the runtime flags the row by this"

    @respx.mock
    async def test_a_401_expires_the_cached_token_so_re_auth_mints_a_new_one(self) -> None:
        """Without this the runtime's one permitted re-authentication replays the
        token that was just rejected."""
        token = mock_token()
        respx.get(LISTING_URL).mock(return_value=httpx.Response(401, json={}))
        connector = build()
        with pytest.raises(AuthError):
            await drain(connector)

        await connector.authenticate()
        await connector.aclose()
        assert token.call_count == 2

    @respx.mock
    async def test_a_private_subreddit_is_a_configuration_error_not_an_auth_error(self) -> None:
        """Reddit answers a revoked token and an unreadable subreddit with the same
        403. Filing the second as auth flags a working account `needs_reauth` and
        sends an operator to fix nothing."""
        mock_token()
        respx.get(LISTING_URL).mock(
            return_value=httpx.Response(403, json={"reason": "private", "error": 403})
        )
        with pytest.raises(ConnectorConfigurationError) as caught:
            await drain(build())
        assert not isinstance(caught.value, AuthError)
        assert caught.value.details["reason"] == "private"


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #


class TestNormalize:
    @pytest.fixture
    def connector(self) -> RedditConnector:
        return build()

    async def test_created_utc_is_epoch_seconds_in_utc(
        self, connector: RedditConnector
    ) -> None:
        """The field is an epoch float. Read as a naive local datetime it shifts
        every trend by the worker's offset and nothing raises."""
        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None
        assert signal.timestamp == P1A
        assert signal.timestamp.tzinfo is not None

    async def test_native_id_is_the_fullname_and_the_id_is_derived_from_it(
        self, connector: RedditConnector
    ) -> None:
        """Rule 1 of §4.1, verbatim rather than hashed, so a DLQ record names
        something a human can paste into Reddit's own UI."""
        from models.signal import signal_id

        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None
        assert signal.lineage.native_id == "t3_p1a"
        assert signal.id == signal_id(Platform.REDDIT, "t3_p1a")

    async def test_author_is_keyed_on_the_t2_id_not_the_handle(
        self, connector: RedditConnector
    ) -> None:
        """Handles are renameable. Keying on one forks an author's history the
        first time they rename, and nothing downstream can tell an id from a
        handle afterwards."""
        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None and signal.author is not None
        assert signal.author.platform_author_id == "t2_9xk3q"
        assert signal.author.handle == "quietoperator"
        assert signal.author.profile_url == "https://www.reddit.com/user/quietoperator"

    async def test_an_author_with_no_stable_id_is_dropped_not_promoted(
        self, connector: RedditConnector
    ) -> None:
        """A handle is not an identity. Losing the author entirely is recoverable;
        a handle sitting in `platform_author_id` is not, because it looks correct."""
        entry = child(PAGE1, "t3_p1a")
        entry["data"].pop("author_fullname")
        signal = await connector.normalize(record(entry))
        assert signal is not None and signal.author is None

    async def test_a_deleted_author_is_dropped_rather_than_failed(
        self, connector: RedditConnector
    ) -> None:
        """Dropping is expected and counted separately; a NormalizationError here
        would bury real mapping bugs under the most common thing a listing holds."""
        assert await connector.normalize(raw(PAGE1, "t3_p1c")) is None

    async def test_a_removed_body_is_dropped(self, connector: RedditConnector) -> None:
        assert await connector.normalize(raw(PAGE1, "t3_p1d")) is None

    async def test_a_removed_link_post_is_dropped(self, connector: RedditConnector) -> None:
        """A link post has no body to blank, so the removal shows up only in
        `removed_by_category`."""
        assert await connector.normalize(raw(PAGE1, "t3_p1e")) is None

    async def test_engagement_carries_the_platform_counters_verbatim(
        self, connector: RedditConnector
    ) -> None:
        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None
        assert signal.engagement.raw == {
            "score": 412,
            "num_comments": 87,
            "upvote_ratio": 0.93,
            "subreddit_subscribers": 512340,
            "crossposts": 3,
        }

    async def test_the_normalized_axes_stay_empty(self, connector: RedditConnector) -> None:
        """The axes are percentiles within a cohort; a connector holding one
        record cannot know a percentile, and a guess would be compared across
        platforms as if it were one."""
        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None
        assert signal.engagement.available_axes() == {}
        assert signal.engagement.compute_score() is None

    async def test_metadata_is_namespaced_and_shallow(
        self, connector: RedditConnector
    ) -> None:
        """Un-namespaced keys collide across connectors in one jsonb column and one
        OpenSearch mapping; deep ones explode that mapping."""
        from models.signal import _json_depth

        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None
        assert signal.metadata == {
            "reddit.subreddit": "selfhosted",
            "reddit.flair": "Guide",
            "reddit.is_self": True,
        }
        assert all(key.startswith("reddit.") for key in signal.metadata)
        assert _json_depth(signal.metadata) <= 3

    async def test_a_false_flag_survives_but_an_absent_one_does_not(
        self, connector: RedditConnector
    ) -> None:
        """`is_self: false` is a fact about the post; a missing flair is the
        absence of one, and storing `None` for it would make every consumer test
        for two kinds of nothing."""
        signal = await connector.normalize(raw(PAGE1, "t3_p1b"))
        assert signal is not None
        assert signal.metadata["reddit.is_self"] is False
        assert "reddit.flair" not in signal.metadata

    async def test_url_is_the_permalink_not_the_outbound_link(
        self, connector: RedditConnector
    ) -> None:
        """The observation is the Reddit post, not the page it links to -- and the
        outbound URL in this fixture carries a tracking parameter that would end up
        in a citation."""
        signal = await connector.normalize(raw(PAGE1, "t3_p1b"))
        assert signal is not None
        assert signal.url == (
            "https://www.reddit.com/r/selfhosted/comments/p1b/benchmarks_for_the_new_mini_pcs/"
        )

    async def test_the_body_is_cleaned_and_the_title_is_kept_separately(
        self, connector: RedditConnector
    ) -> None:
        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None
        assert signal.content.title == "Migrated my whole stack off the cloud, here is what broke"
        assert signal.content.text == (
            "Six weeks in.\n\nPostgres was fine. Object storage was not."
        )
        assert signal.content.char_count == len(signal.content.text)
        assert signal.content.truncated is False

    async def test_lineage_names_the_connector_the_run_and_the_request(
        self, connector: RedditConnector
    ) -> None:
        """Provenance is what turns an assertion in a report into a claim with a
        receipt."""
        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None
        assert signal.lineage.connector_slug == "reddit"
        assert signal.lineage.connector_version == RedditConnector.version
        assert signal.lineage.sync_run_id == "run_1"
        assert signal.lineage.fetched_at == T_NOW
        assert signal.lineage.pipeline_version == "0.0.0", "a connector runs no stage"

    async def test_a_comment_maps_its_body_and_its_parent_post(
        self, connector: RedditConnector
    ) -> None:
        """Without `link_id` a comment is an orphan opinion: nothing can attach it
        to what it is about."""
        signal = await connector.normalize(raw(COMMENTS, "t1_c1a"))
        assert signal is not None
        assert signal.lineage.native_id == "t1_c1a"
        assert signal.content.text == "The object storage part matches my experience exactly."
        assert signal.metadata["reddit.link_id"] == "t3_p1a"
        assert signal.metadata["reddit.flair"] == "homelab"

    async def test_a_removed_comment_is_dropped(self, connector: RedditConnector) -> None:
        assert await connector.normalize(raw(COMMENTS, "t1_c1b")) is None

    async def test_an_unknown_kind_is_a_dlq_record_not_a_drop(
        self, connector: RedditConnector
    ) -> None:
        """A listing holding something other than a post or a comment means the
        endpoint is not what this connector thinks it is -- worth knowing about,
        which a silent drop is not."""
        entry = child(PAGE1, "t3_p1a")
        entry["kind"] = "t5"
        with pytest.raises(NormalizationError):
            await connector.normalize(record(entry))

    async def test_a_fullname_disagreement_is_refused(
        self, connector: RedditConnector
    ) -> None:
        """The runtime keys R2 and Kafka off `native_id` while the stores key off
        `Signal.id`; if the two disagreed, one item would exist under two
        identities and nothing would notice for months."""
        entry = child(PAGE1, "t3_p1a")
        raw = RawRecord(native_id="t3_somethingelse", payload=entry, fetched_at=T_NOW)
        with pytest.raises(NormalizationError, match="fullname"):
            await connector.normalize(raw)

    async def test_a_payload_with_no_timestamp_is_a_dlq_record(
        self, connector: RedditConnector
    ) -> None:
        """Attributable, too: the DLQ record carries the fullname, so a fixed
        mapper can be replayed against it without re-fetching a post that may by
        then be deleted."""
        entry = child(PAGE1, "t3_p1a")
        entry["data"].pop("created_utc")
        with pytest.raises(NormalizationError) as caught:
            await connector.normalize(record(entry))
        assert caught.value.native_id == "t3_p1a"


# --------------------------------------------------------------------------- #
# Pagination
# --------------------------------------------------------------------------- #


class TestPagination:
    @respx.mock
    async def test_pages_with_after_and_emits_oldest_first(self) -> None:
        """Reddit walks backwards in time from the top of /new; the contract
        requires forwards. A newest-first pager that died mid-run would commit a
        watermark past records it never emitted."""
        mock_token()
        listing = mock_listing(
            listing_response(PAGE1),
            listing_response(PAGE2),
        )
        batches = await drain(build())

        assert listing.call_count == 2
        assert "after" not in listing.calls[0].request.url.params
        assert listing.calls[1].request.url.params["after"] == "t3_p1e"

        emitted = [signal.timestamp for signal in signals(batches)]
        assert emitted == sorted(emitted), "the emitted stream must move forward in time"
        assert emitted == [P2B, P2A, P1B, P1A]

    @respx.mock
    async def test_every_request_asks_for_unescaped_json(self) -> None:
        """Without `raw_json=1` Reddit HTML-escapes `&` inside every body, and the
        escape survives cleaning, embedding and quotation in a report."""
        mock_token()
        listing = mock_listing(listing_response(PAGE2))
        await drain(build())
        assert listing.calls[0].request.url.params["raw_json"] == "1"

    @respx.mock
    async def test_the_watermark_advances_once_per_batch_and_never_backwards(self) -> None:
        mock_token()
        mock_listing(
            listing_response(PAGE1),
            listing_response(PAGE2),
        )
        batches = await drain(build())

        watermarks = [batch.cursor.watermark for batch in batches]
        assert watermarks == [P2A, P1A]
        assert all(batch.cursor.page_token is None for batch in batches)

    @respx.mock
    async def test_the_descent_stops_at_the_watermark(self) -> None:
        """The whole point of a cursor: a resumed run must not walk the listing
        back to the beginning every fifteen minutes."""
        mock_token()
        listing = mock_listing(listing_response(PAGE1))
        await drain(build(), Cursor(watermark=datetime(2026, 7, 28, 11, 45, tzinfo=UTC)))
        assert listing.call_count == 1

    @respx.mock
    async def test_a_truncated_descent_parks_progress_instead_of_moving_the_watermark(
        self,
    ) -> None:
        """The expensive silent failure: advancing the watermark to the top of a
        descent that never reached the previous one leaves the records in between
        below the watermark, unemitted, and nothing ever goes back for them."""
        mock_token()
        mock_listing(listing_response(PAGE1))
        batches = await drain(build(make_ctx(max_pages=1)))

        cursor = batches[-1].cursor
        assert cursor.watermark is None, "nothing durable is below the top of the listing yet"
        assert cursor.page_token == "t3_p1e", "resume by descending further, not from the top"
        assert cursor.checkpoint["pending_watermark"] == P1A.isoformat()

    @respx.mock
    async def test_a_parked_descent_resumes_downward_and_then_promotes(self) -> None:
        """The parked value is newer than everything the continuation emits, so
        promoting it before the last page would jump the watermark over records
        this run has not yielded yet."""
        mock_token()
        listing = mock_listing(listing_response(PAGE2))
        parked = Cursor(
            page_token="t3_p1e",
            checkpoint={"pending_watermark": P1A.isoformat()},
        )
        batches = await drain(build(), parked)

        assert listing.calls[0].request.url.params["after"] == "t3_p1e"
        cursor = batches[-1].cursor
        assert cursor.watermark == P1A
        assert cursor.page_token is None
        assert cursor.checkpoint == {}

    @respx.mock
    async def test_the_comments_listing_is_a_separate_endpoint(self) -> None:
        """`/r/x/comments` is the other chronological listing Reddit serves, and
        it returns `t1` items -- a different payload shape behind the same pager."""
        mock_token()
        listing = mock_listing(listing_response(COMMENTS), url=COMMENTS_URL)
        ctx = make_ctx(params={"subreddit": "selfhosted", "listing": "comments"})
        batches = await drain(build(ctx))

        assert listing.call_count == 1
        assert [signal.lineage.native_id for signal in signals(batches)] == ["t1_c1a"]

    @respx.mock
    async def test_deleted_records_are_dropped_not_dlqd(self) -> None:
        """Three of the five posts in the fixture were emptied by Reddit. They are
        expected, so they must not land in the counter that means "a mapping bug
        needs looking at"."""
        mock_token()
        mock_listing(listing_response(PAGE1))
        batches = await drain(build(make_ctx(max_pages=1)))

        stats = batches[-1].stats
        assert (stats["fetched"], stats["emitted"], stats["dropped"], stats["dlq"]) == (5, 2, 3, 0)

    @respx.mock
    async def test_a_rate_limit_slot_is_acquired_before_every_request(self) -> None:
        ctx = make_ctx()
        mock_token()
        mock_listing(
            listing_response(PAGE1),
            listing_response(PAGE2),
        )
        await drain(build(ctx))

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert len(limiter.acquired) == 2
        assert limiter.acquired[0] == [
            "os:rl:reddit",
            "os:rl:reddit:acct_1",
            "os:rl:host:oauth.reddit.com",
        ]

    @respx.mock
    async def test_the_client_is_released_even_when_the_consumer_stops_early(self) -> None:
        """The leak path: a consumer that abandons the generator must still return
        the connection pool."""
        mock_token()
        mock_listing(
            listing_response(PAGE1),
            listing_response(PAGE2),
        )
        connector = build()
        agen = connector.run()
        await agen.__anext__()
        await agen.aclose()
        assert connector._client is None


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

REDDIT_HEADERS = {
    "X-Ratelimit-Used": "42.0",
    "X-Ratelimit-Remaining": "58.0",
    "X-Ratelimit-Reset": "240",
}


class TestRateLimitHeaders:
    def test_the_inherited_parser_cannot_read_reddits_headers(self) -> None:
        """Documented as a test because it is the reason `parse_rate_limit` is
        overridden at all: `int("58.0")` raises, the base degrades to None, and the
        provider's own accounting never reaches the shared bucket."""
        inherited = BaseConnector.parse_rate_limit(build(), REDDIT_HEADERS)
        assert inherited is not None
        assert inherited.remaining is None, "float-valued remaining is dropped"

    def test_the_override_reads_the_float_counters(self) -> None:
        hint = build().parse_rate_limit(REDDIT_HEADERS)
        assert hint is not None
        assert hint.remaining == 58
        assert hint.limit == 100, "Reddit sends no limit header; used + remaining is it"

    def test_remaining_is_truncated_rather_than_rounded(self) -> None:
        """58.9 remaining means 58 requests may be spent; rounding up spends one
        the provider has not granted."""
        hint = build().parse_rate_limit({"X-Ratelimit-Remaining": "58.9"})
        assert hint is not None and hint.remaining == 58

    def test_reset_is_a_delta_and_becomes_an_absolute_epoch(self) -> None:
        """Reddit's reset is seconds *until* the window rolls. Passed through as an
        epoch it reads as a moment in 1970 -- permanently in the past, so every
        consumer concludes the window has already reset."""
        hint = build().parse_rate_limit(REDDIT_HEADERS)
        assert hint is not None
        assert hint.reset_at == T_NOW.timestamp() + 240.0

    def test_retry_after_still_reaches_the_hint(self) -> None:
        hint = build().parse_rate_limit({"Retry-After": "120"})
        assert hint is not None and hint.retry_after_seconds == 120.0

    def test_nothing_useful_means_no_hint(self) -> None:
        assert build().parse_rate_limit({"Content-Type": "application/json"}) is None

    @respx.mock
    async def test_the_provider_count_reaches_the_limiter(self) -> None:
        ctx = make_ctx(max_pages=1)
        mock_token()
        mock_listing(listing_response(PAGE1))
        await drain(build(ctx))

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert [hint.remaining for hint in limiter.observed] == [58]

    @respx.mock
    async def test_only_the_freshest_counters_are_fed_back(self) -> None:
        """The descent is buffered and replayed in reverse, so the *oldest*
        response's headers would otherwise be observed last -- talking the bucket
        back up from a ceiling it had correctly clamped down to."""
        ctx = make_ctx()
        mock_token()
        mock_listing(
            listing_response(PAGE1, **{"X-Ratelimit-Remaining": "58.0"}),
            listing_response(PAGE2, **{"X-Ratelimit-Remaining": "57.0"}),
        )
        await drain(build(ctx))

        limiter = ctx.limiter
        assert isinstance(limiter, FakeLimiter)
        assert [hint.remaining for hint in limiter.observed] == [57]

    @respx.mock
    async def test_only_rate_limit_headers_travel_with_a_page(self) -> None:
        """`FetchPage.raw_headers` is handed to code that may log it, and a Reddit
        response also carries Set-Cookie and CDN identifiers."""
        mock_token()
        mock_listing(listing_response(PAGE2))
        connector = build()
        await connector.authenticate()
        pages: list[FetchPage] = [page async for page in connector.fetch(Cursor())]
        await connector.aclose()

        assert set(pages[0].raw_headers) == {
            "x-ratelimit-used",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
        }

    @respx.mock
    async def test_a_429_inside_the_cap_is_transient(self) -> None:
        """Retryable: the runtime backs off and retries the same page rather than
        writing off the run's progress."""
        mock_token()
        respx.get(LISTING_URL).mock(
            return_value=httpx.Response(429, json={}, headers={"Retry-After": "30"})
        )
        with pytest.raises(TransientError) as caught:
            await drain(build())
        assert caught.value.retryable is True

    @respx.mock
    async def test_a_429_with_a_long_wait_becomes_a_quota_error(self) -> None:
        """A partial success, not a failure: holding a worker for a quarter of an
        hour costs more than checkpointing and rescheduling."""
        mock_token()
        respx.get(LISTING_URL).mock(
            return_value=httpx.Response(429, json={}, headers={"Retry-After": "1800"})
        )
        with pytest.raises(QuotaError) as caught:
            await drain(build())
        assert caught.value.retry_after_seconds == 1800.0


# --------------------------------------------------------------------------- #
# Failure taxonomy
# --------------------------------------------------------------------------- #


class TestFailureModes:
    """The class the runtime catches decides what happens next, so each of these
    asserts the class rather than the message."""

    @respx.mock
    async def test_a_server_error_is_transient(self) -> None:
        mock_token()
        respx.get(LISTING_URL).mock(return_value=httpx.Response(503, json={}))
        with pytest.raises(TransientError):
            await drain(build())

    @respx.mock
    async def test_a_404_is_permanent(self) -> None:
        """A 404 on a configured listing means the configuration is wrong, and
        retrying returns the same 404 five more times."""
        mock_token()
        respx.get(LISTING_URL).mock(return_value=httpx.Response(404, json={}))
        with pytest.raises(PermanentError):
            await drain(build())

    @respx.mock
    async def test_a_connection_failure_is_transient(self) -> None:
        mock_token()
        respx.get(LISTING_URL).mock(side_effect=httpx.ConnectError("connection reset"))
        with pytest.raises(TransientError):
            await drain(build())

    @respx.mock
    async def test_a_non_json_body_is_permanent(self) -> None:
        """Usually an error page from a CDN in front of the API. The body is not
        attached to the error: it can echo a request that carries the token."""
        mock_token()
        respx.get(LISTING_URL).mock(
            return_value=httpx.Response(200, text="<html>gateway error</html>")
        )
        with pytest.raises(PermanentError) as caught:
            await drain(build())
        assert "gateway" not in str(caught.value)

    @respx.mock
    async def test_a_listing_that_is_not_a_listing_is_permanent(self) -> None:
        """§6 files an unparsable page structure as a defect for a human rather
        than as something to back off from."""
        mock_token()
        respx.get(LISTING_URL).mock(return_value=httpx.Response(200, json={"data": {}}))
        with pytest.raises(PermanentError, match="children"):
            await drain(build())

    @respx.mock
    async def test_a_child_with_no_fullname_is_permanent(self) -> None:
        """Manufacturing an id would attach a Signal to an identity nothing can
        resolve back to the provider."""
        mock_token()
        payload = load(PAGE1)
        payload["data"]["children"] = [{"kind": "t3", "data": {"title": "no id at all"}}]
        respx.get(LISTING_URL).mock(return_value=httpx.Response(200, json=payload))
        with pytest.raises(PermanentError, match="fullname"):
            await drain(build())

    @respx.mock
    async def test_an_error_never_carries_a_secret(self) -> None:
        """A ConnectorError is logged with its details; a secret in there reaches
        the aggregator."""
        mock_token()
        respx.get(LISTING_URL).mock(return_value=httpx.Response(500, json={}))
        with pytest.raises(TransientError) as caught:
            await drain(build())
        rendered = repr(caught.value) + json.dumps(caught.value.to_log_fields())
        assert "s3cret" not in rendered and "token-1" not in rendered


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #


class TestDedup:
    @respx.mock
    async def test_the_same_page_twice_emits_n_signals_not_2n(self) -> None:
        """The overlap window deliberately re-fetches; dedup is what makes that
        free rather than a source of duplicate rows."""
        ctx = make_ctx()
        mock_token()
        mock_listing(
            listing_response(PAGE2),
            listing_response(PAGE2),
        )
        first = await drain(build(ctx))
        second = await drain(build(ctx))

        assert len(signals(first)) == 2
        assert len(signals(second)) == 0
        assert second[-1].stats["duplicates"] == 2

    async def test_dedup_keys_are_scoped_to_the_connector(self) -> None:
        """A key shared across connectors would let an RSS copy of a Reddit post
        suppress the Reddit original, or the reverse, depending on arrival order."""
        connector = build()
        signal = await connector.normalize(raw(PAGE1, "t3_p1a"))
        assert signal is not None
        keys = connector.dedup_keys(signal)
        assert keys.identity.startswith("os:dedup:id:reddit:")
        assert keys.content is not None and keys.content.startswith("os:dedup:sha:reddit:")


# --------------------------------------------------------------------------- #
# Isolation
# --------------------------------------------------------------------------- #


class TestIsolation:
    def test_the_module_imports_nothing_from_backend_or_services(self) -> None:
        """The rule that keeps a connector testable with respx and two fakes."""
        source = Path("connectors/social/reddit.py").read_text()
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith(("from backend", "import backend"))
            assert not stripped.startswith(("from services", "import services"))

    @respx.mock
    async def test_nothing_reaches_the_real_reddit(self) -> None:
        """respx raises on an unmocked request, so a route that goes missing fails
        the test instead of opening a socket. Asserted explicitly because the whole
        suite's no-network guarantee rests on it."""
        mock_token()
        mock_listing(listing_response(PAGE2))
        await drain(build())
        assert {call.request.url.host for call in respx.calls} == {
            "www.reddit.com",
            "oauth.reddit.com",
        }
