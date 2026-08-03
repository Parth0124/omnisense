"""The Reddit connector: one chronological listing, walked backwards, emitted forwards.

Phase 1 (`docs/connector-spec.md` §9.1). Reddit is the reference *authenticated*
connector the way RSS is the reference unauthenticated one: a real OAuth2 flow, a
real per-client quota reported on every response, and a listing API whose natural
order is the exact opposite of the one `BaseConnector.fetch()` requires.

Four decisions dominate this module.

**Client-credentials, not the password grant.** A script app can authenticate as
its owning user, and nothing here needs to. Public listings are readable app-only,
and a connector that held a user's password would put the highest-value secret in
the system into a code path that only ever reads public data
(`docs/security-and-privacy.md`, data minimisation). The token is app-only, and
the quota it spends belongs to the OAuth client rather than to the run, which is
why the rate-limit budget below is declared with headroom.

**A descriptive User-Agent is a hard requirement, not a nicety.** Reddit throttles
generic and default user agents far harder than identified ones, and the throttle
arrives as a 429 that looks exactly like a legitimate quota wall. Discovering that
in production means reading rate-limit headers to explain a slowdown that is really
a configuration defect, so `_validated_user_agent` refuses to build a connector
that would earn one.

**The API pages newest-first; the contract demands oldest-first.** Reddit's `after`
parameter walks *backwards* in time from the top of `/new`. `BaseConnector.fetch()`
requires pages oldest-first, because the watermark may only move forward and a
newest-first pager that dies mid-run commits a watermark past records it never
emitted. So the descent is buffered in full and then yielded in reverse. That is
affordable here and nowhere else: Reddit truncates every listing at roughly a
thousand items, so the buffer is bounded by the provider, not by our optimism.

**A descent that hits the page budget must not advance the watermark.** If it did,
the records between the deepest page fetched and the previous watermark would sit
below the new watermark having never been emitted -- a silent hole, and the
expensive kind, because nothing ever goes back for them. Instead the run parks its
progress in `checkpoint["pending_watermark"]`, leaves the watermark where it was,
and stores the deepest `after` in `page_token` so the next run continues *downward*
into the gap. The watermark is promoted only when a descent finally reaches the
previous watermark or the end of the listing. See `_page_cursors`.

Identity is rule 1 of `docs/signal-model.md` §4.1 throughout: `native_id` is the
Reddit fullname (`t3_…` for a post, `t1_…` for a comment), verbatim, so a DLQ
record names something a human can paste into Reddit's own UI. Rule 3 is never
reached, so nothing here makes identity depend on the text cleaner.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Self
from urllib.parse import urlencode

import httpx

from models.base import utcnow
from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal
from connectors.auth.oauth import ClientAuthMethod, OAuth2Client, OAuth2Config, OAuth2Grant
from connectors.auth.token_store import InMemoryTokenStore, TokenStore
from connectors.base import BaseConnector
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    ConnectorError,
    NormalizationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.normalize.mapper import FieldMap, FieldSpec, MappingContext, to_utc_datetime
from connectors.protocol import (
    Credentials,
    Cursor,
    FetchPage,
    RateLimitHint,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)

__all__ = ["RedditConnector"]


# --------------------------------------------------------------------------- #
# Endpoints and provider constants
# --------------------------------------------------------------------------- #

TOKEN_URL: Final = "https://www.reddit.com/api/v1/access_token"
"""Token endpoint. On `www`, not `oauth`: only the API itself lives on the
resource host, and posting the client secret to the wrong one answers 404."""

API_BASE: Final = "https://oauth.reddit.com"
WEB_BASE: Final = "https://www.reddit.com"

CHRONOLOGICAL_LISTINGS: Final[frozenset[str]] = frozenset({"new", "comments"})
"""The only two listings whose order is time.

`hot`, `top`, `rising` and `controversial` are *ranked*. Paging them with `after`
walks popularity rather than time, so the oldest record on one page is not older
than the newest on the next -- which makes a watermark meaningless and silently
skips whatever the ranker had not promoted yet. Refused in `_validated_listing`
rather than merely documented, because the failure is invisible: the run succeeds
and the data is wrong.
"""

MAX_PAGE_SIZE: Final = 100
"""Reddit's ceiling for `limit`. Asking for more is silently clamped, which is
worse than being refused -- the pagination arithmetic would rest on a number the
provider never agreed to."""

MAX_LISTING_PAGES: Final = 10
"""Requests one run may spend descending, before `ctx.max_pages` narrows it.

Bounds the buffer the newest-first-to-oldest-first reversal requires. Ten pages of
a hundred is about as deep as Reddit will serve, so this only bites on a first run
against a very busy subreddit -- which is exactly the case `page_token` exists to
continue.
"""

QUOTA_WAIT_THRESHOLD_SECONDS: Final = 900.0
"""Above this wait a 429 becomes a `QuotaError` rather than a `TransientError`.

`docs/connector-spec.md` §5.2: holding a worker for a quarter of an hour to
preserve in-run retry state costs more than checkpointing and rescheduling.
"""

KIND_POST: Final = "t3"
KIND_COMMENT: Final = "t1"

_DELETED_MARKERS: Final[frozenset[str]] = frozenset({"[deleted]", "[removed]"})
"""What Reddit substitutes for the author and body of a removed item.

These are values, not flags the API sets: the record still has an id, a timestamp,
a subreddit and a score. Only the content is gone, which is why they are *dropped*
rather than raised on -- see `normalize`.
"""


# --------------------------------------------------------------------------- #
# Configuration validation (no I/O, all of it at construction time)
# --------------------------------------------------------------------------- #

_DEFAULT_USER_AGENTS: Final[frozenset[str]] = frozenset(
    {"omnisense/0.1", "omnisense/0.1.0", "omnisense", "python"}
)
"""User agents that identify nobody. `SyncContext.user_agent`'s own default heads
the list on purpose: it is the value an operator gets by not deciding."""

_GENERIC_USER_AGENT_PREFIXES: Final[tuple[str, ...]] = (
    "python-requests",
    "python-urllib",
    "urllib",
    "httpx",
    "aiohttp",
    "curl",
    "wget",
    "go-http-client",
    "java",
    "okhttp",
    "scrapy",
    "mozilla/",
    "praw",
)

_CONTACT_MARKER: Final = re.compile(
    r"(?:/u/[A-Za-z0-9_-]{3,}|https?://\S+|[^\s@]+@[^\s@]+\.[A-Za-z]{2,})"
)
"""Something a Reddit admin could use to reach whoever is making the requests.

Reddit documents the shape `<platform>:<app id>:<version> (by /u/<username>)`. The
check is for *a* contact rather than for that exact shape: rejecting a perfectly
identifying agent because it spells its contact as a URL is pedantry, while
accepting one with no contact at all is what actually gets a client throttled.
"""

_MIN_USER_AGENT_LENGTH: Final = 15

_SUBREDDIT: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_]{1,20}$")
_MAX_MULTIREDDIT_PARTS: Final = 10

_SAFE_REASON: Final = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_ACCESS_REASONS: Final[frozenset[str]] = frozenset(
    {"private", "banned", "quarantined", "gold_only", "gated"}
)
"""403 bodies that mean *this subreddit*, not *this token*.

Reddit answers both a revoked token and an unreadable subreddit with 403. Filing
the second as an `AuthError` would flag a working account `needs_reauth` and send
an operator to re-link credentials that were never the problem.
"""


def _validated_user_agent(ctx: SyncContext, credentials: Credentials) -> str:
    """Resolve and check the User-Agent, refusing anything Reddit will throttle.

    Read from the credential record first because the agent names the *app
    registration* the client id belongs to, and the two are rotated together;
    `SyncContext.user_agent` is the process-wide fallback and is almost always the
    default, which is precisely the value that must not be sent.
    """
    candidate = (
        _as_text(credentials.extra.get("user_agent"))
        or _as_text(credentials.secrets.get("user_agent"))
        or _as_text(ctx.user_agent)
    )
    if not candidate:
        raise ConnectorConfigurationError(
            "no user_agent configured for the Reddit connector; Reddit rate-limits "
            "unidentified clients far harder than identified ones, and the throttle "
            "arrives as a 429 indistinguishable from a real quota wall. Set one on "
            "the connector account, shaped '<platform>:<app id>:<version> "
            "(by /u/<username>)'",
            connector=RedditConnector.slug,
            account_id=credentials.account_id,
        )

    lowered = candidate.casefold()
    if lowered in _DEFAULT_USER_AGENTS or lowered.startswith(_GENERIC_USER_AGENT_PREFIXES):
        raise ConnectorConfigurationError(
            f"user_agent {candidate!r} is a default or a library's own; Reddit "
            "throttles those aggressively and they identify nobody. Set "
            "'<platform>:<app id>:<version> (by /u/<username>)'",
            connector=RedditConnector.slug,
            account_id=credentials.account_id,
        )
    if len(candidate) < _MIN_USER_AGENT_LENGTH or not _CONTACT_MARKER.search(candidate):
        raise ConnectorConfigurationError(
            f"user_agent {candidate!r} carries no contact; Reddit's API terms ask for "
            "one so an admin can reach the operator instead of blocking the client. "
            "Include '/u/<username>', a URL or an email address",
            connector=RedditConnector.slug,
            account_id=credentials.account_id,
        )
    return candidate


def _validated_subreddit(params: Mapping[str, Any]) -> str:
    """Resolve `params['subreddit']` into one path segment, possibly a multireddit.

    A sequence is joined with `+` rather than fanned out into several listings.
    Reddit serves `/r/a+b/new` as one merged chronological listing, and one listing
    is one cursor -- fanning out would give a run several independent watermarks
    under a single `params_hash`, and the runtime persists exactly one
    (`docs/connector-spec.md` §4).
    """
    raw = params.get("subreddit") or params.get("subreddits")
    if isinstance(raw, str):
        parts = raw.split("+")
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        parts = [_as_text(part) for part in raw]
    else:
        parts = []

    parts = [part.strip() for part in parts if part and part.strip()]
    if not parts:
        raise ConnectorConfigurationError(
            "the Reddit connector needs params['subreddit']; there is no sensible "
            "default, and /r/all is a different product decision rather than a "
            "fallback",
            connector=RedditConnector.slug,
        )
    if len(parts) > _MAX_MULTIREDDIT_PARTS:
        raise ConnectorConfigurationError(
            f"{len(parts)} subreddits exceeds the {_MAX_MULTIREDDIT_PARTS} a single "
            "multireddit path may carry; split them across connector accounts so each "
            "gets its own cursor",
            connector=RedditConnector.slug,
        )
    invalid = sorted(part for part in parts if not _SUBREDDIT.match(part))
    if invalid:
        raise ConnectorConfigurationError(
            f"not valid subreddit names: {invalid}. The value is interpolated into the "
            "request path, so anything else is either a 404 or a request for a "
            "resource nobody configured",
            connector=RedditConnector.slug,
        )
    return "+".join(parts)


def _validated_listing(params: Mapping[str, Any]) -> str:
    listing = _as_text(params.get("listing")) or "new"
    if listing not in CHRONOLOGICAL_LISTINGS:
        raise ConnectorConfigurationError(
            f"listing {listing!r} is not chronological; only "
            f"{sorted(CHRONOLOGICAL_LISTINGS)} may be paged incrementally. A ranked "
            "listing paged with `after` walks popularity, so the watermark would "
            "advance past records the ranker had not promoted yet and the run would "
            "succeed while losing them",
            connector=RedditConnector.slug,
        )
    return listing


def _validated_page_size(params: Mapping[str, Any]) -> int:
    raw = params.get("limit", MAX_PAGE_SIZE)
    try:
        size = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConnectorConfigurationError(
            f"params['limit'] must be an integer, got {raw!r}",
            connector=RedditConnector.slug,
        ) from exc
    if not 1 <= size <= MAX_PAGE_SIZE:
        raise ConnectorConfigurationError(
            f"params['limit'] must be between 1 and {MAX_PAGE_SIZE}; Reddit clamps "
            "anything larger silently, which would leave the pagination arithmetic "
            "resting on a number the provider never agreed to",
            connector=RedditConnector.slug,
        )
    return size


# --------------------------------------------------------------------------- #
# The field maps: one per Reddit `kind`
# --------------------------------------------------------------------------- #


def _permalink_url(value: Any) -> str:
    """Absolutize Reddit's site-relative permalink.

    `data.permalink` is `/r/x/comments/…`. Left relative it canonicalizes to the
    empty string (`connectors/normalize/html.py`), which would cost the Signal its
    `url` -- and, if `data.name` were ever absent, push identity down the ladder to
    a rule that hashes cleaned text.
    """
    text = _as_text(value)
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    return f"{WEB_BASE}{text}" if text.startswith("/") else f"{WEB_BASE}/{text}"


def _profile_url(value: Any) -> str:
    handle = _as_text(value)
    return f"{WEB_BASE}/user/{handle}" if handle else ""


# `author_fullname` (t2_…), never `author`. Handles are renameable, and keying an
# author's history on one forks that history the first time they rename --
# silently, because nothing downstream can tell an id from a handle
# (`docs/signal-model.md` §3.1). The handle is still carried, as a label.
_AUTHOR_ID: Final = FieldSpec.at("data.author_fullname")
_AUTHOR_HANDLE: Final = FieldSpec.at("data.author")
_AUTHOR_PROFILE_URL: Final = FieldSpec.at("data.author", transform=_profile_url)

_POST_FIELDS: Final = FieldMap(
    platform=Platform.REDDIT,
    # Epoch *seconds*, as a float. `to_utc_datetime` attaches UTC; reading the
    # value as a naive local datetime would shift every trend by the worker's
    # offset, and nothing downstream could tell.
    timestamp=FieldSpec.at("data.created_utc", required=True),
    item_id=FieldSpec.at("data.name", required=True),
    url=FieldSpec.at("data.permalink", transform=_permalink_url),
    title=FieldSpec.at("data.title"),
    # `selftext` is markdown; `selftext_html` is the HTML rendering. Running the
    # readability extractor over markdown would strip nothing and could mangle
    # code blocks, so `text_is_html` stays false.
    text=FieldSpec.at("data.selftext"),
    engagement={
        "score": FieldSpec.at("data.score"),
        "num_comments": FieldSpec.at("data.num_comments"),
        "upvote_ratio": FieldSpec.at("data.upvote_ratio"),
        "subreddit_subscribers": FieldSpec.at("data.subreddit_subscribers"),
        "crossposts": FieldSpec.at("data.num_crossposts"),
    },
    metadata={
        "reddit.subreddit": FieldSpec.at("data.subreddit"),
        "reddit.flair": FieldSpec.at("data.link_flair_text"),
        "reddit.is_self": FieldSpec.at("data.is_self"),
    },
    # Declared rather than left at the default `text/plain`: a body that still
    # holds link syntax and code fences is not plain text, and the stage that
    # eventually decides whether to strip them reads this field to find out.
    content_type="text/markdown",
    author_id=_AUTHOR_ID,
    author_handle=_AUTHOR_HANDLE,
    author_profile_url=_AUTHOR_PROFILE_URL,
)
"""Link and self posts.

Every counter above is a *raw* platform number. The normalized engagement axes are
percentiles within a `(platform, content_type)` cohort (`docs/signal-model.md`
§3.4), and a connector holding one record cannot know a percentile, so they stay
empty for `services/signal_engine/`.
"""

_COMMENT_FIELDS: Final = FieldMap(
    platform=Platform.REDDIT,
    timestamp=FieldSpec.at("data.created_utc", required=True),
    item_id=FieldSpec.at("data.name", required=True),
    url=FieldSpec.at("data.permalink", transform=_permalink_url),
    text=FieldSpec.at("data.body"),
    engagement={
        "score": FieldSpec.at("data.score"),
        "subreddit_subscribers": FieldSpec.at("data.subreddit_subscribers"),
    },
    metadata={
        "reddit.subreddit": FieldSpec.at("data.subreddit"),
        "reddit.flair": FieldSpec.at("data.author_flair_text"),
        # The post this comment hangs off. Without it a comment is an orphan
        # opinion: the graph layer has nothing to attach it to.
        "reddit.link_id": FieldSpec.at("data.link_id"),
    },
    content_type="text/markdown",
    author_id=_AUTHOR_ID,
    author_handle=_AUTHOR_HANDLE,
    author_profile_url=_AUTHOR_PROFILE_URL,
)

_FIELD_MAPS: Final[dict[str, FieldMap]] = {
    KIND_POST: _POST_FIELDS,
    KIND_COMMENT: _COMMENT_FIELDS,
}


@dataclass(frozen=True, slots=True)
class _Descent:
    """One newest-first walk down the listing, buffered before anything is yielded.

    `complete` is the load-bearing field: it separates "we reached the previous
    watermark or the end of the listing", after which the watermark may move, from
    "we ran out of request budget", after which it may not.
    """

    pages: tuple[tuple[RawRecord, ...], ...]
    """Fetched order: `pages[0]` is the newest slice, `pages[-1]` the oldest."""

    rate_limit_headers: Mapping[str, str]
    """From the *last* response only. The provider's counters describe the budget
    now, not the budget when an earlier page was fetched, so replaying the older
    values afterwards would talk the shared bucket back *up* from a ceiling it had
    correctly clamped down to."""

    deepest_after: str | None
    complete: bool


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #


class RedditConnector(BaseConnector):
    """Posts or comments from one subreddit (or multireddit) chronological listing."""

    slug = "reddit"
    platform = Platform.REDDIT
    category = SourceCategory.SOCIAL
    auth_type = AuthType.OAUTH2

    rate_limit = RateLimitPolicy(requests_per_minute=60, burst=10, concurrency=2)
    """Below Reddit's documented 100 QPM averaged over ten minutes.

    The headroom is not timidity: the same OAuth client also spends requests on the
    token endpoint and may be shared by several accounts, and the budget Reddit
    enforces is the client's, not this run's. Overshooting it is answered with a
    429 that costs more than the requests it saved.
    """

    version = "0.1.0"
    supports_incremental = True

    supports_backfill = False
    """Reddit truncates every listing at roughly a thousand items.

    A backfill mode would therefore differ from an incremental run only in which
    cursor row it wrote, while still being unable to reach anything older. Saying
    so is more useful than a mode that quietly stops at the same wall.
    """

    def __init__(
        self,
        ctx: SyncContext,
        credentials: Credentials,
        *,
        token_store: TokenStore | None = None,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(ctx, credentials)
        # Everything below raises `ConnectorConfigurationError` -- a
        # `PermanentError` -- before a socket exists. §6: configuration defects
        # fail fast, and no cursor is ever created for one.
        self._user_agent = _validated_user_agent(ctx, credentials)
        self._subreddit = _validated_subreddit(ctx.params)
        self._listing = _validated_listing(ctx.params)
        self._page_size = _validated_page_size(ctx.params)
        self._oauth_config = self._build_oauth_config(credentials)

        # Not injected through `SyncContext`, because it carries no port for one.
        # A process-local store means the cross-replica single-flight lock that
        # `connectors/auth/oauth.py` is built around degrades to an asyncio lock,
        # so each worker mints its own token. Tolerable while a token costs one
        # request; the fix is a port on `SyncContext`, not a Redis import here.
        self._token_store = token_store or InMemoryTokenStore()
        self._now = now
        self._client: httpx.AsyncClient | None = None
        self._oauth: OAuth2Client | None = None

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct and validate. No I/O: not even the HTTP client is built."""
        return cls(ctx, credentials)

    def _build_oauth_config(self, credentials: Credentials) -> OAuth2Config:
        """Assemble the client-credentials grant, naming a missing secret clearly.

        `Credentials.require` raises `KeyError`, which is the right exception for a
        mapping and the wrong one for the runtime: it carries no error class, so it
        would surface as an unhandled crash rather than as the configuration defect
        it is.
        """
        try:
            return OAuth2Config(
                token_url=TOKEN_URL,
                client_id=credentials.require("client_id"),
                client_secret=credentials.require("client_secret"),
                grant=OAuth2Grant.CLIENT_CREDENTIALS,
                # Reddit implements RFC 6749 §2.3.1 Basic and rejects the body
                # form, which presents as `invalid_client` -- indistinguishable
                # from a wrong secret, and debugged as one for hours.
                client_auth=ClientAuthMethod.BASIC,
                timeout_seconds=self.ctx.request_timeout_seconds,
            )
        except KeyError as exc:
            raise ConnectorConfigurationError(
                f"Reddit credentials are incomplete: {exc.args[0] if exc.args else exc}",
                connector=self.slug,
                account_id=self.ctx.account_id,
            ) from exc

    # ------------------------------------------------------------- lifecycle --

    async def authenticate(self) -> None:
        """Mint or reuse an app-only access token. Idempotent.

        Idempotence is not a courtesy: the runtime calls this once at the start of
        a run and at most once more after a 401, and `OAuth2Client` answers the
        second call from its store unless `invalidate()` marked the token unusable
        -- which `_access_denied` does exactly when a 401 says it should.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.ctx.request_timeout_seconds,
                headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                # A redirect off `oauth.reddit.com` would carry the bearer token to
                # whatever answered, because httpx strips only *its own* auth on a
                # cross-host redirect and this Authorization header is ours.
                follow_redirects=False,
            )
        if self._oauth is None:
            self._oauth = OAuth2Client(
                self._oauth_config,
                account_id=self.ctx.account_id,
                store=self._token_store,
                # Shared, so renewing a token does not open a second connection
                # pool -- and so `aclose()` has exactly one thing to close.
                client=self._client,
                connector=self.slug,
                user_agent=self._user_agent,
                now=self._now,
            )
        await self._oauth.token()

    async def aclose(self) -> None:
        """Release the client. Called from `run()`'s `finally`, always.

        Both handles are dropped rather than merely closed, so a second `run()` on
        the same instance rebuilds them instead of reusing a closed pool. The token
        survives regardless -- it lives in the store, not in the client -- so the
        rebuild costs no extra mint.
        """
        client, self._client, self._oauth = self._client, None, None
        if client is not None:
            await client.aclose()

    # ----------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Walk the listing backwards, then hand it back forwards.

        The whole descent is buffered before the first yield. That is the opposite
        of what `BaseConnector` prefers -- it yields per page so a crash costs one
        page -- and the provider forces it: the only way to know which Reddit page
        is *oldest* is to have fetched them all, and yielding newest-first would
        commit a watermark past records still to come.
        """
        descent = await self._descend(cursor)
        for page in self._page_cursors(cursor, descent):
            yield page

    async def _descend(self, cursor: Cursor) -> _Descent:
        """Page backwards from the top -- or from a parked `after` -- toward the watermark."""
        watermark = cursor.watermark
        after = _as_text(cursor.page_token) or None
        pages: list[tuple[RawRecord, ...]] = []
        headers: Mapping[str, str] = {}
        complete = False

        for _ in range(self._page_budget()):
            url = self._listing_url()
            params = self._listing_params(after)
            await self.acquire_slot(url)
            payload, headers = await self._request(url, params)

            listing = payload.get("data")
            if not isinstance(listing, Mapping):
                raise self._shape_error("response carries no 'data' object", url)
            children = listing.get("children")
            if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
                raise self._shape_error("'data.children' is not a list", url)

            fingerprint = _request_fingerprint(url, params)
            fetched_at = self._now()
            # Reversed *within* the page as well as between pages. Reddit returns
            # a page newest-first, and a consumer reading the emitted stream in
            # order should see time move forward inside a batch for the same
            # reason it must between them -- half-ordered output is the kind of
            # thing downstream code accidentally relies on being total.
            records = tuple(
                self._to_record(child, fetched_at=fetched_at, fingerprint=fingerprint, url=url)
                for child in reversed(children)
            )
            if records:
                pages.append(records)

            after = _as_text(listing.get("after")) or None
            if not records or after is None:
                # The listing ended. Everything between here and the watermark has
                # been seen, so the descent is complete by exhaustion.
                complete = True
                break
            oldest = _oldest_timestamp(records)
            if watermark is not None and oldest is not None and oldest <= watermark:
                # Crossed into ground a previous run already covered. The overlap
                # is deliberate (`BaseConnector.overlap_seconds`); dedup absorbs it.
                complete = True
                break

        return _Descent(
            pages=tuple(pages),
            rate_limit_headers=headers,
            deepest_after=after,
            complete=complete,
        )

    def _page_cursors(self, cursor: Cursor, descent: _Descent) -> list[FetchPage]:
        """Reverse the descent and attach a legal restart point to every page.

        Three shapes, and the difference between them is the difference between a
        replay and a hole:

        - **Complete descent, nothing parked.** The pages are contiguous from the
          previous watermark upwards, so each may carry its own newest timestamp.
          A crash part-way through leaves the watermark below every record not yet
          emitted, and the next run re-fetches exactly those.
        - **Truncated descent.** The watermark stays put and progress is parked in
          `checkpoint`, because the records between the deepest page fetched and
          the old watermark have not been seen at all. `page_token` carries the
          deepest `after` so the next run continues downward into that gap.
        - **Completing a parked descent.** The parked value is newer than
          everything in this chunk, so promoting it anywhere but on the last page
          would jump the watermark over records this run has not yielded yet.
        """
        pages: list[FetchPage] = []
        running = _parse_pending(cursor)
        parked = running is not None
        ordered = list(reversed(descent.pages))

        for index, records in enumerate(ordered):
            running = _max_datetime(running, _newest_timestamp(records))
            final = index == len(ordered) - 1

            if descent.complete and (final or not parked):
                page_cursor = Cursor(watermark=running, page_token=None, checkpoint={})
            else:
                page_cursor = Cursor(
                    watermark=cursor.watermark,
                    page_token=(
                        descent.deepest_after if not descent.complete else cursor.page_token
                    ),
                    checkpoint=_pending_checkpoint(running),
                )

            pages.append(
                FetchPage(
                    records=records,
                    cursor=page_cursor,
                    # Only the first yielded page carries headers, and they are the
                    # newest response's: see `_Descent.rate_limit_headers`.
                    raw_headers=descent.rate_limit_headers if index == 0 else {},
                )
            )
        return pages

    def _page_budget(self) -> int:
        budget = MAX_LISTING_PAGES
        if self.ctx.max_pages is not None:
            budget = min(budget, max(1, self.ctx.max_pages))
        return budget

    def _listing_url(self) -> str:
        return f"{API_BASE}/r/{self._subreddit}/{self._listing}"

    def _listing_params(self, after: str | None) -> dict[str, str]:
        params = {
            "limit": str(self._page_size),
            # Without this Reddit HTML-escapes `<`, `>` and `&` inside every body,
            # so `&amp;` reaches the cleaner, the embedder, and eventually a
            # sentence quoted in a report.
            "raw_json": "1",
        }
        if after:
            params["after"] = after
        return params

    def _to_record(
        self, child: Any, *, fetched_at: datetime, fingerprint: str, url: str
    ) -> RawRecord:
        """Wrap one listing child verbatim.

        `raw_bytes` is deliberately `None`. `RawRecord` documents it as the exact
        bytes the provider returned, and for an item carved out of a listing those
        bytes do not exist -- one response carried a hundred items. Re-serializing
        the payload to fill the field would produce a digest that changes with the
        json library and break the content-addressed R2 key it feeds.
        """
        if not isinstance(child, Mapping):
            raise self._shape_error("a listing child is not an object", url)
        fullname = _fullname(child)
        if fullname is None:
            # Reddit has never omitted `name`. If it is missing, this is not the
            # endpoint we think it is, and manufacturing an id would attach a
            # Signal to an identity nothing can resolve back to the provider.
            raise self._shape_error("a listing child carries no fullname", url)
        data = child.get("data")
        permalink = data.get("permalink") if isinstance(data, Mapping) else None
        return RawRecord(
            native_id=fullname,
            payload=child,
            fetched_at=fetched_at,
            raw_bytes=None,
            content_type="application/json",
            source_url=_permalink_url(permalink) or None,
            request_fingerprint=fingerprint,
        )

    # ------------------------------------------------------------- normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one listing child onto a Signal, or drop it.

        Dropping is reserved for items Reddit has emptied: a `[deleted]` author or
        a `[removed]` body. Those are not defects -- they are among the most common
        things a subreddit listing contains -- so they are counted as drops rather
        than filed in the DLQ, where they would bury real mapping bugs.
        """
        child = record.payload
        data = child.get("data")
        if not isinstance(data, Mapping):
            raise NormalizationError(
                "listing child carries no 'data' object",
                native_id=record.native_id,
                connector=self.slug,
            )

        kind = _as_text(child.get("kind"))
        field_map = _FIELD_MAPS.get(kind)
        if field_map is None:
            raise NormalizationError(
                f"unsupported Reddit kind {kind!r}; this connector maps posts "
                f"({KIND_POST}) and comments ({KIND_COMMENT}) only",
                native_id=record.native_id,
                connector=self.slug,
            )

        if _is_emptied(data, kind):
            return None

        # The runtime keys the R2 object and the Kafka partition off
        # `RawRecord.native_id`, while every store keys off `Signal.id`, which is
        # derived from `data.name`. If those two disagreed the same item would
        # exist under two identities, so the disagreement is caught here instead of
        # being discovered as duplicate rows months later.
        if _as_text(data.get("name")) != record.native_id:
            raise NormalizationError(
                "payload fullname does not match the fetched record's native_id",
                native_id=record.native_id,
                connector=self.slug,
            )

        return field_map.to_signal(record, self._mapping_context())

    def _mapping_context(self) -> MappingContext:
        return MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=self.ctx.run_id,
        )

    # ------------------------------------------------------------ rate limit --

    def parse_rate_limit(self, headers: Mapping[str, str]) -> RateLimitHint | None:
        """Read Reddit's rate-limit headers, which the inherited parser cannot.

        Two incompatibilities, both silent:

        - `X-Ratelimit-Remaining` arrives as a **float** string (`"58.0"`).
          `BaseConnector.parse_rate_limit` runs `int()` over it, which raises and
          degrades to `None` -- so the provider's own accounting never reaches the
          bucket and the limiter runs on a local guess forever.
        - `X-Ratelimit-Reset` is **seconds until** the window rolls, not an epoch.
          Passed through unchanged it reads as a moment in 1970, permanently in the
          past, so every consumer of `reset_at` concludes the window has reset.

        Reddit sends no `X-Ratelimit-Limit`; the ceiling is `used + remaining`.
        """
        lowered = {key.lower(): value for key, value in headers.items()}
        remaining = _as_float(lowered.get("x-ratelimit-remaining"))
        used = _as_float(lowered.get("x-ratelimit-used"))
        reset_in = _as_float(lowered.get("x-ratelimit-reset"))

        inherited = super().parse_rate_limit(headers)
        retry_after = inherited.retry_after_seconds if inherited is not None else None

        if remaining is None and used is None and reset_in is None and retry_after is None:
            return None
        return RateLimitHint(
            # Truncated, not rounded: 58.9 remaining means 58 requests may be
            # spent, and rounding up spends one the provider has not granted.
            remaining=int(remaining) if remaining is not None else None,
            limit=int(used + remaining) if used is not None and remaining is not None else None,
            reset_at=self._now().timestamp() + reset_in if reset_in is not None else None,
            retry_after_seconds=retry_after,
        )

    # --------------------------------------------------------------- request --

    async def _request(
        self, url: str, params: Mapping[str, str]
    ) -> tuple[Mapping[str, Any], Mapping[str, str]]:
        """Issue one authenticated GET and map every failure onto the taxonomy.

        No retry and no sleep. `docs/connector-spec.md` §1: a connector that
        retries privately makes the shared limiter's accounting wrong and hides the
        failure from metrics.
        """
        if self._client is None or self._oauth is None:
            raise PermanentError(
                "fetch() ran before authenticate(); the six-stage order in "
                "BaseConnector.run() is what guarantees it does not",
                connector=self.slug,
                account_id=self.ctx.account_id,
            )

        auth_headers = await self._oauth.headers()
        try:
            response = await self._client.get(url, params=params, headers=auth_headers)
        except httpx.TransportError as exc:
            raise TransientError(
                "Reddit is unreachable",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                "the listing request could not be issued",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc

        status = response.status_code
        if status == httpx.codes.TOO_MANY_REQUESTS:
            raise self._rate_limited(response)
        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise await self._access_denied(response)
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise TransientError(
                "Reddit returned a server error",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status >= httpx.codes.BAD_REQUEST:
            raise PermanentError(
                "Reddit rejected the listing request",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details={"listing": self._listing},
            )

        try:
            payload = response.json()
        except ValueError as exc:
            # Usually an error page from a CDN in front of the API. The body is not
            # attached: it can echo the request, and the request carries the bearer
            # token in a header (§1 forbids logging either).
            raise PermanentError(
                "Reddit returned a body that is not JSON",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                cause=exc,
            ) from exc
        if not isinstance(payload, Mapping):
            raise self._shape_error("response is not a JSON object", url)

        # Only the rate-limit headers travel onward. `FetchPage.raw_headers` is
        # read by code that may log what it is handed, and a Reddit response also
        # carries Set-Cookie and CDN identifiers.
        return payload, _rate_limit_headers(response.headers)

    def _rate_limited(self, response: httpx.Response) -> ConnectorError:
        """Classify a 429 by how long the provider wants us gone."""
        hint = self.parse_rate_limit(response.headers)
        wait = hint.retry_after_seconds if hint is not None else None
        if wait is None and hint is not None and hint.reset_at is not None:
            wait = max(0.0, hint.reset_at - self._now().timestamp())

        if wait is not None and wait > QUOTA_WAIT_THRESHOLD_SECONDS:
            return QuotaError(
                "Reddit quota is exhausted",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                reset_at=hint.reset_at if hint is not None else None,
                retry_after_seconds=wait,
            )
        # Inside the cap this is a wall to wait behind, not a spent quota: the
        # runtime backs off and retries the same page (§6).
        return TransientError(
            "Reddit rate-limited the request",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=response.status_code,
            details={"retry_after_seconds": wait},
        )

    async def _access_denied(self, response: httpx.Response) -> ConnectorError:
        """Tell a rejected token apart from a subreddit this account may not read."""
        reason = _access_reason(response)
        if reason is not None and reason in _ACCESS_REASONS:
            return ConnectorConfigurationError(
                f"/r/{self._subreddit} is not readable by this account ({reason})",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                details={"reason": reason},
            )
        if self._oauth is not None:
            # Expire the cached token in place so the runtime's single permitted
            # re-authentication actually mints a new one rather than replaying the
            # rejected one. `invalidate()` keeps the refresh material; `delete()`
            # would send a human to a consent screen over a token that merely
            # expired early.
            await self._oauth.invalidate()
        return AuthError(
            "Reddit rejected the access token",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=response.status_code,
        )

    def _shape_error(self, detail: str, url: str) -> PermanentError:
        """A listing that is not shaped like a listing.

        `PermanentError` rather than `TransientError`: a retry returns the same
        bytes, and §6 files an unparsable page structure as a defect for a human
        to look at rather than as something to back off from.
        """
        return PermanentError(
            f"unexpected Reddit listing shape: {detail}",
            connector=self.slug,
            account_id=self.ctx.account_id,
            details={"endpoint": url},
        )


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #


def _fullname(child: Mapping[str, Any]) -> str | None:
    """The Reddit fullname of one listing child: `t3_1abcde`, `t1_ktz9y0`.

    Rule 1 of `docs/signal-model.md` §4.1, verbatim rather than hashed, so a DLQ
    record names something a human can paste into Reddit's own UI.

    `data.name` is the fullname the API sends. Composing `kind` and `data.id` is a
    fallback that produces the same string -- which matters, because a fullname
    built one way in `fetch` and another in `normalize` would fork identity for
    every record.
    """
    data = child.get("data")
    if not isinstance(data, Mapping):
        return None
    name = _as_text(data.get("name"))
    if name:
        return name
    kind = _as_text(child.get("kind"))
    item_id = _as_text(data.get("id"))
    return f"{kind}_{item_id}" if kind and item_id else None


def _is_emptied(data: Mapping[str, Any], kind: str) -> bool:
    """Whether Reddit has removed the content this record used to carry.

    Three signals, because they do not coincide: a user deletion blanks the author,
    a moderator removal blanks the body, and the removal of a *link* post blanks
    neither -- there was no body to blank -- and shows only as
    `removed_by_category`.
    """
    if _as_text(data.get("author")) in _DELETED_MARKERS:
        return True
    body = _as_text(data.get("selftext") if kind == KIND_POST else data.get("body"))
    if body in _DELETED_MARKERS:
        return True
    return bool(_as_text(data.get("removed_by_category")))


def _created_at(record: RawRecord) -> datetime | None:
    """Event time of one record, or `None` when the payload carries no usable one.

    Used only for pagination arithmetic. A record whose `created_utc` is missing or
    unparseable is still emitted -- `normalize` raises for it and it reaches the
    DLQ -- but it must not be allowed to poison the watermark, so it is excluded
    here rather than defaulted to now.
    """
    data = record.payload.get("data")
    if not isinstance(data, Mapping):
        return None
    raw = data.get("created_utc")
    if raw is None:
        return None
    try:
        return to_utc_datetime(raw)
    except ValueError:
        return None


def _newest_timestamp(records: Sequence[RawRecord]) -> datetime | None:
    moments = [moment for moment in map(_created_at, records) if moment is not None]
    return max(moments) if moments else None


def _oldest_timestamp(records: Sequence[RawRecord]) -> datetime | None:
    moments = [moment for moment in map(_created_at, records) if moment is not None]
    return min(moments) if moments else None


def _max_datetime(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


PENDING_WATERMARK_KEY: Final = "pending_watermark"
"""Where an unfinished descent parks progress it may not commit yet.

In `Cursor.checkpoint` rather than in `watermark` because the runtime *interprets*
the watermark -- it schedules and detects gaps from it -- while `checkpoint`
round-trips as opaque JSON (`docs/connector-spec.md` §4).
"""


def _parse_pending(cursor: Cursor) -> datetime | None:
    raw = cursor.checkpoint.get(PENDING_WATERMARK_KEY)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    except ValueError:
        # A checkpoint we cannot read is not worth failing a run over: the
        # watermark is still valid, so the descent simply re-covers ground.
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _pending_checkpoint(moment: datetime | None) -> dict[str, Any]:
    if moment is None:
        return {}
    return {PENDING_WATERMARK_KEY: moment.astimezone(UTC).isoformat()}


def _request_fingerprint(url: str, params: Mapping[str, str]) -> str:
    """Hash of endpoint plus normalized params -- never credentials.

    `lineage.request_fingerprint` is what makes a fetch reproducible: it names the
    exact request that produced a record without naming who made it.
    """
    canonical = urlencode(sorted(params.items()))
    return hashlib.sha256(f"GET {url}?{canonical}".encode()).hexdigest()


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    wanted = ("x-ratelimit-remaining", "x-ratelimit-used", "x-ratelimit-reset", "retry-after")
    return {key: value for key, value in headers.items() if key.lower() in wanted}


def _access_reason(response: httpx.Response) -> str | None:
    """The `reason` field of a Reddit 403 body, and nothing else.

    Character-checked and length-capped before it is allowed into an error message,
    for the same reason `connectors/auth/oauth.py` does it to `error`: a provider
    under load can put anything in a body field, including an echo of the request
    that caused it.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    reason = payload.get("reason")
    if not isinstance(reason, str):
        return None
    reason = reason.strip()
    return reason if _SAFE_REASON.match(reason) else None


def _as_text(value: Any) -> str:
    """Render a scalar as a stripped string; `""` for anything unusable.

    Booleans are refused rather than stringified: `"True"` is never a subreddit, a
    fullname or a user agent, and letting one through turns a type confusion into a
    plausible-looking value.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (Mapping, Sequence, set, frozenset)):
        return ""
    return str(value).strip()


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
