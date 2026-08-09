"""LinkedIn organisation posts, through the Community Management API.

Read the scope limit first, because it bounds everything this module can ever
return: **LinkedIn sells no API that reads third-party organic content.** There is
no public search, no "posts mentioning X", no competitor feed. The Posts API
finder used here answers for exactly one `author` URN, and only when the member
whose token this is holds an `ADMINISTRATOR`, `CONTENT_ADMIN` or
`DIRECT_SPONSORED_CONTENT_POSTER` role on that organisation
(`r_organization_social`). Point it at an organisation nobody authorised and it
answers `403`, forever. Scraping the public feed instead is prohibited and has
been litigated, so `requires_tos_review` stays `True` and
`connectors/registry.py` refuses to enable or instantiate this class until that
flag is cleared in a reviewed change (`docs/connector-spec.md` §9.1).

What is left is genuinely useful and genuinely narrow: an owned or partner-granted
company page, read as a first-party source. Five provider facts shape the module.

**Three-legged OAuth, and no app-only alternative.** LinkedIn issues a
client-credentials token, and it cannot read posts -- content permissions are
member-delegated. So the credential this connector needs is a *refresh token*
produced by a consent flow the API layer ran, and `authenticate()` runs the
refresh grant only. `connectors/auth/oauth.py` deliberately has no
authorization-code grant, for the same reason: a worker cannot open a browser.

**A short page does not mean the last page.** The Posts API documents it
explicitly -- "receive less than the `count` number of results in a page, when
there are more posts available" -- because the finder filters after it slices.
The near-universal `len(elements) < count -> done` heuristic therefore truncates
an organisation's history at the first page containing a deleted or invisible
post, silently, on the very first run. `paging.links[rel=next]` is the only
authority, and `_has_next_page` treats a missing `paging` object as a shape error
rather than as an end, so a response format change fails loudly instead of
committing a watermark over posts nobody fetched.

**`sortBy=CREATED`, not the default.** The default is `LAST_MODIFIED`, which
reorders a post to the head of the listing when someone edits a typo in it. The
watermark is taken from `createdAt`; pairing it with a listing ordered by
modification time makes the "have I reached ground a previous run covered?" test
meaningless, and the descent would stop early against an edit rather than against
the watermark. Sort field and watermark field must be the same field.

**Newest-first, so the watermark is pinned.** Both sort orders are descending and
LinkedIn offers no ascending one. `BaseConnector.fetch` sanctions this exactly
once: yield in provider order provided the watermark is pinned for the whole
descent and only advances on the closing page. That is what `fetch` does --
progress is parked in `checkpoint["pending_watermark"]` and `page_token` carries
the next offset, so a run that dies at page 4 resumes *downward* into the gap
instead of committing a watermark over posts it never emitted. Records are still
reversed *within* a page, because a consumer reading the emitted stream should
see time move forward inside a batch.

**Rate limits are daily, per app and per member, and their values are not
published.** LinkedIn states this outright: limits reset at midnight UTC and the
number for an endpoint is visible only in the Developer Portal's Analytics tab
for the app that made the call. No `X-RateLimit-*` header comes back. So a 429
with no `Retry-After` is treated as a `QuotaError` resetting at the next UTC
midnight -- backing off for thirty seconds into a daily wall would spend the rest
of the day's schedule discovering the same 429.

Identity is rule 1 of `docs/signal-model.md` §4.1: `native_id` is the post URN
(`urn:li:share:…` or `urn:li:ugcPost:…`) verbatim, so a DLQ record names something
a human can paste into LinkedIn's own tooling. Rule 3 is never reached.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Self
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
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)

__all__ = ["LinkedInConnector"]


# --------------------------------------------------------------------------- #
# Endpoints and provider constants
# --------------------------------------------------------------------------- #

TOKEN_URL: Final = "https://www.linkedin.com/oauth/v2/accessToken"
"""Token endpoint. On `www.linkedin.com`, not `api.linkedin.com`: only the
resource server lives on the API host, and posting the grant to the wrong one
answers a 404 that reads as a deleted endpoint rather than a wrong host."""

API_BASE: Final = "https://api.linkedin.com/rest"
"""The *versioned* API. `https://api.linkedin.com/v2` is the legacy unversioned
surface, where `/ugcPosts` lives; the two are not interchangeable and the
versioned one is the only one the Posts API is documented under."""

WEB_BASE: Final = "https://www.linkedin.com"
POSTS_PATH: Final = "/posts"

DEFAULT_API_VERSION: Final = "202506"
"""Value of the mandatory `LinkedIn-Version: YYYYMM` header.

A *floor*, not a promise. LinkedIn sunsets a version roughly a year after it
ships and answers a sunset version with `426 Upgrade Required`, so a constant
compiled into a connector rots on a schedule LinkedIn sets. The deployment
overrides it through `params['api_version']`, and `_raise_for_status` maps the
426 onto a configuration error naming that param -- which is the one message that
turns a mystery outage into a one-line config change.
"""

MAX_COUNT: Final = 100
"""LinkedIn's documented ceiling for `count` on the author finder (default 10)."""

DEFAULT_COUNT: Final = 50
"""Half the ceiling. Offset pagination re-serves the boundary item whenever a new
post lands mid-descent, and a smaller page makes that overlap cheaper without
costing an extra request on the organisations this connector is pointed at --
company pages post in the tens per month, not the thousands."""

MAX_OFFSET_PAGES: Final = 20
"""Requests one run may spend descending, before `ctx.max_pages` narrows it.

Offset pagination over a feed that grows at the head is lossy in one direction:
a post published mid-descent shifts every later offset by one, so the boundary
item is re-served (harmless -- dedup collapses it) and a post *deleted* mid-descent
shifts the other way and is skipped entirely. Keeping a descent short bounds how
much of a moving feed one run walks; `page_token` continues it next run.
"""

QUOTA_WAIT_THRESHOLD_SECONDS: Final = 900.0
"""Above this wait a 429 becomes a `QuotaError` rather than a `TransientError`
(`docs/connector-spec.md` §5.2)."""

PENDING_WATERMARK_KEY: Final = "pending_watermark"
"""Where an unfinished descent parks progress it may not commit yet.

In `Cursor.checkpoint` rather than in `watermark` because the runtime *interprets*
the watermark -- it schedules and detects gaps from it -- while `checkpoint`
round-trips as opaque JSON (`docs/connector-spec.md` §4).
"""

MAX_MENTIONS: Final = 20
"""How many mention URNs one Signal carries into `metadata`.

`metadata` is written to a Postgres `jsonb` column, a Qdrant payload and an
OpenSearch document simultaneously; an unbounded list of URNs from a post that
tagged four hundred people would be paid for three times over.
"""

_RATE_LIMIT_HEADERS: Final = frozenset(
    {"retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}
)
"""The only response headers that leave `fetch()`.

LinkedIn publishes none of them -- its limits are visible solely in the Developer
Portal -- which is precisely why this is an allowlist rather than a blocklist. A
LinkedIn response also carries `Set-Cookie` and `x-li-*` tracing identifiers, and
`FetchPage.raw_headers` is read by code that may log what it is handed
(`docs/connector-spec.md` §1).
"""

_SERVICE_ERROR_CODE: Final = re.compile(r"^[A-Z0-9_]{1,48}$")
"""Shape of the `code` field in a LinkedIn error body.

Character-checked and length-capped before it is allowed into an error message,
for the same reason `connectors/auth/oauth.py` does it to `error`: a provider
under load can put anything in a body field, including an echo of the request --
and this request carries a bearer token in a header.
"""

_AUTHOR_URN: Final = re.compile(r"^urn:li:(organization|organizationBrand|person):[A-Za-z0-9_-]{1,64}$")
"""Author URNs this connector will interpolate into a query string.

Restricted by pattern rather than merely documented: the value is URL-encoded into
the finder's `author` parameter, and anything else is either a 400 or a request for
a resource nobody configured.
"""

_API_VERSION: Final = re.compile(r"^(20[2-9][0-9])(0[1-9]|1[0-2])$")

_PUBLISHED: Final = "PUBLISHED"
"""The only `lifecycleState` that is a public observation.

`DRAFT`, `PUBLISH_REQUESTED`, `PROCESSING` and `PUBLISH_FAILED` describe content
no reader has seen. They reach this connector only under `viewContext=AUTHOR`,
which is why that parameter is pinned to `READER` below -- but the state is still
checked, because the default `viewContext` is a LinkedIn decision and it has
changed before.
"""


# --------------------------------------------------------------------------- #
# Configuration validation (no I/O, all of it at construction time)
# --------------------------------------------------------------------------- #


def _validated_author_urn(params: Mapping[str, Any]) -> str:
    """Resolve the one author this account syncs, as a full URN.

    A bare `organization_id` is accepted and promoted, because that is the number
    an operator reads off the company page's admin URL, and making them type
    `urn:li:organization:` in a config file is how the colon-count goes wrong.
    """
    urn = _as_text(params.get("author_urn") or params.get("organization_urn"))
    if not urn:
        organization_id = _as_text(params.get("organization_id"))
        if organization_id:
            urn = f"urn:li:organization:{organization_id}"
    if not urn:
        raise ConnectorConfigurationError(
            "the LinkedIn connector needs params['author_urn'] (or "
            "params['organization_id']). The Posts API has no search and no public "
            "feed: it answers for exactly one author, so there is no default and an "
            "unbounded query is not a thing this API offers",
            connector=LinkedInConnector.slug,
        )
    if not _AUTHOR_URN.match(urn):
        raise ConnectorConfigurationError(
            f"params['author_urn'] {urn!r} is not an organization or person URN; "
            "it is URL-encoded straight into the finder's `author` parameter, so "
            "anything else is a 400 at best. Expected "
            "'urn:li:organization:{id}'",
            connector=LinkedInConnector.slug,
        )
    return urn


def _validated_api_version(params: Mapping[str, Any]) -> str:
    """Read `LinkedIn-Version`, refusing anything that is not `YYYYMM`.

    Checked here rather than discovered as a 400, because the header is sent on
    *every* request: a malformed version turns the whole connector off, and the
    error LinkedIn returns for it names the header rather than the config key an
    operator would have to change.
    """
    version = _as_text(params.get("api_version")) or DEFAULT_API_VERSION
    if not _API_VERSION.match(version):
        raise ConnectorConfigurationError(
            f"params['api_version'] {version!r} must be a LinkedIn API version in "
            "YYYYMM form, e.g. '202506'. It is sent as the mandatory "
            "LinkedIn-Version header on every request",
            connector=LinkedInConnector.slug,
        )
    return version


def _validated_count(params: Mapping[str, Any]) -> int:
    raw = params.get("count", DEFAULT_COUNT)
    try:
        count = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConnectorConfigurationError(
            f"params['count'] must be an integer, got {raw!r}",
            connector=LinkedInConnector.slug,
        ) from exc
    if not 1 <= count <= MAX_COUNT:
        raise ConnectorConfigurationError(
            f"params['count'] must be between 1 and {MAX_COUNT}; the Posts API "
            "author finder caps it there, and the offset arithmetic in fetch() "
            "steps by exactly this value",
            connector=LinkedInConnector.slug,
        )
    return count


# --------------------------------------------------------------------------- #
# "little" text format
# --------------------------------------------------------------------------- #

_MENTION_TEMPLATE: Final = re.compile(r"@\[([^\]\n]{1,200})\]\((urn:li:[A-Za-z]+:[^)\s]{1,120})\)")
"""`@[Devtestco](urn:li:organization:2414183)` -- an annotated mention."""

_HASHTAG_TEMPLATE: Final = re.compile(r"\{hashtag\|\\?#\|([^}|\s]{1,140})\}")
r"""`{hashtag|\#|coding}` -- how a hashtag comes back out of the Posts API."""

_LITTLE_ESCAPE: Final = re.compile(r"\\([^\w\s])")
r"""A backslash escaping a punctuation character in LinkedIn's "little" format.

Deliberately keyed on "non-word, non-space" rather than on LinkedIn's exact list
of reserved characters. The list has grown before, and the failure modes are
asymmetric: unescaping one character LinkedIn did not reserve costs a stray
backslash in one body, while missing one LinkedIn *did* reserve puts `\(` into
every post that used a parenthesis -- and therefore into the embedding, the BM25
index and any sentence quoted in a report. `\n` and `\word` are left alone.
"""


def _from_little_text(value: Any) -> str:
    r"""Render `commentary` as the prose a reader saw.

    Three passes, in this order, because the templates contain their own escapes:
    the hashtag template embeds `\#`, so unescaping first would destroy it and
    leave `{hashtag|#|coding}` in the body.

    Not cosmetic. `content.text` is what gets embedded, indexed and quoted, and it
    is what the layer-2 dedup hash is taken over -- so a body still carrying
    `{hashtag|\#|ai}` hashes differently from the same sentence syndicated
    anywhere else, and cross-platform duplicate detection quietly stops working
    for every post that used a tag.
    """
    text = _as_text(value)
    if not text:
        return ""
    text = _MENTION_TEMPLATE.sub(r"\1", text)
    text = _HASHTAG_TEMPLATE.sub(r"#\1", text)
    return _LITTLE_ESCAPE.sub(r"\1", text)


def _mentioned_urns(commentary: Any) -> list[str]:
    """Entity URNs a post annotated, in first-seen order.

    Kept because they are the one *resolved* entity reference in the payload:
    LinkedIn has already decided that this text refers to
    `urn:li:organization:2414183`, which is a stronger claim than anything
    `services/signal_engine/entities.py` can make from the prose alone. They ride
    in `metadata` rather than in `entities`, because filling `entities` is the
    enrichment pipeline's job (`docs/connector-spec.md` §2.4).
    """
    seen: dict[str, None] = {}
    for match in _MENTION_TEMPLATE.finditer(_as_text(commentary)):
        seen.setdefault(match.group(2), None)
    return list(seen)[:MAX_MENTIONS]


def _post_url(value: Any) -> str:
    """Permalink for a post URN, in the form LinkedIn documents.

    `https://www.linkedin.com/feed/update/urn:li:share:{id}/` -- the URN sits in
    the path, colons and all. Built here rather than read from the payload
    because the Posts API returns no permalink field at all, and a Signal with no
    `url` cannot be cited in a report.
    """
    urn = _as_text(value)
    return f"{WEB_BASE}/feed/update/{urn}/" if urn else ""


def _author_url(value: Any) -> str:
    """Public page of the authoring entity, when one can be derived.

    Only for organisations: `/company/{id}` resolves. A person URN's public
    profile lives at a vanity slug that the URN does not contain and that this
    connector has no permission to look up, so it returns `""` rather than a
    plausible URL that 404s -- an unresolvable citation is worse than a missing
    one (`models/signal.py::MediaRef` makes the same trade).
    """
    urn = _as_text(value)
    if urn.startswith("urn:li:organization:") or urn.startswith("urn:li:organizationBrand:"):
        return f"{WEB_BASE}/company/{urn.rsplit(':', 1)[-1]}/"
    return ""


# --------------------------------------------------------------------------- #
# The field map
# --------------------------------------------------------------------------- #

_FIELD_MAP: Final = FieldMap(
    platform=Platform.LINKEDIN,
    # Epoch *milliseconds*. `to_utc_datetime` tells milliseconds from seconds by
    # magnitude, so no per-connector arithmetic decides it. `createdAt` first and
    # `publishedAt` only as a fallback: they differ for a scheduled post, and the
    # finder is sorted by `CREATED`, so watermarking on the other one would order
    # the cursor differently from the listing it is walking.
    timestamp=FieldSpec.at("createdAt", "publishedAt", required=True),
    # Rule 1, verbatim: `urn:li:share:6844785523593134080`.
    item_id=FieldSpec.at("id", required=True),
    url=FieldSpec.at("id", transform=_post_url),
    # The article's own headline when the post shares a link. A LinkedIn post has
    # no title of its own, so this is the only titled thing in the payload.
    title=FieldSpec.at("content.article.title"),
    text=FieldSpec.at("commentary", transform=_from_little_text),
    metadata={
        "linkedin.lifecycle_state": FieldSpec.at("lifecycleState"),
        "linkedin.visibility": FieldSpec.at("visibility"),
        # `NONE` marks a dark post -- created for an ad campaign and never shown
        # on the page. It is still a real utterance by the organisation, so it is
        # emitted, but a trend built from page activity has to be able to exclude
        # it or paid distribution reads as organic momentum.
        "linkedin.feed_distribution": FieldSpec.at("distribution.feedDistribution"),
        "linkedin.article_url": FieldSpec.at("content.article.source"),
        "linkedin.reshare_of": FieldSpec.at("reshareContext.parent"),
        "linkedin.is_sponsored": FieldSpec.at("adContext.isDsc"),
        "linkedin.edited": FieldSpec.at("lifecycleStateInfo.isEditedByAuthor"),
    },
    # No `engagement` block, and that is the provider's fault rather than an
    # omission: the Posts API returns no reaction or comment counts. They live
    # behind `/rest/socialActions/{urn}`, one request per post, against a *daily*
    # per-app quota -- so a page of 50 posts would cost 51 requests instead of 1
    # and exhaust the day's budget on a few hundred posts.
    author_id=FieldSpec.at("author"),
    author_profile_url=FieldSpec.at("author", transform=_author_url),
    # No `author_handle`: the payload carries the URN and nothing else. Resolving
    # it to a name needs `/rest/organizations/{id}`, which is another request per
    # distinct author -- and there is exactly one author per connector account, so
    # `services/` can resolve it once for the whole corpus instead.
    content_type="text/plain",
)
"""Organic and sponsored posts alike; the Posts API returns them in one listing.

`text_is_html` stays false. `commentary` is LinkedIn's "little" format -- plain
text with backslash escapes and two annotation templates -- not markup, and
running the readability extractor over it would strip nothing while risking the
templates `_from_little_text` has already resolved.
"""


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #


class LinkedInConnector(BaseConnector):
    """Posts authored by one LinkedIn organisation, walked newest-first."""

    slug: ClassVar[str] = "linkedin"
    platform: ClassVar[Platform] = Platform.LINKEDIN
    category: ClassVar[SourceCategory] = SourceCategory.SOCIAL
    auth_type: ClassVar[AuthType] = AuthType.OAUTH2
    version: ClassVar[str] = "0.1.0"

    requires_tos_review: ClassVar[bool] = True
    """Set, and it must stay set until a documented review says otherwise.

    `connectors/registry.py` refuses to `enable()` *or* `create()` a class
    carrying this flag, so in practice an operator meets the registry's message
    before any message in this module. That is the intended order: the flag is a
    statement that no lawful path to LinkedIn's public content exists, and this
    connector's narrow first-party surface does not change that
    (`docs/connector-spec.md` §9.1, `docs/security-and-privacy.md`).
    """

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=30, burst=1, concurrency=1
    )
    """Serialized, because there is no per-minute limit to encode.

    LinkedIn's documented limits are two *daily* budgets -- one per application,
    one per member per application -- resetting at midnight UTC, and their values
    are deliberately unpublished: they are visible only on the Analytics tab of
    the app that made the call. No `X-RateLimit-*` header comes back either, so
    `parse_rate_limit` has nothing to reconcile against and the bucket runs on a
    local estimate for the whole day.

    `burst=1` and `concurrency=1` are the load-bearing parts. Against an unmetered
    daily cap the only defensible posture is to spend it evenly, and a burst is
    answered with a 429 that costs more than the requests it saved. The real
    budget is the poll cadence in `workers/scheduler.py`; this policy only stops a
    single run from sprinting.
    """

    supports_incremental: ClassVar[bool] = True

    supports_backfill: ClassVar[bool] = False
    """Offset pagination over a feed that grows at the head cannot be trusted for
    a multi-run historical crawl: every post published between two runs shifts
    every offset by one, so a resumed crawl re-reads one item and skips none --
    until a post is deleted, at which point it skips one and nothing notices.

    A first incremental run already descends to the end of the listing, which is
    the whole history a company page has. A backfill mode would differ only in
    which cursor row it wrote, while being strictly less correct.
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
        # `PermanentError` -- before a socket exists (`docs/connector-spec.md` §6:
        # configuration defects fail fast, and no cursor is created for one).
        self._author_urn = _validated_author_urn(ctx.params)
        self._api_version = _validated_api_version(ctx.params)
        self._count = _validated_count(ctx.params)
        self._base_url = str(ctx.params.get("base_url") or API_BASE).rstrip("/")
        self._oauth_config = self._build_oauth_config(credentials)

        # Process-local, because `SyncContext` carries no port for a token store.
        # The cross-replica single-flight lock `connectors/auth/oauth.py` is built
        # around therefore degrades to an asyncio lock and each worker refreshes
        # its own token. Tolerable while a refresh costs one request; the fix is a
        # port on `SyncContext`, not a Redis import here.
        self._token_store = token_store or InMemoryTokenStore()
        self._now = now
        self._client: httpx.AsyncClient | None = None
        self._oauth: OAuth2Client | None = None
        self._mapping = MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=ctx.run_id,
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct and validate. No I/O: not even the HTTP client is built."""
        return cls(ctx, credentials)

    def _build_oauth_config(self, credentials: Credentials) -> OAuth2Config:
        """Assemble the refresh-token grant, naming what is missing and why.

        The *refresh* grant, not client credentials, and that is not a preference.
        LinkedIn will happily mint an app-only token from a client id and secret,
        and that token cannot read a single post: content permissions
        (`r_organization_social`) are delegated by a member, so the only usable
        credential is one that came out of a three-legged consent flow. A
        connector configured with the two client secrets alone would authenticate
        successfully and then 403 on every fetch -- which is why the absence is
        caught here, with a message that says what to go and get.
        """
        missing = [
            key
            for key in ("client_id", "client_secret", "refresh_token")
            if not credentials.secrets.get(key)
        ]
        if missing:
            raise ConnectorConfigurationError(
                f"LinkedIn credentials are incomplete: {missing} not set for account "
                f"{credentials.account_id!r}. This connector needs a three-legged "
                "OAuth 2.0 refresh token minted for a LinkedIn app that has been "
                "granted the Community Management API product (Marketing Developer "
                "Platform) with the r_organization_social permission, authorized by a "
                "member who holds an ADMINISTRATOR, CONTENT_ADMIN or "
                "DIRECT_SPONSORED_CONTENT_POSTER role on the target organisation. "
                "client_id/client_secret alone are not enough -- LinkedIn's app-only "
                "token cannot read posts. Set secrets['client_id'], "
                "secrets['client_secret'] (LINKEDIN_CLIENT_ID / "
                "LINKEDIN_CLIENT_SECRET in .env.example) and secrets['refresh_token'] "
                "from the consent flow the API layer runs",
                connector=self.slug,
                account_id=credentials.account_id,
            )
        return OAuth2Config(
            token_url=TOKEN_URL,
            client_id=credentials.require("client_id"),
            client_secret=credentials.require("client_secret"),
            refresh_token=credentials.require("refresh_token"),
            grant=OAuth2Grant.REFRESH_TOKEN,
            # LinkedIn documents the credentials in the form body and rejects the
            # Basic header, which comes back as `invalid_client` -- indistinguishable
            # from a wrong secret, and debugged as one for hours.
            client_auth=ClientAuthMethod.POST,
            timeout_seconds=self.ctx.request_timeout_seconds,
        )

    # ------------------------------------------------------------- lifecycle --

    async def authenticate(self) -> None:
        """Refresh the member token. Idempotent.

        Idempotence is not a courtesy: the runtime calls this once at the start of
        a run and at most once more after a 401, and `OAuth2Client` answers the
        second call from its store unless `invalidate()` marked the token unusable
        -- which `_access_denied` does exactly when a 401 says it should.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.ctx.request_timeout_seconds,
                headers={
                    "User-Agent": self.ctx.user_agent,
                    "Accept": "application/json",
                    # Both are mandatory on every versioned call. Without
                    # `X-Restli-Protocol-Version` LinkedIn parses the request under
                    # Rest.li 1.0 rules, where `List(...)` syntax and URN encoding
                    # mean something else, and answers a 400 about a field that
                    # looks correct.
                    "LinkedIn-Version": self._api_version,
                    "X-Restli-Protocol-Version": "2.0.0",
                },
                # A redirect off api.linkedin.com would carry the member's bearer
                # token to whatever answered: httpx strips only *its own* auth on a
                # cross-host redirect, and this Authorization header is ours.
                follow_redirects=False,
            )
        if self._oauth is None:
            self._oauth = OAuth2Client(
                self._oauth_config,
                account_id=self.ctx.account_id,
                store=self._token_store,
                # Shared, so refreshing does not open a second connection pool --
                # and so `aclose()` has exactly one thing to close.
                client=self._client,
                connector=self.slug,
                user_agent=self.ctx.user_agent,
                now=self._now,
            )
        await self._oauth.token()

    async def aclose(self) -> None:
        """Release the client. Called from `run()`'s `finally`, always.

        Both handles are dropped rather than merely closed, so a second `run()` on
        the same instance rebuilds them instead of reusing a closed pool. The token
        survives regardless -- it lives in the store, not in the client.
        """
        client, self._client, self._oauth = self._client, None, None
        if client is not None:
            await client.aclose()

    # ----------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Descend the author listing, pinning the watermark until it completes.

        Pages stream rather than being buffered whole -- unlike
        `connectors/social/reddit.py`, which has to buffer because its restart
        token is only known after the descent. Here the restart point is an
        integer offset that is known *before* the request, so each page can carry
        a legal cursor of its own and a crash costs one page instead of a run.

        The pinning is what makes streaming safe. Every page below the closing one
        reports the watermark this run started from and parks its real progress in
        `checkpoint`, so a run killed at page 4 has committed nothing above records
        it never emitted. `BaseConnector.run()` raises that pinned value back to
        the un-rewound watermark on the way out (`_guard_watermark`), which is why
        this method never has to undo the `overlap_seconds` shift itself.
        """
        watermark = cursor.watermark
        start = _as_offset(cursor.page_token)
        running = _parse_pending(cursor)
        endpoint = f"{self._base_url}{POSTS_PATH}"
        fetched = 0
        pages = 0

        while True:
            count = self._request_budget(fetched)
            if count == 0:
                # A run given a zero record budget. Nothing was fetched, so there
                # is nothing to commit, and yielding a cursor here would report
                # progress that did not happen.
                return

            params = self._finder_params(start, count)
            await self.acquire_slot(endpoint)
            body, headers = await self._get(endpoint, params)

            elements = _elements(body, endpoint, self.slug, self.ctx.account_id)
            fingerprint = _request_fingerprint(endpoint, params)
            fetched_at = self._now()
            # Reversed *within* the page as well as between pages: LinkedIn returns
            # a page newest-first, and a consumer reading the emitted stream in
            # order should see time move forward inside a batch for the same reason
            # it must between them -- half-ordered output is the kind of thing
            # downstream code accidentally relies on being total.
            records = tuple(
                self._to_record(
                    element, fetched_at=fetched_at, fingerprint=fingerprint, endpoint=endpoint
                )
                for element in reversed(elements)
            )
            fetched += len(records)
            pages += 1
            running = _max_datetime(running, _newest_timestamp(records))

            oldest = _oldest_timestamp(records)
            complete = (
                not records
                # The provider's own end-of-listing signal, and the *only* one.
                # See the module docstring: a short page does not mean the last
                # page on this API.
                or not _has_next_page(body, endpoint, self.slug, self.ctx.account_id)
                # Crossed into ground a previous run already covered. The overlap
                # is deliberate (`BaseConnector.overlap_seconds`); dedup absorbs it.
                or (watermark is not None and oldest is not None and oldest <= watermark)
            )

            yield FetchPage(
                records=records,
                cursor=(
                    Cursor(watermark=running, page_token=None, checkpoint={})
                    if complete
                    else Cursor(
                        watermark=cursor.watermark,
                        page_token=str(start + count),
                        checkpoint=_pending_checkpoint(running),
                    )
                ),
                raw_headers=headers,
            )
            if complete or self._budget_reached(fetched, pages):
                return
            start += count

    def _finder_params(self, start: int, count: int) -> dict[str, str]:
        """Build the author finder query. Nothing secret goes in here.

        `sortBy=CREATED` rather than LinkedIn's `LAST_MODIFIED` default, because
        the watermark is `createdAt`; see the module docstring.

        `viewContext=READER` is sent explicitly even though it is the documented
        default. The default is LinkedIn's to change, and the other value returns
        drafts and unpublished revisions -- content no reader ever saw, which would
        enter the corpus as though it had.
        """
        return {
            "q": "author",
            "author": self._author_urn,
            "start": str(start),
            "count": str(count),
            "sortBy": "CREATED",
            "viewContext": "READER",
        }

    def _request_budget(self, fetched: int) -> int:
        """`count` for the next request, narrowed by `ctx.max_records`.

        Applied before the request rather than after the page: a run capped at ten
        records should cost one request for ten, not one for fifty that are then
        discarded against a *daily* quota.

        Counted on records fetched rather than emitted, since a connector cannot
        see what survived dedup. Fetched is an upper bound on emitted, so the run
        stops at or before the ceiling and the cursor commits either way.
        """
        if self.ctx.max_records is None:
            return self._count
        return _clamp(self.ctx.max_records - fetched, 0, self._count)

    def _budget_reached(self, fetched: int, pages: int) -> bool:
        if pages >= MAX_OFFSET_PAGES:
            return True
        if self.ctx.max_pages is not None and pages >= self.ctx.max_pages:
            return True
        return self.ctx.max_records is not None and fetched >= self.ctx.max_records

    def _to_record(
        self, element: Any, *, fetched_at: datetime, fingerprint: str, endpoint: str
    ) -> RawRecord:
        """Wrap one `elements[]` entry verbatim.

        `raw_bytes` is deliberately `None`. `RawRecord` documents it as the exact
        bytes the provider returned, and for an item carved out of a listing those
        bytes do not exist -- one response carried fifty posts. Re-serializing the
        payload to fill the field would produce a digest that changes with the json
        library and break the content-addressed R2 key it feeds.
        """
        if not isinstance(element, Mapping):
            raise self._shape_error("a post element is not an object", endpoint)
        urn = _as_text(element.get("id"))
        if not urn:
            # LinkedIn has never omitted `id`. If it is missing, this is not the
            # endpoint we think it is, and manufacturing an identity would attach a
            # Signal to something nothing can resolve back to the provider.
            raise self._shape_error("a post element carries no id URN", endpoint)
        return RawRecord(
            native_id=urn,
            payload=element,
            fetched_at=fetched_at,
            raw_bytes=None,
            content_type="application/json",
            source_url=_post_url(urn) or None,
            request_fingerprint=fingerprint,
        )

    # ------------------------------------------------------------- normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one post onto a Signal, or drop it.

        Two drops, both of them ordinary rather than defective. An unpublished
        `lifecycleState` is content no reader saw. A post with neither commentary
        nor an article headline is an image-only or video-only share, and this
        connector does not resolve LinkedIn's media URNs to files (see `_FIELD_MAP`)
        -- so emitting it would put an empty document into the embedding queue and
        the search index, which is what `connectors/news/gdelt.py` refuses for
        title-less articles for the same reason.
        """
        payload = record.payload

        state = _as_text(payload.get("lifecycleState"))
        if state and state != _PUBLISHED:
            return None

        commentary = _from_little_text(payload.get("commentary"))
        if not commentary and not _as_text(_lookup(payload, "content.article.title")):
            return None

        # The runtime keys the R2 object and the Kafka partition off
        # `RawRecord.native_id`, while every store keys off `Signal.id`, which is
        # derived from `id`. If those two disagreed the same post would exist under
        # two identities, so the disagreement is caught here rather than discovered
        # as duplicate rows months later.
        if _as_text(payload.get("id")) != record.native_id:
            raise NormalizationError(
                "payload id does not match the fetched record's native_id",
                native_id=record.native_id,
                connector=self.slug,
            )

        mentions = _mentioned_urns(payload.get("commentary"))
        return _FIELD_MAP.to_signal(
            record,
            self._mapping,
            extra_metadata={"linkedin.mentions": mentions} if mentions else None,
        )

    # --------------------------------------------------------------- request --

    async def _get(
        self, url: str, params: Mapping[str, str]
    ) -> tuple[Mapping[str, Any], dict[str, str]]:
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

        headers = await self._oauth.headers()
        # LinkedIn's Rest.li router dispatches on this header, not on the query
        # string. Omitting it lets a finder be read as a BATCH_GET with no ids,
        # which answers 200 with an empty `results` object -- a successful run that
        # returns nothing, forever.
        headers["X-RestLi-Method"] = "FINDER"
        try:
            response = await self._client.get(url, params=params, headers=headers)
        except httpx.TransportError as exc:
            raise TransientError(
                "LinkedIn is unreachable",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                "the Posts request could not be issued",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc

        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise await self._raise_for_status(response)

        try:
            payload = response.json()
        except ValueError as exc:
            # Usually a login page served by an edge in front of the API. The body
            # is not attached: it can echo the request, and the request carries the
            # member's bearer token in a header (§1 forbids logging either).
            raise PermanentError(
                "LinkedIn returned a body that is not JSON",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(payload, Mapping):
            raise self._shape_error("response is not a JSON object", url)
        return payload, _rate_limit_headers(response.headers)

    async def _raise_for_status(self, response: httpx.Response) -> ConnectorError:
        """Classify a 4xx/5xx. Always returns an exception for the caller to raise.

        LinkedIn is unusually clean about the 401/403 split and this maps it
        faithfully instead of collapsing the pair the way
        `connectors/social/reddit.py` has to. `401` is `EMPTY_ACCESS_TOKEN` or an
        expired one -- a credential problem, so `AuthError`, so the account is
        flagged `needs_reauth`. `403` is `ACCESS_DENIED`: the token is fine and
        either the app lacks `r_organization_social` or the authorizing member has
        no role on this organisation. Filing that as `AuthError` would send an
        operator to re-link credentials that were never the problem, over and over,
        because re-linking cannot grant a company-page role.
        """
        status = response.status_code
        code = _service_error_code(response)
        details: dict[str, Any] = {"linkedin_code": code} if code else {}

        if status == httpx.codes.UNAUTHORIZED:
            if self._oauth is not None:
                # Expire the cached token in place so the runtime's single permitted
                # re-authentication mints a new one rather than replaying the
                # rejected one. `invalidate()` keeps the refresh material;
                # `delete()` would send a human to a consent screen over a token
                # that merely expired early.
                await self._oauth.invalidate()
            return AuthError(
                "LinkedIn rejected the access token",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status == httpx.codes.FORBIDDEN:
            return ConnectorConfigurationError(
                f"LinkedIn refused to read {self._author_urn}: the token is valid but "
                "not authorised for it. Either the app has not been granted "
                "r_organization_social through the Community Management API product, "
                "or the member who authorized it holds no ADMINISTRATOR / "
                "CONTENT_ADMIN / DIRECT_SPONSORED_CONTENT_POSTER role on that "
                "organisation. Re-linking the account cannot fix either one",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status == httpx.codes.UPGRADE_REQUIRED:
            return ConnectorConfigurationError(
                f"LinkedIn has sunset API version {self._api_version!r}. Versions are "
                "retired roughly a year after release and this one is past it; set "
                "params['api_version'] to a supported YYYYMM value. Nothing else is "
                "wrong -- the credential and the organisation are fine",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status == httpx.codes.TOO_MANY_REQUESTS:
            return self._throttled(response, details)
        if status == httpx.codes.CONFLICT:
            # Documented as "a write conflict occurred; retry the request". Rare on
            # a read path, but it is explicitly retryable and misfiling it as
            # permanent would DLQ a page that would have succeeded.
            return TransientError(
                "LinkedIn reported a write conflict",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            return TransientError(
                "LinkedIn returned a server error",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        return PermanentError(
            "LinkedIn rejected the Posts request",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=status,
            details={**details, "author": self._author_urn},
        )

    def _throttled(self, response: httpx.Response, details: dict[str, Any]) -> ConnectorError:
        """Classify a 429 by how long the provider actually wants us gone.

        LinkedIn's limits are *daily* and reset at midnight UTC, and it sends no
        `Retry-After`. So the honest default when the header is absent is the time
        to that reset, not a thirty-second backoff -- backing off would spend the
        rest of the day's schedule rediscovering the same 429, and each rediscovery
        costs another request against the budget that is already gone.

        `QuotaError` rather than `TransientError` is the whole point: it is a
        *partial success*, so the records this run already emitted stay emitted and
        the cursor commits (`connectors/exceptions.py`).
        """
        hint = self.parse_rate_limit(response.headers)
        wait = hint.retry_after_seconds if hint is not None else None
        if wait is None:
            wait = _seconds_to_utc_midnight(self._now())
        if wait > QUOTA_WAIT_THRESHOLD_SECONDS:
            return QuotaError(
                "LinkedIn's daily throttle limit is reached",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                retry_after_seconds=wait,
                reset_at=self._now().timestamp() + wait,
                details=details,
            )
        return TransientError(
            "LinkedIn throttled the request",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=response.status_code,
            details={**details, "retry_after_seconds": wait},
        )

    def _shape_error(self, detail: str, endpoint: str) -> PermanentError:
        """A listing that is not shaped like a listing.

        `PermanentError` rather than `TransientError`: a retry returns the same
        bytes, and §6 files an unparsable page structure as a defect for a human to
        look at rather than as something to back off from.
        """
        return PermanentError(
            f"unexpected LinkedIn Posts response: {detail}",
            connector=self.slug,
            account_id=self.ctx.account_id,
            details={"endpoint": endpoint},
        )


# --------------------------------------------------------------------------- #
# Response helpers
# --------------------------------------------------------------------------- #


def _elements(
    body: Mapping[str, Any], endpoint: str, slug: str, account_id: str
) -> list[Mapping[str, Any]]:
    """The `elements` array of a finder response.

    An absent `elements` is a shape change rather than an empty result: the finder
    returns `"elements": []` for an organisation with no posts, so its absence
    means this is not a finder response at all -- most often a BATCH_GET answered
    because the `X-RestLi-Method` header went missing.
    """
    elements = body.get("elements")
    if not isinstance(elements, Sequence) or isinstance(elements, (str, bytes)):
        raise PermanentError(
            "unexpected LinkedIn Posts response: no 'elements' array",
            connector=slug,
            account_id=account_id,
            details={"endpoint": endpoint},
        )
    return [item for item in elements if isinstance(item, Mapping)]


def _has_next_page(
    body: Mapping[str, Any], endpoint: str, slug: str, account_id: str
) -> bool:
    """Whether LinkedIn says another page exists.

    `paging.links` carrying `rel: "next"` is the authority, and the *only* one --
    see the module docstring on short pages. A missing `paging` object raises
    rather than reading as "no more": treating a shape change as the end of a
    listing would commit a watermark over every post below it, and nothing would
    ever go back for them.
    """
    paging = body.get("paging")
    if not isinstance(paging, Mapping):
        raise PermanentError(
            "unexpected LinkedIn Posts response: no 'paging' object, so there is no "
            "way to tell a last page from a short one",
            connector=slug,
            account_id=account_id,
            details={"endpoint": endpoint},
        )
    links = paging.get("links")
    if links is None:
        return False
    if not isinstance(links, Sequence) or isinstance(links, (str, bytes)):
        raise PermanentError(
            "unexpected LinkedIn Posts response: 'paging.links' is not a list",
            connector=slug,
            account_id=account_id,
            details={"endpoint": endpoint},
        )
    return any(
        isinstance(link, Mapping) and _as_text(link.get("rel")) == "next" for link in links
    )


def _created_at(record: RawRecord) -> datetime | None:
    """Event time of one record, or `None` when the payload carries no usable one.

    Used only for pagination arithmetic. A post whose `createdAt` is missing or
    unparseable is still emitted -- `normalize` raises for it and it reaches the
    DLQ -- but it must not be allowed to poison the watermark, so it is excluded
    here rather than defaulted to now.
    """
    raw = record.payload.get("createdAt") or record.payload.get("publishedAt")
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


def _as_offset(page_token: str | None) -> int:
    """Read the parked `start` offset, treating anything unusable as a fresh start.

    `docs/connector-spec.md` §4.1 rule 4 makes the page token advisory: a connector
    that finds it rejected re-pages from the top rather than failing the run. A
    negative offset is refused for the same reason a garbage one is -- LinkedIn
    answers it with a 400 that costs a request against a daily cap.
    """
    if page_token is None:
        return 0
    try:
        offset = int(str(page_token).strip())
    except ValueError:
        return 0
    return offset if offset > 0 else 0


def _service_error_code(response: httpx.Response) -> str | None:
    """The `code` field of a LinkedIn error body, and nothing else.

    Never `message`: LinkedIn's error messages routinely quote the request that
    caused them, and this request carries a bearer token
    (`docs/connector-spec.md` §1).
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    code = payload.get("code")
    if not isinstance(code, str):
        return None
    code = code.strip()
    return code if _SERVICE_ERROR_CODE.match(code) else None


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in _RATE_LIMIT_HEADERS}


def _request_fingerprint(url: str, params: Mapping[str, str]) -> str:
    """Hash of endpoint plus normalized params -- never credentials.

    `lineage.request_fingerprint` is what makes a fetch reproducible: it names the
    exact request that produced a record without naming who made it.
    """
    canonical = urlencode(sorted(params.items()))
    return hashlib.sha256(f"GET {url}?{canonical}".encode()).hexdigest()


def _seconds_to_utc_midnight(now: datetime) -> float:
    """Time until LinkedIn's daily budgets reset."""
    moment = now.astimezone(UTC)
    reset = (moment + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(0.0, (reset - moment).total_seconds())


def _lookup(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for segment in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(segment)
    return current


def _as_text(value: Any) -> str:
    """Render a scalar as a stripped string; `""` for anything unusable.

    Booleans are refused rather than stringified: `"True"` is never a URN, a
    lifecycle state or an error code, and letting one through turns a type
    confusion into a plausible-looking value.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (Mapping, Sequence, set, frozenset)):
        return ""
    return str(value).strip()


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
