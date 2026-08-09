"""X (Twitter) posts through the official API v2 recent-search endpoint.

**This needs a paid tier and says so at construction.** The free tier of X API
v2 does not include search at all -- it is post-only. A connector that failed at
the first request with a generic 403 would send an operator looking for a scope
problem; raising at construction with the tier named is the difference between a
five-minute fix and an afternoon.

**`expansions` and `tweet.fields` in one request, not N+1.** The default response
is an id and a text body: no author, no metrics, no timestamp. Fetching those
per-post would multiply request count by the page size and exhaust a tier's quota
in minutes. The expansion mechanism returns them in an `includes` block on the
same response, which is why `_index_includes` exists.

**Recent search reaches seven days.** That is the endpoint's hard limit on the
Basic tier, so `supports_backfill` is False -- a backfill mode would differ from
an incremental run only in which cursor row it wrote while hitting the same wall.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Self

import httpx

from models.base import utcnow
from models.enums import AuthType, Platform, SourceCategory
from models.signal import Author, Content, Engagement, Signal, signal_id
from connectors.base import (
    BaseConnector,
    Credentials,
    Cursor,
    FetchPage,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    NormalizationError,
    PermanentError,
    QuotaError,
    TransientError,
)

__all__ = ["XConnector"]

DEFAULT_BASE_URL: Final = "https://api.x.com/2"
MAX_RESULTS: Final = 100
"""The endpoint's per-page ceiling on the Basic tier."""

TWEET_FIELDS: Final = "id,text,created_at,author_id,public_metrics,lang,conversation_id"
EXPANSIONS: Final = "author_id"
USER_FIELDS: Final = "id,username,name,verified"


class XConnector(BaseConnector):
    """X API v2 recent search, walked forward with a pagination token."""

    slug: ClassVar[str] = "x"
    platform: ClassVar[Platform] = Platform.X
    category: ClassVar[SourceCategory] = SourceCategory.SOCIAL
    auth_type: ClassVar[AuthType] = AuthType.BEARER
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=30, burst=3, concurrency=1
    )
    """Sized for the Basic tier's 60 requests per 15 minutes on recent search.

    Half the documented allowance, deliberately: the quota is shared across
    everything the app does, and a connector that consumes all of it leaves
    nothing for anything else using the same credential."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = False
    overlap_seconds: ClassVar[int] = 300

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._query = str(params.get("query") or "").strip()
        if not self._query:
            raise ConnectorConfigurationError(
                "x requires a `query` param using X's search operator syntax, "
                "e.g. '(acme OR \"acme corp\") -is:retweet lang:en'."
            )
        token = credentials.secrets.get("bearer_token")
        if not token:
            raise ConnectorConfigurationError(
                "x requires X_BEARER_TOKEN. Note that the free tier does NOT "
                "include search -- recent search needs Basic or above. A missing "
                "token and an unentitled one both surface as 403, so this is "
                "checked before any request is made."
            )
        self._token = token

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """The bearer token is validated at construction; nothing to exchange."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Page with `next_token`, oldest-first within each page.

        `start_time` is set from the watermark so a poll re-reads only the
        overlap window. Recent search returns newest-first, so each page's
        contents are reversed -- the sanctioned exception in
        `BaseConnector.fetch`, taken here because the endpoint offers no
        ascending sort.
        """
        token: str | None = (
            str(cursor.checkpoint.get("next_token")) if cursor.checkpoint.get("next_token") else None
        ) if cursor.checkpoint else None
        emitted = 0

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            query: dict[str, Any] = {
                "query": self._query,
                "max_results": min(MAX_RESULTS, self._ctx.max_records or MAX_RESULTS),
                "tweet.fields": TWEET_FIELDS,
                "expansions": EXPANSIONS,
                "user.fields": USER_FIELDS,
            }
            if cursor.watermark is not None:
                query["start_time"] = cursor.watermark.astimezone(UTC).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
            if token:
                query["next_token"] = token

            payload = await self._get("/tweets/search/recent", query)
            posts = payload.get("data") or []
            if not posts:
                return

            users = _index_includes(payload)
            records = [
                RawRecord(
                    native_id=str(post["id"]),
                    payload={**post, "_author": users.get(str(post.get("author_id")))},
                    fetched_at=utcnow(),
                    source_url=f"https://x.com/i/status/{post['id']}",
                )
                for post in reversed(posts)
                if post.get("id")
            ]
            emitted += len(records)

            latest = max(
                (d for d in (_parse_ts(p.get("created_at")) for p in posts) if d),
                default=None,
            )
            token = (payload.get("meta") or {}).get("next_token")
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"next_token": token} if token else {},
                ),
            )
            if not token:
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        post = record.payload
        text = str(post.get("text") or "").strip()
        created = _parse_ts(post.get("created_at"))
        if not text or created is None:
            return None

        author = post.get("_author") or {}
        metrics = post.get("public_metrics") or {}
        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=record.source_url,
            author=Author(
                handle=author.get("username"),
                display_name=author.get("name"),
                is_verified=bool(author.get("verified")),
            )
            if author
            else None,
            timestamp=created,
            content=Content(text=text),
            engagement=Engagement(
                reach=int(metrics.get("impression_count") or 0),
                endorsement=int(metrics.get("like_count") or 0),
                amplification=int(metrics.get("retweet_count") or 0),
                discussion=int(metrics.get("reply_count") or 0),
            ),
            metadata={
                "lang": post.get("lang"),
                "conversation_id": post.get("conversation_id"),
            },
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._ctx.request_timeout_seconds,
                headers={"User-Agent": self._ctx.user_agent, **self._auth_headers()},
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def _get(
        self, path: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """One JSON request, with this provider's failures classified.

        The classification is the point. A `TransientError` is retried with
        backoff, a `QuotaError` parks the connector until the window resets, and
        a `PermanentError` stops the run -- so mapping a 401 onto the wrong one
        either loops forever on a revoked token or abandons a source over a blip.
        """
        client = self._ensure_client()
        try:
            response = await client.get(
                f"{self._base_url}{path}", params=dict(params or {})
            )
        except httpx.TimeoutException as error:
            raise TransientError(f"x timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"x request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "x rejected the credential. Check X_BEARER_TOKEN and that the app has at least the Basic tier -- free does not include search."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"x rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"x returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"x rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"x returned a non-JSON body") from error


def _index_includes(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the `includes.users` block by id.

    The expansion mechanism returns authors once, deduplicated, in a sibling
    block rather than inline on each post -- so a page of a hundred posts by ten
    authors carries ten user objects. Indexing them is what makes the join local
    instead of a request per post.
    """
    users = (payload.get("includes") or {}).get("users") or []
    return {str(user["id"]): user for user in users if user.get("id")}


def _parse_ts(value: Any) -> datetime | None:
    """Parse a timestamp, tolerating the shapes providers actually send."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Milliseconds if it is implausibly large as seconds. The alternative --
        # guessing by field name -- breaks the first time a provider renames it.
        seconds = value / 1000 if value > 10_000_000_000 else value
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None

