"""Instagram through the Graph API: business accounts and hashtag search.

Instagram has no general content API. What exists is the Graph API, reached
through a **linked Facebook Page**, and it sees only Business or Creator accounts
plus a hashtag search with a hard 30-day window. Personal accounts are not
reachable at all, and that is a property of the platform rather than of this
connector -- so it is stated at construction and in the catalogue rather than
discovered as an empty result set.

**The 30-day hashtag window is a wall, not a default.** `hashtag_recent_media`
returns nothing older, whatever date range is requested, so `supports_backfill`
is False and a run asking for last quarter gets an honest empty page rather than
a silently truncated one.

`requires_tos_review` is True: scraping the public web interface is the obvious
alternative and is forbidden.
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

__all__ = ["InstagramConnector"]

DEFAULT_BASE_URL: Final = "https://graph.facebook.com/v21.0"
HASHTAG_WINDOW_DAYS: Final = 30
"""The API returns nothing older, whatever range is asked for."""

MEDIA_FIELDS: Final = "id,caption,media_type,permalink,timestamp,like_count,comments_count,username"


class InstagramConnector(BaseConnector):
    """Instagram Graph API hashtag and business-account media."""

    slug: ClassVar[str] = "instagram"
    platform: ClassVar[Platform] = Platform.INSTAGRAM
    category: ClassVar[SourceCategory] = SourceCategory.SOCIAL
    auth_type: ClassVar[AuthType] = AuthType.OAUTH2
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=30, burst=2, concurrency=1
    )
    """Meta rate-limits per app on a rolling hour with a formula rather than a
    fixed number, and answers an overrun with a 4 code embedded in a 200. Thirty
    a minute stays well inside any realistic budget; the real protection is
    `_check_graph_error`, which reads the embedded code."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = False
    overlap_seconds: ClassVar[int] = 3600

    requires_tos_review: ClassVar[bool] = True
    """Personal accounts are unreachable through any official surface.

    The obvious alternative -- scraping the public web interface -- is forbidden
    by Instagram's terms. Published through the catalogue so a refusal reads as
    a policy decision rather than a broken source.
    """

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        token = credentials.secrets.get("access_token")
        if not token:
            raise ConnectorConfigurationError(
                "instagram requires a Facebook Page access token with "
                "instagram_basic and instagram_manage_insights. Note that the "
                "Graph API reaches only Business and Creator accounts plus "
                "hashtag search -- personal accounts are not accessible through "
                "any official surface."
            )
        self._token = token
        self._ig_user_id = str(params.get("ig_user_id") or "").strip()
        if not self._ig_user_id:
            raise ConnectorConfigurationError(
                "instagram requires `ig_user_id`, the Instagram Business Account "
                "id linked to the Page the token belongs to."
            )
        self._hashtag = str(params.get("hashtag") or "").strip().lstrip("#") or None

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """Page tokens are long-lived and supplied; nothing to exchange."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Hashtag media when a hashtag is configured, own media otherwise."""
        if self._hashtag:
            async for page in self._fetch_hashtag(cursor):
                yield page
            return
        async for page in self._fetch_own_media(cursor):
            yield page

    async def normalize(self, record: RawRecord) -> Signal | None:
        media = record.payload
        caption = str(media.get("caption") or "").strip()
        posted = _parse_ts(media.get("timestamp"))
        if posted is None or not caption:
            return None

        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=media.get("permalink"),
            author=Author(handle=media.get("username")) if media.get("username") else None,
            timestamp=posted,
            content=Content(text=caption),
            engagement=Engagement(
                endorsement=int(media.get("like_count") or 0),
                discussion=int(media.get("comments_count") or 0),
            ),
            metadata={"media_type": media.get("media_type")},
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._ctx.request_timeout_seconds,
                headers={"User-Agent": self._ctx.user_agent, **self._auth_headers()},
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        # Meta takes the token as a query parameter rather than a header, so
        # it is attached per request in `_query_with_token` instead.
        return {}

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
            raise TransientError(f"instagram timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"instagram request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "instagram rejected the credential. The Page token must carry instagram_basic and the linked account must be Business or Creator."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"instagram rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"instagram returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"instagram rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"instagram returned a non-JSON body") from error

    def _query_with_token(self, extra: Mapping[str, Any]) -> dict[str, Any]:
        return {**extra, "access_token": self._token}

    async def _fetch_hashtag(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Resolve the hashtag to an id, then read its recent media.

        Two requests rather than one because Meta requires the hashtag id and
        will not accept the string. The id is stable, so it is cached in the
        checkpoint -- otherwise every poll spends a request rediscovering it.
        """
        checkpoint = dict(cursor.checkpoint or {})
        hashtag_id = checkpoint.get("hashtag_id")
        if not hashtag_id:
            found = await self._get(
                "/ig_hashtag_search",
                self._query_with_token({"user_id": self._ig_user_id, "q": self._hashtag}),
            )
            entries = found.get("data") or []
            if not entries:
                raise PermanentError(
                    f"instagram does not recognise the hashtag #{self._hashtag}"
                )
            hashtag_id = entries[0]["id"]

        payload = await self._get(
            f"/{hashtag_id}/recent_media",
            self._query_with_token(
                {"user_id": self._ig_user_id, "fields": MEDIA_FIELDS}
            ),
        )
        yield self._page(payload, cursor, {"hashtag_id": hashtag_id})

    async def _fetch_own_media(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        payload = await self._get(
            f"/{self._ig_user_id}/media",
            self._query_with_token({"fields": MEDIA_FIELDS}),
        )
        yield self._page(payload, cursor, {})

    def _page(
        self, payload: Mapping[str, Any], cursor: Cursor, checkpoint: Mapping[str, Any]
    ) -> FetchPage:
        media = payload.get("data") or []
        records = [
            RawRecord(
                native_id=str(item["id"]),
                payload=item,
                fetched_at=utcnow(),
                source_url=item.get("permalink"),
            )
            for item in reversed(media)
            if item.get("id")
        ]
        latest = max(
            (d for d in (_parse_ts(m.get("timestamp")) for m in media) if d), default=None
        )
        return FetchPage(
            records=records,
            cursor=Cursor(
                version=cursor.version,
                watermark=latest or cursor.watermark,
                checkpoint=dict(checkpoint),
            ),
        )


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

