"""TikTok via the Research API, which requires an approved application.

There is no lawful way to collect TikTok content at scale without
credentials. The Research API exists, is the correct surface, and requires an
application that is reviewed and can be refused -- so this connector's most
important behaviour is what it does *without* those credentials: it refuses at
construction, naming the exact programme to apply to.

`requires_tos_review` is True. `docs/security-and-privacy.md` §7 makes that a
hard gate, and the catalogue endpoint publishes it so an operator can tell a
policy refusal from an outage. Scraping the public web interface would be the
obvious alternative and is forbidden by TikTok's terms; it is not implemented and
should not be.

**Query syntax is a structured object, not a string.** The Research API takes a
JSON condition tree, which is why the `query` param here is a mapping rather than
the search string every other social connector takes.
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

__all__ = ["TikTokConnector"]

DEFAULT_BASE_URL: Final = "https://open.tiktokapis.com/v2"
MAX_COUNT: Final = 100


class TikTokConnector(BaseConnector):
    """TikTok Research API video query. Research-tier credentials required."""

    slug: ClassVar[str] = "tiktok"
    platform: ClassVar[Platform] = Platform.TIKTOK
    category: ClassVar[SourceCategory] = SourceCategory.SOCIAL
    auth_type: ClassVar[AuthType] = AuthType.OAUTH2
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=20, burst=1, concurrency=1
    )
    """Conservative: the Research API's published quotas are per-application
    and vary by approval, so there is no single documented figure to size
    against. Serialised, per `docs/connector-spec.md` §9.5."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = True
    overlap_seconds: ClassVar[int] = 3600

    requires_tos_review: ClassVar[bool] = True
    """No lawful bulk access without approved Research API credentials.

    Published through the catalogue so a refusal to collect reads as a policy
    decision rather than a broken source. See the module docstring.
    """

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        token = credentials.secrets.get("access_token")
        if not token:
            raise ConnectorConfigurationError(
                "tiktok requires an approved TikTok Research API access token. "
                "Apply at developers.tiktok.com/products/research-api -- the "
                "application is reviewed and can be refused. There is no lawful "
                "alternative surface for this data, and scraping the web "
                "interface violates TikTok's terms."
            )
        self._token = token
        self._query = params.get("query")
        if not isinstance(self._query, Mapping):
            raise ConnectorConfigurationError(
                "tiktok requires a `query` param as a structured condition object, "
                "not a search string -- the Research API takes a JSON condition "
                "tree, e.g. {'and': [{'operation': 'IN', 'field_name': 'region_code', "
                "'field_values': ['GB']}]}."
            )
        self._region = str(params.get("region") or "")

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """The token is supplied; refresh is the credential store's job."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Query videos in a date window, paging with the API's cursor.

        The Research API pages with an integer `cursor` plus a `search_id` that
        pins the result set -- both must be carried together, because a cursor
        from one search is meaningless against another. They travel in the
        connector checkpoint as a pair for that reason.
        """
        client = self._ensure_client()
        checkpoint = dict(cursor.checkpoint or {})
        api_cursor = int(checkpoint.get("cursor", 0))
        search_id = checkpoint.get("search_id")
        emitted = 0

        start = (cursor.watermark or utcnow() - timedelta(days=7)).astimezone(UTC)
        end = utcnow()

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            body: dict[str, Any] = {
                "query": dict(self._query),
                "start_date": start.strftime("%Y%m%d"),
                "end_date": end.strftime("%Y%m%d"),
                "max_count": MAX_COUNT,
                "cursor": api_cursor,
            }
            if search_id:
                body["search_id"] = search_id

            try:
                response = await client.post(
                    f"{self._base_url}/research/video/query/"
                    "?fields=id,create_time,username,video_description,like_count,"
                    "comment_count,share_count,view_count,region_code",
                    json=body,
                )
            except httpx.HTTPError as error:
                raise TransientError(f"tiktok request failed: {error}") from error

            if response.status_code in (401, 403):
                raise AuthError(
                    "tiktok rejected the credential. Research API tokens expire; "
                    "confirm the application is still approved."
                )
            if response.status_code == 429:
                raise QuotaError("tiktok research API quota exhausted")
            if response.status_code >= 500:
                raise TransientError(f"tiktok returned {response.status_code}")
            if response.status_code >= 400:
                raise PermanentError(
                    f"tiktok rejected the query: {response.text[:200]}"
                )

            data = (response.json() or {}).get("data") or {}
            videos = data.get("videos") or []
            if not videos:
                return

            records = [
                RawRecord(
                    native_id=str(video["id"]),
                    payload=video,
                    fetched_at=utcnow(),
                    source_url=f"https://www.tiktok.com/@{video.get('username','')}/video/{video['id']}",
                )
                for video in videos
                if video.get("id")
            ]
            emitted += len(records)

            latest = max(
                (d for d in (_parse_ts(v.get("create_time")) for v in videos) if d),
                default=None,
            )
            api_cursor = int(data.get("cursor") or api_cursor + len(videos))
            search_id = data.get("search_id") or search_id

            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"cursor": api_cursor, "search_id": search_id},
                ),
            )
            if not data.get("has_more"):
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        video = record.payload
        text = str(video.get("video_description") or "").strip()
        created = _parse_ts(video.get("create_time"))
        if created is None:
            return None
        if not text:
            # A video with no description carries no text to retrieve against.
            # Dropped rather than stored as an empty body.
            return None

        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=record.source_url,
            author=Author(handle=video.get("username")) if video.get("username") else None,
            timestamp=created,
            content=Content(text=text),
            engagement=Engagement(
                reach=int(video.get("view_count") or 0),
                endorsement=int(video.get("like_count") or 0),
                amplification=int(video.get("share_count") or 0),
                discussion=int(video.get("comment_count") or 0),
            ),
            metadata={"region_code": video.get("region_code")},
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
            raise TransientError(f"tiktok timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"tiktok request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "tiktok rejected the credential. Research API tokens expire and the application must remain approved."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"tiktok rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"tiktok returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"tiktok rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"tiktok returned a non-JSON body") from error


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

