"""Confluence Cloud pages and blog posts through REST v2.

**Storage format is XHTML, not text.** Confluence returns page bodies as its
own XHTML dialect with custom macro elements. Emitting that raw would put markup
into the Signal body, where it gets embedded, retrieved and eventually quoted in
a report — so the body goes through `connectors/normalize/html.py`, the same
sanitiser the RSS connector uses, rather than a regex that strips angle brackets.

Same authentication as Jira: Atlassian Cloud Basic auth with the account email as
the username.

**Ordered by `modified-date` ascending** for the reason the Jira connector gives
about JQL — v2 paginates with a cursor, but an unordered cursor over a
concurrently-edited space still walks a shifting set.
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

__all__ = ["ConfluenceConnector"]

DEFAULT_BASE_URL: Final = "https://example.atlassian.net/wiki/api/v2"
MAX_PAGE_SIZE: Final = 100


class ConfluenceConnector(BaseConnector):
    """Confluence Cloud pages, with storage-format bodies converted to text."""

    slug: ClassVar[str] = "confluence"
    platform: ClassVar[Platform] = Platform.CONFLUENCE
    category: ClassVar[SourceCategory] = SourceCategory.ENTERPRISE
    auth_type: ClassVar[AuthType] = AuthType.BASIC
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=60, burst=5, concurrency=2
    )
    """Shares Atlassian Cloud's per-user cost budget with the Jira connector,
    so a deployment running both against one account draws from one allowance --
    which is why neither is set near the ceiling."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = True
    overlap_seconds: ClassVar[int] = 300

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        site = str(params.get("site_url") or "").strip().rstrip("/")
        if not site:
            raise ConnectorConfigurationError(
                "confluence requires `site_url`, e.g. 'https://acme.atlassian.net'."
            )
        self._base_url = f"{site}/wiki/api/v2"
        self._site = site
        email = credentials.secrets.get("email")
        token = credentials.secrets.get("api_token")
        if not email or not token:
            raise ConnectorConfigurationError(
                "confluence requires `email` and `api_token` -- Atlassian Cloud "
                "Basic auth uses the account email as the username."
            )
        self._basic = base64.b64encode(f"{email}:{token}".encode()).decode()
        self._space_id = str(params.get("space_id") or "").strip() or None

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """Stateless Basic auth."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Cursor-paged pages, oldest modification first."""
        next_cursor = (
            str(cursor.checkpoint.get("cursor")) if (cursor.checkpoint or {}).get("cursor") else None
        )
        emitted = 0

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            query: dict[str, Any] = {
                "limit": MAX_PAGE_SIZE,
                "body-format": "storage",
                "sort": "modified-date",
            }
            if self._space_id:
                query["space-id"] = self._space_id
            if next_cursor:
                query["cursor"] = next_cursor

            payload = await self._get("/pages", query)
            pages = payload.get("results") or []
            if not pages:
                return

            records = [
                RawRecord(
                    native_id=str(page["id"]),
                    payload=page,
                    fetched_at=utcnow(),
                    source_url=f"{self._site}/wiki/spaces/{page.get('spaceId','')}/pages/{page['id']}",
                )
                for page in pages
                if page.get("id")
            ]
            emitted += len(records)

            latest = max(
                (
                    d
                    for d in (
                        _parse_ts((p.get("version") or {}).get("createdAt")) for p in pages
                    )
                    if d
                ),
                default=None,
            )
            link = (payload.get("_links") or {}).get("next")
            next_cursor = _cursor_from_link(link)
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"cursor": next_cursor} if next_cursor else {},
                ),
            )
            if not next_cursor:
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Convert the storage-format body to text before it becomes a Signal."""
        from connectors.normalize.html import html_to_text

        page = record.payload
        title = str(page.get("title") or "").strip()
        raw_body = (((page.get("body") or {}).get("storage") or {}).get("value")) or ""
        text = html_to_text(raw_body).strip() if raw_body else ""
        if not title:
            return None

        version = page.get("version") or {}
        updated = _parse_ts(version.get("createdAt"))
        if updated is None:
            return None

        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=record.source_url,
            timestamp=updated,
            content=Content(title=title, text=text or title),
            metadata={
                "page_id": record.native_id,
                "space_id": page.get("spaceId"),
                "version": version.get("number"),
                "status": page.get("status"),
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
        return {"Authorization": f"Basic {self._basic}", "Accept": "application/json"}

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
            raise TransientError(f"confluence timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"confluence request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "confluence rejected the credential. Atlassian Cloud needs base64(email:api_token) and the account must have space access."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"confluence rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"confluence returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"confluence rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"confluence returned a non-JSON body") from error


def _cursor_from_link(link: Any) -> str | None:
    """Pull the `cursor` query parameter out of the `next` link.

    Confluence v2 returns a relative URL rather than a bare token, so the token
    has to be extracted. Parsed properly rather than split on `cursor=`: the
    value is percent-encoded and contains characters that a naive split would
    truncate at.
    """
    if not isinstance(link, str) or not link:
        return None
    from urllib.parse import parse_qs, urlparse

    values = parse_qs(urlparse(link).query).get("cursor")
    return values[0] if values else None


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

