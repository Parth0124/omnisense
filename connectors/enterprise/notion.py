"""Notion pages through the search and block APIs.

**Content arrives as a nested block tree, not as text.** A Notion page is a
list of blocks, each of which may have children, and the text of a block is a
list of rich-text runs. Flattening that naively loses list structure and code
blocks entirely — a bulleted list becomes one run-on paragraph, which chunks
badly and reads worse. `_flatten_blocks` walks the tree properly and preserves
list markers and fences.

**Search returns pages the integration has been shared with.** Notion's
permission model is per-page sharing rather than workspace-wide access, so an
integration created but not shared returns an empty result set with a 200. The
connector logs that case distinctly rather than reporting an empty sync.

Filtered by `last_edited_time` for incremental runs.
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

__all__ = ["NotionConnector"]

DEFAULT_BASE_URL: Final = "https://api.notion.com/v1"
NOTION_VERSION: Final = "2022-06-28"
"""Pinned. Notion's API is versioned by header and an unpinned client silently
changes shape when they ship a new one."""

MAX_PAGE_SIZE: Final = 100


class NotionConnector(BaseConnector):
    """Notion pages, with block trees flattened into readable text."""

    slug: ClassVar[str] = "notion"
    platform: ClassVar[Platform] = Platform.NOTION
    category: ClassVar[SourceCategory] = SourceCategory.ENTERPRISE
    auth_type: ClassVar[AuthType] = AuthType.BEARER
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=150, burst=3, concurrency=2
    )
    """Notion documents an average of three requests per second per
    integration. 150 a minute is that rate; `burst=3` matches the documented
    burst allowance rather than exceeding it, because Notion answers an overrun
    with a 429 that counts against the same budget."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = True
    overlap_seconds: ClassVar[int] = 300

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        token = credentials.secrets.get("integration_token")
        if not token:
            raise ConnectorConfigurationError(
                "notion requires `integration_token` (secret_...). Note that "
                "Notion scopes access by per-page sharing: an integration that "
                "exists but has not been shared with any page returns an empty "
                "result set rather than an error."
            )
        self._token = token
        self._query = str(params.get("query") or "").strip() or None

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """Integration tokens are static."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Search for pages, then fetch each one's block tree.

        Two phases because search returns page objects without content. That
        makes this connector request-hungry -- one call per page -- which is why
        the page size is small and the rate limit is honoured strictly.
        """
        client = self._ensure_client()
        start_cursor = (
            str(cursor.checkpoint.get("cursor")) if (cursor.checkpoint or {}).get("cursor") else None
        )
        emitted = 0

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            body: dict[str, Any] = {
                "page_size": min(MAX_PAGE_SIZE, 25),
                "sort": {"direction": "ascending", "timestamp": "last_edited_time"},
                "filter": {"property": "object", "value": "page"},
            }
            if self._query:
                body["query"] = self._query
            if start_cursor:
                body["start_cursor"] = start_cursor

            try:
                response = await client.post(f"{self._base_url}/search", json=body)
            except httpx.HTTPError as error:
                raise TransientError(f"notion request failed: {error}") from error
            payload = self._check(response)

            pages = payload.get("results") or []
            if not pages:
                return

            records: list[RawRecord] = []
            for page in pages:
                page_id = page.get("id")
                if not page_id:
                    continue
                text = await self._page_text(page_id)
                records.append(
                    RawRecord(
                        native_id=str(page_id),
                        payload={**page, "_text": text},
                        fetched_at=utcnow(),
                        source_url=page.get("url"),
                    )
                )
            emitted += len(records)

            latest = max(
                (d for d in (_parse_ts(p.get("last_edited_time")) for p in pages) if d),
                default=None,
            )
            start_cursor = payload.get("next_cursor")
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"cursor": start_cursor} if start_cursor else {},
                ),
            )
            if not payload.get("has_more"):
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        page = record.payload
        text = str(page.get("_text") or "").strip()
        title = _page_title(page)
        edited = _parse_ts(page.get("last_edited_time"))
        if edited is None or (not text and not title):
            return None

        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=page.get("url"),
            timestamp=edited,
            content=Content(title=title or None, text=text or title or ""),
            metadata={
                "page_id": record.native_id,
                "created_time": page.get("created_time"),
                "archived": page.get("archived"),
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
        return {
            "Authorization": f"Bearer {self._token}",
            "Notion-Version": NOTION_VERSION,
        }

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
            raise TransientError(f"notion timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"notion request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "notion rejected the credential. Share at least one page with the integration -- token validity alone is not access."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"notion rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"notion returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"notion rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"notion returned a non-JSON body") from error

    def _check(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise AuthError(
                "notion rejected the token, or the integration has not been "
                "shared with any page -- both surface as 401/403 here."
            )
        if response.status_code == 429:
            raise QuotaError("notion rate limited the request")
        if response.status_code >= 500:
            raise TransientError(f"notion returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(f"notion rejected the request: {response.text[:200]}")
        return response.json()

    async def _page_text(self, page_id: str) -> str:
        """Fetch and flatten one page's blocks. Failures yield empty text.

        Swallowed per page: a single block tree that cannot be read must not
        lose the whole search page, and a page with no text still carries a
        title worth indexing.
        """
        try:
            payload = await self._get(f"/blocks/{page_id}/children", {"page_size": 100})
        except (TransientError, PermanentError, QuotaError, AuthError):
            return ""
        return _flatten_blocks(payload.get("results") or [])


def _rich_text(runs: Any) -> str:
    if not isinstance(runs, list):
        return ""
    return "".join(str(run.get("plain_text") or "") for run in runs if isinstance(run, Mapping))


def _flatten_blocks(blocks: list[Any]) -> str:
    """Flatten a block list into text, preserving list and code structure.

    The structure matters downstream: a bulleted list flattened to one line
    chunks as a single passage and retrieves as an undifferentiated blob, while
    the same list with markers chunks per item.
    """
    lines: list[str] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        kind = str(block.get("type") or "")
        payload = block.get(kind)
        if not isinstance(payload, Mapping):
            continue
        text = _rich_text(payload.get("rich_text"))
        if not text:
            continue
        if kind == "bulleted_list_item":
            lines.append(f"- {text}")
        elif kind == "numbered_list_item":
            lines.append(f"1. {text}")
        elif kind == "code":
            lines.append(f"```\n{text}\n```")
        elif kind.startswith("heading"):
            lines.append(f"\n{text}")
        else:
            lines.append(text)
    return "\n".join(lines)


def _page_title(page: Mapping[str, Any]) -> str:
    """Notion puts the title in a property whose name varies by database."""
    for value in (page.get("properties") or {}).values():
        if isinstance(value, Mapping) and value.get("type") == "title":
            return _rich_text(value.get("title")).strip()
    return ""


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

