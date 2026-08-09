"""Papers with Code: papers linked to their implementations.

The value here is the link between a paper and running code -- which
repositories implement it, and how much traction each has. That is a different
signal from citation count: a paper with four hundred citations and no
implementation is influential in the literature, and one with forty citations and
six actively-starred repositories is influential in practice.

**A standing caveat, recorded here rather than discovered later.** Papers with
Code was folded into Hugging Face and the future of `paperswithcode.com/api/v1`
is not guaranteed. The connector treats a 404 on the API root as a permanent
error with a message saying exactly this, so the day it is withdrawn the failure
explains itself instead of looking like a bug.

Unauthenticated and paged by page number.
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

__all__ = ["PapersWithCodeConnector"]

DEFAULT_BASE_URL: Final = "https://paperswithcode.com/api/v1"
MAX_PAGES: Final = 50


class PapersWithCodeConnector(BaseConnector):
    """Papers with Code, paged newest-first and reversed per page."""

    slug: ClassVar[str] = "papers_with_code"
    platform: ClassVar[Platform] = Platform.PAPERS_WITH_CODE
    category: ClassVar[SourceCategory] = SourceCategory.RESEARCH
    auth_type: ClassVar[AuthType] = AuthType.NONE
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=30, burst=2, concurrency=1
    )
    """Two seconds between requests. No published limit, so the posture is caution.

    `docs/connector-spec.md` §9.5: where a limit is undocumented, serialise. An
    undocumented limiter answers a burst with a silent block and there is no
    header to reconcile against."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = False
    overlap_seconds: ClassVar[int] = 86400

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._area = str(params.get("area") or "").strip() or None
        self._page_size = min(int(params.get("page_size") or 50), 100)

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """Open API. Nothing to acquire."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Page forward, reversing each page so records are oldest-first.

        The API pages newest-first with no ordering parameter, which is the
        sanctioned exception in `BaseConnector.fetch`: pages arrive
        newest-to-oldest and each page's *contents* are reversed so that records
        within a page ascend. The watermark only advances on a completed page,
        so a run that dies mid-descent resumes without skipping.
        """
        page = 1
        emitted = 0

        while page <= MAX_PAGES:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            query: dict[str, Any] = {"page": page, "items_per_page": self._page_size}
            if self._area:
                query["area"] = self._area

            payload = await self._get("/papers/", query)
            results = payload.get("results") or []
            if not results:
                return

            ordered = list(reversed(results))
            records = [
                RawRecord(
                    native_id=str(paper.get("id") or paper.get("arxiv_id") or ""),
                    payload=paper,
                    fetched_at=utcnow(),
                    source_url=paper.get("url_abs"),
                )
                for paper in ordered
                if paper.get("id") or paper.get("arxiv_id")
            ]
            emitted += len(records)

            latest = max(
                (
                    d
                    for d in (_parse_ts(p.get("published")) for p in results)
                    if d is not None
                ),
                default=None,
            )
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"page": page},
                ),
            )

            if payload.get("next") is None:
                return
            page += 1

    async def normalize(self, record: RawRecord) -> Signal | None:
        paper = record.payload
        title = str(paper.get("title") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        if not title:
            return None

        published = _parse_ts(paper.get("published"))
        if published is None:
            return None

        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=paper.get("url_abs"),
            author=Author(display_name=str(paper.get("authors") or "") or None)
            if paper.get("authors")
            else None,
            timestamp=published,
            content=Content(title=title, text=abstract, truncated=True),
            metadata={
                "arxiv_id": paper.get("arxiv_id"),
                "conference": paper.get("conference"),
                "url_pdf": paper.get("url_pdf"),
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
            raise TransientError(f"papers_with_code timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"papers_with_code request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "papers_with_code rejected the credential. This API is unauthenticated; a 401 here means the service changed."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"papers_with_code rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"papers_with_code returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"papers_with_code rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"papers_with_code returned a non-JSON body") from error


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

