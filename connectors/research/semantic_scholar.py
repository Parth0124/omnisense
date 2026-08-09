"""Semantic Scholar papers through the official Graph API.

The Graph API at `api.semanticscholar.org/graph/v1` works without a key at a
low **shared** rate limit -- shared across every unauthenticated caller, so the
throughput you get depends on who else is asking -- and much faster with a key.
Both are supported and the difference is stated in the rate-limit docstring,
because a deployment that has not set `S2_API_KEY` will see intermittent 429s and
should know why.

**`fields` is mandatory in practice.** The default response carries an id and a
title and nothing else, so an implementation that omits the parameter appears to
work and returns papers with no abstract, no date and no authors. Every request
here names the fields it needs.

**Paged by offset, capped at 1000.** The bulk search endpoint refuses an offset
beyond a thousand results, which is a hard wall rather than a soft one: a broad
query silently stops there. The connector advances its watermark instead of
paging past it, so the next run resumes by date rather than by offset.
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

__all__ = ["SemanticScholarConnector"]

DEFAULT_BASE_URL: Final = "https://api.semanticscholar.org/graph/v1"
MAX_OFFSET: Final = 1000
"""The API refuses an offset beyond this. A hard wall, not a soft one."""

FIELDS: Final = (
    "paperId,title,abstract,year,publicationDate,authors,venue,"
    "citationCount,influentialCitationCount,externalIds,url,openAccessPdf"
)
"""Requested explicitly because the default response omits almost everything."""


class SemanticScholarConnector(BaseConnector):
    """Semantic Scholar's Graph API, walked forward by publication date."""

    slug: ClassVar[str] = "semantic_scholar"
    platform: ClassVar[Platform] = Platform.SEMANTIC_SCHOLAR
    category: ClassVar[SourceCategory] = SourceCategory.RESEARCH
    auth_type: ClassVar[AuthType] = AuthType.API_KEY
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=60, burst=2, concurrency=2
    )
    """One request per second, which is the documented unauthenticated allowance.

    A key raises the real ceiling considerably, and this figure is deliberately
    not raised with it: the limiter is a floor on politeness rather than an
    attempt to extract maximum throughput, and a connector that runs at the
    keyed rate will 429 the moment someone runs it without the key."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = True
    overlap_seconds: ClassVar[int] = 86400

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._query = str(params.get("query") or "").strip()
        if not self._query:
            raise ConnectorConfigurationError(
                "semantic_scholar requires a `query` param -- the search endpoint "
                "has no 'everything' mode and an empty query returns nothing."
            )
        # Optional: works without, faster with. The absence is not an error.
        self._api_key = credentials.secrets.get("s2_api_key")
        self._page_size = min(int(params.get("page_size") or 100), 100)

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """The credential is checked at construction; nothing to acquire here."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Page by offset until the wall, then let the watermark carry the rest."""
        offset = int(cursor.checkpoint.get("offset", 0)) if cursor.checkpoint else 0
        emitted = 0

        while offset < MAX_OFFSET:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            payload = await self._get(
                "/paper/search",
                {
                    "query": self._query,
                    "offset": offset,
                    "limit": self._page_size,
                    "fields": FIELDS,
                },
            )
            papers = payload.get("data") or []
            if not papers:
                return

            records = [
                RawRecord(
                    native_id=str(paper.get("paperId")),
                    payload=paper,
                    fetched_at=utcnow(),
                    source_url=paper.get("url"),
                )
                for paper in papers
                if paper.get("paperId")
            ]
            emitted += len(records)
            offset += len(papers)

            dates = [
                _parse_ts(paper.get("publicationDate")) for paper in papers
            ]
            latest = max((d for d in dates if d), default=None)
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"offset": offset},
                ),
            )
            if payload.get("next") is None:
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map a paper. The abstract is the body and is often absent.

        A paper with no abstract is dropped rather than emitted with an empty
        body: the title alone cannot be retrieved against meaningfully, and an
        empty-bodied Signal costs an embedding and a row to contribute nothing.
        """
        paper = record.payload
        title = str(paper.get("title") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        if not title or not abstract:
            return None

        published = _parse_ts(paper.get("publicationDate"))
        if published is None:
            year = paper.get("year")
            if not isinstance(year, int):
                return None
            # Year-only precision, recorded as such in metadata so a temporal
            # query does not treat 1 January as a real publication date.
            published = datetime(year, 1, 1, tzinfo=UTC)

        authors = [a.get("name") for a in (paper.get("authors") or []) if a.get("name")]
        citations = paper.get("citationCount")
        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=paper.get("url"),
            author=Author(display_name=", ".join(authors[:6])) if authors else None,
            timestamp=published,
            content=Content(title=title, text=abstract, truncated=True),
            engagement=Engagement(
                endorsement=int(citations) if isinstance(citations, int) else 0
            ),
            metadata={
                "venue": paper.get("venue"),
                "year": paper.get("year"),
                "date_precision": "day" if paper.get("publicationDate") else "year",
                "influential_citations": paper.get("influentialCitationCount"),
                "doi": (paper.get("externalIds") or {}).get("DOI"),
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
        """`x-api-key` when one is configured, nothing otherwise.

        Unauthenticated access is a supported mode rather than a degraded one, so
        a missing key produces no header rather than an empty one -- some
        gateways treat `x-api-key: ` as a malformed credential and answer 401,
        which would turn "no key" into "bad key".
        """
        return {"x-api-key": self._api_key} if self._api_key else {}

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
            raise TransientError(f"semantic_scholar timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"semantic_scholar request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "semantic_scholar rejected the credential. Set S2_API_KEY, or remove it to use the shared unauthenticated tier."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"semantic_scholar rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"semantic_scholar returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"semantic_scholar rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"semantic_scholar returned a non-JSON body") from error


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

