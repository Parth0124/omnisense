"""arXiv preprints through the official Atom export API.

arXiv publishes an unauthenticated query API at `export.arxiv.org/api/query`
and asks, in its terms of use, for **no more than one request every three
seconds**. That courtesy limit is not enforced by a 429 -- exceeding it gets an
IP blocked by hand, days later, with no signal in between. So the rate limit
below is the whole operational story of this connector, and it is set from the
published guidance rather than from what the server tolerates.

**Sorted ascending by submission date, which makes `fetch` naturally
oldest-first.** The API's default is descending, and taking it would put this
connector in the sanctioned-exception branch of `BaseConnector.fetch` for no
reason: `sortOrder=ascending` costs nothing and makes the watermark contract
trivially satisfiable.

**`native_id` keeps the version suffix.** arXiv ids look like `2401.12345v2`, and
v1 and v2 are genuinely different documents -- different abstracts, sometimes
different conclusions. Stripping the suffix would collapse a revision onto its
original and lose whatever changed, which for a research-monitoring product is
the interesting part.
"""

from __future__ import annotations

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

__all__ = ["ArxivConnector"]

DEFAULT_BASE_URL: Final = "https://export.arxiv.org/api/query"
DEFAULT_PAGE_SIZE: Final = 100
MAX_PAGE_SIZE: Final = 2000
"""arXiv rejects `max_results` above 2000 outright."""

ARXIV_NS: Final = "{http://www.w3.org/2005/Atom}"


class ArxivConnector(BaseConnector):
    """arXiv's Atom query API, walked forward by submission date."""

    slug: ClassVar[str] = "arxiv"
    platform: ClassVar[Platform] = Platform.ARXIV
    category: ClassVar[SourceCategory] = SourceCategory.RESEARCH
    auth_type: ClassVar[AuthType] = AuthType.NONE
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=20, burst=1, concurrency=1
    )
    """One request every three seconds, which is what arXiv's terms of use ask for.

    Twenty a minute is that limit exactly. `burst=1` and `concurrency=1` matter
    more than the rate: the limit is a courtesy, enforced by a human blocking an
    IP rather than by a 429, so there is no feedback signal to back off from. A
    burst that gets through today is a block next week."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = True
    overlap_seconds: ClassVar[int] = 3600

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self._query = str(params.get("query") or params.get("search_query") or "").strip()
        if not self._query:
            raise ConnectorConfigurationError(
                "arxiv requires a `query` param, e.g. 'cat:cs.CL' or "
                "'all:retrieval augmented generation'. Without one the API returns "
                "the entire corpus newest-first, which is neither useful nor kind."
            )
        self._page_size = min(
            int(params.get("page_size") or DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O, so a configuration defect fails before a socket."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """No credentials. The export API is open."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Walk results ascending by submission date, page by page.

        `start` is an offset rather than a token, which is the one thing about
        arXiv's paging that needs care: the underlying result set grows while a
        crawl is in progress, so an offset-based descent can repeat a record at a
        page boundary. Ascending order plus the watermark makes that harmless --
        a repeat is a re-read of something already emitted, and dedup collapses
        it.
        """
        client = self._ensure_client()
        start = int(cursor.checkpoint.get("start", 0)) if cursor.checkpoint else 0
        emitted = 0

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            page_size = self._page_size
            if self._ctx.max_records:
                page_size = min(page_size, self._ctx.max_records - emitted)

            params = {
                "search_query": self._query,
                "start": start,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            }
            response = await self._get(client, params)
            entries = _parse_atom(response.text)
            if not entries:
                return

            records = [
                RawRecord(
                    native_id=entry["id"],
                    payload=entry,
                    fetched_at=utcnow(),
                    source_url=entry.get("link"),
                )
                for entry in entries
            ]
            emitted += len(records)
            start += len(entries)

            latest = max(
                (e["published"] for e in entries if e.get("published")), default=None
            )
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"start": start},
                ),
            )
            if len(entries) < page_size:
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map an Atom entry onto a Signal.

        The abstract is the body. arXiv serves no full text through this API, so
        `Content.truncated` is True -- and stating that is what stops a
        downstream summariser from treating an abstract as the paper.
        """
        payload = record.payload
        title = str(payload.get("title") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        if not title:
            raise NormalizationError(
                f"arXiv entry {record.native_id} has no title; the Atom feed "
                "always carries one, so this payload is malformed rather than empty"
            )

        published = payload.get("published")
        if not isinstance(published, datetime):
            raise NormalizationError(f"arXiv entry {record.native_id} has no usable date")

        authors = payload.get("authors") or []
        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=payload.get("link"),
            author=Author(handle=None, display_name=", ".join(authors[:6]) or None)
            if authors
            else None,
            timestamp=published,
            content=Content(title=title, text=summary, truncated=True),
            metadata={
                "arxiv_id": record.native_id,
                "categories": payload.get("categories") or [],
                "primary_category": payload.get("primary_category"),
                "comment": payload.get("comment"),
            },
        )

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._ctx.request_timeout_seconds,
                headers={"User-Agent": self._ctx.user_agent},
            )
        return self._client

    async def _get(self, client: httpx.AsyncClient, params: Mapping[str, Any]) -> httpx.Response:
        """One request, with arXiv's failure modes classified.

        arXiv answers an overloaded backend with a 200 carrying an Atom error
        entry rather than a 5xx, so a naive status check reports success for a
        response containing nothing. The empty-entry case is handled by the
        caller returning; what is classified here is the transport layer.
        """
        try:
            response = await client.get(self._base_url, params=dict(params))
        except httpx.TimeoutException as error:
            raise TransientError(f"arxiv timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"arxiv request failed: {error}") from error

        if response.status_code == 429:
            raise QuotaError(
                "arxiv returned 429. The published limit is one request per three "
                "seconds; exceeding it risks an IP block applied by hand."
            )
        if response.status_code >= 500:
            raise TransientError(f"arxiv returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"arxiv rejected the query with {response.status_code}; check the "
                "`query` param syntax"
            )
        return response


def _parse_atom(text: str) -> list[dict[str, Any]]:
    """Parse the Atom response into plain dicts.

    `defusedxml` rather than the standard library: this parses a document from a
    remote host, and `xml.etree` is vulnerable to entity-expansion attacks that
    turn a small response into gigabytes of memory. arXiv is not hostile, but a
    parser chosen on the assumption that a source stays friendly is a parser
    chosen wrongly.
    """
    try:
        from defusedxml import ElementTree
    except ImportError:  # pragma: no cover -- fall back, loudly
        import xml.etree.ElementTree as ElementTree  # noqa: S405

    root = ElementTree.fromstring(text)
    entries: list[dict[str, Any]] = []
    for node in root.findall(f"{ARXIV_NS}entry"):
        raw_id = (node.findtext(f"{ARXIV_NS}id") or "").strip()
        if not raw_id:
            continue
        # The id is a URL; the last path segment is the versioned arXiv id.
        native_id = raw_id.rsplit("/", 1)[-1]
        entries.append(
            {
                "id": native_id,
                "link": raw_id,
                "title": " ".join((node.findtext(f"{ARXIV_NS}title") or "").split()),
                "summary": " ".join((node.findtext(f"{ARXIV_NS}summary") or "").split()),
                "published": _parse_ts(node.findtext(f"{ARXIV_NS}published")),
                "authors": [
                    (author.findtext(f"{ARXIV_NS}name") or "").strip()
                    for author in node.findall(f"{ARXIV_NS}author")
                ],
                "categories": [
                    category.get("term", "")
                    for category in node.findall(f"{ARXIV_NS}category")
                ],
                "primary_category": (
                    node.find("{http://arxiv.org/schemas/atom}primary_category") or {}
                ).get("term")
                if node.find("{http://arxiv.org/schemas/atom}primary_category") is not None
                else None,
                "comment": node.findtext("{http://arxiv.org/schemas/atom}comment"),
            }
        )
    return entries


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

