"""Amazon product data through the Product Advertising API v5.

**Amazon has no review-text API, and this connector does not scrape one.**
That is the single most important fact about this module. Review text is
available only through the web interface, whose terms of service forbid
automated collection, so what PA-API v5 offers instead is product metadata and a
*ratings summary* -- the star average and the review count, not the reviews.

A market-intelligence product that silently returned an empty review list here
would be worse than one that returns nothing: the caller would conclude the
product has no complaints. So the connector emits the ratings summary as a
Signal, states in `metadata` that review text is unavailable, and
`requires_tos_review` is True.

**PA-API requires SigV4 request signing and an Associates account in good
standing.** The signing is not optional and the account is revoked for
insufficient sales, which is a failure mode no other connector here has -- a
working integration can stop working for commercial reasons.
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

__all__ = ["AmazonConnector"]

DEFAULT_BASE_URL: Final = "https://webservices.amazon.com/paapi5"
MAX_ITEMS_PER_REQUEST: Final = 10
"""PA-API's GetItems ceiling."""


class AmazonConnector(BaseConnector):
    """Amazon PA-API v5 product metadata and ratings summary. No review text."""

    slug: ClassVar[str] = "amazon"
    platform: ClassVar[Platform] = Platform.AMAZON
    category: ClassVar[SourceCategory] = SourceCategory.REVIEWS
    auth_type: ClassVar[AuthType] = AuthType.API_KEY
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=8, burst=1, concurrency=1
    )
    """PA-API starts every account at one request per second and scales the
    allowance with attributed sales -- so a new integration has the tightest
    budget and no way to raise it quickly. Eight a minute stays inside the floor."""

    supports_incremental: ClassVar[bool] = False
    supports_backfill: ClassVar[bool] = False
    overlap_seconds: ClassVar[int] = 3600

    requires_tos_review: ClassVar[bool] = True
    """Review text is unavailable through any lawful API, and scraping it
    violates Amazon's terms. The connector returns a ratings summary and says so
    rather than returning an empty review list that reads as 'no complaints'."""

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        for name in ("access_key", "secret_key", "partner_tag"):
            if not credentials.secrets.get(name):
                raise ConnectorConfigurationError(
                    f"amazon requires `{name}`. PA-API v5 needs an Associates "
                    "account in good standing plus SigV4 signing keys; note that "
                    "an account is revoked for insufficient attributed sales, so "
                    "a working integration can stop working commercially."
                )
        self._access_key = credentials.secrets["access_key"]
        self._secret_key = credentials.secrets["secret_key"]
        self._partner_tag = credentials.secrets["partner_tag"]
        asins = params.get("asins") or []
        if not isinstance(asins, (list, tuple)) or not asins:
            raise ConnectorConfigurationError(
                "amazon requires an `asins` param listing the products to track. "
                "There is no search-by-keyword mode here: PA-API's SearchItems "
                "returns catalogue results, not review signal."
            )
        self._asins = [str(a) for a in asins][:100]

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """Credentials are validated at construction. SigV4 signs per request."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """One GetItems batch per ten ASINs.

        Not incremental: PA-API has no change feed and no modified-since filter,
        so every run re-reads the configured products in full. The watermark is
        still advanced so downstream dedup collapses the re-read -- a ratings
        summary that has not moved produces the same content hash and is
        suppressed before it reaches the pipeline.
        """
        for start in range(0, len(self._asins), MAX_ITEMS_PER_REQUEST):
            batch = self._asins[start : start + MAX_ITEMS_PER_REQUEST]
            payload = await self._get_items(batch)
            items = ((payload.get("ItemsResult") or {}).get("Items")) or []
            if not items:
                continue

            fetched = utcnow()
            records = [
                RawRecord(
                    native_id=str(item.get("ASIN")),
                    payload=item,
                    fetched_at=fetched,
                    source_url=(item.get("DetailPageURL")),
                )
                for item in items
                if item.get("ASIN")
            ]
            yield FetchPage(
                records=records,
                cursor=Cursor(version=cursor.version, watermark=fetched),
            )

    async def normalize(self, record: RawRecord) -> Signal | None:
        """A ratings summary as a Signal, with the limitation stated in metadata.

        `Content.text` carries the summary in words rather than being left empty,
        because an empty body is invisible to retrieval -- and the fact that a
        product has 4.1 stars over 12,000 ratings is genuine signal even without
        the review text.
        """
        item = record.payload
        info = (item.get("ItemInfo") or {})
        title = (((info.get("Title") or {}).get("DisplayValue")) or "").strip()
        if not title:
            raise NormalizationError(
                f"amazon item {record.native_id} has no title; PA-API always "
                "returns one for a valid ASIN"
            )

        reviews = ((item.get("CustomerReviews") or {}))
        star = (reviews.get("StarRating") or {}).get("Value")
        count = (reviews.get("Count") or {}).get("Value")

        summary = (
            f"{title} holds {star} stars across {count} ratings."
            if star is not None and count is not None
            else f"{title}: no ratings summary available."
        )
        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=record.source_url,
            timestamp=record.fetched_at,
            content=Content(title=title, text=summary, truncated=True),
            engagement=Engagement(endorsement=int(count) if isinstance(count, int) else 0),
            metadata={
                "asin": record.native_id,
                "star_rating": star,
                "rating_count": count,
                "review_text_available": False,
                "review_text_note": (
                    "Amazon exposes no review-text API. This Signal carries the "
                    "ratings summary only."
                ),
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
        # PA-API authenticates by signing each request rather than with a static
        # header, so nothing is attached at client construction.
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
            raise TransientError(f"amazon timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"amazon request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "amazon rejected the credential. Check the SigV4 keys and that the Associates account is still active."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"amazon rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"amazon returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"amazon rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"amazon returned a non-JSON body") from error

    async def _get_items(self, asins: list[str]) -> dict[str, Any]:
        """A signed GetItems call.

        SigV4 signing is delegated to `botocore` rather than hand-rolled. A
        hand-written signer is roughly eighty lines of canonicalisation that
        fails on exactly one input shape -- a query parameter needing a
        particular escape -- and the failure is a 403 that looks like a bad
        credential.
        """
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials as AwsCredentials
        except ImportError as error:  # pragma: no cover
            raise ConnectorConfigurationError(
                "amazon needs `botocore` for PA-API SigV4 signing. Add it to "
                "requirements.txt, or disable this connector."
            ) from error

        import json as _json

        body = _json.dumps(
            {
                "ItemIds": asins,
                "Resources": [
                    "ItemInfo.Title",
                    "CustomerReviews.StarRating",
                    "CustomerReviews.Count",
                    "Offers.Listings.Price",
                ],
                "PartnerTag": self._partner_tag,
                "PartnerType": "Associates",
                "Marketplace": "www.amazon.com",
            }
        )
        request = AWSRequest(
            method="POST",
            url=f"{self._base_url}/getitems",
            data=body,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Amz-Target": "com.amazon.paapi5.v1.ProductAdvertisingAPIv1.GetItems",
            },
        )
        SigV4Auth(
            AwsCredentials(self._access_key, self._secret_key),
            "ProductAdvertisingAPI",
            "us-east-1",
        ).add_auth(request)

        client = self._ensure_client()
        try:
            response = await client.post(
                f"{self._base_url}/getitems",
                content=body,
                headers=dict(request.headers),
            )
        except httpx.HTTPError as error:
            raise TransientError(f"amazon request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "amazon rejected the signature or the Associates account. PA-API "
                "revokes access for insufficient attributed sales."
            )
        if response.status_code == 429:
            raise QuotaError("amazon PA-API throttled the request")
        if response.status_code >= 500:
            raise TransientError(f"amazon returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(f"amazon rejected the request: {response.text[:200]}")
        return response.json()


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

