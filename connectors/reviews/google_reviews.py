"""Google reviews through the Places API, capped at five per place.

**Five reviews. That is the entire ceiling, and it is the most important
thing about this connector.** Place Details returns at most five reviews per
place, selected by Google, with no pagination and no way to ask for more. There
is no other lawful API for Google review content.

A market-intelligence system that sampled five reviews and presented the result
as coverage would be actively misleading -- five reviews chosen by someone else's
relevance ranking is not a sample, it is a highlight reel. So every Signal this
connector emits carries `sample_of_total` in metadata, and the count is available
from the same response, which is what lets a downstream consumer say "5 of 1,284
reviews" rather than implying it saw them all.

`requires_tos_review` is True for that reason rather than a licensing one.
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

__all__ = ["GoogleReviewsConnector"]

DEFAULT_BASE_URL: Final = "https://places.googleapis.com/v1"
MAX_REVIEWS_PER_PLACE: Final = 5
"""Google's hard ceiling. Not a page size -- there is no second page."""


class GoogleReviewsConnector(BaseConnector):
    """Google Places API reviews. At most five per place, chosen by Google."""

    slug: ClassVar[str] = "google_reviews"
    platform: ClassVar[Platform] = Platform.GOOGLE_REVIEWS
    category: ClassVar[SourceCategory] = SourceCategory.REVIEWS
    auth_type: ClassVar[AuthType] = AuthType.API_KEY
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=60, burst=3, concurrency=2
    )
    """Places API bills per request and enforces a per-project QPS. Sixty a
    minute is comfortably inside the default and, more to the point, this
    connector cannot usefully make many requests: five reviews per place is the
    whole payload."""

    supports_incremental: ClassVar[bool] = False
    supports_backfill: ClassVar[bool] = False
    overlap_seconds: ClassVar[int] = 3600

    requires_tos_review: ClassVar[bool] = True
    """Five reviews per place, selected by Google, with no pagination.
    Presenting that as coverage would mislead; every Signal states the sample
    size against the true total."""

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        key = credentials.secrets.get("api_key")
        if not key:
            raise ConnectorConfigurationError(
                "google_reviews requires a Google Places API key with the Places "
                "API (New) enabled. Note the endpoint returns at most five "
                "reviews per place and cannot be paginated."
            )
        self._api_key = key
        places = params.get("place_ids") or []
        if not isinstance(places, (list, tuple)) or not places:
            raise ConnectorConfigurationError(
                "google_reviews requires `place_ids`. Resolve them once with the "
                "Places text-search endpoint; they are stable."
            )
        self._place_ids = [str(p) for p in places][:200]

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """The key is checked at construction and sent as a header per request."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """One request per place. No pagination exists to do."""
        for place_id in self._place_ids:
            payload = await self._get(f"/places/{place_id}", {})
            reviews = payload.get("reviews") or []
            if not reviews:
                continue

            total = payload.get("userRatingCount")
            place_name = ((payload.get("displayName") or {}).get("text"))
            fetched = utcnow()
            records = [
                RawRecord(
                    native_id=str(review.get("name") or f"{place_id}:{index}"),
                    payload={
                        **review,
                        "_place_id": place_id,
                        "_place_name": place_name,
                        "_total_reviews": total,
                    },
                    fetched_at=fetched,
                )
                for index, review in enumerate(reviews[:MAX_REVIEWS_PER_PLACE])
            ]
            yield FetchPage(
                records=records,
                cursor=Cursor(version=cursor.version, watermark=fetched),
            )

    async def normalize(self, record: RawRecord) -> Signal | None:
        review = record.payload
        text = str(((review.get("text") or {}).get("text")) or "").strip()
        if not text:
            return None
        posted = _parse_ts(review.get("publishTime"))
        if posted is None:
            return None

        author = (review.get("authorAttribution") or {})
        total = review.get("_total_reviews")
        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=review.get("googleMapsUri"),
            author=Author(display_name=author.get("displayName")) if author else None,
            timestamp=posted,
            content=Content(text=text),
            metadata={
                "place_id": review.get("_place_id"),
                "place_name": review.get("_place_name"),
                "star_rating": review.get("rating"),
                # The honesty field. Without it a consumer counting five reviews
                # has no way to know it is looking at five of twelve hundred.
                "sample_of_total": total,
                "sampling_note": (
                    "Google Places returns at most five reviews per place, chosen "
                    "by Google. This is not a representative sample."
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
        return {
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": "id,displayName,rating,userRatingCount,reviews",
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
            raise TransientError(f"google_reviews timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"google_reviews request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "google_reviews rejected the credential. Enable the Places API (New) on the project and check key restrictions."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"google_reviews rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"google_reviews returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"google_reviews rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"google_reviews returned a non-JSON body") from error


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

