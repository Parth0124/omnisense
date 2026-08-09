"""Google Play reviews through the Play Developer API.

**This returns reviews only for apps you own.** The `androidpublisher` API is
a *developer* API: it authenticates as the publisher and scopes every read to
that publisher's packages. A competitor's reviews are not reachable through it,
and there is no other lawful bulk surface.

That is a real limitation for a market-intelligence product and it is stated
here, at construction and in the catalogue, rather than being discovered as a
403 on the first competitor package someone configures.

**Reviews are retained for 7 days by this endpoint.** `reviews.list` returns only
recent reviews regardless of pagination, so a poll gap longer than a week loses
data permanently -- which is why `overlap_seconds` is generous and why the
scheduler interval for this connector should stay well under a day.
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

__all__ = ["PlayStoreConnector"]

DEFAULT_BASE_URL: Final = "https://androidpublisher.googleapis.com/androidpublisher/v3"
REVIEW_RETENTION_DAYS: Final = 7
"""`reviews.list` returns nothing older. A poll gap beyond this loses data."""


class PlayStoreConnector(BaseConnector):
    """Google Play Developer API reviews, for packages you publish."""

    slug: ClassVar[str] = "play_store"
    platform: ClassVar[Platform] = Platform.PLAY_STORE
    category: ClassVar[SourceCategory] = SourceCategory.REVIEWS
    auth_type: ClassVar[AuthType] = AuthType.OAUTH2
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=30, burst=2, concurrency=1
    )
    """Play Developer API quotas are per-project and generous relative to this
    workload -- one app's reviews is a handful of requests. Thirty a minute is
    politeness rather than a constraint."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = False
    overlap_seconds: ClassVar[int] = 7200

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        token = credentials.secrets.get("access_token")
        if not token:
            raise ConnectorConfigurationError(
                "play_store requires a Google service-account access token with "
                "the androidpublisher scope. Note this API returns reviews only "
                "for packages the authenticated account publishes -- competitor "
                "reviews are not reachable through it."
            )
        self._token = token
        self._package = str(params.get("package_name") or "").strip()
        if not self._package:
            raise ConnectorConfigurationError(
                "play_store requires `package_name`, e.g. 'com.example.app'. It "
                "must be a package this account publishes."
            )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """The service-account token is supplied and refreshed upstream."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Page reviews with the API's token, reversing each page.

        `reviews.list` returns newest-first with no ordering parameter -- the
        sanctioned exception in `BaseConnector.fetch`. Each page's contents are
        reversed so records ascend within a page.
        """
        token: str | None = (
            str(cursor.checkpoint.get("token")) if cursor.checkpoint.get("token") else None
        ) if cursor.checkpoint else None
        emitted = 0

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            query: dict[str, Any] = {"maxResults": 100}
            if token:
                query["token"] = token

            payload = await self._get(f"/applications/{self._package}/reviews", query)
            reviews = payload.get("reviews") or []
            if not reviews:
                return

            records = [
                RawRecord(
                    native_id=str(review["reviewId"]),
                    payload=review,
                    fetched_at=utcnow(),
                )
                for review in reversed(reviews)
                if review.get("reviewId")
            ]
            emitted += len(records)

            latest = max(
                (d for d in (_review_time(r) for r in reviews) if d), default=None
            )
            token = ((payload.get("tokenPagination") or {}).get("nextPageToken"))
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"token": token} if token else {},
                ),
            )
            if not token:
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        """The most recent comment on a review becomes the Signal body.

        A Play review is a list of `comments` -- the user's text plus any
        developer reply. Only the user comment is taken: a developer reply is our
        own content, and ingesting it would let a product's own support responses
        register as customer sentiment about itself.
        """
        review = record.payload
        comments = review.get("comments") or []
        user_comment = next(
            (c.get("userComment") for c in comments if c.get("userComment")), None
        )
        if not user_comment:
            return None

        text = str(user_comment.get("text") or "").strip()
        if not text:
            return None
        posted = _review_time(review)
        if posted is None:
            return None

        rating = user_comment.get("starRating")
        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            author=Author(display_name=review.get("authorName")) if review.get("authorName") else None,
            timestamp=posted,
            content=Content(text=text),
            engagement=Engagement(
                endorsement=int(user_comment.get("thumbsUpCount") or 0)
            ),
            metadata={
                "package_name": self._package,
                "star_rating": rating,
                "app_version": (user_comment.get("appVersionName")),
                "device": user_comment.get("device"),
                "reviewer_language": user_comment.get("reviewerLanguage"),
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
            raise TransientError(f"play_store timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"play_store request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "play_store rejected the credential. The service account needs the androidpublisher scope and access to this package."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"play_store rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"play_store returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"play_store rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"play_store returned a non-JSON body") from error


def _review_time(review: Mapping[str, Any]) -> datetime | None:
    """The user comment's timestamp, from Google's seconds/nanos shape."""
    for comment in review.get("comments") or []:
        user = comment.get("userComment")
        if not user:
            continue
        stamp = (user.get("lastModified") or {}).get("seconds")
        if stamp is not None:
            try:
                return datetime.fromtimestamp(int(stamp), tz=UTC)
            except (TypeError, ValueError, OSError):
                return None
    return None


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

