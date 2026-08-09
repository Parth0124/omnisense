"""HubSpot CRM objects through the v3 search API.

**The search endpoint stops at 10,000 results, and it stops silently.**
Paging past that offset returns an error rather than more rows, so a naive
implementation collects ten thousand records and reports success. The way around
it is not deeper paging — it is to sort ascending by `hs_lastmodifieddate` and
advance the watermark to the last record seen, then start a fresh search from
there. That is what `fetch` does, and it is the whole reason this connector
re-queries rather than paging straight through.

**A private-app token, not an API key.** HubSpot retired API keys; a legacy key
now fails with a 401 that says nothing about why. The construction error names
the replacement.

Objects are configurable — `tickets` for support signal, `notes` for sales
conversations — because which one carries market intelligence differs by how the
CRM is used.
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

__all__ = ["HubSpotConnector"]

DEFAULT_BASE_URL: Final = "https://api.hubapi.com"
SEARCH_RESULT_CEILING: Final = 10_000
"""HubSpot refuses an offset beyond this. Worked around by re-querying from the
watermark rather than paging deeper."""

MAX_PAGE_SIZE: Final = 100


class HubSpotConnector(BaseConnector):
    """HubSpot CRM search, walked forward by last-modified date."""

    slug: ClassVar[str] = "hubspot"
    platform: ClassVar[Platform] = Platform.HUBSPOT
    category: ClassVar[SourceCategory] = SourceCategory.ENTERPRISE
    auth_type: ClassVar[AuthType] = AuthType.BEARER
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=100, burst=5, concurrency=2
    )
    """HubSpot allows roughly 100-190 requests per 10 seconds depending on
    tier, and separately caps search at 4 requests per second per token. The
    search cap is the binding one here, and 100 a minute stays inside it with
    room for the burst."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = True
    overlap_seconds: ClassVar[int] = 300

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        token = credentials.secrets.get("access_token")
        if not token:
            raise ConnectorConfigurationError(
                "hubspot requires a private-app `access_token`. HubSpot retired "
                "API keys -- a legacy hapikey now fails with a bare 401. Create "
                "a private app and grant it the crm.objects.* read scopes."
            )
        self._token = token
        self._object = str(params.get("object_type") or "tickets").strip()
        self._properties = list(
            params.get("properties")
            or ["subject", "content", "hs_lastmodifieddate", "createdate", "hs_pipeline_stage"]
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """Private-app tokens are static and long-lived."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Search ascending by modified date, re-querying past the 10k ceiling."""
        client = self._ensure_client()
        watermark = cursor.watermark
        emitted = 0
        after: str | None = None
        seen_in_query = 0

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return

            body: dict[str, Any] = {
                "limit": MAX_PAGE_SIZE,
                "properties": self._properties,
                "sorts": [{"propertyName": "hs_lastmodifieddate", "direction": "ASCENDING"}],
            }
            if watermark is not None:
                body["filterGroups"] = [
                    {
                        "filters": [
                            {
                                "propertyName": "hs_lastmodifieddate",
                                "operator": "GT",
                                "value": str(int(watermark.timestamp() * 1000)),
                            }
                        ]
                    }
                ]
            if after:
                body["after"] = after

            try:
                response = await client.post(
                    f"{self._base_url}/crm/v3/objects/{self._object}/search", json=body
                )
            except httpx.HTTPError as error:
                raise TransientError(f"hubspot request failed: {error}") from error
            payload = self._check(response)

            results = payload.get("results") or []
            if not results:
                return

            records = [
                RawRecord(
                    native_id=str(item["id"]),
                    payload=item,
                    fetched_at=utcnow(),
                )
                for item in results
                if item.get("id")
            ]
            emitted += len(records)
            seen_in_query += len(results)

            latest = max(
                (
                    d
                    for d in (
                        _parse_ts((r.get("properties") or {}).get("hs_lastmodifieddate"))
                        for r in results
                    )
                    if d
                ),
                default=None,
            )
            if latest is not None:
                watermark = latest

            after = ((payload.get("paging") or {}).get("next") or {}).get("after")
            yield FetchPage(
                records=records,
                cursor=Cursor(version=cursor.version, watermark=watermark, checkpoint={}),
            )

            if not after:
                return
            # The ceiling. Rather than paging into an error, drop the offset and
            # start a fresh search from the advanced watermark -- which is why
            # ascending order and a modified-date filter are both required.
            if seen_in_query >= SEARCH_RESULT_CEILING - MAX_PAGE_SIZE:
                after = None
                seen_in_query = 0

    async def normalize(self, record: RawRecord) -> Signal | None:
        item = record.payload
        properties = item.get("properties") or {}
        subject = str(properties.get("subject") or "").strip()
        content = str(properties.get("content") or "").strip()
        if not subject and not content:
            return None

        modified = _parse_ts(properties.get("hs_lastmodifieddate")) or _parse_ts(
            properties.get("createdate")
        )
        if modified is None:
            return None

        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            timestamp=modified,
            content=Content(title=subject or None, text=content or subject),
            metadata={
                "object_type": self._object,
                "object_id": record.native_id,
                "pipeline_stage": properties.get("hs_pipeline_stage"),
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
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

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
            raise TransientError(f"hubspot timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"hubspot request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "hubspot rejected the credential. Use a private-app token with crm.objects read scopes; API keys are retired."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"hubspot rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"hubspot returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"hubspot rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"hubspot returned a non-JSON body") from error

    def _check(self, response: httpx.Response) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise AuthError(
                "hubspot rejected the token. Private apps need the "
                "crm.objects.<type>.read scope granted explicitly -- a token "
                "without it fails identically to an invalid one."
            )
        if response.status_code == 429:
            raise QuotaError(
                "hubspot rate limited the request; the search endpoint is capped "
                "at four requests per second regardless of tier"
            )
        if response.status_code >= 500:
            raise TransientError(f"hubspot returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(f"hubspot rejected the request: {response.text[:200]}")
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

