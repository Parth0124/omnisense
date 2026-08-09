"""Jira Cloud issues through the REST v3 search endpoint.

**`ORDER BY updated ASC` is not cosmetic.** JQL without an explicit order
returns rows in an unspecified sequence, and Jira paginates by offset — so under
concurrent edits, which is the normal state of a live project, a record can shift
between pages and be skipped or repeated. Ordering by `updated` ascending makes
paging stable and makes `fetch` naturally oldest-first, which is what the
watermark contract needs.

**`updated`, not `created`.** An issue reopened after six months is new signal
even though it was created long ago, and a `created`-ordered crawl would never
see it again after the first pass.

Basic auth with an API token, which is what Atlassian Cloud uses — the header is
`Basic base64(email:token)`, and the email is part of the credential rather than
a separate field.
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

__all__ = ["JiraConnector"]

DEFAULT_BASE_URL: Final = "https://example.atlassian.net/rest/api/3"
MAX_PAGE_SIZE: Final = 100
ISSUE_FIELDS: Final = "summary,description,created,updated,status,reporter,priority,labels,issuetype"


class JiraConnector(BaseConnector):
    """Jira Cloud issues, ordered by update time so paging is stable."""

    slug: ClassVar[str] = "jira"
    platform: ClassVar[Platform] = Platform.JIRA
    category: ClassVar[SourceCategory] = SourceCategory.ENTERPRISE
    auth_type: ClassVar[AuthType] = AuthType.BASIC
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=60, burst=5, concurrency=2
    )
    """Atlassian Cloud rate-limits per user on a cost budget rather than a
    request count, and answers an overrun with a 429 carrying `Retry-After`.
    Sixty a minute is well inside a normal budget; the real protection is
    honouring the header, which `_get` does by raising `QuotaError`."""

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
                "jira requires `site_url`, e.g. 'https://acme.atlassian.net'."
            )
        self._base_url = f"{site}/rest/api/3"
        email = credentials.secrets.get("email")
        token = credentials.secrets.get("api_token")
        if not email or not token:
            raise ConnectorConfigurationError(
                "jira requires `email` and `api_token`. Atlassian Cloud uses "
                "Basic auth with the account email as the username -- a token "
                "alone is not a credential there."
            )
        self._basic = base64.b64encode(f"{email}:{token}".encode()).decode()
        self._jql = str(params.get("jql") or "").strip()
        if not self._jql:
            raise ConnectorConfigurationError(
                "jira requires a `jql` param, e.g. 'project = SUP AND "
                "type = Bug'. The connector appends its own ORDER BY."
            )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """Basic auth is stateless; the header is built at construction."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Page by offset over an explicitly-ordered JQL query."""
        start_at = int(cursor.checkpoint.get("start_at", 0)) if cursor.checkpoint else 0
        emitted = 0

        jql = self._jql
        if cursor.watermark is not None:
            # Jira's JQL date literal is minute-precision, so the bound is
            # deliberately floored -- asking for a second-precision value
            # produces a parse error rather than a narrower window.
            stamp = cursor.watermark.astimezone(UTC).strftime("%Y-%m-%d %H:%M")
            jql = f"({jql}) AND updated >= '{stamp}'"
        jql = f"{jql} ORDER BY updated ASC"

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            payload = await self._get(
                "/search",
                {
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": MAX_PAGE_SIZE,
                    "fields": ISSUE_FIELDS,
                },
            )
            issues = payload.get("issues") or []
            if not issues:
                return

            records = [
                RawRecord(
                    native_id=str(issue["key"]),
                    payload=issue,
                    fetched_at=utcnow(),
                    source_url=f"{self._base_url.rsplit('/rest/', 1)[0]}/browse/{issue['key']}",
                )
                for issue in issues
                if issue.get("key")
            ]
            emitted += len(records)
            start_at += len(issues)

            latest = max(
                (
                    d
                    for d in (
                        _parse_ts((i.get("fields") or {}).get("updated")) for i in issues
                    )
                    if d
                ),
                default=None,
            )
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={"start_at": start_at},
                ),
            )
            if start_at >= int(payload.get("total") or 0):
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Summary plus description. ADF is flattened to text.

        Jira v3 returns the description as Atlassian Document Format -- a nested
        node tree, not a string. Rendering it naively with `str()` produces a
        Python dict repr in the Signal body, which then gets embedded and
        retrieved. `_flatten_adf` walks it properly.
        """
        issue = record.payload
        fields = issue.get("fields") or {}
        summary = str(fields.get("summary") or "").strip()
        if not summary:
            return None

        description = _flatten_adf(fields.get("description"))
        updated = _parse_ts(fields.get("updated")) or _parse_ts(fields.get("created"))
        if updated is None:
            return None

        reporter = fields.get("reporter") or {}
        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            url=record.source_url,
            author=Author(display_name=reporter.get("displayName")) if reporter else None,
            timestamp=updated,
            content=Content(title=summary, text=description or summary),
            metadata={
                "issue_key": record.native_id,
                "status": ((fields.get("status") or {}).get("name")),
                "priority": ((fields.get("priority") or {}).get("name")),
                "issue_type": ((fields.get("issuetype") or {}).get("name")),
                "labels": fields.get("labels") or [],
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
            raise TransientError(f"jira timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"jira request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "jira rejected the credential. Atlassian Cloud needs Basic auth as base64(email:api_token), not a bare token."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"jira rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"jira returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"jira rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"jira returned a non-JSON body") from error


def _flatten_adf(node: Any) -> str:
    """Flatten Atlassian Document Format into plain text.

    Recursive because ADF nests arbitrarily -- a paragraph inside a list item
    inside a panel. Block-level nodes contribute a newline so the flattened text
    keeps its paragraph structure; without that a long description becomes one
    run-on line, which chunks badly and reads worse.
    """
    if node is None:
        return ""
    if isinstance(node, str):
        return node
    if isinstance(node, list):
        return "".join(_flatten_adf(child) for child in node)
    if not isinstance(node, Mapping):
        return ""

    kind = node.get("type")
    if kind == "text":
        return str(node.get("text") or "")
    if kind == "hardBreak":
        return "\n"

    inner = _flatten_adf(node.get("content"))
    block = {"paragraph", "heading", "listItem", "blockquote", "codeBlock", "panel"}
    return f"{inner}\n" if kind in block else inner


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

