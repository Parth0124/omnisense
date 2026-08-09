"""Slack channel history through the Web API.

**The bot sees only channels it has been invited to.** That is not a
permission to grant once — someone has to add the app to each channel, and a
channel nobody added it to returns `not_in_channel` rather than an empty history.
The connector translates that specific error into a message saying so, because
the generic form ("Slack rejected the request") sends an operator to check
scopes that are already correct.

**Threads are fetched separately and are usually where the content is.** A
channel's top-level history is often a one-line question with the actual
discussion in the replies. `conversations.history` does not return them, so
`conversations.replies` is called per threaded message — which is why the rate
limit here is set against Slack's tiered budget rather than a round number.

Slack answers an overrun with `Retry-After` and expects it to be honoured; the
tier for `conversations.history` is roughly 50 requests per minute.
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

__all__ = ["SlackConnector"]

DEFAULT_BASE_URL: Final = "https://slack.com/api"
MAX_PAGE_SIZE: Final = 200


class SlackConnector(BaseConnector):
    """Slack channel history plus thread replies, oldest-first."""

    slug: ClassVar[str] = "slack"
    platform: ClassVar[Platform] = Platform.SLACK
    category: ClassVar[SourceCategory] = SourceCategory.ENTERPRISE
    auth_type: ClassVar[AuthType] = AuthType.BEARER
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=40, burst=3, concurrency=1
    )
    """Slack's Tier 3 allowance for `conversations.history` is about 50 per
    minute, and thread fetching multiplies request count against the same
    budget. Forty leaves headroom for the replies calls rather than spending the
    whole tier on top-level history."""

    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = True
    overlap_seconds: ClassVar[int] = 300

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        self._base_url = str(params.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
        self._client: httpx.AsyncClient | None = None
        token = credentials.secrets.get("bot_token")
        if not token:
            raise ConnectorConfigurationError(
                "slack requires `bot_token` (xoxb-...) with channels:history and "
                "channels:read. The bot must also be invited to each channel -- "
                "scopes alone do not grant access."
            )
        self._token = token
        self._channel = str(params.get("channel_id") or "").strip()
        if not self._channel:
            raise ConnectorConfigurationError(
                "slack requires `channel_id`, e.g. 'C0123456789'."
            )
        self._include_threads = bool(params.get("include_threads", True))

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct without I/O. A configuration defect fails before a socket exists."""
        return cls(ctx, credentials)

    async def authenticate(self) -> None:
        """Bot tokens do not expire; validated on first use."""
        return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Walk history forward from the watermark, then fetch threads.

        Slack's `oldest` parameter takes a float epoch and the API is inclusive
        of it, so the watermark is nudged by a microsecond -- otherwise every
        poll re-reads the last message it already emitted. Dedup would collapse
        it, but paying a request per poll to fetch a known duplicate is waste
        the cursor exists to avoid.
        """
        oldest = (
            f"{cursor.watermark.timestamp() + 0.000001:.6f}"
            if cursor.watermark is not None
            else None
        )
        next_cursor: str | None = None
        emitted = 0

        while True:
            if self._ctx.max_records and emitted >= self._ctx.max_records:
                return
            query: dict[str, Any] = {
                "channel": self._channel,
                "limit": MAX_PAGE_SIZE,
                "inclusive": "false",
            }
            if oldest:
                query["oldest"] = oldest
            if next_cursor:
                query["cursor"] = next_cursor

            payload = await self._slack("/conversations.history", query)
            messages = payload.get("messages") or []
            if not messages:
                return

            # Slack returns newest-first; reversed so records ascend.
            ordered = list(reversed(messages))
            if self._include_threads:
                ordered = await self._expand_threads(ordered)

            records = [
                RawRecord(
                    native_id=f"{self._channel}:{message['ts']}",
                    payload={**message, "_channel": self._channel},
                    fetched_at=utcnow(),
                )
                for message in ordered
                if message.get("ts")
            ]
            emitted += len(records)

            latest = max(
                (d for d in (_slack_ts(m.get("ts")) for m in ordered) if d), default=None
            )
            next_cursor = ((payload.get("response_metadata") or {}).get("next_cursor")) or None
            yield FetchPage(
                records=records,
                cursor=Cursor(
                    version=cursor.version,
                    watermark=latest or cursor.watermark,
                    checkpoint={},
                ),
            )
            if not next_cursor:
                return

    async def normalize(self, record: RawRecord) -> Signal | None:
        message = record.payload
        text = str(message.get("text") or "").strip()
        posted = _slack_ts(message.get("ts"))
        if not text or posted is None:
            return None
        # Join/leave notices and other system events carry a subtype and are not
        # content. Ingesting them fills the corpus with "X has joined the
        # channel", which retrieves against nothing and dilutes everything.
        if message.get("subtype"):
            return None

        return Signal(
            id=signal_id(self.platform, record.native_id),
            source=self.category,
            platform=self.platform,
            author=Author(handle=message.get("user")) if message.get("user") else None,
            timestamp=posted,
            content=Content(text=text),
            engagement=Engagement(
                endorsement=sum(
                    int(reaction.get("count") or 0)
                    for reaction in (message.get("reactions") or [])
                ),
                discussion=int(message.get("reply_count") or 0),
            ),
            metadata={
                "channel_id": message.get("_channel"),
                "thread_ts": message.get("thread_ts"),
                "is_reply": bool(message.get("thread_ts"))
                and message.get("thread_ts") != message.get("ts"),
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
            raise TransientError(f"slack timed out: {error}") from error
        except httpx.HTTPError as error:
            raise TransientError(f"slack request failed: {error}") from error

        if response.status_code in (401, 403):
            raise AuthError(
                "slack rejected the credential. Check the xoxb- token's scopes and that the bot is invited to the channel."
            )
        if response.status_code == 429:
            raise QuotaError(
                f"slack rate limit hit; retry-after="
                f"{response.headers.get('retry-after', 'unset')}"
            )
        if response.status_code >= 500:
            raise TransientError(f"slack returned {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"slack rejected the request with {response.status_code}: "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise TransientError(f"slack returned a non-JSON body") from error

    async def _slack(self, path: str, query: Mapping[str, Any]) -> dict[str, Any]:
        """A Slack call, with its unusual error convention handled.

        Slack answers errors with **HTTP 200** and `{"ok": false, "error": ...}`.
        A connector that only checked the status code would treat every failure
        as an empty page and silently collect nothing -- which is the single
        most common Slack integration bug.
        """
        payload = await self._get(path, query)
        if payload.get("ok"):
            return payload

        error = str(payload.get("error") or "unknown")
        if error == "not_in_channel":
            raise PermanentError(
                f"the bot is not a member of {self._channel}. Scopes are not "
                "enough -- invite the app to the channel with /invite."
            )
        if error in ("invalid_auth", "token_revoked", "account_inactive"):
            raise AuthError(f"slack rejected the token: {error}")
        if error == "ratelimited":
            raise QuotaError("slack rate limited the request")
        raise PermanentError(f"slack returned ok=false: {error}")

    async def _expand_threads(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inline thread replies after their parent.

        Where the content usually is: a channel's top-level history is often a
        one-line question, and the discussion lives in the replies. Failures are
        swallowed per thread -- losing one thread's replies is far better than
        losing the whole page over a single deleted parent.
        """
        expanded: list[dict[str, Any]] = []
        for message in messages:
            expanded.append(message)
            if not message.get("thread_ts") or not message.get("reply_count"):
                continue
            try:
                payload = await self._slack(
                    "/conversations.replies",
                    {"channel": self._channel, "ts": message["thread_ts"], "limit": 100},
                )
            except (TransientError, PermanentError, QuotaError):
                continue
            replies = payload.get("messages") or []
            expanded.extend(reply for reply in replies[1:] if reply.get("ts"))
        return expanded


def _slack_ts(value: Any) -> datetime | None:
    """Slack's `ts` is a string epoch with microseconds: "1712345678.000100"."""
    if not value:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (TypeError, ValueError, OSError):
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

