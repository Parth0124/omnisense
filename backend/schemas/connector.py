"""Wire shapes for `/api/v1/connectors` (`docs/api-reference.md` §4.5).

This is the most security-sensitive schema in the API, and the sensitivity is
almost entirely about what is *absent*.

**No credential field exists in either direction.** Not on the way out, obviously
-- but also not on the way in, on a response model, or in a `params` passthrough.
`docs/security-and-privacy.md` §8.2: credentials are written through the
credential endpoint into an encrypted column and are never read back, not even
redacted. A redacted credential in a response is still a confirmation that one
exists and a hint at its shape, and the field would eventually be populated by
someone who thought redaction made it safe.

**No endpoint or base-URL field, either.** A connector is addressed by slug, from
a closed registry. A client-supplied URL would turn this endpoint into a
server-side request forgery primitive: authenticated, tenant-scoped, and pointed
at whatever the caller names -- including cloud metadata services.

**`requires_tos_review` is published.** Several platforms have no lawful
third-party API for the data this system would want. A connector that refuses to
collect for that reason must be visibly distinguishable from one that is merely
broken, or an operator will spend a day debugging a refusal that is working as
intended.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Final

from pydantic import Field, field_validator

from backend.schemas.common import RequestModel, ResponseModel

__all__ = [
    "MAX_PARAM_KEYS",
    "ConnectorDetail",
    "ConnectorHealth",
    "ConnectorItem",
    "ConnectorStatusName",
    "SyncRequest",
    "SyncAccepted",
]

MAX_PARAM_KEYS: Final = 20
"""Cap on per-account connector parameters.

`params` is a genuine passthrough -- a subreddit list, a feed URL for RSS, a JQL
string -- so it cannot be a closed schema. The cap and the value-length limit are
what stop it being used as arbitrary storage, and its contents are hashed into
`params_hash` to key the cursor, so an unbounded dict would also mean an
unbounded hash input on every sync.
"""

MAX_PARAM_VALUE_CHARS: Final = 2000


class ConnectorStatusName(enum.StrEnum):
    """Operational state of one configured account."""

    ACTIVE = "active"
    DISABLED = "disabled"
    NEEDS_REAUTH = "needs_reauth"
    """The stored credential no longer works. Distinct from `error`.

    A failing sync and an expired OAuth token look identical in a log and need
    completely different responses -- one is waited out, the other needs a human
    to re-authorise. Collapsing them means every token expiry is investigated as
    an outage.
    """

    ERROR = "error"
    TOS_BLOCKED = "tos_blocked"
    """Refused because the platform has no lawful API for this data.

    Not a failure. Published as its own state so an operator does not debug a
    deliberate refusal.
    """


class ConnectorHealth(ResponseModel):
    """Recent sync outcome. What an operator looks at first."""

    last_sync_at: datetime | None = None
    next_sync_at: datetime | None = None
    consecutive_failures: int = 0
    last_error: str | None = Field(
        default=None,
        max_length=1000,
        description=(
            "The most recent failure, truncated. Never contains a credential -- "
            "connector errors are scrubbed before storage, because a 401 body from "
            "a provider routinely echoes the token that failed."
        ),
    )

    @property
    def is_backing_off(self) -> bool:
        return self.consecutive_failures > 0


class ConnectorItem(ResponseModel):
    """One connector as the catalogue lists it.

    Describes the *capability*, not a configured account: this is what the
    deployment could collect from, which is the question `GET /connectors`
    answers.
    """

    slug: str
    platform: str
    category: str
    enabled: bool = Field(
        description="Whether this deployment has turned the connector on."
    )
    configured: bool = Field(
        default=False,
        description=(
            "Whether an account with credentials exists. Distinct from `enabled`: "
            "a connector can be enabled with nothing configured, which is the most "
            "common reason a source silently returns nothing."
        ),
    )
    requires_tos_review: bool = False
    supports_incremental: bool = True
    supports_backfill: bool = False
    auth_type: str = "none"
    version: str = "0.1.0"


class ConnectorDetail(ConnectorItem):
    """One connector with its configured account's state."""

    status: ConnectorStatusName = ConnectorStatusName.DISABLED
    health: ConnectorHealth = Field(default_factory=ConnectorHealth)
    sync_interval_seconds: int | None = None
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Non-secret per-account configuration. Credentials are never here and "
            "never returned in any form -- see the module docstring."
        ),
    )
    rate_limit_per_minute: int | None = None


class SyncRequest(RequestModel):
    """The `POST /connectors/{slug}/sync` body.

    Note what cannot be requested: a URL, an endpoint, a credential, or an
    unbounded record count. The connector is named by the path slug and resolved
    from the registry; everything here only *narrows* what that connector would
    have done anyway.
    """

    max_records: int | None = Field(
        default=None,
        ge=1,
        le=50_000,
        description=(
            "Ceiling for this run. Bounded because a sync consumes a third-party "
            "rate limit shared with every other investigation in the deployment."
        ),
    )
    max_pages: int | None = Field(default=None, ge=1, le=1000)
    params: dict[str, Any] = Field(default_factory=dict, max_length=MAX_PARAM_KEYS)
    reset_cursor: bool = Field(
        default=False,
        description=(
            "Discard the stored watermark and re-fetch from the beginning. "
            "Destructive to incremental state and expensive: it re-reads history "
            "the connector has already seen, and dedup is what stops that becoming "
            "duplicate signals."
        ),
    )

    @field_validator("params")
    @classmethod
    def _bounded_params(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Bound the passthrough, and refuse anything that smells like a credential.

        The key check is not security theatre. `params` is stored unencrypted and
        returned by `GET /connectors/{slug}` -- so a caller who puts a token in it,
        reasonably assuming the system would protect it, has just written a
        credential into a readable column. Refusing the obvious names catches the
        honest mistake, which is the one that actually happens.
        """
        forbidden = {"token", "secret", "password", "api_key", "apikey", "credential",
                     "access_token", "refresh_token", "private_key", "client_secret"}
        for key, item in value.items():
            lowered = key.strip().casefold()
            if lowered in forbidden or lowered.endswith("_token") or lowered.endswith("_secret"):
                raise ValueError(
                    f"params[{key!r}] looks like a credential. params is stored "
                    "unencrypted and returned by GET /connectors; use the credential "
                    "endpoint instead."
                )
            if isinstance(item, str) and len(item) > MAX_PARAM_VALUE_CHARS:
                raise ValueError(
                    f"params[{key!r}] exceeds {MAX_PARAM_VALUE_CHARS} characters"
                )
        return value


class SyncAccepted(ResponseModel):
    """The `202` body: the sync was queued, not completed."""

    slug: str
    run_id: str
    accepted_at: datetime
    message: str = Field(
        default="Sync queued. Progress is reported through connector health.",
        description=(
            "202 rather than 200 because a sync takes minutes. A synchronous "
            "response would hold the connection through a paginated third-party "
            "crawl and die on the first proxy idle timeout."
        ),
    )
