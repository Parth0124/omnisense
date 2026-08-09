"""The connector domain model: a source, its account, and its watermark.

`connectors/` implements sources; `models/orm/connector_account.py` stores their
configuration. This is the shape services pass around -- and the home of the one
invariant that matters most in the ingestion path.

**A watermark may only move forward.** `Cursor.advanced_to` refuses to move it
backwards, because a backwards watermark is the most damaging bug this layer can
have: it silently re-fetches history the connector already emitted, and the
duplicate suppression that catches it lives one layer away. The connector base
class already guards this at run time; having the rule on the value type means
anything constructing a cursor gets it too.

**Credentials are not here.** No field, no optional, no redacted form. They live
encrypted in the account row and are read only by the connector runtime.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Final

from pydantic import Field, model_validator

from models.base import LenientModel, StrictModel, UtcDatetime, utcnow
from models.enums import AuthType, Platform, SourceCategory

__all__ = [
    "MAX_CONSECUTIVE_FAILURES",
    "ConnectorAccount",
    "ConnectorHealth",
    "ConnectorState",
    "Cursor",
    "SyncOutcome",
]

MAX_CONSECUTIVE_FAILURES: Final = 10
"""After this many, a source is broken rather than busy.

Backoff caps out around here anyway; past it the connector should be reported as
needing attention rather than retried on a lengthening interval forever.
"""


class ConnectorState(enum.StrEnum):
    ACTIVE = "active"
    DISABLED = "disabled"
    NEEDS_REAUTH = "needs_reauth"
    """The stored credential no longer works.

    Distinct from `ERROR` because the responses differ completely: one is waited
    out, the other needs a human to re-authorise. They look identical in a log,
    which is why the distinction has to be in the type.
    """

    ERROR = "error"
    TOS_BLOCKED = "tos_blocked"
    """No lawful API for this data. A policy decision, not a failure."""

    @property
    def is_runnable(self) -> bool:
        return self is ConnectorState.ACTIVE


class Cursor(StrictModel):
    """Where a connector resumed from, and where it may resume next."""

    watermark: UtcDatetime | None = None
    page_token: str | None = None
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)

    def advanced_to(self, watermark: datetime | None) -> Cursor:
        """Return a cursor moved forward, never backward.

        The single most important rule in the ingestion path. A watermark that
        moves backwards re-fetches history the connector already emitted; a
        watermark that moves *forward past unemitted records* loses them
        permanently. Silently keeping the later of the two is correct and is why
        this returns a value rather than raising -- a backward move is usually an
        idle poll subtracting an overlap, not a bug worth failing a run over.
        """
        if watermark is None:
            return self
        if self.watermark is not None and watermark <= self.watermark:
            return self
        return self.model_copy(update={"watermark": watermark})


class ConnectorHealth(LenientModel):
    """Recent operational state. `LenientModel` because workers extend it."""

    last_sync_at: UtcDatetime | None = None
    next_sync_at: UtcDatetime | None = None
    consecutive_failures: int = Field(default=0, ge=0)
    last_error: str | None = Field(default=None, max_length=1000)

    @property
    def needs_attention(self) -> bool:
        return self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES

    @property
    def is_backing_off(self) -> bool:
        return self.consecutive_failures > 0


class ConnectorAccount(StrictModel):
    """One configured source for one tenant. Carries no credential."""

    id: str
    tenant_id: str
    connector_slug: str = Field(min_length=1, max_length=64)
    platform: Platform
    category: SourceCategory
    auth_type: AuthType = AuthType.NONE
    display_name: str = ""
    state: ConnectorState = ConnectorState.DISABLED
    enabled: bool = False
    params: dict[str, Any] = Field(default_factory=dict)
    params_hash: str = ""
    sync_interval_seconds: int = Field(default=3600, ge=60)
    health: ConnectorHealth = Field(default_factory=ConnectorHealth)
    cursor: Cursor = Field(default_factory=Cursor)
    created_at: UtcDatetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _no_credentials_in_params(self) -> ConnectorAccount:
        """`params` is stored unencrypted, so a credential in it is a leak.

        Checked on the value type rather than only at the API edge, because the
        scheduler and the worker construct these too -- and the honest mistake
        (putting a token where it seems to belong) does not only happen over HTTP.
        """
        forbidden = {"token", "secret", "password", "api_key", "apikey", "credential",
                     "access_token", "refresh_token", "private_key", "client_secret"}
        for key in self.params:
            lowered = key.strip().casefold()
            if lowered in forbidden or lowered.endswith(("_token", "_secret", "_key")):
                raise ValueError(
                    f"params[{key!r}] looks like a credential. params is stored "
                    "unencrypted; credentials belong in the encrypted column."
                )
        return self

    @property
    def is_runnable(self) -> bool:
        return self.enabled and self.state.is_runnable


class SyncOutcome(StrictModel):
    """What one sync run produced."""

    connector_slug: str
    run_id: str
    started_at: UtcDatetime = Field(default_factory=utcnow)
    duration_ms: int = Field(default=0, ge=0)
    fetched: int = Field(default=0, ge=0)
    emitted: int = Field(default=0, ge=0)
    dropped: int = Field(default=0, ge=0)
    duplicates: int = Field(default=0, ge=0)
    error: str | None = None
    cursor: Cursor | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None

    @property
    def yield_rate(self) -> float:
        """Emitted over fetched.

        The number that catches a normalisation regression. A connector fetching
        five hundred records and emitting four is not "quiet" -- it is broken, and
        the raw emitted count alone looks identical to a genuinely quiet source.
        """
        return self.emitted / self.fetched if self.fetched else 0.0
