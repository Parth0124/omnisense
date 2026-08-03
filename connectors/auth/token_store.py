"""Where a connector's live token lives between calls -- and who is allowed to decrypt it.

Two facts decide the shape of this module. `connectors/` may not import
`backend/` or `services/` (`docs/architecture.md` §6.2 rule 2, asserted by
`tests/unit/connectors/test_base.py`), and `CREDENTIAL_ENCRYPTION_KEY` lives in
`backend/core/config.py`. So this module defines the *interface* through which a
token is persisted, plus the in-memory implementation the tests use, and it holds
no key material and performs no cryptography at all.

The production store subclasses `TokenStore` in `services/connector_service.py`,
where the Fernet key is available: it encrypts inside `save()` and decrypts
inside `load()`, and `StoredToken.key_version` rides along as opaque metadata so
that subclass can do multi-key-decrypt / single-key-encrypt rotation without this
module ever learning what a key is (`docs/security-and-privacy.md` §4.1). Nothing
here inspects `key_version`; it is round-tripped and logged, never interpreted.

`docs/connector-spec.md` §8.1 says this module "is the only module that
decrypts". Taken literally that is unbuildable -- decrypting needs the key, the
key needs `backend.core.config`, and that import is forbidden here. The
obligation the sentence is actually making is that *exactly one* place decrypts
and that plaintext exists only in memory for the duration of a run; a single
`TokenStore` subclass in `services/` satisfies both.

Three decisions are encoded below.

- **`lock()` is abstract rather than defaulted.** A per-process `asyncio.Lock` is
  the obvious default and it is silently wrong in production: the refresh
  stampede this exists to prevent happens *across* worker replicas, so the real
  lock is `os:auth:refresh:{account_id}` in Redis (`docs/connector-spec.md`
  §8.2). Leaving it abstract makes the author of the production store confront
  that; a default would let them inherit a lock that guards nothing.
- **`save()` takes a whole `StoredToken`.** A rotating refresh token must be
  persisted in the *same* transaction as the access token it arrived with,
  because a crash between the two bricks the account (§8.2 row 2). One call is a
  seam an implementation can make atomic; two setters are not.
- **`StoredToken` renders redacted.** Same reason `Credentials` in
  `connectors/protocol.py` does: a `ConnectorError` carrying one in `details`,
  or a bare f-string in a debug line, would otherwise print the token.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from models.base import utcnow

__all__ = [
    "DEFAULT_REFRESH_MARGIN_SECONDS",
    "REDACTED",
    "InMemoryTokenStore",
    "StoredToken",
    "TokenStore",
]

REDACTED = "<redacted>"
"""What secret material renders as. A visible marker, not an omission.

Dropping the field entirely makes a redacted record indistinguishable from one
that never had a token, which is exactly the distinction someone reading the log
is trying to make (`backend/core/logging.py` makes the same trade)."""

DEFAULT_REFRESH_MARGIN_SECONDS = 300
"""How long before `expires_at` a token is already treated as expired.

Five minutes, from `docs/connector-spec.md` §8.2. The margin has to cover clock
skew between us and the provider plus the duration of the longest request that
might be issued with the token, because the failure it prevents is a request that
was valid when it left and expired in flight.
"""


@dataclass(frozen=True, slots=True)
class StoredToken:
    """One provider access token and everything needed to replace it.

    Frozen because it is shared: several coroutines in a run may hold the same
    instance while one of them refreshes. A mutable token would let a refresh
    rewrite a value another coroutine had already decided to send.
    """

    access_token: str
    token_type: str = "Bearer"
    expires_at: datetime | None = None
    """Absolute, not `expires_in`. A relative lifetime only means anything next
    to the instant it was issued, and that instant is lost the moment the value
    crosses a process boundary or sits in Postgres for an hour."""

    refresh_token: str | None = None
    scope: str | None = None
    obtained_at: datetime = field(default_factory=utcnow)
    key_version: int | None = None
    """Which encryption key the persisted ciphertext was written under.

    Opaque here. It exists so the `services/` subclass can decrypt under several
    keys while encrypting under one, which is what makes key rotation an online
    operation rather than a flag day (`docs/security-and-privacy.md` §4.1)."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Non-secret provider extras -- tenant ids, instance URLs, granted features.

    Salesforce returns the org's API host in its token response and every
    subsequent call needs it, so discarding everything unrecognised would mean a
    second lookup for something we were already handed."""

    def needs_refresh(
        self,
        *,
        margin_seconds: int = DEFAULT_REFRESH_MARGIN_SECONDS,
        now: datetime | None = None,
    ) -> bool:
        """Whether this token should be replaced *before* being used again.

        A token with no `expires_at` never does. Some providers issue
        non-expiring tokens and some simply omit `expires_in`; in both cases the
        only way to learn about expiry is a 401, which the runtime already
        handles by calling `authenticate()` once more. Inventing a lifetime here
        would instead mean discarding a perfectly good token on a fixed timer
        forever.
        """
        if self.expires_at is None:
            return False
        moment = now if now is not None else utcnow()
        return moment + timedelta(seconds=margin_seconds) >= self.expires_at

    def seconds_remaining(self, *, now: datetime | None = None) -> float | None:
        """Seconds of validity left, or `None` for a token that does not expire."""
        if self.expires_at is None:
            return None
        moment = now if now is not None else utcnow()
        return (self.expires_at - moment).total_seconds()

    def authorization_header(self) -> str:
        """The `Authorization` value this token should be sent as.

        `bearer` is normalised to `Bearer`. RFC 6750 makes the scheme
        case-insensitive, most providers return it lower-cased in the token
        response, and a handful of resource servers compare it literally -- so
        echoing back verbatim what we were told is how the same token works
        against one endpoint and 401s against another on the same provider.
        """
        scheme = "Bearer" if self.token_type.lower() == "bearer" else self.token_type
        return f"{scheme} {self.access_token}"

    def to_log_fields(self) -> dict[str, Any]:
        """Structured fields safe to log. No token material, by construction.

        The key names dodge the redaction regex in `backend/core/logging.py` on
        purpose: it matches `token` anywhere in a key, so the obvious names
        `token_type` and `has_refresh_token` would reach the aggregator as
        `***redacted***` and tell nobody anything. These values are not secret
        and are the ones worth reading when a refresh misbehaves.
        """
        return {
            "scheme": self.token_type,
            "scope": self.scope,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "obtained_at": self.obtained_at.isoformat(),
            "refreshable": self.refresh_token is not None,
            "key_version": self.key_version,
        }

    def __repr__(self) -> str:
        expiry = self.expires_at.isoformat() if self.expires_at else "never"
        return (
            f"StoredToken(access_token={REDACTED}, token_type={self.token_type!r}, "
            f"expires_at={expiry}, refreshable={self.refresh_token is not None})"
        )

    __str__ = __repr__


class TokenStore(abc.ABC):
    """The port through which OAuth tokens are persisted and single-flighted.

    Abstract rather than concrete because the two things a production store must
    do -- encrypt at rest, and lock across processes -- both need imports this
    package is not allowed to make. A connector receives an implementation; it
    never constructs one.

    Keyed by `account_id`, the connector-account identity, which is unique across
    connectors (one Reddit account and one Slack account are two rows). That is
    the granularity of the Redis lock named in `docs/connector-spec.md` §8.2,
    `os:auth:refresh:{account_id}`.
    """

    @abc.abstractmethod
    async def load(self, account_id: str) -> StoredToken | None:
        """Return the stored token, or `None` if this account has never had one.

        Returning `None` rather than raising is deliberate: "no token yet" is the
        normal first-run state, and an exception would push every caller into
        wrapping the happy path in a `try`.
        """

    @abc.abstractmethod
    async def save(self, account_id: str, token: StoredToken) -> None:
        """Persist access token, refresh token and expiry **as one unit**.

        Atomicity is the entire point of the signature. With a rotating refresh
        token the provider invalidates the old one the instant it issues the new
        one, so a crash after writing the access token but before the refresh
        token leaves the account holding a credential nothing can renew
        (`docs/connector-spec.md` §8.2).
        """

    @abc.abstractmethod
    async def delete(self, account_id: str) -> None:
        """Forget this account's token entirely.

        For de-authorisation, not for expiry. Expiring a token in place keeps the
        refresh token that can mint its replacement; deleting throws that away
        and sends a human back through the consent screen.
        """

    @abc.abstractmethod
    def lock(self, account_id: str) -> AbstractAsyncContextManager[None]:
        """Serialise token refreshes for one account.

        Returns a context manager rather than being an `async def` so callers
        read as `async with store.lock(account_id):` and cannot forget to
        release.

        The contract is mutual exclusion across *every* concurrent refresher of
        this account, which in production means every worker replica and
        therefore Redis. It is held only around the token request itself: a lock
        that also spanned fetching would serialise the whole run.
        """


class InMemoryTokenStore(TokenStore):
    """Process-local store. For tests, and for a single-process local run.

    Deliberately not the production store. Two limits, both structural rather
    than fixable here:

    - Tokens sit in the heap as plaintext, so nothing is encrypted at rest --
      there is no "at rest" for a dict.
    - `lock()` is an `asyncio.Lock`, which serialises coroutines inside one event
      loop and nothing else. Eight worker replicas hold eight instances and all
      eight refresh, which is precisely the stampede the lock exists to prevent.
      The cross-process lock is Redis and lives in `services/`.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, StoredToken] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def load(self, account_id: str) -> StoredToken | None:
        return self._tokens.get(account_id)

    async def save(self, account_id: str, token: StoredToken) -> None:
        self._tokens[account_id] = token

    async def delete(self, account_id: str) -> None:
        self._tokens.pop(account_id, None)

    @asynccontextmanager
    async def _locked(self, account_id: str) -> AsyncIterator[None]:
        # `setdefault` is safe here only because there is no `await` between the
        # lookup and the insert -- an event loop cannot switch tasks
        # mid-statement. Adding an await above this line would let two tasks
        # create two locks for one account and let both of them "win".
        guard = self._locks.setdefault(account_id, asyncio.Lock())
        async with guard:
            yield

    def lock(self, account_id: str) -> AbstractAsyncContextManager[None]:
        return self._locked(account_id)

    def __repr__(self) -> str:
        return f"InMemoryTokenStore(accounts={len(self._tokens)}, tokens={REDACTED})"

    __str__ = __repr__
