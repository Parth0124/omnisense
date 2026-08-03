"""OAuth2 token acquisition: client-credentials and refresh-token grants.

Two flows, one entry point. `OAuth2Client.headers()` is the only method a
connector calls, and it returns headers carrying a token that is valid *now* --
refreshing first if it will not be valid for much longer.

Three decisions dominate this module.

**Refresh happens before expiry, never on the 401.** A reactive refresh looks
cheaper and is a self-inflicted outage: every worker holding a copy of the same
token discovers its expiry within the same few milliseconds, and all of them
POST the token endpoint at once. Token endpoints are the most aggressively
rate-limited part of an API precisely because they are cheap to abuse, so the
stampede answers 429, every worker treats that as a quota failure, and the
account stops syncing entirely -- for a token that had not actually gone bad.
Refreshing `DEFAULT_REFRESH_MARGIN_SECONDS` early turns a synchronised cliff into
a spread of independent, uncontended refreshes (`docs/connector-spec.md` §8.2).

**Concurrent refreshers are single-flighted.** The margin spreads refreshes
across *time*; the lock collapses the ones that still coincide. `token()` reads
the store, takes `store.lock(account_id)`, then reads the store *again* --
whoever lost the race finds the winner's token already written and returns it
without issuing a request at all. The double read is the entire mechanism, and
dropping it is invisible in a single-threaded test.

**A failed refresh is terminal.** No retry, no backoff, not even once. That is
partly `docs/connector-spec.md` §1 (a connector never retries internally), but
mostly that a rejected grant is not going to be accepted the second time: a
revoked refresh token stays revoked, and a client retrying rejected credentials
in a loop earns an application-level ban rather than an account-level one. The
run raises `AuthError`, the account is flagged `needs_reauth`, and a human
re-links it.

The module talks to `httpx` and to a `TokenStore`, and to nothing else -- no
Redis, no database, no config. That is what lets the whole flow be tested against
`respx` with an in-memory store (`docs/architecture.md` §6.2 rule 2).
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, NoReturn
from urllib.parse import quote, urlsplit

import httpx

from models.base import utcnow
from connectors.auth.apikey import BasicAuth
from connectors.auth.token_store import (
    DEFAULT_REFRESH_MARGIN_SECONDS,
    REDACTED,
    StoredToken,
    TokenStore,
)
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.protocol import Credentials

__all__ = [
    "ClientAuthMethod",
    "OAuth2Client",
    "OAuth2Config",
    "OAuth2Grant",
]


class OAuth2Grant(enum.StrEnum):
    """The grants a *server-side, unattended* connector can use.

    Authorization-code is absent on purpose: it needs a browser and a human, so
    it belongs to the API layer that runs the consent redirect. What arrives here
    is its output -- a refresh token -- which is why `REFRESH_TOKEN` exists but
    `AUTHORIZATION_CODE` does not. A connector that could run the code exchange
    would be a connector that could pop a browser inside a worker.
    """

    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"


class ClientAuthMethod(enum.StrEnum):
    """How the client proves its own identity to the token endpoint.

    RFC 6749 §2.3.1 says servers MUST support Basic and MAY support the body
    form, so Basic is the default. The body form exists because a real minority
    of providers only accept that, and discovering which is which is a per-vendor
    fact rather than something to negotiate at runtime.
    """

    BASIC = "client_secret_basic"
    POST = "client_secret_post"


_TERMINAL_ERRORS = frozenset(
    {"invalid_grant", "invalid_client", "unauthorized_client", "access_denied", "invalid_token"}
)
"""RFC 6749 §5.2 codes meaning *this credential is no longer good*.

Terminal, and the account needs a human. Retrying any of these is the loop that
gets an application banned."""

_CONFIGURATION_ERRORS = frozenset(
    {"invalid_scope", "unsupported_grant_type", "invalid_request", "unsupported_response_type"}
)
"""RFC 6749 §5.2 codes meaning *whoever configured this got it wrong*.

Separated from the terminal set because the remedy differs and so should the
alert: `invalid_scope` is not fixed by re-linking the account, and filing it as
`needs_reauth` sends an operator to re-authorise an integration that will fail
again identically."""

_MAX_ERROR_CODE_LENGTH = 64
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})


@dataclass(frozen=True, slots=True)
class OAuth2Config:
    """Everything static about one provider's token endpoint.

    Frozen and separate from `OAuth2Client` so it can be built once from a
    connector's declaration, validated, and reused across runs, while the client
    -- which owns a socket and a cache -- is per-run.
    """

    token_url: str
    client_id: str
    client_secret: str | None = None
    grant: OAuth2Grant = OAuth2Grant.CLIENT_CREDENTIALS
    refresh_token: str | None = None
    """Seed refresh token, from the consent flow the API layer ran.

    Only consulted when the store holds nothing. Once a token has been minted the
    store's copy wins, because a rotating provider has already invalidated this
    one."""

    scope: str | None = None
    audience: str | None = None
    """Auth0-style resource indicator. Omitted when `None`; sending an empty
    `audience` is not the same as sending none, and some servers 400 on it."""

    client_auth: ClientAuthMethod = ClientAuthMethod.BASIC
    refresh_margin_seconds: int = DEFAULT_REFRESH_MARGIN_SECONDS
    timeout_seconds: float = 30.0
    extra_params: Mapping[str, str] = field(default_factory=dict)
    """Non-standard body parameters some providers require, e.g. Reddit's
    `duration=permanent`."""

    def __post_init__(self) -> None:
        if not self.token_url.strip():
            raise ConnectorConfigurationError("token_url is empty")
        parts = urlsplit(self.token_url)
        host = (parts.hostname or "").lower()
        if parts.scheme != "https" and host not in _LOCAL_HOSTS:
            # A client secret sent over plaintext HTTP is a secret disclosed to
            # every hop on the path, and it is disclosed silently -- the request
            # succeeds. Loopback is exempted so a locally mocked provider does
            # not force a certificate on a developer.
            raise ConnectorConfigurationError(
                f"token_url must be https (got {parts.scheme!r} for host {host!r})"
            )
        if not self.client_id.strip():
            raise ConnectorConfigurationError("client_id is empty")
        if self.grant is OAuth2Grant.CLIENT_CREDENTIALS and not self.client_secret:
            raise ConnectorConfigurationError(
                "the client_credentials grant requires a client_secret"
            )
        if self.refresh_margin_seconds < 0:
            raise ConnectorConfigurationError("refresh_margin_seconds must not be negative")

    @classmethod
    def from_credentials(
        cls,
        credentials: Credentials,
        *,
        token_url: str,
        grant: OAuth2Grant = OAuth2Grant.CLIENT_CREDENTIALS,
        **overrides: Any,
    ) -> OAuth2Config:
        """Build from a decrypted `Credentials`.

        Uses `require` for the client id -- its `KeyError` names the account and
        the missing key, which a bare `secrets["client_id"]` does not -- but
        plain `.get()` for the secret and refresh token, because which of those
        two is mandatory depends on the grant and `__post_init__` already knows
        the rule.
        """
        return cls(
            token_url=token_url,
            grant=grant,
            client_id=credentials.require("client_id"),
            client_secret=credentials.secrets.get("client_secret"),
            refresh_token=credentials.secrets.get("refresh_token"),
            **overrides,
        )

    def __repr__(self) -> str:
        return (
            f"OAuth2Config(token_url={self.token_url!r}, client_id={self.client_id!r}, "
            f"grant={self.grant.value!r}, scope={self.scope!r}, "
            f"client_secret={REDACTED}, refresh_token={REDACTED})"
        )

    __str__ = __repr__


class OAuth2Client:
    """Holds a provider access token valid *now*, refreshing pre-emptively.

    Not a `HeaderAuth`: its `headers()` is `async`, and that asymmetry is the
    point. A synchronous accessor would have to serve whatever it last cached,
    which is how a request goes out with a token that expired in between. Making
    the caller await is what forces the freshness check onto the request path.
    """

    def __init__(
        self,
        config: OAuth2Config,
        *,
        account_id: str,
        store: TokenStore,
        client: httpx.AsyncClient | None = None,
        connector: str | None = None,
        user_agent: str = "omnisense/0.1",
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._config = config
        self._account_id = account_id
        self._store = store
        self._connector = connector
        self._user_agent = user_agent
        self._now = now
        # Injection matters beyond testing: a connector that already holds a
        # client for this provider should not open a second connection pool
        # purely to renew a token. Whoever created it also closes it -- see
        # `aclose`.
        self._client = client or httpx.AsyncClient(timeout=config.timeout_seconds)
        self._owns_client = client is None

    # ------------------------------------------------------------- public API --

    async def token(self) -> StoredToken:
        """Return a token good for at least the refresh margin, minting if not.

        The optimistic read outside the lock is what keeps the common case free:
        the overwhelming majority of calls find a valid token and never contend
        on anything.
        """
        cached = await self._store.load(self._account_id)
        if cached is not None and not self._is_stale(cached):
            return cached

        async with self._store.lock(self._account_id):
            # Second read, under the lock. Whoever won the race has already
            # written a fresh token; everyone else must use it. Without this the
            # lock only serialises the stampede instead of collapsing it -- every
            # worker still POSTs, just politely, one after another.
            cached = await self._store.load(self._account_id)
            if cached is not None and not self._is_stale(cached):
                return cached

            minted = await self._mint(cached)
            # One `save` for access token, refresh token and expiry together: a
            # rotating provider has already invalidated the old refresh token, so
            # a crash between two separate writes would leave the account holding
            # a credential nothing can renew (`docs/connector-spec.md` §8.2).
            await self._store.save(self._account_id, minted)
            return minted

    async def headers(self) -> dict[str, str]:
        """`Authorization` for the resource server. The connector's usual entry."""
        current = await self.token()
        return {"Authorization": current.authorization_header()}

    async def invalidate(self) -> None:
        """Mark the cached access token unusable after a resource-server 401.

        Expires it in place rather than deleting it. The refresh token is the
        only thing that can mint a replacement, and `delete()` would discard that
        too -- turning "the provider revoked one access token early" into "this
        account needs a human at the consent screen".

        This does not count attempts. `docs/connector-spec.md` §2.1 allows
        exactly one re-authentication per run and the runtime enforces it;
        counting here as well would put the same policy in two places and let
        them disagree.
        """
        cached = await self._store.load(self._account_id)
        if cached is None:
            return
        await self._store.save(self._account_id, replace(cached, expires_at=self._now()))

    async def aclose(self) -> None:
        """Close the HTTP client, but only if this object opened it.

        Closing an injected client would tear the connection pool out from under
        the connector that lent it.
        """
        if self._owns_client:
            await self._client.aclose()

    def __repr__(self) -> str:
        return (
            f"OAuth2Client(account_id={self._account_id!r}, "
            f"grant={self._config.grant.value!r}, token={REDACTED})"
        )

    __str__ = __repr__

    # -------------------------------------------------------------- internals --

    def _is_stale(self, token: StoredToken) -> bool:
        return token.needs_refresh(
            margin_seconds=self._config.refresh_margin_seconds, now=self._now()
        )

    async def _mint(self, cached: StoredToken | None) -> StoredToken:
        """Run the configured grant once and turn the response into a token."""
        config = self._config
        carried_refresh_token = (cached.refresh_token if cached else None) or config.refresh_token

        data: dict[str, str] = {"grant_type": config.grant.value}
        if config.grant is OAuth2Grant.REFRESH_TOKEN:
            if not carried_refresh_token:
                # Never linked, or the refresh token was lost. Nothing to send,
                # and nothing a retry could produce.
                raise self._auth_error("no refresh token is available for this account")
            data["refresh_token"] = carried_refresh_token
        if config.scope:
            data["scope"] = config.scope
        if config.audience:
            data["audience"] = config.audience
        data.update(config.extra_params)

        payload = await self._post_token(data)
        return self._token_from_payload(payload, carried_refresh_token=carried_refresh_token)

    async def _post_token(self, data: dict[str, str]) -> Mapping[str, Any]:
        """POST the token endpoint and map every failure onto the taxonomy."""
        config = self._config
        headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
            # Explicit rather than left to httpx: some token endpoints reject a
            # request whose Content-Type they did not expect, and httpx's default
            # for `data=` is already this -- stating it makes the wire format a
            # decision rather than a library detail.
            "Content-Type": "application/x-www-form-urlencoded",
        }
        body = dict(data)
        if config.client_auth is ClientAuthMethod.BASIC:
            headers.update(self._client_basic_header())
        else:
            body["client_id"] = config.client_id
            if config.client_secret:
                body["client_secret"] = config.client_secret

        try:
            response = await self._client.post(config.token_url, data=body, headers=headers)
        except httpx.TransportError as exc:
            # Timeouts, connection resets, DNS. Retryable, but not by us: the
            # runtime owns backoff (`docs/connector-spec.md` §1).
            raise TransientError(
                "token endpoint is unreachable",
                connector=self._connector,
                account_id=self._account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            # Malformed URL, unsupported protocol: identical on every retry.
            raise PermanentError(
                "token request could not be issued",
                connector=self._connector,
                account_id=self._account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc

        if response.status_code != httpx.codes.OK:
            self._raise_for_token_error(response)
        payload = self._decode(response)
        if "error" in payload:
            # A non-trivial minority of providers answer a *rejected* grant with
            # 200 and an error body. Letting that through would fail later on
            # "no access_token" as a `PermanentError`, which leaves the account
            # unflagged and quietly failing every run until someone reads a log.
            self._raise_for_token_error(response)
        return payload

    def _client_basic_header(self) -> Mapping[str, str]:
        """RFC 6749 §2.3.1 client authentication.

        The id and secret are form-urlencoded *before* base64, which the RFC
        requires and which almost every hand-rolled implementation skips. It only
        bites when a secret contains `:` or `+` or `%` -- and then it bites as an
        `invalid_client` that looks exactly like a wrong password.
        """
        return BasicAuth(
            username=quote(self._config.client_id, safe=""),
            password=quote(self._config.client_secret or "", safe=""),
        ).headers()

    def _decode(self, response: httpx.Response) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            # An HTML error page from a proxy is the usual cause. The body is not
            # attached to the error: it can echo the request, and the request
            # carries the client secret.
            raise PermanentError(
                "token endpoint returned a body that is not JSON",
                connector=self._connector,
                account_id=self._account_id,
                status_code=response.status_code,
                cause=exc,
            ) from exc
        if not isinstance(payload, dict):
            raise PermanentError(
                "token endpoint returned a JSON value that is not an object",
                connector=self._connector,
                account_id=self._account_id,
                status_code=response.status_code,
            )
        return payload

    def _token_from_payload(
        self, payload: Mapping[str, Any], *, carried_refresh_token: str | None
    ) -> StoredToken:
        """Map an RFC 6749 §5.1 success body onto a `StoredToken`."""
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise PermanentError(
                "token response carried no access_token",
                connector=self._connector,
                account_id=self._account_id,
            )

        now = self._now()
        expires_at: datetime | None = None
        lifetime = _as_int(payload.get("expires_in"))
        if lifetime is not None:
            expires_at = now + timedelta(seconds=lifetime)

        # RFC 6749 §6: a refresh response MAY omit `refresh_token`, and when it
        # does the existing one stays valid. Writing `payload.get(...)` straight
        # into the field therefore erases a working refresh token on every
        # non-rotating provider -- and the account only breaks later, when the
        # access token expires and there is nothing left to renew it with.
        new_refresh_token = payload.get("refresh_token")
        refresh_token = (
            new_refresh_token
            if isinstance(new_refresh_token, str) and new_refresh_token
            else carried_refresh_token
        )

        token_type = payload.get("token_type")
        scope = payload.get("scope")
        # `id_token` is dropped rather than kept in `extra`. It is an OIDC bearer
        # credential describing a *user*, no server-to-server connector has any
        # use for one, and storing a credential we never send is pure liability
        # (`docs/security-and-privacy.md` §6.1, data minimisation).
        known = {
            "access_token",
            "refresh_token",
            "expires_in",
            "token_type",
            "scope",
            "id_token",
        }
        return StoredToken(
            access_token=access_token,
            token_type=token_type if isinstance(token_type, str) and token_type else "Bearer",
            expires_at=expires_at,
            refresh_token=refresh_token,
            scope=scope if isinstance(scope, str) else None,
            obtained_at=now,
            extra={k: v for k, v in payload.items() if k not in known},
        )

    def _raise_for_token_error(self, response: httpx.Response) -> NoReturn:
        """Translate a rejected token request onto the taxonomy. Always raises.

        Reached for any non-200, and also for a 200 whose body carries an
        `error` -- which is why `httpx.codes.OK` appears in the credential band
        below rather than being unreachable.

        The `error` code is the only thing taken from the body. `error_description`
        is free text that providers routinely fill with an echo of the request,
        and the request is a form containing the client secret.
        """
        status = response.status_code
        code = _error_code(response)
        details: dict[str, Any] = {"oauth_error": code} if code else {}

        if status == httpx.codes.TOO_MANY_REQUESTS:
            # A quota wall, not bad credentials. Classifying it as `AuthError`
            # would flag a perfectly good account `needs_reauth` and send an
            # operator to fix nothing.
            raise QuotaError(
                "token endpoint rate-limited the client",
                connector=self._connector,
                account_id=self._account_id,
                status_code=status,
                retry_after_seconds=_retry_after_seconds(response.headers),
                details=details,
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise TransientError(
                "token endpoint failed",
                connector=self._connector,
                account_id=self._account_id,
                status_code=status,
                details=details,
            )
        if code in _CONFIGURATION_ERRORS:
            raise ConnectorConfigurationError(
                "token request was rejected as malformed or out of scope",
                connector=self._connector,
                account_id=self._account_id,
                status_code=status,
                details=details,
            )
        if code in _TERMINAL_ERRORS or status in (
            httpx.codes.OK,
            httpx.codes.BAD_REQUEST,
            httpx.codes.UNAUTHORIZED,
            httpx.codes.FORBIDDEN,
        ):
            # The code is checked *as well as* the status band because the status
            # a provider attaches to `invalid_grant` is not reliable -- 400, 401
            # and 403 are all in the wild, and a gateway in front of the token
            # endpoint can rewrite it to something else entirely. Everything else
            # in the 400/401/403 band is treated as a credential problem too,
            # including an unrecognised or absent `error`. Defaulting to terminal
            # is the safe direction: a false `needs_reauth` costs a human glance,
            # a false "retry this" costs an application-level ban.
            raise self._auth_error(
                "token request was rejected", status_code=status, details=details
            )
        raise PermanentError(
            "token endpoint returned an unexpected status",
            connector=self._connector,
            account_id=self._account_id,
            status_code=status,
            details=details,
        )

    def _auth_error(
        self,
        message: str,
        *,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> AuthError:
        """Build the terminal error, always attributed to this account.

        `account_id` on the exception is what lets the runtime flag the right row
        `needs_reauth`; an `AuthError` without it halts the run and tells nobody
        which account to fix.
        """
        return AuthError(
            message,
            connector=self._connector,
            account_id=self._account_id,
            status_code=status_code,
            details=details or {},
        )


# --------------------------------------------------------------------------- #
# Response parsing helpers
# --------------------------------------------------------------------------- #


def _as_int(value: Any) -> int | None:
    """Coerce `expires_in`, which arrives as a number from some providers and a
    string from others. A `TypeError` here would fail a run over a token that is
    perfectly usable, so an unparseable lifetime degrades to "unknown expiry"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _error_code(response: httpx.Response) -> str | None:
    """Extract the RFC 6749 §5.2 `error` code, and nothing else.

    Length-capped and character-restricted before it is allowed anywhere near a
    log line. The field is defined as a small enum of `%x20-21 / %x23-5B / %x5D-7E`
    tokens, but a provider under load can put anything there -- including, on at
    least one large platform, the offending request.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    code = payload.get("error")
    if not isinstance(code, str):
        return None
    code = code.strip()[:_MAX_ERROR_CODE_LENGTH]
    return code if code.replace("_", "").replace("-", "").isalnum() else None


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """`Retry-After` in either RFC 9110 form.

    Duplicated from the private helper in `connectors/base.py` rather than
    imported: reaching into another module's underscore name to share nine lines
    is worse than the nine lines. Both belong in
    `connectors/ratelimit/backoff.py`, which is still a stub.
    """
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return max(0.0, (when - utcnow()).total_seconds())
