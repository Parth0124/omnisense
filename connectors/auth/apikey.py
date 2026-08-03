"""Header-shaped credentials: API key, bearer token, HTTP Basic -- and none.

Everything here is pure and synchronous, because none of these schemes involves a
round trip: an API key is a constant. OAuth2 is deliberately *not* in this file.
Acquiring a bearer token is I/O with its own failure taxonomy, and hiding that
behind the same synchronous `headers()` call is exactly how a request goes out
carrying a token that expired forty seconds ago -- see `connectors/auth/oauth.py`,
whose equivalent method is `async` for that reason alone.

Four rules are enforced by construction rather than by review.

- **The secret never renders.** Every class overrides `__repr__` and `__str__`.
  `Credentials` in `connectors/protocol.py` does the same, for the same reason: a
  `ConnectorError` carrying a strategy in `details`, or a stray f-string in a
  debug line, would otherwise print the key
  (`docs/security-and-privacy.md` §4.2).
- **The secret goes in a header, never a query string.** There is no
  query-parameter strategy here and that is a decision, not an omission. URLs are
  logged by every proxy in the path, land in `Referer` on any redirect, and are
  what `ConnectorError` is explicitly forbidden from carrying. A provider that
  only accepts `?key=` forces the connector to build that URL itself, in one
  place, visibly.
- **Values are validated at construction.** A credential with a trailing newline
  -- the shape you get from `cat key.txt` -- fails deep inside httpx's header
  encoder, and the exception names the header, which then gets logged. Better to
  reject it here where nothing has been sent yet.
- **There is a `NoAuth`.** So a connector can hold a strategy unconditionally
  instead of branching on `is None` at every call site, which is where the branch
  eventually gets forgotten.
"""

from __future__ import annotations

import abc
import base64
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from models.enums import AuthType
from connectors.auth.token_store import REDACTED
from connectors.exceptions import ConnectorConfigurationError
from connectors.protocol import Credentials

__all__ = [
    "ApiKeyAuth",
    "BasicAuth",
    "BearerAuth",
    "HeaderAuth",
    "NoAuth",
]

_FINGERPRINT_LENGTH = 12
"""Hex characters of a credential fingerprint. Enough to distinguish `active`
from `next` during a dual-key rotation, short enough that nobody mistakes it for
the credential itself."""

_ILLEGAL_HEADER_CHARS = frozenset("\r\n\0")
"""Characters that would split one header into two.

Header injection is only a real attack when part of the value is
attacker-controlled, which a credential is not. The practical failure is duller
and far more common: a secret copied out of a file with its trailing newline
attached."""


def _digest(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:_FINGERPRINT_LENGTH]


def _clean_secret(value: str, *, what: str) -> str:
    """Strip surrounding whitespace and reject anything unsendable.

    Stripping rather than rejecting outer whitespace is a considered asymmetry:
    an operator who pasted a key with a trailing newline made a typo, not a
    security decision, and failing their sync over it helps nobody. Embedded
    control characters are different -- they change the meaning of the request,
    so they raise.
    """
    if not value.strip():
        raise ConnectorConfigurationError(f"{what} is empty")
    cleaned = value.strip()
    if _ILLEGAL_HEADER_CHARS & set(cleaned):
        # The message names the field, never the value.
        raise ConnectorConfigurationError(f"{what} contains a control character")
    return cleaned


class HeaderAuth(abc.ABC):
    """A credential that is applied purely by adding request headers.

    `headers()` returns a fresh mapping on every call rather than a cached one.
    The cost is trivial and the alternative invites a caller to mutate the shared
    dict -- adding `Content-Type` to it once, and thereafter sending that
    `Content-Type` on every request the connector makes.
    """

    __slots__ = ()

    auth_type: ClassVar[AuthType]
    """Matches the `auth_type` a connector declares, so the registry can check
    that a connector claiming `AuthType.BEARER` was actually handed one."""

    @abc.abstractmethod
    def headers(self) -> Mapping[str, str]:
        """Headers to attach to every outbound request."""

    def fingerprint(self) -> str:
        """A stable, non-reversing id for *which* credential this is.

        Rotation needs it: `docs/connector-spec.md` §8.2 promotes the `next` API
        key on the first successful call, and afterwards the only way to answer
        "which key did that succeed with" is a fingerprint, because the key
        itself may never be logged.

        Digesting the rendered header is safe here only because API keys and
        bearer tokens are machine-generated and high-entropy. `BasicAuth`
        overrides this; see the note there.
        """
        rendered = "\n".join(f"{name}:{value}" for name, value in sorted(self.headers().items()))
        return _digest(rendered)


@dataclass(frozen=True, slots=True)
class NoAuth(HeaderAuth):
    """No credential. RSS, GDELT, arXiv -- anything open."""

    auth_type: ClassVar[AuthType] = AuthType.NONE

    def headers(self) -> Mapping[str, str]:
        return {}

    def fingerprint(self) -> str:
        return "none"


@dataclass(frozen=True, slots=True)
class ApiKeyAuth(HeaderAuth):
    """A shared secret sent in a provider-named header.

    The header name is a parameter because there is no convention: NewsAPI wants
    `X-Api-Key`, Semantic Scholar wants `x-api-key`, others want `Api-Key` or a
    prefixed `Authorization`. Hard-coding one and letting connectors "adapt"
    means each of them re-implements this class slightly differently.
    """

    auth_type: ClassVar[AuthType] = AuthType.API_KEY

    key: str
    header: str = "X-API-Key"
    prefix: str = ""
    """Scheme-ish prefix some providers require, e.g. `"Token "` or `"ApiKey "`.
    Kept separate from the key so the stored credential is exactly what the
    provider's console showed, and rotating it does not require remembering to
    re-type the prefix."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _clean_secret(self.key, what="api key"))
        header = self.header.strip()
        if not header or _ILLEGAL_HEADER_CHARS & set(header) or ":" in header:
            raise ConnectorConfigurationError(f"invalid API key header name {header!r}")
        object.__setattr__(self, "header", header)

    def headers(self) -> Mapping[str, str]:
        return {self.header: f"{self.prefix}{self.key}"}

    @classmethod
    def from_credentials(
        cls,
        credentials: Credentials,
        *,
        secret_key: str = "api_key",
        header: str = "X-API-Key",
        prefix: str = "",
    ) -> ApiKeyAuth:
        """Build from a decrypted `Credentials`.

        Goes through `Credentials.require`, whose `KeyError` names both the
        missing key and the account. The alternative -- `secrets["api_key"]` --
        raises a bare `KeyError('api_key')` from inside an auth flow, which tells
        an operator running twelve connectors nothing about which one to fix.
        """
        return cls(key=credentials.require(secret_key), header=header, prefix=prefix)

    def __repr__(self) -> str:
        return f"ApiKeyAuth(header={self.header!r}, key={REDACTED})"

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class BearerAuth(HeaderAuth):
    """`Authorization: Bearer <token>` (RFC 6750).

    Holds a token someone else obtained -- a personal access token, a Slack bot
    token, or the output of `connectors/auth/oauth.py`. It has no idea whether
    that token is still valid, which is why an OAuth2 connector asks
    `OAuth2Client` for headers per request rather than constructing one of these
    once at start-up and keeping it.
    """

    auth_type: ClassVar[AuthType] = AuthType.BEARER

    token: str
    scheme: str = "Bearer"

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _clean_secret(self.token, what="bearer token"))
        scheme = self.scheme.strip()
        if not scheme or _ILLEGAL_HEADER_CHARS & set(scheme) or " " in scheme:
            raise ConnectorConfigurationError(f"invalid authorization scheme {scheme!r}")
        object.__setattr__(self, "scheme", scheme)

    def headers(self) -> Mapping[str, str]:
        return {"Authorization": f"{self.scheme} {self.token}"}

    @classmethod
    def from_credentials(
        cls, credentials: Credentials, *, secret_key: str = "access_token", scheme: str = "Bearer"
    ) -> BearerAuth:
        return cls(token=credentials.require(secret_key), scheme=scheme)

    def __repr__(self) -> str:
        return f"BearerAuth(scheme={self.scheme!r}, token={REDACTED})"

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class BasicAuth(HeaderAuth):
    """`Authorization: Basic base64(user:password)` (RFC 7617).

    Still the OAuth2 client-authentication method most providers prefer
    (RFC 6749 §2.3.1), which is why Reddit's app-only token request is a Basic
    request, so this is not a legacy path.
    """

    auth_type: ClassVar[AuthType] = AuthType.BASIC

    username: str
    password: str

    def __post_init__(self) -> None:
        username = _clean_secret(self.username, what="basic auth username")
        if ":" in username:
            # RFC 7617 §2: the colon is the field separator, so a username
            # containing one is unrepresentable. Encoding it anyway produces a
            # credential the server silently splits in the wrong place and
            # rejects as a bad password, which is an hour of debugging the
            # wrong thing.
            raise ConnectorConfigurationError("basic auth username may not contain ':'")
        object.__setattr__(self, "username", username)
        object.__setattr__(
            self, "password", _clean_secret(self.password, what="basic auth password")
        )

    def headers(self) -> Mapping[str, str]:
        # UTF-8, per the RFC 7617 `charset` parameter. The historical default was
        # ISO-8859-1, which does not raise on a non-ASCII password -- it encodes
        # a different one, and the provider answers 401 with no hint as to why.
        raw = f"{self.username}:{self.password}".encode()
        return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}

    def fingerprint(self) -> str:
        """Digest of the *username* only.

        The inherited implementation digests the rendered header, which for Basic
        is `base64(user:password)`. Publishing a digest of a human-chosen
        password into a log aggregator is a dictionary attack waiting to happen,
        not a fingerprint. The username already answers the only question a
        fingerprint is asked -- which credential is this -- so it is the whole
        input here.
        """
        return _digest(self.username)

    @classmethod
    def from_credentials(
        cls,
        credentials: Credentials,
        *,
        username_key: str = "client_id",
        password_key: str = "client_secret",
    ) -> BasicAuth:
        return cls(
            username=credentials.require(username_key),
            password=credentials.require(password_key),
        )

    def __repr__(self) -> str:
        return f"BasicAuth(username={self.username!r}, password={REDACTED})"

    __str__ = __repr__
