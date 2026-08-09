"""CORS configuration, and the one thing it must never be.

`backend/main.py` installs Starlette's `CORSMiddleware` directly. This module
exists to own the *policy* -- what origins, what methods, what headers -- so that
the rule below is stated once, in a place a reviewer can find, rather than being
an argument list buried in an application factory.

**`allow_origins` is never `*`, and never `*` with credentials.** The second is
not merely bad practice: browsers reject it outright, so a deployment configured
that way fails every cross-origin request with a message about credentials rather
than about the wildcard, and the hours go into debugging the wrong thing.
`backend/core/config.py` already refuses to start a staging or production process
with a wildcard; this module refuses to *build* the configuration, so the same
rule holds for anything that assembles middleware without going through Settings
validation.

**Credentials are allowed, which is what forces exact origins.** The frontend
authenticates with a bearer token from the same browser session, so
`allow_credentials=True` is required. That is precisely the setting that makes a
wildcard both illegal and dangerous -- with credentials, a permissive origin
policy means any site the user visits can make authenticated calls to this API
using their session.

**Preflight caching is set deliberately.** Without `max_age`, a browser preflights
every non-simple request, doubling the round trips on an API where nearly every
call carries `Authorization`. Ten minutes is long enough to eliminate that and
short enough that a policy change propagates within a deploy cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import FastAPI

    from backend.core.config import Settings

__all__ = [
    "ALLOWED_HEADERS",
    "ALLOWED_METHODS",
    "EXPOSED_HEADERS",
    "PREFLIGHT_MAX_AGE_SECONDS",
    "CorsPolicy",
    "build_cors_policy",
    "install_cors",
]

logger = get_logger(__name__)

ALLOWED_METHODS: Final[tuple[str, ...]] = ("GET", "POST", "PATCH", "DELETE", "OPTIONS")
"""No `PUT`, because the API has no idempotent whole-resource replacement, and no
`HEAD`, which Starlette derives from `GET`. Listing methods the API does not
implement advertises a surface that does not exist."""

ALLOWED_HEADERS: Final[tuple[str, ...]] = (
    "Authorization",
    "Content-Type",
    "Idempotency-Key",
    "X-Request-ID",
    "Last-Event-ID",
)
"""An explicit list rather than `*`.

`*` is legal here even with credentials in modern browsers, and it is still the
wrong answer: the list is documentation of what the API actually reads, and a
request carrying a header nobody declared is a client doing something the server
does not expect. `Last-Event-ID` is present for SSE reconnection.
"""

EXPOSED_HEADERS: Final[tuple[str, ...]] = ("X-Request-ID", "Retry-After")
"""Headers JavaScript may read off a cross-origin response.

Not exposing `X-Request-ID` is a small omission with a large cost: it is the
correlation id a user would quote in a bug report, and without this the browser
can see it in devtools but the application cannot read it to display.
`Retry-After` matters for the same practical reason -- a client that cannot read
it cannot honour a 429 correctly.
"""

PREFLIGHT_MAX_AGE_SECONDS: Final = 600


@dataclass(frozen=True, slots=True)
class CorsPolicy:
    """The resolved policy. Frozen so nothing mutates it after validation."""

    allow_origins: tuple[str, ...]
    allow_credentials: bool = True
    allow_methods: tuple[str, ...] = ALLOWED_METHODS
    allow_headers: tuple[str, ...] = ALLOWED_HEADERS
    expose_headers: tuple[str, ...] = EXPOSED_HEADERS
    max_age: int = PREFLIGHT_MAX_AGE_SECONDS

    def as_middleware_kwargs(self) -> dict[str, Any]:
        return {
            "allow_origins": list(self.allow_origins),
            "allow_credentials": self.allow_credentials,
            "allow_methods": list(self.allow_methods),
            "allow_headers": list(self.allow_headers),
            "expose_headers": list(self.expose_headers),
            "max_age": self.max_age,
        }


def build_cors_policy(settings: Settings | None = None) -> CorsPolicy:
    """Resolve and validate the policy from settings.

    Raises rather than falling back to something permissive. A CORS
    misconfiguration that fails closed costs one clear startup error; one that
    fails open costs a cross-origin credential leak that nothing reports.
    """
    from backend.core.config import get_settings

    resolved = settings or get_settings()
    origins = tuple(resolved.app.cors_origin_list)

    if "*" in origins:
        raise ConfigurationError(
            "CORS_ORIGINS contains '*'. With allow_credentials=True a wildcard is "
            "rejected by every browser, so the API would fail every cross-origin "
            "request with a misleading error -- and if credentials were disabled to "
            "'fix' it, any site the user visits could call this API. List exact "
            "origins.",
            details={"origins": list(origins)},
        )

    if not origins:
        # Not an error. A worker or a headless deployment has no browser client,
        # and an empty list is the correct, most restrictive policy -- the
        # middleware simply never matches an Origin.
        logger.info(
            "cors.no_origins_configured",
            consequence="cross-origin browser requests will be refused",
        )

    for origin in origins:
        if not origin.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"CORS origin {origin!r} has no scheme. An origin is "
                "scheme+host+port; a bare hostname never matches anything, so the "
                "policy would silently refuse the client it was meant to allow.",
                details={"origin": origin},
            )
        if origin.endswith("/"):
            raise ConfigurationError(
                f"CORS origin {origin!r} has a trailing slash. Browsers send the "
                "Origin header without one, so this entry can never match.",
                details={"origin": origin},
            )

    return CorsPolicy(allow_origins=origins)


def install_cors(app: FastAPI, settings: Settings | None = None) -> CorsPolicy:
    """Attach the CORS middleware and return the policy that was applied."""
    from fastapi.middleware.cors import CORSMiddleware

    policy = build_cors_policy(settings)
    app.add_middleware(CORSMiddleware, **policy.as_middleware_kwargs())
    logger.info("cors.installed", origins=list(policy.allow_origins))
    return policy
