"""Shared request-scoped dependencies: sessions, identity, tenancy, paging, idempotency.

Everything in this module exists so that a route body contains no plumbing. That
is not tidiness -- `docs/security-and-privacy.md` §3.2 makes it a rule with a
reason: *"Authorization is enforced in a FastAPI dependency in
`backend/api/deps.py`, not inside route bodies. A route with no auth dependency
must fail a test, not silently be public."* An `if principal.scopes` written into
a handler is a check that the next handler can forget to write.

Four things are decided here rather than per route.

**Tenancy comes from the credential and never from the request**
(`docs/api-reference.md` §3.1). There is no `X-Tenant-ID` header and no
`tenant_id` body field anywhere in the schemas, and there must never be one: a
tenant id a caller can type is a tenant id a caller can change. Every service is
constructed with `tenant_id=principal.tenant_id`, and every service already
carries the tenant into its `WHERE` clause, so cross-tenant reads return
`404` -- not a filtered-empty `200`, and not a `403`, which would turn the
endpoint into an existence oracle for other tenants' ids.

**Sessions come from a factory, and the factory is the test seam.** The
application's factory is the process-wide one in `backend/db/session.py`;
`tests/unit/backend/test_routes.py` overrides `get_session_factory` with an
in-memory SQLite factory and gets the real SQL, the real services and the real
handlers with no database running. Overriding a *session* instead would not work:
several services open more than one session per call, and each takes a factory
for exactly that reason.

**A datastore that is down produces a documented status, never a 500.** An
unhandled `ConnectionRefusedError` from a driver reaches the catch-all in
`backend/api/errors.py` and becomes `500 internal_error`, which tells a client to
open a bug rather than to retry. `upstream()` below is the narrow wrapper that
turns it into `502 upstream_unavailable`, which is what §4.5 and §4.7 document
for exactly this case.

**Idempotency fails open.** §3.5 backs `Idempotency-Key` with Redis, and Redis is
not on the list of dependencies that may take the API out of rotation
(`backend/api/v1/health.py`: only PostgreSQL is required). So a Redis outage
degrades deduplication to "not deduplicated" -- the documented behaviour for a
request that carries no key at all -- rather than failing every `POST` in the
API. The cost is a possible duplicate investigation during an outage; the
alternative cost is no investigations at all.

A note on the split with `backend/core/security.py`. The cryptography -- signing,
constant-time comparison, the closed algorithm table that cannot be talked into
`alg: none` -- lives there. What stays here is the *policy* around it: that every
verification failure produces the same bare `401` with the reason logged rather
than returned, and that the algorithm comes from settings rather than from the
token's own header. `decode_jws` carries its reason on the exception precisely so
that this layer can make that choice instead of having it made in the kernel.

Layer note: `backend/api/` (L4). May import `services/`, `models/`, `backend/`.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Annotated, Any, Final

from fastapi import Depends, Header, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.config import Settings, get_settings
from backend.core.exceptions import (
    ConflictError,
    ExternalServiceError,
    OmniSenseError,
    PermissionDeniedError,
    UnauthenticatedError,
    ValidationError,
)
from backend.core.logging import get_logger
from backend.core.security import (
    SUPPORTED_JWT_ALGORITHMS,
    JwsError,
    decode_jws,
    encode_jws,
)
from backend.schemas.common import DEFAULT_PAGE_LIMIT, MAX_PAGE_LIMIT
from models.base import utcnow
from models.orm.mixins import DEFAULT_TENANT
from services.investigation_service import InvestigationService
from services.signal_service import SignalService

__all__ = [
    "ROLE_SCOPES",
    "SCOPES",
    "CursorPage",
    "FeatureUnavailableError",
    "IdempotencyKeyReuseError",
    "IdempotencyOutcome",
    "IdempotencyStore",
    "Principal",
    "UpstreamUnavailableError",
    "csv_enum",
    "current_principal",
    "get_db_session",
    "get_idempotency_store",
    "get_investigation_service",
    "get_session_factory",
    "get_signal_service",
    "idempotency_key",
    "mint_access_token",
    "pagination",
    "request_id",
    "require_scopes",
    "trace_id",
    "upstream",
]

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# API-level problem codes
# --------------------------------------------------------------------------- #


class UpstreamUnavailableError(ExternalServiceError):
    """A datastore the endpoint cannot degrade without is unreachable.

    `502 upstream_unavailable` from the §3.3 catalogue. A subclass rather than a
    new member of `backend/core/exceptions.py` because the distinction is an
    *HTTP* one: `services/` already raises `DependencyUnavailableError` (503) for
    a store it needs, and the two say different things to a caller. 503 means
    "this replica is not ready, try another"; 502 means "this replica is fine and
    something behind it is not", which is the honest answer when Neo4j is down and
    the rest of the API is serving normally.

    `services/signal_service.py::MalformedCursorError` sets the precedent for
    declaring a narrow code next to the code that raises it.
    """

    status_code = 502
    code = "upstream_unavailable"
    default_message = "A required dependency is unavailable."


class FeatureUnavailableError(OmniSenseError):
    """A documented capability whose backing store has not been built yet.

    `501`, and deliberately not one of the statuses in §6. None of them is true:
    `422` blames the caller for a request the contract explicitly permits, `503`
    and `502` both promise that retrying will eventually work, and `500` claims a
    bug. The only honest answer to "filter signals by `has_media`" while there is
    no `signal_media` table is "the server does not implement that", which is what
    501 means.

    `details` must always name the missing piece. A 501 that does not say what is
    missing is indistinguishable from a 501 that is a bug.
    """

    status_code = 501
    code = "not_implemented"
    default_message = "This capability is documented but not implemented yet."


class IdempotencyKeyReuseError(ConflictError):
    """The same `Idempotency-Key` arrived with a different body (§3.5).

    Distinct from a plain `409 conflict`, which §3.5 uses for a replay that lands
    while the first request is still in flight. A client must be able to tell
    "you sent me two different things under one key" (a bug in the client, do not
    retry) from "your earlier call has not finished" (retry shortly), and a
    shared code would make both look like the second.
    """

    code = "idempotency_key_reuse"
    default_message = "This Idempotency-Key was already used with a different request body."


@asynccontextmanager
async def upstream(dependency: str) -> AsyncIterator[None]:
    """Convert a driver failure into `502 upstream_unavailable`.

    Wrap the **call**, never the projection that follows it. The guard catches
    `Exception`, which is right for "the socket to Neo4j died" and wrong for a
    `KeyError` in the code that shapes the result: the first is a documented 502
    and the second is a bug that belongs in the 500 handler with its traceback
    intact. Keeping the block to the I/O itself is what preserves that
    distinction, and it is the reason this is a context manager rather than a
    decorator on the handler.

    `OmniSenseError` passes through untouched. A service that has already decided
    the failure is a `404`, a `422` or a `409` has more information than this
    wrapper does, and re-labelling it as an upstream outage would send a client
    to retry a request that can never succeed.
    """
    try:
        yield
    except OmniSenseError:
        raise
    except Exception as exc:
        logger.warning("api.upstream_unavailable", dependency=dependency, error=str(exc))
        raise UpstreamUnavailableError(
            f"{dependency} is unavailable.",
            details={"dependency": dependency},
            cause=exc,
        ) from exc


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #


async def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """The process-wide session factory. **The single test seam for storage.**

    A factory and not a session, because every service in `services/` takes one:
    a method there is one short transaction, and a service holding an open
    session would pin a pooled connection for as long as anything referenced the
    service.

    Declared `async` purely so FastAPI treats it as a coroutine dependency and
    does not push it to the threadpool. It performs no I/O -- `get_sessionmaker()`
    builds the engine lazily on first use and opens no socket
    (`backend/db/session.py`).
    """
    from backend.db.session import get_sessionmaker

    return get_sessionmaker()


async def get_db_session(
    factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
) -> AsyncIterator[AsyncSession]:
    """One session for the life of one request, for handlers that read directly.

    Deliberately does **not** commit, matching `backend/db/session.py::get_session`:
    a handler that changed something commits explicitly, which keeps "this
    endpoint writes" visible at the call site instead of implied by a dependency.
    Rollback on exception is automatic, so a failed request cannot leak a
    half-applied transaction.
    """
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


# --------------------------------------------------------------------------- #
# Identity, tenancy and scopes
# --------------------------------------------------------------------------- #


SCOPES: Final[frozenset[str]] = frozenset(
    {
        "investigations:read",
        "investigations:write",
        "reports:read",
        "signals:read",
        "graph:read",
        "connectors:write",
        "agents:run",
    }
)
"""The scope vocabulary of `docs/api-reference.md` §3.1.

A closed set, checked when a token is parsed. An unknown scope in a token is
dropped with a log line rather than carried: carrying it would let a typo in an
issuer (`signals:reads`) look like a scope the API might one day honour, and the
day it is added the typo silently becomes a grant.
"""

ROLE_SCOPES: Final[Mapping[str, frozenset[str]]] = {
    "viewer": frozenset(
        {"investigations:read", "reports:read", "signals:read", "graph:read"}
    ),
    "analyst": frozenset(
        {
            "investigations:read",
            "investigations:write",
            "reports:read",
            "signals:read",
            "graph:read",
            "connectors:write",
        }
    ),
    "service": frozenset(
        {"investigations:read", "investigations:write", "reports:read", "signals:read",
         "graph:read", "connectors:write"}
    ),
    "admin": SCOPES,
}
"""Role -> scope expansion, reconciling the two halves of the specification.

`docs/api-reference.md` §3.1 grants access by *scope*; `docs/security-and-privacy.md`
§3.2 grants it by *role* (`viewer`, `analyst`, `admin`, `service`). Both are
normative and they are not in conflict -- roles are bundles of scopes -- so a
token may carry either and this table is where a role becomes the scopes the
endpoints actually check. Without it, an issuer that follows the security
document produces tokens that every endpoint rejects.

`agents:run` is admin-only on purpose. §4.6 calls `POST /agents/run` a debugging
and evaluation surface that runs a model outside the orchestrated flow with no
step budget above it, which is not something a `viewer` role should imply.
"""


_API_KEY_PATTERN: Final = re.compile(r"^om_[A-Za-z0-9]{4,64}_[A-Za-z0-9_\-]{16,128}$")
"""Shape of a service API key, `om_<id>_<secret>` (`docs/security-and-privacy.md` §3.1)."""


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is making this request, and what they may do.

    Frozen: a handler that could mutate `scopes` could grant itself one. The
    authorization decision is made once, in `require_scopes`, against an object
    nothing downstream can edit.
    """

    subject: str
    tenant_id: str
    scopes: frozenset[str]
    credential: str
    """`jwt` or `api_key`. Recorded for logging and for §3.6's per-credential rate
    limiting, which prices a service key differently from a browser session."""

    def has(self, scope: str) -> bool:
        return scope in self.scopes

    def missing(self, required: Iterable[str]) -> list[str]:
        """Which of `required` this principal lacks, sorted for a stable message."""
        return sorted(scope for scope in required if scope not in self.scopes)




def _decode_jwt(token: str, settings: Settings) -> dict[str, Any]:
    """Verify a session token and return its claims.

    The cryptography lives in `backend/core/security.py`; this function owns the
    *policy* around it, which is the split that module's docstring asks for. What
    stays here is the pair of decisions the kernel deliberately refuses to make:

    **Every failure raises the same bare `UnauthenticatedError`.** The exception
    itself states why: *"Never include the reason in `details`. 'Signature
    invalid' versus 'token expired' tells an attacker which half of a forgery
    attempt succeeded."* `decode_jws` carries the reason on `JwsError` precisely
    so that this layer can log it and withhold it, rather than having the choice
    made for it.

    **The algorithm comes from settings, never from the token.** Passed in
    explicitly, so a header claiming `HS512` against an HS256 deployment fails
    even though both are supported.
    """
    try:
        return decode_jws(
            token,
            secret=settings.security.secret_key.get_secret_value(),
            algorithm=settings.security.jwt_algorithm,
            now=utcnow().timestamp(),
        )
    except JwsError as error:
        logger.info("api.auth.rejected", reason=error.reason)
        raise UnauthenticatedError() from None


def _scopes_from_claims(claims: Mapping[str, Any]) -> frozenset[str]:
    """Read scopes from `scope` (space-delimited, RFC 8693) or `scopes` (a list).

    Both spellings are accepted because both are in the wild and the issuer is
    not written yet; whichever `backend/core/security.py` settles on, tokens that
    already exist keep working. A `role` claim expands through `ROLE_SCOPES`,
    which is what reconciles the role table in `docs/security-and-privacy.md` §3.2
    with the scope table in `docs/api-reference.md` §3.1.

    Unknown scopes are dropped rather than carried -- see `SCOPES`.
    """
    granted: set[str] = set()

    raw = claims.get("scope")
    if isinstance(raw, str):
        granted.update(raw.split())
    listed = claims.get("scopes")
    if isinstance(listed, (list, tuple)):
        granted.update(str(item) for item in listed)

    role = claims.get("role")
    if isinstance(role, str):
        granted.update(ROLE_SCOPES.get(role.lower(), frozenset()))

    unknown = granted - SCOPES
    if unknown:
        logger.info("api.auth.unknown_scopes", scopes=sorted(unknown))
    return frozenset(granted & SCOPES)


def mint_access_token(
    *,
    subject: str,
    tenant_id: str = DEFAULT_TENANT,
    scopes: Iterable[str] = (),
    role: str | None = None,
    ttl_seconds: int | None = None,
    settings: Settings | None = None,
) -> str:
    """Sign a session token the API will accept.

    This is a *verifier's* helper, not an authentication service. It exists
    because there is nowhere else to mint a token: there is no login endpoint and
    no identity provider, so without it the API is a surface nobody -- including
    its own test suite and `scripts/sync_connector.py` -- can call. The signing
    itself is `backend/core/security.encode_jws`, so the issuer and the verifier
    share one implementation and cannot disagree about the encoding.

    It signs with `SECRET_KEY`, which is exactly what a Phase 1 HS256 issuer
    would do (`docs/security-and-privacy.md` §3.1). The symmetric key is also why
    §3.1 already schedules the Phase 7 migration to RS256/EdDSA: with HS256 every
    verifier holds signing material, so any process that can check a token can
    also issue one.
    """
    resolved = settings or get_settings()
    algorithm = resolved.security.jwt_algorithm.upper()
    if algorithm not in SUPPORTED_JWT_ALGORITHMS:
        raise ValueError(
            f"JWT_ALGORITHM={algorithm!r} is not supported; this issuer signs "
            f"only {sorted(SUPPORTED_JWT_ALGORITHMS)}"
        )

    issued = int(utcnow().timestamp())
    ttl = resolved.security.access_token_ttl_seconds if ttl_seconds is None else ttl_seconds
    claims: dict[str, Any] = {
        "sub": subject,
        "tenant": tenant_id,
        "iat": issued,
        "exp": issued + ttl,
    }
    if scopes:
        claims["scope"] = " ".join(sorted(scopes))
    if role is not None:
        claims["role"] = role

    return encode_jws(
        claims,
        secret=resolved.security.secret_key.get_secret_value(),
        algorithm=algorithm,
    )


async def current_principal(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> Principal:
    """Resolve the caller from the credential they presented.

    Two credential types, distinguished at the header level so they can never be
    confused (§3.1): a browser session JWT on `Authorization: Bearer`, and a
    service key on `X-API-Key`. A key presented on `Authorization` is rejected
    rather than sniffed -- header position is the cheapest possible way to keep
    "long-lived, high-privilege, hashed in a database" apart from "short-lived,
    signed, self-contained", and sniffing throws that away.

    Both present is refused. It is not a request a legitimate client makes, and
    picking one silently makes the effective identity depend on the order of two
    `if` statements.
    """
    settings = get_settings()
    bearer = _bearer_token(authorization)

    if bearer is not None and x_api_key is not None:
        logger.info("api.auth.rejected", reason="two_credentials")
        raise UnauthenticatedError()

    if bearer is not None:
        claims = _decode_jwt(bearer, settings)
        subject = str(claims.get("sub") or "").strip()
        if not subject:
            logger.info("api.auth.rejected", reason="no_subject")
            raise UnauthenticatedError()
        tenant = str(claims.get("tenant") or claims.get("tenant_id") or DEFAULT_TENANT)
        return Principal(
            subject=subject,
            tenant_id=tenant,
            scopes=_scopes_from_claims(claims),
            credential="jwt",
        )

    if x_api_key is not None:
        # Shape-checked, then refused. Verification needs the bcrypt hash of the
        # key, which §3.1 stores in PostgreSQL -- and there is no `api_keys` table
        # in `models/orm/`, no `passlib` in `requirements.txt`, and therefore no
        # key in existence that could be verified. A 401 is the correct answer to
        # an unverifiable credential; the alternative -- accepting a
        # well-shaped key because it looks right -- would make the header a
        # password-less authentication bypass.
        shape = "well_formed" if _API_KEY_PATTERN.fullmatch(x_api_key) else "malformed"
        logger.info(
            "api.auth.rejected",
            reason="api_key_verification_unavailable",
            shape=shape,
            missing="models/orm/api_key.py (bcrypt hashes, docs/security-and-privacy.md 3.1)",
        )
        raise UnauthenticatedError()

    raise UnauthenticatedError()


def _bearer_token(authorization: str | None) -> str | None:
    """Extract the token from `Authorization: Bearer <jwt>`, case-insensitively.

    RFC 7235 makes the scheme name case-insensitive, and clients do send
    `bearer`. Returning `None` for any other scheme rather than raising lets the
    caller fall through to the API-key branch and produce one uniform 401.
    """
    if not authorization:
        return None
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = credentials.strip()
    return token or None


def require_scopes(*required: str) -> Callable[[Principal], Awaitable[Principal]]:
    """Build the dependency that enforces `required` on a route.

    A factory returning a dependency, so the scopes a route needs are visible in
    its decorator (`Depends(require_scopes("signals:read"))`) rather than buried
    in its body. That visibility is the property `docs/security-and-privacy.md`
    §3.2 asks for: a route with no auth dependency is missing something you can
    see, and `tests/unit/backend/test_routes.py` asserts every non-probe route
    has one.

    Unknown scope names raise at import time. A typo in a route decorator would
    otherwise produce a requirement no token can ever satisfy -- a 403 for every
    caller, indistinguishable from a deliberate lockout.
    """
    unknown = sorted(set(required) - SCOPES)
    if unknown:
        raise ValueError(
            f"unknown scope(s) {unknown}; the vocabulary of docs/api-reference.md "
            f"§3.1 is {sorted(SCOPES)}"
        )

    async def dependency(
        principal: Annotated[Principal, Depends(current_principal)],
    ) -> Principal:
        missing = principal.missing(required)
        if missing:
            # The missing scopes are named because the caller can act on them --
            # they are the caller's own authorization, not a fact about the
            # system. Contrast `UnauthenticatedError`, which says nothing.
            raise PermissionDeniedError(
                f"this endpoint requires {sorted(required)}",
                details={"required": sorted(required), "missing": missing},
            )
        return principal

    return dependency


# --------------------------------------------------------------------------- #
# Service construction
# --------------------------------------------------------------------------- #


async def get_investigation_service(
    principal: Annotated[Principal, Depends(current_principal)],
    factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
) -> InvestigationService:
    """An investigation service already scoped to the caller's tenant.

    The tenant is bound *here*, at construction, and not passed per call. That is
    what makes a cross-tenant read impossible to write by forgetting an argument:
    every method on the service reaches for `self._tenant_id`, and there is no
    call shape in which a handler supplies one.
    """
    return InvestigationService(factory, tenant_id=principal.tenant_id)


async def get_signal_service(
    factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
) -> SignalService:
    """The signal read facade.

    `SignalService` takes its tenant on the `SignalQuery` rather than on the
    constructor, so the handler sets `tenant_id=principal.tenant_id` on the query
    it builds. That asymmetry is the service's, not this module's.
    """
    return SignalService(factory)


# --------------------------------------------------------------------------- #
# Cursor pagination
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CursorPage:
    """The `limit` / `cursor` pair every collection endpoint accepts (§3.4)."""

    limit: int
    cursor: str | None


async def pagination(
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_PAGE_LIMIT,
            description="Page size. Values above the maximum are rejected, not clamped.",
        ),
    ] = DEFAULT_PAGE_LIMIT,
    cursor: Annotated[
        str | None,
        Query(
            description="Opaque resume token from a previous `page.next_cursor`. "
            "Repeat every filter and sort parameter unchanged alongside it.",
            max_length=2048,
        ),
    ] = None,
) -> CursorPage:
    """Validate and carry the pagination parameters.

    `le=MAX_PAGE_LIMIT` produces the `422` §3.4 requires for an oversized limit,
    through FastAPI's own validation rather than through a check in each handler
    -- four handlers means four chances to clamp instead of reject, and a clamp
    makes a client believe it has reached the end of a collection when it has
    only reached the end of a page.

    `max_length` on the cursor is not decoration. The token is echoed into a
    `MalformedCursorError` path and into logs; an unbounded query parameter is a
    free way to write megabytes into a log pipeline.
    """
    return CursorPage(limit=limit, cursor=cursor)


# --------------------------------------------------------------------------- #
# Correlation
# --------------------------------------------------------------------------- #


async def request_id(request: Request) -> str:
    """The correlation id for this request, set by `backend/middleware/request_id.py`.

    Falls back to minting one so that a handler is never handed an empty string
    when the middleware is absent -- which is the case in a unit test that builds
    a bare `APIRouter`. A missing correlation id would otherwise surface as a
    `KeyError` in a response body rather than as the plumbing gap it is.
    """
    existing = getattr(request.state, "request_id", None)
    return str(existing) if existing else uuid.uuid4().hex


async def trace_id(request: Request) -> str:
    """The W3C trace id for this request, set by `backend/middleware/tracing.py`."""
    existing = getattr(request.state, "trace_id", None)
    return str(existing) if existing else uuid.uuid4().hex


# --------------------------------------------------------------------------- #
# Idempotency (§3.5)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class IdempotencyOutcome:
    """What `IdempotencyStore.begin()` decided about this request.

    `replay` carries the original response when one was recorded; `None` means
    "this is the first time, go and do the work". `token` is what `finish()`
    needs in order to write the result back under the same key.
    """

    token: str | None
    replay: dict[str, Any] | None = None
    status_code: int = 200


@dataclass(slots=True)
class IdempotencyStore:
    """Redis-backed `(tenant, endpoint, key)` deduplication with a 24h window.

    §3.5 fixes the semantics and each clause has a distinct failure it prevents:

    - *Same key, same body* replays the original response. Without it, a client
      that retried on a socket timeout starts a second investigation and pays for
      it twice.
    - *Same key, different body* is `409 idempotency_key_reuse`. Silently
      replaying the first response would tell the client its second, different
      request succeeded when nothing ran.
    - *Same key while the first is in flight* is `409` with `retry_after`, which
      is what makes a fast double-submit safe: the second request is refused
      rather than racing the first to create a duplicate row.

    The body fingerprint is a SHA-256 of the canonicalized JSON, so key order and
    whitespace do not make an identical request look different -- a client
    re-serializing its own retry from a dict would otherwise trip the reuse
    conflict every time.

    Every Redis call is wrapped. Redis is not a required dependency
    (`backend/api/v1/health.py`), so an outage degrades this to "no deduplication"
    -- exactly what a request carrying no key already gets -- rather than failing
    every write endpoint in the API.
    """

    ttl_seconds: int = 24 * 60 * 60
    in_flight_seconds: int = 120
    _memory: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Process-local mirror, used only when Redis is unreachable.

    It does **not** make deduplication correct during an outage: with several API
    replicas a retry can land on a different process and miss. It is here because
    the single-process case -- a unit test, a local `make api`, a double-click in
    a browser hitting one worker -- is both common and cheap to get right, and
    because degrading to a *wrong* answer (replaying nothing, ever) makes the
    endpoint's behaviour untestable without a container.
    """

    async def begin(
        self, *, tenant_id: str, endpoint: str, key: str | None, body: Any
    ) -> IdempotencyOutcome:
        """Claim the key, or report what the first request already produced.

        Returns immediately with `token=None` when the caller sent no key: §3.5
        is explicit that requests without the header are never deduplicated, so
        two identical `POST /investigations` calls create two investigations.
        """
        if not key:
            return IdempotencyOutcome(token=None)

        token = f"idem:{tenant_id}:{endpoint}:{key}"
        fingerprint = _fingerprint(body)
        claim = {"fingerprint": fingerprint, "state": "in_flight"}

        stored = await self._set_if_absent(token, claim, ttl=self.in_flight_seconds)
        if stored:
            return IdempotencyOutcome(token=token)

        existing = await self._get(token)
        if existing is None:
            # The claim expired between the failed SET NX and this read, or Redis
            # dropped it. Treat the request as fresh: re-running is the documented
            # cost of not having a key at all, and refusing would strand a caller
            # behind a key that no longer exists.
            return IdempotencyOutcome(token=token)

        if existing.get("fingerprint") != fingerprint:
            raise IdempotencyKeyReuseError(
                "this Idempotency-Key was used with a different request body; "
                "generate a new key for a new request",
                details={"endpoint": endpoint},
            )
        if existing.get("state") != "done":
            raise ConflictError(
                "an earlier request with this Idempotency-Key is still in flight; "
                "retry shortly rather than re-submitting",
                details={"endpoint": endpoint, "retry_after_seconds": 2},
            )
        return IdempotencyOutcome(
            token=token,
            replay=existing.get("response"),
            status_code=int(existing.get("status", 200)),
        )

    async def finish(
        self, outcome: IdempotencyOutcome, *, status_code: int, body: Any
    ) -> None:
        """Record the response so a replay can return it for the next 24 hours.

        Called only after the work is durable. Recording before the commit would
        let a crash between the two leave a key that replays a success for an
        investigation that was never created.
        """
        if outcome.token is None:
            return
        record = {
            "fingerprint": (await self._get(outcome.token) or {}).get("fingerprint"),
            "state": "done",
            "status": status_code,
            "response": body,
        }
        await self._set(outcome.token, record, ttl=self.ttl_seconds)

    # -- storage, degrading to process memory ------------------------------- #

    async def _get(self, token: str) -> dict[str, Any] | None:
        try:
            from backend.db.redis import get_redis

            raw = await get_redis().get(token)
        except Exception as exc:
            logger.warning("api.idempotency.degraded", operation="get", error=str(exc))
            return self._memory.get(token)
        if raw is None:
            return None
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            logger.warning("api.idempotency.corrupt_record")
            return None
        return decoded if isinstance(decoded, dict) else None

    async def _set(self, token: str, record: dict[str, Any], *, ttl: int) -> None:
        self._memory[token] = record
        try:
            from backend.db.redis import get_redis

            await get_redis().set(token, json.dumps(record), ex=ttl)
        except Exception as exc:
            logger.warning("api.idempotency.degraded", operation="set", error=str(exc))

    async def _set_if_absent(self, token: str, record: dict[str, Any], *, ttl: int) -> bool:
        """`SET NX EX`: claim the key, reporting whether this caller won.

        One command rather than GET-then-SET. Redis executes commands one at a
        time, so two replicas racing on the same key cannot both be told they are
        first -- which is the entire mechanism behind the in-flight `409`.
        """
        try:
            from backend.db.redis import get_redis

            claimed = await get_redis().set(token, json.dumps(record), nx=True, ex=ttl)
        except Exception as exc:
            logger.warning("api.idempotency.degraded", operation="setnx", error=str(exc))
            if token in self._memory:
                return False
            self._memory[token] = record
            return True
        if claimed:
            self._memory[token] = record
        return bool(claimed)


_IDEMPOTENCY_STORE = IdempotencyStore()
"""One store per process, so the in-memory fallback survives between requests."""


async def get_idempotency_store() -> IdempotencyStore:
    """The process-wide idempotency store, overridable in tests."""
    return _IDEMPOTENCY_STORE


async def idempotency_key(
    idempotency_key: Annotated[
        str | None,
        Header(
            description="Client-generated key. Scoped to (tenant, endpoint, key) "
            "and retained for 24 hours (docs/api-reference.md §3.5).",
            max_length=255,
        ),
    ] = None,
) -> str | None:
    """Read and sanity-check `Idempotency-Key`.

    Only length and emptiness are checked. §3.5 recommends a UUID but does not
    require one, and rejecting a client's well-behaved ULID or hash for failing a
    UUID parser would break a caller that is doing nothing wrong. The key is
    concatenated into a Redis key, so control characters are refused -- a newline
    in a key name is how one tenant's namespace becomes another's.
    """
    if idempotency_key is None:
        return None
    candidate = idempotency_key.strip()
    if not candidate or any(character.isspace() for character in candidate):
        raise ValidationError(
            "Idempotency-Key must be a non-empty token containing no whitespace",
            details={"header": "Idempotency-Key"},
        )
    return candidate


def _fingerprint(body: Any) -> str:
    """Canonical SHA-256 of a request body, stable across re-serialization.

    `sort_keys` and the tight separators are what make it stable: a client that
    rebuilds its retry from a dict emits the same fields in a different order,
    and a fingerprint over the raw bytes would call that a different request and
    answer `409 idempotency_key_reuse` to a correct retry.
    """
    canonical = json.dumps(_jsonable(body), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    """Reduce a request body to plain JSON types for fingerprinting."""
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    return value


def csv_enum(
    raw: str | None,
    *,
    allowed: Sequence[str],
    default: Sequence[str],
    parameter: str,
) -> frozenset[str]:
    """Parse a documented `include`-style CSV enum parameter.

    Shared because three endpoints take one (`include` on investigations, reports
    and signals) and each documents the same failure: an unknown member is a
    `422`, not a silently ignored token. Ignoring it is worse than it sounds --
    `include=citation` would return a report without citations and no indication
    that the caller asked for them.
    """
    if raw is None:
        return frozenset(default)
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted({part for part in requested if part not in allowed})
    if unknown:
        raise ValidationError(
            f"unknown {parameter} value(s) {unknown}; permitted values are "
            f"{sorted(allowed)}",
            details={"parameter": parameter, "unknown": unknown, "allowed": sorted(allowed)},
        )
    return frozenset(requested)
