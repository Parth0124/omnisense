"""Application exception hierarchy.

Every error raised deliberately by OmniSense derives from `OmniSenseError` and
carries three things the HTTP layer needs in order to produce an RFC 7807
`application/problem+json` response without a translation table:

`status_code`
    The HTTP status this maps to. Declared on the exception class rather than
    decided by the handler, so an error raised deep in `services/` cannot be
    mis-classified by a route that has no idea what went wrong.

`code`
    A stable, machine-readable slug (`signal_not_found`). Clients branch on this.
    It is part of the public API contract: renaming one is a breaking change.

`details`
    Structured, **non-sensitive** context. This is serialized into the response
    body and into logs, so it must never contain credentials, raw fetched content
    or personal data -- see `docs/security-and-privacy.md`.

Note the deliberate omission: connector failures are **not** here. Their taxonomy
lives in `connectors/exceptions.py` because `connectors/` may not import
`backend/` at all (`docs/architecture.md` §6.2 rule 2). `services/` translates a
`ConnectorError` into the appropriate subclass below when one needs to surface
over HTTP.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ConfigurationError",
    "ConflictError",
    "DependencyUnavailableError",
    "ExternalServiceError",
    "InvestigationFailedError",
    "NotFoundError",
    "OmniSenseError",
    "PermissionDeniedError",
    "RateLimitedError",
    "UnauthenticatedError",
    "ValidationError",
]


class OmniSenseError(Exception):
    """Base class for every deliberate OmniSense failure.

    Catching this catches "something we anticipated went wrong" while letting
    genuine bugs -- `KeyError`, `AttributeError` -- propagate to the 500 handler
    where they belong. A bare `except Exception` in a request path hides both.
    """

    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details: dict[str, Any] = details or {}
        self.cause = cause
        super().__init__(self.message)

    def to_problem(self, instance: str | None = None) -> dict[str, Any]:
        """Render as an RFC 7807 problem document.

        `type` is a stable URI-shaped identifier rather than a resolvable URL:
        it groups errors for clients without committing the project to hosting
        documentation at that address forever.
        """
        problem: dict[str, Any] = {
            "type": f"https://omnisense.dev/errors/{self.code}",
            "title": self.code.replace("_", " "),
            "status": self.status_code,
            "detail": self.message,
        }
        if self.details:
            problem["details"] = self.details
        if instance:
            problem["instance"] = instance
        return problem

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


# --------------------------------------------------------------------------- #
# 4xx -- the caller can fix it
# --------------------------------------------------------------------------- #


class ValidationError(OmniSenseError):
    """Input was syntactically valid but semantically wrong.

    Distinct from Pydantic's request-shape validation, which FastAPI handles
    before a handler runs. This is for rules the schema cannot express: a time
    range whose end precedes its start, a filter naming an unknown platform.
    """

    status_code = 422
    code = "validation_error"
    default_message = "The request was well-formed but semantically invalid."


class UnauthenticatedError(OmniSenseError):
    """No credential, or an unreadable one.

    Never include the reason in `details`. "Signature invalid" versus "token
    expired" tells an attacker which half of a forgery attempt succeeded.
    """

    status_code = 401
    code = "unauthenticated"
    default_message = "Authentication is required."


class PermissionDeniedError(OmniSenseError):
    """Authenticated, but not allowed to do this."""

    status_code = 403
    code = "permission_denied"
    default_message = "You do not have permission to perform this action."


class NotFoundError(OmniSenseError):
    """The addressed resource does not exist, or is not visible to this caller.

    Deliberately conflates "absent" and "not yours": distinguishing them turns
    the endpoint into an existence oracle for ids belonging to other tenants.
    """

    status_code = 404
    code = "not_found"
    default_message = "The requested resource was not found."

    @classmethod
    def for_resource(cls, kind: str, identifier: str) -> NotFoundError:
        """Build a typed not-found for a specific resource."""
        return cls(
            f"{kind} {identifier!r} was not found.",
            details={"resource": kind, "id": identifier},
        )


class ConflictError(OmniSenseError):
    """The request conflicts with current state.

    Raised for a duplicate idempotency key carrying a different payload, or an
    attempt to cancel an investigation that has already reached a terminal state.
    """

    status_code = 409
    code = "conflict"
    default_message = "The request conflicts with the current state of the resource."


class RateLimitedError(OmniSenseError):
    """The caller exceeded their inbound rate limit.

    `retry_after_seconds` becomes the `Retry-After` header. Always populate it --
    a 429 without it tells a client to back off but not by how much, and most
    clients respond by retrying immediately.
    """

    status_code = 429
    code = "rate_limited"
    default_message = "Too many requests."

    def __init__(
        self,
        message: str | None = None,
        *,
        retry_after_seconds: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)
        self.retry_after_seconds = retry_after_seconds
        if retry_after_seconds is not None:
            self.details.setdefault("retry_after_seconds", retry_after_seconds)


# --------------------------------------------------------------------------- #
# 5xx -- we have to fix it
# --------------------------------------------------------------------------- #


class ConfigurationError(OmniSenseError):
    """The process is misconfigured.

    Raised at startup by `backend/core/config.py` and by client bootstrap in
    `backend/db/`. Reaching a *request* with this error means a startup check is
    missing -- the goal is that this never surfaces over HTTP.
    """

    status_code = 500
    code = "configuration_error"
    default_message = "The service is misconfigured."


class DependencyUnavailableError(OmniSenseError):
    """A datastore or internal dependency is unreachable.

    503 rather than 500: this is retryable, and the difference is what lets a
    client and a load balancer behave sensibly. `docs/architecture.md` §7.3
    defines which dependencies degrade gracefully and which are fatal -- only the
    fatal ones should reach this.
    """

    status_code = 503
    code = "dependency_unavailable"
    default_message = "A required dependency is temporarily unavailable."

    @classmethod
    def for_store(cls, store: str, cause: Exception | None = None) -> DependencyUnavailableError:
        return cls(
            f"{store} is unavailable.",
            details={"dependency": store},
            cause=cause,
        )


class ExternalServiceError(OmniSenseError):
    """A third-party service failed in a way we cannot paper over.

    Covers the LLM provider and object storage. Connector-side third-party
    failures use `connectors/exceptions.py` instead and are handled inside the
    ingestion path rather than surfacing to a caller.
    """

    status_code = 502
    code = "external_service_error"
    default_message = "An upstream service returned an error."


class InvestigationFailedError(OmniSenseError):
    """An investigation terminated without producing a report.

    Distinct from `completed_with_findings`, which is a *success* whose report
    ships with unresolved Critic findings attached (`docs/agent-system.md` §13).
    This is the case where no report exists at all.
    """

    status_code = 500
    code = "investigation_failed"
    default_message = "The investigation could not be completed."
