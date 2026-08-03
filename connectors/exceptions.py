"""Connector failure taxonomy.

Four families, and the family is part of the contract rather than an
implementation detail: the runtime decides what to do purely from the class it
caught (`docs/connector-spec.md` §6).

| Class              | Retried | Cursor       | Outcome                              |
| ------------------ | ------- | ------------ | ------------------------------------ |
| `AuthError`        | no      | untouched    | halt run, flag account `needs_reauth` |
| `QuotaError`       | later   | checkpointed | partial success, reschedule at reset  |
| `TransientError`   | yes     | untouched    | backoff with jitter, then escalate    |
| `PermanentError`   | no      | untouched    | record to DLQ, continue or abort      |

Two of those rows are easy to get backwards and both are expensive.

`QuotaError` is a **partial success**, not a failure. Records already emitted
stay emitted and the cursor is committed, because throwing away an hour of
successful pagination just because the 4,000th request hit a quota wall is how a
connector never finishes a backfill.

`AuthError` is terminal on the *first* recurrence, deliberately. The runtime
re-authenticates once after a 401 and gives up if the next call also fails.
Looping on auth is how an integration gets an application-level ban rather than
an account-level one.

This module imports nothing from `backend/` or `services/`
(`docs/architecture.md` §6.2 rule 2), so it stays usable by a connector under
test with nothing but `respx`.
"""

from __future__ import annotations

from typing import Any, ClassVar

from models.enums import ConnectorErrorClass

__all__ = [
    "AuthError",
    "CircuitOpenError",
    "ConnectorConfigurationError",
    "ConnectorError",
    "NormalizationError",
    "PermanentError",
    "QuotaError",
    "TransientError",
]


class ConnectorError(Exception):
    """Base class for every anticipated connector failure.

    Catching this catches "the source misbehaved in a way we planned for" while
    letting real bugs -- `KeyError` in a mapper, `AttributeError` on a typo --
    propagate. A bare `except Exception` in a fetch loop hides both, and the
    second kind then looks like a flaky provider forever.
    """

    error_class: ClassVar[ConnectorErrorClass] = ConnectorErrorClass.UNKNOWN
    retryable: ClassVar[bool] = False

    def __init__(
        self,
        message: str,
        *,
        connector: str | None = None,
        account_id: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.message = message
        self.connector = connector
        self.account_id = account_id
        self.status_code = status_code
        self.details: dict[str, Any] = details or {}
        self.cause = cause
        super().__init__(message)

    def to_log_fields(self) -> dict[str, Any]:
        """Structured fields safe to log.

        Note what is absent: no response body, no request headers, no
        credentials. A provider error message can echo the request that caused
        it, and connector requests carry tokens in headers and fetched content in
        bodies -- `docs/connector-spec.md` §1 forbids logging either.
        """
        return {
            "error_class": self.error_class.value,
            "error_type": type(self).__name__,
            "connector": self.connector,
            "account_id": self.account_id,
            "status_code": self.status_code,
            "retryable": self.retryable,
            **{k: v for k, v in self.details.items() if k not in {"body", "headers"}},
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(connector={self.connector!r}, "
            f"status={self.status_code!r}, message={self.message!r})"
        )


class AuthError(ConnectorError):
    """Credentials are missing, invalid, expired or revoked (401, 403).

    Terminal for the run: no retry, no backoff. The account row is flagged
    `needs_reauth`, the cursor is left untouched, and the run is recorded as
    failed with zero partial credit. A second 401 after a successful
    re-authenticate is also terminal -- the runtime never loops on auth.
    """

    error_class = ConnectorErrorClass.AUTH
    retryable = False


class QuotaError(ConnectorError):
    """The provider's quota is exhausted (429 with a long reset, or an explicit cap).

    A *partial success*. The cursor is checkpointed and the run is rescheduled at
    `reset_at`; records already emitted remain emitted.

    `reset_at` is a UNIX timestamp rather than a `datetime` so it can cross a
    Kafka envelope and a Postgres column without a serialization decision, and
    so a provider's `X-RateLimit-Reset` header maps onto it directly.
    """

    error_class = ConnectorErrorClass.QUOTA
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        reset_at: float | None = None,
        retry_after_seconds: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.reset_at = reset_at
        self.retry_after_seconds = retry_after_seconds
        if retry_after_seconds is not None:
            self.details.setdefault("retry_after_seconds", retry_after_seconds)
        if reset_at is not None:
            self.details.setdefault("reset_at", reset_at)


class TransientError(ConnectorError):
    """A failure that may succeed on retry: timeout, connection reset, 5xx.

    The only retryable family. The connector raises and stops there -- backoff is
    the runtime's job, because `docs/connector-spec.md` §1 forbids a connector
    from sleeping or retrying internally. A connector that retries privately
    makes the shared limiter's accounting wrong and the failure invisible to
    metrics.
    """

    error_class = ConnectorErrorClass.TRANSIENT
    retryable = True


class PermanentError(ConnectorError):
    """A failure that will recur identically on retry: 404, 400, schema change.

    Not retried. Depending on scope it either sends one record to the DLQ and
    continues, or aborts the run -- a 404 on a single item is a deleted post, a
    404 on the whole feed means the configuration is wrong.
    """

    error_class = ConnectorErrorClass.PERMANENT
    retryable = False


class NormalizationError(PermanentError):
    """A payload the connector cannot map onto a Signal.

    Scoped to one record: it goes to `omnisense.dlq` with the raw payload
    attached and the run continues.

    Distinct from a connector returning `None` from `normalize()`, which is the
    sanctioned way to *drop* a record silently -- a deleted comment, an empty
    feed entry. Dropping is expected and counted separately; failing to map is a
    defect worth looking at. Conflating them buries real mapping bugs in a
    drop counter nobody reads.
    """

    def __init__(self, message: str, *, native_id: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.native_id = native_id
        if native_id is not None:
            self.details.setdefault("native_id", native_id)


class CircuitOpenError(ConnectorError):
    """The circuit breaker is open for this account after repeated failures.

    Five consecutive failures open it for ten minutes
    (`docs/connector-spec.md` §5.2). Classified as `QUOTA` rather than
    `TRANSIENT` on purpose: the correct response is to stop scheduling and come
    back later, which is exactly the quota response, whereas a transient
    classification would have the runtime retry into a source already known to
    be failing.
    """

    error_class = ConnectorErrorClass.QUOTA
    retryable = False

    def __init__(self, message: str, *, opens_until: float | None = None, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.opens_until = opens_until
        if opens_until is not None:
            self.details.setdefault("opens_until", opens_until)


class ConnectorConfigurationError(PermanentError):
    """The connector is misconfigured: missing params, unknown feed, bad slug.

    A `PermanentError` because retrying cannot fix it, but distinguished so the
    runtime can surface it to the operator who configured the account instead of
    filing it as a provider fault.
    """
