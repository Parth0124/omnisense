"""Structured logging: one pipeline for the API, every worker and every script.

`docs/observability.md` §1 states the constraint this module exists to satisfy:
the four observability pillars share exactly one join key, `correlation_id`, and
a pillar that cannot be joined on it is decoration. A user's question becomes an
HTTP request, a Kafka event, an ingestion job, an enrichment job and ten agent
steps across three processes. Without one id threaded through all of it, that is
five unrelated log streams and no way to explain a failed investigation.

Three decisions are encoded here.

**The correlation id lives in a `ContextVar`, not in a parameter.** Threading it
through every call signature is the version of this that rots -- one function
forgets, and the chain silently breaks at exactly the boundary you needed. A
context variable is inherited by every task spawned from the current context, so
`asyncio.TaskGroup` fan-out keeps the id for free, and `bind_correlation_id()`
at the process edge (`backend/middleware/request_id.py` for HTTP,
`workers/runtime/base_worker.py` for a consumed envelope) is the only place that
has to remember.

**Redaction runs last, and it is a backstop, not a licence.** structlog will
serialize whatever it is handed, and the things it gets handed include connector
credential dicts, `Authorization` headers echoed from a failed request, and
provider error payloads. `docs/coding-standards.md` §2.8 forbids passing them in;
this processor is what stands between a mistake and a credential sitting in a log
aggregator forever. It is **key-based**: a secret passed as the *value* of an
innocuous key, or embedded in an exception message, is not caught. That is the
reason `POSTGRES_ECHO_SQL` is rejected in production by `backend/core/config.py`
-- SQLAlchemy logs a fully-formatted statement string, and no key-based filter
can find a password inside it.

**Standard-library logging is routed through the same processor chain.** uvicorn,
SQLAlchemy, aiokafka and httpx all log through `logging`. If they bypassed this
pipeline, half of production output would be unstructured lines missing the
correlation id -- and it is invariably the third-party line that explains the
outage.

Layer note: this is the **L1k kernel** (`docs/architecture.md` §6.1) -- importable
by `services/`, `agents/`, `workers/`, `backend/api/` and `scripts/`, but never by
`connectors/`.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import structlog
from structlog.types import EventDict, Processor, WrappedLogger

from backend.core.config import LogFormat, Settings, get_settings

__all__ = [
    "REDACTED",
    "UNBOUND_CORRELATION_ID",
    "bind_correlation_id",
    "clear_correlation_id",
    "configure_logging",
    "correlation_scope",
    "get_correlation_id",
    "get_logger",
    "new_correlation_id",
    "redact_processor",
    "reset_logging",
]


REDACTED = "***redacted***"
"""Replacement value for a sensitive field.

The value is replaced rather than the key dropped. Dropping makes a redacted
field indistinguishable from one that was never set, which turns "did the
connector send an auth header?" into an unanswerable question during an incident.
A visible marker keeps the shape of the record and makes the redaction auditable.
"""

UNBOUND_CORRELATION_ID = "-"
"""What `correlation_id` reads as outside any bound scope.

The field is always present so that a log query can filter on it without
worrying about missing keys. A `-` means the record was emitted outside a request
or message scope -- process startup, a migration, a scheduler tick that has not
begun a job -- which is information, not an error.
"""

MAX_REDACTION_DEPTH = 8
"""Recursion ceiling for the redactor.

Bounds both pathological nesting and self-referential structures. A structure
deeper than this is not something anyone reads out of a log line anyway; it gets
replaced with a marker rather than silently truncated.
"""

DEPTH_ELIDED = "***depth-limit***"

_SENSITIVE_KEY_RE = re.compile(
    r"password|passwd|secret|token|credential|authorization|cookie"
    r"|api[_-]?key|_key$|^key$",
    re.IGNORECASE,
)
"""Keys whose values never reach an aggregator.

The union of `docs/observability.md` §2.1 (`*_secret`, `*_token`, `*_key`,
`*password*`, `authorization`, `cookie`) and `docs/security-and-privacy.md` §4.2.

`_key$` deliberately over-matches: `idempotency_key`, `partition_key` and
`cache_key` are redacted along with `api_key` and `encryption_key`. That trade is
made on purpose -- a redacted idempotency key costs one debugging session, a
leaked encryption key costs a credential rotation across every connector. Note
that the plural `keywords` (a real Signal field) does not match, because the
anchor requires the key to *end* in `_key`.
"""

_PASSTHROUGH_KEYS = frozenset({"_record", "_from_structlog", "exc_info", "stack_info"})
"""Processor plumbing the redactor must not touch.

`_record` is a `logging.LogRecord` and `exc_info` is a `(type, value, traceback)`
tuple; both are consumed by `ProcessorFormatter` and the renderer downstream and
must arrive intact.
"""

_LIBRARY_LEVELS: dict[str, int] = {
    # `docker/entrypoints/api.sh` passes `--no-access-log`; this is the belt to
    # that braces, because a stray `uvicorn --reload` in development would
    # otherwise emit a second, less useful line per request.
    "uvicorn.access": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "aiokafka": logging.WARNING,
    "asyncio": logging.WARNING,
    "neo4j": logging.WARNING,
    "opensearch": logging.WARNING,
}
"""Third-party loggers pinned below the application level.

These are chatty at INFO and say nothing an operator acts on. Deliberately
absent: `sqlalchemy.engine`. `echo=True` works by setting that logger to INFO
itself, so pinning it here would make `POSTGRES_ECHO_SQL` silently do nothing.
"""

_HANDLER_OWNERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "gunicorn", "gunicorn.error")
"""Loggers that install their own handlers and must be stripped of them.

uvicorn configures handlers at startup from its own dict-config. Left in place
they duplicate every line: once in our format, once in uvicorn's.
"""


_correlation_id: ContextVar[str] = ContextVar(
    "omnisense_correlation_id", default=UNBOUND_CORRELATION_ID
)

_configured = False


# --------------------------------------------------------------------------- #
# Correlation id
# --------------------------------------------------------------------------- #


def new_correlation_id() -> str:
    """Mint an id for a chain that has none.

    A bare uuid4 hex rather than a ULID: the only place a ULID's sortability
    would matter is the request-id middleware, and that accepts a well-formed
    client-supplied ULID anyway (`docs/observability.md` §2.3 step 1). Adding a
    dependency to generate ids we mostly receive is not worth it.
    """
    return uuid.uuid4().hex


def get_correlation_id() -> str:
    """Return the id bound to the current context, or `UNBOUND_CORRELATION_ID`.

    Exposed as a plain accessor rather than left inside structlog's context
    because non-logging code needs the value too: the Kafka envelope builder in
    `services/events/` has to copy it into every message, and reaching into a
    logging library's private context dict for it would be fragile.
    """
    return _correlation_id.get()


def bind_correlation_id(correlation_id: str | None = None) -> str:
    """Bind an id to the current context for the remainder of its life.

    Called at a process edge: the HTTP middleware on request ingress, the worker
    runtime on message consumption. A worker that *mints* an id for a message it
    received has broken the chain -- pass the envelope's `correlation_id` in.

    Returns the bound value so a caller can echo it on the response.
    """
    value = correlation_id or new_correlation_id()
    _correlation_id.set(value)
    return value


def clear_correlation_id() -> None:
    """Reset the current context back to unbound.

    Needed by the worker runtime: a consumer loop reuses one task across many
    messages, so an id left bound after a handler returns would be attributed to
    the *next* message.
    """
    _correlation_id.set(UNBOUND_CORRELATION_ID)


@contextmanager
def correlation_scope(correlation_id: str | None = None) -> Iterator[str]:
    """Bind an id for the duration of a block, restoring the previous one after.

    Preferred over `bind_correlation_id()` wherever the scope is lexical -- a
    scheduled job, a script, a test -- because it restores via the `ContextVar`
    token and therefore nests correctly.
    """
    value = correlation_id or new_correlation_id()
    token = _correlation_id.set(value)
    try:
        yield value
    finally:
        _correlation_id.reset(token)


# --------------------------------------------------------------------------- #
# Processors
# --------------------------------------------------------------------------- #


def _add_correlation_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Stamp every record with the current correlation id.

    `setdefault` rather than assignment: a caller that passes an explicit
    `correlation_id=` is talking about some *other* chain -- a replayed DLQ
    message, a parent investigation -- and overwriting it would be a lie.
    """
    event_dict.setdefault("correlation_id", _correlation_id.get())
    return event_dict


def _make_static_field_processor(service: str, environment: str) -> Processor:
    """Build the processor that adds `service` and `env` (§2.2 required fields).

    Closed over at configuration time so that `get_settings()` is not called once
    per log record.
    """

    def add_static_fields(
        logger: WrappedLogger, method_name: str, event_dict: EventDict
    ) -> EventDict:
        event_dict.setdefault("service", service)
        event_dict.setdefault("env", environment)
        return event_dict

    return add_static_fields


def _add_error_fields(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Split an exception into the `error.type` / `error.message` fields of §2.2.

    Read non-destructively: `exc_info` stays in the dict for whichever renderer
    is downstream to format the traceback. The point of the separate fields is
    that alerting on an exception *class* is a one-line query, whereas grepping a
    rendered traceback for the same thing is not.
    """
    exc = event_dict.get("exc_info")
    if exc is True:
        exc = sys.exc_info()

    if isinstance(exc, BaseException):
        exc_type: type[BaseException] | None = type(exc)
        exc_value: BaseException | None = exc
    elif isinstance(exc, tuple) and len(exc) == 3:
        exc_type, exc_value = exc[0], exc[1]
    else:
        return event_dict

    if exc_type is None:
        return event_dict

    event_dict.setdefault("error", {"type": exc_type.__name__, "message": str(exc_value)})
    return event_dict


def _is_sensitive(key: object) -> bool:
    """Whether a mapping key names something that must never be logged."""
    return isinstance(key, str) and _SENSITIVE_KEY_RE.search(key) is not None


def _redact(value: Any, depth: int) -> Any:
    """Recursively replace sensitive values inside an arbitrary structure.

    Recurses through dicts, lists and tuples because credentials do not arrive at
    the top level -- they arrive as `connector={"credentials": {"api_key": ...}}`
    or as a list of header pairs. Tuples are rebuilt as tuples so that anything
    downstream expecting a tuple still gets one.
    """
    if depth >= MAX_REDACTION_DEPTH:
        return DEPTH_ELIDED

    if isinstance(value, dict):
        return {
            key: REDACTED if _is_sensitive(key) else _redact(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, depth + 1) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact(item, depth + 1) for item in value)
    return value


def redact_processor(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Drop the value of any sensitive key, at any nesting depth.

    Runs **last** in the chain, so it also sees fields added by
    `merge_contextvars` and by every processor above it. Placing it earlier would
    leave a hole exactly the width of the remaining processors.
    """
    for key in list(event_dict):
        if key in _PASSTHROUGH_KEYS:
            continue
        if _is_sensitive(key):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _redact(event_dict[key], 1)
    return event_dict


def _shared_processors(service: str, environment: str) -> list[Processor]:
    """The chain applied to structlog-native *and* stdlib records alike.

    Order matters at the end: `redact_processor` is last so that nothing added
    along the way -- including fields merged in from the context -- escapes it.

    `structlog.stdlib.filter_by_level` is deliberately **not** in here even
    though it belongs at the front of the native chain. `ProcessorFormatter`
    calls a `foreign_pre_chain` with `logger=None`, and `filter_by_level` would
    immediately raise `AttributeError: 'NoneType' object has no attribute
    'isEnabledFor'` on the first uvicorn line. Stdlib records have already been
    level-filtered by `logging` anyway.
    """
    return [
        structlog.contextvars.merge_contextvars,
        _add_correlation_id,
        _make_static_field_processor(service, environment),
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _add_error_fields,
        redact_processor,
    ]


def _renderer_chain(log_format: LogFormat) -> list[Processor]:
    """The final, format-specific processors, run inside `ProcessorFormatter`."""
    if log_format is LogFormat.JSON:
        # `format_exc_info` collapses the traceback into one `exception` string,
        # which keeps a JSON log line to exactly one line. A multi-line traceback
        # in JSON output is what breaks line-oriented log shippers.
        return [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]

    # ConsoleRenderer formats `exc_info` itself, with colour and alignment, so
    # `format_exc_info` must NOT run ahead of it.
    return [structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def configure_logging(settings: Settings | None = None, *, force: bool = False) -> None:
    """Configure structlog and the stdlib root logger. Idempotent.

    Called once per process, from the FastAPI lifespan and from each worker's
    entrypoint. Idempotent because those two are not mutually exclusive -- a
    script that imports the app and then starts a worker would otherwise stack a
    second handler on the root logger and duplicate every line.

    Args:
        settings: Injected for tests. Defaults to the process-wide settings.
        force: Reconfigure even if this has already run. Only tests need it.
    """
    global _configured
    if _configured and not force:
        return

    settings = settings or get_settings()
    level = logging.getLevelNamesMapping()[settings.app.log_level.value]
    shared = _shared_processors(
        service=settings.observability.otel_service_name,
        environment=settings.app.environment.value,
    )

    structlog.configure(
        # Every structlog record is handed to stdlib logging rather than written
        # directly, so that application and third-party output share one handler,
        # one destination and one format. Two writers to stdout interleave badly
        # under concurrency and produce torn lines.
        processors=[
            # First, so a suppressed DEBUG record never pays for the timestamp,
            # the traceback formatting or the redaction walk.
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        # Applied to records that did *not* come from structlog, so a SQLAlchemy
        # or uvicorn line ends up with the same timestamp, level, service and
        # correlation id as ours.
        foreign_pre_chain=shared,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *_renderer_chain(settings.app.log_format),
        ],
    )

    # stdout only. No files, no rotation (`docs/observability.md` §2.1): the
    # container runtime owns log shipping, and a process that writes its own log
    # files in a container is a process that fills a disk nobody is watching.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    for name in _HANDLER_OWNERS:
        library_logger = logging.getLogger(name)
        library_logger.handlers.clear()
        library_logger.propagate = True

    for name, library_level in _LIBRARY_LEVELS.items():
        logging.getLogger(name).setLevel(library_level)

    _configured = True


def reset_logging() -> None:
    """Undo `configure_logging()`. For tests only.

    Without this a test that reconfigures logging leaks its handler into every
    subsequent test in the session, and assertions about captured output start
    depending on test order.
    """
    global _configured
    structlog.reset_defaults()
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    _configured = False


def get_logger(name: str | None = None) -> Any:
    """Return a bound logger. The accessor every other module uses.

    Safe at module scope: structlog returns a lazy proxy that resolves the
    configuration on its first *call*, not on creation, so
    `logger = get_logger(__name__)` at the top of a module does not force
    `configure_logging()` to have run at import time.

    Typed as `Any` because structlog's proxy is only a `BoundLogger` after it
    resolves; annotating it as one would be a lie that mypy cannot check and that
    would make every `logger.info("event", custom_field=1)` call site look wrong.
    """
    return structlog.get_logger(name)
