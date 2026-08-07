"""W3C trace-context ingress: continue an incoming trace, or start one.

`docs/observability.md` §4 requires an HTTP server span per request and requires
that trace context survive every hop, including the asynchronous ones: *"The
producer injects W3C `traceparent` into the envelope (§2.3); the consumer extracts
it and continues the trace. Without this the asynchronous half of every pipeline
is a separate, unjoinable trace."* An investigation is almost entirely
asynchronous half -- API to Kafka to worker to orchestrator -- so a request that
starts a fresh trace instead of continuing the caller's produces a set of
disconnected fragments that no backend can stitch together after the fact.

This module does the part that must happen at the edge, and only that part:
parse, validate, decide sampling, expose the ids to the request, and echo the
outgoing context.

**It does not export spans, and that is a stated gap rather than an oversight.**
The OpenTelemetry SDK is not in `requirements.txt`; `backend/core/telemetry.py`,
which `docs/observability.md` §4 names as the place the exporter is configured,
is still a stub. Adding the SDK here would put an exporter, a sampler and a
processor in a middleware module, which is exactly the layering
`backend/core/telemetry.py` exists to avoid. What this module *does* provide is
the contract everything else depends on: `request.state.trace_id` and
`request.state.span_id` are populated on every request, so log lines, problem
documents and resource payloads carry the same ids the eventual exporter will
emit, and turning tracing on later changes no call site.

Two decisions worth their lines.

**Validation is strict, and an invalid header starts a new trace rather than
failing the request.** `traceparent` is `00-<32 hex>-<16 hex>-<2 hex>`, and an
all-zero trace or span id is explicitly invalid in the specification. A malformed
value is a bug in some upstream proxy, not something the caller can act on, so
answering `400` would take an endpoint down over a header nobody reads. Adopting
it unvalidated is worse: a zero trace id collides with every other zero trace id
in the backend, silently merging unrelated requests into one enormous trace.

**Sampling is deterministic in the trace id, not random.** `OTEL_SAMPLE_RATIO`
decides whether a *trace* is recorded, and the decision has to be the same in
every service that sees it -- otherwise the API records a span, the worker
declines, and the resulting trace is missing its middle. Hashing the trace id
gives every participant the same answer without any coordination, which is what
"head-based sampling" means in §4. A `random()` call per hop would instead
produce traces sampled at the API and dropped at the worker, at exactly the rate
that makes the gap hard to notice.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.core.config import get_settings
from backend.core.logging import get_logger

__all__ = [
    "TRACEPARENT_HEADER",
    "TraceContext",
    "TracingMiddleware",
    "format_traceparent",
    "parse_traceparent",
]

logger = get_logger(__name__)

TRACEPARENT_HEADER: Final = "traceparent"

_TRACEPARENT_PATTERN: Final = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<span_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)
"""Lowercase hex only, per the W3C trace-context grammar.

Uppercase is invalid in the specification and is not normalized here on purpose:
a peer emitting uppercase is emitting something no conforming backend will join,
and quietly repairing it hides a real interoperability bug behind a trace that
looks fine.
"""

_INVALID_TRACE_ID: Final = "0" * 32
_INVALID_SPAN_ID: Final = "0" * 16

_SAMPLED_FLAG: Final = 0x01


class TraceContext:
    """The trace ids in force for one request.

    A small class rather than three loose strings because they travel together
    everywhere they go -- onto `request.state`, into the Kafka envelope, into a
    log line -- and a tuple of three hex strings is trivially reordered at a call
    site into something that still typechecks and is silently wrong.
    """

    __slots__ = ("parent_span_id", "sampled", "span_id", "trace_id")

    def __init__(
        self,
        *,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        sampled: bool,
    ) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.sampled = sampled

    def traceparent(self) -> str:
        """This request's outgoing context: same trace, *this* span as the parent."""
        return format_traceparent(self.trace_id, self.span_id, sampled=self.sampled)


def parse_traceparent(value: str | None) -> tuple[str, str, bool] | None:
    """Parse a `traceparent` into `(trace_id, parent_span_id, sampled)`.

    Returns `None` for anything unusable, which the caller reads as "start a new
    trace". Version `ff` is rejected outright (the specification forbids it);
    other unknown versions are accepted for their first four fields, which is what
    the specification's forward-compatibility rule requires -- a future version
    appends fields rather than changing the ones already defined.
    """
    if not value:
        return None
    match = _TRACEPARENT_PATTERN.match(value.strip().split(",")[0].strip())
    if match is None:
        return None
    if match["version"] == "ff":
        return None
    if match["trace_id"] == _INVALID_TRACE_ID or match["span_id"] == _INVALID_SPAN_ID:
        return None
    return match["trace_id"], match["span_id"], bool(int(match["flags"], 16) & _SAMPLED_FLAG)


def format_traceparent(trace_id: str, span_id: str, *, sampled: bool) -> str:
    """Render a version-00 `traceparent` header value."""
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def _new_id(byte_length: int) -> str:
    """A random trace or span id.

    `secrets` rather than `random`: the `random` module is seeded from a
    predictable source, and two processes started in the same second from the
    same image can mint identical ids -- which merges two unrelated traces in the
    backend and is essentially impossible to diagnose from the resulting
    waterfall.
    """
    return secrets.token_hex(byte_length)


def _is_sampled(trace_id: str, ratio: float) -> bool:
    """Head-based sampling decision, a pure function of the trace id.

    The low 8 bytes of a SHA-256 of the trace id are compared against the ratio
    scaled over the same range. Hashing rather than reading the id's own bytes
    matters when the upstream mints ids with structure -- a timestamp prefix, a
    fixed host suffix -- which would otherwise correlate the sampling decision
    with time or with the host and sample one machine at 100% and another at 0%.
    """
    if ratio >= 1.0:
        return True
    if ratio <= 0.0:
        return False
    digest = hashlib.sha256(trace_id.encode("ascii")).digest()[-8:]
    return int.from_bytes(digest, "big") < int(ratio * float(1 << 64))


class TracingMiddleware:
    """Continue or start a trace, and publish its ids on the request.

    Raw ASGI for the same reasons as `backend/middleware/request_id.py`:
    `BaseHTTPMiddleware` would buffer the SSE stream in
    `backend/api/v1/stream.py` and would break `ContextVar` propagation.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = parse_traceparent(_header(scope, TRACEPARENT_HEADER))
        if incoming is None:
            trace_id, parent_span_id = _new_id(16), None
            sampled = _is_sampled(trace_id, get_settings().observability.otel_sample_ratio)
        else:
            trace_id, parent_span_id, sampled = incoming
            # The upstream's decision is honoured rather than re-evaluated. A
            # participant that re-samples produces a trace with holes in it, and
            # the specification makes the sampled flag a property of the trace
            # rather than of each hop for exactly that reason.

        context = TraceContext(
            trace_id=trace_id,
            span_id=_new_id(8),
            parent_span_id=parent_span_id,
            sampled=sampled,
        )

        state = scope.setdefault("state", {})
        state["trace_context"] = context
        state["trace_id"] = context.trace_id
        state["span_id"] = context.span_id

        async def send_with_traceparent(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                message["headers"] = [
                    (name, value)
                    for name, value in headers
                    if name.lower() != TRACEPARENT_HEADER.encode("latin-1")
                ]
                message["headers"].append(
                    (
                        TRACEPARENT_HEADER.encode("latin-1"),
                        context.traceparent().encode("latin-1"),
                    )
                )
            await send(message)

        await self.app(scope, receive, send_with_traceparent)


def _header(scope: Scope, name: str) -> str | None:
    """Read one request header from an ASGI scope. See `request_id._header`."""
    target = name.encode("latin-1")
    for key, value in scope.get("headers", ()):
        if key.lower() == target:
            return value.decode("latin-1").strip()
    return None
