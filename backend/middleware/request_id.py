"""Correlation-id ingress: accept one, or mint one, and echo it on the way out.

`docs/observability.md` §2.3 calls correlation propagation "the single most
important piece of plumbing in this document", and this module is step 1 of it.
One id enters here and is carried, unchanged, through every log line the request
produces, into the Kafka envelope of every event it emits, into the worker that
consumes that event, and back out on the response header the client keeps. When
someone reports "my investigation returned nothing at 09:14", that id is the only
thing that turns a hundred thousand log lines into the eleven that describe their
request.

Three decisions, each preventing a specific failure.

**A client-supplied id is validated, not trusted.** §2.3 step 1: *"It never trusts
an arbitrary-length client string into log fields."* The value lands in every log
record, in a response header, and in `instance`-adjacent problem context, so an
unvalidated one is a log-injection primitive -- a newline in it splits one JSON
log line into two, and the second can be shaped to look like a record the system
wrote itself. It is also a cardinality bomb for any metric or index that groups
by it. So the header is accepted only when it is a well-formed UUID or ULID, and
otherwise silently replaced. Silently, because the request is not wrong in any way
the caller can fix by retrying, and a `400` here would break clients that send a
perfectly good id in a shape we did not anticipate.

**It is bound with a `ContextVar`, not passed as an argument.** Every log line in
the request's lifetime has to carry it, including lines written eight frames deep
in `services/` and `connectors/`, and threading a parameter through those layers
would put an observability concern in every signature. `correlation_scope()`
(`backend/core/logging.py`) sets the var and restores it via its token, so
concurrent requests on one event loop cannot see each other's id -- a plain
module global would give the last request to arrive ownership of every log line
the others were still writing.

**It is written by wrapping `send`, so it survives an error.** A response header
added by a route dependency reaches only the responses that route produced;
`401`, `404`, `422` and `500` are all built by the handlers in
`backend/api/errors.py`, which never see the route. Those are precisely the
responses a caller quotes when they file a bug, so they are the ones that most
need the id. Wrapping the ASGI `send` catches every response the application can
emit, including one from a middleware above the router.

Implemented as raw ASGI rather than `BaseHTTPMiddleware` deliberately.
`BaseHTTPMiddleware` runs the downstream app in a separate anyio task, which both
breaks `ContextVar` propagation back to the caller and buffers streaming
responses -- and `backend/api/v1/stream.py` is an SSE endpoint whose whole
contract is that events arrive as they happen (`docs/api-reference.md` §5).
"""

from __future__ import annotations

import re
import uuid
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.core.logging import correlation_scope, get_logger

__all__ = ["REQUEST_ID_HEADER", "RequestIdMiddleware", "is_acceptable_request_id"]

logger = get_logger(__name__)

REQUEST_ID_HEADER: Final = "x-request-id"
"""Lowercase because ASGI headers are byte-lowercased by the server, always."""

_UUID_PATTERN: Final = re.compile(
    r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?"
    r"[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
)
"""UUID with or without dashes.

Both spellings are accepted because both are produced by things that talk to this
API: `models/orm/investigation.py` ids are dashed, `new_correlation_id()` in
`backend/core/logging.py` is a bare hex, and a browser's `crypto.randomUUID()` is
dashed. Rejecting one of them would silently discard a perfectly good id and
break the chain at its first hop.
"""

_ULID_PATTERN: Final = re.compile(r"^[0-7][0-9ABCDEFGHJKMNPQRSTVWXYZ]{25}$")
"""Crockford base32, 26 characters, first character bounded to keep the 48-bit
timestamp in range. `docs/observability.md` §2.3 shows ULIDs in the envelope, so
a client that already mints them must be able to hand one in."""


def is_acceptable_request_id(value: str) -> bool:
    """Whether a client-supplied id may be adopted as the correlation id.

    Shape only. There is nothing to authenticate here -- a correlation id is not
    a credential and grants nothing -- so the check exists purely to bound what
    can reach a log field: a fixed alphabet and a fixed length, which between
    them exclude newlines, control characters, ANSI escapes and unbounded values.
    """
    return bool(_UUID_PATTERN.match(value) or _ULID_PATTERN.match(value))


class RequestIdMiddleware:
    """Bind a correlation id for the request and echo it on the response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Lifespan and websocket scopes have no headers and no response to
        # decorate. Passing them straight through is not an optimization: reading
        # `scope["headers"]` on a lifespan scope raises `KeyError`, which would
        # make the application fail to start.
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = _header(scope, REQUEST_ID_HEADER)
        if supplied is not None and is_acceptable_request_id(supplied):
            request_id = supplied
        else:
            if supplied is not None:
                # Logged without the value. Echoing a rejected string into a log
                # line would perform exactly the injection the rejection exists
                # to prevent.
                logger.info("api.request_id.rejected", length=len(supplied))
            request_id = uuid.uuid4().hex

        # `Request.state` reads `scope["state"]`, so this is how a handler and a
        # dependency see the id without re-parsing the header.
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                # Filtered rather than appended. A downstream handler that also
                # set the header would otherwise produce two `X-Request-ID`
                # values, and a client picking either one at random is worse than
                # a client that has none.
                message["headers"] = [
                    (name, value)
                    for name, value in headers
                    if name.lower() != REQUEST_ID_HEADER.encode("latin-1")
                ]
                message["headers"].append(
                    (REQUEST_ID_HEADER.encode("latin-1"), request_id.encode("latin-1"))
                )
            await send(message)

        # The scope restores the previous value via the ContextVar token on exit,
        # which matters because one worker task serves many requests in sequence:
        # an id left bound after a response would be attributed to the next
        # request that happened to reuse the task.
        with correlation_scope(request_id):
            await self.app(scope, receive, send_with_request_id)


def _header(scope: Scope, name: str) -> str | None:
    """Read one request header from an ASGI scope, decoded defensively.

    Latin-1 is the encoding HTTP header bytes are defined in (RFC 7230), and it
    cannot fail -- every byte is a valid code point. Using UTF-8 here would raise
    on a malformed header and turn a client's bad byte into a 500 from the
    middleware, before any handler could answer.
    """
    target = name.encode("latin-1")
    for key, value in scope.get("headers", ()):
        if key.lower() == target:
            return value.decode("latin-1").strip()
    return None
