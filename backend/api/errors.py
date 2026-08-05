"""Exception handlers: every failure leaves as RFC 7807 `application/problem+json`.

One shape for every error the API can return, so a client writes one parser
instead of three. `OmniSenseError` already carries `status_code`, a stable `code`
and non-sensitive `details` (`backend/core/exceptions.py`), so the handler is a
translation rather than a decision -- an error raised deep in `services/` cannot
be re-classified by a route that has no idea what went wrong.

The validation handler is the one worth reading. FastAPI's default 422 body
**echoes the offending input**, and in this system that input can be ingested
third-party content: a malformed connector payload, a passage retrieved from
Reddit, a filter value an agent copied out of a document. Echoing it back turns
the error channel into a reflection surface for content we did not author. So the
handler reports the *location and rule* that failed and never the value.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.core.exceptions import OmniSenseError, RateLimitedError
from backend.core.logging import get_logger

__all__ = ["PROBLEM_CONTENT_TYPE", "install_exception_handlers"]

logger = get_logger(__name__)

PROBLEM_CONTENT_TYPE = "application/problem+json"


def _problem_response(
    problem: dict[str, Any], *, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=int(problem["status"]),
        content=problem,
        media_type=PROBLEM_CONTENT_TYPE,
        headers=headers,
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register every handler on `app`."""

    @app.exception_handler(OmniSenseError)
    async def _omnisense(request: Request, exc: OmniSenseError) -> JSONResponse:
        problem = exc.to_problem(instance=str(request.url.path))
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitedError) and exc.retry_after_seconds is not None:
            # A 429 without Retry-After tells a client to back off but not by how
            # much, and most clients respond by retrying immediately.
            headers["Retry-After"] = str(int(exc.retry_after_seconds))

        # 5xx is ours to fix, so it is logged with the cause; 4xx is the caller's
        # and would otherwise fill the log with other people's typos.
        if exc.status_code >= 500:
            logger.error(
                "api.error",
                code=exc.code,
                status=exc.status_code,
                path=request.url.path,
                exc_info=exc.cause or exc,
            )
        else:
            logger.info("api.client_error", code=exc.code, status=exc.status_code)
        return _problem_response(problem, headers=headers or None)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # `loc` and `msg` only. `input` is deliberately dropped -- see the module
        # docstring: it can be third-party content, and this is a response body.
        errors = [
            {
                "location": ".".join(str(part) for part in err.get("loc", ())),
                "rule": err.get("msg", ""),
            }
            for err in exc.errors()
        ]
        return _problem_response(
            {
                "type": "https://omnisense.dev/errors/validation_error",
                "title": "validation error",
                "status": 422,
                "detail": "The request did not match the expected schema.",
                "details": {"errors": errors},
                "instance": str(request.url.path),
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # 404s and 405s raised by the router itself, so they match the shape the
        # rest of the API uses rather than Starlette's `{"detail": ...}`.
        return _problem_response(
            {
                "type": "https://omnisense.dev/errors/http_error",
                "title": "http error",
                "status": exc.status_code,
                "detail": str(exc.detail),
                "instance": str(request.url.path),
            }
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # The catch-all says nothing about what broke. An unhandled exception is
        # by definition one we did not anticipate, so its message may contain
        # anything -- a connection string, a row, a fragment of a fetched page.
        logger.error("api.unhandled", path=request.url.path, exc_info=exc)
        return _problem_response(
            {
                "type": "https://omnisense.dev/errors/internal_error",
                "title": "internal error",
                "status": 500,
                "detail": "An unexpected error occurred.",
                "instance": str(request.url.path),
            }
        )
