"""Application factory and ASGI entrypoint (`backend.main:app`).

Thin by design. The gateway parses, authenticates, validates, delegates to
`services/` or `agents/`, and serialises. Business logic that appears here
belongs one layer down -- `backend/` is a *process*, and the same work has to be
reachable from a worker and a script that have no HTTP server.

Two things the lifespan gets right and a naive version gets wrong.

**Nothing connects at import.** Every client in `backend/db/` is a lazily-created
singleton, so importing this module opens no socket. That is what lets the test
suite build the app and exercise routes with no datastore running, and it means
an unreachable dependency degrades `/readyz` rather than crashing startup.

**Shutdown disposes, and disposal is ordered last-in-first-out.** Each
`dispose_*()` is awaited on the way down; skipping them leaks connections on
every reload, and in a worker that restarts on a schedule the leak is unbounded.
Disposal is also wrapped individually -- one client failing to close must not
prevent the other five from closing.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.errors import install_exception_handlers
from backend.api.router import api_router
from backend.core.config import Settings, get_settings
from backend.core.logging import configure_logging, get_logger

__all__ = ["app", "create_app"]

logger = get_logger(__name__)


async def _dispose_all() -> None:
    """Close every client that was lazily created, tolerating individual failures.

    Imported inside the function rather than at module scope so that importing
    `backend.main` does not pull in every datastore driver -- a test that only
    needs the route table should not need `neo4j` and `qdrant-client` installed.
    """
    from backend.db.neo4j import dispose_driver
    from backend.db.opensearch import dispose_opensearch
    from backend.db.qdrant import dispose_qdrant
    from backend.db.r2 import dispose_r2
    from backend.db.redis import dispose_redis
    from backend.db.session import dispose_engine

    disposers: list[tuple[str, Callable[[], Awaitable[None]]]] = [
        ("opensearch", dispose_opensearch),
        ("qdrant", dispose_qdrant),
        ("neo4j", dispose_driver),
        ("r2", dispose_r2),
        ("redis", dispose_redis),
        ("postgres", dispose_engine),
    ]
    for name, dispose in disposers:
        try:
            await dispose()
        except Exception:  # noqa: BLE001 -- one bad close must not block the rest
            logger.warning("shutdown.dispose_failed", client=name, exc_info=True)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging on the way up, dispose every client on the way down."""
    settings: Settings = get_settings()
    configure_logging()
    logger.info(
        "api.startup",
        environment=settings.app.environment.value,
        service=settings.observability.otel_service_name,
    )
    try:
        yield
    finally:
        await _dispose_all()
        logger.info("api.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application.

    A factory rather than a module-level singleton so a test can build an app
    against overridden settings without mutating global state, and so
    `create_app()` can be called twice in one process.
    """
    resolved = settings or get_settings()

    application = FastAPI(
        title="OmniSense",
        version="0.1.0",
        description="Autonomous multi-agent market intelligence.",
        lifespan=lifespan,
        # The generated docs describe the contract in `docs/api-reference.md`.
        docs_url="/docs",
        openapi_url="/openapi.json",
    )

    application.add_middleware(
        CORSMiddleware,
        # Exact origins from settings, never `*` -- `Settings` refuses to start a
        # staging or production process configured with a wildcard.
        allow_origins=resolved.app.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    install_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()
"""The ASGI app uvicorn serves: `uvicorn backend.main:app`."""
