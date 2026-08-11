"""Router aggregation and versioning.

Everything the product exposes lives under `/api/v1` (`docs/api-reference.md`
§3). The version is in the path rather than a header because it has to survive a
copy-pasted URL, a browser address bar and a log line -- a header-versioned API
whose URL is shared loses its version somewhere in that chain.

`/health` and `/readyz` are the deliberate exception: they are mounted at the
root, unversioned. A liveness probe is configured once in a deployment manifest
and must keep working across an API version bump; pointing it at `/api/v1/health`
would mean editing the manifest to ship `v2`, and forgetting to is an outage.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.v1 import (
    agents,
    connectors,
    graph,
    health,
    investigations,
    projects,
    reports,
    signals,
    stream,
)

__all__ = ["API_V1_PREFIX", "api_router"]

API_V1_PREFIX = "/api/v1"

api_router = APIRouter()

# Unversioned, at the root. See the module docstring.
api_router.include_router(health.router)

v1 = APIRouter(prefix=API_V1_PREFIX)

# Resource routers are attached here as they land. Each one is mounted
# explicitly rather than discovered, so a half-finished module cannot become
# reachable by being saved to disk -- the same reasoning as the explicit
# connector registration in `connectors/__init__.py`.
#
# Order matters for one pair only: `stream` declares
# `/investigations/{id}/stream` and `investigations` declares
# `/investigations/{id}`. Starlette matches in registration order, and the
# streaming path is more specific, so it is mounted first -- otherwise
# `/investigations/{investigation_id}` would capture `abc/stream` on some path
# shapes and answer a stream request with a JSON detail body.
v1.include_router(stream.router)
v1.include_router(investigations.router)
v1.include_router(signals.router)
v1.include_router(reports.router)
v1.include_router(graph.router)
v1.include_router(connectors.router)
v1.include_router(projects.router)
v1.include_router(agents.router)

api_router.include_router(v1)
