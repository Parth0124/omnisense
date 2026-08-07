"""`GET /health` and `GET /readyz` -- two endpoints that must not be one endpoint.

**`/health` is liveness and checks nothing.** It answers "is this process able to
serve a request", and the only honest way to answer that is to return. It
deliberately touches no dependency: a liveness probe that fails when PostgreSQL
is unreachable causes Kubernetes to *kill and reschedule* every API replica
during a database blip, turning a degradation into an outage and removing the
capacity that would have served the endpoints not backed by PostgreSQL.

**`/readyz` is readiness and checks everything.** It answers "should traffic be
routed here", so it probes each datastore and reports per-dependency status.
Failing readiness removes the replica from the load balancer and leaves it
running, which is the correct response to a dependency it cannot reach.

The probes run **concurrently**. Each is bounded at
`HEALTH_PROBE_TIMEOUT_SECONDS` (5s, `backend/db/__init__.py`), so six serial
probes against a half-open network path would take 30s -- long enough for the
liveness probe sharing that deadline to fire and kill the pod, which is exactly
the failure `/readyz` exists to prevent. Gathered, the endpoint costs the slowest
single probe.

Which dependencies are fatal comes from `docs/architecture.md` §7.3 rather than
from a uniform rule: PostgreSQL down means nothing works, while Qdrant down means
keyword-only retrieval with reduced recall. So a Qdrant outage reports
`degraded`, stays in rotation, and lets the system serve the answers it still
can.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Final

from fastapi import APIRouter, Response

from backend.core.config import get_settings
from backend.db import HEALTH_PROBE_TIMEOUT_SECONDS
from backend.db.neo4j import check_neo4j
from backend.db.opensearch import check_opensearch
from backend.db.qdrant import check_qdrant
from backend.db.redis import check_redis
from backend.db.session import check_postgres

__all__ = ["REQUIRED_DEPENDENCIES", "router"]

router = APIRouter(tags=["health"])

REQUIRED_DEPENDENCIES: Final[frozenset[str]] = frozenset({"postgres"})
"""Dependencies whose absence makes the process unfit to serve traffic.

Only PostgreSQL. `docs/architecture.md` §7.3: it is the commit point and the
checkpoint store, so without it an investigation can neither run nor resume.
Everything else degrades -- Qdrant down is keyword-only retrieval, Neo4j down is
a report without graph context and a lower stated confidence. Marking those
required would take the whole API out of rotation to protect a feature that was
designed to survive their loss.
"""

_PROBES: Final[dict[str, Callable[[], Awaitable[bool]]]] = {
    "postgres": check_postgres,
    "redis": check_redis,
    "qdrant": check_qdrant,
    "neo4j": check_neo4j,
    "opensearch": check_opensearch,
}


@router.get("/health", summary="Liveness. Checks no dependency.")
async def health() -> dict[str, object]:
    """Return unconditionally. See the module docstring for why."""
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.observability.otel_service_name,
        "environment": settings.app.environment.value,
    }


@router.get("/readyz", summary="Readiness. Probes every dependency concurrently.")
async def readyz(response: Response) -> dict[str, object]:
    """Probe every datastore and report per-dependency status.

    Returns 503 when a *required* dependency is down, 200 otherwise -- including
    when an optional one is degraded, because a replica that can still answer
    keyword queries belongs in rotation.
    """
    started = time.perf_counter()

    async def probe(name: str, check: Callable[[], Awaitable[bool]]) -> tuple[str, bool, float]:
        begin = time.perf_counter()
        try:
            # Bounded a second time here as well as inside each client. The
            # per-probe timeout protects against a slow dependency; this
            # protects against a probe that fails to honour its own.
            async with asyncio.timeout(HEALTH_PROBE_TIMEOUT_SECONDS + 1):
                ok = await check()
        except Exception:  # noqa: BLE001 -- readiness never raises
            ok = False
        return name, ok, (time.perf_counter() - begin) * 1000

    results = await asyncio.gather(*(probe(n, c) for n, c in _PROBES.items()))

    checks = {
        name: {"status": "ok" if ok else "fail", "latency_ms": round(ms, 1)}
        for name, ok, ms in results
    }
    failed_required = [n for n, ok, _ in results if not ok and n in REQUIRED_DEPENDENCIES]
    degraded = [n for n, ok, _ in results if not ok and n not in REQUIRED_DEPENDENCIES]

    if failed_required:
        response.status_code = 503
        status = "unavailable"
    elif degraded:
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "checks": checks,
        "degraded": degraded,
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }
