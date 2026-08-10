"""Recomputes graph analytics on a schedule.

Two jobs that look unrelated and belong together: both are **batch computations
whose results are read from a property rather than computed on request**, and
both are wrong in the same way if they are allowed to drift.

`graph/analytics/` writes `pagerank_score` and `community_id` onto nodes so the
retrieval path reads a number instead of running an algorithm. A PageRank
computed before an entity absorbed forty new mentions is not detectably wrong --
it is a plausible number describing a graph that no longer exists. Same for a
cached forecast: it looks current, and nothing in it says when it was fitted.

So the invariant this worker maintains is `computed_at`. Every result it writes
carries the timestamp it was produced at, `services/graph_service.py` exposes
`analytics_are_stale` from it, and the UI can say "computed 6 hours ago" instead
of implying "now".

**Recompute is full, not incremental, and that is a deliberate current choice.**
Incremental PageRank is possible and considerably more code, and the graph is
well under the size where a full pass is expensive.
`graph/queries/cypher.stale_analytics_nodes` exists for the incremental path when
that stops being true; until then, a full pass is simpler and cannot drift from
the graph it describes.

**One replica does the work.** Two replicas running PageRank simultaneously
produce identical results and pay twice, and the writes interleave so
`computed_at` ends up describing neither pass cleanly. An advisory lock is the
right shape here -- unlike the scheduler, there is no way to partition the work,
so serialising is the whole answer.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Final

from backend.core.logging import get_logger
from workers.runtime.base_worker import PeriodicWorker, run_worker

if TYPE_CHECKING:  # pragma: no cover
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from backend.core.config import Settings
    from workers.runtime.health import DependencyProbe

__all__ = [
    "ANALYTICS_LOCK_KEY",
    "DEFAULT_INTERVAL_SECONDS",
    "AnalyticsWorker",
    "main",
]

logger = get_logger(__name__)

WORKER_NAME: Final = "analytics"

DEFAULT_INTERVAL_SECONDS: Final = 6 * 60 * 60.0
"""Six hours.

Matched to how fast the answer actually changes. Entity importance in a mention
graph moves over days, not minutes, and recomputing hourly would spend real
compute to produce the same ranking six times. The `computed_at` stamp is what
makes a six-hour cadence honest rather than hidden.
"""

ANALYTICS_LOCK_KEY: Final = 0x4F_53_41_4E
"""PostgreSQL advisory lock key ("OSAN"). One holder recomputes; the rest skip.

A constant rather than a hash of the worker name, because an advisory lock key
must be stable across deploys and a hash changes the moment somebody renames the
worker -- at which point two replicas hold different locks and both run.
"""

MIN_NODES_FOR_ANALYTICS: Final = 10
"""Below this, centrality and communities describe noise.

PageRank over six nodes returns six numbers that are arithmetically correct and
mean nothing, and a community detected among them is a coincidence. Refusing is
better than publishing a ranking nobody should read.
"""


class AnalyticsWorker(PeriodicWorker):
    """Recomputes graph centrality, communities and cached forecasts."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        graph_reader: Any | None = None,
        graph_writer: Any | None = None,
        forecast_service: Any | None = None,
        tenant_id: str = "default",
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
        settings: Settings | None = None,
        name: str = WORKER_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name, interval_seconds=interval_seconds, settings=settings, **kwargs
        )
        self._session_factory = session_factory
        self._graph_reader = graph_reader
        self._graph_writer = graph_writer
        self._forecast_service = forecast_service
        self._tenant_id = tenant_id
        self.analytics_runs = 0
        self.skipped_locked = 0

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        from backend.db.neo4j import check_neo4j
        from backend.db.session import check_postgres

        return {"postgres": check_postgres, "neo4j": check_neo4j}

    async def tick(self) -> None:
        """Take the lock, recompute, stamp. Skip cleanly if another replica holds it."""
        async with self._advisory_lock() as acquired:
            if not acquired:
                self.skipped_locked += 1
                logger.debug("analytics.skipped_another_replica_holds_the_lock")
                return
            await self._recompute_graph_analytics()

    # ------------------------------------------------------------ internals --

    def _advisory_lock(self):  # noqa: ANN202 -- an async context manager
        """`pg_try_advisory_lock`, released on exit.

        `try` rather than the blocking form: a replica that cannot get the lock
        should return immediately and tick again in six hours, not hold a
        connection open waiting for a pass that is already being done.
        """
        from contextlib import asynccontextmanager

        from sqlalchemy import text

        session_factory = self._session_factory
        key = ANALYTICS_LOCK_KEY

        @asynccontextmanager
        async def _lock():
            async with session_factory() as session:
                acquired = bool(
                    (
                        await session.execute(
                            text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
                        )
                    ).scalar()
                )
                try:
                    yield acquired
                finally:
                    if acquired:
                        # Released explicitly rather than relying on the session
                        # closing. A session returned to the pool keeps its
                        # backend alive, and a session-scoped advisory lock goes
                        # with the backend -- so the lock would outlive the work
                        # by however long the connection stays pooled.
                        await session.execute(
                            text("SELECT pg_advisory_unlock(:key)"), {"key": key}
                        )
                        await session.commit()

        return _lock()

    async def _recompute_graph_analytics(self) -> None:
        """Project the graph, score it, write the scores back."""
        if self._graph_reader is None or self._graph_writer is None:
            logger.debug("analytics.no_graph_configured")
            return

        from graph.analytics.centrality import pagerank, projection_from_rows
        from graph.analytics.communities import louvain

        edges = await self._graph_reader.edges(tenant_id=self._tenant_id)
        isolated = await self._graph_reader.isolated_entity_ids(tenant_id=self._tenant_id)
        projection = projection_from_rows(edges, isolated_nodes=isolated)

        if projection.size < MIN_NODES_FOR_ANALYTICS:
            logger.info(
                "analytics.graph_too_small",
                nodes=projection.size,
                minimum=MIN_NODES_FOR_ANALYTICS,
                reason="a ranking over this few nodes describes noise",
            )
            return

        centrality = pagerank(projection)
        if not centrality.converged:
            # Written anyway -- a non-converged PageRank is still a better
            # ranking than none -- but recorded, because the scores are less
            # trustworthy and there is no way to tell from the numbers.
            logger.warning(
                "analytics.pagerank_not_converged",
                iterations=centrality.iterations,
                nodes=centrality.node_count,
            )

        communities = louvain(projection)
        computed_at = datetime.now(UTC)

        await self._graph_writer.write_analytics(
            tenant_id=self._tenant_id,
            pagerank=centrality.scores,
            communities=communities.assignment,
            computed_at=computed_at,
        )
        self.analytics_runs += 1
        logger.info(
            "analytics.recomputed",
            nodes=projection.size,
            edges=projection.edge_count,
            communities=len(communities.communities),
            modularity=round(communities.modularity, 4),
            unassigned=len(communities.unassigned),
            converged=centrality.converged,
        )


def main() -> None:  # pragma: no cover -- process entry point
    from backend.db.session import get_sessionmaker

    run_worker(AnalyticsWorker(session_factory=get_sessionmaker()))


if __name__ == "__main__":  # pragma: no cover
    main()
