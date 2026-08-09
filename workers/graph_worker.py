"""Writes the knowledge graph: step 7 of `docs/data-stores.md` §5.1.

Consumes `omnisense.signals.enriched` in its **own consumer group**, separate
from `workers/indexing_worker.py`. That separation is the point of the design:
Neo4j is not a hard dependency (`docs/architecture.md` §7.3), so a graph outage
must accrue lag on this group while indexing keeps up. Sharing a group would make
the two compete for partitions, and a Neo4j outage would stall vector and keyword
indexing too -- turning a degraded feature into a degraded system.

Why the Signal is re-read from PostgreSQL
------------------------------------------
The same reason `workers/indexing_worker.py` gives. `SignalEnrichedEvent` carries
identity and version, never content, so the row is the source of truth and the
row may have been reprocessed since the message was produced. Reading the topic's
copy would write a graph that disagrees with PostgreSQL in a way no reconciler
could detect, because every reconciler compares *against* PostgreSQL.

Resolution runs before the write, not after
--------------------------------------------
An extracted entity name is not an entity. "Acme", "Acme Corp" and "ACME
Corporation" are three surface forms of one company, and writing them as three
nodes creates duplicates that no later pass can merge without losing the edges
already attached to each. `graph/resolution/` maps a mention to a canonical id
first; the write is then a `MERGE` on that id, which is idempotent under
redelivery.

`graphed_at` is stamped after Neo4j acknowledges
--------------------------------------------------
Never before. A crash between the two leaves the column `NULL`, the sweep in
`docs/data-stores.md` §6 finds it and re-drives the message, and the `MERGE`
redoes the work harmlessly. Stamping first inverts that: the row claims to be
graphed, every reconciler looks for `NULL`, and the Signal is permanently absent
from the graph with nothing able to notice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from backend.core.logging import get_logger
from models.base import utcnow
from models.enums import EdgeType, EntityType
from services.events.consumer import ConsumedMessage
from services.events.schemas import SignalEnrichedEvent
from services.events.topics import TopicRole
from workers.runtime.base_worker import ConsumerWorker, run_worker
from workers.runtime.index_state import IndexState, stamp_index_state

if TYPE_CHECKING:  # pragma: no cover -- import-cycle and driver-weight avoidance
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from backend.core.config import Settings
    from graph.ingest.batcher import WriteBatcher
    from graph.ingest.writer import GraphWriter
    from workers.runtime.health import DependencyProbe

__all__ = ["WORKER_NAME", "GraphWorker", "main"]

logger = get_logger(__name__)

WORKER_NAME: Final = "graph"

DEFAULT_TENANT: Final = "default"
"""Until multi-tenancy lands (Phase 7), everything is written under one tenant.

Named rather than an empty string because the property is required on every node
and edge, and an empty tenant would match nothing on read -- so the graph would
fill up and every query would return nothing, which looks like a query bug.
"""

MIN_MENTION_SALIENCE: Final = 0.1
"""Below this, a mention is not worth an edge.

A `MENTIONS` edge for every entity the extractor glanced at makes the most
popular Topic node a hub with a hundred thousand inbound edges -- which slows
every traversal that crosses it and adds no information, because an edge that
weak never survives ranking anyway.
"""


class GraphWorker(ConsumerWorker):
    """Resolves entities and writes nodes, stubs and edges for one Signal.

    Every collaborator is injected, so the unit suite drives the real sequence --
    read row, resolve, batch, write, stamp -- against in-memory fakes with the
    ordering production has.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        writer: GraphWriter | None = None,
        batcher: WriteBatcher | None = None,
        resolver: Any | None = None,
        tenant_id: str = DEFAULT_TENANT,
        settings: Settings | None = None,
        name: str = WORKER_NAME,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=name,
            topics=[TopicRole.SIGNALS],
            settings=settings,
            **kwargs,
        )
        self._session_factory = session_factory
        self._writer = writer
        self._batcher = batcher
        self._resolver = resolver
        self._tenant_id = tenant_id

        if writer is None and batcher is None:
            # A graph worker with nowhere to write consumes the topic, commits
            # offsets and produces nothing -- and reports healthy batches the
            # whole time. Saying so at construction beats discovering it as "the
            # graph is empty" weeks later.
            logger.warning(
                "graph.no_writer_configured",
                reason="neither a GraphWriter nor a WriteBatcher was supplied",
                consequence="messages are consumed and committed but nothing is written",
            )

    # -------------------------------------------------------------- health --

    def readiness_probes(self) -> Mapping[str, DependencyProbe]:
        """PostgreSQL and Neo4j both, and both are required *for this worker*.

        Different from the API, where Neo4j is optional. The API can answer from
        vector and keyword hits with a lower stated confidence; this worker's
        entire output is graph writes, so a replica that cannot reach Neo4j has
        nothing to contribute and should leave rotation rather than consume
        messages it will fail.
        """
        from backend.db.neo4j import check_neo4j
        from backend.db.session import check_postgres

        return {"postgres": check_postgres, "neo4j": check_neo4j}

    # ------------------------------------------------------------ handling --

    async def handle(self, message: ConsumedMessage) -> None:
        """Write one Signal's graph contribution, then stamp it.

        Idempotent throughout: node ids are canonical and `MERGE`d, the signal
        stub is keyed by signal id, edges carry a deterministic `evidence_key`,
        and the stamp is an absolute assignment. Processing the same message
        twice yields one node per entity and the same timestamp.
        """
        event = message.envelope.payload_as(SignalEnrichedEvent)
        view = await self._load(event.signal_id)

        if view is None:
            # The event is published after the commit, so this is not a race with
            # the writer -- the row was erased in between. Erasure owns removing
            # the derived copies; re-driving would graph a Signal that no longer
            # exists.
            logger.info(
                "graph.signal_missing",
                signal_id=event.signal_id,
                reason="no signals row; erased after the enriched event was published",
            )
            return

        if not view.get("is_canonical", True):
            # `docs/signal-model.md` §4.3: a press release syndicated to six
            # platforms is one thing that happened. Graphing every duplicate
            # would inflate `source_count` sixfold, and `source_count` is the
            # cheapest anti-hallucination signal a report has.
            logger.debug("graph.duplicate_skipped", signal_id=event.signal_id)
            return

        batch = await self._build_batch(view)
        if batch is None:
            # Nothing extracted. Still stamped: the Signal *has* been processed
            # for the graph, and leaving the column NULL would make the sweeper
            # re-drive it forever, once per sweep, for a Signal that will never
            # produce an entity.
            await self._stamp(event.signal_id)
            return

        await self._write(batch)
        await self._stamp(event.signal_id)

    # ------------------------------------------------------------ internals --

    async def _load(self, signal_id: str) -> dict[str, Any] | None:
        """Read the Signal's entity extraction from the commit point."""
        from sqlalchemy import select

        from models.orm.signal import SignalRow

        async with self._session_factory() as session:
            row = (
                await session.execute(select(SignalRow).where(SignalRow.id == signal_id))
            ).scalar_one_or_none()
            if row is None:
                return None
            return {
                "id": row.id,
                "published_at": getattr(row, "published_at", None),
                "source": getattr(row, "source", None),
                "platform": getattr(row, "platform", None),
                "is_canonical": getattr(row, "is_canonical", True),
                "entities": getattr(row, "entities", None) or [],
            }

    async def _build_batch(self, view: Mapping[str, Any]) -> Any | None:
        """Resolve mentions to canonical ids and assemble one `GraphBatch`.

        Returns `None` when the Signal produced no entity worth writing -- which
        is common and not a failure. Plenty of signals are noise, and a batch
        containing only a stub costs a transaction to record that nothing was
        found.
        """
        from graph.ingest.writer import EdgeWrite, GraphBatch, NodeWrite, SignalStub

        mentions = [m for m in view["entities"] if isinstance(m, Mapping)]
        salient = [
            mention
            for mention in mentions
            if _as_float(mention.get("salience"), default=1.0) >= MIN_MENTION_SALIENCE
        ]
        if not salient:
            return None

        observed_at = view.get("published_at") or utcnow()
        nodes: list[NodeWrite] = []
        edges: list[EdgeWrite] = []
        seen_ids: set[str] = set()

        for mention in salient:
            resolved = await self._resolve(mention)
            if resolved is None:
                continue
            entity_id, entity_type, canonical_name, normalized_name = resolved

            if entity_id not in seen_ids:
                seen_ids.add(entity_id)
                nodes.append(
                    NodeWrite(
                        entity_type=entity_type,
                        id=entity_id,
                        tenant_id=self._tenant_id,
                        canonical_name=canonical_name,
                        normalized_name=normalized_name,
                        observed_at=observed_at,
                        aliases=_aliases_of(mention, canonical_name),
                        # One Signal contributes exactly one to an entity's
                        # source_count however many times it names it. Counting
                        # mentions instead would let a single article that
                        # repeats a company name forty times look like forty
                        # independent sources.
                        new_signal_count=1,
                    )
                )

            edges.append(
                EdgeWrite(
                    edge_type=EdgeType.MENTIONS,
                    source_label="Signal",
                    source_id=view["id"],
                    target_label=entity_type.value,
                    target_id=entity_id,
                    tenant_id=self._tenant_id,
                    valid_from=observed_at,
                    observed_at=observed_at,
                    confidence=_as_float(mention.get("confidence"), default=0.5),
                    extractor=str(mention.get("extractor") or "llm"),
                    source_signal_ids=[view["id"]],
                    new_evidence=1,
                    # Deterministic: the same Signal mentioning the same entity
                    # twice is one piece of evidence, and a redelivery must not
                    # increment `evidence_count` again.
                    evidence_key=f"{view['id']}:{entity_id}",
                    properties={"salience": _as_float(mention.get("salience"), default=0.0)},
                )
            )

        if not nodes:
            return None

        stub = SignalStub(
            id=view["id"],
            tenant_id=self._tenant_id,
            published_at=observed_at,
            source=str(view.get("source") or "unknown"),
            platform=str(getattr(view.get("platform"), "value", view.get("platform")) or "unknown"),
        )
        return GraphBatch(nodes=tuple(nodes), signals=(stub,), edges=tuple(edges))

    async def _resolve(
        self, mention: Mapping[str, Any]
    ) -> tuple[str, EntityType, str, str] | None:
        """Map one extracted mention to a canonical entity.

        Without a resolver this falls back to a deterministic id derived from the
        normalized name, which is correct-but-naive: it merges exact normalized
        matches and nothing else. That is the right fallback because it is
        *stable* -- the same name always produces the same id, so a redelivery
        still upserts one node rather than creating a second.
        """
        raw_name = mention.get("name") or mention.get("text")
        if not isinstance(raw_name, str) or not raw_name.strip():
            return None

        raw_type = mention.get("type") or mention.get("entity_type")
        entity_type = EntityType(raw_type) if isinstance(raw_type, str) else EntityType.UNKNOWN
        if entity_type is EntityType.UNKNOWN:
            # A label this build does not recognise has no node spec, so the
            # write would fail. Skipping the mention keeps the rest of the batch
            # writable rather than failing the whole Signal over one unknown type.
            logger.debug("graph.unknown_entity_type_skipped", raw_type=raw_type)
            return None

        if self._resolver is not None:
            resolved = await self._resolver.resolve(
                name=raw_name, entity_type=entity_type, tenant_id=self._tenant_id
            )
            if resolved is not None:
                return (
                    resolved.entity_id,
                    entity_type,
                    resolved.canonical_name,
                    resolved.normalized_name,
                )

        from graph.resolution.blocking import normalize_name

        normalized = normalize_name(raw_name)
        if not normalized:
            return None
        return _deterministic_id(entity_type, normalized), entity_type, raw_name.strip(), normalized

    async def _write(self, batch: Any) -> None:
        """Hand the batch to the batcher, or straight to the writer."""
        if self._batcher is not None:
            for node in batch.nodes:
                await self._batcher.add(node)
            for stub in batch.signals:
                await self._batcher.add(stub)
            for edge in batch.edges:
                await self._batcher.add(edge)
            return
        if self._writer is not None:
            outcome = await self._writer.apply(batch)
            logger.debug(
                "graph.batch_written",
                nodes=outcome.nodes_written,
                edges=outcome.edges_written,
            )

    async def _stamp(self, signal_id: str) -> None:
        """Record that Neo4j holds this Signal. Only after the write succeeded."""
        await stamp_index_state(self._session_factory, signal_id, IndexState.GRAPH)


def _deterministic_id(entity_type: EntityType, normalized_name: str) -> str:
    """A stable canonical id for an unresolved mention.

    UUIDv5 over (label, normalized name), matching the shape
    `graph/schema/nodes.py` documents for `id`. Deterministic so a redelivery --
    or a second Signal naming the same company -- lands on the same node instead
    of creating a twin.
    """
    import uuid

    namespace = uuid.UUID("6f2a1c9e-0000-5000-8000-6f6d6e697365")
    return f"ent_{uuid.uuid5(namespace, f'{entity_type.value}:{normalized_name}')}"


def _aliases_of(mention: Mapping[str, Any], canonical_name: str) -> list[str]:
    """Surface forms worth recording, excluding the canonical one itself."""
    raw = mention.get("aliases")
    if not isinstance(raw, (list, tuple)):
        return []
    return [
        alias
        for alias in raw
        if isinstance(alias, str) and alias and alias != canonical_name
    ][:20]


def _as_float(value: Any, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def build_default_worker(**overrides: Any) -> GraphWorker:
    """Compose the worker from settings. The process entry point's only wiring.

    Imports live inside the function so that importing this module does not pull
    in the Neo4j driver -- a test that only needs `GraphWorker` for its handler
    logic should not require the package.
    """
    from backend.db.neo4j import write_session
    from backend.db.session import get_sessionmaker

    from graph.ingest.writer import GraphWriter, runner_from_session_factory

    return GraphWorker(
        session_factory=overrides.pop("session_factory", None) or get_sessionmaker(),
        writer=overrides.pop("writer", None)
        or GraphWriter(runner_from_session_factory(write_session)),
        **overrides,
    )


def main() -> None:  # pragma: no cover -- process entry point
    run_worker(build_default_worker())


if __name__ == "__main__":  # pragma: no cover
    main()
