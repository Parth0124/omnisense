"""ORM table for Signal metadata.

PostgreSQL is the **commit point** of the ingestion pipeline and the source of
truth from which every derived store can be rebuilt (`docs/data-stores.md` §3.1,
§6). That shapes what this table does and does not hold:

**Holds** — every canonical Signal field except the ones that live better
elsewhere, plus the index-state columns that make reconciliation possible.

**Never holds** — the raw payload body (R2; a `TEXT` column of scraped HTML turns
the write-ahead log into a bandwidth problem), embedding vectors (Qdrant), or the
entity graph (Neo4j).

The `indexed_vector_at` / `indexed_keyword_at` / `graphed_at` columns are the
mechanism behind the rebuild guarantee. Because five stores are written without a
distributed transaction, PostgreSQL records *what it believes* has been indexed
where. A sweeper compares that against reality and re-drives the difference, so a
crash between the Postgres commit and the Qdrant upsert self-heals instead of
leaving a Signal permanently unsearchable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from models.enums import Platform, SignalStatus, SourceCategory
from models.orm.base import Base, JSONVariant, TolerantEnumType
from models.orm.mixins import TenantMixin, TimestampMixin

__all__ = ["SignalRow"]


class SignalRow(Base, TimestampMixin, TenantMixin):
    """One row per Signal. Primary key is the derived `Signal.id`.

    Deliberately **not** soft-deletable: a deletion request must actually remove
    the text, and a tombstone carrying `content_text` would defeat that.
    """

    __tablename__ = "signals"

    # -- identity ----------------------------------------------------------
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    """The derived id (`sig_` + uuid5 hex). Never database-assigned."""

    native_id: Mapped[str] = mapped_column(String(512), nullable=False)
    """The platform's own item id. With `platform`, uniquely identifies the source item."""

    # -- source ------------------------------------------------------------
    source: Mapped[SourceCategory] = mapped_column(
        TolerantEnumType(SourceCategory, 32), nullable=False
    )
    platform: Mapped[Platform] = mapped_column(
        TolerantEnumType(Platform, 32), nullable=False, index=True
    )
    url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # -- author (flattened) ------------------------------------------------
    # Flattened rather than JSON because `author_platform_id` is joined against
    # for per-author aggregates and filtered in retrieval; a JSON path lookup
    # cannot use a plain btree index.
    author_platform_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    author_handle: Mapped[str | None] = mapped_column(String(256), nullable=True)
    author_payload: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)
    """The remaining Author fields. Read whole or not at all, so JSON is right here."""

    # -- time --------------------------------------------------------------
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    """Event time at the source. The axis every trend and forecast query uses."""

    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    """Ingestion time, mirrored from lineage for query convenience."""

    # -- content -----------------------------------------------------------
    content_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    """The *cleaned* body. Bounded by the connector; the raw original is in R2."""

    content_char_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_truncated: Mapped[bool] = mapped_column(nullable=False, default=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False, default="text/plain")

    raw_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # -- enrichment --------------------------------------------------------
    language_code: Mapped[str] = mapped_column(String(16), nullable=False, default="und")
    """BCP-47 or `und`. Indexed: retrieval filters on it and excludes `und`."""

    language_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    entities: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONVariant, nullable=False, default=list
    )
    topics: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONVariant, nullable=False, default=list
    )
    keywords: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONVariant, nullable=False, default=list
    )
    embeddings: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONVariant, nullable=False, default=list
    )
    """EmbeddingRef records — model, dimensions, collection, point id. Never vectors.

    Keeping the refs here is what makes an embedding-model migration tractable:
    a query over this column finds every Signal still carrying vectors from the
    previous model.
    """

    sentiment: Mapped[dict[str, Any] | None] = mapped_column(JSONVariant, nullable=True)

    # -- scoring -----------------------------------------------------------
    engagement: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, default=dict
    )
    engagement_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    """Promoted out of the JSON blob because it is an ORDER BY target."""

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0, index=True)

    # -- overflow and provenance -------------------------------------------
    signal_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONVariant, nullable=False, default=dict
    )
    """Named `signal_metadata` in Python: `metadata` is reserved on Declarative
    classes and would shadow `Base.metadata`. The column is still `metadata`."""

    lineage: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)

    # -- lifecycle ---------------------------------------------------------
    status: Mapped[SignalStatus] = mapped_column(
        TolerantEnumType(SignalStatus, 32), nullable=False, default=SignalStatus.RAW
    )
    dedup_cluster_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    duplicate_of: Mapped[str | None] = mapped_column(String(64), nullable=True)

    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    pipeline_version: Mapped[str] = mapped_column(String(32), nullable=False, default="0.0.0")
    connector_slug: Mapped[str] = mapped_column(String(64), nullable=False)
    sync_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    """Groups a connector run so a bad run can be identified and reverted wholesale."""

    # -- derived-store index state ------------------------------------------
    indexed_vector_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    indexed_keyword_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    graphed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    enrichment_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    """Drives the `partial` -> `quarantined` transition after repeated failures."""

    __table_args__ = (
        # One Signal per source item. The derived id already guarantees this, but
        # the constraint makes it true at the storage layer too -- a connector
        # bug cannot produce two rows for one item.
        # Unnamed on purpose: the convention in `models/orm/base.py` generates
        # `uq_signals_platform_native_id`. Passing an explicit `name=` would
        # bypass the convention and reintroduce the reversibility problem it exists
        # to prevent.
        UniqueConstraint("platform", "native_id"),
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="confidence_range"),
        CheckConstraint(
            "language_confidence >= 0.0 AND language_confidence <= 1.0",
            name="language_confidence_range",
        ),
        CheckConstraint("content_char_count >= 0", name="char_count_non_negative"),
        # A duplicate must point at its canonical member, and must not point at
        # itself -- the same invariant Lineage enforces in Python, restated here
        # because a backfill script can write rows without going through Pydantic.
        CheckConstraint(
            "(status <> 'duplicate') OR "
            "(duplicate_of IS NOT NULL AND duplicate_of <> id)",
            name="duplicate_points_elsewhere",
        ),
        # Retrieval's hot path: recent, retrievable signals for a platform.
        Index("ix_signals_tenant_platform_ts", "tenant_id", "platform", "timestamp"),
        # Backlog sweeps for each derived store. Partial indexes would be tighter
        # but are PostgreSQL-only; these stay portable for the SQLite unit suite.
        Index("ix_signals_vector_backlog", "indexed_vector_at", "status"),
        Index("ix_signals_keyword_backlog", "indexed_keyword_at", "status"),
        Index("ix_signals_graph_backlog", "graphed_at", "status"),
        Index("ix_signals_status_ts", "status", "timestamp"),
        Index("ix_signals_language", "language_code"),
    )
