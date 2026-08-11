"""Reusable ORM mixins: timestamps, tenant scoping and soft delete.

`TenantMixin` is applied from day one even though multi-tenancy is Phase 7
(`docs/roadmap.md`). Adding `tenant_id` later would touch the primary key of
every table, the Qdrant payload filter and every OpenSearch query simultaneously
-- the roadmap calls this out explicitly as the thing to design for early so
Phase 7 is not a rewrite. Carrying an unused column costs 16 bytes a row.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Index, String, func
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

__all__ = ["SoftDeleteMixin", "TenantMixin", "TimestampMixin"]

DEFAULT_TENANT = "default"
"""Tenant id used until Phase 7 introduces real tenancy.

A literal rather than `NULL`: a nullable tenant column makes every query say
`WHERE tenant_id = :t OR tenant_id IS NULL`, and the day that `OR` is forgotten
is the day one tenant reads another's data.
"""


class TimestampMixin:
    """`created_at` and `updated_at`, clocked by the database.

    `server_default=func.now()` rather than a Python default: the database clock
    is the single source of truth. With workers on several hosts, Python-side
    timestamps skew, and `ORDER BY created_at` stops being a stable ordering.

    Both columns are timezone-aware. `DateTime(timezone=True)` maps to
    `TIMESTAMPTZ` on PostgreSQL, which stores an instant rather than a wall-clock
    reading -- consistent with the tz-aware rule enforced in `models/base.py`.

    **The two columns are not maintained the same way, and the difference
    matters.** `created_at` is a true server default: the database fills it in
    however the row arrives. `updated_at` uses `onupdate=`, which is a
    *client-side* SQLAlchemy construct -- it emits `now()` only on an UPDATE that
    SQLAlchemy itself issues. It produces no DDL and installs no trigger.

    So any write that bypasses the ORM leaves `updated_at` stale, and the
    ingestion path is designed to do exactly that: enrichment upserts Signals
    with `ON CONFLICT (id) DO UPDATE` for idempotency. Such statements, hand-run
    SQL, and restores must set `updated_at` explicitly.

    A `BEFORE UPDATE` trigger would make this unconditional and is the better
    long-term answer; it is deliberately not added yet because no write path
    exists to need it. Tracked as an open item so the claim in this docstring
    stays true rather than aspirational.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TenantMixin:
    """`tenant_id` on every row, indexed, ready for Phase 7.

    Declared with `declared_attr` so each table gets its own `Column` object --
    sharing one across mappings raises at configure time.
    """

    @declared_attr
    @classmethod
    def tenant_id(cls) -> Mapped[str]:
        return mapped_column(
            String(64),
            default=DEFAULT_TENANT,
            server_default=DEFAULT_TENANT,
            nullable=False,
            index=True,
        )


class SoftDeleteMixin:
    """`deleted_at`, for rows that must survive deletion as tombstones.

    Applied only where an audit trail or a foreign-key target must outlive the
    logical delete -- not to `signals`, where erasure has to be genuine to honour
    a deletion request (`docs/security-and-privacy.md`). Soft-deleting a Signal
    would leave the text in PostgreSQL after the user asked for it to be gone.
    """

    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None


def tenant_scoped_index(table_name: str, *columns: str, unique: bool = False) -> Index:
    """Build an index prefixed by `tenant_id`.

    Every multi-tenant query filters on `tenant_id` first, so it belongs at the
    front of the index. An index on `(platform, timestamp)` alone cannot serve
    `WHERE tenant_id = ? AND platform = ?` efficiently once there is more than
    one tenant.
    """
    suffix = "_".join(columns)
    return Index(
        f"{'uq' if unique else 'ix'}_{table_name}_tenant_{suffix}",
        "tenant_id",
        *columns,
        unique=unique,
    )
