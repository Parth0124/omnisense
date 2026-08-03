"""Initial schema: signals, connectors, investigations, reports and run accounting.

Revision ID: 0001
Revises:
Created: 2026-08-03

This revision is the whole PostgreSQL schema as of Phase 1. It was derived from
`models/orm/*` rather than typed by hand: the metadata was compiled and the
resulting DDL translated into the calls below. The round trip was then checked
mechanically rather than by reading -- this revision and
`Base.metadata.create_all()` were each applied to a SQLite database with
`omnisense` attached as a schema and the two catalogues diffed, and the
PostgreSQL DDL from `alembic upgrade head --sql` was diffed against the DDL the
metadata compiles to. Repeat both after editing this file or any mapping; a model
change that skips this revision is invisible until the next deploy.

Three things here are deliberate and worth reading before changing them.

**The types are restated, not imported.** `_jsonb()` below duplicates
`JSONVariant` from `models/orm/base.py` instead of importing it. A migration is a
record of what the schema looked like at a point in time; importing the live
models makes an applied revision mean something different after the next model
edit, and "revision 0001 no longer produces the database revision 0001 produced"
is the one property a migration history cannot lose. The same reasoning applies
to the schema names, which are string literals here.

**`TolerantEnumType` is `VARCHAR`.** It is a `TypeDecorator` over `String`, so it
emits no `CREATE TYPE` and no `CHECK`; that is the entire point of it
(`models/orm/base.py`). Nothing is lost by writing `sa.String` here -- the
tolerance is Python-side behaviour on read, not a storage constraint.

**`investigations.report_id` is added last, except where it cannot be.**
`investigations` and `reports` reference each other, so no creation order
satisfies both. The models express this with `use_alter=True`, which resolves
differently per dialect -- separate `ALTER TABLE` where the dialect supports one,
folded back into the `CREATE TABLE` where it does not. `_inline_report_fk()` and
`_create_deferred_foreign_key()` reproduce both branches.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


SCHEMA = "omnisense"
CHECKPOINT_SCHEMA = "checkpoints"


def _jsonb() -> sa.types.TypeEngine[object]:
    """`JSONB` on PostgreSQL, portable `JSON` elsewhere.

    A fresh instance per column: a SQLAlchemy type object is attached to the
    column it is used on, and sharing one across a dozen columns is a pattern
    that works until the day something mutates it.
    """
    return sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.String()), "postgresql")


def _enum(length: int = 32) -> sa.String:
    """The storage type behind `TolerantEnumType` -- a plain, unconstrained VARCHAR."""
    return sa.String(length=length)


def _now() -> sa.sql.functions.Function[object]:
    """`server_default` for the timestamp mixin: the *database* clock.

    `sa.func.now()` rather than a literal `sa.text("now()")` so the default
    compiles to `now()` on PostgreSQL and `CURRENT_TIMESTAMP` on SQLite. A
    hard-coded `now()` would produce a table SQLite accepts and then fails to
    insert into, which is a poor way to discover that the unit suite runs on
    SQLite.
    """
    return sa.func.now()


def _create_schemas() -> None:
    """Create both schemas if they are absent.

    `migrations/env.py` already does this before the version table is created --
    it has to, or Alembic cannot record that this revision ran. Repeating it here
    is not redundancy for its own sake: it makes `alembic upgrade --sql` output a
    complete script a DBA can apply to an empty database, and it means a
    deployment where `docker/local/postgres/01-extensions.sql` never ran still
    gets a working schema.

    Skipped on SQLite, which has no `CREATE SCHEMA`; there a schema is an
    `ATTACH`ed database the caller has already set up.
    """
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{SCHEMA}"')
    op.execute(f'CREATE SCHEMA IF NOT EXISTS "{CHECKPOINT_SCHEMA}"')


def upgrade() -> None:
    _create_schemas()

    # ----------------------------------------------------------------- ingest
    op.create_table(
        "connector_accounts",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("connector_slug", sa.String(length=64), nullable=False),
        sa.Column("platform", _enum(32), nullable=False),
        sa.Column("display_name", sa.String(length=256), nullable=False),
        sa.Column("auth_type", _enum(32), nullable=False),
        sa.Column("encrypted_credentials", sa.LargeBinary(), nullable=True),
        sa.Column("credential_key_version", sa.Integer(), nullable=False),
        sa.Column("credential_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", _enum(32), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("params", _jsonb(), nullable=False),
        sa.Column("params_hash", sa.String(length=64), nullable=False),
        sa.Column("last_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sync_interval_seconds", sa.Integer(), nullable=False),
        sa.Column("consecutive_failures", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(status <> 'disabled') OR (NOT enabled)",
            name=op.f("ck_connector_accounts_disabled_implies_not_enabled"),
        ),
        sa.CheckConstraint(
            "consecutive_failures >= 0",
            name=op.f("ck_connector_accounts_failures_non_negative"),
        ),
        sa.CheckConstraint(
            "credential_key_version >= 1",
            name=op.f("ck_connector_accounts_key_version_positive"),
        ),
        sa.CheckConstraint(
            "encrypted_credentials IS NULL OR credential_updated_at IS NOT NULL",
            name=op.f("ck_connector_accounts_credential_has_timestamp"),
        ),
        sa.CheckConstraint(
            "sync_interval_seconds > 0",
            name=op.f("ck_connector_accounts_sync_interval_positive"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connector_accounts")),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_connector_accounts_due",
        "connector_accounts",
        ["enabled", "next_sync_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_connector_accounts_tenant_slug",
        "connector_accounts",
        ["tenant_id", "connector_slug"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_connector_accounts_tenant_status",
        "connector_accounts",
        ["tenant_id", "status"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_connector_accounts_deleted_at"),
        "connector_accounts",
        ["deleted_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_connector_accounts_tenant_id"),
        "connector_accounts",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )
    # Partial, so a soft-deleted account releases the configuration key instead
    # of blocking a re-create forever. Both dialects get the predicate; SQLite
    # implements partial indexes too, which is what keeps the unit suite honest.
    op.create_index(
        "uq_connector_accounts_live_config",
        "connector_accounts",
        ["tenant_id", "connector_slug", "params_hash"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("deleted_at IS NULL"),
        sqlite_where=sa.text("deleted_at IS NULL"),
    )

    # -------------------------------------------------------- investigations
    # `report_id` is a bare column here on PostgreSQL; its foreign key is
    # attached at the end of upgrade(), once `reports` exists. See the module
    # docstring and `_inline_report_fk()`.
    op.create_table(
        "investigations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("created_by", sa.String(length=256), nullable=True),
        sa.Column("status", _enum(32), nullable=False),
        sa.Column("plan", _jsonb(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("step_count", sa.Integer(), nullable=False),
        sa.Column("token_input", sa.Integer(), nullable=False),
        sa.Column("token_output", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=14, scale=6, asdecimal=False), nullable=False),
        sa.Column("report_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint(
            "completed_at IS NULL OR (started_at IS NOT NULL AND completed_at >= started_at)",
            name=op.f("ck_investigations_completed_after_started"),
        ),
        sa.CheckConstraint("cost_usd >= 0", name=op.f("ck_investigations_cost_non_negative")),
        sa.CheckConstraint(
            "step_count >= 0", name=op.f("ck_investigations_step_count_non_negative")
        ),
        sa.CheckConstraint(
            "token_input >= 0", name=op.f("ck_investigations_token_input_non_negative")
        ),
        sa.CheckConstraint(
            "token_output >= 0", name=op.f("ck_investigations_token_output_non_negative")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_investigations")),
        *_inline_report_fk(),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_investigations_status_started",
        "investigations",
        ["status", "started_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_investigations_tenant_status_created",
        "investigations",
        ["tenant_id", "status", "created_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_investigations_tenant_id"),
        "investigations",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )

    # ---------------------------------------------------------------- signals
    op.create_table(
        "signals",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("native_id", sa.String(length=512), nullable=False),
        sa.Column("source", _enum(32), nullable=False),
        sa.Column("platform", _enum(32), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("author_platform_id", sa.String(length=256), nullable=True),
        sa.Column("author_handle", sa.String(length=256), nullable=True),
        sa.Column("author_payload", _jsonb(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_title", sa.Text(), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=False),
        sa.Column("content_char_count", sa.Integer(), nullable=False),
        sa.Column("content_truncated", sa.Boolean(), nullable=False),
        sa.Column("content_type", sa.String(length=64), nullable=False),
        sa.Column("raw_object_key", sa.Text(), nullable=True),
        sa.Column("raw_sha256", sa.String(length=64), nullable=True),
        sa.Column("language_code", sa.String(length=16), nullable=False),
        sa.Column("language_confidence", sa.Float(), nullable=False),
        sa.Column("entities", _jsonb(), nullable=False),
        sa.Column("topics", _jsonb(), nullable=False),
        sa.Column("keywords", _jsonb(), nullable=False),
        sa.Column("embeddings", _jsonb(), nullable=False),
        sa.Column("sentiment", _jsonb(), nullable=True),
        sa.Column("engagement", _jsonb(), nullable=False),
        sa.Column("engagement_score", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        # Named `metadata` in the database and `signal_metadata` in Python;
        # `metadata` is reserved on a Declarative class (`models/orm/signal.py`).
        sa.Column("metadata", _jsonb(), nullable=False),
        sa.Column("lineage", _jsonb(), nullable=False),
        sa.Column("status", _enum(32), nullable=False),
        sa.Column("dedup_cluster_id", sa.String(length=64), nullable=True),
        # Intentionally not a self-referential foreign key: a deletion request
        # must be able to remove the canonical row, and neither CASCADE nor
        # RESTRICT is an acceptable answer to that.
        sa.Column("duplicate_of", sa.String(length=64), nullable=True),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("pipeline_version", sa.String(length=32), nullable=False),
        sa.Column("connector_slug", sa.String(length=64), nullable=False),
        sa.Column("sync_run_id", sa.String(length=64), nullable=True),
        sa.Column("indexed_vector_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("indexed_keyword_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("graphed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("enrichment_attempts", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint(
            "(status <> 'duplicate') OR (duplicate_of IS NOT NULL AND duplicate_of <> id)",
            name=op.f("ck_signals_duplicate_points_elsewhere"),
        ),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name=op.f("ck_signals_confidence_range")
        ),
        sa.CheckConstraint(
            "content_char_count >= 0", name=op.f("ck_signals_char_count_non_negative")
        ),
        sa.CheckConstraint(
            "language_confidence >= 0.0 AND language_confidence <= 1.0",
            name=op.f("ck_signals_language_confidence_range"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_signals")),
        sa.UniqueConstraint("platform", "native_id", name=op.f("uq_signals_platform_native_id")),
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_signals_confidence"),
        "signals",
        ["confidence"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_signals_dedup_cluster_id"),
        "signals",
        ["dedup_cluster_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_signals_engagement_score"),
        "signals",
        ["engagement_score"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_signals_platform"),
        "signals",
        ["platform"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_signals_sync_run_id"),
        "signals",
        ["sync_run_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_signals_tenant_id"),
        "signals",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_signals_timestamp"),
        "signals",
        ["timestamp"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_signals_graph_backlog",
        "signals",
        ["graphed_at", "status"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_signals_keyword_backlog",
        "signals",
        ["indexed_keyword_at", "status"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_signals_language", "signals", ["language_code"], unique=False, schema=SCHEMA
    )
    op.create_index(
        "ix_signals_status_ts", "signals", ["status", "timestamp"], unique=False, schema=SCHEMA
    )
    op.create_index(
        "ix_signals_tenant_platform_ts",
        "signals",
        ["tenant_id", "platform", "timestamp"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_signals_vector_backlog",
        "signals",
        ["indexed_vector_at", "status"],
        unique=False,
        schema=SCHEMA,
    )

    # ------------------------------------------------------- run accounting
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("investigation_id", sa.String(length=64), nullable=True),
        sa.Column("agent", _enum(32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("prompt_id", sa.String(length=128), nullable=False),
        sa.Column("prompt_version", sa.String(length=16), nullable=False),
        sa.Column("prompt_hash", sa.String(length=64), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("cached_tokens", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Numeric(precision=14, scale=6, asdecimal=False), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("status", _enum(32), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint(
            "cached_tokens >= 0", name=op.f("ck_agent_runs_cached_tokens_non_negative")
        ),
        sa.CheckConstraint("cost_usd >= 0", name=op.f("ck_agent_runs_cost_non_negative")),
        sa.CheckConstraint(
            "input_tokens >= 0", name=op.f("ck_agent_runs_input_tokens_non_negative")
        ),
        sa.CheckConstraint("latency_ms >= 0", name=op.f("ck_agent_runs_latency_non_negative")),
        sa.CheckConstraint(
            "output_tokens >= 0", name=op.f("ck_agent_runs_output_tokens_non_negative")
        ),
        # SET NULL, not CASCADE: money that was spent stays spent even after the
        # investigation it paid for is erased (`models/orm/run.py`).
        sa.ForeignKeyConstraint(
            ["investigation_id"],
            [f"{SCHEMA}.investigations.id"],
            name=op.f("fk_agent_runs_investigation_id_investigations"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_runs")),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_agent_runs_investigation_started",
        "agent_runs",
        ["investigation_id", "started_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_agent_runs_model_started",
        "agent_runs",
        ["model", "started_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_agent_runs_prompt_hash", "agent_runs", ["prompt_hash"], unique=False, schema=SCHEMA
    )
    op.create_index(
        "ix_agent_runs_tenant_started",
        "agent_runs",
        ["tenant_id", "started_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_agent_runs_tenant_id"),
        "agent_runs",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )

    op.create_table(
        "connector_cursors",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("connector_slug", sa.String(length=64), nullable=False),
        sa.Column("account_id", sa.String(length=64), nullable=False),
        sa.Column("params_hash", sa.String(length=64), nullable=False),
        sa.Column("cursor", _jsonb(), nullable=False),
        sa.Column("cursor_version", sa.Integer(), nullable=False),
        sa.Column("watermark", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint(
            "cursor_version >= 1", name=op.f("ck_connector_cursors_cursor_version_positive")
        ),
        sa.ForeignKeyConstraint(
            ["account_id"],
            [f"{SCHEMA}.connector_accounts.id"],
            name=op.f("fk_connector_cursors_account_id_connector_accounts"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_connector_cursors")),
        sa.UniqueConstraint(
            "connector_slug",
            "account_id",
            "params_hash",
            name=op.f("uq_connector_cursors_connector_slug_account_id_params_hash"),
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_connector_cursors_account",
        "connector_cursors",
        ["account_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_connector_cursors_tenant_id"),
        "connector_cursors",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )

    op.create_table(
        "investigation_steps",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("investigation_id", sa.String(length=64), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("agent", _enum(32), nullable=False),
        sa.Column("status", _enum(32), nullable=False),
        sa.Column("input", _jsonb(), nullable=True),
        sa.Column("output", _jsonb(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("token_input", sa.Integer(), nullable=False),
        sa.Column("token_output", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name=op.f("ck_investigation_steps_duration_non_negative"),
        ),
        sa.CheckConstraint(
            "sequence >= 0", name=op.f("ck_investigation_steps_sequence_non_negative")
        ),
        sa.CheckConstraint(
            "token_input >= 0", name=op.f("ck_investigation_steps_token_input_non_negative")
        ),
        sa.CheckConstraint(
            "token_output >= 0", name=op.f("ck_investigation_steps_token_output_non_negative")
        ),
        sa.ForeignKeyConstraint(
            ["investigation_id"],
            [f"{SCHEMA}.investigations.id"],
            name=op.f("fk_investigation_steps_investigation_id_investigations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_investigation_steps")),
        sa.UniqueConstraint(
            "investigation_id",
            "sequence",
            name=op.f("uq_investigation_steps_investigation_id_sequence"),
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_investigation_steps_agent_started",
        "investigation_steps",
        ["agent", "started_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_investigation_steps_tenant_id"),
        "investigation_steps",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )

    # ---------------------------------------------------------------- reports
    op.create_table(
        "reports",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("investigation_id", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("format", _enum(16), nullable=False),
        sa.Column("object_key", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("status", _enum(32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("superseded_by", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0", name=op.f("ck_reports_confidence_range")
        ),
        sa.CheckConstraint(
            "superseded_by IS NULL OR superseded_by <> id",
            name=op.f("ck_reports_supersede_points_elsewhere"),
        ),
        sa.CheckConstraint("version >= 1", name=op.f("ck_reports_version_positive")),
        sa.ForeignKeyConstraint(
            ["investigation_id"],
            [f"{SCHEMA}.investigations.id"],
            name=op.f("fk_reports_investigation_id_investigations"),
            ondelete="CASCADE",
        ),
        # Self-reference: legal in a CREATE TABLE because the table exists by the
        # time the constraint is checked.
        sa.ForeignKeyConstraint(
            ["superseded_by"],
            [f"{SCHEMA}.reports.id"],
            name=op.f("fk_reports_superseded_by_reports"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
        sa.UniqueConstraint(
            "investigation_id", "version", name=op.f("uq_reports_investigation_id_version")
        ),
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_reports_tenant_id"),
        "reports",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_reports_status_created",
        "reports",
        ["status", "created_at"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_reports_tenant_created",
        "reports",
        ["tenant_id", "created_at"],
        unique=False,
        schema=SCHEMA,
    )

    op.create_table(
        "report_sections",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("report_id", sa.String(length=64), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name=op.f("ck_report_sections_confidence_range"),
        ),
        sa.CheckConstraint("ordinal >= 0", name=op.f("ck_report_sections_ordinal_non_negative")),
        sa.ForeignKeyConstraint(
            ["report_id"],
            [f"{SCHEMA}.reports.id"],
            name=op.f("fk_report_sections_report_id_reports"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_report_sections")),
        sa.UniqueConstraint(
            "report_id", "ordinal", name=op.f("uq_report_sections_report_id_ordinal")
        ),
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_report_sections_tenant_id"),
        "report_sections",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )

    # ----------------------------------------------------------------- traces
    op.create_table(
        "traces",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("trace_id", sa.String(length=32), nullable=False),
        sa.Column("span_id", sa.String(length=16), nullable=False),
        sa.Column("parent_span_id", sa.String(length=16), nullable=True),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("attributes", _jsonb(), nullable=False),
        sa.Column("agent_run_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint("length(span_id) = 16", name=op.f("ck_traces_span_id_length")),
        sa.CheckConstraint("length(trace_id) = 32", name=op.f("ck_traces_trace_id_length")),
        sa.CheckConstraint(
            "parent_span_id IS NULL OR length(parent_span_id) = 16",
            name=op.f("ck_traces_parent_span_id_length"),
        ),
        sa.CheckConstraint(
            "parent_span_id IS NULL OR parent_span_id <> span_id",
            name=op.f("ck_traces_parent_is_not_self"),
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            [f"{SCHEMA}.agent_runs.id"],
            name=op.f("fk_traces_agent_run_id_agent_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_traces")),
        sa.UniqueConstraint("trace_id", "span_id", name=op.f("uq_traces_trace_id_span_id")),
        schema=SCHEMA,
    )
    op.create_index(
        op.f("ix_omnisense_traces_tenant_id"), "traces", ["tenant_id"], unique=False, schema=SCHEMA
    )
    op.create_index("ix_traces_agent_run", "traces", ["agent_run_id"], unique=False, schema=SCHEMA)
    op.create_index(
        "ix_traces_parent_span", "traces", ["parent_span_id"], unique=False, schema=SCHEMA
    )

    # -------------------------------------------------------------- citations
    # `signal_id` is deliberately not a foreign key -- see `models/orm/report.py`:
    # erasure must be able to hard-delete a Signal, and both CASCADE (rewrites a
    # published report) and RESTRICT (blocks erasure) are wrong answers.
    op.create_table(
        "citations",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("report_id", sa.String(length=64), nullable=False),
        sa.Column("section_id", sa.String(length=64), nullable=True),
        sa.Column("signal_id", sa.String(length=64), nullable=False),
        sa.Column("quote", sa.Text(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False),
        sa.Column("char_end", sa.Integer(), nullable=False),
        sa.Column("relevance", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_now(), nullable=False),
        sa.Column("tenant_id", sa.String(length=64), server_default="default", nullable=False),
        sa.CheckConstraint("char_end >= char_start", name=op.f("ck_citations_char_range_ordered")),
        sa.CheckConstraint("char_start >= 0", name=op.f("ck_citations_char_start_non_negative")),
        sa.CheckConstraint(
            "relevance >= 0.0 AND relevance <= 1.0", name=op.f("ck_citations_relevance_range")
        ),
        sa.ForeignKeyConstraint(
            ["report_id"],
            [f"{SCHEMA}.reports.id"],
            name=op.f("fk_citations_report_id_reports"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["section_id"],
            [f"{SCHEMA}.report_sections.id"],
            name=op.f("fk_citations_section_id_report_sections"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_citations")),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_citations_report_section",
        "citations",
        ["report_id", "section_id"],
        unique=False,
        schema=SCHEMA,
    )
    op.create_index(
        "ix_citations_section", "citations", ["section_id"], unique=False, schema=SCHEMA
    )
    op.create_index("ix_citations_signal", "citations", ["signal_id"], unique=False, schema=SCHEMA)
    op.create_index(
        op.f("ix_omnisense_citations_tenant_id"),
        "citations",
        ["tenant_id"],
        unique=False,
        schema=SCHEMA,
    )

    _create_deferred_foreign_key()


def downgrade() -> None:
    """Drop everything this revision created, in reverse dependency order.

    Indexes and constraints are not dropped individually: `DROP TABLE` takes the
    table's own indexes, checks and unique constraints with it on both
    PostgreSQL and SQLite. Listing them would be forty lines that add nothing and
    that can silently drift out of step with `upgrade()`. The one exception is
    the deferred foreign key, which has to go first because it is what makes
    `investigations` and `reports` mutually dependent.

    The schemas themselves are **not** dropped, deliberately. `omnisense` holds
    `alembic_version`, and dropping it would take the row recording this
    downgrade with it -- leaving a database Alembic believes is still at head.
    `checkpoints` belongs to LangGraph and was never this revision's to own; it
    is created here only so the bootstrap is complete.
    """
    _drop_deferred_foreign_key()

    for table in (
        "citations",
        "traces",
        "report_sections",
        "reports",
        "investigation_steps",
        "connector_cursors",
        "agent_runs",
        "signals",
        "investigations",
        "connector_accounts",
    ):
        op.drop_table(table, schema=SCHEMA)


def _report_fk() -> sa.ForeignKeyConstraint:
    """The `investigations.report_id -> reports.id` constraint.

    `SET NULL`, not `CASCADE`: deleting a report must not erase the record that
    the investigation ran, what it cost and what it concluded
    (`models/orm/investigation.py`).
    """
    return sa.ForeignKeyConstraint(
        ["report_id"],
        [f"{SCHEMA}.reports.id"],
        name=op.f("fk_investigations_report_id_reports"),
        ondelete="SET NULL",
    )


def _inline_report_fk() -> tuple[sa.ForeignKeyConstraint, ...]:
    """The cycle-breaking constraint, but only on a dialect that cannot ALTER.

    This is what `use_alter=True` actually means, and it is not "skip the
    constraint on SQLite" -- that was the assumption, and diffing the two
    catalogues disproved it. `create_all()` passes a `filter_fn` to
    `sort_tables_and_constraints` that returns False when `supports_alter` is
    false, which folds the constraint *back into* the `CREATE TABLE` rather than
    dropping it. SQLite accepts a forward reference to a table that does not
    exist yet, so inlining is legal there and separating is not.

    Emitting it in the same place keeps the migration and the models producing
    byte-identical SQLite catalogues, which is the only reason the round-trip
    check can be an equality test instead of a list of tolerated exceptions.
    """
    if op.get_bind().dialect.supports_alter:
        return ()
    return (_report_fk(),)


def _create_deferred_foreign_key() -> None:
    """Attach the cycle-breaking constraint once both tables exist.

    Only on a dialect that supports `ALTER TABLE ... ADD CONSTRAINT`; elsewhere
    `_inline_report_fk()` has already emitted it with the table.
    """
    if not op.get_bind().dialect.supports_alter:
        return
    op.create_foreign_key(
        op.f("fk_investigations_report_id_reports"),
        "investigations",
        "reports",
        ["report_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="SET NULL",
    )


def _drop_deferred_foreign_key() -> None:
    """Detach the cycle-breaking constraint before the tables it spans are dropped."""
    if not op.get_bind().dialect.supports_alter:
        return
    op.drop_constraint(
        op.f("fk_investigations_report_id_reports"),
        "investigations",
        schema=SCHEMA,
        type_="foreignkey",
    )
