"""ORM tables for connector accounts, credentials and sync cursors.

Two tables live here because they have genuinely different lifecycles.

`connector_accounts` is *configuration*: an operator-created record saying "sync
this connector, with these credentials, against these targets, this often". It is
the only place in PostgreSQL where third-party credentials exist, and they exist
only as Fernet ciphertext (`docs/security-and-privacy.md` §4.1). Nothing in this
module ever sees plaintext -- `connectors/auth/token_store.py` is the sole reader
and writer of `encrypted_credentials`.

`connector_cursors` is *derived state*: the incremental-sync position, owned by
the connector and opaque to everything else (`docs/connector-spec.md` §4). It is
separated from the account row for two reasons. First, one account produces many
cursors -- the key is `(connector_slug, account_id, params_hash)`, so a Reddit
account watching two subreddits has two independent watermarks. Second, the
cursor is written on every successful page commit while the account row is
written almost never; keeping them apart stops a hot, high-churn row from sharing
a page with long-lived credential ciphertext.

`ConnectorAccountRow` is the one table in `models/orm/` that carries
`SoftDeleteMixin`. A cursor references the account, and every Signal it produced
records its `connector_slug`; deleting the account outright would either orphan
that history or cascade it away. A tombstone keeps the foreign-key target alive
while taking the account out of the scheduler's view. That is safe here precisely
because the row holds no scraped content -- only ciphertext, which the purge job
in `workers/scheduler.py` clears separately.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from models.base import TolerantStrEnum
from models.enums import AuthType, Platform
from models.orm.base import Base, JSONVariant, TolerantEnumType
from models.orm.mixins import SoftDeleteMixin, TenantMixin, TimestampMixin

__all__ = [
    "ConnectorAccountRow",
    "ConnectorAccountStatus",
    "ConnectorCursorRow",
]


class ConnectorAccountStatus(TolerantStrEnum):
    """Health verdict the runtime records on a configured connector account.

    Deliberately distinct from the `enabled` flag on the same row. `enabled` is
    the operator's intent ("I want this syncing"); `status` is the system's
    observation ("I cannot -- the credential is dead"). Collapsing them would make
    an expired OAuth token indistinguishable from a deliberate pause, and the
    reauthentication prompt in the UI has to tell those two apart.

    `NEEDS_REAUTH` is set by the runtime on `ConnectorErrorClass.AUTH`, which is
    terminal for a run and never retried (`docs/connector-spec.md` §2.1).

    Defined here rather than in `models/enums.py` because there is no domain-level
    `ConnectorAccount` model yet -- this vocabulary currently has exactly one
    consumer, the table below. It should move to `models/enums.py` the moment a
    second module needs it.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"
    NEEDS_REAUTH = "needs_reauth"
    UNKNOWN = "unknown"


class ConnectorAccountRow(Base, TimestampMixin, TenantMixin, SoftDeleteMixin):
    """One configured instance of one connector, with its encrypted credentials."""

    __tablename__ = "connector_accounts"

    # -- identity ----------------------------------------------------------
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    """Service-assigned (`acct_` + uuid4 hex).

    Unlike `Signal.id` this is *not* derived from its content: two accounts can
    legitimately hold different credentials for the same connector and the same
    targets (two Reddit apps, two rate-limit budgets), so there is nothing stable
    to derive an id from.
    """

    connector_slug: Mapped[str] = mapped_column(String(64), nullable=False)
    """Registry key resolved by `connectors/registry.py`, e.g. `reddit`."""

    platform: Mapped[Platform] = mapped_column(TolerantEnumType(Platform, 32), nullable=False)
    """Denormalized from the connector class so that "which platforms am I
    ingesting" is a query on this table rather than an import of the registry."""

    display_name: Mapped[str] = mapped_column(String(256), nullable=False)

    # -- credentials -------------------------------------------------------
    auth_type: Mapped[AuthType] = mapped_column(
        TolerantEnumType(AuthType, 32), nullable=False, default=AuthType.NONE
    )

    encrypted_credentials: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    """Fernet ciphertext (AES-128-CBC + HMAC-SHA256), never plaintext.

    `LargeBinary` rather than `Text` on purpose: a base64 `TEXT` column invites
    someone to `SELECT` it during a debugging session and paste the result
    somewhere, and it silently tolerates a plaintext write. `BYTEA` does neither.

    Nullable because an account legitimately exists without a usable credential:
    `auth_type=none` connectors (RSS) never have one, and an OAuth account exists
    with `status=needs_reauth` between the operator creating it and completing
    the consent flow. Credential presence is therefore checked in
    `connectors/auth/token_store.py`, not by a NOT NULL constraint.
    """

    credential_key_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    """Which `CREDENTIAL_ENCRYPTION_KEY` generation the ciphertext is under.

    Required by `docs/security-and-privacy.md` §4.1: rotation is multi-key
    decrypt / single-key encrypt, so a row must say which key opens it. Without
    this column, rotating the encryption key is an offline, all-or-nothing
    re-encryption of every account.
    """

    credential_updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    """Last write of `encrypted_credentials`. Drives the 90-day rotation reminder
    and is the `last_rotated_at` the API is allowed to expose (§4.2)."""

    # -- lifecycle ---------------------------------------------------------
    status: Mapped[ConnectorAccountStatus] = mapped_column(
        TolerantEnumType(ConnectorAccountStatus, 32),
        nullable=False,
        default=ConnectorAccountStatus.ENABLED,
    )
    enabled: Mapped[bool] = mapped_column(nullable=False, default=True)

    # -- configuration -----------------------------------------------------
    params: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    """Connector-specific targets: `{"subreddits": ["devops", "sre"]}`.

    Opaque here by design -- the schema belongs to the connector, which validates
    it. Keeping it JSON is what stops `models/orm/` from growing platform-shaped
    columns (`models/signal.py`: no platform-shaped code above `connectors/`).
    """

    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    """SHA-256 of the canonicalized `params` (`docs/connector-spec.md` §4).

    Stored rather than recomputed on read because it is half of the cursor key: a
    change to `params` must produce a *different* cursor, not silently inherit
    the watermark belonging to a different set of targets.
    """

    # -- scheduling --------------------------------------------------------
    last_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    next_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    """When the scheduler may next claim this account. `NULL` means "never
    scheduled", which the scheduler reads as due immediately."""

    sync_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=3600)

    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    """Reset to zero by any successful run. The scheduler backs `next_sync_at` off
    exponentially against this, so a permanently broken feed stops consuming a
    rate-limit budget a healthy one could be using."""

    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Message from the last failed run, already redacted by the connector
    exception types (`docs/security-and-privacy.md` §4.2 -- they carry
    `(source, status_code, request_id)`, never headers or query strings)."""

    __table_args__ = (
        # Two *live* accounts with the same connector and the same targets are a
        # misconfiguration: they double-fetch and then fight over one cursor key.
        # Expressed as a partial unique index rather than a UniqueConstraint
        # because of SoftDeleteMixin -- a tombstoned row would otherwise occupy
        # the key forever and re-creating a deleted account would fail. Both
        # PostgreSQL and SQLite implement partial indexes, so the unit suite
        # still exercises this.
        Index(
            "uq_connector_accounts_live_config",
            "tenant_id",
            "connector_slug",
            "params_hash",
            unique=True,
            postgresql_where=text("deleted_at IS NULL"),
            sqlite_where=text("deleted_at IS NULL"),
        ),
        CheckConstraint("sync_interval_seconds > 0", name="sync_interval_positive"),
        CheckConstraint("consecutive_failures >= 0", name="failures_non_negative"),
        CheckConstraint("credential_key_version >= 1", name="key_version_positive"),
        # A stored credential must record when it was stored, or the rotation job
        # has no way to find the rows that are overdue.
        CheckConstraint(
            "encrypted_credentials IS NULL OR credential_updated_at IS NOT NULL",
            name="credential_has_timestamp",
        ),
        # `status=disabled` is the system agreeing with the operator; it must not
        # coexist with `enabled=true`. The reverse pairing is legal and common:
        # `enabled=true, status=needs_reauth` is exactly an expired token on an
        # account the operator still wants running.
        CheckConstraint(
            "(status <> 'disabled') OR (NOT enabled)", name="disabled_implies_not_enabled"
        ),
        # The scheduler's claim query: enabled accounts whose next_sync_at has
        # passed, oldest first.
        Index("ix_connector_accounts_due", "enabled", "next_sync_at"),
        # `GET /api/v1/connectors` -- one tenant's accounts, grouped by connector.
        Index("ix_connector_accounts_tenant_slug", "tenant_id", "connector_slug"),
        # The health panel: which of this tenant's accounts need reauthentication.
        Index("ix_connector_accounts_tenant_status", "tenant_id", "status"),
    )


class ConnectorCursorRow(Base, TimestampMixin, TenantMixin):
    """The incremental-sync position for one (connector, account, params) triple.

    Not soft-deletable: a cursor carries no history worth preserving. If the
    account it belongs to is genuinely purged, the correct next run is a full
    backfill -- which is exactly what a missing cursor row means to the runtime.
    """

    __tablename__ = "connector_cursors"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    connector_slug: Mapped[str] = mapped_column(String(64), nullable=False)

    account_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("connector_accounts.id", ondelete="CASCADE"),
        nullable=False,
    )
    """`CASCADE` fires only on a *hard* delete. The normal path sets
    `connector_accounts.deleted_at` and leaves the cursor intact, so re-enabling
    an account resumes where it stopped instead of re-crawling history."""

    params_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    """SHA-256 of the canonicalized `SyncContext.params` for the run, which is not
    necessarily the account's configured `params_hash`:
    `POST /connectors/{slug}/sync` can narrow the targets for a single run
    (`docs/api-reference.md` §4.3).

    `docs/connector-spec.md` §4 rule 5 also requires a backfill to run against a
    *separate* cursor row, while the same section fixes the key as
    `(connector_slug, account_id, params_hash)`. Those two only reconcile if the
    runtime folds the sync mode into the params it hashes -- and it must, or a
    long historical crawl silently clobbers the live incremental watermark.
    """

    cursor: Mapped[dict[str, Any]] = mapped_column(JSONVariant, nullable=False, default=dict)
    """Connector-private state: `page_token`, ETags, per-feed offsets.

    Nothing outside the owning connector may read a key out of this blob. The
    runtime persists and returns it verbatim; interpreting it here would turn the
    cursor shape into a cross-module contract, which is exactly what §4 keeps it
    from being.
    """

    cursor_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    """Bumped by the connector when the `cursor` shape changes
    (`docs/connector-spec.md` §4 rule 6). The runtime treats a version it does not
    recognize as "no cursor" and starts a bounded re-sync, rather than feeding old
    state to a paginator that will misread it."""

    watermark: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    """Event time of the newest record durably emitted.

    The one part of the cursor the runtime *does* understand: it rejects a commit
    that moves the watermark backwards (§4 rule 2) and starts the next run at
    `watermark - overlap` (rule 3). `NULL` means nothing has been emitted yet.
    """

    __table_args__ = (
        # The cursor key from `docs/connector-spec.md` §4, enforced in the storage
        # layer so that a scheduling bug cannot create a second cursor and
        # silently halve the effective watermark.
        UniqueConstraint("connector_slug", "account_id", "params_hash"),
        CheckConstraint("cursor_version >= 1", name="cursor_version_positive"),
        # "Every cursor for this account", and the index PostgreSQL needs to make
        # the ON DELETE CASCADE above a lookup rather than a sequential scan --
        # the unique constraint leads with `connector_slug`, so it cannot serve a
        # probe by `account_id`.
        Index("ix_connector_cursors_account", "account_id"),
    )
