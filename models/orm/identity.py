"""Storage for `models/identity.py`: a human, and the accounts attached to them.

Two tables. `identities` holds one row per human; `identity_links` attaches
`people` rows to them and records *how* each attachment was decided.

**Why a link table rather than a column on `people`.** A nullable
`people.identity_id` would store the conclusion and throw away the argument --
no method, no confidence, nothing to review and nothing to undo cleanly. The join
costs a lookup on an indexed column; the alternative costs the ability to tell a
confirmation from a coincidence, which is the entire point of the feature.

**Why the link's primary key is `person_id`.** One account belongs to at most one
human, so the constraint belongs in the schema rather than in whichever service
happens to write next. Two rows for one account would mean a commit counted twice
in one person's history and once in another's, and the symptom -- slightly wrong
totals -- is one nobody investigates.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from models.enums import Platform
from models.identity import LinkMethod
from models.orm.base import Base, JSONVariant, TolerantEnumType
from models.orm.mixins import TenantMixin, TimestampMixin, tenant_scoped_index

__all__ = ["IdentityLinkRow", "IdentityRow"]


class IdentityRow(TenantMixin, TimestampMixin, Base):
    """One human. Holds no platform fields -- those live on the accounts."""

    __tablename__ = "identities"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    display_name: Mapped[str] = mapped_column(String(256), nullable=False)
    primary_email: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_bot: Mapped[bool] = mapped_column(default=False, nullable=False)

    identity_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, default=dict
    )

    __table_args__ = (
        # Listing humans is the one query this table serves on its own -- the
        # `people` roster, and the "who is this?" screen behind a suggestion.
        tenant_scoped_index("identities", "display_name"),
    )


class IdentityLinkRow(TenantMixin, TimestampMixin, Base):
    """One account attached to one human, with the reasoning kept.

    `ON DELETE CASCADE` from both sides, and for different reasons. A deleted
    identity should take its links with it -- they assert nothing without the
    human they point at. A deleted person likewise: the link claims an account
    belongs to somebody, and an account that no longer exists cannot.
    """

    __tablename__ = "identity_links"

    person_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("people.id", ondelete="CASCADE"),
        primary_key=True,
    )
    identity_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("identities.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    platform: Mapped[Platform] = mapped_column(TolerantEnumType(Platform), nullable=False)

    method: Mapped[LinkMethod] = mapped_column(
        TolerantEnumType(LinkMethod), nullable=False, index=True
    )
    """Indexed because the useful filter is by *trust*.

    "Show me everything that was guessed" is the review queue -- the screen where
    somebody accepts or rejects inferences -- and it is a scan of this column.
    """

    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    confirmed_by: Mapped[str | None] = mapped_column(String(256), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # Walking one human's accounts: "every artifact by Parth" resolves the
        # identity, then fans out to its people, then to their artifacts.
        tenant_scoped_index("identity_links", "identity_id"),
    )
