"""Storage for `models/feature.py`: versions, features, and artifact membership.

Three tables mirroring the three concepts. The only one with a subtlety is
`feature_links`, whose primary key is the *pair* -- one artifact can belong to
several features, and usually does. A commit that adds an image-upload route to
the deploy pipeline is honestly part of both, and forcing a single owner would
mean picking one and being wrong half the time.

Contrast `identity_links`, keyed on `person_id` alone: an account belongs to
exactly one human, and there the schema should say so. The two tables look alike
and their keys differ, deliberately.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from models.feature import FeatureState, MembershipMethod, VersionState
from models.orm.base import Base, JSONVariant, TolerantEnumType
from models.orm.mixins import TenantMixin, TimestampMixin, tenant_scoped_index

__all__ = ["FeatureLinkRow", "FeatureRow", "VersionRow"]


class VersionRow(TenantMixin, TimestampMixin, Base):
    """A milestone within one project."""

    __tablename__ = "versions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(64),
        # RESTRICT, matching `artifacts.source_id`: deleting a project that still
        # holds versions is a decision about their history, and it has to be made
        # explicitly rather than taken as a side effect.
        ForeignKey("projects.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[VersionState] = mapped_column(
        TolerantEnumType(VersionState), nullable=False, default=VersionState.PLANNED
    )
    shipped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    version_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, default=dict
    )

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_versions_project_id_name"),
        tenant_scoped_index("versions", "project_id", "state"),
    )


class FeatureRow(TenantMixin, TimestampMixin, Base):
    """A capability within one project, optionally assigned to a version."""

    __tablename__ = "features"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    version_id: Mapped[str | None] = mapped_column(
        String(64),
        # SET NULL, not CASCADE. Deleting `v1.1` means the release was cancelled,
        # not that image upload never happened -- cascading would delete the
        # feature and, through it, every record of which work belonged to it.
        ForeignKey("versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    state: Mapped[FeatureState] = mapped_column(
        TolerantEnumType(FeatureState), nullable=False, default=FeatureState.PLANNED
    )

    keywords: Mapped[list[str]] = mapped_column(JSONVariant, nullable=False, default=list)
    feature_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, default=dict
    )

    __table_args__ = (
        UniqueConstraint("project_id", "name", name="uq_features_project_id_name"),
        tenant_scoped_index("features", "project_id", "state"),
    )


class FeatureLinkRow(TenantMixin, TimestampMixin, Base):
    """One artifact's membership of one feature.

    Keyed on the pair rather than on the artifact: membership is genuinely
    many-to-many. A commit adding an image-upload route to the deploy pipeline
    belongs to both features, and a single-owner key would force a wrong answer
    for one of them.
    """

    __tablename__ = "feature_links"

    feature_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("features.id", ondelete="CASCADE"), primary_key=True
    )
    artifact_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("artifacts.id", ondelete="CASCADE"), primary_key=True
    )

    method: Mapped[MembershipMethod] = mapped_column(
        TolerantEnumType(MembershipMethod), nullable=False, index=True
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[str | None] = mapped_column(String(256), nullable=True)

    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[str | None] = mapped_column(String(256), nullable=True)

    __table_args__ = (
        # "Everything in image upload", which is the query the whole layer exists
        # to serve.
        tenant_scoped_index("feature_links", "feature_id", "method"),
        # And the reverse: "which features does this artifact belong to", asked
        # whenever one is displayed.
        tenant_scoped_index("feature_links", "artifact_id"),
    )
