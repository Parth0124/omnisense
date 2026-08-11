"""Storage for `models/project.py`: one table, and the column that hangs sources off it.

The `sources.project_id` foreign key lives here rather than in
`models/orm/artifact.py` for a reason worth stating: it is what makes `sources`
depend on `projects`, and a foreign key declared from the side that does not own
the relationship is the kind of thing that reads correctly and creates a circular
import the first time either module needs the other's type.

`Project` is small on purpose. It is a grouping and a description, not a
configuration object -- sync intervals, credentials and connector settings belong
to the connector account that fetches, not to the thing being asked about.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Boolean, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from models.orm.base import Base, JSONVariant
from models.orm.mixins import TenantMixin, TimestampMixin

__all__ = ["ProjectRow"]


class ProjectRow(TenantMixin, TimestampMixin, Base):
    """A project. Sources point at it; artifacts reach it through their source."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    project_metadata: Mapped[dict[str, Any]] = mapped_column(
        JSONVariant, nullable=False, default=dict
    )

    __table_args__ = (
        # Scoped to the tenant rather than global: two tenants may both have a
        # project called `api`, and they are different projects.
        UniqueConstraint("tenant_id", "slug", name="uq_projects_tenant_id_slug"),
    )
