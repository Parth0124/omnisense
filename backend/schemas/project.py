"""Request and response bodies for `/api/v1/projects`.

Separate from `models/project.py` for the reason every schema module here is:
the domain model is what the system stores, and the wire contract is what clients
depend on. Returning the domain object directly makes every internal field a
public promise, and the first one somebody renames is a breaking change nobody
intended to make.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from backend.schemas.common import RequestModel, ResponseModel
from models.project import MAX_SLUG_LENGTH

__all__ = [
    "ProjectCreateRequest",
    "ProjectDetail",
    "ProjectSourceItem",
    "ProjectSummary",
]


class ProjectCreateRequest(RequestModel):
    """What a client sends to create a project."""

    name: str = Field(min_length=1, max_length=256, description="Display name, freely renameable.")
    slug: str | None = Field(
        default=None,
        max_length=MAX_SLUG_LENGTH,
        description=(
            "Command-line handle. Derived from the name when omitted, so 'OmniSense "
            "API' becomes 'omnisense-api'. Stable: renaming the project leaves it alone."
        ),
    )
    description: str | None = Field(
        default=None,
        max_length=4000,
        description=(
            "What this project is, in your words. Read by the agents when they plan -- "
            "'the scheduler service, owns job leasing' tells a planner considerably "
            "more than a repository name does."
        ),
    )


class ProjectSummary(ResponseModel):
    """A project without its sources. What a list returns."""

    id: str
    slug: str
    name: str
    description: str | None = None
    is_active: bool = True
    created_at: datetime | None = None


class ProjectSourceItem(ResponseModel):
    """One source inside a project."""

    source_id: str
    project_id: str | None = None
    platform: str
    name: str = Field(description="Canonical name: 'omnisense/api', '#eng-scheduler'.")
    display_name: str | None = None
    url: str | None = None
    is_active: bool = True
    artifact_count: int = Field(
        default=0,
        description=(
            "How much has actually been ingested from this source. Zero on a source "
            "that has been attached but never synced, which is the difference between "
            "'configured' and 'working' and is worth being able to see."
        ),
    )


class ProjectDetail(ProjectSummary):
    """A project and everything it owns."""

    sources: list[ProjectSourceItem] = Field(default_factory=list)
    artifact_count: int = Field(default=0, description="Summed across every source.")
