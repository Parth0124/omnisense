"""A project: the thing a question is asked *about*.

`models/artifact.py` records what happened and where it came from. A `Source`
answers "which repository", and for a single-repo product that is almost enough.
It stops being enough the moment a product spans three repositories, a Slack
channel and a Notion space -- because then "what happened this week" is a question
about none of those individually and all of them together.

A project is that grouping, and nothing more:

    project "omnisense"
      ├── source  omnisense/api      (github)
      ├── source  omnisense/web      (github)
      ├── source  #eng-scheduler     (slack)
      └── source  Engineering space  (notion)

Every artifact reaches its project through its source, so `WHERE project_id = ?`
covers every repository and channel the project owns, and adding a fourth source
changes no query anywhere.

One project per source, and why
-------------------------------
A source belongs to exactly one project -- a foreign key on `sources`, not a join
table. The many-to-many version would let a shared library repository sit in two
products at once, which is a real arrangement, and it is still the wrong model
here: it puts a join in front of every read of the most-queried table in the
system to serve a case this product does not have yet.

The case that *does* come up is the reverse -- one repository containing several
products, a monorepo. That is not many-to-many either. A monorepo is one source,
and slicing it by module is a question about file paths, which commits already
carry. Keeping those two axes separate is what stops "which project is this
commit in" from becoming a query that has to consult both.

If a genuinely shared source appears later, the change is a join table and an
update to the handful of reads that resolve a project -- bounded, and much
cheaper than carrying the join from the start.

Layer note: **L0 `models/`** -- imports only `models/` and the standard library.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, field_validator

from models.base import StrictModel

__all__ = [
    "PROJECT_ID_PREFIX",
    "Project",
    "ProjectSource",
    "normalize_slug",
    "project_id",
]

PROJECT_ID_PREFIX = "prj_"

_PROJECT_NAMESPACE = uuid.UUID("8c4d2e1f-6a9b-4d3e-b7c5-1f0a8e2d4b69")
"""Fixed namespace for deterministic project ids. Never regenerate it."""

SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$")
"""Lowercase, digits and hyphens. What a person types on a command line.

Constrained because the slug is the handle: `omnisense catch-up --project
omnisense`. A slug with a space or a capital in it is one the user has to quote
and remember the casing of, and both are ways to be told "no such project" for a
project that exists.
"""

MAX_SLUG_LENGTH = 64


def normalize_slug(value: str) -> str:
    """Turn a display name into a usable slug.

    Applied when the caller supplies none, so `omnisense init` can accept "OmniSense
    API" and produce `omnisense-api` rather than refusing and asking again.
    """
    lowered = value.strip().lower()
    hyphenated = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    return hyphenated[:MAX_SLUG_LENGTH].rstrip("-")


def project_id(tenant_id: str, slug: str) -> str:
    """Deterministic project id from the tenant and the slug.

    Derived rather than random so the CLI can name a project it has not created
    yet, and so re-running `omnisense init` against the same slug addresses the
    same project instead of making a second one beside it.

    Keyed on the *slug* rather than the display name because the name is expected
    to change -- renaming a project should not orphan its sources.
    """
    if not slug:
        raise ValueError("slug must be non-empty; identity cannot be derived")
    return PROJECT_ID_PREFIX + uuid.uuid5(_PROJECT_NAMESPACE, f"{tenant_id}:{slug}").hex


class Project(StrictModel):
    """A product or effort, and the sources that record work on it."""

    id: str
    tenant_id: str = "default"

    slug: str = Field(description="The handle used on the command line. Stable; the name is not.")
    name: str = Field(description="Display name. Freely renameable.")
    description: str | None = Field(
        default=None,
        description="What this project is, in the user's words. Read by the agents: "
        "'the scheduler service' tells a planner more than a repository name does.",
    )

    is_active: bool = Field(
        default=True,
        description="Inactive projects stop syncing but keep their history. Deleting "
        "a project would take its artifacts with it, and 'we stopped working on this' "
        "is not the same as 'this never happened'.",
    )

    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("slug")
    @classmethod
    def _slug_is_usable(cls, value: str) -> str:
        if not SLUG_PATTERN.match(value):
            raise ValueError(
                f"{value!r} is not a usable slug: lowercase letters, digits and "
                "hyphens only, starting and ending with a letter or digit"
            )
        return value


class ProjectSource(StrictModel):
    """One source, with the project it belongs to resolved.

    A read model rather than a table. `sources.project_id` is the storage; this is
    what a caller gets back when it asks what a project contains, so the CLI can
    print a project and its repositories without a second round trip.
    """

    source_id: str
    project_id: str | None
    platform: str
    name: str
    display_name: str | None = None
    url: str | None = None
    is_active: bool = True
    artifact_count: int = 0
