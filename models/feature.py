"""What the work was *for*: versions, features, and which artifacts belong to them.

`Artifact` records that something happened. That is enough to answer "what
happened last week" -- slice by date, done -- and it is not enough for anything
else the product promises. "What is blocking image upload" needs *image upload*
to be a thing that exists; without one, the only available move is to search for
the words every time, which gives a different answer on Tuesday than on Monday
and can never say whether the work is finished.

Two levels, because one does not do both jobs
---------------------------------------------
    Version   v1, v1.1        answers "is it shipped?"
      └─ Feature  image upload   answers "what is it?"
           └─ Artifact  commit, PR, CI run

A version alone cannot say what is left; a feature alone cannot say whether it
shipped. Both are askable at their own grain -- `blocking v1.1` and
`blocking "image upload"` are the same query, zoomed differently.

Membership is a guess until somebody says otherwise
---------------------------------------------------
Nothing in a commit says which feature it belongs to. The system infers it from
titles, branch names and paths, and `FeatureLink` keeps *how* it decided right
next to the claim -- exactly as `models/identity.py` does for accounts, and for
the same reason. Assigning by hand is accurate and nobody keeps it up; assigning
silently is effortless and wrong in ways that never surface. Proposing and then
being corrected is the only version of this that stays true for more than a week.

Layer note: **L0 model.**
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from models.base import StrictModel

__all__ = [
    "FEATURE_ID_PREFIX",
    "VERSION_ID_PREFIX",
    "Feature",
    "FeatureLink",
    "FeatureState",
    "MembershipMethod",
    "Version",
    "VersionState",
    "feature_id",
    "version_id",
]

VERSION_ID_PREFIX = "ver_"
FEATURE_ID_PREFIX = "ftr_"

_FEATURE_NAMESPACE = uuid.UUID("2c9e77b1-4a35-4de6-9f18-0b6a3c5d7e21")


class VersionState(enum.StrEnum):
    """Where a version is. Strict -- this drives what gets reported as outstanding."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    SHIPPED = "shipped"
    ABANDONED = "abandoned"
    """Shipped and abandoned are both endings, and conflating them loses the only
    distinction anybody asks about later."""


class FeatureState(enum.StrEnum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    DROPPED = "dropped"


class MembershipMethod(enum.StrEnum):
    """How an artifact came to belong to a feature.

    Mirrors `LinkMethod` in `models/identity.py` deliberately: the same problem
    -- an inference that must not be mistaken for a fact -- deserves the same
    shape, so that a reader who has understood one already understands the other.
    """

    CONFIRMED = "confirmed"
    """A person said so."""

    EXCLUDED = "excluded"
    """A person said the opposite.

    Stored rather than deleted, and that is the whole reason this member exists.
    A rejected guess that is merely removed comes straight back on the next pass,
    and the person who rejected it has to reject it again, forever. Recording the
    rejection is what makes correcting the system worth a person's time.
    """

    BRANCH = "branch"
    """The branch name matched the feature. The strongest inference available --
    somebody chose that name on purpose."""

    TITLE = "title"
    """The title or message mentioned it."""

    PATH = "path"
    """A changed file sits under a path associated with the feature."""

    @property
    def is_decided(self) -> bool:
        """Whether a person settled this, either way."""
        return self in (MembershipMethod.CONFIRMED, MembershipMethod.EXCLUDED)


DEFAULT_MEMBERSHIP_CONFIDENCE: dict[MembershipMethod, float] = {
    MembershipMethod.CONFIRMED: 1.0,
    MembershipMethod.EXCLUDED: 1.0,
    MembershipMethod.BRANCH: 0.8,
    MembershipMethod.TITLE: 0.55,
    MembershipMethod.PATH: 0.4,
}
"""What each signal is worth.

Branch names score highest because they are *chosen*: `feature/image-upload` is a
person declaring intent, not a coincidence of vocabulary. A title mention is
weaker -- "fixed the dotenv issue for image upload" is about deployment. A shared
path is weakest of all, since most changes touch files several features share.
"""


class Version(StrictModel):
    """A milestone. Holds features; answers whether they shipped."""

    id: str
    tenant_id: str = "default"
    project_id: str

    name: str = Field(min_length=1, max_length=64, description="v1, v1.1, 'launch'.")
    description: str | None = Field(default=None, max_length=2000)
    state: VersionState = VersionState.PLANNED

    shipped_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shipping_needs_a_date(self) -> Version:
        """A shipped version without a date cannot be placed in a timeline, which
        is most of what anybody wants a shipped version for."""
        if self.state is VersionState.SHIPPED and self.shipped_at is None:
            raise ValueError("a shipped version needs shipped_at")
        return self


class Feature(StrictModel):
    """A capability. Holds artifacts; answers what the work was."""

    id: str
    tenant_id: str = "default"
    project_id: str

    version_id: str | None = Field(
        default=None,
        description=(
            "Nullable, and that is a real state: work often starts before anybody "
            "decides which release it lands in, and forcing a choice then would "
            "mean inventing a version to hold it."
        ),
    )

    name: str = Field(min_length=1, max_length=128)
    description: str | None = Field(default=None, max_length=2000)
    state: FeatureState = FeatureState.PLANNED

    keywords: list[str] = Field(
        default_factory=list,
        max_length=20,
        description=(
            "Extra terms that mean this feature. 'cloudinary' for image upload -- "
            "the word the commits actually used, which is rarely the feature's name."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureLink(StrictModel):
    """One artifact's membership of one feature, and how it was decided."""

    feature_id: str
    artifact_id: str

    method: MembershipMethod
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str | None = Field(
        default=None,
        max_length=256,
        description=(
            "The matched text itself. A person can judge 'branch: feature/upload' "
            "instantly and cannot judge '0.8'."
        ),
    )

    decided_at: datetime | None = None
    decided_by: str | None = None

    @model_validator(mode="after")
    def _a_decision_is_certain_and_an_inference_is_not(self) -> FeatureLink:
        """Same guard as `IdentityLink`, for the same reason: `method` and
        `confidence` state the same thing, so a pair that disagrees leaves every
        reader to invent its own interpretation."""
        if self.method.is_decided and self.confidence < 1.0:
            raise ValueError(f"{self.method.value!r} is a decision and carries confidence 1.0")
        if not self.method.is_decided and self.confidence >= 1.0:
            raise ValueError(
                f"confidence 1.0 is reserved for a person's decision; "
                f"{self.method.value!r} is an inference"
            )
        return self


def version_id(tenant_id: str, project_id: str, name: str) -> str:
    """Deterministic, so re-declaring `v1` lands on the row that already exists."""
    if not name:
        raise ValueError("name must be non-empty; identity cannot be derived")
    return (
        VERSION_ID_PREFIX
        + uuid.uuid5(
            _FEATURE_NAMESPACE, f"version:{tenant_id}:{project_id}:{name.strip().casefold()}"
        ).hex
    )


def feature_id(tenant_id: str, project_id: str, name: str) -> str:
    if not name:
        raise ValueError("name must be non-empty; identity cannot be derived")
    return (
        FEATURE_ID_PREFIX
        + uuid.uuid5(
            _FEATURE_NAMESPACE, f"feature:{tenant_id}:{project_id}:{name.strip().casefold()}"
        ).hex
    )
