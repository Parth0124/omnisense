"""One human, several accounts.

`Person` is one account on one platform, and that is the right shape for it --
GitHub's `node_id` is a fact, and a row keyed on it is never wrong. But every
question worth asking crosses platforms: *what has Parth been working on* is
answered by a GitHub login, a Slack member id and a Jira account key, and
nothing on any of the three points at the other two.

So this module adds the layer above: an `Identity` is a human, and
`IdentityLink` attaches accounts to it. What matters is not that the link
exists but **how it was arrived at**.

A guess is stored as a guess
----------------------------
There is no shared key between platforms. Email is usually private on GitHub;
display names collide and change; handles are reused by different people on
different services. So every link except one is an *inference*, and the whole
design turns on keeping that visible:

    method=CONFIRMED   a human said so. Certain. Never overwritten by a guess.
    method=EMAIL       two accounts share a verified address. Strong.
    method=HANDLE      identical handle on both platforms. Weak.
    method=DISPLAY_NAME  identical display name. Weakest, and often wrong.

Storing a `HANDLE` match as though it were a fact is how "what has Parth been
working on" quietly starts including somebody else's commits -- and a wrong merge
is far worse than a missing one, because the answer still looks complete.
`services/identity_service.py` therefore *suggests* and never auto-merges above
a threshold; the merge itself is always somebody's decision.

Why this is not a foreign key on `people`
-----------------------------------------
A nullable `identity_id` column would hold the conclusion and lose the argument:
no confidence, no method, no way to review what was inferred or to undo one bad
merge without hunting for its consequences. A link table costs one join and keeps
the reasoning attached to the claim.

Layer note: **L0 model.** Imports `models/` only.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import Field, model_validator

from models.base import StrictModel
from models.enums import Platform

__all__ = [
    "IDENTITY_ID_PREFIX",
    "Identity",
    "IdentityLink",
    "LinkMethod",
    "identity_id",
]

IDENTITY_ID_PREFIX = "idn_"

_IDENTITY_NAMESPACE = uuid.UUID("6f1f0f3a-0f2f-4b1e-9c3d-2a7b5e8c1d40")
"""Fixed namespace for derived identity ids.

Its own, not the artifact one: an identity is not derived from a platform id, and
sharing a namespace would mean a future change to how artifact ids are built
silently reassigned every human in the database.
"""


class LinkMethod(enum.StrEnum):
    """How an account came to be attached to a human.

    A plain `StrEnum`, not a tolerant one -- same reasoning as `ArtifactKind`.
    This value decides whether a link is trusted, and a `LinkMethod` that
    degraded to `UNKNOWN` on an unrecognised string would silently downgrade a
    confirmation into a guess, which is the exact failure the type exists to
    prevent.
    """

    CONFIRMED = "confirmed"
    """A person said so. The only kind that is not an inference."""

    EMAIL = "email"
    """Both accounts carry the same address."""

    HANDLE = "handle"
    """Same username on both platforms. Common, and commonly wrong."""

    DISPLAY_NAME = "display_name"
    """Same display name. The weakest signal, kept only to suggest with."""

    @property
    def is_confirmed(self) -> bool:
        return self is LinkMethod.CONFIRMED


DEFAULT_CONFIDENCE: dict[LinkMethod, float] = {
    LinkMethod.CONFIRMED: 1.0,
    LinkMethod.EMAIL: 0.9,
    LinkMethod.HANDLE: 0.6,
    LinkMethod.DISPLAY_NAME: 0.35,
}
"""What each method is worth when nothing more specific is known.

Deliberately spread wide rather than clustered near one. These numbers are read
by a person deciding whether to accept a suggestion, and three methods all
scoring 0.8-something would tell them nothing.
"""


class Identity(StrictModel):
    """One human, across every platform they appear on.

    Holds no platform fields of its own. A display name here is a label chosen
    for reading, not a value synced from anywhere -- the moment it were, the
    identity would silently become "the GitHub account, plus some others".
    """

    id: str
    tenant_id: str = "default"

    display_name: str = Field(
        min_length=1,
        max_length=256,
        description="What to call this human in output. Chosen, not synced.",
    )
    primary_email: str | None = Field(
        default=None,
        description="Only ever an address a platform actually reported. Never inferred.",
    )
    is_bot: bool = Field(
        default=False,
        description=(
            "Bots get identities too -- dependabot commits and comments across "
            "several platforms -- but a briefing usually wants them out, and that "
            "filter needs somewhere to hang."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class IdentityLink(StrictModel):
    """One account attached to one human, and the reasoning behind it."""

    identity_id: str
    person_id: str
    platform: Platform

    method: LinkMethod
    confidence: float = Field(ge=0.0, le=1.0)

    confirmed_at: datetime | None = Field(
        default=None, description="When a human accepted it. Absent on an inference."
    )
    confirmed_by: str | None = None
    note: str | None = Field(
        default=None,
        max_length=500,
        description="Why, when the reason is not obvious from the method alone.",
    )

    @model_validator(mode="after")
    def _confirmation_is_all_or_nothing(self) -> IdentityLink:
        """A confirmed link must carry full confidence, and vice versa.

        The two fields say the same thing, so letting them disagree creates a
        state nobody can act on: a `CONFIRMED` link at 0.6 is either a
        confirmation being second-guessed or a guess wearing a confirmation's
        name, and no reader can tell which. Rejecting the pair at construction is
        cheaper than every consumer having to decide.
        """
        if self.method.is_confirmed and self.confidence < 1.0:
            raise ValueError("a confirmed link carries confidence 1.0")
        if not self.method.is_confirmed and self.confidence >= 1.0:
            raise ValueError(
                f"confidence 1.0 is reserved for {LinkMethod.CONFIRMED.value!r}; "
                f"{self.method.value!r} is an inference"
            )
        return self


def identity_id(tenant_id: str, seed: str) -> str:
    """Deterministic identity id from a tenant and a seed.

    The seed is whichever account the identity was first created around -- its
    `person_id`. Deterministic so that re-running discovery on an unchanged
    database produces the same identities rather than a second set beside the
    first, exactly as `artifact_id` makes a re-sync an upsert.

    Note what this means: an identity's id does *not* change when accounts are
    added to it, which is what allows a link to be added or withdrawn without
    disturbing anything that already referenced the human.
    """
    if not seed:
        raise ValueError("seed must be non-empty; identity cannot be derived")
    return IDENTITY_ID_PREFIX + uuid.uuid5(_IDENTITY_NAMESPACE, f"{tenant_id}:{seed}").hex
