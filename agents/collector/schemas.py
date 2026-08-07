"""Collector input and output schemas.

The Collector is the only agent that causes *outbound network traffic to third
parties*, which makes its schemas a security boundary rather than a data
contract. `docs/security-and-privacy.md` §8.2 is explicit: the Collector
dispatches connectors **by slug**, never by URL and never with a credential. A
model that could name a URL to fetch is a model that can be prompt-injected into
exfiltrating whatever it has seen to an attacker-controlled host, and the
scraped content it reads is exactly where such an instruction would arrive.

So `CollectionRequest.connector_slug` is a slug, validated against a pattern that
cannot express a URL. There is no `url` field, no `endpoint`, no `headers`, and
adding one is a decision that needs its own review -- not a convenience.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import Field, field_validator

from models.base import StrictModel

__all__ = [
    "MAX_COLLECTION_REQUESTS",
    "CollectionRequest",
    "CollectorInput",
    "CollectorOutput",
    "CollectorPlan",
]

MAX_COLLECTION_REQUESTS: Final = 8
"""Ceiling on connectors dispatched in one run.

Each dispatch costs minutes of wall clock and a slice of a third-party rate
limit shared with every other investigation in the deployment. Eight is already
generous; the failure mode of no cap is one investigation exhausting a daily
quota that the next fifty runs then have to do without.
"""

_SLUG: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
"""Deliberately narrow: lowercase, no dots, no slashes, no colons.

A pattern that cannot match `http://…`, `//evil.host`, `../`, or anything with a
scheme. The narrowness is the control -- validating that a string "looks like a
slug" with a loose pattern is how `reddit/../../etc/passwd` gets through.
"""


class CollectionRequest(StrictModel):
    """One connector the model wants run.

    No URL. No credentials. No headers. See the module docstring -- this is the
    boundary `docs/security-and-privacy.md` §8.2 draws, and the absence of those
    fields is the mechanism, not an oversight.
    """

    connector_slug: str = Field(min_length=1, max_length=64)
    reason: str = Field(
        min_length=1,
        max_length=300,
        description="Why this source is needed for the plan. Recorded, not acted on.",
    )
    query: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Search terms passed to the connector, where it accepts them. Capped "
            "because it reaches a third-party API as a parameter."
        ),
    )

    @field_validator("connector_slug")
    @classmethod
    def _must_be_a_slug(cls, value: str) -> str:
        if not _SLUG.match(value):
            raise ValueError(
                f"{value!r} is not a connector slug. Connectors are dispatched by "
                "slug only -- never by URL -- so that a prompt-injected instruction "
                "in scraped content cannot direct a fetch at an arbitrary host."
            )
        return value


class CollectorPlan(StrictModel):
    """What the model decided to collect, and what it deliberately skipped."""

    requests: list[CollectionRequest] = Field(
        default_factory=list, max_length=MAX_COLLECTION_REQUESTS
    )
    skipped: list[str] = Field(
        default_factory=list,
        max_length=32,
        description=(
            "Connectors considered and rejected, with the slug only. Recorded so "
            "a thin result set can be explained -- 'we had no Reddit coverage' is "
            "a different finding from 'Reddit had nothing'."
        ),
    )
    rationale: str | None = Field(default=None, max_length=1000)


class CollectorInput(StrictModel):
    """The projection of the state the Collector needs."""

    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    available_connectors: list[str] = Field(default_factory=list, max_length=64)
    fresh_data_steps: list[str] = Field(
        default_factory=list,
        max_length=16,
        description="Plan step descriptions that asked for fresh data.",
    )
    seconds_remaining: float | None = None


class CollectorOutput(StrictModel):
    """The result of dispatching. Counts and errors, never content.

    Signal *content* never passes through the agent state -- it goes to
    PostgreSQL through the ingestion pipeline, and the Retriever reads it back
    from there. Carrying it here would put megabytes of scraped text in every
    checkpoint and make a resume a full re-read.
    """

    dispatched: int = 0
    emitted: int = Field(default=0, description="Raw records the connectors produced.")
    failures: list[str] = Field(default_factory=list, max_length=MAX_COLLECTION_REQUESTS)
    skipped: list[str] = Field(default_factory=list, max_length=32)
    rationale: str | None = None

    @property
    def collected_anything(self) -> bool:
        return self.emitted > 0
