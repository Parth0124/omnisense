"""Wire shapes for `/api/v1/signals` (`docs/api-reference.md` §4.7).

`models/signal.py::SignalView` is already the consumer-facing projection, so the
temptation is to return it directly. These exist instead, and the difference is
one deliberate decision: **`SignalView` is a `LenientModel`.** It ignores unknown
fields so a reader running older code can still parse a Signal written by a newer
`pipeline_version` during a rolling deploy. That tolerance is exactly right
inside the system and exactly wrong at the boundary -- returning it would mean a
field added by a newer producer silently appears in the public API before anyone
decided it should.

The response models below inherit `ResponseModel`, which forbids extras on the
way out. A new internal field therefore has to be added *here* to become public,
which is a one-line diff somebody reviews rather than a deploy nobody noticed.

Two things are deliberately absent from every response:

**`lineage`.** It records which pipeline version, which prompt hash and which
stage degraded. That is operational detail; publishing it would make internal
processing decisions part of the contract, and the first refactor of the
enrichment pipeline would be a breaking API change.

**`raw_sha256` and dedup internals.** A caller gets `is_canonical` and
`duplicate_of` because those change how a result should be read; the hash that
produced them is bookkeeping.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Final

from pydantic import Field

from backend.schemas.common import ResponseModel

__all__ = [
    "MAX_SEARCH_CHARS",
    "MAX_TEXT_CHARS",
    "AuthorRef",
    "EngagementCounts",
    "EntityMentionItem",
    "SentimentBand",
    "SentimentSummary",
    "SignalDetail",
    "SignalItem",
    "SignalPageResponse",
    "TopicScoreItem",
]

MAX_TEXT_CHARS: Final = 4000
"""Cap on body text in a *list* response.

A signals page is fifty items. Fifty full articles is megabytes over the wire for
a view that renders a snippet, and it is the single easiest way to make this
endpoint slow. `SignalDetail` returns the full text; the list returns a prefix
and says so via `text_truncated`.
"""

MAX_SEARCH_CHARS: Final = 500
"""Cap on the `q` parameter.

The string reaches OpenSearch's query parser, where a pathological input costs
the *server* time proportional to something the client controls.
"""


class SentimentBand(enum.StrEnum):
    """Coarse sentiment, for a badge.

    Alongside the raw score rather than instead of it. The number is what
    filtering and aggregation use; the band is what a UI renders, and letting a
    UI pick its own thresholds means two screens in the same product disagree
    about whether -0.15 is negative.
    """

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"


class AuthorRef(ResponseModel):
    """Who published it, as much as may be published.

    No follower counts, no bio, no location, no profile URL beyond the handle.
    `docs/security-and-privacy.md`: a private individual posting on a forum is not
    a subject to be profiled, and an API that returns an aggregated author record
    makes building that profile trivial for anyone with a token.
    """

    handle: str | None = None
    display_name: str | None = None
    is_verified: bool = False


class EngagementCounts(ResponseModel):
    reach: int = 0
    endorsement: int = 0
    amplification: int = 0
    discussion: int = 0
    score: float | None = Field(
        default=None,
        description=(
            "Weighted composite, 0-1, comparable only within a platform. Cross-"
            "platform comparison is meaningless -- a hundred Reddit upvotes and a "
            "hundred X likes are not the same quantity of anything."
        ),
    )


class SentimentSummary(ResponseModel):
    score: float = Field(description="-1 to 1.")
    band: SentimentBand
    confidence: float | None = None


class EntityMentionItem(ResponseModel):
    entity_id: str | None = None
    name: str
    type: str
    salience: float | None = None


class TopicScoreItem(ResponseModel):
    topic: str
    score: float


class SignalItem(ResponseModel):
    """One signal in a list. Snippet, not full text."""

    id: str
    platform: str
    source: str
    url: str | None = None
    timestamp: datetime
    title: str | None = None
    text: str = Field(description=f"Body, truncated to {MAX_TEXT_CHARS} characters.")
    text_truncated: bool = Field(
        description=(
            "True when the body was cut for this response. A client showing a "
            "snippet must not present it as the whole document -- and cannot tell "
            "without being told, since a truncated article reads like a short one."
        )
    )
    language: str | None = None
    author: AuthorRef | None = None
    sentiment: SentimentSummary | None = None
    engagement: EngagementCounts = Field(default_factory=EngagementCounts)
    entities: list[EntityMentionItem] = Field(default_factory=list, max_length=20)
    topics: list[TopicScoreItem] = Field(default_factory=list, max_length=10)
    confidence: float = 0.0
    is_canonical: bool = True
    duplicate_of: str | None = Field(
        default=None,
        description=(
            "Set when this is a duplicate of another signal. A press release "
            "syndicated to six platforms is one thing that happened, and a client "
            "counting all six is overcounting its own evidence."
        ),
    )


class SignalDetail(SignalItem):
    """One signal in full. `text` is complete and `text_truncated` is always false."""

    keywords: list[str] = Field(default_factory=list, max_length=30)
    media_count: int = 0
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Platform-specific fields. Shape varies by platform by design.",
    )


class SignalPageResponse(ResponseModel):
    """A page of signals.

    No total count, matching `docs/api-reference.md` §3.4 and
    `services/signal_service.SignalPage`. `COUNT(*)` over a filtered slice of a
    continuously-written table is unbounded work and stale before it renders --
    and a UI given a number will do arithmetic on it.
    """

    items: list[SignalItem]
    limit: int
    next_cursor: str | None = None
    has_more: bool = False
