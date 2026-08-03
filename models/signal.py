
"""The canonical Signal: one observation, from one source, normalized.

Design Doc §6 fixes the field list; `docs/signal-model.md` defines the semantics.
This module is the single most load-bearing contract in OmniSense -- connectors
produce it, the enrichment pipeline decorates it, five stores persist slices of
it, retrieval ranks it, and every claim in a generated report traces back to one.

Two rules drive most of the design here:

**No platform-shaped code above `connectors/`.** Nothing in `retrieval/`,
`graph/` or `agents/` may branch on which platform data came from except by
reading `Signal.platform`. Anything that does not fit the canonical fields goes
into `metadata`, opaque to everything except the connector that wrote it.

**Identity is derived, never assigned.** `Signal.id` is a pure function of
`(platform, native_id)`. Re-fetching the same item can therefore never create a
second Signal, and all five stores can be written idempotently without
coordination.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, Self

from pydantic import Field, model_validator

from models.base import (
    LenientModel,
    NonEmptyStr,
    Score,
    Sha256Hex,
    StrictModel,
    UtcDatetime,
)
from models.entity import EntityMention
from models.enums import (
    PLATFORM_CATEGORY,
    MediaKind,
    Platform,
    SentimentLabel,
    SourceCategory,
)
from models.lineage import Lineage

__all__ = [
    "ENGAGEMENT_AXIS_WEIGHTS",
    "LANGUAGE_CONFIDENCE_FLOOR",
    "SIGNAL_ID_NAMESPACE",
    "SIGNAL_ID_PREFIX",
    "Author",
    "Content",
    "EmbeddingRef",
    "Engagement",
    "Keyword",
    "Language",
    "MediaRef",
    "Polarity",
    "Sentiment",
    "SentimentTarget",
    "Signal",
    "TopicScore",
    "signal_id",
]


Polarity = Annotated[float, Field(ge=-1.0, le=1.0)]
"""Sentiment polarity: -1.0 maximally negative, +1.0 maximally positive."""


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #

SIGNAL_ID_NAMESPACE = uuid.UUID("6f2a1c9e-0000-5000-8000-6f6d6e697365")
"""Fixed UUIDv5 namespace for Signal identity. Never rotated.

Rotating it would re-derive every id in the system, orphaning every stored
vector, index entry, graph edge and report citation simultaneously. Changing
identity derivation requires a new `schema_version` and a full re-ingest
(`docs/signal-model.md` §7) -- it is explicitly not migratable in place.

NOTE: `docs/signal-model.md` §4.1 printed this namespace with a 14-character
final group, which is not a valid UUID and raises `ValueError` in `uuid.UUID()`.
The value here is the corrected 12-character form; the doc has been fixed to
match. The trailing bytes spell "omnise" in ASCII.
"""

SIGNAL_ID_PREFIX = "sig_"
"""Human-readable prefix so a Signal id is never mistaken for a bare UUID.

`docs/api-reference.md` §3.2 distinguishes prefixed opaque ids from raw UUIDs at
the API boundary; keeping the prefix in the id itself means the distinction
survives being copied into a log line or a report citation.
"""


def signal_id(platform: Platform | str, native_id: str) -> str:
    """Derive the deterministic Signal id (`docs/signal-model.md` §4.1).

    Pure and total: the same inputs always produce the same id, on any machine,
    in any process, forever. That property is what lets `scripts/reindex.py`
    rebuild Qdrant and OpenSearch from PostgreSQL without a coordination step.

    `native_id` is chosen by the connector using the first rule that applies:
      1. the platform's own stable item id (Reddit `t3_1abcde`, arXiv `2401.01234v2`);
      2. sha256 of the canonicalized URL, for feeds without a guid;
      3. sha256 of platform | author | timestamp | simhash(text), as a last resort.

    Rule 3 makes identity depend on cleaned text, so a change to the cleaner will
    fork identity. Connectors that need it must say so in their module docstring.
    """
    if not native_id:
        raise ValueError("native_id must be non-empty; identity cannot be derived")
    platform_value = platform.value if isinstance(platform, Platform) else str(platform)
    return SIGNAL_ID_PREFIX + uuid.uuid5(
        SIGNAL_ID_NAMESPACE, f"{platform_value}:{native_id}"
    ).hex


# --------------------------------------------------------------------------- #
# Nested content types
# --------------------------------------------------------------------------- #


class Author(StrictModel):
    """Who produced the observation.

    Never a bare string. `platform_author_id` is the platform's stable
    identifier, *not* the display handle -- handles are renameable, and keying on
    one would silently fork an author's history the first time they renamed.
    """

    platform_author_id: NonEmptyStr
    handle: str | None = None
    display_name: str | None = None
    profile_url: str | None = None
    follower_count: int | None = Field(default=None, ge=0)
    verified: bool = False
    account_age_days: int | None = Field(default=None, ge=0)

    # `follower_count` and `account_age_days` feed the `source_credibility`
    # component of confidence and are snapshotted at fetch time, never
    # refreshed: a report must reflect what was true when it was written.


class Content(StrictModel):
    """The observation's text, cleaned, plus a pointer to the immutable original."""

    title: str | None = None
    text: str = Field(
        description="The cleaned body: markup stripped, boilerplate removed, "
        "whitespace collapsed. May be empty for media-only posts."
    )
    char_count: int = Field(default=0, ge=0)
    truncated: bool = Field(
        default=False,
        description="True when the connector could not obtain the full body "
        "(paywall, API excerpt limit). Caps the content_integrity component.",
    )
    content_type: str = "text/plain"
    raw_ref: str | None = Field(
        default=None,
        description="R2 object key of the immutable original, so a cleaning bug "
        "is repairable by reprocessing rather than re-fetching. Re-fetching is "
        "lossy: posts get deleted and API windows expire.",
    )
    raw_sha256: Sha256Hex | None = None

    @model_validator(mode="before")
    @classmethod
    def _derive_char_count(cls, data: Any) -> Any:
        """Fill `char_count` from `text` when the caller did not supply it.

        Runs `before` rather than `after` so it does not fight
        `validate_assignment`: an after-validator that assigns to a field
        re-enters validation.
        """
        if isinstance(data, dict) and "text" in data and not data.get("char_count"):
            text = data.get("text")
            if isinstance(text, str):
                data = {**data, "char_count": len(text)}
        return data

    @property
    def is_empty(self) -> bool:
        """Whether there is any text to embed, index or analyze."""
        return not self.text.strip()


class MediaRef(StrictModel):
    """An image, video, audio clip or document attached to the observation.

    `source_url` is where it came from; `object_key` is where OmniSense put it.
    Both are kept: the source URL rots, and the archived copy is what a citation
    six months from now actually resolves against.
    """

    kind: MediaKind
    source_url: str | None = None
    object_key: str | None = None
    mime_type: str | None = None
    bytes: int | None = Field(default=None, ge=0)
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    duration_s: float | None = Field(default=None, ge=0.0)
    transcript_ref: str | None = Field(
        default=None,
        description="R2 key of a derived transcript or caption, when one exists. "
        "This is what makes a video searchable by the text pipeline.",
    )


LANGUAGE_CONFIDENCE_FLOOR = 0.70
"""Below this detector confidence, language is recorded as `und` rather than guessed.

`docs/signal-model.md` §3.3: `und` signals are *excluded* from language-filtered
retrieval rather than being silently treated as English, which is what a naive
default would do to every short or code-switched post.
"""


class Language(StrictModel):
    """Detected language of `content.text`, with the detector's own confidence."""

    code: str = Field(
        default="und",
        description="BCP-47 code, or 'und' when detection was inconclusive.",
    )
    confidence: Score = 0.0
    detector: str | None = None
    script: str | None = Field(default=None, description="ISO 15924, e.g. 'Latn'.")
    is_machine_translated: bool = False

    @classmethod
    def detected(
        cls,
        code: str,
        confidence: float,
        detector: str,
        script: str | None = None,
    ) -> Self:
        """Build a `Language` applying the confidence floor policy.

        A factory rather than a validator: the model must be able to represent
        what the detector actually reported (including a low-confidence guess,
        which is worth keeping in `lineage` for tuning). The *policy* of refusing
        to act on a weak guess belongs at the call site in
        `services/signal_engine/language.py`, and this is that call site's helper.
        """
        if confidence < LANGUAGE_CONFIDENCE_FLOOR:
            return cls(code="und", confidence=confidence, detector=detector, script=script)
        return cls(code=code, confidence=confidence, detector=detector, script=script)

    @property
    def is_determinate(self) -> bool:
        """Whether this language may be used as a retrieval filter."""
        return self.code != "und"


class EmbeddingRef(StrictModel):
    """A *reference* to a vector, never the vector itself.

    Raw float arrays never travel inside a Signal. A 1536-dimension vector is
    ~6 KB of JSON; carrying one per chunk through Kafka, Postgres and every API
    response would multiply the size of the system's hottest object for no
    reader's benefit. Qdrant holds the vectors; this holds the address.
    """

    model: NonEmptyStr
    dimensions: int = Field(gt=0)
    chunk_index: int = Field(ge=0)
    collection: NonEmptyStr
    point_id: NonEmptyStr


class SentimentTarget(StrictModel):
    """Polarity toward one specific entity, rather than the post overall.

    "The hardware is great but the software is unusable" is neutral overall and
    useless at that granularity. Per-target stance is what makes it actionable.
    """

    entity_id: NonEmptyStr
    polarity: Polarity


class Sentiment(StrictModel):
    """Sentiment of the observation, overall and optionally per entity."""

    polarity: Polarity
    label: SentimentLabel
    subjectivity: Score | None = None
    targets: list[SentimentTarget] = Field(default_factory=list)
    model: str | None = None
    confidence: Score | None = None


class TopicScore(StrictModel):
    """A scored assignment from the *closed* topic vocabulary.

    Distinct from `Keyword`: topics are a curated closed set and are safe to
    aggregate across sources; keywords are open-vocabulary and are not.
    """

    topic: NonEmptyStr
    score: Score


class Keyword(StrictModel):
    """An open-vocabulary salient term, used to boost BM25 queries."""

    term: NonEmptyStr
    weight: Score


ENGAGEMENT_AXIS_WEIGHTS: dict[str, float] = {
    "reach": 0.30,
    "endorsement": 0.30,
    "amplification": 0.20,
    "discussion": 0.20,
}
"""Default weights for combining engagement axes into a single score.

Renormalized over whichever axes a platform actually supports, so an RSS item
with no `endorsement` axis is not penalized for lacking one. Changing these is a
`pipeline_version` bump and a backfill, not a schema migration
(`docs/signal-model.md` §7).
"""


class Engagement(StrictModel):
    """Platform counters, plus cross-platform comparable axes.

    A Reddit score of 400 and a YouTube view count of 400 are not the same event.
    `raw` preserves the platform's own counters verbatim; the four normalized
    axes are the only thing cross-platform code is allowed to read.

    Each axis is the empirical percentile of the raw value within the same
    `(platform, content_type)` cohort over a trailing window -- so a 400-point
    Reddit post is scored against other Reddit posts, never against YouTube.

    Review *ratings* are not engagement. A 1-star rating is polarity and belongs
    in `Sentiment`; `helpful_votes` on that review is endorsement.
    """

    raw: dict[str, float | int | None] = Field(
        default_factory=dict,
        description="The platform's own counters, verbatim and un-normalized.",
    )
    reach: Score | None = None
    endorsement: Score | None = None
    amplification: Score | None = None
    discussion: Score | None = None
    score: Score | None = None
    normalized_at: UtcDatetime | None = None
    baseline_window: str | None = Field(
        default=None,
        description="Cohort and window used, e.g. 'reddit:text_post:30d'.",
    )

    def available_axes(self) -> dict[str, float]:
        """The axes this platform actually populated."""
        return {
            name: value
            for name, value in (
                ("reach", self.reach),
                ("endorsement", self.endorsement),
                ("amplification", self.amplification),
                ("discussion", self.discussion),
            )
            if value is not None
        }

    def compute_score(
        self, weights: dict[str, float] | None = None
    ) -> float | None:
        """Weighted mean of the available axes, weights renormalized.

        Returns `None` when no axis is populated -- an honest "unknown" rather
        than a misleading 0.0, which retrieval would read as "nobody engaged".
        """
        axes = self.available_axes()
        if not axes:
            return None
        table = weights or ENGAGEMENT_AXIS_WEIGHTS
        total_weight = sum(table.get(name, 0.0) for name in axes)
        if total_weight <= 0.0:
            return None
        weighted = sum(value * table.get(name, 0.0) for name, value in axes.items())
        return round(weighted / total_weight, 6)


# --------------------------------------------------------------------------- #
# The Signal
# --------------------------------------------------------------------------- #


def _json_depth(value: Any, current: int = 0) -> int:
    """Number of nested *container* levels in a JSON-like structure.

    Scalars are depth 0, so a flat mapping is depth 1:

        {"reddit.subreddit": "selfhosted"}          -> 1
        {"reddit.awards": {"gold": 2}}              -> 2
        {"reddit.awards": {"gold": [1, 2]}}         -> 3
        {"a": {"b": {"c": {"d": 1}}}}               -> 4  (rejected)

    Lists count as a level: OpenSearch flattens an array of objects into
    per-path fields exactly as it does a nested object, so exempting them would
    let the mapping explode through the back door.
    """
    if isinstance(value, dict):
        return max(
            (_json_depth(v, current + 1) for v in value.values()), default=current + 1
        )
    if isinstance(value, (list, tuple)):
        return max((_json_depth(v, current + 1) for v in value), default=current + 1)
    return current


class Signal(StrictModel):
    """The canonical unit of data in OmniSense (Design Doc §6).

    Field order below follows the design document exactly, so the two can be
    diffed against each other. `schema_version` and `status` are carried inside
    `lineage` rather than as top-level fields, keeping the §6 field set intact.
    """

    # 1-8: set by the connector
    id: NonEmptyStr
    source: SourceCategory
    platform: Platform
    url: str | None = None
    author: Author | None = None
    timestamp: UtcDatetime = Field(
        description="Event time at the source, not ingestion time. Trend and "
        "forecast agents key off this field exclusively; ingestion time lives in "
        "lineage.fetched_at."
    )
    content: Content
    media: list[MediaRef] = Field(default_factory=list)

    # 9-14: added by the enrichment pipeline
    language: Language = Field(default_factory=Language)
    entities: list[EntityMention] = Field(default_factory=list)
    topics: list[TopicScore] = Field(default_factory=list)
    keywords: list[Keyword] = Field(default_factory=list)
    embeddings: list[EmbeddingRef] = Field(default_factory=list)
    sentiment: Sentiment | None = None

    # 15-16: scoring
    engagement: Engagement = Field(default_factory=Engagement)
    confidence: Score = Field(
        default=0.0,
        description="How much an agent should trust a claim resting on this "
        "Signal alone. Not sentiment confidence, not retrieval relevance, and "
        "not the confidence shown on a report -- that is computed by the Critic "
        "over a whole evidence set.",
    )

    # 17-18: overflow and provenance
    metadata: dict[str, Any] = Field(default_factory=dict)
    lineage: Lineage

    # ------------------------------------------------------------ validators --

    @model_validator(mode="after")
    def _check_identity(self) -> Self:
        """`id` must be the derived function of `(platform, lineage.native_id)`.

        This is the invariant that makes every store idempotent, so it is
        enforced rather than trusted. Without the check, a connector that
        assigned `uuid4()` would work perfectly in tests and silently create
        duplicate Signals on every re-sync in production.

        Use `Signal.create()` to build one correctly.
        """
        expected = signal_id(self.platform, self.lineage.native_id)
        if self.id != expected:
            raise ValueError(
                f"Signal.id {self.id!r} is not derived from "
                f"(platform={self.platform.value!r}, native_id="
                f"{self.lineage.native_id!r}); expected {expected!r}. "
                "Ids are derived, never assigned -- use Signal.create()."
            )
        return self

    @model_validator(mode="after")
    def _check_source_matches_platform(self) -> Self:
        """`source` must be the category `platform` belongs to.

        Both are separate fields in Design Doc §6, which means they can disagree.
        `PLATFORM_CATEGORY` is the single place that decides, so a connector
        cannot declare `platform='reddit', source='news'` and quietly corrupt
        every category-level aggregate downstream.
        """
        expected = PLATFORM_CATEGORY.get(self.platform, SourceCategory.UNKNOWN)
        if self.source is not expected:
            raise ValueError(
                f"source {self.source.value!r} does not match platform "
                f"{self.platform.value!r}, which belongs to {expected.value!r}"
            )
        return self

    @model_validator(mode="after")
    def _check_metadata_shape(self) -> Self:
        """`metadata` must stay shallow and JSON-serializable.

        It is persisted as a Postgres `jsonb` column, a Qdrant payload and an
        OpenSearch object simultaneously. Deep nesting explodes the OpenSearch
        mapping (every path becomes a field) and makes payload filters
        unwritable, so depth is capped at 3 per `docs/signal-model.md` §2.
        """
        depth = _json_depth(self.metadata)
        if depth > 3:
            raise ValueError(
                f"metadata nests {depth} levels deep; the limit is 3. Flatten it "
                "or store the structure in R2 and reference the key."
            )
        return self

    # --------------------------------------------------------------- factory --

    @classmethod
    def create(
        cls,
        *,
        platform: Platform,
        native_id: str,
        timestamp: Any,
        content: Content,
        lineage: Lineage,
        **fields: Any,
    ) -> Self:
        """Build a Signal with a correctly derived `id` and consistent `source`.

        The sanctioned constructor for connectors. Callers supply `native_id`
        once, on `lineage`, and both the id and the source category follow from
        it rather than being restated (and eventually mis-stated) by hand.
        """
        if lineage.native_id != native_id:
            raise ValueError(
                f"native_id {native_id!r} disagrees with lineage.native_id "
                f"{lineage.native_id!r}; they are the same value"
            )
        return cls(
            id=signal_id(platform, native_id),
            source=PLATFORM_CATEGORY.get(platform, SourceCategory.UNKNOWN),
            platform=platform,
            timestamp=timestamp,
            content=content,
            lineage=lineage,
            **fields,
        )

    # ------------------------------------------------------------- accessors --

    @property
    def is_retrievable(self) -> bool:
        """Whether retrieval may return this Signal (`docs/signal-model.md` §5.4)."""
        return self.lineage.status.is_retrievable

    @property
    def is_canonical(self) -> bool:
        """Whether this is the indexed member of its dedup cluster.

        A Signal with no cluster is trivially canonical. Only canonical members
        are embedded into Qdrant and indexed into OpenSearch, so retrieval
        returns one hit for a press release that appeared in six places (§4.3).
        """
        return self.lineage.duplicate_of is None

    def resolved_entity_ids(self) -> list[str]:
        """Distinct canonical entity ids mentioned, preserving first-seen order."""
        seen: dict[str, None] = {}
        for mention in self.entities:
            if mention.resolved_id is not None:
                seen.setdefault(mention.resolved_id, None)
        return list(seen)


class SignalView(LenientModel):
    """Read-only projection of a Signal for consumers.

    `docs/signal-model.md` §7: producers validate strictly, consumers validate
    leniently. `retrieval/`, `agents/` and `backend/schemas/signal.py` read
    through this model so that a Signal written by a newer `pipeline_version`
    -- carrying a field this process has never heard of -- is still readable
    during a rolling deploy instead of raising.

    Deliberately omits the identity, source/platform and metadata-shape
    validators: a consumer is not the right place to discover that a producer
    violated an invariant, and raising there would take down the reader rather
    than the writer that caused it.
    """

    id: str
    source: SourceCategory
    platform: Platform
    url: str | None = None
    author: Author | None = None
    timestamp: UtcDatetime
    content: Content
    media: list[MediaRef] = Field(default_factory=list)
    language: Language = Field(default_factory=Language)
    entities: list[EntityMention] = Field(default_factory=list)
    topics: list[TopicScore] = Field(default_factory=list)
    keywords: list[Keyword] = Field(default_factory=list)
    embeddings: list[EmbeddingRef] = Field(default_factory=list)
    sentiment: Sentiment | None = None
    engagement: Engagement = Field(default_factory=Engagement)
    confidence: Score = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    lineage: Lineage
