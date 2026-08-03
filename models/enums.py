"""Closed vocabularies shared across every OmniSense module.

Every enum whose membership can grow as the platform grows derives from
`TolerantStrEnum` and defines an `UNKNOWN` member, so that a consumer running
older code never raises on a value a newer producer wrote
(`docs/signal-model.md` §7). Enums whose membership is genuinely fixed by the
design -- `StageStatus`, `StageName`, `SignalStatus` -- deliberately omit
`UNKNOWN` so an unexpected value fails loudly instead of being swallowed.

Because these are `StrEnum` members, they compare equal to their string value
(`Platform.REDDIT == "reddit"` is `True`) and serialize as plain strings in JSON,
which keeps Kafka envelopes, Qdrant payloads and OpenSearch documents readable.
"""

from __future__ import annotations

from models.base import TolerantStrEnum

__all__ = [
    "PLATFORM_CATEGORY",
    "AgentName",
    "AuthType",
    "ConnectorErrorClass",
    "EdgeType",
    "EntityType",
    "InvestigationStatus",
    "MediaKind",
    "Platform",
    "SentimentLabel",
    "SignalStatus",
    "SourceCategory",
    "StageName",
    "StageStatus",
]


# --------------------------------------------------------------------------- #
# Source taxonomy (Design Doc §5)
# --------------------------------------------------------------------------- #


class SourceCategory(TolerantStrEnum):
    """Coarse source category. Drives credibility priors and default retention.

    Fixed by Design Doc §5, which groups every connector into exactly one of
    these five buckets.
    """

    SOCIAL = "social"
    REVIEWS = "reviews"
    ENTERPRISE = "enterprise"
    RESEARCH = "research"
    NEWS = "news"
    UNKNOWN = "unknown"


class Platform(TolerantStrEnum):
    """The concrete origin of a Signal -- one member per connector module.

    Adding a member here is a backward-compatible change (`docs/signal-model.md`
    §7) precisely because every reader tolerates unknown members.
    """

    # connectors/social/
    REDDIT = "reddit"
    X = "x"
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    LINKEDIN = "linkedin"

    # connectors/reviews/
    AMAZON = "amazon"
    PLAY_STORE = "play_store"
    APP_STORE = "app_store"
    TRUSTPILOT = "trustpilot"
    GOOGLE_REVIEWS = "google_reviews"

    # connectors/enterprise/
    SLACK = "slack"
    JIRA = "jira"
    CONFLUENCE = "confluence"
    NOTION = "notion"
    GITHUB = "github"
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"

    # connectors/research/
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PAPERS_WITH_CODE = "papers_with_code"

    # connectors/news/
    RSS = "rss"
    GDELT = "gdelt"
    NEWS_API = "news_api"

    UNKNOWN = "unknown"


PLATFORM_CATEGORY: dict[Platform, SourceCategory] = {
    Platform.REDDIT: SourceCategory.SOCIAL,
    Platform.X: SourceCategory.SOCIAL,
    Platform.YOUTUBE: SourceCategory.SOCIAL,
    Platform.INSTAGRAM: SourceCategory.SOCIAL,
    Platform.TIKTOK: SourceCategory.SOCIAL,
    Platform.LINKEDIN: SourceCategory.SOCIAL,
    Platform.AMAZON: SourceCategory.REVIEWS,
    Platform.PLAY_STORE: SourceCategory.REVIEWS,
    Platform.APP_STORE: SourceCategory.REVIEWS,
    Platform.TRUSTPILOT: SourceCategory.REVIEWS,
    Platform.GOOGLE_REVIEWS: SourceCategory.REVIEWS,
    Platform.SLACK: SourceCategory.ENTERPRISE,
    Platform.JIRA: SourceCategory.ENTERPRISE,
    Platform.CONFLUENCE: SourceCategory.ENTERPRISE,
    Platform.NOTION: SourceCategory.ENTERPRISE,
    Platform.GITHUB: SourceCategory.ENTERPRISE,
    Platform.SALESFORCE: SourceCategory.ENTERPRISE,
    Platform.HUBSPOT: SourceCategory.ENTERPRISE,
    Platform.ARXIV: SourceCategory.RESEARCH,
    Platform.SEMANTIC_SCHOLAR: SourceCategory.RESEARCH,
    Platform.PAPERS_WITH_CODE: SourceCategory.RESEARCH,
    Platform.RSS: SourceCategory.NEWS,
    Platform.GDELT: SourceCategory.NEWS,
    Platform.NEWS_API: SourceCategory.NEWS,
    Platform.UNKNOWN: SourceCategory.UNKNOWN,
}
"""Canonical platform -> category mapping.

`Signal.source` and `Signal.platform` are separate fields in Design Doc §6, which
means they can disagree. This table is the single place that decides, so a
connector cannot declare `platform="reddit", source="news"`. `Signal` enforces it
in a model validator.
"""


# --------------------------------------------------------------------------- #
# Signal lifecycle
# --------------------------------------------------------------------------- #


class SignalStatus(TolerantStrEnum):
    """Lifecycle state of a Signal (`docs/signal-model.md` §5.4).

    Only `ENRICHED` and `PARTIAL` are retrievable. `DUPLICATE` members of a dedup
    cluster still exist and still contribute graph edges and trend volume -- they
    are simply not indexed for retrieval, because the canonical member is
    returned instead (§4.3).

    No `UNKNOWN` member: this is a closed set owned entirely by the pipeline, and
    an unrecognized status means a real bug rather than a version skew.
    """

    RAW = "raw"
    ENRICHED = "enriched"
    PARTIAL = "partial"
    DUPLICATE = "duplicate"
    QUARANTINED = "quarantined"

    @property
    def is_retrievable(self) -> bool:
        """Whether a Signal in this state may be returned by retrieval."""
        return self in (SignalStatus.ENRICHED, SignalStatus.PARTIAL)


class StageName(TolerantStrEnum):
    """The enrichment pipeline stages, in execution order (Design Doc §6).

    `SCORING` is stage 6b and `KEYWORDS` folds into stage 4; both appear here
    because `lineage.stages[]` records them individually
    (`docs/signal-model.md` §5).
    """

    CLEAN = "clean"
    NORMALIZE = "normalize"
    LANGUAGE = "language"
    ENTITIES = "entities"
    SENTIMENT = "sentiment"
    EMBEDDING = "embedding"
    SCORING = "scoring"
    STORE = "store"


class StageStatus(TolerantStrEnum):
    """Outcome of a single enrichment stage. Closed set -- no `UNKNOWN`."""

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


#: Stages whose failure is fatal to the Signal (`docs/signal-model.md` §5.2).
#: Everything else degrades to a documented empty value and stores as `partial`.
FATAL_STAGES: frozenset[StageName] = frozenset(
    {StageName.CLEAN, StageName.NORMALIZE, StageName.STORE}
)

#: Per-stage contribution to the `extraction_quality` confidence component
#: (`docs/signal-model.md` §3.5). Fatal stages carry no weight because a Signal
#: cannot exist without them. Weights sum to 1.0.
STAGE_QUALITY_WEIGHTS: dict[StageName, float] = {
    StageName.LANGUAGE: 0.15,
    StageName.ENTITIES: 0.35,
    StageName.SENTIMENT: 0.20,
    StageName.EMBEDDING: 0.30,
}


# --------------------------------------------------------------------------- #
# Content descriptors
# --------------------------------------------------------------------------- #


class MediaKind(TolerantStrEnum):
    """Kind of media attached to a Signal."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class SentimentLabel(TolerantStrEnum):
    """Discrete sentiment label accompanying the continuous `polarity` score.

    `MIXED` is distinct from `NEUTRAL`: a review that praises the hardware and
    condemns the software is strongly polarized in both directions, and
    averaging it to neutral would erase exactly the signal a Competitor or
    Insight agent is looking for.
    """

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Knowledge graph vocabulary (Design Doc §7)
# --------------------------------------------------------------------------- #


class EntityType(TolerantStrEnum):
    """Knowledge-graph node labels.

    Values are capitalized because they are used verbatim as Neo4j labels --
    `MATCH (n:Company)`. Lower-casing them here would force a translation table
    in `graph/schema/nodes.py`.
    """

    COMPANY = "Company"
    PRODUCT = "Product"
    PERSON = "Person"
    TOPIC = "Topic"
    TECHNOLOGY = "Technology"
    REGION = "Region"
    EVENT = "Event"
    UNKNOWN = "Unknown"


class EdgeType(TolerantStrEnum):
    """Knowledge-graph relationship types.

    The six from Design Doc §7 plus two structural edges the design implies:
    `SAME_AS` records an entity-resolution merge reversibly
    (`docs/knowledge-graph.md`), and `DUPLICATE_OF` mirrors
    `lineage.duplicate_of` into the graph so cluster membership is traversable.

    Values are `SCREAMING_SNAKE` to match Cypher convention.
    """

    MENTIONS = "MENTIONS"
    COMPETES_WITH = "COMPETES_WITH"
    ACQUIRED = "ACQUIRED"
    USES = "USES"
    COMPLAINS_ABOUT = "COMPLAINS_ABOUT"
    LAUNCHED_BY = "LAUNCHED_BY"
    SAME_AS = "SAME_AS"
    DUPLICATE_OF = "DUPLICATE_OF"
    UNKNOWN = "UNKNOWN"


# --------------------------------------------------------------------------- #
# Connector vocabulary (Design Doc §5)
# --------------------------------------------------------------------------- #


class AuthType(TolerantStrEnum):
    """How a connector authenticates. Declared as a `ClassVar` on each connector."""

    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"
    OAUTH2 = "oauth2"
    UNKNOWN = "unknown"


class ConnectorErrorClass(TolerantStrEnum):
    """Connector failure taxonomy (`docs/connector-spec.md` §2, §10).

    The class decides the runtime's response, so it is part of the contract
    rather than an implementation detail:

    | Class       | Retried | Cursor      | Outcome                          |
    | ----------- | ------- | ----------- | -------------------------------- |
    | `AUTH`      | no      | untouched   | account flagged `needs_reauth`   |
    | `QUOTA`     | later   | checkpointed| rescheduled at `reset_at`        |
    | `TRANSIENT` | yes     | untouched   | backoff, then escalate           |
    | `PERMANENT` | no      | untouched   | record to DLQ, continue or abort |
    """

    AUTH = "auth"
    QUOTA = "quota"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Orchestration vocabulary (Design Doc §9, §10)
# --------------------------------------------------------------------------- #


class AgentName(TolerantStrEnum):
    """The ten agents of Design Doc §9. One member per `agents/<name>/` package."""

    PLANNER = "planner"
    COLLECTOR = "collector"
    RETRIEVER = "retriever"
    TREND = "trend"
    COMPETITOR = "competitor"
    FORECAST = "forecast"
    INSIGHT = "insight"
    STRATEGY = "strategy"
    CRITIC = "critic"
    REPORT = "report"
    UNKNOWN = "unknown"


class InvestigationStatus(TolerantStrEnum):
    """Lifecycle of a long-running investigation.

    `COMPLETED_WITH_FINDINGS` is the terminal state when the Critic loop hit
    `MAX_CRITIC_REVISIONS` without reaching an `accept` verdict
    (`docs/agent-system.md` §13): the report ships with its unresolved findings
    surfaced rather than being withheld or silently presented as clean.
    """

    QUEUED = "queued"
    PLANNING = "planning"
    RUNNING = "running"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    COMPLETED_WITH_FINDINGS = "completed_with_findings"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_terminal(self) -> bool:
        """Whether no further state transition is possible."""
        return self in (
            InvestigationStatus.COMPLETED,
            InvestigationStatus.COMPLETED_WITH_FINDINGS,
            InvestigationStatus.FAILED,
            InvestigationStatus.CANCELLED,
        )
