"""Stage 6b -- Scoring: engagement normalization and the confidence composite.

Two jobs that look unrelated and are not. Both exist so that an agent can
compare two observations that came from different worlds.

**Engagement** (`docs/signal-model.md` §3.4). Platforms expose incomparable
counters -- a Reddit score of 400 and a YouTube view count of 400 are not the
same event. `engagement.raw` keeps the platform's numbers verbatim; the four
normalized axes are the only thing cross-platform code may read. Each axis is
the empirical percentile of the raw value inside its own `(platform,
content_type)` cohort over a trailing window, so a 400-point Reddit post is
scored against other Reddit posts and never against YouTube.

**Confidence** (`docs/signal-model.md` §3.5) answers exactly one question: how
much should an agent trust a claim resting on *this Signal alone*? Not sentiment
confidence, not retrieval relevance, and not the confidence printed on a report
-- that one is the Critic's, computed over a whole evidence set.

The composite is a **weighted geometric mean**, and that choice is the single
most load-bearing decision in this module:

    confidence = Π max(component, 0.05) ** weight

An arithmetic mean lets a healthy majority *average away* a dead component. Hold
a research publisher's Signal fixed at `source_credibility` 0.80, clean
extraction 1.0 and full corroboration 1.0, and vary only the body. With the body
intact (`content_integrity` 1.0) the two means agree: a weighted sum gives 0.93
and `compose_confidence` gives 0.924872. With the body gone -- a media-only post
whose transcript never arrived, so `content_integrity_of` returns 0.0 and the
floor clamps it to 0.05 -- the weighted sum sags only to 0.73 and still reads as
a usable Signal, while the composite drops to 0.508014, because multiplying by a
number close to zero drags the product down no matter what the other factors
are. A title-only scrap (`content_integrity` 0.2) sits between them at 0.670328
against a weighted sum of 0.77.

Those five figures are `compose_confidence` outputs, not hand arithmetic, and
`TestTheModuleDocstringWorkedExample` re-derives them so a weight change cannot
leave this paragraph quietly wrong.

Confidence is a conjunction: a claim is trustworthy if the source is credible
**and** the extraction worked **and** the body is intact **and** something
corroborates it. A mean that lets three strengths compensate for one fatal
weakness is measuring the wrong thing.

`FLOOR = 0.05` keeps one dead component from annihilating the score outright. A
true zero would make confidence exactly 0.0, and a 0.0 is indistinguishable from
"never scored" -- both to the API and to a `WHERE confidence > 0` filter -- so a
Signal that is merely weak would drop out of retrieval entirely rather than
ranking last.

Everything here is deterministic given its inputs (`docs/signal-model.md` §5.1),
which is why the stage records no model id and why replaying it drifts by
nothing. The one external dependency is the cohort baseline store, which is
PostgreSQL and does not exist yet -- so it arrives as a port.

Degradation: this stage never raises for missing data. An axis with no cohort is
omitted (§5.1, "degrade → axis omitted"), and a missing component takes its
documented value. It raises only when something is genuinely wrong, in which
case the pipeline records the failure and stores the Signal `partial`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Protocol, runtime_checkable

from models.base import utcnow
from models.enums import Platform, SourceCategory, StageName
from models.lineage import ConfidenceComponents
from models.signal import Engagement, Signal
from services.signal_engine.pipeline import EnrichmentContext

__all__ = [
    "COHORT_CONTENT_TYPE_KEY",
    "CONFIDENCE_FLOOR",
    "CONFIDENCE_WEIGHTS",
    "MAX_CORROBORATING_SOURCES",
    "MIN_COHORT_SAMPLES",
    "PLATFORM_AXIS_COUNTERS",
    "PLATFORM_CREDIBILITY_PRIOR",
    "SCORING_STAGE_VERSION",
    "SOURCE_CREDIBILITY_PRIOR",
    "ClusterCorroboration",
    "CohortBaseline",
    "CohortObservation",
    "CohortPercentile",
    "ColdStartPolicy",
    "CorroborationSource",
    "InMemoryCohortBaseline",
    "InMemoryCorroborationIndex",
    "ScoringStage",
    "compose_confidence",
    "content_integrity_of",
    "corroboration_of",
    "source_credibility_of",
]


SCORING_STAGE_VERSION: Final = "1.0.0"


# --------------------------------------------------------------------------- #
# The confidence composite (`docs/signal-model.md` §3.5)
# --------------------------------------------------------------------------- #

CONFIDENCE_WEIGHTS: Final[dict[str, float]] = {
    "source_credibility": 0.35,
    "extraction_quality": 0.25,
    "content_integrity": 0.20,
    "corroboration": 0.20,
}
"""Exponents of the weighted geometric mean. Sum to 1.0, as they must.

Weights that did not sum to 1.0 would leave the composite outside [0, 1] --
above it for a sum below one, below it for a sum above one -- and `Score`
validation would reject the Signal at the assignment, which is a confusing place
to discover an arithmetic mistake. `_check_weights()` asserts it at import.

Changing these is a `pipeline_version` bump and a backfill, not a schema
migration (`docs/signal-model.md` §7): every stored score was computed under the
old weights and is no longer comparable to a new one.
"""

CONFIDENCE_FLOOR: Final = 0.05
"""Lower clamp applied to each component before exponentiation. See the module docstring."""


def _check_weights() -> None:
    total = sum(CONFIDENCE_WEIGHTS.values())
    if abs(total - 1.0) > 1e-9:  # pragma: no cover -- a constant, checked at import
        raise ValueError(
            f"CONFIDENCE_WEIGHTS sum to {total}, not 1.0; the geometric mean would "
            "no longer land in [0, 1]"
        )


_check_weights()


def compose_confidence(components: ConfidenceComponents) -> float:
    """Weighted geometric mean of the four components (`docs/signal-model.md` §3.5).

    Kept a module-level function rather than a method so that `agents/critic/`
    and the UI can explain a stored score by recomputing it from the stored
    components, without constructing a stage or its ports.
    """
    product = 1.0
    for name, weight in CONFIDENCE_WEIGHTS.items():
        component: float = getattr(components, name)
        product *= max(component, CONFIDENCE_FLOOR) ** weight
    # Rounded because this value is serialized into five stores and compared
    # across them during reconciliation (`docs/data-stores.md` §6); float noise
    # in the 15th digit would make identical scores look divergent.
    return min(1.0, round(product, 6))


# --------------------------------------------------------------------------- #
# source_credibility
# --------------------------------------------------------------------------- #

SOURCE_CREDIBILITY_PRIOR: Final[dict[SourceCategory, float]] = {
    SourceCategory.RESEARCH: 0.80,
    SourceCategory.ENTERPRISE: 0.75,
    SourceCategory.NEWS: 0.65,
    SourceCategory.REVIEWS: 0.50,
    SourceCategory.SOCIAL: 0.45,
    SourceCategory.UNKNOWN: 0.30,
}
"""Per-category credibility prior. **Unmeasured starting points, not findings.**

The ordering encodes editorial distance: research and internal enterprise
records carry attribution and review, news carries an editor, reviews and social
posts carry neither and are the easiest to fabricate at scale. `UNKNOWN` sits
lowest deliberately -- an unrecognized platform is one this deployment has never
evaluated, and defaulting it to the middle would let a new connector inject
mid-confidence Signals into reports before anyone had looked at its data.
"""

PLATFORM_CREDIBILITY_PRIOR: Final[dict[Platform, float]] = {
    # A preprint is not peer review. Scoring arXiv at the research prior would
    # let an unreviewed submission carry the same weight as a published paper.
    Platform.ARXIV: 0.70,
    # Bot density and the cost of a fresh account are both higher here than on
    # the rest of social, and author signals below cannot fully correct for it.
    Platform.X: 0.35,
    Platform.UNKNOWN: 0.20,
}
"""Per-platform overrides. Absent platforms take their category's prior.

A table rather than branching logic, and consulted only in this module: nothing
in `retrieval/`, `graph/` or `agents/` may branch on platform
(`models/signal.py`), which is precisely why the branch has to happen once, here,
and be baked into a number those layers can compare.
"""

_VERIFIED_BONUS: Final = 0.10
_FOLLOWER_BONUS: Final = 0.10
_ACCOUNT_AGE_BONUS: Final = 0.08
_FOLLOWER_SATURATION: Final = 100_000
_ACCOUNT_AGE_SATURATION_DAYS: Final = 1_825


def source_credibility_of(signal: Signal) -> float:
    """Per-platform prior modulated by author signals (`docs/signal-model.md` §3.5).

    Bonuses are applied as `prior + bonus * (1 - prior)`, which has two
    properties a plain sum does not: the result can never leave [0, 1], and the
    same author evidence moves a weak platform much further than a strong one.
    A verified account on an anonymous forum is genuinely more informative than
    a verified account on a research index, where verification is the norm.

    A missing author signal contributes **nothing**, neither bonus nor penalty.
    Most connectors cannot see follower counts at all, and penalizing absence
    would score the whole of RSS as untrustworthy for being RSS.
    """
    prior = PLATFORM_CREDIBILITY_PRIOR.get(
        signal.platform, SOURCE_CREDIBILITY_PRIOR.get(signal.source, 0.30)
    )
    author = signal.author
    if author is None:
        return round(prior, 6)

    bonus = 0.0
    if author.verified:
        bonus += _VERIFIED_BONUS
    if author.follower_count:
        bonus += _FOLLOWER_BONUS * _log_saturate(author.follower_count, _FOLLOWER_SATURATION)
    if author.account_age_days:
        bonus += _ACCOUNT_AGE_BONUS * _log_saturate(
            author.account_age_days, _ACCOUNT_AGE_SATURATION_DAYS
        )
    return round(min(1.0, prior + bonus * (1.0 - prior)), 6)


def _log_saturate(value: float, saturation: float) -> float:
    """Map a count onto [0, 1], log-scaled, saturating at `saturation`.

    Log-scaled because the interesting difference is between 100 followers and
    10,000, not between 900,000 and 910,000. Linear scaling would make every
    account below celebrity scale indistinguishable from a brand-new one.
    """
    if value <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(saturation))


# --------------------------------------------------------------------------- #
# content_integrity
# --------------------------------------------------------------------------- #


def content_integrity_of(signal: Signal) -> float:
    """1.0 full body, 0.5 truncated, 0.2 title-only (`docs/signal-model.md` §3.5).

    A fourth case the doc does not enumerate: neither text nor title, i.e. a
    media-only post whose transcript has not been produced. That scores 0.0 and
    is floored to `CONFIDENCE_FLOOR` in the composite. Giving it the title-only
    0.2 would be more generous than the evidence supports -- there is no text at
    all for `services/evidence_service.py` to verify a quote against, so no claim
    should rest on it, which is exactly what a near-floor confidence says.
    """
    has_text = not signal.content.is_empty
    has_title = bool(signal.content.title and signal.content.title.strip())

    if not has_text:
        return 0.2 if has_title else 0.0
    if signal.content.truncated:
        return 0.5
    return 1.0


# --------------------------------------------------------------------------- #
# corroboration
# --------------------------------------------------------------------------- #

MAX_CORROBORATING_SOURCES: Final = 8
"""Count at which corroboration saturates at 1.0. **Unmeasured.**

Independent *platforms*, not cluster members: `docs/signal-model.md` §4.3 dedups
per platform when counting spread, because six copies inside one subreddit are
one community talking, while the same story on the wire, a blog, X and Reddit is
four independent observations. Counting members instead would let a single
platform manufacture corroboration for its own claim.
"""


@dataclass(frozen=True, slots=True)
class ClusterCorroboration:
    """How widely a dedup cluster has spread, at the moment of scoring."""

    members: int
    independent_platforms: int


@runtime_checkable
class CorroborationSource(Protocol):
    """Lookup for a dedup cluster's spread.

    A port because near-duplicate clustering (`connectors/dedup/hashing.py`)
    writes its clusters to PostgreSQL, and stage 6b must remain runnable with no
    database. Returning `None` means "this cluster is unknown here", which is
    different from "this cluster has one member" only in intent -- both score as
    a lone observation, and the difference is not worth a second code path.
    """

    async def lookup(self, cluster_id: str) -> ClusterCorroboration | None:
        """Spread of `cluster_id`, or `None` when it is not known."""
        ...


@dataclass(slots=True)
class InMemoryCorroborationIndex:
    """Process-local `CorroborationSource`, for tests and single-process runs."""

    clusters: dict[str, ClusterCorroboration] = field(default_factory=dict)

    async def lookup(self, cluster_id: str) -> ClusterCorroboration | None:
        return self.clusters.get(cluster_id)


def corroboration_of(spread: ClusterCorroboration | None) -> float:
    """Log-scaled count of independent near-duplicates (`docs/signal-model.md` §3.5).

    A Signal with no known duplicates counts as one independent source rather
    than zero -- it is itself an observation -- so it scores `log(2)/log(9) ≈
    0.32` rather than bottoming out. That matters: at enrichment time almost
    nothing is corroborated yet, because the corroborating copies have not been
    fetched. Corroboration is the one component that rises afterwards
    (§3.5), and whether already-stored Signals are rescored when a cluster grows
    is open question 2 in §9. This function is deliberately cheap and pure so
    that recomputing is an option whenever that is decided.
    """
    independent = 1 if spread is None else max(1, spread.independent_platforms)
    return round(
        min(1.0, math.log1p(independent) / math.log1p(MAX_CORROBORATING_SOURCES)), 6
    )


# --------------------------------------------------------------------------- #
# Engagement normalization (`docs/signal-model.md` §3.4)
# --------------------------------------------------------------------------- #

PLATFORM_AXIS_COUNTERS: Final[dict[Platform, dict[str, str]]] = {
    Platform.REDDIT: {
        "reach": "subreddit_subscribers",
        "endorsement": "score",
        "amplification": "crossposts",
        "discussion": "num_comments",
    },
    Platform.X: {
        "reach": "impressions",
        "endorsement": "likes",
        "amplification": "reposts",
        "discussion": "replies",
    },
    Platform.YOUTUBE: {
        "reach": "views",
        "endorsement": "likes",
        "amplification": "shares",
        "discussion": "comments",
    },
    Platform.APP_STORE: {"endorsement": "helpful_votes", "discussion": "developer_replies"},
    Platform.PLAY_STORE: {"endorsement": "helpful_votes", "discussion": "developer_replies"},
    Platform.RSS: {"amplification": "syndication_count", "discussion": "comments"},
}
"""Which raw counter feeds which axis, per platform (`docs/signal-model.md` §3.4).

The keys are exactly the names connectors write into `engagement.raw` -- see the
`engagement` block of `_POST_FIELDS` in `connectors/social/reddit.py`. An absent
platform contributes no axes and therefore no `engagement.score`, which is
`None` rather than `0.0`: retrieval reads a zero as "nobody engaged", and there
is a real difference between an unpopular post and a platform whose counters we
do not map yet.

Note what is *not* here. A review's star rating is polarity and belongs in
`sentiment`; only `helpful_votes` on that review is endorsement (§3.4).
"""

MIN_COHORT_SAMPLES: Final = 100
"""Observations a cohort needs before its percentiles are trusted. **Unmeasured.**

With ten samples a percentile has a resolution of 10 points and swings with
every new arrival; the axis would encode sampling noise and then be averaged
into `engagement.score` as though it were signal.
"""


class ColdStartPolicy(StrEnum):
    """What to do when a cohort is too small to give a trustworthy percentile.

    `docs/signal-model.md` §9 open question 4 leaves this genuinely undecided:
    "What a new platform's first 1,000 signals score against is undecided; a
    global prior and a flag-as-provisional approach are both plausible." Rather
    than pick silently -- and a silent pick here means every early Signal on a
    new connector carries a fabricated number that nothing downstream can
    distinguish from a measured one -- both are implemented and the choice is
    explicit at construction.

    What is *not* offered is a third option that some pipelines take: defaulting
    a cold axis to 0.5. That is the worst of both, an invented value wearing the
    costume of a measurement, and it biases every cross-platform comparison
    toward whichever platform is newest.
    """

    OMIT = "omit"
    """Leave the axis unset until the cohort matures. The conservative default."""

    PROVISIONAL = "provisional"
    """Use the thin percentile, and mark the baseline label `:provisional`."""


@dataclass(frozen=True, slots=True)
class CohortPercentile:
    """Where one raw value falls inside its cohort."""

    value: float
    sample_size: int
    window: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"percentile must be in [0, 1], got {self.value}")
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be non-negative, got {self.sample_size}")

    @property
    def is_provisional(self) -> bool:
        return self.sample_size < MIN_COHORT_SAMPLES


@runtime_checkable
class CohortBaseline(Protocol):
    """The trailing-window percentile store (`docs/signal-model.md` §3.4).

    Backed by PostgreSQL in production and **not built yet**, which is exactly
    why stage 6b takes it as a constructor argument. The alternative -- reaching
    for a session inside the stage -- would make this stage untestable without a
    database and would put a query on the ingest hot path that nobody could
    substitute during a backfill.
    """

    async def percentile(
        self, *, platform: Platform, content_type: str, axis: str, value: float
    ) -> CohortPercentile | None:
        """Percentile of `value` in the `(platform, content_type)` cohort for `axis`.

        `None` when the cohort holds nothing at all. Implementations must not
        invent a value for an empty cohort; see `ColdStartPolicy`.
        """
        ...


@dataclass(frozen=True, slots=True)
class CohortObservation:
    """One historical raw value, for the in-memory baseline."""

    platform: Platform
    content_type: str
    axis: str
    value: float


@dataclass(slots=True)
class InMemoryCohortBaseline:
    """Process-local `CohortBaseline` over observations held in a list.

    Stands in for the PostgreSQL trailing-window store until it exists, and is
    what the unit suite runs against. The percentile is the plain empirical one
    -- the fraction of the cohort at or below the value -- which is the
    definition §3.4 uses, so a test written against this fake asserts the same
    quantity production will compute.

    Not a production implementation: it holds every observation forever, has no
    window, and is per process. `window` is therefore reported as `all` rather
    than `30d`, so a Signal scored against it is not mistaken for one scored
    against the real trailing baseline.
    """

    window: str = "all"
    observations: list[CohortObservation] = field(default_factory=list)

    def add(self, observation: CohortObservation) -> None:
        self.observations.append(observation)

    def extend(self, platform: Platform, content_type: str, axis: str, values: list[float]) -> None:
        """Bulk-load one cohort. Convenience for fixtures."""
        for value in values:
            self.add(CohortObservation(platform, content_type, axis, value))

    async def percentile(
        self, *, platform: Platform, content_type: str, axis: str, value: float
    ) -> CohortPercentile | None:
        cohort = [
            observation.value
            for observation in self.observations
            if observation.platform is platform
            and observation.content_type == content_type
            and observation.axis == axis
        ]
        if not cohort:
            return None
        at_or_below = sum(1 for other in cohort if other <= value)
        return CohortPercentile(
            value=at_or_below / len(cohort), sample_size=len(cohort), window=self.window
        )


COHORT_CONTENT_TYPE_KEY: Final = "cohort.content_type"
"""Metadata key a connector may set to choose a finer cohort than `content_type`.

`docs/signal-model.md` §3.4 writes the cohort as `reddit:text_post:30d`, but
"text post versus link post" is platform-shaped knowledge, and nothing above
`connectors/` is allowed to branch on it (`models/signal.py`). The connector --
which already reads `data.is_self` -- may therefore publish the cohort
discriminator into `metadata` under this key. Everything else falls back to
`content.content_type`, which is canonical and always present.
"""


# --------------------------------------------------------------------------- #
# The stage
# --------------------------------------------------------------------------- #


class ScoringStage:
    """Stage 6b. Satisfies `services.signal_engine.pipeline.Stage`.

    Runs after stage 6 and before stage 7, which is not arbitrary:
    `extraction_quality` reads `lineage.stages[]`, so every degradable stage must
    already have recorded its outcome, and `confidence` must be final before the
    Signal is written to five stores.
    """

    name: StageName = StageName.SCORING
    version: str = SCORING_STAGE_VERSION

    def __init__(
        self,
        *,
        baseline: CohortBaseline,
        corroboration: CorroborationSource | None = None,
        cold_start: ColdStartPolicy = ColdStartPolicy.OMIT,
        weights: dict[str, float] | None = None,
    ) -> None:
        """Take the cohort store; never open one.

        `corroboration` defaults to an empty in-memory index, which scores every
        Signal as a single independent source. That is the honest reading at
        enrichment time -- no near-duplicates are known -- and it keeps the stage
        constructible before `connectors/dedup/` has a persistent cluster store.
        """
        self._baseline = baseline
        self._corroboration = corroboration or InMemoryCorroborationIndex()
        self._cold_start = cold_start
        self._axis_weights = weights

    @property
    def model_id(self) -> str | None:
        """Always `None`. Stage 6b is deterministic (`docs/signal-model.md` §5.1).

        Given the same counters, the same cohort and the same stage statuses it
        reproduces its output exactly, so there is no model version to record and
        replaying it introduces no drift.
        """
        return None

    async def apply(self, ctx: EnrichmentContext) -> None:
        """Normalize engagement, compose confidence, and record both in lineage."""
        signal = ctx.require_signal()

        engagement = await self._normalize_engagement(signal)
        components = await self._components(signal)

        signal.engagement = engagement
        signal.confidence = compose_confidence(components)
        signal.lineage.confidence_components = components
        signal.lineage.engagement_baseline = engagement.baseline_window

    # ------------------------------------------------------------ internals --

    async def _components(self, signal: Signal) -> ConfidenceComponents:
        spread = (
            await self._corroboration.lookup(signal.lineage.dedup_cluster_id)
            if signal.lineage.dedup_cluster_id
            else None
        )
        return ConfidenceComponents(
            source_credibility=source_credibility_of(signal),
            # Not recomputed here: `Lineage` already owns the stage-weighted
            # arithmetic, and a second copy of it in this module would drift from
            # `STAGE_QUALITY_WEIGHTS` the first time a stage was added.
            extraction_quality=signal.lineage.compute_extraction_quality(),
            content_integrity=content_integrity_of(signal),
            corroboration=corroboration_of(spread),
        )

    async def _normalize_engagement(self, signal: Signal) -> Engagement:
        """Turn raw counters into percentile axes, then into one score.

        Returns a fresh `Engagement` rather than mutating the existing one:
        `raw` is the connector's, and rebuilding around it makes it structurally
        impossible for this stage to overwrite a platform counter with a
        normalized value -- a corruption that would be invisible afterwards,
        because a percentile and an upvote ratio are both plausible floats.
        """
        raw = dict(signal.engagement.raw)
        counters = PLATFORM_AXIS_COUNTERS.get(signal.platform, {})
        content_type = self._cohort_content_type(signal)

        axes: dict[str, float] = {}
        windows: set[str] = set()
        provisional = False
        cold_started = False

        for axis, counter in counters.items():
            value = _as_float(raw.get(counter))
            if value is None:
                continue
            percentile = await self._baseline.percentile(
                platform=signal.platform, content_type=content_type, axis=axis, value=value
            )
            if percentile is None:
                cold_started = True
                continue
            if percentile.is_provisional and self._cold_start is ColdStartPolicy.OMIT:
                cold_started = True
                continue
            provisional = provisional or percentile.is_provisional
            axes[axis] = percentile.value
            windows.add(percentile.window)

        if not axes:
            # Nothing normalized: either the platform has no mapped counters, the
            # record carried none, or the cohort is empty. The axes stay `None`
            # and `score` stays `None`, which reads downstream as "unknown"
            # rather than as "nobody engaged". The label still records *why*,
            # because "no engagement" and "no baseline" demand different fixes.
            return Engagement(
                raw=raw,
                baseline_window=(
                    self._baseline_label(signal, content_type, "none", cold_start=True)
                    if cold_started
                    else None
                ),
            )

        window = sorted(windows)[0] if len(windows) == 1 else "mixed"
        engagement = Engagement(
            raw=raw,
            reach=axes.get("reach"),
            endorsement=axes.get("endorsement"),
            amplification=axes.get("amplification"),
            discussion=axes.get("discussion"),
            normalized_at=utcnow(),
            baseline_window=self._baseline_label(
                signal,
                content_type,
                window,
                provisional=provisional,
                cold_start=cold_started,
            ),
        )
        # `Engagement.compute_score` owns the renormalization over available
        # axes, so an RSS item with no `endorsement` is not penalized for lacking
        # one. Reimplementing it here would fork the weighting the moment
        # `ENGAGEMENT_AXIS_WEIGHTS` changed.
        engagement.score = engagement.compute_score(self._axis_weights)
        return engagement

    def _cohort_content_type(self, signal: Signal) -> str:
        """The cohort's second key. See `COHORT_CONTENT_TYPE_KEY`."""
        declared = signal.metadata.get(COHORT_CONTENT_TYPE_KEY)
        if isinstance(declared, str) and declared.strip():
            return declared.strip()
        return signal.content.content_type

    @staticmethod
    def _baseline_label(
        signal: Signal,
        content_type: str,
        window: str,
        *,
        provisional: bool = False,
        cold_start: bool = False,
    ) -> str:
        """`platform:content_type:window`, plus a suffix when the cohort is thin.

        `:provisional` means at least one axis was normalized against a cohort
        below `MIN_COHORT_SAMPLES`. `:coldstart` means at least one axis was
        *dropped* for want of a cohort, so `engagement.score` is a mean over
        fewer axes than the platform actually supports.

        Stored on both `engagement.baseline_window` and
        `lineage.engagement_baseline` because §3.4 is explicit that "a score
        computed against a cold-start cohort is not comparable to one computed
        against a mature cohort". Without the suffix the two are the same float
        and nothing downstream could tell them apart.
        """
        label = f"{signal.platform.value}:{content_type}:{window}"
        if provisional:
            label += ":provisional"
        if cold_start:
            label += ":coldstart"
        return label


def _as_float(value: object) -> float | None:
    """Coerce a raw counter. `None` for anything that is not a real number.

    `engagement.raw` explicitly permits `None` -- a connector writes it for a
    counter the provider did not return -- and reading that as `0.0` would rank
    an unreported field at the bottom of its cohort, which is a measurement
    invented out of a missing value.

    The bool branch is the second line of defence. `isinstance(True, int)` is
    `True` in Python, so a boolean flag mapped into `engagement.raw` ranks as the
    number 1. Pydantic coerces it to `1.0` at the model boundary before this
    function ever sees it, which is worse rather than better -- see the note in
    the module summary -- but this still guards a `raw` dict assembled without
    validation, such as a `jsonb` column read straight back out of PostgreSQL.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
