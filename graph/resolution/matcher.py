"""Pairwise similarity scoring: does this pair of records name the same thing?

Blocking (`graph/resolution/blocking.py`) proposes; this module disposes. It
takes two `ResolutionRecord`s and returns a `MatchScore` -- never a bare float.

That return type is the central decision of this module. A merge collapses two
nodes into one, rewires every edge, and changes the answer to every query that
touched either of them. Six months later somebody will ask why "Acme Analytics"
and "Acme Analytica" are one node, and "the score was 0.94" is not an answer:
nobody can tell whether that came from a genuine alias hit, from two embeddings
that happen to sit near each other, or from a name feature saturating on a shared
word. **A merge nobody can explain is a merge nobody can safely undo**, and
`docs/knowledge-graph.md` §6 makes un-merging the sanctioned correction path, so
the explanation is not documentation -- it is an operational dependency.
`MatchScore` therefore carries the per-feature breakdown, which features were
unavailable, and which hard rule (if any) overrode the arithmetic.

Four feature families (`docs/knowledge-graph.md` §6), each in `[0, 1]`:

`name`
    A blend of `rapidfuzz` token-set, token-sort and Jaro-Winkler, multiplied by
    a token-coverage penalty. The penalty is not optional -- see
    `_name_similarity`, where the "Apple Inc" / "Apple Bakery" failure lives.
`alias`
    Best similarity across the two records' declared surfaces. The only feature
    that can bridge names with no character overlap ("Big Blue" / "IBM").
`embedding`
    Cosine over context embeddings, when both records carry one.
`context`
    Jaccard over co-mentioned entity ids. Two records mentioned by the same
    signals in the same company are more likely to be one thing.

**Missing features are renormalized away, never scored zero.** Most incoming
mentions have no embedding and no context yet; scoring those as 0.0 would drag
every fresh mention below the merge band and quietly turn resolution off for new
data, while the tests -- written against fully-populated fixtures -- stayed green.
A feature that could not be computed is excluded from both the numerator and the
denominator, so the combined score always means "agreement across what we could
actually measure".

**`alias` and `context` are corroborating features: they raise a score, they
never sink one.** This asymmetry is deliberate and it is not general
squeamishness about negative evidence -- `name` and `embedding` both pull
downward, because a low value there is a real measurement over a dense space.
Alias sets and co-mention sets are *sparse*, and a low value over a sparse set is
dominated by what nobody recorded:

- Two mentions of the same company harvested from different articles share no
  co-mentioned entities at all. Disjoint context is the ordinary case for a true
  match, not evidence against it, so a Jaccard of 0.0 carried at weight 0.20
  would veto merges that the name feature had already settled at 1.0.
- Alias lists are human-curated and mostly absent. Two records that each declare
  one unrelated alias would otherwise score worse than two records that declare
  none, which is backwards: adding a true alias to an entity must never make it
  harder to resolve.

So both features are reported as *unavailable* when they carry no positive
evidence, with the reason recorded on the `FeatureScore` -- not scored zero, and
not silently dropped either.

Hard rules override the arithmetic in both directions (§6). A shared `domain` or
`ticker` forces a match because those are externally assigned and near-unique;
conflicting ones force a non-match; a `Company` and a `Product` never merge no
matter how identical their names are, because "Stripe" the company and "Stripe"
the product are genuinely two nodes.

Pure and synchronous: no I/O, no clock, no configuration lookup. Scoring a pair
must give the same answer in a worker, a test and a backfill script, which rules
out anything that could vary between them.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from graph.resolution.blocking import (
    BlockingKey,
    BlockKind,
    ResolutionRecord,
    cosine_similarity,
    name_tokens,
    normalize_name,
    pair_key,
)
from models.enums import EntityType

__all__ = [
    "AUTO_MERGE_THRESHOLD",
    "EXTRA_TOKEN_PENALTY",
    "MATCH_WEIGHTS",
    "NAME_BLEND_WEIGHTS",
    "NAME_EVIDENCE_FLOOR",
    "REVIEW_THRESHOLD",
    "TOKEN_MATCH_RATIO",
    "FeatureScore",
    "MatchDecision",
    "MatchScore",
    "MatchScorer",
    "score_pair",
]


# --------------------------------------------------------------------------- #
# Weights and thresholds
# --------------------------------------------------------------------------- #


MATCH_WEIGHTS: Final[Mapping[str, float]] = {
    "name": 0.35,
    "alias": 0.20,
    "embedding": 0.25,
    "context": 0.20,
}
"""Contribution of each feature family to the combined score. Sums to 1.0.

Why each weight is where it is -- these are the numbers
`docs/knowledge-graph.md` §6 specifies, and the reasoning behind them:

`name` **0.35, the largest single weight.**
    It is the only feature available for *every* pair. An incoming mention has a
    surface string and often nothing else; embedding and context arrive later,
    if at all. A feature that is always present must carry the most weight or the
    typical pair is decided by whichever optional feature happened to exist.
    It is not larger than 0.35 because names are also the feature most prone to
    confident falsehood: "Apple Inc" and "Apple Bakery" agree on 100% of the
    smaller name's tokens.

`embedding` **0.25, second.**
    The only feature that sees *meaning*. It is what merges "Big Blue" with "IBM"
    when nobody recorded the alias. Ranked below `name` because it is only as
    good as the text the vector was built from -- a one-line description produces
    an embedding that says "this is a software company" and little else, and such
    vectors are near each other by construction.

`alias` **0.20.**
    An exact alias hit is nearly proof, and if that were the whole story this
    would outweigh everything. It is not: alias sets are sparse, human-curated
    and frequently absent, and this feature is unavailable for most pairs. A
    weight is a claim about the *average* pair, and a strong-but-rare signal is
    better expressed by a hard rule (which `domain` and `ticker` get) than by a
    large weight that mostly does not apply.

`context` **0.20.**
    Co-mention overlap is real evidence, but it is also the feature most easily
    faked by topic: two rival databases discussed in the same threads by the same
    people have high context overlap and are emphatically not the same thing.
    Weighted equal to `alias` and never permitted to carry a merge on its own --
    see `NAME_EVIDENCE_FLOOR`.

These are **starting points to be tuned against a labelled pair set**
(`docs/knowledge-graph.md`, open question 3). That set does not exist yet, which
is why they live here as a named, documented constant rather than in
`backend/core/config.py`: they are not an environment-varying knob, they are a
model, and changing one changes which entities exist. A change here needs an
evaluation run (`pytest -m eval`), not a redeploy.
"""

NAME_BLEND_WEIGHTS: Final[Mapping[str, float]] = {
    "token_set": 0.40,
    "token_sort": 0.25,
    "jaro_winkler": 0.15,
    "coverage": 0.20,
}
"""Internal composition of the `name` feature. Sums to 1.0.

Four measures because each is blind to something the others catch:

- `token_set` ignores word order and duplication ("Acme Cloud" / "Cloud Acme").
  It is also the one that saturates -- it returns 100 whenever one token set is a
  subset of the other -- so it cannot be the only measure.
- `token_sort` restores length sensitivity: it drops as the extra tokens pile up.
- `jaro_winkler` is character-level and prefix-weighted, which is what catches
  typos and truncations that destroy tokens outright ("Datadog" / "Datadg").
- `coverage` is the antidote to `token_set` saturation; see `_token_coverage`.
"""

TOKEN_MATCH_RATIO: Final = 85.0
"""`fuzz.ratio` above which two tokens count as the same token for coverage.

At 85 "datadog"/"datadoghq" align (87.5) but "apple"/"bakery" do not (18). Set
lower and coverage stops discriminating; set higher and every typo reads as an
uncovered token, which double-penalizes exactly the pairs the character-level
measures exist to rescue.
"""

EXTRA_TOKEN_PENALTY: Final = 0.5
"""How hard uncovered tokens cut the name score.

The name blend is multiplied by `1 - EXTRA_TOKEN_PENALTY * (1 - coverage)`, so
completely disjoint token sets keep half their blended score and identical ones
keep all of it. Half rather than all: an uncovered token is strong evidence of a
different entity but not proof -- "Acme" and "Acme Payments" may well be the
same company under two writings -- and a full cut would make the character-level
measures unable to rescue any pair whose tokenization differs.
"""

AUTO_MERGE_THRESHOLD: Final = 0.92
"""At or above this combined score, merge without asking (`docs/knowledge-graph.md` §6)."""

REVIEW_THRESHOLD: Final = 0.75
"""Below this, the pair is distinct. Between the two, a human decides.

The band exists because the cost of the two errors is wildly asymmetric. A
missed merge leaves two nodes that a later pass, a new alias or a human can still
join. A wrong merge rewrites history: edges are rewired, reports cite the merged
node, and undoing it needs the un-merge path in
`graph/resolution/entity_resolution.py`. So the automatic action is taken only
where confidence is high, and the whole ambiguous middle is deferred rather than
guessed.
"""

NAME_EVIDENCE_FLOOR: Final = 0.55
"""Minimum `name` feature value for an automatic merge on score alone.

Without this floor a pair with no name similarity at all can still reach 0.92 on
embedding plus context, because renormalization lets two features carry the whole
score. That is how two competing products discussed in the same forum threads,
described in similar words, get merged into one node -- with a high score and no
string evidence whatsoever. Below the floor the pair is capped at `REVIEW`; an
exact alias hit or an identifier rule still merges it, because those are direct
evidence of identity rather than an inference from surroundings.
"""

MAX_ALIAS_COMPARISONS: Final = 64
"""Ceiling on surface-pair comparisons inside the alias feature.

An entity that has absorbed fifty merges carries fifty aliases, and the cross
product against another such entity is 2,500 fuzzy comparisons for one pair --
inside a loop that already runs once per candidate pair. Surfaces are compared in
sorted order so the truncation is deterministic rather than dependent on which
alias happened to be inserted first.
"""


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #


class MatchDecision(enum.StrEnum):
    """What the scorer recommends for a pair.

    `REVIEW` is a first-class outcome, not a failure to decide. The review queue
    is where `docs/knowledge-graph.md` §6 puts the 0.75-0.92 band, and collapsing
    it into either neighbour -- merging on suspicion or discarding real evidence
    -- is worse than carrying the ambiguity forward.
    """

    MERGE = "merge"
    REVIEW = "review"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class FeatureScore:
    """One feature's contribution, including the case where it has none.

    `value is None` means *not measurable* -- no embedding, no aliases on either
    side -- and is deliberately distinct from `value == 0.0`, which means
    measured and disagreeing. Conflating them is the bug that makes every
    embedding-less mention unmergeable.
    """

    name: str
    value: float | None
    weight: float
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.value is not None

    def __str__(self) -> str:
        if self.value is None:
            return f"{self.name}=n/a"
        return f"{self.name}={self.value:.3f}(w{self.weight:.2f})"


@dataclass(frozen=True, slots=True)
class MatchScore:
    """The full, replayable verdict on one pair.

    Persisted onto the `SAME_AS` edge and into the merge audit record so that a
    merge can be explained and reversed months later without re-running the
    matcher against data that has since changed.
    """

    left_id: str
    right_id: str
    combined: float
    decision: MatchDecision
    features: tuple[FeatureScore, ...]
    applied_rule: str | None = None

    @property
    def pair(self) -> tuple[str, str]:
        """The canonically-ordered pair identity."""
        return pair_key(self.left_id, self.right_id)

    def feature(self, name: str) -> FeatureScore | None:
        """One named feature, or `None` if the scorer did not emit it."""
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def value_of(self, name: str) -> float | None:
        """Convenience for `feature(name).value` without the `None` dance."""
        feature = self.feature(name)
        return feature.value if feature is not None else None

    def explain(self) -> str:
        """One line a human can read in a review queue or an audit log."""
        parts = ", ".join(str(feature) for feature in self.features)
        rule = f" rule={self.applied_rule}" if self.applied_rule else ""
        return (
            f"{self.left_id} ~ {self.right_id}: {self.decision.value} "
            f"score={self.combined:.3f}{rule} [{parts}]"
        )

    def as_dict(self) -> dict[str, object]:
        """Serializable form for the `SAME_AS` edge and the merge audit node.

        Feature values are kept as a name -> value map with `None` preserved,
        because "we had no embedding for this pair" is exactly the fact a future
        reader needs in order to judge whether the merge was well-founded.
        """
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "combined": round(self.combined, 6),
            "decision": self.decision.value,
            "applied_rule": self.applied_rule,
            "features": {
                feature.name: (
                    None if feature.value is None else round(feature.value, 6)
                )
                for feature in self.features
            },
        }


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #


def _token_coverage(left_tokens: Sequence[str], right_tokens: Sequence[str]) -> float:
    """Fraction of tokens on both sides that have a partner on the other side.

    This is the fix for the single most dangerous property of `token_set_ratio`:
    it returns 100 whenever one token set is a subset of the other. "Apple Inc"
    normalizes to "apple", whose token set is a subset of "apple bakery"'s, so
    token-set similarity is a perfect 1.0 for a pair that is obviously two
    different businesses. Every ratio-based measure the library offers is
    similarly blind to *what was left over*, so the leftover is measured here
    directly.

    Coverage is symmetric -- `(covered_left + covered_right) / (n_left +
    n_right)` -- rather than one-directional. A one-directional version scores
    "apple" against "apple bakery" as 1.0 again, since every token of the shorter
    name is covered, which is the exact hole being closed.

    Partners are matched fuzzily at `TOKEN_MATCH_RATIO` so that a typo inside a
    token does not read as an entirely uncovered token.
    """
    if not left_tokens or not right_tokens:
        return 0.0

    def covered(source: Sequence[str], target: Sequence[str]) -> int:
        return sum(
            1
            for token in source
            if any(fuzz.ratio(token, other) >= TOKEN_MATCH_RATIO for other in target)
        )

    total = len(left_tokens) + len(right_tokens)
    return (covered(left_tokens, right_tokens) + covered(right_tokens, left_tokens)) / total


def _name_similarity(left: str, right: str) -> float | None:
    """Blended, coverage-penalized string similarity in `[0, 1]`.

    `None` when either side normalizes to nothing -- a name of pure punctuation
    or of characters the normalizer strips. Returning 0.0 there would assert that
    two unnameable records were measured and found different, and the caller
    would then weight that non-observation as evidence.
    """
    left_norm = normalize_name(left)
    right_norm = normalize_name(right)
    if not left_norm or not right_norm:
        return None
    if left_norm == right_norm:
        # Short-circuit for the common case, and a guarantee the blend cannot
        # quite provide: identical normalized names must score exactly 1.0, or
        # "Acme Corp." and "acme corp" would sit fractionally below the merge
        # threshold and never join.
        return 1.0

    coverage = _token_coverage(name_tokens(left), name_tokens(right))
    blend = (
        NAME_BLEND_WEIGHTS["token_set"] * fuzz.token_set_ratio(left_norm, right_norm) / 100.0
        + NAME_BLEND_WEIGHTS["token_sort"] * fuzz.token_sort_ratio(left_norm, right_norm) / 100.0
        + NAME_BLEND_WEIGHTS["jaro_winkler"] * JaroWinkler.similarity(left_norm, right_norm)
        + NAME_BLEND_WEIGHTS["coverage"] * coverage
    )
    # Coverage enters twice, and that is intentional: once as a component of the
    # blend (partial credit for what *did* align) and once as a multiplier (a
    # cut for what did not). The multiplier is what makes the penalty bite on a
    # pair whose token-set ratio has saturated at 1.0.
    penalty = 1.0 - EXTRA_TOKEN_PENALTY * (1.0 - coverage)
    return max(0.0, min(1.0, blend * penalty))


def _alias_feature(left: ResolutionRecord, right: ResolutionRecord) -> FeatureScore:
    """Best similarity between any declared surface of one and of the other.

    Available only when at least one side declares an alias *and* the best
    cross-surface similarity clears `REVIEW_THRESHOLD`. Two gates, two different
    reasons:

    - With no aliases anywhere the only comparable surfaces are the two
      canonical names, and scoring those here would double-count the `name`
      feature under a second weight -- inflating agreement for pairs that
      produced exactly one piece of evidence. The canonical-to-canonical
      comparison is excluded from the cross product for the same reason.
    - Below the threshold there is no alias *evidence*, only two lists that fail
      to overlap. Reporting that as 0.0 would mean an entity becomes harder to
      resolve every time somebody records a true alias for it, which would make
      the alias table actively harmful to the system that reads it.

    Reusing `REVIEW_THRESHOLD` rather than inventing a private constant is
    deliberate: the question "is this pair of surfaces close enough to mean
    anything" is the same question the band answers for a pair of records.
    """
    weight = MATCH_WEIGHTS["alias"]
    if not left.aliases and not right.aliases:
        return FeatureScore("alias", None, weight, "neither side declares an alias")

    left_surfaces = sorted(left.normalized_surfaces())
    right_surfaces = sorted(right.normalized_surfaces())
    if not left_surfaces or not right_surfaces:
        return FeatureScore("alias", None, weight, "no comparable surfaces")

    left_canonical = normalize_name(left.name)
    right_canonical = normalize_name(right.name)

    # A shared surface that is *only* the two canonical names agreeing is not an
    # alias hit -- it is the `name` feature, restated under a second weight. It
    # has to be excluded explicitly, because when the canonical names normalize
    # identically they are members of both surface sets and the intersection is
    # never empty, so every identical-name pair would report a perfect alias hit
    # no matter what its actual alias lists contained.
    shared = set(left_surfaces) & set(right_surfaces)
    if left_canonical == right_canonical:
        shared.discard(left_canonical)
    if shared:
        hit = sorted(shared)[0]
        return FeatureScore("alias", 1.0, weight, f"exact surface hit {hit!r}")

    best = 0.0
    comparisons = 0
    for left_surface in left_surfaces:
        for right_surface in right_surfaces:
            if left_surface == left_canonical and right_surface == right_canonical:
                continue
            if comparisons >= MAX_ALIAS_COMPARISONS:
                break
            comparisons += 1
            similarity = _name_similarity(left_surface, right_surface)
            if similarity is not None and similarity > best:
                best = similarity
        if comparisons >= MAX_ALIAS_COMPARISONS:
            break
    if comparisons == 0:
        return FeatureScore("alias", None, weight, "no surface pair to compare")
    if best < REVIEW_THRESHOLD:
        return FeatureScore(
            "alias", None, weight, f"no alias evidence (best {best:.3f} of {comparisons})"
        )
    return FeatureScore("alias", best, weight, f"best of {comparisons} surface pairs")


def _embedding_feature(left: ResolutionRecord, right: ResolutionRecord) -> FeatureScore:
    """Cosine over the two context embeddings, clamped to `[0, 1]`.

    Negative cosine is clamped to 0 rather than rescaled from `[-1, 1]`. Rescaling
    maps orthogonal vectors -- the textbook "unrelated" case -- to 0.5, which
    would hand every pair half of a 0.25-weight feature for free and lift the
    floor of the combined score by an eighth. Anti-correlation and
    unrelatedness are both "no evidence of sameness" here, and 0.0 says that.
    """
    weight = MATCH_WEIGHTS["embedding"]
    if left.embedding is None or right.embedding is None:
        return FeatureScore("embedding", None, weight, "one or both vectors missing")
    if len(left.embedding) != len(right.embedding):
        # Different embedding models, or a dimension change mid-corpus. Comparing
        # them is meaningless, and silently truncating to the shorter one would
        # produce a number that looks authoritative.
        return FeatureScore(
            "embedding",
            None,
            weight,
            f"dimension mismatch {len(left.embedding)} vs {len(right.embedding)}",
        )
    cosine = cosine_similarity(left.embedding, right.embedding)
    if cosine is None:
        return FeatureScore("embedding", None, weight, "zero-magnitude vector")
    return FeatureScore("embedding", max(0.0, min(1.0, cosine)), weight, "cosine")


def _context_feature(left: ResolutionRecord, right: ResolutionRecord) -> FeatureScore:
    """Jaccard over co-mentioned entity ids and source domains.

    Unavailable in two situations, both of which would otherwise be scored 0.0
    and both of which would be wrong:

    - **Either side has no context**, the normal state of a freshly-extracted
      mention. Scoring an empty set against a full one as 0.0 penalizes newness
      -- the pairs resolution most needs to get right.
    - **The sets are disjoint.** Co-mention sets are sparse: two articles about
      the same company routinely mention no other entity in common. Zero overlap
      is therefore the ordinary case for a true match, and treating it as
      disagreement lets a 0.20-weight feature veto a pair whose names are
      character-for-character identical. A *non-empty* intersection is real
      evidence and is scored as such; its absence is not evidence of anything.
    """
    weight = MATCH_WEIGHTS["context"]
    if not left.context or not right.context:
        return FeatureScore("context", None, weight, "one or both context sets empty")
    intersection = len(left.context & right.context)
    if intersection == 0:
        return FeatureScore("context", None, weight, "no shared context (uninformative)")
    union = len(left.context | right.context)
    return FeatureScore("context", intersection / union, weight, f"{intersection}/{union} shared")


# --------------------------------------------------------------------------- #
# Hard rules
# --------------------------------------------------------------------------- #


def _type_conflict(left: ResolutionRecord, right: ResolutionRecord) -> bool:
    """Whether the two records' labels forbid a merge.

    `EntityType.UNKNOWN` is a wildcard rather than an eighth label. Enrichment
    degrades an unrecognized type to `UNKNOWN` rather than dropping the mention
    (`services/signal_engine/entities.py`), so treating `UNKNOWN` as a distinct
    label would make every degraded mention permanently unresolvable -- it could
    only ever match other degraded mentions. The resolver adopts the surviving
    typed label when a cluster contains one.
    """
    if left.type is right.type:
        return False
    return EntityType.UNKNOWN not in (left.type, right.type)


def _identifier_verdict(
    left: ResolutionRecord, right: ResolutionRecord
) -> tuple[bool, str] | None:
    """`(is_match, rule)` when a strong identifier decides, else `None`.

    Conflict is checked before agreement, and that order matters: two records can
    share a `domain` while carrying different `ticker`s (a subsidiary listed
    separately, or a stale domain on an acquired company), and in that situation
    the disagreement is the more reliable signal. Externally-assigned identifiers
    do not collide by accident, so a disagreement is either two different
    entities or a data error -- and merging on a data error is unrecoverable
    without the un-merge path, while declining to merge costs one duplicate node.

    Platform handles (`handle:reddit`, ...) never force a match on their own:
    the same handle on two platforms is two accounts, and squatting means the
    same handle on one platform over time can be two people. They still
    contribute a blocking key, which is where their value is.
    """
    shared = set(left.identifiers) & set(right.identifiers)
    strong = sorted(key for key in shared if not key.startswith("handle:"))

    for key in strong:
        if left.identifiers[key] != right.identifiers[key]:
            return (False, f"identifier_conflict:{key}")
    for key in strong:
        if left.identifiers[key] == right.identifiers[key]:
            return (True, f"identifier_match:{key}")
    return None


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


class MatchScorer:
    """Scores pairs against one set of weights and thresholds, with memoization.

    A class rather than a bare function for two reasons. First, a resolution pass
    scores the same pair more than once -- once from the blocking candidate set,
    again during cluster refinement -- and the second computation is pure waste.
    Second, thresholds are an experiment parameter: `pytest -m eval` sweeps them
    against a labelled pair set, and threading them through every call site as
    keyword arguments makes that sweep unreadable.

    **The memo assumes records are immutable and ids are stable.** They are:
    `ResolutionRecord` is frozen, and merging produces a new record with a new
    identity rather than editing one in place. A scorer must not be carried
    across a merge boundary -- reusing one would answer with a score computed
    against the pre-merge record while the caller believes it describes the
    merged one. `entity_resolution.resolve()` builds one scorer per pass and
    discards it.
    """

    __slots__ = ("_auto_merge", "_memo", "_review", "_weights")

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        auto_merge_threshold: float = AUTO_MERGE_THRESHOLD,
        review_threshold: float = REVIEW_THRESHOLD,
    ) -> None:
        if review_threshold > auto_merge_threshold:
            raise ValueError(
                "review_threshold must not exceed auto_merge_threshold; "
                f"got review={review_threshold} auto={auto_merge_threshold}"
            )
        self._weights = dict(weights or MATCH_WEIGHTS)
        self._auto_merge = auto_merge_threshold
        self._review = review_threshold
        self._memo: dict[tuple[str, str], MatchScore] = {}

    @property
    def auto_merge_threshold(self) -> float:
        return self._auto_merge

    @property
    def review_threshold(self) -> float:
        return self._review

    @property
    def comparisons(self) -> int:
        """How many distinct pairs this scorer has evaluated. For cost reporting."""
        return len(self._memo)

    def score(
        self,
        left: ResolutionRecord,
        right: ResolutionRecord,
        *,
        shared_keys: Sequence[BlockingKey] = (),
    ) -> MatchScore:
        """Score one pair, reusing a previous result for the same id pair.

        Records are passed in the caller's order but the result is stored under
        the canonical pair key, so `score(a, b)` and `score(b, a)` share a memo
        entry. The returned `MatchScore` keeps whichever orientation was asked
        for first; consumers use `MatchScore.pair` when they need identity.
        """
        key = pair_key(left.id, right.id)
        cached = self._memo.get(key)
        if cached is not None:
            return cached
        score = self._compute(left, right, shared_keys)
        self._memo[key] = score
        return score

    def _compute(
        self,
        left: ResolutionRecord,
        right: ResolutionRecord,
        shared_keys: Sequence[BlockingKey],
    ) -> MatchScore:
        """The scoring pipeline: hard rules, then features, then banding."""
        name_value = _name_similarity(left.name, right.name)
        features = (
            FeatureScore(
                "name",
                name_value,
                self._weights["name"],
                "blended token/character similarity",
            ),
            _alias_feature(left, right),
            _embedding_feature(left, right),
            _context_feature(left, right),
        )

        if _type_conflict(left, right):
            # Checked first and returned immediately: no combination of name,
            # embedding and context should be able to merge a Company into a
            # Product, and computing a high score for such a pair only invites
            # somebody to override the rule later on the strength of the number.
            return MatchScore(
                left.id,
                right.id,
                0.0,
                MatchDecision.DISTINCT,
                features,
                f"type_conflict:{left.type.value}!={right.type.value}",
            )

        identifier = _identifier_verdict(left, right)
        if identifier is not None:
            is_match, rule = identifier
            return MatchScore(
                left.id,
                right.id,
                1.0 if is_match else 0.0,
                MatchDecision.MERGE if is_match else MatchDecision.DISTINCT,
                features,
                rule,
            )

        combined, available = self._combine(features)
        if not available:
            return MatchScore(
                left.id,
                right.id,
                0.0,
                MatchDecision.DISTINCT,
                features,
                "no_comparable_features",
            )

        decision = self._band(combined)
        applied_rule: str | None = None

        alias = next(f for f in features if f.name == "alias")
        exact_alias_hit = alias.value == 1.0 and alias.detail.startswith("exact surface")
        # A shared *handle* is the only identifier that reaches here: agreeing
        # strong identifiers already returned a match above, and conflicting ones
        # already returned a non-match.
        shared_handle = any(
            key.kind is BlockKind.IDENTIFIER and key.value.startswith("handle:")
            for key in shared_keys
        )

        if decision is MatchDecision.MERGE and (
            (name_value is None or name_value < NAME_EVIDENCE_FLOOR)
            and not exact_alias_hit
            and not shared_handle
        ):
            # High score, no string evidence. Almost always two related-but-
            # distinct things sitting near each other in embedding space and
            # sharing a neighbourhood; demote rather than merge.
            decision = MatchDecision.REVIEW
            applied_rule = "name_evidence_floor"
        elif decision is MatchDecision.DISTINCT and exact_alias_hit:
            # One record's canonical name is literally a declared surface of the
            # other -- "Big Blue" against IBM's alias list. The weighted score
            # buries that whenever the two canonical names share no characters,
            # because the name feature (the heaviest, and always available) is
            # near zero for exactly the pairs an alias table exists to rescue.
            #
            # Promoted to review rather than to merge: aliases are absorbed
            # automatically by previous merges, so one bad absorption would
            # otherwise cascade -- entity C merges into B because B wrongly holds
            # C's name as an alias, and every future record named like C follows.
            # A human breaks that chain; an auto-merge extends it.
            decision = MatchDecision.REVIEW
            applied_rule = "exact_alias_hit"

        return MatchScore(left.id, right.id, combined, decision, features, applied_rule)

    def _combine(self, features: Sequence[FeatureScore]) -> tuple[float, bool]:
        """Weighted mean over available features. `(0.0, False)` when none are.

        The denominator is the sum of *available* weights, which is what makes a
        missing feature neutral instead of negative. The alternative -- dividing
        by the full weight total -- caps a name-only pair at 0.35 and makes the
        0.92 threshold unreachable for every record that lacks an embedding.
        """
        numerator = 0.0
        denominator = 0.0
        for feature in features:
            if feature.value is None:
                continue
            weight = self._weights.get(feature.name, feature.weight)
            numerator += weight * feature.value
            denominator += weight
        if denominator <= 0.0:
            return (0.0, False)
        return (numerator / denominator, True)

    def _band(self, combined: float) -> MatchDecision:
        """Map a combined score onto the three-way action (§6)."""
        if combined >= self._auto_merge:
            return MatchDecision.MERGE
        if combined >= self._review:
            return MatchDecision.REVIEW
        return MatchDecision.DISTINCT


def score_pair(
    left: ResolutionRecord,
    right: ResolutionRecord,
    *,
    shared_keys: Sequence[BlockingKey] = (),
) -> MatchScore:
    """Score one pair with the default weights. Convenience for callers and tests.

    Builds a throwaway `MatchScorer`, so it carries no memo. Scoring a batch this
    way recomputes shared work; use `MatchScorer` directly for that.
    """
    return MatchScorer().score(left, right, shared_keys=shared_keys)
