"""Citation resolution, quote verification and confidence aggregation.

This module is the mechanism that makes "evidence-backed" true rather than
decorative. Design Doc §2 requires every report claim to carry a citation;
`docs/security-and-privacy.md` §3 states the rule this module enforces --
**citations are verified, not trusted** -- and `docs/agent-system.md` §13 makes
the Critic's `broken_citation` and `misquote` findings the output of the check
implemented here.

Why verification is not optional
--------------------------------
An unverified citation is worse than no citation. A claim with no citation is
visibly unsupported and a reader discounts it. A claim carrying a signal id, a
URL, an author and a quote *looks* like proof, and the only way to discover that
the quoted sentence does not appear in that document is to go and read it --
which is exactly the work the citation existed to save. A language model that
paraphrases while believing it is quoting produces this failure constantly and
reports it never, so the check has to be mechanical.

That is why nothing here degrades a failed match into a pass. There is no
similarity threshold, no token overlap ratio, no "close enough" branch. A fuzzy
matcher tuned to accept a reflowed quote also accepts a quote with a negation
dropped -- "we evaluated Competitor X and rejected it" against a claim built on
"we evaluated Competitor X" -- and the citation then points at a passage that
supports the opposite of the claim. The two failure modes are not symmetric:
rejecting a real quote costs one re-quote by the agent that produced it, while
accepting a false one puts a fabricated fact into a report under the appearance
of provenance.

What *is* normalized, and why exactly that much
-----------------------------------------------
Only whitespace. A model reflows a quote as it renders it -- a newline becomes a
space, a run of spaces collapses, a non-breaking space appears where an HTML
entity was -- and none of those changes a single word. Rejecting on them would
make verification fail on the overwhelmingly common harmless case and push
whoever owns the pipeline to loosen the comparison in some far less safe way.

Nothing else is folded. Not case, not punctuation, not Unicode compatibility
forms. `services/signal_engine/cleaning.py` makes the same call from the other
side: NFKC folding is right for a dedup hash, where the output is never shown to
anyone, and wrong for a body a report quotes verbatim. Each of those foldings
individually looks harmless and each one widens what counts as "the same text";
whitespace is the only class of difference that is *provably* not a difference in
wording.

Offsets are returned against the original stored text, never against the
normalized form. The normalized form is an internal comparison artifact; a
character range computed in it would point at the wrong place in the document the
UI actually highlights, and would drift further the more whitespace the source
contained.

Layer note: `services/` (L2). Reads Signals through a narrow `SignalReader`
protocol rather than importing a concrete store, so the unit suite substitutes a
ten-line fake and nothing here opens a socket.
"""

from __future__ import annotations

import enum
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Protocol, runtime_checkable

from backend.core.logging import get_logger
from models.base import utcnow
from models.orm.mixins import DEFAULT_TENANT
from models.signal import SignalView

__all__ = [
    "Citation",
    "ConfidenceBand",
    "EvidenceConfidence",
    "EvidenceService",
    "QuoteSpan",
    "QuoteVerification",
    "ResolvedCitation",
    "SignalReader",
    "VerificationOutcome",
    "find_quote",
    "normalize_whitespace",
]

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Quote matching -- pure functions, no I/O
# --------------------------------------------------------------------------- #


def _is_space(character: str) -> bool:
    """Whether a character is whitespace for quote-comparison purposes.

    `str.isspace()` plus the Unicode `Zs` category. The two nearly coincide, and
    the gap is the point: `str.isspace()` is False for U+180E MONGOLIAN VOWEL
    SEPARATOR on modern Python but the character still renders as a space, and a
    quote that travelled through a text box may carry one. `Zs` also covers the
    typographic spaces (U+2007 FIGURE SPACE, U+202F NARROW NO-BREAK SPACE) that
    appear in scraped prose around numbers.

    Zero-width characters are deliberately **not** included. U+200B, U+200C and
    U+200D are not whitespace: the joiners are load-bearing in emoji sequences
    and in Persian and Indic orthography (`services/signal_engine/cleaning.py`),
    so folding them away here would silently equate two different words.
    """
    return character.isspace() or unicodedata.category(character) == "Zs"


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace runs to one space and strip the ends.

    Exported because a caller that wants to display "what was actually compared"
    needs the same function, and a second implementation is how two answers to
    one question appear.
    """
    normalized, _, _ = _normalize_with_offsets(text)
    return normalized


def _normalize_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """Normalize, keeping a per-character map back into the original string.

    Returns `(normalized, starts, ends)` where for every index `i` in
    `normalized`, `text[starts[i]:ends[i]]` is the original run that produced it.
    A collapsed whitespace run maps to its *whole* extent, so a span that begins
    or ends on one still covers the original characters.

    The map is why this is not a one-line `" ".join(text.split())`. Verification
    has to answer "where in the stored document is this quote", and the answer
    has to be an offset into the document the UI highlights and the API returns
    as `char_range` (`docs/api-reference.md` §4.4) -- not into a temporary string
    that exists only inside this comparison. Recomputing offsets afterwards by
    searching for the raw quote would fail on exactly the reflowed quotes this
    normalization exists to accept.
    """
    normalized: list[str] = []
    starts: list[int] = []
    ends: list[int] = []

    index = 0
    length = len(text)
    while index < length:
        if _is_space(text[index]):
            run_start = index
            while index < length and _is_space(text[index]):
                index += 1
            if normalized:
                # Leading whitespace produces nothing at all, which is what makes
                # the result stripped without a second pass that would desync the
                # offset map.
                normalized.append(" ")
                starts.append(run_start)
                ends.append(index)
            continue
        normalized.append(text[index])
        starts.append(index)
        ends.append(index + 1)
        index += 1

    # Trailing whitespace collapsed to a space that nothing follows; drop it.
    if normalized and normalized[-1] == " ":
        normalized.pop()
        starts.pop()
        ends.pop()

    return "".join(normalized), starts, ends


@dataclass(frozen=True, slots=True)
class QuoteSpan:
    """Where a quote was found in the stored text.

    `char_start` / `char_end` are half-open offsets into the *original* text, in
    the same coordinate space as `EntityMention` offsets and the chunk spans
    retrieval stores -- `Signal.content.text`, the cleaned body. Resolving them
    against the raw payload would land in a different place entirely.
    """

    char_start: int
    char_end: int
    occurrences: int = 1
    """How many times the quote appears in the document.

    Kept rather than discarded because more than one occurrence means the offsets
    are not uniquely determined by the quote. That is not a verification failure
    -- the text really is there -- but a citation whose span was reconstructed by
    search rather than recorded by retrieval may point at the wrong instance, and
    a reader following the highlight to the wrong paragraph loses the context the
    claim depended on.
    """

    exact: bool = True
    """Whether the stored text matched the quote character for character.

    `False` means whitespace differed and normalization reconciled it -- normal,
    and worth recording anyway: a corpus where quotes routinely need reflowing is
    one where the chunker and the renderer disagree about whitespace, which is a
    fixable upstream problem rather than a fact of life.
    """


def find_quote(content: str, quote: str) -> QuoteSpan | None:
    """Locate `quote` inside `content`, or return `None`.

    Whitespace-insensitive and nothing else. Returns the **first** occurrence, so
    the result is deterministic: a verifier that returned an arbitrary occurrence
    would make two runs over the same evidence disagree about `char_range`, and
    the report would then change between renders for no visible reason.

    `None` is a rejection and callers must treat it as one. There is deliberately
    no second attempt with a looser comparison -- see the module docstring.
    """
    if not quote.strip() or not content:
        return None

    normalized_content, starts, ends = _normalize_with_offsets(content)
    normalized_quote = normalize_whitespace(quote)
    if not normalized_quote:
        return None

    first = normalized_content.find(normalized_quote)
    if first < 0:
        return None

    # Count every occurrence, not just the first. `str.count` would do, but it
    # counts non-overlapping matches only; for a repeated phrase that is the
    # right count anyway, and the loop below keeps the two answers consistent.
    occurrences = 0
    cursor = first
    while cursor >= 0:
        occurrences += 1
        cursor = normalized_content.find(normalized_quote, cursor + 1)

    char_start = starts[first]
    char_end = ends[first + len(normalized_quote) - 1]
    return QuoteSpan(
        char_start=char_start,
        char_end=char_end,
        occurrences=occurrences,
        exact=content[char_start:char_end] == quote,
    )


# --------------------------------------------------------------------------- #
# Citation vocabulary
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Citation:
    """A claim's pointer at a Signal: the quote plus where it was said to be.

    Mirrors `models/orm/report.py::CitationRow` field for field, as a plain
    dataclass rather than the ORM entity. An agent building a citation must not
    have to import a mapped class -- `docs/architecture.md` §6.1 keeps `agents/`
    off the persistence layer -- and the Critic verifies citations that have not
    been written to any table yet.

    Belongs in `models/evidence.py` once that module exists; it is a stub today,
    and defining the shape here rather than nowhere is what lets the verification
    path be real code.

    `char_start` / `char_end` are what retrieval *recorded*. They are treated as
    a hint, not as truth: a re-crawl that changed the body shifts every offset in
    the document, which is precisely the case verification has to notice.
    """

    signal_id: str
    quote: str
    char_start: int | None = None
    char_end: int | None = None
    relevance: float = 0.0
    citation_id: str | None = None


class VerificationOutcome(enum.StrEnum):
    """The result of checking one citation against the stored Signal.

    Five outcomes rather than a boolean, because the Critic routes on *why* a
    citation failed (`docs/agent-system.md` §13): a missing Signal is a
    `broken_citation` and the retrieval step should re-run, while a quote that is
    not in the text is a `misquote` and the writing step should. Collapsing them
    would send every failure to the same node, and half of those re-runs cannot
    fix anything.
    """

    VERIFIED = "verified"
    """The quote is in the stored text at the offsets the citation claimed."""

    RELOCATED = "relocated"
    """The quote is in the stored text, but not where the citation said.

    Still evidence -- the words are there -- so this is not a rejection. It does
    mean the recorded span is stale, which happens when a re-fetch changed the
    body or when a re-chunk moved a boundary, and the corrected offsets are
    returned so the caller can persist them instead of the wrong ones.
    """

    MISQUOTED = "misquoted"
    """The Signal exists and the quote does not appear anywhere in its text."""

    SIGNAL_MISSING = "signal_missing"
    """No such Signal in this tenant. Erased, never ingested, or hallucinated."""

    EMPTY_QUOTE = "empty_quote"
    """The citation carries no quotable text, so there is nothing to check.

    Its own outcome rather than a `MISQUOTED`, because it is a defect in the
    citation rather than a disagreement with the source, and the fix is different:
    the agent must quote something, not re-read the Signal.
    """

    @property
    def is_verified(self) -> bool:
        """Whether a claim may rest on this citation."""
        return self in (VerificationOutcome.VERIFIED, VerificationOutcome.RELOCATED)

    @property
    def critic_finding(self) -> str | None:
        """The `docs/agent-system.md` §13 finding slug, or `None` when verified."""
        if self.is_verified:
            return None
        if self is VerificationOutcome.MISQUOTED:
            return "misquote"
        return "broken_citation"


@dataclass(frozen=True, slots=True)
class QuoteVerification:
    """What verification concluded about one citation."""

    signal_id: str
    quote: str
    outcome: VerificationOutcome
    span: QuoteSpan | None = None
    claimed_start: int | None = None
    claimed_end: int | None = None
    detail: str = ""

    @property
    def verified(self) -> bool:
        return self.outcome.is_verified

    @property
    def char_range(self) -> tuple[int, int] | None:
        """The *corrected* offsets, for `char_range` in the report API.

        `None` when nothing verified. Returning the claimed offsets in that case
        would hand the renderer a highlight into text that does not contain the
        quote -- a citation that looks resolved and points at the wrong sentence,
        which is the exact failure this module exists to prevent.
        """
        if self.span is None:
            return None
        return (self.span.char_start, self.span.char_end)


@dataclass(frozen=True, slots=True)
class ResolvedCitation:
    """A citation, its verification, and the Signal it resolved to."""

    citation: Citation
    verification: QuoteVerification
    signal: SignalView | None = None

    @property
    def verified(self) -> bool:
        return self.verification.verified


# --------------------------------------------------------------------------- #
# Confidence aggregation
# --------------------------------------------------------------------------- #


class ConfidenceBand(enum.StrEnum):
    """How a confidence score is rendered (`docs/glossary.md`).

    A band rather than a bare number because two decimal places imply a precision
    the inputs do not have: source credibility is a per-platform prior and
    corroboration is a count, so 0.71 and 0.68 are the same answer. The
    boundaries are fixed here because nothing else defines them, and they are
    chosen to agree with the one worked example in `docs/api-reference.md` §4.4,
    where 0.71 renders as `moderate`.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


HIGH_BAND_FLOOR: Final = 0.75
MODERATE_BAND_FLOOR: Final = 0.45

CONFIDENCE_WEIGHTS: Final[dict[str, float]] = {
    "verification": 0.30,
    "corroboration": 0.20,
    "independence": 0.20,
    "source_quality": 0.20,
    "recency": 0.10,
}
"""Weights for the evidence-set confidence, summing to 1.0.

Deliberately the same *shape* as the per-Signal weights in
`docs/signal-model.md` §3.5 -- a weighted geometric mean over components in
`[0, 1]` -- so the two numbers are comparable in kind and a reader who
understands one understands the other. The components differ because the
questions differ: §3.5 asks how much to trust one Signal, this asks how much to
trust a *set* of them, where corroboration and source independence are the whole
point and a single document cannot supply either.

`verification` leads because it is the only component that can be zero for a
reason that invalidates everything else: a set whose quotes do not check out is
not a weakly-supported claim, it is an unsupported one.
"""

COMPONENT_FLOOR: Final = 0.05
"""Floor applied to each component before the geometric mean.

Without it a single zero component zeroes the product, which would make "no
non-English coverage" indistinguishable from "no evidence at all". With it, one
dead component drags the score down hard without collapsing it -- the same
trade-off, and the same constant, as `docs/signal-model.md` §3.5.

It is applied to *components*, never to the result. An evidence set with nothing
verified in it scores exactly 0.0, because the floor exists to stop one weak
dimension from erasing four strong ones, not to grant a baseline to a claim with
no evidence behind it.
"""

CORROBORATION_TARGET: Final = 6
"""Independent Signals at which corroboration is considered saturated.

Log-scaled to that target, so the second source is worth far more than the
twelfth -- which matches how corroboration actually works. Six because
`docs/retrieval.md` uses "reported by 6 sources" as the example of a
well-corroborated claim.
"""

INDEPENDENCE_TARGET: Final = 3
"""Distinct platforms at which source independence is considered saturated.

Separate from the count above because they answer different questions. Six
Reddit threads quoting one another are six Signals and one source; the
`corroboration` component cannot tell the difference and this one can. Without
it, a press release syndicated to six outlets scores as strongly as six
independent investigations, which is the single most common way an evidence set
overstates itself.
"""

RECENCY_HALF_LIFE_DAYS: Final = 90.0
"""Age at which evidence contributes half its recency weight.

Ninety days because that is the default retrieval window in
`docs/api-reference.md` §4.1 -- evidence at the edge of what the system looks at
by default should be worth about half of fresh evidence, not almost nothing and
not the same.
"""

DEGRADED_CEILING: Final = 0.6
"""Cap applied when retrieval ran with a backend missing.

`docs/architecture.md` §7.3 requires a degraded run to be *visible in the report
rather than silent*, and §7.3's own examples are a Qdrant outage (keyword-only,
recall drops) and an OpenSearch outage (vector-only, exact-term recall drops). A
cap rather than a multiplier, because the honest statement is not "this answer is
15% worse" -- the missing recall is unmeasurable, since the evidence that was
never retrieved cannot be counted. What can be said is that such a run must not
present as `high` confidence, and a ceiling below `HIGH_BAND_FLOOR` is exactly
that statement expressed as a number.
"""


@dataclass(frozen=True, slots=True)
class EvidenceConfidence:
    """An aggregated confidence, with the breakdown that explains it.

    The components travel with the score for the same reason
    `docs/signal-model.md` §3.5 stores them next to `Signal.confidence`: a bare
    0.42 is not actionable, and "0.42, limited by single-platform evidence" tells
    the Critic which node to re-run and tells a reader what would change it.
    """

    score: float
    components: dict[str, float] = field(default_factory=dict)
    verified_count: int = 0
    total_count: int = 0
    distinct_signals: int = 0
    distinct_platforms: int = 0
    degraded: bool = False

    @property
    def band(self) -> ConfidenceBand:
        if self.score >= HIGH_BAND_FLOOR:
            return ConfidenceBand.HIGH
        if self.score >= MODERATE_BAND_FLOOR:
            return ConfidenceBand.MODERATE
        return ConfidenceBand.LOW

    @property
    def rationale(self) -> str:
        """A deterministic one-line explanation naming the limiting factor.

        Deterministic on purpose. `docs/api-reference.md` §4.4 shows a prose
        `rationale`, and a model writes the good one -- but a model-written
        rationale is not available when the model is what failed, and it is not
        reproducible for the evaluation harness. This is the floor: it always
        exists, it always agrees with the number beside it, and it names the
        component that is actually holding the score down rather than the one
        that reads best.
        """
        if self.total_count == 0:
            return "no evidence was cited"
        if self.verified_count == 0:
            return f"none of the {self.total_count} citations could be verified"
        weakest = min(self.components, key=lambda name: self.components[name])
        parts = [
            f"{self.verified_count} of {self.total_count} citations verified across "
            f"{self.distinct_signals} signals and {self.distinct_platforms} platforms",
            f"limited by {weakest.replace('_', ' ')}",
        ]
        if self.degraded:
            parts.append("capped because retrieval ran with a backend unavailable")
        return "; ".join(parts)


def _log_scaled(count: int, target: int) -> float:
    """`count` mapped into `[0, 1]`, saturating at `target`.

    Log-scaled rather than linear because the marginal value of corroboration
    falls off fast: the difference between one source and two is the difference
    between an assertion and a corroborated fact, while the difference between
    eleven and twelve is nothing.
    """
    if count <= 0:
        return 0.0
    return min(1.0, math.log1p(count) / math.log1p(target))


def aggregate_confidence(
    resolved: Sequence[ResolvedCitation],
    *,
    as_of: datetime | None = None,
    retrieval_degraded: bool = False,
) -> EvidenceConfidence:
    """Score a whole evidence set for the Critic.

    **Only verified citations contribute.** An unverified citation is not weak
    evidence, it is not evidence, so it is excluded from every component except
    `verification` -- where it lowers the score by its absence twice over: once
    directly, and once because the Signal it named no longer counts toward
    corroboration or independence. Averaging a broken citation in as though it
    were merely a poor one is how a set of four citations, three of them
    fabricated, still scores as corroborated.

    Returns `0.0` for an empty or wholly unverified set. See `COMPONENT_FLOOR`
    for why the floor deliberately does not apply there.
    """
    total = len(resolved)
    verified = [item for item in resolved if item.verified]
    signals = {item.citation.signal_id for item in verified}
    platforms = {
        item.signal.platform for item in verified if item.signal is not None
    }

    if total == 0 or not verified:
        return EvidenceConfidence(
            score=0.0,
            components=dict.fromkeys(CONFIDENCE_WEIGHTS, 0.0),
            verified_count=0,
            total_count=total,
            degraded=retrieval_degraded,
        )

    components = {
        "verification": len(verified) / total,
        "corroboration": _log_scaled(len(signals), CORROBORATION_TARGET),
        "independence": _log_scaled(len(platforms), INDEPENDENCE_TARGET),
        "source_quality": _mean_signal_confidence(verified),
        "recency": _recency(verified, as_of=as_of or utcnow()),
    }

    score = math.prod(
        max(components[name], COMPONENT_FLOOR) ** weight
        for name, weight in CONFIDENCE_WEIGHTS.items()
    )
    if retrieval_degraded:
        score = min(score, DEGRADED_CEILING)

    return EvidenceConfidence(
        score=round(min(1.0, max(0.0, score)), 4),
        components={name: round(value, 4) for name, value in components.items()},
        verified_count=len(verified),
        total_count=total,
        distinct_signals=len(signals),
        distinct_platforms=len(platforms),
        degraded=retrieval_degraded,
    )


def _mean_signal_confidence(verified: Sequence[ResolvedCitation]) -> float:
    """Mean `Signal.confidence` over the *distinct* Signals cited.

    Distinct, because citing one strong document four times must not out-score
    citing four moderate ones -- the arithmetic would otherwise reward an agent
    for quoting the same paragraph repeatedly, which is the cheapest way to
    inflate a report.

    A citation whose Signal could not be loaded contributes nothing here, and
    cannot: it was already excluded by `verified`, since a Signal that is not
    there cannot have its text checked.
    """
    by_signal = {
        item.citation.signal_id: item.signal.confidence
        for item in verified
        if item.signal is not None
    }
    if not by_signal:
        return 0.0
    return sum(by_signal.values()) / len(by_signal)


def _recency(verified: Sequence[ResolvedCitation], *, as_of: datetime) -> float:
    """Exponential decay on the *median* age of the evidence.

    Median rather than newest, because one fresh source does not make a set of
    2019 documents current, and a report that claimed otherwise would be
    describing a market that no longer exists. Median rather than mean, because
    a single very old anchor citation -- a founding announcement, a spec -- is
    normal and legitimate, and a mean would let it drag an otherwise current set
    into the low band.

    Evidence with no `published_at` is skipped rather than treated as ancient.
    An unknown date is unknown; scoring it as old would silently penalise sources
    whose connector does not expose a timestamp.
    """
    ages: list[float] = []
    for item in verified:
        if item.signal is None:
            continue
        published = item.signal.timestamp
        age_days = (as_of - published).total_seconds() / 86_400.0
        ages.append(max(0.0, age_days))
    if not ages:
        return 0.0
    ages.sort()
    middle = len(ages) // 2
    median = ages[middle] if len(ages) % 2 else (ages[middle - 1] + ages[middle]) / 2
    return float(0.5 ** (median / RECENCY_HALF_LIFE_DAYS))


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #


@runtime_checkable
class SignalReader(Protocol):
    """The only thing this service needs from the Signal store.

    A protocol rather than a `SignalService` import, for the reason
    `services/signal_engine/store.py` declares `SignalPublisher`: the unit suite
    substitutes a fake with one method and no database, and this module gains no
    dependency on how Signals are persisted. `SignalService.get_signals`
    satisfies it as written.
    """

    async def get_signals(
        self, signal_ids: Sequence[str], *, tenant_id: str = DEFAULT_TENANT
    ) -> list[SignalView]: ...


class EvidenceService:
    """Resolve citations to Signals, verify their quotes, score the set.

    Stateless per call. One instance is shared across concurrent investigations,
    so everything about one evidence set lives on the stack.
    """

    def __init__(
        self, signals: SignalReader, *, tenant_id: str = DEFAULT_TENANT
    ) -> None:
        self._signals = signals
        self._tenant_id = tenant_id

    async def verify_quote(
        self,
        signal_id: str,
        quote: str,
        *,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> QuoteVerification:
        """Confirm `quote` appears in `signal_id`'s stored text, and say where.

        The single-citation form of `resolve_citations`. Verifying one quote is
        what `agents/tools/retrieval_tools.py::fetch_passage` needs during the
        Critic loop, where citations arrive one claim at a time.
        """
        resolved = await self.resolve_citations(
            [Citation(signal_id=signal_id, quote=quote,
                      char_start=char_start, char_end=char_end)]
        )
        return resolved[0].verification

    async def resolve_citations(
        self, citations: Sequence[Citation]
    ) -> list[ResolvedCitation]:
        """Verify a whole citation list against the corpus, in one read.

        Batched deliberately. The Critic verifies every citation in an artifact,
        a report carries dozens, and a per-citation fetch would issue dozens of
        round trips for a set that overwhelmingly cites the same handful of
        Signals -- the reason `retrieval/hybrid.py` resolves passages in one
        batched step rather than per backend.

        Output order matches input order, because the caller's order is the
        order the claims appear in the artifact and the Critic reports findings
        against it.
        """
        if not citations:
            return []

        wanted = list(dict.fromkeys(c.signal_id for c in citations))
        signals = await self._signals.get_signals(wanted, tenant_id=self._tenant_id)
        by_id = {signal.id: signal for signal in signals}

        missing = [i for i in wanted if i not in by_id]
        if missing:
            # Worth a log line rather than only a finding: a citation naming a
            # Signal that is not in the corpus is either a hallucinated id or a
            # Signal erased since the report was written, and the two need
            # different responses from whoever is on call.
            logger.warning(
                "evidence.citation.signal_missing",
                signal_ids=missing,
                tenant_id=self._tenant_id,
            )

        return [
            ResolvedCitation(
                citation=citation,
                verification=_verify(citation, by_id.get(citation.signal_id)),
                signal=by_id.get(citation.signal_id),
            )
            for citation in citations
        ]

    async def score_citations(
        self,
        citations: Sequence[Citation],
        *,
        as_of: datetime | None = None,
        retrieval_degraded: bool = False,
    ) -> tuple[list[ResolvedCitation], EvidenceConfidence]:
        """Resolve, verify and score in one call -- what the Critic actually wants.

        Returns both halves rather than the score alone: the findings are built
        from the per-citation outcomes, and recomputing them from the score is
        impossible. A method that returned only the number would force every
        caller to run the resolution twice.
        """
        resolved = await self.resolve_citations(citations)
        confidence = aggregate_confidence(
            resolved, as_of=as_of, retrieval_degraded=retrieval_degraded
        )
        return resolved, confidence


def _verify(citation: Citation, signal: SignalView | None) -> QuoteVerification:
    """Check one citation against one Signal. The heart of the module.

    Order of checks is the order of decreasing trust in what the citation says:
    first the claimed span, then the whole document, then rejection. Searching
    the whole document *first* would be simpler and would lose the distinction
    between a citation whose offsets are right and one whose offsets have drifted
    -- and drift is a real, recurring event here, because a re-fetch that changes
    one character of the body shifts every offset after it.
    """
    base = {
        "signal_id": citation.signal_id,
        "quote": citation.quote,
        "claimed_start": citation.char_start,
        "claimed_end": citation.char_end,
    }

    if not citation.quote.strip():
        return QuoteVerification(
            outcome=VerificationOutcome.EMPTY_QUOTE,
            detail="the citation carries no quotable text",
            **base,
        )
    if signal is None:
        return QuoteVerification(
            outcome=VerificationOutcome.SIGNAL_MISSING,
            detail="no Signal with this id exists in this tenant",
            **base,
        )

    content = signal.content.text
    span = _within_claimed_span(content, citation)
    if span is not None:
        return QuoteVerification(outcome=VerificationOutcome.VERIFIED, span=span, **base)

    found = find_quote(content, citation.quote)
    if found is None:
        return QuoteVerification(
            outcome=VerificationOutcome.MISQUOTED,
            detail=(
                "the quoted text does not occur in the stored content, allowing for "
                "whitespace differences only"
            ),
            **base,
        )

    detail = "the quote is present but not at the offsets the citation recorded"
    if citation.char_start is None or citation.char_end is None:
        detail = "the citation recorded no offsets; they were resolved by search"
    return QuoteVerification(
        outcome=VerificationOutcome.RELOCATED, span=found, detail=detail, **base
    )


def _within_claimed_span(content: str, citation: Citation) -> QuoteSpan | None:
    """Try to verify inside `[char_start, char_end)`, as `docs/retrieval.md` §8 does.

    The spec's wording is that verification "re-reads `[char_start, char_end)`
    from the stored signal and confirms the quoted text is a substring" -- a
    substring, not an equality, because a claim frequently quotes one sentence
    out of a 512-token chunk.

    Out-of-range or inverted offsets return `None` rather than raising. They are
    a symptom of the exact drift this function exists to detect, and the caller's
    next step -- searching the whole document -- is the right response to them.
    """
    start, end = citation.char_start, citation.char_end
    if start is None or end is None:
        return None
    if start < 0 or end > len(content) or end <= start:
        return None

    window = content[start:end]
    inner = find_quote(window, citation.quote)
    if inner is None:
        return None
    return QuoteSpan(
        char_start=start + inner.char_start,
        char_end=start + inner.char_end,
        # Occurrences are counted within the claimed window, which is the span the
        # citation actually points at. Counting across the whole document here
        # would report an ambiguity the citation had already resolved.
        occurrences=inner.occurrences,
        exact=inner.exact,
    )
