"""Unit tests for `services/evidence_service.py`: verification, not trust.

The property under test is the one `docs/security-and-privacy.md` §3 states and
that nothing else in the system enforces: **a citation whose quote is not in the
stored text is rejected**. Every other test here exists to stop that rejection
from being quietly widened.

Four groups, ordered by what a regression would cost:

1. **A quote that is not there does not verify.** A dropped negation, a
   paraphrase, a substituted number -- each one is a citation that *looks* like
   proof while pointing at text that does not say what the claim says. These are
   asserted individually rather than as one "unhappy path" case, because the way
   this module breaks is not "verification stops working"; it is someone adding a
   similarity threshold to make a reflowed quote pass and taking the negation
   cases with it.
2. **A reflowed quote does verify, and its offsets point into the original.** The
   only permitted normalization is whitespace, and offsets are asserted against
   the *stored* string -- an offset computed in the normalized form points at the
   wrong character in the document the UI highlights, and drifts further the more
   whitespace the source contained.
3. **The outcome distinguishes why.** `docs/agent-system.md` §13 routes
   `broken_citation` (re-run retrieval) differently from `misquote` (re-run the
   writer), so collapsing the outcomes to a boolean sends half the re-runs
   somewhere that cannot fix anything.
4. **Aggregation counts only verified evidence, and counts it once.** Four
   citations of one document must not score as four corroborating sources, and an
   unverified citation must not average in as a weak one.

The service is driven through a fake `SignalReader` for the pure cases, and
through the real `SignalService` over the in-memory SQLite from
`tests/conftest.py` for the wiring case -- so "the protocol is satisfied by the
thing production passes" is asserted rather than assumed. No network, no broker,
no container.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from models.enums import Platform, SignalStatus, SourceCategory
from models.lineage import Lineage
from models.orm.mixins import DEFAULT_TENANT
from models.orm.signal import SignalRow
from models.signal import Content, SignalView
from services.evidence_service import (
    COMPONENT_FLOOR,
    DEGRADED_CEILING,
    HIGH_BAND_FLOOR,
    Citation,
    ConfidenceBand,
    EvidenceConfidence,
    EvidenceService,
    QuoteVerification,
    SignalReader,
    VerificationOutcome,
    aggregate_confidence,
    find_quote,
    normalize_whitespace,
)
from services.signal_service import SignalService

pytestmark = pytest.mark.unit

NOW = datetime(2026, 7, 1, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fixtures and helpers
# --------------------------------------------------------------------------- #


def make_signal(
    signal_id: str,
    text: str,
    *,
    platform: Platform = Platform.REDDIT,
    source: SourceCategory = SourceCategory.NEWS,
    confidence: float = 0.8,
    timestamp: datetime | None = None,
) -> SignalView:
    """A `SignalView` carrying one body, with everything else at a plausible default."""
    return SignalView(
        id=signal_id,
        source=source,
        platform=platform,
        url=f"https://example.test/{signal_id}",
        timestamp=timestamp if timestamp is not None else NOW - timedelta(days=1),
        content=Content(text=text),
        confidence=confidence,
        lineage=Lineage(
            pipeline_version="1.0.0",
            connector_slug="rss",
            connector_version="1.0.0",
            sync_run_id="run-1",
            fetched_at=NOW,
            native_id=signal_id,
            status=SignalStatus.ENRICHED,
        ),
    )


class FakeSignalReader:
    """The whole dependency: one batched read, recorded so batching is assertable.

    Ten lines because the protocol is three. A test that had to stand up a
    database to check a string comparison would be testing SQLAlchemy.
    """

    def __init__(self, *signals: SignalView) -> None:
        self._by_id = {signal.id: signal for signal in signals}
        self.calls: list[tuple[list[str], str]] = []

    async def get_signals(
        self, signal_ids: Sequence[str], *, tenant_id: str = DEFAULT_TENANT
    ) -> list[SignalView]:
        self.calls.append((list(signal_ids), tenant_id))
        return [self._by_id[i] for i in signal_ids if i in self._by_id]


ARTICLE = (
    "Acme raised prices in March. We evaluated Competitor X and rejected it "
    "after a three-week trial. Latency did not improve."
)


@pytest.fixture
def service() -> EvidenceService:
    return EvidenceService(FakeSignalReader(make_signal("sig_a", ARTICLE)))


async def verify(
    service: EvidenceService, quote: str, *, signal_id: str = "sig_a", **spans: int | None
) -> QuoteVerification:
    return await service.verify_quote(signal_id, quote, **spans)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 1. A quote that is not in the text is rejected
# --------------------------------------------------------------------------- #


async def test_dropped_negation_is_rejected(service: EvidenceService) -> None:
    """The case a fuzzy matcher gets wrong, and the reason there is no threshold.

    "We evaluated Competitor X" is in the article; "We evaluated Competitor X and
    adopted it" is not, and differs from the stored sentence by one word that
    reverses its meaning. Any similarity score high enough to accept a reflowed
    quote accepts this one too, and the citation then supports the opposite of
    what the source says.
    """
    result = await verify(service, "We evaluated Competitor X and adopted it")

    assert result.outcome is VerificationOutcome.MISQUOTED
    assert result.verified is False
    assert result.char_range is None


async def test_inverted_negation_is_rejected(service: EvidenceService) -> None:
    """"Latency did improve" against a body that says it did not."""
    result = await verify(service, "Latency did improve")

    assert result.outcome is VerificationOutcome.MISQUOTED
    assert result.span is None


async def test_paraphrase_is_rejected(service: EvidenceService) -> None:
    """Same meaning, different words: still not a quote."""
    result = await verify(service, "Acme increased prices in March")

    assert result.outcome is VerificationOutcome.MISQUOTED


async def test_substituted_number_is_rejected(service: EvidenceService) -> None:
    """"three-week" became "six-week" -- the most quotable thing in the sentence."""
    result = await verify(service, "after a six-week trial")

    assert result.outcome is VerificationOutcome.MISQUOTED


async def test_case_difference_is_rejected(service: EvidenceService) -> None:
    """Case is *not* folded, and that is a decision rather than an oversight.

    Only whitespace is provably not a difference in wording. Case folding is one
    more class of difference that "looks harmless", and the module docstring
    declines every one of them so that the set of accepted transformations cannot
    grow by increments.
    """
    result = await verify(service, "acme raised prices in march")

    assert result.outcome is VerificationOutcome.MISQUOTED


async def test_a_rejected_quote_is_not_softened_into_a_span(
    service: EvidenceService,
) -> None:
    """A failed verification yields no offsets at all.

    Returning the claimed offsets on failure would hand the renderer a highlight
    into text that does not contain the quote -- a citation that looks resolved
    and points at the wrong sentence.
    """
    result = await verify(service, "Latency improved dramatically", char_start=0, char_end=27)

    assert result.verified is False
    assert result.span is None
    assert result.char_range is None
    assert result.claimed_start == 0 and result.claimed_end == 27


async def test_quote_from_a_different_document_is_rejected() -> None:
    """Real quote, real Signal, wrong pairing -- the citation still fails."""
    evidence = EvidenceService(
        FakeSignalReader(
            make_signal("sig_a", ARTICLE),
            make_signal("sig_b", "Unrelated body about shipping delays."),
        )
    )

    result = await evidence.verify_quote("sig_b", "Acme raised prices in March")

    assert result.outcome is VerificationOutcome.MISQUOTED


# --------------------------------------------------------------------------- #
# 2. A quote that *is* there verifies, with offsets into the stored text
# --------------------------------------------------------------------------- #


async def test_exact_quote_verifies_with_offsets_into_the_stored_text(
    service: EvidenceService,
) -> None:
    result = await verify(service, "Acme raised prices in March")

    assert result.verified is True
    assert result.char_range == (0, 27)
    assert ARTICLE[0:27] == "Acme raised prices in March"
    assert result.span is not None and result.span.exact is True


async def test_reflowed_quote_verifies_and_offsets_cover_the_original_run() -> None:
    """A model rewrapping a quote does not invalidate it.

    The stored body carries the three shapes a scraped body actually has -- a
    newline, a run of spaces, and a U+00A0 left behind by an `&nbsp;` -- where
    the quote has single spaces. Nothing about the wording differs, so this
    must pass, and the returned offsets must be into the *stored* string,
    whitespace and all, or the highlight lands short of the quote by exactly
    the number of characters the normalization removed.
    """
    stored = "Revenue grew\n \u00a0  forty percent year over year."
    evidence = EvidenceService(FakeSignalReader(make_signal("sig_a", stored)))

    result = await evidence.verify_quote("sig_a", "Revenue grew forty percent year over year.")

    assert result.outcome is VerificationOutcome.RELOCATED
    assert result.verified is True
    start, end = result.char_range or (0, 0)
    assert stored[start:end] == stored
    assert result.span is not None and result.span.exact is False


async def test_offsets_are_not_the_normalized_positions() -> None:
    """The offsets must differ from the ones a naive implementation returns.

    Searching the normalized string and returning *its* indices is the easy bug,
    and it is invisible in any body whose whitespace is already single spaces.
    Here the leading run makes the two answers differ by five characters.
    """
    stored = "Intro.\n\n\n\n\nThe outage lasted six hours."
    normalized = normalize_whitespace(stored)
    evidence = EvidenceService(FakeSignalReader(make_signal("sig_a", stored)))

    result = await evidence.verify_quote("sig_a", "The outage lasted six hours.")

    start, end = result.char_range or (-1, -1)
    assert stored[start:end] == "The outage lasted six hours."
    assert start != normalized.find("The outage lasted six hours.")


async def test_verification_never_normalizes_away_a_word() -> None:
    """Whitespace collapses; words do not.

    `normalize_whitespace` is the only transformation applied, so a quote missing
    a word from the middle of the sentence cannot be reconciled by it.
    """
    assert normalize_whitespace("a \n\t b   c") == "a b c"
    assert find_quote("the quick brown fox", "the quick fox") is None
    assert find_quote("the quick brown fox", "the   quick\nbrown fox") is not None


async def test_zero_width_joiners_are_not_treated_as_whitespace() -> None:
    """A ZWJ is orthography, not spacing.

    Folding U+200D away would equate two different words in Persian and Indic
    scripts and would break emoji sequences, so it is excluded from the
    whitespace class deliberately.
    """
    assert find_quote("mid‍word", "mid word") is None
    assert find_quote("mid‍word", "mid‍word") is not None


async def test_repeated_quote_reports_that_its_position_is_ambiguous() -> None:
    """Two occurrences means the offsets are not determined by the quote alone.

    Not a failure -- the words really are there -- but a reader following the
    highlight to the wrong paragraph loses the context the claim depended on, so
    the count travels with the span.
    """
    stored = "Prices rose. Nothing else happened. Prices rose."
    evidence = EvidenceService(FakeSignalReader(make_signal("sig_a", stored)))

    result = await evidence.verify_quote("sig_a", "Prices rose.")

    assert result.span is not None
    assert result.span.occurrences == 2
    # First occurrence, so two runs over the same evidence agree.
    assert result.char_range == (0, 12)


async def test_the_same_evidence_verifies_identically_twice(
    service: EvidenceService,
) -> None:
    """Determinism, because a report that renders differently each time is unreadable."""
    first = await verify(service, "after a three-week trial")
    second = await verify(service, "after a three-week trial")

    assert first.char_range == second.char_range


# --------------------------------------------------------------------------- #
# 3. Claimed offsets are a hint, and the outcome says which one applied
# --------------------------------------------------------------------------- #


async def test_correct_claimed_span_verifies_within_it(service: EvidenceService) -> None:
    """Retrieval's recorded span is checked first, and confirms the offsets."""
    result = await verify(service, "raised prices", char_start=0, char_end=27)

    assert result.outcome is VerificationOutcome.VERIFIED
    assert result.char_range == (5, 18)
    assert ARTICLE[5:18] == "raised prices"


async def test_stale_offsets_relocate_rather_than_fail() -> None:
    """A re-fetch that shifts the body must not turn every citation into a misquote.

    The words are still in the document, so this is evidence; what changed is
    where. The corrected offsets come back so the caller can persist them
    instead of the wrong ones.
    """
    stored = "Alpha beta. The pricing doubled in March."
    evidence = EvidenceService(FakeSignalReader(make_signal("sig_a", stored)))

    result = await evidence.verify_quote(
        "sig_a", "The pricing doubled", char_start=0, char_end=11
    )

    assert result.outcome is VerificationOutcome.RELOCATED
    assert result.verified is True
    assert result.char_range == (12, 31)
    assert stored[12:31] == "The pricing doubled"
    assert result.claimed_start == 0 and result.claimed_end == 11


async def test_out_of_range_offsets_fall_back_to_search_without_raising(
    service: EvidenceService,
) -> None:
    """Nonsense offsets are a symptom of drift, not a reason to crash the Critic."""
    result = await verify(service, "Acme raised prices", char_start=9_000, char_end=9_100)

    assert result.verified is True
    assert result.char_range == (0, 18)


async def test_inverted_offsets_fall_back_to_search(service: EvidenceService) -> None:
    result = await verify(service, "Acme raised prices", char_start=40, char_end=10)

    assert result.verified is True
    assert result.char_range == (0, 18)


async def test_missing_offsets_are_resolved_by_search_and_said_so(
    service: EvidenceService,
) -> None:
    result = await verify(service, "Acme raised prices")

    assert result.outcome is VerificationOutcome.RELOCATED
    assert "recorded no offsets" in result.detail


# --------------------------------------------------------------------------- #
# 4. Outcomes distinguish why a citation failed
# --------------------------------------------------------------------------- #


async def test_unknown_signal_is_a_broken_citation_not_a_misquote() -> None:
    """`docs/agent-system.md` §13 routes these to different re-runs.

    A hallucinated or erased signal id means retrieval must run again; a quote
    that is not in a real document means the writer must. One outcome for both
    would send half the failures to a node that cannot fix them.
    """
    evidence = EvidenceService(FakeSignalReader(make_signal("sig_a", ARTICLE)))

    result = await evidence.verify_quote("sig_nope", "Acme raised prices in March")

    assert result.outcome is VerificationOutcome.SIGNAL_MISSING
    assert result.outcome.critic_finding == "broken_citation"
    assert result.verified is False


async def test_misquote_maps_to_the_misquote_finding(service: EvidenceService) -> None:
    result = await verify(service, "Acme lowered prices in March")

    assert result.outcome.critic_finding == "misquote"


async def test_empty_quote_is_its_own_outcome(service: EvidenceService) -> None:
    """A citation with nothing quoted is a defect in the citation, not a disagreement."""
    result = await verify(service, "   \n\t  ")

    assert result.outcome is VerificationOutcome.EMPTY_QUOTE
    assert result.outcome.critic_finding == "broken_citation"
    assert result.verified is False


async def test_empty_quote_is_checked_before_the_signal_is_loaded() -> None:
    """Order matters: the fix for an empty quote is not "re-read the Signal"."""
    evidence = EvidenceService(FakeSignalReader())

    result = await evidence.verify_quote("sig_missing", "")

    assert result.outcome is VerificationOutcome.EMPTY_QUOTE


async def test_verified_outcomes_are_exactly_verified_and_relocated() -> None:
    verified = {o for o in VerificationOutcome if o.is_verified}

    assert verified == {VerificationOutcome.VERIFIED, VerificationOutcome.RELOCATED}
    assert all(o.critic_finding is None for o in verified)


# --------------------------------------------------------------------------- #
# 5. Batch resolution
# --------------------------------------------------------------------------- #


async def test_resolution_preserves_input_order() -> None:
    """The caller's order is the order the claims appear in the artifact."""
    evidence = EvidenceService(
        FakeSignalReader(
            make_signal("sig_a", ARTICLE),
            make_signal("sig_b", "Shipping delays continued into April."),
        )
    )
    citations = [
        Citation(signal_id="sig_b", quote="Shipping delays continued"),
        Citation(signal_id="sig_nope", quote="anything"),
        Citation(signal_id="sig_a", quote="Acme raised prices"),
    ]

    resolved = await evidence.resolve_citations(citations)

    assert [item.citation.signal_id for item in resolved] == [
        "sig_b",
        "sig_nope",
        "sig_a",
    ]
    assert [item.verified for item in resolved] == [True, False, True]


async def test_repeated_signal_ids_are_read_once() -> None:
    """A report cites the same handful of Signals dozens of times."""
    reader = FakeSignalReader(make_signal("sig_a", ARTICLE))
    evidence = EvidenceService(reader)

    await evidence.resolve_citations(
        [
            Citation(signal_id="sig_a", quote="Acme raised prices"),
            Citation(signal_id="sig_a", quote="Latency did not improve."),
            Citation(signal_id="sig_a", quote="after a three-week trial"),
        ]
    )

    assert len(reader.calls) == 1
    assert reader.calls[0][0] == ["sig_a"]


async def test_empty_citation_list_reads_nothing() -> None:
    reader = FakeSignalReader()
    evidence = EvidenceService(reader)

    assert await evidence.resolve_citations([]) == []
    assert reader.calls == []


async def test_tenant_is_carried_into_the_read() -> None:
    """Cross-tenant citation resolution would be a disclosure, not a bug report."""
    reader = FakeSignalReader(make_signal("sig_a", ARTICLE))
    evidence = EvidenceService(reader, tenant_id="acme")

    await evidence.verify_quote("sig_a", "Acme raised prices")

    assert reader.calls[0][1] == "acme"


# --------------------------------------------------------------------------- #
# 6. Confidence aggregation
# --------------------------------------------------------------------------- #


async def test_empty_evidence_scores_zero_not_the_component_floor() -> None:
    """The floor stops one weak dimension erasing four strong ones.

    It does not grant a baseline to a claim with no evidence behind it, so an
    empty set is exactly 0.0 rather than `COMPONENT_FLOOR`.
    """
    confidence = aggregate_confidence([])

    assert confidence.score == 0.0
    assert confidence.score < COMPONENT_FLOOR
    assert confidence.band is ConfidenceBand.LOW
    assert confidence.rationale == "no evidence was cited"


async def test_wholly_unverified_evidence_scores_zero() -> None:
    """Four fabricated citations are not a weakly-supported claim."""
    evidence = EvidenceService(FakeSignalReader(make_signal("sig_a", ARTICLE)))

    _, confidence = await evidence.score_citations(
        [
            Citation(signal_id="sig_a", quote="Acme slashed prices"),
            Citation(signal_id="sig_ghost", quote="Acme raised prices"),
        ]
    )

    assert confidence.score == 0.0
    assert confidence.verified_count == 0
    assert confidence.total_count == 2
    assert confidence.rationale == "none of the 2 citations could be verified"


async def test_unverified_citations_drag_the_score_down() -> None:
    """The `verification` component is the ratio, not a filter applied silently."""
    signals = [make_signal(f"sig_{i}", f"Body number {i} says something.") for i in range(4)]
    evidence = EvidenceService(FakeSignalReader(*signals))

    _, clean = await evidence.score_citations(
        [Citation(signal_id=f"sig_{i}", quote=f"Body number {i}") for i in range(4)]
    )
    _, mixed = await evidence.score_citations(
        [Citation(signal_id=f"sig_{i}", quote=f"Body number {i}") for i in range(4)]
        + [Citation(signal_id="sig_0", quote="a sentence nobody wrote")]
    )

    assert clean.components["verification"] == 1.0
    assert mixed.components["verification"] == pytest.approx(4 / 5)
    assert mixed.score < clean.score


async def test_citing_one_document_repeatedly_does_not_corroborate_it() -> None:
    """The cheapest way to inflate a report, priced at zero.

    Four quotes from one article are one source. Four quotes from four articles
    are four. If the arithmetic could not tell them apart, quoting the same
    paragraph again would raise confidence.
    """
    one_doc = make_signal("sig_a", ARTICLE)
    four_docs = [
        make_signal(f"sig_{i}", f"Distinct body {i} with a quotable sentence.")
        for i in range(4)
    ]

    single = EvidenceService(FakeSignalReader(one_doc))
    _, from_one = await single.score_citations(
        [
            Citation(signal_id="sig_a", quote="Acme raised prices in March."),
            Citation(signal_id="sig_a", quote="We evaluated Competitor X"),
            Citation(signal_id="sig_a", quote="after a three-week trial"),
            Citation(signal_id="sig_a", quote="Latency did not improve."),
        ]
    )

    many = EvidenceService(FakeSignalReader(*four_docs))
    _, from_four = await many.score_citations(
        [Citation(signal_id=f"sig_{i}", quote=f"Distinct body {i}") for i in range(4)]
    )

    assert from_one.verified_count == from_four.verified_count == 4
    assert from_one.distinct_signals == 1
    assert from_four.distinct_signals == 4
    assert from_one.score < from_four.score


async def test_one_platform_is_not_independent_corroboration() -> None:
    """Six Reddit threads quoting one another are six Signals and one source."""
    same_platform = [
        make_signal(f"sig_{i}", f"Distinct body {i} with a quotable sentence.",
                    platform=Platform.REDDIT)
        for i in range(3)
    ]
    spread = [
        make_signal(f"sig_{i}", f"Distinct body {i} with a quotable sentence.",
                    platform=platform)
        for i, platform in enumerate((Platform.REDDIT, Platform.GITHUB, Platform.X))
    ]
    citations = [Citation(signal_id=f"sig_{i}", quote=f"Distinct body {i}") for i in range(3)]

    _, narrow = await EvidenceService(FakeSignalReader(*same_platform)).score_citations(
        citations
    )
    _, wide = await EvidenceService(FakeSignalReader(*spread)).score_citations(citations)

    assert narrow.distinct_platforms == 1
    assert wide.distinct_platforms == 3
    assert narrow.components["independence"] < wide.components["independence"]
    assert narrow.score < wide.score


async def test_recency_uses_the_median_so_one_old_anchor_does_not_dominate() -> None:
    """A founding announcement among fresh sources is normal, not a stale set.

    A mean would let a single 2019 document drag an otherwise current set into
    the low band; a median treats it as the outlier it is.
    """
    signals = [
        make_signal("sig_0", "Distinct body 0 here.", timestamp=NOW - timedelta(days=1)),
        make_signal("sig_1", "Distinct body 1 here.", timestamp=NOW - timedelta(days=2)),
        make_signal("sig_2", "Distinct body 2 here.", timestamp=NOW - timedelta(days=2200)),
    ]
    evidence = EvidenceService(FakeSignalReader(*signals))

    _, confidence = await evidence.score_citations(
        [Citation(signal_id=f"sig_{i}", quote=f"Distinct body {i}") for i in range(3)],
        as_of=NOW,
    )

    assert confidence.components["recency"] > 0.95


async def test_old_evidence_scores_lower_than_fresh_evidence() -> None:
    fresh = [
        make_signal(f"sig_{i}", f"Distinct body {i} here.", timestamp=NOW - timedelta(days=2))
        for i in range(3)
    ]
    stale = [
        make_signal(f"sig_{i}", f"Distinct body {i} here.", timestamp=NOW - timedelta(days=400))
        for i in range(3)
    ]
    citations = [Citation(signal_id=f"sig_{i}", quote=f"Distinct body {i}") for i in range(3)]

    _, new = await EvidenceService(FakeSignalReader(*fresh)).score_citations(
        citations, as_of=NOW
    )
    _, old = await EvidenceService(FakeSignalReader(*stale)).score_citations(
        citations, as_of=NOW
    )

    assert old.components["recency"] < new.components["recency"]
    assert old.score < new.score


async def test_a_single_dead_component_lowers_without_collapsing() -> None:
    """`COMPONENT_FLOOR` in action: ancient evidence is weak, not absent."""
    ancient = make_signal("sig_a", ARTICLE, timestamp=NOW - timedelta(days=4000))
    evidence = EvidenceService(FakeSignalReader(ancient))

    _, confidence = await evidence.score_citations(
        [Citation(signal_id="sig_a", quote="Acme raised prices in March.")], as_of=NOW
    )

    assert confidence.components["recency"] < COMPONENT_FLOOR
    assert confidence.score > 0.0
    assert confidence.band is not ConfidenceBand.HIGH


async def test_a_degraded_retrieval_run_cannot_present_as_high_confidence() -> None:
    """`docs/architecture.md` §7.3: a degraded run is visible, not silent.

    A ceiling rather than a multiplier, because the missing recall is
    unmeasurable -- what can be said is that the answer must not look certain.
    """
    signals = [
        make_signal(f"sig_{i}", f"Distinct body {i} here.", confidence=0.95,
                    platform=platform, timestamp=NOW - timedelta(hours=6))
        for i, platform in enumerate(
            (Platform.REDDIT, Platform.GITHUB, Platform.X, Platform.YOUTUBE,
             Platform.LINKEDIN, Platform.TRUSTPILOT)
        )
    ]
    citations = [Citation(signal_id=f"sig_{i}", quote=f"Distinct body {i}") for i in range(6)]
    evidence = EvidenceService(FakeSignalReader(*signals))

    _, healthy = await evidence.score_citations(citations, as_of=NOW)
    _, degraded = await evidence.score_citations(
        citations, as_of=NOW, retrieval_degraded=True
    )

    assert healthy.score >= HIGH_BAND_FLOOR
    assert healthy.band is ConfidenceBand.HIGH
    assert degraded.score <= DEGRADED_CEILING
    assert degraded.band is not ConfidenceBand.HIGH
    assert "backend unavailable" in degraded.rationale


async def test_rationale_names_the_limiting_component() -> None:
    """A bare 0.42 is not actionable; the component holding it down is."""
    signals = [
        make_signal("sig_0", "Distinct body 0 here.", timestamp=NOW - timedelta(days=2000)),
    ]
    evidence = EvidenceService(FakeSignalReader(*signals))

    _, confidence = await evidence.score_citations(
        [Citation(signal_id="sig_0", quote="Distinct body 0")], as_of=NOW
    )

    assert "1 of 1 citations verified" in confidence.rationale
    assert "limited by recency" in confidence.rationale


async def test_bands_follow_the_published_floors() -> None:
    assert EvidenceConfidence(score=0.71).band is ConfidenceBand.MODERATE
    assert EvidenceConfidence(score=HIGH_BAND_FLOOR).band is ConfidenceBand.HIGH
    assert EvidenceConfidence(score=0.0).band is ConfidenceBand.LOW


async def test_score_citations_returns_the_outcomes_alongside_the_number() -> None:
    """The Critic builds findings from the outcomes; a bare score cannot be re-expanded."""
    evidence = EvidenceService(FakeSignalReader(make_signal("sig_a", ARTICLE)))

    resolved, confidence = await evidence.score_citations(
        [
            Citation(signal_id="sig_a", quote="Acme raised prices"),
            Citation(signal_id="sig_a", quote="Acme cut prices"),
        ]
    )

    assert [item.verification.outcome for item in resolved] == [
        VerificationOutcome.RELOCATED,
        VerificationOutcome.MISQUOTED,
    ]
    assert confidence.total_count == 2
    assert confidence.verified_count == 1


# --------------------------------------------------------------------------- #
# 7. Wiring: the real SignalService satisfies the reader protocol
# --------------------------------------------------------------------------- #


@pytest.fixture
async def session_factory(
    orm_engine: AsyncEngine,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    yield async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


async def test_signal_service_satisfies_the_reader_protocol_and_verifies_a_stored_row(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The protocol is not a fiction maintained only by the fake.

    A quote is verified end to end against a row read out of the database, so a
    signature change in `SignalService.get_signals` fails here rather than in
    production the first time the Critic runs.
    """
    body = "Acme shipped the CLI on Tuesday.\nSupport tickets fell by half."
    async with session_factory() as session:
        session.add(
            SignalRow(
                id="sig_db",
                native_id="native-1",
                source=SourceCategory.NEWS,
                platform=Platform.RSS,
                url="https://example.test/1",
                timestamp=NOW - timedelta(days=1),
                fetched_at=NOW,
                content_title="Acme CLI",
                content_text=body,
                content_char_count=len(body),
                content_type="text/plain",
                language_code="en",
                language_confidence=0.99,
                entities=[],
                topics=[],
                keywords=[],
                embeddings=[],
                engagement={},
                confidence=0.7,
                signal_metadata={},
                lineage={"connector_version": "1.2.3", "stages": []},
                status=SignalStatus.ENRICHED,
                schema_version=1,
                pipeline_version="1.0.0",
                connector_slug="rss",
                sync_run_id="run-1",
            )
        )
        await session.commit()

    signals = SignalService(session_factory)
    assert isinstance(signals, SignalReader)

    evidence = EvidenceService(signals)
    good = await evidence.verify_quote(
        "sig_db", "Acme shipped the CLI on Tuesday. Support tickets fell by half."
    )
    bad = await evidence.verify_quote("sig_db", "Support tickets doubled")

    assert good.verified is True
    start, end = good.char_range or (0, 0)
    assert body[start:end] == body
    assert bad.outcome is VerificationOutcome.MISQUOTED
