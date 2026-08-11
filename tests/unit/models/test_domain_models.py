"""Unit tests for the `models/` domain layer.

These types exist to make certain states unrepresentable. Each test below names
the state and the reason it must not be constructible -- if a validator is
removed, the test that fails says what the removal permits.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from models.connector import (
    ConnectorAccount,
    ConnectorHealth,
    ConnectorState,
    Cursor,
    SyncOutcome,
)
from models.enums import InvestigationStatus, Platform, SourceCategory
from models.evidence import Citation, EvidenceReference, VerificationOutcome
from models.investigation import (
    TERMINAL_STATUSES,
    Investigation,
    InvestigationStep,
    TokenUsage,
    can_transition,
)
from models.report import (
    ConfidenceBand,
    Report,
    ReportClaim,
    ReportSection,
    SectionKind,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _citation(outcome: VerificationOutcome = VerificationOutcome.VERIFIED) -> Citation:
    return Citation(
        reference=EvidenceReference(signal_id="sig_1"),
        quote="a quoted span",
        outcome=outcome,
        detail=None if outcome is VerificationOutcome.VERIFIED else "store unreachable",
    )


class TestInvestigationTransitions:
    def test_forward_transitions_are_legal(self) -> None:
        assert can_transition(InvestigationStatus.QUEUED, InvestigationStatus.PLANNING)
        assert can_transition(InvestigationStatus.RUNNING, InvestigationStatus.COMPLETED)

    def test_terminal_states_are_terminal(self) -> None:
        for status in TERMINAL_STATUSES:
            assert not can_transition(status, InvestigationStatus.RUNNING)

    def test_reflecting_returns_to_running(self) -> None:
        """The Critic loop sends the graph round again. Without this edge the
        guard would fire on the system's own designed behaviour."""
        assert can_transition(InvestigationStatus.REFLECTING, InvestigationStatus.RUNNING)

    def test_a_repeated_transition_is_allowed(self) -> None:
        """At-least-once delivery means a worker re-applies the same transition
        after a redelivery. Rejecting it turns ordinary redelivery into an error."""
        assert can_transition(InvestigationStatus.RUNNING, InvestigationStatus.RUNNING)

    def test_a_terminal_run_must_record_when_it_ended(self) -> None:
        """Otherwise 'how long do investigations take' is unanswerable for
        exactly the runs that ended badly."""
        with pytest.raises(Exception, match="completed_at"):
            Investigation(
                id="i", tenant_id="t", query="q", status=InvestigationStatus.FAILED
            )

    def test_completed_with_findings_is_a_success(self) -> None:
        """A `status == COMPLETED` check discards every honestly-degraded run,
        which are the majority of real ones."""
        run = Investigation(
            id="i",
            tenant_id="t",
            query="q",
            status=InvestigationStatus.COMPLETED_WITH_FINDINGS,
            completed_at=NOW,
        )
        assert run.succeeded
        assert run.is_terminal

    def test_duration_needs_both_ends(self) -> None:
        run = Investigation(id="i", tenant_id="t", query="q", started_at=NOW)
        assert run.duration_seconds is None


class TestTokenUsage:
    def test_cache_hit_rate_survives_a_zero_denominator(self) -> None:
        assert TokenUsage().cache_hit_rate == 0.0

    def test_cache_hit_rate_is_over_input_tokens(self) -> None:
        assert TokenUsage(input_tokens=100, cached_tokens=75).cache_hit_rate == 0.75


class TestInvestigationStep:
    def test_completed_at_is_derived_from_duration(self) -> None:
        """The table stores a duration, not an end timestamp."""
        step = InvestigationStep(
            id="s", investigation_id="i", sequence=0,
            agent="planner", started_at=NOW, duration_ms=1500,
        )
        assert step.completed_at == NOW + timedelta(milliseconds=1500)

    def test_completed_at_is_none_while_running(self) -> None:
        step = InvestigationStep(
            id="s", investigation_id="i", sequence=0, agent="planner", started_at=NOW
        )
        assert step.completed_at is None


class TestEvidence:
    def test_an_inverted_span_is_rejected(self) -> None:
        """A UI highlighting [400, 200] renders no highlight, and the citation
        looks unremarkable while pointing nowhere."""
        with pytest.raises(Exception, match="inverted"):
            EvidenceReference(signal_id="s", char_start=400, char_end=200)

    def test_an_empty_span_is_rejected(self) -> None:
        with pytest.raises(Exception):
            EvidenceReference(signal_id="s", char_start=10, char_end=10)

    def test_a_failed_verification_must_explain_itself(self) -> None:
        with pytest.raises(Exception, match="detail"):
            Citation(
                reference=EvidenceReference(signal_id="s"),
                quote="q",
                outcome=VerificationOutcome.NOT_FOUND,
            )

    def test_only_verified_citations_are_printable(self) -> None:
        """'We could not check' is not a reason to print it anyway."""
        assert _citation(VerificationOutcome.VERIFIED).is_printable
        assert not _citation(VerificationOutcome.UNVERIFIABLE).is_printable
        assert not _citation(VerificationOutcome.NOT_FOUND).is_printable

    def test_unverifiable_is_distinct_from_not_found(self) -> None:
        """An unreachable store must not turn into a report full of fabrication
        findings."""
        assert VerificationOutcome.UNVERIFIABLE is not VerificationOutcome.NOT_FOUND


class TestReport:
    def test_a_claim_cannot_exist_without_citations(self) -> None:
        with pytest.raises(Exception):
            ReportClaim(id="c", text="t", citations=[])

    def test_a_claim_with_one_bad_citation_is_not_printable(self) -> None:
        """All, not any. Two citations where one is fabricated reads as
        doubly-sourced and is partly invented."""
        claim = ReportClaim(
            id="c",
            text="t",
            citations=[_citation(), _citation(VerificationOutcome.NOT_FOUND)],
        )
        assert not claim.is_printable

    def test_recorded_gaps_must_be_rendered(self) -> None:
        section = ReportSection(kind=SectionKind.FINDINGS, heading="F")
        with pytest.raises(Exception, match="GAPS"):
            Report(
                id="r", investigation_id="i", tenant_id="t", title="T",
                sections=[section], gaps=["unestablished"],
            )

    def test_a_report_with_a_gaps_section_is_valid(self) -> None:
        report = Report(
            id="r", investigation_id="i", tenant_id="t", title="T",
            sections=[ReportSection(kind=SectionKind.GAPS, heading="Gaps")],
            gaps=["unestablished"],
        )
        assert report.gaps

    def test_unprintable_claims_are_surfaced_not_filtered(self) -> None:
        """Dropping them silently leaves prose implying support it no longer
        has."""
        section = ReportSection(
            kind=SectionKind.FINDINGS,
            heading="F",
            claims=[
                ReportClaim(id="ok", text="t", citations=[_citation()]),
                ReportClaim(
                    id="bad", text="t",
                    citations=[_citation(VerificationOutcome.NOT_FOUND)],
                ),
            ],
        )
        report = Report(
            id="r", investigation_id="i", tenant_id="t", title="T", sections=[section]
        )
        assert [c.id for c in report.unprintable_claims] == ["bad"]
        assert report.citation_count == 2

    def test_confidence_bands_are_coarse(self) -> None:
        assert ConfidenceBand.from_score(0.63) is ConfidenceBand.MODERATE
        assert ConfidenceBand.from_score(0.9) is ConfidenceBand.HIGH
        assert ConfidenceBand.from_score(0.2) is ConfidenceBand.LOW


class TestConnector:
    def test_a_watermark_moves_forward(self) -> None:
        cursor = Cursor(watermark=NOW)
        assert cursor.advanced_to(NOW + timedelta(hours=1)).watermark == NOW + timedelta(hours=1)

    def test_a_watermark_never_moves_backward(self) -> None:
        """The most damaging bug in the ingestion path: it silently re-fetches
        history the connector already emitted."""
        cursor = Cursor(watermark=NOW)
        assert cursor.advanced_to(NOW - timedelta(hours=1)).watermark == NOW

    def test_an_absent_watermark_leaves_the_cursor_alone(self) -> None:
        cursor = Cursor(watermark=NOW)
        assert cursor.advanced_to(None).watermark == NOW

    def test_a_first_watermark_is_accepted(self) -> None:
        assert Cursor().advanced_to(NOW).watermark == NOW

    def test_credentials_cannot_hide_in_params(self) -> None:
        """`params` is stored unencrypted. The honest mistake -- putting a token
        where it seems to belong -- does not only happen over HTTP."""
        for key in ("api_key", "access_token", "client_secret", "github_token"):
            with pytest.raises(Exception, match="credential"):
                ConnectorAccount(
                    id="a", tenant_id="t", connector_slug="rss",
                    platform=Platform.RSS, category=SourceCategory.NEWS,
                    params={key: "value"},
                )

    def test_ordinary_params_are_accepted(self) -> None:
        account = ConnectorAccount(
            id="a", tenant_id="t", connector_slug="rss",
            platform=Platform.RSS, category=SourceCategory.NEWS,
            params={"feed_url": "https://example.com/feed", "subreddits": ["x"]},
        )
        assert account.params["feed_url"]

    def test_needs_reauth_is_distinct_from_error(self) -> None:
        """They look identical in a log and need completely different
        responses."""
        assert ConnectorState.NEEDS_REAUTH is not ConnectorState.ERROR
        assert not ConnectorState.NEEDS_REAUTH.is_runnable

    def test_health_flags_a_persistently_failing_source(self) -> None:
        assert ConnectorHealth(consecutive_failures=12).needs_attention
        assert not ConnectorHealth(consecutive_failures=1).needs_attention

    def test_yield_rate_catches_a_normalisation_regression(self) -> None:
        """500 fetched and 4 emitted is not a quiet source -- it is broken, and
        the emitted count alone looks identical to quiet."""
        assert SyncOutcome(
            connector_slug="x", run_id="r", fetched=500, emitted=4
        ).yield_rate == pytest.approx(0.008)

    def test_yield_rate_survives_a_zero_denominator(self) -> None:
        assert SyncOutcome(connector_slug="x", run_id="r").yield_rate == 0.0
