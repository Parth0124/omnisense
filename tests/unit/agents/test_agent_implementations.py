"""Unit tests for the ten agent implementations.

No model and no network. Each agent's `build_input` and `to_delta` are pure, so
they are tested directly; `execute` is exercised through a fake router that
returns a canned schema instance, which is enough to verify the validation and
merge logic that sits around the model call.

The properties worth testing here are the ones that stop a bad answer reaching a
reader: citations that resolve, numbers that came from tools, hedges that survive
into the output, and gaps that cannot be omitted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agents.collector.agent import CollectorAgent
from agents.collector.schemas import CollectionRequest, CollectorOutput
from agents.competitor.agent import CompetitorAgent
from agents.competitor.schemas import CompetitiveBasis, CompetitorFinding
from agents.critic.agent import SOURCE_CONCENTRATION_THRESHOLD, CriticAgent
from agents.critic.schemas import CriticOutput, Finding, FindingKind, Severity
from agents.forecast.agent import ForecastAgent
from agents.insight.agent import InsightAgent
from agents.insight.schemas import Insight, InsightKind, InsightOutput
from agents.planner.agent import PlannerAgent
from agents.planner.schemas import PlannedStep, PlannerOutput
from agents.report.agent import ReportAgent
from agents.report.schemas import (
    ConfidenceBand,
    ReportClaim,
    ReportOutput,
    ReportSection,
    SectionKind,
)
from agents.retriever.agent import RetrieverAgent
from agents.state import EvidenceRef, PlanStep, SubQuestion, new_state
from agents.strategy.agent import StrategyAgent
from agents.strategy.schemas import Horizon, Recommendation, StrategyOutput, Urgency
from agents.tools.registry import AGENT_TOOL_ALLOWLIST
from agents.trend.agent import TrendAgent
from agents.trend.schemas import DetectedTrend, TrendDirection
from models.enums import AgentName, InvestigationStatus

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

ALL_AGENTS = [
    PlannerAgent,
    CollectorAgent,
    RetrieverAgent,
    TrendAgent,
    CompetitorAgent,
    ForecastAgent,
    InsightAgent,
    StrategyAgent,
    CriticAgent,
    ReportAgent,
]


def _state(**overrides):
    state = new_state(
        investigation_id="inv_1",
        tenant_id="tenant-1",
        query="How is Acme's battery strategy performing?",
        deadline_at=NOW + timedelta(minutes=10),
        trace_id="trace-1",
    )
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _bare(cls):
    """An agent instance without the provider/registry wiring.

    `build_input` and `to_delta` are pure by contract -- that is why they are
    separate methods -- so they can be called on an uninitialised instance. If
    either ever touches `self._provider`, this fails, which is the point.
    """
    return cls.__new__(cls)


class TestContract:
    """Properties every agent must have, checked over all ten."""

    @pytest.mark.parametrize("agent_cls", ALL_AGENTS)
    def test_declares_the_full_contract(self, agent_cls) -> None:
        for attribute in ("name", "tier", "output_model", "tools"):
            assert hasattr(agent_cls, attribute), f"{agent_cls.__name__} lacks {attribute}"

    @pytest.mark.parametrize("agent_cls", ALL_AGENTS)
    def test_class_allowlist_matches_the_registry_grant(self, agent_cls) -> None:
        """Two allowlists written in different files. A divergence means an agent
        either calls a tool it never declared or declares one it was never
        granted -- security-relevant drift that should fail here rather than be
        found by reading two files side by side."""
        assert agent_cls.tools == AGENT_TOOL_ALLOWLIST[agent_cls.name]

    def test_every_agent_name_has_an_implementation(self) -> None:
        implemented = {agent.name for agent in ALL_AGENTS}
        expected = {name for name in AgentName if name is not AgentName.UNKNOWN}
        assert implemented == expected

    def test_only_the_planner_is_blocking(self) -> None:
        """No plan means no branches, no sub-questions and nothing to check
        coverage against. Everything else degrades into a smaller answer."""
        blocking = {agent.name for agent in ALL_AGENTS if agent.blocking}
        assert blocking == {AgentName.PLANNER}


class TestPlanner:
    def test_writes_the_plan_and_opens_the_run(self) -> None:
        output = PlannerOutput(
            objective="Establish whether Acme's battery strategy is working",
            steps=[PlannedStep(id="s1", description="retrieve", agent=AgentName.RETRIEVER)],
        )
        delta = _bare(PlannerAgent).to_delta(output, _state())
        assert delta["status"] is InvestigationStatus.RUNNING
        assert delta["plan"][0].agent is AgentName.RETRIEVER
        assert isinstance(delta["plan"][0], PlanStep)

    def test_model_output_is_converted_not_stored_directly(self) -> None:
        """The model's type and the state's type are separate, so a field added
        to the state for bookkeeping does not become prompt-settable."""
        output = PlannerOutput(
            objective="o", steps=[PlannedStep(id="s1", description="d", agent=AgentName.TREND)]
        )
        delta = _bare(PlannerAgent).to_delta(output, _state())
        assert type(delta["plan"][0]) is PlanStep


class TestCollector:
    def test_only_fresh_data_steps_reach_the_input(self) -> None:
        """A Collector that saw the whole plan would collect for analysis steps
        meant to run off existing evidence."""
        state = _state(
            plan=[
                PlanStep(id="a", description="fresh", agent=AgentName.COLLECTOR,
                         requires_fresh_data=True),
                PlanStep(id="b", description="stale", agent=AgentName.TREND),
            ]
        )
        assert _bare(CollectorAgent).build_input(state).fresh_data_steps == ["fresh"]

    def test_a_url_cannot_be_a_connector(self) -> None:
        """The exfiltration boundary. Scraped content is attacker-influenceable;
        if it could name a fetch target this would be a channel out."""
        for hostile in ("http://evil.host", "reddit/../etc", "//evil", "a b"):
            with pytest.raises(Exception):
                CollectionRequest(connector_slug=hostile, reason="r")

    def test_a_plain_slug_is_accepted(self) -> None:
        assert CollectionRequest(connector_slug="reddit", reason="r").connector_slug == "reddit"

    def test_failures_become_state_not_exceptions(self) -> None:
        """One rate-limited third-party API must not take down every
        investigation that touches it."""
        output = CollectorOutput(dispatched=1, failures=["reddit: 429 rate limited"])
        delta = _bare(CollectorAgent).to_delta(output, _state())
        assert delta["collection_results"][0].connector_slug == "reddit"
        assert delta["collection_results"][0].error


class TestRetriever:
    def test_evidence_is_appended_as_references(self) -> None:
        from agents.retriever.schemas import RetrievedItem, RetrieverOutput

        output = RetrieverOutput(
            items=[RetrievedItem(signal_id="sig_1", relevance=0.9, sub_question_id="q1")]
        )
        state = _state(sub_questions=[SubQuestion(id="q1", question="does it work?")])
        delta = _bare(RetrieverAgent).to_delta(output, state)
        assert isinstance(delta["evidence"][0], EvidenceRef)
        assert delta["evidence"][0].signal_id == "sig_1"

    def test_covered_sub_questions_are_marked_answered(self) -> None:
        from agents.retriever.schemas import RetrievedItem, RetrieverOutput

        output = RetrieverOutput(
            items=[RetrievedItem(signal_id="sig_1", sub_question_id="q1")]
        )
        state = _state(
            sub_questions=[
                SubQuestion(id="q1", question="a"),
                SubQuestion(id="q2", question="b"),
            ]
        )
        delta = _bare(RetrieverAgent).to_delta(output, state)
        answered = {q.id: q.answered for q in delta["sub_questions"]}
        assert answered == {"q1": True, "q2": False}

    def test_degradation_is_visible(self) -> None:
        """A keyword-only result set looks identical to a full one and is
        materially weaker."""
        from agents.retriever.schemas import RetrieverOutput

        assert RetrieverOutput(degraded_backends=["vector"]).is_degraded


class TestTrend:
    def test_two_points_cannot_be_a_direction(self) -> None:
        with pytest.raises(Exception, match="observations"):
            DetectedTrend(
                topic="t", direction=TrendDirection.RISING, observation_count=2, summary="s"
            )

    def test_volatile_is_allowed_on_a_short_series(self) -> None:
        """Volatility is itself a statement that the series is too noisy to
        call, so the count floor does not apply."""
        assert DetectedTrend(
            topic="t", direction=TrendDirection.VOLATILE, observation_count=2, summary="s"
        )

    def test_a_claimed_rise_on_a_flat_series_is_corrected(self) -> None:
        """The check that matters: a model handed a wobble will call it rising,
        because rising sounds like a finding."""
        agent = _bare(TrendAgent)
        series = [{"subject": "battery", "points": [(1, 10.0), (2, 10.4), (3, 9.8), (4, 10.1)]}]
        claimed = DetectedTrend(
            topic="battery", direction=TrendDirection.RISING, observation_count=4,
            summary="up", confidence=0.9,
        )
        fixed = agent._verify_against_series(claimed, series)
        assert fixed.direction is TrendDirection.STABLE
        assert fixed.confidence == pytest.approx(0.45)

    def test_a_real_rise_survives_intact(self) -> None:
        agent = _bare(TrendAgent)
        series = [{"subject": "battery", "points": [(1, 10.0), (2, 14.0), (3, 18.0)]}]
        claimed = DetectedTrend(
            topic="battery", direction=TrendDirection.RISING, observation_count=3,
            summary="up", confidence=0.9,
        )
        kept = agent._verify_against_series(claimed, series)
        assert kept.direction is TrendDirection.RISING
        assert kept.confidence == 0.9
        assert kept.change_pct == pytest.approx(80.0)

    def test_a_trend_about_an_unmeasured_subject_is_dropped(self) -> None:
        """A fabricated subject is not recoverable by softening the direction."""
        agent = _bare(TrendAgent)
        series = [{"subject": "battery", "points": [(1, 1.0), (2, 2.0), (3, 3.0)]}]
        invented = DetectedTrend(
            topic="moon phases", direction=TrendDirection.STABLE, summary="s"
        )
        assert agent._verify_against_series(invented, series) is None

    def test_a_zero_baseline_yields_no_percentage(self) -> None:
        """'Up 400%' from one prior mention is arithmetically true and
        substantively meaningless."""
        agent = _bare(TrendAgent)
        series = [{"subject": "battery", "points": [(1, 0.0), (2, 3.0), (3, 5.0)]}]
        claimed = DetectedTrend(
            topic="battery", direction=TrendDirection.RISING, observation_count=3, summary="s"
        )
        assert agent._verify_against_series(claimed, series).change_pct is None


class TestCompetitor:
    def test_a_stated_rivalry_needs_a_source(self) -> None:
        """`stated` licenses an unhedged sentence; without a citation it is an
        inference wearing the label."""
        with pytest.raises(Exception, match="stated"):
            CompetitorFinding(name="Globex", basis=CompetitiveBasis.STATED)

    def test_an_inference_needs_none(self) -> None:
        assert CompetitorFinding(name="Globex", basis=CompetitiveBasis.INFERRED)

    def test_merge_keeps_the_strongest_basis_and_unions_citations(self) -> None:
        """One row per company. Listing Globex twice because it came from two
        sources is false corroboration arriving through a different door."""
        agent = _bare(CompetitorAgent)
        merged = agent._merge(
            [CompetitorFinding(name="Globex", basis=CompetitiveBasis.GRAPH, strength=0.4)],
            [
                CompetitorFinding(
                    name="globex", basis=CompetitiveBasis.STATED, signal_ids=["sig_1"],
                    strength=0.9,
                )
            ],
        )
        assert len(merged) == 1
        assert merged[0].basis is CompetitiveBasis.STATED
        assert merged[0].strength == 0.9
        assert "sig_1" in merged[0].signal_ids


class TestForecast:
    def test_subjects_come_from_measured_trends(self) -> None:
        """A subject the Trend agent could not measure has no series, so
        forecasting it means fitting to nothing."""
        state = _state(trends=[{"topic": "battery complaints"}, {"topic": "battery complaints"}])
        assert _bare(ForecastAgent).build_input(state).subjects == ["battery complaints"]

    def test_the_narrative_schema_cannot_carry_a_number(self) -> None:
        """A stronger control than a prompt asking the model not to: there is
        nowhere to put one."""
        from agents.forecast.agent import ForecastNarrative

        assert "points" not in ForecastNarrative.model_fields
        assert "value" not in ForecastNarrative.model_fields


class TestInsight:
    def test_a_causal_claim_needs_two_independent_signals(self) -> None:
        """On one source it is that source's framing restated as analysis."""
        with pytest.raises(Exception, match="causal"):
            Insight(
                id="i1", kind=InsightKind.CAUSAL_HYPOTHESIS, statement="x because y",
                reasoning="r", signal_ids=["sig_1"],
            )

    def test_a_causal_claim_cannot_be_near_certain(self) -> None:
        with pytest.raises(Exception, match="confidence"):
            Insight(
                id="i1", kind=InsightKind.CAUSAL_HYPOTHESIS, statement="s", reasoning="r",
                signal_ids=["sig_1", "sig_2"], confidence=0.95,
            )

    def test_an_insight_cannot_exist_without_evidence(self) -> None:
        with pytest.raises(Exception):
            Insight(id="i1", kind=InsightKind.OBSERVATION, statement="s", reasoning="r",
                    signal_ids=[])

    def test_insights_append_as_the_increment(self) -> None:
        output = InsightOutput(
            insights=[
                Insight(id="i1", kind=InsightKind.OBSERVATION, statement="s", reasoning="r",
                        signal_ids=["sig_1"])
            ]
        )
        delta = _bare(InsightAgent).to_delta(output, _state())
        assert len(delta["insights"]) == 1
        assert delta["insights"][0]["id"] == "i1"


class TestStrategy:
    def _rec(self, **overrides) -> Recommendation:
        return Recommendation(
            **{
                "id": "r1",
                "action": "a",
                "rationale": "r",
                "urgency": Urgency.NEAR_TERM,
                "horizon": Horizon.WEEKS,
                "confidence": 0.6,
                "based_on_insight_ids": ["i1"],
                "assumptions": ["market holds"],
                "risks": ["may not"],
                **overrides,
            }
        )

    def test_an_urgent_action_needs_conviction(self) -> None:
        """Asking someone to act now on a coin flip transfers an unpriced risk."""
        with pytest.raises(Exception, match="immediate"):
            self._rec(urgency=Urgency.IMMEDIATE, confidence=0.3)

    def test_a_recommendation_must_descend_from_an_insight(self) -> None:
        with pytest.raises(Exception):
            self._rec(based_on_insight_ids=[])

    def test_assumptions_and_risks_are_mandatory(self) -> None:
        with pytest.raises(Exception):
            self._rec(assumptions=[])
        with pytest.raises(Exception):
            self._rec(risks=[])

    def test_silence_must_be_explained(self) -> None:
        """An unexplained empty list is indistinguishable from a crash."""
        with pytest.raises(Exception, match="no reason given"):
            StrategyOutput(recommendations=[])
        assert StrategyOutput(withheld_reason="evidence supports no action")


class TestCritic:
    def test_a_broken_citation_cannot_be_downgraded(self) -> None:
        """Otherwise a model that wants the run to proceed marks a fabricated
        citation minor and the report ships with it."""
        with pytest.raises(Exception, match="blocking"):
            Finding(
                kind=FindingKind.BROKEN_CITATION, severity=Severity.MINOR,
                target="i1", detail="d",
            )

    def test_approval_over_a_blocking_finding_is_unrepresentable(self) -> None:
        with pytest.raises(Exception, match="approve"):
            CriticOutput(
                findings=[
                    Finding(kind=FindingKind.MISQUOTE, severity=Severity.BLOCKING,
                            target="i1", detail="d")
                ],
                summary="looks fine",
                approved=True,
            )

    async def test_unresolvable_citations_are_found_mechanically(self) -> None:
        from agents.critic.schemas import CriticInput

        agent = _bare(CriticAgent)
        request = CriticInput(
            query="q",
            tenant_id="t",
            insights=[{"id": "i1", "signal_ids": ["sig_ghost"]}],
            evidence_ids=["sig_real"],
        )
        findings = await agent._mechanical_pass(request, ctx=None)  # type: ignore[arg-type]
        kinds = {finding.kind for finding in findings}
        assert FindingKind.BROKEN_CITATION in kinds

    async def test_source_concentration_is_arithmetic_not_judgement(self) -> None:
        """Forty citations tracing to one signal look like abundant evidence in
        every other check."""
        from agents.critic.schemas import CriticInput

        agent = _bare(CriticAgent)
        request = CriticInput(
            query="q",
            tenant_id="t",
            insights=[{"id": f"i{n}", "signal_ids": ["sig_1"]} for n in range(5)],
            evidence_ids=["sig_1"],
        )
        findings = await agent._mechanical_pass(request, ctx=None)  # type: ignore[arg-type]
        assert FindingKind.SOURCE_CONCENTRATION in {f.kind for f in findings}

    async def test_a_balanced_evidence_base_raises_no_concentration_finding(self) -> None:
        from agents.critic.schemas import CriticInput

        agent = _bare(CriticAgent)
        request = CriticInput(
            query="q",
            tenant_id="t",
            insights=[{"id": f"i{n}", "signal_ids": [f"sig_{n}"]} for n in range(5)],
            evidence_ids=[f"sig_{n}" for n in range(5)],
        )
        findings = await agent._mechanical_pass(request, ctx=None)  # type: ignore[arg-type]
        assert FindingKind.SOURCE_CONCENTRATION not in {f.kind for f in findings}

    def test_the_critique_is_kept_and_historied(self) -> None:
        """A single overwritten verdict makes an improving report and a stuck
        one look identical."""
        output = CriticOutput(summary="ok", confidence=0.7, approved=True)
        delta = _bare(CriticAgent).to_delta(output, _state())
        assert delta["critique"]["confidence"] == 0.7
        assert len(delta["critique_history"]) == 1
        assert delta["confidence"] == 0.7

    def test_unverifiable_is_not_the_same_as_false(self) -> None:
        """A payload shape the reader cannot parse must not count as a pass --
        that would turn a parsing bug into silent approval of every citation."""
        from agents.critic.agent import _is_verified

        assert _is_verified({"verified": True})
        assert not _is_verified({"verified": False})
        assert not _is_verified({"unexpected": "shape"})


class TestReport:
    def _section(self, **overrides) -> ReportSection:
        return ReportSection(
            **{
                "kind": SectionKind.FINDINGS,
                "title": "Findings",
                "body": "body",
                "claims": [ReportClaim(id="c1", text="t", citations=["sig_1"])],
                **overrides,
            }
        )

    def test_a_claim_cannot_exist_without_citations(self) -> None:
        with pytest.raises(Exception):
            ReportClaim(id="c1", text="t", citations=[])

    def test_recorded_gaps_must_be_rendered(self) -> None:
        """An unrendered gap is a gap the reader never learns about."""
        with pytest.raises(Exception, match="GAPS"):
            ReportOutput(
                title="t", executive_summary="s", sections=[self._section()],
                gaps=["something unestablished"],
            )

    def test_gaps_are_assembled_from_state_not_written_by_the_model(self) -> None:
        """A model writing its own limitations section omits the most damaging
        item, every time."""
        from agents.report.schemas import ReportInput

        agent = _bare(ReportAgent)
        gaps = agent._assemble_gaps(
            ReportInput(
                query="q",
                tenant_id="t",
                investigation_id="inv_1",
                unanswered_sub_questions=["did pricing change?"],
                degraded_backends=["vector"],
                collection_failures=["reddit: 429"],
                critique={
                    "findings": [
                        {"severity": "major", "kind": "unsupported_claim", "detail": "d"}
                    ]
                },
            )
        )
        joined = " ".join(gaps)
        assert "did pricing change?" in joined
        assert "degraded" in joined
        assert "reddit" in joined
        assert "unsupported_claim" in joined

    def test_gaps_are_deduplicated(self) -> None:
        from agents.report.schemas import ReportInput

        agent = _bare(ReportAgent)
        gaps = agent._assemble_gaps(
            ReportInput(
                query="q", tenant_id="t", investigation_id="i",
                unanswered_sub_questions=["a", "a"],
            )
        )
        assert len(gaps) == 1

    def test_claims_citing_unknown_signals_are_dropped(self) -> None:
        """The last place a fabricated reference can be caught."""
        agent = _bare(ReportAgent)
        section = self._section(
            claims=[
                ReportClaim(id="ok", text="t", citations=["sig_real"]),
                ReportClaim(id="bad", text="t", citations=["sig_ghost"]),
            ]
        )
        cleaned = agent._clean_section(section, {"sig_real"})
        assert [claim.id for claim in cleaned.claims] == ["ok"]

    def test_a_section_whose_every_claim_fails_is_removed(self) -> None:
        """Keeping the prose leaves a section that reads as authoritative with
        nothing behind it."""
        agent = _bare(ReportAgent)
        assert agent._clean_section(self._section(), {"sig_other"}) is None

    def test_gaps_change_the_terminal_status(self) -> None:
        """Lets a caller filter for runs that answered fully without opening
        each one."""
        agent = _bare(ReportAgent)
        with_gaps = ReportOutput(
            title="t",
            executive_summary="s",
            sections=[self._section(kind=SectionKind.GAPS, title="Gaps")],
            gaps=["g"],
        )
        without = ReportOutput(title="t", executive_summary="s", sections=[self._section()])
        assert (
            agent.to_delta(with_gaps, _state())["status"]
            is InvestigationStatus.COMPLETED_WITH_FINDINGS
        )
        assert agent.to_delta(without, _state())["status"] is InvestigationStatus.COMPLETED

    def test_confidence_bands_are_coarse_on_purpose(self) -> None:
        """A reader shown 0.63 treats the second digit as meaningful, and it is
        not."""
        assert ConfidenceBand.from_score(0.9) is ConfidenceBand.HIGH
        assert ConfidenceBand.from_score(0.5) is ConfidenceBand.MODERATE
        assert ConfidenceBand.from_score(0.1) is ConfidenceBand.LOW


class TestPrompts:
    @pytest.mark.parametrize("agent_cls", ALL_AGENTS)
    def test_every_agent_has_a_real_prompt(self, agent_cls) -> None:
        from prompts.loader import load_prompt

        rendered = load_prompt(agent_cls.name)
        assert "TODO" not in rendered.text
        assert len(rendered.text) > 2000

    @pytest.mark.parametrize("agent_cls", ALL_AGENTS)
    def test_every_prompt_carries_the_injection_boundary(self, agent_cls) -> None:
        """The fragment saying retrieved text is data, never instruction. The
        agents most exposed to injected text are the mechanical ones."""
        from prompts.loader import load_prompt

        assert "OMNISENSE_UNTRUSTED_DATA" in load_prompt(agent_cls.name).text

    def test_evidence_bearing_agents_get_the_citation_rules(self) -> None:
        from prompts.loader import fragments_for

        for agent_cls in ALL_AGENTS:
            fragments = fragments_for(agent_cls.name)
            if agent_cls.name in {AgentName.PLANNER, AgentName.COLLECTOR}:
                assert "citation_rules" not in fragments
            else:
                assert "citation_rules" in fragments
                assert "confidence_rubric" in fragments
