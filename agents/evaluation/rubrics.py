"""Scoring rubrics: what "good" means for each agent, written down.

Without this module, agent quality is assessed by reading output and forming an
impression. That is not a measurement -- it does not survive a prompt change, it
cannot be compared across weeks, and it is dominated by fluency. A more
articulate wrong answer scores higher than a plainer right one, every time,
because that is what impressions respond to.

**Every criterion here is mechanically checkable or explicitly marked as not.**
The split is the point. `citations_resolve` is arithmetic: either the signal ids
exist or they do not. `insight_is_non_obvious` is a judgement, and no amount of
wanting it to be mechanical makes it so. Mixing the two silently -- scoring both
with an LLM and reporting one number -- produces a metric that looks objective
and is not, which is worse than an honest subjective score.

**The weights are stated and they are guesses.** `docs/retrieval.md` §3 says the
same thing about the retrieval defaults, and it is equally true here: nobody has
tuned these against a labelled set, because no labelled set exists yet. They are
written as constants with their reasoning so the first person with real data can
change them deliberately rather than discovering them buried in a scoring loop.

**Grounding is weighted highest everywhere it applies.** Not because it is the
most interesting property but because it is the one whose failure is invisible.
An unhelpful answer is obviously unhelpful; an ungrounded one reads exactly like
a grounded one, and the reader has no way to tell without doing the work the
system was supposed to do for them.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from models.enums import AgentName

__all__ = [
    "PASS_THRESHOLD",
    "AgentRubric",
    "Criterion",
    "CriterionKind",
    "RUBRICS",
    "RubricScore",
    "rubric_for",
]

PASS_THRESHOLD: Final = 0.7
"""Weighted score at or above which output is acceptable.

0.7 rather than 0.8 or 0.5 for a reason that is about the *distribution* rather
than the number: with grounding weighted at 0.35-0.40, an output that fails
grounding outright cannot reach 0.7 however good everything else is. The
threshold is chosen so that the one criterion that matters most is effectively
a gate, without having to special-case it in the arithmetic.
"""


class CriterionKind(enum.StrEnum):
    """How a criterion is scored. The honest distinction.

    Kept explicit so a report can say which half of a score was measured and
    which was judged. A single blended number invites the reader to trust the
    subjective half as much as the objective one.
    """

    MECHANICAL = "mechanical"
    """Computable from the output and the evidence. No model involved."""

    JUDGED = "judged"
    """Requires reading. Scored by an LLM judge or a human, and noisy."""


@dataclass(frozen=True, slots=True)
class Criterion:
    """One thing an agent's output is scored on."""

    key: str
    description: str
    weight: float
    kind: CriterionKind
    guidance: str = ""
    """What a judge should look for. Empty for mechanical criteria.

    Written as instructions to a reader rather than a definition, because a judge
    -- human or model -- scores against what it is told to look for, and a
    one-line definition gets interpreted differently every time.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.weight <= 1.0:
            raise ValueError(f"{self.key}: weight must be in (0, 1], got {self.weight}")
        if self.kind is CriterionKind.JUDGED and not self.guidance:
            raise ValueError(
                f"{self.key} is judged but carries no guidance. A judged criterion "
                "without instructions is scored differently by every judge and by "
                "the same judge twice."
            )


@dataclass(frozen=True, slots=True)
class AgentRubric:
    """The full rubric for one agent."""

    agent: AgentName
    criteria: tuple[Criterion, ...]

    def __post_init__(self) -> None:
        total = sum(criterion.weight for criterion in self.criteria)
        if abs(total - 1.0) > 1e-6:
            # Enforced rather than normalised. Silent normalisation means adding a
            # criterion quietly dilutes every existing one, and the score moves
            # for reasons unrelated to the output being scored -- which destroys
            # comparability across the exact change you added it to measure.
            raise ValueError(
                f"{self.agent.value} rubric weights sum to {total:.4f}, not 1.0. "
                "Adjust them deliberately; normalising here would silently reweight "
                "every existing criterion whenever one is added."
            )
        keys = [criterion.key for criterion in self.criteria]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{self.agent.value} rubric has duplicate criterion keys")

    def criterion(self, key: str) -> Criterion:
        for candidate in self.criteria:
            if candidate.key == key:
                return candidate
        raise KeyError(f"{self.agent.value} rubric has no criterion {key!r}")

    @property
    def mechanical_weight(self) -> float:
        """How much of the score is measured rather than judged.

        Worth reporting alongside the score: a 0.8 that is 70% mechanical means
        something quite different from a 0.8 that is 70% judged.
        """
        return sum(
            criterion.weight
            for criterion in self.criteria
            if criterion.kind is CriterionKind.MECHANICAL
        )

    def score(self, values: Mapping[str, float]) -> RubricScore:
        """Combine per-criterion scores into a weighted total.

        A missing criterion scores 0 rather than being skipped. Skipping it would
        renormalise the remaining weights, so an evaluation that failed to
        measure grounding would report a *higher* score than one that measured it
        and found it lacking -- exactly backwards.
        """
        breakdown: dict[str, float] = {}
        total = 0.0
        for criterion in self.criteria:
            raw = values.get(criterion.key, 0.0)
            clamped = max(0.0, min(1.0, float(raw)))
            breakdown[criterion.key] = clamped
            total += clamped * criterion.weight
        missing = tuple(
            criterion.key for criterion in self.criteria if criterion.key not in values
        )
        return RubricScore(
            agent=self.agent, total=total, breakdown=breakdown, missing=missing
        )


@dataclass(frozen=True, slots=True)
class RubricScore:
    """One scored output."""

    agent: AgentName
    total: float
    breakdown: Mapping[str, float]
    missing: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.total >= PASS_THRESHOLD

    @property
    def is_complete(self) -> bool:
        """Whether every criterion was actually measured.

        A score with missing criteria is a lower bound, not a measurement, and
        reporting it as the latter understates the agent while looking precise.
        """
        return not self.missing

    def weakest(self, rubric: AgentRubric, n: int = 3) -> list[tuple[str, float]]:
        """The criteria dragging the score down, by weighted contribution lost.

        Weighted, not raw. A 0.5 on something worth 0.4 costs more than a 0.2 on
        something worth 0.05, and ranking by raw score would send whoever is
        fixing this at the wrong criterion.
        """
        losses = [
            (key, (1.0 - value) * rubric.criterion(key).weight)
            for key, value in self.breakdown.items()
        ]
        losses.sort(key=lambda item: (-item[1], item[0]))
        return [(key, self.breakdown[key]) for key, _ in losses[:n]]


def _c(
    key: str,
    description: str,
    weight: float,
    kind: CriterionKind = CriterionKind.MECHANICAL,
    guidance: str = "",
) -> Criterion:
    return Criterion(
        key=key, description=description, weight=weight, kind=kind, guidance=guidance
    )


_JUDGED = CriterionKind.JUDGED


RUBRICS: Final[Mapping[AgentName, AgentRubric]] = {
    AgentName.PLANNER: AgentRubric(
        AgentName.PLANNER,
        (
            _c("plan_is_routable", "Every step names an agent the router can dispatch.", 0.20),
            _c("dependencies_are_acyclic", "No cycles and no dangling references.", 0.15),
            _c(
                "decomposition_covers_the_question",
                "The sub-questions together answer what was asked.",
                0.30,
                _JUDGED,
                guidance=(
                    "Read the query, then read only the sub-questions. Could you "
                    "answer the query from complete answers to them? Score down for "
                    "a facet the query clearly implies and no sub-question reaches."
                ),
            ),
            _c(
                "no_redundant_steps",
                "No two steps do substantially the same work.",
                0.15,
                _JUDGED,
                guidance=(
                    "Near-duplicate sub-questions retrieve the same passages at full "
                    "cost. Score down for paraphrases; do not penalise two steps that "
                    "genuinely differ in facet or in time window."
                ),
            ),
            _c(
                "fresh_data_is_justified",
                "requires_fresh_data is set only where the corpus plausibly predates the question.",
                0.20,
            ),
        ),
    ),
    AgentName.RETRIEVER: AgentRubric(
        AgentName.RETRIEVER,
        (
            _c("evidence_returned", "The pass returned usable evidence.", 0.20),
            _c("no_duplicate_references", "No signal/chunk pair appears twice.", 0.15),
            _c("sub_questions_tagged", "Each item names the sub-question it serves.", 0.20),
            _c("degradation_reported", "Unavailable backends are recorded.", 0.15),
            _c(
                "queries_match_document_language",
                "Searches use the words documents use, not the words questions use.",
                0.30,
                _JUDGED,
                guidance=(
                    "People write 'battery drains overnight', not 'battery longevity "
                    "concerns'. Score down for question-shaped queries that would miss "
                    "the way the corpus actually phrases the topic."
                ),
            ),
        ),
    ),
    AgentName.TREND: AgentRubric(
        AgentName.TREND,
        (
            _c("every_number_from_a_series", "No figure appears that a tool did not return.", 0.40),
            _c("direction_matches_the_data", "Claimed direction survives verification.", 0.25),
            _c("observation_counts_present", "Every trend states how many points support it.", 0.15),
            _c(
                "volatility_used_honestly",
                "Noisy series are called volatile rather than given a direction.",
                0.20,
                _JUDGED,
                guidance=(
                    "'Rising' sounds like a finding and 'volatile' does not, so the "
                    "failure is one-directional. Score down for a direction asserted "
                    "over a series that visibly wobbles."
                ),
            ),
        ),
    ),
    AgentName.COMPETITOR: AgentRubric(
        AgentName.COMPETITOR,
        (
            _c("stated_claims_are_cited", "Every 'stated' rivalry carries a signal id.", 0.35),
            _c("basis_is_accurate", "Inferences are not labelled as stated.", 0.25),
            _c("graph_degradation_reported", "An unreadable graph is disclosed.", 0.15),
            _c(
                "overlap_is_specific",
                "Says where the two compete, not merely that they do.",
                0.25,
                _JUDGED,
                guidance=(
                    "Two firms competing in one line and partnering in another is the "
                    "interesting shape. Score down for a bare list of names with no "
                    "statement of the market they contest."
                ),
            ),
        ),
    ),
    AgentName.FORECAST: AgentRubric(
        AgentName.FORECAST,
        (
            _c("no_invented_numbers", "Every point came from fit_forecast.", 0.40),
            _c("intervals_well_formed", "Every band contains its own estimate.", 0.15),
            _c("history_sufficient", "Short series are refused, not fitted.", 0.20),
            _c(
                "caveats_are_specific",
                "Caveats name what would break this projection.",
                0.25,
                _JUDGED,
                guidance=(
                    "'Conditions may change' is true of everything and warns nobody. "
                    "Score down for generic caveats; score up for one naming the "
                    "assumption this particular fit depends on."
                ),
            ),
        ),
    ),
    AgentName.INSIGHT: AgentRubric(
        AgentName.INSIGHT,
        (
            _c("citations_resolve", "Every cited signal is in the run's evidence.", 0.35),
            _c("causal_claims_corroborated", "Causal claims cite two independent signals.", 0.15),
            _c("unanswered_questions_named", "Questions the evidence missed are listed.", 0.10),
            _c(
                "reasoning_is_traceable",
                "The stated derivation actually leads from the evidence to the claim.",
                0.25,
                _JUDGED,
                guidance=(
                    "Read the reasoning field, then the cited passages. Does the "
                    "reasoning use them, or restate the conclusion in other words? "
                    "A derivation that could support any conclusion supports none."
                ),
            ),
            _c(
                "insight_is_non_obvious",
                "Says something the evidence does not say on its face.",
                0.15,
                _JUDGED,
                guidance=(
                    "Restating one passage is not synthesis. Score up for a claim that "
                    "requires two or more pieces of evidence taken together."
                ),
            ),
        ),
    ),
    AgentName.STRATEGY: AgentRubric(
        AgentName.STRATEGY,
        (
            _c("provenance_resolves", "Every recommendation cites real insight ids.", 0.30),
            _c("assumptions_and_risks_present", "Both are stated for every action.", 0.20),
            _c("urgency_matches_confidence", "No immediate action below 0.5 confidence.", 0.15),
            _c(
                "actions_are_actionable",
                "A reader could start on Monday.",
                0.20,
                _JUDGED,
                guidance=(
                    "'Monitor the competitive landscape' is not an action. Score up "
                    "for a specific thing a specific person could do."
                ),
            ),
            _c(
                "withholding_used_when_warranted",
                "Declines rather than padding when evidence does not support action.",
                0.15,
                _JUDGED,
                guidance=(
                    "Producing three recommendations because the section expects three "
                    "is the failure. Score up for an explicit, reasoned refusal."
                ),
            ),
        ),
    ),
    AgentName.CRITIC: AgentRubric(
        AgentName.CRITIC,
        (
            _c("broken_citations_found", "Fabricated citations are caught.", 0.30),
            _c("blocking_severity_respected", "Factual defects are not downgraded.", 0.20),
            _c("no_approval_over_blocking", "Never approves with a blocking finding.", 0.15),
            _c(
                "findings_are_real",
                "Findings describe actual defects rather than style.",
                0.20,
                _JUDGED,
                guidance=(
                    "A Critic that reports everything trains readers to skip it. Score "
                    "down for findings that are preferences, and for severity inflation."
                ),
            ),
            _c(
                "confidence_is_calibrated",
                "The stated confidence matches the evidence quality.",
                0.15,
                _JUDGED,
                guidance=(
                    "Check against prompts/shared/confidence_rubric.md. Single-source "
                    "findings above 0.85 and causal claims above 0.8 are miscalibrated."
                ),
            ),
        ),
    ),
    AgentName.REPORT: AgentRubric(
        AgentName.REPORT,
        (
            _c("every_claim_cited", "No claim reaches the document uncited.", 0.35),
            _c("gaps_rendered", "Recorded gaps appear in the document.", 0.20),
            _c("no_invented_numbers", "No figure that is not in the state.", 0.20),
            _c(
                "hedging_survives",
                "Hedged findings stay hedged in the prose.",
                0.15,
                _JUDGED,
                guidance=(
                    "Compare each claim's `hedged` flag against how the sentence reads. "
                    "Score down where a tentative finding is written as a fact to make "
                    "the document read better."
                ),
            ),
            _c(
                "summary_is_honest_about_limits",
                "The executive summary reflects the gaps, not just the findings.",
                0.10,
                _JUDGED,
                guidance=(
                    "Most readers read only the summary. Score down for one that reads "
                    "confident while the gaps section undercuts it."
                ),
            ),
        ),
    ),
    AgentName.COLLECTOR: AgentRubric(
        AgentName.COLLECTOR,
        (
            _c("slugs_are_real", "Every dispatched slug exists in the registry.", 0.35),
            _c("no_urls_requested", "No fetch target came from anywhere but the registry.", 0.35),
            _c("skips_recorded", "Considered-and-rejected sources are named.", 0.10),
            _c(
                "sources_match_the_question",
                "The chosen sources plausibly carry the answer.",
                0.20,
                _JUDGED,
                guidance=(
                    "A press-release feed does not answer a question about developer "
                    "sentiment. Score down for shotgunning every available connector."
                ),
            ),
        ),
    ),
}


def rubric_for(agent: AgentName) -> AgentRubric:
    """The rubric for an agent, raising on one that has none.

    Raising rather than returning a default. A default rubric would score an
    unmeasured agent as if it had been measured, which is the failure this whole
    module exists to prevent.
    """
    try:
        return RUBRICS[agent]
    except KeyError:
        raise KeyError(
            f"no rubric for {agent.value}. Add one to RUBRICS rather than scoring "
            "against a default -- an unmeasured agent must not look measured."
        ) from None
