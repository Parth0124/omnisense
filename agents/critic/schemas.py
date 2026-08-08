"""Critic input and output schemas.

`docs/agent-system.md` §13 fixes the finding vocabulary, and it is a closed enum
here rather than free text for one reason: the router branches on severity. A
finding the Critic cannot categorise is a finding the graph cannot route, and a
free-text `kind` would let the model invent a category that silently maps to "no
action".

The severity scale is what decides whether a run revises, degrades or ships.
`BLOCKING` means the report cannot be published as written -- a fabricated
citation, a quote that does not appear in its source. That is not a quality
opinion; it is a factual defect, and the distinction from `MAJOR` is exactly the
distinction between "this is wrong" and "this is weak".
"""

from __future__ import annotations

import enum
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel

__all__ = [
    "MAX_FINDINGS",
    "BLOCKING_KINDS",
    "CriticInput",
    "CriticOutput",
    "Finding",
    "FindingKind",
    "Severity",
]

MAX_FINDINGS: Final = 20


class FindingKind(enum.StrEnum):
    """The closed vocabulary of `docs/agent-system.md` §13."""

    BROKEN_CITATION = "broken_citation"
    """A cited signal does not exist or cannot be resolved. Always blocking."""

    MISQUOTE = "misquote"
    """A quoted span does not appear in the cited source. Always blocking.

    The failure `services/evidence_service.py` exists to catch: a model that
    paraphrases while believing it is quoting produces this constantly and
    reports it never.
    """

    UNSUPPORTED_CLAIM = "unsupported_claim"
    OVERSTATED_CONFIDENCE = "overstated_confidence"
    MISSING_COVERAGE = "missing_coverage"
    CONTRADICTION = "contradiction"
    STALE_EVIDENCE = "stale_evidence"
    SOURCE_CONCENTRATION = "source_concentration"
    """Too much of the conclusion rests on one source or one outlet.

    Worth its own kind because it is invisible in every other check: forty
    citations all tracing to one syndicated wire story looks like abundant
    evidence right up until someone follows the links.
    """

    BIAS = "bias"


BLOCKING_KINDS: Final[frozenset[FindingKind]] = frozenset(
    {FindingKind.BROKEN_CITATION, FindingKind.MISQUOTE}
)
"""Findings that are factual defects rather than quality judgements.

These two are mechanically verifiable and always blocking. Everything else is a
matter of degree, and its severity depends on what the claim is load-bearing for.
"""


class Severity(enum.StrEnum):
    BLOCKING = "blocking"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class Finding(StrictModel):
    """One defect the Critic found."""

    kind: FindingKind
    severity: Severity
    target: str = Field(
        min_length=1,
        max_length=200,
        description="What this is about: an insight id, a claim id, a section name.",
    )
    detail: str = Field(min_length=1, max_length=1000)
    suggested_fix: str | None = Field(default=None, max_length=500)
    signal_ids: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _factual_defects_are_blocking(self) -> Finding:
        """A broken citation cannot be downgraded to a suggestion.

        Without this, a model that wants the run to proceed marks a fabricated
        citation `minor` and the report ships with it. The two kinds in
        `BLOCKING_KINDS` are mechanically verifiable facts, not judgements, so
        their severity is not the model's to choose.
        """
        if self.kind in BLOCKING_KINDS and self.severity is not Severity.BLOCKING:
            raise ValueError(
                f"{self.kind.value} is a factual defect and is always blocking; it "
                f"cannot be reported as {self.severity.value}"
            )
        return self


class CriticInput(StrictModel):
    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    insights: list[dict] = Field(default_factory=list, max_length=12)
    recommendations: list[dict] = Field(default_factory=list, max_length=8)
    trends: list[dict] = Field(default_factory=list, max_length=10)
    report: dict | None = None
    evidence_ids: list[str] = Field(default_factory=list, max_length=60)
    sub_questions: list[str] = Field(default_factory=list, max_length=8)
    unanswered_sub_questions: list[str] = Field(default_factory=list, max_length=8)
    degraded_backends: list[str] = Field(default_factory=list, max_length=4)
    revision_count: int = 0
    is_final_pass: bool = Field(
        default=False,
        description=(
            "True when the run has exhausted its revision budget. The Critic still "
            "reports, but the router will ship with findings rather than loop."
        ),
    )


class CriticOutput(StrictModel):
    findings: list[Finding] = Field(default_factory=list, max_length=MAX_FINDINGS)
    confidence: Score = Field(
        default=0.0,
        description=(
            "Overall confidence in the report as it stands. Drives the badge the "
            "UI renders and is deliberately the Critic's number rather than the "
            "Report's -- an author scoring their own work grades generously."
        ),
    )
    summary: str = Field(min_length=1, max_length=2000)
    approved: bool = Field(
        default=False, description="Whether this can be published as written."
    )

    @model_validator(mode="after")
    def _cannot_approve_over_a_blocking_finding(self) -> CriticOutput:
        """Approval and a blocking finding are contradictory.

        A model asked both to critique and to approve will, when the run is long
        and the findings are minor-looking, do both. This makes that
        unrepresentable.
        """
        blocking = [f for f in self.findings if f.severity is Severity.BLOCKING]
        if self.approved and blocking:
            raise ValueError(
                f"cannot approve with {len(blocking)} blocking finding(s): "
                f"{', '.join(f.kind.value for f in blocking)}"
            )
        return self

    @property
    def blocking_findings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCKING]

    @property
    def requires_revision(self) -> bool:
        return bool(self.blocking_findings) or any(
            f.severity is Severity.MAJOR for f in self.findings
        )
