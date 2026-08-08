"""Competitor agent input and output schemas.

The distinction this module exists to preserve is `basis`: whether a rivalry was
*stated* in a document or *inferred* from co-occurrence. They are not the same
claim and must never render the same way. "Acme names Globex as its principal
competitor" is a sourced fact; "Acme and Globex are mentioned together often" is
a statistical observation that is frequently true of a company and its largest
customer, its supplier, and the analyst who covers both.

A report that flattens the two overstates its evidence in the direction that
matters most -- competitive positioning is what the reader acts on.
"""

from __future__ import annotations

import enum
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel

__all__ = [
    "MAX_COMPETITORS",
    "CompetitiveBasis",
    "CompetitorFinding",
    "CompetitorInput",
    "CompetitorOutput",
    "PositioningAxis",
]

MAX_COMPETITORS: Final = 12


class CompetitiveBasis(enum.StrEnum):
    """How the rivalry was established. See the module docstring."""

    STATED = "stated"
    """A document said so. The only basis that supports an unhedged claim."""

    INFERRED = "inferred"
    """Derived from co-occurrence or shared market signals. A hypothesis."""

    GRAPH = "graph"
    """A COMPETES_WITH edge, whose own basis is recorded in the graph."""


class PositioningAxis(StrictModel):
    """One dimension along which competitors are compared.

    Axes are named by the model rather than fixed, because the meaningful axis
    differs entirely by market -- price/performance for hardware, breadth/depth
    for software, coverage/latency for services. A fixed pair would force every
    comparison onto the same chart and make most of them meaningless.
    """

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)


class CompetitorFinding(StrictModel):
    """One rival, and what is actually known about the rivalry."""

    entity_id: str | None = Field(
        default=None,
        description=(
            "Graph id when the competitor resolved to a known entity. Null for a "
            "name mentioned in evidence that resolution has not yet seen -- which "
            "is a real and common case, not an error."
        ),
    )
    name: str = Field(min_length=1, max_length=200)
    basis: CompetitiveBasis
    strength: Score = Field(default=0.0, description="0-1 rivalry intensity.")
    confidence: Score = 0.0
    overlap: str | None = Field(
        default=None, max_length=400, description="Where the two actually compete."
    )
    differentiators: list[str] = Field(default_factory=list, max_length=6)
    signal_ids: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _stated_claims_need_a_source(self) -> CompetitorFinding:
        """A `stated` rivalry with no signal ids is an unsourced sourced-claim.

        The most consequential validation in this module. `stated` is the basis
        that licenses an unhedged sentence in the report; allowing one without a
        citation means the strongest claim the system can make is also the one
        with the least evidence behind it.
        """
        if self.basis is CompetitiveBasis.STATED and not self.signal_ids:
            raise ValueError(
                f"{self.name!r} is marked 'stated' but cites no signal. A stated "
                "rivalry is a sourced claim -- without a source it is an inference, "
                "and must be labelled one."
            )
        return self


class CompetitorInput(StrictModel):
    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    subject: str | None = Field(
        default=None, max_length=200, description="The company or product under analysis."
    )
    seed_entity_ids: list[str] = Field(default_factory=list, max_length=32)
    evidence_count: int = 0


class CompetitorOutput(StrictModel):
    subject: str | None = None
    competitors: list[CompetitorFinding] = Field(
        default_factory=list, max_length=MAX_COMPETITORS
    )
    axes: list[PositioningAxis] = Field(default_factory=list, max_length=4)
    summary: str | None = Field(default=None, max_length=2000)
    graph_available: bool = Field(
        default=True,
        description=(
            "False when the knowledge graph could not be read. The competitive "
            "picture is then built from retrieved text alone, which systematically "
            "misses rivals nobody wrote about in the retrieved window -- and the "
            "report must say so rather than presenting a partial list as complete."
        ),
    )

    @property
    def stated_count(self) -> int:
        return sum(1 for item in self.competitors if item.basis is CompetitiveBasis.STATED)
