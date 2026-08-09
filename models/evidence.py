"""Evidence: a reference to a passage, and the verification that makes it a citation.

The distinction this module draws is the product's central one. A **reference**
points at a passage. A **citation** is a reference whose quoted text has been
checked against the stored source. They are different types here because they are
different claims, and a system that uses one word for both will eventually print
the first while meaning the second.

**No passage text is stored on a reference.** `agents/state.py` explains the size
consequence -- a run gathers hundreds of passages and a checkpoint should be
kilobytes. The deeper reason is freshness: text copied into a reference is a
snapshot, and a snapshot cannot be re-verified. Fetching on demand means a
reference to a passage that has since been redacted fails loudly instead of
quoting something that no longer exists.

**`quote` exists but is bounded and optional.** One sentence, only once a
downstream agent has committed to citing it. That is the single place text is
allowed to cross into the state, and the bound is what keeps it a citation rather
than a copy of the corpus.
"""

from __future__ import annotations

import enum
from typing import Final

from pydantic import Field, model_validator

from models.base import Score, StrictModel, UtcDatetime, utcnow
from models.enums import AgentName

__all__ = [
    "MAX_QUOTE_CHARS",
    "Citation",
    "EvidenceReference",
    "VerificationOutcome",
]

MAX_QUOTE_CHARS: Final = 500
"""The only place text enters the state, and how much of it.

A sentence or two. Enough to render a citation inline; far too little to become a
copy of the document, which is what an unbounded field would gradually become.
"""


class VerificationOutcome(enum.StrEnum):
    """Whether a quote was found in its cited source.

    Three states, not two, and the third is the important one. `UNVERIFIABLE`
    means the store could not be reached -- which is *not* the same as the quote
    being wrong, and treating it as such would turn a Postgres blip into a report
    full of fabrication findings.
    """

    VERIFIED = "verified"
    NOT_FOUND = "not_found"
    UNVERIFIABLE = "unverifiable"

    @property
    def is_citable(self) -> bool:
        """Whether this may be printed as a citation.

        Only `VERIFIED`. `UNVERIFIABLE` is deliberately excluded: an unchecked
        quote printed as a checked one is exactly the failure verification
        exists to prevent, and "we could not check" is not a reason to print it
        anyway.
        """
        return self is VerificationOutcome.VERIFIED


class EvidenceReference(StrictModel):
    """A pointer to a passage. Not yet a citation -- nothing has been verified."""

    signal_id: str = Field(min_length=1)
    chunk_id: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    relevance: Score = 0.0
    retrieved_by: AgentName = AgentName.RETRIEVER
    retrieved_at: UtcDatetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _range_is_coherent(self) -> EvidenceReference:
        """An inverted or empty span points at nothing.

        Caught here because the failure downstream is silent: a UI highlighting
        `[400, 200]` renders no highlight, and the citation looks unremarkable
        while pointing nowhere.
        """
        if self.char_start is not None and self.char_end is not None:
            if self.char_end <= self.char_start:
                raise ValueError(
                    f"char range [{self.char_start}, {self.char_end}) is empty or "
                    "inverted; it would highlight nothing"
                )
        return self

    @property
    def has_span(self) -> bool:
        return self.char_start is not None and self.char_end is not None


class Citation(StrictModel):
    """A reference whose quote has been checked against the stored source.

    Constructing one asserts the check happened. That is why `outcome` is
    required and why `is_printable` gates on it -- there is no way to build a
    `Citation` that claims verification it did not receive.
    """

    reference: EvidenceReference
    quote: str = Field(min_length=1, max_length=MAX_QUOTE_CHARS)
    outcome: VerificationOutcome
    verified_at: UtcDatetime = Field(default_factory=utcnow)
    detail: str | None = Field(
        default=None,
        max_length=500,
        description="Why verification failed, when it did. For a Critic finding.",
    )

    @model_validator(mode="after")
    def _failures_explain_themselves(self) -> Citation:
        if self.outcome is not VerificationOutcome.VERIFIED and not self.detail:
            raise ValueError(
                f"outcome is {self.outcome.value} but no detail was given; a failed "
                "verification that does not say why cannot be acted on"
            )
        return self

    @property
    def signal_id(self) -> str:
        return self.reference.signal_id

    @property
    def is_printable(self) -> bool:
        """Whether a report may render this.

        The gate. An unverified quote reaching a document is indistinguishable to
        a reader from a verified one, and there is no recovery once it is read.
        """
        return self.outcome.is_citable
