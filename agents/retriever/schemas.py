"""Retriever input and output schemas.

The Retriever's output is the evidence every downstream agent reasons over, so
the one decision that matters here is what it is *not* allowed to carry:
passage text.

`EvidenceRef` in `agents/state.py` says it at length -- a run gathers hundreds of
passages, and inlining them makes every checkpoint megabytes and every resume a
full re-read. The `RetrievedItem` below mirrors that: ids, offsets, scores, and a
quote only when a downstream agent has already committed to citing it. The text
is fetched on demand through `services/evidence_service.py`, which re-verifies
the quote against the stored signal -- so a reference to text that has since been
redacted fails loudly instead of silently citing something that no longer exists.
"""

from __future__ import annotations

from typing import Final

from pydantic import Field

from models.base import Score, StrictModel
from models.enums import AgentName

__all__ = [
    "MAX_EVIDENCE_ITEMS",
    "MAX_QUOTE_CHARS",
    "RetrievalQuery",
    "RetrievedItem",
    "RetrieverInput",
    "RetrieverOutput",
]

MAX_EVIDENCE_ITEMS: Final = 60
"""Ceiling on evidence returned in one retrieval pass.

Sized against the *context window*, not against recall. Sixty references at a few
hundred characters each is what the downstream analysis agents can actually read
alongside their own instructions; returning three hundred would mean the
Insight agent silently truncates, and which forty it kept would depend on
serialisation order.
"""

MAX_QUOTE_CHARS: Final = 500
"""Matches `agents.state.EvidenceRef.quote`. The one place text enters the state."""

MAX_SUB_QUERIES: Final = 6
"""Ceiling on query decomposition.

Each sub-query is a full hybrid retrieval -- three backends, a fusion pass and a
rerank. Six is already a second or two of latency and a rerank bill; the failure
mode of no cap is a model decomposing "what is happening with Acme" into fifteen
near-identical paraphrases, each retrieving the same passages.
"""


class RetrievalQuery(StrictModel):
    """One search the Retriever decided to run."""

    text: str = Field(min_length=1, max_length=500)
    rationale: str | None = Field(default=None, max_length=300)
    sub_question_id: str | None = Field(
        default=None,
        max_length=40,
        description=(
            "Which sub-question this search serves. Carried so the Critic can "
            "check coverage per question rather than in aggregate -- a report "
            "with forty passages that all answer one of six sub-questions is "
            "well-evidenced and incomplete at the same time."
        ),
    )


class RetrievedItem(StrictModel):
    """One piece of evidence. A reference, not the passage.

    See the module docstring for why there is no `text` field.
    """

    signal_id: str = Field(min_length=1)
    chunk_id: str | None = None
    quote: str | None = Field(default=None, max_length=MAX_QUOTE_CHARS)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    relevance: Score = 0.0
    sub_question_id: str | None = Field(default=None, max_length=40)


class RetrieverInput(StrictModel):
    """The projection of the state the Retriever needs."""

    query: str = Field(min_length=1)
    objective: str = ""
    tenant_id: str
    sub_questions: list[str] = Field(default_factory=list, max_length=8)
    sub_question_ids: list[str] = Field(default_factory=list, max_length=8)
    seed_entity_ids: list[str] = Field(default_factory=list, max_length=32)
    already_retrieved: int = Field(
        default=0,
        description=(
            "Evidence already in the state from an earlier pass. Read so a "
            "re-retrieval after a Critic finding does not return the same set."
        ),
    )


class RetrieverOutput(StrictModel):
    """Evidence gathered, plus what the retrieval could not do.

    `degraded_backends` is not diagnostics. `docs/architecture.md` §7.3 lets
    retrieval continue with keyword-only results when Qdrant is down, and a
    report built on that basis is genuinely weaker. Recording which backends were
    missing is what lets the Critic lower confidence for a reason it can name,
    instead of the run silently producing a thinner answer that looks the same.
    """

    items: list[RetrievedItem] = Field(default_factory=list, max_length=MAX_EVIDENCE_ITEMS)
    queries_run: list[str] = Field(default_factory=list, max_length=MAX_SUB_QUERIES)
    degraded_backends: list[str] = Field(default_factory=list, max_length=4)
    total_candidates: int = 0
    retrieved_by: AgentName = AgentName.RETRIEVER

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded_backends)

    @property
    def covered_sub_questions(self) -> set[str]:
        """Which sub-questions this pass found evidence for."""
        return {
            item.sub_question_id for item in self.items if item.sub_question_id is not None
        }
