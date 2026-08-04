"""Assemble reranked passages and graph facts into a budgeted evidence pack.

`docs/retrieval.md` §9 hands the agents in `agents/` one object, not two streams,
and §9's budgeting table is the reason this module exists at all: the pack is
budgeted to `token_budget` (24,000 by default) which sits *far* below the model
context window, because that window must also hold the system prompt, the tool
schemas, the agent's scratchpad and the output. A pack that merely "fits the
model" overflows the moment the agent thinks.

Three decisions here are load-bearing, and each of them is a failure mode that
would otherwise be invisible in the finished report.

**Breadth before depth.** Budgeting purely greedily by `final_score` is wrong in
a way that reads as correct: one verbose, highly-ranked source produces six
consecutive top-ranked chunks, fills the passage allocation on its own, and five
corroborating sources never enter the pack. The report then rests on a single
outlet while looking well-evidenced. So selection runs in two phases -- one
passage per distinct `signal_id` first, in score order, and only then a second
from any signal. Corroboration across independent sources is the strongest
evidence available (it is why `Passage.duplicate_of_count` is kept at all), and
the budget must not be spent on redundancy while breadth is still available.

**Whole passages, never truncated.** Half a passage produces a quote that
`services/evidence_service.py` cannot verify against its stored span, and an
unverifiable citation is worse than a missing one because it looks like evidence.
A passage that does not fit is skipped and the fill continues -- a later, smaller
passage may still fit, and stranding the tail of the budget to preserve a strict
score order costs recall for nothing.

**Every drop is counted.** `dropped_passages` / `dropped_facts` travel with the
pack so a report can say "analysis based on 12 of 47 retrieved passages" instead
of implying exhaustiveness. `EvidencePack.is_complete` is what the Critic reads
to decide whether the answer is allowed to sound complete.

Two things are deliberately *not* here. Token counting is a port
(`TokenCounter`): §9 requires counts from the active provider's tokenizer, which
lives behind `services/llm/router.py`, and `retrieval/` is layer L1 and may not
import `services/` (`docs/architecture.md` §6.1). And `entity_cards` /
`community_summaries` from the §9 sketch have no field on
`retrieval.types.EvidencePack`; their share of the budget is accounted for below
rather than reserved for something the pack cannot yet carry.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from models.enums import Platform
from retrieval.types import EvidencePack, GraphFact, Passage, RetrievalResult

__all__ = [
    "DEFAULT_CHARS_PER_TOKEN",
    "FACT_SHARE",
    "PASSAGE_SHARE",
    "RESERVE_SHARE",
    "ApproximateTokenCounter",
    "BudgetAllocation",
    "ContextBuilder",
    "TokenCounter",
    "approximate_token_count",
    "render_fact",
    "render_passage",
]


# --------------------------------------------------------------------------- #
# The budget shares of docs/retrieval.md §9
# --------------------------------------------------------------------------- #

PASSAGE_SHARE: Final = 0.65
"""Share of the budget for passages. 15,600 tokens at the 24,000 default."""

FACT_SHARE: Final = 0.15
"""Share for graph facts. Small, because a fact is one line and a passage is not."""

RESERVE_SHARE: Final = 0.05
"""Never allocated. Absorbs tokenizer variance.

The counts here come from a tokenizer that is not necessarily the one the serving
model uses -- a cached count, a different provider, an approximation in
development. A pack sized to exactly 100% of the window is one rounding
disagreement away from a context-overflow error mid-investigation, which costs
the whole run rather than one passage.
"""

UNCLAIMED_SHARE: Final = 0.15
"""Entity cards (5%) + community summaries (10%) from §9.

`EvidencePack` has no field for either -- entity cards are unbuilt and
`retrieval/graphrag/community.py` cannot produce summaries until graph analytics
exist. Reserving their share would drop real passages to protect space nothing
can occupy, so it goes to passages until those components arrive, at which point
this constant shrinks rather than the passage share.
"""

DEFAULT_CHARS_PER_TOKEN: Final = 4.0
"""The usual English rule of thumb, used only by `approximate_token_count`."""


# --------------------------------------------------------------------------- #
# Ports
# --------------------------------------------------------------------------- #


@runtime_checkable
class TokenCounter(Protocol):
    """Counts tokens for a batch of rendered strings. One count per string.

    Batched and async because the real implementation is a provider tokenizer
    reached through `services/llm/router.py`, possibly over the network. Counting
    fifty passages one at a time would turn pack assembly into fifty round trips
    on the request path.

    It takes the *rendered* string, not a `Passage`, because what costs tokens is
    what is sent to the model -- citation handles and fact lines included.
    Counting raw passage text would undercount every pack by its own headers and
    quietly reintroduce the overflow the reserve exists to prevent.
    """

    async def __call__(self, texts: Sequence[str]) -> Sequence[int]:
        """Return one token count per text, in the order the texts were given."""
        ...


def approximate_token_count(text: str) -> int:
    """Characters ÷ 4, rounded up. A development and test fallback only.

    `docs/retrieval.md` §9 requires the provider's tokenizer, "never a character
    heuristic", and this is exactly that heuristic. It exists so the builder can
    be exercised without a model, and it is named explicitly at the call site
    rather than defaulted, so it cannot ship to production by omission. It
    under-counts CJK text and code badly, which is one reason the reserve share
    is not optional.
    """
    return math.ceil(len(text) / DEFAULT_CHARS_PER_TOKEN)


class ApproximateTokenCounter:
    """`TokenCounter` over `approximate_token_count`. Development and tests only."""

    async def __call__(self, texts: Sequence[str]) -> Sequence[int]:
        return [approximate_token_count(text) for text in texts]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

PassageRenderer = Callable[[Passage], str]
FactRenderer = Callable[[GraphFact], str]


def render_passage(passage: Passage) -> str:
    """One passage as the agent sees it: a citation handle, then the text.

    The handle leads with `chunk_id` because that is the join key back to
    PostgreSQL, OpenSearch and Qdrant (`retrieval.types.chunk_id_for`), so a
    model quoting this block gives `services/evidence_service.py` everything it
    needs to re-fetch the span and verify the quote. The corroboration count is
    included when non-zero: "reported by 6 sources" is a claim the report is
    entitled to make, and it is derivable from nothing else in the block.
    """
    parts = [f"chunk={passage.chunk_id}", f"signal={passage.signal_id}"]
    if passage.title:
        parts.append(f"title={passage.title}")
    if passage.published_at is not None:
        parts.append(f"published={passage.published_at.date().isoformat()}")
    if passage.platform is not Platform.UNKNOWN:
        parts.append(f"platform={passage.platform.value}")
    if passage.url:
        parts.append(f"url={passage.url}")
    if passage.duplicate_of_count:
        parts.append(f"corroborated_by={passage.duplicate_of_count + 1}")
    return f"[{' | '.join(parts)}]\n{passage.text}"


def render_fact(fact: GraphFact) -> str:
    """One graph edge as a compact line, per the `FACT ...` form in §9.

    A line rather than prose, on purpose: prose blurs an inferred edge into a
    human-written sentence, and a claim the graph *derived* must never be citable
    as though a source had said it. Validity travels with the fact so an edge
    that ended in 2024 cannot be read as current.
    """
    if fact.valid_from is not None:
        end = "present" if fact.is_current else fact.valid_to.date().isoformat()  # type: ignore[union-attr]
        validity = f"valid {fact.valid_from.date().isoformat()} → {end}"
    elif fact.is_current:
        validity = "valid ? → present"
    else:
        validity = f"valid ? → {fact.valid_to.date().isoformat()}"  # type: ignore[union-attr]
    return (
        f"FACT {fact.subject_name} {fact.predicate} {fact.object_name} "
        f"[{validity}, confidence {fact.confidence:.2f}, "
        f"{len(fact.supporting_signal_ids)} signals]"
    )


# --------------------------------------------------------------------------- #
# Budgeting
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class BudgetAllocation:
    """How one `token_budget` splits across the pack's components.

    Computed rather than configured, so a caller that lowers the budget for a
    cheaper model tier cannot forget to rescale the per-component caps and
    overflow the smaller window.
    """

    passages: int
    facts: int
    reserve: int

    @classmethod
    def for_budget(cls, token_budget: int) -> BudgetAllocation:
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")
        facts = int(token_budget * FACT_SHARE)
        reserve = int(token_budget * RESERVE_SHARE)
        # Passages take their share plus the unclaimed one. Floor division loses
        # a token or two, which lands in the reserve rather than in an overflow.
        passages = int(token_budget * (PASSAGE_SHARE + UNCLAIMED_SHARE))
        return cls(passages=passages, facts=facts, reserve=reserve)

    @property
    def allocated(self) -> int:
        """Everything a pack may actually spend. Always below `token_budget`."""
        return self.passages + self.facts


class ContextBuilder:
    """Packs passages and graph facts into an `EvidencePack` that fits a budget.

    Stateless apart from its ports; one instance per process, safe to drive
    concurrently.
    """

    def __init__(
        self,
        count_tokens: TokenCounter,
        *,
        passage_renderer: PassageRenderer = render_passage,
        fact_renderer: FactRenderer = render_fact,
    ) -> None:
        self._count_tokens = count_tokens
        self._render_passage = passage_renderer
        self._render_fact = fact_renderer

    async def build_from_result(self, result: RetrievalResult) -> EvidencePack:
        """Pack a retrieval result using the budget its own request asked for."""
        return await self.build(
            result.passages,
            result.graph_facts,
            token_budget=result.request.token_budget,
        )

    async def build(
        self,
        passages: Sequence[Passage],
        graph_facts: Sequence[GraphFact] = (),
        *,
        token_budget: int,
    ) -> EvidencePack:
        """Fit passages and facts to `token_budget`, counting everything dropped.

        Facts are fitted first. They are one line each and they are the only
        thing in the pack that states a *relationship* rather than an excerpt, so
        per token they carry more; whatever their allocation leaves unused flows
        to passages instead of going unspent.
        """
        allocation = BudgetAllocation.for_budget(token_budget)

        # Passages that cannot be quoted are removed before budgeting rather than
        # packed: `is_citable` is false when the text is empty or the span is
        # degenerate, and a citation built from one cannot be verified. They are
        # still counted as dropped -- a pack that silently discarded them would
        # claim a completeness it does not have.
        citable = [p for p in passages if p.is_citable]
        uncitable = len(passages) - len(citable)

        rendered_passages = [self._render_passage(p) for p in citable]
        rendered_facts = [self._render_fact(f) for f in graph_facts]

        # One batched call for everything: the port may be a network round trip
        # and pack assembly sits on the request path.
        counts = await self._count([*rendered_passages, *rendered_facts])
        passage_costs = counts[: len(rendered_passages)]
        fact_costs = counts[len(rendered_passages) :]

        kept_facts, facts_used = _select_facts(graph_facts, fact_costs, allocation.facts)
        passage_limit = allocation.passages + (allocation.facts - facts_used)
        kept_passages, passages_used = _select_passages(citable, passage_costs, passage_limit)

        return EvidencePack(
            passages=tuple(kept_passages),
            graph_facts=tuple(kept_facts),
            token_count=passages_used + facts_used,
            token_budget=token_budget,
            dropped_passages=len(citable) - len(kept_passages) + uncitable,
            dropped_facts=len(graph_facts) - len(kept_facts),
        )

    async def _count(self, texts: Sequence[str]) -> list[int]:
        """Token counts for every rendered block, validated against the input.

        A short or long answer would zip costs onto the wrong blocks and size the
        pack against something it does not contain, so it is fatal here rather
        than an overflow at generation time, three agent steps later.
        """
        if not texts:
            return []
        counts = list(await self._count_tokens(texts))
        if len(counts) != len(texts):
            raise ValueError(
                f"token counter returned {len(counts)} counts for {len(texts)} texts; "
                "costs would be attributed to the wrong blocks"
            )
        if any(count < 0 for count in counts):
            raise ValueError(f"token counter returned a negative count: {counts!r}")
        return [int(count) for count in counts]


# --------------------------------------------------------------------------- #
# Selection -- pure, so it is testable without a tokenizer
# --------------------------------------------------------------------------- #


def _select_passages(
    passages: Sequence[Passage], costs: Sequence[int], limit: int
) -> tuple[list[Passage], int]:
    """Greedy fill by `final_score`, one passage per signal before any second.

    Phase 1 walks the passages in score order and takes the best *affordable*
    passage of each distinct `signal_id`. Phase 2 walks what is left, still in
    score order, and adds whatever still fits. Without phase 1 a single verbose
    source crowds out every corroborating one (see the module docstring).

    A passage that does not fit is skipped rather than ending the fill: passages
    are packed whole, so skipping is the only alternative to truncating, and
    stopping at the first oversized passage would strand the rest of the budget.
    The cost is that a smaller, lower-scored passage can take a place a larger
    one wanted -- a budgeting decision, not a ranking claim, and the pack is still
    emitted in ranked order.

    Returns the selected passages in score order and the tokens they spend.
    """
    order = sorted(range(len(passages)), key=lambda i: (-passages[i].final_score, i))

    kept: set[int] = set()
    seen_signals: set[str] = set()
    remaining = limit

    for i in order:  # phase 1: breadth -- one per distinct signal
        signal_id = passages[i].signal_id
        if signal_id in seen_signals:
            continue
        if costs[i] > remaining:
            # Not marked as seen: a shorter passage from the same signal may
            # still be able to represent it.
            continue
        kept.add(i)
        seen_signals.add(signal_id)
        remaining -= costs[i]

    for i in order:  # phase 2: depth -- second and later passages per signal
        if i in kept or costs[i] > remaining:
            continue
        kept.add(i)
        remaining -= costs[i]

    return [passages[i] for i in order if i in kept], limit - remaining


def _select_facts(
    facts: Sequence[GraphFact], costs: Sequence[int], limit: int
) -> tuple[list[GraphFact], int]:
    """Greedy fill by confidence -- §9's "lowest confidence first" eviction, read
    forwards.

    Current facts outrank ended ones at equal confidence: a pack answers a
    question about now unless the request said otherwise, and a superseded edge
    sitting next to a live one invites the model to state it in the present
    tense.
    """
    order = sorted(
        range(len(facts)),
        key=lambda i: (-facts[i].confidence, not facts[i].is_current, i),
    )

    kept: set[int] = set()
    remaining = limit
    for i in order:
        if costs[i] > remaining:
            continue
        kept.add(i)
        remaining -= costs[i]

    return [facts[i] for i in order if i in kept], limit - remaining
