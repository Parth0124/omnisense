"""Unit tests for `retrieval/hybrid.py` -- fan out, fuse, dedupe, rerank, resolve.

Ranking quality is not what these tests are for. The orchestrator's job is to
hold a specific *shape* of failure at bay, and every one of those failures
produces a plausible-looking result rather than an exception:

**A backend that is down must cost recall, not the request.** `docs/retrieval.md`
§12 and `docs/architecture.md` §7.3 both say a Qdrant outage means keyword-only
retrieval, not a failed investigation. The dangerous half is the second one: the
run must *say* it was degraded, because five passages presented as twelve is
worse than an error. So the tests assert on `RetrievalDiagnostics.degraded` and
`backends_failed` as hard as they assert on the passages.

**A chunk that no longer exists must be skipped, not raised on.** An erasure
between indexing and querying leaves a live chunk id in Qdrant pointing at
nothing. Raising there lets one deleted record break every query that happens to
match it.

**The reranker is optional infrastructure.** When it fails the fused RRF order is
a coherent ranking; a half-reranked list is not, because rerank scores and fused
scores are different scales and a passage's position would depend on which slice
it landed in.

**Fan-out is concurrent.** Three sequential round trips make a hybrid query cost
the sum rather than the max of its backends. That is asserted structurally --
with a barrier the backends can only clear together -- rather than by timing,
because a timing assertion on a loaded CI box is a flake generator.

Every backend, the resolver, the expander and the reranker are fakes. No network,
no services, no datastore.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pytest

from models.enums import Platform, SourceCategory
from retrieval.hybrid import (
    GraphExpander,
    HybridRetriever,
    PassageResolver,
    Reranker,
    SearchBackend,
)
from retrieval.types import (
    Backend,
    Candidate,
    Filter,
    GraphFact,
    Passage,
    RetrievalRequest,
    chunk_id_for,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def make_candidates(backend: Backend, signal_ids: Sequence[str]) -> list[Candidate]:
    """One candidate per signal, ranked in the order given, chunk 0 of each."""
    return [
        Candidate(
            chunk_id=chunk_id_for(signal_id, 0),
            backend=backend,
            rank=rank,
            raw_score=1.0 / rank,
        )
        for rank, signal_id in enumerate(signal_ids, start=1)
    ]


def make_passage(chunk_id: str, *, text: str | None = None) -> Passage:
    """A citable passage with enough provenance that a citation would resolve."""
    signal_id, index = chunk_id.rsplit(":", 1)
    body = text if text is not None else f"body of {chunk_id}"
    return Passage(
        chunk_id=chunk_id,
        signal_id=signal_id,
        text=body,
        char_start=0,
        char_end=len(body),
        platform=Platform.RSS,
        source=SourceCategory.NEWS,
        url=f"https://example.test/{signal_id}/{index}",
        title=f"title {signal_id}",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        signal_confidence=0.7,
    )


class FakeBackend:
    """A `SearchBackend` that answers from a script, or refuses to answer at all.

    `gate` exists for the concurrency test: when set, the backend waits on it
    before returning, so three backends can only all return if all three were
    started before any of them finished.
    """

    def __init__(
        self,
        backend: Backend,
        candidates: Sequence[Candidate] = (),
        *,
        fails: BaseException | None = None,
        gate: asyncio.Barrier | None = None,
    ) -> None:
        self.backend = backend
        self._candidates = list(candidates)
        self._fails = fails
        self._gate = gate
        self.calls: list[tuple[RetrievalRequest, int]] = []

    async def search(
        self, request: RetrievalRequest, *, limit: int
    ) -> Sequence[Candidate]:
        self.calls.append((request, limit))
        if self._gate is not None:
            await self._gate.wait()
        if self._fails is not None:
            raise self._fails
        return self._candidates[:limit]


class FakeResolver:
    """A `PassageResolver` over an in-memory table, recording every batch.

    Ids absent from the table are simply absent from the answer -- exactly what
    the real resolver does for a chunk whose Signal has been erased.
    """

    def __init__(self, passages: Sequence[Passage] = ()) -> None:
        self.table = {p.chunk_id: p for p in passages}
        self.batches: list[list[str]] = []

    async def resolve(self, chunk_ids: Sequence[str]) -> Mapping[str, Passage]:
        self.batches.append(list(chunk_ids))
        return {cid: self.table[cid] for cid in chunk_ids if cid in self.table}


class FakeExpander:
    """A `GraphExpander` that can fail on either half independently.

    Independently, because the two halves fail differently: losing expansion
    costs alias recall before the fan-out, losing facts costs graph context
    after it, and neither may take the query with it.
    """

    def __init__(
        self,
        *,
        aliases: Sequence[str] = (),
        facts: Sequence[GraphFact] = (),
        expand_fails: bool = False,
        facts_fail: bool = False,
    ) -> None:
        self._aliases = list(aliases)
        self._facts = list(facts)
        self._expand_fails = expand_fails
        self._facts_fail = facts_fail
        self.expand_calls = 0
        self.fact_calls: list[list[str]] = []

    async def expand_query(self, request: RetrievalRequest) -> Sequence[str]:
        self.expand_calls += 1
        if self._expand_fails:
            raise ConnectionError("neo4j unreachable")
        return self._aliases

    async def facts_for(
        self, request: RetrievalRequest, signal_ids: Sequence[str]
    ) -> Sequence[GraphFact]:
        self.fact_calls.append(sorted(signal_ids))
        if self._facts_fail:
            raise TimeoutError("fact lookup timed out")
        return self._facts


class FakeReranker:
    """A `Reranker` that reverses the fused order, or fails.

    Reversal is deliberate: it is the ordering fusion would never produce, so a
    test cannot pass by accident when the reranker's output is ignored.
    """

    def __init__(self, *, fails: bool = False) -> None:
        self._fails = fails
        self.calls: list[tuple[str, list[str], int]] = []

    async def rerank(
        self, query: str, passages: Sequence[Passage], *, top_k: int
    ) -> Sequence[Passage]:
        self.calls.append((query, [p.chunk_id for p in passages], top_k))
        if self._fails:
            raise RuntimeError("reranker endpoint returned 503")
        import dataclasses

        reversed_passages = list(reversed(passages))
        return [
            dataclasses.replace(passage, rerank_score=float(len(passages) - i))
            for i, passage in enumerate(reversed_passages)
        ][:top_k]


def build_retriever(
    *,
    backends: Sequence[SearchBackend],
    resolver: PassageResolver,
    expander: GraphExpander | None = None,
    reranker: Reranker | None = None,
    simhash_of: object | None = None,
) -> HybridRetriever:
    return HybridRetriever(
        backends,
        resolver,
        expander=expander,
        reranker=reranker,
        simhash_of=simhash_of,
    )


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #


def test_fakes_satisfy_the_protocols_they_stand_in_for() -> None:
    """A fake that has drifted from the protocol tests nothing about the real thing."""
    assert isinstance(FakeBackend(Backend.KEYWORD), SearchBackend)
    assert isinstance(FakeResolver(), PassageResolver)
    assert isinstance(FakeExpander(), GraphExpander)
    assert isinstance(FakeReranker(), Reranker)


def test_two_backends_claiming_one_slot_is_refused_at_construction() -> None:
    """Silently keeping the last one would drop a whole backend's recall.

    The failure is invisible at query time -- the fan-out simply never asks the
    shadowed backend -- so it has to be caught where the mistake is made.
    """
    with pytest.raises(ValueError, match="same Backend value"):
        HybridRetriever(
            [FakeBackend(Backend.KEYWORD), FakeBackend(Backend.KEYWORD)],
            FakeResolver(),
        )


# --------------------------------------------------------------------------- #
# Healthy fan-out
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_healthy_fan_out_fuses_all_three_backends() -> None:
    """The happy path, asserted on the thing fusion exists for: agreement.

    `sig_b` is ranked second, third and first by the three backends and never
    first twice, yet it must outrank `sig_a`, which one backend ranked first and
    the others never returned. That is the whole point of fusing before
    truncating, and a pipeline that quietly dropped a backend would still return
    "results" -- just with `sig_a` on top.
    """
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, ["sig_a", "sig_b"]))
    vector = FakeBackend(Backend.VECTOR, make_candidates(Backend.VECTOR, ["sig_c", "sig_b"]))
    graph = FakeBackend(Backend.GRAPH, make_candidates(Backend.GRAPH, ["sig_b", "sig_c"]))
    resolver = FakeResolver(
        [make_passage(chunk_id_for(s, 0)) for s in ("sig_a", "sig_b", "sig_c")]
    )
    retriever = build_retriever(
        backends=[keyword, vector, graph], resolver=resolver
    )

    result = await retriever.retrieve(RetrievalRequest(query="acme layoffs"))

    assert [p.signal_id for p in result.passages][0] == "sig_b"
    assert result.signal_ids == ["sig_b", "sig_c", "sig_a"]

    found_by = {p.signal_id: p.found_by for p in result.passages}
    assert found_by["sig_b"] == frozenset(Backend)
    assert found_by["sig_a"] == frozenset({Backend.KEYWORD})

    # Ranks survive fusion keyed by backend *name*, because the diagnostic
    # question after a recall regression is "which backend stopped contributing".
    ranks = {p.signal_id: dict(p.ranks) for p in result.passages}
    assert ranks["sig_b"] == {"keyword": 2, "vector": 2, "graph": 1}

    assert result.diagnostics.per_backend_counts == {"keyword": 2, "vector": 2, "graph": 2}
    assert set(result.diagnostics.per_backend_latency_ms) == {"keyword", "vector", "graph"}
    assert result.diagnostics.backends_failed == ()
    assert result.diagnostics.degraded is False
    assert result.diagnostics.fused_pool_size == 3
    assert result.diagnostics.total_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_each_backend_gets_its_own_limit_and_the_whole_filter() -> None:
    """Filters are pushed down, per backend, with that backend's k.

    A filter applied *after* a fixed-size ANN result is not the same operation:
    asking for 100 neighbours and keeping the 3 in-date ones silently makes the
    last month of data unreachable. Nothing downstream can detect that, so the
    only place to assert it is at the call boundary.
    """
    keyword = FakeBackend(Backend.KEYWORD)
    vector = FakeBackend(Backend.VECTOR)
    graph = FakeBackend(Backend.GRAPH)
    retriever = build_retriever(
        backends=[keyword, vector, graph], resolver=FakeResolver()
    )
    filters = Filter(
        published_after=datetime(2026, 1, 1, tzinfo=UTC),
        platforms=frozenset({Platform.RSS}),
        tenant_id="acme",
    )
    request = RetrievalRequest(
        query="q", filters=filters, k_keyword=77, k_vector=55, k_graph=11
    )

    await retriever.retrieve(request)

    assert keyword.calls[0][1] == 77
    assert vector.calls[0][1] == 55
    assert graph.calls[0][1] == 11
    for backend in (keyword, vector, graph):
        assert backend.calls[0][0].filters is filters


@pytest.mark.asyncio
async def test_a_backend_not_in_the_request_is_never_consulted() -> None:
    """Degraded operation is a request-level decision, not a silent catch.

    `RetrievalRequest.with_backends` is how a caller that already knows Qdrant is
    down asks for keyword-only retrieval -- and the caller that made that choice
    is the one that must report the reduced confidence, so the omission is not
    recorded as a failure here.
    """
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, ["sig_a"]))
    vector = FakeBackend(Backend.VECTOR, make_candidates(Backend.VECTOR, ["sig_b"]))
    resolver = FakeResolver([make_passage(chunk_id_for("sig_a", 0))])
    retriever = build_retriever(backends=[keyword, vector], resolver=resolver)

    request = RetrievalRequest(query="q").with_backends(Backend.KEYWORD)
    result = await retriever.retrieve(request)

    assert vector.calls == []
    assert result.diagnostics.per_backend_counts == {"keyword": 1}
    assert result.diagnostics.backends_failed == ()
    assert result.signal_ids == ["sig_a"]


@pytest.mark.asyncio
async def test_chunk_ids_are_resolved_in_one_batch() -> None:
    """Resolution is batched once, not once per backend.

    Resolving inside each backend would fetch the same passage three times in
    exactly the case the design hopes for -- all three backends agreeing -- so
    the batching is load-bearing rather than an optimisation.
    """
    shared = ["sig_a", "sig_b", "sig_c"]
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, shared))
    vector = FakeBackend(Backend.VECTOR, make_candidates(Backend.VECTOR, shared))
    resolver = FakeResolver([make_passage(chunk_id_for(s, 0)) for s in shared])
    retriever = build_retriever(backends=[keyword, vector], resolver=resolver)

    await retriever.retrieve(RetrievalRequest(query="q"))

    assert len(resolver.batches) == 1
    assert sorted(resolver.batches[0]) == sorted(chunk_id_for(s, 0) for s in shared)


@pytest.mark.asyncio
async def test_backends_are_queried_concurrently() -> None:
    """Three backends must overlap; sequential fan-out costs the sum, not the max.

    Asserted with a barrier rather than a stopwatch: each backend blocks until
    all three have arrived, so a sequential implementation deadlocks and the
    `wait_for` fails the test, while a timing assertion would flake on a loaded
    runner.
    """
    gate = asyncio.Barrier(3)
    backends = [
        FakeBackend(name, make_candidates(name, ["sig_a"]), gate=gate)
        for name in (Backend.KEYWORD, Backend.VECTOR, Backend.GRAPH)
    ]
    resolver = FakeResolver([make_passage(chunk_id_for("sig_a", 0))])
    retriever = build_retriever(backends=backends, resolver=resolver)

    result = await asyncio.wait_for(
        retriever.retrieve(RetrievalRequest(query="q")), timeout=5
    )

    assert result.signal_ids == ["sig_a"]


@pytest.mark.asyncio
async def test_pool_max_truncates_after_fusion_and_k_final_after_reranking() -> None:
    """Two different cuts, in the right order.

    `pool_max` bounds what the expensive stages see; `k_final` bounds what the
    agent sees. Truncating before fusion would deny a chunk the chance to be
    rescued by agreement from another backend, which is the one thing fusion is
    for.
    """
    signals = [f"sig_{i:02d}" for i in range(20)]
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, signals))
    resolver = FakeResolver([make_passage(chunk_id_for(s, 0)) for s in signals])
    retriever = build_retriever(backends=[keyword], resolver=resolver)

    result = await retriever.retrieve(
        RetrievalRequest(query="q", pool_max=8, k_final=3)
    )

    assert result.diagnostics.fused_pool_size == 8
    assert len(result.passages) == 3
    assert result.signal_ids == signals[:3]


# --------------------------------------------------------------------------- #
# Degradation: one backend down, then all of them
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_one_backend_down_degrades_rather_than_failing() -> None:
    """A Qdrant outage is keyword-only retrieval, not a failed investigation.

    Two assertions, and the second matters more than the first: the surviving
    backends still produce passages, *and* the result says it was degraded. A
    thin answer presented as a complete one is the failure this whole module is
    arranged to prevent.
    """
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, ["sig_a", "sig_b"]))
    vector = FakeBackend(Backend.VECTOR, fails=ConnectionError("qdrant refused"))
    graph = FakeBackend(Backend.GRAPH, make_candidates(Backend.GRAPH, ["sig_b"]))
    resolver = FakeResolver(
        [make_passage(chunk_id_for(s, 0)) for s in ("sig_a", "sig_b")]
    )
    retriever = build_retriever(backends=[keyword, vector, graph], resolver=resolver)

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert result.signal_ids == ["sig_b", "sig_a"]
    assert result.diagnostics.backends_failed == ("vector",)
    assert result.diagnostics.degraded is True
    # Counted as zero rather than omitted: "the vector backend contributed
    # nothing" and "the vector backend was never asked" are different incidents.
    assert result.diagnostics.per_backend_counts["vector"] == 0
    assert "vector" in result.diagnostics.per_backend_latency_ms


@pytest.mark.asyncio
async def test_all_backends_down_returns_an_empty_result_without_raising() -> None:
    """`docs/retrieval.md` §12: an empty pack with the filters echoed.

    The agent's contract is to say "no evidence found" and never improvise, and
    it can only do that if it is handed an empty result rather than an
    exception. The request travels back with the result so the caller can report
    which filters produced the emptiness.
    """
    backends = [
        FakeBackend(name, fails=ConnectionError(f"{name} down"))
        for name in (Backend.KEYWORD, Backend.VECTOR, Backend.GRAPH)
    ]
    resolver = FakeResolver()
    reranker = FakeReranker()
    expander = FakeExpander(facts=[_fact()])
    retriever = build_retriever(
        backends=backends, resolver=resolver, reranker=reranker, expander=expander
    )
    request = RetrievalRequest(query="q", seed_entity_ids=("e_1",))

    result = await retriever.retrieve(request)

    assert list(result.passages) == []
    assert list(result.graph_facts) == []
    assert result.request is request
    assert sorted(result.diagnostics.backends_failed) == ["graph", "keyword", "vector"]
    assert result.diagnostics.degraded is True
    assert result.diagnostics.fused_pool_size == 0
    # Nothing to resolve and nothing to rerank: the expensive stages must not be
    # entered just because they were configured.
    assert resolver.batches == []
    assert reranker.calls == []
    assert expander.fact_calls == []


@pytest.mark.asyncio
async def test_backends_returning_nothing_is_not_a_failure() -> None:
    """An empty answer is a hard query; a raised answer is an outage.

    Conflating them would make every zero-result query look like an incident and
    lower the confidence of a report that was correctly cautious.
    """
    backends = [FakeBackend(name) for name in (Backend.KEYWORD, Backend.VECTOR)]
    retriever = build_retriever(backends=backends, resolver=FakeResolver())

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert list(result.passages) == []
    assert result.diagnostics.backends_failed == ()
    assert result.diagnostics.degraded is False
    assert result.diagnostics.per_backend_counts == {"keyword": 0, "vector": 0}


@pytest.mark.asyncio
async def test_expansion_failure_costs_aliases_not_the_query() -> None:
    """Losing expansion narrows recall; it must not narrow the result set to zero."""
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, ["sig_a"]))
    resolver = FakeResolver([make_passage(chunk_id_for("sig_a", 0))])
    expander = FakeExpander(expand_fails=True)
    retriever = build_retriever(
        backends=[keyword], resolver=resolver, expander=expander
    )

    result = await retriever.retrieve(
        RetrievalRequest(query="DDOG", seed_entity_ids=("e_datadog",))
    )

    assert result.signal_ids == ["sig_a"]
    assert "expansion" in result.diagnostics.backends_failed
    assert result.diagnostics.degraded is True


@pytest.mark.asyncio
async def test_fact_lookup_failure_leaves_the_passages_intact() -> None:
    """Graph context is an enrichment. Losing it must not lose the evidence."""
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, ["sig_a"]))
    resolver = FakeResolver([make_passage(chunk_id_for("sig_a", 0))])
    expander = FakeExpander(facts_fail=True)
    retriever = build_retriever(
        backends=[keyword], resolver=resolver, expander=expander
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert result.signal_ids == ["sig_a"]
    assert list(result.graph_facts) == []
    assert "graph_facts" in result.diagnostics.backends_failed


@pytest.mark.asyncio
async def test_graph_facts_are_requested_for_the_surviving_signals() -> None:
    """Facts hang off the passages that survived, not off the query's seeds."""
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, ["sig_a", "sig_b"]))
    resolver = FakeResolver(
        [make_passage(chunk_id_for(s, 0)) for s in ("sig_a", "sig_b")]
    )
    fact = _fact()
    expander = FakeExpander(facts=[fact])
    retriever = build_retriever(
        backends=[keyword], resolver=resolver, expander=expander
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert expander.fact_calls == [["sig_a", "sig_b"]]
    assert list(result.graph_facts) == [fact]


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reranker_output_replaces_the_fused_order() -> None:
    """If the reranker ran, its opinion is the one that ships.

    The fake reverses the order precisely so this cannot pass by coincidence: a
    pipeline that discarded the reranker's return value would still emit a
    ranked list, just the fused one.
    """
    signals = ["sig_a", "sig_b", "sig_c"]
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, signals))
    resolver = FakeResolver([make_passage(chunk_id_for(s, 0)) for s in signals])
    reranker = FakeReranker()
    retriever = build_retriever(
        backends=[keyword], resolver=resolver, reranker=reranker
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert result.signal_ids == ["sig_c", "sig_b", "sig_a"]
    assert all(p.rerank_score is not None for p in result.passages)
    assert [p.final_score for p in result.passages] == [3.0, 2.0, 1.0]
    assert result.diagnostics.reranked > 0


@pytest.mark.asyncio
async def test_only_rerank_depth_passages_are_scored() -> None:
    """The cost control, asserted at the boundary where the cost is incurred.

    The cross-encoder reads query and passage together and is roughly two orders
    of magnitude more expensive per item than the arithmetic fusion it refines.
    Handing it the whole pool would let one query's tail dominate the latency
    budget for a gain that falls off sharply past the top few dozen.
    """
    signals = [f"sig_{i:02d}" for i in range(30)]
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, signals))
    resolver = FakeResolver([make_passage(chunk_id_for(s, 0)) for s in signals])
    reranker = FakeReranker()
    retriever = build_retriever(
        backends=[keyword], resolver=resolver, reranker=reranker
    )

    await retriever.retrieve(RetrievalRequest(query="q", rerank_depth=5, k_final=3))

    _, scored_ids, top_k = reranker.calls[0]
    assert len(scored_ids) == 5
    assert scored_ids == [chunk_id_for(s, 0) for s in signals[:5]]
    assert top_k == 3


@pytest.mark.asyncio
async def test_rerank_disabled_on_the_request_skips_the_reranker() -> None:
    """`rerank=False` is a latency choice the caller is allowed to make."""
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, ["sig_a", "sig_b"]))
    resolver = FakeResolver(
        [make_passage(chunk_id_for(s, 0)) for s in ("sig_a", "sig_b")]
    )
    reranker = FakeReranker()
    retriever = build_retriever(
        backends=[keyword], resolver=resolver, reranker=reranker
    )

    result = await retriever.retrieve(RetrievalRequest(query="q", rerank=False))

    assert reranker.calls == []
    assert result.signal_ids == ["sig_a", "sig_b"]
    assert result.diagnostics.reranked == 0


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_to_the_fused_order() -> None:
    """`docs/retrieval.md` §12: reranker unavailable -> fused RRF order.

    Whole-list fallback, not per-passage: a mixture of reranked and fused scores
    is not a ranking, because the two scales are unrelated and a passage's
    position would depend on which slice of the list it happened to be in. So
    every passage comes back with `rerank_score=None` and the fused order
    intact, and the run is marked degraded so the Critic lowers its confidence.
    """
    signals = ["sig_a", "sig_b", "sig_c"]
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, signals))
    resolver = FakeResolver([make_passage(chunk_id_for(s, 0)) for s in signals])
    reranker = FakeReranker(fails=True)
    retriever = build_retriever(
        backends=[keyword], resolver=resolver, reranker=reranker
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert result.signal_ids == signals
    assert all(p.rerank_score is None for p in result.passages)
    assert all(p.fused_score > 0 for p in result.passages)
    assert "rerank" in result.diagnostics.backends_failed
    assert result.diagnostics.degraded is True


# --------------------------------------------------------------------------- #
# Resolution gaps and near-duplicates
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_chunk_erased_from_storage_is_skipped_not_raised_on() -> None:
    """An erasure between indexing and querying must cost one passage, not the query.

    Qdrant and OpenSearch keep the chunk id after the PostgreSQL row is deleted
    for a GDPR erasure -- `docs/retrieval.md` §12 makes the PostgreSQL row
    authoritative and the drift a reindex trigger. Raising here would let one
    deleted record break every query that happened to match it, which is a
    denial of service authored by a single subject-access request.
    """
    signals = ["sig_live", "sig_erased", "sig_also_live"]
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, signals))
    vector = FakeBackend(Backend.VECTOR, make_candidates(Backend.VECTOR, ["sig_erased"]))
    resolver = FakeResolver(
        [make_passage(chunk_id_for(s, 0)) for s in ("sig_live", "sig_also_live")]
    )
    retriever = build_retriever(backends=[keyword, vector], resolver=resolver)

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert result.signal_ids == ["sig_live", "sig_also_live"]
    # The erased chunk was fused -- it was top of the vector list -- and only
    # disappeared at resolution, so the pool size still counts it. That gap
    # between `fused_pool_size` and the passages returned is the reindex signal.
    assert result.diagnostics.fused_pool_size == 3
    assert result.diagnostics.after_dedupe == 2
    assert chunk_id_for("sig_erased", 0) in resolver.batches[0]


@pytest.mark.asyncio
async def test_every_chunk_erased_leaves_an_empty_result_and_no_facts() -> None:
    """The degenerate case of the above: a whole result set that no longer exists."""
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, ["sig_a"]))
    expander = FakeExpander(facts=[_fact()])
    retriever = build_retriever(
        backends=[keyword], resolver=FakeResolver(), expander=expander
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert list(result.passages) == []
    assert list(result.graph_facts) == []
    assert expander.fact_calls == []
    # Nothing failed: the backends answered correctly about an index that is
    # ahead of storage, so this must not be reported as an outage.
    assert result.diagnostics.degraded is False


@pytest.mark.asyncio
async def test_near_duplicates_collapse_and_keep_the_corroboration_count() -> None:
    """Six outlets carrying one press release is evidence, not noise.

    The collapse itself belongs to `retrieval/rerank/fusion.py`; what is tested
    here is that the orchestrator runs it *after* fusion and *before* reranking,
    so the surviving passage carries three backends' worth of agreement and the
    cross-encoder budget is not spent scoring three copies of one story.
    """
    signals = ["sig_a", "sig_b", "sig_c"]
    keyword = FakeBackend(Backend.KEYWORD, make_candidates(Backend.KEYWORD, signals))
    resolver = FakeResolver(
        [
            make_passage(chunk_id_for("sig_a", 0), text="Acme acquires Globex"),
            make_passage(chunk_id_for("sig_b", 0), text="Acme acquires Globex"),
            make_passage(chunk_id_for("sig_c", 0), text="unrelated earnings note"),
        ]
    )
    reranker = FakeReranker()
    retriever = build_retriever(
        backends=[keyword],
        resolver=resolver,
        reranker=reranker,
        simhash_of=lambda p: hash(p.text) & ((1 << 64) - 1),
    )

    result = await retriever.retrieve(RetrievalRequest(query="q"))

    assert result.diagnostics.fused_pool_size == 3
    assert result.diagnostics.after_dedupe == 2
    survivor = next(p for p in result.passages if p.signal_id == "sig_a")
    assert survivor.duplicate_of_count == 1
    assert list(survivor.collapsed_signal_ids) == ["sig_b"]
    # The reranker sees the collapsed set, not the raw pool.
    assert len(reranker.calls[0][1]) == 2


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _fact() -> GraphFact:
    return GraphFact(
        subject_id="e_acme",
        subject_name="Acme Corp",
        predicate="COMPETES_WITH",
        object_id="e_globex",
        object_name="Globex",
        valid_from=datetime(2025, 3, 1, tzinfo=UTC),
        confidence=0.82,
        supporting_signal_ids=("sig_a",),
    )
