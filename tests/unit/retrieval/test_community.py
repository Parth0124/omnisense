"""Unit tests for `retrieval/graphrag/community.py`.

No model is called. `SummaryWriter` is a one-argument Protocol precisely so the
batching, caching and refusal logic can be exercised against a function that
returns a fixed string.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from graph.analytics.centrality import Projection
from graph.analytics.communities import Community, CommunityResult, louvain
from retrieval.graphrag.community import (
    MAX_SUMMARY_CHARS,
    CommunityMember,
    CommunitySummarizer,
    SummaryRequest,
    members_from_rows,
    render_community_prompt,
    select_representatives,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 6, tzinfo=UTC)


def _community(size: int = 4, *, conductance_external: float = 1.0) -> Community:
    members = tuple(f"e{index}" for index in range(size))
    return Community(
        community_id="com_test",
        members=members,
        internal_weight=10.0,
        external_weight=conductance_external,
    )


def _members(size: int = 4, **overrides: object) -> dict[str, CommunityMember]:
    return {
        f"e{index}": CommunityMember(
            entity_id=f"e{index}",
            name=f"Company {index}",
            entity_type="Company",
            description=f"description {index}",
            importance=float(size - index),
            signal_ids=(f"sig_{index}", "sig_shared"),
            **overrides,  # type: ignore[arg-type]
        )
        for index in range(size)
    }


def _writer(response: str = "TITLE: Battery supply\nSUMMARY: These firms share suppliers."):
    calls: list[str] = []

    async def write(prompt: str) -> str:
        calls.append(prompt)
        return response

    write.calls = calls  # type: ignore[attr-defined]
    return write


class TestRepresentativeSelection:
    def test_picks_the_most_important(self) -> None:
        chosen = select_representatives(list(_members(6).values()), limit=2)
        assert [member.entity_id for member in chosen] == ["e0", "e1"]

    def test_is_stable_when_importance_ties(self) -> None:
        """PageRank ties constantly on small graphs. An unstable sort would send
        a different subset to the model every run and produce a different summary
        for a cluster that did not change."""
        members = [
            CommunityMember(entity_id=f"e{i}", name=f"n{i}", importance=1.0) for i in range(5)
        ]
        first = select_representatives(members, limit=3)
        second = select_representatives(list(reversed(members)), limit=3)
        assert [m.entity_id for m in first] == [m.entity_id for m in second]


class TestPrompt:
    def test_lists_the_representatives(self) -> None:
        members = tuple(_members(3).values())
        request = SummaryRequest(_community(3), members, members)
        prompt = render_community_prompt(request)
        assert "Company 0" in prompt
        assert "description 0" in prompt

    def test_states_the_true_size_when_members_are_omitted(self) -> None:
        """A model told about 25 of 60 members writes 'this cluster of 25
        companies', and the number lands in a report."""
        members = tuple(_members(10).values())
        request = SummaryRequest(_community(10), members, members[:3])
        prompt = render_community_prompt(request)
        assert "contains 10 entities" in prompt
        assert "7 less-connected members are omitted" in prompt

    def test_does_not_claim_omissions_when_there_are_none(self) -> None:
        members = tuple(_members(3).values())
        prompt = render_community_prompt(SummaryRequest(_community(3), members, members))
        assert "omitted" not in prompt

    def test_permits_the_model_to_report_no_theme(self) -> None:
        """Without this instruction a model always finds a theme, because that is
        what it was asked for."""
        members = tuple(_members(3).values())
        prompt = render_community_prompt(SummaryRequest(_community(3), members, members))
        assert "no clear common theme" in prompt


class TestSummarization:
    async def test_writes_a_summary_with_citations(self) -> None:
        summarizer = CommunitySummarizer(_writer())
        summary = await summarizer.summarize(_community(), _members(), now=NOW)
        assert summary.is_written
        assert summary.title == "Battery supply"
        assert "share suppliers" in summary.summary
        assert summary.entity_ids == ("e0", "e1", "e2", "e3")

    async def test_signal_ids_are_deduplicated(self) -> None:
        """Every member carries `sig_shared`; a citation list repeating it four
        times reads as four independent sources."""
        summary = await CommunitySummarizer(_writer()).summarize(_community(), _members())
        assert summary.signal_ids.count("sig_shared") == 1

    async def test_a_diffuse_group_is_refused(self) -> None:
        """Conductance above the floor means most connections point outside the
        group. A model asked to find a theme will produce one anyway, phrased
        confidently."""
        diffuse = _community(conductance_external=100.0)
        summary = await CommunitySummarizer(_writer()).summarize(diffuse, _members())
        assert not summary.is_written
        assert "conductance" in (summary.skipped_reason or "")

    async def test_missing_metadata_is_distinguished_from_a_weak_cluster(self) -> None:
        """The two look identical in a log otherwise, and one is a wiring bug."""
        summary = await CommunitySummarizer(_writer()).summarize(_community(), {})
        assert "no member metadata" in (summary.skipped_reason or "")

    async def test_a_failing_writer_does_not_raise(self) -> None:
        async def failing(prompt: str) -> str:
            raise RuntimeError("provider down")

        summary = await CommunitySummarizer(failing).summarize(_community(), _members())
        assert not summary.is_written
        assert "provider down" in (summary.skipped_reason or "")

    async def test_an_empty_response_is_a_skip_not_an_empty_summary(self) -> None:
        summary = await CommunitySummarizer(_writer("   ")).summarize(_community(), _members())
        assert not summary.is_written

    async def test_summary_is_truncated_to_the_budget(self) -> None:
        """A summary that displaces the passages it was meant to frame makes the
        model answer from a paraphrase of a paraphrase."""
        long_response = "TITLE: T\nSUMMARY: " + "x" * (MAX_SUMMARY_CHARS * 2)
        summary = await CommunitySummarizer(_writer(long_response)).summarize(
            _community(), _members()
        )
        assert len(summary.summary) <= MAX_SUMMARY_CHARS

    async def test_unlabelled_response_falls_back_to_first_line_as_title(self) -> None:
        """A formatting slip must not lose the batch."""
        summary = await CommunitySummarizer(
            _writer("Battery cluster\nThey share suppliers.")
        ).summarize(_community(), _members())
        assert summary.title == "Battery cluster"
        assert "share suppliers" in summary.summary


class TestCaching:
    async def test_an_unchanged_community_is_not_re_summarised(self) -> None:
        """Content-addressed ids are what make this safe: a cluster that gained a
        member gets a new id, so the cache cannot serve last week's summary for
        a group that now contains different companies."""
        writer = _writer()
        summarizer = CommunitySummarizer(writer)
        await summarizer.summarize(_community(), _members())
        await summarizer.summarize(_community(), _members())
        assert len(writer.calls) == 1  # type: ignore[attr-defined]

    async def test_a_skip_is_not_cached(self) -> None:
        """A skip is usually caused by something outside the community. Caching
        it would mean the cluster is never summarised again for the process
        lifetime, long after the cause was fixed."""
        writer = _writer()
        summarizer = CommunitySummarizer(writer)
        await summarizer.summarize(_community(), {})
        summary = await summarizer.summarize(_community(), _members())
        assert summary.is_written

    async def test_priming_admits_only_written_summaries(self) -> None:
        summarizer = CommunitySummarizer(_writer())
        skipped = await summarizer.summarize(_community(), {})
        summarizer.prime([skipped])
        assert (await summarizer.summarize(_community(), _members())).is_written


class TestBatch:
    async def test_summarises_every_community(self) -> None:
        projection = Projection()
        names = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)]
        for group in (names[:5], names[5:]):
            for i, left in enumerate(group):
                for right in group[i + 1 :]:
                    projection.add_edge(left, right, 1.0)
        projection.add_edge("a0", "b0", 1.0)
        result = louvain(projection)

        members = {
            name: CommunityMember(entity_id=name, name=name.upper(), importance=1.0)
            for name in names
        }
        summaries = await CommunitySummarizer(_writer()).summarize_all(result, members)
        assert len(summaries) == len(result.communities) == 2
        assert all(summary.is_written for summary in summaries)

    async def test_one_failure_does_not_lose_the_batch(self) -> None:
        state = {"calls": 0}

        async def flaky(prompt: str) -> str:
            state["calls"] += 1
            if state["calls"] == 1:
                raise RuntimeError("rate limited")
            return "TITLE: T\nSUMMARY: body"

        result = CommunityResult(
            communities=(
                Community("com_a", ("e0", "e1", "e2"), 5.0, 1.0),
                Community("com_b", ("e3",), 5.0, 1.0),
            ),
            unassigned=(),
            modularity=0.4,
            passes=1,
        )
        members = _members(4)
        summaries = await CommunitySummarizer(flaky, max_concurrency=1).summarize_all(
            result, members
        )
        assert len(summaries) == 2
        assert sum(1 for s in summaries if s.is_written) == 1

    async def test_concurrency_is_bounded(self) -> None:
        """Two hundred simultaneous requests get rate-limited and retry into a
        longer wall-clock time than the bounded version would have taken."""
        active = 0
        peak = 0

        async def slow(prompt: str) -> str:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return "TITLE: T\nSUMMARY: body"

        result = CommunityResult(
            communities=tuple(
                Community(f"com_{i}", (f"e{i % 4}",), 5.0, 1.0) for i in range(10)
            ),
            unassigned=(),
            modularity=0.4,
            passes=1,
        )
        await CommunitySummarizer(slow, max_concurrency=2).summarize_all(result, _members())
        assert peak <= 2


class TestRetrievableForm:
    async def test_includes_entity_names_for_lexical_matching(self) -> None:
        """A user asking about lithium suppliers will not match a summary that
        discusses the topic without naming the companies."""
        summary = await CommunitySummarizer(_writer()).summarize(_community(), _members())
        text = summary.as_retrievable_text()
        assert "Company 0" in text
        assert "Battery supply" in text

    async def test_a_skipped_community_has_no_retrievable_text(self) -> None:
        summary = await CommunitySummarizer(_writer()).summarize(_community(), {})
        assert summary.as_retrievable_text() == ""


class TestMembersFromRows:
    def test_builds_the_join_index(self) -> None:
        members = members_from_rows(
            [{"id": "e1", "name": "Acme", "type": "Company", "pagerank_score": 0.4}]
        )
        assert members["e1"].name == "Acme"
        assert members["e1"].importance == 0.4

    def test_rows_without_an_id_are_skipped(self) -> None:
        assert members_from_rows([{"name": "no id"}]) == {}

    def test_missing_importance_defaults_to_zero(self) -> None:
        assert members_from_rows([{"id": "e1"}])["e1"].importance == 0.0

    def test_falls_back_to_the_id_when_unnamed(self) -> None:
        assert members_from_rows([{"id": "e1"}])["e1"].name == "e1"
