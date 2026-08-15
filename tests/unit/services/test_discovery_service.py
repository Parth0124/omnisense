"""Finding sources, and refusing to read from one nobody approved.

The premise is that you never tell the system where to look. The cost of that
premise is that discovery finds *everything* -- a real run against one account
surfaced 110 repositories, of which about six were current work and the rest were
coursework from 2023. So the tests here are mostly about the gate: nothing is
ingested until a person says so, and nothing they rejected is ever proposed
again.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from backend.core.exceptions import NotFoundError
from connectors.exceptions import QuotaError
from models.artifact import WatchStatus, source_id
from models.enums import Platform
from models.orm.artifact import SourceRow
from services.discovery_service import QUIET_AFTER_DAYS, Candidate, DiscoveryService

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


class FakeReader:
    """Yields fixed repository payloads, and can run out partway."""

    def __init__(self, repos: list[dict], fail_after: int | None = None) -> None:
        self._repos = repos
        self._fail_after = fail_after
        self.saw_forks_flag: bool | None = None

    async def viewer_repositories(self, *, max_pages: int = 10, include_forks: bool = False):
        self.saw_forks_flag = include_forks
        for index, repo in enumerate(self._repos):
            if self._fail_after is not None and index >= self._fail_after:
                raise QuotaError("rate limit exhausted")
            if repo.get("fork") and not include_forks:
                continue
            yield repo


def repo(name: str, node: str, **overrides) -> dict:
    payload = {
        "node_id": node,
        "full_name": name,
        "html_url": f"https://github.com/{name}",
        "default_branch": "main",
        "private": False,
        "archived": False,
        "fork": False,
        "pushed_at": "2026-08-01T00:00:00Z",
        "description": "a thing",
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


def service(factory, repos: list[dict], fail_after: int | None = None) -> DiscoveryService:
    return DiscoveryService(
        factory, reader=FakeReader(repos, fail_after=fail_after), tenant_id="local"
    )


class TestTheGate:
    """Nothing is read from a source nobody approved. The whole point."""

    async def test_everything_found_lands_pending(self, factory) -> None:
        await service(factory, [repo("me/a", "R_a"), repo("me/b", "R_b")]).discover_github()

        pending = await service(factory, []).candidates(WatchStatus.PENDING)
        assert {c.name for c in pending} == {"me/a", "me/b"}

    async def test_a_pending_source_is_not_offered_to_sync(self, factory) -> None:
        """`included_source_ids` is the only thing sync reads. If a pending source
        leaked into it, the review queue would be decorative."""
        await service(factory, [repo("me/a", "R_a")]).discover_github()

        assert await service(factory, []).included_source_ids() == []

    async def test_approving_one_makes_it_syncable(self, factory) -> None:
        await service(factory, [repo("me/a", "R_a")]).discover_github()
        instance = service(factory, [])

        await instance.decide(source="me/a", include=True)

        assert await instance.included_source_ids() == [source_id(Platform.GITHUB, "R_a")]

    async def test_a_skipped_source_is_never_syncable(self, factory) -> None:
        await service(factory, [repo("me/a", "R_a")]).discover_github()
        instance = service(factory, [])

        await instance.decide(source="me/a", include=False)

        assert await instance.included_source_ids() == []


class TestRejectionsStick:
    async def test_an_excluded_source_is_not_proposed_again(self, factory) -> None:
        """A person who rejects the same forked tutorial every week stops reading
        the list -- and then the one that mattered goes past unread too."""
        payload = [repo("me/junk", "R_junk")]
        await service(factory, payload).discover_github()
        await service(factory, []).decide(source="me/junk", include=False)

        report = await service(factory, payload).discover_github()

        assert report.new == 0
        assert report.previously_excluded == 1
        assert await service(factory, []).candidates(WatchStatus.PENDING) == []

    async def test_discovery_never_revises_a_decision(self, factory) -> None:
        payload = [repo("me/a", "R_a")]
        await service(factory, payload).discover_github()
        await service(factory, []).decide(source="me/a", include=True)

        await service(factory, payload).discover_github()

        assert await service(factory, []).included_source_ids() == [
            source_id(Platform.GITHUB, "R_a")
        ]


class TestCounting:
    async def test_waiting_and_decided_are_counted_apart(self, factory) -> None:
        """Merged, the number read "109 already decided" on a run where 107 had
        never been looked at -- an encouraging sentence describing an untouched
        backlog."""
        payload = [repo("me/a", "R_a"), repo("me/b", "R_b"), repo("me/c", "R_c")]
        await service(factory, payload).discover_github()
        await service(factory, []).decide(source="me/a", include=True)

        report = await service(factory, payload).discover_github()

        assert report.new == 0
        assert report.still_pending == 2
        assert report.already_decided == 1

    async def test_a_first_run_reports_everything_as_new(self, factory) -> None:
        report = await service(
            factory, [repo("me/a", "R_a"), repo("me/b", "R_b")]
        ).discover_github()

        assert (report.found, report.new) == (2, 2)


class TestForks:
    async def test_forks_are_skipped_by_default(self, factory) -> None:
        """A fork is usually somebody else's project cloned to send one patch, and
        ingesting its history files their work under your name."""
        await service(
            factory, [repo("me/mine", "R_1"), repo("them/theirs", "R_2", fork=True)]
        ).discover_github()

        names = {c.name for c in await service(factory, []).candidates()}
        assert names == {"me/mine"}

    async def test_forks_can_be_asked_for(self, factory) -> None:
        await service(factory, [repo("them/theirs", "R_2", fork=True)]).discover_github(
            include_forks=True
        )

        assert len(await service(factory, []).candidates()) == 1


class TestPartialFailure:
    async def test_what_was_found_before_the_wall_is_kept(self, factory) -> None:
        """Discovery that found sixty repositories and then hit the rate limit has
        done sixty repositories of useful work. Throwing it away means starting
        over against the same wall."""
        payload = [repo(f"me/r{i}", f"R_{i}") for i in range(5)]

        report = await service(factory, payload, fail_after=3).discover_github()

        assert report.error is not None
        assert report.new == 3
        assert len(await service(factory, []).candidates()) == 3


class TestRenames:
    async def test_a_renamed_repository_keeps_its_row_and_updates_its_name(self, factory) -> None:
        """Keyed on the node id, so a rename must follow rather than fork the
        history in two."""
        await service(factory, [repo("me/old-name", "R_a")]).discover_github()
        await service(factory, []).decide(source="me/old-name", include=True)

        await service(factory, [repo("me/new-name", "R_a")]).discover_github()

        candidates = await service(factory, []).candidates()
        assert [c.name for c in candidates] == ["me/new-name"]
        assert candidates[0].status is WatchStatus.INCLUDED


class TestReviewQueue:
    async def test_the_queue_is_newest_first(self, factory) -> None:
        await service(
            factory,
            [
                repo("me/old", "R_old", pushed_at="2024-01-01T00:00:00Z"),
                repo("me/new", "R_new", pushed_at="2026-08-01T00:00:00Z"),
            ],
        ).discover_github()

        assert [c.name for c in await service(factory, []).candidates()] == [
            "me/new",
            "me/old",
        ]

    async def test_undated_sources_sort_last_not_first(self, factory) -> None:
        """`None` at the top would put the least informative rows where the eye
        lands."""
        await service(
            factory,
            [
                repo("me/dated", "R_1", pushed_at="2026-08-01T00:00:00Z"),
                repo("me/undated", "R_2", pushed_at=None),
            ],
        ).discover_github()

        assert [c.name for c in await service(factory, []).candidates()][-1] == "me/undated"

    async def test_flags_carry_through(self, factory) -> None:
        """The three columns somebody skims a queue of a hundred by."""
        await service(factory, [repo("me/a", "R_a", private=True, archived=True)]).discover_github()

        candidate = (await service(factory, []).candidates())[0]
        assert candidate.private
        assert candidate.archived

    def test_dormancy_is_measured_not_guessed(self) -> None:
        fresh = Candidate(
            source_id="s",
            platform=Platform.GITHUB,
            name="me/a",
            status=WatchStatus.PENDING,
            last_activity=datetime.now(UTC) - timedelta(days=QUIET_AFTER_DAYS - 5),
        )
        stale = Candidate(
            source_id="s",
            platform=Platform.GITHUB,
            name="me/b",
            status=WatchStatus.PENDING,
            last_activity=datetime.now(UTC) - timedelta(days=QUIET_AFTER_DAYS + 5),
        )
        assert not fresh.is_dormant
        assert stale.is_dormant

    def test_a_source_that_never_moved_counts_as_dormant(self) -> None:
        never = Candidate(
            source_id="s", platform=Platform.GITHUB, name="me/c", status=WatchStatus.PENDING
        )
        assert never.is_dormant


class TestBulk:
    async def test_everything_pending_can_be_settled_at_once(self, factory) -> None:
        """A first run on a real account surfaced 110 repositories. Sixty commands
        to say "yes, all of them" is how somebody abandons the review."""
        await service(factory, [repo(f"me/r{i}", f"R_{i}") for i in range(5)]).discover_github()

        assert await service(factory, []).decide_all_pending(include=True) == 5
        assert len(await service(factory, []).included_source_ids()) == 5

    async def test_bulk_does_not_touch_what_was_already_decided(self, factory) -> None:
        await service(factory, [repo("me/a", "R_a"), repo("me/b", "R_b")]).discover_github()
        instance = service(factory, [])
        await instance.decide(source="me/a", include=False)

        await instance.decide_all_pending(include=True)

        assert await instance.included_source_ids() == [source_id(Platform.GITHUB, "R_b")]


class TestResolving:
    async def test_a_full_name_resolves(self, factory) -> None:
        await service(factory, [repo("me/a", "R_a")]).discover_github()
        assert (await service(factory, []).decide(source="me/a", include=True)).name == "me/a"

    async def test_an_id_prefix_resolves(self, factory) -> None:
        await service(factory, [repo("me/a", "R_a")]).discover_github()
        prefix = source_id(Platform.GITHUB, "R_a")[:12]

        assert (await service(factory, []).decide(source=prefix, include=True)).name == "me/a"

    async def test_an_unknown_reference_is_reported(self, factory) -> None:
        with pytest.raises(NotFoundError):
            await service(factory, []).decide(source="me/nothing", include=True)

    async def test_an_empty_reference_matches_nothing(self, factory) -> None:
        await service(factory, [repo("me/a", "R_a")]).discover_github()
        with pytest.raises(NotFoundError):
            await service(factory, []).decide(source="   ", include=True)


class TestExistingSources:
    async def test_a_hand_added_source_is_included_without_review(self, factory) -> None:
        """Somebody who typed `add-repo owner/name` already made the decision the
        queue exists to collect. Asking again would be a worse interface, not a
        safer one."""
        async with factory() as session:
            session.add(
                SourceRow(
                    id="src_manual",
                    tenant_id="local",
                    platform=Platform.GITHUB,
                    external_id="R_manual",
                    name="me/manual",
                )
            )
            await session.commit()

        assert "src_manual" in await service(factory, []).included_source_ids()
