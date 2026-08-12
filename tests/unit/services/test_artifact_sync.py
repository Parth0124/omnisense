"""Storing artifacts, and deciding what to ask for next time.

The watermark tests are the important ones. A sync that advances its watermark
after an interrupted pass leaves a permanent hole in the history at exactly the
point it stopped -- and nothing ever reports it, because from then on the missing
range is simply never requested again. That failure cannot be found by looking at
the data, which is why it is pinned here instead.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from connectors.exceptions import AuthError, PermanentError, QuotaError
from models.artifact import (
    Artifact,
    ArtifactKind,
    ArtifactProvenance,
    ArtifactState,
    Person,
    artifact_id,
    person_id,
    source_id,
)
from models.enums import Platform
from models.orm.artifact import ArtifactRow, PersonRow, SourceRow
from services.artifact_store import ArtifactStore
from services.artifact_sync import ArtifactSync, SyncReport

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)

NAIVE_NOW = NOW.replace(tzinfo=None)
"""What SQLite gives back for `NOW`.

SQLite has no timezone-aware type, so a `DateTime(timezone=True)` column returns
what it was given with the offset dropped. The same convention as
`tests/unit/models/test_orm.py`; PostgreSQL preserves it, and the distinction is
the storage engine's rather than the mapping's."""
SRC = source_id(Platform.GITHUB, "R_1")


@pytest.fixture
def factory(orm_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=orm_engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
    )


@pytest.fixture
async def source(factory) -> str:
    async with factory() as session:
        session.add(
            SourceRow(
                id=SRC,
                tenant_id="local",
                platform=Platform.GITHUB,
                external_id="R_1",
                name="omnisense/api",
            )
        )
        await session.commit()
    return SRC


def person(external: str = "U_1") -> Person:
    return Person(
        id=person_id(Platform.GITHUB, external),
        tenant_id="local",
        platform=Platform.GITHUB,
        external_id=external,
        handle="dsokolov",
    )


def artifact(native: str, *, actor: str | None = None, **overrides) -> Artifact:
    fields = {
        "id": artifact_id(Platform.GITHUB, native),
        "tenant_id": "local",
        "kind": ArtifactKind.COMMIT,
        "source_id": SRC,
        "actor_id": actor,
        "platform": Platform.GITHUB,
        "native_id": native,
        "occurred_at": NOW,
        "provenance": ArtifactProvenance(connector_slug="github", fetched_at=NOW),
    }
    fields.update(overrides)
    return Artifact(**fields)


class TestStore:
    async def test_writing_the_same_artifact_twice_updates_rather_than_duplicates(
        self, factory, source
    ) -> None:
        """What makes a sync safe to run twice -- and safe to resume after the rate
        limit stops it halfway, which is the normal outcome on a real repository."""
        store = ArtifactStore(factory)
        await store.write([artifact("c1", title="first")])
        await store.write([artifact("c1", title="second")])

        async with factory() as session:
            count = (
                await session.execute(select(func.count()).select_from(ArtifactRow))
            ).scalar_one()
            stored = await session.get(ArtifactRow, artifact_id(Platform.GITHUB, "c1"))

        assert count == 1
        assert stored is not None
        assert stored.title == "second"

    async def test_a_changing_pull_request_has_its_state_refreshed(self, factory, source) -> None:
        """Most of these genuinely change: a pull request opens, is reviewed, gains
        commits and merges, and every sync sees a different state for one row."""
        store = ArtifactStore(factory)
        await store.write(
            [artifact("pr1", kind=ArtifactKind.PULL_REQUEST, state=ArtifactState.OPEN)]
        )
        await store.write(
            [artifact("pr1", kind=ArtifactKind.PULL_REQUEST, state=ArtifactState.MERGED)]
        )

        async with factory() as session:
            stored = await session.get(ArtifactRow, artifact_id(Platform.GITHUB, "pr1"))
        assert stored is not None
        assert stored.state is ArtifactState.MERGED

    async def test_occurred_at_is_never_moved_by_a_later_write(self, factory, source) -> None:
        """When a thing happened does not change. Letting a re-read move it would
        let one bad payload silently relocate history."""
        store = ArtifactStore(factory)
        await store.write([artifact("c1", occurred_at=NOW)])
        await store.write([artifact("c1", occurred_at=NOW + timedelta(days=30))])

        async with factory() as session:
            stored = await session.get(ArtifactRow, artifact_id(Platform.GITHUB, "c1"))
        assert stored is not None
        assert stored.occurred_at == NAIVE_NOW

    async def test_one_author_on_many_commits_does_not_break_the_batch(
        self, factory, source
    ) -> None:
        """PostgreSQL refuses an ON CONFLICT statement that touches the same row
        twice -- so a batch of a hundred commits by one author would fail outright
        without deduplicating the people first."""
        store = ArtifactStore(factory)
        author = person()
        result = await store.write(
            [artifact(f"c{i}", actor=author.id) for i in range(50)],
            [author] * 50,
        )

        assert result.people_written == 1
        assert result.artifacts_written == 50

    async def test_a_persons_details_are_refreshed_not_frozen(self, factory, source) -> None:
        """A display name that changed since first sighting should follow. `DO
        NOTHING` would freeze it at whatever it was the first time."""
        store = ArtifactStore(factory)
        await store.write([], [person()])
        renamed = person().model_copy(update={"handle": "dmitri"})
        await store.write([], [renamed])

        async with factory() as session:
            stored = await session.get(PersonRow, person().id)
        assert stored is not None
        assert stored.handle == "dmitri"

    async def test_people_are_written_before_the_artifacts_that_reference_them(
        self, factory, source
    ) -> None:
        """The foreign key makes the order load-bearing: artifacts first fails."""
        store = ArtifactStore(factory)
        author = person()
        await store.write([artifact("c1", actor=author.id)], [author])

        async with factory() as session:
            stored = await session.get(ArtifactRow, artifact_id(Platform.GITHUB, "c1"))
        assert stored is not None
        assert stored.actor_id == author.id

    async def test_an_empty_write_is_not_an_error(self, factory, source) -> None:
        """A quiet repository legitimately produces nothing, and a sync should not
        have to check before calling."""
        assert (await ArtifactStore(factory).write([], [])).artifacts_written == 0


class FakeReader:
    """A reader that yields fixed payloads and can be told to run out."""

    def __init__(self, commits=(), pulls=(), runs=(), reviews=()) -> None:
        self._commits = list(commits)
        self._pulls = list(pulls)
        self._runs = list(runs)
        self._reviews = list(reviews)
        self.reviewed: list[int] = []
        """Which pull requests were asked for reviews, in order.

        The reviews stream is the only one that costs a request per item, so
        *which* items it picked and in what order is the whole behaviour."""
        self.rate_limit = type("R", (), {"describe": lambda self: "fake"})()

    async def commits(self, owner, repo, *, since=None, max_pages=100):
        for item in self._commits:
            yield item

    async def pull_requests(self, owner, repo, *, since=None, max_pages=100):
        for item in self._pulls:
            yield item

    async def reviews(self, owner, repo, number):
        self.reviewed.append(number)
        for item in self._reviews:
            # `node_id` varied, not `id`: identity is the node id everywhere in
            # this connector, so leaving it fixed makes every review the same
            # artifact and the upsert collapses them into one.
            yield {**item, "node_id": f"{item.get('node_id', 'PRR')}_{number}"}

    async def workflow_runs(self, owner, repo, *, since=None, max_pages=50):
        for item in self._runs:
            yield item

    async def run_jobs(self, owner, repo, run_id):
        return []


def commit(node: str, at: str) -> dict:
    return {
        "sha": node,
        "node_id": node,
        "commit": {
            "message": f"work {node}",
            "author": {"name": "a", "date": at},
            "committer": {"name": "a", "date": at},
        },
        "author": {"login": "a", "node_id": "U_1", "type": "User"},
        "parents": [],
    }


def pull(number: int, updated: str) -> dict:
    """A pull request, newest-updated first as GitHub returns them."""
    return {
        "number": number,
        "node_id": f"PR_{number}",
        "title": f"pull {number}",
        "state": "open",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": updated,
        "user": {"login": "a", "node_id": "U_1", "type": "User"},
        "base": {"ref": "main"},
        "head": {"ref": f"feature-{number}"},
    }


def review(verdict: str = "APPROVED") -> dict:
    return {
        "id": "rv",
        "node_id": "PRR_1",
        "state": verdict,
        "submitted_at": "2026-08-10T10:00:00Z",
        "user": {"login": "b", "node_id": "U_2", "type": "User"},
        "body": "looks right",
    }


class TestReviewFanout:
    """The one stream that costs a request *per item* rather than per page.

    A first pass over `pallets/click` found 1,639 pull requests touched inside the
    ninety-day window. Unbounded, that is 1,639 requests for one stream of one
    repository -- a third of an authenticated hourly budget. Bounded carelessly,
    it never finishes. Both failures are pinned here.
    """

    @staticmethod
    def _pulls(count: int) -> list[dict]:
        # Newest-updated first, as the endpoint returns them.
        return [pull(n, f"2026-08-{10 - (n - 1) % 28:02d}T10:00:00Z") for n in range(1, count + 1)]

    async def test_the_fanout_is_bounded(self, factory, source, monkeypatch) -> None:
        """Without a bound, a busy repository's first pass spends the whole budget
        on this one stream before any other has run."""
        monkeypatch.setattr("services.artifact_sync.REVIEW_FANOUT_LIMIT", 5)
        reader = FakeReader(pulls=self._pulls(40), reviews=[review()])

        report = await ArtifactSync(factory, reader=reader).sync_source(SRC, streams=["reviews"])

        assert len(reader.reviewed) == 5
        assert report.streams[0].complete is False, "a capped pass is not a complete one"

    async def test_the_chunk_is_taken_oldest_first_so_successive_runs_converge(
        self, factory, source, monkeypatch
    ) -> None:
        """The property the whole design turns on.

        The pull-request endpoint returns *newest* first. Taking the newest `n`
        and stopping would re-read that same newest `n` on every run -- the
        watermark could never move past them, and the older ones would never be
        reached at all. Reversed, each run eats the oldest outstanding chunk and
        the watermark walks forward.
        """
        monkeypatch.setattr("services.artifact_sync.REVIEW_FANOUT_LIMIT", 3)
        pulls = [
            pull(1, "2026-08-09T10:00:00Z"),
            pull(2, "2026-08-08T10:00:00Z"),
            pull(3, "2026-08-07T10:00:00Z"),
            pull(4, "2026-08-06T10:00:00Z"),
            pull(5, "2026-08-05T10:00:00Z"),
        ]
        reader = FakeReader(pulls=pulls, reviews=[review()])
        sync = ArtifactSync(factory, reader=reader)

        await sync.sync_source(SRC, streams=["reviews"])

        assert reader.reviewed == [5, 4, 3], "oldest first, or it never converges"

        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
        cursor = datetime.fromisoformat(stored.source_metadata["cursors"]["reviews"])
        # Advanced past the chunk it finished -- which is what lets the next run
        # pick up 2 and 1 instead of repeating 5, 4 and 3 forever.
        assert cursor > datetime(2026, 8, 6, tzinfo=UTC)

    async def test_a_truncated_pull_request_walk_leaves_the_watermark_alone(
        self, factory, source
    ) -> None:
        """The page bound cuts the *tail* of a newest-first list -- so a truncated
        walk is missing exactly the oldest pull requests, which are the ones this
        stream would otherwise mark as done."""
        reader = FakeReader(pulls=self._pulls(100), reviews=[review()])

        report = await ArtifactSync(factory, reader=reader).sync_source(
            SRC, streams=["reviews"], max_pages=1
        )

        assert report.streams[0].complete is False
        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
        assert "reviews" not in (stored.source_metadata or {}).get("cursors", {})

    async def test_a_pass_within_the_bound_is_complete(self, factory, source) -> None:
        reader = FakeReader(pulls=self._pulls(3), reviews=[review()])

        report = await ArtifactSync(factory, reader=reader).sync_source(SRC, streams=["reviews"])

        assert report.streams[0].complete is True
        assert report.streams[0].written == 3


class TestWatermarks:
    async def test_a_clean_pass_advances_the_watermark(self, factory, source) -> None:
        reader = FakeReader(commits=[commit("c1", "2026-08-10T10:00:00Z")])
        sync = ArtifactSync(factory, reader=reader)

        await sync.sync_source(SRC, streams=["commits"])

        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
        assert stored is not None
        assert "commits" in (stored.source_metadata or {}).get("cursors", {})

    async def test_an_interrupted_pass_leaves_the_watermark_alone(self, factory, source) -> None:
        """The most important test in this file.

        Advancing after a partial read leaves a permanent hole at exactly the
        point the sync stopped: the missing range is never requested again, and
        nothing reports it -- the data simply is not there and never will be.
        Re-reading instead is free, because every write is an upsert.
        """
        # More items than the page bound allows, so the pass is incomplete.
        reader = FakeReader(commits=[commit(f"c{i}", "2026-08-10T10:00:00Z") for i in range(100)])
        sync = ArtifactSync(factory, reader=reader)

        report = await sync.sync_source(SRC, streams=["commits"], max_pages=1)

        assert report.streams[0].complete is False
        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
        assert stored is not None
        assert (stored.source_metadata or {}).get("cursors", {}) == {}

    async def test_the_watermark_is_set_slightly_behind_the_newest_item(
        self, factory, source
    ) -> None:
        """GitHub timestamps have second resolution, so two commits can share one.
        Advancing to exactly the newest drops whichever fell on the far side of a
        page boundary."""
        reader = FakeReader(commits=[commit("c1", "2026-08-10T10:00:00Z")])
        sync = ArtifactSync(factory, reader=reader)
        await sync.sync_source(SRC, streams=["commits"])

        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
        cursor = datetime.fromisoformat(stored.source_metadata["cursors"]["commits"])
        assert cursor < datetime(2026, 8, 10, 10, 0, tzinfo=UTC)

    async def test_each_stream_advances_independently(self, factory, source) -> None:
        """Commits arrive constantly and pull requests do not. One watermark for
        the repository would either re-read the quiet streams or skip the busy
        ones."""
        reader = FakeReader(commits=[commit("c1", "2026-08-10T10:00:00Z")])
        sync = ArtifactSync(factory, reader=reader)
        await sync.sync_source(SRC, streams=["commits", "ci_runs"])

        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
        cursors = stored.source_metadata["cursors"]
        assert "commits" in cursors
        assert "ci_runs" not in cursors, "an empty stream must not claim progress"

    async def test_an_unreadable_cursor_starts_over_rather_than_crashing(
        self, factory, source
    ) -> None:
        """Worse than no cursor would be one silently treated as "now", which
        would skip everything forever."""
        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
            stored.source_metadata = {"cursors": {"commits": "not-a-timestamp"}}
            await session.commit()

        reader = FakeReader(commits=[commit("c1", "2026-08-10T10:00:00Z")])
        report = await ArtifactSync(factory, reader=reader).sync_source(SRC, streams=["commits"])
        assert report.streams[0].written == 1


class TestBackfillWindow:
    """`--days` versus the watermark, and which one wins.

    The watermark wins, on purpose -- otherwise a nightly sync re-reads a year
    every night. The cost is that widening the window later does nothing at all,
    silently, which is exactly the sort of thing somebody discovers a month after
    they thought they had the data.
    """

    async def test_a_first_sync_reaches_back_the_requested_window(self, factory, source) -> None:
        seen: list[datetime] = []

        class Recording(FakeReader):
            async def commits(self, owner, repo, *, since=None, max_pages=100):
                seen.append(since)
                return
                yield  # pragma: no cover -- makes this an async generator

        await ArtifactSync(factory, reader=Recording()).sync_source(
            SRC, streams=["commits"], backfill_days=730
        )

        assert seen[0] is not None
        reach = (datetime.now(UTC) - seen[0]).days
        assert 729 <= reach <= 730

    async def test_an_existing_watermark_outranks_a_wider_window(self, factory, source) -> None:
        """Asking for two years when the cursor sits at yesterday reads from
        yesterday. Not a bug -- but invisible, which is why `--reset` exists."""
        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
            stored.source_metadata = {"cursors": {"commits": "2026-08-01T00:00:00+00:00"}}
            await session.commit()

        seen: list[datetime] = []

        class Recording(FakeReader):
            async def commits(self, owner, repo, *, since=None, max_pages=100):
                seen.append(since)
                return
                yield  # pragma: no cover

        await ArtifactSync(factory, reader=Recording()).sync_source(
            SRC, streams=["commits"], backfill_days=730
        )

        assert seen[0] == datetime(2026, 8, 1, tzinfo=UTC), "the 730 must be ignored"

    async def test_resetting_the_cursor_makes_the_wider_window_apply(self, factory, source) -> None:
        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
            stored.source_metadata = {"cursors": {"commits": "2026-08-01T00:00:00+00:00"}}
            await session.commit()

        sync = ArtifactSync(factory, reader=FakeReader())
        cleared = await sync.reset_cursors(SRC, ["commits"])

        assert cleared == ["commits"]
        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
        assert (stored.source_metadata or {}).get("cursors", {}) == {}

    async def test_resetting_one_stream_leaves_the_others_where_they_were(
        self, factory, source
    ) -> None:
        """`--reset --stream commits` must not silently re-read pull requests too --
        that is the expensive stream, and the surprise would be a spent budget."""
        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
            stored.source_metadata = {
                "cursors": {
                    "commits": "2026-08-01T00:00:00+00:00",
                    "pull_requests": "2026-08-05T00:00:00+00:00",
                }
            }
            await session.commit()

        await ArtifactSync(factory, reader=FakeReader()).reset_cursors(SRC, ["commits"])

        async with factory() as session:
            stored = await session.get(SourceRow, SRC)
        cursors = stored.source_metadata["cursors"]
        assert "commits" not in cursors
        assert cursors["pull_requests"] == "2026-08-05T00:00:00+00:00"

    async def test_resetting_a_stream_that_never_ran_is_not_an_error(self, factory, source) -> None:
        assert await ArtifactSync(factory, reader=FakeReader()).reset_cursors(SRC) == []


class TestStopping:
    """A stopped run has to say *why*, because the two reasons want opposite acts."""

    async def _sync_raising(self, factory, error: Exception) -> SyncReport:
        class Failing(FakeReader):
            async def commits(self, owner, repo, *, since=None, max_pages=100):
                raise error
                yield  # pragma: no cover -- makes this an async generator

        return await ArtifactSync(factory, reader=Failing()).sync_source(
            SRC, streams=["commits", "pull_requests"]
        )

    async def test_a_rate_limit_stops_the_whole_run_not_just_one_stream(
        self, factory, source
    ) -> None:
        """One budget covers all four streams, so continuing would spend three more
        requests learning the same thing three more times."""
        report = await self._sync_raising(factory, QuotaError("exhausted"))

        assert report.stop_reason == "quota"
        assert [s.stream for s in report.streams] == ["commits"]

    async def test_an_auth_failure_is_not_reported_as_a_rate_limit(self, factory, source) -> None:
        """The advice differs entirely: a rate limit resumes by itself, and a token
        without the right scope produces the identical 403 forever. Telling someone
        to "run it again" sends them round the same loop."""
        report = await self._sync_raising(factory, AuthError("bad token"))

        assert report.stop_reason == "auth"
        assert report.stopped_early is True

    async def test_a_stream_specific_failure_does_not_stop_the_others(
        self, factory, source
    ) -> None:
        """A 404 on `/actions` means CI is not configured -- which says nothing
        about whether commits can be read."""
        report = await self._sync_raising(factory, PermanentError("no such endpoint"))

        assert report.stop_reason is None
        assert [s.stream for s in report.streams] == ["commits", "pull_requests"]

    async def test_a_clean_run_reports_no_stop_reason(self, factory, source) -> None:
        report = await ArtifactSync(factory, reader=FakeReader()).sync_source(
            SRC, streams=["commits"]
        )
        assert report.stopped_early is False


class TestFailureIsolation:
    async def test_a_repository_name_that_is_not_owner_slash_repo_is_reported(
        self, factory
    ) -> None:
        async with factory() as session:
            session.add(
                SourceRow(
                    id="src_bad",
                    tenant_id="local",
                    platform=Platform.GITHUB,
                    external_id="X",
                    name="not-a-pair",
                )
            )
            await session.commit()

        report = await ArtifactSync(factory, reader=FakeReader()).sync_source("src_bad")
        assert report.failed
        assert "owner/repo" in report.failed[0].error

    async def test_syncing_a_source_that_does_not_exist_raises_clearly(self, factory) -> None:
        with pytest.raises(LookupError, match="no source"):
            await ArtifactSync(factory, reader=FakeReader()).sync_source("src_ghost")
