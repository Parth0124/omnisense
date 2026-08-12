"""Sync one repository's artifacts: decide what is new, fetch it, store it.

`connectors/github/` fetches and maps. `services/artifact_store.py` writes. This
is what decides *what to ask for* and *when to stop*, which is where the actual
difficulty lives -- a sync that asks for everything works perfectly on a
week-old repository and exhausts an hourly budget on a real one.

The watermark, and why it is per stream
---------------------------------------
Each stream advances independently. Commits arrive constantly, workflow runs
arrive more often still, and a pull request opened in March is updated in
August -- so one watermark for the repository would either re-read the quiet
streams every time or skip changes in the busy ones. They are stored together on
the source row, under `source_metadata["cursors"]`, keyed by stream.

**A watermark is only advanced on a clean pass.** If the rate limit runs out
halfway through the commits, the watermark stays where it was: the next run
re-reads what it already stored, which is free -- every write is an upsert -- and
misses nothing. Advancing it optimistically would leave a permanent hole
in the history at exactly the point the sync was interrupted, and nothing would
ever notice.

**Overlap is deliberate.** The watermark is set slightly behind the newest thing
seen, because GitHub's timestamps have second resolution and two commits can
share one. Advancing to exactly the newest would drop whichever of them the page
boundary happened to cut off.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.logging import get_logger
from connectors.exceptions import (
    AuthError,
    ConnectorError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.github.mapping import (
    map_commit,
    map_person,
    map_pull_request,
    map_review,
    map_workflow_run,
)
from connectors.github.reader import GitHubReader, parse_time
from models.artifact import Artifact, Person
from models.orm.artifact import SourceRow
from services.artifact_store import BATCH_SIZE, ArtifactStore, build_artifact_store

__all__ = ["STREAMS", "ArtifactSync", "StreamResult", "SyncReport", "build_artifact_sync"]

_log = get_logger(__name__)

STREAMS = ("commits", "pull_requests", "reviews", "ci_runs")

WATERMARK_OVERLAP = timedelta(seconds=1)
"""How far behind the newest item a watermark is set.

GitHub timestamps have second resolution, so two commits can share one. Setting
the watermark to exactly the newest drops whichever of them fell on the far side
of a page boundary; one second of overlap re-reads a handful of rows that upsert
to no effect.
"""

REVIEW_FANOUT_LIMIT = 150
"""How many pull requests get their reviews read in one pass.

The only per-item cost in the whole sync. Every other stream pays one request per
*hundred* items; reviews pay one per pull request, because they hang off one and
GitHub exposes no repository-wide list. `pallets/click` had 1,639 pull requests
touched inside the ninety-day window on a first pass -- unbounded, that is 1,639
requests for one stream of one repository, a third of an authenticated hourly
budget and twenty-seven times an unauthenticated one.

Bounded, the remainder is not lost: the chunk is taken oldest-first and the
watermark advances past it, so the next run continues from where this one
stopped. See `_sync_reviews` for why the ordering is what makes that terminate.
"""

DEFAULT_BACKFILL_DAYS = 90
"""How far back a first sync reaches when nothing says otherwise.

A bound rather than a preference. "Everything" on a repository with eight years
of history is tens of thousands of requests before a single row is stored, and
the answer to "what happened while I was away" has never needed 2019.
"""


@dataclass(slots=True)
class StreamResult:
    """What one stream did. Reported per stream because they fail independently."""

    stream: str
    fetched: int = 0
    written: int = 0
    complete: bool = False
    """Whether the stream reached the end of what it was asked for.

    The flag the watermark hangs on. False means the pass stopped early -- a page
    bound or a rate limit -- so the watermark must not move.
    """
    error: str | None = None


@dataclass(slots=True)
class SyncReport:
    """The outcome of syncing one source."""

    source_name: str
    streams: list[StreamResult] = field(default_factory=list)
    rate_limit: str = ""

    stop_reason: str | None = None
    """Why the run gave up before finishing, if it did: `"quota"` or `"auth"`.

    Recorded apart from the error text because the two want opposite advice.
    "Run it again and it resumes" is exactly right after a rate limit and exactly
    wrong after an auth failure, where running it again produces the same 403 --
    the fix is a token with the missing scope.
    """

    @property
    def stopped_early(self) -> bool:
        return self.stop_reason is not None

    @property
    def written(self) -> int:
        return sum(stream.written for stream in self.streams)

    @property
    def failed(self) -> list[StreamResult]:
        return [stream for stream in self.streams if stream.error]


class ArtifactSync:
    """Read one repository's four streams and store what is new."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        reader: GitHubReader,
        store: ArtifactStore | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._reader = reader
        self._store = store or ArtifactStore(session_factory)

    # ------------------------------------------------------------ watermarks --

    async def _load_source(self, source_id: str) -> SourceRow:
        async with self._session_factory() as session:
            source = await session.get(SourceRow, source_id)
            if source is None:
                raise LookupError(f"no source {source_id!r}")
            return source

    @staticmethod
    def _cursor(source: SourceRow, stream: str) -> datetime | None:
        raw = (source.source_metadata or {}).get("cursors", {}).get(stream)
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except ValueError:
            # A cursor we cannot parse is worse than none: it would either crash
            # the sync or be silently treated as "now" and skip everything. Start
            # over instead, which costs a backfill and loses nothing.
            _log.warning("sync.cursor_unreadable", stream=stream, value=raw)
            return None

    async def _advance(self, source_id: str, stream: str, newest: datetime) -> None:
        """Move one stream's watermark. Only ever called after a clean pass."""
        async with self._session_factory() as session:
            source = await session.get(SourceRow, source_id)
            if source is None:
                return
            # Replaced wholesale rather than mutated in place: SQLAlchemy does not
            # see a mutation inside a JSON column and would not write it back.
            metadata = dict(source.source_metadata or {})
            cursors = dict(metadata.get("cursors", {}))
            cursors[stream] = (newest - WATERMARK_OVERLAP).isoformat()
            metadata["cursors"] = cursors
            source.source_metadata = metadata
            await session.commit()

    async def reset_cursors(self, source_id: str, streams: Sequence[str] = STREAMS) -> list[str]:
        """Forget where these streams got to. Returns the ones that had a position.

        Needed because a watermark *outranks* `backfill_days` -- deliberately, so
        a nightly sync does not re-read a year every night. The cost is that
        widening the window afterwards silently does nothing: asking for two years
        when the cursor already sits at yesterday reads yesterday onwards, and the
        older year never arrives. Clearing the cursor is what makes the wider
        window mean something.

        Safe to call: every write is an upsert keyed on GitHub's own id, so a
        re-read updates rows rather than duplicating them. It costs requests, not
        data.
        """
        async with self._session_factory() as session:
            source = await session.get(SourceRow, source_id)
            if source is None:
                return []
            metadata = dict(source.source_metadata or {})
            cursors = dict(metadata.get("cursors", {}))
            cleared = [stream for stream in streams if stream in cursors]
            for stream in cleared:
                del cursors[stream]
            metadata["cursors"] = cursors
            source.source_metadata = metadata
            await session.commit()
        if cleared:
            _log.info("sync.cursors_reset", source_id=source_id, streams=cleared)
        return cleared

    # --------------------------------------------------------------- syncing --

    async def sync_source(
        self,
        source_id: str,
        *,
        streams: Sequence[str] = STREAMS,
        max_pages: int = 20,
        backfill_days: int = DEFAULT_BACKFILL_DAYS,
    ) -> SyncReport:
        """Sync one repository. Never raises for a stream failure.

        A stream that fails is recorded and the rest still run: a repository with
        Actions disabled should not stop its commits from being read, and that is
        a 404 on one endpoint rather than a broken repository.
        """
        source = await self._load_source(source_id)
        owner, _, repo = source.name.partition("/")
        report = SyncReport(source_name=source.name)

        if not owner or not repo:
            report.streams.append(
                StreamResult("all", error=f"{source.name!r} is not an owner/repo pair")
            )
            return report

        floor = datetime.now(UTC) - timedelta(days=backfill_days)

        for stream in streams:
            since = self._cursor(source, stream) or floor
            try:
                result = await self._sync_stream(
                    stream,
                    owner=owner,
                    repo=repo,
                    source_id=source_id,
                    since=since,
                    max_pages=max_pages,
                )
            except (QuotaError, AuthError) as fatal:
                report.streams.append(StreamResult(stream, error=str(fatal)))
                # Every later stream would hit the same wall -- one budget and one
                # token cover all four -- so stop rather than producing four
                # identical errors.
                report.stop_reason = "quota" if isinstance(fatal, QuotaError) else "auth"
                break
            except ConnectorError as error:
                # Transient and permanent are both per-stream: a 404 on `/actions`
                # means CI is not configured, which says nothing about commits.
                report.streams.append(StreamResult(stream, error=str(error)))
                continue

            report.streams.append(result)

        report.rate_limit = self._reader.rate_limit.describe()
        return report

    async def _sync_stream(
        self,
        stream: str,
        *,
        owner: str,
        repo: str,
        source_id: str,
        since: datetime,
        max_pages: int,
    ) -> StreamResult:
        handlers = {
            "commits": self._sync_commits,
            "pull_requests": self._sync_pull_requests,
            "reviews": self._sync_reviews,
            "ci_runs": self._sync_ci_runs,
        }
        handler = handlers.get(stream)
        if handler is None:
            return StreamResult(stream, error=f"unknown stream {stream!r}")
        return await handler(
            owner=owner, repo=repo, source_id=source_id, since=since, max_pages=max_pages
        )

    async def _flush(self, artifacts: list[Artifact], people: list[Person]) -> int:
        if not artifacts:
            return 0
        result = await self._store.write(artifacts, people)
        return result.artifacts_written

    async def _sync_commits(
        self, *, owner: str, repo: str, source_id: str, since: datetime, max_pages: int
    ) -> StreamResult:
        result = StreamResult("commits")
        artifacts: list[Artifact] = []
        people: list[Person] = []
        newest: datetime | None = None

        async for payload in self._reader.commits(owner, repo, since=since, max_pages=max_pages):
            result.fetched += 1
            artifact = map_commit(payload, source_id=source_id)
            if artifact is None:
                continue
            artifacts.append(artifact)
            for key in ("author", "committer"):
                person = map_person(payload.get(key))
                if person:
                    people.append(person)
            newest = max(newest or artifact.occurred_at, artifact.occurred_at)

        result.written = await self._flush(artifacts, people)
        # Complete unless the page bound was the thing that stopped it. Anything
        # less than a full set of pages means the stream genuinely ran out.
        result.complete = result.fetched < max_pages * 100
        if newest and result.complete:
            await self._advance(source_id, "commits", newest)
        return result

    async def _sync_pull_requests(
        self, *, owner: str, repo: str, source_id: str, since: datetime, max_pages: int
    ) -> StreamResult:
        result = StreamResult("pull_requests")
        artifacts: list[Artifact] = []
        people: list[Person] = []
        newest: datetime | None = None

        async for payload in self._reader.pull_requests(
            owner, repo, since=since, max_pages=max_pages
        ):
            result.fetched += 1
            artifact = map_pull_request(payload, source_id=source_id)
            if artifact is None:
                continue
            artifacts.append(artifact)
            person = map_person(payload.get("user"))
            if person:
                people.append(person)
            updated = artifact.updated_at_source or artifact.occurred_at
            newest = max(newest or updated, updated)

        result.written = await self._flush(artifacts, people)
        result.complete = result.fetched < max_pages * 100
        if newest and result.complete:
            # Watermarked on `updated_at`, not `created_at`. The endpoint is
            # sorted and filtered by update time, and a watermark on creation
            # would never see an old pull request that was just merged.
            await self._advance(source_id, "pull_requests", newest)
        return result

    async def _sync_reviews(
        self, *, owner: str, repo: str, source_id: str, since: datetime, max_pages: int
    ) -> StreamResult:
        """Reviews on pull requests that changed since the watermark.

        The only stream that costs a request *per item* rather than per page --
        reviews hang off a pull request and there is no repository-wide endpoint --
        which makes it the one that can quietly eat a whole hourly budget. A first
        pass over `pallets/click` found 1,639 pull requests touched inside the
        ninety-day window, and an unbounded fan-out would have asked for 1,639
        pages of reviews in a single run.

        So this stream converges in chunks, and the ordering is what makes that
        work:

        **Oldest changed first, not newest.** The pull-request endpoint returns
        newest-updated first, which is right for every other purpose and wrong
        here. Taking the newest `n` and stopping would re-read the same newest `n`
        on every run and never reach the rest -- the watermark could never move
        past them. Reversed, each run consumes the oldest chunk of outstanding
        work and the watermark advances past it, so successive runs walk forward
        and finish.

        **The watermark is a pull request's `updated_at`, not a review's
        timestamp.** What this stream is really tracking is how far through the
        changed-pull-request list it has got. A watermark on review time would
        jump to the newest review found in the chunk and skip every older pull
        request still waiting.
        """
        result = StreamResult("reviews")

        changed: list[tuple[int, str, datetime]] = []
        walked = 0
        async for payload in self._reader.pull_requests(
            owner, repo, since=since, max_pages=max_pages
        ):
            walked += 1
            number, node_id = payload.get("number"), payload.get("node_id")
            updated = parse_time(payload.get("updated_at"))
            if number and node_id and updated:
                changed.append((int(number), str(node_id), updated))

        # If the walk itself was truncated it is missing the *oldest* changed pull
        # requests -- the list is newest-first, so a page bound cuts the tail. The
        # watermark must not move in that case, or those pull requests are skipped
        # for good.
        walk_complete = walked < max_pages * 100

        changed.sort(key=lambda item: item[2])
        batch = changed[:REVIEW_FANOUT_LIMIT]
        skipped = len(changed) - len(batch)
        if skipped:
            _log.info("sync.reviews_deferred", deferred=skipped, taken=len(batch))

        artifacts: list[Artifact] = []
        people: list[Person] = []
        watermark: datetime | None = None

        for number, node_id, updated in batch:
            async for payload in self._reader.reviews(owner, repo, number):
                result.fetched += 1
                artifact = map_review(
                    payload,
                    source_id=source_id,
                    pull_number=number,
                    pull_node_id=node_id,
                )
                if artifact is None:
                    continue
                artifacts.append(artifact)
                person = map_person(payload.get("user"))
                if person:
                    people.append(person)

            # Flushed as it goes, per pull request. Held to the end instead, a run
            # that spends its whole budget and then hits the wall on the last
            # request stores nothing at all -- which is the opposite of the
            # partial-success this module is built around.
            if len(artifacts) >= BATCH_SIZE:
                result.written += await self._flush(artifacts, people)
                artifacts, people = [], []

            # Only after the pull request is fully read. Moving it before would
            # let a failure midway through leave the watermark past reviews that
            # were never stored.
            watermark = updated

        result.written += await self._flush(artifacts, people)
        result.complete = walk_complete and not skipped
        if watermark and walk_complete:
            # Advanced even when the fan-out was capped -- that is what lets the
            # next run pick up the remainder instead of repeating this chunk.
            await self._advance(source_id, "reviews", watermark)
        return result

    async def _sync_ci_runs(
        self, *, owner: str, repo: str, source_id: str, since: datetime, max_pages: int
    ) -> StreamResult:
        result = StreamResult("ci_runs")
        artifacts: list[Artifact] = []
        people: list[Person] = []
        newest: datetime | None = None

        async for payload in self._reader.workflow_runs(
            owner, repo, since=since, max_pages=max_pages
        ):
            result.fetched += 1

            # Jobs only for runs that did not succeed. A green run's jobs are all
            # green, and paying a request each to learn that would double the cost
            # of the noisiest stream to add nothing.
            jobs: list[dict[str, Any]] = []
            if payload.get("conclusion") not in (None, "success", "skipped"):
                try:
                    jobs = await self._reader.run_jobs(owner, repo, int(payload["id"]))
                except (TransientError, PermanentError, KeyError, ValueError):
                    # The run is still worth storing without its jobs -- "it
                    # failed" is most of the value; "which job" is the rest.
                    #
                    # Quota and auth are deliberately *not* caught here. Swallowing
                    # them would keep the loop running against a wall it has
                    # already hit, spending one doomed request per remaining run.
                    jobs = []

            artifact = map_workflow_run(payload, source_id=source_id, jobs=jobs)
            if artifact is None:
                continue
            artifacts.append(artifact)
            person = map_person(payload.get("actor"))
            if person:
                people.append(person)
            newest = max(newest or artifact.occurred_at, artifact.occurred_at)

        result.written = await self._flush(artifacts, people)
        result.complete = result.fetched < max_pages * 100
        if newest and result.complete:
            await self._advance(source_id, "ci_runs", newest)
        return result

    async def sync_project(
        self,
        source_ids: Sequence[str],
        *,
        streams: Sequence[str] = STREAMS,
        max_pages: int = 20,
        backfill_days: int = 90,
        reset: bool = False,
    ) -> list[SyncReport]:
        """Sync every source in a project, in order.

        Sequential rather than concurrent, and that is the rate limit talking: the
        budget is per *token*, so four repositories in parallel exhaust it four
        times as fast and the failure lands in the middle of all four at once
        rather than cleanly at the end of the first.
        """
        reports = []
        for source_id in source_ids:
            if reset:
                await self.reset_cursors(source_id, streams)
            report = await self.sync_source(
                source_id,
                streams=streams,
                max_pages=max_pages,
                backfill_days=backfill_days,
            )
            reports.append(report)
            if report.stopped_early:
                break
        return reports


def build_artifact_sync(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    *,
    token: str | None = None,
) -> ArtifactSync:
    if session_factory is None:
        from backend.db.session import get_sessionmaker

        session_factory = get_sessionmaker()
    return ArtifactSync(
        session_factory,
        reader=GitHubReader(token=token),
        store=build_artifact_store(session_factory),
    )
