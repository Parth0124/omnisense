"""Read the streams a repository actually records: commits, PRs, reviews, CI runs.

`connectors/enterprise/github.py` reads issues, discussions and releases and
turns them into `Signal`s. This reads the other four streams and turns them into
`Artifact`s, and the split is on purpose rather than an oversight.

**Why not extend the existing connector.** `BaseConnector`'s contract is
Signal-shaped from end to end -- `normalize()` returns a `Signal`, `dedup_keys()`
takes one, and the runtime around it archives, dedups and enriches on that
assumption. A commit is not a Signal and a CI run is emphatically not one; making
them fit would mean either widening that contract for every connector or lying in
`normalize()`. Reading them separately costs a second HTTP layer and keeps both
honest, which is the trade `docs/architecture.md` already makes between the two
data paths.

**What this module does and does not do.** It fetches and it maps. It does not
store, does not decide what is new, and does not manage credentials -- those are
`services/artifact_sync.py`'s job. Keeping fetching separate from deciding is
what makes every mapper below testable against a recorded payload with no
database and no network.

Three things about GitHub's API that shape the code
---------------------------------------------------
**Only two of the four streams can be filtered server-side.** `/commits` takes
`since` and `/actions/runs` takes `created`. `/pulls` takes neither -- it can only
be sorted -- so a pull-request walk has to page until it sees something older
than the watermark and then stop itself. Reviews are worse: they hang off a pull
request, so they cost one request each and are only fetched for pull requests
that have actually changed.

**A pull request's `updated_at` moves for things that are not the pull request.**
A comment, a label, a re-run all bump it. That makes it a *safe* watermark -- it
never misses a change -- and a noisy one, so re-reading an unchanged pull request
has to be cheap and idempotent rather than prevented.

**`node_id` is the identity, never the number.** Issue and pull-request numbers
repeat across repositories, and a URL changes when a repository is renamed. Every
artifact below is keyed on the node id for the same reason
`connectors/enterprise/github.py` keys authors on theirs.

Failures use `connectors/exceptions.py`, unchanged
------------------------------------------------
This reads a different data path but it is still a connector, and the four
families there already encode the decision a caller has to make: `QuotaError` is
a *partial success* whose cursor is kept, `AuthError` is terminal, `TransientError`
is worth retrying, `PermanentError` is not. Inventing a parallel pair of GitHub
exceptions -- which this module briefly did -- forces every caller to learn a
second vocabulary for the same four outcomes, and quietly loses the distinction
between "your token cannot see this repository" and "github.com is down", which
are a fixable configuration problem and a wait-and-retry respectively.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

import httpx

from connectors.exceptions import AuthError, PermanentError, QuotaError, TransientError

__all__ = [
    "GITHUB_API",
    "GitHubReader",
    "RateLimitState",
    "parse_link_header",
    "parse_time",
]

CONNECTOR_SLUG: Final = "github"

GITHUB_API: Final = "https://api.github.com"

PER_PAGE: Final = 100
"""GitHub's maximum. Fewer means more round trips against the same rate limit."""

_LINK_NEXT = re.compile(r'<([^>]+)>;\s*rel="next"')


@dataclass(slots=True)
class RateLimitState:
    """What GitHub last said about the budget.

    Tracked rather than merely obeyed, because the useful number is not "am I
    limited now" but "how much of the hour have I spent" -- a backfill that will
    exhaust the budget in ten minutes should be visible before it stops, not
    after.
    """

    limit: int | None = None
    remaining: int | None = None
    resets_at: datetime | None = None
    requests_made: int = 0

    def observe(self, headers: Mapping[str, str]) -> None:
        self.requests_made += 1
        limit = headers.get("x-ratelimit-limit")
        remaining = headers.get("x-ratelimit-remaining")
        reset = headers.get("x-ratelimit-reset")
        if limit and limit.isdigit():
            self.limit = int(limit)
        if remaining and remaining.isdigit():
            self.remaining = int(remaining)
        if reset and reset.isdigit():
            self.resets_at = datetime.fromtimestamp(int(reset), tz=UTC)

    @property
    def is_exhausted(self) -> bool:
        return self.remaining is not None and self.remaining <= 0

    def describe(self) -> str:
        if self.remaining is None or self.limit is None:
            return f"{self.requests_made} requests"
        return f"{self.requests_made} requests, {self.remaining}/{self.limit} budget left"


def parse_link_header(value: str | None) -> str | None:
    """The `rel="next"` URL from a Link header, or `None` at the last page.

    Followed rather than counted. GitHub's page count is not exposed anywhere
    reliable, and constructing `?page=n+1` walks off the end of a list that is
    shrinking while it is read -- a repository where commits are being pushed
    during a backfill.
    """
    if not value:
        return None
    match = _LINK_NEXT.search(value)
    return match.group(1) if match else None


@dataclass(slots=True)
class GitHubReader:
    """Paged, rate-limit-aware reads of one repository.

    Holds no per-repository state, so one reader serves a whole sync across many
    repositories -- which matters for the rate limit, since the budget is per
    *token* rather than per repository and a reader per repository would each
    think it had a full hour to spend.
    """

    token: str | None = None
    client: httpx.AsyncClient | None = None
    rate_limit: RateLimitState = field(default_factory=RateLimitState)
    timeout_seconds: float = 30.0

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def _get(self, url: str, params: Mapping[str, Any] | None = None) -> httpx.Response:
        owns = self.client is None
        http = self.client or httpx.AsyncClient(timeout=self.timeout_seconds)
        try:
            response = await http.get(url, params=params, headers=self._headers())
        except httpx.HTTPError as error:
            raise TransientError(
                f"could not reach github.com: {error}", connector=CONNECTOR_SLUG, cause=error
            ) from error
        finally:
            if owns:
                await http.aclose()

        self.rate_limit.observe(response.headers)
        status = response.status_code

        if status == 429 or (status == 403 and self.rate_limit.is_exhausted):
            # A 403 is two problems wearing one status code, and only the budget
            # header separates them. Backing off from a permissions error waits
            # forever; failing hard on a rate limit throws away a backfill that
            # would have resumed in twenty minutes.
            raise self._quota_error(status)
        if status in (401, 403):
            raise AuthError(
                "GitHub refused the credentials -- the token is missing, expired, or "
                "lacks read access to this repository",
                connector=CONNECTOR_SLUG,
                status_code=status,
            )
        if 400 <= status < 500:
            raise PermanentError(
                f"GitHub answered {status} for {url}: {response.text[:200]}",
                connector=CONNECTOR_SLUG,
                status_code=status,
            )
        if status >= 500:
            raise TransientError(
                f"GitHub answered {status} for {url}",
                connector=CONNECTOR_SLUG,
                status_code=status,
            )
        return response

    def _quota_error(self, status: int) -> QuotaError:
        """The budget is gone, carrying when it returns so a caller can say so.

        `reset_at` is a UNIX timestamp because that is what `QuotaError` takes and
        what `x-ratelimit-reset` gives -- the two were built to meet.

        The *message* is in local time, though, because a person reads it. Held in
        UTC it said "until 14:19" to someone whose clock read 19:41, which reads
        as five hours ago rather than eight minutes away -- so the one number the
        message exists to convey was the one thing it got wrong.
        """
        resets_at = self.rate_limit.resets_at
        when = f" until {resets_at.astimezone():%H:%M}" if resets_at else ""
        return QuotaError(
            f"GitHub rate limit exhausted{when}",
            connector=CONNECTOR_SLUG,
            status_code=status,
            reset_at=resets_at.timestamp() if resets_at else None,
        )

    async def _paginate(
        self, path: str, params: Mapping[str, Any] | None = None, *, max_pages: int = 100
    ) -> AsyncIterator[dict[str, Any]]:
        """Walk `Link: rel="next"` until it runs out or `max_pages` is reached.

        `max_pages` is a stop, not a target. Without one, a first backfill of a
        very old repository walks tens of thousands of commits and spends the
        whole hourly budget before anything is stored -- so the caller bounds it
        and records how far it got. Silently returning a prefix would be worse
        than either: it reads as "that is all there is".
        """
        url = f"{GITHUB_API}{path}"
        query: dict[str, Any] | None = {"per_page": PER_PAGE, **(params or {})}

        for _ in range(max_pages):
            response = await self._get(url, query)
            payload = response.json()
            if isinstance(payload, dict):
                # `/actions/runs` wraps its list; the list endpoints do not.
                payload = payload.get("workflow_runs") or payload.get("items") or []
            if not payload:
                return
            for item in payload:
                yield item

            next_url = parse_link_header(response.headers.get("link"))
            if not next_url:
                return
            # The next URL already carries every parameter; passing them again
            # would double `page` and silently re-read the same page forever.
            url, query = next_url, None

    # --------------------------------------------------------------- streams --

    async def repository(self, owner: str, repo: str) -> dict[str, Any]:
        """The repository itself, for its node id and default branch."""
        response = await self._get(f"{GITHUB_API}/repos/{owner}/{repo}")
        payload: dict[str, Any] = response.json()
        return payload

    async def commits(
        self, owner: str, repo: str, *, since: datetime | None = None, max_pages: int = 100
    ) -> AsyncIterator[dict[str, Any]]:
        """Commits on the default branch, newest first.

        `since` is honoured by GitHub, so this is the cheap stream: an incremental
        sync of a quiet repository costs one request that returns nothing.

        Only the default branch. Commits on unmerged branches are reachable but
        cost a request per branch to enumerate, and a commit that never merged is
        rarely what "what happened" means. Merged work arrives through the merge.
        """
        params: dict[str, Any] = {}
        if since:
            params["since"] = _iso(since)
        async for commit in self._paginate(
            f"/repos/{owner}/{repo}/commits", params, max_pages=max_pages
        ):
            yield commit

    async def pull_requests(
        self, owner: str, repo: str, *, since: datetime | None = None, max_pages: int = 100
    ) -> AsyncIterator[dict[str, Any]]:
        """Pull requests, most recently updated first, stopping at the watermark.

        The endpoint takes no `since`, so the stop is client-side: sorted by
        `updated`, the first item older than the watermark means every item after
        it is too. Reading the whole list and filtering afterwards would work and
        would cost the entire history on every sync.
        """
        params = {"state": "all", "sort": "updated", "direction": "desc"}
        async for pull in self._paginate(
            f"/repos/{owner}/{repo}/pulls", params, max_pages=max_pages
        ):
            if since:
                updated = parse_time(pull.get("updated_at"))
                if updated and updated <= since:
                    return
            yield pull

    async def reviews(self, owner: str, repo: str, number: int) -> AsyncIterator[dict[str, Any]]:
        """Reviews on one pull request.

        One request per pull request, which is why the caller only asks for pull
        requests that have actually changed. Asking for all of them would turn a
        five-request incremental sync into one request per open pull request,
        every time.
        """
        async for review in self._paginate(
            f"/repos/{owner}/{repo}/pulls/{number}/reviews", max_pages=5
        ):
            yield review

    async def workflow_runs(
        self, owner: str, repo: str, *, since: datetime | None = None, max_pages: int = 50
    ) -> AsyncIterator[dict[str, Any]]:
        """Actions runs, newest first.

        The highest-volume stream by a distance -- every push to every branch
        triggers one -- which is why its page bound is lower than the others.
        """
        params: dict[str, Any] = {}
        if since:
            # `created` takes a range expression, not a bare timestamp.
            params["created"] = f">={_iso(since)}"
        async for run in self._paginate(
            f"/repos/{owner}/{repo}/actions/runs", params, max_pages=max_pages
        ):
            yield run

    async def run_jobs(self, owner: str, repo: str, run_id: int) -> list[dict[str, Any]]:
        """The jobs inside one run -- which is where "what actually broke" lives.

        Fetched only for runs that did not succeed. A green run's jobs are all
        green, and paying a request each to learn that would double the cost of
        the noisiest stream to add nothing.
        """
        return [
            job
            async for job in self._paginate(
                f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs", max_pages=3
            )
        ]


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(value: Any) -> datetime | None:
    """A GitHub timestamp, or `None` if it is missing or unparseable.

    GitHub emits `2026-08-11T13:52:03Z`, and `fromisoformat` did not accept the
    trailing `Z` before Python 3.11. Kept as one function because a missing
    timestamp is normal -- a pull request with no `merged_at`, a run still going --
    and every caller wants the same "absent rather than exploded" answer.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
