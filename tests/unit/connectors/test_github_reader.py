"""Paging, rate limits, and knowing when to stop.

The reader's job is not "fetch" -- `httpx` does that. It is to stop at the right
moment: at the end of a list, at a watermark, at a page bound, or at an exhausted
budget. Each of those stops is a different decision and three of them are silent
if they go wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from connectors.exceptions import AuthError, PermanentError, QuotaError, TransientError
from connectors.github.reader import GitHubReader, parse_link_header

pytestmark = pytest.mark.unit


class TestLinkHeader:
    def test_the_next_url_is_extracted(self) -> None:
        header = (
            '<https://api.github.com/repos/a/b/commits?page=2>; rel="next", '
            '<https://api.github.com/repos/a/b/commits?page=9>; rel="last"'
        )
        assert parse_link_header(header) == "https://api.github.com/repos/a/b/commits?page=2"

    def test_the_last_page_has_no_next(self) -> None:
        """Which is how paging knows to stop. Constructing `?page=n+1` instead
        would walk off the end of a list that is growing while it is read."""
        assert parse_link_header('<https://api.github.com/x?page=1>; rel="prev"') is None
        assert parse_link_header(None) is None
        assert parse_link_header("") is None


class TestPaging:
    async def test_it_follows_next_until_it_runs_out(self) -> None:
        pages = {
            "1": ([{"id": 1}, {"id": 2}], '<https://api.github.com/x?page=2>; rel="next"'),
            "2": ([{"id": 3}], ""),
        }
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            page = request.url.params.get("page", "1")
            seen.append(page)
            body, link = pages[page]
            return httpx.Response(200, json=body, headers={"link": link} if link else {})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reader = GitHubReader(token="t", client=client)
            items = [c async for c in reader.commits("a", "b")]

        assert [i["id"] for i in items] == [1, 2, 3]
        assert seen == ["1", "2"]

    async def test_the_page_bound_stops_a_runaway_backfill(self) -> None:
        """A stop, not a target. Without one, a first sync of an old repository
        walks tens of thousands of commits and spends the whole hourly budget
        before a single row is stored."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[{"id": 1}],
                headers={"link": '<https://api.github.com/x?page=99>; rel="next"'},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reader = GitHubReader(token="t", client=client)
            items = [c async for c in reader.commits("a", "b", max_pages=3)]

        assert len(items) == 3

    async def test_an_empty_page_ends_the_walk(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=[]))
        ) as client:
            reader = GitHubReader(token="t", client=client)
            assert [c async for c in reader.commits("a", "b")] == []

    async def test_workflow_runs_are_unwrapped_from_their_envelope(self) -> None:
        """`/actions/runs` wraps its list; the other endpoints do not."""
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, json={"workflow_runs": [{"id": 7}]})
            )
        ) as client:
            reader = GitHubReader(token="t", client=client)
            runs = [w async for w in reader.workflow_runs("a", "b")]
        assert [r["id"] for r in runs] == [7]


class TestPullRequestWatermark:
    async def test_it_stops_at_the_first_item_older_than_the_watermark(self) -> None:
        """The endpoint takes no `since`, so the stop is client-side. Sorted by
        update time, the first older item means every later one is too -- reading
        the whole list and filtering afterwards would cost the entire history on
        every sync."""
        payload = [
            {"number": 3, "updated_at": "2026-08-10T00:00:00Z"},
            {"number": 2, "updated_at": "2026-08-05T00:00:00Z"},
            {"number": 1, "updated_at": "2026-07-01T00:00:00Z"},
        ]
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload))
        ) as client:
            reader = GitHubReader(token="t", client=client)
            pulls = [
                p
                async for p in reader.pull_requests(
                    "a", "b", since=datetime(2026, 8, 6, tzinfo=UTC)
                )
            ]

        assert [p["number"] for p in pulls] == [3]


class TestFailureClassification:
    """Which of the four families a failure lands in decides what happens next.

    `QuotaError` keeps what was read and reschedules, `AuthError` halts, and
    `TransientError` is retried -- so a misclassification is not a cosmetic
    difference in an error message, it is the wrong response to an outage.
    """

    async def test_an_exhausted_budget_is_a_quota_error_carrying_its_reset(self) -> None:
        """The same 403, opposite responses: backing off from a permissions error
        waits forever, and failing hard on a rate limit throws away a backfill
        that would have resumed in twenty minutes."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={"message": "rate limit"},
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1789000000"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reader = GitHubReader(token="t", client=client)
            with pytest.raises(QuotaError) as caught:
                [c async for c in reader.commits("a", "b")]

        assert caught.value.reset_at == 1789000000.0

    async def test_the_reset_time_is_reported_on_the_readers_clock(self) -> None:
        """`x-ratelimit-reset` is UTC and the person reading the message is not.

        Printed in UTC it said "until 14:19" to a reader whose clock showed 19:41 --
        five hours in the past rather than eight minutes ahead, which inverts the
        one thing the message exists to say.
        """
        reset = 1789000000

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403,
                json={},
                headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(QuotaError) as caught:
                [c async for c in GitHubReader(token="t", client=client).commits("a", "b")]

        local = datetime.fromtimestamp(reset, tz=UTC).astimezone()
        assert f"{local:%H:%M}" in str(caught.value)

    async def test_a_429_is_a_quota_error_too(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(429, json={}))
        ) as client:
            with pytest.raises(QuotaError):
                [c async for c in GitHubReader(token="t", client=client).commits("a", "b")]

    async def test_a_permission_403_is_an_auth_error_not_a_quota_one(self) -> None:
        """The distinction the user acts on: a token that cannot see a private
        repository is fixed by regranting scopes, and no amount of waiting helps."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                403, json={"message": "Forbidden"}, headers={"x-ratelimit-remaining": "42"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reader = GitHubReader(token="t", client=client)
            with pytest.raises(AuthError):
                [c async for c in reader.commits("a", "b")]

    async def test_a_404_is_permanent_because_retrying_cannot_fix_it(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(404, json={}))
        ) as client:
            with pytest.raises(PermanentError) as caught:
                [c async for c in GitHubReader(token="t", client=client).commits("a", "b")]
        assert caught.value.retryable is False

    async def test_a_500_is_transient_because_retrying_might(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(502, text="bad gateway"))
        ) as client:
            with pytest.raises(TransientError) as caught:
                [c async for c in GitHubReader(token="t", client=client).commits("a", "b")]
        assert caught.value.retryable is True


class TestRateLimit:
    async def test_the_budget_is_tracked_across_requests(self) -> None:
        """The useful number is not "am I limited now" but "how much of the hour
        have I spent" -- a backfill that will exhaust it should be visible before
        it stops, not after."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=[],
                headers={"x-ratelimit-limit": "5000", "x-ratelimit-remaining": "4997"},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reader = GitHubReader(token="t", client=client)
            [c async for c in reader.commits("a", "b")]

        assert reader.rate_limit.remaining == 4997
        assert "4997/5000" in reader.rate_limit.describe()

    async def test_a_network_failure_is_its_own_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reader = GitHubReader(token="t", client=client)
            with pytest.raises(TransientError, match="could not reach"):
                [c async for c in reader.commits("a", "b")]


class TestAuth:
    async def test_the_token_is_sent_when_present(self) -> None:
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json=[])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await GitHubReader(token="secret", client=client).repository("a", "b")
        assert seen["authorization"] == "Bearer secret"

    async def test_public_reads_work_without_one(self) -> None:
        """GitHub serves public repositories unauthenticated, so a missing token
        narrows what can be seen rather than refusing outright."""
        seen: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(request.headers)
            return httpx.Response(200, json=[])

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await GitHubReader(token=None, client=client).repository("a", "b")
        assert "authorization" not in seen
