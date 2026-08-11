"""Validating a repository, and saying the right thing when it cannot be read.

Almost every test here is about a *failure*, because that is what this module is
for. Confirming a readable repository is one line; the value is in distinguishing
"you typed it wrong" from "your token cannot see it" from "an owner has not
approved your token yet" -- three problems that GitHub reports with two status
codes, and that have completely different next actions.
"""

from __future__ import annotations

import httpx
import pytest

from cli.github_probe import RepoStatus, parse_repo_reference, probe_repository

pytestmark = pytest.mark.unit


def transport(status: int, body: dict | None = None, headers: dict | None = None):
    """A client that answers every request the same way."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body or {}, headers=headers or {})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestParsing:
    @pytest.mark.parametrize(
        "given",
        [
            "omnisense/api",
            "omnisense/api/",
            "  omnisense/api  ",
            "github.com/omnisense/api",
            "www.github.com/omnisense/api",
            "https://github.com/omnisense/api",
            "http://github.com/omnisense/api",
            "https://github.com/omnisense/api.git",
            "https://github.com/omnisense/api/",
        ],
    )
    def test_every_way_somebody_types_a_repository(self, given: str) -> None:
        """All of these are the same repository. Refusing any of them would be a
        pointless argument with somebody who has just pasted from their browser."""
        assert parse_repo_reference(given) == ("omnisense", "api")

    @pytest.mark.parametrize(
        "given",
        ["", "omnisense", "not a repo", "https://gitlab.com/omnisense/api", "a/b/c", "/api"],
    )
    def test_what_is_not_a_repository(self, given: str) -> None:
        assert parse_repo_reference(given) is None

    def test_dots_and_underscores_are_legal_in_a_name(self) -> None:
        """`docs.rs`, `my_repo` and `some.thing` are all real repository names, and
        a validator that rejected them would be wrong more often than the user."""
        assert parse_repo_reference("rust-lang/docs.rs") == ("rust-lang", "docs.rs")
        assert parse_repo_reference("a/my_repo") == ("a", "my_repo")


class TestReadable:
    async def test_a_repository_that_can_be_read(self) -> None:
        async with transport(
            200,
            {
                "full_name": "omnisense/api",
                "node_id": "R_kgDOABCD1M",
                "default_branch": "main",
                "private": True,
                "archived": False,
            },
        ) as client:
            probe = await probe_repository("omnisense/api", token="t", client=client)

        assert probe.ok
        assert probe.node_id == "R_kgDOABCD1M"
        assert probe.default_branch == "main"
        assert probe.is_private

    async def test_the_node_id_is_captured_because_names_change(self) -> None:
        """The node id is what a source is keyed on. A repository renamed on GitHub
        keeps it, and every artifact follows; keyed on the name, a rename forks the
        history in two."""
        async with transport(
            200, {"full_name": "omnisense/renamed", "node_id": "R_stable", "default_branch": "main"}
        ) as client:
            probe = await probe_repository("omnisense/old-name", token="t", client=client)

        assert probe.node_id == "R_stable"
        assert probe.full_name == "omnisense/renamed"

    async def test_an_archived_repository_is_reported_not_refused(self) -> None:
        """Archived means read-only, not invisible -- and its history is often
        exactly what somebody is asking about."""
        async with transport(
            200, {"full_name": "omnisense/old", "node_id": "R_1", "archived": True}
        ) as client:
            probe = await probe_repository("omnisense/old", token="t", client=client)

        assert probe.ok
        assert probe.is_archived
        assert "archived" in probe.message


class TestUnreadable:
    async def test_a_bad_token_says_the_token_is_bad(self) -> None:
        async with transport(401) as client:
            probe = await probe_repository("omnisense/api", token="wrong", client=client)

        assert probe.status is RepoStatus.BAD_TOKEN
        assert "token" in probe.fix.lower()

    async def test_a_404_with_a_token_names_both_possibilities(self) -> None:
        """GitHub answers "no such repository" and "you may not see it" with the
        same 404, deliberately -- otherwise a 404-versus-403 difference would let
        anyone enumerate private repositories. Saying only "not found" sends
        somebody to check their spelling when the problem is a token scope.
        """
        async with transport(404) as client:
            probe = await probe_repository("omnisense/api", token="t", client=client)

        assert probe.status is RepoStatus.NOT_FOUND
        assert "spelling" in probe.fix
        assert "private" in probe.fix
        assert "organisation" in probe.fix

    async def test_a_404_without_a_token_blames_the_missing_token_first(self) -> None:
        """Because that is overwhelmingly the reason. Giving the same message as
        the authenticated case would bury the one thing that is actually wrong."""
        async with transport(404) as client:
            probe = await probe_repository("omnisense/api", token=None, client=client)

        assert probe.status is RepoStatus.NO_TOKEN
        assert "GITHUB_TOKEN" in probe.fix

    async def test_a_permission_403_is_not_a_rate_limit(self) -> None:
        """The same status code, opposite responses: backing off from a
        permissions error waits forever, and failing hard on a rate limit throws
        away a run that would have succeeded in sixty seconds."""
        async with transport(403, {"message": "Resource not accessible"}) as client:
            probe = await probe_repository("omnisense/api", token="t", client=client)

        assert probe.status is RepoStatus.FORBIDDEN
        assert "approved" in probe.fix

    async def test_a_rate_limit_403_is_recognised_by_its_body(self) -> None:
        async with transport(
            403, {"message": "API rate limit exceeded"}, {"x-ratelimit-remaining": "0"}
        ) as client:
            probe = await probe_repository("omnisense/api", token="t", client=client)

        assert probe.status is RepoStatus.RATE_LIMITED
        assert "wait" in probe.fix.lower()

    async def test_a_rate_limit_is_recognised_from_the_header_alone(self) -> None:
        """The body wording has changed before; the header has not."""
        async with transport(
            403, {"message": "Forbidden"}, {"x-ratelimit-remaining": "0"}
        ) as client:
            probe = await probe_repository("omnisense/api", token="t", client=client)

        assert probe.status is RepoStatus.RATE_LIMITED

    async def test_saml_gets_its_own_message(self) -> None:
        """ "Authorise the token for the organisation" is a specific, findable
        setting, and no amount of re-issuing the token will help without it."""
        async with transport(
            403, {"message": "Resource protected by organization SAML enforcement"}
        ) as client:
            probe = await probe_repository("omnisense/api", token="t", client=client)

        assert probe.status is RepoStatus.FORBIDDEN
        assert "SAML" in probe.message or "SAML" in probe.fix

    async def test_a_network_failure_is_reported_not_raised(self) -> None:
        """This is called in a loop while somebody types. An exception per typo
        would mean the wizard crashes or wraps everything in a try/except that
        flattens the distinctions this module exists to draw."""

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("dns failure")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            probe = await probe_repository("omnisense/api", token="t", client=client)

        assert probe.status is RepoStatus.UNREACHABLE
        assert not probe.ok

    async def test_a_malformed_reference_never_reaches_the_network(self) -> None:
        """No client is passed, so a request would fail loudly."""
        probe = await probe_repository("not a repo", token="t")
        assert probe.status is RepoStatus.MALFORMED

    async def test_every_failure_carries_something_to_do_about_it(self) -> None:
        """The point of the module. A status with no next action is a status the
        reader has to go and research."""
        cases = [
            (401, {}, {}),
            (404, {}, {}),
            (403, {"message": "nope"}, {}),
            (429, {}, {}),
            (500, {}, {}),
        ]
        for status, body, headers in cases:
            async with transport(status, body, headers) as client:
                probe = await probe_repository("omnisense/api", token="t", client=client)
            assert not probe.ok
            assert probe.message, status
            assert probe.fix, f"{status} has no suggested fix"
