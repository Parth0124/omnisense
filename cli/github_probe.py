"""Check one GitHub repository, and say precisely what is wrong when it is not readable.

Used by `omnisense init` to validate a repository the moment it is typed, rather
than accepting it and discovering at the first sync that the token cannot read
it. That distinction is the whole reason this exists: a wrong answer here is
found in one second by the person who can fix it, and the same wrong answer found
at sync time is an empty result with no obvious cause.

**Why not reuse `GitHubConnector`.** The connector is built to walk a repository:
it holds cursors, watermarks, rate-limit budgets and a sync context. Constructing
one to ask "does this exist" would mean assembling a sync run to make a single
GET, and every one of those moving parts is a way for validation to fail for a
reason that has nothing to do with the repository.

**404 is the interesting case.** GitHub answers "no such repository" and "you may
not see this repository" with the *same* 404 -- deliberately, so a private
repository's existence cannot be probed. That means the honest message names both
possibilities, and says which is more likely given the token that was used. A
message saying only "not found" sends somebody to check their spelling when the
real problem is a token scope.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass

import httpx

__all__ = [
    "RepoProbe",
    "RepoStatus",
    "parse_repo_reference",
    "probe_repository",
]

GITHUB_API = "https://api.github.com"

_REPO_REFERENCE = re.compile(
    r"""
    ^(?:(?:https?://)?(?:www\.)?github\.com/)?  # a pasted URL, with or without scheme
    (?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)
    /
    (?P<repo>[A-Za-z0-9._-]{1,100}?)
    (?:\.git)?                              # a clone URL
    /?$
    """,
    re.VERBOSE,
)
"""What a person actually types when asked for a repository.

All of these are the same repository, and refusing any of them would be a
pointless argument with somebody who has just pasted from their browser:

    omnisense/api
    https://github.com/omnisense/api
    https://github.com/omnisense/api.git
    github.com/omnisense/api/
    omnisense/api/
"""


class RepoStatus(enum.StrEnum):
    """The outcome of a probe. Each one has a different next action."""

    OK = "ok"
    NOT_FOUND = "not_found"
    NO_TOKEN = "no_token"
    BAD_TOKEN = "bad_token"
    FORBIDDEN = "forbidden"
    RATE_LIMITED = "rate_limited"
    UNREACHABLE = "unreachable"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class RepoProbe:
    """What was learned about a repository, and what to do if it is unusable."""

    status: RepoStatus
    reference: str
    message: str
    fix: str = ""

    node_id: str | None = None
    full_name: str | None = None
    default_branch: str | None = None
    is_private: bool | None = None
    is_archived: bool | None = None

    @property
    def ok(self) -> bool:
        return self.status is RepoStatus.OK


def parse_repo_reference(value: str) -> tuple[str, str] | None:
    """Pull `(owner, repo)` out of whatever was typed, or `None` if it is not a repo."""
    match = _REPO_REFERENCE.match(value.strip())
    if match is None:
        return None
    return match.group("owner"), match.group("repo")


async def probe_repository(
    reference: str,
    *,
    token: str | None,
    client: httpx.AsyncClient | None = None,
    timeout_seconds: float = 15.0,
) -> RepoProbe:
    """Ask GitHub about one repository and classify the answer.

    Returns rather than raises. This is called in a loop while somebody types, and
    an exception per typo would mean the wizard either crashes or wraps every call
    in a try/except that flattens the distinctions this function exists to draw.
    """
    parsed = parse_repo_reference(reference)
    if parsed is None:
        return RepoProbe(
            status=RepoStatus.MALFORMED,
            reference=reference,
            message=f"{reference!r} does not look like a repository",
            fix="Expected owner/name, e.g. omnisense/api -- a full GitHub URL is fine too",
        )

    owner, repo = parsed
    full_name = f"{owner}/{repo}"

    # No token is not a refusal. GitHub serves public repositories
    # unauthenticated, so a missing token only narrows what can be *seen* -- and
    # short-circuiting here meant the wizard could not validate a single
    # repository until a token existed, including the public ones it can read
    # perfectly well. The absence is carried into the 404 message instead, where
    # it is the most likely explanation and worth saying.
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=timeout_seconds)
    try:
        response = await http.get(f"{GITHUB_API}/repos/{owner}/{repo}", headers=headers)
    except httpx.HTTPError as error:
        return RepoProbe(
            status=RepoStatus.UNREACHABLE,
            reference=reference,
            full_name=full_name,
            message=f"could not reach github.com: {error}",
            fix="Check your network, then try again",
        )
    finally:
        if owns_client:
            await http.aclose()

    return _classify(response, reference=reference, full_name=full_name, authenticated=bool(token))


def _classify(
    response: httpx.Response, *, reference: str, full_name: str, authenticated: bool
) -> RepoProbe:
    """Turn one HTTP response into an outcome with an action attached."""
    if response.status_code == 200:
        body = response.json()
        archived = bool(body.get("archived"))
        return RepoProbe(
            status=RepoStatus.OK,
            reference=reference,
            full_name=body.get("full_name") or full_name,
            node_id=body.get("node_id"),
            default_branch=body.get("default_branch"),
            is_private=bool(body.get("private")),
            is_archived=archived,
            message=(
                f"{body.get('full_name')} — {'private' if body.get('private') else 'public'}"
                + (", archived" if archived else "")
            ),
        )

    if response.status_code == 401:
        return RepoProbe(
            status=RepoStatus.BAD_TOKEN,
            reference=reference,
            full_name=full_name,
            message="GitHub rejected the token",
            fix="The token is wrong, expired or revoked. Issue a new one at "
            "github.com/settings/personal-access-tokens",
        )

    if response.status_code == 404:
        # GitHub answers "no such repository" and "you may not see this
        # repository" identically, on purpose -- otherwise a 404-versus-403
        # difference would let anyone enumerate private repositories. So the
        # message has to name both, in the order they are likely.
        if not authenticated:
            return RepoProbe(
                status=RepoStatus.NO_TOKEN,
                reference=reference,
                full_name=full_name,
                message=f"{full_name} is not public, and there is no token to check it with",
                fix="If the repository is private, set GITHUB_TOKEN in .env "
                "(github.com/settings/personal-access-tokens). If it is public, "
                "check the spelling.",
            )
        return RepoProbe(
            status=RepoStatus.NOT_FOUND,
            reference=reference,
            full_name=full_name,
            message=f"{full_name} not found, or your token cannot see it",
            fix=(
                "GitHub returns the same 404 for both, so check in this order: "
                "the spelling; then, if it is private, that the token grants access "
                "to this repository; then, if it belongs to an organisation, that an "
                "owner has approved the token"
            ),
        )

    if response.status_code == 403:
        body = response.text.lower()
        # A 403 is two very different problems wearing one status code, and the
        # body is the only thing that separates them. Backing off from a
        # permissions error waits forever; failing hard on a rate limit throws
        # away a run that would have succeeded in sixty seconds.
        if "rate limit" in body or response.headers.get("x-ratelimit-remaining") == "0":
            reset = response.headers.get("x-ratelimit-reset", "")
            return RepoProbe(
                status=RepoStatus.RATE_LIMITED,
                reference=reference,
                full_name=full_name,
                message="GitHub rate limit reached",
                fix=f"Wait and try again{f' (resets at {reset})' if reset else ''}",
            )
        if "saml" in body:
            return RepoProbe(
                status=RepoStatus.FORBIDDEN,
                reference=reference,
                full_name=full_name,
                message="the organisation requires SAML authorisation for this token",
                fix="Authorise the token for the organisation in your GitHub settings, "
                "under Personal access tokens",
            )
        return RepoProbe(
            status=RepoStatus.FORBIDDEN,
            reference=reference,
            full_name=full_name,
            message="the token is valid but not permitted to read this repository",
            fix="Check the token grants access to this repository, and that an "
            "organisation owner has approved it",
        )

    if response.status_code == 429:
        return RepoProbe(
            status=RepoStatus.RATE_LIMITED,
            reference=reference,
            full_name=full_name,
            message="GitHub is rate limiting this token",
            fix="Wait a minute and try again",
        )

    return RepoProbe(
        status=RepoStatus.UNREACHABLE,
        reference=reference,
        full_name=full_name,
        message=f"GitHub answered {response.status_code}",
        fix="Unexpected. Try again; if it persists, check githubstatus.com",
    )
