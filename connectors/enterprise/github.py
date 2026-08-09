"""GitHub: issues (and pull requests), discussions and releases from one repository.

Phase 6 (`docs/connector-spec.md` §9.3). GitHub is the first connector in the
tree that reads *three* different provider surfaces under one slug, and the first
whose records are **mutable documents** rather than immutable posts. Almost every
decision below follows from one of those two facts.

**Three streams, three watermarks, one cursor.** An issue list, a release list
and a discussion connection advance at completely different rates, and the
runtime persists exactly one `Cursor` per (connector, account, params) triple
(§4). So each stream keeps its own watermark inside `Cursor.checkpoint["streams"]`
and the top-level `Cursor.watermark` is the **minimum** across the selected
streams -- the only value for which the sentence the runtime reads it as ("nothing
older than this is unemitted") is true. Taking the maximum would declare the
slowest stream caught up because the fastest one was.

**Issues are paged by advancing `since`, not by walking page numbers.** GitHub
offers `since=` + `sort=updated&direction=asc`, which is why oldest-first is
natural here rather than reconstructed. It is *page numbers* that are unsafe: an
issue edited while the walk is in flight moves to the end of an ascending
`updated` ordering, every issue behind it shifts down one index, and the item that
slid across the page boundary is never returned. Re-issuing the query with
`since` set to the newest `updated_at` of the previous page makes the pager
keyset-based, so an edited issue can only be *re-read*, never skipped. The cost is
re-reading the boundary second on every page, which dedup collapses. The one thing
a keyset walk cannot survive is a saturated second -- more items sharing one
`updated_at` than a page holds, which a bulk label operation produces -- so that
case steps one second forward and records the step in the cursor, exactly as
`connectors/news/gdelt.py` does for the same reason.

**Discussions have no REST surface at all.** They exist only in GraphQL
(`repository.discussions`), so this connector speaks both protocols. The
connection is ordered `UPDATED_AT DESC` and buffered-then-reversed rather than
walked ascending, because descending is the direction in which a concurrent edit
*duplicates* a node instead of hiding one: an updated discussion jumps to the
front of a DESC ordering, pushing later nodes one place deeper, and a node seen
twice costs a dedup hit while a node skipped costs a permanent hole.

**Releases document no ordering and offer no `since`.** The REST list endpoint
takes neither a sort nor a filter parameter and its ordering is not part of the
documented contract, so the only reading that is safe is a *bounded full listing*
that this module sorts itself. That is affordable because releases are the rarest
of the three by orders of magnitude; when the listing is longer than the bound,
the release watermark deliberately does not advance, because a prefix of an
unordered list proves nothing about what is behind it.

**The secondary rate limit is the one that surprises people.** The primary budget
(5,000 requests/hour for a PAT or an App installation) is reported on every
response in `x-ratelimit-remaining`, so it is easy to respect. The secondary
limits -- concurrent requests, points per minute, content-creation per minute --
are *not* reported anywhere, are enforced with a `403` rather than a `429`, and
arrive while `x-ratelimit-remaining` still reads in the thousands. Two
consequences are load-bearing here:

- `concurrency=1`. GitHub's own guidance is to make requests serially; a burst of
  parallel reads is the fastest way to trip a limit no header warns about.
- A `403` is classified by *reading it* (`_denied`). Filing a secondary-limit 403
  as an `AuthError` would flag the account `needs_reauth` and stop the connector
  until a human re-links a credential that was never broken.

**Identity is rule 1 through `node_id`, not the numeric `id`.** GitHub's numeric
ids are unique per resource type, so an issue and a release can both be `4711`;
`node_id` is the global relay id and is the *same* string the GraphQL API returns
as `Discussion.id`. Using it means all three streams share one identity space with
no hand-rolled prefixing, and a discussion fetched over GraphQL cannot collide
with an issue fetched over REST.

**A mutable document needs a revision in its dedup key.** `dedup_keys` is
overridden to fold `updated_at` into the identity key. Without that, layer 1 of
`docs/signal-model.md` §4.2 collapses every re-read of an issue -- including the
read that exists precisely *because* the issue changed -- and an issue closed
three hours after it was filed would never reach the pipeline as closed. The
`Signal.id` itself stays keyed on the item, so the five stores still hold one row
per issue and upsert it.

Not done here: file contents, commits and code review comments. Those are a
different kind of object with a different licence position (§9.3 notes per-repo
licences apply to file content), and each would be an N+1 request per issue.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Self
from urllib.parse import urlencode

import httpx
import jwt

from models.base import utcnow
from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal
from connectors.auth.token_store import (
    DEFAULT_REFRESH_MARGIN_SECONDS,
    InMemoryTokenStore,
    StoredToken,
    TokenStore,
)
from connectors.base import BaseConnector
from connectors.exceptions import (
    AuthError,
    ConnectorConfigurationError,
    ConnectorError,
    NormalizationError,
    PermanentError,
    QuotaError,
    TransientError,
)
from connectors.normalize.mapper import FieldMap, FieldSpec, MappingContext
from connectors.protocol import (
    Credentials,
    Cursor,
    DedupKeys,
    FetchPage,
    RateLimitHint,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)

__all__ = ["ENVELOPE_KEY", "STREAMS", "GitHubConnector"]

_SLUG: Final = "github"

API_BASE: Final = "https://api.github.com"
GRAPHQL_PATH: Final = "/graphql"

API_VERSION: Final = "2022-11-28"
"""Value of `X-GitHub-Api-Version`.

GitHub versions the REST API by date and serves the *newest* version to a client
that sends no header. Pinning it is what stops a breaking change to a payload
shape from arriving on a Tuesday with no deploy of ours in between.
"""

STREAMS: Final[tuple[str, ...]] = ("issues", "discussions", "releases")
"""The three surfaces this connector reads, in the order it reads them.

Order is fixed rather than taken from `params` so that a page budget truncates the
same stream every run. A rotating order would starve whichever stream happened to
sort last on a busy repository.
"""

ENVELOPE_KEY: Final = "omnisense"
"""Payload key under which this connector stores what it computed itself.

Four values cannot come from the item: the repository (a repo-scoped issue payload
does not name its own repo -- only `repository_url`, an API URL nobody can paste),
which stream produced it, whether an issue is really a pull request, and the
GraphQL-vs-REST provenance a DLQ record needs to be attributable. Namespacing them
under one obviously-ours key keeps "what GitHub said" and "what we decided"
legible in a payload someone is reading at 3am, exactly as `connectors/news/rss.py`
does.
"""

MAX_PAGE_SIZE: Final = 100
"""GitHub's ceiling for `per_page` on every REST collection, and for `first` on a
GraphQL connection. Asking for more is answered with a 422 on REST and a
validation error on GraphQL, not with a silent clamp."""

DEFAULT_LOOKBACK_DAYS: Final = 30
"""How far back a first run reaches for a stream with no watermark.

A month rather than the whole history: a cold start against a repository with
40,000 issues would otherwise spend an entire hourly budget before the first
incremental poll ever ran. `params["lookback_days"]` widens it, and because that
changes `params_hash` it gets its own cursor row -- which is the separation §4.1
rule 5 requires between a backfill and the live watermark.
"""

MAX_DISCUSSION_PAGES: Final = 10
"""Requests one run may spend descending the discussions connection.

Bounds the buffer the newest-first-to-oldest-first reversal needs. A truncated
descent parks its progress rather than advancing the watermark; see
`_discussion_pages`.
"""

MAX_RELEASE_PAGES: Final = 5
"""Pages of the (unordered, unfiltered) release listing one run will read.

500 releases. Beyond that the listing is treated as unread rather than partially
read -- see the module docstring.
"""

GITHUB_OVERLAP_SECONDS: Final = 600
"""How far back before its watermark each stream restarts (§4.1 rule 3).

Twice the default, for the two streams that cannot express a server-side `since`
bound. GitHub serves list endpoints and the GraphQL connection from read replicas,
so an item's `updated_at` is assigned before the row carrying it is visible to the
query that would have returned it. The issues walk re-reads its boundary second
regardless, so this window only has to cover replication lag plus the duration of
one run.
"""

QUOTA_WAIT_THRESHOLD_SECONDS: Final = 900.0
"""Above this wait a throttle becomes a `QuotaError` rather than a `TransientError`.

§5.2: holding a worker for a quarter of an hour to preserve in-run retry state
costs more than checkpointing and rescheduling. A primary-limit exhaustion is
almost always on the far side of this line -- the window is hourly -- and a
secondary-limit block is almost always on the near side.
"""

JWT_LIFETIME_SECONDS: Final = 540
"""Lifetime of the App JWT used to mint an installation token.

Nine minutes, under GitHub's ten-minute ceiling. A JWT at exactly 600 seconds is
rejected the moment GitHub's clock is one second ahead of ours, which presents as
an intermittent 401 on a credential that is perfectly valid.
"""

JWT_BACKDATE_SECONDS: Final = 60
"""How far `iat` is backdated. GitHub rejects a JWT issued in its own future, and
a worker whose clock runs thirty seconds fast would otherwise fail every mint."""

_RATE_LIMIT_HEADERS: Final[frozenset[str]] = frozenset(
    {
        "retry-after",
        "x-ratelimit-limit",
        "x-ratelimit-remaining",
        "x-ratelimit-reset",
        "x-ratelimit-used",
        "x-ratelimit-resource",
    }
)
"""The only response headers that leave `fetch()`.

An allowlist, not a denylist. `FetchPage.raw_headers` travels with the batch into
code that may log what it is handed, and a GitHub response also carries
`set-cookie`, CDN request ids and an echo of the authorization scheme
(`docs/connector-spec.md` §1).
"""

_CORE_RESOURCE: Final = "core"
"""The value of `x-ratelimit-resource` whose counters describe *this* budget.

GitHub reports several independent budgets through identically-named headers:
`core` is 5,000 requests/hour, `search` is 30 *requests per minute*, and `graphql`
is 5,000 *points* per hour. Feeding a `search` hint into the shared bucket would
clamp the whole connector to 28 remaining requests, and feeding a `graphql` hint
would talk it up to 4,999 in a unit that is not requests at all. See
`parse_rate_limit`.
"""

_SECONDARY_MARKERS: Final[tuple[str, ...]] = (
    "secondary rate limit",
    "abuse detection",
)
"""Substrings GitHub puts in the body of a secondary-limit 403/429.

Matched against the message rather than the status code because the status code
is the same one used for "your token may not read this repository", and the two
demand opposite responses -- back off, versus stop and tell a human.
"""

_PERMISSION_MARKERS: Final[tuple[str, ...]] = (
    "not accessible by integration",
    "must have admin rights",
    "resource protected by organization saml",
    "forbidden",
)
"""403 bodies that mean *this installation lacks a permission*, not *this token is
invalid*. A missing App permission is fixed in the App's settings page, so it is a
`ConnectorConfigurationError`; flagging the account `needs_reauth` would send an
operator to re-issue a credential that would be rejected identically."""

_OWNER: Final = "owner"
_REPO: Final = "repo"

_STREAM_CHECKPOINT_KEY: Final = "streams"
_MAX_SATURATED_MARKS: Final = 20


# --------------------------------------------------------------------------- #
# Configuration validation (no I/O, all of it at construction time)
# --------------------------------------------------------------------------- #

_OWNER_PATTERN: Final = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")
"""GitHub logins: alphanumerics and single internal hyphens, 39 characters max."""

_REPO_PATTERN: Final = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def _validated_repository(params: Mapping[str, Any]) -> tuple[str, str]:
    """Resolve `params` to exactly one `(owner, repo)` pair.

    One repository per account, not a list. A list would give one `params_hash`
    several independent watermarks and the runtime persists one
    (`docs/connector-spec.md` §4) -- the same reason `connectors/social/reddit.py`
    merges a multireddit into a single listing rather than fanning out. Several
    repositories are several connector accounts, which also gives each its own
    rate-limit sub-bucket.

    Both halves are validated because both are interpolated into the request path.
    GitHub's own naming rules are narrower than "anything without a slash", and a
    value that escapes the path segment turns a misconfiguration into a request
    for a resource nobody configured.
    """
    raw = _as_text(params.get("repository") or params.get("repo_full_name"))
    if raw:
        owner, _, repo = raw.partition("/")
    else:
        owner, repo = _as_text(params.get(_OWNER)), _as_text(params.get(_REPO))

    owner, repo = owner.strip(), repo.strip()
    if not owner or not repo:
        raise ConnectorConfigurationError(
            "the GitHub connector needs params['repository'] as 'owner/name' (or "
            "params['owner'] and params['repo']); there is no sensible default and "
            "an unscoped crawl of GitHub is a different product decision",
            connector=_SLUG,
        )
    if not _OWNER_PATTERN.match(owner) or not _REPO_PATTERN.match(repo):
        raise ConnectorConfigurationError(
            f"{owner + '/' + repo!r} is not a valid GitHub repository path. The "
            "value is interpolated into the request path, so anything else is "
            "either a 404 or a request for a resource nobody configured",
            connector=_SLUG,
        )
    return owner, repo


def _validated_streams(params: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve which of `STREAMS` this account reads, preserving the fixed order.

    Selecting a subset is worth supporting because the three surfaces have
    different permission requirements: discussions need `read:discussion` on a
    classic PAT, and a repository with discussions disabled answers the GraphQL
    query with a `NOT_FOUND` that would otherwise fail every run.
    """
    raw = params.get("streams") or params.get("stream")
    if raw is None:
        return STREAMS
    if isinstance(raw, str):
        requested = [part.strip() for part in raw.split(",")]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        requested = [_as_text(part) for part in raw]
    else:
        requested = []

    chosen = {part.lower() for part in requested if part}
    unknown = sorted(chosen - set(STREAMS))
    if unknown:
        raise ConnectorConfigurationError(
            f"unknown GitHub stream(s) {unknown}; this connector reads "
            f"{list(STREAMS)}. A silently-ignored stream name is a stream an "
            "operator believes is being ingested",
            connector=_SLUG,
        )
    if not chosen:
        raise ConnectorConfigurationError(
            "params['streams'] selected nothing; omit it to read all of "
            f"{list(STREAMS)}",
            connector=_SLUG,
        )
    return tuple(name for name in STREAMS if name in chosen)


def _validated_page_size(params: Mapping[str, Any]) -> int:
    raw = params.get("per_page", MAX_PAGE_SIZE)
    try:
        size = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConnectorConfigurationError(
            f"params['per_page'] must be an integer, got {raw!r}", connector=_SLUG
        ) from exc
    if not 1 <= size <= MAX_PAGE_SIZE:
        raise ConnectorConfigurationError(
            f"params['per_page'] must be between 1 and {MAX_PAGE_SIZE}; GitHub "
            "answers a larger value with 422 rather than clamping it",
            connector=_SLUG,
        )
    return size


# --------------------------------------------------------------------------- #
# Field maps: one per stream
# --------------------------------------------------------------------------- #


def _label_names(value: Any) -> list[str] | None:
    """`labels` as a list of names.

    GitHub returns label *objects* (`{"id":…, "name":"bug", "color":…}`) and, on
    some legacy payloads, bare strings. Storing the objects would blow past the
    depth-3 cap `Signal` enforces on `metadata`; storing the names keeps the field
    filterable in OpenSearch, which is the only thing anything downstream does
    with it.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    names = [
        _as_text(item.get("name")) if isinstance(item, Mapping) else _as_text(item)
        for item in value
    ]
    kept = [name for name in names if name]
    return kept or None


def _asset_downloads(value: Any) -> int | None:
    """Total downloads across a release's binary assets.

    Summed rather than carried per asset: `Engagement.raw` is a flat counter map,
    and "how many people took this build" is the question, not "which of the seven
    platform tarballs was most popular".
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    total = 0
    seen = False
    for asset in value:
        if not isinstance(asset, Mapping):
            continue
        count = asset.get("download_count")
        if isinstance(count, int) and not isinstance(count, bool):
            total += count
            seen = True
    return total if seen else None


_ENVELOPE_METADATA: Final[dict[str, FieldSpec]] = {
    "github.repository": FieldSpec.at(f"{ENVELOPE_KEY}.repository"),
    "github.stream": FieldSpec.at(f"{ENVELOPE_KEY}.stream"),
}

_ISSUE_FIELDS: Final = FieldMap(
    platform=Platform.GITHUB,
    # `created_at`, not `updated_at`, even though `updated_at` is what the cursor
    # walks. `Signal.timestamp` is event time at the source and every trend and
    # forecast agent keys off it exclusively; stamping an issue with the moment
    # somebody added a label would file a three-year-old bug report as today's
    # news. The update time is metadata, and it is what `dedup_keys` reads.
    timestamp=FieldSpec.at("created_at", required=True),
    item_id=FieldSpec.at("node_id", required=True),
    # `html_url`, never `url`: the latter is the API endpoint. `Signal.url` is the
    # permalink a report citation resolves against, and an api.github.com URL
    # answers a browser with JSON.
    url=FieldSpec.at("html_url"),
    title=FieldSpec.at("title"),
    # GitHub-Flavored Markdown. Not run through `extract_readable`: the extractor
    # would strip nothing from markdown and could mangle fenced code blocks, which
    # is most of what a bug report is made of.
    text=FieldSpec.at("body"),
    content_type="text/markdown",
    # `user.node_id`, not `user.login`. GitHub logins are renameable and the
    # rename rewrites every URL that contained one, so keying an author's history
    # on a handle forks it silently the first time somebody rebrands
    # (`docs/signal-model.md` §3.1).
    author_id=FieldSpec.at("user.node_id"),
    author_handle=FieldSpec.at("user.login"),
    author_profile_url=FieldSpec.at("user.html_url"),
    engagement={
        "comments": FieldSpec.at("comments"),
        "reactions": FieldSpec.at("reactions.total_count"),
        "reactions_plus_one": FieldSpec.at("reactions.+1"),
    },
    metadata={
        **_ENVELOPE_METADATA,
        "github.number": FieldSpec.at("number"),
        "github.state": FieldSpec.at("state"),
        "github.state_reason": FieldSpec.at("state_reason"),
        "github.labels": FieldSpec.at("labels", transform=_label_names),
        # Read by `dedup_keys`, which is why it is required rather than optional:
        # without it a mutable document silently reverts to identity-only dedup and
        # every edit after the first is dropped inside the TTL window.
        "github.updated_at": FieldSpec.at("updated_at", required=True),
        "github.is_pull_request": FieldSpec.at(f"{ENVELOPE_KEY}.is_pull_request"),
    },
)
"""Issues and pull requests.

The REST issues endpoint returns both -- a pull request *is* an issue in GitHub's
data model, distinguishable only by the presence of a `pull_request` key. They are
kept rather than filtered because a PR description is the single richest statement
of intent a repository produces, and `github.is_pull_request` is set in `fetch()`
so a consumer can separate them without re-deriving the rule.

Every counter is the platform's raw number. The normalized engagement axes are
percentiles within a `(platform, content_type)` cohort (`docs/signal-model.md`
§3.4) and a connector holding one record cannot know a percentile.
"""

_DISCUSSION_FIELDS: Final = FieldMap(
    platform=Platform.GITHUB,
    timestamp=FieldSpec.at("createdAt", required=True),
    # The GraphQL `id` *is* the REST `node_id`; see the module docstring.
    item_id=FieldSpec.at("id", required=True),
    url=FieldSpec.at("url"),
    title=FieldSpec.at("title"),
    text=FieldSpec.at("body"),
    content_type="text/markdown",
    author_id=FieldSpec.at("author.id"),
    author_handle=FieldSpec.at("author.login"),
    author_profile_url=FieldSpec.at("author.url"),
    engagement={
        "upvotes": FieldSpec.at("upvoteCount"),
        "comments": FieldSpec.at("comments.totalCount"),
        "reactions": FieldSpec.at("reactions.totalCount"),
    },
    metadata={
        **_ENVELOPE_METADATA,
        "github.number": FieldSpec.at("number"),
        "github.category": FieldSpec.at("category.name"),
        "github.is_answered": FieldSpec.at("isAnswered"),
        "github.updated_at": FieldSpec.at("updatedAt", required=True),
    },
)

_RELEASE_FIELDS: Final = FieldMap(
    platform=Platform.GITHUB,
    # `published_at`, not `created_at`: `created_at` is when the *tag* was cut,
    # which for a release published from an old tag can be years earlier. The
    # observation is the announcement, and drafts -- which have no `published_at`
    # at all -- are dropped in `normalize` rather than dated by a fallback.
    timestamp=FieldSpec.at("published_at", required=True),
    item_id=FieldSpec.at("node_id", required=True),
    url=FieldSpec.at("html_url"),
    title=FieldSpec.at("name", "tag_name"),
    text=FieldSpec.at("body"),
    content_type="text/markdown",
    author_id=FieldSpec.at("author.node_id"),
    author_handle=FieldSpec.at("author.login"),
    author_profile_url=FieldSpec.at("author.html_url"),
    engagement={
        "reactions": FieldSpec.at("reactions.total_count"),
        "mentions": FieldSpec.at("mentions_count"),
        "asset_downloads": FieldSpec.at("assets", transform=_asset_downloads),
    },
    metadata={
        **_ENVELOPE_METADATA,
        "github.tag_name": FieldSpec.at("tag_name"),
        "github.prerelease": FieldSpec.at("prerelease"),
        # A release payload carries no `updated_at`. `published_at` stands in so
        # `dedup_keys` has a revision to fold in, with the consequence that an
        # edited release body is not re-emitted inside the dedup TTL window --
        # there is nothing in the payload that would let us notice the edit.
        "github.updated_at": FieldSpec.at("published_at", required=True),
    },
)

_FIELD_MAPS: Final[dict[str, FieldMap]] = {
    "issues": _ISSUE_FIELDS,
    "discussions": _DISCUSSION_FIELDS,
    "releases": _RELEASE_FIELDS,
}


_DISCUSSIONS_QUERY: Final = """
query Discussions($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(
      first: $first
      after: $after
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id
        number
        title
        body
        url
        createdAt
        updatedAt
        upvoteCount
        isAnswered
        category { name slug }
        comments { totalCount }
        reactions { totalCount }
        author {
          login
          url
          ... on Node { id }
        }
      }
    }
  }
  rateLimit { limit cost remaining resetAt }
}
"""
"""The whole discussions read, in one round trip.

`... on Node { id }` is not decoration: `author` is typed as the `Actor`
interface, which declares `login`, `url` and `avatarUrl` but *not* `id`. Without
the inline fragment the node id never arrives, `author_id` resolves to nothing and
the mapper -- correctly -- refuses to promote a renameable handle into
`platform_author_id`, so every discussion would be authorless.

`rateLimit` is requested on the same query so the point budget is observed without
spending a second request to ask about it; see `_graphql`.
"""


# --------------------------------------------------------------------------- #
# Per-stream cursor state
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _StreamState:
    """Resume state for one of the three surfaces.

    `pending` is the load-bearing field, and it means the same thing it means in
    `connectors/social/reddit.py`: progress a *truncated* newest-first descent has
    made but may not commit, because the records between the deepest page fetched
    and the previous watermark have not been seen at all. Promoting it early would
    leave those records below the watermark, unemitted, with nothing ever going
    back for them.
    """

    watermark: datetime | None = None
    pending: datetime | None = None
    page_token: str | None = None
    saturated: tuple[str, ...] = ()
    """Seconds the issues walk had to step over. See `_advance_since`."""

    def to_json(self) -> dict[str, Any]:
        return {
            "watermark": _iso(self.watermark),
            "pending": _iso(self.pending),
            "page_token": self.page_token,
            "saturated": list(self.saturated),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> _StreamState:
        marks = value.get("saturated")
        saturated = (
            tuple(str(mark) for mark in marks)[-_MAX_SATURATED_MARKS:]
            if isinstance(marks, Sequence) and not isinstance(marks, (str, bytes))
            else ()
        )
        return cls(
            watermark=_parse_moment(value.get("watermark")),
            pending=_parse_moment(value.get("pending")),
            page_token=_as_text(value.get("page_token")) or None,
            saturated=saturated,
        )


def _load_streams(cursor: Cursor) -> dict[str, _StreamState]:
    """Read per-stream state out of the checkpoint, tolerating anything.

    An unreadable checkpoint costs a re-sync, never a run: `Cursor.checkpoint` is
    documented as opaque JSON the connector owns, and a connector that raised on
    its own historical state could never change that state's shape.
    """
    raw = cursor.checkpoint.get(_STREAM_CHECKPOINT_KEY)
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(name): _StreamState.from_json(value)
        for name, value in raw.items()
        if isinstance(value, Mapping)
    }


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #


class GitHubConnector(BaseConnector):
    """Issues, discussions and releases from one GitHub repository."""

    slug: ClassVar[str] = _SLUG
    platform: ClassVar[Platform] = Platform.GITHUB
    category: ClassVar[SourceCategory] = SourceCategory.ENTERPRISE
    auth_type: ClassVar[AuthType] = AuthType.BEARER
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=80, burst=5, concurrency=1
    )
    """Under the documented 5,000 requests/hour, spent serially.

    5,000/hour is 83.3/minute sustained, so 80 leaves room for the token mint and
    for a second account sharing the same App installation -- the budget GitHub
    enforces belongs to the *token*, not to this run.

    `concurrency=1` is the part that matters and it is not about the primary
    budget at all. The secondary limits (concurrent requests, ~900 points/minute)
    are reported in no header, are enforced with a 403 while
    `x-ratelimit-remaining` still reads in the thousands, and GitHub's own guidance
    is to make requests serially. `burst=5` keeps a page-turn from queueing behind
    a full minute of tokens without ever putting five requests on the wire at once.
    """

    supports_incremental: ClassVar[bool] = True

    supports_backfill: ClassVar[bool] = True
    """History is reachable for issues, through `params["lookback_days"]`.

    `SyncContext` carries no `since`/`until`, so a historical crawl expresses its
    window as a large lookback, which gives it a different `params_hash` and
    therefore its own cursor row (§4.1 rule 5). Discussions and releases are
    bounded by their own page budgets in either mode -- neither surface accepts a
    server-side time filter, so "backfill" for them means "keep descending across
    runs", which the parked `page_token` already does.
    """

    overlap_seconds: ClassVar[int] = GITHUB_OVERLAP_SECONDS

    def __init__(
        self,
        ctx: SyncContext,
        credentials: Credentials,
        *,
        token_store: TokenStore | None = None,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(ctx, credentials)
        # All of the below raises `ConnectorConfigurationError` -- a
        # `PermanentError` -- before a socket exists. §6: configuration defects
        # fail fast, and no cursor is ever created for one.
        self._owner, self._repo = _validated_repository(ctx.params)
        self._streams = _validated_streams(ctx.params)
        self._page_size = _validated_page_size(ctx.params)
        self._lookback_days = max(
            1, _as_int(ctx.params.get("lookback_days"), DEFAULT_LOOKBACK_DAYS)
        )
        self._base_url = _as_text(ctx.params.get("base_url")) or API_BASE
        self._app = _AppCredentials.from_credentials(credentials)
        self._token = "" if self._app is not None else _validated_token(credentials)

        # Not injected through `SyncContext`, which carries no port for one. A
        # process-local store means each replica mints its own installation token;
        # tolerable while a mint costs one request against a budget of 5,000, and
        # the fix is a port on `SyncContext`, not a Redis import here.
        self._token_store = token_store or InMemoryTokenStore()
        self._now = now
        self._client: httpx.AsyncClient | None = None
        self._state: dict[str, _StreamState] = {}
        self._fetched = 0
        self._pages = 0

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct and validate. No I/O: not even the HTTP client is built."""
        return cls(ctx, credentials)

    # ------------------------------------------------------------- lifecycle --

    async def authenticate(self) -> None:
        """Build the client and make sure a usable token exists. Idempotent.

        For a PAT there is nothing to acquire, so this only builds the session. For
        a GitHub App it signs a short-lived JWT and exchanges it for an installation
        token, which is the whole reason the App path is preferred: the installation
        token expires in an hour and is scoped to the repositories the installation
        actually covers, while a PAT carries its owner's entire account for as long
        as nobody rotates it (`docs/security-and-privacy.md`, least privilege).

        Idempotence is not a courtesy -- the runtime calls this once at the start of
        a run and at most once more after a 401 -- so the stored token is reused
        unless it is inside its refresh margin.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.ctx.request_timeout_seconds,
                headers={
                    "User-Agent": self.ctx.user_agent,
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": API_VERSION,
                },
                # A redirect off api.github.com would carry our Authorization
                # header to whatever answered: httpx strips only the auth *it*
                # added, and this header is ours.
                follow_redirects=False,
            )
        if self._app is not None:
            self._token = await self._installation_token(self._app)

    async def aclose(self) -> None:
        """Release the client. Called from `run()`'s `finally`, always.

        The handle is dropped rather than merely closed so a second `run()` on the
        same instance rebuilds it instead of reusing a closed pool. The installation
        token survives in the store, so the rebuild costs no extra mint.
        """
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def _installation_token(self, app: _AppCredentials) -> str:
        """Return a live installation token, minting one only when required.

        The lock is held around the mint and nothing else. GitHub invalidates
        nothing when a second token is minted -- both stay valid -- so a stampede
        is wasteful rather than destructive, but it is wasteful against the same
        5,000/hour budget the run needs.
        """
        stored = await self._token_store.load(self.ctx.account_id)
        if stored is not None and not stored.needs_refresh(now=self._now()):
            return stored.access_token

        async with self._token_store.lock(self.ctx.account_id):
            stored = await self._token_store.load(self.ctx.account_id)
            if stored is not None and not stored.needs_refresh(now=self._now()):
                return stored.access_token
            minted = await self._mint_installation_token(app)
            await self._token_store.save(self.ctx.account_id, minted)
            return minted.access_token

    async def _mint_installation_token(self, app: _AppCredentials) -> StoredToken:
        """Sign an App JWT and exchange it for an installation access token."""
        client = self._client
        if client is None:  # pragma: no cover -- authenticate() builds it first
            raise PermanentError("github token mint ran with no HTTP client", connector=_SLUG)

        issued = int(self._now().timestamp())
        try:
            assertion = jwt.encode(
                {
                    "iat": issued - JWT_BACKDATE_SECONDS,
                    "exp": issued + JWT_LIFETIME_SECONDS,
                    "iss": app.app_id,
                },
                app.private_key,
                algorithm="RS256",
            )
        except (ValueError, TypeError, jwt.PyJWTError) as exc:
            # The key material never appears in the message: an operator needs to
            # know *which* field is unusable, not what is in it.
            raise ConnectorConfigurationError(
                "the GitHub App private key could not sign a JWT; it must be the "
                f"unmodified PEM downloaded from the App settings page ({type(exc).__name__})",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc

        path = f"/app/installations/{app.installation_id}/access_tokens"
        await self.acquire_slot(self._base_url)
        try:
            response = await client.post(
                path, headers={"Authorization": f"Bearer {assertion}"}
            )
        except httpx.TransportError as exc:
            raise TransientError(
                "GitHub is unreachable while minting an installation token",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc

        if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            raise AuthError(
                "GitHub rejected the App JWT; the app id, the private key and the "
                "installation id must all belong to the same App installation",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
            )
        if response.status_code >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise TransientError(
                "GitHub returned a server error while minting an installation token",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
            )
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise ConnectorConfigurationError(
                "GitHub refused to mint an installation token; check that "
                "installation_id names an installation of this App",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
            )

        body = _json_object(response, connector=_SLUG)
        token = _as_text(body.get("token"))
        if not token:
            raise AuthError(
                "GitHub's installation-token response carried no token",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
            )
        return StoredToken(
            access_token=token,
            token_type="Bearer",
            # Absolute, from the provider. An installation token lives one hour and
            # `StoredToken.needs_refresh` applies the five-minute margin that covers
            # clock skew plus the longest request the token might be sent on.
            expires_at=_parse_moment(body.get("expires_at")),
        )

    # ----------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Walk each selected stream in turn, oldest-first within each.

        Streams are walked sequentially rather than concurrently. That is not
        laziness: the secondary rate limit counts concurrent requests, and three
        parallel walks against one repository is exactly the shape that trips it.
        """
        self._state = _load_streams(cursor)
        self._fetched = 0
        self._pages = 0

        for stream in self._streams:
            if self._budget_reached():
                return
            if stream == "issues":
                async for page in self._fetch_issues():
                    yield page
            elif stream == "discussions":
                async for page in self._fetch_discussions():
                    yield page
            else:
                async for page in self._fetch_releases():
                    yield page

    # -- issues ------------------------------------------------------------- --

    async def _fetch_issues(self) -> AsyncIterator[FetchPage]:
        """Walk `updated_at` forward with `since`, oldest-first.

        Each page is a legal restart point the moment it is durable, because the
        ordering is total and ascending: everything older than the page's newest
        `updated_at` has already been yielded. That is the property a newest-first
        pager cannot offer and the reason this stream needs no parked watermark.
        """
        state = self._state.get("issues", _StreamState())
        # Resolved to a concrete instant here rather than left `None`: `since` is
        # both a request parameter and the left edge of the pager's arithmetic, and
        # a nullable value would put a `None` check inside the loop that decides
        # whether the pager moved.
        since = self._window_start(state) or (
            self._now() - timedelta(days=self._lookback_days)
        )
        saturated = list(state.saturated)

        while True:
            budget = self._request_budget()
            if budget == 0:
                return

            params = {
                # Closed issues are signal too -- a bug closed as `not_planned` is
                # a product decision. `state=all` is not the endpoint's default.
                "state": "all",
                "sort": "updated",
                "direction": "asc",
                "per_page": str(budget),
                "since": _github_time(since),
            }
            path = f"/repos/{self._owner}/{self._repo}/issues"
            payload, headers = await self._get(path, params)
            issues = _json_array(payload, path, connector=_SLUG)

            fingerprint = _fingerprint("GET", path, params)
            records = [
                self._record(
                    issue,
                    stream="issues",
                    fingerprint=fingerprint,
                    url=_as_text(issue.get("html_url")) or None,
                    extra={"is_pull_request": "pull_request" in issue},
                )
                for issue in issues
            ]
            newest = _max_moment(*(_parse_moment(i.get("updated_at")) for i in issues))
            full = len(issues) >= budget

            next_since, saturated = _advance_since(since, newest, full, saturated)
            state = replace(
                state,
                watermark=_max_moment(state.watermark, newest),
                pending=None,
                page_token=None,
                saturated=tuple(saturated),
            )
            self._state["issues"] = state
            self._count(records)
            yield self._page(records, headers)

            if not full or next_since <= since or self._budget_reached():
                return
            since = next_since

    # -- discussions -------------------------------------------------------- --

    async def _fetch_discussions(self) -> AsyncIterator[FetchPage]:
        """Descend the GraphQL connection newest-first, then hand it back forwards.

        The whole descent is buffered before the first yield, which is the opposite
        of what `BaseConnector` prefers -- it yields per page so a crash costs one
        page -- and the provider forces it: the only way to know which page is
        *oldest* is to have fetched them all. The buffer is bounded by
        `MAX_DISCUSSION_PAGES`, and a descent that hits that bound parks rather than
        advancing the watermark.
        """
        state = self._state.get("discussions", _StreamState())
        floor = self._window_start(state)
        pages, headers, deepest, complete = await self._descend_discussions(state, floor)

        ordered = list(reversed(pages))
        running = state.pending
        for index, records in enumerate(ordered):
            running = _max_moment(running, *(_discussion_updated(r) for r in records))
            final = index == len(ordered) - 1

            if complete and final:
                # The descent reached the previous watermark or the end of the
                # connection, so everything between them has now been yielded and
                # the parked value may finally be promoted.
                state = replace(
                    state,
                    watermark=_max_moment(state.watermark, running),
                    pending=None,
                    page_token=None,
                )
            else:
                # Either the descent was truncated, or there are older pages still
                # to come in this run. In both cases records below `running` have
                # not all been emitted, so the watermark stays where it was and the
                # progress is parked in the checkpoint instead.
                state = replace(
                    state,
                    pending=running,
                    page_token=deepest if not complete else state.page_token,
                )
            self._state["discussions"] = state
            self._count(records)
            yield self._page(records, headers if index == 0 else {})

        if not ordered:
            # Nothing came back, but a parked token may still have been cleared by
            # a completed descent; persist that so the next run starts from the top.
            if complete and state.page_token is not None:
                self._state["discussions"] = replace(state, page_token=None)

    async def _descend_discussions(
        self, state: _StreamState, floor: datetime | None
    ) -> tuple[list[tuple[RawRecord, ...]], dict[str, str], str | None, bool]:
        """Page backwards from the top -- or from a parked cursor -- toward `floor`."""
        after = state.page_token
        pages: list[tuple[RawRecord, ...]] = []
        headers: dict[str, str] = {}
        complete = False

        for _ in range(self._discussion_page_budget()):
            budget = self._request_budget()
            if budget == 0:
                break
            variables: dict[str, Any] = {
                "owner": self._owner,
                "name": self._repo,
                "first": budget,
                "after": after,
            }
            body, headers = await self._graphql(_DISCUSSIONS_QUERY, variables)
            connection = _discussion_connection(body, self._owner, self._repo)
            nodes = [node for node in connection.get("nodes") or [] if isinstance(node, Mapping)]

            fingerprint = _fingerprint("POST", GRAPHQL_PATH, {"after": after or "", "q": "d"})
            records = tuple(
                self._record(
                    node,
                    stream="discussions",
                    fingerprint=fingerprint,
                    url=_as_text(node.get("url")) or None,
                )
                # Reversed within the page as well as between pages: a consumer
                # reading the emitted stream should see time move forward inside a
                # batch for the same reason it must between them.
                for node in reversed(nodes)
            )
            if records:
                pages.append(records)

            page_info = connection.get("pageInfo")
            info = page_info if isinstance(page_info, Mapping) else {}
            after = _as_text(info.get("endCursor")) or None
            if not nodes or not info.get("hasNextPage") or after is None:
                complete = True
                break
            oldest = _min_moment(*(_parse_moment(n.get("updatedAt")) for n in nodes))
            if floor is not None and oldest is not None and oldest <= floor:
                # Crossed into ground a previous run already covered. The overlap is
                # deliberate (`overlap_seconds`); dedup absorbs it.
                complete = True
                break
            if _graphql_budget_spent(body):
                # The GraphQL point budget is separate from the REST one, so
                # exhausting it must not fail a run whose other two streams still
                # have budget. Ending the descent parks the cursor and the next run
                # continues downward from it.
                break

        return pages, headers, after, complete

    def _discussion_page_budget(self) -> int:
        budget = MAX_DISCUSSION_PAGES
        if self.ctx.max_pages is not None:
            budget = min(budget, max(1, self.ctx.max_pages - self._pages))
        return max(1, budget)

    # -- releases ----------------------------------------------------------- --

    async def _fetch_releases(self) -> AsyncIterator[FetchPage]:
        """Read a bounded full listing, sort it ourselves, emit what is new.

        The endpoint accepts no `since` and documents no ordering, so nothing about
        a prefix of it can be trusted. Reading it whole and sorting locally is the
        only shape that yields oldest-first honestly. When the listing turns out to
        be longer than `MAX_RELEASE_PAGES`, the watermark deliberately stays put:
        the records behind the bound are of unknown age, and advancing past them
        would bury them permanently.
        """
        state = self._state.get("releases", _StreamState())
        floor = self._window_start(state)
        path = f"/repos/{self._owner}/{self._repo}/releases"

        collected: list[Mapping[str, Any]] = []
        headers: dict[str, str] = {}
        complete = False
        for page in range(1, MAX_RELEASE_PAGES + 1):
            if self._request_budget() == 0:
                break
            params = {"per_page": str(self._page_size), "page": str(page)}
            payload, headers = await self._get(path, params)
            releases = _json_array(payload, path, connector=_SLUG)
            collected.extend(releases)
            if len(releases) < self._page_size:
                complete = True
                break

        fingerprint = _fingerprint("GET", path, {"per_page": str(self._page_size)})
        published = [
            (moment, release)
            for release in collected
            for moment in (_parse_moment(release.get("published_at")),)
            if moment is not None
        ]
        published.sort(key=lambda item: item[0])

        fresh = [
            self._record(
                release,
                stream="releases",
                fingerprint=fingerprint,
                url=_as_text(release.get("html_url")) or None,
            )
            for moment, release in published
            if floor is None or moment > floor
        ]
        newest = published[-1][0] if published else None

        if complete:
            state = replace(state, watermark=_max_moment(state.watermark, newest))
        self._state["releases"] = state
        self._count(fresh)
        yield self._page(fresh, headers)

    # -- shared page plumbing ------------------------------------------------ --

    def _page(self, records: Sequence[RawRecord], headers: Mapping[str, str]) -> FetchPage:
        """One page plus the cursor that resumes after it.

        The top-level watermark is the minimum across the selected streams, and it
        is `None` until every one of them has reported. `BaseConnector.run()` clamps
        a `None` back up to the watermark the run started from, which is exactly the
        behaviour wanted while the first stream of a resumed run is still walking:
        no progress is claimed, and none is lost.
        """
        self._pages += 1
        return FetchPage(
            records=records,
            cursor=Cursor(
                watermark=self._min_watermark(),
                # Unused at the top level: every stream keeps its own token inside
                # the checkpoint, because one shared slot cannot describe three
                # independent pagers.
                page_token=None,
                checkpoint={
                    _STREAM_CHECKPOINT_KEY: {
                        name: state.to_json() for name, state in self._state.items()
                    }
                },
            ),
            raw_headers=dict(headers),
        )

    def _min_watermark(self) -> datetime | None:
        moments: list[datetime] = []
        for name in self._streams:
            watermark = self._state.get(name, _StreamState()).watermark
            if watermark is None:
                # One stream has never reported. Claiming any watermark at all
                # would assert that nothing older is unemitted, which is false for
                # exactly that stream.
                return None
            moments.append(watermark)
        return min(moments) if moments else None

    def _window_start(self, state: _StreamState) -> datetime | None:
        """Where a stream resumes: its own watermark, rewound by the overlap.

        `BaseConnector._effective_start` applies the overlap to the *top-level*
        watermark, which this connector does not query on. Applying it per stream
        here is what keeps each surface's restart point honest -- the alternative,
        rewinding the minimum, would drag the fastest stream back to the slowest
        one's position on every poll.
        """
        if state.watermark is None:
            return None
        return state.watermark - timedelta(seconds=self.overlap_seconds)

    def _record(
        self,
        payload: Mapping[str, Any],
        *,
        stream: str,
        fingerprint: str,
        url: str | None,
        extra: Mapping[str, Any] | None = None,
    ) -> RawRecord:
        """Wrap one provider object verbatim, plus the envelope described above.

        `raw_bytes` is `None` on purpose. `RawRecord` documents it as the exact
        bytes the provider returned, and for one item carved out of a hundred-item
        listing those bytes do not exist; re-serializing the payload to fill the
        field would produce a digest that changes with the json library and break
        the content-addressed R2 key it feeds.
        """
        envelope: dict[str, Any] = {
            "repository": f"{self._owner}/{self._repo}",
            "stream": stream,
            **(extra or {}),
        }
        return RawRecord(
            native_id=_node_id(payload) or _fallback_identity(payload),
            payload={**payload, ENVELOPE_KEY: envelope},
            fetched_at=self._now(),
            raw_bytes=None,
            content_type="application/json",
            source_url=url,
            request_fingerprint=fingerprint,
        )

    def _count(self, records: Sequence[RawRecord]) -> None:
        self._fetched += len(records)

    def _request_budget(self) -> int:
        """Page size for the next request, narrowed by `ctx.max_records`.

        Applied *before* the request rather than after the page: `run()`'s own
        ceiling stops the loop only once a hundred-record page has already been
        fetched, normalized and hashed to serve a budget of ten.
        """
        if self._budget_reached():
            return 0
        if self.ctx.max_records is None:
            return self._page_size
        return max(0, min(self._page_size, self.ctx.max_records - self._fetched))

    def _budget_reached(self) -> bool:
        if self.ctx.max_pages is not None and self._pages >= self.ctx.max_pages:
            return True
        return self.ctx.max_records is not None and self._fetched >= self.ctx.max_records

    # ------------------------------------------------------------- normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one GitHub object onto a Signal, or drop it.

        Dropping is reserved for objects GitHub has emptied or never published: a
        draft release has no `published_at` and is visible only to maintainers, and
        a ghost-authored issue whose body was redacted carries no observation. Those
        are expected states, not defects, so they are counted as drops rather than
        filed in the DLQ where they would bury real mapping bugs.
        """
        payload = record.payload
        envelope = payload.get(ENVELOPE_KEY)
        if not isinstance(envelope, Mapping):
            raise NormalizationError(
                "GitHub payload carries no omnisense envelope; it did not come "
                "through this connector's fetch()",
                native_id=record.native_id,
                connector=_SLUG,
            )

        stream = _as_text(envelope.get("stream"))
        field_map = _FIELD_MAPS.get(stream)
        if field_map is None:
            raise NormalizationError(
                f"unknown GitHub stream {stream!r}; this connector maps {list(STREAMS)}",
                native_id=record.native_id,
                connector=_SLUG,
            )

        if stream == "releases" and (
            payload.get("draft") is True or not _as_text(payload.get("published_at"))
        ):
            # An unpublished draft. It has a node id and a body, so it maps
            # perfectly well; it simply has not happened yet, and dating it by
            # `created_at` would announce a release that may never ship.
            return None

        # The runtime keys the R2 object and the Kafka partition off
        # `RawRecord.native_id`, while every store keys off `Signal.id`, which is
        # derived from `node_id`. If those two disagreed the same item would exist
        # under two identities, so the disagreement is caught here rather than
        # discovered as duplicate rows months later.
        node_id = _node_id(payload)
        if node_id is None or node_id != record.native_id:
            raise NormalizationError(
                "payload node id does not match the fetched record's native_id",
                native_id=record.native_id,
                connector=_SLUG,
            )

        return field_map.to_signal(record, self._mapping_context())

    def _mapping_context(self) -> MappingContext:
        return MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=self.ctx.run_id,
        )

    # ------------------------------------------------------------------ dedup --

    def dedup_keys(self, signal: Signal) -> DedupKeys:
        """Fold the item's revision into the identity key.

        Issues, discussions and releases are *mutable documents*, unlike a Reddit
        post or a news article. The inherited identity key is `Signal.id` alone, so
        it collapses every re-read of an issue -- including the read that exists
        because the issue changed -- and an issue closed three hours after it was
        filed would never reach the pipeline as closed.

        `Signal.id` itself is untouched, so the five stores still hold exactly one
        row per issue and upsert it. What changes is only which *observations* the
        seen-set lets through: one per distinct `updated_at`, which is precisely how
        many times GitHub says the document changed.
        """
        keys = super().dedup_keys(signal)
        revision = _as_text(signal.metadata.get("github.updated_at"))
        if not revision:
            return keys
        return DedupKeys(
            identity=f"{keys.identity}:{revision}",
            content=keys.content,
            simhash=keys.simhash,
        )

    # ------------------------------------------------------------ rate limit --

    def parse_rate_limit(self, headers: Mapping[str, str]) -> RateLimitHint | None:
        """Feed back only the counters that describe the bucket we are spending.

        GitHub reports several independent budgets through identically-named
        headers and distinguishes them with `x-ratelimit-resource`. A `search`
        response reports `remaining: 28` out of 30 *per minute*, and a `graphql`
        response reports points rather than requests. `RateLimiter.observe` clamps
        the bucket to whatever it is told, so passing either through unfiltered
        would throttle the whole connector to a number measured in a different
        unit.

        `Retry-After` always survives the filter: it is an instruction, not an
        accounting figure, and ignoring one is how an integration earns a block.
        """
        hint = super().parse_rate_limit(headers)
        if hint is None:
            return None
        resource = _as_text(
            {key.lower(): value for key, value in headers.items()}.get("x-ratelimit-resource")
        )
        if not resource or resource == _CORE_RESOURCE:
            return hint
        if hint.retry_after_seconds is None:
            return None
        return RateLimitHint(retry_after_seconds=hint.retry_after_seconds)

    # --------------------------------------------------------------- requests --

    async def _get(
        self, path: str, params: Mapping[str, str]
    ) -> tuple[Any, dict[str, str]]:
        """One authenticated REST GET. Raises; never retries, never sleeps."""
        client = self._require_client()
        await self.acquire_slot(self._base_url)
        try:
            response = await client.get(path, params=params, headers=self._auth_headers())
        except httpx.TransportError as exc:
            raise TransientError(
                "GitHub is unreachable",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                "the GitHub request could not be issued",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc

        self._raise_for_status(response, path)
        return _json_body(response, connector=_SLUG), _rate_limit_headers(response.headers)

    async def _graphql(
        self, query: str, variables: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], dict[str, str]]:
        """One authenticated GraphQL POST.

        GraphQL answers `200` for most failures and puts them in an `errors` array,
        so the status check below is necessary but nowhere near sufficient --
        `_discussion_connection` is where a field-level error becomes an exception
        of the right class.
        """
        client = self._require_client()
        await self.acquire_slot(self._base_url)
        try:
            response = await client.post(
                GRAPHQL_PATH,
                json={"query": query, "variables": dict(variables)},
                headers=self._auth_headers(),
            )
        except httpx.TransportError as exc:
            raise TransientError(
                "GitHub GraphQL is unreachable",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__},
                cause=exc,
            ) from exc

        self._raise_for_status(response, GRAPHQL_PATH)
        return _json_object(response, connector=_SLUG), _rate_limit_headers(response.headers)

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise PermanentError(
                "fetch() ran before authenticate(); the six-stage order in "
                "BaseConnector.run() is what guarantees it does not",
                connector=_SLUG,
                account_id=self.ctx.account_id,
            )
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        """The Authorization header, built fresh per request.

        Fresh rather than cached on the client, because the App path replaces the
        token mid-run when it approaches expiry, and a header baked into the client
        at construction would keep sending the one that is about to stop working.
        """
        return {"Authorization": f"Bearer {self._token}"}

    def _raise_for_status(self, response: httpx.Response, path: str) -> None:
        status = response.status_code
        if status < httpx.codes.BAD_REQUEST:
            return
        if status == httpx.codes.TOO_MANY_REQUESTS or status == httpx.codes.FORBIDDEN:
            raise self._denied(response, path)
        if status == httpx.codes.UNAUTHORIZED:
            raise AuthError(
                "GitHub rejected the credential",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status == httpx.codes.NOT_FOUND:
            raise ConnectorConfigurationError(
                f"GitHub has no {path} visible to this credential; a 404 on a whole "
                "collection means the repository name is wrong or the token cannot "
                "see it, and a private repository answers 404 rather than 403",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status == httpx.codes.GONE:
            raise ConnectorConfigurationError(
                f"{path} is disabled for this repository; GitHub answers 410 for the "
                "issues collection of a repository with issues turned off. Narrow "
                "params['streams'] rather than polling a surface that does not exist",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            raise TransientError(
                f"GitHub returned {status}",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=status,
            )
        raise PermanentError(
            f"GitHub rejected the request with {status}; the same request will be "
            "rejected identically",
            connector=_SLUG,
            account_id=self.ctx.account_id,
            status_code=status,
            details={"endpoint": path},
        )

    def _denied(self, response: httpx.Response, path: str) -> ConnectorError:
        """Tell the three meanings of a GitHub 403 apart.

        GitHub overloads one status across a primary-budget exhaustion, a
        secondary-limit block and a genuine permission failure, and the correct
        response to each is different: reschedule, back off, or stop and tell a
        human. Guessing wrong in the third direction is the expensive one -- an
        `AuthError` flags the account `needs_reauth`, which stops every future run
        until somebody re-links a credential that was working.
        """
        hint = self.parse_rate_limit(response.headers)
        lowered = {key.lower(): value for key, value in response.headers.items()}
        remaining = _as_int(lowered.get("x-ratelimit-remaining"), -1)
        message = _error_message(response)

        if remaining == 0:
            # Primary budget. The window is hourly, so the wait is nearly always on
            # the far side of the quota threshold; `QuotaError` is a *partial
            # success*, so everything already emitted stays emitted.
            reset_at = hint.reset_at if hint is not None else None
            wait = max(0.0, reset_at - self._now().timestamp()) if reset_at else None
            return QuotaError(
                "GitHub's hourly request budget is exhausted",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                reset_at=reset_at,
                retry_after_seconds=wait,
            )

        if any(marker in message for marker in _SECONDARY_MARKERS) or (
            hint is not None and hint.retry_after_seconds is not None
        ):
            wait = hint.retry_after_seconds if hint is not None else None
            if wait is not None and wait > QUOTA_WAIT_THRESHOLD_SECONDS:
                return QuotaError(
                    "GitHub applied a long secondary rate limit",
                    connector=_SLUG,
                    account_id=self.ctx.account_id,
                    status_code=response.status_code,
                    retry_after_seconds=wait,
                )
            # No `Retry-After` on a secondary block is common; GitHub's guidance is
            # to wait at least a minute. The runtime owns that wait
            # (`docs/connector-spec.md` §1 forbids sleeping here), which is the
            # whole reason this is a `TransientError` and not a local pause.
            return TransientError(
                "GitHub applied a secondary rate limit; it counts concurrency and "
                "request bursts, and it reports neither in a header",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                details={"retry_after_seconds": wait, "endpoint": path},
            )

        if any(marker in message for marker in _PERMISSION_MARKERS):
            return ConnectorConfigurationError(
                f"this credential may not read {path}; grant the App installation "
                "the Issues/Discussions/Contents read permissions it needs, or add "
                "read:discussion to the personal access token",
                connector=_SLUG,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
            )
        return AuthError(
            "GitHub refused the credential",
            connector=_SLUG,
            account_id=self.ctx.account_id,
            status_code=response.status_code,
        )


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _AppCredentials:
    """The three values a GitHub App needs to mint an installation token.

    `__repr__` is overridden for the same reason `Credentials` overrides it: a
    `ConnectorError` carrying this object in `details` would otherwise render a
    PEM private key straight into a log line.
    """

    app_id: str
    installation_id: str
    private_key: str

    @classmethod
    def from_credentials(cls, credentials: Credentials) -> _AppCredentials | None:
        """Build the App triple, or `None` when this account uses a PAT.

        Presence of the private key is what selects the App path. A half-configured
        App -- key but no installation id -- is refused rather than silently falling
        back to a PAT that is probably absent too, because the fallback would report
        "no credential configured" for an account that plainly has one.
        """
        private_key = _as_text(credentials.secrets.get("private_key"))
        if not private_key:
            return None
        app_id = _as_text(credentials.secrets.get("app_id")) or _as_text(
            credentials.secrets.get("client_id")
        )
        installation_id = _as_text(credentials.secrets.get("installation_id"))
        if not app_id or not installation_id:
            raise ConnectorConfigurationError(
                "a GitHub App credential needs private_key, app_id (or client_id) "
                "and installation_id; the JWT is signed with the key and issued by "
                "the app, and only the installation knows which repositories it covers",
                connector=_SLUG,
                account_id=credentials.account_id,
            )
        return cls(app_id=app_id, installation_id=installation_id, private_key=private_key)

    def __repr__(self) -> str:
        return f"_AppCredentials(app_id={self.app_id!r}, private_key=<redacted>)"

    __str__ = __repr__


def _validated_token(credentials: Credentials) -> str:
    """The personal access token, checked for the shapes that fail unhelpfully.

    A token with an embedded newline -- the shape you get from `cat token.txt` --
    fails inside httpx's header encoder with an exception that names the header,
    which then gets logged. Rejecting it here means nothing has been sent yet and
    the message names the field rather than the value.
    """
    for key in ("access_token", "token", "personal_access_token"):
        token = _as_text(credentials.secrets.get(key))
        if token:
            if any(char in token for char in "\r\n\0"):
                raise ConnectorConfigurationError(
                    f"the GitHub credential {key!r} contains a control character; it "
                    "was probably copied with a trailing newline",
                    connector=_SLUG,
                    account_id=credentials.account_id,
                )
            return token
    raise ConnectorConfigurationError(
        "the GitHub connector needs either a personal access token "
        "(secrets['access_token']) or a GitHub App credential "
        "(secrets['private_key'], ['app_id'], ['installation_id']). The App path is "
        "preferred: its token expires in an hour and is scoped to the installation, "
        "while a PAT carries its owner's whole account until somebody rotates it",
        connector=_SLUG,
        account_id=credentials.account_id,
    )


# --------------------------------------------------------------------------- #
# Payload and response helpers
# --------------------------------------------------------------------------- #


def _advance_since(
    since: datetime,
    newest: datetime | None,
    was_full: bool,
    saturated: Sequence[str],
) -> tuple[datetime, list[str]]:
    """Where the next `since` sits, and what that cost.

    Normally the newest `updated_at` in the page: re-reading the boundary second is
    cheap because dedup collapses it, whereas skipping past it would drop every
    issue that shares that second with the last one returned.

    The exception is a *saturated second*. GitHub timestamps have one-second
    resolution and a bulk label operation stamps hundreds of issues with the same
    value, so one second can hold more issues than a page. The newest `updated_at`
    of a full page then equals the start of the window and the pager cannot move.
    Stepping one second past it is the only way to make progress, and it loses
    whatever did not fit. That loss is *recorded* in the cursor rather than
    swallowed: a silent gap in an issue history is indistinguishable from a quiet
    week.
    """
    if newest is None:
        return since, list(saturated)
    if newest > since:
        return newest, list(saturated)
    if not was_full:
        # No progress and no pressure: the window simply ends here.
        return since, list(saturated)
    marks = [*saturated, since.astimezone(UTC).isoformat()][-_MAX_SATURATED_MARKS:]
    return since + timedelta(seconds=1), marks


def _discussion_connection(
    body: Mapping[str, Any], owner: str, repo: str
) -> Mapping[str, Any]:
    """Pull the discussions connection out of a GraphQL response, or raise.

    GraphQL answers `200` with an `errors` array for everything from "this
    repository has discussions disabled" to "your token lacks read:discussion", so
    the error *type* is what decides the exception class here. Only the type is
    read -- never the message -- because a GraphQL error can echo the query, and
    the query carries the repository an operator configured.
    """
    for error in body.get("errors") or []:
        if not isinstance(error, Mapping):
            continue
        kind = _as_text(error.get("type")).upper()
        if kind == "RATE_LIMITED":
            raise TransientError(
                "GitHub GraphQL reported the point budget exhausted",
                connector=_SLUG,
                details={"error_type": kind},
            )
        if kind == "FORBIDDEN":
            raise AuthError(
                "GitHub GraphQL refused the credential for repository.discussions",
                connector=_SLUG,
                details={"error_type": kind},
            )
        raise ConnectorConfigurationError(
            f"GitHub GraphQL rejected the discussions query on {owner}/{repo} "
            f"({kind or 'unspecified'}); a repository with Discussions turned off "
            "answers NOT_FOUND. Narrow params['streams'] rather than polling a "
            "surface that does not exist",
            connector=_SLUG,
            details={"error_type": kind},
        )

    data = body.get("data")
    repository = data.get("repository") if isinstance(data, Mapping) else None
    if not isinstance(repository, Mapping):
        raise PermanentError(
            f"GitHub GraphQL returned no repository object for {owner}/{repo}",
            connector=_SLUG,
        )
    connection = repository.get("discussions")
    if not isinstance(connection, Mapping):
        raise PermanentError(
            "GitHub GraphQL returned no discussions connection; the schema changed",
            connector=_SLUG,
        )
    return connection


def _graphql_budget_spent(body: Mapping[str, Any]) -> bool:
    """Whether the GraphQL point budget cannot fund another query of this size.

    The `rateLimit` block rides on the same response, so this costs nothing extra.
    Compared against `cost` rather than against zero because the budget is spent in
    points and the next query costs what this one did; stopping at
    `remaining < cost` is the last moment at which stopping is still voluntary.
    """
    limit = body.get("rateLimit")
    if not isinstance(limit, Mapping):
        return False
    remaining = _as_int(limit.get("remaining"), -1)
    cost = _as_int(limit.get("cost"), 1)
    return remaining >= 0 and remaining < max(1, cost)


def _node_id(payload: Mapping[str, Any]) -> str | None:
    """The global relay id: `node_id` over REST, `id` over GraphQL.

    Both spellings name the same string for the same object, which is what lets
    three streams and two protocols share one identity space. The GraphQL `id` is
    only accepted when it is not a bare integer, so a numeric REST `id` can never
    be mistaken for one.
    """
    node_id = _as_text(payload.get("node_id"))
    if node_id:
        return node_id
    candidate = _as_text(payload.get("id"))
    return candidate if candidate and not candidate.isdigit() else None


def _fallback_identity(payload: Mapping[str, Any]) -> str:
    """An id for a payload GitHub did not identify.

    GitHub has never omitted `node_id`, so reaching this means the response is not
    what we think it is. The record still needs *an* id in order to become an
    attributable DLQ entry -- `normalize` rejects it a moment later -- and the
    digest is taken over the API URL, which is the one field that is certainly
    unique per object.
    """
    material = _as_text(payload.get("url")) or repr(sorted(payload))
    return f"unidentified:{hashlib.sha256(material.encode('utf-8')).hexdigest()[:32]}"


def _discussion_updated(record: RawRecord) -> datetime | None:
    return _parse_moment(record.payload.get("updatedAt"))


def _json_body(response: httpx.Response, *, connector: str) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        # Usually an error page from the CDN in front of the API. The body is not
        # attached: it can echo the request, and the request carries the token in a
        # header (`docs/connector-spec.md` §1 forbids logging either).
        raise PermanentError(
            f"GitHub returned {len(response.content)} bytes that are not JSON",
            connector=connector,
            status_code=response.status_code,
            cause=exc,
        ) from exc


def _json_object(response: httpx.Response, *, connector: str) -> Mapping[str, Any]:
    body = _json_body(response, connector=connector)
    if not isinstance(body, Mapping):
        raise PermanentError(
            f"GitHub returned a JSON {type(body).__name__} where an object was "
            "expected; the response shape changed",
            connector=connector,
            status_code=response.status_code,
        )
    return body


def _json_array(body: Any, path: str, *, connector: str) -> list[Mapping[str, Any]]:
    if not isinstance(body, Sequence) or isinstance(body, (str, bytes)):
        raise PermanentError(
            f"GitHub returned a {type(body).__name__} where {path} promises an array",
            connector=connector,
        )
    return [item for item in body if isinstance(item, Mapping)]


def _error_message(response: httpx.Response) -> str:
    """The provider's own `message` field, lower-cased, for classification only.

    Never rendered into an error. It is used to tell a secondary-limit 403 from a
    permission 403, and GitHub's message on the first is the only place that
    distinction is published.
    """
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, Mapping):
        return ""
    return _as_text(body.get("message")).lower()


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in _RATE_LIMIT_HEADERS}


def _fingerprint(method: str, path: str, params: Mapping[str, str]) -> str:
    """Hash of method, endpoint and normalized params -- never the credential.

    `lineage.request_fingerprint` is what makes a fetch reproducible: it names the
    exact request that produced a record without naming who made it.
    """
    canonical = urlencode(sorted(params.items()))
    return hashlib.sha256(f"{method} {path}?{canonical}".encode()).hexdigest()[:32]


def _github_time(moment: datetime) -> str:
    """`since`, in the only spelling GitHub documents: ISO 8601 with a `Z`.

    An offset-bearing form (`+00:00`) is accepted today, but the documented
    parameter is `YYYY-MM-DDTHH:MM:SSZ`, and a filter GitHub fails to parse is
    ignored silently -- which returns the whole issue history rather than an error.
    """
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).isoformat() if moment is not None else None


def _parse_moment(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp, or `None`. Never raises.

    Used for pagination arithmetic only. A record whose timestamp is missing or
    unparseable is still emitted -- `normalize` raises for it and it reaches the
    DLQ with its identity attached -- but it must not be allowed to poison a
    watermark, so it is excluded here rather than defaulted to now.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _max_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return max(present) if present else None


def _min_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return min(present) if present else None


def _as_text(value: Any) -> str:
    """Render a scalar as a stripped string; `""` for anything unusable.

    Booleans are refused rather than stringified: `"True"` is never a node id or a
    stream name, and letting one through turns a type confusion into a
    plausible-looking value.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (Mapping, Sequence, set, frozenset)):
        return ""
    return str(value).strip()


def _as_int(value: Any, default: int) -> int:
    if value is None or isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
