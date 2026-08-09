"""YouTube Data API v3: a *unit* budget wearing the shape of a request API.

Phase 2 (`docs/connector-spec.md` §9.1). Every other connector in this repo is
limited in requests per unit time. YouTube is not. Google bills the Data API in
**quota units per day** -- 10,000 by default -- and the cost of a call has almost
nothing to do with how much data it returns:

| Call                    | Units | Returns                                  |
| ----------------------- | ----- | ---------------------------------------- |
| `search.list`           |  100  | up to 50 video *ids* and a stub snippet  |
| `playlistItems.list`    |    1  | up to 50 video ids from one playlist     |
| `videos.list`           |    1  | up to 50 *fully hydrated* videos         |
| `commentThreads.list`   |    1  | up to 100 comment threads                |
| `channels.list`         |    1  | one channel                              |

A hundred `search.list` calls is the entire day. That single fact shapes this
module, and four decisions follow from it.

**Discovery prefers the uploads playlist over search, by two orders of
magnitude.** Following a channel through `channels.list` -> its
`contentDetails.relatedPlaylists.uploads` id -> `playlistItems.list` costs 1 unit
per fifty videos. Answering the same question with `search.list?channelId=` costs
100. So `params['channel_id']` is the cheap path and `params['query']` -- a
free-text search, which nothing else can answer -- is the expensive one, and the
connector says which it is spending on. The resolved uploads playlist id is
cached in the cursor checkpoint so the `channels.list` unit is paid once per
account rather than once per run.

**Discovery and hydration are separate calls on purpose.** `search.list` and
`playlistItems.list` return no statistics at all, and `search.list` truncates
`snippet.description` to roughly 160 characters. Emitting those directly would
put a clipped description into `content.text` and no engagement anywhere. So a
page of ids is hydrated with one `videos.list` -- 1 unit for the whole page,
never one call per video, which is the N+1 that turns a 50-video page into 50
units.

**A run spends from a ledger, not from optimism.** `_UnitLedger` refuses to issue
a call whose cost exceeds what is left of `params['quota_units_per_run']`, and
the check happens *before* the request. `ctx.max_records` and `ctx.max_pages` are
applied the same way (`docs/connector-spec.md` §2.2). A run that stops on budget
commits its cursor and the next one resumes, so the daily quota is divided by the
schedule rather than consumed by whichever account syncs first.

**Resuming an unfinished descent differs by mode, because the providers differ.**
Both pagers walk newest-first and `BaseConnector.fetch()` requires oldest-first,
so a descent is buffered and reversed exactly as `connectors/social/reddit.py`
does, and a descent that runs out of budget may *not* advance the watermark --
the records between the deepest page reached and the previous watermark have not
been seen at all. Where the two modes part company is how the next run gets back
there:

- `search.list` accepts `publishedBefore`, so a parked descent resumes exactly at
  its floor and re-walks nothing. It therefore carries its progress forward in
  `checkpoint['covered_to']`, which may only be promoted to the watermark on the
  page that finally closes the descent.
- `playlistItems.list` accepts no such bound, so a parked descent re-walks from
  the top of the playlist. That is affordable -- 1 unit per fifty -- and it is
  also *safer* than parking an opaque `pageToken`, because a playlist token
  encodes a position and the position shifts when the channel uploads. The
  re-walk skips the `videos.list` hydration for any page lying wholly inside the
  already-covered interval, which is where the units would actually go.

**Transcripts are not implemented and must not be.** `docs/connector-spec.md`
§9.1: there is no public endpoint for the transcript of a video you do not own,
and the third-party timedtext scrapers are a ToS violation. `contentDetails.caption`
is carried in metadata so a downstream consumer can see that captions *exist*;
`MediaRef.transcript_ref` stays `None` until something lawful fills it.

Identity is rule 1 of `docs/signal-model.md` §4.1 throughout: the video id
(`dQw4w9WgXcQ`) or the top-level comment id, verbatim, so a DLQ record names
something a human can paste into YouTube. Rule 3 is never reached.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Self
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from models.base import utcnow
from models.enums import AuthType, MediaKind, Platform, SourceCategory
from models.signal import MediaRef, Signal
from connectors.auth.apikey import ApiKeyAuth
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
from connectors.normalize.mapper import FieldMap, FieldSpec, MappingContext, to_utc_datetime
from connectors.protocol import (
    Credentials,
    Cursor,
    FetchPage,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)

__all__ = ["YouTubeConnector"]


# --------------------------------------------------------------------------- #
# Endpoints, costs and provider constants
# --------------------------------------------------------------------------- #

DEFAULT_BASE_URL: Final = "https://www.googleapis.com/youtube/v3"
WATCH_URL: Final = "https://www.youtube.com/watch"
CHANNEL_URL: Final = "https://www.youtube.com/channel"

SEARCH_PATH: Final = "/search"
VIDEOS_PATH: Final = "/videos"
PLAYLIST_ITEMS_PATH: Final = "/playlistItems"
CHANNELS_PATH: Final = "/channels"
COMMENT_THREADS_PATH: Final = "/commentThreads"

UNIT_COST: Final[Mapping[str, int]] = {
    SEARCH_PATH: 100,
    VIDEOS_PATH: 1,
    PLAYLIST_ITEMS_PATH: 1,
    CHANNELS_PATH: 1,
    COMMENT_THREADS_PATH: 1,
}
"""Google's published per-call quota cost, per endpoint.

Kept as a table beside the endpoints rather than passed at each call site: the
whole design of this module is an argument about these five numbers, and a cost
inlined at a call site is a number that drifts the day someone adds a `part`.
The cost is per *call*, not per item, which is why hydration batches 50 ids.
"""

DAILY_QUOTA_UNITS: Final = 10_000
"""The default per-project daily allocation. Resets at midnight Pacific."""

DEFAULT_RUN_UNIT_BUDGET: Final = 500
"""Units one run may spend, before `params['quota_units_per_run']` overrides it.

Twenty runs a day per project at the default, which is a poll every seventy-odd
minutes with room for a second account. It is deliberately far below
`DAILY_QUOTA_UNITS`: the connector cannot see what other accounts or other
services on the same Google Cloud project have already spent, so the only safe
posture is to take a slice and leave the rest.
"""

QUOTA_RESET_ZONE: Final = "America/Los_Angeles"
"""Where Google's quota day ends. Documented as midnight Pacific Time, not UTC.

Getting this wrong is a `QuotaError` rescheduled up to eight hours early or late,
and the early case simply spends the retry re-discovering that the quota is still
exhausted.
"""

MAX_LIST_RESULTS: Final = 50
"""`maxResults` ceiling for `search.list`, `playlistItems.list` and `videos.list`."""

MAX_COMMENT_RESULTS: Final = 100
"""`maxResults` ceiling for `commentThreads.list`. Genuinely different from 50."""

MAX_DISCOVERY_PAGES: Final = 20
"""Discovery pages one run may buffer before reversal.

A thousand video ids and their publish times -- the buffer holds `_Discovered`
tuples, not payloads, so the reversal `BaseConnector.fetch()` forces costs almost
nothing here. Narrowed further by `ctx.max_pages` and by the unit ledger.
"""

MAX_COMMENT_PAGES: Final = 10
"""Comment pages one run may buffer. Lower than `MAX_DISCOVERY_PAGES` because a
comment descent buffers whole thread payloads rather than ids, and a thousand
threads is already several megabytes held while nothing has been emitted."""

SEARCH_RESULT_CEILING: Final = 500
"""Roughly how deep `search.list` will page before it stops issuing tokens.

Not an error and not worth detecting: the pager simply finds no
`nextPageToken`. It is documented here because it is the reason a *backfill*
configured with `params['query']` silently reaches less history than one
configured with `params['channel_id']`, which can walk an uploads playlist to
the channel's first upload.
"""

YOUTUBE_OVERLAP_SECONDS: Final = 900
"""§4.1 rule 3's eventually-consistent value, chosen for the search index.

A freshly uploaded video is visible on its channel's uploads playlist immediately
and in `search.list` results some minutes later. Resuming exactly at the
watermark in search mode therefore queries a window Google has not finished
indexing, gets nothing, and advances past those videos permanently. Fifteen
minutes plus dedup is what catches them; it costs channel mode nothing but a
handful of re-read ids that the identity key collapses.

It is not a *complete* fix for search mode -- Google publishes no bound on
indexing lag -- which is the strongest argument for following a channel rather
than searching when the deployment cares about completeness.
"""

QUOTA_WAIT_THRESHOLD_SECONDS: Final = 900.0
"""Above this wait a throttle becomes a `QuotaError` rather than a held worker (§5.2)."""

CONTENT_VIDEOS: Final = "videos"
CONTENT_COMMENTS: Final = "comments"
_CONTENT_KINDS: Final[frozenset[str]] = frozenset({CONTENT_VIDEOS, CONTENT_COMMENTS})

COVERED_FROM_KEY: Final = "covered_from"
COVERED_TO_KEY: Final = "covered_to"
"""The interval an unfinished descent has already emitted but may not commit.

Two keys rather than one because the two ends do different jobs: `covered_from`
is where a search descent resumes (`publishedBefore`) and where a playlist
re-walk starts skipping hydration, while `covered_to` is the watermark a search
descent is holding back and the upper edge of the skip window. Both live in
`Cursor.checkpoint` rather than in `watermark`, because the runtime *interprets*
the watermark -- it schedules and detects gaps from it -- while `checkpoint`
round-trips as opaque JSON (`docs/connector-spec.md` §4).
"""

UPLOADS_PLAYLIST_KEY: Final = "uploads_playlist_id"
"""Cache of the channel -> uploads-playlist resolution.

The mapping never changes for a channel, and re-deriving it costs a unit every
run. The widely-repeated shortcut of rewriting the `UC` prefix to `UU` is
undocumented, so this asks `channels.list` once and remembers the answer instead
of relying on a convention Google never promised.
"""

_RATE_LIMIT_HEADERS: Final = frozenset({"retry-after"})
"""The only response headers that leave `fetch()`.

Google's Data API expresses its budget in the response *body* (`error.errors[].reason`),
not in `X-RateLimit-*`, so the allowlist is nearly empty -- which is precisely why
it is an allowlist. A Google response also carries `Set-Cookie`, `Alt-Svc` and
project identifiers, and `FetchPage.raw_headers` is read by code that may log what
it is handed (`docs/connector-spec.md` §1).
"""

# 403 `reason` codes, grouped by what the operator has to *do* about them. Google
# answers a spent daily quota with 403 rather than 429, which is the single
# easiest mistake to make here: filed as an auth failure it flags a working
# account `needs_reauth`, and filed as a transient it retries into a wall that
# will not move until midnight Pacific.
_QUOTA_REASONS: Final[frozenset[str]] = frozenset({"quotaExceeded", "dailyLimitExceeded"})
_THROTTLE_REASONS: Final[frozenset[str]] = frozenset(
    {"rateLimitExceeded", "userRateLimitExceeded", "backendError"}
)
_KEY_REASONS: Final[frozenset[str]] = frozenset({"keyInvalid", "keyExpired", "authError"})
_PROJECT_REASONS: Final[frozenset[str]] = frozenset(
    {"accessNotConfigured", "ipRefererBlocked", "forbidden", "servicesUnavailable"}
)
_RESOURCE_REASONS: Final[frozenset[str]] = frozenset(
    {
        "commentsDisabled",
        "videoNotFound",
        "channelNotFound",
        "playlistNotFound",
        "channelClosed",
        "channelSuspended",
        "processingFailure",
        "commentThreadNotFound",
    }
)

_SAFE_REASON: Final = re.compile(r"^[A-Za-z0-9_.-]{1,48}$")
"""Character-checked before a provider string is allowed into an error message.

Same rule `connectors/social/reddit.py` applies to a 403 body: a provider under
load can put anything in a body field, including an echo of the request.
"""

_CHANNEL_ID: Final = re.compile(r"^UC[A-Za-z0-9_-]{22}$")
_HANDLE: Final = re.compile(r"^@[A-Za-z0-9._-]{3,30}$")
_PLAYLIST_ID: Final = re.compile(r"^[A-Za-z0-9_-]{13,64}$")
_VIDEO_ID: Final = re.compile(r"^[A-Za-z0-9_-]{11}$")

_ISO_DURATION: Final = re.compile(
    r"^P(?:(?P<days>\d+)D)?"
    r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)
"""`contentDetails.duration`, which is ISO-8601 (`PT4M13S`), never seconds.

A live stream reports `P0D`, which parses to zero here rather than raising -- it
is a true statement about a broadcast with no fixed length.
"""

_THUMBNAIL_PREFERENCE: Final = ("maxres", "standard", "high", "medium", "default")


# --------------------------------------------------------------------------- #
# Configuration validation (no I/O, all of it at construction time)
# --------------------------------------------------------------------------- #


def _validated_content(params: Mapping[str, Any]) -> str:
    content = _as_str(params.get("content")) or CONTENT_VIDEOS
    if content not in _CONTENT_KINDS:
        raise ConnectorConfigurationError(
            f"params['content'] must be one of {sorted(_CONTENT_KINDS)}, got {content!r}",
            connector=YouTubeConnector.slug,
        )
    return content


def _validated_channel(params: Mapping[str, Any]) -> str:
    """Accept a `UC…` channel id or an `@handle`; reject the vanity paths.

    `/c/SomeName` and `/user/SomeName` are legacy URL forms with no lookup
    parameter on `channels.list` -- `forUsername` addresses the long-dead legacy
    username namespace, not the vanity path -- so a connector that accepted them
    would spend a unit to learn nothing and then 404. Naming the two forms that
    do work is more useful than a generic "invalid channel".
    """
    raw = _as_str(params.get("channel_id") or params.get("channel"))
    if not raw:
        return ""
    if _CHANNEL_ID.match(raw) or _HANDLE.match(raw):
        return raw
    raise ConnectorConfigurationError(
        f"params['channel_id'] must be a 'UC…' channel id or an '@handle', got {raw!r}. "
        "A /c/ or /user/ vanity path is not resolvable through the Data API",
        connector=YouTubeConnector.slug,
    )


def _validated_playlist(params: Mapping[str, Any]) -> str:
    raw = _as_str(params.get("playlist_id"))
    if not raw:
        return ""
    if not _PLAYLIST_ID.match(raw):
        raise ConnectorConfigurationError(
            f"params['playlist_id'] {raw!r} is not a playlist id",
            connector=YouTubeConnector.slug,
        )
    return raw


def _validated_video(params: Mapping[str, Any]) -> str:
    raw = _as_str(params.get("video_id"))
    if not raw:
        return ""
    if not _VIDEO_ID.match(raw):
        raise ConnectorConfigurationError(
            f"params['video_id'] {raw!r} is not an eleven-character video id",
            connector=YouTubeConnector.slug,
        )
    return raw


def _validated_page_size(params: Mapping[str, Any], ceiling: int) -> int:
    """`maxResults`, checked here rather than discovered as a 400.

    The Data API rejects an out-of-range `maxResults` instead of clamping it, and
    in search mode that rejection costs 100 units to receive.
    """
    raw = params.get("max_results", ceiling)
    try:
        size = int(raw)
    except (TypeError, ValueError) as exc:
        raise ConnectorConfigurationError(
            f"params['max_results'] must be an integer, got {raw!r}",
            connector=YouTubeConnector.slug,
        ) from exc
    if not 1 <= size <= ceiling:
        raise ConnectorConfigurationError(
            f"params['max_results'] must be between 1 and {ceiling} for this mode; "
            "the Data API rejects anything larger rather than clamping it",
            connector=YouTubeConnector.slug,
        )
    return size


# --------------------------------------------------------------------------- #
# The field maps: one for a hydrated video, one for a comment thread
# --------------------------------------------------------------------------- #


def _watch_url(video_id: Any) -> str:
    text = _as_str(video_id)
    return f"{WATCH_URL}?v={text}" if text else ""


def _channel_url(channel_id: Any) -> str:
    text = _as_str(channel_id)
    return f"{CHANNEL_URL}/{text}" if text else ""


def _comment_url(top_level: Any) -> str:
    """The deep link to one comment: `watch?v=<video>&lc=<comment>`.

    Built from the thread rather than declared as a path because it needs two
    values from two levels of the payload. Without the `lc` parameter the URL
    addresses the video, so every comment on a video would carry the same
    permalink and `Signal.url` would stop being a citation.
    """
    if not isinstance(top_level, Mapping):
        return ""
    comment_id = _as_str(top_level.get("id"))
    snippet = top_level.get("snippet")
    video_id = _as_str(snippet.get("videoId")) if isinstance(snippet, Mapping) else ""
    if not comment_id or not video_id:
        return ""
    return f"{WATCH_URL}?v={video_id}&lc={comment_id}"


_VIDEO_FIELDS: Final = FieldMap(
    platform=Platform.YOUTUBE,
    # RFC 3339 with a `Z`, which `to_utc_datetime` parses through
    # `datetime.fromisoformat`. This is the *upload* time from `videos.list`, not
    # the `playlistItems` `snippet.publishedAt` used for pagination -- see
    # `_Discovered`.
    timestamp=FieldSpec.at("snippet.publishedAt", required=True),
    # On a hydrated `videos.list` item `id` is the bare video id string. On a
    # `search.list` result it is an *object* (`{"kind":…, "videoId":…}`), which is
    # one of the reasons search results are never normalized directly.
    item_id=FieldSpec.at("id", required=True),
    url=FieldSpec.at("id", transform=_watch_url),
    title=FieldSpec.at("snippet.title"),
    # The full description, which only `videos.list` returns; `search.list`
    # truncates it to roughly 160 characters with no marker to say it did.
    text=FieldSpec.at("snippet.description"),
    engagement={
        "view_count": FieldSpec.at("statistics.viewCount"),
        # Absent, not zero, when the uploader hides likes. The mapper drops a
        # `None` counter rather than filing it as 0, which would put a hidden
        # count into the same percentile cohort as a genuine zero.
        "like_count": FieldSpec.at("statistics.likeCount"),
        "comment_count": FieldSpec.at("statistics.commentCount"),
        # No `dislikeCount`: YouTube stopped returning it in December 2021, and no
        # `favoriteCount`, which is vestigial and reported as "0" for every video.
    },
    metadata={
        "youtube.channel_id": FieldSpec.at("snippet.channelId"),
        "youtube.channel_title": FieldSpec.at("snippet.channelTitle"),
        "youtube.category_id": FieldSpec.at("snippet.categoryId"),
        "youtube.duration": FieldSpec.at("contentDetails.duration"),
        "youtube.definition": FieldSpec.at("contentDetails.definition"),
        # "live" | "upcoming" | "none". A premiere announced but not yet streamed
        # is still an observation -- a company scheduling an event is signal -- so
        # it is emitted rather than dropped, and this is how a consumer tells it
        # apart from a video that exists.
        "youtube.live_broadcast_content": FieldSpec.at("snippet.liveBroadcastContent"),
        # "true" | "false": whether *any* caption track exists. Not a transcript
        # and not a pointer to one; see the module docstring.
        "youtube.caption_available": FieldSpec.at("contentDetails.caption"),
        # The uploader's declared audio language, which is a claim rather than a
        # detector result. `Signal.language` is filled by
        # `services/signal_engine/language.py` from the text it actually has;
        # copying this into it would fabricate a detection with no confidence
        # behind it (`docs/signal-model.md` §3.3).
        "youtube.default_audio_language": FieldSpec.at(
            "snippet.defaultAudioLanguage", "snippet.defaultLanguage"
        ),
    },
    # `channelId` rather than `channelTitle`: a channel can be renamed and its id
    # cannot, and keying an author's history on the title forks it on the first
    # rebrand (`docs/signal-model.md` §3.1). The title is carried as a label.
    author_id=FieldSpec.at("snippet.channelId"),
    author_handle=FieldSpec.at("snippet.channelTitle"),
    author_display_name=FieldSpec.at("snippet.channelTitle"),
    author_profile_url=FieldSpec.at("snippet.channelId", transform=_channel_url),
)
"""A hydrated `videos.list` item.

`truncated` stays False and that is a deliberate reading of the field: the
description *is* the body this API serves, and we get all of it. What is missing
is the spoken content of the video, which has no lawful endpoint at all -- an
absence recorded by `MediaRef.transcript_ref` being `None`, not by pretending the
description was clipped.
"""

_COMMENT_FIELDS: Final = FieldMap(
    platform=Platform.YOUTUBE,
    # `publishedAt`, never `updatedAt`. `Signal.timestamp` is event time at the
    # source, and an edited comment that re-dated itself would move in the
    # timeline every time its author fixed a typo.
    timestamp=FieldSpec.at("snippet.topLevelComment.snippet.publishedAt", required=True),
    # The *comment* id, not the thread id. They are equal today for a top-level
    # comment, and declaring the one that identifies the thing being emitted is
    # what keeps that true if they ever diverge.
    item_id=FieldSpec.at("snippet.topLevelComment.id", "id", required=True),
    url=FieldSpec.at("snippet.topLevelComment", transform=_comment_url),
    # Plain text because the request asks for `textFormat=plainText`; see
    # `_comment_params`. Left at the default, `textDisplay` is HTML and would
    # carry `<a href>` and `&#39;` into the cleaned body.
    text=FieldSpec.at(
        "snippet.topLevelComment.snippet.textOriginal",
        "snippet.topLevelComment.snippet.textDisplay",
    ),
    engagement={
        "like_count": FieldSpec.at("snippet.topLevelComment.snippet.likeCount"),
        "reply_count": FieldSpec.at("snippet.totalReplyCount"),
    },
    metadata={
        "youtube.video_id": FieldSpec.at("snippet.videoId"),
        "youtube.channel_id": FieldSpec.at("snippet.channelId"),
        "youtube.can_reply": FieldSpec.at("snippet.canReply"),
        "youtube.is_public": FieldSpec.at("snippet.isPublic"),
        "youtube.updated_at": FieldSpec.at("snippet.topLevelComment.snippet.updatedAt"),
    },
    # A nested object, which the mapper's dotted path walks: `authorChannelId` is
    # `{"value": "UC…"}`, and the bare `authorDisplayName` beside it is a
    # renameable label rather than an identity.
    author_id=FieldSpec.at("snippet.topLevelComment.snippet.authorChannelId.value"),
    author_handle=FieldSpec.at("snippet.topLevelComment.snippet.authorDisplayName"),
    author_display_name=FieldSpec.at("snippet.topLevelComment.snippet.authorDisplayName"),
    author_profile_url=FieldSpec.at("snippet.topLevelComment.snippet.authorChannelUrl"),
)


# --------------------------------------------------------------------------- #
# Run state
# --------------------------------------------------------------------------- #


class _UnitLedger:
    """What this run has spent of its quota slice.

    A plain object rather than a counter on the connector so the invariant has
    somewhere to live: nothing may be requested that cannot be paid for, and the
    payment happens before the request. Units are charged whether or not the call
    succeeds, because a request that times out may still have been served and
    counted by Google -- a ledger that only counted successes would overspend the
    daily budget precisely on the day the network is flaky.
    """

    __slots__ = ("_budget", "_spent")

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._spent = 0

    @property
    def spent(self) -> int:
        return self._spent

    @property
    def remaining(self) -> int:
        return max(0, self._budget - self._spent)

    def can_afford(self, cost: int) -> bool:
        return cost <= self.remaining

    def charge(self, cost: int) -> None:
        self._spent += cost


@dataclass(frozen=True, slots=True)
class _Discovered:
    """One video id and the publish time the *discovery* call reported.

    Held instead of the discovery payload because the payload is thrown away:
    everything a Signal needs comes from the later `videos.list` hydration. The
    timestamp is kept for pagination arithmetic alone -- which page is older,
    where the descent has reached -- exactly as `connectors/social/reddit.py`
    keeps `created_utc` out of the mapping path.
    """

    video_id: str
    published_at: datetime | None


@dataclass(frozen=True, slots=True)
class _Descent:
    """One newest-first walk, buffered before anything is yielded.

    `complete` is the load-bearing field: it separates "we reached the previous
    watermark or the end of the listing", after which the watermark may move,
    from "we ran out of units, pages or records", after which it may not.
    """

    pages: tuple[tuple[Any, ...], ...]
    """Fetched order: `pages[0]` is the newest slice, `pages[-1]` the oldest."""

    complete: bool
    resumes_downward: bool
    """Whether the next run continues *below* this descent rather than re-walking
    it from the top. True only for search, which can bound a query with
    `publishedBefore`. It decides whether a parked `covered_to` may be carried
    forward as a watermark -- carrying it across a re-walk would claim coverage of
    an interval this run has fetched but not yet emitted."""

    headers: Mapping[str, str]
    """From the *last* response only. The provider's counters describe the budget
    now, not when an earlier page was fetched, so replaying older values would
    talk the shared bucket back up from a ceiling it correctly clamped down to."""


# --------------------------------------------------------------------------- #
# The connector
# --------------------------------------------------------------------------- #


class YouTubeConnector(BaseConnector):
    """Videos from a channel, playlist or search, or comment threads."""

    slug: ClassVar[str] = "youtube"
    platform: ClassVar[Platform] = Platform.YOUTUBE
    category: ClassVar[SourceCategory] = SourceCategory.SOCIAL
    auth_type: ClassVar[AuthType] = AuthType.API_KEY
    version: ClassVar[str] = "0.1.0"

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=60, burst=1, concurrency=1
    )
    """A per-minute bucket cannot express this provider's limit at all.

    The documented budget is 10,000 *units* per day and Google publishes no
    per-minute request figure for an API-key client -- the per-project rate
    ceiling is a Cloud console setting, not an API contract. So the bucket is
    sized only to stop a burst, `burst=1` and `concurrency=1` serialize the run
    per `docs/connector-spec.md` §5.1's posture for undocumented limits, and the
    budget that actually matters is enforced by `_UnitLedger` against
    `params['quota_units_per_run']`.
    """

    supports_incremental: ClassVar[bool] = True

    supports_backfill: ClassVar[bool] = True
    """Reachable in channel and playlist mode, which can walk an uploads playlist
    back to a channel's first video at 1 unit per fifty. A backfill configured
    with `params['query']` instead stops at the search endpoint's
    `SEARCH_RESULT_CEILING`, having spent 100 units a page to get there. The
    backfill run has its own `params_hash` and therefore its own cursor row,
    which is the separation §4.1 rule 5 requires."""

    overlap_seconds: ClassVar[int] = YOUTUBE_OVERLAP_SECONDS

    def __init__(self, ctx: SyncContext, credentials: Credentials) -> None:
        super().__init__(ctx, credentials)
        params = ctx.params
        # Everything below raises `ConnectorConfigurationError` -- a
        # `PermanentError` -- before a socket exists. §6: configuration defects
        # fail fast, and no cursor is ever created for one.
        self._base_url = (_as_str(params.get("base_url")) or DEFAULT_BASE_URL).rstrip("/")
        self._content = _validated_content(params)
        self._channel = _validated_channel(params)
        self._playlist = _validated_playlist(params)
        self._video = _validated_video(params)
        self._query = _as_str(params.get("query") or params.get("q"))
        self._page_size = _validated_page_size(
            params,
            MAX_COMMENT_RESULTS if self._content == CONTENT_COMMENTS else MAX_LIST_RESULTS,
        )
        self._unit_budget = max(1, _as_int(params.get("quota_units_per_run"), DEFAULT_RUN_UNIT_BUDGET))
        self._lookback_hours = max(1, _as_int(params.get("lookback_hours"), 24))

        self._auth: ApiKeyAuth | None = None
        self._client: httpx.AsyncClient | None = None
        self._units = _UnitLedger(self._unit_budget)
        self._uploads_playlist: str = ""
        self._mapping = MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=ctx.run_id,
        )

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Construct and check that the account describes a run that can happen.

        No I/O: not even the HTTP client is built. Both checks below exist because
        the alternative is discovering the same thing from the API for 1 to 100
        units, on every scheduled run, forever.
        """
        connector = cls(ctx, credentials)
        connector._check_target()
        connector._check_unit_budget()
        return connector

    def _check_target(self) -> None:
        if self._content == CONTENT_COMMENTS:
            if not (self._video or self._channel):
                raise ConnectorConfigurationError(
                    "youtube comments need params['video_id'] or params['channel_id']; "
                    "commentThreads.list addresses one video or one channel and has no "
                    "unbounded form",
                    connector=self.slug,
                    account_id=self.ctx.account_id,
                )
            return
        if not (self._channel or self._playlist or self._query):
            raise ConnectorConfigurationError(
                "youtube needs params['channel_id'], params['playlist_id'] or "
                "params['query']; there is no default feed, and search.list refuses "
                "an empty query",
                connector=self.slug,
                account_id=self.ctx.account_id,
            )

    def _check_unit_budget(self) -> None:
        """Refuse a budget too small to complete one discovery-plus-hydration cycle.

        A run that can pay for the search but not for the `videos.list` that makes
        its results usable spends 100 units and emits nothing, every time it is
        scheduled. That is not a slow sync, it is a quota leak, and it looks like
        a healthy zero-record run in every metric.
        """
        required = self._discovery_cost() + (
            0 if self._content == CONTENT_COMMENTS else UNIT_COST[VIDEOS_PATH]
        )
        if self._channel and self._content != CONTENT_COMMENTS and not self._playlist:
            required += UNIT_COST[CHANNELS_PATH]
        if self._unit_budget < required:
            raise ConnectorConfigurationError(
                f"params['quota_units_per_run'] is {self._unit_budget}, below the "
                f"{required} units one page of this mode costs; the run would spend "
                "its discovery call and never hydrate it",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"required_units": required, "daily_units": DAILY_QUOTA_UNITS},
            )

    def _discovery_cost(self) -> int:
        """Units for one discovery page in the configured mode.

        This is the number the whole module is organized around: 100 for search,
        1 for everything else.
        """
        return UNIT_COST[self._discovery_path()]

    def _discovery_path(self) -> str:
        if self._content == CONTENT_COMMENTS:
            return COMMENT_THREADS_PATH
        if self._playlist or self._channel:
            return PLAYLIST_ITEMS_PATH
        return SEARCH_PATH

    # ------------------------------------------------------------- lifecycle --

    async def authenticate(self) -> None:
        """Prove the key is present and build the client. Idempotent, no I/O.

        There is no session to establish -- an API key is a constant -- and the
        cheapest validating call still costs a unit. Spending one per run to learn
        what the first real request learns anyway is 20 units a day thrown away
        against a 10,000-unit budget, so this stage only checks that a credential
        exists.

        A missing secret is an `AuthError` rather than a configuration error: it
        is a credential row an operator has to fix, and `AuthError` is what flags
        the account `needs_reauth` (`docs/connector-spec.md` §2.1).
        """
        if self._auth is None:
            try:
                self._auth = ApiKeyAuth.from_credentials(
                    self.credentials,
                    secret_key="api_key",
                    # Google's frontend accepts the key either as `?key=` or in
                    # this header. The header is the only form that keeps the
                    # secret out of proxy logs and out of `Referer` on a redirect,
                    # which is why `connectors/auth/apikey.py` ships no
                    # query-parameter strategy at all.
                    header="X-Goog-Api-Key",
                )
            except KeyError as exc:
                raise AuthError(
                    "youtube account has no 'api_key' secret (YOUTUBE_API_KEY)",
                    connector=self.slug,
                    account_id=self.ctx.account_id,
                    cause=exc,
                ) from exc
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self.ctx.request_timeout_seconds,
                headers={
                    "User-Agent": self.ctx.user_agent,
                    "Accept": "application/json",
                    **self._auth.headers(),
                },
                # A redirect off an API endpoint is a captive portal or a consent
                # page, never a moved resource. Following it would send the key to
                # whatever answered and parse the result as a video list.
                follow_redirects=False,
            )

    async def aclose(self) -> None:
        """Release the client. Idempotent: `run()` closes in a `finally`."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ----------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Walk the chosen listing backwards, then hand it back forwards.

        The ledger is reset here rather than in `__init__` so a connector instance
        driven through `run()` twice gets a fresh slice each time; the budget is
        per run, not per object.
        """
        self._units = _UnitLedger(self._unit_budget)
        if self._content == CONTENT_COMMENTS:
            async for page in self._fetch_comments(cursor):
                yield page
            return
        async for page in self._fetch_videos(cursor):
            yield page

    # ------------------------------------------------------------ videos mode --

    async def _fetch_videos(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Discover ids newest-first, then hydrate and emit oldest-first.

        Hydration deliberately happens *during* the emit loop rather than during
        the descent. A page that is never reached -- because the ledger ran out,
        or `ctx.max_records` did -- then costs no `videos.list` call, and the
        pages that are reached are hydrated in the order they will be emitted, so
        a crash leaves no hydrated-but-unemitted work behind.
        """
        covered_from = _parse_moment(cursor.checkpoint.get(COVERED_FROM_KEY))
        covered_to = _parse_moment(cursor.checkpoint.get(COVERED_TO_KEY))
        source = await self._resolve_video_source(cursor)
        descent = await self._descend_videos(cursor, source, covered_from)

        ordered = list(reversed(descent.pages))
        # Carried forward only when the next run will resume *below* this descent.
        # A re-walking mode re-covers the parked interval itself, and treating the
        # parked value as already-committed there would jump the watermark over
        # pages this run has fetched but not yet yielded.
        carried = covered_to if descent.resumes_downward else None
        running = carried
        floor = _min_moment(
            covered_from if descent.resumes_downward else None,
            _oldest(descent.pages[-1]) if descent.pages else None,
        )
        emitted = 0

        for index, page in enumerate(ordered):
            skip = self._already_covered(page, covered_from, covered_to, descent)
            if not skip and not self._units.can_afford(UNIT_COST[VIDEOS_PATH]):
                # Stop before the call, not after it. Nothing is yielded for this
                # page, so the previous page's cursor is the resume point and the
                # unreached pages -- which are *newer* -- are picked up next run.
                return
            records = () if skip else await self._hydrate(page)
            running = _max_moment(running, *(item.published_at for item in page))
            emitted += len(records)
            final = index == len(ordered) - 1

            yield FetchPage(
                records=records,
                cursor=self._video_cursor(
                    cursor,
                    promote=descent.complete and (final or carried is None),
                    watermark=running,
                    covered_from=floor,
                    covered_to=running if descent.resumes_downward else None,
                ),
                # Only the first yielded page carries headers, and they are the
                # newest response's: see `_Descent.headers`.
                raw_headers=dict(descent.headers) if index == 0 else {},
            )
            if self.ctx.max_records is not None and emitted >= self.ctx.max_records:
                return

    async def _descend_videos(
        self, cursor: Cursor, source: tuple[str, str], covered_from: datetime | None
    ) -> _Descent:
        """Page the discovery endpoint backwards toward the watermark."""
        path, target = source
        downward = path == SEARCH_PATH
        watermark = cursor.watermark
        token: str | None = None
        pages: list[tuple[_Discovered, ...]] = []
        headers: Mapping[str, str] = {}
        complete = False
        cost = UNIT_COST[path]

        for index in range(self._descent_budget(MAX_DISCOVERY_PAGES)):
            if not self._units.can_afford(cost + UNIT_COST[VIDEOS_PATH]):
                # Refuse a discovery page the run could not afford to hydrate.
                # Spending 100 units on ids and stopping is the exact waste
                # `_check_unit_budget` exists to prevent at configuration time.
                break
            params = (
                self._search_params(target, cursor, covered_from, token)
                if downward
                else self._playlist_params(target, token)
            )
            body, headers = await self._get(path, params, cost=cost)

            items = _items(body, path)
            page = tuple(_discovered(item, path) for item in items)
            page = tuple(item for item in page if item.video_id)
            if page:
                pages.append(page)

            token = _as_str(body.get("nextPageToken")) or None
            if not items or token is None:
                # The listing ended: everything between here and the watermark has
                # been seen, so the descent is complete by exhaustion.
                complete = True
                break
            oldest = _oldest(page)
            if watermark is not None and oldest is not None and oldest <= watermark:
                # Crossed into ground a previous run already covered. Both pagers
                # here are strictly ordered by publish time, so the first record
                # at or below the watermark means the rest of the listing is too.
                complete = True
                break
            if index == 0 and not page:
                complete = True
                break

        return _Descent(
            pages=tuple(pages), complete=complete, resumes_downward=downward, headers=headers
        )

    def _already_covered(
        self,
        page: Sequence[_Discovered],
        covered_from: datetime | None,
        covered_to: datetime | None,
        descent: _Descent,
    ) -> bool:
        """Whether a whole page was emitted by an earlier, unfinished descent.

        Only meaningful for the re-walking modes, and only worth checking for a
        *whole* page: `videos.list` costs one unit for up to fifty ids, so
        hydrating forty-nine covered videos alongside one new one costs exactly
        what hydrating the one costs. Skipping a fully-covered page is where the
        units are actually saved during a resumed playlist walk.
        """
        if descent.resumes_downward or covered_from is None or covered_to is None:
            return False
        moments = [item.published_at for item in page if item.published_at is not None]
        if not moments or len(moments) != len(page):
            # A page with an unreadable timestamp cannot be proven covered, and
            # guessing would drop records silently. Hydrate it; dedup is cheap.
            return False
        return covered_from <= min(moments) and max(moments) <= covered_to

    async def _hydrate(self, page: Sequence[_Discovered]) -> tuple[RawRecord, ...]:
        """One `videos.list` for a whole discovery page. 1 unit, up to 50 videos.

        `videos.list` silently omits ids it will not serve -- private, deleted,
        region-blocked, or made members-only since discovery. A short response is
        therefore normal and must not be filed as a defect: those ids simply never
        become records, and nothing is sent to the DLQ for them.

        Records come back oldest-first *within* the page as well as between pages.
        Reddit's docstring argues the reason: half-ordered output is the kind of
        thing downstream code accidentally relies on being total.
        """
        ids = [item.video_id for item in page]
        params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(ids),
            "maxResults": str(min(len(ids), MAX_LIST_RESULTS)),
        }
        body, _ = await self._get(VIDEOS_PATH, params, cost=UNIT_COST[VIDEOS_PATH])
        fingerprint = _fingerprint(VIDEOS_PATH, params)
        fetched_at = utcnow()
        videos = [item for item in _items(body, VIDEOS_PATH) if _as_str(item.get("id"))]
        videos.sort(key=lambda item: (_published_at(item) or datetime.min.replace(tzinfo=UTC)))
        return tuple(
            self._to_record(
                video,
                native_id=_as_str(video.get("id")),
                source_url=_watch_url(video.get("id")),
                fingerprint=fingerprint,
                fetched_at=fetched_at,
            )
            for video in videos
        )

    def _video_cursor(
        self,
        cursor: Cursor,
        *,
        promote: bool,
        watermark: datetime | None,
        covered_from: datetime | None,
        covered_to: datetime | None,
    ) -> Cursor:
        """A legal restart point for one emitted page.

        Two shapes, and the difference between them is the difference between a
        replay and a hole. A *complete* descent is contiguous from the previous
        watermark upwards, so each page may carry its own newest timestamp and a
        crash re-fetches exactly what was not emitted. A *truncated* one leaves the
        watermark where it was and parks the interval it did cover, because the
        records between the deepest page reached and the old watermark have not
        been seen at all.
        """
        if promote:
            return Cursor(
                watermark=watermark,
                page_token=None,
                checkpoint=self._checkpoint(),
            )
        return Cursor(
            watermark=cursor.watermark,
            # No token is parked: a `playlistItems` token encodes a position and
            # the position shifts when the channel uploads, so replaying one after
            # a new upload skips a page's worth of videos. Re-walking costs a unit
            # a page and cannot skip anything (§4.1 rule 4 makes tokens advisory).
            page_token=None,
            checkpoint=self._checkpoint(
                **{COVERED_FROM_KEY: _isoformat(covered_from), COVERED_TO_KEY: _isoformat(covered_to)}
            ),
        )

    async def _resolve_video_source(self, cursor: Cursor) -> tuple[str, str]:
        """Decide which discovery endpoint this run uses, resolving a channel once.

        Returns `(path, target)`. The channel -> uploads-playlist lookup is the
        only I/O here and it happens at most once per account: the answer is
        immutable, so it is cached in the checkpoint and re-read on later runs.
        """
        if self._playlist:
            return PLAYLIST_ITEMS_PATH, self._playlist
        if self._channel:
            cached = _as_str(cursor.checkpoint.get(UPLOADS_PLAYLIST_KEY))
            if cached:
                self._uploads_playlist = cached
                return PLAYLIST_ITEMS_PATH, cached
            self._uploads_playlist = await self._lookup_uploads_playlist()
            return PLAYLIST_ITEMS_PATH, self._uploads_playlist
        return SEARCH_PATH, self._query

    async def _lookup_uploads_playlist(self) -> str:
        """`channels.list` -> `contentDetails.relatedPlaylists.uploads`. One unit."""
        params = {"part": "contentDetails", "maxResults": "1"}
        if self._channel.startswith("@"):
            # `forHandle` is the supported lookup for the `@name` form.
            # `forUsername` addresses the long-retired legacy username namespace
            # and answers an empty list for a modern handle, which reads exactly
            # like "this channel does not exist".
            params["forHandle"] = self._channel
        else:
            params["id"] = self._channel
        body, _ = await self._get(CHANNELS_PATH, params, cost=UNIT_COST[CHANNELS_PATH])
        for item in _items(body, CHANNELS_PATH):
            details = item.get("contentDetails")
            related = details.get("relatedPlaylists") if isinstance(details, Mapping) else None
            uploads = _as_str(related.get("uploads")) if isinstance(related, Mapping) else ""
            if uploads:
                return uploads
        raise ConnectorConfigurationError(
            f"no channel resolves to {self._channel!r}, or it publishes no uploads "
            "playlist; the API answered with an empty item list rather than a 404",
            connector=self.slug,
            account_id=self.ctx.account_id,
        )

    def _search_params(
        self,
        query: str,
        cursor: Cursor,
        covered_from: datetime | None,
        token: str | None,
    ) -> dict[str, str]:
        """One `search.list` page. 100 units -- the expensive call in this module.

        `publishedBefore` is what makes a parked search descent resumable without
        re-walking: it is set to the floor an earlier run reached, so the next run
        starts exactly below it. On a fresh descent it is left open and the
        endpoint answers from now backwards.
        """
        params = {
            "part": "snippet",
            "q": query,
            # Without `type=video` the response mixes channels and playlists, whose
            # `id` objects carry no `videoId`, and the hydration call would be
            # built from ids that do not exist.
            "type": "video",
            # The only order that is a timeline. `relevance` and `viewCount`
            # re-rank between requests, so page 2 of one run and page 2 of the next
            # describe different sets and no watermark can span them.
            "order": "date",
            "maxResults": str(self._page_size),
        }
        start = cursor.watermark or utcnow() - timedelta(hours=self._lookback_hours)
        params["publishedAfter"] = _rfc3339(start)
        if covered_from is not None:
            params["publishedBefore"] = _rfc3339(covered_from)
        if token:
            params["pageToken"] = token
        return params

    def _playlist_params(self, playlist_id: str, token: str | None) -> dict[str, str]:
        params = {
            # `contentDetails` is not optional here and costs nothing extra:
            # `snippet.publishedAt` on a playlist item is when the item was *added
            # to the playlist*, which for a re-added video is not the upload time.
            # `contentDetails.videoPublishedAt` is the upload time, and it is what
            # the watermark has to be built from.
            "part": "snippet,contentDetails,status",
            "playlistId": playlist_id,
            "maxResults": str(self._page_size),
        }
        if token:
            params["pageToken"] = token
        return params

    # ---------------------------------------------------------- comments mode --

    async def _fetch_comments(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Comment threads, newest-first from the provider, emitted oldest-first.

        No checkpoint is parked here and that is a decision, not an omission. A
        comment listing costs 1 unit per hundred threads and there is no hydration
        step, so re-walking an unfinished descent from the top next run costs a
        unit or two and dedup collapses the re-read. Parking a `pageToken` would
        buy those units back at the price of replaying a token whose position
        shifts every time somebody comments.
        """
        descent = await self._descend_comments(cursor)
        ordered = list(reversed(descent.pages))
        running: datetime | None = None
        emitted = 0

        for index, page in enumerate(ordered):
            running = _max_moment(running, *(_comment_published_at(item) for item in page))
            final = index == len(ordered) - 1
            emitted += len(page)
            yield FetchPage(
                records=page,
                cursor=Cursor(
                    # Promoted only when the descent actually reached the previous
                    # watermark or the end of the listing. Promoting a truncated
                    # descent would put the watermark above threads nobody fetched.
                    watermark=running if (descent.complete and final) else cursor.watermark,
                    page_token=None,
                    checkpoint=self._checkpoint(),
                ),
                raw_headers=dict(descent.headers) if index == 0 else {},
            )
            if self.ctx.max_records is not None and emitted >= self.ctx.max_records:
                return

    async def _descend_comments(self, cursor: Cursor) -> _Descent:
        watermark = cursor.watermark
        token: str | None = None
        pages: list[tuple[RawRecord, ...]] = []
        headers: Mapping[str, str] = {}
        complete = False
        cost = UNIT_COST[COMMENT_THREADS_PATH]

        for _ in range(self._descent_budget(MAX_COMMENT_PAGES)):
            if not self._units.can_afford(cost):
                break
            params = self._comment_params(token)
            body, headers = await self._get(COMMENT_THREADS_PATH, params, cost=cost)

            items = _items(body, COMMENT_THREADS_PATH)
            fingerprint = _fingerprint(COMMENT_THREADS_PATH, params)
            fetched_at = utcnow()
            page = tuple(
                self._to_record(
                    thread,
                    native_id=_comment_identity(thread),
                    source_url=_comment_url(_top_level(thread)) or None,
                    fingerprint=fingerprint,
                    fetched_at=fetched_at,
                )
                for thread in reversed(items)
            )
            if page:
                pages.append(page)

            token = _as_str(body.get("nextPageToken")) or None
            if not items or token is None:
                complete = True
                break
            newest = _max_moment(*(_comment_published_at(record) for record in page))
            if watermark is not None and newest is not None and newest <= watermark:
                # The *newest* thread on the page, not the oldest. `order=time`
                # sorts threads by activity, so a thread that received a reply can
                # sit above older ones; stopping at the first already-seen record
                # would end the descent one page early and lose whatever sat below
                # the bubbled thread. Requiring the whole page to be old costs one
                # extra unit and removes the failure.
                complete = True
                break

        return _Descent(
            pages=tuple(pages), complete=complete, resumes_downward=False, headers=headers
        )

    def _comment_params(self, token: str | None) -> dict[str, str]:
        params = {
            "part": "snippet",
            "order": "time",
            # Default is `html`, which puts `<a href="…">` and `&#39;` into
            # `textDisplay` and therefore into the cleaned body, the embedding and
            # eventually a sentence quoted in a report. Asking the provider for
            # plain text is better than stripping markup we asked to be given.
            "textFormat": "plainText",
            "maxResults": str(self._page_size),
        }
        if self._video:
            params["videoId"] = self._video
        else:
            # Threads across everything on the channel, in one pager. The
            # alternative -- a `commentThreads` call per discovered video -- is one
            # unit per video, which is the N+1 this connector avoids everywhere
            # else.
            params["allThreadsRelatedToChannelId"] = self._channel
        if token:
            params["pageToken"] = token
        # `replies` is deliberately not requested. The reply preview a thread
        # carries is capped at a handful with no token of its own, so emitting it
        # would produce a record set that silently changes shape between polls and
        # can never be completed without a `comments.list` per thread.
        return params

    # --------------------------------------------------------------- plumbing --

    def _descent_budget(self, ceiling: int) -> int:
        budget = ceiling
        if self.ctx.max_pages is not None:
            budget = min(budget, max(1, self.ctx.max_pages))
        return budget

    def _checkpoint(self, **extra: Any) -> dict[str, Any]:
        """Checkpoint for an emitted cursor, always carrying the cached playlist id.

        Built rather than merged because the promoting branch of `_video_cursor`
        constructs a `Cursor` directly instead of going through
        `Cursor.advanced_to`, and dropping the cached id there would buy back the
        `channels.list` unit on every single run.
        """
        state: dict[str, Any] = {}
        if self._uploads_playlist:
            state[UPLOADS_PLAYLIST_KEY] = self._uploads_playlist
        state.update({key: value for key, value in extra.items() if value is not None})
        return state

    def _to_record(
        self,
        payload: Mapping[str, Any],
        *,
        native_id: str,
        source_url: str | None,
        fingerprint: str,
        fetched_at: datetime,
    ) -> RawRecord:
        """Wrap one API item verbatim, with the bytes that will be archived.

        Per-record provider bytes do not exist -- one response carries up to fifty
        items -- so they are synthesized here in a canonical encoding, once.
        `lineage.raw_sha256` is then taken over exactly the bytes the runtime PUTs
        to R2, which is what content-addressing requires; re-serializing later, in
        another process on another library version, is what `RawRecord.raw_bytes`
        exists to prevent.
        """
        return RawRecord(
            native_id=native_id or _unidentified(payload),
            payload=payload,
            fetched_at=fetched_at,
            raw_bytes=json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
            content_type="application/json",
            source_url=source_url or None,
            request_fingerprint=fingerprint,
        )

    async def _get(
        self, path: str, params: Mapping[str, str], *, cost: int
    ) -> tuple[Mapping[str, Any], dict[str, str]]:
        """Issue one API call: acquire a slot, charge the ledger, map the failure.

        Every outbound request in this module goes through here, which is what
        makes "a slot is always acquired and a unit is always charged" a property
        of the code rather than of four call sites remembering. No retry and no
        sleep -- `docs/connector-spec.md` §1 gives backoff to the runtime, because
        a connector that retries privately makes the shared limiter's accounting
        wrong and hides the failure from metrics.
        """
        client = self._client
        if client is None:  # pragma: no cover -- run() always authenticates first
            raise PermanentError(
                "youtube fetch ran before authenticate(); there is no HTTP client",
                connector=self.slug,
                account_id=self.ctx.account_id,
            )

        await self.acquire_slot(f"{self._base_url}{path}")
        self._units.charge(cost)
        try:
            response = await client.get(path, params=params)
        except httpx.TransportError as exc:
            raise TransientError(
                "youtube is unreachable",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__, "endpoint": path},
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            raise PermanentError(
                "the youtube request could not be issued",
                connector=self.slug,
                account_id=self.ctx.account_id,
                details={"reason": type(exc).__name__, "endpoint": path},
                cause=exc,
            ) from exc

        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise self._failure(response, path)

        try:
            body = response.json()
        except ValueError as exc:
            # Usually an error page from a proxy in front of the API. The body is
            # not attached: it can echo the request, and the request carries the
            # API key in a header (§1 forbids logging either).
            raise PermanentError(
                f"youtube returned {len(response.content)} bytes that are not JSON",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=response.status_code,
                details={"endpoint": path},
                cause=exc,
            ) from exc
        if not isinstance(body, Mapping):
            raise PermanentError(
                f"youtube returned a JSON {type(body).__name__} where an object was "
                "expected; the response shape changed",
                connector=self.slug,
                status_code=response.status_code,
                details={"endpoint": path},
            )
        return body, _rate_limit_headers(response.headers)

    def _failure(self, response: httpx.Response, path: str) -> ConnectorError:
        """Classify a Data API error by its `reason`, not by its status alone.

        The status is nearly uninformative here: Google answers an exhausted daily
        quota, a per-second throttle, a disabled comment section, a wrong API key
        and a project that never enabled the API all with **403**. Reading the
        status alone means one of two expensive mistakes -- flagging a working
        account `needs_reauth` when the quota simply ran out, or backing off for
        seconds against a wall that does not move until midnight Pacific.
        """
        status = response.status_code
        reason = _error_reason(response)
        details = {"endpoint": path, "reason": reason or "unknown"}

        if reason in _QUOTA_REASONS:
            return QuotaError(
                "youtube daily quota units are exhausted",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                reset_at=_next_quota_reset(utcnow()),
                details={**details, "units_spent_this_run": self._units.spent},
            )
        if reason in _THROTTLE_REASONS or status == httpx.codes.TOO_MANY_REQUESTS:
            hint = self.parse_rate_limit(response.headers)
            wait = hint.retry_after_seconds if hint is not None else None
            if wait is not None and wait > QUOTA_WAIT_THRESHOLD_SECONDS:
                return QuotaError(
                    "youtube asked for a long wait",
                    connector=self.slug,
                    account_id=self.ctx.account_id,
                    status_code=status,
                    retry_after_seconds=wait,
                    details=details,
                )
            return TransientError(
                "youtube throttled the request",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if reason in _PROJECT_REASONS or reason in _RESOURCE_REASONS or status == httpx.codes.NOT_FOUND:
            # The credential works; the *configuration* does not -- the API is not
            # enabled on the project, the key is restricted to another referrer,
            # or the channel has comments turned off. Filing these as `AuthError`
            # would send an operator to rotate a key that was never the problem.
            return ConnectorConfigurationError(
                f"youtube refused this request ({reason or status}); the credential is "
                "accepted but the project or the resource is not usable as configured",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if reason in _KEY_REASONS or status == httpx.codes.UNAUTHORIZED:
            return AuthError(
                "youtube rejected the API key",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        if status >= httpx.codes.INTERNAL_SERVER_ERROR:
            return TransientError(
                f"youtube returned {status}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                status_code=status,
                details=details,
            )
        return PermanentError(
            f"youtube rejected the request ({reason or status}); it will be rejected "
            "identically on retry",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=status,
            details=details,
        )

    # ------------------------------------------------------------- normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one video or comment thread onto a Signal, or drop it."""
        payload = record.payload
        kind = _as_str(payload.get("kind"))
        if kind == "youtube#commentThread":
            return self._normalize_comment(record)
        if kind == "youtube#video":
            return self._normalize_video(record)
        raise NormalizationError(
            f"unsupported YouTube resource kind {kind!r}; this connector maps "
            "youtube#video and youtube#commentThread",
            native_id=record.native_id,
            connector=self.slug,
        )

    def _normalize_video(self, record: RawRecord) -> Signal | None:
        payload = record.payload
        if not isinstance(payload.get("snippet"), Mapping):
            raise NormalizationError(
                "video carries no 'snippet'; the requested part was not returned",
                native_id=record.native_id,
                connector=self.slug,
            )
        # The runtime keys the R2 object and the Kafka partition off
        # `RawRecord.native_id`, while every store keys off `Signal.id`, which is
        # derived from `id`. If those disagreed the same video would exist under
        # two identities, so the disagreement is caught here rather than
        # discovered as duplicate rows months later.
        if _as_str(payload.get("id")) != record.native_id:
            raise NormalizationError(
                "payload video id does not match the fetched record's native_id",
                native_id=record.native_id,
                connector=self.slug,
            )

        signal = _VIDEO_FIELDS.to_signal(record, self._mapping)
        signal.media = _video_media(payload)
        return signal

    def _normalize_comment(self, record: RawRecord) -> Signal | None:
        payload = record.payload
        top_level = _top_level(payload)
        if top_level is None:
            raise NormalizationError(
                "comment thread carries no 'snippet.topLevelComment'",
                native_id=record.native_id,
                connector=self.slug,
            )
        snippet = top_level.get("snippet")
        text = _as_str(snippet.get("textDisplay")) if isinstance(snippet, Mapping) else ""
        text = text or (
            _as_str(snippet.get("textOriginal")) if isinstance(snippet, Mapping) else ""
        )
        if not text:
            # A comment with no text carries no observation. Dropping is right
            # rather than a DLQ record: the payload is well-formed, it simply says
            # nothing, and emitting it would put an empty document into the
            # embedding queue and the search index.
            return None
        return _COMMENT_FIELDS.to_signal(record, self._mapping)


# --------------------------------------------------------------------------- #
# Payload helpers
# --------------------------------------------------------------------------- #


def _items(body: Mapping[str, Any], path: str) -> list[Mapping[str, Any]]:
    """The `items` array.

    An absent `items` is zero results, which the Data API returns for a query that
    matched nothing. A present but non-list value is a shape change and raises,
    because everything downstream would then silently iterate a string.
    """
    items = body.get("items")
    if items is None:
        return []
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise PermanentError(
            f"youtube {path} response has a non-list 'items'; the shape changed"
        )
    return [item for item in items if isinstance(item, Mapping)]


def _discovered(item: Mapping[str, Any], path: str) -> _Discovered:
    """Pull an id and a publish time out of one discovery item.

    The two endpoints spell both differently, and the `playlistItems` spelling is
    the trap: `snippet.publishedAt` there is when the video was added to the
    playlist. For an uploads playlist that is usually the upload time and
    occasionally is not, and the occasional case walks the watermark to the wrong
    place with nothing to show for it.
    """
    if path == SEARCH_PATH:
        identifier = item.get("id")
        video_id = (
            _as_str(identifier.get("videoId")) if isinstance(identifier, Mapping) else ""
        )
        return _Discovered(video_id=video_id, published_at=_published_at(item))

    details = item.get("contentDetails")
    snippet = item.get("snippet")
    video_id = _as_str(details.get("videoId")) if isinstance(details, Mapping) else ""
    if not video_id and isinstance(snippet, Mapping):
        resource = snippet.get("resourceId")
        video_id = _as_str(resource.get("videoId")) if isinstance(resource, Mapping) else ""
    moment = None
    if isinstance(details, Mapping):
        moment = _parse_moment(details.get("videoPublishedAt"))
    if moment is None:
        moment = _published_at(item)

    status = item.get("status")
    privacy = _as_str(status.get("privacyStatus")) if isinstance(status, Mapping) else ""
    if privacy in {"private", "privacyStatusUnspecified"}:
        # A private entry in a public playlist is a tombstone: `videos.list` will
        # not serve it, so carrying its id would spend a slot in the hydration
        # batch for a video that can never arrive.
        return _Discovered(video_id="", published_at=moment)
    return _Discovered(video_id=video_id, published_at=moment)


def _published_at(item: Mapping[str, Any]) -> datetime | None:
    snippet = item.get("snippet")
    if not isinstance(snippet, Mapping):
        return None
    return _parse_moment(snippet.get("publishedAt"))


def _top_level(thread: Mapping[str, Any]) -> Mapping[str, Any] | None:
    snippet = thread.get("snippet")
    if not isinstance(snippet, Mapping):
        return None
    top = snippet.get("topLevelComment")
    return top if isinstance(top, Mapping) else None


def _comment_identity(thread: Mapping[str, Any]) -> str:
    """Rule 1: the top-level comment's own id, verbatim.

    Falls back to the thread id, which is the same string today. A thread with
    neither is filed under a digest of itself so two arrivals of the same broken
    payload are recognisable as one DLQ entry; it never becomes a Signal, because
    `item_id` is required in the field map.
    """
    top = _top_level(thread)
    identifier = _as_str(top.get("id")) if top is not None else ""
    return identifier or _as_str(thread.get("id")) or _unidentified(thread)


def _comment_published_at(record: RawRecord) -> datetime | None:
    """Event time of one thread, for pagination arithmetic only.

    Never raises. A thread with an unreadable timestamp still has to reach
    `normalize()`, which is the stage allowed to DLQ it with its identity
    attached, but it must not be allowed to poison the watermark.
    """
    top = _top_level(record.payload)
    snippet = top.get("snippet") if top is not None else None
    if not isinstance(snippet, Mapping):
        return None
    return _parse_moment(snippet.get("publishedAt"))


def _video_media(payload: Mapping[str, Any]) -> list[MediaRef]:
    """The video itself, plus its poster frame.

    Assigned after mapping rather than through `MediaMap`, which addresses a list
    and this payload carries neither the video nor the thumbnail in one.
    `transcript_ref` stays `None`: there is no lawful endpoint for the transcript
    of a video we do not own (`docs/connector-spec.md` §9.1), and a null pointer
    is an honest gap where a scraped one would be a liability.
    """
    refs: list[MediaRef] = []
    video_id = _as_str(payload.get("id"))
    details = payload.get("contentDetails")
    duration = (
        _duration_seconds(details.get("duration")) if isinstance(details, Mapping) else None
    )
    if video_id:
        refs.append(
            MediaRef(kind=MediaKind.VIDEO, source_url=_watch_url(video_id), duration_s=duration)
        )
    snippet = payload.get("snippet")
    thumbnail = _best_thumbnail(snippet) if isinstance(snippet, Mapping) else ""
    if thumbnail:
        refs.append(MediaRef(kind=MediaKind.IMAGE, source_url=thumbnail))
    return refs


def _best_thumbnail(snippet: Mapping[str, Any]) -> str:
    """The largest thumbnail the payload offers.

    Resolution tiers are populated inconsistently -- `maxres` exists only for
    videos uploaded above 720p -- so the preference list falls through rather than
    addressing one key and finding nothing.
    """
    thumbnails = snippet.get("thumbnails")
    if not isinstance(thumbnails, Mapping):
        return ""
    for name in _THUMBNAIL_PREFERENCE:
        entry = thumbnails.get(name)
        if isinstance(entry, Mapping):
            url = _as_str(entry.get("url"))
            if url:
                return url
    return ""


def _duration_seconds(value: Any) -> float | None:
    """Parse `contentDetails.duration`, which is ISO-8601 rather than seconds."""
    text = _as_str(value)
    if not text:
        return None
    match = _ISO_DURATION.match(text)
    if match is None:
        return None
    parts = {key: int(raw) for key, raw in match.groupdict(default="0").items()}
    return float(
        parts["days"] * 86_400 + parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"]
    )


def _error_reason(response: httpx.Response) -> str | None:
    """The `error.errors[0].reason` of a Data API error body, and nothing else.

    Character-checked and length-capped before it is allowed into an error
    message, for the same reason `connectors/auth/oauth.py` does it to `error`: a
    provider under load can put anything in a body field, including an echo of the
    request that caused it. `error.status` is read as a fallback because the newer
    error shape omits `errors[]` for some failures.
    """
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, Mapping):
        return None
    error = payload.get("error")
    if not isinstance(error, Mapping):
        return None
    errors = error.get("errors")
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        for entry in errors:
            if isinstance(entry, Mapping):
                reason = _as_str(entry.get("reason"))
                if reason and _SAFE_REASON.match(reason):
                    return reason
    status = _as_str(error.get("status"))
    return status if status and _SAFE_REASON.match(status) else None


def _next_quota_reset(now: datetime) -> float | None:
    """Epoch seconds of the next midnight Pacific, when the unit budget refills.

    Returns `None` when the platform has no tzdata rather than approximating with
    a fixed offset: an eight-hour guess that is an hour wrong across a DST
    boundary reschedules the run before the quota resets, and a `QuotaError` with
    no `reset_at` is scheduled by the runtime's own backoff, which is the honest
    outcome.
    """
    try:
        pacific = ZoneInfo(QUOTA_RESET_ZONE)
    except ZoneInfoNotFoundError:  # pragma: no cover -- depends on the host image
        return None
    local = now.astimezone(pacific)
    tomorrow = (local + timedelta(days=1)).date()
    return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=pacific).timestamp()


def _oldest(page: Sequence[_Discovered]) -> datetime | None:
    moments = [item.published_at for item in page if item.published_at is not None]
    return min(moments) if moments else None


def _max_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return max(present) if present else None


def _min_moment(*moments: datetime | None) -> datetime | None:
    present = [moment for moment in moments if moment is not None]
    return min(present) if present else None


def _parse_moment(value: Any) -> datetime | None:
    """Read an RFC 3339 timestamp, or a checkpoint's ISO string, tolerantly.

    A checkpoint we cannot read is not worth failing a run over: the watermark is
    still valid, so the descent simply re-covers ground that dedup collapses.
    """
    if value is None:
        return None
    try:
        moment = to_utc_datetime(value)
    except ValueError:
        return None
    return moment


def _isoformat(moment: datetime | None) -> str | None:
    return moment.astimezone(UTC).isoformat() if moment is not None else None


def _rfc3339(moment: datetime) -> str:
    """`publishedAfter` / `publishedBefore`, as the Data API spells them.

    Sub-second precision is dropped: the endpoint accepts it, but a watermark
    round-tripped through a checkpoint at microsecond precision and back into a
    query is a boundary that lands differently on each run for no benefit.
    """
    return moment.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in _RATE_LIMIT_HEADERS}


def _fingerprint(path: str, params: Mapping[str, str]) -> str:
    """Hash of endpoint plus normalized params -- never the credential.

    `lineage.request_fingerprint` is what makes a fetch reproducible: it names the
    exact request that produced a record without naming who made it. The API key
    travels in a header, so there is nothing here to omit.
    """
    canonical = urlencode(sorted(params.items()))
    return hashlib.sha256(f"GET {path}?{canonical}".encode()).hexdigest()[:32]


def _unidentified(payload: Mapping[str, Any]) -> str:
    material = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"unidentified:{hashlib.sha256(material).hexdigest()}"


def _as_str(value: Any) -> str:
    """Render a scalar as a stripped string; `""` for anything unusable.

    Booleans are refused rather than stringified: `"True"` is never a video id or
    a page token, and letting one through turns a type confusion into a
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
