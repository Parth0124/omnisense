"""RSS/Atom feeds: the Phase 1 reference connector (`docs/connector-spec.md` §11).

RSS is the source with the least API and the most edge cases, which is why it is
the one other connector authors should read first. There is no auth, no
pagination, no provider quota and no rate-limit header -- and consequently
nothing to hide behind. Every decision below is a decision about politeness,
identity or failure isolation, and each is the decision a connector against a
real API would also have to make, only here it is not obscured by an SDK.

**One account, many origins.** Feed URLs come from `ctx.params`, seeded from
`RSS_FEED_URLS` in `.env.example`, so a single account polls dozens of unrelated
servers. That is why every request takes a *per-host* bucket via
`BaseConnector.host_rate_limit_key()` in addition to the connector and account
buckets: a connector-wide limit would either throttle a thousand hosts as though
they were one server, or let sixty requests a minute land on one hobbyist's VPS.

**Conditional GET is the point.** Every request carries `If-None-Match` /
`If-Modified-Since` built from per-feed validators stored in
`Cursor.checkpoint["feeds"]`, and a `304 Not Modified` is a *successful* poll
with zero records, never an error. This is the single largest politeness win
available to a feed reader and most clients skip it. It also makes failure cheap
in a way the rest of this module depends on: a run that dies half-way is retried
by the runtime at the cost of one 304 per already-synced feed.

**Identity: rules 1 and 2 of `docs/signal-model.md` §4.1, and rule 3 only under
protest.** The entry's `<guid>`/`<id>` is used verbatim when present (rule 1);
otherwise `native_id` is the sha256 of the canonicalized `<link>` (rule 2). §4.1
requires a connector that can reach rule 3 to say so, so: an entry carrying
neither a guid nor a canonicalizable link falls to
`sha256(platform | author | timestamp | simhash64(cleaned_text))`, which means
its identity depends on the output of `extract_readable()` and a change to the
cleaner forks it. That path is rare and deliberately not smoothed over.

Identity is derived once, here in `fetch()`, and carried into the payload under
the `omnisense` envelope so `normalize()` reads it back as rule 1. Deriving it
twice -- once for `RawRecord.native_id`, which becomes the Kafka reference
(`docs/connector-spec.md` §2.6), and again inside the field map for
`Signal.lineage.native_id` -- would let the message on the bus and the Signal it
points at disagree about which item they are.

**Entries with no date go to the DLQ.** `Signal.timestamp` is event time at the
source and every trend and forecast agent keys off it exclusively; substituting
`fetched_at` would file a two-year-old post as today's news and corrupt exactly
the aggregate the field exists for. The feed's own `<lastBuildDate>` is not a
substitute either -- it would stamp every undated item with the same instant and
move them all forward on every poll. So such an entry raises
`NormalizationError` and lands in the DLQ, per `docs/connector-spec.md` §11.2
step 8. The cost is honest and bounded: a feed that never dates its entries
produces one DLQ record per entry per poll for as long as the entry stays in the
feed's trailing window, which is the pressure that gets the feed fixed or
removed. Dropping them silently would instead leave the operator believing the
feed is being ingested.

**One dead feed is not a dead run.** A feed that times out, 404s, or returns
something that is not a feed becomes a single marker record whose `normalize()`
raises `NormalizationError` -- a DLQ record, attributable to that feed URL --
and the loop moves to the next feed. Aborting would mean one unreachable host
stops the other forty-nine from syncing on every poll until a human notices. The
15-minute scheduler cadence is that feed's retry; it does not need the runtime's
backoff. The one thing that *is* raised is total failure: if no feed produced a
usable response, the first error is re-raised, because "every feed failed" is a
fact about us, not about them, and reporting success over an outage is worse
than failing.

Not done here, deliberately: `robots.txt` is not consulted (it is one fetch per
host per day, shared across every connector, so it belongs to a runtime-level
policy cache rather than to each feed poll); linked articles are never fetched
for full text (that is a second request per entry against a publisher who
offered a *feed*, and it would make `normalize()` depend on what the network
answered at that instant, which `docs/connector-spec.md` §2.4 forbids). The
second is what `Content.truncated` is for, and this connector sets it.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from time import struct_time
from typing import Any, ClassVar, Final, Self
from urllib.parse import urlsplit

import feedparser
import httpx

from models.enums import AuthType, Platform, SourceCategory
from models.signal import Signal
from connectors.base import BaseConnector
from connectors.exceptions import (
    ConnectorConfigurationError,
    ConnectorError,
    NormalizationError,
    PermanentError,
    TransientError,
)
from connectors.normalize.html import canonicalize_url, extract_readable
from connectors.normalize.mapper import (
    FieldMap,
    FieldSpec,
    MappingContext,
    MediaMap,
    derive_native_id,
    to_utc_datetime,
)
from connectors.protocol import (
    Credentials,
    Cursor,
    FetchPage,
    RateLimitPolicy,
    RawRecord,
    SyncContext,
)

__all__ = [
    "ENVELOPE_KEY",
    "EXCERPT_CHAR_THRESHOLD",
    "FEED_PARAM_KEYS",
    "MAX_FEEDS",
    "MAX_FEED_BYTES",
    "RssConnector",
]

_SLUG: Final = "rss"

FEED_PARAM_KEYS: Final[tuple[str, ...]] = ("feeds", "feed_urls", "rss_feed_urls")
"""Accepted spellings of the feed list in `SyncContext.params`.

`docs/connector-spec.md` §11.2 step 2 names `feeds`; `.env.example` seeds the
same list as `RSS_FEED_URLS`. Both are accepted because the alternative is an
operator whose feeds are configured, spelled the other way, and silently never
polled -- a failure that looks exactly like a healthy connector with nothing to
say.
"""

ENVELOPE_KEY: Final = "omnisense"
"""Payload key under which this connector stores what it computed itself.

Everything else in `RawRecord.payload` is feedparser's view of the provider's
bytes, verbatim. Three values have to live somewhere and cannot come from the
entry: the feed URL (without it a DLQ record is unattributable -- an entry does
not name the feed it came from), the derived `native_id`, and the host-scoped
author identity. Namespacing them under one obviously-ours key keeps the
boundary between "what the publisher said" and "what we decided" legible in a
payload someone is reading at 3am.
"""

EXCERPT_CHAR_THRESHOLD: Final = 400
"""Cleaned-body length below which a summary-sourced entry is `truncated`.

The figure is `docs/connector-spec.md` §11.2 step 6's own threshold for "this is
a teaser, not an article". `Content.truncated` caps the `content_integrity`
component of confidence (`docs/signal-model.md` §3.5), so the flag is the
difference between an excerpt being trusted like a body and being discounted
like one.
"""

MAX_FEEDS: Final = 200
"""Default ceiling on feeds per account, overridable with `params["max_feeds"]`.

A ceiling exists because one run holds one worker: a thousand feeds at one
request each is a run nobody can schedule around. Exceeding it raises rather
than truncating -- silently polling the first two hundred of three hundred
configured feeds is invisible data loss, and the operator would have no way to
learn which hundred stopped.
"""

MAX_FEED_BYTES: Final = 32 * 1024 * 1024
"""Largest feed document this connector will hand to feedparser.

Bounds the *parse*, not the download: feedparser holds several derived copies of
the document, so a hostile or broken 500 MB response is an OOM in a worker
shared with other connectors. The download itself is bounded only by the
request timeout; capping that needs a streaming read the runtime does not
currently ask for.
"""

_ACCEPT: Final = (
    "application/atom+xml, application/rss+xml;q=0.9, "
    "application/xml;q=0.8, text/xml;q=0.8, */*;q=0.5"
)

_EPOCH: Final = datetime.min.replace(tzinfo=UTC)

_TIMESTAMP_PATHS: Final[tuple[str, ...]] = (
    "published_parsed",
    "updated_parsed",
    "created_parsed",
    "published",
    "updated",
    "created",
)
"""Where an entry's event time comes from, best interpretation first.

All three of feedparser's *parsed* forms are tried before any raw string,
because a feed that carries an unparseable `pubDate` and a perfectly good
`<atom:updated>` is common, and `FieldSpec` returns the first *present* value:
ordering `published` above `updated_parsed` would let the broken field shadow
the working one and send the entry to the DLQ.

`published` outranks `updated` because `Signal.timestamp` is when the
observation happened, not when the publisher last touched its CMS.
"""

_TIMESTAMP_SPEC: Final = FieldSpec.at(*_TIMESTAMP_PATHS)
"""Shared by the field map and by `fetch()`'s ordering and cutoff logic.

One object rather than two identical declarations: `fetch()` sorts and filters
on the timestamp it computes, and `normalize()` stamps the one the map computes.
If those two could drift, a feed would be ordered by one clock and emitted with
another, and the watermark would stop meaning anything.
"""


def _entry_timestamp(payload: Mapping[str, Any]) -> datetime | None:
    """Event time for one entry, or `None` when there is none we can use.

    Returns `None` for both "absent" and "unparseable" on purpose: `fetch()`
    only needs to know whether the entry can be ordered. The *distinction*
    between the two matters to a human triaging the DLQ, and the field map
    preserves it there.
    """
    raw = _TIMESTAMP_SPEC.resolve(payload, name="timestamp", strict=False)
    if raw is None:
        return None
    try:
        return to_utc_datetime(raw)
    except ValueError:
        return None


def _category_terms(value: Any) -> list[str] | None:
    """Flatten feedparser's `tags` into a list of category strings.

    Kept flat because `Signal.metadata` is a Postgres `jsonb` column, a Qdrant
    payload and an OpenSearch object at once, and every nested path becomes a
    field in the last of those (`docs/signal-model.md` §2). The `scheme` and
    `label` siblings are dropped: no consumer reads them, and carrying them
    would spend a mapping field per feed taxonomy.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    terms = [
        item["term"].strip()
        for item in value
        if isinstance(item, Mapping)
        and isinstance(item.get("term"), str)
        and item["term"].strip()
    ]
    return terms or None


_METADATA: Final[Mapping[str, FieldSpec]] = {
    "rss.feed_url": FieldSpec.at(f"{ENVELOPE_KEY}.feed_url"),
    "rss.feed_title": FieldSpec.at(f"{ENVELOPE_KEY}.feed_title"),
    # The provider's own guid, kept even though it is usually also the
    # `native_id`: when it is *not* -- a feed with no guid, identified by URL --
    # this is how you tell the two cases apart six months later.
    "rss.guid": FieldSpec.at("id"),
    "rss.categories": FieldSpec.at("tags", transform=_category_terms),
}


def _build_map(*, text_paths: tuple[str, ...], truncated: bool) -> FieldMap:
    """One field map per body-provenance case. See `_map_for`.

    A factory rather than three literals so the twenty shared lines are written
    once: three near-identical maps that drift apart is how a connector ends up
    mapping `author` on full-content entries and not on summary ones.
    """
    return FieldMap(
        platform=Platform.RSS,
        timestamp=_TIMESTAMP_SPEC,
        # Rule 1 against an id this connector derived itself in `fetch()`. See
        # the module docstring: one derivation, so the Kafka reference and the
        # Signal cannot disagree.
        item_id=FieldSpec.at(f"{ENVELOPE_KEY}.native_id"),
        url=FieldSpec.at("link"),
        title=FieldSpec.at("title"),
        text=FieldSpec.at(*text_paths),
        # Both RSS `<description>` and Atom `<content type="html">` are markup
        # far more often than not, and `extract_readable` passes plain text
        # through untouched (`looks_like_html` guards it), so declaring HTML is
        # right for both and costs nothing when it is wrong.
        text_is_html=True,
        author_id=FieldSpec.at(f"{ENVELOPE_KEY}.author_id"),
        author_display_name=FieldSpec.at("author_detail.name", "author"),
        author_profile_url=FieldSpec.at("author_detail.href"),
        metadata=_METADATA,
        # `<enclosure>` only. feedparser also surfaces the `media:` extension as
        # `media_content`/`media_thumbnail`, and news feeds routinely list the
        # same asset in both; mapping both would emit two `MediaRef`s for one
        # image and double every attachment count downstream. The extension
        # fields stay in the payload for a reprocess that wants them.
        media=MediaMap(container="enclosures", url="href", mime_type="type"),
        truncated=truncated,
        # `Content.content_type` describes the *cleaned* body, which is always
        # text by the time `extract_readable` is done with it.
        content_type="text/plain",
    )


_CONTENT_MAP: Final = _build_map(text_paths=("content.0.value",), truncated=False)
_SUMMARY_MAP: Final = _build_map(text_paths=("summary",), truncated=False)
_EXCERPT_MAP: Final = _build_map(text_paths=("summary",), truncated=True)


_RATE_LIMIT_HEADERS: Final = frozenset(
    {"retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"}
)
"""The only response headers that leave `fetch()`.

Mirrors the allowlist in `news_api.py`, `gdelt.py` and `reddit.py` so all four
connectors filter identically. `FetchPage.raw_headers` travels with the batch and
a whole header map carries `Set-Cookie` and echoed authorization --
`docs/connector-spec.md` S1 forbids logging either. RSS needs this as much as the
authenticated connectors do: private feeds use HTTP basic auth (`_auth_for`), and
an unfiltered map is the one surface where those credentials can travel back out
of the connector.
"""


def _rate_limit_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() in _RATE_LIMIT_HEADERS}



class RssConnector(BaseConnector):
    """Polls a set of RSS 2.0 / Atom 1.0 feeds and emits one Signal per entry."""

    slug: ClassVar[str] = _SLUG
    platform: ClassVar[Platform] = Platform.RSS
    category: ClassVar[SourceCategory] = SourceCategory.NEWS
    auth_type: ClassVar[AuthType] = AuthType.NONE
    """No auth. HTTP basic is supported for private feeds but is a property of
    an individual feed, not of the source, so the declaration stays `NONE` --
    see `_basic_auth_hosts` for why the credential is not simply attached to
    every request."""

    rate_limit: ClassVar[RateLimitPolicy] = RateLimitPolicy(
        requests_per_minute=60, burst=10, concurrency=4
    )
    """The connector-wide budget only. The limit that actually protects a
    publisher is the per-host bucket, which
    `.env.example::CONNECTOR_DEFAULT_RATE_LIMIT_PER_MINUTE` sets."""

    version: ClassVar[str] = "0.1.0"
    supports_incremental: ClassVar[bool] = True
    supports_backfill: ClassVar[bool] = False
    """A feed is a trailing window, not an archive. Declaring backfill support
    would have the scheduler plan a historical crawl that can only ever
    re-fetch the same last fifty items."""

    requires_tos_review: ClassVar[bool] = False

    overlap_seconds: ClassVar[int] = 300

    def __init__(
        self, ctx: SyncContext, credentials: Credentials, feeds: Sequence[str]
    ) -> None:
        super().__init__(ctx, credentials)
        self._feeds: tuple[str, ...] = tuple(feeds)
        self._client: httpx.AsyncClient | None = None
        self._basic_auth_hosts: frozenset[str] = _basic_auth_hosts(credentials, self._feeds)
        self._mapping = MappingContext(
            connector_slug=self.slug,
            connector_version=self.version,
            sync_run_id=ctx.run_id,
        )

    # ----------------------------------------------------------- lifecycle --

    @classmethod
    def from_config(cls, ctx: SyncContext, credentials: Credentials) -> Self:
        """Validate the feed list. No I/O, per `BaseConnector.from_config`.

        Everything that can be wrong with the *configuration* is discovered
        here, before a socket exists: a `file://` URL, a URL with credentials
        embedded in it, an empty list. A run that opens a connection and then
        finds out the config is broken has already spent a scheduling slot and
        looks, in metrics, exactly like a provider fault.
        """
        return cls(ctx, credentials, _configured_feeds(ctx.params))

    async def authenticate(self) -> None:
        """Build the HTTP client. Idempotent; there is no session to acquire.

        The client is built here rather than in `from_config` because it owns a
        connection pool that has to be closed. The scheduler constructs
        connectors merely to read their declaration
        (`docs/connector-spec.md` §3), and a construction that allocated a pool
        would leak one every time it did.
        """
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.ctx.request_timeout_seconds),
            # Feeds move. Following the redirect keeps the poll working; the
            # *identity* of the entries is unaffected because `native_id` is
            # derived from the entry's own link, never from the response URL.
            follow_redirects=True,
            headers={
                # An identifying User-Agent is a condition of use for most
                # publishers and the first thing an operator blocks when it is
                # absent (`docs/connector-spec.md` §9.5).
                "User-Agent": self.ctx.user_agent,
                "Accept": _ACCEPT,
            },
        )

    async def aclose(self) -> None:
        """Release the connection pool. Called by `run()` even on early exit."""
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    # ---------------------------------------------------------------- fetch --

    async def fetch(self, cursor: Cursor) -> AsyncIterator[FetchPage]:
        """Poll every configured feed, one `FetchPage` per feed, oldest-first.

        One page per feed rather than one page for the run: the page cursor is
        the runtime's commit point, so a crash after feed seventeen must leave
        feeds one through seventeen durable and their validators stored.

        The three interesting behaviours are documented on the helpers they live
        in -- `_ordered_payloads` (ordering), `_conditional_state` (304s) and
        `_failure_page` (isolation).
        """
        states = _feed_states(cursor)
        current = cursor
        succeeded = 0
        first_failure: ConnectorError | None = None

        for feed_url in self._feeds:
            key = _feed_key(feed_url)
            state = states.get(key, {})

            # Acquired around every outbound call, including the ones that turn
            # out to be 304s: a conditional request is still a request, and a
            # limiter that only counted the expensive ones would let a thousand
            # unchanged feeds hammer one host for free.
            await self.acquire_slot(feed_url)
            try:
                response = await self._get(feed_url, state)
                parsed = None if response.status_code == 304 else await self._parse(response)
            except ConnectorError as exc:
                if first_failure is None:
                    first_failure = exc
                yield self._failure_page(feed_url, exc, current)
                continue

            succeeded += 1
            states[key] = {
                **_conditional_state(response, state),
                **({"newest_seen": state["newest_seen"]} if state.get("newest_seen") else {}),
            }

            if parsed is None:
                # 304 Not Modified: a successful poll that found nothing. The
                # empty page still carries a cursor, so the refreshed validators
                # are committed and the next poll is conditional too.
                current = current.advanced_to(feeds=dict(states))
                yield FetchPage(
                    records=(), cursor=current, raw_headers=_rate_limit_headers(response.headers)
                )
                continue

            records, newest = self._records_for(feed_url, parsed, state)
            if newest is not None:
                states[key]["newest_seen"] = _to_iso(newest)
            current = current.advanced_to(watermark=newest, feeds=dict(states))
            yield FetchPage(
                records=records, cursor=current, raw_headers=_rate_limit_headers(response.headers)
            )

        if succeeded == 0 and first_failure is not None:
            # Nothing worked. One dead feed is that feed's problem; every feed
            # dead at once is ours -- DNS, egress, a bad User-Agent -- and
            # reporting a successful run with zero records would hide an outage
            # behind a healthy-looking connector.
            raise first_failure

    def _records_for(
        self,
        feed_url: str,
        parsed: Any,
        state: Mapping[str, str],
    ) -> tuple[tuple[RawRecord, ...], datetime | None]:
        """Turn one parsed feed into oldest-first records and a new watermark.

        The cutoff is taken from *this feed's* `newest_seen`, never from the
        run-wide watermark. A monthly newsletter polled alongside a wire service
        would otherwise be filtered against the wire's timestamps and never emit
        anything again.
        """
        raw_bytes = parsed.get("_omnisense_bytes")
        cutoff = _cutoff(state, self.overlap_seconds)
        newest = _from_iso(state.get("newest_seen"))
        records: list[RawRecord] = []

        for index, (moment, payload) in enumerate(
            _ordered_payloads(self._entry_payloads(feed_url, parsed))
        ):
            if moment is not None:
                if cutoff is not None and moment <= cutoff:
                    # Already emitted on an earlier poll. Dedup would catch it
                    # anyway, but skipping here saves a normalize and two Redis
                    # round-trips for the ~90% of a feed that is unchanged.
                    continue
                if newest is None or moment > newest:
                    newest = moment

            try:
                native_id = _native_id(payload, moment)
            except NormalizationError as exc:
                # No guid, no link, and not enough text to hash: the entry
                # cannot be named, so it cannot be a Signal. It becomes a DLQ
                # marker rather than vanishing, because an entry we silently
                # skipped is one nobody can discover we skipped.
                records.append(
                    self._marker(
                        native_id=f"rss:unidentified:{_feed_key(feed_url)}#{index}",
                        feed_url=feed_url,
                        message=exc.message,
                        error_class=type(exc).__name__,
                    )
                )
                continue

            payload[ENVELOPE_KEY]["native_id"] = native_id
            records.append(
                RawRecord(
                    native_id=native_id,
                    payload=payload,
                    # The feed document, attached to every entry that came out
                    # of it. RSS's unit of retrieval is the document, not the
                    # item -- there are no per-item provider bytes short of
                    # re-serializing, which would break the digest promise
                    # `RawRecord` makes. The R2 key is content-addressed, so N
                    # entries from one poll reference one stored object.
                    raw_bytes=raw_bytes,
                    content_type=str(parsed.get("_omnisense_content_type") or "application/xml"),
                    source_url=feed_url,
                    request_fingerprint=_fingerprint(feed_url),
                )
            )

        return tuple(records), newest

    def _entry_payloads(self, feed_url: str, parsed: Any) -> list[dict[str, Any]]:
        """Project feedparser's entries into JSON-safe payloads with an envelope.

        JSON-safe is not cosmetic: the runtime PUTs `payload` to R2 as JSON
        (`docs/connector-spec.md` §2.6), and feedparser hands back `struct_time`
        objects that no JSON encoder accepts. They are rendered as ISO-8601 UTC
        strings, which is lossless -- feedparser has already normalized them to
        UTC -- and which `to_utc_datetime` reads back.
        """
        feed = parsed.get("feed") or {}
        bozo = bool(parsed.get("bozo"))
        payloads: list[dict[str, Any]] = []
        for entry in parsed.get("entries") or ():
            payload = _jsonable(dict(entry))
            # `enclosures` is a *computed* member of feedparser's dict subclass,
            # filtered out of `links` by `rel`, so `dict(entry)` loses it
            # entirely. Materializing it keeps the payload self-contained: a DLQ
            # replay reads the stored JSON, not a live parse, and would
            # otherwise map an article's attachments to nothing.
            payload["enclosures"] = _jsonable(list(entry.get("enclosures") or ()))
            payload[ENVELOPE_KEY] = {
                "feed_url": feed_url,
                "feed_title": _clean(feed.get("title")) or None,
                "feed_link": _clean(feed.get("link")) or None,
                # Recorded rather than acted on: feedparser recovered enough to
                # produce entries, and refusing them because the document also
                # had a stray ampersand would reject half the real-world web.
                # It is here so a DLQ record from this feed is explicable.
                "bozo": bozo,
                "author_id": _author_identity(entry, feed_url),
            }
            payloads.append(payload)
        return payloads

    # ------------------------------------------------------------ normalize --

    async def normalize(self, record: RawRecord) -> Signal | None:
        """Map one entry onto a Signal, or `None` to drop it.

        Three outcomes, and the difference between the last two is the whole
        reason `docs/connector-spec.md` §2.4 separates them: a marker record
        raises (a defect somebody should see), an entry with nothing to observe
        returns `None` (expected, counted as a drop), and everything else maps.
        """
        payload = record.payload
        envelope = payload.get(ENVELOPE_KEY)
        envelope = envelope if isinstance(envelope, Mapping) else {}

        failure = envelope.get("error")
        if failure:
            raise NormalizationError(
                str(failure),
                native_id=record.native_id,
                details={
                    "feed_url": envelope.get("feed_url"),
                    "cause_class": envelope.get("error_class"),
                },
            )

        if not _has_observation(payload):
            # An entry with an id and nothing else: a placeholder, or an item
            # whose body was pulled. `docs/connector-spec.md` §2.4 names the
            # empty feed entry as the canonical silent drop.
            return None

        return _map_for(payload).to_signal(record, self._mapping)

    # -------------------------------------------------------------- HTTP --

    async def _get(self, feed_url: str, state: Mapping[str, str]) -> httpx.Response:
        """One conditional GET. Raises a `ConnectorError` scoped to this feed."""
        client = self._client
        if client is None:  # pragma: no cover -- run() always authenticates first
            raise ConnectorConfigurationError(
                "authenticate() must run before fetch(); the HTTP client is built there",
                connector=self.slug,
            )

        headers: dict[str, str] = {}
        if state.get("etag"):
            headers["If-None-Match"] = state["etag"]
        if state.get("last_modified"):
            headers["If-Modified-Since"] = state["last_modified"]

        try:
            response = await client.get(
                feed_url, headers=headers, auth=self._auth_for(feed_url)
            )
        except httpx.TimeoutException as exc:
            raise TransientError(
                f"timed out fetching feed {feed_url}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc
        except httpx.TransportError as exc:
            # DNS failure, connection reset, TLS error. Retryable in principle,
            # which is why the class matters even though this connector
            # isolates it: `succeeded == 0` re-raises it as-is.
            raise TransientError(
                f"could not reach feed {feed_url}: {type(exc).__name__}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc
        except httpx.HTTPError as exc:
            # Malformed URL, redirect loop. Retrying reproduces it exactly.
            raise PermanentError(
                f"feed {feed_url} is not fetchable: {type(exc).__name__}",
                connector=self.slug,
                account_id=self.ctx.account_id,
                cause=exc,
            ) from exc

        if response.status_code in (200, 304):
            return response
        raise self._status_error(feed_url, response)

    def _status_error(self, feed_url: str, response: httpx.Response) -> ConnectorError:
        """Classify a non-2xx/304 status for one feed.

        Note what is *not* here: `AuthError`. A 401 from a feed is a fact about
        that feed's configuration, not about the account -- `auth_type` is
        `NONE`, so there is no credential to flag `needs_reauth`, and raising
        `AuthError` would take the whole run down over one private URL in a list
        of fifty public ones.

        Nor `QuotaError` on a 429. A per-feed `Retry-After` cannot be honoured
        by a run-level reschedule when the other forty-nine feeds are fine, so
        the instruction is routed to the limiter instead of to the scheduler:
        the headers ride along on the failure page and `run()` feeds them to
        `RateLimiter.observe`, which clamps the bucket.
        """
        status = response.status_code
        # `to_log_fields()` drops a `headers` key, which is what makes it safe
        # to carry one here: it reaches the limiter and never a log line.
        details: dict[str, Any] = {
            "headers": _rate_limit_headers(response.headers),
            "feed_url": feed_url,
        }
        kind = TransientError if status >= 500 or status == 429 else PermanentError
        return kind(
            f"feed {feed_url} returned HTTP {status}",
            connector=self.slug,
            account_id=self.ctx.account_id,
            status_code=status,
            details=details,
        )

    async def _parse(self, response: httpx.Response) -> Any:
        """Parse a feed document off the event loop.

        feedparser is a synchronous, regex-heavy parser that holds the loop for
        tens of milliseconds on a large feed; this connector is expected to run
        alongside dozens of other in-flight connectors in one worker, so the
        parse goes to a thread. That is not the internal sleeping
        `docs/connector-spec.md` §1 forbids -- nothing waits on a clock.

        Bytes are passed, never the URL: `feedparser.parse("http://...")` would
        fetch it itself, outside the rate limiter and outside every timeout set
        above.
        """
        body = response.content
        if len(body) > MAX_FEED_BYTES:
            raise PermanentError(
                f"feed body is {len(body)} bytes, over the {MAX_FEED_BYTES} byte "
                "parse ceiling",
                connector=self.slug,
                status_code=response.status_code,
            )

        parsed = await asyncio.to_thread(
            feedparser.parse, body, response_headers=dict(response.headers)
        )
        if not parsed.get("entries") and parsed.get("bozo"):
            # Nothing parsed *and* the parser complained: an HTML error page, a
            # captive portal, a truncated response. An empty feed that parses
            # cleanly is legal and is not this.
            raise PermanentError(
                "response body is not a parseable RSS or Atom document",
                connector=self.slug,
                status_code=response.status_code,
                details={"bozo_exception": type(parsed.get("bozo_exception")).__name__},
            )

        # Carried alongside the parse so `_records_for` can attach the exact
        # bytes without re-reading a response it does not hold.
        parsed["_omnisense_bytes"] = body
        parsed["_omnisense_content_type"] = (
            response.headers.get("content-type", "application/xml").split(";")[0].strip()
        )
        return parsed

    def _auth_for(self, feed_url: str) -> httpx.BasicAuth | None:
        """HTTP basic, but only for hosts the credential was declared for.

        A client-level `auth=` would send the operator's password to every host
        in the feed list, including the forty-nine public ones. See
        `_basic_auth_hosts` for how the declaration is made.
        """
        host = (urlsplit(feed_url).hostname or "").lower()
        if host not in self._basic_auth_hosts:
            return None
        return httpx.BasicAuth(
            self.credentials.require("username"), self.credentials.require("password")
        )

    # ----------------------------------------------------------- assembly --

    def _failure_page(
        self, feed_url: str, exc: ConnectorError, cursor: Cursor
    ) -> FetchPage:
        """One dead feed, expressed as a DLQ record instead of a dead run.

        A connector holds no logger the runtime reads and no store it can write
        to, so its only channel for "this feed is broken" is a record whose
        `normalize()` raises. That is the whole trick: the failure becomes one
        DLQ entry, attributable to the feed URL, and the loop continues.

        The cursor is passed through *unchanged*. The feed's validators and
        `newest_seen` stay exactly as they were, so a feed that recovers on the
        next poll resumes where it left off rather than re-emitting its window.
        """
        headers = exc.details.get("headers")
        return FetchPage(
            records=(
                self._marker(
                    native_id=f"rss:feed-error:{_feed_key(feed_url)}",
                    feed_url=feed_url,
                    message=exc.message,
                    error_class=type(exc).__name__,
                ),
            ),
            cursor=cursor,
            # Filtered even though the caller has usually filtered already: this
            # is the last point before the headers leave the connector, and a
            # second application of an allowlist costs nothing.
            raw_headers=_rate_limit_headers(headers) if isinstance(headers, Mapping) else {},
        )

    def _marker(
        self, *, native_id: str, feed_url: str, message: str, error_class: str
    ) -> RawRecord:
        """A record that exists only to fail `normalize()` and reach the DLQ.

        `raw_bytes` is `None` because there are none worth keeping -- the
        response either did not arrive or was not a feed -- and the message is
        this connector's own text, never a response body
        (`docs/connector-spec.md` §1).
        """
        return RawRecord(
            native_id=native_id,
            payload={
                ENVELOPE_KEY: {
                    "feed_url": feed_url,
                    "error": message,
                    "error_class": error_class,
                }
            },
            content_type="application/json",
            source_url=feed_url,
            request_fingerprint=_fingerprint(feed_url),
        )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def _configured_feeds(params: Mapping[str, Any]) -> tuple[str, ...]:
    """Read, validate and de-duplicate the feed list from `SyncContext.params`."""
    raw: Any = None
    for key in FEED_PARAM_KEYS:
        if key in params:
            raw = params[key]
            break
    if raw is None:
        raise ConnectorConfigurationError(
            f"the rss connector needs feed URLs: set one of {list(FEED_PARAM_KEYS)} "
            "in the account's params (seeded from RSS_FEED_URLS in .env.example)",
            connector=_SLUG,
        )

    seen: dict[str, str] = {}
    for candidate in _split(raw):
        url = _validated_feed_url(candidate)
        # Keyed by canonical form, but the *configured* spelling is what gets
        # requested: two spellings of one feed must not be polled twice, and
        # rewriting an operator's URL before sending it would make a request
        # they cannot reproduce with curl.
        seen.setdefault(_feed_key(url), url)

    if not seen:
        raise ConnectorConfigurationError(
            "the rss connector was configured with an empty feed list",
            connector=_SLUG,
        )

    ceiling = int(params.get("max_feeds", MAX_FEEDS))
    if len(seen) > ceiling:
        raise ConnectorConfigurationError(
            f"{len(seen)} feeds configured, over the ceiling of {ceiling}. Raise "
            "max_feeds or split the account: polling only the first {ceiling} "
            "would be invisible data loss",
            connector=_SLUG,
        )
    return tuple(seen.values())


def _split(raw: Any) -> list[str]:
    """Accept both the list form and the comma-separated string form.

    `.env.example` seeds `RSS_FEED_URLS` as one comma-separated line, and a
    settings layer that has not split it yet is the common case rather than a
    misuse.
    """
    if isinstance(raw, str):
        return [part.strip() for part in raw.replace("\n", ",").split(",") if part.strip()]
    if isinstance(raw, Iterable):
        return [str(part).strip() for part in raw if str(part).strip()]
    raise ConnectorConfigurationError(
        f"feed list must be a string or a sequence of strings, got {type(raw).__name__}",
        connector=_SLUG,
    )


def _validated_feed_url(url: str) -> str:
    """Reject a URL this connector must not fetch."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ConnectorConfigurationError(
            f"feed URL {url!r} uses scheme {parts.scheme!r}; only http and https are "
            "allowed. A file:// or data: entry would turn a feed list into a local "
            "file read",
            connector=_SLUG,
        )
    if not parts.hostname:
        raise ConnectorConfigurationError(
            f"feed URL {url!r} has no host", connector=_SLUG
        )
    if parts.username or parts.password:
        raise ConnectorConfigurationError(
            "credentials must not be embedded in a feed URL: "
            "BaseConnector.host_rate_limit_key() builds its Redis key from the "
            "netloc, which includes userinfo, so the password would end up in a "
            "bucket name and in every metric derived from it. Put them in the "
            "connector account's secrets instead",
            connector=_SLUG,
        )
    return url


def _basic_auth_hosts(credentials: Credentials, feeds: Sequence[str]) -> frozenset[str]:
    """Which hosts the account's HTTP basic credential may be sent to.

    Declared explicitly via `credentials.extra["basic_auth_hosts"]`, or inferred
    when every configured feed is on one host -- which is the shape a
    private-feed account actually has. Anything else raises, because the
    alternative is guessing, and the wrong guess sends a password to a host that
    never asked for one.
    """
    if not credentials.secrets.get("password"):
        return frozenset()
    if not credentials.secrets.get("username"):
        # Caught here rather than at request time: a half-configured credential
        # would otherwise surface as a `KeyError` from inside the fetch loop,
        # attributed to a provider fault rather than to the operator who
        # configured the account.
        raise ConnectorConfigurationError(
            "the account carries an HTTP basic password but no username",
            connector=_SLUG,
        )

    declared = credentials.extra.get("basic_auth_hosts")
    if isinstance(declared, str):
        declared = [declared]
    if isinstance(declared, Iterable):
        hosts = {str(host).strip().lower() for host in declared if str(host).strip()}
        if hosts:
            return frozenset(hosts)

    inferred = {(urlsplit(feed).hostname or "").lower() for feed in feeds}
    if len(inferred) == 1:
        return frozenset(inferred)
    raise ConnectorConfigurationError(
        "this account carries an HTTP basic credential but polls more than one "
        "host; declare credentials.extra['basic_auth_hosts'] so the password is "
        "not sent to every publisher in the feed list",
        connector=_SLUG,
    )


# --------------------------------------------------------------------------- #
# Cursor state
# --------------------------------------------------------------------------- #


def _feed_states(cursor: Cursor) -> dict[str, dict[str, str]]:
    """Read `checkpoint["feeds"]` defensively.

    Every value is coerced to a non-empty string and anything else is dropped.
    The checkpoint round-trips through JSON in Postgres, so a corrupted or
    hand-edited row must degrade to "no validators, poll unconditionally" rather
    than crash a run -- an unreadable cursor costs one full poll, an exception
    costs every poll until someone fixes the row.
    """
    raw = cursor.checkpoint.get("feeds")
    if not isinstance(raw, Mapping):
        return {}
    states: dict[str, dict[str, str]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            continue
        states[str(key)] = {
            str(name): field
            for name, field in value.items()
            if isinstance(field, str) and field
        }
    return states


def _conditional_state(
    response: httpx.Response, previous: Mapping[str, str]
) -> dict[str, str]:
    """The validators to send with the next request for this feed.

    A 200 *replaces* them with exactly what the response carried: a server that
    has stopped issuing `ETag`s must not keep receiving a stale
    `If-None-Match` forever. A 304 *merges*, because RFC 9110 lets a 304 omit
    validators that have not changed, and dropping them there would silently
    turn every subsequent poll into a full download.
    """
    fresh = {
        name: value
        for name, value in (
            ("etag", response.headers.get("etag")),
            ("last_modified", response.headers.get("last-modified")),
        )
        if value
    }
    if response.status_code == 304:
        return {**{k: v for k, v in previous.items() if k != "newest_seen"}, **fresh}
    return fresh


def _cutoff(state: Mapping[str, str], overlap_seconds: int) -> datetime | None:
    """This feed's incremental floor, with the overlap window applied.

    The overlap has to be re-applied here rather than inherited from
    `BaseConnector._effective_start`, which only shifts the run-wide watermark:
    per-feed state travels in `checkpoint`, which the base class treats as
    opaque. Without it, a publisher whose CMS indexes a post a minute after its
    own `pubDate` loses that post on every poll.
    """
    seen = _from_iso(state.get("newest_seen"))
    if seen is None:
        return None
    return seen - timedelta(seconds=overlap_seconds)


# --------------------------------------------------------------------------- #
# Entry handling
# --------------------------------------------------------------------------- #


def _ordered_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> list[tuple[datetime | None, dict[str, Any]]]:
    """Oldest-first, with undated entries ahead of everything.

    `BaseConnector.fetch` requires oldest-first and it is not stylistic: the
    watermark may only move forward, so a newest-first pager that dies mid-run
    commits a watermark past records it never emitted and they are never fetched
    again.

    Feeds are conventionally newest-first, so the list is reversed *before* a
    stable sort. That is what puts entries sharing one timestamp -- a publisher
    that stamps a whole batch with the same minute -- in oldest-first order too,
    instead of preserving the newest-first order of the document.

    Undated entries lead because they can never move the watermark and are
    headed for the DLQ; sorting them anywhere else would mean reasoning about a
    record with no position on the timeline.
    """
    prepared = [(_entry_timestamp(payload), dict(payload)) for payload in payloads]
    prepared.reverse()
    prepared.sort(key=lambda item: (item[0] is not None, item[0] or _EPOCH))
    return prepared


def _native_id(payload: Mapping[str, Any], moment: datetime | None) -> str:
    """Derive `native_id` by the §4.1 ladder, computing rule 3 only if forced.

    Rules 1 and 2 are string operations. Rule 3 needs the *cleaned* body, which
    means running `extract_readable` a second time on this entry, so it is
    computed only for the entry that has neither a guid nor a canonicalizable
    link -- rather than for every entry in every feed on the chance one of them
    might need it.
    """
    item_id = payload.get("id")
    link = payload.get("link")
    envelope = payload.get(ENVELOPE_KEY)
    author_id = envelope.get("author_id") if isinstance(envelope, Mapping) else None

    text: str | None = None
    if not _clean(item_id) and not canonicalize_url(_clean(link)):
        text = _body_text(payload)

    return derive_native_id(
        platform=Platform.RSS,
        item_id=item_id,
        url=link if isinstance(link, str) else None,
        author_id=author_id if isinstance(author_id, str) else None,
        timestamp=moment,
        text=text,
    )


def _map_for(payload: Mapping[str, Any]) -> FieldMap:
    """Pick the field map whose `truncated` flag tells the truth.

    Atom `<content>` is the full body by definition, so it is never truncated.
    A `<summary>`/`<description>` may be either the whole article or a teaser,
    and the two cannot be told apart except by length -- so the §11.2 threshold
    decides, and the cleaned length is used rather than the raw one because
    three hundred characters of markup around a hundred characters of prose is
    still a teaser.

    That costs one extra `extract_readable` over a short string. `truncated`
    caps a confidence component for the life of the Signal; being right about it
    is worth two passes over a few hundred bytes.
    """
    if _first_content_value(payload):
        return _CONTENT_MAP
    summary = payload.get("summary")
    if isinstance(summary, str) and len(extract_readable(summary)) >= EXCERPT_CHAR_THRESHOLD:
        return _SUMMARY_MAP
    return _EXCERPT_MAP


def _body_text(payload: Mapping[str, Any]) -> str:
    """The cleaned body the chosen field map will produce, for rule 3."""
    source = _first_content_value(payload) or payload.get("summary") or ""
    return extract_readable(source) if isinstance(source, str) else ""


def _first_content_value(payload: Mapping[str, Any]) -> str:
    """`content[0].value`, matching the field map's path exactly.

    Index 0 only, and not "the first non-empty entry in `content`": the map
    reads `content.0.value`, and a chooser that scanned further would select the
    full-content map for an entry whose body the map then resolves to `""`.
    """
    content = payload.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return ""
    if not content:
        return ""
    first = content[0]
    if not isinstance(first, Mapping):
        return ""
    value = first.get("value")
    return value.strip() if isinstance(value, str) else ""


def _has_observation(payload: Mapping[str, Any]) -> bool:
    """Whether the entry says anything at all beyond having an identity."""
    return bool(
        _clean(payload.get("title"))
        or _first_content_value(payload)
        or _clean(payload.get("summary"))
        or _clean(payload.get("link"))
    )


def _author_identity(entry: Mapping[str, Any], feed_url: str) -> str | None:
    """A host-scoped author identity, or `None` when the entry names nobody.

    RSS has no author ids, only bylines, so `Author.platform_author_id` has to
    be built from one -- `connectors/normalize/mapper.py` requires a connector
    that does this to own the decision explicitly, and this is that decision.

    The byline is scoped by the *feed's* host because "John Smith" writes for
    more than one publication. Scoping forks one real author across two feeds;
    not scoping merges two real authors into one. `connectors/normalize/html.py`
    states the asymmetry that settles it: a fork is annoying and recoverable, a
    merge is silent and is not.
    """
    detail = entry.get("author_detail")
    detail = detail if isinstance(detail, Mapping) else {}
    byline = _clean(detail.get("email")) or _clean(detail.get("name")) or _clean(
        entry.get("author")
    )
    if not byline:
        return None
    host = (urlsplit(feed_url).hostname or "").lower()
    return f"{host}:{byline}" if host else byline


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #


def _jsonable(value: Any) -> Any:
    """Render feedparser's output as something `json.dumps` accepts.

    `struct_time` becomes an ISO-8601 UTC string -- lossless, because feedparser
    normalizes the parsed date fields to UTC before handing them over. Anything
    exotic that is left becomes its `str()` rather than being dropped: the
    payload is the input to a DLQ replay, and a field nobody maps today is still
    evidence tomorrow.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, struct_time):
        return _struct_time_iso(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _struct_time_iso(value: struct_time) -> str | None:
    """Render a parsed date, or `None` when the parser produced a nonsense one.

    Returning `None` rather than raising lets `_TIMESTAMP_SPEC` fall through to
    the next path -- usually the raw `published` string, which our own parser
    may well handle.
    """
    try:
        return _to_iso(datetime(*value[:6], tzinfo=UTC))
    except (TypeError, ValueError):
        return None


def _to_iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _from_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return to_utc_datetime(value)
    except ValueError:
        return None


def _clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _feed_key(url: str) -> str:
    """Checkpoint key for one feed: its canonical URL.

    Canonical rather than verbatim so that a trailing slash added to the config
    does not orphan a feed's ETag and re-download its whole window.
    """
    return canonicalize_url(url) or url


def _fingerprint(feed_url: str) -> str:
    """`lineage.request_fingerprint`: the request, never the credentials."""
    return hashlib.sha256(f"GET {_feed_key(feed_url)}".encode()).hexdigest()
