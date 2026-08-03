"""Declarative payload -> Signal mapping, and the `native_id` rule ladder.

Stage 4 of the connector lifecycle (`docs/connector-spec.md` §2.4) is the same
shape for every source: pull a handful of values out of a provider payload,
derive an identity, assemble provenance, hand back a `Signal`. Written by hand
per connector it is twenty lines of `payload.get(...)` chains that each get the
same three things subtly wrong -- a naive datetime, a `metadata` key that is not
namespaced, a `KeyError` escaping as a crash instead of a DLQ record.

So the *shape* of the mapping is declared as data (`FieldMap`) and the
interpretation lives here, once. That buys three things a hand-written mapper
does not:

- **Every missing required field raises `NormalizationError` carrying the
  `native_id`.** That is the difference between a DLQ record that can be replayed
  and one nobody can even attribute to a source (`docs/connector-spec.md` §6).
- **Identity is derived before required fields are enforced.** The three-rule
  ladder runs first precisely so that the error raised for a *missing timestamp*
  can still name the item it happened to.
- **Pydantic's `ValidationError` never escapes.** `Signal` enforces real
  invariants -- derived id, source/platform agreement, metadata depth -- and a
  violation is one bad record, not a dead run. `BaseConnector.run()` only catches
  `NormalizationError`, so anything else aborts the page and blocks every
  well-formed record behind it.

**Rule 3 makes identity depend on cleaned text.** `docs/signal-model.md` §4.1
requires connectors that reach rule 3 to say so in their module docstring; the
same obligation applies to this module, which is why `simhash64` below is frozen
and deliberately not shared with `connectors/dedup/hashing.py`.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, tzinfo
from email.utils import parsedate_to_datetime
from time import struct_time
from typing import Any, Final

from pydantic import ValidationError

from models.enums import MediaKind, Platform
from models.lineage import Lineage
from models.signal import Author, Content, Engagement, MediaRef, Signal
from connectors.exceptions import NormalizationError
from connectors.normalize.html import canonicalize_url, collapse_whitespace, extract_readable
from connectors.protocol import RawRecord

__all__ = [
    "UNENRICHED_PIPELINE_VERSION",
    "FieldMap",
    "FieldSpec",
    "MappingContext",
    "MediaMap",
    "build_lineage",
    "derive_native_id",
    "simhash64",
    "to_utc_datetime",
]

UNENRICHED_PIPELINE_VERSION: Final = "0.0.0"
"""What a connector stamps on `lineage.pipeline_version`.

`Lineage` requires the field, but a connector has run none of the enrichment
pipeline and does not know its version. Stamping the zero version says exactly
that, and `services/signal_engine/pipeline.py` overwrites it at stage 1. A
plausible-looking `"1.0.0"` here would instead claim an enrichment that never
happened, and `docs/signal-model.md` §7 makes that field the basis for deciding
whether a stored Signal needs reprocessing.
"""


# --------------------------------------------------------------------------- #
# Identity: the three-rule ladder of docs/signal-model.md §4.1
# --------------------------------------------------------------------------- #


def derive_native_id(
    *,
    platform: Platform,
    item_id: Any = None,
    url: str | None = None,
    author_id: str | None = None,
    timestamp: datetime | None = None,
    text: str | None = None,
    url_resolver: Callable[[str], str | None] | None = None,
) -> str:
    r"""Derive `native_id` by the first rule that applies (§4.1).

    | Rank | Rule                                                          |
    | ---- | ------------------------------------------------------------- |
    | 1    | the platform's own stable item id                             |
    | 2    | `sha256` of the canonicalized URL                             |
    | 3    | `sha256(platform \| author \| timestamp \| simhash64(text))`   |

    Order is not a preference, it is a correctness requirement: the rules are
    ranked by how stable they are under re-fetch. A provider that starts
    returning a guid it previously omitted would fork every item's identity if a
    connector were allowed to prefer the URL, so the ladder is evaluated top-down
    and never re-entered.

    Rule 1 returns the provider's id **verbatim rather than hashed**. Hashing
    would cost nothing in correctness and everything in operability: a DLQ record
    for `t3_1abcde` can be pasted straight into the provider's own UI, and a
    64-character digest cannot.

    Raises `NormalizationError` when no rule applies. No `native_id` is attached
    to that error because none exists -- that *is* the failure -- so the message
    names all three rules and what each was missing instead.
    """
    candidate = _as_clean_str(item_id)
    if candidate:
        return candidate

    if url:
        canonical = canonicalize_url(url, resolver=url_resolver)
        if canonical:
            return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    cleaned = (text or "").strip()
    if timestamp is not None and cleaned:
        # `|` as the separator, and every component rendered in exactly one way,
        # because this string *is* the identity: a different separator or a
        # different timestamp spelling is a different Signal for the same item.
        material = "|".join(
            (
                platform.value,
                author_id or "",
                _identity_timestamp(timestamp),
                f"{simhash64(cleaned):016x}",
            )
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    raise NormalizationError(
        "cannot derive native_id: rule 1 needs a platform item id (absent), "
        "rule 2 needs a resolvable absolute URL "
        f"({'unusable: ' + url if url else 'absent'}), rule 3 needs both a "
        f"timestamp ({'present' if timestamp else 'absent'}) and non-empty "
        f"cleaned text ({'present' if cleaned else 'absent'})",
        details={"platform": platform.value},
    )


def _identity_timestamp(moment: datetime) -> str:
    """The one spelling of a timestamp that may enter an identity.

    Normalized to UTC and rendered with a `Z` suffix, at full microsecond
    precision. Truncating to seconds would be tempting -- provider clocks are
    coarse -- but two comments posted in the same second by the same author would
    then collide into one Signal, and rule 3 exists precisely for sources whose
    only distinguishing feature is their text.
    """
    if moment.tzinfo is None:
        raise NormalizationError(
            "refusing to derive identity from a naive timestamp: it is ambiguous "
            "across process boundaries and would change meaning on replay"
        )
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


_TOKEN = re.compile(r"[0-9a-z]+")
_SHINGLE_SIZE: Final = 3
_SIMHASH_BITS: Final = 64


def simhash64(text: str) -> int:
    """64-bit SimHash over 3-gram token shingles.

    **Frozen.** This function feeds rule 3, so changing the shingle size, the
    tokenizer or the hash re-derives the id of every Signal that reached rule 3 --
    which `docs/signal-model.md` §7 lists as "not migratable in place".

    That is also why it lives here rather than being imported from
    `connectors/dedup/hashing.py`. The near-duplicate SimHash there is a *tuning
    knob*: `docs/connector-spec.md` open question 3 says its Hamming threshold
    "needs to be tuned against a labelled corpus", and tuning it implies moving
    the tokenization with it. One shared implementation would let a
    clustering-recall experiment silently re-identify the corpus.

    Tokens are NFKC-folded, case-folded and reduced to alphanumerics, so the same
    body extracted by two different backends -- which differ in whitespace and
    punctuation, not in words -- hashes identically.
    """
    tokens = _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())
    if not tokens:
        return 0
    if len(tokens) < _SHINGLE_SIZE:
        shingles = tokens
    else:
        shingles = [
            " ".join(tokens[i : i + _SHINGLE_SIZE])
            for i in range(len(tokens) - _SHINGLE_SIZE + 1)
        ]

    weights = [0] * _SIMHASH_BITS
    for shingle in shingles:
        value = int.from_bytes(
            hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(_SIMHASH_BITS):
            weights[bit] += 1 if value >> bit & 1 else -1

    result = 0
    for bit, weight in enumerate(weights):
        if weight > 0:
            result |= 1 << bit
    return result


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MappingContext:
    """Who is doing the mapping, for the acquisition group of `Lineage`.

    Passed in rather than read off a connector instance so this module never
    imports `connectors/base.py`. The mapper is then usable from `workers/dlq.py`
    replaying a fixed field map against a historical payload, with no connector
    instance in sight.
    """

    connector_slug: str
    connector_version: str
    sync_run_id: str
    pipeline_version: str = UNENRICHED_PIPELINE_VERSION


def build_lineage(
    record: RawRecord,
    *,
    native_id: str,
    ctx: MappingContext,
    raw_object_key: str | None = None,
) -> Lineage:
    """Assemble the three lineage groups a connector can know (§3.6).

    Acquisition, raw payload and identity. The processing group keeps its
    defaults and the scoring group stays empty, because a connector runs no
    enrichment stage and computes no confidence component.

    `raw_object_key` is normally `None` here even though its value is
    predictable: the R2 key is content-addressed, so it *could* be computed from
    the digest below. It is not, because the connector does not perform the PUT
    (`docs/connector-spec.md` §2.6), and a lineage pointer to an object that was
    never written is worse than a null one -- it turns a failed upload into a 404
    at citation time instead of a visible gap.
    """
    raw_bytes = record.raw_bytes

    return Lineage(
        # -- Acquisition ---------------------------------------------------
        connector_slug=ctx.connector_slug,
        connector_version=ctx.connector_version,
        sync_run_id=ctx.sync_run_id,
        fetched_at=record.fetched_at,
        request_fingerprint=record.request_fingerprint,
        # -- Raw payload ---------------------------------------------------
        # The digest is taken over the bytes the provider actually returned,
        # never over a re-serialization of `payload`: `json.dumps` orders keys
        # and escapes non-ASCII differently across versions, and the R2 key is
        # content-addressed off exactly this value.
        raw_object_key=raw_object_key,
        raw_sha256=hashlib.sha256(raw_bytes).hexdigest() if raw_bytes else None,
        raw_bytes=len(raw_bytes) if raw_bytes is not None else None,
        raw_content_type=record.content_type,
        # -- Identity ------------------------------------------------------
        native_id=native_id,
        # -- Processing ----------------------------------------------------
        pipeline_version=ctx.pipeline_version,
    )


# --------------------------------------------------------------------------- #
# The declarative field map
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Where one value comes from, and what to do if it is not there.

    Several `paths` are tried in order because provider payloads are unions in
    practice, not schemas: an Atom entry carries `content`, an RSS 2.0 item
    carries `description`, and the same feedparser dict may hold either. A
    connector that hard-codes one and falls over on the other has a mapping bug
    that surfaces on the day a publisher switches format, not before.
    """

    paths: tuple[str, ...]
    required: bool = False
    default: Any = None
    transform: Callable[[Any], Any] | None = None

    @classmethod
    def at(
        cls,
        *paths: str,
        required: bool = False,
        default: Any = None,
        transform: Callable[[Any], Any] | None = None,
    ) -> FieldSpec:
        """Readable constructor: `FieldSpec.at("data.title", required=True)`."""
        if not paths:
            raise ValueError("a FieldSpec needs at least one path")
        return cls(paths=paths, required=required, default=default, transform=transform)

    def resolve(
        self,
        payload: Mapping[str, Any],
        *,
        name: str,
        native_id: str | None = None,
        strict: bool = True,
    ) -> Any:
        """First present value among `paths`, else `default`.

        "Present" excludes `None` and the blank string, but *not* `0` or `False`:
        a score of zero is a fact about the item, whereas `"summary": ""` is the
        provider saying it has no summary. Conflating them either loses real
        zeros or pins every empty field to the first path that happens to exist.

        `strict=False` suppresses the required-field check. `FieldMap.to_signal`
        uses it for the first pass so that identity can be derived before any
        `NormalizationError` is raised -- see that method.
        """
        for path in self.paths:
            value = _lookup(payload, path)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return self.transform(value) if self.transform else value
        if self.required and strict:
            raise NormalizationError(
                f"required field {name!r} is absent; tried {list(self.paths)}",
                native_id=native_id,
                details={"field": name},
            )
        return self.default


@dataclass(frozen=True, slots=True)
class MediaMap:
    """Where attached media live in the payload.

    `container` addresses a list; the other paths are relative to each item in
    it. An attachment with no URL is skipped rather than emitted with a null one:
    a `MediaRef` carrying neither `source_url` nor `object_key` points at
    nothing, and would survive into a report as an unresolvable citation.
    """

    container: str
    url: str
    mime_type: str | None = None
    kind: str | None = None
    default_kind: MediaKind = MediaKind.UNKNOWN


#: Field-map attributes that are plain `FieldSpec | None` slots, in the order
#: they are resolved. Kept as data so `_iter_specs` cannot drift out of sync with
#: the dataclass the way a hand-maintained list of `self.x` accesses would.
_SPEC_FIELDS: Final[tuple[str, ...]] = (
    "timestamp",
    "item_id",
    "url",
    "title",
    "text",
    "author_id",
    "author_handle",
    "author_display_name",
    "author_profile_url",
    "author_follower_count",
    "author_verified",
)


@dataclass(frozen=True, slots=True)
class FieldMap:
    """A declarative description of one provider's payload shape.

    Only the fields a connector can know are here. `language`, `entities`,
    `topics`, `keywords`, `embeddings` and `sentiment` are absent by design --
    they belong to `services/signal_engine/`, and a connector that filled them
    would be doing enrichment inside the ingest path (`docs/connector-spec.md`
    §1).

    Validation happens in `__post_init__`, at import time, because everything it
    catches is wrong with the *map* rather than with any particular payload. A
    metadata key that is not namespaced is wrong for a million records or for
    none of them; finding that out at import costs nothing, finding it out at 3am
    costs a run.
    """

    platform: Platform
    timestamp: FieldSpec
    item_id: FieldSpec | None = None
    url: FieldSpec | None = None
    title: FieldSpec | None = None
    text: FieldSpec | None = None
    text_is_html: bool = False
    """Whether `text` needs `extract_readable()` before it is a cleaned body."""

    author_id: FieldSpec | None = None
    author_handle: FieldSpec | None = None
    author_display_name: FieldSpec | None = None
    author_profile_url: FieldSpec | None = None
    author_follower_count: FieldSpec | None = None
    author_verified: FieldSpec | None = None

    engagement: Mapping[str, FieldSpec] = field(default_factory=dict)
    """Platform counters, verbatim, into `Engagement.raw`.

    Never the normalized axes: those are percentiles within a cohort
    (`docs/signal-model.md` §3.4), and a connector holding one record cannot know
    a percentile. A star rating is not engagement either -- it is polarity, and
    belongs to the sentiment stage.
    """

    metadata: Mapping[str, FieldSpec] = field(default_factory=dict)
    media: MediaMap | None = None

    assume_timezone: tzinfo | None = None
    """Timezone to attach to naive provider timestamps, when one is *known*.

    Left `None` on purpose. `models/base.py` rejects naive datetimes because
    guessing "silently shift[s] a trend by hours", so a connector states what the
    provider documents rather than letting this module assume UTC. A naive
    payload timestamp with no declaration is a `NormalizationError`.
    """

    truncated: bool = False
    """Set by connectors whose API returns an excerpt (NewsAPI, paywalled RSS).

    Caps `content_integrity` in the confidence model (`docs/signal-model.md`
    §3.5), so declaring it is what stops an excerpt being trusted like a body.
    """

    content_type: str = "text/plain"

    def __post_init__(self) -> None:
        if not isinstance(self.platform, Platform):
            raise ValueError(
                f"platform must be a Platform member, got {type(self.platform).__name__}"
            )
        if self.platform is Platform.UNKNOWN:
            raise ValueError(
                "platform 'unknown' has no category and would derive identity "
                "under a name no connector owns"
            )

        prefix = f"{self.platform.value}."
        unnamespaced = sorted(key for key in self.metadata if not key.startswith(prefix))
        if unnamespaced:
            raise ValueError(
                f"metadata keys must be namespaced by platform ({prefix}...): "
                f"{unnamespaced}. Un-namespaced keys collide across connectors in "
                "one jsonb column and one OpenSearch mapping "
                "(docs/signal-model.md §2)"
            )

        shadowed = sorted(set(_SPEC_FIELDS) & set(self.engagement))
        if shadowed:
            raise ValueError(
                f"engagement counters may not reuse a Signal field name: {shadowed}. "
                "Resolution is keyed by name, so the counter would overwrite the "
                "field and the mapping would fail somewhere unrelated"
            )

        if self.item_id is None and self.url is None and self.text is None:
            # Rule 1 needs an id, rule 2 a URL, rule 3 text. With none of the
            # three declared, every record this map will ever see fails identity
            # derivation -- so fail now, loudly, with the map in hand.
            raise ValueError(
                "this FieldMap can never derive a native_id: none of item_id "
                "(rule 1), url (rule 2) or text (rule 3) is declared "
                "(docs/signal-model.md §4.1)"
            )

    # --------------------------------------------------------------- mapping --

    def to_signal(
        self,
        record: RawRecord,
        ctx: MappingContext,
        *,
        url_resolver: Callable[[str], str | None] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> Signal:
        """Map one raw record onto a canonical Signal.

        The order is deliberate: everything is resolved leniently, then identity
        is derived, and only then are required fields enforced. Enforcing first
        would produce a `NormalizationError` with no `native_id` for the single
        most common malformed payload -- an entry with no date -- and a DLQ record
        nobody can attribute is one nobody can replay.
        """
        payload = record.payload
        resolved = {
            name: spec.resolve(payload, name=name, strict=False)
            for name, spec in self._iter_specs()
        }

        body = self._body(resolved.get("text"))
        url = self._canonical_url(resolved.get("url"), url_resolver)
        author_id = _as_clean_str(resolved.get("author_id"))
        moment, timestamp_error = self._timestamp(resolved.get("timestamp"))

        native_id = derive_native_id(
            platform=self.platform,
            item_id=resolved.get("item_id"),
            url=url,
            author_id=author_id,
            timestamp=moment,
            text=body,
            url_resolver=url_resolver,
        )

        self._enforce_required(resolved, native_id)
        if moment is None:
            raise NormalizationError(
                timestamp_error
                or f"required field 'timestamp' is absent; tried {list(self.timestamp.paths)}",
                native_id=native_id,
                details={"field": "timestamp"},
            )

        content = Content(
            title=_clean_text(resolved.get("title")) or None,
            text=body,
            truncated=self.truncated,
            content_type=self.content_type,
            # `raw_ref` stays None: the R2 key is the runtime's to assign. The
            # digest is set here because the connector is the only component that
            # still holds the bytes.
            raw_sha256=(
                hashlib.sha256(record.raw_bytes).hexdigest() if record.raw_bytes else None
            ),
        )

        metadata = {
            key: value
            for key, value in ((key, resolved.get(key)) for key in self.metadata)
            if value is not None
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        try:
            return Signal.create(
                platform=self.platform,
                native_id=native_id,
                timestamp=moment,
                content=content,
                lineage=build_lineage(record, native_id=native_id, ctx=ctx),
                url=url or None,
                author=self._author(resolved, author_id),
                media=self._media(payload),
                engagement=Engagement(raw=self._engagement(resolved)),
                metadata=metadata,
            )
        except ValidationError as exc:
            # `Signal` enforces the derived id, source/platform agreement and
            # metadata depth. A violation here is a defect in this map, but it is
            # a defect in *one record's* data; letting `ValidationError` escape
            # would abort the whole page in `BaseConnector.run()`, which catches
            # only `NormalizationError`.
            #
            # Pydantic's `msg` is carried through but its `input` is not: the
            # message is our own validator's text, while the input is payload
            # content, and `docs/connector-spec.md` §1 forbids logging that.
            problems = [_describe(err) for err in exc.errors()]
            raise NormalizationError(
                "payload does not satisfy the Signal contract: " + "; ".join(problems[:3]),
                native_id=native_id,
                details={"fields": sorted({_location(err) for err in exc.errors()})},
                cause=exc,
            ) from exc

    # ------------------------------------------------------------- internals --

    def _iter_specs(self) -> Iterator[tuple[str, FieldSpec]]:
        """Every declared spec, keyed by the name its errors will carry."""
        for name in _SPEC_FIELDS:
            spec = getattr(self, name)
            if spec is not None:
                yield name, spec
        yield from self.engagement.items()
        yield from self.metadata.items()

    def _enforce_required(self, resolved: Mapping[str, Any], native_id: str) -> None:
        """Raise for the first required field that resolved to nothing.

        Separated from resolution so the error can carry `native_id`, which does
        not exist until the ladder has run.
        """
        for name, spec in self._iter_specs():
            if spec.required and resolved.get(name) is None:
                raise NormalizationError(
                    f"required field {name!r} is absent; tried {list(spec.paths)}",
                    native_id=native_id,
                    details={"field": name},
                )

    def _body(self, raw: Any) -> str:
        """Clean the body, so that identity sees cleaned text.

        Rule 3 hashes whatever ends up in `content.text`. Deriving identity from
        raw markup instead would fork every item the day a publisher changed its
        page template, without a word of the article changing.
        """
        if raw is None:
            return ""
        text = raw if isinstance(raw, str) else str(raw)
        return extract_readable(text) if self.text_is_html else collapse_whitespace(text)

    def _canonical_url(
        self, raw: Any, url_resolver: Callable[[str], str | None] | None
    ) -> str:
        """Canonicalize the permalink with the same function identity uses.

        Field 4 of the Signal is the permalink "after redirect resolution and
        tracking-parameter stripping" (`docs/signal-model.md` §2), and rule 2
        hashes the canonical URL. Two different canonicalizations would mean
        `Signal.url` and `Signal.id` disagreed about which page this is, so there
        is exactly one.
        """
        text = _as_clean_str(raw)
        return canonicalize_url(text, resolver=url_resolver) if text else ""

    def _timestamp(self, raw: Any) -> tuple[datetime | None, str | None]:
        """Return `(timestamp, error)`; at most one is ever non-None.

        Absent and unparseable are different defects -- one is a feed with no
        date, the other is a format this mapper does not know -- and the second
        deserves a message naming the value rather than the paths.
        """
        if raw is None:
            return None, None
        try:
            return to_utc_datetime(raw, assume_timezone=self.assume_timezone), None
        except ValueError as exc:
            return None, f"field 'timestamp' is unusable: {exc}"

    def _author(self, resolved: Mapping[str, Any], author_id: str) -> Author | None:
        """Build the author, or `None` when the payload names no stable id.

        A display handle is *not* promoted into `platform_author_id` when the id
        is missing. Handles are renameable (`docs/signal-model.md` §3.1), so
        keying on one forks an author's history the first time they rename -- and
        silently, because nothing downstream can tell an id from a handle. A
        connector whose provider genuinely has no other identifier declares the
        handle path on `author_id` itself, and owns that decision explicitly.
        """
        if not author_id:
            return None
        return Author(
            platform_author_id=author_id,
            handle=_as_clean_str(resolved.get("author_handle")) or None,
            display_name=_as_clean_str(resolved.get("author_display_name")) or None,
            profile_url=_as_clean_str(resolved.get("author_profile_url")) or None,
            follower_count=_as_count(resolved.get("author_follower_count")),
            verified=bool(resolved.get("author_verified")),
        )

    def _engagement(self, resolved: Mapping[str, Any]) -> dict[str, float | int | None]:
        counters: dict[str, float | int | None] = {}
        for name in self.engagement:
            value = _as_number(resolved.get(name))
            if value is not None:
                counters[name] = value
        return counters

    def _media(self, payload: Mapping[str, Any]) -> list[MediaRef]:
        if self.media is None:
            return []
        container = _lookup(payload, self.media.container)
        if not isinstance(container, Sequence) or isinstance(container, (str, bytes)):
            return []
        refs: list[MediaRef] = []
        for item in container:
            if not isinstance(item, Mapping):
                continue
            source_url = _as_clean_str(_lookup(item, self.media.url))
            if not source_url:
                continue
            mime = (
                _as_clean_str(_lookup(item, self.media.mime_type))
                if self.media.mime_type
                else ""
            )
            declared = (
                _as_clean_str(_lookup(item, self.media.kind)) if self.media.kind else ""
            )
            refs.append(
                MediaRef(
                    kind=_media_kind(declared, mime, source_url, self.media.default_kind),
                    # Falls back to the raw URL: a media URL that will not
                    # canonicalize (relative, or hostless) is still worth keeping,
                    # because unlike `native_id` nothing derives identity from it.
                    source_url=canonicalize_url(source_url) or source_url,
                    mime_type=mime or None,
                )
            )
        return refs


# --------------------------------------------------------------------------- #
# Value coercion
# --------------------------------------------------------------------------- #


def to_utc_datetime(value: Any, *, assume_timezone: tzinfo | None = None) -> datetime:
    """Coerce a provider timestamp to a timezone-aware UTC datetime.

    Handles the five shapes providers actually send: `datetime`, epoch seconds or
    milliseconds, ISO-8601, RFC 2822 (every RSS 2.0 `pubDate`) and feedparser's
    `struct_time`. Raises `ValueError` on anything else, and on a naive value when
    the field map declared no `assume_timezone` -- guessing would shift a trend by
    hours, which is why `models/base.py` rejects naive datetimes at all.
    """
    if isinstance(value, datetime):
        return _attach_timezone(value, assume_timezone)
    if isinstance(value, struct_time) or (
        isinstance(value, tuple) and len(value) == 9 and all(isinstance(v, int) for v in value)
    ):
        # feedparser normalizes `published_parsed` to UTC before handing it over,
        # so this branch is not making the assumption the naive-datetime branch
        # refuses to make.
        year, month, day, hour, minute, second = (int(part) for part in tuple(value)[:6])
        return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
    if isinstance(value, bool):
        raise ValueError("a boolean is not a timestamp")
    if isinstance(value, (int, float)):
        # Millisecond epochs are common and indistinguishable from a year-5138
        # second epoch, so the boundary sits where no realistic event time falls:
        # anything past ~2286 read as seconds is milliseconds.
        seconds = value / 1000.0 if abs(value) > 100_000_000_000 else float(value)
        return datetime.fromtimestamp(seconds, tz=UTC)
    if isinstance(value, str):
        return _attach_timezone(_parse_datetime_string(value), assume_timezone)
    raise ValueError(f"unsupported timestamp type {type(value).__name__}")


def _parse_datetime_string(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        parsed = None
    if parsed is not None:
        return parsed
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S", "%Y%m%d", "%Y-%m-%d"):
        try:
            # Naive by construction; `_attach_timezone` decides what to do with it.
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    raise ValueError(f"unrecognized timestamp format {text!r}")


def _attach_timezone(moment: datetime, assume_timezone: tzinfo | None) -> datetime:
    if moment.tzinfo is not None:
        return moment.astimezone(UTC)
    if assume_timezone is not None:
        return moment.replace(tzinfo=assume_timezone).astimezone(UTC)
    raise ValueError(
        "timestamp is timezone-naive and the field map declares no "
        "assume_timezone; set it to what the provider documents rather than "
        "letting UTC be guessed"
    )


def _location(error: Mapping[str, Any]) -> str:
    """Dotted field path of one pydantic error.

    Model-level validators -- the identity check, the source/platform check, the
    metadata-depth check -- report an empty location, so they get a name of their
    own rather than silently vanishing from the DLQ record.
    """
    return ".".join(str(part) for part in error.get("loc", ())) or "<signal>"


def _describe(error: Mapping[str, Any]) -> str:
    return f"{_location(error)}: {error.get('msg', 'rejected')}"


def _lookup(payload: Any, path: str) -> Any:
    """Read a dotted path, with integer segments indexing sequences.

    `"data.children.0.data.title"` walks mappings and lists alike, because
    provider payloads nest one inside the other and a mapper that only handled
    mappings would push list indexing back into every connector.
    """
    current: Any = payload
    for segment in path.split("."):
        if current is None:
            return None
        if isinstance(current, Mapping):
            if segment not in current:
                return None
            current = current[segment]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                index = int(segment)
            except ValueError:
                return None
            if not -len(current) <= index < len(current):
                return None
            current = current[index]
            continue
        current = getattr(current, segment, None)
    return current


def _as_clean_str(value: Any) -> str:
    """Render a scalar as a stripped string; `""` for anything unusable.

    Provider ids arrive as ints as often as strings, and a mapper that accepted
    only strings would silently drop identity for half the catalogue. Booleans
    and containers are refused rather than stringified -- `"True"` is never a
    meaningful id, and `"['a', 'b']"` is a bug wearing a value's clothes.
    """
    if value is None or isinstance(value, bool):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (Mapping, Sequence, set, frozenset)):
        return ""
    return str(value).strip()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return collapse_whitespace(value if isinstance(value, str) else str(value))


def _as_number(value: Any) -> float | int | None:
    """Coerce a counter, rejecting booleans.

    `True` is an `int` in Python, so an unguarded coercion files `"is_self":
    true` as an engagement counter of 1 and lets it into a percentile cohort.
    Flags belong in `metadata`.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _as_count(value: Any) -> int | None:
    """Coerce a non-negative count, dropping anything that is not one."""
    number = _as_number(value)
    if number is None:
        return None
    count = int(number)
    return count if count >= 0 else None


_MEDIA_EXTENSIONS: Final[dict[MediaKind, tuple[str, ...]]] = {
    MediaKind.IMAGE: (".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".bmp", ".svg"),
    MediaKind.VIDEO: (".mp4", ".m4v", ".mov", ".webm", ".mkv", ".avi"),
    MediaKind.AUDIO: (".mp3", ".m4a", ".aac", ".ogg", ".oga", ".wav", ".flac"),
    MediaKind.DOCUMENT: (".pdf", ".doc", ".docx", ".ppt", ".pptx", ".epub"),
}


def _media_kind(declared: str, mime: str, url: str, default: MediaKind) -> MediaKind:
    """Classify an attachment: declared kind, then MIME type, then extension.

    Falls back to the map's `default_kind` rather than guessing `IMAGE`, because
    `MediaKind` is tolerant and a wrong-but-plausible kind is far harder to
    notice downstream than an honest `unknown`.
    """
    for candidate in (declared, mime):
        if candidate:
            kind = MediaKind(candidate.split("/")[0].strip().lower())
            if kind is not MediaKind.UNKNOWN:
                return kind
    lowered = url.lower().split("?")[0]
    for kind, extensions in _MEDIA_EXTENSIONS.items():
        if lowered.endswith(extensions):
            return kind
    return default
