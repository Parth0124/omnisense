"""What a keyword document *is*, and how the chunk index is created and evolved.

`backend/db/opensearch.py` owns bootstrap: the connection singleton and a
language-neutral mapping good enough for the process to start indexing. This
module sits one layer above and owns the parts that are retrieval decisions
rather than boot decisions -- which languages get their own analyzer, what a
document is allowed to contain, and what happens when the index that already
exists is not the index this code expects.

Three decisions are encoded here. All three fail *quietly* if they are made the
other way, which is why each one is stated rather than left to a default.

**One document per chunk, `_id = chunk_id`.** `docs/data-stores.md` §3.5 and
§5.2. The `_id` is the idempotency key: re-indexing a chunk overwrites in place
instead of adding a second copy, and hybrid fusion joins the OpenSearch and
Qdrant candidate lists on exactly this string (`retrieval/rerank/fusion.py`).
Indexing one document per Signal would leave fusion with nothing to join on and
would break citation, whose unit is the chunk span `[char_start, char_end)`.

**`dynamic: "strict"`.** An indexer that leaks an unexpected field would
otherwise add it to the mapping permanently -- a change that cannot be undone in
place, and that in the worst case ("this string looks like a date") makes every
subsequent well-formed document unindexable. Strict turns an irreversible schema
mutation into a rejected document naming the field.

**Per-language analysis happens in a *selective* field, not a multi-field.**
`docs/retrieval.md` §4 asks for "analyzer selected per `language`", which
OpenSearch cannot do per document: an analyzer is a property of a field. The two
ways to get there are not equivalent.

- A multi-field (`text.de`) copies the parent value, so *every* document is
  analyzed by *every* language analyzer. That is N x the index-time CPU, and
  worse, it poisons the corpus statistics the ranking depends on: `text.de` would
  contain every English document run through a German stemmer, so the IDF of a
  German term is computed against a corpus that is mostly not German.
- A sibling field (`text_de`) written *only* when the document is German costs one
  extra analysis per document and gives `text_de` a posting list whose statistics
  are genuinely German.

The second is what this module defines. The price is that the set of supported
languages is fixed at index creation and adding one is a reindex -- so the set
below is deliberately wider than today's connectors need.

Layer note: L1 (`docs/architecture.md` §6.1). Imports `models/` and
`backend/core` + `backend/db` (the L1k kernel); nothing above it.
"""

from __future__ import annotations

import copy
import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final

from opensearchpy.exceptions import RequestError

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError
from backend.core.logging import get_logger
from backend.db.opensearch import SIGNAL_INDEX_ANALYSIS, SIGNAL_INDEX_MAPPINGS
from models.base import utcnow
from models.enums import Platform, SourceCategory
from retrieval.types import chunk_id_for

__all__ = [
    "EXACT_SUBFIELD",
    "LANGUAGE_ANALYZERS",
    "TEXT_FIELD_PREFIX",
    "ChunkDocument",
    "ChunkField",
    "IndexSpec",
    "IndexState",
    "MappingDrift",
    "chunk_index_mappings",
    "chunk_index_spec",
    "ensure_chunk_index",
    "inspect_chunk_index",
    "language_text_field",
    "mapping_drift",
    "primary_subtag",
    "swap_alias",
]

_log = get_logger(__name__)

TEXT_FIELD_PREFIX: Final[str] = "text_"
"""Prefix of the per-language sibling fields, e.g. `text_de`.

A prefix on a flat field name rather than a nested object, because a `bool` query
names fields as flat strings and `text_*` stays usable as a debugging wildcard.
"""

EXACT_SUBFIELD: Final[str] = "exact"
"""Name of the keyword-ish multi-field under `text` and `title`.

Its analyzer is declared in `backend/db/opensearch.py`: tokenize, lowercase and
asciifold, but do **not** stem and do **not** drop stopwords. That is what makes
a phrase boost mean anything -- a stemmer collapses distinct product names onto
one root, and a stopword filter deletes the "by" from "connection reset by peer".
"""


class ChunkField(enum.StrEnum):
    """Every field a chunk document may carry. There are no others.

    A `StrEnum` rather than bare string constants so the mapping, the document
    builder and the query builder cannot drift apart on a spelling. A `bool`
    query naming a field that was never mapped matches zero documents and raises
    nothing -- indistinguishable from "no results for that query" -- so a typo
    that is an `AttributeError` at import is strictly better than one that is a
    silently empty result list.

    Per-language text fields are deliberately *not* members: their names are
    derived from `LANGUAGE_ANALYZERS` by `language_text_field()`, and enumerating
    them twice would be two places to forget a language.
    """

    CHUNK_ID = "chunk_id"
    SIGNAL_ID = "signal_id"
    CHUNK_INDEX = "chunk_index"
    TENANT_ID = "tenant_id"

    TITLE = "title"
    TEXT = "text"
    CHAR_START = "char_start"
    CHAR_END = "char_end"

    SOURCE = "source"
    PLATFORM = "platform"
    URL = "url"
    AUTHOR = "author"
    PUBLISHED_AT = "published_at"
    LANGUAGE = "language"
    KEYWORDS = "keywords"
    TOPICS = "topics"
    ENTITY_IDS = "entity_ids"

    SENTIMENT_POLARITY = "sentiment_polarity"
    ENGAGEMENT_SCORE = "engagement_score"
    CONFIDENCE = "confidence"

    PIPELINE_VERSION = "pipeline_version"
    INDEXED_AT = "indexed_at"
    METADATA = "metadata"


LANGUAGE_ANALYZERS: Final[Mapping[str, str]] = {
    "en": "english",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "pt": "portuguese",
    "it": "italian",
    "nl": "dutch",
    "ru": "russian",
    "ar": "arabic",
    "tr": "turkish",
    # `cjk` is the built-in fallback: character bigrams rather than real
    # morphological segmentation. Correct but blunt. Proper Japanese and Korean
    # analysis needs the `kuromoji` and `nori` plugins, which the local
    # single-node cluster in `docker-compose.yml` does not install -- naming them
    # here would make index creation fail on a developer laptop with an error
    # about an unknown analyzer rather than about a missing plugin.
    "ja": "cjk",
    "zh": "cjk",
    "ko": "cjk",
}
"""ISO 639-1 code -> built-in OpenSearch analyzer, for the sibling text fields.

Fixed at index creation: adding a language means a new field, and a new field is
empty for every document already indexed. Adding one is therefore a reindex into
a new index plus an alias swap (`docs/data-stores.md` §3.5), which is why this
set is wider than the connectors currently produce.

Only *built-in* analyzers appear here, for the reason in the `cjk` comment: a
plugin analyzer turns index creation into a boot failure on every cluster that
lacks the plugin.
"""


def primary_subtag(code: str) -> str:
    """The ISO 639-1 primary subtag of a BCP-47 tag: `pt-BR` -> `pt`.

    Both the stored `language` field and the filter DSL are normalised through
    this one function, and they *must* be, because they are compared with a
    `terms` clause. `docs/retrieval.md` §7 defines the filter as an "ISO 639-1
    include set", so storing `pt-BR` while filtering on `pt` would make a
    Portuguese filter silently miss every Brazilian document -- a `terms` clause
    matches the keyword exactly or not at all, and there is no error either way.
    The regional subtag is not lost to the system; PostgreSQL keeps the full
    `Language` model (`models/signal.py`).
    """
    return code.strip().lower().replace("_", "-").split("-")[0] or "und"


def language_text_field(code: str) -> str | None:
    """The sibling text field for a language, or None when it has no analyzer.

    Returning None rather than raising is the point: `und` (detection was
    inconclusive -- `models/signal.py`) and any unsupported language are *normal*
    rather than exceptional. Those documents stay searchable through the
    language-neutral `text` field, which is why that field is populated for every
    document regardless of language.
    """
    primary = primary_subtag(code)
    if primary not in LANGUAGE_ANALYZERS:
        return None
    return f"{TEXT_FIELD_PREFIX}{primary}"


def _exact_analyzer_name() -> str:
    """The custom analyzer name declared in the bootstrap analysis block.

    Read out of `SIGNAL_INDEX_ANALYSIS` rather than hardcoded, so renaming it in
    the kernel module cannot leave this module referring to an analyzer the index
    does not define -- which OpenSearch rejects at create time, at boot, on every
    replica at once.
    """
    analyzers = SIGNAL_INDEX_ANALYSIS.get("analyzer", {})
    if len(analyzers) != 1:
        raise ConfigurationError(
            "SIGNAL_INDEX_ANALYSIS is expected to declare exactly one custom "
            f"analyzer for the exact sub-field; found {sorted(analyzers)}. "
            "retrieval/keyword/index.py cannot guess which of them is the "
            "keyword-ish one."
        )
    return str(next(iter(analyzers)))


def chunk_index_mappings() -> dict[str, Any]:
    """The full mapping: the bootstrap mapping plus the per-language text fields.

    Built by extending `SIGNAL_INDEX_MAPPINGS` rather than restating it, as
    `backend/db/opensearch.py` intends. Two copies of a mapping drift, and the
    drift is undetectable from either side -- both create an index successfully,
    and only queries against the fields they disagree about come back empty.

    Note what is *not* here: a field-level `boost`. An index-time boost is baked
    into the length norms at write time, so changing one requires a reindex, and
    OpenSearch inherits Elasticsearch's deprecation of them for exactly that
    reason. The title boost from `docs/retrieval.md` §4 is applied per query in
    `retrieval/keyword/query_builder.py`, where it can be tuned against the
    evaluation harness without touching the corpus.
    """
    mappings = copy.deepcopy(SIGNAL_INDEX_MAPPINGS)
    properties: dict[str, Any] = mappings["properties"]

    # A phrase match on the title is the strongest signal available for the query
    # class this index exists for -- product names, error strings, exact quotes --
    # and the bootstrap mapping gives only `text` an exact sub-field.
    title: dict[str, Any] = properties[ChunkField.TITLE.value]
    title.setdefault("fields", {})[EXACT_SUBFIELD] = {
        "type": "text",
        "analyzer": _exact_analyzer_name(),
    }

    # `norms` are left enabled on the sibling fields. They cost about a byte per
    # document per field and they are what makes a match in a 12-word title
    # outrank the same match in a 4000-word article; disabling them to save that
    # byte flattens ranking on exactly the long documents where length
    # normalisation matters most, and nothing reports that it happened.
    properties.update(
        {
            f"{TEXT_FIELD_PREFIX}{code}": {"type": "text", "analyzer": analyzer}
            for code, analyzer in LANGUAGE_ANALYZERS.items()
        }
    )
    return mappings


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """The full definition of the chunk index.

    Frozen and passed explicitly rather than read from settings at each use,
    because the migration path in `docs/data-stores.md` §3.5 is *build a second
    index alongside the live one and swap an alias*. Code that reads global
    settings at every call cannot express "this indexer writes the new index
    while that searcher reads the old one".
    """

    name: str
    number_of_shards: int = 1
    number_of_replicas: int = 1
    """1 by default because in a deployed cluster a lost shard is a lost index.

    The local single-node cluster cannot allocate the replica and therefore sits
    permanently yellow, which `backend.db.opensearch.check_opensearch()` accepts
    by design. Pass 0 to keep a single-node cluster green.
    """

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("index name must not be empty")
        if self.number_of_shards < 1:
            raise ValueError(
                f"number_of_shards must be positive, got {self.number_of_shards}"
            )
        if self.number_of_replicas < 0:
            raise ValueError(
                f"number_of_replicas must be non-negative, got {self.number_of_replicas}"
            )

    def mappings(self) -> dict[str, Any]:
        return chunk_index_mappings()

    def settings(self) -> dict[str, Any]:
        return {
            "index": {
                "number_of_shards": self.number_of_shards,
                "number_of_replicas": self.number_of_replicas,
            },
            "analysis": copy.deepcopy(SIGNAL_INDEX_ANALYSIS),
        }

    def body(self) -> dict[str, Any]:
        """The create-index request body."""
        return {"settings": self.settings(), "mappings": self.mappings()}


def chunk_index_spec(
    *,
    name: str | None = None,
    number_of_shards: int | None = None,
    number_of_replicas: int | None = None,
) -> IndexSpec:
    """The configured index, with overrides for a reindex-and-swap migration.

    Everyday callers pass nothing: the name is `OPENSEARCH_SIGNAL_INDEX`.
    """
    settings = get_settings()
    return IndexSpec(
        name=name or settings.opensearch.signal_index,
        number_of_shards=1 if number_of_shards is None else number_of_shards,
        number_of_replicas=1 if number_of_replicas is None else number_of_replicas,
    )


# --------------------------------------------------------------------------- #
# The document
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ChunkDocument:
    """One chunk, ready to write. A closed set of fields, by design.

    A dataclass rather than a `dict[str, Any]` because the mapping is
    `dynamic: "strict"`: a caller that invents a field gets a rejected *bulk
    item*, which surfaces as a partial success buried in a 200-response body.
    Closing the set here names the offending field at the call site instead.

    Nothing here is authoritative. OpenSearch is a derived store
    (`docs/data-stores.md` §1) and `text` is the one deliberately duplicated
    payload in the system -- an inverted index needs the tokens. A citation is
    resolved from PostgreSQL by `chunk_id`, never rendered out of this document.
    """

    signal_id: str
    chunk_index: int
    text: str
    char_start: int = 0
    char_end: int = 0

    tenant_id: str = "default"
    title: str | None = None
    source: SourceCategory = SourceCategory.UNKNOWN
    platform: Platform = Platform.UNKNOWN
    url: str | None = None
    author_handle: str | None = None
    published_at: datetime | None = None
    language: str = "und"
    keywords: Sequence[str] = ()
    topics: Sequence[str] = ()
    entity_ids: Sequence[str] = ()
    sentiment_polarity: float | None = None
    engagement_score: float | None = None
    confidence: float = 0.0

    pipeline_version: int = 0
    """`pipeline_version` as the sortable ordinal from `models/lineage.py`.

    An integer, not the `"1.10.0"` string: compared as text, `'1.10.0' >= '1.9.0'`
    is False. This value becomes the *external document version*
    (`docs/data-stores.md` §5.2), so a string here would let a stale backfill
    overwrite newer enrichment the moment a version component reached 10 -- the
    exact corruption the ordinal exists to prevent.
    """

    metadata: Mapping[str, Any] = field(default_factory=dict)
    indexed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.signal_id:
            raise ValueError("signal_id must not be empty; it is half of the chunk id")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be non-negative, got {self.chunk_index}")
        if not self.text.strip() and not (self.title or "").strip():
            # A chunk with neither title nor body can never satisfy a query and
            # can never be cited (`Passage.is_citable`). Indexing it costs a slot
            # and buys a document that only ever surfaces when a filter-only
            # browse pads out the result set.
            raise ValueError(
                f"chunk {chunk_id_for(self.signal_id, self.chunk_index)!r} has no "
                "title and no text; an empty chunk is neither retrievable nor "
                "citable, so indexing it can only dilute results"
            )
        if self.char_start < 0:
            raise ValueError(f"char_start must be non-negative, got {self.char_start}")
        if self.char_end < self.char_start:
            raise ValueError(
                f"chunk {chunk_id_for(self.signal_id, self.chunk_index)!r} has "
                f"char_end {self.char_end} before char_start {self.char_start}; "
                "services/evidence_service.py verifies a quote by re-reading that "
                "span, and an inverted span verifies nothing while looking checked"
            )

    @property
    def chunk_id(self) -> str:
        """The document `_id`, and the key hybrid fusion joins on."""
        return chunk_id_for(self.signal_id, self.chunk_index)

    @property
    def language_field(self) -> str | None:
        """The sibling text field this document populates, if any."""
        return language_text_field(self.language)

    def to_document(self) -> dict[str, Any]:
        """Serialise to the `_source` body.

        `published_at` must carry a timezone. OpenSearch parses a naive datetime
        as UTC, which is a *guess*: for a Signal timestamped in Asia/Tokyo the
        guess moves it by up to a day, and a day is enough to cross the boundary
        of a `[start, end)` window filter. Nothing downstream can detect that it
        happened -- the document is simply missing from a result set it belonged
        in -- so the write is refused instead.
        """
        if self.published_at is not None and self.published_at.tzinfo is None:
            raise ValueError(
                f"published_at for chunk {self.chunk_id!r} is timezone-naive; "
                "OpenSearch would assume UTC and could move the Signal across a "
                "time-window filter boundary undetectably"
            )

        document: dict[str, Any] = {
            ChunkField.CHUNK_ID.value: self.chunk_id,
            ChunkField.SIGNAL_ID.value: self.signal_id,
            ChunkField.CHUNK_INDEX.value: self.chunk_index,
            ChunkField.TENANT_ID.value: self.tenant_id,
            ChunkField.TITLE.value: self.title,
            # Always populated, whatever the language. This is the field that
            # keeps `und` and unsupported-language documents retrievable at all.
            ChunkField.TEXT.value: self.text,
            ChunkField.CHAR_START.value: self.char_start,
            ChunkField.CHAR_END.value: self.char_end,
            ChunkField.SOURCE.value: str(self.source),
            ChunkField.PLATFORM.value: str(self.platform),
            ChunkField.URL.value: self.url,
            ChunkField.AUTHOR.value: {"handle": self.author_handle},
            ChunkField.PUBLISHED_AT.value: (
                self.published_at.isoformat() if self.published_at is not None else None
            ),
            # Normalised to the primary subtag: see `primary_subtag()` for why
            # the stored value and the filter value must agree exactly.
            ChunkField.LANGUAGE.value: primary_subtag(self.language),
            # Lists, never bare strings. A `terms` filter against a scalar works,
            # but a one-element list and a string round-trip differently through
            # `_source`, and the difference shows up only as a filter that
            # matches nothing.
            ChunkField.KEYWORDS.value: list(self.keywords),
            ChunkField.TOPICS.value: list(self.topics),
            ChunkField.ENTITY_IDS.value: list(self.entity_ids),
            ChunkField.SENTIMENT_POLARITY.value: self.sentiment_polarity,
            ChunkField.ENGAGEMENT_SCORE.value: self.engagement_score,
            ChunkField.CONFIDENCE.value: float(self.confidence),
            ChunkField.PIPELINE_VERSION.value: int(self.pipeline_version),
            ChunkField.INDEXED_AT.value: (self.indexed_at or utcnow()).isoformat(),
            ChunkField.METADATA.value: dict(self.metadata),
        }

        sibling = self.language_field
        if sibling is not None:
            # The one place the "selective sibling field" decision from the
            # module docstring is enacted: exactly one language field per
            # document, so its posting list carries honest corpus statistics.
            document[sibling] = self.text
        return document


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MappingDrift:
    """How a live mapping differs from the one this build expects.

    Split into three kinds because they have different consequences, and
    collapsing them into "the mapping is wrong" would hide the worst one behind
    the harmless one.
    """

    missing_fields: Sequence[str] = ()
    """Expected by this build, absent from the index.

    The dangerous kind. A query naming an unmapped field is *accepted* by
    OpenSearch and matches nothing, so a missing `text_de` is indistinguishable
    from "no German document matched", permanently.
    """

    conflicting_fields: Sequence[str] = ()
    """Present in both, with a different `type` or `analyzer`.

    Unfixable in place: OpenSearch cannot change a mapped field's type, and
    changing an analyzer does not re-analyze the documents already written.
    """

    extra_fields: Sequence[str] = ()
    """Present in the index, unknown to this build. Informational only.

    An index created by a newer deployment mid-rolling-upgrade looks exactly like
    this, and it answers every query this build makes correctly.
    """

    dynamic: str | None = None
    """The live `dynamic` setting when it is not `strict`, else None."""

    @property
    def is_breaking(self) -> bool:
        """Whether the difference makes this build unsafe against this index."""
        return bool(self.missing_fields or self.conflicting_fields or self.dynamic)

    def describe(self) -> str:
        parts: list[str] = []
        if self.missing_fields:
            parts.append(f"missing fields {sorted(self.missing_fields)}")
        if self.conflicting_fields:
            parts.append(f"incompatible fields {sorted(self.conflicting_fields)}")
        if self.dynamic:
            parts.append(f"dynamic={self.dynamic!r} rather than 'strict'")
        return "; ".join(parts) or "no breaking difference"


@dataclass(frozen=True, slots=True)
class IndexState:
    """The outcome of `ensure_chunk_index()`."""

    name: str
    created: bool
    drift: MappingDrift = field(default_factory=MappingDrift)


def mapping_drift(live: Mapping[str, Any], expected: Mapping[str, Any]) -> MappingDrift:
    """Compare two mapping bodies. Pure, so the comparison is testable offline.

    Compares `type` and `analyzer`, and nothing else. A live mapping is full of
    server-supplied defaults (`fielddata`, `similarity`, index options) that the
    create request never sent, so a whole-body equality check would report drift
    against a freshly created index -- and an alarm that fires every time is an
    alarm nobody reads.
    """
    missing: list[str] = []
    conflicting: list[str] = []
    extra: list[str] = []
    _walk_properties(
        live.get("properties") or {},
        expected.get("properties") or {},
        prefix="",
        missing=missing,
        conflicting=conflicting,
        extra=extra,
    )

    live_dynamic = str(live.get("dynamic", "true")).lower()
    return MappingDrift(
        missing_fields=tuple(missing),
        conflicting_fields=tuple(conflicting),
        extra_fields=tuple(extra),
        dynamic=None if live_dynamic == "strict" else live_dynamic,
    )


def _walk_properties(
    live: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    prefix: str,
    missing: list[str],
    conflicting: list[str],
    extra: list[str],
) -> None:
    """Recurse through `properties` and multi-`fields`, recording differences.

    Both nestings are walked because both can hide a breaking difference: a
    missing `author.handle` breaks an author filter, and a missing `text.exact`
    silently removes the phrase boost that makes exact-match queries work.
    """
    for name, want in expected.items():
        path = f"{prefix}{name}"
        have = live.get(name)
        if have is None:
            missing.append(path)
            continue
        if _differs(have, want):
            conflicting.append(path)
        for nesting in ("properties", "fields"):
            _walk_properties(
                have.get(nesting) or {},
                want.get(nesting) or {},
                prefix=f"{path}.",
                missing=missing,
                conflicting=conflicting,
                extra=extra,
            )

    for name in live:
        if name not in expected:
            extra.append(f"{prefix}{name}")


def _differs(have: Mapping[str, Any], want: Mapping[str, Any]) -> bool:
    """Whether two field definitions are incompatible.

    A live `text` field reports no `analyzer` when it uses `standard`, because
    the server does not echo its own defaults. Treating that silence as a
    conflict would make every index this module creates immediately report drift
    against itself.
    """
    want_type = want.get("type", "object")
    have_type = have.get("type", "object")
    if want_type != have_type:
        return True
    if want_type != "text":
        return False
    return bool(want.get("analyzer", "standard") != have.get("analyzer", "standard"))


async def inspect_chunk_index(client: Any, spec: IndexSpec | None = None) -> MappingDrift:
    """Report how the live index differs from the spec. Never raises on drift.

    The read-only half of `ensure_chunk_index()`, split out so `scripts/` and an
    operator can see the difference without a process refusing to start over it.
    """
    resolved = spec or chunk_index_spec()
    response = await client.indices.get_mapping(index=resolved.name)
    return mapping_drift(_single_mapping(response, resolved.name), resolved.mappings())


def _single_mapping(response: Mapping[str, Any], requested: str) -> Mapping[str, Any]:
    """Pull the one mapping body out of a `get_mapping` response.

    The response is keyed by *concrete* index name, not by what was asked for:
    `requested` may be an alias, in which case indexing the response by that name
    raises `KeyError` against a perfectly healthy cluster. Exactly one concrete
    index is expected -- more than one means the alias spans a half-finished
    reindex, and picking one arbitrarily would compare against whichever the dict
    happened to yield first.
    """
    bodies = {name: value.get("mappings", {}) for name, value in response.items()}
    if not bodies:
        raise ConfigurationError(
            f"OpenSearch returned no mapping for {requested!r}; the index or alias "
            "does not exist"
        )
    if len(bodies) > 1:
        raise ConfigurationError(
            f"{requested!r} resolves to {sorted(bodies)} -- more than one concrete "
            "index. An alias spanning several indices is a reindex that was never "
            "finished; complete the alias swap before starting writers."
        )
    return next(iter(bodies.values()))


async def ensure_chunk_index(client: Any, spec: IndexSpec | None = None) -> IndexState:
    """Create the chunk index if absent; refuse to run against a divergent one.

    Idempotent in the way that matters under concurrency: every worker replica
    calls this at boot, they all see the index missing, and they all issue a
    create. Exactly one wins and the losers get
    `resource_already_exists_exception`, which is the success case rather than an
    error -- but the loser then *verifies*, because "someone else created it a
    millisecond ago" and "someone else created it last release with last
    release's mapping" arrive as the same response.

    An existing index is never mutated. `put_mapping` looks harmless -- adding a
    field is legal and additive -- and that is the trap: the new field is empty
    for every document already written, so queries against it return nothing
    while the mapping reports it present. `docs/data-stores.md` §3.5 prescribes
    reindex-into-a-new-index plus an alias swap, and this function's job is to
    make the operator do that rather than to paper over the gap.

    Raises:
        ConfigurationError: the live mapping is missing a field this build
            queries, has a field with an incompatible type or analyzer, or is not
            `dynamic: strict`.
    """
    resolved = spec or chunk_index_spec()

    # The existence check is an optimisation; the exception handler below is the
    # correctness guarantee. Checking first only avoids a 400 in the cluster log
    # on every process start.
    if not await client.indices.exists(index=resolved.name):
        try:
            await client.indices.create(index=resolved.name, body=resolved.body())
        except RequestError as exc:
            if getattr(exc, "error", None) != "resource_already_exists_exception":
                raise
        else:
            _log.info(
                "opensearch.chunk_index.created",
                index=resolved.name,
                languages=sorted(LANGUAGE_ANALYZERS),
            )
            return IndexState(name=resolved.name, created=True)

    drift = await inspect_chunk_index(client, resolved)
    if drift.is_breaking:
        raise ConfigurationError(
            f"OpenSearch index {resolved.name!r} does not match the mapping this "
            f"build expects: {drift.describe()}. A mapping cannot be changed in "
            "place -- adding the fields now would leave them empty for every "
            "document already indexed, and queries against them would come back "
            "empty without erroring. Reindex into a new index with "
            "scripts/reindex.py and swap the alias.",
            details={
                "index": resolved.name,
                "missing_fields": list(drift.missing_fields),
                "conflicting_fields": list(drift.conflicting_fields),
                "dynamic": drift.dynamic,
            },
        )
    if drift.extra_fields:
        # Not fatal: an index written by a newer build during a rolling upgrade
        # looks exactly like this and answers every query here correctly.
        _log.info(
            "opensearch.chunk_index.extra_fields",
            index=resolved.name,
            fields=sorted(drift.extra_fields),
        )
    return IndexState(name=resolved.name, created=False, drift=drift)


async def swap_alias(client: Any, alias: str, *, to_index: str) -> Sequence[str]:
    """Point `alias` at `to_index` atomically. Returns the indices detached.

    The other half of the lifecycle `ensure_chunk_index()` refuses to shortcut.
    One `update_aliases` call rather than remove-then-add, because between two
    calls the alias resolves to nothing and every search against it fails with
    `index_not_found_exception` -- a self-inflicted outage in the middle of what
    is supposed to be a zero-downtime migration.

    The detached indices are *not* deleted. Deleting them here would make the
    rollback -- swap back -- impossible at exactly the moment it is needed.
    """
    try:
        existing = await client.indices.get_alias(name=alias)
    except Exception:  # noqa: BLE001 -- absence is the ordinary first-swap case
        existing = {}

    detached = [name for name in existing if name != to_index]
    actions: list[dict[str, Any]] = [
        {"remove": {"index": name, "alias": alias}} for name in detached
    ]
    actions.append({"add": {"index": to_index, "alias": alias}})

    await client.indices.update_aliases(body={"actions": actions})
    _log.info("opensearch.alias.swapped", alias=alias, to=to_index, detached=detached)
    return tuple(detached)
