"""What the Qdrant collection *is*: geometry, payload schema, and payload indexes.

`backend/db/qdrant.py` owns the two parameters that cannot be changed after
creation -- vector size and distance metric -- because getting those wrong is a
boot-time configuration failure, not a retrieval concern. This module sits one
layer above and owns everything that *can* be changed later: which payload fields
exist, and which of them are indexed.

Two decisions are encoded here, and both of them fail quietly rather than loudly.

**Every field used as a filter must have a payload index.** Qdrant will happily
evaluate a filter on an unindexed payload field -- by reading it off every point
in the segment. The query still returns correct results, so nothing alerts; it
just costs O(collection) instead of O(matches), and the symptom at 10M points is
"search got slow", which reads like a slow disk or a bad HNSW parameter and sends
the investigation to the wrong place entirely. `docs/retrieval.md` §5 lists
`tenant_id`, `source`, `language`, `published_at` and `entity_ids` as required
indexes; this module indexes those plus every other field
`retrieval.types.Filter` can constrain, because an unindexed filter is a latent
performance cliff and the index is cheap.

**The payload is a filter index, not a document store.** `docs/data-stores.md`
§3.3 forbids Signal bodies, `metadata` blobs and anything read for display from
living here: Qdrant is a *derived* store, rebuildable from PostgreSQL plus R2, and
the moment a UI renders a field straight out of a Qdrant payload that rebuild
guarantee is gone. `ChunkPayload` is therefore a closed dataclass rather than a
free `dict[str, Any]` -- the schema is the enforcement.

The one field here that is neither a filter nor provenance is `pipeline_version`
(as the sortable ordinal from `models/lineage.py`). It exists so "which store is
behind" is answerable by querying Qdrant, the way `docs/data-stores.md` §5.2
stores it in OpenSearch, and so a slow reprocess can be recognised as stale
rather than accepted as newer.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from qdrant_client import models

from backend.core.config import VectorDistance, get_settings
from backend.core.logging import get_logger
from backend.db.qdrant import ensure_collection as ensure_collection_geometry
from models.enums import Platform, SourceCategory
from retrieval.types import chunk_id_for
from retrieval.vector.qdrant_client import VectorStore, get_vector_store

__all__ = [
    "DEFAULT_HNSW_EF_CONSTRUCT",
    "DEFAULT_HNSW_M",
    "PAYLOAD_INDEXES",
    "ChunkPayload",
    "CollectionSpec",
    "PayloadField",
    "ensure_payload_indexes",
    "ensure_signal_collection",
    "signal_collection_spec",
]

_log = get_logger(__name__)


class PayloadField(enum.StrEnum):
    """Every key a point payload may carry. There are no others.

    A `StrEnum` rather than bare string constants so the filter compiler and the
    indexer cannot drift apart on a spelling: the member is a `str` at the wire
    boundary, but a typo is an `AttributeError` at import rather than a filter
    that matches nothing at runtime. A filter naming a payload key that was never
    written matches zero points and raises nothing -- indistinguishable from "no
    results for that query".
    """

    SIGNAL_ID = "signal_id"
    CHUNK_INDEX = "chunk_index"
    PLATFORM = "platform"
    SOURCE = "source"
    PUBLISHED_AT = "published_at"
    LANGUAGE = "language"
    ENTITY_IDS = "entity_ids"
    CONFIDENCE = "confidence"
    TENANT_ID = "tenant_id"
    PIPELINE_VERSION = "pipeline_version"


PAYLOAD_INDEXES: Final[Mapping[PayloadField, models.PayloadSchemaType]] = {
    # Not a filter in the retrieval path, but the key `delete_signal()` selects
    # on for erasure and for demoting a Signal that lost a canonical election
    # (`retrieval/vector/indexer.py`). Unindexed, an erasure would scan the whole
    # collection -- and erasure runs against a deadline someone else set.
    PayloadField.SIGNAL_ID: models.PayloadSchemaType.KEYWORD,
    # Indexed so a re-chunk can delete the orphaned tail with a range condition
    # instead of enumerating point ids it no longer knows.
    PayloadField.CHUNK_INDEX: models.PayloadSchemaType.INTEGER,
    PayloadField.PLATFORM: models.PayloadSchemaType.KEYWORD,
    PayloadField.SOURCE: models.PayloadSchemaType.KEYWORD,
    # DATETIME rather than KEYWORD: a keyword index cannot answer a range, so a
    # time-windowed query -- which is nearly every OmniSense query -- would fall
    # back to a scan while the collection info still reports the field as
    # indexed.
    PayloadField.PUBLISHED_AT: models.PayloadSchemaType.DATETIME,
    PayloadField.LANGUAGE: models.PayloadSchemaType.KEYWORD,
    # A list-valued field. Qdrant indexes each element, so `MatchAny` over entity
    # ids is a posting-list lookup rather than a per-point list membership test.
    PayloadField.ENTITY_IDS: models.PayloadSchemaType.KEYWORD,
    PayloadField.CONFIDENCE: models.PayloadSchemaType.FLOAT,
    # Mandatory on every query (`docs/retrieval.md` §7), so it becomes the most
    # selective condition in the collection the moment there is a second tenant.
    PayloadField.TENANT_ID: models.PayloadSchemaType.KEYWORD,
    PayloadField.PIPELINE_VERSION: models.PayloadSchemaType.INTEGER,
}
"""Field -> index type, for every field that may appear in a filter.

Deliberately covers *all* of `PayloadField`. The alternative -- index only what
today's queries filter on -- makes tomorrow's filter a silent full scan, and a
keyword index over a field with a few dozen distinct values costs nothing next to
a 1536-dimensional vector.
"""

DEFAULT_HNSW_M: Final[int] = 16
DEFAULT_HNSW_EF_CONSTRUCT: Final[int] = 128
"""HNSW build parameters (`docs/retrieval.md` §5): Qdrant's defaults, unmeasured.

Stated explicitly rather than left to the server so a Qdrant upgrade that changes
its defaults cannot quietly change recall on an existing deployment -- the
collection would be built one way and every document here would describe another.
"""


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """The full definition of the signal-chunk collection.

    Frozen and passed around explicitly rather than read from settings at each
    use, because the re-embedding swap in `.env.example` §11 builds a *second*
    collection with a different vector size alongside the live one. Code that
    reads global settings at every call cannot express "this indexer writes the
    new collection while that searcher reads the old one".
    """

    name: str
    vector_size: int
    distance: VectorDistance
    hnsw_m: int = DEFAULT_HNSW_M
    hnsw_ef_construct: int = DEFAULT_HNSW_EF_CONSTRUCT

    def __post_init__(self) -> None:
        if self.vector_size <= 0:
            raise ValueError(f"vector_size must be positive, got {self.vector_size}")
        if not self.name:
            raise ValueError("collection name must not be empty")

    def vectors_config(self) -> models.VectorParams:
        """The geometry, in Qdrant's vocabulary. Fixed at creation, forever."""
        return models.VectorParams(
            size=self.vector_size,
            distance=_QDRANT_DISTANCE[self.distance],
        )

    def hnsw_config(self) -> models.HnswConfigDiff:
        """Index-build parameters. Alterable later, unlike the geometry above."""
        return models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct)

    @property
    def payload_indexes(self) -> Mapping[PayloadField, models.PayloadSchemaType]:
        return PAYLOAD_INDEXES


# Duplicated from `backend/db/qdrant.py` rather than imported: that mapping is
# private to the kernel module, and the whole point of mapping explicitly is that
# a new `VectorDistance` member must fail loudly in every place that translates
# it rather than defaulting to cosine somewhere.
_QDRANT_DISTANCE: Final[dict[VectorDistance, models.Distance]] = {
    VectorDistance.COSINE: models.Distance.COSINE,
    VectorDistance.DOT: models.Distance.DOT,
    VectorDistance.EUCLID: models.Distance.EUCLID,
    VectorDistance.MANHATTAN: models.Distance.MANHATTAN,
}


def signal_collection_spec(
    *,
    name: str | None = None,
    vector_size: int | None = None,
    distance: VectorDistance | None = None,
) -> CollectionSpec:
    """The configured collection, with overrides for a re-embedding migration.

    Everyday callers pass nothing: the name is `QDRANT_COLLECTION`, the size is
    `EMBEDDING_DIMENSIONS`, the metric is `QDRANT_DISTANCE`. The triple
    (provider, model, dimensions) fixes the collection identity
    (`docs/retrieval.md` §5), so a size passed here that disagrees with the live
    collection is caught by `ensure_signal_collection()` rather than by the first
    upsert -- which is after the embedding provider has already been paid.
    """
    settings = get_settings()
    return CollectionSpec(
        name=name or settings.qdrant.collection,
        vector_size=(
            vector_size if vector_size is not None else settings.embedding.dimensions
        ),
        distance=distance or settings.qdrant.distance,
    )


@dataclass(frozen=True, slots=True)
class ChunkPayload:
    """The payload written with every point. A closed set, by design.

    Everything here is either filtered on or used to decide whether a point is
    stale. Nothing here is for display: a citation is resolved from PostgreSQL by
    `chunk_id`, never out of this payload (`docs/data-stores.md` §3.3).
    """

    signal_id: str
    chunk_index: int
    tenant_id: str = "default"
    platform: Platform = Platform.UNKNOWN
    source: SourceCategory = SourceCategory.UNKNOWN
    published_at: datetime | None = None
    language: str | None = None
    entity_ids: Sequence[str] = ()
    confidence: float = 0.0
    pipeline_version: int = 0
    """`pipeline_version` as the sortable ordinal from `models/lineage.py`.

    An integer, not the `"1.10.0"` string. Compared as text `'1.10.0' >= '1.9.0'`
    is False, so a staleness filter would invert the moment a version component
    reached 10 -- silently preferring the older write, exactly the corruption the
    ordinal exists to prevent.
    """

    @property
    def chunk_id(self) -> str:
        return chunk_id_for(self.signal_id, self.chunk_index)

    def to_payload(self) -> dict[str, Any]:
        """Serialise to the wire form Qdrant stores.

        `published_at` is emitted as an RFC 3339 string because that is what a
        `DATETIME` payload index parses. Timezone-naive input is rejected rather
        than assumed to be UTC: guessing the zone can move a Signal across a
        time-window boundary by up to a day, and nothing downstream can detect
        that it happened.
        """
        if self.published_at is not None and self.published_at.tzinfo is None:
            raise ValueError(
                f"published_at for chunk {self.chunk_id!r} is timezone-naive; "
                "Qdrant range filters compare instants, and assuming UTC here "
                "would silently move the Signal across a time-window boundary."
            )
        return {
            PayloadField.SIGNAL_ID.value: self.signal_id,
            PayloadField.CHUNK_INDEX.value: self.chunk_index,
            PayloadField.TENANT_ID.value: self.tenant_id,
            PayloadField.PLATFORM.value: str(self.platform),
            PayloadField.SOURCE.value: str(self.source),
            PayloadField.PUBLISHED_AT.value: (
                self.published_at.isoformat() if self.published_at is not None else None
            ),
            PayloadField.LANGUAGE.value: self.language,
            # A list, never a bare string: Qdrant treats a string payload value
            # and a one-element list differently for `MatchAny`, and the
            # difference only shows up as a filter that matches nothing.
            PayloadField.ENTITY_IDS.value: list(self.entity_ids),
            PayloadField.CONFIDENCE.value: float(self.confidence),
            PayloadField.PIPELINE_VERSION.value: int(self.pipeline_version),
        }


async def ensure_signal_collection(
    client: VectorStore | None = None,
    spec: CollectionSpec | None = None,
) -> CollectionSpec:
    """Create or verify the collection, then create any missing payload indexes.

    Idempotent and safe to call from every worker replica at boot, which is the
    only way it is ever called.

    Geometry is delegated to `backend/db/qdrant.ensure_collection()`: the L1k
    kernel owns the creation-race handling and the mismatch guard, and a second
    implementation here would eventually disagree with it about which one
    refuses.  What this function adds is the payload indexes, which that module
    deliberately leaves to the retrieval layer.

    Raises:
        ConfigurationError: propagated from the geometry check when the live
            collection's vector size or distance metric differs from `spec`.
    """
    resolved = spec or signal_collection_spec()
    await ensure_collection_geometry(
        resolved.name,
        vector_size=resolved.vector_size,
        distance=resolved.distance,
    )
    await ensure_payload_indexes(client or get_vector_store(), resolved)
    return resolved


async def ensure_payload_indexes(
    client: VectorStore, spec: CollectionSpec
) -> list[PayloadField]:
    """Create the payload indexes that are missing. Returns the ones created.

    Existing indexes are read from the collection info and skipped rather than
    re-created blindly. Re-creating is mostly harmless -- Qdrant accepts an
    identical definition -- but on a large collection it can trigger a rebuild,
    and during a rolling restart every replica calling this at boot would then
    serialise behind each other's rebuilds.

    A field indexed with the *wrong* type is reported, not corrected. Dropping
    and re-creating an index in place removes the one live queries are currently
    using; the right response is an operator rebuilding from PostgreSQL, not a
    boot path silently deoptimising every concurrent query.
    """
    # By keyword, like every other call into the client: `qdrant_client.py` makes the
    # point that a positional call breaks on the next release that inserts a parameter,
    # and it breaks by binding the wrong argument rather than by raising.
    info = await client.get_collection(collection_name=spec.name)
    existing = dict(getattr(info, "payload_schema", None) or {})

    created: list[PayloadField] = []
    for field, schema in spec.payload_indexes.items():
        present = existing.get(field.value)
        if present is not None:
            _warn_on_type_mismatch(spec.name, field, schema, present)
            continue
        await client.create_payload_index(
            collection_name=spec.name,
            field_name=field.value,
            field_schema=schema,
            wait=True,
        )
        created.append(field)
    return created


def _warn_on_type_mismatch(
    collection: str,
    field: PayloadField,
    expected: models.PayloadSchemaType,
    present: Any,
) -> None:
    """Log a payload index whose type is not the one this code assumes.

    The case that matters: `published_at` indexed as `keyword` by an older build.
    Range filters against it still return *correct* results -- by scanning -- so
    the only symptom is latency, which is why this is worth a log line naming the
    field rather than nothing at all.
    """
    actual = getattr(present, "data_type", present)
    if str(actual) == str(expected):
        return
    _log.warning(
        "qdrant.payload_index.type_mismatch",
        collection=collection,
        field=field.value,
        expected=str(expected),
        actual=str(actual),
        detail=(
            "filters on this field may degrade to a full scan; drop the "
            "collection and re-index from PostgreSQL to correct it"
        ),
    )
