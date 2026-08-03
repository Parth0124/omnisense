"""Qdrant client lifecycle and collection-geometry enforcement.

Qdrant holds one point per embedded chunk (`docs/data-stores.md` §3.3). It is a
*derived* store: everything in it is rebuildable from PostgreSQL plus R2, which
is why this module is free to be opinionated about refusing to start against a
collection whose shape is wrong -- dropping and rebuilding is always a legal
recovery, and proceeding is not.

Structure follows `backend/db/session.py`: a lazily-built module-level singleton,
no I/O at import time, `get_qdrant()` / `check_qdrant()` / `dispose_qdrant()`.

Two decisions are encoded here.

**Geometry is verified, not assumed.** `ensure_collection()` refuses to proceed
when an existing collection's vector size or distance metric differs from what
this process is configured for. Qdrant itself only notices a dimension mismatch
at *upsert* time -- by then the embedding provider has already been paid for the
batch, and the failure surfaces in a worker as a rejected write rather than at
boot as a configuration problem. `docs/signal-model.md` §9 question 1 is exactly
this hazard: the provider/model/dimension triple fixes the collection geometry
and changing it after ingestion begins requires re-embedding every Signal.

**There is no `require_qdrant()`,** deliberately, and that is the difference from
`session.py`. `docs/architecture.md` §7.3 makes Qdrant a degrade-not-fail
dependency: without it retrieval falls back to BM25 plus graph and ingestion
queues embeddings. A helper that turns "Qdrant is down" into a 503 would invite
call sites that hard-fail where the design says they must degrade. Callers ask
`check_qdrant()` and take the fallback path.

Scope: this module owns the connection and the *minimum viable* collection --
vector size and distance, the two parameters that cannot be changed after
creation. ANN tuning that can be changed later (HNSW parameters, quantization,
payload field indexes) belongs to the retrieval layer in
`retrieval/vector/collections.py`, which sits above this one.

Layer note: **L1k kernel** (`docs/architecture.md` §6.1) -- importable by
`services/`, `agents/`, `workers/`, `backend/api/` and `scripts/`, but never by
`connectors/`.
"""

from __future__ import annotations

import asyncio
from typing import Final

# `grpc` arrives transitively with qdrant-client and ships no type stubs, and it
# is not in the `ignore_missing_imports` override list in `pyproject.toml`. The
# ignore is narrower than adding it there, since this is the only module that
# touches gRPC directly. Why it is needed at all: see the race handler in
# `ensure_collection()`.
from grpc import RpcError  # type: ignore[import-untyped]
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.core.config import VectorDistance, get_settings
from backend.core.exceptions import ConfigurationError

__all__ = [
    "check_qdrant",
    "dispose_qdrant",
    "ensure_collection",
    "get_qdrant",
]

_client: AsyncQdrantClient | None = None

# Mapped explicitly rather than by title-casing the enum value. The two
# vocabularies are independent -- ours is lowercase config, Qdrant's is
# capitalized wire format -- and a silent mismatch here would build a collection
# with the wrong metric, which is only detectable by noticing that search
# results are subtly bad.
_DISTANCE: Final[dict[VectorDistance, models.Distance]] = {
    VectorDistance.COSINE: models.Distance.COSINE,
    VectorDistance.DOT: models.Distance.DOT,
    VectorDistance.EUCLID: models.Distance.EUCLID,
    VectorDistance.MANHATTAN: models.Distance.MANHATTAN,
}

# `QDRANT_TIMEOUT_SECONDS` (default 30) is sized for bulk upserts, not for
# answering "are you there?". A readiness probe that can block for 30 seconds is
# indistinguishable from a failing one to any orchestrator, while still holding a
# request slot, so the probe gets its own budget.
_PROBE_TIMEOUT_SECONDS: Final[float] = 5.0


def get_qdrant() -> AsyncQdrantClient:
    """Return the process-wide Qdrant client, creating it on first use.

    Construction opens no socket, so importing this module -- or calling this in a
    test -- does not require a running Qdrant.
    """
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncQdrantClient(
            url=settings.qdrant.url,
            api_key=_api_key(),
            prefer_grpc=settings.qdrant.prefer_grpc,
            timeout=settings.qdrant.timeout_seconds,
            # The client otherwise spawns a daemon thread in its constructor to
            # fetch the server version and compare it against its own. That is a
            # blocking HTTP call on a thread we do not control, it makes object
            # construction depend on the network, and when Qdrant is down it
            # prints a warning from a background thread during unit tests that
            # never intended to touch it.
            check_compatibility=False,
        )
    return _client


def _api_key() -> str | None:
    """The configured API key, with a blank value normalized to "absent".

    `.env.example` ships `QDRANT_API_KEY=` because the local container has no
    auth, and pydantic reads that as `SecretStr("")` rather than `None`. The
    client treats any non-`None` key as "authenticate": it would attach an empty
    `api-key` header and warn that a key is being used over an insecure
    connection. Under `filterwarnings = ["error"]` (`pyproject.toml`) that warning
    is an exception, so the blank default would break the test suite rather than
    mean what it plainly says.
    """
    api_key = get_settings().qdrant.api_key
    if api_key is None:
        return None
    return api_key.get_secret_value() or None


async def ensure_collection(
    collection: str | None = None,
    *,
    vector_size: int | None = None,
    distance: VectorDistance | None = None,
) -> bool:
    """Create the collection if it is missing; verify its geometry if it exists.

    Idempotent, and safe to call concurrently from every worker at boot.

    The overrides exist for the re-embedding swap described in `.env.example`
    §11: a migration to a new embedding model builds the new collection alongside
    the live one, which means naming a size and metric that are not (yet) the
    configured ones. Everyday callers pass nothing.

    Args:
        collection: Collection name. Defaults to `QDRANT_COLLECTION`.
        vector_size: Vector dimensionality. Defaults to `EMBEDDING_DIMENSIONS`.
        distance: Similarity metric. Defaults to `QDRANT_DISTANCE`.

    Returns:
        True if this call created the collection, False if it already existed
        with a matching geometry.

    Raises:
        ConfigurationError: The collection exists with a different vector size or
            distance metric, or with named vectors this code cannot address.
            Also raised when creation fails for a reason other than losing a
            race, since at bootstrap that is a deployment problem, not a request.

    A Qdrant that is simply *unreachable* surfaces as the underlying
    `qdrant_client` transport exception rather than a typed one. That is
    deliberate: the caller's response is to retry later, which is a different
    decision from the ones above, and flattening both into one exception type
    would hide it.
    """
    settings = get_settings()
    name = collection or settings.qdrant.collection
    size = vector_size if vector_size is not None else settings.embedding.dimensions
    metric = _to_qdrant_distance(distance or settings.qdrant.distance)

    client = get_qdrant()
    if not await client.collection_exists(name):
        try:
            await client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=size, distance=metric),
            )
        except (UnexpectedResponse, RpcError) as exc:
            # Losing a creation race is not an error: `scripts/init_databases.py`
            # and every indexing worker replica call this during startup. The two
            # transports report the loser differently (HTTP 409 versus gRPC
            # ALREADY_EXISTS), so rather than matching on either code we ask the
            # server. If the collection is there now we fall through to the
            # geometry check below, which is the part that actually matters --
            # and if the winner created it with a different shape, that check is
            # what catches it.
            if not await _exists_quietly(client, name):
                raise ConfigurationError(
                    f"Could not create Qdrant collection {name!r}: {exc}",
                    details={"collection": name},
                    cause=exc,
                ) from exc
        else:
            return True

    await _assert_geometry(client, name, size, metric)
    return False


async def check_qdrant() -> bool:
    """Probe Qdrant for `/readyz`. Never raises.

    `get_collections()` rather than a bare TCP connect: it exercises the same
    REST or gRPC path real queries use and it fails when the API key is wrong,
    which a socket check would report as healthy right up until the first search.

    Returns a bool rather than raising because readiness aggregates several
    dependencies and one being down must not prevent reporting on the others
    (`docs/observability.md`).
    """
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            await get_qdrant().get_collections()
    except Exception:
        return False
    return True


async def dispose_qdrant() -> None:
    """Close the client and reset the singleton. Called from lifespan shutdown."""
    global _client
    # Detach before closing. `close()` can raise on a gRPC channel that never
    # connected, and if that propagated with the singleton still installed, every
    # later `get_qdrant()` would hand back a closed client instead of building a
    # fresh one.
    client, _client = _client, None
    if client is not None:
        await client.close()


def _to_qdrant_distance(distance: VectorDistance) -> models.Distance:
    """Translate the configured metric into Qdrant's vocabulary."""
    try:
        return _DISTANCE[distance]
    except KeyError as exc:  # A new VectorDistance member added without a mapping.
        raise ConfigurationError(
            f"QDRANT_DISTANCE={distance.value!r} has no Qdrant equivalent in "
            f"backend/db/qdrant.py; mapped metrics are "
            f"{', '.join(sorted(m.value for m in _DISTANCE))}.",
            details={"distance": distance.value},
            cause=exc,
        ) from exc


async def _exists_quietly(client: AsyncQdrantClient, name: str) -> bool:
    """Existence check that reports "no" instead of raising.

    Used only from the creation-race handler, where a second failure must not
    replace the original creation failure with a less informative one.
    """
    try:
        return await client.collection_exists(name)
    except Exception:
        return False


async def _assert_geometry(
    client: AsyncQdrantClient,
    name: str,
    expected_size: int,
    expected_distance: models.Distance,
) -> None:
    """Raise `ConfigurationError` unless the live collection matches expectations.

    This is the guard described in the module docstring. Vector size and distance
    are fixed at creation: Qdrant will not alter either, so a mismatch is never
    something the process can adapt to. Left unchecked, the size mismatch is
    discovered by the first upsert -- after the embedding provider has been paid
    for that batch -- and the metric mismatch is not discovered at all, it just
    returns quietly worse results forever.
    """
    params = (await client.get_collection(name)).config.params.vectors

    if params is None:
        raise ConfigurationError(
            f"Qdrant collection {name!r} exists but declares no dense vector "
            f"configuration, so it cannot hold {expected_size}-dimensional "
            "embeddings. It was created by something other than OmniSense; drop "
            "it or point QDRANT_COLLECTION elsewhere.",
            details={"collection": name, "expected_size": expected_size},
        )

    if isinstance(params, dict):
        # Named vectors. Every point would then have to key its vectors by name,
        # and OmniSense writes a single unnamed vector per chunk.
        raise ConfigurationError(
            f"Qdrant collection {name!r} exists with named vectors "
            f"({', '.join(sorted(params))}), but OmniSense writes a single "
            "unnamed vector per chunk. Drop the collection or point "
            "QDRANT_COLLECTION at one created by ensure_collection().",
            details={"collection": name, "named_vectors": sorted(params)},
        )

    if params.size == expected_size and params.distance == expected_distance:
        return

    raise ConfigurationError(
        f"Qdrant collection {name!r} already exists with "
        f"size={params.size} distance={params.distance.value}, but this process "
        f"is configured for size={expected_size} "
        f"distance={expected_distance.value} (EMBEDDING_DIMENSIONS and "
        "QDRANT_DISTANCE). Neither is alterable in place. Refusing to continue: "
        "the size mismatch would otherwise surface as a rejected upsert in a "
        "worker, after the embedding provider had already charged for the batch, "
        "and a metric mismatch would not surface at all. Either restore the "
        "previous configuration, or create the new collection under a different "
        "QDRANT_COLLECTION and re-embed from PostgreSQL with scripts/reindex.py.",
        details={
            "collection": name,
            "actual_size": params.size,
            "actual_distance": params.distance.value,
            "expected_size": expected_size,
            "expected_distance": expected_distance.value,
        },
    )
