"""The narrow Qdrant port the retrieval layer talks to, and the search budget.

There is deliberately **no second client here**. `backend/db/qdrant.py` owns the
process-wide `AsyncQdrantClient`, its lifecycle and its disposal; a second
connection pool built in the retrieval layer would double the socket count,
survive `dispose_qdrant()` and keep an event loop alive at shutdown. What this
module adds is the *shape* of the dependency rather than the dependency itself.

`VectorStore` is the subset of `AsyncQdrantClient` that `retrieval/vector/`
actually uses -- five methods out of roughly sixty. Two things follow from
declaring it:

- A unit test fakes five methods instead of importing a client library that
  wants a URL. `docs/testing-strategy.md` fixes the unit suite as "no external
  services", and a protocol is what makes that cheap rather than a mocking
  exercise.
- The blast radius of a `qdrant-client` upgrade is one file. When `search()` was
  superseded by `query_points()` that was a signature change in a library we do
  not control; a codebase calling the client directly from four modules
  discovers such a change four times.

The search budget lives here too, because `hnsw_ef` is the one search parameter
that trades recall for latency at *query* time. It belongs next to the client
rather than buried in a default argument of a search function, where nobody
tuning recall would think to look.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Protocol, runtime_checkable

from qdrant_client import models

from backend.db.qdrant import get_qdrant

__all__ = [
    "DEFAULT_SEARCH_HNSW_EF",
    "VectorStore",
    "get_vector_store",
    "search_params",
]

DEFAULT_SEARCH_HNSW_EF: Final[int] = 128
"""How many candidates HNSW keeps in flight while descending the graph.

`docs/retrieval.md` §5: raise for recall, lower for latency, unmeasured. The
failure mode worth naming is that a too-low value neither errors nor returns
fewer results -- it returns *different, worse* neighbours at the same count,
which is invisible without the evaluation harness in `retrieval/evaluation/`.
"""


@runtime_checkable
class VectorStore(Protocol):
    """The Qdrant surface `retrieval/vector/` depends on. Nothing wider.

    Structurally satisfied by `AsyncQdrantClient`, so `get_vector_store()` needs
    no adapter, and satisfied by a five-method fake in tests. Arguments are named
    the way the client names them and passed by keyword at every call site;
    positional calls would break on the next release that inserts a parameter.
    """

    async def query_points(
        self,
        collection_name: str,
        query: Any = None,
        *,
        query_filter: models.Filter | None = ...,
        search_params: models.SearchParams | None = ...,
        limit: int = ...,
        with_payload: Any = ...,
        with_vectors: Any = ...,
        **kwargs: Any,
    ) -> Any:
        """ANN search. Returns an object carrying `.points`."""
        ...

    async def upsert(
        self,
        collection_name: str,
        points: Sequence[Any],
        *,
        wait: bool = ...,
        **kwargs: Any,
    ) -> Any:
        """Insert or overwrite points, keyed by point id."""
        ...

    async def delete(
        self,
        collection_name: str,
        points_selector: Any,
        *,
        wait: bool = ...,
        **kwargs: Any,
    ) -> Any:
        """Delete by point ids or by filter."""
        ...

    async def get_collection(self, collection_name: str) -> Any:
        """Collection info, including `payload_schema`."""
        ...

    async def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: Any = None,
        *,
        wait: bool = ...,
        **kwargs: Any,
    ) -> Any:
        """Create a payload index on one field."""
        ...


def get_vector_store() -> VectorStore:
    """The shared client, typed down to what the retrieval layer may use.

    Narrowing rather than wrapping: the object returned *is* the singleton from
    `backend/db/qdrant.py`, so `dispose_qdrant()` still closes it and nothing
    here holds a reference that outlives the application's lifespan hooks.
    """
    return get_qdrant()


def search_params(
    hnsw_ef: int | None = None, *, exact: bool = False
) -> models.SearchParams:
    """Per-query ANN parameters.

    `exact=True` bypasses HNSW for a brute-force scan. It exists for the
    evaluation harness, which needs the *true* nearest neighbours to measure what
    fraction of them the approximate index returned -- recall@k measured against
    the approximate index measures nothing at all. It must never appear on a
    request path: the cost is linear in collection size.
    """
    return models.SearchParams(
        hnsw_ef=hnsw_ef if hnsw_ef is not None else DEFAULT_SEARCH_HNSW_EF,
        exact=exact,
    )
