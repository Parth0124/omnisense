"""The keyword backend and the chunk writer: the two halves that touch the cluster.

`retrieval/keyword/index.py` says what a document is, `query_builder.py` says what a
query is, and neither of them performs I/O. This module is the only place in
`retrieval/keyword/` that talks to OpenSearch, which is what keeps the other two
testable as pure functions.

**There is no second client here.** `backend/db/opensearch.py` owns the process-wide
`AsyncOpenSearch`, its connection pool and its disposal. A client constructed in the
retrieval layer would double the socket count, survive `dispose_opensearch()` and hold
an `aiohttp` session open past shutdown -- which surfaces as "Unclosed client session"
on stderr after logging is gone. What this module adds is the *shape* of the
dependency: `KeywordStore` is the three methods of the client that `retrieval/keyword/`
uses, out of the roughly two hundred the OpenSearch API exposes. Declaring it buys a
unit suite that fakes three methods rather than mocking a library that wants a URL
(`docs/testing-strategy.md`), and it confines the blast radius of an `opensearch-py`
upgrade to one file.

Three failure modes shape the code, and every one of them arrives as **HTTP 200**.

**A bulk request that rejected half its documents is a 200.** The per-item statuses
live in the response body, so a caller that checks only for a raised exception records
a successful indexing run for documents that were never written. `dynamic: "strict"`
(`index.py`) makes an unexpected field a rejected *item*, so the strictness that module
buys is worth nothing unless someone reads the items -- which is what
`_read_bulk_response()` exists to do.

**A search that lost a shard is also a 200.** `_shards.failed > 0` and
`timed_out: true` both return the hits from the surviving shards and nothing else.
Recall silently drops by the fraction of the corpus that shard held, and the query
looks healthy. That is logged loudly rather than raised: partial results are still
results, and `HybridRetriever` can only see a backend as up or down.

**A stale write that overwrote newer enrichment is a 200 too.** `scripts/reindex.py`
replaying last month's corpus and a live enrichment update for the same Signal are two
writers to one `_id` with no ordering between them, and last-write-wins resolves that
by wall clock -- which is exactly backwards when the backfill is the slow one. Every
write therefore carries the `pipeline_version` ordinal as the external document version
(`docs/data-stores.md` §5.2), so the cluster rejects the older writer with a 409 that
this module counts as a *success of the mechanism* rather than an error.

Layer note: L1 (`docs/architecture.md` §6.1). Imports `models/` and the `backend/core`
+ `backend/db` kernel; nothing above it.
"""

from __future__ import annotations

import enum
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from backend.core.exceptions import ExternalServiceError
from backend.core.logging import get_logger
from backend.db.opensearch import get_opensearch
from models.signal import Keyword
from retrieval.keyword.index import ChunkDocument, ChunkField, IndexSpec, chunk_index_spec
from retrieval.keyword.query_builder import (
    DEFAULT_OPTIONS,
    QueryOptions,
    build_search_body,
    candidates_from_response,
)
from retrieval.types import Backend, Candidate, RetrievalRequest, split_chunk_id

__all__ = [
    "DEFAULT_BULK_BATCH_SIZE",
    "BulkIndexError",
    "BulkItemFailure",
    "ExpansionProvider",
    "IndexOutcome",
    "KeywordBackend",
    "KeywordIndexer",
    "KeywordStore",
    "VersionType",
    "get_keyword_store",
]

_log = get_logger(__name__)

DEFAULT_BULK_BATCH_SIZE: Final[int] = 500
"""Documents per bulk request.

Sized by request *body*, not by document count: a chunk is roughly 1-2 KB of text plus
its metadata, so 500 documents is a ~1 MB request -- large enough to amortise the round
trip, small enough that a timeout re-sends 1 MB rather than 50. The usual guidance is
5-15 MB per bulk; this stays under it deliberately, because the retry cost of an
oversized batch is paid at exactly the moment the cluster is already struggling.
"""

_VERSION_CONFLICT: Final[str] = "version_conflict_engine_exception"
_OK_STATUS: Final[int] = 300
_CONFLICT_STATUS: Final[int] = 409

ExpansionProvider = Callable[[RetrievalRequest], Awaitable[Sequence[str]]]
"""How the caller supplies alias and canonical-name expansions.

A callable rather than a graph client, for the same reason `retrieval/vector/search.py`
takes a `QueryEmbedder`: `retrieval/` reads `models/` and takes everything else as an
argument (`docs/architecture.md` §6.1). In production this is backed by
`retrieval/graph_retrieval/expansion.py`; in an evaluation replay it is a dict lookup;
in a test it is a lambda.
"""


@runtime_checkable
class KeywordStore(Protocol):
    """The OpenSearch surface `retrieval/keyword/` depends on. Nothing wider.

    Structurally satisfied by `AsyncOpenSearch`, so `get_keyword_store()` needs no
    adapter. Every call site passes arguments by keyword; the client's methods are
    generated from the REST spec and gain parameters between releases, so a positional
    call is a break waiting for the next upgrade.
    """

    async def search(self, **kwargs: Any) -> Any:
        """Run a `_search`. Returns the decoded response body."""
        ...

    async def bulk(self, **kwargs: Any) -> Any:
        """Apply a newline-delimited batch of index/delete actions."""
        ...

    async def delete_by_query(self, **kwargs: Any) -> Any:
        """Delete every document matching a query."""
        ...


def get_keyword_store() -> KeywordStore:
    """The shared client, typed down to what the retrieval layer may use.

    Narrowing rather than wrapping: the object returned *is* the singleton from
    `backend/db/opensearch.py`, so `dispose_opensearch()` still closes it and nothing
    here holds a reference that outlives the application's lifespan hooks.
    """
    return get_opensearch()


# --------------------------------------------------------------------------- #
# Read path
# --------------------------------------------------------------------------- #


class KeywordBackend:
    """BM25 retrieval over the chunk index. Satisfies `SearchBackend`.

    Holds a client, an index spec and the scoring knobs; no per-request state, so one
    instance per process serves concurrent queries.

    The index name is held rather than read from settings at call time so a
    reindex-and-swap migration (`docs/data-stores.md` §3.5) can point a searcher at the
    old index while an indexer fills the new one. Code that consults global settings on
    every call cannot express two index names at once, which is the entire migration.
    """

    backend: Backend = Backend.KEYWORD

    def __init__(
        self,
        client: KeywordStore,
        spec: IndexSpec | None = None,
        *,
        options: QueryOptions = DEFAULT_OPTIONS,
        expand: ExpansionProvider | None = None,
    ) -> None:
        self._client = client
        self._spec = spec or chunk_index_spec()
        self._options = options
        self._expand = expand

    @property
    def index(self) -> str:
        return self._spec.name

    @property
    def options(self) -> QueryOptions:
        return self._options

    async def search(self, request: RetrievalRequest, *, limit: int) -> Sequence[Candidate]:
        """Expand, then run one filtered BM25 query.

        Exceptions propagate. `HybridRetriever` catches them, records the backend as
        failed and continues on the remaining two (`docs/architecture.md` §7.3), so
        swallowing a cluster outage here would produce an empty candidate list
        indistinguishable from a query with no lexical match -- and the diagnostics
        would report a healthy run that happened to find nothing, at full confidence.
        """
        return await self.search_with_terms(
            request, limit=limit, expansions=await self._expansions(request)
        )

    async def search_with_terms(
        self,
        request: RetrievalRequest,
        *,
        limit: int,
        expansions: Sequence[str] = (),
        keywords: Sequence[Keyword] = (),
    ) -> list[Candidate]:
        """Query with expansions and keyword boosts the caller already holds.

        The entry point for anything that must not pay for a graph round trip: the
        evaluation harness sweeping `QueryOptions`, and a planner that already resolved
        the entities it cares about.

        `keywords` only ever boost. They are passed into the `should` half of the query
        (`query_builder.py`), where they cannot satisfy `minimum_should_match` and so
        cannot widen the result set -- a document whose extracted keyword list happens
        to contain the query term is not a lexical match.
        """
        body = build_search_body(
            request,
            limit=limit,
            options=self._options,
            expansions=expansions,
            keywords=keywords,
        )
        response = await self._client.search(index=self._spec.name, body=body)
        self._warn_on_partial(response, request)

        candidates = candidates_from_response(response)
        _log.debug(
            "opensearch.search",
            index=self._spec.name,
            limit=limit,
            returned=len(candidates),
            filters=_filter_count(body),
            expansions=len(expansions),
            tenant_id=request.filters.tenant_id,
        )
        return candidates

    # ------------------------------------------------------------ internals --

    async def _expansions(self, request: RetrievalRequest) -> Sequence[str]:
        """Alias expansion, which must never be able to fail the query.

        Expansion is recall on top of a query that already works: "DDOG" and "Datadog
        Inc" are one entity to the graph and two unrelated strings to BM25. Letting a
        Neo4j timeout propagate would mark the *keyword* backend as failed in
        `RetrievalDiagnostics` and drop a third of the fan-out over an optimisation --
        `HybridRetriever` treats its own expander the same way, for the same reason.
        Logged at warning rather than swallowed, because expansion that has been
        quietly off for a week looks exactly like a corpus with poor alias coverage.
        """
        if self._expand is None or not request.seed_entity_ids:
            return ()
        try:
            return await self._expand(request)
        except Exception as exc:  # noqa: BLE001 -- expansion is an optimisation
            _log.warning(
                "opensearch.expansion.failed",
                index=self._spec.name,
                error=type(exc).__name__,
                detail="querying without alias expansion; recall on aliases is reduced",
            )
            return ()

    def _warn_on_partial(self, response: Any, request: RetrievalRequest) -> None:
        """Notice the 200 that only searched part of the corpus.

        A search whose shards partly failed, or that hit `timeout`, returns the hits
        from the shards that answered and a `200`. Recall drops by whatever fraction of
        the index those shards held, ranking still looks plausible, and nothing raises.

        Not fatal: partial results are results, and `HybridRetriever` has only two
        states for a backend. Raising would turn a 20% recall loss into a 100% one.
        """
        if not isinstance(response, Mapping):
            return
        shards = response.get("_shards")
        failed = shards.get("failed", 0) if isinstance(shards, Mapping) else 0
        timed_out = bool(response.get("timed_out", False))
        if not failed and not timed_out:
            return
        _log.warning(
            "opensearch.search.partial_results",
            index=self._spec.name,
            failed_shards=failed,
            total_shards=shards.get("total") if isinstance(shards, Mapping) else None,
            timed_out=timed_out,
            query=request.query[:120],
            detail=(
                "the cluster answered from a subset of shards; recall is reduced by "
                "the share of the corpus the missing shards hold, and the response "
                "reports success"
            ),
        )


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #


class VersionType(enum.StrEnum):
    """How the cluster compares an incoming write against the stored document."""

    EXTERNAL = "external"
    """Accept only a *strictly greater* version.

    Rejects a re-delivery of the same `pipeline_version` with a 409. That is safe --
    the stored document is byte-identical to the one being written -- but it means
    at-least-once delivery produces conflicts in normal operation, so a caller that
    treats a conflict as an error would page on a healthy Kafka rebalance.
    """

    EXTERNAL_GTE = "external_gte"
    """Accept an equal or greater version. The default here.

    Re-processing one Signal at the same `pipeline_version` -- a partition rebalance, a
    reconciler pass, a `scripts/reindex.py` run against unchanged code -- rewrites the
    same content in place and succeeds, which is what `docs/data-stores.md` §5.1 means
    by "at-least-once delivery, idempotent by id". A strictly *older* write is still
    rejected, which is the property the version exists for.
    """


@dataclass(frozen=True, slots=True)
class BulkItemFailure:
    """One rejected document, lifted out of the 200 response body."""

    chunk_id: str
    status: int
    type: str
    reason: str

    def __str__(self) -> str:
        return f"{self.chunk_id}: {self.status} {self.type} -- {self.reason}"


class BulkIndexError(ExternalServiceError):
    """OpenSearch rejected documents inside an otherwise successful bulk request.

    A distinct type because the handling is distinct: a rejected *item* is a property
    of that document (a field the strict mapping does not know, a date that will not
    parse), not of the cluster, so the retry that fixes a `ConnectionError` will fail
    forever here. `workers/` routes these to the DLQ by `chunk_id`, which is why the
    failures are carried on the exception rather than only formatted into its message.
    """

    code = "opensearch_bulk_index_error"
    default_message = "OpenSearch rejected one or more documents in a bulk request."

    def __init__(self, failures: Sequence[BulkItemFailure], *, index: str) -> None:
        shown = list(failures[:5])
        super().__init__(
            f"OpenSearch rejected {len(failures)} document(s) in index {index!r}: "
            + "; ".join(str(f) for f in shown)
            + ("; ..." if len(failures) > len(shown) else ""),
            details={
                "index": index,
                "rejected": len(failures),
                "chunk_ids": [f.chunk_id for f in shown],
            },
        )
        self.failures = tuple(failures)
        self.index = index


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    """What one `index_chunks()` call did. Returned for the reconciler's log.

    `indexed` and `conflicts` are separate counters because they mean opposite things.
    A conflict is the version guard doing its job -- a backfill losing to newer
    enrichment -- so a run that is *all* conflicts is a correct no-op, while a run that
    unexpectedly reports zero conflicts during a backfill means the guard is not armed.
    Collapsing them into "documents processed" would hide both.
    """

    indexed: int = 0
    conflicts: int = 0
    batches: int = 0
    index: str = ""
    chunk_ids: Sequence[str] = field(default_factory=tuple)
    conflicted_chunk_ids: Sequence[str] = field(default_factory=tuple)

    @property
    def submitted(self) -> int:
        return self.indexed + self.conflicts


class KeywordIndexer:
    """Upserts and deletes chunk documents in one index.

    Holds a client and a spec, no other state, so one instance per process is fine and
    concurrent calls are independent.
    """

    def __init__(
        self,
        client: KeywordStore,
        spec: IndexSpec | None = None,
        *,
        batch_size: int = DEFAULT_BULK_BATCH_SIZE,
        version_type: VersionType = VersionType.EXTERNAL_GTE,
        refresh: bool | str = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self._client = client
        self._spec = spec or chunk_index_spec()
        self._batch_size = batch_size
        self._version_type = version_type
        # `refresh=False` by default: a refresh is a Lucene segment flush, and forcing
        # one per bulk on an indexing run collapses throughput while producing tiny
        # segments the merge policy then has to clean up. Nothing needs
        # read-your-write from a derived store -- the API reads PostgreSQL -- so the
        # index refresh interval is allowed to do its job. Tests and
        # `scripts/reindex.py` pass `"wait_for"` because they assert on what is
        # searchable next.
        self._refresh = refresh

    @property
    def index(self) -> str:
        return self._spec.name

    @property
    def spec(self) -> IndexSpec:
        return self._spec

    async def index_chunks(self, documents: Iterable[ChunkDocument]) -> IndexOutcome:
        """Upsert chunk documents in batches, keyed by `chunk_id` and version-guarded.

        Idempotent twice over: the `_id` is the `chunk_id`, so a replay overwrites in
        place rather than adding a second copy of the corpus; and the external version
        is the `pipeline_version` ordinal, so a replay of *older* output is rejected
        instead of reinstating stale enrichment over newer.

        Raises:
            ValueError: a document with a non-positive `pipeline_version`, or two
                documents for one chunk in a single call.
            BulkIndexError: the cluster rejected documents for any reason other than
                the version guard.
        """
        pending: list[dict[str, Any]] = []
        batch_ids: list[str] = []
        all_ids: list[str] = []
        conflicted: list[str] = []
        seen: set[str] = set()
        indexed = 0
        conflicts = 0
        batches = 0

        for document in documents:
            self._assert_versioned(document)
            if document.chunk_id in seen:
                # Two entries for one chunk in a single call means the caller chunked
                # twice or merged two batches wrongly. The cluster would apply both
                # and keep whichever landed last -- a coin flip between two different
                # bodies for one citation span, which is not a thing to resolve
                # silently. Under `EXTERNAL` it is worse: the second item 409s and is
                # counted as a healthy version conflict.
                raise ValueError(
                    f"chunk {document.chunk_id!r} appears twice in one index_chunks() "
                    "call; the last write would win arbitrarily"
                )
            seen.add(document.chunk_id)
            pending.extend(self._index_action(document))
            batch_ids.append(document.chunk_id)
            all_ids.append(document.chunk_id)

            if len(batch_ids) >= self._batch_size:
                ok, conflict_ids = await self._flush(pending, batch_ids)
                indexed += ok
                conflicts += len(conflict_ids)
                conflicted.extend(conflict_ids)
                batches += 1
                pending, batch_ids = [], []

        if batch_ids:
            ok, conflict_ids = await self._flush(pending, batch_ids)
            indexed += ok
            conflicts += len(conflict_ids)
            conflicted.extend(conflict_ids)
            batches += 1

        return IndexOutcome(
            indexed=indexed,
            conflicts=conflicts,
            batches=batches,
            index=self._spec.name,
            chunk_ids=tuple(all_ids),
            conflicted_chunk_ids=tuple(conflicted),
        )

    async def delete_signal(self, signal_id: str) -> int:
        """Delete every chunk of a Signal. Returns how many documents went.

        Two operations that look different and are the same one:

        - **Erasure** (`docs/security-and-privacy.md`). A deletion request must reach
          the derived stores, and this index has no other record of what it holds for
          a Signal.
        - **Demotion.** A Signal that loses a canonical election becomes `DUPLICATE`
          and stops being retrievable (`models/enums.py`), so its documents must leave
          the index while its row stays in PostgreSQL.

        By query rather than by enumerating `_id`s, because the caller does not know
        how many chunks there were -- the chunk count is a property of the text *at the
        time it was chunked*. A caller re-deriving ids from a current chunk count
        leaves the tail behind whenever a re-chunk produced fewer chunks, and those
        orphans stay searchable forever, pointing at spans that no longer exist.

        `conflicts="proceed"` is load-bearing. `delete_by_query` takes a snapshot and
        then deletes; a document rewritten underneath it raises a version conflict, and
        the *default* is to abort the whole run -- leaving an erasure that deleted some
        of a Signal's chunks and reported failure, on the one operation where partial
        completion is a compliance incident. Proceeding lets the pass finish, and the
        chunk that was rewritten mid-flight is caught by the next reconciler sweep.

        No external version accompanies a delete: erasure is not enrichment, and it
        must win against every version, including a newer one still in flight.
        """
        response = await self._client.delete_by_query(
            index=self._spec.name,
            body={"query": {"term": {ChunkField.SIGNAL_ID.value: signal_id}}},
            conflicts="proceed",
            refresh=bool(self._refresh),
        )
        deleted = int((response or {}).get("deleted", 0))
        version_conflicts = int((response or {}).get("version_conflicts", 0))
        _log.info(
            "opensearch.delete_signal",
            index=self._spec.name,
            signal_id=signal_id,
            deleted=deleted,
            version_conflicts=version_conflicts,
        )
        return deleted

    async def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Delete specific chunks by `_id`. Returns how many the cluster removed.

        The narrow case: a chunk known to be individually wrong, and the tail trim
        after a re-chunk. Whole-Signal removal goes through `delete_signal()`, which
        does not depend on the caller knowing the chunk count.

        A 404 for an id that is not there is *not* an error: the caller asked for the
        document to be gone, and it is. Only that status is tolerated -- anything else
        is a genuine rejection and is raised.
        """
        actions: list[dict[str, Any]] = []
        wanted: list[str] = []
        for chunk_id in dict.fromkeys(chunk_ids):
            # Validates the shape and refuses an id this indexer could not have
            # written; a malformed id here would delete nothing while reporting a
            # successful erasure.
            split_chunk_id(chunk_id)
            actions.append({"delete": {"_index": self._spec.name, "_id": chunk_id}})
            wanted.append(chunk_id)
        if not wanted:
            return 0

        response = await self._call_bulk(actions)
        items = _items(response)
        if len(items) != len(wanted):
            # The API returns one item per action. A mismatch means the ids below are
            # matched to the wrong statuses, so the count returned and any error
            # raised would both name the wrong chunk.
            _log.warning(
                "opensearch.bulk.item_count_mismatch",
                index=self._spec.name,
                submitted=len(wanted),
                items=len(items),
            )

        deleted = 0
        failures: list[BulkItemFailure] = []
        for position, item in enumerate(items):
            fallback = wanted[position] if position < len(wanted) else "?"
            chunk_id = _item_id(item) or fallback
            status, error = _item_status(item)
            if status < _OK_STATUS:
                deleted += 1
            elif status == 404:
                # Not an error: the caller asked for the document to be gone, and it
                # is. An erasure re-run must not fail on the chunks it already removed.
                continue
            else:
                failures.append(_failure(chunk_id, status, error))
        if failures:
            raise BulkIndexError(failures, index=self._spec.name)
        return deleted

    # ------------------------------------------------------------ internals --

    def _index_action(self, document: ChunkDocument) -> tuple[dict[str, Any], dict[str, Any]]:
        """The action/source pair for one document.

        `_id` is the `chunk_id` (`docs/data-stores.md` §5.2): the idempotency key, and
        the key hybrid fusion joins the OpenSearch and Qdrant candidate lists on.
        """
        return (
            {
                "index": {
                    "_index": self._spec.name,
                    "_id": document.chunk_id,
                    "version": document.pipeline_version,
                    "version_type": self._version_type.value,
                }
            },
            document.to_document(),
        )

    def _assert_versioned(self, document: ChunkDocument) -> None:
        """Refuse a document that carries no usable version.

        `pipeline_version` here is the ordinal from `models.lineage`, and
        `pipeline_version_ordinal()` returns **0** for a version string it cannot
        parse. Writing that 0 as the external version would mean "older than every
        other write": under `EXTERNAL_GTE` two such writes would race with no ordering
        between them, and under `EXTERNAL` the second would be rejected as stale
        against a document that is not newer in any meaningful sense. Either way the
        guard that this whole write path exists for would be silently disarmed, so the
        write is refused where the offending chunk can still be named.
        """
        if document.pipeline_version < 1:
            raise ValueError(
                f"chunk {document.chunk_id!r} has pipeline_version "
                f"{document.pipeline_version}, which cannot be used as an external "
                "document version. 0 is what pipeline_version_ordinal() returns for "
                "an unparseable version string; writing it would disarm the guard "
                "that stops a stale backfill overwriting newer enrichment."
            )

    async def _flush(
        self, actions: Sequence[Mapping[str, Any]], chunk_ids: Sequence[str]
    ) -> tuple[int, list[str]]:
        response = await self._call_bulk(actions)
        return self._read_bulk_response(response, chunk_ids)

    async def _call_bulk(self, actions: Sequence[Mapping[str, Any]]) -> Any:
        """Send one bulk request.

        `refresh` is passed only when it was asked for: sending `refresh=false`
        explicitly is the cluster default anyway, and an absent parameter keeps the
        request line honest about what this indexer actually requests.
        """
        kwargs: dict[str, Any] = {"body": list(actions)}
        if self._refresh:
            kwargs["refresh"] = self._refresh
        return await self._client.bulk(**kwargs)

    def _read_bulk_response(
        self, response: Any, chunk_ids: Sequence[str]
    ) -> tuple[int, list[str]]:
        """Split a bulk response into successes, version conflicts and real failures.

        The whole point of the module docstring's first failure mode. A bulk request
        that rejected every document still returns HTTP 200 with `errors: true`, so
        this is the only place the strictness of `dynamic: "strict"` becomes visible to
        a caller.

        Items are matched to chunk ids by *position*: the response preserves request
        order, and the item body carries `_id` anyway, so the id is read from the item
        when present and falls back to the positional one when the cluster elided it.
        """
        items = _items(response)
        if not items:
            if isinstance(response, Mapping) and response.get("errors"):
                # `errors: true` with no items is a malformed response, and reporting
                # the batch as written would lose every document in it.
                raise BulkIndexError(
                    [
                        BulkItemFailure(
                            chunk_id=chunk_id,
                            status=0,
                            type="malformed_bulk_response",
                            reason="response reported errors but carried no items",
                        )
                        for chunk_id in chunk_ids
                    ],
                    index=self._spec.name,
                )
            # An empty, error-free response for an empty batch. Nothing to count.
            return 0, []

        indexed = 0
        conflicted: list[str] = []
        failures: list[BulkItemFailure] = []
        for position, item in enumerate(items):
            fallback = chunk_ids[position] if position < len(chunk_ids) else "?"
            chunk_id = _item_id(item) or fallback
            status, error = _item_status(item)
            if status < _OK_STATUS:
                indexed += 1
            elif status == _CONFLICT_STATUS and _error_type(error) == _VERSION_CONFLICT:
                # The guard working. The stored document is newer than the one being
                # written, which is precisely what a backfill racing a live update
                # looks like, so this is counted rather than raised.
                conflicted.append(chunk_id)
            else:
                failures.append(_failure(chunk_id, status, error))

        if failures:
            _log.error(
                "opensearch.bulk.rejected",
                index=self._spec.name,
                rejected=len(failures),
                submitted=len(chunk_ids),
                first=str(failures[0]),
            )
            raise BulkIndexError(failures, index=self._spec.name)

        if conflicted:
            _log.info(
                "opensearch.bulk.version_conflicts",
                index=self._spec.name,
                conflicts=len(conflicted),
                submitted=len(chunk_ids),
                detail="older pipeline_version rejected; newer enrichment kept",
            )
        return indexed, conflicted


# --------------------------------------------------------------------------- #
# Response shapes
# --------------------------------------------------------------------------- #


def _filter_count(body: Mapping[str, Any]) -> int:
    """How many filter clauses the body actually carries, for the query log.

    Unwraps the optional `function_score` first. Reading `query.bool` directly would
    report zero filters for every recency-decayed query -- and "the filter count in the
    log is zero" is exactly the observation someone would use to conclude that filters
    are not being pushed down, on the one code path where they demonstrably are.
    """
    query = body.get("query", {})
    if isinstance(query, Mapping) and "function_score" in query:
        inner = query["function_score"]
        query = inner.get("query", {}) if isinstance(inner, Mapping) else {}
    inner_bool = query.get("bool") if isinstance(query, Mapping) else None
    if not isinstance(inner_bool, Mapping):
        return 0
    return len(inner_bool.get("filter", ()))


def _items(response: Any) -> Sequence[Mapping[str, Any]]:
    """The `items` array of a bulk response, tolerating a client that returns None."""
    if not isinstance(response, Mapping):
        return ()
    items = response.get("items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return ()
    return [item for item in items if isinstance(item, Mapping)]


def _operation(item: Mapping[str, Any]) -> Mapping[str, Any]:
    """The single `{"index": {...}}` / `{"delete": {...}}` body inside a bulk item.

    Read by taking the one value rather than by looking up `"index"`, because the key
    is the *operation* -- `index`, `create`, `delete`, `update` -- and hardcoding one
    of them would make a delete response parse as an empty success.
    """
    for value in item.values():
        if isinstance(value, Mapping):
            return value
    return {}


def _item_id(item: Mapping[str, Any]) -> str | None:
    value = _operation(item).get("_id")
    return value if isinstance(value, str) and value else None


def _item_status(item: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
    """`(status, error)` for one item.

    A missing status defaults to 500 rather than 200. The default is reached only for
    a response shape that is not documented, and treating an unreadable item as
    written would report a successful index for a document that may not exist.
    """
    operation = _operation(item)
    raw = operation.get("status")
    status = raw if isinstance(raw, int) and not isinstance(raw, bool) else 500
    error = operation.get("error")
    return status, error if isinstance(error, Mapping) else {}


def _error_type(error: Mapping[str, Any]) -> str:
    value = error.get("type")
    return value if isinstance(value, str) else ""


def _failure(chunk_id: str, status: int, error: Mapping[str, Any]) -> BulkItemFailure:
    reason = error.get("reason")
    return BulkItemFailure(
        chunk_id=chunk_id,
        status=status,
        type=_error_type(error) or "unknown",
        reason=reason if isinstance(reason, str) else "no reason reported",
    )
