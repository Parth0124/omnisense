"""OpenSearch client bootstrap and the signal index definition.

OpenSearch is the BM25 half of hybrid retrieval (`docs/retrieval.md` §4). It is a
**derived** store: everything in it can be rebuilt from PostgreSQL by
`scripts/reindex.py`, which is what licenses the two decisions encoded here.

**One document per chunk, not per Signal.** The index is addressed by
`chunk_id` (`{signal_id}:{chunk_index}`) -- exactly the granularity of the Qdrant
points (`docs/data-stores.md` §3.5, `docs/retrieval.md` §4). Hybrid fusion joins
the keyword and vector candidate lists on that key, so indexing one document per
Signal would leave the two backends with nothing to fuse on and would break
citation, whose unit is the chunk span `[char_start, char_end)`.

**The mapping is explicit and `dynamic: "strict"`.** Dynamic mapping would let a
single indexer that leaks an unexpected field permanently add it to the index
mapping -- a change that cannot be undone in place and that in the worst case
("this string looks like a date") makes subsequent well-formed documents
unindexable. Strict turns that into a loud, immediate rejection of the one bad
document instead of a silent, irreversible schema change. The one escape hatch is
`metadata`, deliberately `enabled: false`; see the comment on it.

Layer note: **L1k kernel** (`docs/architecture.md` §6.1) -- importable by
`services/`, `agents/`, `workers/`, `backend/api/` and `scripts/`, never by
`connectors/`.

Ownership note: `docs/data-stores.md` §3.5 places index *templates and ISM
lifecycle policies* in `retrieval/keyword/index.py`. This module owns only what
bootstrap needs -- the connection singleton and an idempotent create of the one
index the system cannot start indexing without. The constants below are exported
so `retrieval/keyword/index.py` can build on them rather than restate them.
"""

from __future__ import annotations

from typing import Any

from opensearchpy import AsyncHttpConnection, AsyncOpenSearch
from opensearchpy.exceptions import RequestError

from backend.core.config import get_settings

__all__ = [
    "SIGNAL_INDEX_ANALYSIS",
    "SIGNAL_INDEX_MAPPINGS",
    "check_opensearch",
    "dispose_opensearch",
    "ensure_index",
    "get_opensearch",
]

_client: AsyncOpenSearch | None = None

_HEALTH_TIMEOUT_SECONDS = 5
"""Client-side budget for the readiness probe.

Deliberately far below `OPENSEARCH_TIMEOUT_SECONDS` (30s, and a sane ceiling for
a real search). A readiness probe is called on a fixed short interval by the
orchestrator; letting it inherit the search timeout means an unreachable cluster
turns `/readyz` into a 30-second request that the probe has already given up on.
"""


# --------------------------------------------------------------------------- #
# Index definition
# --------------------------------------------------------------------------- #

_EXACT_ANALYZER = "os_exact"

SIGNAL_INDEX_ANALYSIS: dict[str, Any] = {
    "analyzer": {
        # `docs/retrieval.md` §4 asks for a "keyword-ish" analyzer behind
        # `text.exact` for phrase boosts. Keyword-ish means: tokenize and fold
        # case/accents so "OpenSearch" matches "opensearch", but do **not** stem
        # and do **not** drop stopwords. Stemming is what makes the standard
        # analyzer useless for the queries this field exists for -- product
        # names, error strings, exact phrases -- because a stemmer collapses
        # distinct product names onto one root and a stopword filter deletes the
        # "by" from "connection reset by peer".
        _EXACT_ANALYZER: {
            "type": "custom",
            "tokenizer": "standard",
            "filter": ["lowercase", "asciifolding"],
        }
    }
}

SIGNAL_INDEX_MAPPINGS: dict[str, Any] = {
    # See the module docstring: an unexpected field must be a rejected document,
    # not a mutated mapping.
    "dynamic": "strict",
    "properties": {
        # --- identity -----------------------------------------------------
        # `chunk_id` is also the document `_id` (`docs/data-stores.md` §5.2), so
        # it is duplicated into a field only because `_id` cannot be aggregated
        # or filtered on as cheaply as a real keyword field.
        "chunk_id": {"type": "keyword"},
        "signal_id": {"type": "keyword"},
        "chunk_index": {"type": "integer"},
        # Mandatory filter from Phase 7 onward (`docs/retrieval.md` §4). Present
        # from day one because retrofitting a tenant discriminator means a full
        # reindex, and this index is the large one.
        "tenant_id": {"type": "keyword"},
        # --- searchable body ----------------------------------------------
        # The analyzer here is language-neutral on purpose. `docs/retrieval.md`
        # §4 wants the analyzer selected per `language`, which needs one
        # sub-field per supported language and a decision about which languages
        # are supported -- that belongs to `retrieval/keyword/index.py`, not to
        # bootstrap. `language` is indexed below so routing can be added later
        # without re-deriving anything.
        "title": {"type": "text", "analyzer": "standard"},
        "text": {
            "type": "text",
            "analyzer": "standard",
            "fields": {"exact": {"type": "text", "analyzer": _EXACT_ANALYZER}},
        },
        # The citation span, as offsets into the *source Signal content* rather
        # than into `text`. `services/evidence_service.py` re-reads this range to
        # verify that a quoted passage really appears in the original
        # (`docs/retrieval.md` §9), which only works against source offsets.
        "char_start": {"type": "integer"},
        "char_end": {"type": "integer"},
        # --- filterable metadata (mirrors the Qdrant payload) --------------
        "source": {"type": "keyword"},
        "platform": {"type": "keyword"},
        # Not analyzed: a URL is a filter and a display value, never a query
        # target (`docs/data-stores.md` §4). `ignore_above` makes a pathological
        # tracking URL an unindexed field rather than a rejected document.
        "url": {"type": "keyword", "ignore_above": 2048},
        "author": {
            "dynamic": "strict",
            "properties": {"handle": {"type": "keyword"}},
        },
        "published_at": {"type": "date"},
        "language": {"type": "keyword"},
        # Extracted terms, matched literally in a `terms` boost clause rather
        # than analyzed -- they have already been through extraction, so
        # re-tokenizing them would only reintroduce the noise that step removed.
        "keywords": {"type": "keyword"},
        "topics": {"type": "keyword"},
        "entity_ids": {"type": "keyword"},
        # `docs/data-stores.md` §4 lists sentiment, engagement and confidence as
        # numeric fields here. They exist for range filters and score functions;
        # the authoritative values stay in PostgreSQL.
        "sentiment_polarity": {"type": "float"},
        "engagement_score": {"type": "float"},
        "confidence": {"type": "float"},
        # --- reconciliation ------------------------------------------------
        # The same value used as the external document version on write
        # (`version_type=external`, `docs/data-stores.md` §5.2). Stored so that
        # "which store is behind" is answerable by querying this index rather
        # than by diffing it against PostgreSQL row by row.
        "pipeline_version": {"type": "integer"},
        "indexed_at": {"type": "date"},
        # The one unstructured field, and the reason `dynamic: strict` is
        # survivable at all. `metadata` is per-platform and open-ended by design
        # (`models/signal.py`), so mapping it would defeat the point of a strict
        # mapping. `enabled: false` keeps it in `_source` -- readable,
        # returnable, never parsed and never mapped -- so an unexpected key
        # inside it can neither explode the mapping nor reject the document.
        "metadata": {"type": "object", "enabled": False},
    },
}


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #


def get_opensearch() -> AsyncOpenSearch:
    """Return the process-wide OpenSearch client, creating it on first use.

    Constructing the client opens no socket -- `opensearchpy` connects lazily on
    the first request -- so importing this module stays free, which is what keeps
    the unit suite runnable without a cluster.
    """
    global _client
    if _client is None:
        settings = get_settings()
        opensearch = settings.opensearch

        options: dict[str, Any] = {
            # Explicit rather than relying on the transport default: this client
            # is awaited from the event loop, and a synchronous connection class
            # would block it while looking every bit like a slow cluster.
            "connection_class": AsyncHttpConnection,
            "timeout": opensearch.timeout_seconds,
            "verify_certs": opensearch.verify_certs,
            # When an operator has deliberately turned certificate verification
            # off (a self-signed cluster in dev), the accompanying urllib3
            # warning is noise -- and `filterwarnings = ["error"]` in
            # pyproject.toml would promote that noise to a test failure.
            "ssl_show_warn": opensearch.verify_certs,
            # One retry, on a different node where there is one. Aggressively
            # retrying a timed-out search is the wrong trade -- the cluster is
            # already struggling -- so this is deliberately not a retry policy.
            "max_retries": 1,
            "retry_on_timeout": False,
        }

        # `docs/data-stores.md` §3.5: the local single-node cluster runs with the
        # security plugin disabled, so `OPENSEARCH_USER`/`OPENSEARCH_PASSWORD`
        # are empty in `.env.example`. Sending an empty-string basic auth header
        # to such a cluster is not harmless -- it is a malformed credential --
        # so auth is attached only when both halves are actually present.
        if opensearch.user and opensearch.password is not None:
            options["http_auth"] = (
                opensearch.user,
                opensearch.password.get_secret_value(),
            )

        _client = AsyncOpenSearch(hosts=opensearch.url, **options)
    return _client


async def check_opensearch() -> bool:
    """Probe OpenSearch for `/readyz`. Never raises.

    Cluster health rather than `ping()`, because a cluster that answers `HEAD /`
    while its primary shards are unassigned is not usable and must not be
    reported ready. **Yellow counts as healthy**: `docs/data-stores.md` §3.5
    documents the local cluster as single-node with zero replicas, which is
    permanently yellow, so treating yellow as failure would mean the local stack
    never reports ready.

    Returns a bool rather than raising because readiness aggregates several
    dependencies and one being down must not prevent reporting on the others
    (`docs/observability.md`).
    """
    try:
        health = await get_opensearch().cluster.health(request_timeout=_HEALTH_TIMEOUT_SECONDS)
    except Exception:
        return False
    return str(health.get("status", "")).lower() in ("green", "yellow")


async def ensure_index(
    index: str | None = None,
    *,
    number_of_shards: int = 1,
    number_of_replicas: int = 1,
) -> bool:
    """Create the signal index if it is absent. Returns whether it created it.

    Idempotent in the way that matters under concurrency: several workers start
    at once, all see the index missing, and all issue a create. Exactly one wins;
    the losers get `resource_already_exists_exception`, which is swallowed here
    because "someone else created it a millisecond ago" is the success case, not
    an error.

    An **existing** index is left exactly as it is. That is deliberate rather
    than lazy: OpenSearch cannot change the type of a mapped field in place, and
    `docs/data-stores.md` §3.5 prescribes reindex-into-a-new-index plus an alias
    swap for mapping changes. Attempting a `put_mapping` here would half-apply a
    schema change -- the additive parts succeeding, the incompatible parts
    failing -- and leave the index in a state nobody designed.

    Args:
        index: Index name. Defaults to `OPENSEARCH_SIGNAL_INDEX`.
        number_of_shards: Fixed at creation; changing it later requires a
            reindex.
        number_of_replicas: Defaults to 1 because in a deployed cluster a lost
            shard is a lost index. The local single-node cluster cannot allocate
            the replica and sits yellow, which `check_opensearch()` accepts by
            design; pass 0 to keep a single-node cluster green.
    """
    name = index or get_settings().opensearch.signal_index
    client = get_opensearch()

    body: dict[str, Any] = {
        "settings": {
            "index": {
                "number_of_shards": number_of_shards,
                "number_of_replicas": number_of_replicas,
            },
            "analysis": SIGNAL_INDEX_ANALYSIS,
        },
        "mappings": SIGNAL_INDEX_MAPPINGS,
    }

    # The existence check is an optimization, not the correctness guarantee --
    # the exception handler below is that. Checking first only avoids a 400 in
    # the cluster log on every process start.
    if await client.indices.exists(index=name):
        return False

    try:
        await client.indices.create(index=name, body=body)
    except RequestError as exc:
        if exc.error != "resource_already_exists_exception":
            raise
        return False
    return True


async def dispose_opensearch() -> None:
    """Close the connection pool and reset the singleton.

    Called from lifespan shutdown. Skipping it leaves `aiohttp` sessions open,
    which surfaces as "Unclosed client session" on stderr during interpreter
    teardown -- after logging has been torn down, so it never appears in the logs
    where anyone would look for it.
    """
    global _client
    if _client is not None:
        await _client.close()
    _client = None
