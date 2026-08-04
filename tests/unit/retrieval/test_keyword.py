"""Unit tests for `retrieval/keyword/` -- the BM25 side of hybrid retrieval.

Everything here runs against a fake cluster. `docs/testing-strategy.md` fixes the unit
suite as "no external services", and in any case the properties worth pinning are
properties of the *request we send* and of *how we read the reply*, not of Lucene's
scoring.

The fake is deliberately not a stub returning canned hits. It does the two things
OpenSearch does that this code depends on:

**It applies the `filter` clauses before truncating to `size`.** That ordering is the
whole reason filters are pushed down. A fake that returned its corpus unfiltered would
let every assertion in `TestFilterPushdown` pass against a backend that sent no filter
at all and trimmed the results afterwards -- which is precisely the bug those tests
exist to catch, because post-filtering a fixed-size result set silently converts
`k=100` into "however many of the top 100 survived".

**It enforces external document versioning.** A stale backfill overwriting newer
enrichment is a successful-looking 200 in every store that does not check, so the fake
stores a version per `_id` and rejects an older write with the same
`version_conflict_engine_exception` a real cluster returns -- inside a 200 response
body, where an unwary caller will not look.

It also refuses any clause it does not understand rather than ignoring it. A fake that
silently skipped an unrecognised filter clause would report success for a filter that
was never applied.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from opensearchpy.exceptions import RequestError

from backend.core.exceptions import ConfigurationError
from models.enums import Platform, SourceCategory
from models.lineage import pipeline_version_ordinal
from models.signal import Keyword
from retrieval.hybrid import SearchBackend
from retrieval.keyword.index import (
    EXACT_SUBFIELD,
    LANGUAGE_ANALYZERS,
    TEXT_FIELD_PREFIX,
    ChunkDocument,
    ChunkField,
    IndexSpec,
    chunk_index_mappings,
    ensure_chunk_index,
    inspect_chunk_index,
    language_text_field,
    mapping_drift,
    primary_subtag,
    swap_alias,
)
from retrieval.keyword.opensearch_client import (
    BulkIndexError,
    IndexOutcome,
    KeywordBackend,
    KeywordIndexer,
    KeywordStore,
    VersionType,
)
from retrieval.keyword.query_builder import (
    MAX_RESULT_WINDOW,
    QueryOptions,
    build_filter_clauses,
    build_search_body,
    candidates_from_response,
)
from retrieval.types import Backend, Filter, RetrievalRequest, chunk_id_for

pytestmark = pytest.mark.unit

INDEX = "test-chunks"
SPEC = IndexSpec(name=INDEX, number_of_shards=1, number_of_replicas=0)

JAN = datetime(2026, 1, 15, tzinfo=UTC)
FEB = datetime(2026, 2, 15, tzinfo=UTC)
MAR = datetime(2026, 3, 15, tzinfo=UTC)

TEXT_EXACT = f"{ChunkField.TEXT.value}.{EXACT_SUBFIELD}"
"""The phrase-boost field, spelled once so a rename cannot half-land in these tests."""

V1_9 = pipeline_version_ordinal("1.9.0")
V1_10 = pipeline_version_ordinal("1.10.0")


# --------------------------------------------------------------------------- #
# A fake that behaves like OpenSearch where it matters
# --------------------------------------------------------------------------- #


class FakeQueryError(AssertionError):
    """The fake was handed a clause it does not implement.

    An `AssertionError` rather than a returned empty list: a fake that quietly ignored
    an unknown clause would report a passing test for a query whose filter was never
    evaluated, which is the exact class of bug this suite is for.
    """


def _analyze(value: Any) -> list[str]:
    """Standard-analyzer-ish tokenisation: lowercase, split on non-alphanumerics.

    A list rather than a set, because the scorer below counts term frequency. Scoring
    on set overlap instead would make "revenue revenue revenue" and "revenue" score
    identically, and a tie is resolved by the sort's second key -- which quietly turns
    every ranking assertion in this file into an assertion about document *ids*.
    """
    out: list[str] = []
    if not isinstance(value, str):
        return out
    current = ""
    for char in value.lower():
        if char.isalnum():
            current += char
        elif current:
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


def _as_instant(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _matches_filter(clause: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    """Evaluate one `filter` clause. Restrictive, non-scoring, like the real thing."""
    if "term" in clause:
        field, expected = next(iter(clause["term"].items()))
        return source.get(field) == expected
    if "terms" in clause:
        field, wanted = next(iter(clause["terms"].items()))
        held = source.get(field)
        if isinstance(held, list):
            # A `terms` clause against a list field is an intersection test: the
            # `any_of` semantics of `docs/retrieval.md` §7.
            return any(value in wanted for value in held)
        return held in wanted
    if "range" in clause:
        field, bounds = next(iter(clause["range"].items()))
        held = source.get(field)
        instant = _as_instant(held)
        if instant is not None:
            for op, bound in bounds.items():
                edge = _as_instant(bound)
                if edge is None:
                    raise FakeQueryError(f"unparseable range bound {bound!r}")
                if op == "gte" and instant < edge:
                    return False
                if op == "lt" and instant >= edge:
                    return False
                if op not in ("gte", "lt"):
                    raise FakeQueryError(f"unsupported range operator {op!r}")
            return True
        if not isinstance(held, (int, float)):
            return False
        for op, bound in bounds.items():
            if op == "gte" and held < bound:
                return False
            if op == "lt" and held >= bound:
                return False
            if op not in ("gte", "lt"):
                raise FakeQueryError(f"unsupported range operator {op!r}")
        return True
    raise FakeQueryError(f"fake does not implement filter clause {sorted(clause)}")


def _score_clause(clause: Mapping[str, Any], source: Mapping[str, Any]) -> float:
    """Score one scoring clause. 0.0 means "did not match"."""
    if "match_all" in clause:
        return 1.0
    if "match" in clause:
        field, spec = next(iter(clause["match"].items()))
        wanted = set(_analyze(spec["query"]))
        # Term frequency, not set overlap: a document that says "revenue" three times
        # must outrank one that says it once, or the fake cannot distinguish a strong
        # match from a weak one and every ranking test degenerates into a tie-break.
        frequency = sum(1 for token in _analyze(source.get(field)) if token in wanted)
        return frequency * float(spec.get("boost", 1.0))
    if "match_phrase" in clause:
        field, spec = next(iter(clause["match_phrase"].items()))
        # `text.exact` and `title.exact` hold the same value as their parent; the
        # sub-field differs only in analysis, which a substring test stands in for.
        parent = field.split(".")[0]
        held = source.get(parent)
        if not isinstance(held, str):
            return 0.0
        return float(spec.get("boost", 1.0)) if spec["query"].lower() in held.lower() else 0.0
    if "term" in clause:
        field, spec = next(iter(clause["term"].items()))
        held = source.get(field)
        values = held if isinstance(held, list) else [held]
        return float(spec.get("boost", 1.0)) if spec["value"] in values else 0.0
    raise FakeQueryError(f"fake does not implement scoring clause {sorted(clause)}")


def _gauss(source: Mapping[str, Any], spec: Mapping[str, Any], now: datetime) -> float:
    """The `gauss` decay, computed the way Lucene computes it.

    A document with no `published_at` scores 1.0 -- unknown recency is not old --
    which is the behaviour `query_builder._with_recency_decay` documents and which
    would otherwise be untested.
    """
    published = _as_instant(source.get(ChunkField.PUBLISHED_AT.value))
    if published is None:
        return 1.0
    scale_days = float(str(spec["scale"]).rstrip("d"))
    decay = float(spec["decay"])
    age_days = abs((now - published).total_seconds()) / 86_400
    sigma_squared = -(scale_days**2) / (2 * math.log(decay))
    return math.exp(-(age_days**2) / (2 * sigma_squared))


class FakeOpenSearch:
    """The three `KeywordStore` methods plus an `indices` namespace, all recording.

    Holds one corpus that both the search path and the write path see, so a test can
    index documents and then search for them without a second source of truth.
    """

    def __init__(self, *, now: datetime | None = None) -> None:
        self.sources: dict[str, dict[str, Any]] = {}
        self.versions: dict[str, int] = {}
        self.searches: list[dict[str, Any]] = []
        self.bulk_bodies: list[list[dict[str, Any]]] = []
        self.bulk_params: list[dict[str, Any]] = []
        self.delete_by_query_calls: list[dict[str, Any]] = []
        self.indices = FakeIndices()
        self.now = now or datetime(2026, 3, 15, tzinfo=UTC)

        self.search_error: Exception | None = None
        self.canned_response: dict[str, Any] | None = None
        self.reject: dict[str, tuple[int, dict[str, Any]]] = {}
        """`chunk_id -> (status, error)` for simulating a rejected bulk item."""

        self.shards: dict[str, Any] = {"total": 1, "successful": 1, "failed": 0}
        self.timed_out = False

    # -- reads ---------------------------------------------------------------

    def seed(self, chunk_id: str, source: Mapping[str, Any]) -> None:
        self.sources[chunk_id] = dict(source)

    async def search(self, **kwargs: Any) -> dict[str, Any]:
        self.searches.append(kwargs)
        if self.search_error is not None:
            raise self.search_error
        if self.canned_response is not None:
            return self.canned_response

        body = kwargs["body"]
        query = body["query"]
        decay: Mapping[str, Any] | None = None
        if "function_score" in query:
            function_score = query["function_score"]
            decay = function_score["functions"][0]["gauss"][ChunkField.PUBLISHED_AT.value]
            query = function_score["query"]

        clauses = query["bool"]
        scored: list[tuple[float, str]] = []
        for chunk_id, source in self.sources.items():
            # Filters first, and only then `size`. Reversing these two lines is the
            # bug the pushdown tests are written against.
            if not all(_matches_filter(c, source) for c in clauses.get("filter", ())):
                continue
            score = self._score(clauses, source)
            if score is None:
                continue
            if decay is not None:
                score *= _gauss(source, decay, self.now)
            scored.append((score, chunk_id))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        hits = [
            {"_id": chunk_id, "_score": score} for score, chunk_id in scored[: body["size"]]
        ]
        return {
            "took": 1,
            "timed_out": self.timed_out,
            "_shards": dict(self.shards),
            "hits": {"hits": hits},
        }

    def _score(self, clauses: Mapping[str, Any], source: Mapping[str, Any]) -> float | None:
        """Total score, or None when the document does not match at all."""
        total = 0.0
        for must in clauses.get("must", ()):
            inner = must.get("bool")
            if inner is None:
                total += _score_clause(must, source)
                continue
            hits = [_score_clause(c, source) for c in inner.get("should", ())]
            matched = [s for s in hits if s > 0]
            if len(matched) < inner.get("minimum_should_match", 0):
                return None
            total += sum(matched)
        # Outer `should` clauses boost, and can never make a non-matching document
        # match -- the property `query_builder` separates recall from boosting for.
        total += sum(_score_clause(c, source) for c in clauses.get("should", ()))
        return total

    # -- writes --------------------------------------------------------------

    async def bulk(self, **kwargs: Any) -> dict[str, Any]:
        body = list(kwargs["body"])
        self.bulk_bodies.append(body)
        self.bulk_params.append({k: v for k, v in kwargs.items() if k != "body"})

        items: list[dict[str, Any]] = []
        errors = False
        position = 0
        while position < len(body):
            action = body[position]
            operation, meta = next(iter(action.items()))
            chunk_id = meta["_id"]
            if operation == "delete":
                position += 1
                items.append(self._apply_delete(chunk_id))
            else:
                source = body[position + 1]
                position += 2
                items.append(self._apply_index(chunk_id, meta, source))
            errors = errors or items[-1][operation if operation == "delete" else "index"][
                "status"
            ] >= 300
        return {"took": 1, "errors": errors, "items": items}

    def _apply_index(
        self, chunk_id: str, meta: Mapping[str, Any], source: Mapping[str, Any]
    ) -> dict[str, Any]:
        if chunk_id in self.reject:
            status, error = self.reject[chunk_id]
            return {"index": {"_index": INDEX, "_id": chunk_id, "status": status, "error": error}}

        version = meta["version"]
        version_type = meta["version_type"]
        stored = self.versions.get(chunk_id)
        if stored is not None:
            newer = version > stored if version_type == "external" else version >= stored
            if not newer:
                return {
                    "index": {
                        "_index": INDEX,
                        "_id": chunk_id,
                        "status": 409,
                        "error": {
                            "type": "version_conflict_engine_exception",
                            "reason": (
                                f"[{chunk_id}]: version conflict, current version "
                                f"[{stored}] is higher or equal to the one provided "
                                f"[{version}]"
                            ),
                        },
                    }
                }
        self.sources[chunk_id] = dict(source)
        self.versions[chunk_id] = version
        return {
            "index": {
                "_index": INDEX,
                "_id": chunk_id,
                "status": 201 if stored is None else 200,
                "_version": version,
            }
        }

    def _apply_delete(self, chunk_id: str) -> dict[str, Any]:
        if chunk_id in self.reject:
            status, error = self.reject[chunk_id]
            return {"delete": {"_index": INDEX, "_id": chunk_id, "status": status, "error": error}}
        existed = self.sources.pop(chunk_id, None) is not None
        self.versions.pop(chunk_id, None)
        return {
            "delete": {
                "_index": INDEX,
                "_id": chunk_id,
                "status": 200 if existed else 404,
                "result": "deleted" if existed else "not_found",
            }
        }

    async def delete_by_query(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_by_query_calls.append(kwargs)
        clause = kwargs["body"]["query"]
        doomed = [
            chunk_id
            for chunk_id, source in self.sources.items()
            if _matches_filter(clause, source)
        ]
        for chunk_id in doomed:
            self.sources.pop(chunk_id, None)
            self.versions.pop(chunk_id, None)
        return {"deleted": len(doomed), "version_conflicts": 0, "failures": []}


class FakeIndices:
    """The `client.indices` namespace used by the lifecycle functions."""

    def __init__(self) -> None:
        self.mappings: dict[str, dict[str, Any]] = {}
        self.created: list[dict[str, Any]] = []
        self.aliases: dict[str, list[str]] = {}
        self.alias_actions: list[list[dict[str, Any]]] = []
        self.create_error: Exception | None = None

    async def exists(self, *, index: str) -> bool:
        return index in self.mappings

    async def create(self, *, index: str, body: Mapping[str, Any]) -> dict[str, Any]:
        self.created.append({"index": index, "body": body})
        if self.create_error is not None:
            raise self.create_error
        self.mappings[index] = dict(body["mappings"])
        return {"acknowledged": True}

    async def get_mapping(self, *, index: str) -> dict[str, Any]:
        names = self.aliases.get(index) or ([index] if index in self.mappings else [])
        return {name: {"mappings": self.mappings[name]} for name in names}

    async def get_alias(self, *, name: str) -> dict[str, Any]:
        if name not in self.aliases:
            raise KeyError(name)
        return {index: {"aliases": {name: {}}} for index in self.aliases[name]}

    async def update_aliases(self, *, body: Mapping[str, Any]) -> dict[str, Any]:
        actions = list(body["actions"])
        self.alias_actions.append(actions)
        for action in actions:
            if "remove" in action:
                alias = action["remove"]["alias"]
                self.aliases[alias] = [
                    i for i in self.aliases.get(alias, []) if i != action["remove"]["index"]
                ]
            else:
                alias = action["add"]["alias"]
                self.aliases.setdefault(alias, []).append(action["add"]["index"])
        return {"acknowledged": True}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Doc:
    """Shorthand for seeding the fake corpus."""

    chunk_id: str
    text: str = "quarterly revenue grew"
    title: str | None = None
    published_at: datetime | None = FEB
    platform: Platform = Platform.RSS
    source: SourceCategory = SourceCategory.NEWS
    language: str = "en"
    entity_ids: Sequence[str] = ()
    keywords: Sequence[str] = ()
    confidence: float = 0.8
    tenant_id: str = "default"

    def as_source(self) -> dict[str, Any]:
        return {
            ChunkField.CHUNK_ID.value: self.chunk_id,
            ChunkField.TEXT.value: self.text,
            ChunkField.TITLE.value: self.title,
            ChunkField.PUBLISHED_AT.value: (
                self.published_at.isoformat() if self.published_at else None
            ),
            ChunkField.PLATFORM.value: str(self.platform),
            ChunkField.SOURCE.value: str(self.source),
            ChunkField.LANGUAGE.value: self.language,
            ChunkField.ENTITY_IDS.value: list(self.entity_ids),
            ChunkField.KEYWORDS.value: list(self.keywords),
            ChunkField.CONFIDENCE.value: self.confidence,
            ChunkField.TENANT_ID.value: self.tenant_id,
        }


def seed(client: FakeOpenSearch, *docs: Doc) -> None:
    for doc in docs:
        client.seed(doc.chunk_id, doc.as_source())


def a_document(**overrides: Any) -> ChunkDocument:
    """A valid `ChunkDocument`, version-stamped so the indexer will accept it."""
    fields: dict[str, Any] = {
        "signal_id": "sig1",
        "chunk_index": 0,
        "text": "quarterly revenue grew",
        "char_start": 0,
        "char_end": 22,
        "published_at": FEB,
        "pipeline_version": V1_9,
    }
    fields.update(overrides)
    return ChunkDocument(**fields)


def bodies_of(client: FakeOpenSearch) -> list[dict[str, Any]]:
    return [call["body"] for call in client.searches]


def inner_bool(body: Mapping[str, Any]) -> dict[str, Any]:
    """The `bool` carrying the filters, unwrapping an optional `function_score`."""
    query = body["query"]
    if "function_score" in query:
        query = query["function_score"]["query"]
    return query["bool"]


def scoring_clauses(body: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The inner `should` clauses -- the ones that decide whether a document matches."""
    must = inner_bool(body)["must"][0]
    return list(must.get("bool", {}).get("should", []))


# --------------------------------------------------------------------------- #
# index.py -- the mapping
# --------------------------------------------------------------------------- #


class TestChunkMapping:
    def test_mapping_is_strict(self) -> None:
        """An unexpected field must be a rejected document, not a mutated mapping.

        The alternative is irreversible: a dynamically added field cannot be removed
        in place, and one that guessed `date` from a string makes every subsequent
        well-formed document unindexable.
        """
        assert chunk_index_mappings()["dynamic"] == "strict"

    def test_author_object_is_strict_too(self) -> None:
        """Nested objects inherit nothing: `dynamic` is per-object."""
        properties = chunk_index_mappings()["properties"]
        assert properties[ChunkField.AUTHOR.value]["dynamic"] == "strict"

    def test_required_fields_are_mapped(self) -> None:
        properties = chunk_index_mappings()["properties"]
        for spec_field in (
            ChunkField.CHUNK_ID,
            ChunkField.SIGNAL_ID,
            ChunkField.TEXT,
            ChunkField.TITLE,
            ChunkField.SOURCE,
            ChunkField.PLATFORM,
            ChunkField.PUBLISHED_AT,
            ChunkField.LANGUAGE,
            ChunkField.ENTITY_IDS,
            ChunkField.CONFIDENCE,
            ChunkField.TENANT_ID,
        ):
            assert spec_field.value in properties, spec_field

    def test_text_and_title_carry_an_exact_subfield(self) -> None:
        """The phrase boost has nowhere to land without them.

        A `match_phrase` against a stemmed, stopworded field cannot express
        "connection reset by peer" -- the stopword filter deletes the "by".
        """
        properties = chunk_index_mappings()["properties"]
        for name in (ChunkField.TEXT.value, ChunkField.TITLE.value):
            assert EXACT_SUBFIELD in properties[name]["fields"], name
            analyzer = properties[name]["fields"][EXACT_SUBFIELD]["analyzer"]
            assert analyzer != "standard"

    def test_language_fields_are_siblings_not_multifields(self) -> None:
        """Per-language analysis must not copy every document into every analyzer.

        A multi-field would run English text through the German stemmer, so the IDF
        of a German term would be computed against a corpus that is mostly not
        German -- ranking damage that no error reports.
        """
        properties = chunk_index_mappings()["properties"]
        for code, analyzer in LANGUAGE_ANALYZERS.items():
            name = f"{TEXT_FIELD_PREFIX}{code}"
            assert properties[name] == {"type": "text", "analyzer": analyzer}
        assert set(properties[ChunkField.TEXT.value]["fields"]) == {EXACT_SUBFIELD}

    @pytest.mark.parametrize(
        ("code", "expected"),
        [("pt-BR", "pt"), ("PT_br", "pt"), ("en", "en"), ("", "und"), ("  De ", "de")],
    )
    def test_primary_subtag(self, code: str, expected: str) -> None:
        assert primary_subtag(code) == expected

    def test_language_field_for_unsupported_language_is_none(self) -> None:
        """`und` and unsupported languages stay searchable through `text`."""
        assert language_text_field("und") is None
        assert language_text_field("xx") is None
        assert language_text_field("pt-BR") == f"{TEXT_FIELD_PREFIX}pt"


class TestMappingDrift:
    def test_a_fresh_mapping_reports_no_drift(self) -> None:
        """An alarm that fires against a correct index is an alarm nobody reads."""
        drift = mapping_drift(chunk_index_mappings(), chunk_index_mappings())
        assert not drift.is_breaking
        assert drift.describe() == "no breaking difference"

    def test_server_supplied_default_analyzer_is_not_drift(self) -> None:
        """A live `text` field omits `analyzer` when it is `standard`."""
        live = {"dynamic": "strict", "properties": {"text": {"type": "text"}}}
        expected = {
            "dynamic": "strict",
            "properties": {"text": {"type": "text", "analyzer": "standard"}},
        }
        assert not mapping_drift(live, expected).is_breaking

    def test_missing_field_is_breaking(self) -> None:
        """A query naming an unmapped field matches nothing and raises nothing."""
        live = {"dynamic": "strict", "properties": {"text": {"type": "text"}}}
        expected = {
            "dynamic": "strict",
            "properties": {"text": {"type": "text"}, "text_de": {"type": "text"}},
        }
        drift = mapping_drift(live, expected)
        assert drift.missing_fields == ("text_de",)
        assert drift.is_breaking

    def test_missing_multifield_is_breaking(self) -> None:
        """`text.exact` vanishing removes the phrase boost silently."""
        live = {"dynamic": "strict", "properties": {"text": {"type": "text"}}}
        expected = {
            "dynamic": "strict",
            "properties": {
                "text": {"type": "text", "fields": {"exact": {"type": "text"}}}
            },
        }
        assert mapping_drift(live, expected).missing_fields == ("text.exact",)

    def test_type_and_analyzer_conflicts_are_breaking(self) -> None:
        live = {
            "dynamic": "strict",
            "properties": {
                "confidence": {"type": "keyword"},
                "text_de": {"type": "text", "analyzer": "standard"},
            },
        }
        expected = {
            "dynamic": "strict",
            "properties": {
                "confidence": {"type": "float"},
                "text_de": {"type": "text", "analyzer": "german"},
            },
        }
        drift = mapping_drift(live, expected)
        assert sorted(drift.conflicting_fields) == ["confidence", "text_de"]

    def test_non_strict_dynamic_is_breaking(self) -> None:
        live = {"dynamic": "true", "properties": {}}
        drift = mapping_drift(live, {"dynamic": "strict", "properties": {}})
        assert drift.dynamic == "true"
        assert drift.is_breaking

    def test_extra_fields_are_informational(self) -> None:
        """A newer build's index mid-rolling-upgrade answers our queries correctly."""
        live = {
            "dynamic": "strict",
            "properties": {"text": {"type": "text"}, "novelty": {"type": "float"}},
        }
        expected = {"dynamic": "strict", "properties": {"text": {"type": "text"}}}
        drift = mapping_drift(live, expected)
        assert drift.extra_fields == ("novelty",)
        assert not drift.is_breaking


class TestEnsureChunkIndex:
    async def test_creates_when_absent(self) -> None:
        client = FakeOpenSearch()
        state = await ensure_chunk_index(client, SPEC)

        assert state.created is True
        body = client.indices.created[0]["body"]
        assert body["mappings"]["dynamic"] == "strict"
        assert body["settings"]["index"]["number_of_replicas"] == 0
        assert "analysis" in body["settings"]

    async def test_is_idempotent(self) -> None:
        """Every worker replica calls this at boot; only one may create."""
        client = FakeOpenSearch()
        first = await ensure_chunk_index(client, SPEC)
        second = await ensure_chunk_index(client, SPEC)

        assert (first.created, second.created) == (True, False)
        assert len(client.indices.created) == 1

    async def test_lost_create_race_still_verifies(self) -> None:
        """`resource_already_exists_exception` is the success case -- but only after
        checking, because "created a millisecond ago" and "created last release with
        last release's mapping" arrive as the same response."""
        client = FakeOpenSearch()
        client.indices.mappings[INDEX] = chunk_index_mappings()
        client.indices.create_error = RequestError(
            400, "resource_already_exists_exception", {}
        )
        # Force the create path even though the index exists, as the race does.
        client.indices.exists = _returning(False)  # type: ignore[method-assign]

        state = await ensure_chunk_index(client, SPEC)
        assert state.created is False

    async def test_other_create_errors_propagate(self) -> None:
        client = FakeOpenSearch()
        client.indices.create_error = RequestError(400, "invalid_index_name_exception", {})
        with pytest.raises(RequestError):
            await ensure_chunk_index(client, SPEC)

    async def test_refuses_to_run_against_a_divergent_mapping(self) -> None:
        """Never silently reconciled: `put_mapping` would add a field that is empty
        for every document already written, so queries against it return nothing
        while the mapping reports it present."""
        client = FakeOpenSearch()
        stale = chunk_index_mappings()
        del stale["properties"][f"{TEXT_FIELD_PREFIX}de"]
        client.indices.mappings[INDEX] = stale

        with pytest.raises(ConfigurationError) as caught:
            await ensure_chunk_index(client, SPEC)

        assert "text_de" in str(caught.value.details["missing_fields"])
        assert "reindex" in str(caught.value).lower()

    async def test_refuses_a_non_strict_existing_index(self) -> None:
        client = FakeOpenSearch()
        loose = chunk_index_mappings()
        loose["dynamic"] = "true"
        client.indices.mappings[INDEX] = loose

        with pytest.raises(ConfigurationError):
            await ensure_chunk_index(client, SPEC)

    async def test_tolerates_extra_fields(self) -> None:
        client = FakeOpenSearch()
        newer = chunk_index_mappings()
        newer["properties"]["novelty_score"] = {"type": "float"}
        client.indices.mappings[INDEX] = newer

        state = await ensure_chunk_index(client, SPEC)
        assert state.created is False
        assert state.drift.extra_fields == ("novelty_score",)

    async def test_alias_spanning_two_indices_is_refused(self) -> None:
        """A half-finished reindex must not be compared against an arbitrary half."""
        client = FakeOpenSearch()
        client.indices.mappings["chunks-v1"] = chunk_index_mappings()
        client.indices.mappings["chunks-v2"] = chunk_index_mappings()
        client.indices.aliases[INDEX] = ["chunks-v1", "chunks-v2"]
        client.indices.mappings[INDEX] = chunk_index_mappings()

        with pytest.raises(ConfigurationError) as caught:
            await inspect_chunk_index(client, SPEC)
        assert "more than one concrete" in str(caught.value)

    async def test_missing_index_reports_clearly(self) -> None:
        client = FakeOpenSearch()
        with pytest.raises(ConfigurationError):
            await inspect_chunk_index(client, SPEC)


class TestSwapAlias:
    async def test_swap_is_one_atomic_call(self) -> None:
        """Remove-then-add leaves a window where the alias resolves to nothing and
        every search 404s -- a self-inflicted outage inside a zero-downtime move."""
        client = FakeOpenSearch()
        client.indices.aliases["chunks"] = ["chunks-v1"]

        detached = await swap_alias(client, "chunks", to_index="chunks-v2")

        assert detached == ("chunks-v1",)
        assert len(client.indices.alias_actions) == 1
        actions = client.indices.alias_actions[0]
        assert actions[-1] == {"add": {"index": "chunks-v2", "alias": "chunks"}}
        assert client.indices.aliases["chunks"] == ["chunks-v2"]

    async def test_first_swap_needs_no_existing_alias(self) -> None:
        client = FakeOpenSearch()
        detached = await swap_alias(client, "chunks", to_index="chunks-v1")
        assert detached == ()


# --------------------------------------------------------------------------- #
# index.py -- the document
# --------------------------------------------------------------------------- #


class TestChunkDocument:
    def test_id_is_the_chunk_id(self) -> None:
        """The join key for fusion and the idempotency key for re-indexing."""
        assert a_document(signal_id="sig1", chunk_index=3).chunk_id == "sig1:3"
        assert a_document().chunk_id == chunk_id_for("sig1", 0)

    def test_every_emitted_field_exists_in_the_mapping(self) -> None:
        """The test that makes `dynamic: strict` survivable.

        A field the builder emits and the mapping lacks is a rejected bulk *item* --
        a partial success buried in a 200 body. Checking the two against each other
        here catches it at the commit that introduces it.
        """
        document = a_document(language="de", title="t", entity_ids=["e1"]).to_document()
        allowed = set(chunk_index_mappings()["properties"])
        assert set(document) <= allowed, set(document) - allowed

    def test_language_sibling_is_populated_selectively(self) -> None:
        german = a_document(language="de").to_document()
        assert german[f"{TEXT_FIELD_PREFIX}de"] == german[ChunkField.TEXT.value]
        assert f"{TEXT_FIELD_PREFIX}en" not in german

        unknown = a_document(language="und").to_document()
        siblings = [k for k in unknown if k.startswith(TEXT_FIELD_PREFIX)]
        assert siblings == []
        assert unknown[ChunkField.TEXT.value]

    def test_language_is_stored_as_the_primary_subtag(self) -> None:
        """Stored and filtered values are compared as exact keywords."""
        assert a_document(language="pt-BR").to_document()[ChunkField.LANGUAGE.value] == "pt"

    def test_naive_published_at_is_refused(self) -> None:
        """OpenSearch would assume UTC -- a guess that can move a Signal across a
        window boundary undetectably."""
        with pytest.raises(ValueError, match="timezone-naive"):
            a_document(published_at=datetime(2026, 2, 15)).to_document()  # noqa: DTZ001

    def test_empty_chunk_is_refused(self) -> None:
        with pytest.raises(ValueError, match="neither retrievable nor"):
            a_document(text="   ", title=None)

    def test_inverted_span_is_refused(self) -> None:
        """An inverted span verifies nothing while looking checked."""
        with pytest.raises(ValueError, match="char_end"):
            a_document(char_start=50, char_end=10)

    def test_missing_signal_id_is_refused(self) -> None:
        with pytest.raises(ValueError, match="signal_id"):
            a_document(signal_id="")

    def test_list_fields_are_lists(self) -> None:
        """A one-element list and a bare string round-trip differently."""
        document = a_document(keywords=("a",), topics=("b",), entity_ids=("c",)).to_document()
        assert document[ChunkField.KEYWORDS.value] == ["a"]
        assert document[ChunkField.TOPICS.value] == ["b"]
        assert document[ChunkField.ENTITY_IDS.value] == ["c"]


# --------------------------------------------------------------------------- #
# query_builder.py -- filters are pushed down
# --------------------------------------------------------------------------- #


class TestFilterPushdown:
    def test_tenant_is_always_filtered(self) -> None:
        """Not optional, even in single-tenant Phase 1: a filter that is conditional
        is a filter that will one day be conditionally absent."""
        clauses = build_filter_clauses(Filter())
        assert {"term": {ChunkField.TENANT_ID.value: "default"}} in clauses

    def test_every_dimension_compiles_to_a_filter_clause(self) -> None:
        filters = Filter(
            published_after=JAN,
            published_before=MAR,
            platforms=frozenset({Platform.RSS}),
            sources=frozenset({SourceCategory.NEWS}),
            languages=frozenset({"pt-BR"}),
            entity_ids=frozenset({"ent-1"}),
            min_confidence=0.5,
            tenant_id="acme",
        )
        clauses = build_filter_clauses(filters)

        assert {"term": {ChunkField.TENANT_ID.value: "acme"}} in clauses
        assert {"terms": {ChunkField.PLATFORM.value: ["rss"]}} in clauses
        assert {"terms": {ChunkField.SOURCE.value: ["news"]}} in clauses
        # Normalised through the same function the indexer stores with; `pt-BR` here
        # would match zero documents and raise nothing.
        assert {"terms": {ChunkField.LANGUAGE.value: ["pt"]}} in clauses
        assert {"terms": {ChunkField.ENTITY_IDS.value: ["ent-1"]}} in clauses
        assert {"range": {ChunkField.CONFIDENCE.value: {"gte": 0.5}}} in clauses

    def test_time_window_is_half_open(self) -> None:
        """`[start, end)`: a closed upper bound puts a midnight document in two
        adjacent trend buckets at once."""
        clauses = build_filter_clauses(Filter(published_after=JAN, published_before=MAR))
        window = next(c for c in clauses if "range" in c)["range"]
        assert set(window[ChunkField.PUBLISHED_AT.value]) == {"gte", "lt"}

    @pytest.mark.parametrize("bound", ["published_after", "published_before"])
    def test_naive_bounds_are_refused(self, bound: str) -> None:
        with pytest.raises(ValueError, match="timezone-naive"):
            build_filter_clauses(Filter(**{bound: datetime(2026, 1, 1)}))  # noqa: DTZ001

    def test_filters_land_in_the_filter_clause_not_in_must(self) -> None:
        """A `filter` is restrictive and non-scoring. The same conditions in `must`
        return identical results with different scores, so a five-platform filter
        would rank differently from a one-platform filter for reasons unrelated to
        relevance."""
        request = RetrievalRequest(
            query="revenue", filters=Filter(platforms=frozenset({Platform.RSS}))
        )
        body = build_search_body(request, limit=10)

        clauses = inner_bool(body)["filter"]
        assert {"terms": {ChunkField.PLATFORM.value: ["rss"]}} in clauses
        assert all("terms" not in clause for clause in scoring_clauses(body))

    async def test_the_filter_reaches_the_cluster_and_is_applied_first(self) -> None:
        """The headline property: filtered *inside* the search, not after it.

        The out-of-window documents here score higher than the in-window ones and
        would fill a `size=2` result set entirely. A backend that fetched the top 2
        and then dropped what failed the filter would return nothing at all; one that
        never filtered would return the wrong two. Only pushdown returns the right
        two.
        """
        client = FakeOpenSearch()
        seed(
            client,
            Doc("old:0", text="revenue revenue revenue", published_at=JAN),
            Doc("old:1", text="revenue revenue revenue", published_at=JAN),
            Doc("new:0", text="revenue", published_at=FEB),
            Doc("new:1", text="revenue", published_at=FEB),
        )
        backend = KeywordBackend(client, SPEC)
        request = RetrievalRequest(
            query="revenue",
            filters=Filter(published_after=datetime(2026, 2, 1, tzinfo=UTC)),
        )

        candidates = await backend.search(request, limit=2)

        assert [c.chunk_id for c in candidates] == ["new:0", "new:1"]
        # And the filter was in the body rather than applied to the reply.
        body = bodies_of(client)[0]
        assert any("range" in clause for clause in inner_bool(body)["filter"])
        assert body["size"] == 2

    async def test_tenant_isolation_is_enforced_by_the_query(self) -> None:
        """An unfiltered query returns another tenant's chunks and reports nothing."""
        client = FakeOpenSearch()
        seed(
            client,
            Doc("mine:0", tenant_id="acme"),
            Doc("theirs:0", tenant_id="other"),
        )
        backend = KeywordBackend(client, SPEC)
        request = RetrievalRequest(
            query="quarterly revenue", filters=Filter(tenant_id="acme")
        )

        candidates = await backend.search(request, limit=10)
        assert [c.chunk_id for c in candidates] == ["mine:0"]

    async def test_entity_filter_is_any_of(self) -> None:
        client = FakeOpenSearch()
        seed(
            client,
            Doc("a:0", entity_ids=["ent-1"]),
            Doc("b:0", entity_ids=["ent-9"]),
        )
        backend = KeywordBackend(client, SPEC)
        request = RetrievalRequest(
            query="quarterly", filters=Filter(entity_ids=frozenset({"ent-1", "ent-2"}))
        )

        candidates = await backend.search(request, limit=10)
        assert [c.chunk_id for c in candidates] == ["a:0"]


# --------------------------------------------------------------------------- #
# query_builder.py -- the query
# --------------------------------------------------------------------------- #


class TestQueryStructure:
    def test_minimum_should_match_is_explicit(self) -> None:
        """Without it, a `bool` with `filter` and `should` but no `must` matches every
        document that passes the filter, scored 0 -- so a query with no lexical match
        answers with `k` arbitrary in-window documents at ranks 1..k."""
        body = build_search_body(RetrievalRequest(query="revenue"), limit=10)
        assert inner_bool(body)["must"][0]["bool"]["minimum_should_match"] == 1

    def test_title_is_boosted(self) -> None:
        body = build_search_body(RetrievalRequest(query="revenue"), limit=10)
        title = next(
            c["match"][ChunkField.TITLE.value]
            for c in scoring_clauses(body)
            if "match" in c and ChunkField.TITLE.value in c["match"]
        )
        assert title["boost"] == 2.0

    def test_phrase_clauses_target_the_exact_subfields(self) -> None:
        body = build_search_body(RetrievalRequest(query="connection reset by peer"), limit=10)
        phrase_fields = {
            next(iter(c["match_phrase"])) for c in scoring_clauses(body) if "match_phrase" in c
        }
        assert phrase_fields == {
            f"{ChunkField.TEXT.value}.{EXACT_SUBFIELD}",
            f"{ChunkField.TITLE.value}.{EXACT_SUBFIELD}",
        }

    def test_empty_query_is_an_explicit_match_all(self) -> None:
        """A filter-only browse is legitimate but must be stated, not defaulted."""
        body = build_search_body(RetrievalRequest(query="   "), limit=5)
        assert inner_bool(body)["must"] == [{"match_all": {}}]

    def test_language_clauses_only_when_a_language_is_pinned(self) -> None:
        """Querying every sibling field would run the text through fourteen
        morphologies to add recall in languages nobody asked about."""
        plain = build_search_body(RetrievalRequest(query="laufen"), limit=10)
        assert not [
            c
            for c in scoring_clauses(plain)
            if "match" in c and next(iter(c["match"])).startswith(TEXT_FIELD_PREFIX)
        ]

        pinned = build_search_body(
            RetrievalRequest(query="laufen", filters=Filter(languages=frozenset({"de", "de-AT"}))),
            limit=10,
        )
        sibling = [
            c
            for c in scoring_clauses(pinned)
            if "match" in c and next(iter(c["match"])).startswith(TEXT_FIELD_PREFIX)
        ]
        # Deduped through `primary_subtag`: `de` and `de-AT` are one field.
        assert len(sibling) == 1
        assert next(iter(sibling[0]["match"])) == f"{TEXT_FIELD_PREFIX}de"

    def test_unsupported_pinned_language_adds_no_clause(self) -> None:
        body = build_search_body(
            RetrievalRequest(query="x", filters=Filter(languages=frozenset({"xx"}))), limit=10
        )
        assert not [
            c
            for c in scoring_clauses(body)
            if "match" in c and next(iter(c["match"])).startswith(TEXT_FIELD_PREFIX)
        ]

    def test_expansions_enter_as_underweighted_phrases(self) -> None:
        """An alias is a graph inference about what the user meant, not something
        they typed, so it must never outrank a literal match. Phrases rather than
        `match`, or "Datadog Inc" would match every document containing "Inc"."""
        body = build_search_body(
            RetrievalRequest(query="datadog"),
            limit=10,
            expansions=["Datadog Inc", "datadog inc", "  ", "DDOG"],
        )
        expansion = [
            c
            for c in scoring_clauses(body)
            if "match_phrase" in c
            and c["match_phrase"].get(TEXT_EXACT, {}).get("boost") == 0.6
        ]
        # Deduped case-insensitively; blanks dropped.
        assert len(expansion) == 2
        assert all(
            clause["match_phrase"][f"{ChunkField.TEXT.value}.{EXACT_SUBFIELD}"]["boost"] < 1.0
            for clause in expansion
        )

    def test_expansion_clauses_are_capped(self) -> None:
        """`max_clause_count` is 1024, and breaching it fails the whole query -- which
        degrades into an unreproducible keyword-backend outage."""
        options = QueryOptions(max_expansion_clauses=3)
        body = build_search_body(
            RetrievalRequest(query="q"),
            limit=10,
            options=options,
            expansions=[f"alias {n}" for n in range(50)],
        )
        phrases = [
            c
            for c in scoring_clauses(body)
            if "match_phrase" in c
            and c["match_phrase"].get(TEXT_EXACT, {}).get("boost") == 0.6
        ]
        assert len(phrases) == 3

    def test_keywords_boost_but_cannot_widen_the_result_set(self) -> None:
        """A document whose extracted keyword list happens to contain the query term
        is not a lexical match, so keyword clauses hang off the *outer* should where
        they cannot satisfy `minimum_should_match`."""
        body = build_search_body(
            RetrievalRequest(query="revenue"),
            limit=10,
            keywords=[Keyword(term="Guidance", weight=0.8)],
        )
        outer = inner_bool(body)["should"]
        assert outer == [
            {
                "term": {
                    ChunkField.KEYWORDS.value: {
                        "value": "guidance",
                        "boost": pytest.approx(1.5 * 0.8),
                    }
                }
            }
        ]
        assert all("term" not in clause for clause in scoring_clauses(body))

    def test_keyword_cap_keeps_the_most_salient(self) -> None:
        """Capping by emission order would drop whichever terms the extractor
        happened to produce last rather than the marginal ones."""
        keywords = [Keyword(term=f"k{n}", weight=n / 100) for n in range(1, 11)]
        body = build_search_body(
            RetrievalRequest(query="q"),
            limit=10,
            options=QueryOptions(max_keyword_clauses=2),
            keywords=keywords,
        )
        terms = [
            next(iter(c["term"].values()))["value"] for c in inner_bool(body)["should"]
        ]
        assert terms == ["k10", "k9"]

    def test_recency_decay_is_a_scoring_function_not_a_filter(self) -> None:
        """An old document that is the only real answer must be demoted, not deleted."""
        body = build_search_body(
            RetrievalRequest(query="q"), limit=10, options=QueryOptions(recency_scale_days=30)
        )
        function_score = body["query"]["function_score"]
        gauss = function_score["functions"][0]["gauss"][ChunkField.PUBLISHED_AT.value]
        assert gauss["scale"] == "30d"
        assert gauss["origin"] == "now"
        assert function_score["boost_mode"] == "multiply"
        # The filters survive the wrapping.
        assert inner_bool(body)["filter"]

    def test_no_decay_by_default(self) -> None:
        body = build_search_body(RetrievalRequest(query="q"), limit=10)
        assert "function_score" not in body["query"]

    def test_zero_recency_scale_is_refused(self) -> None:
        with pytest.raises(ValueError, match="recency_scale_days"):
            QueryOptions(recency_scale_days=0)

    @pytest.mark.parametrize("decay", [0.0, 1.0, 1.5])
    def test_invalid_decay_is_refused(self, decay: float) -> None:
        with pytest.raises(ValueError, match="recency_decay"):
            QueryOptions(recency_decay=decay)

    def test_source_is_not_fetched(self) -> None:
        """The passage text is re-read from PostgreSQL; returning it here would ship
        every chunk body over the wire for data that is thrown away."""
        body = build_search_body(RetrievalRequest(query="q"), limit=10)
        assert body["_source"] is False
        assert body["track_total_hits"] is False

    @pytest.mark.parametrize("limit", [0, -1])
    def test_non_positive_limit_is_refused(self, limit: int) -> None:
        with pytest.raises(ValueError, match="limit"):
            build_search_body(RetrievalRequest(query="q"), limit=limit)

    def test_limit_past_the_result_window_is_refused(self) -> None:
        """The cluster rejects it rather than clamping, and the rejection arrives as
        a shard failure from a request nobody can see."""
        with pytest.raises(ValueError, match="max_result_window"):
            build_search_body(RetrievalRequest(query="q"), limit=MAX_RESULT_WINDOW + 1)

    async def test_recency_decay_demotes_without_excluding(self) -> None:
        """End-to-end through the fake, which computes the gauss the way Lucene does."""
        client = FakeOpenSearch(now=MAR)
        seed(
            client,
            Doc("old:0", text="revenue revenue", published_at=datetime(2025, 3, 15, tzinfo=UTC)),
            Doc("new:0", text="revenue", published_at=MAR),
        )
        backend = KeywordBackend(
            client, SPEC, options=QueryOptions(recency_scale_days=30, recency_decay=0.5)
        )

        candidates = await backend.search(RetrievalRequest(query="revenue"), limit=10)

        # The old document scores twice as well lexically and is still demoted -- but
        # it is present, which a range filter could never have achieved.
        assert [c.chunk_id for c in candidates] == ["new:0", "old:0"]


# --------------------------------------------------------------------------- #
# query_builder.py -- the response
# --------------------------------------------------------------------------- #


class TestCandidates:
    def test_ranks_are_one_based_and_dense(self) -> None:
        response = {
            "hits": {"hits": [{"_id": f"sig{n}:0", "_score": 10 - n} for n in range(5)]}
        }
        candidates = candidates_from_response(response)

        assert [c.rank for c in candidates] == [1, 2, 3, 4, 5]
        assert all(c.backend is Backend.KEYWORD for c in candidates)
        assert [c.signal_id for c in candidates] == [f"sig{n}" for n in range(5)]

    def test_a_skipped_hit_does_not_leave_a_hole_in_the_ranks(self) -> None:
        """RRF scores `1 / (k + rank)`, so a hole would hand the survivors a fused
        score that reflects a document which was discarded."""
        response = {
            "hits": {
                "hits": [
                    {"_id": "sig1:0", "_score": 9.0},
                    {"_id": "not-a-chunk-id", "_score": 8.0},
                    {"_id": "sig2:0", "_score": 7.0},
                ]
            }
        }
        candidates = candidates_from_response(response)
        assert [(c.chunk_id, c.rank) for c in candidates] == [("sig1:0", 1), ("sig2:0", 2)]

    def test_duplicate_chunk_ids_are_dropped(self) -> None:
        """Reachable when the searched name is an alias spanning a half-finished
        reindex. Passing both on lets one backend manufacture agreement with itself."""
        response = {
            "hits": {
                "hits": [
                    {"_id": "sig1:0", "_score": 9.0},
                    {"_id": "sig1:0", "_score": 8.0},
                ]
            }
        }
        assert len(candidates_from_response(response)) == 1

    def test_missing_score_is_zero_not_a_failure(self) -> None:
        """`_score` is null when the request sorts by a field. Raw scores are
        diagnostics; fusion consumes the rank."""
        response = {"hits": {"hits": [{"_id": "sig1:0", "_score": None}]}}
        assert candidates_from_response(response)[0].raw_score == 0.0

    @pytest.mark.parametrize(
        "response",
        [
            {"hits": {"hits": []}},
            {"hits": {}},
            {},
            {"hits": {"hits": "nonsense"}},
        ],
    )
    def test_empty_or_odd_responses_are_not_errors(self, response: dict[str, Any]) -> None:
        """"No lexical match" is the normal answer to a semantic question. Raising
        would mark the whole backend failed and lower the report's confidence."""
        assert candidates_from_response(response) == []


# --------------------------------------------------------------------------- #
# opensearch_client.py -- the backend
# --------------------------------------------------------------------------- #


class TestKeywordBackend:
    def test_satisfies_the_search_backend_protocol(self) -> None:
        backend = KeywordBackend(FakeOpenSearch(), SPEC)
        assert isinstance(backend, SearchBackend)
        assert backend.backend is Backend.KEYWORD
        assert backend.index == INDEX

    def test_the_fake_satisfies_the_store_protocol(self) -> None:
        """If the fake drifts from the port, these tests stop testing the port."""
        assert isinstance(FakeOpenSearch(), KeywordStore)

    async def test_searches_the_configured_index(self) -> None:
        client = FakeOpenSearch()
        await KeywordBackend(client, SPEC).search(RetrievalRequest(query="q"), limit=3)
        assert client.searches[0]["index"] == INDEX

    async def test_no_results_is_not_an_error(self) -> None:
        client = FakeOpenSearch()
        candidates = await KeywordBackend(client, SPEC).search(
            RetrievalRequest(query="nothing matches this"), limit=10
        )
        assert candidates == []

    async def test_cluster_failure_propagates(self) -> None:
        """`HybridRetriever` records the backend as failed and continues on the other
        two. Swallowing the error here would produce an empty candidate list
        indistinguishable from a hard query, at full reported confidence."""
        client = FakeOpenSearch()
        client.search_error = RuntimeError("connection refused")
        with pytest.raises(RuntimeError):
            await KeywordBackend(client, SPEC).search(RetrievalRequest(query="q"), limit=5)

    async def test_expansions_are_used_when_seeds_are_present(self) -> None:
        client = FakeOpenSearch()
        seen: list[str] = []

        async def expand(request: RetrievalRequest) -> Sequence[str]:
            seen.append(request.query)
            return ["Datadog Inc"]

        backend = KeywordBackend(client, SPEC, expand=expand)
        await backend.search(
            RetrievalRequest(query="ddog", seed_entity_ids=["ent-1"]), limit=5
        )

        assert seen == ["ddog"]
        body = bodies_of(client)[0]
        assert any(
            "match_phrase" in c and "Datadog Inc" in str(c) for c in scoring_clauses(body)
        )

    async def test_expansion_is_skipped_without_seeds(self) -> None:
        client = FakeOpenSearch()
        called = False

        async def expand(request: RetrievalRequest) -> Sequence[str]:
            nonlocal called
            called = True
            return []

        await KeywordBackend(client, SPEC, expand=expand).search(
            RetrievalRequest(query="q"), limit=5
        )
        assert called is False

    async def test_expansion_failure_does_not_fail_the_query(self) -> None:
        """Expansion is recall on top of a query that already works. Letting a Neo4j
        timeout propagate would drop a third of the fan-out over an optimisation."""
        client = FakeOpenSearch()
        seed(client, Doc("sig1:0", text="quarterly revenue"))

        async def expand(request: RetrievalRequest) -> Sequence[str]:
            raise TimeoutError("neo4j is slow")

        candidates = await KeywordBackend(client, SPEC, expand=expand).search(
            RetrievalRequest(query="quarterly", seed_entity_ids=["ent-1"]), limit=5
        )
        assert [c.chunk_id for c in candidates] == ["sig1:0"]

    async def test_partial_shard_failure_still_returns_results(self) -> None:
        """A lost shard is a 200 with fewer hits. Raising would turn a 20% recall
        loss into a 100% one; the loss is logged instead."""
        client = FakeOpenSearch()
        client.shards = {"total": 5, "successful": 4, "failed": 1}
        seed(client, Doc("sig1:0", text="quarterly revenue"))

        candidates = await KeywordBackend(client, SPEC).search(
            RetrievalRequest(query="quarterly"), limit=5
        )
        assert len(candidates) == 1

    async def test_search_with_terms_takes_keywords_without_a_graph_call(self) -> None:
        client = FakeOpenSearch()
        seed(client, Doc("sig1:0", keywords=["guidance"]))
        backend = KeywordBackend(client, SPEC)

        candidates = await backend.search_with_terms(
            RetrievalRequest(query="quarterly"),
            limit=5,
            keywords=[Keyword(term="guidance", weight=1.0)],
        )
        assert [c.chunk_id for c in candidates] == ["sig1:0"]

    async def test_options_are_carried_into_the_body(self) -> None:
        client = FakeOpenSearch()
        backend = KeywordBackend(client, SPEC, options=QueryOptions(title_boost=7.0))
        await backend.search(RetrievalRequest(query="q"), limit=5)

        title = next(
            c["match"][ChunkField.TITLE.value]
            for c in scoring_clauses(bodies_of(client)[0])
            if "match" in c and ChunkField.TITLE.value in c["match"]
        )
        assert title["boost"] == 7.0


# --------------------------------------------------------------------------- #
# opensearch_client.py -- the indexer
# --------------------------------------------------------------------------- #


class TestKeywordIndexer:
    async def test_documents_are_keyed_by_chunk_id_and_versioned(self) -> None:
        client = FakeOpenSearch()
        outcome = await KeywordIndexer(client, SPEC).index_chunks([a_document()])

        action, source = client.bulk_bodies[0]
        assert action == {
            "index": {
                "_index": INDEX,
                "_id": "sig1:0",
                "version": V1_9,
                "version_type": "external_gte",
            }
        }
        assert source[ChunkField.CHUNK_ID.value] == "sig1:0"
        assert outcome == IndexOutcome(
            indexed=1, conflicts=0, batches=1, index=INDEX, chunk_ids=("sig1:0",)
        )

    async def test_a_stale_backfill_cannot_overwrite_newer_enrichment(self) -> None:
        """The reason external versioning is here at all.

        `scripts/reindex.py` replaying last month's corpus and a live enrichment
        update are two writers to one `_id` with no ordering between them. Under
        last-write-wins the backfill silently reinstates stale output; under the
        version guard the cluster rejects it, inside a 200 body that this indexer
        reads.
        """
        client = FakeOpenSearch()
        indexer = KeywordIndexer(client, SPEC)

        await indexer.index_chunks([a_document(text="new enrichment", pipeline_version=V1_10)])
        outcome = await indexer.index_chunks(
            [a_document(text="stale backfill", pipeline_version=V1_9)]
        )

        assert outcome.indexed == 0
        assert outcome.conflicts == 1
        assert outcome.conflicted_chunk_ids == ("sig1:0",)
        assert client.sources["sig1:0"][ChunkField.TEXT.value] == "new enrichment"

    async def test_the_ordinal_is_what_makes_the_guard_correct(self) -> None:
        """Compared as text, `'1.10.0' >= '1.9.0'` is False -- so a string version
        would invert the guard the moment a component reached 10, accepting the stale
        write and rejecting the new one."""
        assert V1_10 > V1_9
        assert not "1.10.0" >= "1.9.0"

    async def test_equal_version_replay_succeeds_under_external_gte(self) -> None:
        """At-least-once delivery is the normal case: a partition rebalance re-sends
        a Signal at the same `pipeline_version`, and that must be a rewrite rather
        than a permanent failure a retry loop hammers."""
        client = FakeOpenSearch()
        indexer = KeywordIndexer(client, SPEC)

        await indexer.index_chunks([a_document()])
        outcome = await indexer.index_chunks([a_document()])

        assert (outcome.indexed, outcome.conflicts) == (1, 0)

    async def test_equal_version_replay_conflicts_under_external(self) -> None:
        """The stricter mode, and why it is not the default."""
        client = FakeOpenSearch()
        indexer = KeywordIndexer(client, SPEC, version_type=VersionType.EXTERNAL)

        await indexer.index_chunks([a_document()])
        outcome = await indexer.index_chunks([a_document()])

        assert (outcome.indexed, outcome.conflicts) == (0, 1)

    async def test_unversioned_documents_are_refused(self) -> None:
        """0 is what `pipeline_version_ordinal()` returns for an unparseable version.
        Writing it would disarm the guard silently."""
        client = FakeOpenSearch()
        with pytest.raises(ValueError, match="pipeline_version"):
            await KeywordIndexer(client, SPEC).index_chunks([a_document(pipeline_version=0)])
        assert client.bulk_bodies == []

    async def test_duplicate_chunk_in_one_call_is_refused(self) -> None:
        """Two bodies for one citation span, resolved by a coin flip."""
        client = FakeOpenSearch()
        with pytest.raises(ValueError, match="twice"):
            await KeywordIndexer(client, SPEC).index_chunks(
                [a_document(text="one"), a_document(text="two")]
            )

    async def test_a_rejected_item_inside_a_200_is_raised(self) -> None:
        """The failure mode the whole write path is shaped around: `dynamic: strict`
        buys nothing unless somebody reads the per-item statuses."""
        client = FakeOpenSearch()
        client.reject["sig1:0"] = (
            400,
            {
                "type": "strict_dynamic_mapping_exception",
                "reason": "mapping set to strict, dynamic introduction of [novelty] "
                "within [_doc] is not allowed",
            },
        )

        with pytest.raises(BulkIndexError) as caught:
            await KeywordIndexer(client, SPEC).index_chunks([a_document()])

        error = caught.value
        assert error.failures[0].chunk_id == "sig1:0"
        assert error.failures[0].type == "strict_dynamic_mapping_exception"
        assert error.details["rejected"] == 1
        assert error.index == INDEX

    async def test_conflicts_and_successes_are_counted_separately(self) -> None:
        """A run that is all conflicts is a correct no-op; a backfill reporting zero
        conflicts means the guard is not armed. Collapsing them hides both."""
        client = FakeOpenSearch()
        indexer = KeywordIndexer(client, SPEC)
        await indexer.index_chunks([a_document(chunk_index=0, pipeline_version=V1_10)])

        outcome = await indexer.index_chunks(
            [
                a_document(chunk_index=0, pipeline_version=V1_9),
                a_document(chunk_index=1, pipeline_version=V1_9),
            ]
        )
        assert (outcome.indexed, outcome.conflicts, outcome.submitted) == (1, 1, 2)

    async def test_batching_splits_into_several_requests(self) -> None:
        client = FakeOpenSearch()
        indexer = KeywordIndexer(client, SPEC, batch_size=2)

        outcome = await indexer.index_chunks(
            [a_document(chunk_index=n) for n in range(5)]
        )

        assert outcome.batches == 3
        assert [len(body) // 2 for body in client.bulk_bodies] == [2, 2, 1]
        assert outcome.indexed == 5

    async def test_empty_input_sends_nothing(self) -> None:
        client = FakeOpenSearch()
        outcome = await KeywordIndexer(client, SPEC).index_chunks([])
        assert client.bulk_bodies == []
        assert outcome.batches == 0

    async def test_refresh_is_omitted_unless_requested(self) -> None:
        """Forcing a segment flush per bulk collapses indexing throughput."""
        client = FakeOpenSearch()
        await KeywordIndexer(client, SPEC).index_chunks([a_document()])
        assert "refresh" not in client.bulk_params[0]

        await KeywordIndexer(client, SPEC, refresh="wait_for").index_chunks(
            [a_document(chunk_index=1)]
        )
        assert client.bulk_params[1]["refresh"] == "wait_for"

    async def test_bad_batch_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            KeywordIndexer(FakeOpenSearch(), SPEC, batch_size=0)

    async def test_indexed_documents_are_searchable(self) -> None:
        """Write path and read path agree on the document shape.

        Worth an end-to-end pass through the fake because the two halves were written
        against the mapping independently: a field the indexer emits under one name
        and the query names under another produces an empty result set, never an
        error.
        """
        client = FakeOpenSearch()
        await KeywordIndexer(client, SPEC).index_chunks(
            [
                a_document(
                    signal_id="sig9",
                    text="datadog acquired an observability startup",
                    title="Datadog acquisition",
                    entity_ids=["ent-ddog"],
                    language="en",
                )
            ]
        )

        candidates = await KeywordBackend(client, SPEC).search(
            RetrievalRequest(
                query="observability startup",
                filters=Filter(
                    entity_ids=frozenset({"ent-ddog"}),
                    languages=frozenset({"en"}),
                    published_after=JAN,
                    min_confidence=0.0,
                ),
            ),
            limit=5,
        )
        assert [c.chunk_id for c in candidates] == ["sig9:0"]


class TestDeletes:
    async def test_delete_signal_is_by_query_and_proceeds_past_conflicts(self) -> None:
        """The caller does not know how many chunks there were -- the count is a
        property of the text at the time it was chunked. And an erasure that aborts
        halfway on a version conflict is a compliance incident, not a retry."""
        client = FakeOpenSearch()
        seed(client, Doc("sig1:0"), Doc("sig1:1"), Doc("sig2:0"))
        # `Doc` seeds the search corpus; give the deleter its join key.
        for chunk_id in ("sig1:0", "sig1:1", "sig2:0"):
            client.sources[chunk_id][ChunkField.SIGNAL_ID.value] = chunk_id.split(":")[0]

        deleted = await KeywordIndexer(client, SPEC).delete_signal("sig1")

        assert deleted == 2
        call = client.delete_by_query_calls[0]
        assert call["conflicts"] == "proceed"
        assert call["body"] == {"query": {"term": {ChunkField.SIGNAL_ID.value: "sig1"}}}
        assert set(client.sources) == {"sig2:0"}

    async def test_delete_chunks_tolerates_absent_documents(self) -> None:
        """The caller asked for the document to be gone, and it is."""
        client = FakeOpenSearch()
        await KeywordIndexer(client, SPEC).index_chunks([a_document()])

        deleted = await KeywordIndexer(client, SPEC).delete_chunks(["sig1:0", "sig1:99"])

        assert deleted == 1
        assert "sig1:0" not in client.sources

    async def test_delete_chunks_raises_on_a_real_rejection(self) -> None:
        client = FakeOpenSearch()
        await KeywordIndexer(client, SPEC).index_chunks([a_document()])
        client.reject["sig1:0"] = (403, {"type": "cluster_block_exception", "reason": "read-only"})

        with pytest.raises(BulkIndexError):
            await KeywordIndexer(client, SPEC).delete_chunks(["sig1:0"])

    async def test_delete_chunks_refuses_a_malformed_id(self) -> None:
        """A malformed id would delete nothing while reporting a successful erasure."""
        with pytest.raises(ValueError, match="malformed chunk_id"):
            await KeywordIndexer(FakeOpenSearch(), SPEC).delete_chunks(["not-a-chunk-id"])

    async def test_delete_chunks_with_nothing_to_do_sends_nothing(self) -> None:
        client = FakeOpenSearch()
        assert await KeywordIndexer(client, SPEC).delete_chunks([]) == 0
        assert client.bulk_bodies == []


def _returning(value: Any) -> Callable[..., Awaitable[Any]]:
    """An async stub with a fixed answer, for forcing a code path the fake cannot."""

    async def _call(**kwargs: Any) -> Any:
        return value

    return _call
