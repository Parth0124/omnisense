"""Unit tests for `graph/client.py` and `graph/queries/cypher.py`.

No driver and no database. `QueryRunner` is a one-method Protocol precisely so
that the retry classification, the deadline and the value normalisation can be
tested against a callable that returns whatever the test needs -- including
raising a class named `TransientError` that has nothing to do with Neo4j, which
is the point of classifying by name.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from graph.client import (
    GraphClient,
    GraphQueryError,
    GraphUnavailableError,
    is_transient,
    normalize_record,
    normalize_value,
    read_runner_from_session_factory,
)
from graph.queries import cypher as q
from graph.schema.nodes import GraphSchemaError
from models.enums import EdgeType, EntityType

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 6, 12, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _Neo4jDateTime:
    """Stands in for `neo4j.time.DateTime`: converts via `.to_native()`."""

    def __init__(self, value: datetime) -> None:
        self._value = value

    def to_native(self) -> datetime:
        return self._value


class TransientError(Exception):
    """Name-matched by `is_transient`. Deliberately not the driver's class."""


class ServiceUnavailable(Exception):
    pass


class ClientError(Exception):
    pass


class CypherSyntaxError(ClientError):
    """A `ClientError` subclass that must NOT be retried."""


def _runner(rows: list[dict[str, Any]] | None = None, *, raises: list[BaseException] | None = None):
    """A `QueryRunner` that replays a script of failures then succeeds."""
    calls: list[tuple[str, dict[str, Any]]] = []
    pending = list(raises or [])

    async def run(cypher: str, parameters: Any = None) -> list[dict[str, Any]]:
        calls.append((cypher, dict(parameters or {})))
        if pending:
            raise pending.pop(0)
        return list(rows or [])

    run.calls = calls  # type: ignore[attr-defined]
    return run


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


class TestNormalization:
    def test_driver_datetime_becomes_stdlib(self) -> None:
        """`neo4j.time.DateTime` prints like a datetime and compares unequal to
        one, so the bug lands several layers away -- in a Pydantic field that
        rejects it, or a subtraction that yields an unserialisable Duration."""
        assert normalize_value(_Neo4jDateTime(NOW)) == NOW

    def test_nested_containers_are_walked(self) -> None:
        """Cypher returns them: `collect()` yields a list, a map projection a
        dict, and a driver type inside either is exactly as wrong."""
        value = {"rows": [{"at": _Neo4jDateTime(NOW)}], "n": 1}
        assert normalize_value(value) == {"rows": [{"at": NOW}], "n": 1}

    def test_stdlib_datetime_is_untouched(self) -> None:
        assert normalize_value(NOW) is NOW

    def test_primitives_pass_through(self) -> None:
        for value in ("s", 1, 1.5, True, None):
            assert normalize_value(value) is value

    def test_unconvertible_value_is_returned_as_is(self) -> None:
        class Exploding:
            def to_native(self) -> Any:
                raise RuntimeError("nope")

        exploding = Exploding()
        assert normalize_value(exploding) is exploding

    def test_record_normalisation_returns_a_plain_dict(self) -> None:
        record = normalize_record({"at": _Neo4jDateTime(NOW), "id": "x"})
        assert record == {"at": NOW, "id": "x"}
        assert type(record) is dict


# --------------------------------------------------------------------------- #
# Retry classification
# --------------------------------------------------------------------------- #


class TestTransientClassification:
    def test_transient_error_is_retryable(self) -> None:
        assert is_transient(TransientError("leader moved"))

    def test_service_unavailable_is_retryable(self) -> None:
        assert is_transient(ServiceUnavailable("down"))

    def test_unknown_error_is_permanent(self) -> None:
        """Wrong in the safe direction: a new driver exception fails loudly
        instead of being retried into a timeout that misdirects the diagnosis
        towards the network."""
        assert not is_transient(ValueError("something new"))

    def test_client_error_naming_a_leader_condition_is_retryable(self) -> None:
        assert is_transient(ClientError("Not the leader of this database"))

    def test_client_error_for_bad_cypher_is_permanent(self) -> None:
        """Retrying a syntax error three times produces three identical stack
        traces with the real message buried under them."""
        assert not is_transient(CypherSyntaxError("Invalid input 'MERGE'"))

    def test_subclass_of_a_transient_class_is_retryable(self) -> None:
        class Derived(TransientError):
            pass

        assert is_transient(Derived("x"))


# --------------------------------------------------------------------------- #
# GraphClient
# --------------------------------------------------------------------------- #


class TestGraphClient:
    async def test_fetch_normalises_every_row(self) -> None:
        client = GraphClient(_runner([{"at": _Neo4jDateTime(NOW)}]))
        rows = await client.fetch(q.entity_by_id(tenant_id="t", entity_id="e"))
        assert rows == [{"at": NOW}]

    async def test_fetch_one_returns_none_on_empty(self) -> None:
        """'Not in the graph' is an ordinary answer here; turning it into a 404
        is a decision for the service layer, which knows whether the caller asked
        for something that ought to exist."""
        client = GraphClient(_runner([]))
        assert await client.fetch_one(q.entity_by_id(tenant_id="t", entity_id="e")) is None

    async def test_fetch_value_extracts_a_column(self) -> None:
        client = GraphClient(_runner([{"id": "e1", "computed_at": None}]))
        value = await client.fetch_value(q.entity_by_id(tenant_id="t", entity_id="e"), "id")
        assert value == "e1"

    async def test_parameters_reach_the_runner_unaltered(self) -> None:
        run = _runner([])
        client = GraphClient(run)
        await client.fetch(q.entity_by_id(tenant_id="tenant-9", entity_id="e1"))
        _, parameters = run.calls[0]  # type: ignore[attr-defined]
        assert parameters == {"tenant_id": "tenant-9", "entity_id": "e1"}

    async def test_transient_failure_is_retried_then_succeeds(self) -> None:
        run = _runner([{"id": "ok"}], raises=[TransientError("blip")])
        client = GraphClient(run, max_attempts=3)
        rows = await client.fetch(q.entity_by_id(tenant_id="t", entity_id="e"))
        assert rows == [{"id": "ok"}]
        assert len(run.calls) == 2  # type: ignore[attr-defined]

    async def test_permanent_failure_is_not_retried(self) -> None:
        run = _runner(raises=[CypherSyntaxError("bad cypher")] * 5)
        client = GraphClient(run, max_attempts=3)
        with pytest.raises(GraphQueryError):
            await client.fetch(q.entity_by_id(tenant_id="t", entity_id="e"))
        assert len(run.calls) == 1  # type: ignore[attr-defined]

    async def test_exhausted_retries_raise_unavailable(self) -> None:
        run = _runner(raises=[TransientError("still down")] * 5)
        client = GraphClient(run, max_attempts=3)
        with pytest.raises(GraphUnavailableError, match="after 3 attempts"):
            await client.fetch(q.entity_by_id(tenant_id="t", entity_id="e"))
        assert len(run.calls) == 3  # type: ignore[attr-defined]

    async def test_deadline_covers_the_whole_retry_loop(self) -> None:
        """Bounding each attempt separately would let three attempts plus two
        backoffs take three times the stated timeout -- the caller budgeted once
        and would wait three times as long."""

        async def slow(cypher: str, parameters: Any = None) -> list[dict[str, Any]]:
            await asyncio.sleep(10)
            return []

        client = GraphClient(slow, timeout_seconds=0.05, max_attempts=3)
        started = asyncio.get_running_loop().time()
        with pytest.raises(GraphUnavailableError, match="exceeded"):
            await client.fetch(q.entity_by_id(tenant_id="t", entity_id="e"))
        assert asyncio.get_running_loop().time() - started < 1.0

    async def test_timeout_is_unavailability_not_a_query_error(self) -> None:
        """The statement may have been perfectly valid; the graph just did not
        answer. Classifying it as a query error would stop the retrieval path
        from degrading past it."""

        async def slow(cypher: str, parameters: Any = None) -> list[dict[str, Any]]:
            await asyncio.sleep(10)
            return []

        client = GraphClient(slow, timeout_seconds=0.05)
        with pytest.raises(GraphUnavailableError):
            await client.fetch(q.entity_by_id(tenant_id="t", entity_id="e"))

    async def test_degrade_context_swallows_unavailable(self) -> None:
        client = GraphClient(_runner(raises=[ServiceUnavailable("down")] * 5), max_attempts=1)
        async with client.degrade_on_unavailable("expansion"):
            await client.fetch(q.entity_by_id(tenant_id="t", entity_id="e"))
            raise AssertionError("unreachable")  # pragma: no cover

    async def test_degrade_context_does_not_swallow_query_errors(self) -> None:
        """A malformed query is a bug; degrading past it hides it permanently
        rather than for the duration of an outage."""
        client = GraphClient(_runner(raises=[CypherSyntaxError("bad")] * 5), max_attempts=1)
        with pytest.raises(GraphQueryError):
            async with client.degrade_on_unavailable("expansion"):
                await client.fetch(q.entity_by_id(tenant_id="t", entity_id="e"))

    def test_invalid_configuration_is_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError):
            GraphClient(_runner([]), timeout_seconds=0)
        with pytest.raises(ValueError):
            GraphClient(_runner([]), max_attempts=0)


class TestReadRunnerAdapter:
    async def test_uses_execute_read_and_materialises_rows(self) -> None:
        """A Neo4j result stream dies with its transaction. Returning the stream
        itself hands back a closed cursor, and the failure appears at first
        iteration, far from the cause."""
        seen: dict[str, Any] = {}

        class FakeResult:
            def __init__(self, rows: list[dict[str, Any]]) -> None:
                self._rows = rows

            def __aiter__(self):
                async def gen():
                    for row in self._rows:
                        yield type("Record", (), {"data": lambda _s, r=row: r})()

                return gen()

        class FakeTx:
            async def run(self, cypher: str, parameters: Any = None) -> FakeResult:
                seen["cypher"] = cypher
                seen["parameters"] = parameters
                return FakeResult([{"id": "x"}])

        class FakeSession:
            async def execute_read(self, work):
                seen["mode"] = "read"
                return await work(FakeTx())

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        runner = read_runner_from_session_factory(lambda: FakeSession())
        rows = await runner("MATCH (n) RETURN n", {"a": 1})
        assert rows == [{"id": "x"}]
        assert seen["mode"] == "read"
        assert seen["parameters"] == {"a": 1}


# --------------------------------------------------------------------------- #
# Cypher templates
# --------------------------------------------------------------------------- #


def _params_in(cypher: str) -> set[str]:
    return set(re.findall(r"\$([a-z_][a-z0-9_]*)", cypher))


ALL_BUILDERS = [
    lambda: q.entity_neighbours(tenant_id="t", entity_id="e"),
    lambda: q.paths_between(tenant_id="t", source_id="a", target_id="b"),
    lambda: q.subgraph_edges(tenant_id="t", entity_ids=["a", "b"]),
    lambda: q.neighbourhood_signals(
        tenant_id="t", seed_ids=["a"], start=NOW - timedelta(days=1), end=NOW
    ),
    lambda: q.neighbourhood_signals(
        tenant_id="t", seed_ids=["a"], start=NOW - timedelta(days=1), end=NOW, use_apoc=True
    ),
    lambda: q.entity_search(tenant_id="t", query="ibm"),
    lambda: q.entity_by_id(tenant_id="t", entity_id="e"),
    lambda: q.signals_mentioning(tenant_id="t", entity_id="e"),
    lambda: q.topic_activity(tenant_id="t", since=NOW - timedelta(days=1), until=NOW),
    lambda: q.stale_analytics_nodes(
        tenant_id="t", entity_type=EntityType.COMPANY, older_than=NOW
    ),
]


class TestTemplateInvariants:
    @pytest.mark.parametrize("build", ALL_BUILDERS)
    def test_every_referenced_parameter_is_supplied(self, build) -> None:
        """The failure this catches is a `ParameterMissing` at runtime, on the
        read path, for a query nobody exercised with a database attached."""
        query = build()
        assert _params_in(query.cypher) == set(query.parameters)

    def test_no_caller_value_reaches_query_text(self) -> None:
        """Every builder, called with values distinctive enough to find.

        The one property that makes this module a security boundary: a caller's
        string must arrive as a parameter, never as query text. A distinctive
        sentinel is used because the shared `tenant_id="t"` of the other tests is
        a single character that occurs in every Cypher keyword.
        """
        sentinel = "TENANT_SENTINEL_ZZZ"
        needle = "NAME_SENTINEL_ZZZ"
        builders = [
            lambda: q.entity_neighbours(tenant_id=sentinel, entity_id=needle),
            lambda: q.paths_between(tenant_id=sentinel, source_id=needle, target_id=needle),
            lambda: q.subgraph_edges(tenant_id=sentinel, entity_ids=[needle]),
            lambda: q.neighbourhood_signals(
                tenant_id=sentinel,
                seed_ids=[needle],
                start=NOW - timedelta(days=1),
                end=NOW,
            ),
            lambda: q.entity_search(tenant_id=sentinel, query=needle),
            lambda: q.entity_by_id(tenant_id=sentinel, entity_id=needle),
            lambda: q.signals_mentioning(tenant_id=sentinel, entity_id=needle),
        ]
        for build in builders:
            query = build()
            assert sentinel not in query.cypher
            assert needle not in query.cypher
            # ...and it did reach the parameters, so the test is not passing
            # merely because the value was dropped on the floor.
            assert sentinel in query.parameters.values() or any(
                sentinel in str(v) for v in query.parameters.values()
            )

    def test_an_injection_attempt_stays_a_parameter(self) -> None:
        """The classic. It must appear nowhere in the text and intact in the
        parameters, where the driver sends it as a value."""
        attack = "x') DETACH DELETE n //"
        query = q.entity_search(tenant_id="tenant-1", query=attack)
        assert "DETACH DELETE" not in query.cypher
        assert query.parameters["query"] == attack

    @pytest.mark.parametrize("build", ALL_BUILDERS)
    def test_every_template_is_bounded(self, build) -> None:
        """A LIMIT is the only bound on a graph read, and the memory an unbounded
        one costs is the server's, shared with every other query."""
        assert "LIMIT" in build().cypher


class TestTemplateGuards:
    def test_empty_tenant_is_rejected(self) -> None:
        """An empty string is a valid parameter that matches nothing, so the
        caller reports 'no data' for what is actually a wiring bug."""
        with pytest.raises(GraphSchemaError, match="tenant_id"):
            q.entity_search(tenant_id="", query="x")

    def test_limit_above_the_ceiling_is_rejected(self) -> None:
        with pytest.raises(GraphSchemaError, match="MAX_LIMIT"):
            q.entity_search(tenant_id="t", query="x", limit=q.MAX_LIMIT + 1)

    def test_unseeded_neighbourhood_is_rejected(self) -> None:
        """An unseeded walk is a full graph scan."""
        with pytest.raises(GraphSchemaError, match="seed_ids"):
            q.neighbourhood_signals(tenant_id="t", seed_ids=[], start=NOW, end=NOW)

    def test_inverted_window_is_rejected(self) -> None:
        with pytest.raises(GraphSchemaError, match="inverted"):
            q.neighbourhood_signals(
                tenant_id="t", seed_ids=["a"], start=NOW, end=NOW - timedelta(days=1)
            )

    def test_hop_bound_above_the_ceiling_is_rejected(self) -> None:
        """Path count grows with branching raised to the depth."""
        with pytest.raises(GraphSchemaError, match="max_hops"):
            q.paths_between(
                tenant_id="t", source_id="a", target_id="b", max_hops=q.MAX_HOPS + 1
            )

    def test_unknown_entity_type_cannot_be_a_search_filter(self) -> None:
        """`EntityType` is tolerant on read -- `EntityType('Spaceship')` yields
        UNKNOWN rather than raising -- and UNKNOWN is a well-formed label no node
        carries, so the query would run and match nothing."""
        with pytest.raises(GraphSchemaError, match="UNKNOWN"):
            q.entity_search(tenant_id="t", query="x", entity_types=[EntityType.UNKNOWN])

    def test_unknown_entity_type_cannot_be_scored(self) -> None:
        with pytest.raises(GraphSchemaError):
            q.stale_analytics_nodes(
                tenant_id="t", entity_type=EntityType.UNKNOWN, older_than=NOW
            )

    def test_empty_filter_list_is_rejected(self) -> None:
        """`l IN []` is false for every node, which reads as 'no results' rather
        than 'you filtered everything out'."""
        with pytest.raises(GraphSchemaError, match="pass None"):
            q.entity_search(tenant_id="t", query="x", entity_types=[])

    def test_naive_datetime_is_rejected(self) -> None:
        """A naive value would compare a local wall clock against stored UTC and
        be wrong by the server's offset -- silently, and only outside UTC."""
        with pytest.raises(Exception):
            q.entity_neighbours(tenant_id="t", entity_id="e", as_of=datetime(2026, 8, 6))


class TestTemplateSemantics:
    def test_relationship_traversal_is_undirected(self) -> None:
        """Every edge is stored once, in one canonical orientation.

        A directed match therefore returns roughly half the neighbours, and which
        half depends on the lexical accident of two uuids -- so the result looks
        like sparse data rather than like a bug, and nothing raises.
        """
        for cypher in (
            q.entity_neighbours(tenant_id="t", entity_id="e").cypher,
            q.subgraph_edges(tenant_id="t", entity_ids=["e"]).cypher,
            q.paths_between(tenant_id="t", source_id="a", target_id="b").cypher,
        ):
            assert "]-(" in cypher
            assert "]->(" not in cypher

    def test_confidence_ranking_tolerates_a_missing_confidence(self) -> None:
        """A bare `ORDER BY r.confidence` puts unscored edges wherever the server
        sorts nulls, which is not the same end in every clause it appears.

        Most rule-extracted edges carry no confidence at all, so this decides
        where the bulk of the graph lands in a `LIMIT`-ed read. `coalesce` makes
        an unscored edge the weakest rather than the arbitrary one.
        """
        for cypher in (
            q.entity_neighbours(tenant_id="t", entity_id="e").cypher,
            q.subgraph_edges(tenant_id="t", entity_ids=["e"]).cypher,
        ):
            assert "coalesce(r.confidence, 0.0) DESC" in cypher

    def test_a_window_boundary_is_a_parameter_not_server_clock_arithmetic(self) -> None:
        """`duration()` in the query text means two calls a millisecond apart read
        different windows, so a paginated result drops or repeats rows between
        pages -- with no error and no way to notice from the output."""
        cypher = q.topic_activity(
            tenant_id="t", since=NOW - timedelta(days=7), until=NOW
        ).cypher
        assert "duration(" not in cypher

    def test_a_window_is_injectable_rather_than_read_from_the_clock(self) -> None:
        query = q.topic_activity(tenant_id="t", since=NOW - timedelta(days=7), until=NOW)
        assert query.parameters["until"] == NOW
        assert query.parameters["since"] == NOW - timedelta(days=7)

    def test_neighbourhood_excludes_bookkeeping_edges(self) -> None:
        """Walking SAME_AS makes the expansion traverse a node's own merge
        history and return the same entity under several ids, which fusion then
        reads as independent corroboration."""
        cypher = q.neighbourhood_signals(
            tenant_id="t", seed_ids=["a"], start=NOW - timedelta(days=1), end=NOW
        ).cypher
        assert "SAME_AS" not in cypher
        assert "DUPLICATE_OF" not in cypher

    def test_neighbourhood_defaults_to_pure_cypher(self) -> None:
        """APOC is a plugin. Assuming it means `Unknown procedure` on the read
        path in production, from a query the test suite never exercised because
        the test Neo4j had it installed."""
        cypher = q.neighbourhood_signals(
            tenant_id="t", seed_ids=["a"], start=NOW - timedelta(days=1), end=NOW
        ).cypher
        assert "apoc" not in cypher

    def test_neighbourhood_apoc_form_is_available_on_request(self) -> None:
        cypher = q.neighbourhood_signals(
            tenant_id="t",
            seed_ids=["a"],
            start=NOW - timedelta(days=1),
            end=NOW,
            use_apoc=True,
        ).cypher
        assert "apoc.path.subgraphNodes" in cypher

    def test_neighbourhood_caps_entities_before_expanding_to_signals(self) -> None:
        """A Topic with ten thousand inbound MENTIONS would otherwise expand to
        ten thousand signals and only then be truncated."""
        cypher = q.neighbourhood_signals(
            tenant_id="t", seed_ids=["a"], start=NOW - timedelta(days=1), end=NOW
        ).cypher
        assert cypher.index("$fanout_cap") < cypher.index("MENTIONS]->(node)")

    def test_relationship_pattern_deduplicates(self) -> None:
        cypher = q.neighbourhood_signals(
            tenant_id="t",
            seed_ids=["a"],
            start=NOW - timedelta(days=1),
            end=NOW,
            edge_types=[EdgeType.MENTIONS, EdgeType.MENTIONS],
        ).cypher
        assert "MENTIONS|MENTIONS" not in cypher

    def test_entity_search_over_fetches_to_offset_post_index_filtering(self) -> None:
        """The fulltext index cannot be partitioned by tenant, so the filter runs
        after scoring. Recall suffers; nothing leaks."""
        query = q.entity_search(tenant_id="t", query="x", limit=10)
        assert query.parameters["over_fetch"] > query.parameters["limit"]

    def test_entity_search_names_the_declared_index(self) -> None:
        from graph.schema.constraints import FULLTEXT_INDEX_NAME

        assert f"'{FULLTEXT_INDEX_NAME}'" in q.entity_search(tenant_id="t", query="x").cypher

    def test_as_of_predicate_is_generated_not_handwritten(self) -> None:
        """One copy in the repository. A second is how one query gets `>=` and
        another `>`, so an edge closing at exactly the queried instant is
        returned by one and not the other."""
        from graph.temporal.validity import as_of_cypher

        assert as_of_cypher("r") in q.entity_neighbours(tenant_id="t", entity_id="e").cypher
