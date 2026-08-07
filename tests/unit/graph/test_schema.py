"""Unit tests for `graph/schema/`.

The graph schema is enforced in three places that can silently disagree:
`graph/schema/constraints.py` generates it, `graph/schema/versions/v001_initial.cypher`
is what the migrator applies, and `docker/local/neo4j/01-constraints.cypher` is
what `make init-db` applies to a developer's laptop. Nothing at run time notices
when they drift -- every statement is `IF NOT EXISTS`, so a missing index is a
query plan nobody looks at until production is slow and the laptop is fast. The
drift test below is the only thing that notices, and it is the main reason this
file exists.

The rest pins the two properties the write path depends on:

* an edge between labels it may not connect is refused, because there is no query
  that finds a nonsense relationship after it has been written; and
* no value ever reaches query text. Entity names come from scraped third-party
  content, so Cypher injection here is not hypothetical -- it is a company name
  away.

Everything runs with no Neo4j, no driver and no network: these modules generate
strings and validate maps, which is deliberately all they do.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from graph.schema import constraints, edges, nodes
from graph.schema.edges import (
    SIGNAL_LABEL,
    Cardinality,
    Direction,
    edge_key,
    edge_spec,
    orient,
    validate_edge_properties,
    validate_endpoints,
)
from graph.schema.migrator import (
    GraphMigrator,
    MigrationError,
    discover_versions,
    split_statements,
)
from graph.schema.nodes import (
    GraphSchemaError,
    PropertyOwner,
    PropertyType,
    entity_labels,
    node_spec,
    validate_label,
    validate_node_properties,
)
from models.enums import EdgeType, EntityType

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
BOOTSTRAP = REPO_ROOT / "docker" / "local" / "neo4j" / "01-constraints.cypher"

# A value a scraped entity name could plausibly carry, and which would end an
# identifier and start a new clause if it were ever interpolated.
HOSTILE = "Acme`) DETACH DELETE n //"


def normalise(statement: str) -> str:
    """Collapse whitespace so a three-line statement compares to a one-line one."""
    return " ".join(statement.split())


class TestNodeRegistry:
    """The seven labels, and what every one of them must carry."""

    def test_exactly_the_seven_entity_types_are_registered(self) -> None:
        """`Unknown` must never become a label: it exists for forward tolerance."""
        assert set(nodes.NODE_SPECS) == {
            member for member in EntityType if member is not EntityType.UNKNOWN
        }

    def test_unknown_is_refused_rather_than_defaulted(self) -> None:
        with pytest.raises(GraphSchemaError, match="not a writable node label"):
            node_spec(EntityType.UNKNOWN)

    @pytest.mark.parametrize("entity_type", list(nodes.NODE_SPECS))
    def test_every_label_carries_the_common_block(self, entity_type: EntityType) -> None:
        """id, canonical_name, aliases, tenant_id, first_seen, last_seen -- always.

        Cross-label queries (the fulltext index, the neighbourhood expansion,
        every tenant filter) address these by name on any label. One label
        missing one of them is a result set that is quietly incomplete.
        """
        spec = node_spec(entity_type)
        for name in (
            "id",
            "canonical_name",
            "aliases",
            "tenant_id",
            "first_seen",
            "last_seen",
        ):
            assert spec.has_property(name), f"{spec.label} is missing {name}"

    @pytest.mark.parametrize("entity_type", list(nodes.NODE_SPECS))
    def test_identity_properties_are_required(self, entity_type: EntityType) -> None:
        required = {p.name for p in node_spec(entity_type).required_properties}
        assert {"id", "tenant_id", "canonical_name", "normalized_name"} <= required

    def test_labels_are_capitalised_cypher_identifiers(self) -> None:
        for label in entity_labels():
            assert validate_label(label) == label
            assert label[0].isupper()

    def test_embedding_replaces_rather_than_accumulating(self) -> None:
        """A unioned vector is a corrupt vector.

        `aliases` accumulates because knowledge about an entity is additive.
        Applying the same rule to `embedding` would concatenate two vectors,
        deduplicate the components that happened to be equal, and truncate the
        result -- producing something that still type-checks and quietly ruins
        every resolution decision that reads it.
        """
        spec = node_spec(EntityType.COMPANY)
        assert spec.property_spec("aliases").accumulates is True
        assert spec.property_spec("embedding").accumulates is False

    def test_handles_is_a_list_not_a_map(self) -> None:
        """Neo4j cannot store a map as a property value.

        `docs/knowledge-graph.md` §2 describes `Person.handles` as a map. Writing
        one raises `Property values can only be of primitive types or arrays
        thereof` at the driver, so the schema models it as `platform:handle`
        strings.
        """
        assert node_spec(EntityType.PERSON).property_spec("handles").type is (
            PropertyType.STRING_LIST
        )

    def test_analytics_properties_exist_but_are_not_caller_writable(self) -> None:
        spec = node_spec(EntityType.COMPANY)
        assert spec.property_spec("pagerank_score").owner is PropertyOwner.ANALYTICS
        assert "pagerank_score" not in {p.name for p in spec.caller_properties}


class TestNodePropertyValidation:
    """Neo4j accepts anything; this is where a write is actually checked."""

    def base(self) -> dict[str, object]:
        return {
            "id": "ent_acme",
            "tenant_id": "t1",
            "canonical_name": "Acme",
            "normalized_name": "acme",
        }

    def test_a_valid_map_passes(self) -> None:
        validate_node_properties(EntityType.COMPANY, {**self.base(), "ticker": "ACME"})

    def test_a_typo_is_rejected_rather_than_silently_created(self) -> None:
        """The failure this whole module exists for.

        `canonical_nam` would become a real property on every node the writer
        touches, and entity search would quietly stop returning the entity.
        """
        with pytest.raises(GraphSchemaError, match="canonical_nam"):
            validate_node_properties(
                EntityType.COMPANY, {**self.base(), "canonical_nam": "Acme"}
            )

    def test_a_missing_required_property_is_rejected(self) -> None:
        payload = self.base()
        del payload["normalized_name"]
        with pytest.raises(GraphSchemaError, match="normalized_name is required"):
            validate_node_properties(EntityType.COMPANY, payload)

    def test_a_wrong_type_is_rejected(self) -> None:
        """A datetime stored as a string compares against nothing, forever."""
        with pytest.raises(GraphSchemaError, match="declared datetime but got str"):
            validate_node_properties(
                EntityType.PRODUCT, {**self.base(), "released_at": "2024-01-01"}
            )

    def test_a_bool_does_not_pass_as_an_integer(self) -> None:
        """`bool` subclasses `int` in Python, so this needs an explicit guard."""
        with pytest.raises(GraphSchemaError, match="declared integer but got bool"):
            validate_node_properties(
                EntityType.COMPANY, {**self.base(), "founded_year": True}
            )

    def test_a_value_outside_a_closed_vocabulary_is_rejected(self) -> None:
        with pytest.raises(GraphSchemaError, match="outside its vocabulary"):
            validate_node_properties(
                EntityType.PRODUCT, {**self.base(), "lifecycle_state": "GA"}
            )

    def test_a_caller_cannot_write_an_ingest_owned_counter(self) -> None:
        """An absolute `source_count` discards every other writer's accumulation."""
        with pytest.raises(GraphSchemaError, match="owned by ingest"):
            validate_node_properties(EntityType.COMPANY, {**self.base(), "source_count": 5})

    def test_an_explicit_null_is_allowed_for_an_optional_property(self) -> None:
        """"I have no value" is a legitimate thing for a row to say.

        The generated fragment coalesces it against what is already stored, so a
        mention that knows only a name does not blank the ticker.
        """
        validate_node_properties(EntityType.COMPANY, {**self.base(), "ticker": None})


class TestEdgeRegistry:
    """Direction, cardinality and endpoints -- the modelling decisions."""

    def test_every_edge_type_except_unknown_is_registered(self) -> None:
        assert set(edges.EDGE_SPECS) == {
            member for member in EdgeType if member is not EdgeType.UNKNOWN
        }

    @pytest.mark.parametrize("edge_type", list(edges.EDGE_SPECS))
    def test_every_edge_carries_the_bitemporal_block(self, edge_type: EdgeType) -> None:
        """`valid_from` / `valid_to` is what makes an as-of query possible.

        And `observed_at` is what keeps it from being conflated with transaction
        time, which `docs/knowledge-graph.md` §5 calls the most common temporal
        bug in this layer.
        """
        names = {p.name for p in edge_spec(edge_type).properties}
        assert {"valid_from", "valid_to", "observed_at", "edge_key", "tenant_id"} <= names

    @pytest.mark.parametrize("edge_type", list(edges.EDGE_SPECS))
    def test_valid_from_is_required_and_valid_to_is_not(self, edge_type: EdgeType) -> None:
        """Null `valid_to` means "still true"; a sentinel date would sort as real."""
        spec = edge_spec(edge_type)
        assert spec.property_spec("valid_from").required is True
        assert spec.property_spec("valid_to").required is False

    def test_mentions_connects_a_signal_to_every_entity_label(self) -> None:
        spec = edge_spec(EdgeType.MENTIONS)
        assert spec.source_labels() == (SIGNAL_LABEL,)
        assert set(spec.target_labels()) == set(entity_labels())

    def test_competes_with_is_symmetric_and_many_to_many(self) -> None:
        spec = edge_spec(EdgeType.COMPETES_WITH)
        assert spec.direction is Direction.SYMMETRIC
        assert spec.cardinality is Cardinality.MANY_TO_MANY

    def test_acquired_is_directed_because_reversing_it_inverts_the_fact(self) -> None:
        spec = edge_spec(EdgeType.ACQUIRED)
        assert spec.direction is Direction.DIRECTED
        assert spec.cardinality is Cardinality.ONE_TO_MANY

    def test_complains_about_accepts_a_signal_subject(self) -> None:
        """Most complainants are throwaway accounts that resolve to no Person.

        Requiring one would fill the graph with millions of singleton people that
        never merge with anything.
        """
        assert edge_spec(EdgeType.COMPLAINS_ABOUT).allows(SIGNAL_LABEL, "Product")

    def test_a_symmetric_edge_set_is_closed_under_swapping(self) -> None:
        """Otherwise `orient()` can flip a legal pair into an illegal one.

        The edge would then be accepted or refused depending on the alphabetical
        accident of its two ids, which is the worst possible kind of flake.
        """
        for spec in edges.EDGE_SPECS.values():
            if spec.direction is Direction.SYMMETRIC:
                assert {(t, s) for s, t in spec.endpoints} == spec.endpoints


class TestEndpointValidation:
    """The check that stops "Acme competes with Belgium" from reaching a report."""

    def test_a_legal_pair_passes(self) -> None:
        validate_endpoints(EdgeType.COMPETES_WITH, "Company", "Company")
        validate_endpoints(EdgeType.USES, "Product", "Technology")

    def test_competes_with_between_a_company_and_a_region_is_refused(self) -> None:
        """The example from the brief, and it is refused at the only catchable moment.

        Nothing in Neo4j objects to this relationship, no query detects it
        afterwards, and it surfaces months later inside a report.
        """
        with pytest.raises(GraphSchemaError, match="cannot connect"):
            validate_endpoints(EdgeType.COMPETES_WITH, "Company", "Region")

    def test_a_reversed_pair_says_so(self) -> None:
        """`LAUNCHED_BY` points from the launched thing to the launcher.

        Ingest naturally produces company → product, so this is the mistake that
        will actually be made, and the message is what turns it into a two-minute
        fix rather than a re-read of the schema.
        """
        with pytest.raises(GraphSchemaError, match="probably reversed"):
            validate_endpoints(EdgeType.LAUNCHED_BY, "Company", "Product")

    def test_an_unknown_label_is_refused(self) -> None:
        with pytest.raises(GraphSchemaError):
            validate_endpoints(EdgeType.MENTIONS, SIGNAL_LABEL, "Spaceship")

    def test_a_hostile_label_never_becomes_a_query(self) -> None:
        with pytest.raises(GraphSchemaError, match="not a valid Neo4j label"):
            validate_endpoints(EdgeType.MENTIONS, SIGNAL_LABEL, HOSTILE)

    def test_merge_cypher_cannot_be_built_for_an_illegal_pair(self) -> None:
        """Validation is structural, not a lint pass bolted on beside the writer."""
        with pytest.raises(GraphSchemaError):
            edges.merge_cypher(EdgeType.ACQUIRED, "Person", "Company")


class TestEdgeIdentity:
    """`orient()` and `edge_key()`: why a replay lands on the same relationship."""

    def test_a_symmetric_edge_is_oriented_the_same_way_from_either_side(self) -> None:
        """Stored once. Two mirrored copies drift the moment one side is updated."""
        forward = orient(EdgeType.COMPETES_WITH, "ent_zulu", "Company", "ent_acme", "Product")
        backward = orient(EdgeType.COMPETES_WITH, "ent_acme", "Product", "ent_zulu", "Company")
        assert forward == backward == ("ent_acme", "Product", "ent_zulu", "Company")

    def test_orientation_carries_the_label_with_its_id(self) -> None:
        """Swapping ids without labels would MATCH an id against the wrong label.

        The query would find nothing, `MERGE` would never run, and the edge would
        vanish with no error anywhere.
        """
        from_id, from_label, to_id, to_label = orient(
            EdgeType.COMPETES_WITH, "ent_zulu", "Company", "ent_acme", "Product"
        )
        assert (from_id, from_label) == ("ent_acme", "Product")
        assert (to_id, to_label) == ("ent_zulu", "Company")

    def test_a_directed_edge_is_never_reoriented(self) -> None:
        assert orient(EdgeType.ACQUIRED, "ent_zulu", "Company", "ent_acme", "Company") == (
            "ent_zulu",
            "Company",
            "ent_acme",
            "Company",
        )

    def test_the_key_is_deterministic(self) -> None:
        when = datetime(2024, 1, 1, tzinfo=UTC)
        assert edge_key(EdgeType.MENTIONS, "a", "b", when, "s1") == edge_key(
            EdgeType.MENTIONS, "a", "b", when, "s1"
        )

    def test_the_key_is_timezone_normalised(self) -> None:
        """The same instant in two timezones must not be two edges."""
        import zoneinfo

        utc = datetime(2024, 1, 1, 12, tzinfo=UTC)
        other = utc.astimezone(zoneinfo.ZoneInfo("Asia/Kolkata"))
        assert edge_key(EdgeType.USES, "a", "b", utc) == edge_key(EdgeType.USES, "a", "b", other)

    def test_a_naive_valid_from_is_refused(self) -> None:
        """A key derived from a local wall clock is not reproducible."""
        with pytest.raises(GraphSchemaError, match="naive"):
            edge_key(EdgeType.USES, "a", "b", datetime(2024, 1, 1))

    def test_the_interval_is_part_of_the_key(self) -> None:
        """Competed, stopped, competes again is two edges, not one reopened."""
        first = edge_key(EdgeType.COMPETES_WITH, "a", "b", datetime(2020, 1, 1, tzinfo=UTC))
        second = edge_key(EdgeType.COMPETES_WITH, "a", "b", datetime(2024, 1, 1, tzinfo=UTC))
        assert first != second

    def test_components_cannot_forge_a_boundary(self) -> None:
        """A separator-joined key would collide `("a|b","c")` with `("a","b|c")`."""
        when = datetime(2024, 1, 1, tzinfo=UTC)
        assert edge_key(EdgeType.USES, "a|b", "c", when) != edge_key(
            EdgeType.USES, "a", "b|c", when
        )


class TestEdgePropertyValidation:
    def base(self) -> dict[str, object]:
        return {
            "edge_key": "k",
            "tenant_id": "t1",
            "valid_from": datetime(2024, 1, 1, tzinfo=UTC),
            "observed_at": datetime(2024, 1, 2, tzinfo=UTC),
        }

    def test_a_valid_map_passes(self) -> None:
        validate_edge_properties(EdgeType.COMPETES_WITH, {**self.base(), "basis": "stated"})

    def test_an_inverted_interval_is_refused(self) -> None:
        """An empty interval is a fact that exists and can never be read.

        `[t, t)` contains no instant, so the as-of predicate excludes it at every
        point in time.
        """
        payload = {**self.base(), "valid_to": datetime(2023, 1, 1, tzinfo=UTC)}
        with pytest.raises(GraphSchemaError, match="empty or inverted"):
            validate_edge_properties(EdgeType.COMPETES_WITH, payload)

    def test_a_property_from_another_edge_type_is_refused(self) -> None:
        """`salience` belongs to MENTIONS. On COMPETES_WITH it is a typo."""
        with pytest.raises(GraphSchemaError, match="salience"):
            validate_edge_properties(EdgeType.COMPETES_WITH, {**self.base(), "salience": 0.5})


class TestGeneratedCypherIsParameterised:
    """Cypher injection is as real as SQL injection, and entity names are scraped."""

    def test_no_node_fragment_contains_an_interpolated_value(self) -> None:
        """Only `$rows`, `$batch_id` and `$schema_version` carry data.

        Everything that varies per write is a parameter, which also means Neo4j
        caches one plan per label instead of one per company name.
        """
        for entity_type in nodes.NODE_SPECS:
            query = nodes.merge_cypher(entity_type)
            params = set(re.findall(r"\$(\w+)", query))
            assert params == {"rows", "batch_id", "schema_version"}, entity_type

    def test_no_edge_fragment_contains_an_interpolated_value(self) -> None:
        for edge_type, spec in edges.EDGE_SPECS.items():
            for source, target in spec.endpoints:
                query = edges.merge_cypher(edge_type, source, target)
                params = set(re.findall(r"\$(\w+)", query))
                assert params == {"rows", "batch_id", "schema_version"}, edge_type

    def test_the_only_interpolated_tokens_are_labels_from_the_enum(self) -> None:
        """A label cannot be parameterised in Cypher, so it is checked instead.

        This asserts the *set* of identifiers that appear after a colon in a
        pattern is exactly the known vocabulary -- if a label ever came from data,
        it would show up here as a token nobody declared.
        """
        known = set(entity_labels()) | {SIGNAL_LABEL} | {
            spec.type_name for spec in edges.EDGE_SPECS.values()
        }
        for edge_type, spec in edges.EDGE_SPECS.items():
            for source, target in spec.endpoints:
                query = edges.merge_cypher(edge_type, source, target)
                found = set(re.findall(r"[(\[][a-z]?:(\w+)", query))
                assert found <= known, (edge_type, found - known)

    def test_a_node_merge_keys_on_id_alone(self) -> None:
        """A fuller MERGE map creates a second node the moment a name changes.

        This is the single most common way a Neo4j graph acquires duplicates.
        """
        query = nodes.merge_cypher(EntityType.COMPANY)
        assert "MERGE (n:Company {id: row.id})" in query
        assert "canonical_name:" not in query

    def test_a_node_merge_never_writes_an_analytics_property(self) -> None:
        """Ingest has no opinion about PageRank; setting it would null it hourly."""
        query = nodes.merge_cypher(EntityType.COMPANY)
        for name in ("pagerank_score", "community_id", "computed_at"):
            assert name not in query

    def test_the_counter_is_guarded_against_replay(self) -> None:
        """`SET n.c = n.c + 1` through a managed transaction applies twice on retry.

        `backend/db/neo4j.py` says exactly that. The `replayed` guard is what
        makes the increment safe to route through one anyway, and it covers
        Kafka's at-least-once redelivery in the same stroke.
        """
        query = nodes.merge_cypher(EntityType.COMPANY)
        assert "(n.last_batch_id = $batch_id) AS replayed" in query
        assert "CASE WHEN replayed THEN 0 ELSE" in query

    def test_an_optional_property_is_coalesced_rather_than_overwritten(self) -> None:
        """A row that says nothing about the ticker must not blank it."""
        assert "coalesce(row.ticker, n.ticker)" in nodes.merge_cypher(EntityType.COMPANY)

    def test_the_edge_match_seeks_both_labelled_endpoints(self) -> None:
        """An unlabelled `MATCH (a {id: …})` is an all-nodes scan.

        There is no global id index in Neo4j, so on the write path at 500 rows a
        batch this is the difference between an index seek and reading the store.
        """
        query = edges.merge_cypher(EdgeType.MENTIONS, SIGNAL_LABEL, "Company")
        assert "MATCH (a:Signal {id: row.from_id, tenant_id: row.tenant_id})" in query
        assert "MATCH (b:Company {id: row.to_id, tenant_id: row.tenant_id})" in query

    def test_valid_from_is_only_written_on_create(self) -> None:
        """An interval's start is a fact about when it began.

        A later batch re-reporting the same edge must not move it forward; that
        would erase the history the bitemporal model exists to keep.
        """
        query = edges.merge_cypher(EdgeType.ACQUIRED, "Company", "Company")
        on_create, _, on_match = query.partition("WITH r, row,")
        assert "r.valid_from = row.valid_from" in on_create
        assert "r.valid_from =" not in on_match  # `valid_from_precision` may be
        # ...but closing an interval on match is exactly how a change is recorded.
        assert "r.valid_to = coalesce(row.valid_to, r.valid_to)" in on_match

    def test_no_fragment_calls_apoc(self) -> None:
        """APOC is a plugin, and a write path that needs one fails when it is absent."""
        for entity_type in nodes.NODE_SPECS:
            assert "apoc." not in nodes.merge_cypher(entity_type)

    def test_list_properties_are_capped(self) -> None:
        """An uncapped list property grows a node until reading its name is slow."""
        assert "[..100]" in nodes.merge_cypher(EntityType.COMPANY)
        assert "[..50]" in edges.merge_cypher(EdgeType.MENTIONS, SIGNAL_LABEL, "Company")

    def test_the_fragment_is_byte_identical_across_calls(self) -> None:
        """Neo4j caches a plan keyed by query text; a varying string misses it."""
        assert nodes.merge_cypher(EntityType.TOPIC) is nodes.merge_cypher(EntityType.TOPIC)


class TestConstraints:
    """What the server actually enforces."""

    def test_every_label_has_a_uniqueness_constraint_on_id(self) -> None:
        """Without it, `MERGE` is a read-then-create and concurrency duplicates.

        This is what makes `graph/ingest/writer.py` safe, not an optimisation.
        """
        constrained = {c.label for c in constraints.UNIQUENESS_CONSTRAINTS}
        assert constrained == set(entity_labels())

    def test_the_fulltext_index_spans_all_seven_labels(self) -> None:
        """`GET /graph/search` fails outright without it, rather than being slow."""
        (index,) = constraints.FULLTEXT_INDEXES
        assert index.name == "entity_search"
        assert index.labels == entity_labels()
        assert index.properties == ("canonical_name", "aliases", "description")

    def test_the_temporal_indexes_cover_the_two_hottest_edge_properties(self) -> None:
        declared = {(i.edge_type, i.property) for i in constraints.RELATIONSHIP_INDEXES}
        assert declared == {
            (EdgeType.MENTIONS, "observed_at"),
            (EdgeType.COMPETES_WITH, "valid_from"),
        }

    def test_relationship_indexes_are_undirected(self) -> None:
        """`COMPETES_WITH` is stored once and matched with `-[r:…]-`.

        A directed index would not be usable by that match, and every competitor
        query would scan the type.
        """
        for index in constraints.RELATIONSHIP_INDEXES:
            assert "FOR ()-[r:" in index.to_cypher()
            assert "]->()" not in index.to_cypher()

    def test_every_statement_is_idempotent(self) -> None:
        """A partially-applied version must be safe to re-run, and `make init-db`
        must be a no-op against a live graph."""
        for statement in constraints.statements():
            assert "IF NOT EXISTS" in statement

    def test_the_signal_stub_constraint_is_declared_but_not_applied(self) -> None:
        """A known gap, recorded rather than hidden.

        `graph/ingest/writer.py` merges on `(:Signal {id})` with nothing behind
        it. Applying the constraint here would make this file drift from the
        bootstrap; leaving it undocumented would let it stay unfixed.
        """
        (pending,) = constraints.PENDING_CONSTRAINTS
        assert pending.label == SIGNAL_LABEL
        assert pending.to_cypher() not in constraints.statements()


class TestNoDrift:
    """The three copies of the schema must not diverge. Nothing else notices."""

    def bootstrap_statements(self) -> list[str]:
        return [normalise(s) for s in split_statements(BOOTSTRAP.read_text(encoding="utf-8"))]

    def v001_statements(self) -> list[str]:
        (v001,) = discover_versions()
        return [normalise(s) for s in v001.statements]

    def generated_statements(self) -> list[str]:
        return [normalise(s) for s in constraints.statements()]

    def test_the_bootstrap_file_matches_the_generator(self) -> None:
        """`make init-db` applies the bootstrap; the migrator applies `v001`.

        If they differ, a developer's laptop and production have different
        indexes, and every `IF NOT EXISTS` in both files guarantees neither
        environment ever complains.
        """
        assert self.bootstrap_statements() == self.generated_statements()

    def test_v001_matches_the_generator(self) -> None:
        assert self.v001_statements() == self.generated_statements()

    def test_v001_is_the_authoritative_copy_of_the_bootstrap(self) -> None:
        assert self.v001_statements() == self.bootstrap_statements()


class TestStatementSplitter:
    """`str.split(';')` is wrong, and the ways it is wrong are not theoretical."""

    def test_line_comments_are_dropped(self) -> None:
        assert split_statements("// header\nRETURN 1;") == ("RETURN 1",)

    def test_a_semicolon_inside_a_string_is_not_a_separator(self) -> None:
        """A data migration writing punctuation would otherwise split mid-literal."""
        assert split_statements("SET n.note = 'a; b';") == ("SET n.note = 'a; b'",)

    def test_a_semicolon_inside_a_backticked_identifier_is_not_a_separator(self) -> None:
        assert split_statements("MATCH (n:`odd;name`) RETURN n;") == (
            "MATCH (n:`odd;name`) RETURN n",
        )

    def test_an_escaped_quote_does_not_close_the_literal(self) -> None:
        assert split_statements(r"SET n.x = 'it\'s; fine';") == (r"SET n.x = 'it\'s; fine'",)

    def test_block_comments_are_dropped(self) -> None:
        assert split_statements("/* a; b */ RETURN 1;") == ("RETURN 1",)

    def test_an_unterminated_literal_raises_rather_than_truncating(self) -> None:
        """Silently splitting a broken file produces statements that half-run."""
        with pytest.raises(Exception, match="unterminated"):
            split_statements("SET n.x = 'oops")

    def test_a_trailing_statement_without_a_semicolon_is_kept(self) -> None:
        assert split_statements("RETURN 1;\nRETURN 2") == ("RETURN 1", "RETURN 2")

    def test_empty_and_comment_only_input_yields_nothing(self) -> None:
        """An empty query sent to the server is an error, not a no-op."""
        assert split_statements("// nothing here\n\n;;") == ()


class FakeSchemaGraph:
    """A `CypherRunner` that models the singleton `(:_SchemaVersion)` node.

    Not a Cypher interpreter -- it recognises the migrator's two bookkeeping
    statements by shape and records everything else as "applied". That is enough
    to exercise the three behaviours that matter (skip, order, checksum) without
    a database, and it fails loudly if the migrator ever starts issuing a
    statement this fake does not expect, which is exactly when the test should
    stop being trusted.
    """

    def __init__(self) -> None:
        self.record: dict[str, list[object]] | None = None
        self.applied: list[str] = []
        self.extra_nodes: list[dict[str, object]] = []
        self.fail_on: str | None = None

    async def __call__(
        self,
        query: str,
        parameters: object = None,
    ) -> list[dict[str, object]]:
        params = dict(parameters or {})  # type: ignore[arg-type]

        if query.lstrip().startswith("MATCH (v:_SchemaVersion)"):
            rows: list[dict[str, object]] = []
            if self.record is not None:
                rows.append({"id": "omnisense", **self.record})
            rows.extend(self.extra_nodes)
            return rows

        if "MERGE (v:_SchemaVersion" in query:
            if self.record is None:
                self.record = {
                    "versions": [],
                    "checksums": [],
                    "applied_at": [],
                    "durations_ms": [],
                }
            if params["version"] not in self.record["versions"]:
                self.record["versions"].append(params["version"])
                self.record["checksums"].append(params["checksum"])
                self.record["applied_at"].append(params["applied_at"])
                self.record["durations_ms"].append(params["duration_ms"])
            return [{"current_version": params["version"]}]

        if self.fail_on is not None and self.fail_on in query:
            raise RuntimeError("Neo.ClientError.Schema.ConstraintAlreadyExists")
        self.applied.append(query)
        return []


def write_version(directory: Path, version: int, slug: str, body: str) -> Path:
    path = directory / f"v{version:03d}_{slug}.cypher"
    path.write_text(body, encoding="utf-8")
    return path


class TestMigrator:
    """Forward-only, checksummed, and loud about the three ways it can go wrong."""

    def test_the_repository_ships_exactly_v001(self) -> None:
        (v001,) = discover_versions()
        assert (v001.version, v001.slug) == (1, "initial")
        assert len(v001.statements) == len(constraints.statements())

    async def test_applying_a_version_runs_every_statement_and_records_it(
        self, tmp_path: Path
    ) -> None:
        write_version(tmp_path, 1, "initial", "RETURN 1;\nRETURN 2;\n")
        graph = FakeSchemaGraph()
        migrator = GraphMigrator(graph, versions_dir=tmp_path)

        result = await migrator.apply_all()

        assert result.applied == (1,)
        assert graph.applied == ["RETURN 1", "RETURN 2"]
        (recorded,) = await migrator.applied_versions()
        assert recorded.version == 1
        assert recorded.duration_ms >= 0

    async def test_applying_an_already_applied_version_is_a_no_op(
        self, tmp_path: Path
    ) -> None:
        """Re-running must be reported as skipped, not merely be harmless.

        Every statement is `IF NOT EXISTS`, so a second run would do nothing
        either way -- but an operator watching a deploy needs to tell "already up
        to date" from "applied four versions".
        """
        write_version(tmp_path, 1, "initial", "RETURN 1;")
        graph = FakeSchemaGraph()
        migrator = GraphMigrator(graph, versions_dir=tmp_path)

        first = await migrator.apply_all()
        second = await migrator.apply_all()

        assert first.applied == (1,) and first.skipped == ()
        assert second.applied == () and second.skipped == (1,)
        assert graph.applied == ["RETURN 1"], "the statement ran twice"

    async def test_versions_apply_in_numeric_order(self, tmp_path: Path) -> None:
        """Lexical order agrees with numeric order only while the padding holds."""
        write_version(tmp_path, 2, "second", "RETURN 2;")
        write_version(tmp_path, 10, "tenth", "RETURN 10;")
        write_version(tmp_path, 1, "first", "RETURN 1;")

        graph = FakeSchemaGraph()
        result = await GraphMigrator(graph, versions_dir=tmp_path).apply_all()

        assert result.applied == (1, 2, 10)
        assert graph.applied == ["RETURN 1", "RETURN 2", "RETURN 10"]

    async def test_a_version_appearing_below_the_high_water_mark_is_an_error(
        self, tmp_path: Path
    ) -> None:
        """Two branches each added a version and one was deployed first.

        Applying the straggler now produces a schema that no version file
        describes and no other environment shares, so the only safe move is to
        stop and make a human renumber.
        """
        write_version(tmp_path, 1, "first", "RETURN 1;")
        write_version(tmp_path, 3, "third", "RETURN 3;")
        graph = FakeSchemaGraph()
        migrator = GraphMigrator(graph, versions_dir=tmp_path)
        await migrator.apply_all()

        write_version(tmp_path, 2, "second", "RETURN 2;")
        with pytest.raises(MigrationError, match="forward-only"):
            await migrator.apply_all()

    async def test_an_edited_applied_version_is_an_error(self, tmp_path: Path) -> None:
        """`IF NOT EXISTS` makes re-running an edited file quietly do nothing.

        Without the checksum, two environments would disagree about what `v001`
        was and nothing would ever say so.
        """
        path = write_version(tmp_path, 1, "initial", "RETURN 1;")
        graph = FakeSchemaGraph()
        migrator = GraphMigrator(graph, versions_dir=tmp_path)
        await migrator.apply_all()

        path.write_text("RETURN 1;\nRETURN 2;\n", encoding="utf-8")
        with pytest.raises(MigrationError, match="has changed since it was applied"):
            await migrator.apply_all()

    async def test_even_a_comment_only_edit_is_caught(self, tmp_path: Path) -> None:
        """The checksum is over raw text on purpose.

        Normalising before hashing would make the check tolerant of exactly the
        edits it exists to catch.
        """
        path = write_version(tmp_path, 1, "initial", "RETURN 1;")
        graph = FakeSchemaGraph()
        migrator = GraphMigrator(graph, versions_dir=tmp_path)
        await migrator.apply_all()

        path.write_text("// harmless\nRETURN 1;", encoding="utf-8")
        with pytest.raises(MigrationError, match="has changed"):
            await migrator.pending()

    async def test_a_recorded_version_with_no_file_is_an_error(
        self, tmp_path: Path
    ) -> None:
        write_version(tmp_path, 1, "initial", "RETURN 1;")
        graph = FakeSchemaGraph()
        migrator = GraphMigrator(graph, versions_dir=tmp_path)
        await migrator.apply_all()

        (tmp_path / "v001_initial.cypher").unlink()
        with pytest.raises(MigrationError, match="no file for it exists"):
            await migrator.pending()

    async def test_a_failed_statement_leaves_the_version_unrecorded(
        self, tmp_path: Path
    ) -> None:
        """So the next run retries from the beginning, which is safe.

        Recording a half-applied version would make the rest of it unreachable
        forever, because a recorded version is never re-run.
        """
        write_version(tmp_path, 1, "initial", "RETURN 1;\nCREATE CONSTRAINT x;\n")
        graph = FakeSchemaGraph()
        graph.fail_on = "CREATE CONSTRAINT"
        migrator = GraphMigrator(graph, versions_dir=tmp_path)

        with pytest.raises(MigrationError, match="failed at statement 2 of 2"):
            await migrator.apply_all()
        assert await migrator.applied_versions() == ()

    async def test_a_fresh_graph_reports_nothing_applied(self, tmp_path: Path) -> None:
        """The normal case on a first deploy and in every integration test."""
        write_version(tmp_path, 1, "initial", "RETURN 1;")
        migrator = GraphMigrator(FakeSchemaGraph(), versions_dir=tmp_path)
        assert await migrator.applied_versions() == ()
        assert [f.version for f in await migrator.pending()] == [1]

    async def test_two_schema_version_nodes_abort_the_run(self, tmp_path: Path) -> None:
        """`MERGE` on this node has no uniqueness constraint behind it either.

        With two, there is no way to know which records the truth, so guessing is
        worse than stopping.
        """
        write_version(tmp_path, 1, "initial", "RETURN 1;")
        graph = FakeSchemaGraph()
        graph.record = {
            "versions": [1],
            "checksums": ["x"],
            "applied_at": [None],
            "durations_ms": [1],
        }
        graph.extra_nodes.append(
            {
                "id": "other",
                "versions": [1],
                "checksums": ["y"],
                "applied_at": [None],
                "durations_ms": [1],
            }
        )
        with pytest.raises(MigrationError, match="there must be exactly one"):
            await GraphMigrator(graph, versions_dir=tmp_path).applied_versions()

    async def test_drifted_parallel_arrays_abort_the_run(self, tmp_path: Path) -> None:
        """The cost of Neo4j having no list-of-maps property type.

        With the arrays out of step, version-to-checksum is guesswork, and a
        checksum check that guesses is worse than no check at all.
        """
        write_version(tmp_path, 1, "initial", "RETURN 1;")
        graph = FakeSchemaGraph()
        graph.record = {
            "versions": [1, 2],
            "checksums": ["x"],
            "applied_at": [None, None],
            "durations_ms": [1, 2],
        }
        with pytest.raises(MigrationError, match="drifted"):
            await GraphMigrator(graph, versions_dir=tmp_path).applied_versions()

    async def test_an_empty_version_file_is_an_error(self, tmp_path: Path) -> None:
        """Almost always a file whose contents were never written.

        Recording it as applied would hide that permanently, because a recorded
        version is never re-run.
        """
        write_version(tmp_path, 1, "initial", "// nothing yet\n")
        migrator = GraphMigrator(FakeSchemaGraph(), versions_dir=tmp_path)
        with pytest.raises(MigrationError, match="contains no statements"):
            await migrator.apply_all()

    def test_a_misnamed_file_raises_rather_than_being_skipped(
        self, tmp_path: Path
    ) -> None:
        """A silently skipped migration is worse than a failed deploy."""
        (tmp_path / "v2_thing.cypher").write_text("RETURN 1;", encoding="utf-8")
        with pytest.raises(MigrationError, match="not a valid version filename"):
            discover_versions(tmp_path)

    def test_two_files_claiming_one_version_raise(self, tmp_path: Path) -> None:
        write_version(tmp_path, 1, "initial", "RETURN 1;")
        write_version(tmp_path, 1, "also_initial", "RETURN 2;")
        with pytest.raises(MigrationError, match="two files claim version"):
            discover_versions(tmp_path)

    async def test_recording_is_itself_idempotent(self, tmp_path: Path) -> None:
        """A managed transaction that loses its commit ack is retried by the driver.

        Appending twice would leave the parallel arrays describing a history that
        never happened.
        """
        write_version(tmp_path, 1, "initial", "RETURN 1;")
        graph = FakeSchemaGraph()
        migrator = GraphMigrator(graph, versions_dir=tmp_path)
        (version_file,) = migrator.discover()

        await migrator.apply(version_file)
        # Simulate the driver replaying the record statement verbatim.
        await graph(
            "MERGE (v:_SchemaVersion {id: $node_id})",
            {
                "node_id": "omnisense",
                "version": 1,
                "checksum": version_file.checksum,
                "applied_at": None,
                "duration_ms": 0,
            },
        )
        assert len(await migrator.applied_versions()) == 1

    async def test_apply_returns_false_for_an_already_applied_version(
        self, tmp_path: Path
    ) -> None:
        write_version(tmp_path, 1, "initial", "RETURN 1;")
        migrator = GraphMigrator(FakeSchemaGraph(), versions_dir=tmp_path)
        (version_file,) = migrator.discover()
        assert await migrator.apply(version_file) is True
        assert await migrator.apply(version_file) is False

    def test_a_missing_versions_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(MigrationError, match="does not exist"):
            discover_versions(tmp_path / "nope")
