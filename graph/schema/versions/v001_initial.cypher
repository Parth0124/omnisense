// Knowledge graph schema, version 001 (initial).
//
// Graph schema is versioned and forward-only (Design Doc S15). To change the
// schema, add v002_*.cypher - never edit an applied version in place: the
// checksum recorded on the singleton (:_SchemaVersion) node stops matching,
// graph/schema/migrator.py refuses to run, and the two environments have
// already diverged by the time anyone notices.
//
// Applied by graph/schema/migrator.py, which records applied versions on that
// singleton node. docker/local/neo4j/01-constraints.cypher holds the same
// statements for `make init-db`, and graph/schema/constraints.py generates
// both; tests/unit/graph/test_schema.py asserts all three agree.
//
// Every statement is IF NOT EXISTS, so a partially-applied run is safe to
// repeat. That matters more than atomicity here: Neo4j refuses to mix schema
// commands with data writes in one transaction, so no version can be atomic
// once it does both, and idempotence is the property that replaces it.

CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT technology_id IF NOT EXISTS FOR (n:Technology) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT region_id IF NOT EXISTS FOR (n:Region) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE;
CREATE INDEX company_name IF NOT EXISTS FOR (n:Company) ON (n.canonical_name);
CREATE INDEX product_name IF NOT EXISTS FOR (n:Product) ON (n.canonical_name);
CREATE INDEX topic_name IF NOT EXISTS FOR (n:Topic) ON (n.canonical_name);
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
  FOR (n:Company|Product|Person|Topic|Technology|Region|Event)
  ON EACH [n.canonical_name, n.aliases, n.description];
CREATE INDEX mentions_observed IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.observed_at);
CREATE INDEX competes_validity IF NOT EXISTS FOR ()-[r:COMPETES_WITH]-() ON (r.valid_from);
