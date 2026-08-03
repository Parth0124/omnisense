// Local development bootstrap for the OmniSense knowledge graph.
// Applied by `make init-db` (scripts/init_databases.py), not automatically by Neo4j.
// The authoritative, versioned schema lives in graph/schema/versions/.

// --------------------------------------------------------------- uniqueness --
CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT technology_id IF NOT EXISTS FOR (n:Technology) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT region_id IF NOT EXISTS FOR (n:Region) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (n:Event) REQUIRE n.id IS UNIQUE;

// ------------------------------------------------------------------ lookups --
CREATE INDEX company_name IF NOT EXISTS FOR (n:Company) ON (n.canonical_name);
CREATE INDEX product_name IF NOT EXISTS FOR (n:Product) ON (n.canonical_name);
CREATE INDEX topic_name IF NOT EXISTS FOR (n:Topic) ON (n.canonical_name);
CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
  FOR (n:Company|Product|Person|Topic|Technology|Region|Event)
  ON EACH [n.canonical_name, n.aliases, n.description];

// ----------------------------------------------------------------- temporal --
// Every relationship carries valid_from / valid_to so the graph can be queried
// "as of" any point in time (Design Doc S7: temporal relationships).
CREATE INDEX mentions_observed IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.observed_at);
CREATE INDEX competes_validity IF NOT EXISTS FOR ()-[r:COMPETES_WITH]-() ON (r.valid_from);
