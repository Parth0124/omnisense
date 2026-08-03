// Knowledge graph schema, version 001 (initial).
//
// Graph schema is versioned and forward-only (Design Doc S15). To change the
// schema, add v002_*.cypher - never edit an applied version in place.
// Applied by graph/schema/migrator.py, which records applied versions on a
// singleton (:_SchemaVersion) node.

// TODO: move the constraint and index definitions from
// docker/local/neo4j/01-constraints.cypher here once the migrator exists.
