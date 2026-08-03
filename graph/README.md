# graph/

*The Neo4j knowledge layer: entities, relationships and time (Design Doc §7).*

## Layout

| Path | Purpose |
| --- | --- |
| `client.py` | Async Neo4j driver wrapper. |
| `schema/` | Node labels, edge types, constraints, and the versioned schema migrator. |
| `schema/versions/` | Forward-only `.cypher` migrations. Never edit an applied version. |
| `ingest/` | Idempotent MERGE-based writes with batching and backpressure. |
| `resolution/` | Entity resolution: blocking → scoring → clustering → merge. |
| `temporal/` | `valid_from` / `valid_to` edge intervals and as-of queries. |
| `queries/` | Parameterized Cypher templates. |
| `analytics/` | Centrality and community detection. |

## Schema

**Nodes:** Company, Product, Person, Topic, Technology, Region, Event
**Edges:** MENTIONS, COMPETES_WITH, ACQUIRED, USES, COMPLAINS_ABOUT, LAUNCHED_BY

Every edge is temporal. A query without a time constraint returns the *current*
view; historical questions must go through `temporal/validity.py`.

## Rules

- All writes are idempotent `MERGE`s keyed on the resolved entity id.
- Schema changes are versioned and forward-only (Design Doc §15). Add
  `vNNN_*.cypher`; do not modify a version that has been applied anywhere.
- Entity resolution decisions must be reversible — record the merge, don't
  destroy the inputs.

## See also

[`docs/knowledge-graph.md`](../docs/knowledge-graph.md)
