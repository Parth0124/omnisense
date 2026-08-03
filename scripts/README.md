# scripts/

*Operational scripts for local development.*

| Script | Purpose |
| --- | --- |
| `init_databases.py` | Create schemas, indexes, Qdrant collections and Neo4j constraints. |
| `seed_data.py` | Load sample signals and entities for local development. |
| `sync_connector.py` | Run one connector sync from the command line. |
| `reindex.py` | Rebuild Qdrant and OpenSearch indexes from PostgreSQL. |
| `export_graph_schema.py` | Dump the current Neo4j schema for review. |

Run with the project virtualenv active, from the repository root:

```bash
python scripts/init_databases.py      # or: make init-db
python scripts/seed_data.py           # or: make seed
```

## Rules

- These scripts operate on whatever `DATABASE_URL`, `NEO4J_URI` etc. point at.
  Check your `.env` before running anything destructive.
- `reindex.py` and `init_databases.py` must be idempotent and safe to re-run.

## See also

[`docs/local-development.md`](../docs/local-development.md)
