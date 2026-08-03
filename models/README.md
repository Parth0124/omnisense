# models/

*Canonical domain models shared by every other module.*

This package is the bottom of the dependency graph. It imports nothing else in
the repository, and everything else may import it. If you find yourself needing
`services/` or `backend/` from here, the model belongs somewhere else.

## Layout

| Path | Purpose |
| --- | --- |
| `signal.py` | The canonical `Signal` — the single normalized unit of data (Design Doc §6). |
| `entity.py` | Resolved entity model backing the knowledge graph nodes. |
| `investigation.py` | A long-running, resumable user question and its execution state. |
| `report.py` | Evidence-backed narrative output. |
| `evidence.py` | Evidence and citation links tying claims to source signals. |
| `trend.py` / `forecast.py` | Trend detection and forecast outputs. |
| `connector.py` | Connector descriptors, credential references, sync cursors. |
| `lineage.py` | Provenance chain from raw payload to derived insight. |
| `enums.py` | Shared enumerations. |
| `orm/` | SQLAlchemy mappings for the PostgreSQL-persisted subset. |

## Rules

- Domain models are the API contract. Changing a field is a breaking change —
  see the schema evolution rules in [`docs/signal-model.md`](../docs/signal-model.md).
- `orm/` types are persistence detail and must not leak into agent or connector
  signatures. Convert at the service boundary.
- Every persisted model carries `tenant_id` from day one so Phase 7 multi-tenancy
  is not a rewrite.

## See also

[`docs/signal-model.md`](../docs/signal-model.md) ·
[`docs/data-stores.md`](../docs/data-stores.md)
