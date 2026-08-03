# workers/

*Background processing driven by Kafka/Redpanda events and schedules.*

Ingestion is event-driven (Design Doc §15) so a slow enrichment step never blocks
a connector fetch, and a connector outage never stalls the API.

## Layout

| Path | Purpose |
| --- | --- |
| `runtime/base_worker.py` | Consumer loop: lifecycle, graceful shutdown, metrics. |
| `scheduler.py` | Triggers periodic connector syncs and maintenance jobs. |
| `ingestion_worker.py` | Consumes connector output, emits raw records. |
| `enrichment_worker.py` | Runs the Signal Engine pipeline. |
| `embedding_worker.py` | Batches and generates embeddings. |
| `indexing_worker.py` | Writes to Qdrant and OpenSearch. |
| `graph_worker.py` | Entity resolution and knowledge-graph updates. |
| `forecast_worker.py` | Recomputes trend and forecast aggregates. |
| `report_worker.py` | Renders long-running report jobs. |
| `dlq.py` | Dead-letter handling and replay. |

## Rules

- Every handler is **idempotent**. Delivery is at-least-once; duplicates will
  happen.
- Correlation ids propagate through the event envelope so a trace spans the API
  and every worker that touched the record.
- A poisoned message goes to the DLQ after bounded retries — it never blocks the
  partition.

## Run

```bash
make worker      # enrichment worker
make scheduler   # connector sync scheduler
WORKER_MODULE=workers.graph_worker python -m workers.graph_worker
```

## See also

[`docs/architecture.md`](../docs/architecture.md) ·
[`docs/observability.md`](../docs/observability.md)
