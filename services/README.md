# services/

*The Signal Engine and orchestration-agnostic business logic.*

Services hold the logic that must behave identically whether it is invoked by an
HTTP request, a background worker, or an agent tool. They know about datastores
and domain models; they know nothing about FastAPI or LangGraph.

## Layout

| Path | Purpose |
| --- | --- |
| `signal_engine/` | The seven-stage enrichment pipeline (Design Doc §6). |
| `llm/` | Model-agnostic AI layer: provider interface, routing, caching, embeddings. |
| `storage/` | Cloudflare R2 object storage and media handling. |
| `events/` | Kafka/Redpanda producer, consumer runtime, topics and event schemas. |
| `*_service.py` | Per-aggregate application services (investigation, report, graph, trend, …). |

## The Signal Engine

```
Clean → Normalize → Language Detection → Entity Extraction → Sentiment → Embedding → Store
```

Each stage is independently testable and independently failable. A stage failure
degrades the signal (recording the failure in `lineage`) rather than dropping it,
unless the failure is in `Clean` or `Normalize`.

## Rules

- No FastAPI imports here. No LangGraph imports here.
- All model access goes through `llm/provider.py` so the platform stays
  model-agnostic (Design Doc §15).
- Persistence fan-out to six stores must be idempotent — every write is retried.

## See also

[`docs/signal-model.md`](../docs/signal-model.md) ·
[`docs/architecture.md`](../docs/architecture.md)
