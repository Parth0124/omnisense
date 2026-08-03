# retrieval/

*Hybrid retrieval: keyword + vector + graph + filtering + reranking (Design Doc §8).*

## Layout

| Path | Purpose |
| --- | --- |
| `hybrid.py` | Orchestrates the fan-out, fusion and rerank. Start reading here. |
| `keyword/` | OpenSearch BM25 candidate generation. |
| `vector/` | Qdrant ANN search, collections, indexing. |
| `graph_retrieval/` | Neo4j neighbourhood traversal and query expansion. |
| `filters/` | Metadata filter DSL pushed down into every backend. |
| `rerank/` | Reciprocal rank fusion, then cross-encoder reranking. |
| `chunking/` | Splitting documents for embedding and citation granularity. |
| `graphrag/` | Context builder merging graph neighbours with retrieved passages. |
| `evaluation/` | recall@k, nDCG, MRR, groundedness. |

## Flow

```
query ──┬─► keyword candidates ──┐
        ├─► vector candidates  ──┼─► RRF fusion ─► cross-encoder rerank ─► GraphRAG context pack
        └─► graph candidates   ──┘
              (all constrained by the same metadata filter)
```

## Rules

- Retrieval never mutates state. It reads.
- Every returned passage carries enough provenance to build a citation — a
  passage that cannot be cited is a bug, not a result.
- Retrieval quality changes are validated against `evaluation/` before merge.

## See also

[`docs/retrieval.md`](../docs/retrieval.md) ·
[`docs/knowledge-graph.md`](../docs/knowledge-graph.md)
