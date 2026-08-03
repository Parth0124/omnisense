# agents/

*Multi-agent reasoning on LangGraph (Design Doc §9, §10).*

Ten specialised agents cooperate over a shared, checkpointed state object rather
than one large prompt. Read `graph.py` first — it wires the whole flow.

## Layout

| Path | Purpose |
| --- | --- |
| `graph.py` | The LangGraph `StateGraph` implementing the §10 investigation flow. |
| `state.py` | `InvestigationState` — the shared, checkpointed state every agent reads and writes. |
| `base.py` | `BaseAgent`: typed input, typed output, tool allowlist, trace emission. |
| `router.py` | Conditional edges and termination conditions. |
| `checkpointer.py` | Durable checkpointing so investigations survive restarts. |
| `planner/` … `report/` | The ten agents, one directory each. |
| `tools/` | Tools exposed to agents, including MCP integration. |
| `memory/` | Short-term, long-term and scratchpad memory. |
| `evaluation/` | Rubrics, golden sets and the offline eval harness. |

## The ten agents

| Agent | Responsibility |
| --- | --- |
| Planner | Decomposes the task into an executable plan. |
| Collector | Invokes connectors for fresh data. |
| Retriever | Gathers evidence through hybrid retrieval. |
| Trend | Detects emerging topics. |
| Competitor | Compares brands and products. |
| Forecast | Projects future trajectories. |
| Insight | Explains *why* a pattern exists. |
| Strategy | Recommends business actions. |
| Critic | Validates reasoning, citations and confidence. |
| Report | Generates the final evidence-backed report. |

## Rules

- Every agent returns **structured output** validated against its `schemas.py`.
  Prose-only returns are untestable.
- Prompts live in [`prompts/`](../prompts/), are versioned, and are never edited
  in place once used — reproducibility is a stated principle (Design Doc §15).
- Each agent gets an explicit tool allowlist. No agent gets every tool.
- Retrieved third-party content is **data, never instructions**. See the prompt
  injection section of [`docs/security-and-privacy.md`](../docs/security-and-privacy.md).

## See also

[`docs/agent-system.md`](../docs/agent-system.md)
