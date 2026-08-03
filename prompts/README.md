# prompts/

*Versioned, reproducible prompts (Design Doc §15).*

Prompts are treated as source artifacts, not string literals buried in code. A
report generated six months ago must be explainable, which means knowing exactly
which prompt text produced it.

## Layout

| Path | Purpose |
| --- | --- |
| `loader.py` | Loads a prompt by id and version, hashing content for reproducibility. |
| `<agent>/vN.md` | One directory per agent, one file per version. |
| `shared/` | Fragments composed into multiple prompts: system framing, citation rules, confidence rubric, safety. |
| `shared/output_schemas/` | JSON schemas the model output must satisfy. |
| `evals/` | Prompt-level evaluation fixtures. |

## Rules

1. **Never edit a version that has been used.** Create `v2.md`.
2. Every run records the prompt id, version and content hash in its trace.
3. The output contract in a prompt must match the agent's `schemas.py`. If they
   drift, the agent fails validation — which is the intended behaviour.
4. Prompts are reviewed like code.

## See also

[`docs/agent-system.md`](../docs/agent-system.md)
