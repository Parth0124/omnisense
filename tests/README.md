# tests/

*The test suite, mirroring the source layout.*

| Path | Marker | What belongs here |
| --- | --- | --- |
| `unit/` | `unit` | Fast, isolated, no network, no database. |
| `integration/` | `integration` | Real datastores via `make up`. |
| `e2e/` | `e2e` | A full investigation, end to end. |
| `evals/` | `eval` | Agent quality scoring — tracked, **not** pass/fail. |
| `factories/` | — | Test data builders. |
| `fixtures/` | — | Recorded payloads, sample signals, graph fixtures. |

`unit/` mirrors the source tree: `unit/connectors/` tests `connectors/`, and so on.

## Running

```bash
make test               # unit only — the default inner loop
make test-integration   # requires `make up`
make test-all           # everything, with coverage
make eval               # agent evaluations (non-blocking)
```

## Rules

- A unit test that opens a socket is misfiled. Use `respx` to fake HTTP.
- Agent tests assert on **structured output**, never on prose wording.
- Evaluations measure quality and are expected to fluctuate; they gate nothing,
  but a regression is a signal worth investigating.

## See also

[`docs/testing-strategy.md`](../docs/testing-strategy.md)
