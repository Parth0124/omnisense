# backend/

*The FastAPI API gateway (Design Doc §3, §13).*

Thin by design: parse, authenticate, validate, delegate to `services/` or
`agents/`, serialise. Business logic that appears here belongs in `services/`.

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Application factory and ASGI entrypoint (`backend.main:app`). |
| `api/v1/` | Route handlers, one module per resource. |
| `api/deps.py` | Shared dependencies: sessions, auth, pagination, tenancy. |
| `api/errors.py` | Exception handlers and RFC 7807 problem responses. |
| `core/` | Config, logging, telemetry, security, rate limiting. |
| `db/` | Client bootstrap for all six datastores. |
| `schemas/` | Pydantic request/response DTOs — distinct from `models/`. |
| `middleware/` | Correlation ids, tracing, CORS. |

## Endpoints

| Route | Handler |
| --- | --- |
| `POST /api/v1/investigations` · `GET /api/v1/investigations/{id}` | `api/v1/investigations.py` |
| `POST /api/v1/connectors/sync` | `api/v1/connectors.py` |
| `GET /api/v1/reports/{id}` | `api/v1/reports.py` |
| `GET /api/v1/graph/search` | `api/v1/graph.py` |
| `POST /api/v1/agents/run` | `api/v1/agents.py` |
| `GET /api/v1/signals` | `api/v1/signals.py` |
| `GET /api/v1/health` · `/readyz` | `api/v1/health.py` |
| SSE execution timeline | `api/v1/stream.py` |

## Rules

- Configuration is read **only** through `core/config.py`. No scattered
  `os.environ` lookups.
- `schemas/` are the wire format; `models/` are the domain. Never return an ORM
  object directly.
- No blocking I/O in async handlers.

## Run

```bash
make api    # uvicorn backend.main:app --reload
```

## See also

[`docs/api-reference.md`](../docs/api-reference.md) ·
[`docs/architecture.md`](../docs/architecture.md)
