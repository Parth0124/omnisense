# docker/

*Container images and local service bootstrap.*

| Path | Purpose |
| --- | --- |
| `backend.Dockerfile` | API gateway image. Build from the repo root. |
| `worker.Dockerfile` | Worker image; `WORKER_MODULE` selects which worker runs. |
| `frontend.Dockerfile` | Next.js standalone image. Build from `frontend/`. |
| `entrypoints/` | Container entrypoint scripts. |
| `local/postgres/` | SQL run once on first container start (extensions, schemas). |
| `local/neo4j/` | Cypher constraints applied by `make init-db`. |
| `local/opensearch/`, `local/redpanda/` | Local bootstrap assets. |

The compose file itself lives at the repository root
([`docker-compose.yml`](../docker-compose.yml)) — it is a **local development
stack only**. Everything under `local/` uses throwaway development credentials
and must never be used anywhere else.

## Building

```bash
docker build -f docker/backend.Dockerfile .
docker build -f docker/worker.Dockerfile .
docker build -f docker/frontend.Dockerfile ./frontend
```

## See also

[`docs/local-development.md`](../docs/local-development.md) ·
[`docs/deployment.md`](../docs/deployment.md)
