# OmniSense

*An autonomous multi-agent market intelligence platform: continuously ingest digital signals from across the internet, build organizational knowledge, detect trends, forecast opportunities, and recommend business actions.*

> ## Status: pre-implementation scaffold
>
> **This repository contains structure, configuration and documentation only. There is no implemented behaviour.**
> Every Python module is a docstring stub containing a `TODO`. Every file under `frontend/src/` is a
> placeholder. `migrations/versions/` is empty and no test has been written. The local Docker stack
> starts real datastores, but nothing reads from or writes to them yet.
>
> The documents under [`docs/`](./docs/) are the specification for what must be built — not a description
> of what runs today. See [`docs/roadmap.md`](./docs/roadmap.md) for the phase plan; the repository is at Phase 0.

---

## Quickstart

Requires Python 3.12+, Node 20+, and Docker with Compose.

```bash
make bootstrap    # venv + pip install -r requirements.txt -r requirements-dev.txt, npm install, create .env
make up           # start Postgres, Redis, Neo4j, Qdrant, OpenSearch, Redpanda
make init-db      # create schemas, indexes, Qdrant collections, Neo4j constraints
make api          # FastAPI gateway on :8000
make frontend     # Next.js dev server on :3000
```

`make init-db`, `make api` and `make frontend` execute stubs today: the datastores come up, the
processes start, and neither serves real behaviour. Full walkthrough in
[`docs/getting-started.md`](./docs/getting-started.md); the daily loop is in
[`docs/local-development.md`](./docs/local-development.md).

Run `make help` for the full target list. `make check` runs lint, typecheck and unit tests;
`make down` stops the stack and `make nuke` also deletes its volumes.

| Service | Host port |
| --- | --- |
| API (FastAPI) | 8000 |
| Frontend (Next.js) | 3000 |
| PostgreSQL | 5432 |
| Redis | 6379 |
| Neo4j | 7474 (browser), 7687 (Bolt) |
| Qdrant | 6333 (REST), 6334 (gRPC) |
| OpenSearch | 9200 |
| Redpanda (Kafka API) | 19092 |

Application containers (`api`, `worker`, `frontend`) sit behind the `apps` Compose profile, so the
default `make up` starts datastores only and you run the app from your host with hot reload.

---

## Repository layout

The layout is flat: every directory below is a sibling at the repository root. There is no `src/`.

| Directory | Contents |
| --- | --- |
| [`agents/`](./agents/) | The ten investigation agents (planner, collector, retriever, trend, competitor, forecast, insight, strategy, critic, report), their tools and the LangGraph wiring |
| [`backend/`](./backend/) | FastAPI gateway: routers under `api/v1/`, settings and cross-cutting concerns in `core/`, store clients in `db/`, middleware, Pydantic schemas |
| [`connectors/`](./connectors/) | Source connectors grouped by category (social, reviews, enterprise, research, news) plus shared auth, rate limiting, normalization and dedup |
| [`docker/`](./docker/) | Dockerfiles, container entrypoints and local datastore init material |
| [`docs/`](./docs/) | The engineering documentation set — start at [`docs/README.md`](./docs/README.md) |
| [`frontend/`](./frontend/) | Next.js App Router application: seven pages, component families, stores, hooks and API client |
| [`graph/`](./graph/) | Neo4j layer: schema and versioned migrations, ingest, entity resolution, temporal edges, queries, analytics |
| [`infra/`](./infra/) | Deployment scaffolding for Kubernetes, Modal, Railway, Vercel, Prometheus and Grafana. Design material; nothing here is executed by this repository |
| [`migrations/`](./migrations/) | Alembic configuration and revisions for PostgreSQL |
| [`models/`](./models/) | The canonical `Signal` and the SQLAlchemy ORM models under `orm/` |
| [`prompts/`](./prompts/) | Versioned prompt assets, one directory per agent, plus shared output schemas |
| [`retrieval/`](./retrieval/) | Hybrid retrieval: keyword, vector, graph traversal, filters, fusion, reranking, GraphRAG context building |
| [`scripts/`](./scripts/) | Operational entry points: database init, seeding, reindexing, connector sync, graph schema export |
| [`services/`](./services/) | Shared services: LLM provider abstraction, signal engine, event bus, object storage |
| [`tests/`](./tests/) | `unit/`, `integration/`, `e2e/` and `evals/`, with factories and fixtures |
| [`workers/`](./workers/) | Background consumers: ingestion, enrichment, embedding, indexing, graph, forecast, report, DLQ, scheduler |

---

## Documentation

| Document | For |
| --- | --- |
| [Documentation index](./docs/README.md) | Every document, with a reading path per audience |
| [Design Doc v0.1](./docs/design-doc-v0.1.md) | The source of truth this scaffold was built from |
| [Architecture](./docs/architecture.md) | Components, the ingestion path and the investigation path |
| [Roadmap](./docs/roadmap.md) | Phases 0–7 with exit criteria and non-goals |
| [Getting started](./docs/getting-started.md) | Clone to a running local stack |
| [Local development](./docs/local-development.md) | The day-to-day workflow |
| [Folder structure](./docs/folder-structure.md) | What every directory is for |
| [Signal model](./docs/signal-model.md) | The canonical unit of data |
| [Connector spec](./docs/connector-spec.md) | The contract a new data source must satisfy |
| [Agent system](./docs/agent-system.md) | The LangGraph investigation topology |
| [Retrieval](./docs/retrieval.md) | Hybrid retrieval and GraphRAG |
| [Knowledge graph](./docs/knowledge-graph.md) | The Neo4j entity layer |
| [Data stores](./docs/data-stores.md) | What each of the six stores owns |
| [API reference](./docs/api-reference.md) | The HTTP and SSE contract |
| [Coding standards](./docs/coding-standards.md) | Rules `make check` enforces |
| [Testing strategy](./docs/testing-strategy.md) | What each test layer must prove |
| [Contributing](./docs/contributing.md) | Conventions, the pre-merge gate, when an ADR is required |
| [ADRs](./docs/adr/README.md) | Ten recorded architecture decisions |
| [Observability](./docs/observability.md) · [Security and privacy](./docs/security-and-privacy.md) · [Deployment](./docs/deployment.md) · [Glossary](./docs/glossary.md) | Operational and reference material |

---

## Tech stack

From Design Doc S4.

| Layer | Choice |
| --- | --- |
| Frontend | Next.js, React, TypeScript, TailwindCSS, shadcn/ui, Zustand, TanStack Query |
| Backend | Python 3.12, FastAPI, SQLAlchemy, Alembic, AsyncIO |
| Agents | LangGraph, LangChain (tool wrappers), MCP |
| Databases | PostgreSQL, Neo4j, Qdrant, Redis |
| Storage | Cloudflare R2 |
| Messaging | Kafka/Redpanda |
| Deployment | Docker, Kubernetes, Modal (GPU), Railway, Vercel |
| Monitoring | LangSmith, OpenTelemetry, Prometheus, Grafana |

The AI layer is model-agnostic behind `services/llm/provider.py` and defaults to Anthropic Claude.
Model ids are configuration only — `LLM_MODEL_PLANNER`, `LLM_MODEL_WORKER` and `LLM_MODEL_FAST` in
[`.env.example`](./.env.example) — never literals in application code.

Python dependencies are managed with pip via [`requirements.txt`](./requirements.txt) and
`requirements-dev.txt`; [`pyproject.toml`](./pyproject.toml) holds tool configuration only and the
repository is not installed as a package. The frontend uses npm.

---

## Local only

This repository is local. It has **no configured remote, no CI/CD and no deployment automation.**

- The [`Makefile`](./Makefile) contains no publish, push or deploy target. Every command runs on your machine.
- [`docker-compose.yml`](./docker-compose.yml) is a development stack, not a deployment artifact. Its credentials are throwaway defaults.
- [`infra/`](./infra/) holds design scaffolding for a future production topology; nothing in this repository executes it, and [`docs/deployment.md`](./docs/deployment.md) describes a target rather than a running system.
- All publishing — anything that leaves this machine — is performed manually by the repository owner. Do not add workflow files, remotes or hooks that assume otherwise.
# omnisense
