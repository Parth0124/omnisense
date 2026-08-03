# =============================================================================
# OmniSense - local development commands
#
# Everything here runs LOCALLY. No target in this file deploys, publishes or
# pushes anything to a remote environment.
# =============================================================================

SHELL := /bin/bash
PY    := python3
VENV  := .venv
BIN   := $(VENV)/bin
COMPOSE := docker compose

.DEFAULT_GOAL := help

# ------------------------------------------------------------------- Meta ----

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ Setup ----

.PHONY: venv
venv: ## Create the Python virtual environment
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

.PHONY: install
install: venv ## Install runtime + dev Python dependencies
	$(BIN)/pip install -r requirements.txt -r requirements-dev.txt

.PHONY: install-frontend
install-frontend: ## Install frontend dependencies
	cd frontend && npm install

.PHONY: bootstrap
bootstrap: install install-frontend env ## One-shot local setup
	@echo "Bootstrap complete. Next: make up && make init-db"

.PHONY: env
env: ## Create .env from .env.example if missing
	@test -f .env || (cp .env.example .env && echo "Created .env - fill in your secrets")
	@test -f frontend/.env.local || (cp frontend/.env.local.example frontend/.env.local && echo "Created frontend/.env.local")

# -------------------------------------------------------- Local infrastructure

.PHONY: up
up: ## Start local datastores (Postgres, Neo4j, Qdrant, Redis, OpenSearch, Redpanda)
	$(COMPOSE) up -d

.PHONY: down
down: ## Stop local datastores
	$(COMPOSE) down

.PHONY: nuke
nuke: ## Stop local datastores AND delete their volumes (destructive, local only)
	$(COMPOSE) down -v

.PHONY: logs
logs: ## Tail datastore logs
	$(COMPOSE) logs -f

.PHONY: ps
ps: ## Show local service status
	$(COMPOSE) ps

.PHONY: init-db
init-db: ## Create schemas, indexes, Qdrant collections and Neo4j constraints
	$(BIN)/python scripts/init_databases.py

.PHONY: seed
seed: ## Load sample data for local development
	$(BIN)/python scripts/seed_data.py

# ------------------------------------------------------------------- Run ------

.PHONY: api
api: ## Run the FastAPI gateway with reload
	$(BIN)/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: worker
worker: ## Run the enrichment worker
	$(BIN)/python -m workers.enrichment_worker

.PHONY: scheduler
scheduler: ## Run the connector sync scheduler
	$(BIN)/python -m workers.scheduler

.PHONY: frontend
frontend: ## Run the Next.js dev server
	cd frontend && npm run dev

# --------------------------------------------------------------- Migrations ---

.PHONY: migrate
migrate: ## Apply PostgreSQL migrations
	$(BIN)/alembic -c migrations/alembic.ini upgrade head

.PHONY: migration
migration: ## Create a migration: make migration m="add signals table"
	$(BIN)/alembic -c migrations/alembic.ini revision --autogenerate -m "$(m)"

.PHONY: downgrade
downgrade: ## Roll back one migration
	$(BIN)/alembic -c migrations/alembic.ini downgrade -1

# ------------------------------------------------------------------- Quality --

.PHONY: lint
lint: ## Lint Python and frontend
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	cd frontend && npm run lint

.PHONY: format
format: ## Auto-format Python and frontend
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .
	cd frontend && npm run format

.PHONY: typecheck
typecheck: ## Type-check Python and frontend
	$(BIN)/mypy .
	cd frontend && npm run typecheck

.PHONY: test
test: ## Run unit tests
	$(BIN)/pytest -m unit

.PHONY: test-integration
test-integration: ## Run integration tests (requires `make up`)
	$(BIN)/pytest -m integration

.PHONY: test-all
test-all: ## Run the whole suite with coverage
	$(BIN)/pytest --cov --cov-report=term-missing --cov-report=html

.PHONY: eval
eval: ## Run agent quality evaluations (non-blocking)
	$(BIN)/pytest -m eval

.PHONY: check
check: lint typecheck test ## Everything CI would run, locally

# ------------------------------------------------------------------- Utils ----

.PHONY: lock
lock: ## Freeze current dependency versions
	$(BIN)/pip freeze > requirements.lock.txt

.PHONY: tree
tree: ## Print the project structure
	@command -v tree >/dev/null 2>&1 \
		&& tree -a -I '.git|node_modules|.venv|__pycache__|.next|.ruff_cache|.mypy_cache|.pytest_cache' -L 3 \
		|| find . -maxdepth 3 -type d -not -path '*/.*' -not -path '*/node_modules*'

.PHONY: clean
clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf frontend/.next frontend/out
