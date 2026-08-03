# syntax=docker/dockerfile:1
# =============================================================================
# OmniSense background worker image (ingestion, enrichment, indexing, graph).
# The worker to run is selected with the WORKER_MODULE environment variable.
# Build from the repository root:  docker build -f docker/worker.Dockerfile .
# =============================================================================

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl libpq-dev \
    && rm -rf /var/lib/apt/lists/*

FROM base AS deps
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

FROM base AS runtime

COPY --from=deps /install /usr/local

RUN useradd --create-home --uid 1000 omnisense

COPY --chown=omnisense:omnisense . .

USER omnisense

ENV WORKER_MODULE=workers.enrichment_worker

ENTRYPOINT ["/app/docker/entrypoints/worker.sh"]
