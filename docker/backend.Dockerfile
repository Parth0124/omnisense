# syntax=docker/dockerfile:1
# =============================================================================
# OmniSense API gateway image.
# Build from the repository root:  docker build -f docker/backend.Dockerfile .
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

# ------------------------------------------------------------- dependencies --
FROM base AS deps
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

# ------------------------------------------------------------------ runtime --
FROM base AS runtime

COPY --from=deps /install /usr/local

RUN useradd --create-home --uid 1000 omnisense

COPY --chown=omnisense:omnisense . .

USER omnisense

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:8000/api/v1/health || exit 1

ENTRYPOINT ["/app/docker/entrypoints/api.sh"]
