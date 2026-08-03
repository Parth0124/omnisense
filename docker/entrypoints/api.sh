#!/usr/bin/env sh
# Entrypoint for the OmniSense API container.
set -eu

: "${API_HOST:=0.0.0.0}"
: "${API_PORT:=8000}"
: "${API_WORKERS:=1}"
: "${RUN_MIGRATIONS:=false}"

if [ "$RUN_MIGRATIONS" = "true" ]; then
  echo "[entrypoint] applying database migrations"
  alembic -c migrations/alembic.ini upgrade head
fi

echo "[entrypoint] starting api on ${API_HOST}:${API_PORT} (${API_WORKERS} worker(s))"
exec uvicorn backend.main:app \
  --host "$API_HOST" \
  --port "$API_PORT" \
  --workers "$API_WORKERS" \
  --no-access-log
