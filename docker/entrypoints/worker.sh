#!/usr/bin/env sh
# Entrypoint for OmniSense worker containers.
# Select the worker with WORKER_MODULE, e.g. workers.graph_worker.
set -eu

: "${WORKER_MODULE:=workers.enrichment_worker}"

echo "[entrypoint] starting worker: ${WORKER_MODULE}"
exec python -m "$WORKER_MODULE"
