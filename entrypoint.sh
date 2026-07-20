#!/usr/bin/env bash
# Run the sanitizer reverse proxy and LiteLLM side by side in one container.
# The sanitizer fronts LiteLLM: clients hit ${SANITIZER_PORT}, the sanitizer
# forwards to LiteLLM at ${UPSTREAM_BASE_URL}.
set -euo pipefail

SANITIZER_PORT="${SANITIZER_PORT:-3996}"
LITELLM_PORT="${LITELLM_PORT:-3999}"
UPSTREAM_BASE_URL="${UPSTREAM_BASE_URL:-http://localhost:${LITELLM_PORT}}"
export UPSTREAM_BASE_URL

python -m uvicorn sanitizer.main:app --host 0.0.0.0 --port "${SANITIZER_PORT}" --log-level info &
SANITIZER_PID=$!

sleep 1

litellm --config /app/config.yaml --port "${LITELLM_PORT}" --host 0.0.0.0 &
LITELLM_PID=$!

cleanup() {
    kill "${SANITIZER_PID}" "${LITELLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Exit as soon as either process dies so the container is recycled.
wait -n "${SANITIZER_PID}" "${LITELLM_PID}"
