FROM ghcr.io/berriai/litellm:main-stable

# LiteLLM (:3999) with the sanitizer in front of it (:3996) — the same topology as
# Dockerfile.dev. The sanitizer is what learns each served model's accepted
# reasoning-effort levels (startup probe + the model's own 400s) and publishes
# them on the relayed GET /v1/models as ``effort_levels``; oh-my-gateway reads
# that through model discovery and ChatDRAGON offers exactly those levels. A
# client that talks to :3999 directly gets neither the levels nor the clamp, so
# point ANTHROPIC_BASE_URL at :3996. The sanitizer imports only fastapi, httpx
# and uvicorn, all of which the LiteLLM image already ships.
COPY litellm_config.yaml /app/config.yaml
COPY strip_thinking.py /app/strip_thinking.py
COPY sanitizer /app/sanitizer
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV LITELLM_WORKER_STARTUP_HOOKS=strip_thinking:apply_patch
ENV PYTHONPATH=/app
ENV THINK_OUTPUT_MODE=none
ENV LITELLM_PORT=3999
ENV SANITIZER_PORT=3996
ENV UPSTREAM_BASE_URL=http://localhost:3999

EXPOSE 3999 3996

ENTRYPOINT ["/app/entrypoint.sh"]
