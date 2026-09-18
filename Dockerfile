# One image for both services; compose picks the process via `command`.
FROM ghcr.io/berriai/litellm:main-stable

# Sanitizer-only dependencies, kept apart from LiteLLM's own. The sanitizer
# never imports litellm/pydantic/anthropic/openai — it talks to LiteLLM over HTTP.
RUN pip install --no-cache-dir "fastapi==0.115.*" "uvicorn[standard]==0.32.*" "httpx==0.27.*"

COPY litellm_config.yaml /app/config.yaml
COPY strip_thinking.py /app/strip_thinking.py
COPY sanitizer /app/sanitizer

ENV PYTHONPATH=/app
ENV LITELLM_WORKER_STARTUP_HOOKS=strip_thinking:apply_patch

EXPOSE 3999 3996

# Default process is LiteLLM; compose overrides `command` for the sanitizer.
CMD ["litellm", "--config", "/app/config.yaml", "--port", "3999", "--host", "0.0.0.0"]
