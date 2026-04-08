FROM ghcr.io/berriai/litellm:main-stable

COPY litellm_config.yaml /app/config.yaml
COPY strip_thinking.py /app/strip_thinking.py

ENV LITELLM_WORKER_STARTUP_HOOKS=strip_thinking:apply_patch
ENV PYTHONPATH=/app
ENV THINK_OUTPUT_MODE=none

EXPOSE 3999

ENTRYPOINT ["litellm", "--config", "/app/config.yaml", "--port", "3999", "--host", "0.0.0.0"]
