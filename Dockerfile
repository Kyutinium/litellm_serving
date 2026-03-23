FROM ghcr.io/berriai/litellm:main-latest

COPY litellm_config.yaml /app/config.yaml

EXPOSE 3999

ENTRYPOINT ["litellm", "--config", "/app/config.yaml", "--port", "3999", "--host", "0.0.0.0"]
