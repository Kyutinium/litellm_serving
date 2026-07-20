"""FastAPI application assembly for the sanitizer reverse proxy.

Route registration order is significant — FastAPI matches routes in the order
they are added, so the exact ``/v1/messages`` route must be registered before
the ``/{path:path}`` wildcard passthrough.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from . import routes_messages, routes_passthrough
from .config import (
    get_port,
    get_think_output_mode,
    get_upstream_url,
    is_openai_bridge_enabled,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sanitizer.main")

app = FastAPI(title="LiteLLM Sanitizer", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


# Order matters: exact /v1/messages before the /{path:path} wildcard.
app.include_router(routes_messages.router)
app.include_router(routes_passthrough.router)


@app.on_event("startup")
async def _log_active_config() -> None:
    logger.info(
        "sanitizer started: port=%d upstream=%s bridge=%s think_mode=%s",
        get_port(),
        get_upstream_url(),
        is_openai_bridge_enabled(),
        get_think_output_mode(),
    )
