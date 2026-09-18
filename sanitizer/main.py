"""FastAPI application assembly for the sanitizer reverse proxy.

Route registration order is significant — FastAPI matches routes in the order
they are added, so the exact ``/v1/messages`` route must be registered before
the ``/{path:path}`` wildcard passthrough.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from . import routes_messages, routes_passthrough
from .effort import (
    known_levels,
    level_source,
    list_upstream_models,
    probe_enabled,
    probe_models,
    validate_at_startup,
)
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


@app.get("/sanitizer/effort-levels")
async def effort_levels():
    """What this proxy knows about each model's accepted effort levels, and how."""
    return {
        model: {"levels": list(levels), "source": level_source(model)}
        for model, levels in sorted(known_levels().items())
    }


# Order matters: exact /v1/messages before the /{path:path} wildcard.
app.include_router(routes_messages.router)
app.include_router(routes_passthrough.router)


@app.on_event("startup")
async def _log_active_config() -> None:
    # A typo in the declared effort vocabulary must not go live as a narrower
    # declaration; refuse to start instead (see sanitizer.effort).
    validate_at_startup()
    logger.info(
        "sanitizer started: port=%d upstream=%s bridge=%s think_mode=%s",
        get_port(),
        get_upstream_url(),
        is_openai_bridge_enabled(),
        get_think_output_mode(),
    )
    if probe_enabled():
        asyncio.create_task(_probe_effort_levels())


async def _probe_effort_levels() -> None:
    """Ask the upstream what each served model takes, before any client does.

    Runs in the background so startup never waits on LiteLLM; retries while the
    upstream is still coming up. What it learns rides ``/v1/models`` as
    ``effort_levels`` and drives the request-path clamp. A model the probe cannot
    settle stays verbatim until a real 400 teaches it (``learn_from_upstream_error``).
    """
    upstream = get_upstream_url()
    for attempt in range(1, 31):
        await asyncio.sleep(2 if attempt == 1 else 20)
        try:
            models = await list_upstream_models(upstream)
        except Exception as exc:  # noqa: BLE001 — upstream not up yet; keep trying
            logger.info("effort probe: upstream model list unavailable (attempt %d): %s", attempt, exc)
            continue
        learned = await probe_models(upstream, models)
        logger.info(
            "effort probe done: %d model(s) listed, %d learned: %s",
            len(models),
            len(learned),
            ", ".join(f"{m}={'|'.join(lv)}" for m, lv in sorted(learned.items())) or "-",
        )
        return
    logger.warning("effort probe gave up: upstream model list never became available")
