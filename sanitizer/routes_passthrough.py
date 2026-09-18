"""Wildcard ``/{path:path}`` — byte-for-byte passthrough to upstream.

Everything that is not ``POST /v1/messages`` (``/v1/models``, direct
``/v1/chat/completions``, ``/v1/embeddings``, …) is relayed to the upstream
without parsing or transforming the body: method, headers (minus hop-by-hop),
query string, status, and streaming chunks are all preserved.

**One documented exception**, and only while ``SANITIZER_EFFORT_VOCABULARY``
declares what a model's upstream accepts: a chat-completions POST for such a
model whose ``reasoning_effort`` is outside that set has that one field clamped
(:mod:`sanitizer.effort`). Without it the upstream answers 400 and the caller —
which speaks the OpenAI-standard vocabulary and not the served model's — has no
way to know why (issue #26). Everything about the relay is unchanged: any other
body, an unparseable body, a body already using a supported level, and an
unset switch all send the original bytes.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .config import get_request_timeout_seconds, get_tls_verify, get_upstream_url
from .effort import clamp_to_supported, enrich_models_payload, known_levels, learn_from_upstream_error
from .routes_messages import _clean_request_headers, _clean_response_headers

logger = logging.getLogger("sanitizer.routes_passthrough")

router = APIRouter()

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def _make_client(timeout) -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=get_tls_verify(), timeout=timeout)


def _is_chat_completions(path: str) -> bool:
    """Whether *path* is a chat-completions endpoint (with or without a prefix)."""
    return path.strip("/").endswith("chat/completions")


def _is_models_list(path: str) -> bool:
    return path.strip("/").endswith("models")


def _retry_body_after_learning(body: bytes, error_bytes: bytes) -> Optional[bytes]:
    """Learn from a chat-completions 400 and return the body to retry with, or ``None``."""
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    model = parsed.get("model")
    requested = parsed.get("reasoning_effort")
    if not isinstance(model, str) or not isinstance(requested, str):
        return None
    if not learn_from_upstream_error(model, 400, error_bytes):
        return None
    chosen = clamp_to_supported(requested.strip().lower(), model)
    if chosen is None or chosen == requested:
        return None
    parsed["reasoning_effort"] = chosen
    try:
        return json.dumps(parsed).encode()
    except (TypeError, ValueError):
        return None


def _clamped_body(path: str, method: str, body: bytes) -> Optional[bytes]:
    """Re-encode *body* with a supported ``reasoning_effort``, or ``None``.

    ``None`` means "relay the original bytes" and is the answer for everything
    the exception does not cover: a non-chat-completions path, a non-POST, no
    configured support set, an unparseable or non-object body, no string
    ``reasoning_effort``, and a level that already needs no change. Re-encoding
    is deliberately confined to that one field's value.
    """
    if method != "POST" or not _is_chat_completions(path) or not known_levels():
        return None
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    requested = parsed.get("reasoning_effort")
    if not isinstance(requested, str):
        return None
    model = parsed.get("model")
    chosen = clamp_to_supported(
        requested.strip().lower(), model if isinstance(model, str) else None
    )
    if chosen is None or chosen == requested:
        return None
    parsed["reasoning_effort"] = chosen
    try:
        return json.dumps(parsed).encode()
    except (TypeError, ValueError):
        # A body we cannot round-trip is relayed untouched rather than dropped.
        logger.warning("cannot re-encode body to clamp reasoning_effort; relaying as-is")
        return None


@router.api_route("/{path:path}", methods=_METHODS)
async def passthrough(path: str, request: Request):
    body = await request.body()
    clamped = _clamped_body(path, request.method, body)
    if clamped is not None:
        # ``_clean_request_headers`` already drops content-length, so httpx
        # recomputes it for the re-encoded body rather than truncating it.
        body = clamped
    headers = _clean_request_headers(request.headers)
    timeout = get_request_timeout_seconds()

    url = f"{get_upstream_url()}/{path}"
    query = request.url.query
    if query:
        url = f"{url}?{query}"

    client = _make_client(timeout)
    request_obj = client.build_request(request.method, url, content=body, headers=headers)
    try:
        resp = await client.send(request_obj, stream=True)
    except httpx.TransportError as exc:
        # Narrowed to TransportError on purpose: an upstream HTTP error status is
        # not a transport failure and must surface with its real status, not 502.
        await client.aclose()
        logger.warning("passthrough transport error for %s: %s", path, exc)
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "bad_gateway", "message": str(exc)}},
        )

    # The upstream teaches its effort vocabulary in the 400 it answers an
    # unsupported level with (issue #26). On the relay path too: learn, re-encode
    # that one field, and retry once — the caller speaks the OpenAI ladder, not
    # the served template's. Anything the learning does not cover is relayed as
    # the upstream sent it.
    if resp.status_code == 400 and request.method == "POST" and _is_chat_completions(path):
        error_bytes = await resp.aread()
        await resp.aclose()
        await client.aclose()
        retried = _retry_body_after_learning(body, error_bytes)
        if retried is not None:
            client = _make_client(timeout)
            request_obj = client.build_request(request.method, url, content=retried, headers=headers)
            try:
                resp = await client.send(request_obj, stream=True)
            except httpx.TransportError as exc:
                await client.aclose()
                logger.warning("passthrough retry transport error for %s: %s", path, exc)
                return JSONResponse(status_code=502, content={"error": {"type": "bad_gateway", "message": str(exc)}})
        else:
            return Response(
                content=error_bytes,
                status_code=400,
                headers=_clean_response_headers(resp.headers),
                media_type=resp.headers.get("content-type"),
            )

    resp_headers = _clean_response_headers(resp.headers)
    media_type = resp.headers.get("content-type")

    # ``/v1/models``: publish what this proxy knows about each model's effort
    # levels as ``effort_levels`` on its row, so the gateway in front (discovery)
    # advertises the same set without an operator copying it. Byte-for-byte when
    # nothing is known or the payload is not a JSON model list.
    if request.method == "GET" and _is_models_list(path) and resp.status_code == 200 and known_levels():
        raw = await resp.aread()
        await resp.aclose()
        await client.aclose()
        try:
            payload = json.loads(raw)
        except ValueError:
            return Response(content=raw, status_code=200, headers=resp_headers, media_type=media_type)
        enriched = enrich_models_payload(payload)
        return Response(content=json.dumps(enriched).encode(), status_code=200, headers=resp_headers, media_type="application/json")

    async def relay():
        try:
            async for chunk in resp.aiter_bytes():
                if chunk:  # drop empty byte chunks; strict SSE parsers choke on them
                    yield chunk
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        relay(),
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=media_type,
    )
