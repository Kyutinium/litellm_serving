"""Wildcard ``/{path:path}`` — byte-for-byte passthrough to upstream.

Everything that is not ``POST /v1/messages`` (``/v1/models``, direct
``/v1/chat/completions``, ``/v1/embeddings``, …) is relayed to the upstream
without parsing or transforming the body: method, headers (minus hop-by-hop),
query string, status, and streaming chunks are all preserved.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import get_request_timeout_seconds, get_tls_verify, get_upstream_url
from .routes_messages import _clean_request_headers, _clean_response_headers

logger = logging.getLogger("sanitizer.routes_passthrough")

router = APIRouter()

_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def _make_client(timeout) -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=get_tls_verify(), timeout=timeout)


@router.api_route("/{path:path}", methods=_METHODS)
async def passthrough(path: str, request: Request):
    body = await request.body()
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

    resp_headers = _clean_response_headers(resp.headers)
    media_type = resp.headers.get("content-type")

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
