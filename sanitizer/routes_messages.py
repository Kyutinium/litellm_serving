"""``POST /v1/messages`` — sanitize (and optionally bridge) Anthropic requests.

Parses the Anthropic request body, forwards it upstream (either verbatim to the
upstream ``/v1/messages`` route, or — in bridge mode — translated to the
upstream ``/v1/chat/completions`` route), then normalizes the response so that
spec-conforming Anthropic clients (e.g. the Claude Agent SDK) never see
LiteLLM's malformed SSE.

Never logs raw prompt/response content — only metadata (status, byte counts).
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Dict, Optional

import httpx
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from .config import (
    get_request_timeout_seconds,
    get_think_output_mode,
    get_tls_verify,
    get_upstream_url,
    is_openai_bridge_enabled,
)
from .openai_bridge import (
    anthropic_request_to_openai_body,
    openai_response_to_anthropic_body,
    openai_stream_to_anthropic_events,
)
from .output_transform import transform_events
from .sanitizer import normalize_anthropic_message_body, sanitize_events

logger = logging.getLogger("sanitizer.routes_messages")

router = APIRouter()

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


# --------------------------------------------------------------------------- #
# Header helpers
# --------------------------------------------------------------------------- #


def _clean_request_headers(headers) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in headers.items():
        lk = key.lower()
        if lk in _HOP_BY_HOP or lk in ("accept-encoding", "host", "content-length"):
            continue
        out[key] = value
    return out


def _with_json_content_type(headers: Dict[str, str]) -> Dict[str, str]:
    out = {k: v for k, v in headers.items() if k.lower() != "content-type"}
    out["content-type"] = "application/json"
    return out


def _clean_response_headers(headers) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in headers.items():
        lk = key.lower()
        if lk in _HOP_BY_HOP or lk in ("content-encoding", "content-length", "content-type"):
            continue
        out[key] = value
    return out


# --------------------------------------------------------------------------- #
# SSE helpers
# --------------------------------------------------------------------------- #


def _parse_anthropic_event(event_type: Optional[str], payload: str) -> Optional[Dict]:
    payload = payload.strip()
    if payload and payload != "[DONE]":
        try:
            obj = json.loads(payload)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            if "type" not in obj and event_type:
                obj["type"] = event_type
            return obj
    if event_type:  # type-only event, e.g. `event: ping`
        return {"type": event_type}
    return None


async def _iter_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[Dict]:
    """Parse an Anthropic SSE line stream into event dicts."""
    event_type: Optional[str] = None
    data_lines = []
    async for line in lines:
        if line == "":
            if data_lines or event_type:
                evt = _parse_anthropic_event(event_type, "\n".join(data_lines))
                if evt is not None:
                    yield evt
            event_type = None
            data_lines = []
            continue
        if line.startswith(":"):
            continue  # comment line
        if line.startswith("event:"):
            event_type = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
    if data_lines or event_type:
        evt = _parse_anthropic_event(event_type, "\n".join(data_lines))
        if evt is not None:
            yield evt


async def _iter_openai_sse_chunks(lines: AsyncIterator[str]) -> AsyncIterator[Dict]:
    """Parse an OpenAI SSE line stream into chunk dicts (data: lines only)."""
    async for line in lines:
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def _format_sse(event: Dict) -> str:
    etype = event.get("type", "message")
    data = json.dumps(event, ensure_ascii=False)
    return f"event: {etype}\ndata: {data}\n\n"


# --------------------------------------------------------------------------- #
# httpx client (monkeypatched in tests)
# --------------------------------------------------------------------------- #


def _make_client(timeout) -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=get_tls_verify(), timeout=timeout)


# --------------------------------------------------------------------------- #
# Route handler
# --------------------------------------------------------------------------- #


@router.post("/v1/messages")
async def sanitize_messages(request: Request):
    raw = await request.body()
    headers = _clean_request_headers(request.headers)

    try:
        body = json.loads(raw) if raw else {}
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}

    is_stream = bool(body.get("stream"))
    use_bridge = is_openai_bridge_enabled() and bool(body)
    think_mode = get_think_output_mode()
    timeout = get_request_timeout_seconds()
    upstream = get_upstream_url()

    if use_bridge:
        target_url = f"{upstream}/v1/chat/completions"
        content = json.dumps(anthropic_request_to_openai_body(body)).encode()
        headers = _with_json_content_type(headers)
    else:
        target_url = f"{upstream}/v1/messages"
        content = raw

    if is_stream:
        return await _handle_streaming(
            target_url, content, headers, timeout, use_bridge, think_mode, body
        )
    return await _handle_non_streaming(target_url, content, headers, timeout, use_bridge)


async def _handle_non_streaming(
    url: str, content: bytes, headers: Dict[str, str], timeout, use_bridge: bool
):
    client = _make_client(timeout)
    try:
        request = client.build_request("POST", url, content=content, headers=headers)
        resp = await client.send(request)
        body_bytes = resp.content
        resp_headers = _clean_response_headers(resp.headers)
        media_type = resp.headers.get("content-type")

        if resp.status_code >= 400:
            # Metadata only — never the prompt/response body.
            logger.warning(
                "upstream error status=%d content_type=%r bytes=%d",
                resp.status_code,
                media_type,
                len(body_bytes),
            )

        if 200 <= resp.status_code < 300 and "application/json" in (media_type or ""):
            try:
                parsed_body = json.loads(body_bytes)
            except ValueError:
                parsed_body = None

            if use_bridge and isinstance(parsed_body, dict):
                anthropic_body = openai_response_to_anthropic_body(parsed_body)
                return Response(
                    content=json.dumps(anthropic_body).encode(),
                    status_code=resp.status_code,
                    headers=resp_headers,
                    media_type="application/json",
                )

            if not use_bridge and isinstance(parsed_body, dict):
                normalized_body = normalize_anthropic_message_body(parsed_body)
                if normalized_body is not parsed_body:
                    return Response(
                        content=json.dumps(normalized_body).encode(),
                        status_code=resp.status_code,
                        headers=resp_headers,
                        media_type="application/json",
                    )

        return Response(
            content=body_bytes,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=media_type,
        )
    finally:
        await client.aclose()


async def _handle_streaming(
    url: str,
    content: bytes,
    headers: Dict[str, str],
    timeout,
    use_bridge: bool,
    think_mode: str,
    body: Dict,
):
    client = _make_client(timeout)
    request = client.build_request("POST", url, content=content, headers=headers)
    resp = await client.send(request, stream=True)

    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" not in content_type:
        # Upstream returned a non-SSE payload (typically a JSON error). Relay it
        # verbatim with its status so errors are not swallowed by the SSE parser.
        error_bytes = await resp.aread()
        await resp.aclose()
        await client.aclose()
        logger.warning(
            "upstream non-SSE response status=%d content_type=%r bytes=%d",
            resp.status_code,
            content_type,
            len(error_bytes),
        )
        return Response(
            content=error_bytes,
            status_code=resp.status_code,
            headers=_clean_response_headers(resp.headers),
            media_type=content_type or None,
        )

    model = body.get("model", "")

    async def event_generator() -> AsyncIterator[str]:
        try:
            if use_bridge:
                chunks = _iter_openai_sse_chunks(resp.aiter_lines())
                events = openai_stream_to_anthropic_events(chunks, model)
            else:
                events = _iter_sse_events(resp.aiter_lines())
            events = sanitize_events(events)
            events = transform_events(events, think_mode)
            async for event in events:
                yield _format_sse(event)
        finally:
            await resp.aclose()
            await client.aclose()

    return StreamingResponse(
        event_generator(),
        status_code=resp.status_code,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
