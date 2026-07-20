"""Integration tests for POST /v1/messages (sanitize + bridge)."""

import asyncio
import json

import httpx

from sanitizer.main import app
from sanitizer.routes_messages import _clean_response_headers


def run(coro):
    return asyncio.run(coro)


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _sse(*events):
    out = []
    for e in events:
        out.append(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n")
    return "".join(out).encode()


def _parse_events(text):
    events = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


# --------------------------------------------------------------------------- #
# Passthrough mode (bridge disabled)
# --------------------------------------------------------------------------- #


def test_streaming_rewrite_passthrough(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    monkeypatch.setenv("THINK_OUTPUT_MODE", "default")

    broken = _sse(
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {}},
        {"type": "message_stop"},
    )
    upstream.set_handler(
        lambda request: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=broken)
    )

    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json={"model": "M", "stream": True, "messages": []})

    resp = run(_do())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_events(resp.text)
    starts = [(e["index"], e["content_block"]["type"]) for e in events if e["type"] == "content_block_start"]
    assert starts == [(0, "text"), (1, "thinking"), (2, "text")]
    assert str(upstream.requests[-1].url).endswith("/v1/messages")


def test_non_streaming_passthrough(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    upstream.set_handler(
        lambda request: httpx.Response(200, json={"id": "msg_1", "content": []})
    )

    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json={"model": "M", "messages": []})

    resp = run(_do())
    assert resp.status_code == 200
    assert resp.json() == {"id": "msg_1", "content": []}
    assert str(upstream.requests[-1].url).endswith("/v1/messages")


def test_upstream_base_url_respected(upstream, monkeypatch):
    monkeypatch.setenv("UPSTREAM_BASE_URL", "http://backend:4321")
    upstream.set_handler(lambda request: httpx.Response(200, json={}))

    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json={"model": "M", "messages": []})

    run(_do())
    assert str(upstream.requests[-1].url) == "http://backend:4321/v1/messages"


def test_streaming_json_error_preserves_status(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    # streaming request, but upstream answers with a JSON error (not SSE)
    upstream.set_handler(
        lambda request: httpx.Response(429, json={"error": "rate_limited"})
    )

    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json={"model": "M", "stream": True, "messages": []})

    resp = run(_do())
    assert resp.status_code == 429
    assert resp.json() == {"error": "rate_limited"}


def test_client_accept_encoding_not_forwarded(upstream, monkeypatch):
    # The client's accept-encoding must not be relayed verbatim (httpx adds its
    # own, which it can transparently decode); host/content-length are dropped so
    # httpx recomputes them for the upstream request.
    from sanitizer.routes_messages import _clean_request_headers

    cleaned = _clean_request_headers(
        httpx.Headers(
            {
                "accept-encoding": "gzip, br",
                "host": "client",
                "content-length": "99",
                "connection": "keep-alive",
                "authorization": "Bearer x",
            }
        )
    )
    lower = {k.lower() for k in cleaned}
    assert "accept-encoding" not in lower
    assert "host" not in lower
    assert "content-length" not in lower
    assert "connection" not in lower
    assert "authorization" in lower  # auth is preserved

    captured = {}

    def handler(request):
        captured["ae"] = request.headers.get("accept-encoding")
        return httpx.Response(200, json={})

    upstream.set_handler(handler)

    async def _do():
        async with _client() as client:
            return await client.post(
                "/v1/messages",
                json={"model": "M", "messages": []},
                headers={"accept-encoding": "identity-client-value"},
            )

    run(_do())
    # upstream never sees the client's literal accept-encoding value
    assert captured["ae"] != "identity-client-value"


def test_error_log_omits_body_content(upstream, monkeypatch, caplog):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    secret = "SUPER_SECRET_PROMPT_TEXT"
    upstream.set_handler(
        lambda request: httpx.Response(500, json={"error": secret})
    )

    async def _do():
        async with _client() as client:
            return await client.post(
                "/v1/messages", json={"model": "M", "stream": True, "messages": [{"role": "user", "content": secret}]}
            )

    with caplog.at_level("WARNING"):
        resp = run(_do())
    assert resp.status_code == 500
    assert secret not in caplog.text  # neither prompt nor response body logged


# --------------------------------------------------------------------------- #
# Bridge mode
# --------------------------------------------------------------------------- #


def test_non_streaming_error_logs_metadata_not_body(upstream, monkeypatch, caplog):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    secret = "SECRET_RESPONSE_BODY"
    upstream.set_handler(lambda request: httpx.Response(422, json={"error": secret}))

    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json={"model": "M", "messages": []})

    with caplog.at_level("WARNING"):
        resp = run(_do())
    assert resp.status_code == 422
    assert "status=422" in caplog.text  # metadata logged
    assert secret not in caplog.text  # body never logged


def test_bridge_streaming_full(upstream, monkeypatch):
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")
    monkeypatch.setenv("THINK_OUTPUT_MODE", "default")
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        openai_sse = (
            'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=openai_sse)

    upstream.set_handler(handler)

    async def _do():
        async with _client() as client:
            return await client.post(
                "/v1/messages",
                json={"model": "M", "stream": True, "system": "sys", "messages": [{"role": "user", "content": "hi"}]},
            )

    resp = run(_do())
    assert resp.status_code == 200
    # request routed and converted to the OpenAI chat route
    assert captured["url"].endswith("/v1/chat/completions")
    assert captured["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert captured["body"]["stream_options"] == {"include_usage": True}
    # response reverse-converted to Anthropic SSE
    events = _parse_events(resp.text)
    types = [e["type"] for e in events]
    assert types[0] == "message_start" and types[-1] == "message_stop"
    starts = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
    assert starts == ["thinking", "text"]
    mdelta = [e for e in events if e["type"] == "message_delta"][0]
    assert mdelta["usage"] == {"input_tokens": 5, "output_tokens": 2}


def test_bridge_non_streaming(upstream, monkeypatch):
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")

    def handler(request):
        assert str(request.url).endswith("/v1/chat/completions")
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-x",
                "model": "M",
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    upstream.set_handler(handler)

    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json={"model": "M", "messages": [{"role": "user", "content": "q"}]})

    resp = run(_do())
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "message"
    assert body["content"] == [{"type": "text", "text": "answer"}]
    assert body["id"].startswith("msg_")


def test_content_encoding_stripped_from_response_headers():
    headers = httpx.Headers(
        {"content-encoding": "gzip", "content-length": "10", "connection": "keep-alive", "x-keep": "1"}
    )
    cleaned = _clean_response_headers(headers)
    lower = {k.lower() for k in cleaned}
    assert "content-encoding" not in lower
    assert "content-length" not in lower
    assert "connection" not in lower
    assert "x-keep" in lower
