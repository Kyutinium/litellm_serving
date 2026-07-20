"""Integration tests for the wildcard /{path:path} passthrough route."""

import asyncio

import httpx

from sanitizer.main import app


def run(coro):
    return asyncio.run(coro)


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_get_path_and_query_preserved(upstream, monkeypatch):
    monkeypatch.setenv("UPSTREAM_BASE_URL", "http://backend:9000")
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json={"models": []})

    upstream.set_handler(handler)

    async def _do():
        async with _client() as client:
            return await client.get("/v1/models?limit=5")

    resp = run(_do())
    assert resp.status_code == 200
    assert captured["method"] == "GET"
    assert captured["url"] == "http://backend:9000/v1/models?limit=5"


def test_post_body_and_auth_preserved(upstream):
    captured = {}

    def handler(request):
        captured["body"] = request.content
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    upstream.set_handler(handler)

    async def _do():
        async with _client() as client:
            return await client.post(
                "/v1/chat/completions",
                content=b'{"model":"M"}',
                headers={"authorization": "Bearer secret", "content-type": "application/json"},
            )

    resp = run(_do())
    assert resp.status_code == 200
    assert captured["body"] == b'{"model":"M"}'
    assert captured["auth"] == "Bearer secret"


def test_streaming_relay(upstream):
    def handler(request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"data: a\n\ndata: b\n\n",
        )

    upstream.set_handler(handler)

    async def _do():
        async with _client() as client:
            return await client.get("/v1/stream")

    resp = run(_do())
    assert resp.status_code == 200
    assert resp.text == "data: a\n\ndata: b\n\n"


def test_upstream_error_status_preserved(upstream):
    upstream.set_handler(lambda request: httpx.Response(401, json={"error": "unauthorized"}))

    async def _do():
        async with _client() as client:
            return await client.get("/v1/models")

    resp = run(_do())
    assert resp.status_code == 401
    assert resp.json() == {"error": "unauthorized"}


def test_unreachable_upstream_returns_502(upstream):
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    upstream.set_handler(handler)

    async def _do():
        async with _client() as client:
            return await client.get("/v1/models")

    resp = run(_do())
    assert resp.status_code == 502
    assert resp.json()["error"]["type"] == "bad_gateway"


def test_v1_messages_not_caught_by_passthrough(upstream):
    # POST /v1/messages must be handled by the messages router, forwarded to
    # the /v1/messages upstream route (not treated as a generic passthrough).
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json={})

    upstream.set_handler(handler)

    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json={"model": "M", "messages": []})

    run(_do())
    assert captured["url"].endswith("/v1/messages")


def test_http_status_error_is_not_transport_error():
    # Guards the narrow `except httpx.TransportError` in the passthrough handler:
    # an upstream HTTP error status must not be swallowed into a 502.
    assert not issubclass(httpx.HTTPStatusError, httpx.TransportError)
    assert issubclass(httpx.ConnectError, httpx.TransportError)
