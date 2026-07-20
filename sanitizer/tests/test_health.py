"""Route-priority / liveness tests for the assembled app."""

import asyncio

import httpx

from sanitizer.main import app


def run(coro):
    return asyncio.run(coro)


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def test_health_ok():
    async def _do():
        async with _client() as client:
            return await client.get("/health")

    resp = run(_do())
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_post_v1_messages_hits_messages_router(upstream):
    upstream.set_handler(lambda request: httpx.Response(200, json={"ok": True}))

    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json={"model": "M", "messages": []})

    resp = run(_do())
    assert resp.status_code == 200
    # bridge disabled by default → forwarded to upstream /v1/messages
    assert str(upstream.requests[-1].url).endswith("/v1/messages")


def test_v1_models_hits_passthrough_router(upstream):
    upstream.set_handler(lambda request: httpx.Response(200, json={"data": []}))

    async def _do():
        async with _client() as client:
            return await client.get("/v1/models")

    resp = run(_do())
    assert resp.status_code == 200
    assert str(upstream.requests[-1].url).endswith("/v1/models")


def test_get_v1_messages_falls_through_to_passthrough(upstream):
    # messages router only registers POST /v1/messages; GET must hit passthrough.
    upstream.set_handler(lambda request: httpx.Response(200, json={"via": "passthrough"}))

    async def _do():
        async with _client() as client:
            return await client.get("/v1/messages")

    resp = run(_do())
    assert resp.status_code == 200
    assert resp.json() == {"via": "passthrough"}
