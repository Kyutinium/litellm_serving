"""Follow-up regression coverage for PR #20 review findings."""

import asyncio

import httpx

from sanitizer.main import app
from sanitizer.sanitizer import normalize_anthropic_message_body, sanitize_events


async def _agen(items):
    for item in items:
        yield item


def _sanitize(events):
    async def collect():
        return [event async for event in sanitize_events(_agen(events))]

    return asyncio.run(collect())


def _client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _thinking_start(events):
    return next(
        event["content_block"]
        for event in events
        if event.get("type") == "content_block_start"
        and (event.get("content_block") or {}).get("type") == "thinking"
    )


def test_stream_thinking_start_fills_both_required_fields():
    out = _sanitize(
        [
            {"type": "message_start"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
    )

    assert _thinking_start(out) == {
        "type": "thinking",
        "thinking": "",
        "signature": "",
    }


def test_stream_thinking_start_normalizes_explicit_nulls():
    out = _sanitize(
        [
            {"type": "message_start"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": None,
                    "signature": None,
                },
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
    )

    assert _thinking_start(out) == {
        "type": "thinking",
        "thinking": "",
        "signature": "",
    }


def test_nonstream_body_normalizes_only_invalid_thinking_fields():
    body = {
        "id": "msg_1",
        "content": [
            {"type": "thinking"},
            {
                "type": "thinking",
                "thinking": None,
                "signature": "opaque-native-signature",
            },
            {"type": "text", "text": "answer"},
        ],
    }

    out = normalize_anthropic_message_body(body)

    assert out["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "",
    }
    assert out["content"][1] == {
        "type": "thinking",
        "thinking": "",
        "signature": "opaque-native-signature",
    }
    assert out["content"][2] == {"type": "text", "text": "answer"}
    assert body["content"][0] == {"type": "thinking"}  # no in-place mutation


def test_bridge_off_nonstream_route_normalizes_thinking(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    upstream.set_handler(
        lambda request: httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "content": [
                    {"type": "thinking", "signature": None},
                    {"type": "text", "text": "answer"},
                ],
            },
        )
    )

    async def do_request():
        async with _client() as client:
            return await client.post(
                "/v1/messages",
                json={"model": "M", "messages": [{"role": "user", "content": "q"}]},
            )

    response = asyncio.run(do_request())
    body = response.json()

    assert response.status_code == 200
    assert body["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "",
    }
    assert body["content"][1] == {"type": "text", "text": "answer"}
    assert str(upstream.requests[-1].url).endswith("/v1/messages")
