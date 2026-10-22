"""Unique tool_use ids across a conversation (Kimi repeats ``functions.<name>:0``).

The Claude Code CLI drops a later tool_use/tool_result pair whose id already
appears in the history (pinned against CLI 2.1.283), so a repeated id made the
model loop on the same call, and ChatDRAGON folded the running call into the
first, finished card. The sanitizer replaces a repeated (or missing) id.
"""

import asyncio
import json

import httpx

from sanitizer.main import app
from sanitizer.sanitizer import sanitize_events
from sanitizer.tool_ids import ToolIdAllocator, dedupe_body_tool_ids, history_tool_ids

KIMI = "functions.mcp__probe__slow:0"


def run(coro):
    return asyncio.run(coro)


def _history(*ids):
    """An Anthropic request whose history already used *ids*."""
    messages = [{"role": "user", "content": "go"}]
    for tool_id in ids:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": tool_id, "name": "mcp__probe__slow", "input": {}}],
            }
        )
        messages.append(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "r"}]}
        )
    return messages


# --------------------------------------------------------------------------- #
# Allocator
# --------------------------------------------------------------------------- #


def test_history_ids_cover_tool_use_and_tool_result():
    messages = _history("a") + [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "orphan", "content": ""}]},
        {"role": "user", "content": "plain"},
    ]
    assert history_tool_ids({"messages": messages}) == {"a", "orphan"}


def test_a_new_id_passes_through():
    alloc = ToolIdAllocator({"functions.x:0"})
    assert alloc.assign("call_abc", "x") == "call_abc"
    assert alloc.rewritten == 0


def test_a_repeated_kimi_id_gets_the_next_global_index():
    alloc = ToolIdAllocator({KIMI, "functions.other:3"})
    assert alloc.assign(KIMI, "mcp__probe__slow") == "functions.mcp__probe__slow:4"
    # The replacement is now taken too: a second repeat moves on.
    assert alloc.assign(KIMI, "mcp__probe__slow") == "functions.mcp__probe__slow:5"
    assert alloc.rewritten == 2


def test_a_repeated_foreign_id_or_a_missing_one_gets_a_toolu_id():
    alloc = ToolIdAllocator({"call_1"})
    repeated = alloc.assign("call_1", "x")
    missing = alloc.assign("", "x")
    none = alloc.assign(None, "x")
    for new in (repeated, missing, none):
        assert new.startswith("toolu_")
    assert len({repeated, missing, none}) == 3


def test_ids_within_one_response_stay_unique():
    alloc = ToolIdAllocator()
    assert alloc.assign(KIMI) == KIMI
    assert alloc.assign(KIMI) != KIMI


# --------------------------------------------------------------------------- #
# Streaming rewrite (sanitize_events)
# --------------------------------------------------------------------------- #


async def _agen(items):
    for item in items:
        yield item


def _tool_stream(tool_id):
    return [
        {"type": "message_start", "message": {"id": "m", "content": []}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": tool_id, "name": "mcp__probe__slow", "input": {}},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]


def _started_ids(events):
    return [e["content_block"].get("id") for e in events if e["type"] == "content_block_start"]


def test_sanitize_events_replaces_a_repeated_id():
    alloc = ToolIdAllocator({KIMI})
    out = run(_collect(sanitize_events(_agen(_tool_stream(KIMI)), alloc)))
    assert _started_ids(out) == ["functions.mcp__probe__slow:1"]


def test_sanitize_events_without_allocator_is_unchanged():
    out = run(_collect(sanitize_events(_agen(_tool_stream(KIMI)))))
    assert _started_ids(out) == [KIMI]


async def _collect(agen):
    return [e async for e in agen]


# --------------------------------------------------------------------------- #
# Non-streaming rewrite
# --------------------------------------------------------------------------- #


def test_dedupe_body_rewrites_only_tool_use_blocks():
    body = {
        "content": [
            {"type": "text", "text": "t"},
            {"type": "tool_use", "id": KIMI, "name": "mcp__probe__slow", "input": {}},
        ]
    }
    out = dedupe_body_tool_ids(body, ToolIdAllocator({KIMI}))
    assert out is not body
    assert out["content"][0] == body["content"][0]
    assert out["content"][1]["id"] == "functions.mcp__probe__slow:1"
    assert dedupe_body_tool_ids(body, ToolIdAllocator()) is body


# --------------------------------------------------------------------------- #
# Routes: both modes, both shapes
# --------------------------------------------------------------------------- #


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _post(body):
    async def _do():
        async with _client() as client:
            return await client.post("/v1/messages", json=body)

    return run(_do())


def _events(text):
    return [
        json.loads(line[len("data:") :].strip())
        for block in text.strip().split("\n\n")
        for line in block.splitlines()
        if line.startswith("data:")
    ]


_KIMI_OPENAI_SSE = (
    'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"functions.mcp__probe__slow:0",'
    '"type":"function","function":{"name":"mcp__probe__slow","arguments":""}}]}}]}\n\n'
    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]}}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    "data: [DONE]\n\n"
).encode()


def test_bridge_stream_replaces_an_id_the_history_already_has(upstream, monkeypatch):
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")
    monkeypatch.setenv("THINK_OUTPUT_MODE", "default")
    upstream.set_handler(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_KIMI_OPENAI_SSE)
    )
    resp = _post({"model": "kimi", "stream": True, "messages": _history(KIMI)})
    assert resp.status_code == 200
    tool_ids = [i for i in _started_ids(_events(resp.text)) if i is not None and i.startswith("functions.")]
    assert tool_ids == ["functions.mcp__probe__slow:1"]


def test_bridge_stream_keeps_a_first_call_id(upstream, monkeypatch):
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")
    monkeypatch.setenv("THINK_OUTPUT_MODE", "default")
    upstream.set_handler(
        lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_KIMI_OPENAI_SSE)
    )
    resp = _post({"model": "kimi", "stream": True, "messages": _history()})
    assert KIMI in _started_ids(_events(resp.text))


def test_passthrough_stream_replaces_a_repeated_id(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    monkeypatch.setenv("THINK_OUTPUT_MODE", "default")
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in _tool_stream(KIMI)).encode()
    upstream.set_handler(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse))
    resp = _post({"model": "kimi", "stream": True, "messages": _history(KIMI)})
    assert _started_ids(_events(resp.text)) == ["functions.mcp__probe__slow:1"]


def test_bridge_non_streaming_replaces_a_repeated_id(upstream, monkeypatch):
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")
    upstream.set_handler(
        lambda r: httpx.Response(
            200,
            json={
                "id": "x",
                "model": "kimi",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": KIMI,
                                    "type": "function",
                                    "function": {"name": "mcp__probe__slow", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )
    )
    body = _post({"model": "kimi", "messages": _history(KIMI)}).json()
    assert [b["id"] for b in body["content"] if b["type"] == "tool_use"] == ["functions.mcp__probe__slow:1"]


def test_passthrough_non_streaming_replaces_a_repeated_id(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    upstream.set_handler(
        lambda r: httpx.Response(
            200,
            json={
                "id": "msg_x",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "tool_use", "id": KIMI, "name": "mcp__probe__slow", "input": {}}],
                "stop_reason": "tool_use",
            },
        )
    )
    body = _post({"model": "kimi", "messages": _history(KIMI)}).json()
    assert body["content"][0]["id"] == "functions.mcp__probe__slow:1"


# --------------------------------------------------------------------------- #
# Server tools keep their ids (PR #34 review): their result block in the same
# response names them by tool_use_id, so only client tool_use is rewritten.
# --------------------------------------------------------------------------- #

SRV = "srvtoolu_1"


def _server_tool_stream():
    return [
        {"type": "message_start", "message": {"id": "m", "content": []}},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "server_tool_use", "id": SRV, "name": "web_search", "input": {}},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "web_search_tool_result", "tool_use_id": SRV, "content": []},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": KIMI, "name": "mcp__probe__slow", "input": {}},
        },
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]


def _history_with_server_tool():
    messages = _history(KIMI)
    messages.append(
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": SRV, "name": "web_search", "input": {}},
                {"type": "web_search_tool_result", "tool_use_id": SRV, "content": []},
            ],
        }
    )
    return messages


def _server_link(blocks):
    call = next(b for b in blocks if b["type"] == "server_tool_use")
    result = next(b for b in blocks if b["type"] == "web_search_tool_result")
    client = next(b for b in blocks if b["type"] == "tool_use")
    return call["id"], result["tool_use_id"], client["id"]


def test_history_counts_server_tool_ids_as_taken():
    assert SRV in history_tool_ids({"messages": _history_with_server_tool()})


def test_stream_keeps_a_reused_server_tool_id_and_its_result_link():
    alloc = ToolIdAllocator(history_tool_ids({"messages": _history_with_server_tool()}))
    out = run(_collect(sanitize_events(_agen(_server_tool_stream()), alloc)))
    blocks = [e["content_block"] for e in out if e["type"] == "content_block_start"]
    call_id, result_ref, client_id = _server_link(blocks)
    assert call_id == result_ref == SRV
    assert client_id == "functions.mcp__probe__slow:1"  # the Kimi dedupe still applies


def test_passthrough_stream_route_keeps_the_server_tool_link(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    monkeypatch.setenv("THINK_OUTPUT_MODE", "default")
    sse = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in _server_tool_stream()).encode()
    upstream.set_handler(lambda r: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse))
    resp = _post({"model": "m", "stream": True, "messages": _history_with_server_tool()})
    blocks = [e["content_block"] for e in _events(resp.text) if e["type"] == "content_block_start"]
    call_id, result_ref, client_id = _server_link(blocks)
    assert call_id == result_ref == SRV
    assert client_id == "functions.mcp__probe__slow:1"


def test_non_streaming_keeps_a_reused_server_tool_id_and_its_result_link(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_USE_OPENAI_BRIDGE", raising=False)
    blocks = [e["content_block"] for e in _server_tool_stream() if e["type"] == "content_block_start"]
    upstream.set_handler(
        lambda r: httpx.Response(
            200,
            json={"id": "msg_x", "type": "message", "role": "assistant", "content": blocks, "stop_reason": "tool_use"},
        )
    )
    body = _post({"model": "m", "messages": _history_with_server_tool()}).json()
    call_id, result_ref, client_id = _server_link(body["content"])
    assert call_id == result_ref == SRV
    assert client_id == "functions.mcp__probe__slow:1"
