"""Unit tests for sanitizer.sanitize_events (pure dict-in / dict-out logic)."""

import asyncio

from sanitizer.sanitizer import DELTA_COMPATIBLE_BLOCKS, sanitize_events


def run(coro):
    return asyncio.run(coro)


async def _agen(items):
    for item in items:
        yield item


async def _collect(events):
    return [e async for e in sanitize_events(_agen(events))]


def sanitize(events):
    return run(_collect(events))


def assert_spec_conforming(events):
    """Assert the stream obeys Anthropic Messages SSE invariants."""
    started = 0
    open_type = None
    seen_indices = []
    for e in events:
        etype = e["type"]
        if etype == "message_start":
            started += 1
            assert open_type is None
        elif etype == "content_block_start":
            assert open_type is None, "start without preceding stop"
            open_type = e["content_block"]["type"]
            idx = e["index"]
            if seen_indices:
                assert idx == seen_indices[-1] + 1, "index not monotonic"
            else:
                assert idx == 0, "first index must be 0"
            seen_indices.append(idx)
        elif etype == "content_block_delta":
            assert open_type is not None, "delta with no open block"
            dtype = e["delta"]["type"]
            compatible = DELTA_COMPATIBLE_BLOCKS.get(dtype)
            if compatible is not None:
                assert open_type in compatible, f"{dtype} in {open_type}"
        elif etype == "content_block_stop":
            assert open_type is not None, "stop with no open block"
            open_type = None
        elif etype in ("message_delta", "message_stop"):
            assert open_type is None, "message end with block still open"
    assert started <= 1, "message_start emitted more than once"


def test_clean_passthrough():
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {}},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    texts = [e["delta"]["text"] for e in out if e["type"] == "content_block_delta"]
    assert texts == ["hi"]


def test_canonical_split_pattern():
    # text block that leaks a thinking_delta then a text_delta (LiteLLM #21128).
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "User asks..."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello!"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {}},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    starts = [(e["index"], e["content_block"]["type"]) for e in out if e["type"] == "content_block_start"]
    assert starts == [(0, "text"), (1, "thinking"), (2, "text")]
    # the original text block (index 0) is closed empty; thinking + text follow.
    thinking = [e for e in out if e["type"] == "content_block_delta" and e["delta"]["type"] == "thinking_delta"]
    assert thinking[0]["index"] == 1
    text = [e for e in out if e["type"] == "content_block_delta" and e["delta"]["type"] == "text_delta"]
    assert text[0]["index"] == 2 and text[0]["delta"]["text"] == "Hello!"


def test_duplicate_message_start_dedup():
    events = [
        {"type": "message_start", "message": {"id": "a"}},
        {"type": "message_start", "message": {"id": "b"}},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    starts = [e for e in out if e["type"] == "message_start"]
    assert len(starts) == 1 and starts[0]["message"]["id"] == "a"


def test_dangling_block_auto_close():
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}},
        {"type": "message_stop"},  # no content_block_stop
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    assert any(e["type"] == "content_block_stop" for e in out)


def test_tool_use_delta_transition():
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "let me"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"a\":1}"}},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    starts = [e["content_block"]["type"] for e in out if e["type"] == "content_block_start"]
    assert "tool_use" in starts


def test_server_tool_use_not_split():
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "server_tool_use", "id": "s1", "name": "web_search", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\"q\":1}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    starts = [e["content_block"] for e in out if e["type"] == "content_block_start"]
    # server_tool_use preserved (id/name intact, not split into a synthetic tool_use)
    assert len(starts) == 1 and starts[0]["type"] == "server_tool_use"
    assert starts[0]["id"] == "s1" and starts[0]["name"] == "web_search"


def test_empty_text_delta_in_thinking_dropped():
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "reasoning"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    # empty text_delta must be dropped, not split the thinking block.
    starts = [e["content_block"]["type"] for e in out if e["type"] == "content_block_start"]
    assert starts == ["thinking"]


def test_empty_input_json_delta_dropped():
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ""}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    # no empty tool_use synthesized from the zero-payload input_json_delta.
    starts = [e["content_block"]["type"] for e in out if e["type"] == "content_block_start"]
    assert "tool_use" not in starts


def test_explicit_start_index_remapped_after_split():
    events = [
        {"type": "message_start"},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "t"}},
        # explicit upstream start still claiming index 0
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "final"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    indices = [e["index"] for e in out if e["type"] == "content_block_start"]
    assert indices == [0, 1, 2]  # explicit start reindexed to 2


def test_content_block_stop_with_no_open_block_ignored():
    events = [
        {"type": "message_start"},
        {"type": "content_block_stop", "index": 5},  # spurious
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    assert_spec_conforming(out)
    assert not any(e["type"] == "content_block_stop" for e in out)


def test_ping_and_error_passthrough():
    events = [
        {"type": "message_start"},
        {"type": "ping"},
        {"type": "error", "error": {"type": "overloaded_error"}},
        {"type": "message_stop"},
    ]
    out = sanitize(events)
    types = [e["type"] for e in out]
    assert "ping" in types and "error" in types
