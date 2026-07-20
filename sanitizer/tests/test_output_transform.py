"""Unit tests for output_transform.transform_events (THINK_OUTPUT_MODE)."""

import asyncio

from sanitizer.output_transform import transform_events


def run(coro):
    return asyncio.run(coro)


async def _agen(items):
    for item in items:
        yield item


def transform(events, mode):
    async def _collect():
        return [e async for e in transform_events(_agen(events), mode)]

    return run(_collect())


THINKING_STREAM = [
    {"type": "message_start"},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "reasoning"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_stop"},
]


def test_default_passthrough():
    out = transform(THINKING_STREAM, "default")
    assert out == THINKING_STREAM
    # thinking blocks stay thinking
    assert any(e["type"] == "content_block_start" and e["content_block"]["type"] == "thinking" for e in out)


def test_bridge_passthrough():
    out = transform(THINKING_STREAM, "bridge")
    assert out == THINKING_STREAM


def test_none_mode_promotes_thinking_to_text():
    out = transform(THINKING_STREAM, "none")
    start = [e for e in out if e["type"] == "content_block_start"][0]
    assert start["content_block"]["type"] == "text"  # rewritten
    deltas = [e["delta"] for e in out if e["type"] == "content_block_delta"]
    assert deltas[0] == {"type": "text_delta", "text": "reasoning"}
    assert deltas[1] == {"type": "text_delta", "text": ""}  # signature suppressed
    assert all(d["type"] == "text_delta" for d in deltas)


def test_text_mode_same_as_none():
    assert transform(THINKING_STREAM, "text") == transform(THINKING_STREAM, "none")


def test_think_tag_wraps_thinking():
    out = transform(THINKING_STREAM, "think_tag")
    start = [e for e in out if e["type"] == "content_block_start"][0]
    assert start["content_block"]["type"] == "text"
    texts = [e["delta"]["text"] for e in out if e["type"] == "content_block_delta"]
    joined = "".join(texts)
    assert joined.startswith("◰\n")
    assert "reasoning" in joined
    assert joined.endswith("◱\n\n")


def test_think_tag_closes_on_first_text_delta():
    stream = [
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "r"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "answer"}},
        {"type": "content_block_stop", "index": 1},
    ]
    out = transform(stream, "think_tag")
    texts = [e["delta"]["text"] for e in out if e["type"] == "content_block_delta"]
    joined = "".join(texts)
    assert "◰\n" in joined and "◱\n\n" in joined and joined.endswith("answer")
