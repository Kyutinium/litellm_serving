"""Regression coverage for issue #19 thinking-signature compatibility."""

import asyncio
import json

from sanitizer.openai_bridge import (
    anthropic_request_to_openai_body,
    openai_response_to_anthropic_body,
    openai_stream_to_anthropic_events,
)
from sanitizer.sanitizer import sanitize_events


async def _agen(items):
    for item in items:
        yield item


def _collect_bridge(chunks):
    async def collect():
        return [
            event
            async for event in openai_stream_to_anthropic_events(
                _agen(chunks), "M"
            )
        ]

    return asyncio.run(collect())


def _sanitize(events):
    async def collect():
        return [event async for event in sanitize_events(_agen(events))]

    return asyncio.run(collect())


def _thinking_starts(events):
    return [
        event["content_block"]
        for event in events
        if event.get("type") == "content_block_start"
        and (event.get("content_block") or {}).get("type") == "thinking"
    ]


def test_bridge_stream_thinking_start_has_signature_placeholder():
    out = _collect_bridge(
        [
            {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
    )

    starts = _thinking_starts(out)
    assert starts == [{"type": "thinking", "thinking": "", "signature": ""}]


def test_bridge_nonstream_thinking_block_has_signature_placeholder():
    out = openai_response_to_anthropic_body(
        {
            "model": "M",
            "choices": [
                {
                    "message": {
                        "reasoning_content": "thinking...",
                        "content": "answer",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )

    assert out["content"][0] == {
        "type": "thinking",
        "thinking": "thinking...",
        "signature": "",
    }


def test_sanitizer_synthetic_thinking_start_has_signature_placeholder():
    out = _sanitize(
        [
            {"type": "message_start"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "reasoning"},
            },
            {"type": "message_stop"},
        ]
    )

    starts = _thinking_starts(out)
    assert starts == [{"type": "thinking", "thinking": "", "signature": ""}]


def test_sanitizer_normalizes_unsigned_thinking_start():
    out = _sanitize(
        [
            {"type": "message_start"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "reasoning"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
    )

    assert _thinking_starts(out)[0]["signature"] == ""


def test_sanitizer_preserves_real_thinking_signature():
    out = _sanitize(
        [
            {"type": "message_start"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "thinking",
                    "thinking": "",
                    "signature": "opaque-native-signature",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "reasoning"},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
    )

    assert _thinking_starts(out)[0]["signature"] == "opaque-native-signature"


def test_compat_signature_is_not_forwarded_to_openai_history():
    out = anthropic_request_to_openai_body(
        {
            "model": "M",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "hidden",
                            "signature": "compat-or-native-signature",
                        },
                        {"type": "text", "text": "visible"},
                    ],
                }
            ],
        }
    )

    assert out["messages"] == [{"role": "assistant", "content": "visible"}]
    assert "compat-or-native-signature" not in json.dumps(out)
