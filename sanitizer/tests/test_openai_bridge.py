"""Unit tests for the Anthropic ↔ OpenAI bridge translations."""

import asyncio
import json

from sanitizer.openai_bridge import (
    anthropic_request_to_openai_body,
    openai_response_to_anthropic_body,
    openai_stream_to_anthropic_events,
)


def run(coro):
    return asyncio.run(coro)


async def _agen(items):
    for item in items:
        yield item


def stream_to_events(chunks, model="M"):
    async def _collect():
        return [e async for e in openai_stream_to_anthropic_events(_agen(chunks), model)]

    return run(_collect())


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #


def test_system_merge():
    body = {
        "model": "M",
        "system": "top",
        "messages": [
            {"role": "system", "content": "inner"},
            {"role": "user", "content": "hi"},
        ],
    }
    out = anthropic_request_to_openai_body(body)
    assert out["messages"][0] == {"role": "system", "content": "top\n\ninner"}
    assert out["messages"][1] == {"role": "user", "content": "hi"}


def test_system_as_block_list():
    body = {"model": "M", "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "messages": []}
    out = anthropic_request_to_openai_body(body)
    assert out["messages"][0] == {"role": "system", "content": "a\n\nb"}


def test_assistant_history_tool_use_and_thinking():
    body = {
        "model": "M",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "secret"},
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "tu_1", "name": "Bash", "input": {"command": "ls"}},
                ],
            }
        ],
    }
    out = anthropic_request_to_openai_body(body)
    msg = out["messages"][0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Let me check."  # thinking dropped
    assert msg["tool_calls"][0]["id"] == "tu_1"
    assert msg["tool_calls"][0]["function"]["name"] == "Bash"
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}


def test_assistant_tool_calls_content_is_empty_string_not_none():
    body = {
        "model": "M",
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "B", "input": {}}]}
        ],
    }
    out = anthropic_request_to_openai_body(body)
    msg = out["messages"][0]
    assert msg["content"] == ""  # never None
    assert "tool_calls" in msg


def test_tool_result_becomes_tool_message():
    body = {
        "model": "M",
        "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": "file1\nfile2"}]}
        ],
    }
    out = anthropic_request_to_openai_body(body)
    assert out["messages"][0] == {"role": "tool", "tool_call_id": "tu_1", "content": "file1\nfile2"}


def test_user_text_before_tool_result():
    body = {
        "model": "M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "context"},
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"},
                ],
            }
        ],
    }
    out = anthropic_request_to_openai_body(body)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "tool"]
    assert out["messages"][0]["content"] == "context"


def test_user_turn_image_becomes_multipart_image_url():
    body = {
        "model": "M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                ],
            }
        ],
    }
    out = anthropic_request_to_openai_body(body)
    content = out["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_text_only_user_stays_plain_string():
    body = {"model": "M", "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    out = anthropic_request_to_openai_body(body)
    assert out["messages"][0]["content"] == "hi"


def test_tool_result_image_relocated_to_trailing_user_message():
    # Issue #140: OpenAI tool messages cannot carry images → relocate to a user turn.
    body = {
        "model": "M",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [
                            {"type": "text", "text": "screenshot"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "IMG"}},
                        ],
                    }
                ],
            }
        ],
    }
    out = anthropic_request_to_openai_body(body)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["tool", "user"]
    tool_msg = out["messages"][0]
    assert tool_msg["tool_call_id"] == "tu_1"
    assert "screenshot" in tool_msg["content"]
    assert "attached" in tool_msg["content"].lower()  # pointer note present
    user_msg = out["messages"][1]
    assert isinstance(user_msg["content"], list)
    assert user_msg["content"][0]["type"] == "image_url"
    assert user_msg["content"][0]["image_url"]["url"] == "data:image/png;base64,IMG"


def test_url_image_source_passthrough():
    body = {
        "model": "M",
        "messages": [
            {"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "http://x/y.png"}}]}
        ],
    }
    out = anthropic_request_to_openai_body(body)
    assert out["messages"][0]["content"][0]["image_url"]["url"] == "http://x/y.png"


def test_tools_definition_conversion():
    body = {
        "model": "M",
        "messages": [],
        "tools": [
            {"name": "Bash", "description": "shell", "input_schema": {"type": "object", "properties": {"c": {"type": "string"}}}}
        ],
    }
    out = anthropic_request_to_openai_body(body)
    tool = out["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "Bash"
    assert tool["function"]["description"] == "shell"
    assert tool["function"]["parameters"]["properties"] == {"c": {"type": "string"}}


def test_tool_choice_variants():
    def choice(tc):
        return anthropic_request_to_openai_body({"model": "M", "messages": [], "tool_choice": tc}).get("tool_choice")

    assert choice({"type": "auto"}) == "auto"
    assert choice({"type": "any"}) == "required"
    assert choice({"type": "none"}) == "none"
    assert choice({"type": "tool", "name": "Bash"}) == {"type": "function", "function": {"name": "Bash"}}
    assert "tool_choice" not in anthropic_request_to_openai_body({"model": "M", "messages": []})


def test_stop_sequences_renamed_to_stop():
    out = anthropic_request_to_openai_body({"model": "M", "messages": [], "stop_sequences": ["X"]})
    assert out["stop"] == ["X"]


def test_stream_forces_include_usage():
    out = anthropic_request_to_openai_body({"model": "M", "messages": [], "stream": True})
    assert out["stream"] is True
    assert out["stream_options"] == {"include_usage": True}


def test_thinking_and_metadata_dropped_from_request():
    out = anthropic_request_to_openai_body({"model": "M", "messages": [], "thinking": {"type": "enabled"}, "metadata": {"x": 1}})
    assert "thinking" not in out and "metadata" not in out


# --------------------------------------------------------------------------- #
# Streaming translation
# --------------------------------------------------------------------------- #


def test_stream_reasoning_content_toolcall_flow():
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"reasoning_content": "thinking..."}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "Bash", "arguments": "{\"command\":\"ls\"}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 10}},
    ]
    out = stream_to_events(chunks)
    types = [e["type"] for e in out]
    assert types[0] == "message_start"
    assert types[-1] == "message_stop"
    starts = [(e["index"], e["content_block"]["type"]) for e in out if e["type"] == "content_block_start"]
    assert starts == [(0, "thinking"), (1, "text"), (2, "tool_use")]
    tool_start = [e for e in out if e["type"] == "content_block_start" and e["content_block"]["type"] == "tool_use"][0]
    assert tool_start["content_block"]["id"] == "call_1"
    assert tool_start["content_block"]["name"] == "Bash"
    ijd = [e for e in out if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"][0]
    assert ijd["delta"]["partial_json"] == '{"command":"ls"}'
    mdelta = [e for e in out if e["type"] == "message_delta"][0]
    assert mdelta["delta"]["stop_reason"] == "tool_use"
    assert mdelta["usage"] == {"input_tokens": 50, "output_tokens": 10}


def test_stream_finish_reason_map():
    def stop_reason(fr):
        chunks = [{"choices": [{"delta": {"content": "x"}, "finish_reason": fr}]}]
        out = stream_to_events(chunks)
        return [e for e in out if e["type"] == "message_delta"][0]["delta"]["stop_reason"]

    assert stop_reason("stop") == "end_turn"
    assert stop_reason("length") == "max_tokens"
    assert stop_reason("content_filter") == "stop_sequence"


def test_stream_empty_content_chunk_does_not_split():
    chunks = [
        {"choices": [{"delta": {"content": ""}}]},
        {"choices": [{"delta": {"content": "real"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    out = stream_to_events(chunks)
    starts = [e for e in out if e["type"] == "content_block_start"]
    assert len(starts) == 1 and starts[0]["content_block"]["type"] == "text"


def test_stream_empty_upstream_still_frames_message():
    out = stream_to_events([])
    types = [e["type"] for e in out]
    assert types == ["message_start", "message_delta", "message_stop"]


def test_stream_two_parallel_tool_calls():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "A", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c1", "function": {"name": "B", "arguments": "{}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    out = stream_to_events(chunks)
    tool_starts = [e["content_block"] for e in out if e["type"] == "content_block_start"]
    assert [t["name"] for t in tool_starts] == ["A", "B"]
    assert [t["id"] for t in tool_starts] == ["c0", "c1"]


def test_stream_interleaved_parallel_arguments():
    chunks = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c0", "function": {"name": "A", "arguments": "{\"x\":"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c1", "function": {"name": "B", "arguments": "{\"y\":"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 1, "function": {"arguments": "2}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    out = stream_to_events(chunks)
    ijds = [e for e in out if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta"]
    # each call's fragments concatenated in order → complete JSON per block
    assert ijds[0]["delta"]["partial_json"] == '{"x":1}'
    assert ijds[1]["delta"]["partial_json"] == '{"y":2}'


# --------------------------------------------------------------------------- #
# Non-streaming translation
# --------------------------------------------------------------------------- #


def test_nonstream_block_order():
    body = {
        "id": "chatcmpl-1",
        "model": "M",
        "choices": [
            {
                "message": {
                    "reasoning_content": "thinking",
                    "content": "answer",
                    "tool_calls": [{"id": "t1", "function": {"name": "Bash", "arguments": "{\"c\":\"ls\"}"}}],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 7},
    }
    out = openai_response_to_anthropic_body(body)
    assert [b["type"] for b in out["content"]] == ["thinking", "text", "tool_use"]
    assert out["content"][2]["input"] == {"c": "ls"}
    assert out["stop_reason"] == "tool_use"
    assert out["usage"] == {"input_tokens": 3, "output_tokens": 7}
    assert out["id"].startswith("msg_")  # chatcmpl id replaced with anthropic-style id


def test_nonstream_malformed_arguments_fallback():
    body = {
        "model": "M",
        "choices": [{"message": {"tool_calls": [{"id": "t", "function": {"name": "B", "arguments": "not json"}}]}, "finish_reason": "tool_calls"}],
    }
    out = openai_response_to_anthropic_body(body)
    assert out["content"][0]["input"] == {}
