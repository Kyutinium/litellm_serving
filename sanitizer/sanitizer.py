"""``sanitize_events`` — enforce Anthropic Messages SSE invariants.

Pure, transport-independent async iterator (dict-in / dict-out). Given a stream
of upstream Anthropic SSE events (as decoded dicts), it yields a stream that is
guaranteed to conform to the Anthropic Messages streaming spec:

* ``message_start`` is emitted at most once (duplicates dropped);
* every ``content_block_delta.delta.type`` is compatible with the type of the
  currently open ``content_block_start`` (per the table below), splitting into a
  synthetic block when the upstream block type does not match;
* ``index`` values are monotonically increasing from 0;
* every ``content_block_start`` is paired with a ``content_block_stop`` before
  the next start / ``message_delta`` / ``message_stop``.

This works around LiteLLM's ``AnthropicStreamWrapper`` which leaks
``thinking_delta`` events into open ``text`` blocks, drops ``content_block_stop``
events, and explodes zero-payload ``input_json_delta`` events into empty
``tool_use`` blocks.
"""

from __future__ import annotations

from typing import AsyncIterator, Dict, Optional

# Which block types a given delta type is allowed to live in.
DELTA_COMPATIBLE_BLOCKS = {
    "text_delta": frozenset({"text"}),
    "thinking_delta": frozenset({"thinking"}),
    "signature_delta": frozenset({"thinking"}),
    "input_json_delta": frozenset({"tool_use", "server_tool_use"}),
}

# When a delta forces a synthetic block, which block type to open.
DELTA_PRIMARY_BLOCK = {
    "text_delta": "text",
    "thinking_delta": "thinking",
    "signature_delta": "thinking",
    "input_json_delta": "tool_use",
}


def _is_empty_delta(delta: Dict) -> bool:
    """Return True for zero-payload deltas that should be dropped outright."""
    dtype = delta.get("type")
    if dtype == "text_delta":
        return not delta.get("text")
    if dtype == "thinking_delta":
        return not delta.get("thinking")
    if dtype == "signature_delta":
        return not delta.get("signature")
    if dtype == "input_json_delta":
        return not delta.get("partial_json")
    return False


def _synthetic_block(block_type: str) -> Dict:
    """Minimal spec-valid ``content_block`` payload for a synthesized start."""
    if block_type == "thinking":
        return {"type": "thinking", "thinking": "", "signature": ""}
    if block_type in ("tool_use", "server_tool_use"):
        return {"type": "tool_use", "id": "", "name": "", "input": {}}
    return {"type": "text", "text": ""}


def _normalize_content_block(block_type: str, content_block: Dict) -> Dict:
    """Fill thinking fields required by strict Anthropic clients.

    Claude Agent SDK indexes both ``thinking`` and ``signature`` directly.  Some
    LiteLLM-compatible backends omit one of those keys (or serialize it as null),
    so normalize missing/null values to the empty start placeholder.  Real
    upstream values are preserved unchanged.
    """
    if block_type != "thinking":
        return content_block

    thinking = content_block.get("thinking")
    signature = content_block.get("signature")
    if thinking is not None and signature is not None:
        return content_block

    normalized = dict(content_block)
    if thinking is None:
        normalized["thinking"] = ""
    if signature is None:
        normalized["signature"] = ""
    return normalized


def normalize_anthropic_message_body(body: Dict) -> Dict:
    """Normalize thinking blocks in a non-streaming Anthropic message body.

    Streaming responses pass through :func:`sanitize_events`; bridge-mode
    non-streaming responses are constructed locally.  Passthrough non-streaming
    responses need the same compatibility invariant, so apply the content-block
    normalizer without otherwise rewriting the response body.
    """
    content = body.get("content")
    if not isinstance(content, list):
        return body

    normalized_content = []
    changed = False
    for block in content:
        if not isinstance(block, dict):
            normalized_content.append(block)
            continue
        block_type = block.get("type", "text")
        normalized = _normalize_content_block(block_type, block)
        if normalized is not block:
            changed = True
        normalized_content.append(normalized)

    if not changed:
        return body
    normalized_body = dict(body)
    normalized_body["content"] = normalized_content
    return normalized_body


async def sanitize_events(
    upstream: AsyncIterator[Dict],
) -> AsyncIterator[Dict]:
    message_started = False
    open_type: Optional[str] = None
    open_index: Optional[int] = None
    next_index = 0

    def open_block(block_type: str, content_block: Dict):
        nonlocal open_type, open_index, next_index
        open_type = block_type
        open_index = next_index
        next_index += 1
        return {
            "type": "content_block_start",
            "index": open_index,
            "content_block": _normalize_content_block(block_type, content_block),
        }

    def close_block():
        nonlocal open_type, open_index
        if open_type is None:
            return None
        event = {"type": "content_block_stop", "index": open_index}
        open_type = None
        open_index = None
        return event

    async for event in upstream:
        etype = event.get("type")

        if etype == "message_start":
            if message_started:
                continue
            message_started = True
            yield event
            continue

        if etype == "content_block_start":
            content_block = event.get("content_block", {}) or {}
            block_type = content_block.get("type", "text")
            closed = close_block()
            if closed is not None:
                yield closed
            yield open_block(block_type, content_block)
            continue

        if etype == "content_block_delta":
            delta = event.get("delta", {}) or {}
            if _is_empty_delta(delta):
                continue
            dtype = delta.get("type")
            compatible = DELTA_COMPATIBLE_BLOCKS.get(dtype)
            if compatible is None:
                # Unknown delta type: relay against the current block if open.
                if open_index is not None:
                    yield {
                        "type": "content_block_delta",
                        "index": open_index,
                        "delta": delta,
                    }
                else:
                    yield event
                continue
            if open_type not in compatible:
                closed = close_block()
                if closed is not None:
                    yield closed
                primary = DELTA_PRIMARY_BLOCK[dtype]
                yield open_block(primary, _synthetic_block(primary))
            yield {
                "type": "content_block_delta",
                "index": open_index,
                "delta": delta,
            }
            continue

        if etype == "content_block_stop":
            closed = close_block()
            if closed is not None:
                yield closed
            continue

        if etype in ("message_delta", "message_stop"):
            closed = close_block()
            if closed is not None:
                yield closed
            yield event
            continue

        # ping / error / unknown → pass through untouched.
        yield event
