"""Anthropic ↔ OpenAI bidirectional translation.

The bridge lets the sanitizer call the upstream's *known-good*
``/v1/chat/completions`` route directly and perform the Anthropic ↔ OpenAI
translation in-process, sidestepping LiteLLM's broken ``/v1/messages`` adapter
(dropped reasoning content, truncated / zero-payload ``input_json_delta``).

Three public entry points:

* ``anthropic_request_to_openai_body`` — request body translation;
* ``openai_stream_to_anthropic_events`` — streaming SSE state machine;
* ``openai_response_to_anthropic_body`` — non-streaming one-shot translation.
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator, Dict, List, Optional

# Note attached to a ``tool`` message whose image payload was relocated into a
# trailing user message (OpenAI ``role:"tool"`` messages cannot carry images).
_TOOL_IMAGE_NOTE = "[The tool returned an image; it is attached in the following user message.]"
_RELOCATED_IMAGE_NOTE = "[Image returned by the previous tool call]"

# OpenAI/vLLM reasoning has no Anthropic cryptographic thinking signature.  The
# Agent SDK still requires the field to exist, and Anthropic's own streaming
# protocol starts thinking blocks with an empty signature before a later
# signature_delta fills it.  Keep the compatibility placeholder explicitly
# empty rather than pretending it is a real signature.  Request translation
# drops thinking blocks, so this value is never forwarded upstream.
_UNSIGNED_THINKING_SIGNATURE = ""

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "stop_sequence",
}


# --------------------------------------------------------------------------- #
# Request translation: Anthropic /v1/messages body → OpenAI chat body
# --------------------------------------------------------------------------- #


def _extract_system_texts(system) -> List[str]:
    """Flatten an Anthropic ``system`` value into a list of text strings."""
    if not system:
        return []
    if isinstance(system, str):
        return [system]
    texts: List[str] = []
    if isinstance(system, list):
        for block in system:
            if isinstance(block, str):
                if block:
                    texts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if text:
                    texts.append(text)
    return texts


def _image_block_to_image_url(block: Dict) -> Optional[Dict]:
    """Convert an Anthropic image block to an OpenAI ``image_url`` part."""
    source = block.get("source") or {}
    stype = source.get("type")
    if stype == "base64":
        data = source.get("data")
        if not data:
            return None
        media_type = source.get("media_type", "image/png")
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}
    if stype == "url":
        url = source.get("url")
        if not url:
            return None
        return {"type": "image_url", "image_url": {"url": url}}
    return None


def _extract_tool_result_content(content):
    """Split an Anthropic ``tool_result`` content into (text, [image parts])."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    text_chunks: List[str] = []
    images: List[Dict] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                if block:
                    text_chunks.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    if block.get("text"):
                        text_chunks.append(block["text"])
                elif btype == "image":
                    part = _image_block_to_image_url(block)
                    if part:
                        images.append(part)
    return "\n".join(text_chunks), images


def _convert_user_message(content) -> List[Dict]:
    """Convert an Anthropic user message into one or more OpenAI messages.

    * Plain text stays a plain-string ``user`` message.
    * User-turn images become ``image_url`` parts in a multipart ``user`` message.
    * ``tool_result`` blocks become separate ``tool`` messages; any image inside a
      tool_result is relocated into a trailing ``user`` message (OpenAI tool
      messages cannot carry images) — the fix for gateway issue #140.
    """
    messages: List[Dict] = []
    if isinstance(content, str):
        if content:
            messages.append({"role": "user", "content": content})
        return messages
    if not isinstance(content, list):
        return messages

    text_parts: List[Dict] = []
    image_parts: List[Dict] = []
    tool_results = []  # (tool_call_id, text, [image parts])

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            if block.get("text"):
                text_parts.append({"type": "text", "text": block["text"]})
        elif btype == "image":
            part = _image_block_to_image_url(block)
            if part:
                image_parts.append(part)
        elif btype == "tool_result":
            tr_text, tr_images = _extract_tool_result_content(block.get("content"))
            tool_results.append((block.get("tool_use_id", ""), tr_text, tr_images))

    # Top-level user content (text before images). Plain string unless an image
    # forces the structured multipart shape (vLLM vision requires multipart).
    if image_parts:
        messages.append({"role": "user", "content": text_parts + image_parts})
    elif text_parts:
        messages.append(
            {"role": "user", "content": "\n".join(p["text"] for p in text_parts)}
        )

    # Tool results (+ relocated images as trailing user messages).
    for tool_call_id, tr_text, tr_images in tool_results:
        if tr_images:
            tool_content = f"{tr_text}\n\n{_TOOL_IMAGE_NOTE}".strip() if tr_text else _TOOL_IMAGE_NOTE
        else:
            tool_content = tr_text
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": tool_content}
        )
        if tr_images:
            messages.append(
                {
                    "role": "user",
                    "content": list(tr_images)
                    + [{"type": "text", "text": _RELOCATED_IMAGE_NOTE}],
                }
            )

    return messages


def _convert_assistant_message(content) -> Dict:
    """Convert an Anthropic assistant message into an OpenAI assistant message."""
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_chunks: List[str] = []
    tool_calls: List[Dict] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                if block.get("text"):
                    text_chunks.append(block["text"])
            elif btype == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}) or {}),
                        },
                    }
                )
            # thinking / redacted_thinking blocks are intentionally dropped.

    message = {"role": "assistant", "content": "\n".join(text_chunks)}
    if tool_calls:
        # content must be "" (never None): LiteLLM's exclude_none serialization
        # would drop the key and vLLM answers 422 "content field required".
        message["tool_calls"] = tool_calls
    return message


def _convert_tool(tool: Dict) -> Dict:
    function = {
        "name": tool.get("name", ""),
        "parameters": tool.get("input_schema", {}) or {},
    }
    if tool.get("description"):
        function["description"] = tool["description"]
    return {"type": "function", "function": function}


def _convert_tool_choice(tool_choice):
    if not tool_choice or not isinstance(tool_choice, dict):
        return None
    ctype = tool_choice.get("type")
    if ctype == "auto":
        return "auto"
    if ctype == "any":
        return "required"
    if ctype == "none":
        return "none"
    if ctype == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return None


def anthropic_request_to_openai_body(body: Dict) -> Dict:
    out: Dict = {}
    if body.get("model") is not None:
        out["model"] = body["model"]
    for key in ("max_tokens", "temperature", "top_p"):
        if body.get(key) is not None:
            out[key] = body[key]

    stream = bool(body.get("stream"))
    out["stream"] = stream

    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]

    # Merge top-level system + any system-role messages into one leading system
    # message (vLLM: "System message must be at the beginning.").
    system_texts = _extract_system_texts(body.get("system"))

    converted: List[Dict] = []
    for msg in body.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_texts.extend(_extract_system_texts(content))
        elif role == "assistant":
            converted.append(_convert_assistant_message(content))
        else:  # user (and any unknown role) → user conversion
            converted.extend(_convert_user_message(content))

    messages: List[Dict] = []
    if system_texts:
        messages.append({"role": "system", "content": "\n\n".join(system_texts)})
    messages.extend(converted)
    out["messages"] = messages

    if body.get("tools"):
        out["tools"] = [_convert_tool(t) for t in body["tools"] if isinstance(t, dict)]
    tool_choice = _convert_tool_choice(body.get("tool_choice"))
    if tool_choice is not None:
        out["tool_choice"] = tool_choice

    if stream:
        out["stream_options"] = {"include_usage": True}

    return out


# --------------------------------------------------------------------------- #
# Streaming response translation: OpenAI SSE → Anthropic SSE
# --------------------------------------------------------------------------- #


class _PendingToolCall:
    __slots__ = ("id", "name", "arguments")

    def __init__(self):
        self.id: Optional[str] = None
        self.name: Optional[str] = None
        self.arguments: str = ""


class _StreamState:
    def __init__(self, model: str):
        self.model = model
        self.message_id = "msg_" + uuid.uuid4().hex
        self.open_kind: Optional[str] = None  # "thinking" | "text" | None
        self.open_index: Optional[int] = None
        self.next_index = 0
        self.pending_tool_calls: Dict[int, _PendingToolCall] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.finish_reason: Optional[str] = None


def _close_open(state: _StreamState) -> List[Dict]:
    if state.open_kind is None:
        return []
    event = {"type": "content_block_stop", "index": state.open_index}
    state.open_kind = None
    state.open_index = None
    return [event]


def _ensure_block(state: _StreamState, kind: str, content_block: Dict) -> List[Dict]:
    if state.open_kind == kind:
        return []
    events = _close_open(state)
    state.open_kind = kind
    state.open_index = state.next_index
    state.next_index += 1
    events.append(
        {
            "type": "content_block_start",
            "index": state.open_index,
            "content_block": content_block,
        }
    )
    return events


def _buffer_tool_calls(state: _StreamState, tool_calls) -> None:
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        idx = tc.get("index", 0)
        pending = state.pending_tool_calls.setdefault(idx, _PendingToolCall())
        if tc.get("id"):
            pending.id = tc["id"]
        function = tc.get("function") or {}
        if function.get("name"):
            pending.name = function["name"]
        if function.get("arguments"):
            pending.arguments += function["arguments"]


def _flush_tool_calls(state: _StreamState) -> List[Dict]:
    events = _close_open(state)
    for idx in sorted(state.pending_tool_calls.keys()):
        pending = state.pending_tool_calls[idx]
        block_index = state.next_index
        state.next_index += 1
        events.append(
            {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": pending.id or "",
                    "name": pending.name or "",
                    "input": {},
                },
            }
        )
        if pending.arguments:
            events.append(
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": pending.arguments},
                }
            )
        events.append({"type": "content_block_stop", "index": block_index})
    state.pending_tool_calls = {}
    return events


async def openai_stream_to_anthropic_events(
    chunks: AsyncIterator[Dict], model: str
) -> AsyncIterator[Dict]:
    state = _StreamState(model)

    yield {
        "type": "message_start",
        "message": {
            "id": state.message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }

    async for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        usage = chunk.get("usage")
        if usage:
            if usage.get("prompt_tokens") is not None:
                state.input_tokens = usage["prompt_tokens"]
            if usage.get("completion_tokens") is not None:
                state.output_tokens = usage["completion_tokens"]

        for choice in chunk.get("choices", []) or []:
            if choice.get("finish_reason"):
                state.finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}

            reasoning = delta.get("reasoning_content")
            if reasoning:
                for ev in _ensure_block(
                    state,
                    "thinking",
                    {
                        "type": "thinking",
                        "thinking": "",
                        "signature": _UNSIGNED_THINKING_SIGNATURE,
                    },
                ):
                    yield ev
                yield {
                    "type": "content_block_delta",
                    "index": state.open_index,
                    "delta": {"type": "thinking_delta", "thinking": reasoning},
                }

            content = delta.get("content")
            if content:  # empty-string content chunks must not open a block
                for ev in _ensure_block(state, "text", {"type": "text", "text": ""}):
                    yield ev
                yield {
                    "type": "content_block_delta",
                    "index": state.open_index,
                    "delta": {"type": "text_delta", "text": content},
                }

            if delta.get("tool_calls"):
                _buffer_tool_calls(state, delta["tool_calls"])

    for ev in _flush_tool_calls(state):
        yield ev

    stop_reason = _FINISH_REASON_MAP.get(state.finish_reason, "end_turn")
    yield {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": state.input_tokens, "output_tokens": state.output_tokens},
    }
    yield {"type": "message_stop"}


# --------------------------------------------------------------------------- #
# Non-streaming response translation: OpenAI chat body → Anthropic body
# --------------------------------------------------------------------------- #


def openai_response_to_anthropic_body(body: Dict) -> Dict:
    choices = body.get("choices") or []
    message = choices[0].get("message") or {} if choices else {}

    content_blocks: List[Dict] = []
    reasoning = message.get("reasoning_content")
    if reasoning:
        content_blocks.append(
            {
                "type": "thinking",
                "thinking": reasoning,
                "signature": _UNSIGNED_THINKING_SIGNATURE,
            }
        )
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})
    for tc in message.get("tool_calls") or []:
        function = tc.get("function") or {}
        try:
            args = json.loads(function.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": function.get("name", ""),
                "input": args,
            }
        )

    finish_reason = choices[0].get("finish_reason") if choices else None
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    usage = body.get("usage") or {}
    message_id = body.get("id")
    if not message_id or not str(message_id).startswith("msg_"):
        message_id = "msg_" + uuid.uuid4().hex

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": body.get("model", ""),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
