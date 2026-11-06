"""Chat message-order normalization for strict vLLM chat templates.

Keep stable initial system instructions in the prompt prefix: mid-history system
reminders become user turns at their original point for selected models. Other
models retain the sanitizer's historical hoist-all behavior for compatibility.
Mistral Large 4 additionally requires tool results immediately after the
assistant turn that invoked them. Both operations are pure and idempotent.

Adapted to the sanitizer from JinY0ung-Shin/UniBridge commit
224150e3f0db293a23e32222da8b8625c246e881.
"""

from __future__ import annotations

import re
from typing import Any


def _is_system(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "system"


def _role(message: Any) -> Any:
    return message.get("role") if isinstance(message, dict) else None


def _system_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        )
    return ""


def _merge_system_texts(messages: list[dict]) -> str:
    return "\n\n".join(
        text for text in (_system_text(m.get("content")) for m in messages) if text
    )


def normalize_system_messages(
    messages: list[dict],
    policy: str,
    model: Any = None,
    model_pattern: re.Pattern[str] | None = None,
) -> list[dict]:
    """Normalize system placement; don't mutate inputs.

    For matching models, "user" (default in config) merges an initial system
    run and converts later system turns to user in place. "hoist" puts all
    system text into a single leading system turn. "asis" leaves messages alone.

    Nonmatching/unnamed models always use the old hoist behavior, irrespective
    of policy, so upgrading the sanitizer does not alter their request shape.
    Supply a pattern matching .* to apply the configured policy to all models.
    Without a model_pattern, apply policy unconditionally (unit-test contract).
    """
    if not isinstance(messages, list) or not messages:
        return messages
    if model_pattern is not None and not (
        isinstance(model, str) and model_pattern.search(model)
    ):
        policy = "hoist"

    if policy == "asis":
        return messages
    if policy == "hoist":
        system = [m for m in messages if _is_system(m)]
        if not system:
            return messages
        other = [m for m in messages if not _is_system(m)]
        return [{"role": "system", "content": _merge_system_texts(system)}, *other]

    # Policy "user" (also the fallback for unexpected policy strings).
    head_count = 0
    for message in messages:
        if not _is_system(message):
            break
        head_count += 1

    tail = messages[head_count:]
    if head_count <= 1 and not any(_is_system(m) for m in tail):
        return messages

    result: list[dict] = []
    if head_count:
        head = dict(messages[0])
        head["content"] = _merge_system_texts(messages[:head_count])
        result.append(head)
    for message in tail:
        if _is_system(message):
            result.append({**message, "role": "user"})
        else:
            result.append(message)
    return result


def normalize_tool_messages(
    messages: list[dict],
    model: Any = None,
    model_pattern: re.Pattern[str] | None = None,
) -> list[dict]:
    """Move matching tool results directly after their originating tool calls.

    Between an assistant tool-call turn and the next assistant turn, collect
    only tool messages whose tool_call_id belongs to that assistant. Preserve
    the order of matching results and of all other messages. Unmatched/orphan
    tool messages are deliberately not reassociated with unrelated calls.
    No content is changed; already-normalized input is returned by identity.
    """
    if model_pattern is not None and not (
        isinstance(model, str) and model_pattern.search(model)
    ):
        return messages
    if not isinstance(messages, list):
        return messages

    out: list[dict] = []
    i = 0
    while i < len(messages):
        message = messages[i]
        out.append(message)
        i += 1
        if _role(message) != "assistant" or not message.get("tool_calls"):
            continue

        call_ids = {
            call.get("id")
            for call in message["tool_calls"]
            if isinstance(call, dict) and isinstance(call.get("id"), str)
        }
        if not call_ids:
            continue
        end = i
        while end < len(messages) and _role(messages[end]) != "assistant":
            end += 1
        window = messages[i:end]

        def is_own_result(item: Any) -> bool:
            return _role(item) == "tool" and item.get("tool_call_id") in call_ids

        out.extend(item for item in window if is_own_result(item))
        out.extend(item for item in window if not is_own_result(item))
        i = end

    if len(out) == len(messages) and all(
        first is second for first, second in zip(out, messages)
    ):
        return messages
    return out
