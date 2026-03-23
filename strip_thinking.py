"""
Strip thinking/redacted_thinking content blocks from messages before
they reach non-Anthropic backends (SGLang, vLLM).

Two-layer approach:
1. Monkey-patch litellm.acompletion to strip before the call
2. Register a CustomLogger callback as a fallback

Invoked via LITELLM_WORKER_STARTUP_HOOKS before any requests are handled.
"""

import json
import litellm
from litellm.integrations.custom_logger import CustomLogger


def _strip_thinking_from_messages(messages):
    """Remove thinking/redacted_thinking blocks from a list of messages."""
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
        elif hasattr(msg, "content"):
            content = msg.content
        else:
            cleaned.append(msg)
            continue

        if isinstance(content, list):
            filtered = [
                block for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in ("thinking", "redacted_thinking")
                )
                and not (
                    hasattr(block, "type")
                    and getattr(block, "type", None) in ("thinking", "redacted_thinking")
                )
            ]
            if not filtered:
                # Entire message was thinking — skip it
                continue
            # Flatten single text block to plain string
            if (
                len(filtered) == 1
                and isinstance(filtered[0], dict)
                and filtered[0].get("type") == "text"
            ):
                new_content = filtered[0].get("text", "")
            elif (
                len(filtered) == 1
                and hasattr(filtered[0], "type")
                and getattr(filtered[0], "type", None) == "text"
            ):
                new_content = getattr(filtered[0], "text", "")
            else:
                new_content = filtered

            if isinstance(msg, dict):
                msg = {**msg, "content": new_content}
            else:
                try:
                    msg.content = new_content
                except AttributeError:
                    pass
        cleaned.append(msg)
    return cleaned


def _log_messages(tag, messages):
    """Debug-log message structure (types only, no content)."""
    summary = []
    for i, msg in enumerate(messages):
        if isinstance(msg, dict):
            role = msg.get("role", "?")
            content = msg.get("content")
        else:
            role = getattr(msg, "role", "?")
            content = getattr(msg, "content", None)
        if isinstance(content, list):
            types = []
            for block in content:
                if isinstance(block, dict):
                    types.append(block.get("type", "?"))
                else:
                    types.append(getattr(block, "type", str(type(block).__name__)))
            summary.append(f"  [{i}] role={role} content=[{', '.join(types)}]")
        elif isinstance(content, str):
            summary.append(f"  [{i}] role={role} content=str({len(content)} chars)")
        else:
            summary.append(f"  [{i}] role={role} content={type(content).__name__}")
    print(f"[strip_thinking] {tag}:\n" + "\n".join(summary), flush=True)


# ---------------------------------------------------------------------------
# Layer 1: Monkey-patch litellm.acompletion
# ---------------------------------------------------------------------------

_original_acompletion = litellm.acompletion


async def _patched_acompletion(*args, **kwargs):
    messages = kwargs.get("messages")
    if messages and isinstance(messages, list):
        _log_messages("BEFORE strip (acompletion)", messages)
        kwargs["messages"] = _strip_thinking_from_messages(messages)
        _log_messages("AFTER strip (acompletion)", kwargs["messages"])
    return await _original_acompletion(*args, **kwargs)


# ---------------------------------------------------------------------------
# Layer 2: CustomLogger callback (fallback)
# ---------------------------------------------------------------------------

class StripThinkingCallback(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if "messages" in data and isinstance(data["messages"], list):
            _log_messages("BEFORE strip (callback)", data["messages"])
            data["messages"] = _strip_thinking_from_messages(data["messages"])
            _log_messages("AFTER strip (callback)", data["messages"])
        return data


# ---------------------------------------------------------------------------
# Startup hook
# ---------------------------------------------------------------------------

def apply_patch():
    """Called by LITELLM_WORKER_STARTUP_HOOKS during worker init."""
    # Layer 1: patch acompletion
    litellm.acompletion = _patched_acompletion
    print("[strip_thinking] Patched litellm.acompletion", flush=True)

    # Layer 2: register callback
    callback = StripThinkingCallback()
    litellm.callbacks.append(callback)
    print("[strip_thinking] Registered StripThinkingCallback", flush=True)
