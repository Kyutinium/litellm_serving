"""
Worker startup hook to strip thinking/redacted_thinking content blocks
from messages before they reach non-Anthropic backends (SGLang, vLLM).

The /v1/messages pass-through endpoint bypasses LiteLLM callbacks, so
we monkey-patch litellm.acompletion() directly. This hook is invoked
via LITELLM_WORKER_STARTUP_HOOKS before any requests are handled.
"""

import litellm


def _strip_thinking_from_messages(messages):
    cleaned = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            filtered = [
                block for block in content
                if not (isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"))
            ]
            if not filtered:
                continue
            if isinstance(msg, dict):
                msg = {**msg, "content": filtered}
                # Flatten single text block to string for compatibility
                if len(filtered) == 1 and isinstance(filtered[0], dict) and filtered[0].get("type") == "text":
                    msg = {**msg, "content": filtered[0].get("text", "")}
        cleaned.append(msg)
    return cleaned


_original_acompletion = litellm.acompletion


async def _patched_acompletion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_thinking_from_messages(kwargs["messages"])
    return await _original_acompletion(*args, **kwargs)


def apply_patch():
    """Called by LITELLM_WORKER_STARTUP_HOOKS during worker init."""
    litellm.acompletion = _patched_acompletion
    print("[strip_thinking] Patched litellm.acompletion to strip thinking blocks")
