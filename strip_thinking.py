"""
Strip thinking/redacted_thinking content blocks from messages before
they reach non-Anthropic backends (SGLang, vLLM).

Uses a LiteLLM CustomLogger callback with async_pre_call_hook, which
runs immediately before the API call to the backend — after all format
conversions but before the request is sent.

Invoked via LITELLM_WORKER_STARTUP_HOOKS before any requests are handled.
"""

import litellm
from litellm.integrations.custom_logger import CustomLogger


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _strip_thinking_from_messages(messages):
    """Remove thinking/redacted_thinking blocks from a list of messages."""
    cleaned = []
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)

        if isinstance(content, list):
            filtered = [
                block for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in ("thinking", "redacted_thinking")
                )
            ]
            if not filtered:
                continue
            if isinstance(msg, dict):
                msg = {**msg, "content": filtered}
            else:
                msg.content = filtered
            # Flatten single text block to plain string for compatibility
            content_to_check = msg.get("content") if isinstance(msg, dict) else msg.content
            if (
                isinstance(content_to_check, list)
                and len(content_to_check) == 1
                and isinstance(content_to_check[0], dict)
                and content_to_check[0].get("type") == "text"
            ):
                text_val = content_to_check[0].get("text", "")
                if isinstance(msg, dict):
                    msg["content"] = text_val
                else:
                    msg.content = text_val
        cleaned.append(msg)
    return cleaned


# ---------------------------------------------------------------------------
# LiteLLM CustomLogger callback
# ---------------------------------------------------------------------------

class StripThinkingCallback(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Runs right before the LLM API call — strip thinking blocks."""
        if "messages" in data and isinstance(data["messages"], list):
            data["messages"] = _strip_thinking_from_messages(data["messages"])
        return data


# ---------------------------------------------------------------------------
# Startup hook entry point
# ---------------------------------------------------------------------------

def apply_patch():
    """Called by LITELLM_WORKER_STARTUP_HOOKS during worker init."""
    callback = StripThinkingCallback()
    litellm.callbacks.append(callback)
    print("[strip_thinking] Registered StripThinkingCallback (async_pre_call_hook)")
