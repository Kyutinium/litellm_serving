"""
LiteLLM custom callback to strip thinking/reasoning content blocks
from message history before sending to non-Anthropic backends.

SGLang and other OpenAI-compatible servers don't understand Anthropic's
thinking content blocks ({"type": "thinking", "thinking": "..."}).
This callback filters them out so only text content reaches the backend.
"""

from litellm.integrations.custom_logger import CustomLogger


class StripThinkingCallback(CustomLogger):

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if call_type == "completion" and "messages" in data:
            data["messages"] = _strip_thinking_from_messages(data["messages"])
        return data


def _strip_thinking_from_messages(messages):
    cleaned = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            filtered = [
                block for block in content
                if not (isinstance(block, dict) and block.get("type") in ("thinking", "redacted_thinking"))
            ]
            if not filtered:
                continue
            msg = {**msg, "content": filtered}
            # If only one text block remains, flatten to string
            if len(filtered) == 1 and isinstance(filtered[0], dict) and filtered[0].get("type") == "text":
                msg = {**msg, "content": filtered[0].get("text", "")}
        cleaned.append(msg)
    return cleaned


strip_thinking_callback = StripThinkingCallback()
