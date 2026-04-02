"""
Strip thinking/redacted_thinking content blocks from messages before
they reach non-Anthropic backends (SGLang, vLLM).

Also patches LiteLLM's Anthropic streaming adapter to emit text_delta
instead of thinking_delta for reasoning_content, fixing a format mismatch
that prevents the Claude SDK from streaming.

Invoked via LITELLM_WORKER_STARTUP_HOOKS before any requests are handled.
"""

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


def _patch_streaming_thinking_delta():
    """Patch the Anthropic adapter to emit text_delta instead of thinking_delta.

    LiteLLM's adapter has a bug: the initial content_block_start is always
    type="text", but reasoning_content from models like GLM/DeepSeek gets
    converted to thinking_delta. This mismatch breaks the Claude SDK's
    streaming parser. Fix: convert thinking_delta → text_delta so all
    content is consistent with the text block type.
    """
    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        from litellm.types.llms.anthropic import ContentTextBlockDelta

        original = LiteLLMAnthropicMessagesAdapter._translate_streaming_openai_chunk_to_anthropic

        def patched_translate(self, choices):
            type_of_content, delta = original(self, choices)
            # Convert thinking_delta → text_delta to match the "text" content block
            if type_of_content == "thinking_delta":
                # Drop reasoning content entirely — don't forward to client
                return "text_delta", ContentTextBlockDelta(
                    type="text_delta", text=""
                )
            if type_of_content == "signature_delta":
                # Drop signature deltas — not meaningful for non-Anthropic models
                return "text_delta", ContentTextBlockDelta(type="text_delta", text="")
            return type_of_content, delta

        LiteLLMAnthropicMessagesAdapter._translate_streaming_openai_chunk_to_anthropic = patched_translate
        print("[strip_thinking] Patched thinking_delta → text_delta in streaming adapter", flush=True)
    except Exception as e:
        print(f"[strip_thinking] Failed to patch streaming adapter: {e}", flush=True)


class StripThinkingCallback(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Runs in the proxy pre-call pipeline — strip thinking blocks."""
        if "messages" in data and isinstance(data["messages"], list):
            before_count = len(data["messages"])
            data["messages"] = _strip_thinking_from_messages(data["messages"])
            after_count = len(data["messages"])
            if before_count != after_count:
                print(
                    f"[strip_thinking] Stripped {before_count - after_count} "
                    f"thinking-only message(s) ({before_count} -> {after_count})",
                    flush=True,
                )
        return data


def apply_patch():
    """Called by LITELLM_WORKER_STARTUP_HOOKS during worker init."""
    callback = StripThinkingCallback()
    litellm.callbacks.append(callback)
    print("[strip_thinking] Registered StripThinkingCallback", flush=True)
    _patch_streaming_thinking_delta()
