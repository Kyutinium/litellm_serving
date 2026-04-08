"""
Strip thinking/redacted_thinking content blocks from messages before
they reach non-Anthropic backends (SGLang, vLLM).

Also patches LiteLLM's Anthropic streaming adapter to handle reasoning_content
based on the THINK_OUTPUT_MODE environment variable:

  - "default" : LiteLLM default behavior (thinking_delta passed as-is)
  - "think_tag": Wrap thinking in <think>...</think> tags, output as regular text
  - "text"    : Output thinking as regular text without wrapping
  - "none"    : Don't output thinking content at all  (default value)

Invoked via LITELLM_WORKER_STARTUP_HOOKS before any requests are handled.
"""

import os
import litellm
from litellm.integrations.custom_logger import CustomLogger

THINK_OUTPUT_MODE = os.environ.get("THINK_OUTPUT_MODE", "none").lower()

_VALID_MODES = ("default", "think_tag", "text", "none")
if THINK_OUTPUT_MODE not in _VALID_MODES:
    print(
        f"[strip_thinking] WARNING: invalid THINK_OUTPUT_MODE='{THINK_OUTPUT_MODE}', "
        f"falling back to 'none'. Valid values: {_VALID_MODES}",
        flush=True,
    )
    THINK_OUTPUT_MODE = "none"


# ---------------------------------------------------------------------------
# Input message cleaning (always active)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Streaming adapter patch
# ---------------------------------------------------------------------------

def _patch_streaming_thinking_delta():
    """Patch the Anthropic streaming adapter based on THINK_OUTPUT_MODE.

    LiteLLM's adapter emits thinking_delta for reasoning_content from models
    like GLM / DeepSeek, but the content_block_start is always type="text".
    This mismatch breaks the Claude SDK's streaming parser.

    Depending on the mode we either:
      - default  : leave everything untouched (no patch)
      - think_tag: convert thinking_delta → text_delta wrapped in <think> tags
      - text     : convert thinking_delta → text_delta (pass content through)
      - none     : convert thinking_delta → text_delta with empty text (drop)
    """
    mode = THINK_OUTPUT_MODE

    if mode == "default":
        print(
            "[strip_thinking] THINK_OUTPUT_MODE=default — streaming adapter NOT patched",
            flush=True,
        )
        return

    try:
        from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
            LiteLLMAnthropicMessagesAdapter,
        )
        from litellm.types.llms.anthropic import ContentTextBlockDelta

        original = (
            LiteLLMAnthropicMessagesAdapter
            ._translate_streaming_openai_chunk_to_anthropic
        )

        def patched_translate(self, choices):
            type_of_content, delta = original(self, choices)

            # --- thinking_delta ------------------------------------------------
            if type_of_content == "thinking_delta":
                if mode == "none":
                    return "text_delta", ContentTextBlockDelta(
                        type="text_delta", text=""
                    )

                thinking_text = (
                    getattr(delta, "thinking", "")
                    or getattr(delta, "text", "")
                    or ""
                )

                if mode == "text":
                    return "text_delta", ContentTextBlockDelta(
                        type="text_delta", text=thinking_text
                    )

                if mode == "think_tag":
                    if not getattr(self, "_think_tag_open", False):
                        self._think_tag_open = True
                        thinking_text = "<think>\n" + thinking_text
                    return "text_delta", ContentTextBlockDelta(
                        type="text_delta", text=thinking_text
                    )

            # --- signature_delta -----------------------------------------------
            if type_of_content == "signature_delta":
                if mode == "think_tag" and getattr(self, "_think_tag_open", False):
                    self._think_tag_open = False
                    return "text_delta", ContentTextBlockDelta(
                        type="text_delta", text="\n</think>\n\n"
                    )
                return "text_delta", ContentTextBlockDelta(
                    type="text_delta", text=""
                )

            # --- text_delta / other — close <think> if still open --------------
            if mode == "think_tag" and getattr(self, "_think_tag_open", False):
                self._think_tag_open = False
                text = getattr(delta, "text", "") or ""
                return type_of_content, ContentTextBlockDelta(
                    type="text_delta", text="\n</think>\n\n" + text
                )

            return type_of_content, delta

        LiteLLMAnthropicMessagesAdapter._translate_streaming_openai_chunk_to_anthropic = (
            patched_translate
        )
        print(
            f"[strip_thinking] Patched streaming adapter (mode={mode})", flush=True
        )
    except Exception as e:
        print(
            f"[strip_thinking] Failed to patch streaming adapter: {e}", flush=True
        )


# ---------------------------------------------------------------------------
# CustomLogger callback
# ---------------------------------------------------------------------------

class StripThinkingCallback(CustomLogger):
    """Pre-call: strip thinking blocks from input messages.

    Input-message stripping is always active regardless of THINK_OUTPUT_MODE
    because non-Anthropic backends (SGLang, vLLM) do not understand thinking
    content blocks.
    """

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def apply_patch():
    """Called by LITELLM_WORKER_STARTUP_HOOKS during worker init."""
    print(f"[strip_thinking] THINK_OUTPUT_MODE={THINK_OUTPUT_MODE}", flush=True)
    callback = StripThinkingCallback()
    litellm.callbacks.append(callback)
    print("[strip_thinking] Registered StripThinkingCallback", flush=True)
    _patch_streaming_thinking_delta()
