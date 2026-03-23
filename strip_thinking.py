"""
Strip thinking/redacted_thinking content blocks from messages before
they reach non-Anthropic backends (SGLang, vLLM).

Uses LiteLLM's CustomLogger async_pre_call_hook, which runs in the proxy's
pre-call pipeline — before the request flows to the adapter/acompletion.
This avoids wrapping acompletion and preserves streaming behavior.

Invoked via LITELLM_WORKER_STARTUP_HOOKS before any requests are handled.
"""

import inspect
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


def _patch_proxy_streaming_debug():
    """Monkey-patch the proxy's anthropic_messages response path to log streaming decisions."""
    try:
        from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

        original_is_streaming_response = ProxyBaseLLMRequestProcessing._is_streaming_response

        def debug_is_streaming_response(self, response):
            result = original_is_streaming_response(self, response)
            print(
                f"[DEBUG-STREAM] _is_streaming_response: type={type(response).__name__}, "
                f"is_asyncgen={inspect.isasyncgen(response)}, "
                f"result={result}",
                flush=True,
            )
            return result

        ProxyBaseLLMRequestProcessing._is_streaming_response = debug_is_streaming_response
        print("[DEBUG-STREAM] Patched _is_streaming_response for debug logging", flush=True)
    except Exception as e:
        print(f"[DEBUG-STREAM] Failed to patch: {e}", flush=True)


class StripThinkingCallback(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Runs in the proxy pre-call pipeline — strip thinking blocks."""
        stream_val = data.get("stream")
        print(
            f"[strip_thinking] pre_call_hook: call_type={call_type}, "
            f"stream={stream_val} (type={type(stream_val).__name__}), "
            f"model={data.get('model', '?')}",
            flush=True,
        )
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

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Log response type to diagnose streaming."""
        stream = kwargs.get("stream")
        resp_type = type(response_obj).__name__
        print(
            f"[strip_thinking] success: stream={stream}, response_type={resp_type}",
            flush=True,
        )


def apply_patch():
    """Called by LITELLM_WORKER_STARTUP_HOOKS during worker init."""
    callback = StripThinkingCallback()
    litellm.callbacks.append(callback)
    print("[strip_thinking] Registered StripThinkingCallback", flush=True)
    _patch_proxy_streaming_debug()
