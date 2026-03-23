"""
Worker startup hook to strip thinking/redacted_thinking content blocks
from messages before they reach non-Anthropic backends (SGLang, vLLM).

Two layers of stripping:
1. ASGI middleware: rewrites the raw JSON request body before LiteLLM's
   pydantic validation, so thinking blocks don't cause parse errors.
2. Monkey-patched litellm.acompletion(): catches anything the middleware
   might miss (e.g. internally constructed messages).

Invoked via LITELLM_WORKER_STARTUP_HOOKS before any requests are handled.
"""

import json
import litellm


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _strip_thinking_from_messages(messages):
    """Remove thinking/redacted_thinking blocks from a list of messages."""
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


# ---------------------------------------------------------------------------
# Layer 1 – ASGI middleware (strips before pydantic validation)
# ---------------------------------------------------------------------------

class StripThinkingMiddleware:
    """ASGI middleware that strips thinking blocks from incoming request JSON."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Only process POST requests to message/chat endpoints
        path = scope.get("path", "")
        method = scope.get("method", "")
        if method != "POST" or not any(
            p in path for p in ("/v1/messages", "/chat/completions", "/v1/chat/completions")
        ):
            return await self.app(scope, receive, send)

        # Collect the full request body
        body_chunks = []
        more_body = True
        while more_body:
            message = await receive()
            body_chunks.append(message.get("body", b""))
            more_body = message.get("more_body", False)
        raw_body = b"".join(body_chunks)

        # Try to parse and strip thinking blocks
        try:
            data = json.loads(raw_body)
            if "messages" in data and isinstance(data["messages"], list):
                data["messages"] = _strip_thinking_from_messages(data["messages"])
            raw_body = json.dumps(data).encode("utf-8")
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Not JSON or unexpected shape – pass through unchanged

        # Provide the (possibly modified) body to the downstream app
        body_sent = False

        async def patched_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": raw_body, "more_body": False}
            # After body is consumed, wait for disconnect
            return await receive()

        await self.app(scope, patched_receive, send)


# ---------------------------------------------------------------------------
# Layer 2 – Monkey-patch litellm.acompletion (fallback)
# ---------------------------------------------------------------------------

_original_acompletion = litellm.acompletion


async def _patched_acompletion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_thinking_from_messages(kwargs["messages"])
    return await _original_acompletion(*args, **kwargs)


# ---------------------------------------------------------------------------
# Startup hook entry point
# ---------------------------------------------------------------------------

def apply_patch():
    """Called by LITELLM_WORKER_STARTUP_HOOKS during worker init."""
    # Patch acompletion
    litellm.acompletion = _patched_acompletion
    print("[strip_thinking] Patched litellm.acompletion to strip thinking blocks")

    # Install ASGI middleware on the LiteLLM FastAPI app
    try:
        from litellm.proxy.proxy_server import app as litellm_app
        litellm_app.add_middleware(StripThinkingMiddleware)
        print("[strip_thinking] Installed ASGI middleware to strip thinking blocks from requests")
    except Exception as e:
        print(f"[strip_thinking] WARNING: Could not install ASGI middleware: {e}")
        print("[strip_thinking] Falling back to acompletion patch only")
