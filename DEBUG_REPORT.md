# Debug Report: Claude SDK Streaming Failure with LiteLLM + GLM-5-FP8

## Environment

| Component | Detail |
|-----------|--------|
| **Model** | GLM-5-FP8 (served via SGLang/vLLM on port 8088) |
| **Proxy** | LiteLLM (`ghcr.io/berriai/litellm:main-stable`) on port 3999 |
| **Gateway** | claude-code-gateway (Anthropic Messages API ↔ Claude Code) |
| **Client** | Claude Code (Claude Agent SDK) |

### Request flow

```
Claude Code ──► claude-code-gateway ──► LiteLLM proxy ──► SGLang (GLM-5-FP8)
   (SDK)         (Anthropic API)        (OpenAI→Anthropic)    (OpenAI API)
```

---

## Problem

Claude Code would either **hang indefinitely** or **crash with a type error** when streaming responses from GLM-5-FP8 through LiteLLM.

### Symptoms

1. **SDK streaming parser crash** — The Claude Agent SDK expects SSE events with consistent content block types. It receives a `content_block_start` event with `type: "text"`, but subsequent `content_block_delta` events arrive as `type: "thinking_delta"` instead of `type: "text_delta"`. The SDK cannot reconcile a `thinking_delta` inside a `text` block and throws an error.

2. **Thinking blocks forwarded to non-Anthropic backends** — When Claude Code sends multi-turn conversations, assistant messages may contain `thinking` and `redacted_thinking` content blocks (from the SDK's extended thinking feature). These blocks are Anthropic-specific. When forwarded to SGLang/vLLM, the backend either rejects them or produces malformed output.

3. **`signature_delta` events** — LiteLLM also emits `signature_delta` events for non-Anthropic models. These have no meaning for GLM-5 and also break the SDK's streaming parser.

---

## Root Cause

LiteLLM's **Anthropic Messages API pass-through adapter** (`LiteLLMAnthropicMessagesAdapter`) translates OpenAI-format streaming chunks into Anthropic SSE format. Two bugs exist in this translation:

### Bug 1: `thinking_delta` / `text` type mismatch

The adapter always emits `content_block_start` with `type: "text"` (correct for non-Anthropic models). However, when the OpenAI chunk contains `reasoning_content` (used by models like DeepSeek, GLM), the adapter converts it to `thinking_delta` — a delta type that only makes sense inside a `thinking` content block, not a `text` block.

```
content_block_start  →  type: "text"          ✓
content_block_delta  →  type: "thinking_delta" ✗  (should be "text_delta")
```

The Claude SDK validates that delta types match their parent block type. This mismatch causes a parse/validation error.

### Bug 2: Thinking blocks in multi-turn messages

The Claude SDK includes `thinking` / `redacted_thinking` content blocks in assistant messages for conversation context. LiteLLM passes these through to the backend verbatim. SGLang/vLLM does not understand these block types, causing failures or garbage output.

---

## Solution

A single startup hook file (`strip_thinking.py`) that patches LiteLLM at worker initialization via `LITELLM_WORKER_STARTUP_HOOKS`.

### Fix 1: Patch the streaming adapter

Monkey-patches `_translate_streaming_openai_chunk_to_anthropic` to:
- Convert `thinking_delta` → `text_delta` (extracting the `thinking` field into `text`)
- Convert `signature_delta` → empty `text_delta` (drop meaningless signatures)

This ensures all delta events are `text_delta`, consistent with the `text` content block type.

```python
def patched_translate(self, choices):
    type_of_content, delta = original(self, choices)
    if type_of_content == "thinking_delta":
        thinking_text = getattr(delta, "thinking", "") or delta.get("thinking", "")
        return "text_delta", ContentTextBlockDelta(type="text_delta", text=thinking_text)
    if type_of_content == "signature_delta":
        return "text_delta", ContentTextBlockDelta(type="text_delta", text="")
    return type_of_content, delta
```

### Fix 2: Strip thinking blocks from inbound messages

A `CustomLogger` callback (`StripThinkingCallback`) runs in LiteLLM's pre-call pipeline and removes `thinking` / `redacted_thinking` content blocks from all messages before they reach the backend. Single remaining `text` blocks are flattened to plain strings for maximum compatibility.

### Deployment

The fix is self-contained — no gateway changes required. It's loaded via a Docker environment variable:

```dockerfile
ENV LITELLM_WORKER_STARTUP_HOOKS=strip_thinking:apply_patch
```

---

## Files Changed

| File | Change |
|------|--------|
| `strip_thinking.py` | New — startup hook with streaming adapter patch + thinking block stripper |
| `Dockerfile` | Added `COPY strip_thinking.py` and `LITELLM_WORKER_STARTUP_HOOKS` env var |
| `litellm_config.yaml` | Added `max_tokens: 16384` to prevent truncated responses |

---

## Verification

After applying the fix and rebuilding (`docker compose up -d --build`):

- Claude Code streams responses from GLM-5-FP8 without errors
- Multi-turn conversations work correctly (thinking blocks stripped before reaching SGLang)
- No gateway modifications needed
