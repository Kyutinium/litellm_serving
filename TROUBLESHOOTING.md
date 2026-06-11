# LiteLLM + SGLang Troubleshooting Notes

## Setup

- **LiteLLM** runs in Docker (host network mode) on port `3999`
- **SGLang** runs on the host on port `8088`, serving `glm-5-fp8`
- **claude-wrapper** connects to LiteLLM as a proxy

## Issues Encountered & Fixes

### 1. Proxy / SSL Errors (403 from SGLang)

**Symptom:** LiteLLM couldn't reach SGLang at `localhost:8088`, getting 403 or SSL errors.

**Cause:** Corporate proxy was intercepting requests from inside the container.

**Fix:** Added `NO_PROXY=localhost,127.0.0.1` to the container environment in `docker-compose.yml`.

### 2. Model Name is Case-Sensitive

**Symptom:**
```
400: Invalid model name passed in model=GLM-5-FP8. Call `/v1/models` to view available models for your key.
```

**Cause:** LiteLLM model names are case-sensitive. Config had `glm-5-fp8` but the client sent `GLM-5-FP8`.

**Fix:** Added an uppercase alias in `litellm_config.yaml`:

```yaml
model_list:
  - model_name: glm-5-fp8
    litellm_params:
      model: hosted_vllm/glm-5-fp8
      api_base: http://localhost:8088/v1
      api_key: EMPTY
  - model_name: GLM-5-FP8
    litellm_params:
      model: hosted_vllm/glm-5-fp8
      api_base: http://localhost:8088/v1
      api_key: EMPTY
```

### 3. 500 Internal Server Error from claude-wrapper

**Symptom:**
```
litellm.exceptions.InternalServerError: OpenAIException - Internal Server Error.
Received Model Group=GLM-5-FP8
```

The error traceback passes through `anthropic/experimental_pass_through/messages/handler.py`.

**Cause:** The claude-wrapper sends requests using the **Anthropic Messages API format** (`/v1/messages`). LiteLLM's `experimental_pass_through` handler converts these to the **Responses API** format and calls `litellm.aresponses()`. With the `openai/` model prefix, LiteLLM assumes the backend is actual OpenAI (which supports `/v1/responses`), so it tries to hit `http://localhost:8088/v1/responses` on SGLang — which doesn't exist, causing the 500.

The `SystemMessage(subtype='api_retry')` seen in claude-wrapper logs is the Claude Agent SDK's built-in retry mechanism reacting to the 500 — it's a symptom, not the root cause.

**Fix:** Changed model prefix from `openai/glm-5-fp8` to `hosted_vllm/glm-5-fp8` in `litellm_config.yaml`. The `hosted_vllm/` prefix tells LiteLLM the backend is an OpenAI-compatible server (like SGLang/vLLM) that only supports chat completions, forcing it to use the chat completions fallback path instead of the Responses API.

Rebuild with `docker compose build && docker compose up -d`.

### 4. BadRequestError: Thinking blocks in message content

**Symptom:**
```
litellm.exceptions.BadRequestError: Hosted_vllmException - 19 validation errors:
'Input should be a valid string', 'input': [{'type': 'thinking', 'thinking': '...'}]
```

**Cause:** Multi-turn conversations include previous assistant messages with Anthropic `thinking` content blocks (`{"type": "thinking", "thinking": "..."}`). LiteLLM's Anthropic-to-ChatCompletions adapter doesn't strip these blocks, so SGLang receives them as-is and rejects them (it expects string content, not content block arrays).

**Fix:**
1. Added `drop_params: true` to `litellm_settings` to drop unsupported top-level parameters (like `output_config`, `thinking`)
2. Created `strip_thinking.py` — a custom LiteLLM callback that filters thinking/redacted_thinking blocks from message content arrays before they reach the backend
3. Registered the callback in `litellm_config.yaml` under `litellm_settings.callbacks`
4. Updated Dockerfile to copy `strip_thinking.py` into the container

Rebuild with `docker compose build && docker compose up -d`.

### 5. Claude Code subagent gets 200 OK with empty `response_text`

**Symptom:** Claude Code subagent calls return HTTP 200, but the collected `response_text` is empty even though usage shows non-zero `completion_tokens`.

**Confirmed fix path:** When vLLM/GLM streams visible text as `reasoning_content`, LiteLLM can surface it as `thinking_delta`. In `THINK_OUTPUT_MODE=none`, `strip_thinking.py` now promotes that `thinking_delta` text into an Anthropic `text_delta` instead of replacing it with an empty string. This covers the `content: null` + `reasoning_content: "..."` pattern.

**Important limitation:** If the upstream backend sends no text-bearing delta at all, or sends only `tool_calls` and never follows up with final assistant text after tool results, this patch cannot synthesize a response. In that case, check the raw vLLM/LiteLLM stream for these fields:

- `choices[].delta.content` or `choices[].message.content`
- `choices[].delta.reasoning_content` or `choices[].message.reasoning_content`
- `choices[].delta.tool_calls` / `choices[].message.tool_calls` and `finish_reason`
- whether the next request includes the matching `tool_result`/tool message content

If both `content` and `reasoning_content` are absent/empty across the stream, the empty subagent response is upstream/model/tool-loop behavior rather than `strip_thinking.py` dropping text.

## Quick Test Commands

```bash
# Test LiteLLM directly (OpenAI format — works)
curl http://localhost:3999/v1/chat/completions \
  -H "Authorization: Bearer sk-1234" \
  -H "Content-Type: application/json" \
  -d '{"model": "glm-5-fp8", "messages": [{"role": "user", "content": "Hello!"}]}'

# Test SGLang directly
curl http://localhost:8088/v1/chat/completions \
  -H "Authorization: Bearer EMPTY" \
  -H "Content-Type: application/json" \
  -d '{"model": "glm-5-fp8", "messages": [{"role": "user", "content": "Hello!"}]}'

# List available models on LiteLLM
curl http://localhost:3999/v1/models -H "Authorization: Bearer sk-1234"
```

### 6. Bypass `strip_thinking.py` completely

If another LiteLLM proxy works and you need to verify whether this hook is the difference, set:

```bash
STRIP_THINKING_ENABLED=false
```

This is stronger than `THINK_OUTPUT_MODE=default`: `default` only skips the streaming adapter patch, while `STRIP_THINKING_ENABLED=false` skips both the input-message stripping callback registration and the streaming adapter patch. If bypass mode fixes the issue, compare raw streams with `STRIP_THINKING_ENABLED=true` and false.

