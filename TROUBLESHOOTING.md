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
      model: openai/glm-5-fp8
      api_base: http://localhost:8088/v1
      api_key: EMPTY
  - model_name: GLM-5-FP8
    litellm_params:
      model: openai/glm-5-fp8
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
