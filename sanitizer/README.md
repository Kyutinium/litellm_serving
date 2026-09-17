# Sanitizer

A thin **Anthropic-facing reverse proxy** placed in front of the LiteLLM proxy.
It exposes an Anthropic `/v1/messages` endpoint and works around confirmed bugs
in LiteLLM's own Anthropic adapter, so spec-conforming clients (notably the
Claude Agent SDK) work against LiteLLM-fronted vLLM/SGLang backends.

Implements the specification in issue #15.

## What it does

1. **Malformed Anthropic SSE repair** (`sanitize_events`) — enforces Anthropic
   Messages SSE invariants: single `message_start`, block/delta type
   compatibility (splitting into synthetic blocks when needed), monotonic
   indices, and matched `content_block_start`/`content_block_stop` pairs. Drops
   zero-payload deltas that otherwise cause text↔thinking thrashing and empty
   `{}` tool-call arguments.
2. **Anthropic ↔ OpenAI bridge** (optional, `openai_bridge`) — when
   `SANITIZER_USE_OPENAI_BRIDGE=true`, calls the upstream's known-good
   `/v1/chat/completions` route directly and translates in-process, sidestepping
   LiteLLM's broken `/v1/messages` adapter. Handles system-message merging,
   tool-call buffering/flush, and **relocates images returned inside
   `tool_result` blocks into a trailing user message** (OpenAI `role:"tool"`
   messages cannot carry images — the fix for gateway issue #140).
3. **`THINK_OUTPUT_MODE` post-processing** (`transform_events`) — controls how
   reasoning/thinking content is surfaced (`default` / `none` / `text` /
   `think_tag` / `bridge`).

Every other path (`/v1/models`, direct `/v1/chat/completions`, `/v1/embeddings`,
…) is relayed **byte-for-byte** by the wildcard passthrough route.

**Dependencies:** `fastapi`, `httpx`, `uvicorn[standard]` only. It never imports
`litellm`, `pydantic`, `anthropic`, or `openai` — it talks to LiteLLM over HTTP.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `UPSTREAM_BASE_URL` | `http://localhost:3999` | Upstream LiteLLM base URL |
| `SANITIZER_PORT` | `3996` | Sanitizer listen port |
| `SANITIZER_TLS_VERIFY` | `true` | `true/false/…` or a CA bundle path |
| `SANITIZER_REQUEST_TIMEOUT` | `0` | Seconds; `0`/empty/negative → no timeout |
| `SANITIZER_USE_OPENAI_BRIDGE` | `false` | Enable the OpenAI bridge route |
| `SANITIZER_FORWARD_REASONING_EFFORT` | `true` | Carry Anthropic `output_config.effort` into the OpenAI `reasoning_effort` field |
| `THINK_OUTPUT_MODE` | `default` | `default` / `none` / `text` / `think_tag` / `bridge` |

### Reasoning effort

The bridge carries Anthropic's `output_config.effort` into the OpenAI
`reasoning_effort` field. Before this it was dropped here, so a client that
offered an effort control offered a no-op on this path (issue #24).

The two vocabularies differ. Claude Code / oh-my-gateway send the SDK's levels
(`low`/`medium`/`high`/`xhigh`/`max`); the OpenAI field defines
`minimal`/`low`/`medium`/`high`. So `xhigh` and `max` are **clamped to `high`**
(logged), and anything this bridge cannot map — an unknown level, a non-string,
a non-dict `output_config` — sends **no field at all** rather than a guess.
`none` is not a level on the Anthropic side (it disables extended thinking and
rides `thinking`), so it sends nothing either.

Set `SANITIZER_FORWARD_REASONING_EFFORT=false` if the upstream validates its
request schema strictly and rejects the extra field; the rest of the bridge is
unaffected.

**Whether the backend then honors it is a separate question.** Forwarding the
field is necessary but not sufficient — a model or a vLLM/SGLang build that
ignores `reasoning_effort` still reasons at its default. Advertise a strong
capability upstream only for models where the effort has been observed to change
the request the backend actually runs.

## Topology

```
client / gateway  (Anthropic /v1/messages)
  → :5501  sanitizer   (bridge=true, think_mode=default)
      → :3999  LiteLLM
          → vLLM / SGLang backends
```

## Run

```bash
# Local
PYTHONPATH=. SANITIZER_USE_OPENAI_BRIDGE=true \
  python -m uvicorn sanitizer.main:app --host 0.0.0.0 --port 5501

# Docker (sanitizer + LiteLLM in one container)
docker compose -f docker-compose-dev.yml up -d --build
```

## Tests

```bash
pip install "fastapi==0.115.*" "uvicorn[standard]==0.32.*" "httpx==0.27.*" pytest pytest-asyncio
python -m pytest sanitizer/tests/ -q
```

Tests are pure-logic where possible (dict-in/dict-out); route tests inject an
`httpx.MockTransport` in place of the real upstream. No prompt/response content
is ever logged.
