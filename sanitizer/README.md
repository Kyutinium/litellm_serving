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
| `SANITIZER_EFFORT_SUPPORTED` | *(unset)* | Effort levels the upstream accepts, e.g. `low,medium,xhigh`. Unset = forward verbatim |
| `THINK_OUTPUT_MODE` | `default` | `default` / `none` / `text` / `think_tag` / `bridge` |

### Reasoning effort

The bridge carries Anthropic's `output_config.effort` into the OpenAI
`reasoning_effort` field. Before this it was dropped here, so a client that
offered an effort control offered a no-op on this path (issue #24).

**The level is preserved, not normalized.** Claude Code / oh-my-gateway send the
SDK's levels (`low`/`medium`/`high`/`xhigh`/`max`) and each one is forwarded
verbatim. vLLM's `ChatCompletionRequest` and SGLang's OpenAI protocol both accept
the full `none|minimal|low|medium|high|xhigh|max` string set, and the subset that
actually works is decided **per model** by its chat template: some current Qwen
templates take `low|medium|xhigh` and reject `high`, while `xhigh`/`max` carry
real meaning on others. Rewriting a level in this bridge — which does not know
which model the request lands on — would turn a working `xhigh` request into a
400. Per-model remapping belongs to the layer that knows the model: a LiteLLM
provider transform, or the serving template itself.

`minimal` is accepted as tolerant input (the schemas above define it), but it is
not a Claude Code level and nothing upstream of this bridge emits it. Anything
the bridge does not recognize — an unknown level, a non-string, a non-dict
`output_config` — sends **no field at all** rather than a guess. `none` is not a
level on the Anthropic side (it disables extended thinking and rides `thinking`),
so it sends nothing either.

Set `SANITIZER_FORWARD_REASONING_EFFORT=false` if the upstream validates its
request schema strictly and rejects the extra field; the rest of the bridge is
unaffected.

### Declaring what the upstream accepts

A vLLM build answers a level it does not know with a **400 that takes the whole
turn**, not by ignoring it:

```
Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
```

So `high` — the level a caller using the OpenAI-standard vocabulary is most
likely to send — kills every request on that upstream (issue #26). Which levels
a model accepts comes from its chat template, which this proxy cannot read off
the request, so it is declared:

```bash
SANITIZER_EFFORT_SUPPORTED=low,medium,xhigh
```

With the set declared, an unsupported level is clamped to the nearest supported
one, **preferring upward** — `high` and `max` become `xhigh`, `minimal` becomes
`low` — because reasoning less than the caller asked for is the worse of the two
failures. When nothing above is supported the nearest level below is used. The
clamp is logged.

Unset or empty means **forward verbatim** (the default): normalizing a level
without knowing the model is a guess, and a wrong guess turns a working request
into a 400 — which is what the earlier blanket `xhigh`→`high` mapping did. An
entry outside the known scale is dropped with a warning, and a value that leaves
no usable level also forwards verbatim, so a typo degrades to the default rather
than to 400s.

This applies on both ingress paths. On the bridge it picks the level the
translated body carries; on the byte-for-byte relay it is the **one documented
exception** — a chat-completions POST whose `reasoning_effort` needs changing is
re-encoded with just that field replaced. Any other body, an unparseable body, a
level already supported, and an unset switch all relay the original bytes.

**Whether the field survives LiteLLM, and whether the backend then honors it, are
separate questions.** Forwarding it here is necessary but not sufficient:
`litellm_config.yaml` sets a global `drop_params: true`, so LiteLLM's own
provider-capability judgment can still drop `reasoning_effort` before the
upstream sees it — and a model or a vLLM/SGLang build that ignores the field
reasons at its default anyway. Advertise a strong capability upstream only for
models where the effort has been observed in the request the backend actually
runs.

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
