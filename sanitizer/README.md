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
| `SANITIZER_EFFORT_VOCABULARY` | *(unset)* | **Manual override**: JSON map of model name → effort levels its upstream accepts. Normally unnecessary — the levels are learned (below). A declared entry wins over a learned one |
| `SANITIZER_EFFORT_PROBE` | `true` | At startup, ask the upstream which effort levels each listed model takes (1-token requests, in the background). `false` = learn only from real 400s |
| `SANITIZER_UPSTREAM_API_KEY` | *(`LITELLM_MASTER_KEY` if set)* | Bearer the probe uses for its own upstream requests |
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

### The vocabulary is learned from the upstream

Which levels a served model takes is a property of its chat template, and the
upstream states that fact itself, so nothing has to be written down per model:

- **On a real 400.** A vLLM build answers an unsupported level with
  `Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.`
  The sanitizer reads that sentence, records the model's set, and **retries the
  request once** with the nearest level the model takes (upward first). The
  caller sees a 200; the next request for that model is clamped up front. Both
  ingress paths do this — the bridge (`/v1/messages`) and the byte-for-byte relay
  (`/v1/chat/completions`). A 400 that says nothing about effort is relayed as-is,
  never retried blindly.
- **At startup (`SANITIZER_EFFORT_PROBE`, default on).** In the background the
  sanitizer lists the upstream's models and asks each one: first `minimal` (a
  level every schema defines but templates rarely list — a template that names
  its set answers with the whole vocabulary in one 400), then, only if that was
  inconclusive, one 1-token completion per level (200 = takes it, effort-related
  400 = does not). Models with a declared entry are skipped; an upstream that is
  still coming up is retried for ~10 minutes.
- **Published.** Whatever is known rides the relayed `GET /v1/models` as
  `effort_levels` on each model's row, and `GET /sanitizer/effort-levels` shows
  the table with each entry's source (`declared` / `probe` / `probe-400` /
  `upstream-400`). `oh-my-gateway` reads the rows through its model discovery
  (`MODEL_DISCOVERY_ENABLED=true`) and advertises the same set — so the composer
  offers exactly the levels this model takes without an operator copying a list.

The learned table is process-local; a restart relearns (one retry per model on
the request path, or the probe).

### Declaring what each model's upstream accepts (override)

A vLLM build answers a level it does not know with a **400 that takes the whole
turn**, not by ignoring it:

```
Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
```

So `high` — the level a caller using the OpenAI-standard vocabulary is most
likely to send — kills every request on that upstream (issue #26). Which levels
a model accepts comes from its chat template, and one sanitizer fronts a LiteLLM
that routes to **many** models whose accepted subsets differ, so the declaration
is per model, keyed by the name exactly as clients send it:

```bash
SANITIZER_EFFORT_VOCABULARY='{"qwen3.6-27b": "low,medium,xhigh", "Qwen3.6-27B": "low,medium,xhigh"}'
```

LiteLLM names are case-sensitive and this repository registers case variants as
separate aliases, so list every alias a client may send (the config renderer
expands them from `models.yaml`). A model with no entry is forwarded verbatim.

With a model's set declared, an unsupported level is clamped to the nearest
declared one, **preferring upward** — `high` and `max` become `xhigh`, `minimal`
becomes `low` — because reasoning less than the caller asked for is the worse of
the two failures. When nothing above is declared the nearest level below is
used. The clamp is logged. Note what this does *not* do: on a three-level
upstream `high`/`xhigh`/`max` all arrive as `xhigh`, so they are deliberately
collapsed at the model input. Preventing the 400 is this proxy's job; telling
users which levels a model really offers belongs to the capability surface
upstream of it (the gateway's `/v1/models`).

Unset or empty means **forward verbatim** for every model (the default):
normalizing a level without knowing the model is a guess, and a wrong guess
turns a working request into a 400 — which is what the earlier blanket
`xhigh`→`high` mapping did.

**Validity is all-or-nothing.** A typo (`"low,medium,xhi"`) must not quietly
become a *narrower* declaration that clamps good levels away — dropping the
unknown entry and keeping the rest does exactly that. So an unknown level,
a wrong shape, or unparseable JSON makes the whole setting invalid: the process
**refuses to start**, and a request served under an invalid setting anyway (the
env changed under a running process) is forwarded verbatim for every model with
one warning. Invalid never narrows.

This applies on both ingress paths. On the bridge it picks the level the
translated body carries; on the byte-for-byte relay it is the **one documented
exception** — a chat-completions POST for a declared model whose
`reasoning_effort` needs changing is re-encoded with just that field replaced.
Any other body, an unparseable body, an undeclared model, a level already
declared, and an unset switch all relay the original bytes.

For a model with a declaration the bridge also sends LiteLLM's per-request
`allowed_openai_params: ["reasoning_effort"]`, so the field is forwarded verbatim
even where a provider transform would drop it under `drop_params: true`
(measured: LiteLLM 1.101 forwards it for `hosted_vllm/` and `openai/` entries by
itself; the pin covers other prefixes and versions). Undeclared models keep
LiteLLM's own judgment.

On the gateway side nothing needs declaring either: with `MODEL_DISCOVERY_ENABLED=true`
`oh-my-gateway` reads the `effort_levels` this proxy publishes on `/v1/models`,
advertises `capabilities.reasoning_effort: true` + `effort_levels` for that id, and
answers a level outside them with a 400 before it reaches this proxy. Its
`CLAUDE_CUSTOM_UPSTREAM_EFFORT_MODELS` is the matching manual override. This
sanitizer's clamp stays as the safety net for callers that do not go through the
gateway.

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
