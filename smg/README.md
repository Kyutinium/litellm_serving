# SMG Router (optional, HTTP-only pilot)

This is an **opt-in alternative** to LiteLLM `simple-shuffle` and the existing
`balancer/` (`BALANCER_POOLS`). The default Docker Compose stack is unchanged.
It runs the [official SMG v1.13.0 gateway image](https://lightseek.org/smg/v1.13.0/getting-started/)
and keeps vLLM/SGLang's OpenAI HTTP server responsible for chat templates,
reasoning/DSML parsing, tool calls, and inference.

## 1. Configure same-model replicas

Each pool launches one SMG process with its own local port. All workers in a
pool **must serve the same canonical model ID** and compatible model weights,
tokenizer, prompt template, and prefix-cache settings. Use only endpoints from
the trusted inference network. No API keys are stored in this JSON.

```bash
export SMG_POOLS='{
  "glm-5.1-fp8": {
    "port": 3998,
    "workers": ["http://10.0.0.1:8089", "http://10.0.0.2:8089", "http://10.0.0.3:8089"],
    "policy": "cache_aware"
  },
  "qwen3.6-27b": {
    "port": 4000,
    "workers": ["http://10.0.1.1:8092", "http://10.0.1.2:8092"],
    "policy": "cache_aware"
  }
}'
docker compose -f docker-compose.yml -f docker-compose.smg.yml up -d smg
```

Port `3998` and `4000` are examples; `3996` (sanitizer), `3997` (legacy
balancer) and `3999` (LiteLLM) are reserved. The stack uses host networking,
so SMG binds to **127.0.0.1 only**. No extra GPU or changes to Oh My Gateway
are required. This Compose overlay must run on the same host/network namespace
as LiteLLM. For separate machines, use a trusted internal endpoint and adapt
the binding/network topology explicitly.

The launch supervisor validates **all** pool definitions before starting any
SMG process. One failed SMG child stops the others, allowing Docker to restart
the service. Pools are fixed at startup; update `SMG_POOLS` and recreate the
`smg` service to add/remove workers. In this first stage, HTTP cache-aware
routing tracks approximate prefixes, **not actual KV-cache events**; those
require a separate gRPC worker migration.

## 2. Verify readiness before changing traffic

```bash
curl -fsS http://127.0.0.1:3998/health
curl -fsS http://127.0.0.1:3998/readiness
curl -fsS http://127.0.0.1:3998/workers
```

The workers' `/v1/models` must report the model name LiteLLM sends to them.
For one chosen model in `litellm_config.yaml` (including all case aliases),
change **only** `api_base` to the new local SMG endpoint:

```yaml
- model_name: glm-5.1-fp8
  litellm_params:
    model: hosted_vllm/glm-5.1-fp8
    api_base: http://127.0.0.1:3998/v1
    api_key: EMPTY
    max_tokens: 131072
```

Because `Dockerfile` copies `litellm_config.yaml` into the image, apply the
change with a rebuild after the SMG endpoint is ready:

```bash
docker compose -f docker-compose.yml -f docker-compose.smg.yml up -d --build litellm
```

For the same pool, **remove that model's other LiteLLM deployment entries**
from random selection (or point its aliases at the same SMG URL), so the pool
is selected only once by SMG. Do not chain `balancer/` and SMG together.
The request path remains `gateway → sanitizer(:3996) → LiteLLM(:3999) →
SMG(:3998) → vLLM`, preserving the sanitizer's Anthropic/OpenAI bridge.

SMG is launched with `--disable-retries` to avoid hidden replay below LiteLLM.
Keep the outer LiteLLM retry/fallback policy under review; a request may already
be generating when a timeout happens, so **do not blindly retry streamed
requests**. Verify production client and upstream timeouts are compatible with
SMG's default 1800-second request timeout. Cache-aware HTTP routing parses incoming
request bodies and can add CPU/memory cost on large multimodal prompts.

## 3. A/B performance and compatibility

Keep a separate unmodified endpoint for the baseline, then test one routing
method at a time with identical vLLM workers, concurrency, and input lengths:

1. LiteLLM `simple-shuffle` across the 3–4 replicas.
2. Existing `balancer/` with `BALANCER_POOLS` and one LiteLLM deployment.
3. SMG `cache_aware` and one LiteLLM deployment.

The benchmark works with any OpenAI-compatible `/v1` URL. It sends a common
prefix, unique user suffixes, and streaming completions, printing aggregate
TTFT p50/p95/p99, request throughput, failures, and completion-token throughput
when the provider returns streamed usage. It **does not** claim to measure
decode ITL or actual KV-cache hit rate; collect those directly from vLLM
metrics, alongside GPU utilization and request queues.

```bash
export LITELLM_API_KEY='<key>'
python scripts/benchmark_smg.py \
  --model glm-5.1-fp8 --requests 100 --concurrency 16 \
  --target shuffle=http://127.0.0.1:4010/v1 \
  --target balancer=http://127.0.0.1:4011/v1 \
  --target smg=http://127.0.0.1:4012/v1
```

The `4010`–`4012` URLs above are **illustrative test endpoints**, not services
created by the overlay. Route each to the same model pool through its respective
method. `--prefix-file` accepts a UTF-8 common prefix for your representative
workload. The benchmark needs `httpx` (`pip install httpx`); no request content
or keys are written to its output.

Before production enablement, additionally verify:

- Chat/completions SSE delta order, disconnect/cancellation, and long prefill.
- Thinking/reasoning visibility, model-specific effort levels, and DSML tags.
- Tool calls with incremental JSON arguments and multimodal image requests.
- A dead vLLM replica, LiteLLM retry behavior, and router restart/draining.
- Prefix-caching enabled on every worker; cache reuse and P95/P99 TTFT under
  realistic multi-turn load (not just one repeated prompt).

## 4. Rollback and tests

To roll back, restore the prior `api_base` / deployment list in LiteLLM's
configuration and rebuild/recreate LiteLLM **before** stopping SMG (`config.yaml`
is copied at Docker image build time). The unchanged
`docker-compose.yml` still starts the original stack. Stop only the opt-in
router service afterwards:

```bash
docker compose -f docker-compose.yml -f docker-compose.smg.yml up -d --build litellm
docker compose -f docker-compose.yml -f docker-compose.smg.yml stop smg
```

Configuration and benchmark-format tests (no GPU/SMG needed):

```bash
python -m unittest discover -s smg/tests -v
```

SMG docs: [HTTP workers](https://lightseek.org/smg/v1.13.0/getting-started/multiple-workers/),
[cache-aware routing](https://lightseek.org/smg/v1.13.0/concepts/routing/cache-aware/),
[HTTP streaming](https://lightseek.org/smg/v1.13.0/concepts/performance/request-streaming/).
