"""Compare the same OpenAI-compatible model via LiteLLM shuffle/balancer/SMG.

Only aggregate latencies and counts are printed. No prompts or keys are logged.
Use a stable shared prefix to exercise prefix-cache-aware routing, and compare
against the same load and model on each independent endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from dataclasses import dataclass

@dataclass
class Result:
    ttft: float | None
    elapsed: float
    tokens: int | None
    error: str | None = None


def percentile(values: list[float], percentage: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(len(ordered) * percentage / 100) - 1
    return ordered[max(0, index)]


def streamed_delta(line: str) -> tuple[bool, int | None]:
    """Return (has output, completion token usage) for one SSE data line."""
    if not line.startswith("data:"):
        return False, None
    raw = line[5:].strip()
    if raw == "[DONE]":
        return False, None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, None
    usage = data.get("usage") or {}
    tokens = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None
    for choice in data.get("choices", []):
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"):
            return True, tokens
    return False, tokens


async def run_request(
    client: object, endpoint: str, key: str, model: str,
    common_prefix: str, number: int, max_tokens: int,
) -> Result:
    import httpx

    started = time.perf_counter()
    first: float | None = None
    tokens: int | None = None
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": common_prefix},
            {"role": "user", "content": f"Respond to the request number {number} in one paragraph."},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with client.stream(
            "POST", endpoint.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {key}"}, json=body,
        ) as response:
            if response.status_code != 200:
                return Result(None, time.perf_counter() - started, None, f"HTTP {response.status_code}")
            async for line in response.aiter_lines():
                emitted, count = streamed_delta(line)
                if emitted and first is None:
                    first = time.perf_counter() - started
                if count is not None:
                    tokens = count
        return Result(first, time.perf_counter() - started, tokens,
                      None if first is not None else "no streamed output")
    except httpx.HTTPError as exc:
        return Result(first, time.perf_counter() - started, tokens, type(exc).__name__)


def summary(label: str, results: list[Result], wall_seconds: float) -> dict:
    good = [r for r in results if r.error is None]
    ttfts = [r.ttft * 1000 for r in good if r.ttft is not None]
    durations = [r.elapsed for r in good]
    known_tokens = [r.tokens for r in good if r.tokens is not None]
    result = {
        "target": label,
        "success": len(good),
        "failed": len(results) - len(good),
        "wall_seconds": round(wall_seconds, 3),
        "requests_per_second": round(len(good) / wall_seconds, 3) if wall_seconds else None,
        "ttft_ms_p50": percentile(ttfts, 50),
        "ttft_ms_p95": percentile(ttfts, 95),
        "ttft_ms_p99": percentile(ttfts, 99),
        "mean_response_seconds": round(statistics.mean(durations), 3) if durations else None,
        "completion_tokens_per_second": (
            round(sum(known_tokens) / wall_seconds, 3)
            if len(known_tokens) == len(good) and good and wall_seconds else None
        ),
        "errors": {code: sum(r.error == code for r in results) for code in sorted({r.error for r in results if r.error})},
    }
    return result


async def benchmark(args) -> None:
    import httpx

    prefix = open(args.prefix_file, encoding="utf-8").read() if args.prefix_file else (
        "You are a helpful assistant. Always answer clearly and precisely. " * 200
    )
    key = os.environ.get(args.key_env, "")
    if not key:
        raise SystemExit(f"Set {args.key_env} before benchmarking")
    timeout = httpx.Timeout(args.timeout, connect=10)
    async with httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=args.concurrency + 2)) as client:
        for target in args.target:
            if "=" not in target:
                raise SystemExit("--target must look like label=http://host:port/v1")
            label, url = target.split("=", 1)
            for i in range(args.warmup):
                await run_request(client, url, key, args.model, prefix, -i, args.max_tokens)
            sem = asyncio.Semaphore(args.concurrency)

            async def one(i: int) -> Result:
                async with sem:
                    return await run_request(client, url, key, args.model, prefix, i, args.max_tokens)

            started = time.perf_counter()
            results = await asyncio.gather(*(one(i) for i in range(args.requests)))
            print(json.dumps(summary(label, results, time.perf_counter() - started), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", action="append", required=True, help="label=http://host:port/v1 (repeatable)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--key-env", default="LITELLM_API_KEY")
    parser.add_argument("--prefix-file", help="UTF-8 shared-prefix text; never printed")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if min(args.requests, args.concurrency, args.max_tokens) < 1 or args.warmup < 0:
        parser.error("requests, concurrency and max-tokens must be positive; warmup must be nonnegative")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
