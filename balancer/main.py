"""FastAPI app: ``/<pool>/<path>`` → the least-loaded replica's ``/<path>``.

* Bodies, query strings, status codes and SSE streams are relayed as-is (hop-by-hop
  headers dropped, like the sanitizer's passthrough).
* The reservation taken by :meth:`Pool.acquire` is released when the response is done
  — including a client that disconnects mid-stream.
* A replica that cannot be connected to — refused (``ConnectError``) or no answer within
  ``BALANCER_CONNECT_TIMEOUT`` (``ConnectTimeout``: a DROP firewall, a blackholed IP) —
  is marked down and the request goes to the next one (each replica at most once).
  Nothing was sent to it, so moving on is safe. Anything after the connection is up (a
  5xx, a ``ReadTimeout`` mid-request) is the replica's answer and is not retried here:
  the request may already be running there. LiteLLM's own retry/fallback sees it.
* ``GET /balancer/status`` shows per-replica scores and counters; ``/health`` is liveness.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Dict, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from .config import get_pools, get_tuning
from .pool import Backend, Pool, Weights, parse_vllm_metrics

logging.basicConfig(level=logging.INFO)
# httpx logs every request at INFO — that is one line per replica per scrape interval.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("balancer.main")

_HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
        "accept-encoding",
    }
)
_DROP_FROM_RESPONSE = (_HOP_BY_HOP - {"accept-encoding"}) | {"content-encoding"}
_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]

app = FastAPI(title="vLLM load balancer", version="0.1.0")

POOLS: Dict[str, Pool] = {}
_scraper: Optional[asyncio.Task] = None


# Connecting is the only phase that is safe to fail over: nothing has been sent yet.
_CONNECT_FAILURES = (httpx.ConnectError, httpx.ConnectTimeout)


def _make_client(timeout) -> httpx.AsyncClient:
    """Factory for the relay/scrape client; tests swap in a ``MockTransport``."""
    return httpx.AsyncClient(timeout=timeout)


def build_pools() -> Dict[str, Pool]:
    tuning = get_tuning()
    weights = Weights(waiting=tuning.weight_waiting, running=tuning.weight_running, kv=tuning.weight_kv)
    return {
        name: Pool.from_urls(name, urls, weights=weights, down_seconds=tuning.down_seconds)
        for name, urls in get_pools().items()
    }


async def scrape_once(pools: Dict[str, Pool], client: httpx.AsyncClient) -> None:
    async def one(pool: Pool, b: Backend) -> None:
        try:
            res = await client.get(f"{b.url}/metrics")
        except _CONNECT_FAILURES as exc:
            pool.mark_down(b, f"metrics: {exc.__class__.__name__}")
            b.metrics_ok = False
            return
        except httpx.HTTPError:
            # Slow or odd answer: keep the replica, fall back to our own count.
            b.metrics_ok = False
            return
        if res.status_code != 200:
            pool.record_metrics(b, None)
            return
        pool.record_metrics(b, parse_vllm_metrics(res.text))

    await asyncio.gather(*(one(p, b) for p in pools.values() for b in p.backends))


async def _scrape_forever() -> None:
    tuning = get_tuning()
    timeout = httpx.Timeout(tuning.scrape_timeout, connect=min(tuning.scrape_timeout, tuning.connect_timeout))
    async with _make_client(timeout) as client:
        while True:
            try:
                await scrape_once(POOLS, client)
            except Exception:  # noqa: BLE001 — never let the scraper die
                logger.exception("metrics scrape failed")
            await asyncio.sleep(tuning.scrape_interval)


@app.on_event("startup")
async def _startup() -> None:
    global _scraper
    POOLS.clear()
    POOLS.update(build_pools())
    if not POOLS:
        logger.warning("BALANCER_POOLS is empty — every /<pool>/… request will 404")
    for pool in POOLS.values():
        logger.info("pool %s: %s", pool.name, ", ".join(b.url for b in pool.backends))
    _scraper = asyncio.create_task(_scrape_forever())


@app.on_event("shutdown")
async def _shutdown() -> None:
    if _scraper is not None:
        _scraper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _scraper


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/balancer/status")
async def status():
    return {"pools": [p.status() for p in POOLS.values()]}


def _request_headers(request: Request) -> Dict[str, str]:
    return {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}


def _response_headers(res: httpx.Response) -> Dict[str, str]:
    return {k: v for k, v in res.headers.items() if k.lower() not in _DROP_FROM_RESPONSE}


@app.api_route("/{pool_name}/{path:path}", methods=_METHODS)
async def relay(pool_name: str, path: str, request: Request):
    pool = POOLS.get(pool_name)
    if pool is None:
        return JSONResponse({"error": {"message": f"unknown pool {pool_name!r}", "type": "not_found"}}, 404)
    body = await request.body()
    headers = _request_headers(request)
    query = request.url.query
    tuning = get_tuning()
    # Long reads (an inference may take minutes), short connects (a dead host must not
    # hold the request for the whole read budget before we try the next replica).
    client = _make_client(httpx.Timeout(tuning.request_timeout, connect=tuning.connect_timeout))
    tried: set = set()
    while True:
        backend = pool.acquire(frozenset(tried))
        if backend is None:
            await client.aclose()
            return JSONResponse(
                {"error": {"message": f"pool {pool_name!r}: no replica reachable", "type": "service_unavailable"}},
                503,
            )
        tried.add(backend.url)
        url = f"{backend.url}/{path}" + (f"?{query}" if query else "")
        upstream = client.build_request(request.method, url, headers=headers, content=body)
        try:
            res = await client.send(upstream, stream=True)
        except _CONNECT_FAILURES as exc:
            pool.release(backend)
            pool.mark_down(backend, f"connect: {exc.__class__.__name__}")
            continue
        except BaseException:
            pool.release(backend)
            await client.aclose()
            raise
        pool.mark_ok(backend)
        return _stream_back(pool, backend, client, res)


def _stream_back(pool: Pool, backend: Backend, client: httpx.AsyncClient, res: httpx.Response) -> Response:
    released = False

    async def done() -> None:
        nonlocal released
        if released:
            return
        released = True
        pool.release(backend)
        await res.aclose()
        await client.aclose()

    async def body():
        try:
            # Decoded bytes: content-encoding is dropped from the relayed headers.
            async for chunk in res.aiter_bytes():
                yield chunk
        finally:
            # Runs on normal end, upstream error and client disconnect alike.
            await done()

    return StreamingResponse(
        body(),
        status_code=res.status_code,
        headers=_response_headers(res),
        # Backstop for a response that is never iterated; ``done`` runs once.
        background=BackgroundTask(done),
    )
