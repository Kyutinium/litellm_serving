"""Smoke: the effort path a client actually takes, end to end through a real
sanitizer process.

Boots a fake LiteLLM (vLLM-shaped answers: a Qwen3.x template that takes
``low|medium|xhigh`` and 400s on anything else) and the real sanitizer app in a
subprocess with the production env (bridge on), then checks the contract the
gateway relies on:

1. the startup probe learns the model's levels using ``LITELLM_MASTER_KEY``;
2. ``GET /v1/models`` publishes them as ``effort_levels``;
3. a ``POST /v1/messages`` with ``output_config.effort: high`` (non-stream and
   stream) reaches the upstream on ``/v1/chat/completions`` with
   ``reasoning_effort`` clamped to ``xhigh`` — the level the model takes;
4. with the probe off, the first ``high`` is answered by the upstream's 400,
   learned, and retried once as ``xhigh`` — the client sees a 200.

No real model server is needed; run it anywhere the sanitizer's deps are
installed::

    python scripts/effort_path_smoke.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
QWEN = "qwen3.6-27b"
TAKES = ("low", "medium", "xhigh")
KEY = "sk-smoke"

FAKE_UPSTREAM = r'''
import json, os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
app = FastAPI()
LOG = os.environ["SMOKE_LOG"]
QWEN, TAKES, KEY = %r, %r, %r

def _log(entry):
    with open(LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")

@app.get("/v1/models")
async def models(request: Request):
    _log({"path": "/v1/models", "auth": request.headers.get("authorization")})
    return {"object": "list", "data": [{"id": QWEN, "object": "model"}, {"id": "glm-5-fp8", "object": "model"}]}

@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    effort = body.get("reasoning_effort")
    _log({"path": "/v1/chat/completions", "model": body.get("model"), "reasoning_effort": effort,
          "stream": bool(body.get("stream")), "auth": request.headers.get("authorization")})
    if request.headers.get("authorization") != f"Bearer {KEY}":
        return JSONResponse(status_code=401, content={"error": {"message": "Authentication Error"}})
    if body.get("model") == QWEN and effort is not None and effort not in TAKES:
        return JSONResponse(status_code=400, content={"error": {"message":
            f"Unexpected reasoning effort {effort}. Supported types are xhigh (default), medium, and low.", "type": "invalid_request_error"}})
    text = f"effort={effort}"
    if body.get("stream"):
        async def gen():
            yield "data: " + json.dumps({"id": "c1", "object": "chat.completion.chunk", "model": body["model"],
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]}) + "\n\n"
            yield "data: " + json.dumps({"id": "c1", "object": "chat.completion.chunk", "model": body["model"],
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"id": "c1", "object": "chat.completion", "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
'''


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=1.0)
            return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise SystemExit(f"never came up: {url}")


def _log_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _messages_body(effort: str, stream: bool) -> dict:
    return {"model": QWEN, "max_tokens": 16, "stream": stream,
            "messages": [{"role": "user", "content": "hi"}], "output_config": {"effort": effort}}


def _run_phase(probe: bool, workdir: Path) -> None:
    up_port, san_port = _free_port(), _free_port()
    log = workdir / f"upstream-{'probe' if probe else 'noprobe'}.jsonl"
    fake_py = workdir / "fake_upstream.py"
    fake_py.write_text(FAKE_UPSTREAM % (QWEN, TAKES, KEY))
    env = {**os.environ, "SMOKE_LOG": str(log), "PYTHONPATH": str(ROOT)}
    upstream = subprocess.Popen([sys.executable, "-m", "uvicorn", "fake_upstream:app", "--port", str(up_port),
                                 "--log-level", "warning"], cwd=workdir, env=env)
    san_env = {**env,
               "UPSTREAM_BASE_URL": f"http://127.0.0.1:{up_port}",
               "SANITIZER_USE_OPENAI_BRIDGE": "true",  # production setting under test
               "SANITIZER_EFFORT_PROBE": "true" if probe else "false",
               "LITELLM_MASTER_KEY": KEY,
               "THINK_OUTPUT_MODE": "none"}
    san_env.pop("SANITIZER_EFFORT_VOCABULARY", None)
    sanitizer = subprocess.Popen([sys.executable, "-m", "uvicorn", "sanitizer.main:app", "--port", str(san_port),
                                  "--log-level", "warning"], cwd=ROOT, env=san_env)
    try:
        _wait(f"http://127.0.0.1:{up_port}/v1/models")
        _wait(f"http://127.0.0.1:{san_port}/health")
        base = f"http://127.0.0.1:{san_port}"
        headers = {"authorization": f"Bearer {KEY}", "content-type": "application/json"}
        label = "probe on " if probe else "probe off"

        if probe:
            # 1. the probe learns the model's set (starts ~2 s after boot)
            deadline = time.time() + 30
            table = {}
            while time.time() < deadline:
                table = httpx.get(f"{base}/sanitizer/effort-levels").json()
                if QWEN in table:
                    break
                time.sleep(0.5)
            assert table.get(QWEN, {}).get("levels") == list(TAKES), table
            assert table[QWEN]["source"] in ("probe-400", "probe"), table
            probe_calls = [e for e in _log_entries(log) if e["path"] == "/v1/chat/completions"]
            assert probe_calls and all(e["auth"] == f"Bearer {KEY}" for e in probe_calls), "probe must use LITELLM_MASTER_KEY"
            print(f"[{label}] probe learned {QWEN} = {'|'.join(TAKES)} via {table[QWEN]['source']} ({len(probe_calls)} request(s))")

            # 2. /v1/models publishes it
            rows = {r["id"]: r for r in httpx.get(f"{base}/v1/models", headers=headers).json()["data"]}
            assert rows[QWEN]["effort_levels"] == list(TAKES), rows[QWEN]
            assert "effort_levels" not in rows["glm-5-fp8"] or rows["glm-5-fp8"]["effort_levels"], rows
            print(f"[{label}] GET /v1/models row carries effort_levels={rows[QWEN]['effort_levels']}")

        # 3./4. /v1/messages with a level the template rejects
        for stream in (False, True):
            before = len(_log_entries(log))
            with httpx.stream("POST", f"{base}/v1/messages", headers=headers,
                              json=_messages_body("high", stream), timeout=20.0) as resp:
                status = resp.status_code
                payload = b"".join(resp.iter_bytes())
            assert status == 200, (status, payload[:300])
            sent = [e["reasoning_effort"] for e in _log_entries(log)[before:] if e["path"] == "/v1/chat/completions"]
            if probe:
                assert sent == ["xhigh"], sent  # clamped up front from the learned table
            else:
                # first request: learned from the 400 and retried once
                assert sent == (["high", "xhigh"] if stream is False else ["xhigh"]), sent
            body_text = payload.decode()
            assert "effort=xhigh" in body_text, body_text[:300]
            print(f"[{label}] POST /v1/messages effort=high stream={stream}: upstream saw reasoning_effort={sent}, client got 200 with the model's answer")
    finally:
        for proc in (sanitizer, upstream):
            proc.terminate()
        for proc in (sanitizer, upstream):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        _run_phase(probe=False, workdir=workdir)
        _run_phase(probe=True, workdir=workdir)
    print("OK: /v1/messages output_config.effort -> reasoning_effort -> backend, clamped to the learned set")


if __name__ == "__main__":
    main()
