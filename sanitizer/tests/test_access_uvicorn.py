"""Path gate end to end: real uvicorn in front, a real HTTP upstream behind (#286 review).

The ASGI-scope tests pin ``allowed()``; these pin what actually crosses the wire. Requests
are written as raw bytes on a socket so no client library normalises ``..`` or re-encodes
anything, and the fake upstream records each request-target exactly as it arrived. Pinned:

- public routes reach the upstream with the same path (and query) the client sent;
  streaming passes through unbuffered;
- every refused spelling — sub-path, trailing slash, other method, ``..``, ``//``,
  ``%2e``/``%2f``/``%5c``, double encoding, ``;``, encoded ``?``/``#`` — never reaches
  the upstream;
- with the token, the upstream sees the exact raw path (one decoding: LiteLLM's), the
  internal header is not relayed, and an upstream redirect is relayed, never followed.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

import pytest
import uvicorn

TOKEN = "s3cr3t-internal"
ROUTES = "POST /v1/messages, POST /v1/messages/count_tokens"

_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "m",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_SSE = [
    ("message_start", {"type": "message_start", "message": {**_MESSAGE, "content": [], "stop_reason": None}}),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
    ("message_stop", {"type": "message_stop"}),
]


class _Upstream(BaseHTTPRequestHandler):
    seen: ClassVar[list] = []  # shared across the module's servers; cleared per test

    def log_message(self, *args):
        pass

    def _record(self):
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length) if length else b""
        self.seen.append({"method": self.command, "target": self.path, "headers": dict(self.headers)})
        return body

    def _json(self, status, payload, extra=()):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        for key, value in extra:
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._record()
        if self.path.startswith("/redirect-me"):
            self._json(302, {}, [("location", "http://169.254.169.254/latest/meta-data")])
        else:
            self._json(200, {"data": [{"id": "m"}]})

    def do_POST(self):
        body = json.loads(self._record() or b"{}")
        if self.path.split("?")[0].endswith("/count_tokens"):
            self._json(200, {"input_tokens": 3})
            return
        if not body.get("stream"):
            self._json(200, _MESSAGE)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        for event, data in _SSE:
            chunk = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
            time.sleep(0.02)
        self.wfile.write(b"0\r\n\r\n")


@pytest.fixture(scope="module")
def _servers():
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("UPSTREAM_BASE_URL", f"http://127.0.0.1:{upstream.server_address[1]}")
        mp.setenv("SANITIZER_PUBLIC_ROUTES", ROUTES)
        mp.setenv("SANITIZER_INTERNAL_TOKEN", TOKEN)
        mp.setenv("SANITIZER_EFFORT_PROBE", "false")
        mp.setenv("SANITIZER_USE_OPENAI_BRIDGE", "false")

        from sanitizer.main import app

        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not server.started and time.time() < deadline:
            time.sleep(0.02)
        assert server.started, "uvicorn did not start"
        yield server.servers[0].sockets[0].getsockname()[1]
        server.should_exit = True
        thread.join(5)
    upstream.shutdown()


@pytest.fixture
def stack(_servers):
    _Upstream.seen.clear()
    return _servers, _Upstream.seen


def _raw(port, method, target, body=b"", headers=()):
    """Send exactly these bytes; return (status, headers, body)."""
    head = [f"{method} {target} HTTP/1.1", f"Host: 127.0.0.1:{port}", "Connection: close"]
    head += [f"{k}: {v}" for k, v in headers]
    if body or method in ("POST", "PUT", "PATCH"):
        head += ["content-type: application/json", f"content-length: {len(body)}"]
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall(("\r\n".join(head) + "\r\n\r\n").encode("latin-1") + body)
        data = b""
        while chunk := sock.recv(65536):
            data += chunk
    head_bytes, _, rest = data.partition(b"\r\n\r\n")
    lines = head_bytes.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1])
    hdrs = {k.lower(): v.strip() for k, _, v in (line.partition(":") for line in lines[1:])}
    if hdrs.get("transfer-encoding", "").lower() == "chunked":
        rest = _dechunk(rest)
    return status, hdrs, rest


def _dechunk(data):
    out = b""
    while data:
        size_line, _, data = data.partition(b"\r\n")
        size = int(size_line.split(b";")[0], 16)
        if size == 0:
            break
        out, data = out + data[:size], data[size + 2 :]
    return out


_BODY = json.dumps({"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}).encode()


def test_public_messages_reach_the_upstream_with_the_same_target(stack):
    port, seen = stack
    status, _, body = _raw(port, "POST", "/v1/messages?beta=true", _BODY)
    assert status == 200, body
    assert json.loads(body)["content"][0]["text"] == "hi"
    # the Messages route posts to the upstream's fixed /v1/messages (its query is not relayed —
    # pre-existing routes_messages behaviour, not the gate's)
    assert [s["target"] for s in seen] == ["/v1/messages"]

    status, _, body = _raw(port, "POST", "/v1/messages/count_tokens?beta=true", _BODY)
    assert status == 200 and json.loads(body) == {"input_tokens": 3}
    assert seen[-1]["target"] == "/v1/messages/count_tokens?beta=true"


def test_streaming_passes_through_the_gate(stack):
    port, _ = stack
    payload = json.dumps({**json.loads(_BODY), "stream": True}).encode()
    status, hdrs, body = _raw(port, "POST", "/v1/messages", payload)
    assert status == 200
    assert hdrs["content-type"].startswith("text/event-stream")
    assert b"message_stop" in body and b"text_delta" in body


def test_health_is_exactly_get_health(stack):
    port, seen = stack
    assert _raw(port, "GET", "/health")[0] == 200
    for method, target in [("GET", "/health/"), ("GET", "/health/x"), ("GET", "/healthz"), ("POST", "/health")]:
        assert _raw(port, method, target)[0] == 404, (method, target)
    assert seen == []


@pytest.mark.parametrize(
    ("method", "target"),
    [
        ("POST", "/v1/messages/"),
        ("POST", "/v1/messages/batches"),
        ("GET", "/v1/messages"),
        ("DELETE", "/v1/messages"),
        ("POST", "/V1/messages"),
        ("POST", "/v1/messages/../../model/info"),
        ("POST", "/v1/messages/./count_tokens"),
        ("POST", "//v1/messages"),
        ("POST", "/v1//messages"),
        ("POST", "/v1/messages/%2e%2e/%2e%2e/model/info"),
        ("POST", "/v1/messages%2f..%2f..%2fmodel%2finfo"),
        ("POST", "/v1/messages/..%5c..%5cmodel%5cinfo"),
        ("POST", "/v1/messages/%252e%252e/%252e%252e/model/info"),
        ("POST", "/v1/%6dessages"),
        ("POST", "/v1/messages;x"),
        ("POST", "/v1/messages%3F/../model/info"),
        ("POST", "/v1/messages%23/../model/info"),
        ("GET", "/v1/models"),
        ("GET", "/model/info"),
        ("GET", "/openapi.json"),
        ("GET", "/sanitizer/effort-levels"),
    ],
)
def test_refused_spellings_never_reach_the_upstream(stack, method, target):
    port, seen = stack
    status, _, body = _raw(port, method, target, _BODY if method == "POST" else b"")
    assert status in (400, 404), (target, status)  # 400: uvicorn refused the target itself
    assert b"api_base" not in body and b'"data"' not in body
    assert seen == [], (target, seen)


@pytest.mark.parametrize(
    "target",
    ["/v1/models", "/v1/models?return_wildcard_routes=false", "/model/info", "/v1/models/a~b.c-d_e"],
)
def test_with_the_token_the_upstream_sees_the_raw_path_and_no_header(stack, target):
    port, seen = stack
    status, _, _ = _raw(port, "GET", target, headers=[("X-Sanitizer-Internal", TOKEN)])
    assert status == 200
    assert seen[0]["target"] == target
    assert "x-sanitizer-internal" not in {k.lower() for k in seen[0]["headers"]}


@pytest.mark.parametrize(
    "target",
    ["/model/../v1/messages", "/v1/models/%2e%2e", "/v1/models/%252e", "/v1/models%3Fx", "/v1/models;x",
     "/v1/models/a%20b", "/v1/models/%00", "/v1/models/%C3%A9"],
)
def test_the_token_does_not_open_non_canonical_paths(stack, target):
    port, seen = stack
    status, _, _ = _raw(port, "GET", target, headers=[("X-Sanitizer-Internal", TOKEN)])
    assert status in (400, 404)
    assert seen == []


def test_an_upstream_redirect_is_relayed_not_followed(stack):
    port, seen = stack
    status, hdrs, _ = _raw(port, "GET", "/redirect-me", headers=[("X-Sanitizer-Internal", TOKEN)])
    assert status == 302
    assert hdrs["location"] == "http://169.254.169.254/latest/meta-data"
    assert [s["target"] for s in seen] == ["/redirect-me"]  # one request: not followed


# -- gate OFF: what the raw-path relay changed (#39 review: compatibility) ------
# Ordinary paths are relayed exactly as before. Only percent-encoded reserved
# characters differ: the old code re-embedded the DECODED path, so LiteLLM saw a
# different path than the client sent (``%2e%2e`` even became a real ``..`` and
# climbed out of ``/model``). Literal ``..`` is still normalised by httpx, as before.
_GATE_OFF = [
    # (client target, upstream target now, upstream target before this change)
    ("/v1/models?x=1", "/v1/models?x=1", "/v1/models?x=1"),
    ("/v1/models/a%20b", "/v1/models/a%20b", "/v1/models/a%20b"),
    ("/v1/models/caf%C3%A9", "/v1/models/caf%C3%A9", "/v1/models/caf%C3%A9"),
    ("/v1/models/a+b", "/v1/models/a+b", "/v1/models/a+b"),
    ("/v1/models;p=1", "/v1/models;p=1", "/v1/models;p=1"),
    ("/v1//models", "/v1//models", "/v1//models"),
    ("/model/info/../x", "/model/x", "/model/x"),
    ("/v1/models/a%2Fb", "/v1/models/a%2Fb", "/v1/models/a/b"),
    ("/v1/models%3Fx=1", "/v1/models%3Fx=1", "/v1/models?x=1?x=1"),
    ("/v1/models%23frag", "/v1/models%23frag", "/v1/models"),
    ("/model/%2e%2e/x", "/model/%2e%2e/x", "/x"),
    ("/v1/models/%25", "/v1/models/%25", "/v1/models/%"),
]


@pytest.mark.parametrize(("target", "now", "before"), _GATE_OFF)
def test_gate_off_relay_compatibility(stack, monkeypatch, target, now, before):
    port, seen = stack
    monkeypatch.delenv("SANITIZER_PUBLIC_ROUTES")
    status, _, _ = _raw(port, "GET", target)
    assert status == 200
    assert [s["target"] for s in seen] == [now], (target, seen, "before:", before)
