"""Path gate (oh-my-gateway#286): a Messages-only client stays on the Messages API.

Pinned:
- unset ``SANITIZER_PUBLIC_PATHS`` changes nothing (every path relays as before);
- set, the listed prefixes and ``/health`` answer; every other path — the relay of
  ``/v1/models``, ``/model/info``, LiteLLM admin routes, this proxy's ``/openapi.json``,
  ``/docs``, ``/sanitizer/effort-levels`` — is a plain 404 and never reaches the upstream;
- the internal header with the right token opens them; a wrong or missing token does not,
  and with no token configured nothing opens (fail closed);
- non-canonical paths (``..``, ``//``, encoded ``/``/``.``/``\\``) are refused even under a
  public prefix, so the upstream cannot normalise them into a private path;
- the internal header is never relayed upstream.
"""

import asyncio

import httpx
import pytest

from sanitizer.main import app

TOKEN = "s3cr3t-internal"


def run(coro):
    return asyncio.run(coro)


def _client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def _get(path, **headers):
    async with _client() as client:
        return await client.get(path, headers=headers)


@pytest.fixture
def gated(monkeypatch):
    monkeypatch.setenv("SANITIZER_PUBLIC_PATHS", "/v1/messages")
    monkeypatch.setenv("SANITIZER_INTERNAL_TOKEN", TOKEN)
    monkeypatch.setenv("SANITIZER_EFFORT_PROBE", "false")


def test_unset_gate_changes_nothing(upstream, monkeypatch):
    monkeypatch.delenv("SANITIZER_PUBLIC_PATHS", raising=False)
    upstream.set_handler(lambda request: httpx.Response(200, json={"data": []}))
    assert run(_get("/model/info")).status_code == 200
    assert run(_get("/openapi.json")).status_code == 200
    assert len(upstream.requests) == 1


@pytest.mark.parametrize(
    "path",
    ["/v1/models", "/model/info", "/key/generate", "/user/list", "/openapi.json", "/docs", "/redoc",
     "/sanitizer/effort-levels", "/v1/chat/completions", "/v1/messagesX"],
)
def test_private_paths_are_404_and_never_reach_the_upstream(upstream, gated, path):
    upstream.set_handler(lambda request: httpx.Response(200, json={"leaked": True}))
    resp = run(_get(path))
    assert resp.status_code == 404
    assert "leaked" not in resp.text
    assert upstream.requests == []


def test_public_prefix_and_health_answer(upstream, gated):
    upstream.set_handler(lambda request: httpx.Response(200, json={"input_tokens": 1}))

    async def _do():
        async with _client() as client:
            counted = await client.post("/v1/messages/count_tokens", json={"model": "m", "messages": []})
            health = await client.get("/health")
            return counted, health

    counted, health = run(_do())
    assert counted.status_code == 200 and health.status_code == 200
    assert str(upstream.requests[0].url).endswith("/v1/messages/count_tokens")


def test_the_internal_token_opens_private_paths_and_is_not_relayed(upstream, gated):
    upstream.set_handler(lambda request: httpx.Response(200, json={"data": [{"id": "m"}]}))
    resp = run(_get("/v1/models", **{"X-Sanitizer-Internal": TOKEN}))
    assert resp.status_code == 200
    assert "x-sanitizer-internal" not in {k.lower() for k in upstream.requests[0].headers}
    assert run(_get("/sanitizer/effort-levels", **{"X-Sanitizer-Internal": TOKEN})).status_code == 200


@pytest.mark.parametrize("given", ["", "wrong", TOKEN + "x", TOKEN[:-1]])
def test_a_wrong_or_missing_token_does_not_open(upstream, gated, given):
    headers = {"X-Sanitizer-Internal": given} if given else {}
    assert run(_get("/model/info", **headers)).status_code == 404
    assert upstream.requests == []


def test_without_a_configured_token_nothing_private_opens(upstream, monkeypatch):
    monkeypatch.setenv("SANITIZER_PUBLIC_PATHS", "/v1/messages")
    monkeypatch.delenv("SANITIZER_INTERNAL_TOKEN", raising=False)
    assert run(_get("/v1/models", **{"X-Sanitizer-Internal": ""})).status_code == 404
    assert upstream.requests == []


@pytest.mark.parametrize(
    "raw",
    [
        "/v1/messages/../model/info",
        "/v1/messages/./../key/generate",
        "/v1/messages//count_tokens",
        "/v1/messages/%2e%2e/model/info",
        "/v1/messages%2f..%2fmodel%2finfo",
        "/v1/messages/..%5cmodel",
    ],
)
def test_non_canonical_paths_are_refused_even_under_a_public_prefix(upstream, gated, raw):
    async def _do():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = client.build_request("POST", "http://test/x", json={})
            request.url = httpx.URL("http://test" + raw)
            return await client.send(request)

    resp = run(_do())
    assert resp.status_code == 404, (raw, resp.status_code)
    assert upstream.requests == []


def test_a_malformed_public_list_is_refused(monkeypatch):
    from sanitizer.config import get_public_paths

    monkeypatch.setenv("SANITIZER_PUBLIC_PATHS", "v1/messages")
    with pytest.raises(ValueError):
        get_public_paths()


@pytest.mark.parametrize(
    ("path", "raw"),
    [
        # what uvicorn hands the app: ``path`` percent-decoded but NOT dot-normalised
        ("/v1/messages/../model/info", b"/v1/messages/../model/info"),
        ("/v1/messages/./count_tokens", b"/v1/messages/./count_tokens"),
        ("/v1/messages/..", b"/v1/messages/.."),
        ("/v1/messages//count_tokens", b"/v1/messages//count_tokens"),
        ("/v1/messages/../model/info", b"/v1/messages/%2e%2e/model/info"),
        ("/v1/messages/../model/info", b"/v1/messages%2f..%2fmodel%2finfo"),
        ("/v1/messages/..\\model", b"/v1/messages/..%5cmodel"),
    ],
)
def test_the_gate_refuses_non_canonical_scopes(gated, path, raw):
    """Direct on the ASGI scope: an HTTP client would normalise ``..`` before sending,
    so this is the only honest way to pin what uvicorn passes through."""
    from sanitizer.access import allowed

    scope = {"type": "http", "path": path, "raw_path": raw, "headers": []}
    assert allowed(scope) is False
    # ...even with the internal token: a canonical path is required for every caller
    with_token = {**scope, "headers": [(b"x-sanitizer-internal", TOKEN.encode())]}
    assert allowed(with_token) is False


@pytest.mark.parametrize(
    ("path", "expected"),
    [("/v1/messages", True), ("/v1/messages/", True), ("/v1/messages/count_tokens", True),
     ("/health", True), ("/v1/messagesX", False), ("/v1", False), ("/", False)],
)
def test_public_prefix_matching(gated, path, expected):
    from sanitizer.access import allowed

    assert allowed({"type": "http", "path": path, "raw_path": path.encode(), "headers": []}) is expected
