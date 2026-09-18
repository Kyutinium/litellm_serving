"""Upstream effort vocabulary: issue #26.

A vLLM build answers a level it does not know with a 400 that takes the whole
turn. ``high`` — what a caller using the OpenAI-standard vocabulary sends — is
outside the accepted set of an upstream whose own error names
``xhigh (default), medium, low``. These pin the declared-set clamp and, just as
importantly, that an undeclared set still forwards verbatim (issue #25).
"""

import asyncio
import json

import httpx
import pytest

from sanitizer.effort import EFFORT_SCALE, clamp_to_supported, supported_levels
from sanitizer.main import app
from sanitizer.openai_bridge import anthropic_request_to_openai_body

# The set the issue's upstream reports for ChatDRAGON-Medium (vLLM Qwen3.8-27b).
_VLLM = "low,medium,xhigh"


@pytest.fixture
def vllm_vocab(monkeypatch):
    monkeypatch.setenv("SANITIZER_EFFORT_SUPPORTED", _VLLM)


# --- the clamp itself --------------------------------------------------------


def test_no_declared_set_forwards_every_level_verbatim(monkeypatch):
    """The default must stay #25: normalizing blind is what broke Qwen before."""
    monkeypatch.delenv("SANITIZER_EFFORT_SUPPORTED", raising=False)
    assert supported_levels() == ()
    for level in EFFORT_SCALE:
        assert clamp_to_supported(level) == level


def test_high_clamps_up_to_xhigh_not_down_to_medium(vllm_vocab):
    """The issue's ask, and the direction matters.

    ``high`` sits between the supported ``medium`` and ``xhigh``. Reasoning less
    than the caller asked for is the worse miss, so the clamp goes up.
    """
    assert clamp_to_supported("high") == "xhigh"
    assert clamp_to_supported("max") == "xhigh"
    assert clamp_to_supported("minimal") == "low"


def test_supported_levels_are_untouched(vllm_vocab):
    for level in ("low", "medium", "xhigh"):
        assert clamp_to_supported(level) == level


def test_clamps_down_when_nothing_above_is_supported(monkeypatch):
    monkeypatch.setenv("SANITIZER_EFFORT_SUPPORTED", "low,medium")
    assert clamp_to_supported("xhigh") == "medium"
    assert clamp_to_supported("max") == "medium"


def test_a_level_outside_the_scale_is_not_invented(vllm_vocab):
    assert clamp_to_supported("bogus") is None


def test_declared_set_is_read_in_scale_order_not_env_order(monkeypatch):
    monkeypatch.setenv("SANITIZER_EFFORT_SUPPORTED", "xhigh, low , MEDIUM")
    assert supported_levels() == ("low", "medium", "xhigh")


def test_a_typo_cannot_silently_narrow_the_set(monkeypatch):
    """Dropping the unknown entry must not make the rest clamp good levels away."""
    monkeypatch.setenv("SANITIZER_EFFORT_SUPPORTED", "low,medium,xhi")
    assert supported_levels() == ("low", "medium")

    monkeypatch.setenv("SANITIZER_EFFORT_SUPPORTED", "nonsense,alsononsense")
    assert supported_levels() == ()
    assert clamp_to_supported("high") == "high", "no usable level -> forward as-is"


# --- the bridge path (the deployed one) --------------------------------------


def _messages_body(effort):
    return {
        "model": "ChatDRAGON-Medium",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": effort},
    }


def test_bridge_sends_a_level_the_upstream_accepts(vllm_vocab):
    out = anthropic_request_to_openai_body(_messages_body("high"))
    assert out["reasoning_effort"] == "xhigh"


def test_bridge_still_forwards_verbatim_without_a_declared_set(monkeypatch):
    monkeypatch.delenv("SANITIZER_EFFORT_SUPPORTED", raising=False)
    out = anthropic_request_to_openai_body(_messages_body("high"))
    assert out["reasoning_effort"] == "high"


def test_bridge_none_still_sends_nothing(vllm_vocab):
    """``none`` disables thinking on the Anthropic side; it is not a level."""
    assert "reasoning_effort" not in anthropic_request_to_openai_body(
        _messages_body("none")
    )


# --- the incident, end to end against an upstream that behaves like the real one


def _vllm_upstream(request):
    body = json.loads(request.content or b"{}")
    effort = body.get("reasoning_effort")
    if effort is not None and effort not in ("low", "medium", "xhigh"):
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": f"Unexpected reasoning effort {effort}. "
                    "Supported types are xhigh (default), medium, and low.",
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                }
            },
        )
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "model": body.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _asgi():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _post(path, payload):
    async def _do():
        async with _asgi() as client:
            return await client.post(path, json=payload)

    return asyncio.run(_do())


def test_every_level_survives_the_bridge_once_the_set_is_declared(
    upstream, vllm_vocab, monkeypatch
):
    """Before this, 3 of 5 levels took the turn down with a 400."""
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")
    upstream.set_handler(_vllm_upstream)

    for level in ("low", "medium", "high", "xhigh", "max"):
        resp = _post("/v1/messages", _messages_body(level))
        assert resp.status_code == 200, f"{level} -> {resp.status_code}"

    sent = [json.loads(r.content).get("reasoning_effort") for r in upstream.requests]
    assert sent == ["low", "medium", "xhigh", "xhigh", "xhigh"]


def test_the_issues_own_curl_stops_being_a_400(upstream, vllm_vocab):
    """Direct ``/v1/chat/completions`` — the reproduction filed on the issue.

    This path is the byte-for-byte relay, so the clamp is its one documented
    exception; the rest of the body must arrive unchanged.
    """
    upstream.set_handler(_vllm_upstream)
    payload = {
        "model": "ChatDRAGON-Medium",
        "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "high",
        "max_tokens": 64,
    }

    resp = _post("/v1/chat/completions", payload)

    assert resp.status_code == 200
    forwarded = json.loads(upstream.requests[-1].content)
    assert forwarded["reasoning_effort"] == "xhigh"
    assert forwarded["model"] == "ChatDRAGON-Medium"
    assert forwarded["max_tokens"] == 64
    assert forwarded["messages"] == payload["messages"]


def test_passthrough_relays_original_bytes_when_nothing_needs_clamping(upstream):
    """The relay stays byte-for-byte for every case the exception excludes."""
    upstream.set_handler(_vllm_upstream)
    raw = b'{"model":"m","messages":[],"reasoning_effort":"low","x":  1}'

    async def _do():
        async with _asgi() as client:
            return await client.post(
                "/v1/chat/completions",
                content=raw,
                headers={"content-type": "application/json"},
            )

    assert asyncio.run(_do()).status_code == 200
    assert upstream.requests[-1].content == raw


def test_passthrough_leaves_a_non_chat_completions_body_alone(upstream, vllm_vocab):
    upstream.set_handler(lambda request: httpx.Response(200, json={}))
    raw = b'{"reasoning_effort":"high"}'

    async def _do():
        async with _asgi() as client:
            return await client.post(
                "/v1/embeddings",
                content=raw,
                headers={"content-type": "application/json"},
            )

    assert asyncio.run(_do()).status_code == 200
    assert upstream.requests[-1].content == raw


def test_passthrough_leaves_an_unparseable_body_alone(upstream, vllm_vocab):
    upstream.set_handler(lambda request: httpx.Response(200, json={}))
    raw = b"not json at all"

    async def _do():
        async with _asgi() as client:
            return await client.post(
                "/v1/chat/completions",
                content=raw,
                headers={"content-type": "application/json"},
            )

    assert asyncio.run(_do()).status_code == 200
    assert upstream.requests[-1].content == raw
