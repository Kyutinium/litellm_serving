"""The effort vocabulary is learned from the upstream, not hand-written.

vLLM names its supported levels in the 400 it answers an unsupported one with;
a 1-token completion per level answers it for builds that word the error
differently. These pin: the parser, learn-and-retry-once on both ingress paths,
that a declared override is never overwritten, the startup probe (one-shot via
``minimal`` and the per-level fallback), and the ``/v1/models`` enrichment the
gateway reads.
"""

import asyncio
import json

import httpx
import pytest

from sanitizer.effort import (
    ENV,
    forget_learned,
    known_levels,
    learn_from_upstream_error,
    learned_levels,
    parse_supported_from_error,
    probe_models,
    supported_levels,
)
from sanitizer.main import app

QWEN = "qwen3.6-27b"
VLLM_400 = "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low."


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setenv("SANITIZER_EFFORT_PROBE", "false")
    forget_learned()
    yield
    forget_learned()


# --- parsing --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (VLLM_400, ("low", "medium", "xhigh")),
        ("Unsupported reasoning_effort 'max'. Supported values: low, high, max.", ("low", "high", "max")),
        ("reasoning effort must be one of the supported levels: minimal, low", ("minimal", "low")),
        ("Invalid model name passed in model=foo", ()),  # not about effort
        ("The supported types are low and medium", ()),  # about something else entirely
        ("", ()),
    ],
)
def test_parse_supported_levels_from_upstream_error(text, expected):
    assert parse_supported_from_error(text) == expected


def test_learn_reads_openai_error_envelopes_and_plain_text():
    body = json.dumps({"error": {"message": VLLM_400, "type": "BadRequestError", "code": 400}}).encode()
    assert learn_from_upstream_error(QWEN, 400, body) == ("low", "medium", "xhigh")
    assert supported_levels(QWEN) == ("low", "medium", "xhigh")
    forget_learned()
    assert learn_from_upstream_error(QWEN, 400, json.dumps({"detail": VLLM_400}).encode()) == ("low", "medium", "xhigh")
    forget_learned()
    assert learn_from_upstream_error(QWEN, 400, VLLM_400.encode()) == ("low", "medium", "xhigh")


def test_learn_ignores_other_errors():
    assert learn_from_upstream_error(QWEN, 500, VLLM_400.encode()) == ()
    assert learn_from_upstream_error(QWEN, 400, b'{"error":{"message":"No connected db."}}') == ()
    assert learn_from_upstream_error(None, 400, VLLM_400.encode()) == ()
    assert learned_levels() == {}


def test_a_declared_override_is_never_overwritten_by_learning(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({QWEN: "low,medium"}))
    assert learn_from_upstream_error(QWEN, 400, VLLM_400.encode()) == ("low", "medium")
    assert supported_levels(QWEN) == ("low", "medium")
    assert learned_levels() == {}


# --- the two ingress paths learn from a real 400 and retry once --------------------


def _vllm_like(accepted):
    """Upstream that answers an unsupported level exactly like vLLM does."""
    calls = []

    def handler(request):
        body = json.loads(request.content or b"{}")
        calls.append(body)
        effort = body.get("reasoning_effort")
        if effort is not None and effort not in accepted:
            return httpx.Response(
                400,
                json={"error": {"message": f"Unexpected reasoning effort {effort}. Supported types are " + ", ".join(accepted) + ".", "type": "BadRequestError", "code": 400}},
            )
        return httpx.Response(
            200,
            json={"id": "x", "object": "chat.completion", "model": body.get("model"), "choices": [{"index": 0, "message": {"role": "assistant", "content": f"ok {effort}"}, "finish_reason": "stop"}]},
        )

    return handler, calls


def _asgi():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _post(path, payload):
    async def _do():
        async with _asgi() as client:
            return await client.post(path, json=payload)

    return asyncio.run(_do())


def _get(path):
    async def _do():
        async with _asgi() as client:
            return await client.get(path)

    return asyncio.run(_do())


def test_bridge_learns_from_the_first_400_and_retries_with_a_level_the_model_takes(upstream, monkeypatch):
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")
    handler, calls = _vllm_like(("xhigh", "medium", "low"))
    upstream.set_handler(handler)

    resp = _post("/v1/messages", {"model": QWEN, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "output_config": {"effort": "high"}})
    assert resp.status_code == 200, resp.text
    assert [c.get("reasoning_effort") for c in calls] == ["high", "xhigh"], "one 400, one retry"
    assert supported_levels(QWEN) == ("low", "medium", "xhigh")

    # Learned: the next request is clamped up front — a single upstream call.
    calls.clear()
    resp = _post("/v1/messages", {"model": QWEN, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "output_config": {"effort": "max"}})
    assert resp.status_code == 200
    assert [c.get("reasoning_effort") for c in calls] == ["xhigh"]


def test_bridge_relays_a_400_that_teaches_nothing(upstream, monkeypatch):
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "No connected db."}})

    upstream.set_handler(handler)
    resp = _post("/v1/messages", {"model": QWEN, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}], "output_config": {"effort": "high"}})
    assert resp.status_code == 400
    assert len(calls) == 1, "no blind retry"
    assert learned_levels() == {}


def test_relay_path_learns_and_retries_the_issues_curl(upstream):
    handler, calls = _vllm_like(("xhigh", "medium", "low"))
    upstream.set_handler(handler)
    resp = _post("/v1/chat/completions", {"model": QWEN, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high", "max_tokens": 4})
    assert resp.status_code == 200, resp.text
    assert [c.get("reasoning_effort") for c in calls] == ["high", "xhigh"]
    assert calls[-1]["max_tokens"] == 4 and calls[-1]["model"] == QWEN


# --- what the gateway reads ---------------------------------------------------------


def test_models_list_is_byte_for_byte_until_something_is_known(upstream):
    raw = b'{"object":"list","data":[{"id":"qwen3.6-27b","object":"model"}]}'
    upstream.set_handler(lambda request: httpx.Response(200, content=raw, headers={"content-type": "application/json"}))
    resp = _get("/v1/models")
    assert resp.content == raw


def test_models_list_carries_learned_effort_levels(upstream, monkeypatch):
    learn_from_upstream_error(QWEN, 400, VLLM_400.encode())
    monkeypatch.setenv(ENV, json.dumps({"glm-5-fp8": "low,medium,high,xhigh,max"}))
    upstream.set_handler(lambda request: httpx.Response(200, json={"object": "list", "data": [{"id": QWEN, "object": "model"}, {"id": "glm-5-fp8", "object": "model"}, {"id": "other", "object": "model"}]}))
    rows = {row["id"]: row for row in _get("/v1/models").json()["data"]}
    assert rows[QWEN]["effort_levels"] == ["low", "medium", "xhigh"]
    assert rows["glm-5-fp8"]["effort_levels"] == ["low", "medium", "high", "xhigh", "max"]
    assert "effort_levels" not in rows["other"]

    diag = _get("/sanitizer/effort-levels").json()
    assert diag[QWEN] == {"levels": ["low", "medium", "xhigh"], "source": "upstream-400"}
    assert diag["glm-5-fp8"]["source"] == "declared"


# --- the startup probe --------------------------------------------------------------


def test_probe_learns_in_one_question_when_the_template_names_its_set():
    handler, calls = _vllm_like(("xhigh", "medium", "low"))

    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    learned = asyncio.run(probe_models("http://up", [QWEN], client_factory=factory, headers={}))
    assert learned == {QWEN: ("low", "medium", "xhigh")}
    assert [c["reasoning_effort"] for c in calls] == ["minimal"], "the first 400 named the whole set"
    assert calls[0]["max_tokens"] == 1 and calls[0]["allowed_openai_params"] == ["reasoning_effort"]


def test_probe_falls_back_to_one_token_per_level_when_the_error_does_not_name_the_set():
    accepted = {"low", "high", "max"}
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["reasoning_effort"])
        if body["reasoning_effort"] not in accepted:
            return httpx.Response(400, json={"error": {"message": "reasoning_effort not accepted by this template"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "k"}}]})

    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    learned = asyncio.run(probe_models("http://up", ["kimi-k2"], client_factory=factory, headers={}))
    assert learned == {"kimi-k2": ("low", "high", "max")}
    assert calls == ["minimal", "low", "medium", "high", "xhigh", "max"]
    assert known_levels()["kimi-k2"] == ("low", "high", "max")


def test_probe_skips_declared_models_and_gives_up_on_unrelated_failures(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({QWEN: "low,medium,xhigh"}))
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(503, json={"error": "warming up"})

    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    learned = asyncio.run(probe_models("http://up", [QWEN, "glm-5-fp8"], client_factory=factory, headers={}))
    assert learned == {}
    assert calls == ["glm-5-fp8"], "declared model not probed; 503 ends the other's probe"
    assert learned_levels() == {}
