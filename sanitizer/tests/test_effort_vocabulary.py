"""Per-model effort vocabulary: issue #26.

A vLLM build answers a level it does not know with a 400 that takes the whole
turn. ``high`` — what a caller using the OpenAI-standard vocabulary sends — is
outside the accepted set of an upstream whose own error names
``xhigh (default), medium, low``. One sanitizer fronts many models whose sets
differ, so the declaration is per model. These pin the clamp, its per-model
independence, that an undeclared setting still forwards verbatim (issue #25),
and that an invalid declaration can never narrow anything.
"""

import asyncio
import json

import httpx
import pytest

from sanitizer.effort import (
    EFFORT_SCALE,
    ENV,
    EffortVocabularyError,
    clamp_to_supported,
    parse_vocabulary,
    supported_levels,
    validate_at_startup,
    vocabulary,
)
from sanitizer.main import app
from sanitizer.openai_bridge import anthropic_request_to_openai_body

QWEN = "qwen3.6-27b"  # the issue's upstream: vLLM, template accepts low|medium|xhigh
GLM = "glm-5-fp8"  # a second model behind the same sanitizer, left undeclared
KIMI = "kimi-k2"  # a third with a different vocabulary
_VOCAB = {QWEN: "low,medium,xhigh", "Qwen3.6-27B": ["low", "medium", "xhigh"], KIMI: "low,high,max"}


@pytest.fixture
def declared(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps(_VOCAB))


# --- parsing ------------------------------------------------------------------


def test_unset_or_empty_means_verbatim_everywhere(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert vocabulary() == {}
    for level in EFFORT_SCALE:
        assert clamp_to_supported(level, QWEN) == level
    monkeypatch.setenv(ENV, "   ")
    assert vocabulary() == {}


def test_levels_accept_csv_or_list_and_come_back_in_scale_order():
    vocab = parse_vocabulary(json.dumps({"m": "xhigh, low , MEDIUM", "n": ["max", "low"]}))
    assert vocab == {"m": ("low", "medium", "xhigh"), "n": ("low", "max")}


@pytest.mark.parametrize(
    "raw",
    [
        '{"m": "low,medium,xhi"}',  # typo
        '{"m": "nonsense"}',
        '{"m": ""}',  # declares nothing
        '{"m": 3}',
        '["low", "medium"]',  # not a map
        "not json",
        '{"": "low"}',
    ],
)
def test_any_invalid_entry_invalidates_the_whole_setting(raw):
    with pytest.raises(EffortVocabularyError):
        parse_vocabulary(raw)


def test_a_typo_never_narrows_it_forwards_verbatim_instead(monkeypatch):
    """The failure mode the review named.

    Dropping the unknown ``xhi`` and keeping ``low,medium`` would clamp
    ``high``/``xhigh``/``max`` down to ``medium`` — a typo silently turning into
    a *stricter* declaration. Invalid must never narrow: the whole setting is
    ignored and every model is forwarded as-is.
    """
    monkeypatch.setenv(ENV, json.dumps({QWEN: "low,medium,xhi"}))
    assert vocabulary() == {}
    for level in ("high", "xhigh", "max"):
        assert clamp_to_supported(level, QWEN) == level


def test_startup_refuses_an_invalid_declaration(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({QWEN: "low,medium,xhi"}))
    with pytest.raises(EffortVocabularyError, match="unknown level"):
        validate_at_startup()


def test_startup_accepts_a_valid_declaration_and_unset(monkeypatch, declared):
    assert set(validate_at_startup()) == set(_VOCAB)
    monkeypatch.delenv(ENV, raising=False)
    assert validate_at_startup() == {}


# --- the clamp, per model -------------------------------------------------------


def test_high_clamps_up_to_xhigh_not_down_to_medium(declared):
    """The issue's ask, and the direction matters: never reason less than asked."""
    assert clamp_to_supported("high", QWEN) == "xhigh"
    assert clamp_to_supported("max", QWEN) == "xhigh"
    assert clamp_to_supported("minimal", QWEN) == "low"


def test_declared_levels_are_untouched(declared):
    for level in ("low", "medium", "xhigh"):
        assert clamp_to_supported(level, QWEN) == level


def test_each_model_uses_its_own_vocabulary(declared):
    """Two models behind one sanitizer, two different accepted sets.

    A single proxy-wide set would rewrite Kimi's perfectly valid ``high`` into
    Qwen's vocabulary. Per model, each request lands on its own upstream's
    level.
    """
    assert clamp_to_supported("high", QWEN) == "xhigh"  # Qwen has no high
    assert clamp_to_supported("high", KIMI) == "high"  # Kimi does
    assert clamp_to_supported("medium", KIMI) == "high"  # Kimi has no medium; up
    assert clamp_to_supported("xhigh", KIMI) == "max"  # nor xhigh; up to max


def test_an_undeclared_model_is_forwarded_verbatim(declared):
    assert supported_levels(GLM) == ()
    for level in EFFORT_SCALE:
        assert clamp_to_supported(level, GLM) == level


def test_a_missing_model_name_is_forwarded_verbatim(declared):
    assert clamp_to_supported("high", None) == "high"
    assert clamp_to_supported("high", "") == "high"


def test_clamps_down_when_nothing_above_is_declared(monkeypatch):
    monkeypatch.setenv(ENV, json.dumps({QWEN: "low,medium"}))
    assert clamp_to_supported("xhigh", QWEN) == "medium"
    assert clamp_to_supported("max", QWEN) == "medium"


def test_a_level_outside_the_scale_is_not_invented(declared):
    assert clamp_to_supported("bogus", QWEN) is None


def test_case_variant_aliases_are_separate_keys(declared):
    """LiteLLM names are case-sensitive; each alias is declared on its own."""
    assert clamp_to_supported("high", "Qwen3.6-27B") == "xhigh"
    assert clamp_to_supported("high", "QWEN3.6-27B") == "high", "not declared -> verbatim"


# --- the bridge path (the deployed one) --------------------------------------


def _messages_body(effort, model=QWEN):
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
        "output_config": {"effort": effort},
    }


def test_bridge_sends_a_level_this_models_upstream_accepts(declared):
    assert anthropic_request_to_openai_body(_messages_body("high"))["reasoning_effort"] == "xhigh"
    assert anthropic_request_to_openai_body(_messages_body("high", GLM))["reasoning_effort"] == "high"


def test_bridge_still_forwards_verbatim_without_a_declaration(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert anthropic_request_to_openai_body(_messages_body("high"))["reasoning_effort"] == "high"


def test_bridge_none_still_sends_nothing(declared):
    """``none`` disables thinking on the Anthropic side; it is not a level."""
    assert "reasoning_effort" not in anthropic_request_to_openai_body(_messages_body("none"))


# --- the incident, end to end against upstreams that behave like the real ones


def _upstream_accepting(per_model):
    def handler(request):
        body = json.loads(request.content or b"{}")
        model, effort = body.get("model"), body.get("reasoning_effort")
        accepted = per_model.get(model)
        if accepted is not None and effort is not None and effort not in accepted:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": f"Unexpected reasoning effort {effort}. Supported types are "
                        + ", ".join(accepted)
                        + ".",
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
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
                ],
            },
        )

    return handler


_REAL_UPSTREAMS = {QWEN: ("low", "medium", "xhigh"), "Qwen3.6-27B": ("low", "medium", "xhigh"), KIMI: ("low", "high", "max")}


def _asgi():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def _post(path, payload=None, *, content=None, headers=None):
    async def _do():
        async with _asgi() as client:
            if content is not None:
                return await client.post(path, content=content, headers=headers)
            return await client.post(path, json=payload)

    return asyncio.run(_do())


def test_every_level_survives_the_bridge_for_every_declared_model(upstream, declared, monkeypatch):
    """Before this, 3 of 5 levels took the Qwen turn down with a 400."""
    monkeypatch.setenv("SANITIZER_USE_OPENAI_BRIDGE", "true")
    upstream.set_handler(_upstream_accepting(_REAL_UPSTREAMS))

    for model in (QWEN, KIMI):
        for level in ("low", "medium", "high", "xhigh", "max"):
            resp = _post("/v1/messages", _messages_body(level, model))
            assert resp.status_code == 200, f"{model} {level} -> {resp.status_code}"

    sent = [(json.loads(r.content)["model"], json.loads(r.content).get("reasoning_effort")) for r in upstream.requests]
    assert sent[:5] == [(QWEN, "low"), (QWEN, "medium"), (QWEN, "xhigh"), (QWEN, "xhigh"), (QWEN, "xhigh")]
    assert sent[5:] == [(KIMI, "low"), (KIMI, "high"), (KIMI, "high"), (KIMI, "max"), (KIMI, "max")]


def test_the_issues_own_curl_stops_being_a_400(upstream, declared):
    """Direct ``/v1/chat/completions`` — the reproduction filed on the issue.

    This path is the byte-for-byte relay, so the clamp is its one documented
    exception; the rest of the body must arrive unchanged.
    """
    upstream.set_handler(_upstream_accepting(_REAL_UPSTREAMS))
    payload = {"model": QWEN, "messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "high", "max_tokens": 64}

    resp = _post("/v1/chat/completions", payload)

    assert resp.status_code == 200
    forwarded = json.loads(upstream.requests[-1].content)
    assert forwarded["reasoning_effort"] == "xhigh"
    assert forwarded["model"] == QWEN
    assert forwarded["max_tokens"] == 64
    assert forwarded["messages"] == payload["messages"]


def test_passthrough_relays_original_bytes_for_an_undeclared_model(upstream, declared):
    """The relay stays byte-for-byte for a model with no declaration."""
    upstream.set_handler(lambda request: httpx.Response(200, json={}))
    raw = b'{"model":"glm-5-fp8","messages":[],"reasoning_effort":"high","x":  1}'
    assert _post("/v1/chat/completions", content=raw, headers={"content-type": "application/json"}).status_code == 200
    assert upstream.requests[-1].content == raw


def test_passthrough_relays_original_bytes_when_nothing_needs_clamping(upstream, declared):
    upstream.set_handler(lambda request: httpx.Response(200, json={}))
    raw = b'{"model":"qwen3.6-27b","messages":[],"reasoning_effort":"low","x":  1}'
    assert _post("/v1/chat/completions", content=raw, headers={"content-type": "application/json"}).status_code == 200
    assert upstream.requests[-1].content == raw


def test_passthrough_leaves_a_non_chat_completions_body_alone(upstream, declared):
    upstream.set_handler(lambda request: httpx.Response(200, json={}))
    raw = b'{"model":"qwen3.6-27b","reasoning_effort":"high"}'
    assert _post("/v1/embeddings", content=raw, headers={"content-type": "application/json"}).status_code == 200
    assert upstream.requests[-1].content == raw


def test_passthrough_leaves_an_unparseable_body_alone(upstream, declared):
    upstream.set_handler(lambda request: httpx.Response(200, json={}))
    raw = b"not json at all"
    assert _post("/v1/chat/completions", content=raw, headers={"content-type": "application/json"}).status_code == 200
    assert upstream.requests[-1].content == raw
