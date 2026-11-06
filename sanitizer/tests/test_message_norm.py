"""Regression tests for UniBridge-style system and tool ordering."""

from __future__ import annotations

import copy
import re

import pytest

from sanitizer import config
from sanitizer.message_norm import normalize_system_messages, normalize_tool_messages
from sanitizer.openai_bridge import anthropic_request_to_openai_body


MISTRAL = "mistralai/Mistral-Large-4-1T-A52B"
_GATE = re.compile(r"mistral-large-4", re.IGNORECASE)
CALL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "Read", "arguments": "{}"}},
        {"id": "t2", "type": "function", "function": {"name": "Bash", "arguments": "{}"}},
    ],
}
T1 = {"role": "tool", "tool_call_id": "t1", "content": "one"}
T2 = {"role": "tool", "tool_call_id": "t2", "content": "two"}


def roles(messages):
    return [message["role"] for message in messages]


def test_default_model_gates(monkeypatch):
    monkeypatch.delenv("SANITIZER_MID_SYSTEM_MODEL_PATTERN", raising=False)
    monkeypatch.delenv("SANITIZER_TOOL_NORM_MODEL_PATTERN", raising=False)
    system_gate = config.get_mid_system_model_regex()
    tool_gate = config.get_tool_norm_model_regex()
    for model in (MISTRAL, "Mistral-Large-4", "qwen3.5-32b", "hosted_vllm/Qwen3.6-27B", "qwen3.8-a3b"):
        assert system_gate.search(model)
    for model in (MISTRAL, "hosted_vllm/Mistral-Large-4"):
        assert tool_gate.search(model)
    for model in ("qwen3-8b", "qwen2.5-72b", "gpt-4o", "mistral-large-3"):
        assert not system_gate.search(model)
        assert not tool_gate.search(model)
    assert not tool_gate.search("qwen3.6-27b")


@pytest.mark.parametrize("name, default", [
    ("SANITIZER_MID_SYSTEM_MODEL_PATTERN", config._DEFAULT_MID_SYSTEM_MODEL_PATTERN),
    ("SANITIZER_TOOL_NORM_MODEL_PATTERN", config._DEFAULT_TOOL_NORM_MODEL_PATTERN),
])
def test_invalid_regex_uses_default(monkeypatch, name, default):
    monkeypatch.setenv(name, "(")
    assert config._get_regex(name, default).pattern == default


def test_config_overrides_and_policy(monkeypatch):
    monkeypatch.setenv("SANITIZER_MID_SYSTEM_MODEL_PATTERN", ".*")
    monkeypatch.setenv("SANITIZER_TOOL_NORM_MODEL_PATTERN", "^my-model$")
    monkeypatch.setenv("SANITIZER_MID_SYSTEM_POLICY", "  HOIST  ")
    assert config.get_mid_system_model_regex().search("other-model")
    assert config.get_tool_norm_model_regex().search("my-model")
    assert not config.get_tool_norm_model_regex().search("other-model")
    assert config.get_mid_system_policy() == "hoist"
    monkeypatch.setenv("SANITIZER_MID_SYSTEM_POLICY", "invalid")
    assert config.get_mid_system_policy() == "user"


def test_mid_system_as_user_keeps_initial_kv_prefix():
    before = [{"role": "system", "content": "stable"},
              {"role": "user", "content": "question"},
              {"role": "assistant", "content": "answer"}]
    after = before + [{"role": "system", "content": "late reminder"},
                      {"role": "user", "content": "next question"}]
    normalized_before = normalize_system_messages(before, "user")
    normalized_after = normalize_system_messages(after, "user")
    assert normalized_after[:len(normalized_before)] == normalized_before
    assert normalized_after[-2:] == [
        {"role": "user", "content": "late reminder"},
        {"role": "user", "content": "next question"},
    ]
    assert normalized_after[0]["content"] == "stable"


def test_merge_initial_system_run_not_later_reminders():
    messages = [
        {"role": "system", "content": "base"},
        {"role": "system", "content": [{"type": "text", "text": "start"}]},
        {"role": "user", "content": "q"},
        {"role": "system", "content": "reminder"},
    ]
    out = normalize_system_messages(messages, "user")
    assert out == [
        {"role": "system", "content": "base\\n\\nstart".replace("\\n", "\n")},
        {"role": "user", "content": "q"},
        {"role": "user", "content": "reminder"},
    ]


def test_unmatched_model_hoists_as_before_even_with_user_policy():
    messages = [{"role": "system", "content": "base"},
                {"role": "user", "content": "q"},
                {"role": "system", "content": "late"}]
    pattern = re.compile(r"mistral-large-4", re.IGNORECASE)
    out = normalize_system_messages(messages, "user", "gpt-4o", pattern)
    assert out == [{"role": "system", "content": "base\n\nlate"},
                   {"role": "user", "content": "q"}]


def test_system_policies_asis_and_hoist():
    messages = [{"role": "user", "content": "q"},
                {"role": "system", "content": "late"}]
    assert normalize_system_messages(messages, "asis") is messages
    assert normalize_system_messages(messages, "hoist") == [
        {"role": "system", "content": "late"}, {"role": "user", "content": "q"}]


def test_system_norm_is_pure_and_idempotent():
    messages = [{"role": "system", "content": "base"},
                {"role": "system", "content": "more"},
                {"role": "user", "content": "x"},
                {"role": "system", "content": "late"}]
    original = copy.deepcopy(messages)
    once = normalize_system_messages(messages, "user")
    assert messages == original
    assert normalize_system_messages(once, "user") is once


def test_tool_results_move_before_wedged_user_and_keep_their_order():
    note = {"role": "user", "content": "reminder"}
    messages = [CALL, note, T2, T1]
    original = copy.deepcopy(messages)
    out = normalize_tool_messages(messages, MISTRAL, _GATE)
    assert out == [CALL, T2, T1, note]
    assert messages == original
    assert normalize_tool_messages(out, MISTRAL, _GATE) is out


def test_tools_do_not_cross_another_assistant_or_take_orphan_results():
    orphan = {"role": "tool", "tool_call_id": "unrelated", "content": "x"}
    second = {"role": "assistant", "content": "done"}
    messages = [CALL, {"role": "user", "content": "note"}, T1, orphan, second, T2]
    out = normalize_tool_messages(messages)
    assert out == [CALL, T1, {"role": "user", "content": "note"}, orphan, second, T2]


def test_nonmatching_tool_model_returns_input():
    messages = [CALL, {"role": "user", "content": "note"}, T1]
    assert normalize_tool_messages(messages, "qwen3.6-27b", _GATE) is messages


def test_full_anthropic_bridge_mistral_normalizes_system_then_tools(monkeypatch):
    monkeypatch.delenv("SANITIZER_MID_SYSTEM_POLICY", raising=False)
    monkeypatch.delenv("SANITIZER_MID_SYSTEM_MODEL_PATTERN", raising=False)
    monkeypatch.delenv("SANITIZER_TOOL_NORM_MODEL_PATTERN", raising=False)
    body = {
        "model": MISTRAL,
        "system": "stable base",
        "messages": [
            {"role": "user", "content": "read file"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"path": "foo"}}
            ]},
            {"role": "system", "content": "remember the file scope"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "file content"}
            ]},
        ],
    }
    original = copy.deepcopy(body)
    output = anthropic_request_to_openai_body(body)
    assert roles(output["messages"]) == ["system", "user", "assistant", "tool", "user"]
    assert output["messages"][0]["content"] == "stable base"
    assert output["messages"][3]["tool_call_id"] == "t1"
    assert output["messages"][4]["content"] == "remember the file scope"
    assert body == original


def test_full_bridge_qwen_demotes_late_system_without_changing_prefix():
    body = {"model": "Qwen3.6-27B", "system": "base",
            "messages": [{"role": "user", "content": "q"},
                         {"role": "system", "content": "reminder"}]}
    out = anthropic_request_to_openai_body(body)["messages"]
    assert out == [{"role": "system", "content": "base"},
                   {"role": "user", "content": "q"},
                   {"role": "user", "content": "reminder"}]


def test_full_bridge_other_models_keep_original_hoisting():
    body = {"model": "some-other-model", "system": "base",
            "messages": [{"role": "user", "content": "q"},
                         {"role": "system", "content": "reminder"}]}
    out = anthropic_request_to_openai_body(body)["messages"]
    assert out == [{"role": "system", "content": "base\n\nreminder"},
                   {"role": "user", "content": "q"}]
