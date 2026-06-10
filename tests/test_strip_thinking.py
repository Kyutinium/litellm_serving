import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace


class FakeContentTextBlockDelta:
    def __init__(self, type, text):
        self.type = type
        self.text = text


class StripThinkingPatchTests(unittest.TestCase):
    def _load_strip_thinking(self, mode, original_result):
        for name in list(sys.modules):
            if name == "strip_thinking" or name == "litellm" or name.startswith("litellm."):
                del sys.modules[name]

        os.environ["THINK_OUTPUT_MODE"] = mode

        litellm = types.ModuleType("litellm")
        litellm.callbacks = []
        sys.modules["litellm"] = litellm

        custom_logger_mod = types.ModuleType("litellm.integrations.custom_logger")
        custom_logger_mod.CustomLogger = type("CustomLogger", (), {})
        sys.modules["litellm.integrations"] = types.ModuleType("litellm.integrations")
        sys.modules["litellm.integrations.custom_logger"] = custom_logger_mod

        class FakeAdapter:
            def _translate_streaming_openai_chunk_to_anthropic(self, choices):
                return original_result

        transformation_mod = types.ModuleType(
            "litellm.llms.anthropic.experimental_pass_through.adapters.transformation"
        )
        transformation_mod.LiteLLMAnthropicMessagesAdapter = FakeAdapter

        sys.modules["litellm.llms"] = types.ModuleType("litellm.llms")
        sys.modules["litellm.llms.anthropic"] = types.ModuleType("litellm.llms.anthropic")
        sys.modules[
            "litellm.llms.anthropic.experimental_pass_through"
        ] = types.ModuleType("litellm.llms.anthropic.experimental_pass_through")
        sys.modules[
            "litellm.llms.anthropic.experimental_pass_through.adapters"
        ] = types.ModuleType("litellm.llms.anthropic.experimental_pass_through.adapters")
        sys.modules[
            "litellm.llms.anthropic.experimental_pass_through.adapters.transformation"
        ] = transformation_mod

        anthropic_types_mod = types.ModuleType("litellm.types.llms.anthropic")
        anthropic_types_mod.ContentTextBlockDelta = FakeContentTextBlockDelta
        sys.modules["litellm.types"] = types.ModuleType("litellm.types")
        sys.modules["litellm.types.llms"] = types.ModuleType("litellm.types.llms")
        sys.modules["litellm.types.llms.anthropic"] = anthropic_types_mod

        strip_thinking = importlib.import_module("strip_thinking")
        strip_thinking._patch_streaming_thinking_delta()
        return FakeAdapter

    def test_none_mode_promotes_reasoning_only_thinking_delta(self):
        adapter_cls = self._load_strip_thinking(
            "none", ("thinking_delta", SimpleNamespace(thinking="visible answer"))
        )

        type_of_content, delta = adapter_cls()._translate_streaming_openai_chunk_to_anthropic([])

        self.assertEqual(type_of_content, "text_delta")
        self.assertEqual(delta.type, "text_delta")
        self.assertEqual(delta.text, "visible answer")

    def test_none_mode_still_suppresses_signature_delta(self):
        adapter_cls = self._load_strip_thinking(
            "none", ("signature_delta", SimpleNamespace(signature="sig"))
        )

        type_of_content, delta = adapter_cls()._translate_streaming_openai_chunk_to_anthropic([])

        self.assertEqual(type_of_content, "text_delta")
        self.assertEqual(delta.type, "text_delta")
        self.assertEqual(delta.text, "")

    def test_empty_upstream_text_delta_remains_empty(self):
        adapter_cls = self._load_strip_thinking(
            "none", ("text_delta", SimpleNamespace(type="text_delta", text=""))
        )

        type_of_content, delta = adapter_cls()._translate_streaming_openai_chunk_to_anthropic([])

        self.assertEqual(type_of_content, "text_delta")
        self.assertEqual(delta.text, "")


if __name__ == "__main__":
    unittest.main()
