import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace


class StripThinkingBypassTests(unittest.TestCase):
    def test_disabled_apply_patch_bypasses_callback_and_streaming_patch(self):
        for name in list(sys.modules):
            if (
                name == "strip_thinking"
                or name == "litellm"
                or name.startswith("litellm.")
            ):
                del sys.modules[name]

        os.environ["THINK_OUTPUT_MODE"] = "none"
        os.environ["STRIP_THINKING_ENABLED"] = "false"

        litellm = types.ModuleType("litellm")
        litellm.callbacks = []
        sys.modules["litellm"] = litellm

        custom_logger_mod = types.ModuleType("litellm.integrations.custom_logger")
        custom_logger_mod.CustomLogger = type("CustomLogger", (), {})
        sys.modules["litellm.integrations"] = types.ModuleType("litellm.integrations")
        sys.modules["litellm.integrations.custom_logger"] = custom_logger_mod

        class FakeAdapter:
            def _translate_streaming_openai_chunk_to_anthropic(self, choices):
                return "thinking_delta", SimpleNamespace(thinking="visible answer")

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
        anthropic_types_mod.ContentTextBlockDelta = type(
            "ContentTextBlockDelta",
            (),
            {"__init__": lambda self, type, text: setattr(self, "text", text)},
        )
        sys.modules["litellm.types"] = types.ModuleType("litellm.types")
        sys.modules["litellm.types.llms"] = types.ModuleType("litellm.types.llms")
        sys.modules["litellm.types.llms.anthropic"] = anthropic_types_mod

        strip_thinking = importlib.import_module("strip_thinking")
        strip_thinking.apply_patch()
        type_of_content, delta = FakeAdapter()._translate_streaming_openai_chunk_to_anthropic([])

        self.assertEqual(litellm.callbacks, [])
        self.assertEqual(type_of_content, "thinking_delta")
        self.assertEqual(delta.thinking, "visible answer")


if __name__ == "__main__":
    unittest.main()
