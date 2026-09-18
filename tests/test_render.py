"""scripts/render.py: models.yaml is the single source; everything else is derived."""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import render  # noqa: E402


class RegistryValidation(unittest.TestCase):
    def _load(self, text):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(textwrap.dedent(text))
        try:
            return render.load_registry(Path(f.name))
        finally:
            os.unlink(f.name)

    def test_one_client_name_routes_to_exactly_one_backend(self):
        with self.assertRaisesRegex(render.RegistryError, "declared by both"):
            self._load("""
                models:
                  - {id: a, aliases: [X], roles: [direct], serve: {engine: external, port: 1}, litellm: {provider: openai, max_tokens: 1}}
                  - {id: b, aliases: [X], roles: [direct], serve: {engine: external, port: 2}, litellm: {provider: openai, max_tokens: 1}}
            """)

    def test_unknown_role_and_engine_are_rejected(self):
        with self.assertRaisesRegex(render.RegistryError, "unknown role"):
            self._load("""
                models:
                  - {id: a, roles: [sidecar], serve: {engine: external, port: 1}, litellm: {provider: openai, max_tokens: 1}}
            """)
        with self.assertRaisesRegex(render.RegistryError, "unknown serve.engine"):
            self._load("""
                models:
                  - {id: a, roles: [direct], serve: {engine: ollama, port: 1}, litellm: {provider: openai, max_tokens: 1}}
            """)


class RenderedLiteLLM(unittest.TestCase):
    def setUp(self):
        self.reg = render.load_registry(ROOT / "models.yaml")

    def test_every_alias_becomes_an_entry_with_the_canonical_params(self):
        doc = yaml.safe_load(render.render_litellm(self.reg))
        by_name = {e["model_name"]: e["litellm_params"] for e in doc["model_list"]}
        for m in self.reg["models"]:
            canonical = by_name[m["id"]]
            self.assertEqual(canonical["model"], f"{m['litellm']['provider']}/{m['id']}")
            for alias in m.get("aliases", []):
                self.assertEqual(by_name[alias], canonical, alias)

    def test_master_key_comes_from_the_environment_not_the_file(self):
        doc = yaml.safe_load(render.render_litellm(self.reg))
        self.assertEqual(doc["general_settings"]["master_key"], "os.environ/LITELLM_MASTER_KEY")

    def test_committed_files_match_the_registry(self):
        """The drift check the repo relies on, run as a test too."""
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "render.py"), "--check"],
            capture_output=True, text=True, cwd=ROOT,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class RenderedEnv(unittest.TestCase):
    def test_effort_vocabulary_lists_every_alias_of_a_declaring_model(self):
        reg = render.load_registry(ROOT / "models.yaml")
        env = render.render_env(reg)
        line = next(l for l in env.splitlines() if l.startswith("SANITIZER_EFFORT_VOCABULARY="))
        import json
        vocab = json.loads(line.split("=", 1)[1])
        for m in reg["models"]:
            levels = (m.get("sanitizer") or {}).get("effort_supported")
            for name in render.every_name(m):
                if levels:
                    self.assertEqual(vocab[name], ",".join(levels), name)
                else:
                    self.assertNotIn(name, vocab, name)


class RenderedCompose(unittest.TestCase):
    def test_external_models_get_no_profile_and_launched_ones_do(self):
        reg = render.load_registry(ROOT / "models.yaml")
        for m in reg["models"]:
            text = render.render_compose(m)
            if m["serve"]["engine"] == "external":
                self.assertIsNone(text, m["id"])
            else:
                svc = next(iter(yaml.safe_load(text)["services"].values()))
                self.assertEqual(svc["profiles"], [m["id"]])
                self.assertEqual(svc["ports"], [f"{m['serve']['port']}:{render.CONTAINER_PORT[m['serve']['engine']]}"])


if __name__ == "__main__":
    unittest.main()
