"""Pure configuration tests; no SMG binary or GPU required."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from launch import ConfigError, command, parse_pools  # noqa: E402


class SMGConfigTests(unittest.TestCase):
    def test_three_workers_and_default_cache_policy(self):
        pools = parse_pools(json.dumps({
            "glm-5.1-fp8": {"port": 3998, "workers": [
                "http://10.0.0.1:8089/v1", "http://10.0.0.2:8089", "http://10.0.0.3:8089"
            ]}
        }))
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0].workers[0], "http://10.0.0.1:8089")
        args = command(pools[0])
        self.assertEqual(args[:3], ["smg", "launch", "--worker-urls"])
        self.assertEqual(args[3:6], list(pools[0].workers))
        self.assertIn("cache_aware", args)
        self.assertIn("--disable-retries", args)

    def test_multiple_independent_model_pools(self):
        pools = parse_pools(json.dumps({
            "a": {"port": 3998, "workers": ["http://a1:8000", "http://a2:8000"]},
            "b": {"port": 4000, "policy": "power_of_two", "workers": ["http://b1:8000", "http://b2:8000"]},
        }))
        self.assertEqual([p.port for p in pools], [3998, 4000])
        self.assertEqual(pools[1].policy, "power_of_two")

    def test_rejects_invalid_configuration(self):
        valid = {"a": {"port": 3998, "workers": ["http://a1:8000", "http://a2:8000"]}}
        variants = [
            "", "{}", "[1]", "{invalid}",
            json.dumps({"a": {"port": 3998, "workers": ["http://a1:8000"]}}),
            json.dumps({"a": {"port": 3998, "workers": ["http://a1:8000", "http://a1:8000/v1"]}}),
            json.dumps({"a": {"port": 3999, "workers": valid["a"]["workers"]}}),
            json.dumps({"a": {"port": True, "workers": valid["a"]["workers"]}}),
            json.dumps({"a": {"port": 3998, "policy": "nonsense", "workers": valid["a"]["workers"]}}),
            json.dumps({"a": {"port": 3998, "policy": [], "workers": valid["a"]["workers"]}}),
            json.dumps({"a": {"port": 3998, "workers": ["file:///etc/passwd", "http://a2:8000"]}}),
            json.dumps({"a": {"port": 3998, "workers": ["http://name:secret@a1:8000", "http://a2:8000"]}}),
            json.dumps({"a": {"port": 3998, "workers": ["http://a1:8000/private", "http://a2:8000"]}}),
            json.dumps({"a": valid["a"], "b": {"port": 3998, "workers": ["http://b1:8000", "http://b2:8000"]}}),
        ]
        for case in variants:
            with self.subTest(case=case), self.assertRaises(ConfigError):
                parse_pools(case)


if __name__ == "__main__":
    unittest.main()
