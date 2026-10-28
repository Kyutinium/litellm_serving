import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from benchmark_smg import Result, percentile, streamed_delta, summary  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    def test_stream_reader(self):
        self.assertEqual(streamed_delta('data: {"choices":[{"delta":{"role":"assistant"}}]}'), (False, None))
        self.assertEqual(streamed_delta('data: {"choices":[{"delta":{"content":"hi"}}]}'), (True, None))
        self.assertEqual(streamed_delta('data: {"choices":[],"usage":{"completion_tokens":42}}'), (False, 42))
        self.assertEqual(streamed_delta("data: [DONE]"), (False, None))

    def test_percentiles_and_missing_usage(self):
        self.assertEqual(percentile([1, 2, 3, 4], 95), 4)
        self.assertIsNone(percentile([], 50))
        report = summary("smg", [Result(0.2, 0.5, None), Result(None, 0.1, None, "HTTP 503")], 1)
        self.assertEqual(report["success"], 1)
        self.assertEqual(report["failed"], 1)
        self.assertIsNone(report["completion_tokens_per_second"])


if __name__ == "__main__":
    unittest.main()
