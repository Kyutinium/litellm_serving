import json
import os
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import prompt_capture as pc  # noqa: E402

SYSTEM = {"role": "system", "content": "You are Claude Code. " * 200}
TOOLS = [
    {"type": "function", "function": {"name": "Read", "description": "Read a file", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "mcp__wiki__search", "description": "Search the team wiki\nmore", "parameters": {"type": "object"}}},
]


def kwargs_for(messages, call_id="c1", headers=None):
    """The shape LiteLLM 1.105 hands log_pre_api_call (checked against a live proxy)."""
    body = {
        "model": "served-name",
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 4096,
        "reasoning_effort": "high",
        "tools": TOOLS,
    }
    headers = headers or {
        "host": "localhost:3999",
        "authorization": "Bearer sk-secret",
        "x-api-key": "sk-secret",
        "X-Claude-Code-Session-Id": "sess-1",
        "x-claude-code-agent-type": "Explore",
        "user-agent": "claude-cli/2.1.283",
    }
    return {
        "litellm_call_id": call_id,
        "custom_llm_provider": "hosted_vllm",
        "additional_args": {"api_base": "http://localhost:8088/v1/chat/completions", "complete_input_dict": body, "headers": {}},
        "litellm_params": {"proxy_server_request": {"headers": headers}, "metadata": {"model_group": "glm-5-fp8"}},
    }


class BuildRecordTests(unittest.TestCase):
    def test_keeps_the_exact_body_and_only_allowlisted_headers(self):
        kw = kwargs_for([SYSTEM, {"role": "user", "content": "hi"}])
        record = pc.build_record(kw, now=0)
        self.assertIs(record["body"], kw["additional_args"]["complete_input_dict"])
        self.assertEqual(record["session_id"], "sess-1")
        self.assertEqual(record["agent_type"], "Explore")
        self.assertEqual(record["model_group"], "glm-5-fp8")
        self.assertNotIn("sk-secret", json.dumps(record))
        self.assertEqual(record["ts"][:10], "1970-01-01")

    def test_no_body_no_record(self):
        self.assertIsNone(pc.build_record({"additional_args": {}}))

    def test_falls_back_to_metadata_headers(self):
        kw = kwargs_for([{"role": "user", "content": "hi"}])
        kw["litellm_params"] = {"metadata": {"headers": {"x-claude-code-session-id": "sess-2"}}}
        self.assertEqual(pc.build_record(kw)["session_id"], "sess-2")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = pc.Store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def blob_files(self):
        return [f for _b, _d, fs in os.walk(self.store.blob_root) for f in fs]

    def test_round_trip_is_exact(self):
        messages = [SYSTEM, {"role": "user", "content": [{"type": "text", "text": "안녕"}]}]
        kw = kwargs_for(messages)
        original = json.dumps(kw["additional_args"]["complete_input_dict"], ensure_ascii=False)
        self.store.write(pc.build_record(kw))
        [line] = list(self.store.iter_records(self.store.days()[0]))
        self.assertEqual(line["body"]["messages"][0].keys(), {"$blob"})
        self.assertEqual(json.dumps(self.store.expand(line["body"]), ensure_ascii=False), original)

    def test_repeated_system_tools_and_history_are_stored_once(self):
        history = [SYSTEM, {"role": "user", "content": "one"}]
        self.store.write(pc.build_record(kwargs_for(history, "c1")))
        first = len(self.blob_files())
        history = history + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "two"}]
        self.store.write(pc.build_record(kwargs_for(history, "c2")))
        # system + "one" + tools, then only the two new messages
        self.assertEqual(first, 3)
        self.assertEqual(len(self.blob_files()), 5)

    def test_files_are_private(self):
        self.store.write(pc.build_record(kwargs_for([SYSTEM])))
        day = self.store.day_path(self.store.days()[0])
        self.assertEqual(os.stat(day).st_mode & 0o777, 0o600)
        blob = os.path.join(self.store.blob_root, *[(b, f) for b, _d, fs in os.walk(self.store.blob_root) for f in fs][0])
        self.assertEqual(os.stat(blob).st_mode & 0o777, 0o600)

    def test_prune_drops_old_days_and_their_orphan_blobs(self):
        old = pc.build_record(kwargs_for([{"role": "user", "content": "old only"}], "old"), now=0)
        new = pc.build_record(kwargs_for([SYSTEM], "new"), now=time.time())
        self.store.write(old)
        self.store.write(new)
        past = time.time() - 2 * pc._BLOB_GRACE_SECONDS
        for base, _d, files in os.walk(self.store.blob_root):
            for f in files:
                os.utime(os.path.join(base, f), (past, past))
        with mock.patch.dict(os.environ, {"PROMPT_CAPTURE_RETENTION_DAYS": "7"}):
            self.store.prune()
        self.assertEqual(self.store.days(), [new["ts"][:10]])
        [kept] = list(self.store.iter_records(new["ts"][:10]))
        self.store.expand(kept["body"])  # every piece the kept line needs survived
        self.assertEqual(len(self.blob_files()), 2)  # system + tools; "old only" went

    def test_prune_keeps_fresh_unreferenced_blobs(self):
        # another worker process wrote the blob, its line is not on disk yet
        path = self.store.blob_path("ab" * 32)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write("{}")
        self.store.prune()
        self.assertTrue(os.path.exists(path))

    def test_size_ceiling_drops_oldest_days_first(self):
        for i, day in enumerate(["2026-10-01", "2026-10-02", "2026-10-03"]):
            ts = time.mktime(time.strptime(day, "%Y-%m-%d")) + 43200
            self.store.write(pc.build_record(kwargs_for([{"role": "user", "content": "x" * 400_000 + str(i)}], day), now=ts))
        with mock.patch.dict(os.environ, {"PROMPT_CAPTURE_RETENTION_DAYS": "365", "PROMPT_CAPTURE_MAX_MB": "1"}):
            self.store.prune(today="2026-10-03")
        self.assertEqual(self.store.days(), ["2026-10-02", "2026-10-03"])


class WriterTests(unittest.TestCase):
    def test_full_queue_drops_instead_of_blocking(self):
        with tempfile.TemporaryDirectory() as root:
            writer = pc.Writer.__new__(pc.Writer)
            writer.store = pc.Store(root)
            writer.queue = pc.queue.Queue(maxsize=1)
            writer.dropped = 0
            record = pc.build_record(kwargs_for([{"role": "user", "content": "hi"}]))
            writer.submit(record)
            writer.submit(record)  # must return at once
            self.assertEqual(writer.dropped, 1)

    def test_writer_thread_lands_the_capture(self):
        with tempfile.TemporaryDirectory() as root:
            writer = pc.Writer(root)
            writer.submit(pc.build_record(kwargs_for([SYSTEM], "w1")))
            deadline = time.time() + 5
            while time.time() < deadline and not writer.store.days():
                time.sleep(0.05)
            ids = [r["call_id"] for d in writer.store.days() for r in writer.store.iter_records(d)]
            self.assertEqual(ids, ["w1"])


class InstallTests(unittest.TestCase):
    def test_off_by_default_never_imports_litellm(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMPT_CAPTURE", None)
            with mock.patch.dict(sys.modules, {"litellm": None}):
                pc.install()  # would raise ImportError if it touched litellm

    def test_on_registers_a_callback_that_captures(self):
        litellm = types.ModuleType("litellm")
        litellm.callbacks = []
        custom = types.ModuleType("litellm.integrations.custom_logger")
        custom.CustomLogger = type("CustomLogger", (), {})
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(
            sys.modules,
            {"litellm": litellm, "litellm.integrations": types.ModuleType("litellm.integrations"), "litellm.integrations.custom_logger": custom},
        ), mock.patch.dict(os.environ, {"PROMPT_CAPTURE": "true", "PROMPT_CAPTURE_DIR": root}):
            pc.install()
            [callback] = litellm.callbacks
            callback.log_pre_api_call("served-name", [], kwargs_for([SYSTEM], "cb1"))
            callback.log_pre_api_call("served-name", [], {"additional_args": None})  # never raises
            store = pc.Store(root)
            deadline = time.time() + 5
            while time.time() < deadline and not store.days():
                time.sleep(0.05)
            self.assertEqual([r["call_id"] for d in store.days() for r in store.iter_records(d)], ["cb1"])


class ViewerTests(unittest.TestCase):
    def test_list_and_show(self):
        with tempfile.TemporaryDirectory() as root:
            store = pc.Store(root)
            store.write(pc.build_record(kwargs_for([SYSTEM, {"role": "user", "content": "질문"}], "v1")))
            out = self._run(["--dir", root, "list"])
            self.assertIn("v1", out)
            self.assertIn("session=sess-1", out)
            shown = self._run(["--dir", root, "show", "v1"])
            self.assertIn("mcp__wiki__search: Search the team wiki", shown)
            self.assertIn("질문", shown)
            raw = json.loads(self._run(["--dir", root, "show", "v1", "--raw"]))
            self.assertEqual(raw["tools"], TOOLS)

    def _run(self, argv):
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(pc.main(argv), 0)
        return buf.getvalue()


if __name__ == "__main__":
    unittest.main()
