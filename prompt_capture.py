"""
Capture exactly what each model receives — the request body LiteLLM sends to
the backend (vLLM / SGLang), after every transformation on the way.

Why here: oh-my-gateway and ChatDRAGON only see SDK-level events, and the
sanitizer only sees the Anthropic request before LiteLLM rewrites it. What the
model actually reads — the Claude Code system prompt, every tool schema (MCP
tools included), the full history, plus what LiteLLM itself adds (`max_tokens`,
`stream_options`, the served model name, the stripped thinking blocks) — exists
in one place only: LiteLLM's ``complete_input_dict`` at ``log_pre_api_call``,
which is byte-for-byte the JSON the backend receives.

OFF by default. ``PROMPT_CAPTURE=true`` turns it on. The captures hold user
conversations, tool results and anything the agent read — treat the directory
like a database of user data (volume, permissions, retention).

Storage (``PROMPT_CAPTURE_DIR``, default ``/app/prompt_captures``):

  YYYY-MM-DD.jsonl         one line per backend call (UTC day)
  blobs/ab/<sha256>.json   content-addressed pieces, written once

Claude Code resends the same system prompt, the same tool list and the whole
history on every turn, so each ``messages[i]`` and the ``tools`` list are stored
as blobs and the line keeps ``{"$blob": sha}`` references. The viewer puts the
body back together exactly (``show --raw``).

The request path never waits on disk: the hook serializes the body and hands it
to a single writer thread through a bounded queue. A full queue drops the
capture (counted, logged) — it never blocks or fails the model call.

Viewer::

    python prompt_capture.py list [--day YYYY-MM-DD] [--session ID] [--limit N]
    python prompt_capture.py show CALL_ID [--raw] [--tools-full]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, Iterator, List, Optional

_TRUTHY = ("true", "1", "yes", "on")
DEFAULT_DIR = "/app/prompt_captures"
_BLOB_GRACE_SECONDS = 3600

# Request headers worth keeping to tie a call back to a ChatDRAGON / gateway
# session. An allowlist on purpose: authorization and api keys never land here.
_KEPT_HEADERS = {
    "x-claude-code-session-id": "session_id",
    "x-claude-code-agent-id": "agent_id",
    "x-claude-code-parent-agent-id": "parent_agent_id",
    "x-claude-code-agent-type": "agent_type",
    "x-claude-code-request-class": "request_class",
    "x-claude-code-compaction": "compaction",
    "user-agent": "user_agent",
}


def enabled() -> bool:
    return os.environ.get("PROMPT_CAPTURE", "false").strip().lower() in _TRUTHY


def capture_dir() -> str:
    return os.environ.get("PROMPT_CAPTURE_DIR", "").strip() or DEFAULT_DIR


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default)).strip()))
    except ValueError:
        return default


def retention_days() -> int:
    """Whole UTC days to keep; 0 keeps only today."""
    return _int_env("PROMPT_CAPTURE_RETENTION_DAYS", 7)


def max_bytes() -> int:
    """Total size ceiling for the directory; oldest days go first. 0 = no ceiling."""
    return _int_env("PROMPT_CAPTURE_MAX_MB", 2048) * 1024 * 1024


# --------------------------------------------------------------------------- #
# Record building (pure — no I/O)
# --------------------------------------------------------------------------- #


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_body(body: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Replace each message and the tool list with blob references.

    Returns the reference body and ``{sha: serialized piece}`` for the blobs it
    points at. Everything else in the body is kept inline, untouched.
    """
    blobs: Dict[str, str] = {}

    def ref(piece: Any) -> Dict[str, str]:
        text = _dumps(piece)
        sha = _sha(text)
        blobs[sha] = text
        return {"$blob": sha}

    out = dict(body)
    if isinstance(body.get("messages"), list):
        out["messages"] = [ref(m) for m in body["messages"]]
    if isinstance(body.get("tools"), list) and body["tools"]:
        out["tools"] = ref(body["tools"])
    return out, blobs


def build_record(kwargs: Dict[str, Any], now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The capture line for one ``log_pre_api_call``, or None when there is no body."""
    additional = kwargs.get("additional_args") or {}
    body = additional.get("complete_input_dict")
    if not isinstance(body, dict):
        return None
    litellm_params = kwargs.get("litellm_params") or {}
    proxy_request = litellm_params.get("proxy_server_request") or {}
    headers = proxy_request.get("headers") if isinstance(proxy_request, dict) else None
    if not isinstance(headers, dict):
        headers = (litellm_params.get("metadata") or {}).get("headers") or {}
    metadata = litellm_params.get("metadata") or {}
    record: Dict[str, Any] = {
        "v": 1,
        "ts": _dt.datetime.fromtimestamp(now if now is not None else time.time(), _dt.timezone.utc).isoformat(),
        "call_id": kwargs.get("litellm_call_id"),
        "model_group": metadata.get("model_group") or litellm_params.get("model_group"),
        "provider": kwargs.get("custom_llm_provider"),
        "api_base": additional.get("api_base"),
    }
    lowered = {str(k).lower(): v for k, v in headers.items()}
    for header, field in _KEPT_HEADERS.items():
        if lowered.get(header):
            record[field] = str(lowered[header])
    record["body"] = body
    return record


# --------------------------------------------------------------------------- #
# Store (writer thread only)
# --------------------------------------------------------------------------- #


def _open_private(path: str, flags: int):
    return os.fdopen(os.open(path, flags, 0o600), "a" if flags & os.O_APPEND else "w", encoding="utf-8")


class Store:
    def __init__(self, root: str):
        self.root = root
        self.blob_root = os.path.join(root, "blobs")
        os.makedirs(self.blob_root, mode=0o700, exist_ok=True)

    def blob_path(self, sha: str) -> str:
        return os.path.join(self.blob_root, sha[:2], f"{sha}.json")

    def day_path(self, day: str) -> str:
        return os.path.join(self.root, f"{day}.jsonl")

    def write(self, record: Dict[str, Any]) -> None:
        ref_body, blobs = split_body(record["body"])
        for sha, text in blobs.items():
            path = self.blob_path(sha)
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
            tmp = f"{path}.tmp"
            with _open_private(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC) as fh:
                fh.write(text)
            os.replace(tmp, path)
        line = dict(record, body=ref_body)
        day = record["ts"][:10]
        with _open_private(self.day_path(day), os.O_WRONLY | os.O_CREAT | os.O_APPEND) as fh:
            fh.write(_dumps(line) + "\n")

    # ---- reading --------------------------------------------------------- #

    def days(self) -> List[str]:
        try:
            names = os.listdir(self.root)
        except FileNotFoundError:
            return []
        return sorted(n[:-6] for n in names if n.endswith(".jsonl") and len(n) == 16)

    def iter_records(self, day: str) -> Iterator[Dict[str, Any]]:
        try:
            with open(self.day_path(day), encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue  # a torn last line after a crash
        except FileNotFoundError:
            return

    def load_blob(self, sha: str) -> Any:
        with open(self.blob_path(sha), encoding="utf-8") as fh:
            return json.load(fh)

    def expand(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Put a reference body back together — exactly what the backend got."""

        def resolve(value: Any) -> Any:
            if isinstance(value, dict) and set(value) == {"$blob"}:
                return self.load_blob(value["$blob"])
            return value

        out = dict(body)
        if isinstance(body.get("messages"), list):
            out["messages"] = [resolve(m) for m in body["messages"]]
        if "tools" in body:
            out["tools"] = resolve(body["tools"])
        return out

    # ---- pruning --------------------------------------------------------- #

    def prune(self, today: Optional[str] = None) -> None:
        """Drop days past retention, then oldest days past the size ceiling, then orphan blobs."""
        today = today or _dt.datetime.now(_dt.timezone.utc).date().isoformat()
        cutoff = (_dt.date.fromisoformat(today) - _dt.timedelta(days=retention_days())).isoformat()
        for day in self.days():
            if day < cutoff:
                _remove(self.day_path(day))
        ceiling = max_bytes()
        if ceiling:
            # Size what would REMAIN: the kept days plus the blobs they still reference. Blobs
            # dominate, and an old day's blobs only go at collection — counting the raw directory
            # would keep dropping days that are not what fills it.
            while True:
                days = self.days()
                if len(days) <= 1 or self._live_bytes(days) <= ceiling:
                    break
                _remove(self.day_path(days[0]))
        self._collect_blobs()

    def _refs(self, days: List[str]) -> set:
        live = set()
        for day in days:
            for record in self.iter_records(day):
                body = record.get("body") or {}
                for value in list(body.get("messages") or []) + [body.get("tools")]:
                    if isinstance(value, dict) and "$blob" in value:
                        live.add(value["$blob"])
        return live

    def _live_bytes(self, days: List[str]) -> int:
        total = 0
        for path in [self.day_path(d) for d in days] + [self.blob_path(s) for s in self._refs(days)]:
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
        return total

    def _collect_blobs(self) -> None:
        live = self._refs(self.days())
        # Another LiteLLM worker process may have written a blob whose line is not on disk
        # yet — only collect blobs old enough that their line would have landed long ago.
        settled = time.time() - _BLOB_GRACE_SECONDS
        for base, _dirs, files in os.walk(self.blob_root):
            for name in files:
                path = os.path.join(base, name)
                if not name.endswith(".json") or name[:-5] in live:
                    continue
                try:
                    if os.path.getmtime(path) > settled:
                        continue
                except OSError:
                    continue
                _remove(path)


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Writer thread + LiteLLM hook
# --------------------------------------------------------------------------- #

_PRUNE_EVERY_SECONDS = 3600


class Writer:
    def __init__(self, root: str, maxsize: int = 256):
        self.store = Store(root)
        self.queue: "queue.Queue[str]" = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._last_prune = 0.0
        self._thread = threading.Thread(target=self._run, name="prompt-capture", daemon=True)
        self._thread.start()

    def submit(self, record: Dict[str, Any]) -> None:
        # Serialize now: the body dict belongs to LiteLLM and may change after the hook returns.
        try:
            self.queue.put_nowait(_dumps(record))
        except queue.Full:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 100 == 0:
                print(f"[prompt_capture] writer behind — dropped {self.dropped} capture(s)", flush=True)

    def _run(self) -> None:
        while True:
            if time.time() - self._last_prune >= _PRUNE_EVERY_SECONDS:
                self._last_prune = time.time()
                try:
                    self.store.prune()
                except Exception as exc:  # noqa: BLE001 — pruning must never stop capture
                    print(f"[prompt_capture] prune failed: {exc}", flush=True)
            try:
                line = self.queue.get(timeout=60)
            except queue.Empty:
                continue
            try:
                self.store.write(json.loads(line))
            except Exception as exc:  # noqa: BLE001 — a bad write loses one capture, not the thread
                print(f"[prompt_capture] write failed: {exc}", flush=True)


def install() -> None:
    """LITELLM_WORKER_STARTUP_HOOKS entry point — a no-op unless PROMPT_CAPTURE=true."""
    if not enabled():
        print("[prompt_capture] off (PROMPT_CAPTURE is not true)", flush=True)
        return
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    writer = Writer(capture_dir())

    class PromptCaptureCallback(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            try:
                record = build_record(kwargs)
                if record is not None:
                    writer.submit(record)
            except Exception as exc:  # noqa: BLE001 — capture never fails a model call
                print(f"[prompt_capture] capture failed: {exc}", flush=True)

    litellm.callbacks.append(PromptCaptureCallback())
    print(
        f"[prompt_capture] ON → {capture_dir()} (retention {retention_days()} d, "
        f"ceiling {max_bytes() // (1024 * 1024)} MB)",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Viewer
# --------------------------------------------------------------------------- #


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif part.get("type") in ("image_url", "image"):
                    parts.append("[image]")
                else:
                    parts.append(_dumps(part))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return "" if content is None else _dumps(content)


def _summary(record: Dict[str, Any]) -> str:
    body = record.get("body") or {}
    tools = body.get("tools")
    n_tools = "-" if tools is None else ("blob" if isinstance(tools, dict) else str(len(tools)))
    return "  ".join(
        [
            record.get("ts", "")[:19],
            str(record.get("call_id", "")),
            f"session={record.get('session_id', '-')}",
            f"agent={record.get('agent_type', '-')}",
            f"model={body.get('model', '-')}",
            f"msgs={len(body.get('messages') or [])}",
            f"tools={n_tools}",
        ]
    )


def _render(body: Dict[str, Any], tools_full: bool) -> str:
    lines: List[str] = []
    params = {k: v for k, v in body.items() if k not in ("messages", "tools")}
    lines.append("## params")
    lines.append(json.dumps(params, ensure_ascii=False, indent=2))
    tools = body.get("tools") or []
    lines.append(f"\n## tools ({len(tools)})")
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name", "?")
        if tools_full:
            lines.append(f"- {name}\n{json.dumps(fn, ensure_ascii=False, indent=2)}")
        else:
            desc = (fn.get("description") or "").strip().splitlines()
            lines.append(f"- {name}: {desc[0][:160] if desc else ''}")
    messages = body.get("messages") or []
    lines.append(f"\n## messages ({len(messages)})")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        lines.append(f"\n### [{i}] {role}")
        if not isinstance(msg, dict):
            lines.append(_dumps(msg))
            continue
        text = _text_of(msg.get("content"))
        if text:
            lines.append(text)
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            lines.append(f"→ tool_call {call.get('id', '')} {fn.get('name', '')}({fn.get('arguments', '')})")
        if msg.get("tool_call_id"):
            lines.append(f"(result for {msg['tool_call_id']})")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="prompt_capture", description="Read captured model requests.")
    parser.add_argument("--dir", default=capture_dir())
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list", help="one line per captured call")
    p_list.add_argument("--day", help="YYYY-MM-DD (UTC); default: every kept day")
    p_list.add_argument("--session", help="x-claude-code-session-id to filter on")
    p_list.add_argument("--limit", type=int, default=50, help="newest N (0 = all)")
    p_show = sub.add_parser("show", help="one call, reconstructed")
    p_show.add_argument("call_id")
    p_show.add_argument("--raw", action="store_true", help="the exact JSON body the backend received")
    p_show.add_argument("--tools-full", action="store_true", help="every tool's full schema")
    args = parser.parse_args(argv)

    store = Store(args.dir)
    days = [args.day] if getattr(args, "day", None) else store.days()
    if args.cmd == "list":
        rows = [
            r
            for day in days
            for r in store.iter_records(day)
            if not args.session or r.get("session_id") == args.session
        ]
        if args.limit:
            rows = rows[-args.limit :]
        for r in rows:
            print(_summary(r))
        return 0
    for day in reversed(days):
        for record in store.iter_records(day):
            if record.get("call_id") == args.call_id:
                try:
                    body = store.expand(record.get("body") or {})
                except FileNotFoundError as exc:
                    print(f"capture {args.call_id} lost a piece ({exc.filename})", file=sys.stderr)
                    return 1
                if args.raw:
                    print(json.dumps(body, ensure_ascii=False, indent=2))
                else:
                    meta = {k: v for k, v in record.items() if k != "body"}
                    print(json.dumps(meta, ensure_ascii=False, indent=2))
                    print(_render(body, args.tools_full))
                return 0
    print(f"no capture with call_id {args.call_id}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
