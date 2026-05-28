"""
Optional request/response logger for the LiteLLM proxy.

Activated when LITELLM_LOG_FILE is set to a non-empty path. When unset (or
empty) the logger registers nothing, so there is zero overhead.

Environment variables
---------------------
LITELLM_LOG_FILE         Path to the JSON Lines log file. Empty/unset = OFF.
LITELLM_LOG_LEVEL        "info"  (default) - metadata only (model, user, latency, usage, status)
                         "debug"           - also include request messages and response text
LITELLM_LOG_MAX_BYTES    Rotating handler max bytes per file (default 50_000_000).
LITELLM_LOG_BACKUP_COUNT Number of rotated files to keep (default 5).

Each log line is a single JSON object terminated by a newline:

    {"ts": "...", "event": "success", "model": "...", "user": "...",
     "latency_ms": 1234.5, "usage": {...}, "status": "ok", "request_id": "..."}

Failures share the same shape with "event": "failure" and an "error" field.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

import litellm
from litellm.integrations.custom_logger import CustomLogger


LOG_FILE = os.environ.get("LITELLM_LOG_FILE", "").strip()
LOG_LEVEL = os.environ.get("LITELLM_LOG_LEVEL", "info").strip().lower()
MAX_BYTES = int(os.environ.get("LITELLM_LOG_MAX_BYTES", str(50 * 1024 * 1024)))
BACKUP_COUNT = int(os.environ.get("LITELLM_LOG_BACKUP_COUNT", "5"))

_VALID_LEVELS = ("info", "debug")
if LOG_LEVEL not in _VALID_LEVELS:
    print(
        f"[request_logger] WARNING: invalid LITELLM_LOG_LEVEL='{LOG_LEVEL}', "
        f"falling back to 'info'. Valid values: {_VALID_LEVELS}",
        flush=True,
    )
    LOG_LEVEL = "info"


def _hash_key(api_key: str | None) -> str | None:
    if not api_key:
        return None
    return "sha256:" + hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]


def _latency_ms(start_time: Any, end_time: Any) -> float | None:
    if start_time is None or end_time is None:
        return None
    try:
        delta = end_time - start_time
        return round(delta.total_seconds() * 1000.0, 2)
    except Exception:
        return None


def _extract_usage(response_obj: Any) -> dict | None:
    if response_obj is None:
        return None
    usage = None
    if isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    else:
        usage = getattr(response_obj, "usage", None)
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    for fn in ("model_dump", "dict"):
        if hasattr(usage, fn):
            try:
                return getattr(usage, fn)()
            except Exception:
                pass
    return {
        k: getattr(usage, k, None)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _extract_response_text(response_obj: Any) -> str | None:
    if response_obj is None:
        return None
    try:
        if isinstance(response_obj, dict):
            choices = response_obj.get("choices") or []
        else:
            choices = getattr(response_obj, "choices", None) or []
        if not choices:
            return None
        first = choices[0]
        message = (
            first.get("message")
            if isinstance(first, dict)
            else getattr(first, "message", None)
        )
        if message is None:
            return None
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "".join(parts) or None
    except Exception:
        return None
    return None


def _build_record(
    event: str,
    kwargs: dict,
    response_obj: Any,
    start_time: Any,
    end_time: Any,
    error: Any = None,
) -> dict:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    proxy_metadata = kwargs.get("proxy_server_request") or {}

    record: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "model": kwargs.get("model"),
        "user": _hash_key(kwargs.get("user") or metadata.get("user_api_key")),
        "request_id": (
            kwargs.get("litellm_call_id")
            or metadata.get("request_id")
            or proxy_metadata.get("request_id")
        ),
        "call_type": kwargs.get("call_type"),
        "stream": bool(kwargs.get("stream")),
        "latency_ms": _latency_ms(start_time, end_time),
        "usage": _extract_usage(response_obj),
    }
    if error is not None:
        record["error"] = str(error)[:1000]

    if LOG_LEVEL == "debug":
        messages = kwargs.get("messages")
        if messages is not None:
            record["messages"] = messages
        text = _extract_response_text(response_obj)
        if text is not None:
            record["response_text"] = text

    return record


def _build_handler() -> logging.Handler:
    directory = os.path.dirname(LOG_FILE)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


class _JsonLineLogger:
    def __init__(self) -> None:
        self._logger = logging.getLogger("litellm_serving.request_logger")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if not self._logger.handlers:
            self._logger.addHandler(_build_handler())

    def write(self, record: dict) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except Exception as e:
            line = json.dumps(
                {"ts": record.get("ts"), "event": "log_serialize_error", "error": str(e)}
            )
        self._logger.info(line)


class RequestLoggingCallback(CustomLogger):
    """LiteLLM callback that writes one JSON line per request."""

    def __init__(self, sink: _JsonLineLogger) -> None:
        super().__init__()
        self._sink = sink

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._sink.write(_build_record("success", kwargs, response_obj, start_time, end_time))

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._sink.write(_build_record("success", kwargs, response_obj, start_time, end_time))

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        error = kwargs.get("exception") or kwargs.get("error")
        self._sink.write(
            _build_record("failure", kwargs, response_obj, start_time, end_time, error=error)
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        error = kwargs.get("exception") or kwargs.get("error")
        self._sink.write(
            _build_record("failure", kwargs, response_obj, start_time, end_time, error=error)
        )


def register() -> None:
    """Register the JSON Lines request logger if LITELLM_LOG_FILE is set."""
    if not LOG_FILE:
        print(
            "[request_logger] LITELLM_LOG_FILE not set — request logging DISABLED",
            flush=True,
        )
        return

    try:
        sink = _JsonLineLogger()
    except Exception as e:
        print(
            f"[request_logger] Failed to initialize log file '{LOG_FILE}': {e}",
            file=sys.stderr,
            flush=True,
        )
        return

    callback = RequestLoggingCallback(sink)
    litellm.callbacks.append(callback)
    print(
        f"[request_logger] ENABLED file={LOG_FILE} level={LOG_LEVEL} "
        f"max_bytes={MAX_BYTES} backups={BACKUP_COUNT}",
        flush=True,
    )
