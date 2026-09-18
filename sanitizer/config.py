"""Environment-variable-based configuration for the sanitizer.

Intentionally free of pydantic — the sanitizer keeps its dependency surface to
``fastapi``/``httpx``/``uvicorn`` only. Every setting is resolved lazily from
``os.environ`` so tests can monkeypatch the environment without re-importing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("sanitizer.config")

_DEFAULT_UPSTREAM = "http://localhost:3999"
_DEFAULT_PORT = 3996

_VALID_THINK_MODES = ("default", "none", "text", "think_tag", "bridge")
_TRUTHY = ("true", "1", "yes", "on")
_FALSY = ("false", "0", "no", "off")


def get_upstream_url() -> str:
    """Return the upstream LiteLLM base URL with any trailing slash removed."""
    raw = os.environ.get("UPSTREAM_BASE_URL", _DEFAULT_UPSTREAM).strip()
    if not raw:
        raw = _DEFAULT_UPSTREAM
    return raw.rstrip("/")


def get_port() -> int:
    """Return the sanitizer listen port, falling back to 3996 on invalid input."""
    raw = os.environ.get("SANITIZER_PORT", str(_DEFAULT_PORT)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(
            "invalid SANITIZER_PORT=%r, falling back to %d", raw, _DEFAULT_PORT
        )
        return _DEFAULT_PORT


def get_tls_verify():
    """Return the value passed to httpx's ``verify``.

    Accepts truthy/falsy strings (``true/1/yes/on`` and ``false/0/no/off``,
    case-insensitive) mapping to booleans; anything else is treated as a path
    to a CA bundle (httpx accepts a path string for ``verify``).
    """
    raw = os.environ.get("SANITIZER_TLS_VERIFY", "true").strip()
    if not raw:
        return True
    low = raw.lower()
    if low in _TRUTHY:
        return True
    if low in _FALSY:
        return False
    # Not a boolean literal → treat as a CA bundle path.
    return raw


def get_request_timeout_seconds():
    """Return the per-request timeout in seconds, or ``None`` for no timeout.

    ``0``, empty, negative, or non-numeric values resolve to ``None`` so that
    long-lived streaming responses are never cut off by a client timeout.
    """
    raw = os.environ.get("SANITIZER_REQUEST_TIMEOUT", "0").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("invalid SANITIZER_REQUEST_TIMEOUT=%r, treating as none", raw)
        return None
    if value <= 0:
        return None
    return value


def is_openai_bridge_enabled() -> bool:
    """Return whether the Anthropic↔OpenAI bridge route is enabled."""
    raw = os.environ.get("SANITIZER_USE_OPENAI_BRIDGE", "false").strip().lower()
    return raw in _TRUTHY


def forwards_reasoning_effort() -> bool:
    """Whether the bridge forwards Anthropic ``output_config.effort`` upstream.

    Default **on**: without it the selected effort is silently dropped in the
    bridge, so a client that offers an effort control offers a no-op (issue #24).
    The escape hatch exists because ``reasoning_effort`` is an extra field for an
    upstream that validates its request schema strictly — a deployment whose
    vLLM/SGLang build rejects unknown parameters can turn the forwarding off
    without giving up the bridge itself.
    """
    raw = os.environ.get("SANITIZER_FORWARD_REASONING_EFFORT", "true").strip().lower()
    return raw not in _FALSY


def get_think_output_mode() -> str:
    """Return the sanitizer's validated think-output mode.

    Read from ``SANITIZER_THINK_OUTPUT_MODE`` first. ``THINK_OUTPUT_MODE`` is
    also read by ``strip_thinking.py`` inside LiteLLM, whose modes mean
    something different (it patches LiteLLM's own Anthropic adapter), so a
    deployment running both processes needs to set them independently. The
    shared name remains the fallback so existing deployments keep working.
    """
    raw = os.environ.get("SANITIZER_THINK_OUTPUT_MODE", "").strip().lower()
    source = "SANITIZER_THINK_OUTPUT_MODE"
    if not raw:
        raw = os.environ.get("THINK_OUTPUT_MODE", "default").strip().lower()
        source = "THINK_OUTPUT_MODE"
    if raw not in _VALID_THINK_MODES:
        logger.warning(
            "invalid %s=%r, falling back to 'default'. valid: %s",
            source,
            raw,
            _VALID_THINK_MODES,
        )
        return "default"
    return raw
