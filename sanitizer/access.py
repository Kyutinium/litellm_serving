"""Path gate: keep a Messages-only client to the Messages API (oh-my-gateway#286).

The sanitizer relays every path to LiteLLM. An agent session's CLI reaches it with the
key it needs for ``/v1/messages`` — and with that key it could also read ``/v1/models``,
``/model/info`` (backend api_base URLs, token limits, tier mappings), LiteLLM's admin
routes, and this proxy's own ``/openapi.json`` and ``/sanitizer/effort-levels``. In
production a session did exactly that while mapping the runtime.

With ``SANITIZER_PUBLIC_PATHS`` set (comma-separated path prefixes, e.g. ``/v1/messages``)
only those prefixes — and ``/health`` — answer. Every other path needs the header
``X-Sanitizer-Internal: <SANITIZER_INTERNAL_TOKEN>``, held only by trusted callers (the
gateway's model discovery); without it the answer is a plain 404, so the gate does not
say what exists behind it. With the list set and no token configured, no internal path
opens at all (fail closed). Unset — the default — nothing changes.

A path that is not in canonical form (``..``/``.`` segments, ``//``, a backslash, or a
percent-encoded ``/``, ``.`` or ``\\``) is refused before any matching: the upstream may
normalise ``/v1/messages/../model/info`` into ``/model/info`` after the prefix check.
The internal header itself is never relayed upstream (``routes_messages`` strips it).
"""

from __future__ import annotations

import hmac
import logging
import posixpath

from .config import INTERNAL_HEADER, get_internal_token, get_public_paths

logger = logging.getLogger("sanitizer.access")

_ALWAYS_PUBLIC = ("/health",)
_ENCODED = (b"%2f", b"%2e", b"%5c")


def is_canonical(path: str, raw_path: bytes) -> bool:
    if not path.startswith("/") or "\\" in path or "//" in path:
        return False
    if any(token in raw_path.lower() for token in _ENCODED):
        return False
    trimmed = path.rstrip("/") or "/"
    return posixpath.normpath(trimmed) == trimmed


def is_public(path: str, prefixes) -> bool:
    trimmed = path.rstrip("/") or "/"
    for prefix in (*_ALWAYS_PUBLIC, *prefixes):
        if trimmed == prefix or trimmed.startswith(prefix + "/"):
            return True
    return False


def _header(scope, name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def allowed(scope) -> bool:
    prefixes = get_public_paths()
    if prefixes is None:
        return True
    path = scope.get("path") or "/"
    raw_path = scope.get("raw_path") or path.encode()
    if not is_canonical(path, raw_path):
        return False
    if is_public(path, prefixes):
        return True
    token = get_internal_token()
    given = _header(scope, INTERNAL_HEADER.encode())
    return bool(token) and bool(given) and hmac.compare_digest(given.encode(), token.encode())


class PathGate:
    """Pure ASGI wrapper (no ``BaseHTTPMiddleware``: it buffers streamed responses)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or allowed(scope):
            await self.app(scope, receive, send)
            return
        logger.info("path gate refused %s %s", scope.get("method"), scope.get("path"))
        body = b'{"error":{"type":"not_found","message":"Not Found"}}'
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})
