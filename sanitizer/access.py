"""Path gate: keep a Messages-only client to the Messages API (oh-my-gateway#286).

The sanitizer relays every path to LiteLLM. An agent session's CLI reaches it with the
key it needs for ``/v1/messages`` — and with that key it could also read ``/v1/models``,
``/model/info`` (backend api_base URLs, token limits, tier mappings), LiteLLM's admin
routes, and this proxy's own ``/openapi.json`` and ``/sanitizer/effort-levels``. In
production a session did exactly that while mapping the runtime.

With ``SANITIZER_PUBLIC_ROUTES`` set (comma-separated exact ``METHOD /path`` pairs, e.g.
``POST /v1/messages, POST /v1/messages/count_tokens``) only those requests — and
``GET /health`` — answer. Matching is exact: method, path and the raw bytes the client
sent must all agree, so no sub-path, trailing slash, other method or percent-encoded
spelling of a public route is public. Every other request needs the header
``X-Sanitizer-Internal: <SANITIZER_INTERNAL_TOKEN>``, held only by trusted callers (the
gateway's model discovery); without it the answer is a plain 404, so the gate does not
say what exists behind it. With the list set and no token configured, nothing else opens
at all (fail closed). Unset — the default — nothing changes.

A token-bearing request must still be canonical: ``..``/``.`` segments, ``//``, a
backslash, a percent-encoded ``/``, ``.``, ``\\`` or ``%`` (double encoding), and a
decoded ``?``, ``#``, ``;``, space, control or non-ASCII character are refused, so the upstream cannot
re-interpret the path after the check. The passthrough relays the client's raw path
bytes (``routes_passthrough``), so what LiteLLM decodes is what this gate decoded. The
internal header itself is never relayed upstream (``routes_messages`` strips it).
"""

from __future__ import annotations

import hmac
import logging
import posixpath

from .config import INTERNAL_HEADER, get_internal_token, get_public_routes

logger = logging.getLogger("sanitizer.access")

_ALWAYS_PUBLIC = frozenset({("GET", "/health")})
_ENCODED = (b"%2f", b"%2e", b"%5c", b"%25")
_DECODED_FORBIDDEN = ("\\", "%", "?", "#", ";")


def _path_only(raw_path: bytes) -> bytes:
    # uvicorn's raw_path never carries the query; httpx's ASGITransport does.
    return raw_path.split(b"?", 1)[0]


def is_canonical(path: str, raw_path: bytes) -> bool:
    if not path.startswith("/") or "//" in path:
        return False
    if not path.isascii() or any(ch in path for ch in _DECODED_FORBIDDEN) or any(ord(ch) < 0x21 or ord(ch) == 0x7F for ch in path):
        return False
    raw = _path_only(raw_path)
    if not raw.isascii() or any(token in raw.lower() for token in _ENCODED):
        return False
    trimmed = path.rstrip("/") or "/"
    return posixpath.normpath(trimmed) == trimmed


def is_public(method: str, path: str, raw_path: bytes, routes) -> bool:
    """Exact match on method, decoded path AND raw bytes (no encoded spelling)."""
    try:
        literal = path.encode("ascii")
    except UnicodeEncodeError:
        return False
    if _path_only(raw_path) != literal:
        return False
    key = (method.upper(), path)
    return key in _ALWAYS_PUBLIC or key in routes


def _header(scope, name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def allowed(scope) -> bool:
    routes = get_public_routes()
    if routes is None:
        return True
    path = scope.get("path") or "/"
    raw_path = scope.get("raw_path") or path.encode("utf-8", "surrogateescape")
    if is_public(scope.get("method") or "GET", path, raw_path, routes):
        return True
    if not is_canonical(path, raw_path):
        return False
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
        logger.info("path gate refused %s %r", scope.get("method"), scope.get("raw_path") or scope.get("path"))
        body = b'{"error":{"type":"not_found","message":"Not Found"}}'
        await send(
            {
                "type": "http.response.start",
                "status": 404,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})
