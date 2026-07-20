"""``transform_events`` — ``THINK_OUTPUT_MODE`` post-processing.

Applied *after* :func:`sanitizer.sanitize_events`, so every event it sees is
already spec-conforming. It rewrites how thinking/reasoning content is surfaced:

* ``default`` / ``bridge`` — pass through (spec-accurate thinking; ``bridge`` is
  handled at the route level, not here);
* ``none`` / ``text`` — promote thinking to regular text: ``thinking_delta`` →
  ``text_delta``, ``signature_delta`` → empty ``text_delta`` (suppressed), and
  ``content_block_start`` of type ``thinking`` rewritten to ``text``;
* ``think_tag`` — wrap thinking in the unicode markers ``◰\\n`` / ``\\n◱\\n\\n``
  emitted as ``text_delta`` chunks, closing the tag on the first
  ``signature_delta`` or ``text_delta``; thinking block starts rewritten to text.
"""

from __future__ import annotations

from typing import AsyncIterator, Dict

_THINK_TAG_OPEN = "◰\n"
_THINK_TAG_CLOSE = "\n◱\n\n"


def _rewrite_thinking_start(event: Dict) -> Dict:
    """Return a copy of a ``content_block_start`` with a text block payload."""
    new_event = dict(event)
    new_event["content_block"] = {"type": "text", "text": ""}
    return new_event


async def _passthrough(upstream: AsyncIterator[Dict]) -> AsyncIterator[Dict]:
    async for event in upstream:
        yield event


async def _to_text(upstream: AsyncIterator[Dict]) -> AsyncIterator[Dict]:
    async for event in upstream:
        etype = event.get("type")
        if etype == "content_block_start":
            cb = event.get("content_block", {}) or {}
            if cb.get("type") == "thinking":
                yield _rewrite_thinking_start(event)
                continue
            yield event
        elif etype == "content_block_delta":
            delta = event.get("delta", {}) or {}
            dtype = delta.get("type")
            if dtype == "thinking_delta":
                new_event = dict(event)
                new_event["delta"] = {"type": "text_delta", "text": delta.get("thinking", "")}
                yield new_event
            elif dtype == "signature_delta":
                new_event = dict(event)
                new_event["delta"] = {"type": "text_delta", "text": ""}
                yield new_event
            else:
                yield event
        else:
            yield event


async def _think_tag(upstream: AsyncIterator[Dict]) -> AsyncIterator[Dict]:
    tag_open = False
    async for event in upstream:
        etype = event.get("type")
        if etype == "content_block_start":
            cb = event.get("content_block", {}) or {}
            if cb.get("type") == "thinking":
                yield _rewrite_thinking_start(event)
                continue
            yield event
        elif etype == "content_block_delta":
            delta = event.get("delta", {}) or {}
            dtype = delta.get("type")
            if dtype == "thinking_delta":
                text = delta.get("thinking", "")
                if not tag_open:
                    text = _THINK_TAG_OPEN + text
                    tag_open = True
                new_event = dict(event)
                new_event["delta"] = {"type": "text_delta", "text": text}
                yield new_event
            elif dtype == "signature_delta":
                if tag_open:
                    new_event = dict(event)
                    new_event["delta"] = {"type": "text_delta", "text": _THINK_TAG_CLOSE}
                    tag_open = False
                    yield new_event
                # else: nothing to emit (no open tag to close)
            elif dtype == "text_delta":
                if tag_open:
                    new_event = dict(event)
                    new_event["delta"] = {
                        "type": "text_delta",
                        "text": _THINK_TAG_CLOSE + delta.get("text", ""),
                    }
                    tag_open = False
                    yield new_event
                else:
                    yield event
            else:
                yield event
        else:
            yield event


def transform_events(upstream: AsyncIterator[Dict], mode: str) -> AsyncIterator[Dict]:
    if mode in ("none", "text"):
        return _to_text(upstream)
    if mode == "think_tag":
        return _think_tag(upstream)
    # "default" and "bridge" — no post-processing.
    return _passthrough(upstream)
