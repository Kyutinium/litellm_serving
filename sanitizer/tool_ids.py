"""Keep every ``tool_use`` id unique across the conversation.

vLLM's Kimi tool parser hands back ``functions.<name>:<index>`` ids, and the
model does not reliably keep that index global: calling the same tool on a
later turn can return ``functions.<name>:0`` again. Anthropic ids are unique per
conversation, and the clients behind this proxy rely on that:

* the Claude Code CLI drops a later ``tool_use``/``tool_result`` pair whose id
  already appears in the history, so the model never sees its own repeated call
  or its result and calls the tool again — a loop;
* a UI that keys tool calls by id (ChatDRAGON) folds the new, running call into
  the first, finished card: the call is invisible while it runs and its result
  overwrites the old one.

The request carries the whole history, so the sanitizer knows every id already
in use. :class:`ToolIdAllocator` passes an upstream id through when it is new
and replaces it when it repeats (or is missing) — with the next index in the
Kimi format when the upstream id has that shape, so the model keeps seeing ids
it would have produced, else with a ``toolu_`` id. The client stores the
rewritten id and sends it back on the next request, where it is history like
any other.
"""

from __future__ import annotations

import re
import uuid
from typing import Dict, Iterable, Optional, Set

_KIMI_ID = re.compile(r"^functions\.(?P<name>.+):(?P<index>\d+)$")
_TOOL_USE_TYPES = ("tool_use", "server_tool_use")


def history_tool_ids(body: Dict) -> Set[str]:
    """Every tool id an Anthropic request's messages already use."""
    ids: Set[str] = set()
    messages = body.get("messages") if isinstance(body, dict) else None
    for message in messages or []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in _TOOL_USE_TYPES:
                value = block.get("id")
            elif block.get("type") == "tool_result":
                value = block.get("tool_use_id")
            else:
                continue
            if isinstance(value, str) and value:
                ids.add(value)
    return ids


class ToolIdAllocator:
    """Hands out ids unique against the history and this response."""

    def __init__(self, used: Iterable[str] = ()) -> None:
        self.used: Set[str] = set(used)
        self.rewritten = 0

    def assign(self, tool_id: Optional[str], name: Optional[str] = None) -> str:
        if isinstance(tool_id, str) and tool_id and tool_id not in self.used:
            self.used.add(tool_id)
            return tool_id
        fresh = self._fresh(tool_id, name)
        self.used.add(fresh)
        self.rewritten += 1
        return fresh

    def _fresh(self, tool_id: Optional[str], name: Optional[str]) -> str:
        match = _KIMI_ID.match(tool_id or "")
        if match:
            # Kimi documents the index as a counter over the whole conversation.
            indices = [
                int(m.group("index"))
                for m in (_KIMI_ID.match(used) for used in self.used)
                if m
            ]
            index = max(indices, default=-1) + 1
            while True:
                candidate = f"functions.{match.group('name')}:{index}"
                if candidate not in self.used:
                    return candidate
                index += 1
        while True:
            candidate = "toolu_" + uuid.uuid4().hex[:24]
            if candidate not in self.used:
                return candidate


def dedupe_body_tool_ids(body: Dict, allocator: ToolIdAllocator) -> Dict:
    """Return *body* (a non-streaming Anthropic message) with unique tool ids.

    The same object when nothing changed; a shallow copy otherwise.
    """
    content = body.get("content") if isinstance(body, dict) else None
    if not isinstance(content, list):
        return body
    changed = False
    out = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in _TOOL_USE_TYPES:
            new_id = allocator.assign(block.get("id"), block.get("name"))
            if new_id != block.get("id"):
                block = {**block, "id": new_id}
                changed = True
        out.append(block)
    if not changed:
        return body
    return {**body, "content": out}
