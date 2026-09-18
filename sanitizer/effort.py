"""Reasoning-effort vocabulary: what the upstream accepts, and how to get there.

The vocabularies on the two sides of this proxy do not match, and the mismatch
is fatal rather than cosmetic. Claude Code / oh-my-gateway speak the Anthropic
SDK levels (``low``/``medium``/``high``/``xhigh``/``max``); a vLLM build answers
a level it does not know with a 400 and takes the whole turn with it::

    Unexpected reasoning effort high. Supported types are xhigh (default),
    medium, and low.

So ``high`` — the level a caller using the OpenAI-standard vocabulary is most
likely to send — kills every request on an upstream whose accepted set is
``{low, medium, xhigh}`` (issue #26).

Which levels an upstream accepts is a property of the served model's chat
template, not something this proxy can read off the request, so it is declared:

``SANITIZER_EFFORT_SUPPORTED``
    Comma-separated levels the upstream accepts, e.g. ``low,medium,xhigh``.
    Unset or empty means **forward verbatim** — the default, and the behavior
    before this module existed (issue #25): normalizing a level without knowing
    the model is a guess, and a wrong guess turns a working request into a 400.

With the set declared, an unsupported level is clamped to the nearest supported
one, **preferring upward**: a caller who asked for ``high`` gets ``xhigh``
rather than ``medium``, because reasoning less than asked is the worse of the
two failures. When nothing above is supported the nearest level below is used.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger("sanitizer.effort")

# Ordered weakest → strongest. ``minimal`` is not a Claude Code level but both
# vLLM's and SGLang's schemas define it, so it is placed rather than dropped.
# ``none`` is deliberately absent: on the Anthropic side it disables extended
# thinking instead of naming a level, and it never reaches this module.
EFFORT_SCALE: Tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")

_ENV = "SANITIZER_EFFORT_SUPPORTED"


def supported_levels() -> Tuple[str, ...]:
    """Levels the upstream accepts, in scale order; empty means "forward as-is".

    Entries outside :data:`EFFORT_SCALE` are dropped with a warning — a typo
    must not silently narrow the set and start clamping good levels away. A
    value that leaves nothing usable also means "forward as-is", so a
    misconfiguration degrades to the previous behavior rather than to 400s.
    """
    raw = os.environ.get(_ENV, "").strip()
    if not raw:
        return ()
    wanted = [part.strip().lower() for part in raw.split(",")]
    known = [level for level in EFFORT_SCALE if level in wanted]
    unknown = sorted({p for p in wanted if p and p not in EFFORT_SCALE})
    if unknown:
        logger.warning(
            "ignoring unknown %s entries %s (known: %s)",
            _ENV,
            ", ".join(unknown),
            ", ".join(EFFORT_SCALE),
        )
    if not known:
        logger.warning("%s=%r leaves no usable level; forwarding effort as-is", _ENV, raw)
    return tuple(known)


def clamp_to_supported(level: str) -> Optional[str]:
    """Map *level* onto the configured supported set.

    Returns the level unchanged when no set is configured or it is already
    supported, the nearest supported level otherwise (upward first), and
    ``None`` only for a level outside :data:`EFFORT_SCALE` entirely — the
    caller decides what to do with one of those.
    """
    supported = supported_levels()
    if not supported:
        return level
    if level in supported:
        return level
    if level not in EFFORT_SCALE:
        return None
    index = EFFORT_SCALE.index(level)
    # Upward first: under-reasoning is the worse miss. ``high`` on a
    # {low, medium, xhigh} upstream becomes ``xhigh``, not ``medium``.
    for candidate in EFFORT_SCALE[index + 1 :]:
        if candidate in supported:
            logger.info("clamping reasoning effort %r up to %r", level, candidate)
            return candidate
    for candidate in reversed(EFFORT_SCALE[:index]):
        if candidate in supported:
            logger.info("clamping reasoning effort %r down to %r", level, candidate)
            return candidate
    return level  # unreachable: a non-empty set always has a neighbour
