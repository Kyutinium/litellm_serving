"""Reasoning-effort vocabulary, per model: what each upstream accepts.

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
template, and one sanitizer fronts a LiteLLM that routes to **many** models at
once (GLM, Qwen, Gemma, SAMUEL, …) whose accepted subsets differ. A single
proxy-wide set is therefore not a correctness contract: it would rewrite every
model's ``high`` into one model's vocabulary. The setting is a map, keyed by the
model name exactly as clients send it:

``SANITIZER_EFFORT_VOCABULARY``
    JSON object ``{"<model or alias>": [levels] | "csv"}``, e.g.
    ``{"qwen3.6-27b": "low,medium,xhigh", "Qwen3.6-27B": "low,medium,xhigh"}``.
    LiteLLM names are case-sensitive and this repository registers case
    variants as separate aliases, so every alias a client may send is listed
    (the config renderer expands them). Unset or empty means **forward
    verbatim** for every model — the default, and the behavior before this
    module existed (issue #25). A model with no entry is forwarded verbatim.

With a model's set declared, an unsupported level is clamped to the nearest
declared one, **preferring upward**: a caller who asked for ``high`` gets
``xhigh`` rather than ``medium``, because reasoning less than asked is the
worse of the two failures. When nothing above is declared the nearest level
below is used. On a three-level upstream this deliberately collapses
``high``/``xhigh``/``max`` onto ``xhigh`` at the model input; preventing the 400
is the job here, and telling users which levels a model really has belongs to
the capability surface upstream of this proxy.

Validity is all-or-nothing. An operator-declared capability with a typo in it
(``low,medium,xhi``) must not quietly become a *narrower* declaration that
clamps good levels away — dropping the unknown entry and keeping the rest does
exactly that. So an unknown level, an unknown shape, or unparseable JSON makes
the **whole** setting invalid: :func:`validate_at_startup` refuses to start the
process, and a request served under an invalid setting anyway (the env changed
under a running process) is forwarded verbatim for every model, with one
warning. Invalid never narrows.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger("sanitizer.effort")

# Ordered weakest → strongest. ``minimal`` is not a Claude Code level but both
# vLLM's and SGLang's schemas define it, so it is placed rather than dropped.
# ``none`` is deliberately absent: on the Anthropic side it disables extended
# thinking instead of naming a level, and it never reaches this module.
EFFORT_SCALE: Tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")

ENV = "SANITIZER_EFFORT_VOCABULARY"

Vocabulary = Dict[str, Tuple[str, ...]]


class EffortVocabularyError(ValueError):
    """``SANITIZER_EFFORT_VOCABULARY`` is set but not a valid declaration."""


def _levels(model: str, value: object) -> Tuple[str, ...]:
    if isinstance(value, str):
        parts = [p.strip().lower() for p in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = [str(p).strip().lower() for p in value]
    else:
        raise EffortVocabularyError(
            f"{ENV}[{model!r}] must be a list or a comma-separated string, "
            f"got {type(value).__name__}"
        )
    parts = [p for p in parts if p]
    unknown = sorted({p for p in parts if p not in EFFORT_SCALE})
    if unknown:
        raise EffortVocabularyError(
            f"{ENV}[{model!r}] names unknown level(s) {', '.join(unknown)}; "
            f"known: {', '.join(EFFORT_SCALE)}"
        )
    if not parts:
        raise EffortVocabularyError(f"{ENV}[{model!r}] declares no levels")
    # Scale order, deduplicated; callers rely on the ordering for "nearest".
    return tuple(level for level in EFFORT_SCALE if level in parts)


def parse_vocabulary(raw: Optional[str]) -> Vocabulary:
    """Parse the setting, or raise :class:`EffortVocabularyError`.

    ``None``/empty → ``{}`` (verbatim everywhere). Anything else must be a JSON
    object of model → levels with every level on :data:`EFFORT_SCALE`.
    """
    if raw is None or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise EffortVocabularyError(f"{ENV} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EffortVocabularyError(
            f"{ENV} must be a JSON object of model -> levels, got {type(data).__name__}"
        )
    vocab: Vocabulary = {}
    for model, value in data.items():
        if not isinstance(model, str) or not model.strip():
            raise EffortVocabularyError(f"{ENV} has an empty model name")
        vocab[model] = _levels(model, value)
    return vocab


def validate_at_startup() -> Vocabulary:
    """Parse the live setting and raise so a misconfigured process never starts."""
    vocab = parse_vocabulary(os.environ.get(ENV))
    if vocab:
        logger.info("effort vocabulary declared for %d model name(s)", len(vocab))
    return vocab


def vocabulary() -> Vocabulary:
    """The live setting, or ``{}`` when unset **or invalid**.

    Invalid must never narrow: a bad declaration is treated as no declaration,
    so every model is forwarded verbatim. The startup check is what turns that
    into a hard failure for a fresh deploy.
    """
    try:
        return parse_vocabulary(os.environ.get(ENV))
    except EffortVocabularyError as exc:
        logger.warning("%s; forwarding effort verbatim for every model", exc)
        return {}


def supported_levels(model: Optional[str]) -> Tuple[str, ...]:
    """Levels declared for *model*, in scale order; empty means "forward as-is"."""
    if not model:
        return ()
    return vocabulary().get(model, ())


def clamp_to_supported(level: str, model: Optional[str]) -> Optional[str]:
    """Map *level* onto *model*'s declared set.

    Returns the level unchanged when the model has no declaration or the level
    is already declared, the nearest declared level otherwise (upward first),
    and ``None`` only for a level outside :data:`EFFORT_SCALE` entirely — the
    caller decides what to do with one of those.
    """
    supported = supported_levels(model)
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
            logger.info("clamping %s effort %r up to %r", model, level, candidate)
            return candidate
    for candidate in reversed(EFFORT_SCALE[:index]):
        if candidate in supported:
            logger.info("clamping %s effort %r down to %r", model, level, candidate)
            return candidate
    return level  # unreachable: a non-empty set always has a neighbour
