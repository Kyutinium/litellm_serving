"""Reasoning-effort vocabulary, per model: what each upstream accepts.

**The vocabulary is learned, not hand-written.** Which levels a served model takes
is a fact about its chat template, and the upstream states that fact itself: a
vLLM build answers an unsupported level with a 400 whose text lists the supported
ones, and a 1-token completion per level answers it for builds that word the error
differently. So this module keeps a *learned* table — filled from real 400s on
the request path (the request is then retried once with a level the model takes)
and from an optional startup probe (:func:`probe_models`) — and publishes it on the
relayed ``/v1/models`` as ``effort_levels`` so the gateway in front can advertise
the same set without an operator copying it. ``SANITIZER_EFFORT_VOCABULARY`` stays
as a manual override for an upstream that neither names its levels nor probes.

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
import re
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

import httpx

logger = logging.getLogger("sanitizer.effort")

# Ordered weakest → strongest. ``minimal`` is not a Claude Code level but both
# vLLM's and SGLang's schemas define it, so it is placed rather than dropped.
# ``none`` is deliberately absent: on the Anthropic side it disables extended
# thinking instead of naming a level, and it never reaches this module.
EFFORT_SCALE: Tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")

ENV = "SANITIZER_EFFORT_VOCABULARY"

Vocabulary = Dict[str, Tuple[str, ...]]

# Learned per model name (as clients send it), scale-ordered. Process-local: the
# upstream is the source of truth and re-teaches after a restart (a 400 costs one
# retry; the startup probe costs a handful of 1-token requests).
_learned: Dict[str, Tuple[str, ...]] = {}
_learned_source: Dict[str, str] = {}


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
    """Levels known for *model*, in scale order; empty means "forward as-is".

    A declared entry (``SANITIZER_EFFORT_VOCABULARY``) wins over a learned one —
    the operator override exists for upstreams that cannot be learned from.
    """
    if not model:
        return ()
    declared = vocabulary().get(model)
    if declared:
        return declared
    return _learned.get(model, ())


def learned_levels() -> Dict[str, Tuple[str, ...]]:
    """The learned table (copy), for diagnostics and ``/v1/models`` enrichment."""
    return dict(_learned)


def known_levels() -> Dict[str, Tuple[str, ...]]:
    """Declared ∪ learned, declared winning — everything the proxy knows."""
    out = dict(_learned)
    out.update(vocabulary())
    return out


def level_source(model: str) -> str:
    if model in vocabulary():
        return "declared"
    return _learned_source.get(model, "")


def record_levels(model: str, levels: Iterable[str], source: str) -> Tuple[str, ...]:
    """Store what the upstream told us about *model*; returns the scale-ordered set."""
    ordered = tuple(level for level in EFFORT_SCALE if level in set(levels))
    if not ordered or not model:
        return ()
    previous = _learned.get(model)
    _learned[model] = ordered
    _learned_source[model] = source
    if previous != ordered:
        logger.info("learned effort levels for %s from %s: %s", model, source, ",".join(ordered))
    return ordered


def forget_learned() -> None:
    """Drop the learned table (tests, and an operator-driven relearn)."""
    _learned.clear()
    _learned_source.clear()


# The upstream tells us its vocabulary in the 400 it answers an unsupported level
# with. vLLM's Qwen3.x template: "Unexpected reasoning effort high. Supported types
# are xhigh (default), medium, and low." The parser is deliberately loose about
# the sentence — any "supported …" clause whose remainder names known levels —
# but requires the text to be about effort at all, so an unrelated 400 that
# happens to contain a level word never teaches anything.
_SUPPORTED_RE = re.compile(
    r"supported\s+(?:reasoning[\s_-]*)?(?:effort[\s_-]*)?(?:types?|values?|levels?|options?|efforts?)?\s*(?:are|is|:)?\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


def parse_supported_from_error(text: str) -> Tuple[str, ...]:
    """Levels named in an upstream error *text*, or ``()`` when it does not say."""
    if not text or "effort" not in text.lower():
        return ()
    match = _SUPPORTED_RE.search(text)
    if not match:
        return ()
    tokens = set(re.findall(r"[a-z]+", match.group(1).lower()))
    return tuple(level for level in EFFORT_SCALE if level in tokens)


def _error_text(body: bytes | str) -> str:
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    try:
        data = json.loads(body)
    except ValueError:
        return body
    # OpenAI/LiteLLM: {"error": {"message": …}}; FastAPI: {"detail": …}; flat {"message": …}
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(err, str):
            return err
        for key in ("message", "detail"):
            value = data.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (dict, list)):
                return json.dumps(value)
    return body


def learn_from_upstream_error(
    model: Optional[str], status_code: int, body: bytes | str
) -> Tuple[str, ...]:
    """Read an upstream 4xx for *model*; on a vocabulary statement, learn and return it.

    ``()`` means the error taught nothing (not a 400, not about effort, or no level
    names in it) — the caller relays it as before. A model with a *declared* entry
    is never overwritten (the override stays the operator's).
    """
    if status_code != 400 or not model:
        return ()
    levels = parse_supported_from_error(_error_text(body))
    if not levels:
        return ()
    if model in vocabulary():
        logger.warning(
            "upstream says %s accepts %s but SANITIZER_EFFORT_VOCABULARY declares %s; keeping the declaration",
            model, ",".join(levels), ",".join(vocabulary()[model]),
        )
        return vocabulary()[model]
    return record_levels(model, levels, "upstream-400")


# ---------------------------------------------------------------------------
# Startup probe: ask the upstream what each model takes before any client does.
# ---------------------------------------------------------------------------

ClientFactory = Callable[[], httpx.AsyncClient]

# ``minimal`` is a level every current schema (vLLM, SGLang) defines but that a
# template rarely lists, so it is the cheapest first question: a template that
# names its set answers with the whole vocabulary in one 400. Only when that is
# inconclusive is each level asked individually (1 token each).
_PROBE_LEVELS: Tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")


def probe_enabled() -> bool:
    raw = os.environ.get("SANITIZER_EFFORT_PROBE", "true").strip().lower()
    return raw not in ("false", "0", "no", "off")


def probe_headers() -> Dict[str, str]:
    """Auth for the probe's own requests: the upstream (LiteLLM) key, if configured."""
    key = (os.environ.get("SANITIZER_UPSTREAM_API_KEY") or os.environ.get("LITELLM_MASTER_KEY") or "").strip()
    headers = {"content-type": "application/json"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    return headers


def _probe_body(model: str, level: str) -> Dict[str, object]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
        "reasoning_effort": level,
        # LiteLLM must not drop the field under drop_params — the whole point is
        # to see whether the *model* takes it.
        "allowed_openai_params": ["reasoning_effort"],
    }


async def probe_model(client: httpx.AsyncClient, upstream: str, model: str, headers: Dict[str, str]) -> Tuple[str, ...]:
    """Learn *model*'s vocabulary from the upstream itself; ``()`` when inconclusive."""
    url = f"{upstream.rstrip('/')}/v1/chat/completions"
    accepted: List[str] = []
    for level in _PROBE_LEVELS:
        try:
            resp = await client.post(url, json=_probe_body(model, level), headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("effort probe for %s aborted at %s: %s", model, level, exc)
            return ()
        if resp.status_code == 400:
            named = parse_supported_from_error(_error_text(resp.content))
            if named:
                return record_levels(model, named, "probe-400")
            if "effort" in _error_text(resp.content).lower():
                continue  # this level is rejected; the rest are still open questions
            logger.warning("effort probe for %s: unrelated 400 at %s; giving up", model, level)
            return ()
        if 200 <= resp.status_code < 300:
            accepted.append(level)
            continue
        logger.warning("effort probe for %s: status %d at %s; giving up", model, resp.status_code, level)
        return ()
    # Every level answered 200 or an effort-related 400 without naming the set.
    return record_levels(model, accepted, "probe") if accepted else ()


async def probe_models(
    upstream: str,
    models: Iterable[str],
    client_factory: ClientFactory = httpx.AsyncClient,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[str, ...]]:
    """Probe every model without a declaration; returns what was learned."""
    learned: Dict[str, Tuple[str, ...]] = {}
    declared = vocabulary()
    async with client_factory() as client:
        for model in models:
            if model in declared:
                continue
            levels = await probe_model(client, upstream, model, headers or probe_headers())
            if levels:
                learned[model] = levels
    return learned


async def list_upstream_models(upstream: str, client_factory: ClientFactory = httpx.AsyncClient, headers: Optional[Dict[str, str]] = None) -> List[str]:
    async with client_factory() as client:
        resp = await client.get(f"{upstream.rstrip('/')}/v1/models", headers=headers or probe_headers())
        resp.raise_for_status()
        data = resp.json()
    rows = data.get("data") if isinstance(data, dict) else None
    out: List[str] = []
    for row in rows or []:
        model_id = row.get("id") if isinstance(row, dict) else row
        if isinstance(model_id, str) and model_id.strip():
            out.append(model_id.strip())
    return out


def enrich_models_payload(payload: object) -> object:
    """Add ``effort_levels`` to each ``/v1/models`` row the proxy knows about.

    Rows for unknown models are untouched; a payload that is not a model list is
    returned as-is. This is how the gateway in front learns the vocabulary
    without an operator copying it into a second config.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return payload
    known = known_levels()
    if not known:
        return payload
    for row in payload["data"]:
        if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"] in known:
            row["effort_levels"] = list(known[row["id"]])
    return payload


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
