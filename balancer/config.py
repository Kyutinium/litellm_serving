"""Environment-variable configuration for the balancer.

Like the sanitizer, free of pydantic and resolved lazily from ``os.environ`` so tests
can monkeypatch it. ``BALANCER_POOLS`` is the one required setting:

    BALANCER_POOLS='{"glm-5.1-fp8": ["http://10.0.0.1:8089", "http://10.0.0.2:8089"]}'

Each pool is served under ``/<pool>/…`` (point LiteLLM's ``api_base`` at
``http://localhost:<BALANCER_PORT>/<pool>/v1``). A malformed value refuses to start
rather than serving nothing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger("balancer.config")

_DEFAULT_PORT = 3997
_POOL_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class ConfigError(ValueError):
    """``BALANCER_POOLS`` (or a numeric knob) is malformed."""


def get_pools() -> Dict[str, List[str]]:
    raw = os.environ.get("BALANCER_POOLS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ConfigError(f"BALANCER_POOLS is not JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError("BALANCER_POOLS must be a JSON object of pool name → list of URLs")
    pools: Dict[str, List[str]] = {}
    for name, urls in parsed.items():
        if not isinstance(name, str) or not _POOL_NAME.match(name):
            raise ConfigError(f"invalid pool name {name!r} (letters, digits, . _ - only)")
        if not isinstance(urls, list) or not urls:
            raise ConfigError(f"pool {name!r} needs a non-empty list of backend URLs")
        cleaned: List[str] = []
        for url in urls:
            if not isinstance(url, str) or not re.match(r"^https?://[^\s/]+", url.strip()):
                raise ConfigError(f"pool {name!r} has an invalid backend URL {url!r}")
            u = url.strip().rstrip("/")
            # A trailing /v1 would double up with the client's own /v1 path.
            if u.endswith("/v1"):
                u = u[: -len("/v1")]
            if u not in cleaned:
                cleaned.append(u)
        pools[name] = cleaned
    return pools


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number") from exc
    if value != value or value < minimum:  # NaN or below range
        raise ConfigError(f"{name}={raw!r} must be ≥ {minimum}")
    return value


@dataclass(frozen=True)
class Tuning:
    scrape_interval: float
    scrape_timeout: float
    weight_waiting: float
    weight_running: float
    weight_kv: float
    down_seconds: float
    request_timeout: float
    connect_timeout: float


def get_tuning() -> Tuning:
    return Tuning(
        scrape_interval=_float("BALANCER_SCRAPE_INTERVAL", 1.0, minimum=0.1),
        scrape_timeout=_float("BALANCER_SCRAPE_TIMEOUT", 0.8, minimum=0.05),
        # Waiting requests are the clearest sign a replica is saturated; KV-cache use
        # (0..1) catches long-context load that request counts miss.
        weight_waiting=_float("BALANCER_WEIGHT_WAITING", 4.0),
        weight_running=_float("BALANCER_WEIGHT_RUNNING", 1.0),
        weight_kv=_float("BALANCER_WEIGHT_KV", 8.0),
        down_seconds=_float("BALANCER_DOWN_SECONDS", 10.0, minimum=0.5),
        request_timeout=_float("BALANCER_REQUEST_TIMEOUT", 600.0, minimum=1.0),
        # A replica that never answers the SYN (DROP, blackholed IP) must be given up on
        # in seconds, not after the 600 s an inference read may legitimately take.
        connect_timeout=_float("BALANCER_CONNECT_TIMEOUT", 3.0, minimum=0.1),
    )


def get_port() -> int:
    raw = os.environ.get("BALANCER_PORT", str(_DEFAULT_PORT)).strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid BALANCER_PORT=%r, falling back to %d", raw, _DEFAULT_PORT)
        return _DEFAULT_PORT
