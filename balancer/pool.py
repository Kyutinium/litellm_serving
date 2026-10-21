"""Replica selection: what each vLLM reports, plus what we sent since we last asked.

Three things LiteLLM's ``least-busy`` gets wrong are fixed here:

* **Reservation at selection.** :meth:`Pool.acquire` picks and reserves in one
  synchronous step — no ``await`` between reading the scores and bumping the pick —
  so a burst of concurrent requests spreads instead of all reading the same zeros.
* **Random tie-break.** Equal scores pick uniformly, never "the first replica".
* **Real load.** The score comes from vLLM's ``/metrics``
  (``num_requests_waiting``, ``num_requests_running``, KV-cache use), so one long
  context weighs more than one short prompt. Requests dispatched since the last
  scrape are added on top (vLLM cannot have counted them yet). A replica whose
  metrics cannot be read is scored on our own in-flight count alone.

A replica that refuses a connection is taken out for ``down_seconds``; when every
replica is out the one seen healthy most recently is still tried, so a pool never
answers "nothing to send to" while something might be up.
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("balancer.pool")

# vLLM renamed the KV metric (gpu_cache_usage_perc → kv_cache_usage_perc); read either.
_METRIC_NAMES = {
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_requests_running": "running",
    "vllm:gpu_cache_usage_perc": "kv",
    "vllm:kv_cache_usage_perc": "kv",
}
_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfINFNaN]+)")


def parse_vllm_metrics(text: str) -> Optional[Dict[str, float]]:
    """``{"waiting", "running", "kv"}`` from a Prometheus text page, or ``None``.

    A replica serving several model names (or several engines) reports one sample per
    label set — counts are summed, KV use takes the highest. ``None`` when the page
    carries none of the vLLM gauges (not a vLLM, or metrics disabled).
    """
    found: Dict[str, float] = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        key = _METRIC_NAMES.get(m.group(1))
        if key is None:
            continue
        try:
            value = float(m.group(3))
        except ValueError:
            continue
        if value != value or value in (float("inf"), float("-inf")):
            continue
        if key == "kv":
            found[key] = max(found.get(key, 0.0), value)
        else:
            found[key] = found.get(key, 0.0) + value
    if not found:
        return None
    return {"waiting": found.get("waiting", 0.0), "running": found.get("running", 0.0), "kv": found.get("kv", 0.0)}


@dataclass
class Backend:
    url: str
    in_flight: int = 0
    # Dispatched since the last successful scrape — vLLM has not counted these yet.
    since_scrape: int = 0
    waiting: float = 0.0
    running: float = 0.0
    kv: float = 0.0
    metrics_ok: bool = False
    down_until: float = 0.0
    last_ok: float = 0.0
    failures: int = 0
    picked: int = 0

    def status(self, now: float) -> dict:
        return {
            "url": self.url,
            "in_flight": self.in_flight,
            "waiting": self.waiting,
            "running": self.running,
            "kv": self.kv,
            "metrics_ok": self.metrics_ok,
            "down": self.down_until > now,
            "failures": self.failures,
            "picked": self.picked,
        }


@dataclass
class Weights:
    waiting: float = 4.0
    running: float = 1.0
    kv: float = 8.0


@dataclass
class Pool:
    name: str
    backends: List[Backend]
    weights: Weights = field(default_factory=Weights)
    down_seconds: float = 10.0
    clock: Callable[[], float] = time.monotonic
    rng: random.Random = field(default_factory=random.Random)

    @classmethod
    def from_urls(cls, name: str, urls: List[str], **kwargs) -> "Pool":
        return cls(name=name, backends=[Backend(url=u) for u in urls], **kwargs)

    def score(self, b: Backend) -> float:
        if b.metrics_ok:
            return (
                self.weights.waiting * b.waiting
                + self.weights.running * (b.running + b.since_scrape)
                + self.weights.kv * b.kv
            )
        # No metrics: our own count is all we know.
        return self.weights.running * b.in_flight

    def candidates(self, exclude: frozenset = frozenset()) -> List[Backend]:
        now = self.clock()
        pool = [b for b in self.backends if b.url not in exclude]
        up = [b for b in pool if b.down_until <= now]
        if up:
            return up
        if not pool:
            return []
        # Everything is out: still try the one that answered most recently.
        best = max(pool, key=lambda b: b.last_ok)
        logger.warning("pool %s: every replica is marked down; trying %s anyway", self.name, best.url)
        return [best]

    def acquire(self, exclude: frozenset = frozenset()) -> Optional[Backend]:
        """Pick and reserve a replica in one synchronous step (see module doc)."""
        options = self.candidates(exclude)
        if not options:
            return None
        scores = [self.score(b) for b in options]
        low = min(scores)
        ties = [b for b, s in zip(options, scores) if s - low < 1e-9]
        chosen = self.rng.choice(ties)
        chosen.in_flight += 1
        chosen.since_scrape += 1
        chosen.picked += 1
        return chosen

    def release(self, b: Backend) -> None:
        b.in_flight = max(0, b.in_flight - 1)

    def mark_down(self, b: Backend, reason: str) -> None:
        b.failures += 1
        b.down_until = self.clock() + self.down_seconds
        logger.warning("pool %s: %s out for %.0fs (%s)", self.name, b.url, self.down_seconds, reason)

    def mark_ok(self, b: Backend) -> None:
        b.last_ok = self.clock()
        if b.down_until:
            b.down_until = 0.0

    def record_metrics(self, b: Backend, metrics: Optional[Dict[str, float]]) -> None:
        """Apply one scrape. ``None`` = page reachable but no vLLM gauges."""
        if metrics is None:
            b.metrics_ok = False
        else:
            b.waiting, b.running, b.kv = metrics["waiting"], metrics["running"], metrics["kv"]
            b.metrics_ok = True
        b.since_scrape = 0
        self.mark_ok(b)

    def status(self) -> dict:
        now = self.clock()
        return {"pool": self.name, "backends": [b.status(now) for b in self.backends]}
