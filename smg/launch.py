"""Launch an isolated SMG HTTP router for each explicitly configured model pool.

The official SMG gateway image already includes the `smg launch` command.  This
small supervisor only translates JSON environment configuration into that CLI;
it does not proxy requests or alter the existing LiteLLM/sanitizer processes.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlsplit


_NAME = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_POLICIES = frozenset({"cache_aware", "power_of_two", "least_load", "prefix_hash", "round_robin"})
_RESERVED_PORTS = {3996, 3997, 3999}  # sanitizer, existing balancer, LiteLLM


class ConfigError(ValueError):
    """Invalid SMG_POOLS value; fail closed rather than launching a partial pool."""


@dataclass(frozen=True)
class Pool:
    name: str
    port: int
    workers: tuple[str, ...]
    policy: str = "cache_aware"


def _worker_url(raw: object) -> str:
    if not isinstance(raw, str):
        raise ConfigError("worker URLs must be strings")
    try:
        url = urlsplit(raw.strip())
        hostname = url.hostname
    except ValueError as exc:
        raise ConfigError(f"invalid worker URL {raw!r}") from exc
    if (
        url.scheme not in {"http", "https"}
        or not hostname
        or url.username is not None
        or url.password is not None
        or url.query
        or url.fragment
        or url.path not in {"", "/", "/v1", "/v1/"}
    ):
        raise ConfigError(f"invalid HTTP worker URL {raw!r}; use http(s)://host:port without credentials")
    try:
        port = url.port
    except ValueError as exc:
        raise ConfigError(f"invalid port in worker URL {raw!r}") from exc
    if port is None:
        raise ConfigError(f"worker URL must specify a port: {raw!r}")
    hostname = f"[{hostname}]" if ":" in hostname else hostname
    return f"{url.scheme}://{hostname}:{port}"


def parse_pools(raw: str) -> list[Pool]:
    """Parse per-model HTTP pools; every pool gets its own port/process."""
    if not raw.strip():
        raise ConfigError("SMG_POOLS is empty; provide at least one model pool before enabling the smg profile")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"SMG_POOLS must be valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ConfigError("SMG_POOLS must be a non-empty object of pool names to settings")

    pools: list[Pool] = []
    ports: set[int] = set()
    for name, config in data.items():
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ConfigError(f"invalid pool name {name!r}")
        if not isinstance(config, dict) or set(config) - {"port", "workers", "policy"}:
            raise ConfigError(f"pool {name}: expected only port, workers, optional policy")
        port = config.get("port")
        if type(port) is not int or not (1024 <= port <= 65535) or port in _RESERVED_PORTS or port in ports:
            raise ConfigError(f"pool {name}: port must be a unique unreserved integer from 1024 to 65535")
        ports.add(port)
        policy = config.get("policy", "cache_aware")
        if policy not in _POLICIES:
            raise ConfigError(f"pool {name}: unsupported policy {policy!r}; choose from {sorted(_POLICIES)}")
        workers = config.get("workers")
        if not isinstance(workers, list) or len(workers) < 2:
            raise ConfigError(f"pool {name}: at least two HTTP workers are required")
        urls = tuple(dict.fromkeys(_worker_url(w) for w in workers))
        if len(urls) < 2:
            raise ConfigError(f"pool {name}: at least two distinct workers are required")
        pools.append(Pool(name, port, urls, policy))
    return pools


def command(pool: Pool) -> list[str]:
    # Keep model parsing (DSML, reasoning, tool calls) inside the existing vLLM
    # HTTP workers; SMG only selects the worker.  Disable SMG's own retries to
    # avoid a second retry layer underneath LiteLLM on long/streamed requests.
    return [
        "smg", "launch", "--worker-urls", *pool.workers,
        "--policy", pool.policy, "--host", "127.0.0.1", "--port", str(pool.port),
        "--disable-retries", "--load-monitor-interval", "2",
    ]


def main() -> int:
    try:
        pools = parse_pools(os.environ.get("SMG_POOLS", ""))
    except ConfigError as exc:
        print(f"SMG configuration error: {exc}", file=sys.stderr, flush=True)
        return 2

    children: list[subprocess.Popen] = []
    stopping = False

    def stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for pool in pools:
            print(f"Starting SMG pool {pool.name} at 127.0.0.1:{pool.port} ({len(pool.workers)} workers)", flush=True)
            children.append(subprocess.Popen(command(pool)))
        while not stopping:
            for pool, child in zip(pools, children):
                code = child.poll()
                if code is not None:
                    print(f"SMG pool {pool.name} exited with status {code}; stopping sibling pools", file=sys.stderr)
                    return code if code != 0 else 1
            time.sleep(0.5)
    finally:
        stop()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
