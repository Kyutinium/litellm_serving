"""Relay: bytes and streams pass through, reservations are released, refused
connections fail over, metrics scraping feeds the score (#31)."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from balancer import main
from balancer.config import ConfigError, get_pools


@pytest.fixture
def replicas(monkeypatch):
    """Fake vLLM replicas behind an ``httpx.MockTransport``.

    ``replicas.handlers[url] = fn(request) -> httpx.Response``; a replica listed in
    ``replicas.refuse`` raises ``ConnectError``.
    """

    class Fake:
        def __init__(self):
            self.handlers = {}
            self.refuse = set()
            self.no_answer = set()  # SYN never answered → ConnectTimeout
            self.slow = set()  # connected, then no response → ReadTimeout
            self.seen = []
            self.timeouts = []

        def _handle(self, request: httpx.Request):
            base = f"{request.url.scheme}://{request.url.host}"
            if base in self.refuse:
                raise httpx.ConnectError("refused", request=request)
            if base in self.no_answer:
                raise httpx.ConnectTimeout("no answer", request=request)
            if base in self.slow and request.url.path != "/metrics":
                self.seen.append((base, request.method, request.url.path, request.url.query.decode()))
                raise httpx.ReadTimeout("read timed out", request=request)
            self.seen.append((base, request.method, request.url.path, request.url.query.decode()))
            handler = self.handlers.get(base, lambda r: httpx.Response(200, json={"from": base}))
            return handler(request)

        def factory(self, timeout):
            self.timeouts.append(timeout)
            return httpx.AsyncClient(transport=httpx.MockTransport(self._handle))

    fake = Fake()
    monkeypatch.setattr(main, "_make_client", fake.factory)
    monkeypatch.setenv("BALANCER_POOLS", json.dumps({"glm": ["http://r0", "http://r1/v1"]}))
    monkeypatch.setenv("BALANCER_SCRAPE_INTERVAL", "3600")  # tests drive scrapes themselves
    return fake


def _client():
    return TestClient(main.app)


def test_relays_method_path_query_body_and_status(replicas):
    replicas.handlers["http://r0"] = replicas.handlers["http://r1"] = lambda r: httpx.Response(
        201, json={"echo": json.loads(r.content)}, headers={"x-upstream": "yes"}
    )
    with _client() as c:
        res = c.post("/glm/v1/chat/completions?x=1", json={"model": "glm", "messages": []})
    assert res.status_code == 201
    assert res.json() == {"echo": {"model": "glm", "messages": []}}
    assert res.headers["x-upstream"] == "yes"
    base, method, path, query = replicas.seen[-1]
    assert (method, path, query) == ("POST", "/v1/chat/completions", "x=1"), "a trailing /v1 in the pool URL is not doubled"
    assert all(b.in_flight == 0 for b in main.POOLS["glm"].backends), "the reservation is released"


def test_streams_pass_through_and_release_after_the_stream(replicas):
    chunks = [b"data: {\"a\":1}\n\n", b"data: [DONE]\n\n"]
    replicas.handlers["http://r0"] = replicas.handlers["http://r1"] = lambda r: httpx.Response(
        200, content=b"".join(chunks), headers={"content-type": "text/event-stream"}
    )
    with _client() as c:
        with c.stream("POST", "/glm/v1/chat/completions", json={"stream": True}) as res:
            assert res.headers["content-type"].startswith("text/event-stream")
            assert b"".join(res.iter_raw()) == b"".join(chunks)
    assert all(b.in_flight == 0 for b in main.POOLS["glm"].backends)


def test_a_refused_replica_fails_over_and_is_marked_down(replicas):
    replicas.refuse.add("http://r0")
    with _client() as c:
        for _ in range(4):
            assert c.get("/glm/v1/models").json() == {"from": "http://r1"}
    r0 = main.POOLS["glm"].backends[0]
    assert r0.failures >= 1 and r0.in_flight == 0


def test_every_replica_refusing_is_a_503(replicas):
    replicas.refuse.update({"http://r0", "http://r1"})
    with _client() as c:
        res = c.get("/glm/v1/models")
    assert res.status_code == 503
    assert all(b.in_flight == 0 for b in main.POOLS["glm"].backends)


def test_an_unknown_pool_is_a_404(replicas):
    with _client() as c:
        assert c.get("/nope/v1/models").status_code == 404


def test_scrape_feeds_the_score_and_status_shows_it(replicas):
    pages = {
        "http://r0": "vllm:num_requests_waiting 5\nvllm:num_requests_running 8\nvllm:kv_cache_usage_perc 0.9\n",
        "http://r1": "vllm:num_requests_waiting 0\nvllm:num_requests_running 1\nvllm:kv_cache_usage_perc 0.1\n",
    }

    def handler(r):
        base = f"{r.url.scheme}://{r.url.host}"
        if r.url.path == "/metrics":
            return httpx.Response(200, text=pages[base])
        return httpx.Response(200, json={"from": base})

    replicas.handlers["http://r0"] = replicas.handlers["http://r1"] = handler
    with _client() as c:
        asyncio.run(main.scrape_once(main.POOLS, main._make_client(1)))
        froms = [c.get("/glm/v1/models").json()["from"] for _ in range(6)]
        status = c.get("/balancer/status").json()
    assert froms == ["http://r1"] * 6, "the replica with a queue and full KV cache is avoided"
    r0 = status["pools"][0]["backends"][0]
    assert r0["waiting"] == 5 and r0["metrics_ok"] is True


def test_a_refused_metrics_scrape_marks_the_replica_down(replicas):
    replicas.refuse.add("http://r0")
    with _client():
        asyncio.run(main.scrape_once(main.POOLS, main._make_client(1)))
    r0 = main.POOLS["glm"].backends[0]
    assert r0.failures >= 1 and r0.metrics_ok is False


@pytest.mark.parametrize(
    "raw",
    ["not json", "[]", '{"a": []}', '{"a": ["ftp://x"]}', '{"bad name!": ["http://x"]}'],
)
def test_malformed_pools_refuse(monkeypatch, raw):
    monkeypatch.setenv("BALANCER_POOLS", raw)
    with pytest.raises(ConfigError):
        get_pools()


def test_a_client_that_disconnects_mid_stream_releases_its_reservation():
    """The reservation must not leak when the caller goes away before the stream ends —
    a leaked count would make that replica look busy forever."""
    from balancer.pool import Pool

    class Slow(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: 1\n\n"
            await asyncio.sleep(3600)  # never finishes on its own
            yield b"data: 2\n\n"

    async def run():
        pool = Pool.from_urls("m", ["http://r0"])
        backend = pool.acquire()
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Slow())))
        res = await client.send(client.build_request("GET", "http://r0/x"), stream=True)
        streaming = main._stream_back(pool, backend, client, res)
        body = streaming.body_iterator
        assert await body.__anext__() == b"data: 1\n\n"
        assert backend.in_flight == 1
        await body.aclose()  # what Starlette does when the client disconnects
        return backend.in_flight

    assert asyncio.run(run()) == 0


def test_a_replica_that_never_answers_the_connect_fails_over_and_is_marked_down(replicas):
    """A DROP firewall or blackholed IP ends in ConnectTimeout, not ConnectError — it must
    fail over like a refusal (review P1), not fail the request after the read budget."""
    replicas.no_answer.add("http://r0")
    with _client() as c:
        for _ in range(4):
            res = c.get("/glm/v1/models")
            assert res.status_code == 200 and res.json() == {"from": "http://r1"}
    r0, r1 = main.POOLS["glm"].backends
    assert r0.failures >= 1 and r0.down_until > 0
    assert r0.in_flight == 0 and r1.in_flight == 0


def test_every_replica_timing_out_on_connect_is_a_503(replicas):
    replicas.no_answer.update({"http://r0", "http://r1"})
    with _client() as c:
        res = c.get("/glm/v1/models")
    assert res.status_code == 503
    assert all(b.in_flight == 0 for b in main.POOLS["glm"].backends)


def test_connect_is_bounded_short_while_reads_keep_the_long_budget(replicas, monkeypatch):
    monkeypatch.setenv("BALANCER_CONNECT_TIMEOUT", "2.5")
    monkeypatch.setenv("BALANCER_REQUEST_TIMEOUT", "600")
    with _client() as c:
        c.get("/glm/v1/models")
    relay_timeout = replicas.timeouts[-1]
    assert isinstance(relay_timeout, httpx.Timeout)
    assert relay_timeout.connect == 2.5 and relay_timeout.read == 600


def test_a_read_timeout_is_not_retried_on_another_replica(replicas):
    """Once connected the request may already be running there — resending it would run
    it twice. The replica stays in the pool; LiteLLM's retry decides."""
    replicas.slow.update({"http://r0", "http://r1"})
    with _client() as c:
        with pytest.raises(httpx.ReadTimeout):
            c.get("/glm/v1/models")
    hits = [s for s in replicas.seen if s[2] != "/metrics"]
    assert len(hits) == 1, "sent to exactly one replica"
    assert all(b.failures == 0 and b.in_flight == 0 for b in main.POOLS["glm"].backends)


def test_unroutable_replicas_give_up_within_the_connect_timeout(monkeypatch):
    """Real sockets: TEST-NET-1 addresses (RFC 5737) are never routed, so the connect
    either times out or is refused at once — either way within the connect budget."""
    import time

    monkeypatch.setattr(main, "_make_client", lambda timeout: httpx.AsyncClient(timeout=timeout, trust_env=False))
    monkeypatch.setenv("BALANCER_POOLS", json.dumps({"dead": ["http://192.0.2.1:9", "http://192.0.2.2:9"]}))
    monkeypatch.setenv("BALANCER_SCRAPE_INTERVAL", "3600")
    monkeypatch.setenv("BALANCER_CONNECT_TIMEOUT", "0.3")
    with _client() as c:
        started = time.monotonic()
        res = c.get("/dead/v1/models")
        elapsed = time.monotonic() - started
    assert res.status_code == 503
    assert elapsed < 3, f"gave up after {elapsed:.1f}s"
