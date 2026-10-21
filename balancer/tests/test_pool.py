"""Selection: reservation at pick time, random ties, real load, down/recovery (#31)."""

import random

from balancer.pool import Backend, Pool, Weights, parse_vllm_metrics


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _pool(n=3, seed=0, clock=None):
    return Pool.from_urls(
        "m",
        [f"http://r{i}" for i in range(n)],
        weights=Weights(),
        down_seconds=10,
        clock=clock or Clock(),
        rng=random.Random(seed),
    )


def test_a_burst_spreads_evenly_instead_of_herding():
    """LiteLLM least-busy sent every concurrent request to the first replica: here each
    pick reserves before the next one reads the scores."""
    pool = _pool(3)
    picks = [pool.acquire().url for _ in range(30)]
    assert {u: picks.count(u) for u in set(picks)} == {"http://r0": 10, "http://r1": 10, "http://r2": 10}


def test_ties_are_broken_at_random_not_first_replica():
    firsts = {_pool(4, seed=s).acquire().url for s in range(40)}
    assert len(firsts) > 1, "an idle pool must not always pick the first replica"


def test_the_queue_and_kv_cache_count_not_just_request_counts():
    pool = _pool(2)
    a, b = pool.backends
    # Same running count, but r0 has a queue and a full KV cache (long contexts).
    pool.record_metrics(a, {"waiting": 3, "running": 2, "kv": 0.95})
    pool.record_metrics(b, {"waiting": 0, "running": 2, "kv": 0.10})
    picks = [pool.acquire().url for _ in range(5)]
    assert picks.count("http://r1") == 5


def test_dispatches_since_the_last_scrape_count_until_vllm_reports_them():
    pool = _pool(2)
    a, b = pool.backends
    pool.record_metrics(a, {"waiting": 0, "running": 0, "kv": 0})
    pool.record_metrics(b, {"waiting": 0, "running": 0, "kv": 0})
    first = pool.acquire()
    second = pool.acquire()
    assert first is not second
    pool.record_metrics(first, {"waiting": 0, "running": 1, "kv": 0})
    assert first.since_scrape == 0


def test_without_metrics_our_own_in_flight_count_decides():
    pool = _pool(2)
    a, b = pool.backends
    for _ in range(3):
        a.in_flight += 1
    assert pool.acquire() is b
    pool.release(b)
    assert b.in_flight == 0
    pool.release(b)
    assert b.in_flight == 0, "release never goes negative"


def test_a_down_replica_is_skipped_then_comes_back():
    clock = Clock()
    pool = _pool(2, clock=clock)
    a, b = pool.backends
    pool.mark_down(a, "connect refused")
    assert all(pool.acquire() is b for _ in range(5))
    clock.t += 11
    for x in pool.backends:
        x.in_flight = 0
        x.since_scrape = 0
    assert {pool.acquire().url for _ in range(4)} == {"http://r0", "http://r1"}


def test_when_every_replica_is_down_the_most_recent_healthy_one_is_tried():
    clock = Clock()
    pool = _pool(2, clock=clock)
    a, b = pool.backends
    pool.mark_ok(a)
    clock.t += 1
    pool.mark_ok(b)
    pool.mark_down(a, "x")
    pool.mark_down(b, "y")
    assert pool.acquire() is b


def test_exclude_skips_replicas_already_tried():
    pool = _pool(2)
    a, b = pool.backends
    assert pool.acquire(frozenset({a.url})) is b
    assert pool.acquire(frozenset({a.url, b.url})) is None


def test_parse_vllm_metrics_sums_counts_and_reads_either_kv_name():
    page = """# HELP vllm:num_requests_running Number of requests running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="a"} 3.0
vllm:num_requests_running{engine="1",model_name="a"} 2.0
vllm:num_requests_waiting{model_name="a"} 4.0
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:kv_cache_usage_perc{engine="1"} 0.75
vllm:num_requests_swapped{model_name="a"} 9.0
"""
    assert parse_vllm_metrics(page) == {"waiting": 4.0, "running": 5.0, "kv": 0.75}
    assert parse_vllm_metrics("vllm:gpu_cache_usage_perc 0.5\n") == {"waiting": 0.0, "running": 0.0, "kv": 0.5}
    assert parse_vllm_metrics("process_cpu_seconds_total 1\n") is None
    assert parse_vllm_metrics("vllm:num_requests_running NaN\n") is None
