import asyncio
import time
import uuid

from smart_router.config import PolicyConfig, SchedulerConfig
from smart_router.config.smart_router import build_config
from smart_router.config.utils import build_parser
from smart_router.engine.engine import Engine, EngineRequest, RequestType
from smart_router.policies.policy import PolicyRequest
from smart_router.policies.prefix_aware import PrefixAwarePolicy
from smart_router.policies.prefix_tree import PrefixTree


class FakeWorker:
    def __init__(self, url: str, load: int = 0, rank: int = -1):
        self._url = url
        self._load = load
        self._rank = rank
        self.increment_calls = []

    def url(self) -> str:
        return self._url

    def base_url(self) -> str:
        return self._url

    def dp_rank(self) -> int:
        return self._rank

    def load(self) -> int:
        return self._load

    def increment_load(self, load: int = 1) -> None:
        self.increment_calls.append(load)
        self._load += load


class CountingPrefixTree(PrefixTree):
    def __init__(self):
        super().__init__()
        self.prefix_match_calls = []

    def prefix_match(self, text, available_tenants=None):
        self.prefix_match_calls.append(text)
        return super().prefix_match(text, available_tenants)


class RecordingEngine(Engine):
    def __init__(self, prefill_workers, decode_workers):
        suffix = uuid.uuid4().hex
        super().__init__(
            f"inproc://input-{suffix}",
            f"inproc://output-{suffix}",
            scheduler_config=SchedulerConfig(max_batch_size=4),
        )
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers
        self.responses = []

    def schedule_prefill_batch(self, requests):
        return self.prefill_workers[: len(requests)]

    def schedule_decode_batch(self, requests):
        return self.decode_workers[: len(requests)]

    async def send_response(self, request, msg):
        self.responses.append((request.request_id, msg))


def _engine_request(request_id: str, request_text: str = "hello") -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        identity="test-client",
        request_type=RequestType.SCHEDULE,
        request_text=request_text,
        headers={},
    )


def _make_engine(config: SchedulerConfig | None = None) -> Engine:
    suffix = uuid.uuid4().hex
    return Engine(
        f"inproc://input-{suffix}",
        f"inproc://output-{suffix}",
        scheduler_config=config,
    )


def test_scheduler_config_defaults_and_cli_overrides():
    parser = build_parser()

    default_config = build_config(
        parser.parse_args(
            [
                "--prefill-urls",
                "http://prefill",
                "--decode-urls",
                "http://decode",
            ]
        )
    )
    assert default_config.scheduler_config.max_batch_size == 1
    assert default_config.scheduler_config.batch_wait_timeout_ms == 0

    config = build_config(
        parser.parse_args(
            [
                "--prefill-urls",
                "http://prefill",
                "--decode-urls",
                "http://decode",
                "--scheduler-max-batch-size",
                "8",
                "--scheduler-batch-wait-timeout-ms",
                "25",
            ]
        )
    )
    assert config.scheduler_config.max_batch_size == 8
    assert config.scheduler_config.batch_wait_timeout_ms == 25


def test_prefix_tree_insert_many_matches_sequential_insert_and_ignores_unknown_tenant():
    items = [
        ("hello world", "worker-a", 1.0),
        ("hello there", "worker-b", 2.0),
        ("", "worker-a", 3.0),
        ("goodbye", "worker-a", 4.0),
        ("ignored", "missing-worker", 5.0),
    ]
    available_tenants = ["worker-a", "worker-b", "missing-worker"]
    texts = ["hello world", "hello thomas", "goodbye friend", "", "unknown"]

    sequential = PrefixTree()
    sequential.add_tenants(["worker-a", "worker-b"], time_s=0.0)
    for item in items:
        sequential.insert(*item)

    batched = PrefixTree()
    batched.add_tenants(["worker-a", "worker-b"], time_s=0.0)
    batched.insert_many(items)

    assert batched.tenant_to_char_count == sequential.tenant_to_char_count
    assert "missing-worker" not in batched.tenant_to_char_count
    for text in texts:
        assert batched.prefix_match(text, available_tenants) == sequential.prefix_match(
            text,
            available_tenants,
        )


def test_get_next_batch_defaults_to_single_request():
    async def run_test():
        engine = _make_engine()
        engine.waiting_queue.put_nowait(_engine_request("req-1"))
        engine.waiting_queue.put_nowait(_engine_request("req-2"))

        batch = await engine._get_next_batch()

        assert [request.request_id for request in batch] == ["req-1"]
        assert engine.waiting_queue.qsize() == 1
        await engine.shutdown()

    asyncio.run(run_test())


def test_get_next_batch_drains_ready_requests_until_max_batch_size():
    async def run_test():
        engine = _make_engine(
            SchedulerConfig(max_batch_size=2, batch_wait_timeout_ms=1000)
        )
        engine.waiting_queue.put_nowait(_engine_request("req-1"))
        engine.waiting_queue.put_nowait(_engine_request("req-2"))
        engine.waiting_queue.put_nowait(_engine_request("req-3"))

        batch = await engine._get_next_batch()

        assert [request.request_id for request in batch] == ["req-1", "req-2"]
        assert engine.waiting_queue.qsize() == 1
        await engine.shutdown()

    asyncio.run(run_test())


def test_get_next_batch_waits_within_fixed_time_window_for_more_requests():
    async def run_test():
        engine = _make_engine(
            SchedulerConfig(max_batch_size=3, batch_wait_timeout_ms=100)
        )
        engine.waiting_queue.put_nowait(_engine_request("req-1"))

        async def add_later():
            await asyncio.sleep(0.01)
            engine.waiting_queue.put_nowait(_engine_request("req-2"))

        task = asyncio.create_task(add_later())
        started = time.monotonic()
        batch = await engine._get_next_batch()
        elapsed = time.monotonic() - started

        assert [request.request_id for request in batch] == ["req-1", "req-2"]
        assert elapsed < 0.5
        await task
        await engine.shutdown()

    asyncio.run(run_test())


def test_schedule_batch_sends_one_response_per_request_and_commits_loads():
    async def run_test():
        prefill_a = FakeWorker("http://prefill-a", rank=0)
        prefill_b = FakeWorker("http://prefill-b", rank=1)
        decode = FakeWorker("http://decode", rank=2)
        engine = RecordingEngine(
            prefill_workers=[prefill_a, prefill_b],
            decode_workers=[decode, decode],
        )
        requests = [_engine_request("req-1"), _engine_request("req-2")]

        await engine._schedule_batch(requests)

        assert [request_id for request_id, _ in engine.responses] == [
            "req-1",
            "req-2",
        ]
        assert engine.responses[0][1]["prefill_url"] == "http://prefill-a"
        assert engine.responses[0][1]["decode_url"] == "http://decode"
        assert engine.responses[1][1]["prefill_url"] == "http://prefill-b"
        assert engine.responses[1][1]["decode_rank"] == 2
        assert prefill_a.increment_calls == [1]
        assert prefill_b.increment_calls == [1]
        assert decode.increment_calls == [2]
        await engine.shutdown()

    asyncio.run(run_test())


def test_schedule_batch_requires_prefill_and_decode_before_committing_load():
    async def run_test():
        prefill = FakeWorker("http://prefill", rank=0)
        engine = RecordingEngine(
            prefill_workers=[prefill],
            decode_workers=[None],
        )
        requests = [_engine_request("req-1")]

        await engine._schedule_batch(requests)

        assert [request_id for request_id, _ in engine.responses] == ["req-1"]
        response = engine.responses[0][1]
        assert response["prefill_url"] is None
        assert response["prefill_rank"] == -1
        assert response["decode_url"] is None
        assert response["decode_rank"] == -1
        assert prefill.increment_calls == []
        await engine.shutdown()

    asyncio.run(run_test())


def test_prefix_aware_batch_prefers_existing_prefix_cache_hit():
    policy = PrefixAwarePolicy(
        PolicyConfig(
            policy="prefix_aware",
            cache_threshold=0.5,
            balance_abs_threshold=100,
        )
    )
    worker_a = FakeWorker("worker-a", load=10)
    worker_b = FakeWorker("worker-b", load=0)
    policy._insert_tree("hello world", worker_a.url())

    selected = policy.select_worker_batch(
        [worker_a, worker_b],
        [PolicyRequest(request_text="hello there")],
    )

    assert selected == [worker_a]


def test_prefix_aware_batch_uses_temporary_affinity_within_batch():
    policy = PrefixAwarePolicy(
        PolicyConfig(
            policy="prefix_aware",
            cache_threshold=0.5,
            balance_abs_threshold=100,
        )
    )
    worker_a = FakeWorker("worker-a", load=0)
    worker_b = FakeWorker("worker-b", load=0)

    selected = policy.select_worker_batch(
        [worker_a, worker_b],
        [
            PolicyRequest(request_text="shared prefix alpha"),
            PolicyRequest(request_text="shared prefix beta"),
        ],
    )

    assert selected == [worker_a, worker_a]
    assert policy.tree.prefix_match("shared prefix gamma", [worker_a.url()])[
        1
    ] == [worker_a.url()]


def test_prefix_aware_batch_groups_related_requests_before_tree_match():
    policy = PrefixAwarePolicy(
        PolicyConfig(
            policy="prefix_aware",
            cache_threshold=0.5,
            balance_abs_threshold=100,
        )
    )
    policy.tree = CountingPrefixTree()
    worker_a = FakeWorker("worker-a", load=0)
    worker_b = FakeWorker("worker-b", load=0)

    selected = policy.select_worker_batch(
        [worker_a, worker_b],
        [
            PolicyRequest(request_text="shared prefix alpha"),
            PolicyRequest(request_text="unrelated request"),
            PolicyRequest(request_text="shared prefix beta"),
        ],
    )

    assert selected == [worker_a, worker_b, worker_a]
    assert policy.tree.prefix_match_calls == [
        "shared prefix ",
        "unrelated request",
    ]


def test_prefix_aware_batch_uses_average_group_match_rate_for_cache_hit():
    policy = PrefixAwarePolicy(
        PolicyConfig(
            policy="prefix_aware",
            cache_threshold=0.5,
            balance_abs_threshold=100,
        )
    )
    worker_a = FakeWorker("worker-a", load=10)
    worker_b = FakeWorker("worker-b", load=0)
    policy._insert_tree("abcdefgh", worker_a.url())

    selected = policy.select_worker_batch(
        [worker_a, worker_b],
        [
            PolicyRequest(request_text="abcdefghij"),
            PolicyRequest(request_text="abcdefghij12345678"),
        ],
    )

    assert selected == [worker_a, worker_a]


def test_prefix_aware_batch_prefers_min_effective_load_when_imbalanced():
    policy = PrefixAwarePolicy(
        PolicyConfig(
            policy="prefix_aware",
            cache_threshold=0.5,
            balance_abs_threshold=1,
            balance_rel_threshold=1.1,
        )
    )
    worker_a = FakeWorker("worker-a", load=10)
    worker_b = FakeWorker("worker-b", load=0)
    policy._insert_tree("hello world", worker_a.url())

    selected = policy.select_worker_batch(
        [worker_a, worker_b],
        [PolicyRequest(request_text="hello there")],
    )

    assert selected == [worker_b]
