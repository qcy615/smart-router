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
from smart_router.worker import WorkerType


class FakeWorker:
    def __init__(
        self,
        url: str,
        load: int = 0,
        rank: int = -1,
        worker_type: WorkerType = WorkerType.PREFILL,
        available: bool = True,
    ):
        self._url = url
        self._load = load
        self._rank = rank
        self._worker_type = worker_type
        self._available = available
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

    def decrement_load(self, load: int = 1) -> None:
        self._load = max(0, self._load - load)

    def worker_type(self) -> WorkerType:
        return self._worker_type

    def is_available(self) -> bool:
        return self._available

    def set_available(self, available: bool) -> None:
        self._available = available


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
    assert default_config.scheduler_config.schedule_response_timeout_ms == 5000
    assert default_config.scheduler_config.schedule_response_send_margin_ms == 1000
    assert default_config.scheduler_config.adaptive_interval_enabled is False

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
                "--scheduler-schedule-response-timeout-ms",
                "15000",
                "--scheduler-schedule-response-send-margin-ms",
                "1500",
                "--scheduler-adaptive-interval-enabled",
                "--scheduler-stats-window-size",
                "4",
                "--scheduler-default-forward-time-ms",
                "80",
                "--scheduler-network-latency-ms",
                "5",
                "--scheduler-min-interval-ms",
                "2",
                "--scheduler-max-interval-ms",
                "200",
                "--scheduler-watchdog-multiplier",
                "3",
            ]
        )
    )
    assert config.scheduler_config.max_batch_size == 8
    assert config.scheduler_config.batch_wait_timeout_ms == 25
    assert config.scheduler_config.schedule_response_timeout_ms == 15000
    assert config.scheduler_config.schedule_response_send_margin_ms == 1500
    assert config.scheduler_config.adaptive_interval_enabled is True
    assert config.scheduler_config.stats_window_size == 4
    assert config.scheduler_config.default_forward_time_ms == 80
    assert config.scheduler_config.network_latency_ms == 5
    assert config.scheduler_config.min_interval_ms == 2
    assert config.scheduler_config.max_interval_ms == 200
    assert config.scheduler_config.watchdog_multiplier == 3


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


def test_adaptive_interval_uses_forward_samples_and_available_worker_counts():
    async def run_test():
        engine = _make_engine(
            SchedulerConfig(
                adaptive_interval_enabled=True,
                default_forward_time_ms=100,
                network_latency_ms=10,
            )
        )
        prefill_a = FakeWorker("prefill-a", worker_type=WorkerType.PREFILL)
        prefill_b = FakeWorker("prefill-b", worker_type=WorkerType.PREFILL)
        decode = FakeWorker("decode", worker_type=WorkerType.DECODE)
        engine.worker_registry.register(prefill_a)
        engine.worker_registry.register(prefill_b)
        engine.worker_registry.register(decode)

        engine._record_forward_time(WorkerType.PREFILL, 40)
        engine._record_forward_time(WorkerType.DECODE, 120)

        assert round(engine._current_interval_secs, 3) == 0.130

        decode.set_available(False)

        assert round(engine._recompute_adaptive_interval_secs(), 3) == 0.025
        await engine.shutdown()

    asyncio.run(run_test())


def test_adaptive_get_next_batch_dispatches_first_request_immediately_when_idle():
    async def run_test():
        engine = _make_engine(
            SchedulerConfig(
                max_batch_size=2,
                adaptive_interval_enabled=True,
                default_forward_time_ms=500,
            )
        )
        engine.waiting_queue.put_nowait(_engine_request("req-1"))

        started = time.monotonic()
        batch = await engine._get_next_batch()
        elapsed = time.monotonic() - started

        assert [request.request_id for request in batch] == ["req-1"]
        assert elapsed < 0.05
        await engine.shutdown()

    asyncio.run(run_test())


def test_adaptive_get_next_batch_waits_for_interval_only():
    async def run_test():
        engine = _make_engine(
            SchedulerConfig(
                max_batch_size=2,
                adaptive_interval_enabled=True,
                default_forward_time_ms=50,
            )
        )
        worker = FakeWorker("prefill-a", load=1, worker_type=WorkerType.PREFILL)
        engine.worker_registry.register(worker)
        loop = asyncio.get_running_loop()
        engine._last_dispatch_time = loop.time()
        engine._watchdog_deadline = loop.time() + 1
        engine.waiting_queue.put_nowait(_engine_request("req-1"))
        engine.waiting_queue.put_nowait(_engine_request("req-2"))

        started = time.monotonic()
        batch = await engine._get_next_batch()
        elapsed = time.monotonic() - started

        assert [request.request_id for request in batch] == ["req-1", "req-2"]
        assert elapsed >= 0.035
        assert engine._ready_event.is_set() is False
        await engine.shutdown()

    asyncio.run(run_test())


def test_adaptive_get_next_batch_uses_watchdog_when_release_is_missing():
    async def run_test():
        engine = _make_engine(
            SchedulerConfig(
                adaptive_interval_enabled=True,
                default_forward_time_ms=20,
                watchdog_multiplier=1,
            )
        )
        worker = FakeWorker("prefill-a", load=1, worker_type=WorkerType.PREFILL)
        engine.worker_registry.register(worker)
        loop = asyncio.get_running_loop()
        engine._last_dispatch_time = loop.time()
        engine._watchdog_deadline = engine._last_dispatch_time + 0.02
        engine.waiting_queue.put_nowait(_engine_request("req-1"))

        started = time.monotonic()
        batch = await engine._get_next_batch()
        elapsed = time.monotonic() - started

        assert [request.request_id for request in batch] == ["req-1"]
        assert 0.015 <= elapsed < 0.2
        await engine.shutdown()

    asyncio.run(run_test())


def test_adaptive_get_next_batch_forces_dispatch_before_schedule_timeout():
    async def run_test():
        engine = _make_engine(
            SchedulerConfig(
                adaptive_interval_enabled=True,
                default_forward_time_ms=1000,
                schedule_response_timeout_ms=100,
                schedule_response_send_margin_ms=10,
            )
        )
        worker = FakeWorker("prefill-a", load=1, worker_type=WorkerType.PREFILL)
        engine.worker_registry.register(worker)
        loop = asyncio.get_running_loop()
        engine._last_dispatch_time = loop.time()
        engine._watchdog_deadline = engine._last_dispatch_time + 10
        request = _engine_request("req-1")
        request.enqueue_time = loop.time() - 0.2
        engine.waiting_queue.put_nowait(request)

        started = time.monotonic()
        batch = await engine._get_next_batch()
        elapsed = time.monotonic() - started

        assert [request.request_id for request in batch] == ["req-1"]
        assert elapsed < 0.05
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
