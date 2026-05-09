import asyncio
import logging
from types import SimpleNamespace

import httpx
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from smart_router.config import HealthConfig, SmartRouterConfig
from smart_router.discovery.k8s import DiscoveredWorker
from smart_router.engine.engine import (
    Engine,
    EngineHealthResponse,
    EngineRequest,
    RequestType,
)
from smart_router.entrypoints.serve import api_server
from smart_router.worker import BasicWorker, DPAwareWorker, WorkerRegistry, WorkerType


class FakeWorkerClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def get(self, url, timeout=None):
        self.calls.append({"url": url, "timeout": timeout})
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def _config() -> SmartRouterConfig:
    return SmartRouterConfig(
        prefill_urls=[],
        decode_urls=[],
        health_config=HealthConfig(timeout_secs=1, check_interval_secs=60),
    )


def test_worker_health_check_requires_http_200(monkeypatch):
    fake_client = FakeWorkerClient(
        {
            "http://ok/health": httpx.Response(200),
            "http://bad/health": httpx.Response(500),
            "http://missing/health": httpx.Response(404),
            "http://down/health": RuntimeError("down"),
        }
    )
    monkeypatch.setattr("smart_router.worker.basic_worker.WORKER_CLIENT", fake_client)

    async def run():
        ok = BasicWorker("http://ok", WorkerType.PREFILL, _config())
        bad = BasicWorker("http://bad", WorkerType.PREFILL, _config())
        missing = BasicWorker("http://missing", WorkerType.PREFILL, _config())
        down = BasicWorker("http://down", WorkerType.PREFILL, _config())

        assert await ok.check_health_async() is True
        assert ok.is_healthy() is True

        for worker in (bad, missing, down):
            assert await worker.check_health_async() is False
            assert worker.is_healthy() is False

    asyncio.run(run())


def test_registry_groups_dp_workers_by_base_url_and_filters_health():
    registry = WorkerRegistry()
    config = _config()
    prefill_rank0 = DPAwareWorker("http://prefill", WorkerType.PREFILL, config, 0, 2)
    prefill_rank1 = DPAwareWorker("http://prefill", WorkerType.PREFILL, config, 1, 2)
    decode = BasicWorker("http://decode", WorkerType.DECODE, config)

    registry.register(prefill_rank0)
    registry.register(prefill_rank1)
    registry.register(decode)

    groups = registry.get_health_check_groups()
    group_sizes = {
        (worker_type, base_url): len(workers)
        for worker_type, base_url, workers in groups
    }

    assert group_sizes[(WorkerType.PREFILL, "http://prefill")] == 2
    assert group_sizes[(WorkerType.DECODE, "http://decode")] == 1

    registry.set_group_health(WorkerType.PREFILL, "http://prefill", False)

    assert prefill_rank0.is_healthy() is False
    assert prefill_rank1.is_healthy() is False
    assert registry.get_healthy_by_type(WorkerType.PREFILL) == []
    assert registry.get_healthy_by_type(WorkerType.DECODE) == [decode]


def test_registry_register_is_idempotent_for_duplicate_worker_ids():
    registry = WorkerRegistry()
    config = _config()
    worker_a = BasicWorker("http://prefill", WorkerType.PREFILL, config)
    worker_b = BasicWorker("http://prefill", WorkerType.PREFILL, config)

    registry.register(worker_a)
    registry.register(worker_b)

    assert registry.get_by_type(WorkerType.PREFILL) == [worker_a]
    assert registry.get_all_urls() == ["http://prefill"]


def test_engine_health_response_requires_one_healthy_prefill_and_decode(monkeypatch, caplog):
    fake_client = FakeWorkerClient(
        {
            "http://prefill-a/health": httpx.Response(500),
            "http://prefill-b/health": httpx.Response(200),
            "http://decode-a/health": httpx.Response(200),
        }
    )
    monkeypatch.setattr("smart_router.worker.basic_worker.WORKER_CLIENT", fake_client)

    async def run():
        engine = object.__new__(Engine)
        engine.worker_registry = WorkerRegistry()
        engine._health_check_lock = asyncio.Lock()
        engine.worker_registry.register(
            BasicWorker("http://prefill-a", WorkerType.PREFILL, _config())
        )
        engine.worker_registry.register(
            BasicWorker("http://prefill-b", WorkerType.PREFILL, _config())
        )
        engine.worker_registry.register(
            BasicWorker("http://decode-a", WorkerType.DECODE, _config())
        )

        caplog.set_level(logging.WARNING, logger="smart_router.engine.engine")
        response = await engine.refresh_worker_health(request_id="health-1")

        assert response.status == "ok"
        assert response.prefill_healthy == 1
        assert response.prefill_total == 2
        assert response.decode_healthy == 1
        assert response.decode_total == 1
        assert "Worker is unhealthy after health check" in caplog.text
        assert "http://prefill-a" in caplog.text

        fake_client.responses["http://decode-a/health"] = httpx.Response(503)
        response = await engine.refresh_worker_health(request_id="health-2")

        assert response.status == "unhealthy"
        assert response.decode_healthy == 0
        assert "http://decode-a" in caplog.text

    asyncio.run(run())


def test_engine_schedule_loop_returns_error_when_no_healthy_worker():
    class TestEngine(Engine):
        def __init__(self):
            self.waiting_queue = asyncio.Queue()
            self.sent = []
            self.worker_registry = WorkerRegistry()
            self._health_check_lock = asyncio.Lock()

        def schedule_prefill(self, request_text, headers):
            return None

        def schedule_decode(self, request_text, headers):
            return None

        async def send_response(self, request, msg):
            self.sent.append(msg)

    async def run():
        engine = TestEngine()
        await engine.waiting_queue.put(
            EngineRequest(
                request_id="req-1",
                identity="test",
                request_type=RequestType.SCHEDULE,
            )
        )
        task = asyncio.create_task(engine.schedule_loop())
        for _ in range(20):
            if engine.sent:
                break
            await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert engine.sent[0]["prefill_url"] is None
        assert engine.sent[0]["error"] == "No available prefill workers"

    asyncio.run(run())


def test_discovered_workers_start_unhealthy_then_health_refresh_marks_healthy(monkeypatch):
    fake_client = FakeWorkerClient(
        {
            "http://prefill/health": httpx.Response(200),
        }
    )
    monkeypatch.setattr("smart_router.worker.basic_worker.WORKER_CLIENT", fake_client)

    async def run():
        engine = object.__new__(Engine)
        engine.config = SmartRouterConfig(
            prefill_urls=[],
            decode_urls=[],
            prefill_intra_dp_size=2,
            decode_intra_dp_size=1,
            health_config=HealthConfig(timeout_secs=1, check_interval_secs=60),
            enable_k8s_discovery=True,
            prefill_port=8100,
            decode_port=8200,
        )
        engine.worker_registry = WorkerRegistry()
        engine._health_check_lock = asyncio.Lock()
        engine._health_refresh_task = None

        await engine._upsert_discovered_worker(
            DiscoveredWorker(
                pod_name="prefill-pod",
                worker_type=WorkerType.PREFILL,
                base_url="http://prefill",
            )
        )

        assert engine.worker_registry.get_healthy_by_type(WorkerType.PREFILL) == []
        await asyncio.sleep(0.2)

        healthy = engine.worker_registry.get_healthy_by_type(WorkerType.PREFILL)
        assert len(healthy) == 2
        assert fake_client.calls == [
            {"url": "http://prefill/health", "timeout": 1},
        ]

    asyncio.run(run())


def test_api_health_route_returns_engine_health_status():
    class FakeEngineClient:
        identity = "test-client"

        async def send_request(self, request):
            assert request.request_type == RequestType.HEALTH
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(
                EngineHealthResponse(
                    request_id=request.request_id,
                    status="ok",
                    prefill_healthy=1,
                    prefill_total=2,
                    decode_healthy=1,
                    decode_total=1,
                )
            )
            return future

    app = Starlette(routes=[Route("/health", api_server.health, methods=["GET"])])
    app.state.engine_client = FakeEngineClient()
    app.state.health_timeout_secs = 1

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "prefill_healthy": 1,
        "prefill_total": 2,
        "decode_healthy": 1,
        "decode_total": 1,
    }


def test_api_health_route_registered_for_vllm_and_sglang_apps():
    vllm_app = api_server._build_app(SimpleNamespace(router_type="vllm-pd-disagg"))
    sglang_app = api_server._build_app(
        SimpleNamespace(router_type="sglang-pd-disagg", prefill_bootstrap_ports=[])
    )

    assert "/health" in {route.path for route in vllm_app.routes}
    assert "/health" in {route.path for route in sglang_app.routes}
