import asyncio
import logging
import httpx
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from smart_router.config import HealthConfig, SmartRouterConfig
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


def test_engine_health_response_uses_regular_workers_for_normal_mode(monkeypatch):
    fake_client = FakeWorkerClient(
        {
            "http://regular-a/health": httpx.Response(200),
            "http://regular-b/health": httpx.Response(503),
        }
    )
    monkeypatch.setattr("smart_router.worker.basic_worker.WORKER_CLIENT", fake_client)

    async def run():
        engine = object.__new__(Engine)
        engine.worker_registry = WorkerRegistry()
        engine._health_check_lock = asyncio.Lock()
        engine.worker_registry.register(
            BasicWorker("http://regular-a", WorkerType.REGULAR, _config())
        )
        engine.worker_registry.register(
            BasicWorker("http://regular-b", WorkerType.REGULAR, _config())
        )

        response = await engine.refresh_worker_health(request_id="health-regular-1")

        assert response.status == "ok"
        assert response.regular_healthy == 1
        assert response.regular_total == 2
        assert response.prefill_total == 0
        assert response.decode_total == 0

        fake_client.responses["http://regular-a/health"] = httpx.Response(503)
        response = await engine.refresh_worker_health(request_id="health-regular-2")

        assert response.status == "unhealthy"
        assert response.regular_healthy == 0
        assert response.regular_total == 2

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


def test_engine_health_request_does_not_block_receive_loop():
    class FakeInputSocket:
        def __init__(self, messages):
            self.messages = asyncio.Queue()
            for message in messages:
                self.messages.put_nowait(message)

        async def recv_json(self):
            return await self.messages.get()

    class TestEngine(Engine):
        def __init__(self):
            self.input_socket = FakeInputSocket(
                [
                    EngineRequest(
                        request_id="health-1",
                        identity="test",
                        request_type=RequestType.HEALTH,
                    ).to_dict(),
                    EngineRequest(
                        request_id="schedule-1",
                        identity="test",
                        request_type=RequestType.SCHEDULE,
                    ).to_dict(),
                ]
            )
            self.waiting_queue = asyncio.Queue()
            self.sent = []
            self.worker_registry = WorkerRegistry()
            self._health_check_lock = asyncio.Lock()
            self._background_tasks = set()
            self.health_started = asyncio.Event()
            self.finish_health = asyncio.Event()

        async def refresh_worker_health(self, request_id=""):
            self.health_started.set()
            await self.finish_health.wait()
            return EngineHealthResponse(
                request_id=request_id,
                status="unhealthy",
                prefill_healthy=0,
                prefill_total=0,
                decode_healthy=0,
                decode_total=0,
            )

        async def send_response(self, request, msg):
            self.sent.append(msg)

    async def run():
        engine = TestEngine()
        task = asyncio.create_task(engine.receive_loop())

        await asyncio.wait_for(engine.health_started.wait(), timeout=1)
        for _ in range(20):
            if engine.waiting_queue.qsize() == 1:
                break
            await asyncio.sleep(0)

        assert engine.waiting_queue.qsize() == 1
        queued = await engine.waiting_queue.get()
        assert queued.request_id == "schedule-1"
        assert engine.sent == []

        engine.finish_health.set()
        for _ in range(20):
            if engine.sent:
                break
            await asyncio.sleep(0)

        assert engine.sent[0]["request_id"] == "health-1"

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run())


def test_engine_shutdown_stops_policies():
    class FakeSocket:
        def __init__(self):
            self.closed = False

        def close(self, linger=0):
            self.closed = True
            self.linger = linger

    class FakePolicy:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    async def run():
        engine = object.__new__(Engine)
        engine.input_socket = FakeSocket()
        engine.output_socket = FakeSocket()
        engine.worker_discovery = None
        engine.prefill_policy = FakePolicy()
        engine.decode_policy = FakePolicy()

        await engine.shutdown()

        assert engine.input_socket.closed is True
        assert engine.output_socket.closed is True
        assert engine.prefill_policy.stopped is True
        assert engine.decode_policy.stopped is True

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


def test_api_health_route_includes_regular_counts_for_normal_mode():
    class FakeEngineClient:
        identity = "test-client"

        async def send_request(self, request):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(
                EngineHealthResponse(
                    request_id=request.request_id,
                    status="ok",
                    prefill_healthy=0,
                    prefill_total=0,
                    decode_healthy=0,
                    decode_total=0,
                    regular_healthy=1,
                    regular_total=2,
                )
            )
            return future

    app = Starlette(routes=[Route("/health", api_server.health, methods=["GET"])])
    app.state.engine_client = FakeEngineClient()
    app.state.health_timeout_secs = 1

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["regular_healthy"] == 1
    assert response.json()["regular_total"] == 2


def test_api_health_route_registered_for_vllm_and_sglang_apps():
    vllm_app = api_server._build_app(
        SmartRouterConfig(router_type="vllm", pd_disaggregation=True)
    )
    sglang_app = api_server._build_app(
        SmartRouterConfig(router_type="sglang", pd_disaggregation=True)
    )

    assert "/health" in {route.path for route in vllm_app.routes}
    assert "/health" in {route.path for route in sglang_app.routes}
