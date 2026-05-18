import asyncio

from smart_router.config import SmartRouterConfig
from smart_router.engine.engine import Engine, EngineRequest, RequestType
from smart_router.engine.normal_engine import NormalEngine
from smart_router.worker import BasicWorker, WorkerRegistry, WorkerType


def test_normal_engine_allows_missing_worker_urls(monkeypatch):
    monkeypatch.setattr(Engine, "__init__", lambda self, **kwargs: None)

    engine = NormalEngine(
        SmartRouterConfig(worker_urls=None),
        input_socket_address="tcp://127.0.0.1:5557",
        output_socket_address="tcp://127.0.0.1:5558",
    )

    assert engine.worker_registry.get_all() == []


def test_normal_engine_schedule_loop_returns_error_when_no_healthy_worker():
    class TestEngine(NormalEngine):
        def __init__(self):
            self.waiting_queue = asyncio.Queue()
            self.sent = []
            self.config = SmartRouterConfig(worker_urls=[])
            self.worker_registry = WorkerRegistry()
            self.policy = None

        def schedule_worker(self, request_text, headers):
            return None

        async def send_response(self, request, msg):
            self.sent.append(msg)

    async def wait_until_sent(engine):
        while not engine.sent:
            await asyncio.sleep(0)

    async def run():
        engine = TestEngine()
        await engine.waiting_queue.put(
            EngineRequest(
                request_id="normal-req-1",
                identity="test",
                request_type=RequestType.SCHEDULE,
            )
        )
        task = asyncio.create_task(engine.schedule_loop())
        await asyncio.wait_for(wait_until_sent(engine), timeout=1)
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=1)
        except asyncio.CancelledError:
            pass

        assert engine.sent[0]["worker_url"] == ""
        assert engine.sent[0]["worker_rank"] == -1
        assert engine.sent[0]["error"] == "No available workers"

    asyncio.run(run())


def test_normal_engine_schedule_worker_filters_unhealthy_regular_workers():
    class TestPolicy:
        def __init__(self):
            self.workers = None

        def select_worker(self, workers, request_text=None, headers=None):
            self.workers = workers
            return workers[0] if workers else None

    engine = object.__new__(NormalEngine)
    engine.worker_registry = WorkerRegistry()
    engine.policy = TestPolicy()
    healthy_worker = BasicWorker(
        "http://healthy", WorkerType.REGULAR, SmartRouterConfig()
    )
    unhealthy_worker = BasicWorker(
        "http://unhealthy", WorkerType.REGULAR, SmartRouterConfig()
    )
    unhealthy_worker.set_healthy(False)
    engine.worker_registry.register(healthy_worker)
    engine.worker_registry.register(unhealthy_worker)

    selected = engine.schedule_worker("hello", {})

    assert selected == healthy_worker
    assert engine.policy.workers == [healthy_worker]
