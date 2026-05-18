import asyncio
from types import SimpleNamespace

from smart_router.config import K8SDiscoveryConfig, SmartRouterConfig
from smart_router.discovery import K8SPodDiscovery
from smart_router.engine.engine import Engine
from smart_router.worker import BasicWorker, WorkerRegistry, WorkerType


def _env(values):
    return [SimpleNamespace(name=name, value=value) for name, value in values.items()]


def _pod(
    name,
    uid,
    *,
    phase="Running",
    pod_ip="10.0.0.1",
    labels=None,
    env=None,
    deletion_timestamp=None,
    resource_version="1",
):
    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            uid=uid,
            labels=labels or {"task_id": "task-a"},
            deletion_timestamp=deletion_timestamp,
            resource_version=resource_version,
        ),
        spec=SimpleNamespace(
            containers=[
                SimpleNamespace(env=_env(env or {})),
            ],
        ),
        status=SimpleNamespace(phase=phase, pod_ip=pod_ip),
    )


class FakeCoreV1:
    def __init__(self, pods, router_pod=None):
        self.pods = pods
        self.router_pod = router_pod or _pod("router", "router-uid")
        self.list_calls = []

    def list_namespaced_pod(self, **kwargs):
        self.list_calls.append(kwargs)
        return SimpleNamespace(
            metadata=SimpleNamespace(resource_version="10"),
            items=list(self.pods),
        )

    def read_namespaced_pod(self, name, namespace):
        assert name == self.router_pod.metadata.name
        assert namespace == "default"
        return self.router_pod


class FakeWatch:
    def __init__(self, events):
        self.events = events
        self.kwargs = None

    def stream(self, *args, **kwargs):
        self.kwargs = kwargs
        yield from self.events


def _config(**overrides):
    values = {
        "enabled": True,
        "prefill_port": 8100,
        "decode_port": 8200,
        "regular_port": 8300,
        "task_id": "task-a",
    }
    values.update(overrides)
    discovery = K8SDiscoveryConfig(**values)
    return SmartRouterConfig(
        prefill_intra_dp_size=2,
        decode_intra_dp_size=1,
        k8s_discovery_config=discovery,
    )


def test_k8s_discovery_sync_registers_only_serving_worker_pods():
    registry = WorkerRegistry()
    core = FakeCoreV1(
        [
            _pod("router", "router-uid", env={"WORKERTYPE": "PREFILL"}),
            _pod("prefill", "prefill-uid", pod_ip="10.0.0.2", env={"WORKERTYPE": "PREFILL"}),
            _pod("decode", "decode-uid", pod_ip="fd00::1", env={"WORKERTYPE": "decode"}),
            _pod("regular", "regular-uid", pod_ip="10.0.0.3", env={"WORKERTYPE": "regular"}),
            _pod("headless", "headless-uid", env={"WORKERTYPE": "PREFILL", "HEADLESS": "true"}),
            _pod("pending", "pending-uid", phase="Pending", env={"WORKERTYPE": "DECODE"}),
            _pod("invalid", "invalid-uid", env={"WORKERTYPE": "OTHER"}),
        ]
    )
    discovery = K8SPodDiscovery(
        _config(),
        registry,
        core_v1=core,
        watch_factory=lambda: FakeWatch([]),
        env={"HOSTNAME": "router"},
    )

    discovery.sync_once_blocking()

    assert registry.get_all_urls() == [
        "http://10.0.0.2:8100@0",
        "http://10.0.0.2:8100@1",
        "http://[fd00::1]:8200",
        "http://10.0.0.3:8300",
    ]
    assert registry.get_by_type(WorkerType.REGULAR)[0].base_url() == "http://10.0.0.3:8300"
    assert all(not worker.is_healthy() for worker in registry.get_all())
    assert core.list_calls[0]["label_selector"] == "task_id=task-a"


def test_k8s_discovery_reads_task_id_from_router_pod_label():
    registry = WorkerRegistry()
    core = FakeCoreV1(
        [_pod("prefill", "prefill-uid", env={"WORKERTYPE": "PREFILL"})],
        router_pod=_pod(
            "router",
            "router-uid",
            labels={"custom_task": "from-router"},
        ),
    )
    discovery = K8SPodDiscovery(
        _config(task_id=None, task_label_key="custom_task"),
        registry,
        core_v1=core,
        watch_factory=lambda: FakeWatch([]),
        env={"HOSTNAME": "router"},
    )

    discovery.sync_once_blocking()

    assert discovery.task_id == "from-router"
    assert core.list_calls[0]["label_selector"] == "custom_task=from-router"


def test_k8s_discovery_watch_adds_modifies_and_deletes_workers():
    registry = WorkerRegistry()
    removed = []
    prefill = _pod("prefill", "prefill-uid", pod_ip="10.0.0.2", env={"WORKERTYPE": "PREFILL"})
    headless = _pod(
        "prefill",
        "prefill-uid",
        pod_ip="10.0.0.2",
        env={"WORKERTYPE": "PREFILL", "HEADLESS": "1"},
        resource_version="2",
    )
    watcher = FakeWatch(
        [
            {"type": "ADDED", "object": prefill},
            {"type": "MODIFIED", "object": headless},
            {"type": "DELETED", "object": prefill},
        ]
    )
    discovery = K8SPodDiscovery(
        _config(),
        registry,
        on_workers_removed=removed.extend,
        core_v1=FakeCoreV1([]),
        watch_factory=lambda: watcher,
        env={"HOSTNAME": "router"},
    )

    discovery.watch_once_blocking()

    assert registry.get_all_urls() == []
    assert removed == ["http://10.0.0.2:8100@0", "http://10.0.0.2:8100@1"]
    assert watcher.kwargs["resource_version"] is None


def test_k8s_discovery_notifies_added_workers_once_and_starts_unhealthy():
    registry = WorkerRegistry()
    added = []
    prefill = _pod(
        "prefill",
        "prefill-uid",
        pod_ip="10.0.0.2",
        env={"WORKERTYPE": "PREFILL"},
    )
    discovery = K8SPodDiscovery(
        _config(),
        registry,
        on_workers_added=added.extend,
        core_v1=FakeCoreV1([]),
        watch_factory=lambda: FakeWatch([]),
        env={"HOSTNAME": "router"},
    )

    discovery.apply_pod(prefill)
    discovery.apply_pod(prefill)

    assert added == ["http://10.0.0.2:8100@0", "http://10.0.0.2:8100@1"]
    assert all(not worker.is_healthy() for worker in registry.get_all())


def test_engine_debounces_discovery_health_refresh_requests():
    class TestEngine(Engine):
        def __init__(self):
            self._event_loop = None
            self._background_tasks = set()
            self._debounced_health_refresh_task = None
            self._health_refresh_debounce_secs = 0.01
            self.refresh_calls = 0
            self.refreshed = asyncio.Event()

        async def refresh_worker_health(self, request_id=""):
            self.refresh_calls += 1
            self.refreshed.set()

    async def run():
        engine = TestEngine()
        engine._event_loop = asyncio.get_running_loop()

        engine.request_debounced_health_refresh(["worker-a"])
        engine.request_debounced_health_refresh(["worker-b"])

        await asyncio.wait_for(engine.refreshed.wait(), timeout=1)
        await asyncio.sleep(0.02)

        assert engine.refresh_calls == 1

    asyncio.run(run())


def test_worker_registry_register_is_idempotent_for_same_worker_id():
    registry = WorkerRegistry()
    config = SmartRouterConfig()
    worker = BasicWorker("http://worker", WorkerType.PREFILL, config)

    registry.register(worker)
    registry.register(worker)

    assert registry.get_all_urls() == ["http://worker"]
    assert registry.get_by_type(WorkerType.PREFILL) == [worker]
