import asyncio
from types import SimpleNamespace

from smart_router.discovery.k8s import (
    DiscoveredWorker,
    K8sPodDiscovery,
    parse_worker_pod,
)
from smart_router.worker import WorkerType


def _pod(
    name="worker-a",
    *,
    task_id="task-1",
    worker_type="PREFILL",
    headless=None,
    ready=True,
    phase="Running",
    pod_ip="10.0.0.8",
    resource_version="1",
):
    env = [{"name": "WORKERTYPE", "value": worker_type}]
    if headless is not None:
        env.append({"name": "HEADLESS", "value": headless})

    return {
        "metadata": {
            "name": name,
            "resourceVersion": resource_version,
            "labels": {"task_id": task_id},
        },
        "spec": {"containers": [{"name": "worker", "env": env}]},
        "status": {
            "phase": phase,
            "podIP": pod_ip,
            "conditions": [
                {"type": "Ready", "status": "True" if ready else "False"}
            ],
        },
    }


def _sdk_pod(
    name="worker-a",
    *,
    task_id="task-1",
    worker_type="PREFILL",
    headless=None,
    ready=True,
    phase="Running",
    pod_ip="10.0.0.8",
    resource_version="1",
):
    env = [SimpleNamespace(name="WORKERTYPE", value=worker_type)]
    if headless is not None:
        env.append(SimpleNamespace(name="HEADLESS", value=headless))

    return SimpleNamespace(
        metadata=SimpleNamespace(
            name=name,
            resource_version=resource_version,
            labels={"task_id": task_id},
        ),
        spec=SimpleNamespace(
            containers=[SimpleNamespace(name="worker", env=env)],
        ),
        status=SimpleNamespace(
            phase=phase,
            pod_ip=pod_ip,
            conditions=[
                SimpleNamespace(type="Ready", status="True" if ready else "False")
            ],
        ),
    )


def test_parse_worker_pod_builds_prefill_and_decode_urls():
    prefill = parse_worker_pod(
        _pod(worker_type="PREFILL"),
        task_label_key="task_id",
        task_id="task-1",
        prefill_port=8100,
        decode_port=8200,
    )
    decode = parse_worker_pod(
        _pod(name="worker-b", worker_type="DECODE", pod_ip="10.0.0.9"),
        task_label_key="task_id",
        task_id="task-1",
        prefill_port=8100,
        decode_port=8200,
    )

    assert prefill.worker_type == WorkerType.PREFILL
    assert prefill.base_url == "http://10.0.0.8:8100"
    assert decode.worker_type == WorkerType.DECODE
    assert decode.base_url == "http://10.0.0.9:8200"


def test_parse_worker_pod_accepts_sdk_style_objects():
    worker = parse_worker_pod(
        _sdk_pod(worker_type="DECODE", pod_ip="10.0.0.9"),
        task_label_key="task_id",
        task_id="task-1",
        prefill_port=8100,
        decode_port=8200,
    )

    assert worker.worker_type == WorkerType.DECODE
    assert worker.base_url == "http://10.0.0.9:8200"


def test_parse_worker_pod_filters_headless_not_ready_missing_ip_and_wrong_task():
    common_args = {
        "task_label_key": "task_id",
        "task_id": "task-1",
        "prefill_port": 8100,
        "decode_port": 8200,
    }

    assert parse_worker_pod(_pod(headless="true"), **common_args) is None
    assert parse_worker_pod(_pod(ready=False), **common_args) is None
    assert parse_worker_pod(_pod(pod_ip=None), **common_args) is None
    assert parse_worker_pod(_pod(task_id="other"), **common_args) is None


def test_parse_worker_pod_wraps_ipv6_addresses():
    worker = parse_worker_pod(
        _pod(pod_ip="fd00::1"),
        task_label_key="task_id",
        task_id="task-1",
        prefill_port=8100,
        decode_port=8200,
    )

    assert worker.base_url == "http://[fd00::1]:8100"


def test_apply_events_upserts_updates_and_deletes_workers():
    async def run():
        discovery = K8sPodDiscovery(
            namespace="inference",
            task_label_key="task_id",
            prefill_port=8100,
            decode_port=8200,
        )
        upserts = []
        deletes = []

        async def on_upsert(worker):
            upserts.append(worker)

        async def on_delete(worker):
            deletes.append(worker)

        await discovery._apply_event(
            "ADDED", _pod(name="worker-a"), "task-1", on_upsert, on_delete
        )
        await discovery._apply_event(
            "MODIFIED", _pod(name="worker-a"), "task-1", on_upsert, on_delete
        )
        await discovery._apply_event(
            "MODIFIED",
            _pod(name="worker-a", headless="true"),
            "task-1",
            on_upsert,
            on_delete,
        )
        await discovery._apply_event(
            "ADDED",
            _pod(name="worker-b", worker_type="DECODE", pod_ip="10.0.0.9"),
            "task-1",
            on_upsert,
            on_delete,
        )
        await discovery._apply_event(
            "DELETED",
            _pod(name="worker-b", worker_type="DECODE", pod_ip="10.0.0.9"),
            "task-1",
            on_upsert,
            on_delete,
        )

        assert [worker.base_url for worker in upserts] == [
            "http://10.0.0.8:8100",
            "http://10.0.0.9:8200",
        ]
        assert [worker.base_url for worker in deletes] == [
            "http://10.0.0.8:8100",
            "http://10.0.0.9:8200",
        ]

    asyncio.run(run())


def test_load_self_task_id_uses_core_v1_api(monkeypatch):
    class FakeCoreV1Api:
        def __init__(self):
            self.calls = []

        def read_namespaced_pod(self, name, namespace):
            self.calls.append((name, namespace))
            return _sdk_pod(name=name, task_id="task-1")

    async def run():
        api = FakeCoreV1Api()
        discovery = K8sPodDiscovery(
            namespace="inference",
            task_label_key="task_id",
            prefill_port=8100,
            decode_port=8200,
            core_v1_api=api,
        )
        monkeypatch.setenv("HOSTNAME", "router-pod")

        assert await discovery._load_self_task_id() == "task-1"
        assert api.calls == [("router-pod", "inference")]

    asyncio.run(run())


def test_list_and_sync_uses_core_v1_api_and_resource_version():
    class FakeCoreV1Api:
        def __init__(self):
            self.calls = []

        def list_namespaced_pod(self, namespace, label_selector):
            self.calls.append((namespace, label_selector))
            return SimpleNamespace(
                metadata=SimpleNamespace(resource_version="42"),
                items=[_sdk_pod(name="worker-a")],
            )

    async def run():
        api = FakeCoreV1Api()
        discovery = K8sPodDiscovery(
            namespace="inference",
            task_label_key="task_id",
            prefill_port=8100,
            decode_port=8200,
            core_v1_api=api,
        )
        discovery._pod_workers["old-worker"] = DiscoveredWorker(
            pod_name="old-worker",
            worker_type=WorkerType.PREFILL,
            base_url="http://10.0.0.7:8100",
        )
        upserts = []
        deletes = []

        async def on_upsert(worker):
            upserts.append(worker)

        async def on_delete(worker):
            deletes.append(worker)

        resource_version = await discovery._list_and_sync(
            task_id="task-1",
            on_upsert=on_upsert,
            on_delete=on_delete,
        )

        assert resource_version == "42"
        assert api.calls == [("inference", "task_id=task-1")]
        assert [worker.base_url for worker in upserts] == ["http://10.0.0.8:8100"]
        assert [worker.base_url for worker in deletes] == ["http://10.0.0.7:8100"]

    asyncio.run(run())


def test_watch_uses_sdk_watch_and_applies_events():
    class FakeCoreV1Api:
        def list_namespaced_pod(self, **kwargs):
            return None

    class FakeWatch:
        def __init__(self):
            self.calls = []
            self.stopped = False

        def stream(self, func, **kwargs):
            self.calls.append((func, kwargs))
            yield {"type": "ADDED", "object": _sdk_pod(name="worker-a")}
            yield {"type": "DELETED", "object": _sdk_pod(name="worker-a")}

        def stop(self):
            self.stopped = True

    async def run():
        api = FakeCoreV1Api()
        fake_watch = FakeWatch()
        discovery = K8sPodDiscovery(
            namespace="inference",
            task_label_key="task_id",
            prefill_port=8100,
            decode_port=8200,
            core_v1_api=api,
            watch_factory=lambda: fake_watch,
        )
        upserts = []
        deletes = []

        async def on_upsert(worker):
            upserts.append(worker)

        async def on_delete(worker):
            deletes.append(worker)

        resource_version = await discovery._watch(
            task_id="task-1",
            resource_version="41",
            on_upsert=on_upsert,
            on_delete=on_delete,
        )

        assert resource_version == "1"
        assert fake_watch.stopped is True
        assert discovery._active_watch is None
        assert fake_watch.calls == [
            (
                api.list_namespaced_pod,
                {
                    "namespace": "inference",
                    "label_selector": "task_id=task-1",
                    "resource_version": "41",
                },
            )
        ]
        assert [worker.base_url for worker in upserts] == ["http://10.0.0.8:8100"]
        assert [worker.base_url for worker in deletes] == ["http://10.0.0.8:8100"]

    asyncio.run(run())
