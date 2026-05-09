import asyncio

from smart_router.discovery.k8s import K8sPodDiscovery, parse_worker_pod
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
