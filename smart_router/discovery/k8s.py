from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from smart_router.config import SmartRouterConfig
from smart_router.worker import WorkerType

logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
NAMESPACE_PATH = SERVICE_ACCOUNT_DIR / "namespace"

WorkerCallback = Callable[["DiscoveredWorker"], Awaitable[None]]


class ResourceVersionExpired(RuntimeError):
    """Kubernetes watch resourceVersion is too old and must be relisted."""


@dataclass(frozen=True)
class DiscoveredWorker:
    pod_name: str
    worker_type: WorkerType
    base_url: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def _format_url_host(host: str) -> str:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.version == 6:
        return f"[{host}]"
    return host


def _get_value(obj: Any, *names: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default

    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _pod_metadata(pod: Any) -> Any:
    return _get_value(pod, "metadata", default={}) or {}


def _pod_status(pod: Any) -> Any:
    return _get_value(pod, "status", default={}) or {}


def _pod_spec(pod: Any) -> Any:
    return _get_value(pod, "spec", default={}) or {}


def _pod_name(pod: Any) -> str:
    return str(_get_value(_pod_metadata(pod), "name", default="") or "")


def _pod_resource_version(pod: Any) -> Optional[str]:
    resource_version = _get_value(
        _pod_metadata(pod),
        "resourceVersion",
        "resource_version",
    )
    if resource_version is None:
        return None
    return str(resource_version)


def _pod_labels(pod: Any) -> Dict[str, str]:
    labels = _get_value(_pod_metadata(pod), "labels", default={}) or {}
    return {str(key): str(value) for key, value in labels.items()}


def _pod_is_ready(pod: Any) -> bool:
    conditions = _get_value(_pod_status(pod), "conditions", default=[]) or []
    for condition in conditions:
        if _get_value(condition, "type") == "Ready":
            return _get_value(condition, "status") == "True"
    return False


def _container_env_values(pod: Any) -> Dict[str, list[str]]:
    env_values: Dict[str, list[str]] = {}
    containers = _get_value(_pod_spec(pod), "containers", default=[]) or []
    for container in containers:
        for env in _get_value(container, "env", default=[]) or []:
            name = _get_value(env, "name")
            value = _get_value(env, "value")
            if not name or value is None:
                continue
            env_values.setdefault(str(name), []).append(str(value))
    return env_values


def _worker_type_from_env(env_values: Dict[str, list[str]]) -> Optional[WorkerType]:
    for value in env_values.get("WORKERTYPE", []):
        normalized = value.strip().upper()
        if normalized == "PREFILL":
            return WorkerType.PREFILL
        if normalized == "DECODE":
            return WorkerType.DECODE
    return None


def parse_worker_pod(
    pod: Any,
    *,
    task_label_key: str,
    task_id: str,
    prefill_port: int,
    decode_port: int,
) -> Optional[DiscoveredWorker]:
    labels = _pod_labels(pod)
    if labels.get(task_label_key) != task_id:
        return None

    status = _pod_status(pod)
    if _get_value(status, "phase") != "Running":
        return None
    pod_ip = _get_value(status, "podIP", "pod_ip")
    if not pod_ip:
        return None
    if not _pod_is_ready(pod):
        return None

    env_values = _container_env_values(pod)
    if any(_is_true(value) for value in env_values.get("HEADLESS", [])):
        return None

    worker_type = _worker_type_from_env(env_values)
    if worker_type is None:
        return None

    port = prefill_port if worker_type == WorkerType.PREFILL else decode_port
    host = _format_url_host(str(pod_ip))
    return DiscoveredWorker(
        pod_name=_pod_name(pod),
        worker_type=worker_type,
        base_url=f"http://{host}:{port}",
    )


class K8sPodDiscovery:
    def __init__(
        self,
        *,
        namespace: Optional[str],
        task_label_key: str,
        prefill_port: int,
        decode_port: int,
        core_v1_api: Optional[Any] = None,
        watch_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.namespace = namespace
        self.task_label_key = task_label_key
        self.prefill_port = prefill_port
        self.decode_port = decode_port
        self._api = core_v1_api
        self._watch_factory = watch_factory
        self._active_watch: Optional[Any] = None
        self._pod_workers: Dict[str, DiscoveredWorker] = {}

    @classmethod
    def from_config(cls, config: SmartRouterConfig) -> "K8sPodDiscovery":
        if config.prefill_port is None or config.decode_port is None:
            raise ValueError("Kubernetes discovery requires prefill_port and decode_port")
        return cls(
            namespace=config.k8s_namespace,
            task_label_key=config.k8s_task_label_key,
            prefill_port=config.prefill_port,
            decode_port=config.decode_port,
        )

    async def run(
        self,
        *,
        on_upsert: WorkerCallback,
        on_delete: WorkerCallback,
    ) -> None:
        self.namespace = self.namespace or self._load_namespace()
        self._api = self._api or self._build_core_v1_api()
        task_id = await self._load_self_task_id()
        logger.info(
            "starting Kubernetes pod discovery: namespace=%s %s=%s",
            self.namespace,
            self.task_label_key,
            task_id,
        )

        resource_version: Optional[str] = None
        while True:
            try:
                resource_version = await self._list_and_sync(
                    task_id=task_id,
                    on_upsert=on_upsert,
                    on_delete=on_delete,
                )
                resource_version = await self._watch(
                    task_id=task_id,
                    resource_version=resource_version,
                    on_upsert=on_upsert,
                    on_delete=on_delete,
                )
            except ResourceVersionExpired:
                logger.info("Kubernetes pod watch resourceVersion expired; relisting")
                resource_version = None
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Kubernetes pod discovery failed; retrying")
                await asyncio.sleep(5)

    def _load_namespace(self) -> str:
        if NAMESPACE_PATH.exists():
            return _read_text(NAMESPACE_PATH)
        raise RuntimeError(
            "Kubernetes namespace was not provided and service account namespace is missing"
        )

    def _build_core_v1_api(self) -> Any:
        from kubernetes import client, config
        from kubernetes.config.config_exception import ConfigException

        try:
            config.load_incluster_config()
        except ConfigException:
            logger.info("Falling back to local kubeconfig for Kubernetes discovery")
            config.load_kube_config()
        return client.CoreV1Api()

    def _label_selector(self, task_id: str) -> str:
        return f"{self.task_label_key}={task_id}"

    async def _load_self_task_id(self) -> str:
        pod_name = os.getenv("HOSTNAME")
        if not pod_name:
            raise RuntimeError("HOSTNAME is not set; cannot identify router pod")
        assert self._api is not None
        pod = await asyncio.to_thread(
            self._api.read_namespaced_pod,
            name=pod_name,
            namespace=self.namespace,
        )
        labels = _pod_labels(pod)
        task_id = labels.get(self.task_label_key)
        if not task_id:
            raise RuntimeError(
                f"Router pod {pod_name!r} is missing label {self.task_label_key!r}"
            )
        return task_id

    async def _list_and_sync(
        self,
        *,
        task_id: str,
        on_upsert: WorkerCallback,
        on_delete: WorkerCallback,
    ) -> Optional[str]:
        assert self._api is not None
        pod_list = await asyncio.to_thread(
            self._api.list_namespaced_pod,
            namespace=self.namespace,
            label_selector=self._label_selector(task_id),
        )
        seen_names = set()
        for pod in _get_value(pod_list, "items", default=[]) or []:
            name = _pod_name(pod)
            if name:
                seen_names.add(name)
            await self._apply_event("ADDED", pod, task_id, on_upsert, on_delete)

        for name, worker in list(self._pod_workers.items()):
            if name not in seen_names:
                self._pod_workers.pop(name, None)
                await on_delete(worker)

        return _get_value(
            _get_value(pod_list, "metadata", default={}),
            "resourceVersion",
            "resource_version",
        )

    async def _watch(
        self,
        *,
        task_id: str,
        resource_version: Optional[str],
        on_upsert: WorkerCallback,
        on_delete: WorkerCallback,
    ) -> Optional[str]:
        assert self._api is not None
        loop = asyncio.get_running_loop()

        def _run_watch() -> Optional[str]:
            from kubernetes.client.exceptions import ApiException

            sdk_watch = self._new_watch()
            self._active_watch = sdk_watch
            last_resource_version = resource_version
            try:
                for event in sdk_watch.stream(
                    self._api.list_namespaced_pod,
                    namespace=self.namespace,
                    label_selector=self._label_selector(task_id),
                    resource_version=resource_version,
                ):
                    event_type = event.get("type")
                    pod = event.get("object")
                    if event_type == "ERROR":
                        if _get_value(pod, "code") == 410:
                            raise ResourceVersionExpired()
                        logger.warning("Kubernetes watch error event: %s", pod)
                        continue

                    next_resource_version = _pod_resource_version(pod)
                    if next_resource_version:
                        last_resource_version = next_resource_version

                    future = asyncio.run_coroutine_threadsafe(
                        self._apply_event(
                            str(event_type),
                            pod,
                            task_id,
                            on_upsert,
                            on_delete,
                        ),
                        loop,
                    )
                    future.result()
            except ApiException as exc:
                if exc.status == 410:
                    raise ResourceVersionExpired() from exc
                raise
            finally:
                sdk_watch.stop()
                if self._active_watch is sdk_watch:
                    self._active_watch = None

            return last_resource_version

        try:
            return await asyncio.to_thread(_run_watch)
        except asyncio.CancelledError:
            active_watch = self._active_watch
            if active_watch is not None:
                active_watch.stop()
            raise

    def _new_watch(self) -> Any:
        if self._watch_factory is not None:
            return self._watch_factory()

        from kubernetes import watch

        return watch.Watch()

    async def _apply_event(
        self,
        event_type: str,
        pod: Any,
        task_id: str,
        on_upsert: WorkerCallback,
        on_delete: WorkerCallback,
    ) -> None:
        pod_name = _pod_name(pod)
        if not pod_name:
            return

        if event_type == "DELETED":
            old_worker = self._pod_workers.pop(pod_name, None)
            if old_worker is not None:
                await on_delete(old_worker)
            return

        new_worker = parse_worker_pod(
            pod,
            task_label_key=self.task_label_key,
            task_id=task_id,
            prefill_port=self.prefill_port,
            decode_port=self.decode_port,
        )
        old_worker = self._pod_workers.get(pod_name)
        if new_worker is None:
            if old_worker is not None:
                self._pod_workers.pop(pod_name, None)
                await on_delete(old_worker)
            return

        if old_worker == new_worker:
            return

        if old_worker is not None:
            await on_delete(old_worker)
        self._pod_workers[pod_name] = new_worker
        await on_upsert(new_worker)
