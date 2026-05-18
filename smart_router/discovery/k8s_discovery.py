from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from smart_router.config import SmartRouterConfig
from smart_router.worker import WorkerRegistry, WorkerType
from smart_router.worker.factory import register_workers_for_url

logger = logging.getLogger(__name__)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
WATCH_TIMEOUT_SECONDS = 30
WATCH_REQUEST_TIMEOUT_SECONDS = 35


@dataclass(frozen=True)
class PodWorkerSpec:
    uid: str
    name: str
    url: str
    worker_type: WorkerType


@dataclass
class PodRegistration:
    signature: Tuple[str, str]
    worker_ids: List[str]


class K8SPodDiscovery:
    """Discover prefill/decode workers by watching Kubernetes pods."""

    def __init__(
        self,
        router_config: SmartRouterConfig,
        worker_registry: WorkerRegistry,
        on_workers_removed: Optional[Callable[[List[str]], None]] = None,
        on_workers_added: Optional[Callable[[List[str]], None]] = None,
        core_v1: Any | None = None,
        watch_factory: Any | None = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.router_config = router_config
        self.config = router_config.k8s_discovery_config
        self.worker_registry = worker_registry
        self.on_workers_added = on_workers_added
        self.on_workers_removed = on_workers_removed
        self.core_v1 = core_v1
        self.watch_factory = watch_factory
        self.env = env if env is not None else os.environ

        self.namespace: Optional[str] = self.config.namespace
        self.task_id: Optional[str] = self.config.task_id
        self.own_pod_name: Optional[str] = None
        self.label_selector: Optional[str] = None
        self.resource_version: Optional[str] = None

        self._pod_registrations: Dict[str, PodRegistration] = {}
        self._initialized = False
        self._stop_event = threading.Event()
        self._watcher: Any | None = None

    async def sync_once(self) -> None:
        await asyncio.to_thread(self.sync_once_blocking)

    def sync_once_blocking(self) -> None:
        self._ensure_initialized()
        response = self.core_v1.list_namespaced_pod(
            namespace=self.namespace,
            label_selector=self.label_selector,
        )
        self.resource_version = self._resource_version(response) or self.resource_version

        seen_uids = set()
        for pod in self._items(response):
            uid = self._pod_uid(pod)
            if uid:
                seen_uids.add(uid)
            self.apply_pod(pod)

        for uid in list(self._pod_registrations):
            if uid not in seen_uids:
                self.remove_pod(uid)

        logger.info(
            "K8S discovery sync complete: namespace=%s label_selector=%s workers=%s",
            self.namespace,
            self.label_selector,
            self.worker_registry.get_all_urls(),
        )

    async def run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await asyncio.to_thread(self.watch_once_blocking)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._is_resource_expired(exc):
                    logger.warning("K8S watch resource version expired; resyncing pods")
                    await self.sync_once()
                else:
                    logger.exception("K8S discovery watch failed")
                await asyncio.sleep(2)

    def watch_once_blocking(self) -> None:
        self._ensure_initialized()
        watcher = self._new_watcher()
        self._watcher = watcher
        try:
            stream = watcher.stream(
                self.core_v1.list_namespaced_pod,
                namespace=self.namespace,
                label_selector=self.label_selector,
                resource_version=self.resource_version,
                timeout_seconds=WATCH_TIMEOUT_SECONDS,
                _request_timeout=WATCH_REQUEST_TIMEOUT_SECONDS,
            )
            for event in stream:
                if self._stop_event.is_set():
                    break

                event_type = self._event_type(event)
                pod = self._event_object(event)
                if pod is None:
                    continue

                self.resource_version = (
                    self._resource_version(pod) or self.resource_version
                )
                uid = self._pod_uid(pod)
                if event_type == "DELETED":
                    if uid:
                        self.remove_pod(uid)
                    continue

                self.apply_pod(pod)
        finally:
            self._watcher = None

    def stop(self) -> None:
        self._stop_event.set()
        watcher = self._watcher
        if watcher is not None and hasattr(watcher, "stop"):
            watcher.stop()

    def apply_pod(self, pod: Any) -> None:
        spec = self._worker_spec_from_pod(pod)
        uid = self._pod_uid(pod)
        if spec is None:
            if uid:
                self.remove_pod(uid)
            return

        signature = (spec.worker_type.value, spec.url)
        current = self._pod_registrations.get(spec.uid)
        if current is not None and current.signature == signature:
            return

        if current is not None:
            self.remove_pod(spec.uid)

        worker_ids = register_workers_for_url(
            self.worker_registry,
            spec.url,
            spec.worker_type,
            self.router_config,
        )
        self._set_workers_health(worker_ids, healthy=False)
        self._pod_registrations[spec.uid] = PodRegistration(
            signature=signature,
            worker_ids=worker_ids,
        )
        logger.info(
            "Registered K8S worker pod: name=%s type=%s url=%s worker_ids=%s",
            spec.name,
            spec.worker_type.value,
            spec.url,
            worker_ids,
        )
        if self.on_workers_added is not None:
            try:
                self.on_workers_added(worker_ids)
            except Exception:
                logger.exception(
                    "Failed to notify K8S worker addition: worker_ids=%s",
                    worker_ids,
                )

    def remove_pod(self, uid: str) -> None:
        current = self._pod_registrations.pop(uid, None)
        if current is None:
            return

        removed = []
        for worker_id in current.worker_ids:
            if self.worker_registry.remove(worker_id) is not None:
                removed.append(worker_id)

        if removed:
            logger.info("Removed K8S worker pod uid=%s worker_ids=%s", uid, removed)
            if self.on_workers_removed is not None:
                self.on_workers_removed(removed)

    def registered_worker_ids(self) -> List[str]:
        worker_ids = []
        for registration in self._pod_registrations.values():
            worker_ids.extend(registration.worker_ids)
        return worker_ids

    def _set_workers_health(self, worker_ids: List[str], healthy: bool) -> None:
        for worker_id in worker_ids:
            worker = self.worker_registry.get(worker_id)
            if worker is not None:
                worker.set_healthy(healthy)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        self._ensure_k8s_client()
        self.namespace = self.namespace or self._read_service_account_namespace()
        self.own_pod_name = self.env.get("POD_NAME") or self.env.get("HOSTNAME")
        self.task_id = self.task_id or self._read_own_task_id()
        self.label_selector = f"{self.config.task_label_key}={self.task_id}"
        self._initialized = True

        logger.info(
            "K8S discovery initialized: namespace=%s label_selector=%s own_pod=%s",
            self.namespace,
            self.label_selector,
            self.own_pod_name,
        )

    def _ensure_k8s_client(self) -> None:
        if self.core_v1 is not None and self.watch_factory is not None:
            return

        try:
            from kubernetes import client, config, watch
        except ImportError as exc:
            raise RuntimeError(
                "K8S discovery requires the `kubernetes` Python package."
            ) from exc

        if self.core_v1 is None:
            try:
                config.load_incluster_config()
            except Exception:
                logger.info("Falling back to local kubeconfig for K8S discovery")
                config.load_kube_config()
            self.core_v1 = client.CoreV1Api()

        if self.watch_factory is None:
            self.watch_factory = watch.Watch

    def _read_service_account_namespace(self) -> str:
        env_namespace = self.env.get("POD_NAMESPACE")
        if env_namespace:
            return env_namespace

        namespace_path = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        try:
            with open(namespace_path, "r", encoding="utf-8") as namespace_file:
                namespace = namespace_file.read().strip()
                if namespace:
                    return namespace
        except OSError:
            pass

        return "default"

    def _read_own_task_id(self) -> str:
        if not self.own_pod_name:
            raise RuntimeError(
                "K8S discovery needs POD_NAME or HOSTNAME to read the router pod label."
            )

        pod = self.core_v1.read_namespaced_pod(
            name=self.own_pod_name,
            namespace=self.namespace,
        )
        labels = self._labels(pod)
        task_id = labels.get(self.config.task_label_key)
        if not task_id:
            raise RuntimeError(
                f"Router pod is missing `{self.config.task_label_key}` label."
            )
        return task_id

    def _new_watcher(self) -> Any:
        return self.watch_factory() if callable(self.watch_factory) else self.watch_factory

    def _worker_spec_from_pod(self, pod: Any) -> Optional[PodWorkerSpec]:
        metadata = self._metadata(pod)
        status = self._status(pod)
        name = self._value(metadata, "name", "")
        uid = self._pod_uid(pod)
        if not uid:
            return None

        if name and name == self.own_pod_name:
            return None

        if self._value(metadata, "deletion_timestamp") is not None:
            return None

        if self._value(status, "phase") != "Running":
            return None

        pod_ip = self._value(status, "pod_ip") or self._value(status, "podIP")
        if not pod_ip:
            return None

        env = self._pod_env(pod)
        if self._is_truthy(env.get("HEADLESS")):
            return None

        raw_worker_type = env.get("WORKERTYPE", "").upper()
        worker_type = {
            "PREFILL": WorkerType.PREFILL,
            "DECODE": WorkerType.DECODE,
            "REGULAR": WorkerType.REGULAR,
        }.get(raw_worker_type)
        if worker_type is None:
            return None

        port = self._port_for_worker_type(worker_type)
        if port is None:
            return None

        return PodWorkerSpec(
            uid=uid,
            name=name,
            url=self._build_url(pod_ip, port),
            worker_type=worker_type,
        )

    def _port_for_worker_type(self, worker_type: WorkerType) -> Optional[int]:
        if worker_type == WorkerType.PREFILL:
            return self.config.prefill_port
        if worker_type == WorkerType.DECODE:
            return self.config.decode_port
        if worker_type == WorkerType.REGULAR:
            return self.config.regular_port
        return None

    def _pod_env(self, pod: Any) -> Dict[str, str]:
        values: Dict[str, str] = {}
        spec = self._spec(pod)
        containers = self._value(spec, "containers", []) or []
        for container in containers:
            env_list = self._value(container, "env", []) or []
            for env_var in env_list:
                name = self._value(env_var, "name")
                if not name:
                    continue
                value = self._value(env_var, "value", "") or ""
                values[str(name).upper()] = str(value)
        return values

    def _build_url(self, host: str, port: int) -> str:
        try:
            address = ipaddress.ip_address(host)
            if address.version == 6:
                host = f"[{host}]"
        except ValueError:
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
        return f"{self.config.url_scheme}://{host}:{port}"

    @staticmethod
    def _is_truthy(value: Optional[str]) -> bool:
        if value is None:
            return False
        return value.strip().lower() in TRUTHY_ENV_VALUES

    @staticmethod
    def _is_resource_expired(exc: Exception) -> bool:
        return getattr(exc, "status", None) == 410

    @staticmethod
    def _event_type(event: Any) -> str:
        if isinstance(event, dict):
            return str(event.get("type", ""))
        return str(getattr(event, "type", ""))

    @staticmethod
    def _event_object(event: Any) -> Any:
        if isinstance(event, dict):
            return event.get("object")
        return getattr(event, "object", None)

    @classmethod
    def _items(cls, response: Any) -> Iterable[Any]:
        return cls._value(response, "items", []) or []

    @classmethod
    def _pod_uid(cls, pod: Any) -> str:
        metadata = cls._metadata(pod)
        return cls._value(metadata, "uid") or cls._value(metadata, "name", "")

    @classmethod
    def _resource_version(cls, obj: Any) -> Optional[str]:
        metadata = cls._metadata(obj)
        return cls._value(metadata, "resource_version")

    @classmethod
    def _labels(cls, pod: Any) -> Dict[str, str]:
        labels = cls._value(cls._metadata(pod), "labels", {}) or {}
        return dict(labels)

    @classmethod
    def _metadata(cls, obj: Any) -> Any:
        return cls._value(obj, "metadata")

    @classmethod
    def _spec(cls, obj: Any) -> Any:
        return cls._value(obj, "spec")

    @classmethod
    def _status(cls, obj: Any) -> Any:
        return cls._value(obj, "status")

    @staticmethod
    def _value(obj: Any, name: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)
