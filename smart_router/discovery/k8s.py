from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional

import httpx

from smart_router.config import SmartRouterConfig
from smart_router.worker import WorkerType

logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
TOKEN_PATH = SERVICE_ACCOUNT_DIR / "token"
NAMESPACE_PATH = SERVICE_ACCOUNT_DIR / "namespace"
CA_CERT_PATH = SERVICE_ACCOUNT_DIR / "ca.crt"

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


def _pod_name(pod: dict) -> str:
    return (pod.get("metadata") or {}).get("name", "")


def _pod_labels(pod: dict) -> Dict[str, str]:
    labels = (pod.get("metadata") or {}).get("labels") or {}
    return {str(key): str(value) for key, value in labels.items()}


def _pod_is_ready(pod: dict) -> bool:
    conditions = (pod.get("status") or {}).get("conditions") or []
    for condition in conditions:
        if condition.get("type") == "Ready":
            return condition.get("status") == "True"
    return False


def _container_env_values(pod: dict) -> Dict[str, list[str]]:
    env_values: Dict[str, list[str]] = {}
    containers = (pod.get("spec") or {}).get("containers") or []
    for container in containers:
        for env in container.get("env") or []:
            name = env.get("name")
            if not name or "value" not in env:
                continue
            env_values.setdefault(str(name), []).append(str(env.get("value")))
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
    pod: dict,
    *,
    task_label_key: str,
    task_id: str,
    prefill_port: int,
    decode_port: int,
) -> Optional[DiscoveredWorker]:
    labels = _pod_labels(pod)
    if labels.get(task_label_key) != task_id:
        return None

    status = pod.get("status") or {}
    if status.get("phase") != "Running":
        return None
    pod_ip = status.get("podIP")
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
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.namespace = namespace
        self.task_label_key = task_label_key
        self.prefill_port = prefill_port
        self.decode_port = decode_port
        self._client = http_client
        self._owns_client = http_client is None
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
        self._client = self._client or self._build_client()
        try:
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
        finally:
            if self._owns_client and self._client is not None:
                await self._client.aclose()

    def _load_namespace(self) -> str:
        if NAMESPACE_PATH.exists():
            return _read_text(NAMESPACE_PATH)
        raise RuntimeError(
            "Kubernetes namespace was not provided and service account namespace is missing"
        )

    def _build_client(self) -> httpx.AsyncClient:
        if not TOKEN_PATH.exists():
            raise RuntimeError("Kubernetes service account token is missing")
        token = _read_text(TOKEN_PATH)
        verify: bool | str = str(CA_CERT_PATH) if CA_CERT_PATH.exists() else True
        return httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            timeout=httpx.Timeout(None),
            verify=verify,
        )

    def _api_base(self) -> str:
        host = os.getenv("KUBERNETES_SERVICE_HOST")
        if not host:
            raise RuntimeError("KUBERNETES_SERVICE_HOST is not set")
        port = (
            os.getenv("KUBERNETES_SERVICE_PORT_HTTPS")
            or os.getenv("KUBERNETES_SERVICE_PORT")
            or "443"
        )
        return f"https://{host}:{port}/api/v1"

    def _pods_url(self) -> str:
        return f"{self._api_base()}/namespaces/{self.namespace}/pods"

    def _pod_url(self, pod_name: str) -> str:
        return f"{self._pods_url()}/{pod_name}"

    def _label_selector(self, task_id: str) -> str:
        return f"{self.task_label_key}={task_id}"

    async def _load_self_task_id(self) -> str:
        pod_name = os.getenv("HOSTNAME")
        if not pod_name:
            raise RuntimeError("HOSTNAME is not set; cannot identify router pod")
        assert self._client is not None
        response = await self._client.get(self._pod_url(pod_name))
        response.raise_for_status()
        labels = _pod_labels(response.json())
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
        assert self._client is not None
        response = await self._client.get(
            self._pods_url(),
            params={"labelSelector": self._label_selector(task_id)},
        )
        response.raise_for_status()
        payload = response.json()
        seen_names = set()
        for pod in payload.get("items") or []:
            name = _pod_name(pod)
            if name:
                seen_names.add(name)
            await self._apply_event("ADDED", pod, task_id, on_upsert, on_delete)

        for name, worker in list(self._pod_workers.items()):
            if name not in seen_names:
                self._pod_workers.pop(name, None)
                await on_delete(worker)

        return (payload.get("metadata") or {}).get("resourceVersion")

    async def _watch(
        self,
        *,
        task_id: str,
        resource_version: Optional[str],
        on_upsert: WorkerCallback,
        on_delete: WorkerCallback,
    ) -> Optional[str]:
        assert self._client is not None
        params = {
            "watch": "true",
            "labelSelector": self._label_selector(task_id),
        }
        if resource_version:
            params["resourceVersion"] = resource_version

        last_resource_version = resource_version
        async with self._client.stream("GET", self._pods_url(), params=params) as response:
            if response.status_code == 410:
                raise ResourceVersionExpired()
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed Kubernetes watch line: %s", line)
                    continue

                event_type = event.get("type")
                pod = event.get("object") or {}
                if event_type == "ERROR":
                    if pod.get("code") == 410:
                        raise ResourceVersionExpired()
                    logger.warning("Kubernetes watch error event: %s", pod)
                    continue

                resource_version = (pod.get("metadata") or {}).get("resourceVersion")
                if resource_version:
                    last_resource_version = resource_version

                await self._apply_event(
                    str(event_type),
                    pod,
                    task_id,
                    on_upsert,
                    on_delete,
                )

        return last_resource_version

    async def _apply_event(
        self,
        event_type: str,
        pod: dict,
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
