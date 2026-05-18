import logging
import asyncio
import zmq
import zmq.asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import platform

from smart_router.engine.utils import make_zmq_socket
from smart_router.worker import Worker, WorkerRegistry, WorkerType

logger = logging.getLogger(__name__)
K8S_HEALTH_REFRESH_DEBOUNCE_SECS = 1.0

is_linux = platform.system() == "Linux"
if is_linux:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("enable uvloop event loop poicy!")


class RequestType:
    SCHEDULE = "schedule"
    RELEASE = "release"
    HEALTH = "health"
    WORKERS = "workers"


@dataclass
class EngineRequest:
    request_id: str
    identity: str
    request_type: RequestType  # "schedule" | "release"
    # request_type: schedule
    request_text: str = field(default="")
    headers: Dict[str, str] = field(default_factory=dict)
    request_token_ids: List[int] = field(default_factory=list)
    # request_type: release
    worker_url: str = field(default="")
    worker_rank: int = field(default=-1)


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineRequest":
        return cls(
            request_id=data["request_id"],
            identity=data["identity"],
            request_type=data["request_type"],
            request_text=data.get("request_text", ""),
            headers=data.get("headers", {}),
            request_token_ids=data.get("request_token_ids", []),
            worker_url=data.get("worker_url", ""),
            worker_rank=data.get("worker_rank", -1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "identity": self.identity,
            "request_type": self.request_type,
            "request_text": self.request_text,
            "headers": self.headers,
            "request_token_ids": self.request_token_ids,
            "worker_url": self.worker_url,
            "worker_rank": self.worker_rank,
        }
    

@dataclass
class EngineResponse:
    request_id: str
    prefill_url: Optional[str]
    prefill_rank: int
    decode_url: Optional[str]
    decode_rank: int
    error: str = ""
    # Normal mode: single worker
    worker_url: str = ""
    worker_rank: int = -1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineResponse":
        return cls(
            request_id=data["request_id"],
            prefill_url=data.get("prefill_url", ""),
            prefill_rank=data.get("prefill_rank", -1),
            decode_url=data.get("decode_url", ""),
            decode_rank=data.get("decode_rank", -1),
            worker_url=data.get("worker_url", ""),
            worker_rank=data.get("worker_rank", -1),
            error=data.get("error", ""),
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prefill_url": self.prefill_url,
            "prefill_rank": self.prefill_rank,
            "decode_url": self.decode_url,
            "decode_rank": self.decode_rank,
            "worker_url": self.worker_url,
            "worker_rank": self.worker_rank,
            "error": self.error,
            "response_type": RequestType.SCHEDULE,
        }


@dataclass
class EngineHealthResponse:
    request_id: str
    status: str
    prefill_healthy: int
    prefill_total: int
    decode_healthy: int
    decode_total: int
    regular_healthy: int = 0
    regular_total: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineHealthResponse":
        return cls(
            request_id=data["request_id"],
            status=data.get("status", "unhealthy"),
            prefill_healthy=data.get("prefill_healthy", 0),
            prefill_total=data.get("prefill_total", 0),
            decode_healthy=data.get("decode_healthy", 0),
            decode_total=data.get("decode_total", 0),
            regular_healthy=data.get("regular_healthy", 0),
            regular_total=data.get("regular_total", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "prefill_healthy": self.prefill_healthy,
            "prefill_total": self.prefill_total,
            "decode_healthy": self.decode_healthy,
            "decode_total": self.decode_total,
            "regular_healthy": self.regular_healthy,
            "regular_total": self.regular_total,
            "response_type": RequestType.HEALTH,
        }


@dataclass
class EngineWorkersResponse:
    request_id: str
    urls: List[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineWorkersResponse":
        return cls(
            request_id=data["request_id"],
            urls=[url for url in data.get("urls", []) if isinstance(url, str)],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "urls": self.urls,
            "response_type": RequestType.WORKERS,
        }
    


class Engine:
    def __init__(
        self,
        input_socket_address: str,
        output_socket_address: str,
    ) -> None:
        # Initialize ZeroMQ context and sockets
        ctx = zmq.Context()
        self.async_ctx = zmq.asyncio.Context(ctx)
        self.input_socket: zmq.Socket = make_zmq_socket(self.async_ctx, input_socket_address, zmq.PULL)
        self.output_socket: zmq.Socket = make_zmq_socket(self.async_ctx, output_socket_address, zmq.ROUTER)

        # queues for scheduling
        self.waiting_queue: asyncio.Queue[EngineRequest] = asyncio.Queue()

        self.worker_registry: WorkerRegistry = WorkerRegistry()
        self._health_check_lock = asyncio.Lock()
        self._background_tasks: set[asyncio.Task] = set()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._debounced_health_refresh_task: asyncio.Task | None = None
        self._health_refresh_debounce_secs = K8S_HEALTH_REFRESH_DEBOUNCE_SECS
        self.worker_discovery = None


    async def receive_loop(self):
        while True:
            request = await self.input_socket.recv_json()
            engine_request = EngineRequest.from_dict(request)
            if engine_request.request_type == RequestType.SCHEDULE:
                await self.waiting_queue.put(engine_request)
                logger.debug(f"Received schedule request: {engine_request.request_id}, queue size: {self.waiting_queue.qsize()}")

            elif engine_request.request_type == RequestType.RELEASE:
                if engine_request.worker_rank == -1:
                    worker_id = engine_request.worker_url
                else:
                    worker_id = f"{engine_request.worker_url}@{engine_request.worker_rank}"

                worker = self.worker_registry.get(worker_id)
                if worker is not None:
                    worker.decrement_load()
                else:
                    logger.warning("Release request referenced unknown worker id=%s", worker_id)

                logger.debug(f"Received release request: {engine_request.request_id}, worker id: {worker_id}")

            elif engine_request.request_type == RequestType.HEALTH:
                task = asyncio.create_task(self._handle_health_request(engine_request))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            elif engine_request.request_type == RequestType.WORKERS:
                resp = EngineWorkersResponse(
                    request_id=engine_request.request_id,
                    urls=self.get_worker_base_urls(),
                )
                await self.send_response(engine_request, resp.to_dict())

    async def schedule_loop(self):
        while True:
            request = await self.waiting_queue.get()
            logger.debug(f"Processing prefill for request: {request.request_id}")
            # schedule
            schedule_context = self.build_schedule_context(request)
            prefill_worker = self.schedule_prefill(
                request.request_text,
                request.headers,
                request_token_ids=request.request_token_ids,
                schedule_context=schedule_context,
            )
            decode_worker = self.schedule_decode(
                request.request_text,
                request.headers,
                request_token_ids=request.request_token_ids,
                schedule_context=schedule_context,
            )

            if prefill_worker is None:
                resp = EngineResponse(
                    request_id=request.request_id,
                    prefill_url=None,
                    prefill_rank=-1,
                    decode_url=None,
                    decode_rank=-1,
                    error="No available prefill workers",
                )
                await self.send_response(request, resp.to_dict())
                continue

            if decode_worker is None:
                resp = EngineResponse(
                    request_id=request.request_id,
                    prefill_url=prefill_worker.base_url(),
                    prefill_rank=prefill_worker.dp_rank(),
                    decode_url=None,
                    decode_rank=-1,
                    error="No available decode workers",
                )
                await self.send_response(request, resp.to_dict())
                continue

            prefill_worker.increment_load()
            decode_worker.increment_load()
            # build resp
            resp = EngineResponse(
                request_id=request.request_id,
                prefill_url=prefill_worker.base_url(), 
                prefill_rank=prefill_worker.dp_rank(),
                decode_url=decode_worker.base_url(),
                decode_rank=decode_worker.dp_rank(),
            )
            await self.send_response(request, resp.to_dict())

    async def _handle_health_request(self, request: EngineRequest) -> None:
        try:
            health_response = await self.refresh_worker_health(
                request_id=request.request_id
            )
        except Exception:
            logger.exception("Failed to process health request")
            health_response = self.get_health_response(request_id=request.request_id)

        await self.send_response(request, health_response.to_dict())

    async def refresh_worker_health(self, request_id: str = "") -> EngineHealthResponse:
        async with self._health_check_lock:
            groups = self.worker_registry.get_health_check_groups()
            checks = [
                (
                    worker_type,
                    base_url,
                    asyncio.create_task(workers[0].check_health_async()),
                )
                for worker_type, base_url, workers in groups
                if workers
            ]

            for worker_type, base_url, task in checks:
                try:
                    healthy = await task
                except Exception:
                    logger.exception("Worker health check failed: %s %s", worker_type, base_url)
                    healthy = False

                if not healthy:
                    logger.warning(
                        "Worker is unhealthy after health check: type=%s url=%s",
                        worker_type,
                        base_url,
                    )

                self.worker_registry.set_group_health(worker_type, base_url, healthy)

            return self.get_health_response(request_id=request_id)

    def get_health_response(self, request_id: str = "") -> EngineHealthResponse:
        counts = self.worker_registry.health_counts_by_type()
        prefill_healthy, prefill_total = counts.get(WorkerType.PREFILL, (0, 0))
        decode_healthy, decode_total = counts.get(WorkerType.DECODE, (0, 0))
        regular_healthy, regular_total = counts.get(WorkerType.REGULAR, (0, 0))
        has_pd_workers = prefill_total > 0 or decode_total > 0
        if has_pd_workers:
            healthy = (
                prefill_healthy > 0
                and prefill_total > 0
                and decode_healthy > 0
                and decode_total > 0
            )
        else:
            healthy = regular_healthy > 0 and regular_total > 0
        status = "ok" if healthy else "unhealthy"
        return EngineHealthResponse(
            request_id=request_id,
            status=status,
            prefill_healthy=prefill_healthy,
            prefill_total=prefill_total,
            decode_healthy=decode_healthy,
            decode_total=decode_total,
            regular_healthy=regular_healthy,
            regular_total=regular_total,
        )

    def get_worker_base_urls(self) -> List[str]:
        urls = []
        seen = set()
        for worker in self.worker_registry.get_all():
            base_url = worker.base_url()
            if base_url in seen:
                continue
            seen.add(base_url)
            urls.append(base_url)
        return urls

    def configure_worker_discovery(self, config) -> None:
        discovery_config = getattr(config, "k8s_discovery_config", None)
        if discovery_config is None or not discovery_config.enabled:
            self.worker_discovery = None
            return

        from smart_router.discovery import K8SPodDiscovery

        self.worker_discovery = K8SPodDiscovery(
            config,
            self.worker_registry,
            on_workers_added=self.request_debounced_health_refresh,
            on_workers_removed=self._remove_workers_from_policies,
        )

    def request_debounced_health_refresh(
        self, worker_ids: List[str] | None = None
    ) -> None:
        if worker_ids:
            logger.info(
                "Scheduling debounced health refresh for new K8S workers: %s",
                worker_ids,
            )

        loop = self._event_loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.debug(
                    "Skipping debounced health refresh; event loop is not running"
                )
                return
            self._event_loop = loop

        if loop.is_closed():
            return

        loop.call_soon_threadsafe(self._schedule_debounced_health_refresh)

    def _schedule_debounced_health_refresh(self) -> None:
        current = self._debounced_health_refresh_task
        if current is not None and not current.done():
            current.cancel()

        task = asyncio.create_task(self._debounced_health_refresh())
        self._debounced_health_refresh_task = task
        self._background_tasks.add(task)
        task.add_done_callback(self._finish_debounced_health_refresh)

    def _finish_debounced_health_refresh(self, task: asyncio.Task) -> None:
        if self._debounced_health_refresh_task is task:
            self._debounced_health_refresh_task = None
        self._background_tasks.discard(task)

    async def _debounced_health_refresh(self) -> None:
        try:
            await asyncio.sleep(self._health_refresh_debounce_secs)
            await self.refresh_worker_health()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Debounced worker health refresh failed")

    def _remove_workers_from_policies(self, worker_ids: List[str]) -> None:
        for policy in (
            getattr(self, "prefill_policy", None),
            getattr(self, "decode_policy", None),
        ):
            remover = getattr(policy, "remove_workers", None)
            if remover is not None:
                remover(worker_ids)

    def build_schedule_context(self, request: EngineRequest) -> Dict[str, Any]:
        _ = request
        return {}

    async def health_check_loop(self):
        while True:
            interval_secs = getattr(
                getattr(self, "config", None),
                "health_config",
                None,
            )
            interval_secs = (
                interval_secs.check_interval_secs
                if interval_secs is not None
                else 60
            )
            await asyncio.sleep(interval_secs)
            await self.refresh_worker_health()

    async def send_response(self, request: EngineRequest, msg: Dict[str, Any]) -> None:
        await self.output_socket.send_multipart([
            request.identity.encode("utf-8"),
            b"",
            json.dumps(msg).encode("utf-8"),
        ])

    async def run(self):
        """
        receive_loop: request  -> prefill_waiting_queue 
        prefill_loop: prefill_waiting_queue -> request -> handle prefill -> decode_waiting_queue
        decode_loop: decode_waiting_queue -> request -> handle decode
        """
        self._event_loop = asyncio.get_running_loop()
        tasks = []
        if self.worker_discovery is not None:
            await self.worker_discovery.sync_once()

        await self.refresh_worker_health()
        tasks.extend(
            [
                self.receive_loop(),
                self.schedule_loop(),
                self.health_check_loop(),
            ]
        )
        if self.worker_discovery is not None:
            tasks.append(self.worker_discovery.run_loop())
        await asyncio.gather(*tasks)

    async def shutdown(self):
        """Gracefully shutdown the engine:
        1. stopping receiving new requests;
        2. waiting for in-flight tasks to complete."""
        # stop receiving new requests
        self.input_socket.close(0)

        # closesocket
        self.output_socket.close(0)
        if self.worker_discovery is not None:
            self.worker_discovery.stop()
        task = getattr(self, "_debounced_health_refresh_task", None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._stop_policies()

    def _stop_policies(self) -> None:
        for policy in (
            getattr(self, "prefill_policy", None),
            getattr(self, "decode_policy", None),
        ):
            stopper = getattr(policy, "stop", None)
            if stopper is None:
                continue
            try:
                stopper()
            except Exception:
                logger.exception("Failed to stop policy %s", policy)

    def schedule_prefill(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_token_ids: Optional[List[int]] = None,
        schedule_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Worker]:
        raise NotImplementedError
    
    def schedule_decode(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_token_ids: Optional[List[int]] = None,
        schedule_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Worker]:
        raise NotImplementedError
    
    
