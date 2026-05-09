import logging
import asyncio
import zmq
import zmq.asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import platform

from smart_router.engine.utils import make_zmq_socket
from smart_router.worker import BasicWorker, DPAwareWorker, Worker, WorkerRegistry, WorkerType

logger = logging.getLogger(__name__)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineResponse":
        return cls(
            request_id=data["request_id"],
            prefill_url=data.get("prefill_url"),
            prefill_rank=data.get("prefill_rank", -1),
            decode_url=data.get("decode_url"),
            decode_rank=data.get("decode_rank", -1),
            error=data.get("error", ""),
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prefill_url": self.prefill_url,
            "prefill_rank": self.prefill_rank,
            "decode_url": self.decode_url,
            "decode_rank": self.decode_rank,
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineHealthResponse":
        return cls(
            request_id=data["request_id"],
            status=data.get("status", "unhealthy"),
            prefill_healthy=data.get("prefill_healthy", 0),
            prefill_total=data.get("prefill_total", 0),
            decode_healthy=data.get("decode_healthy", 0),
            decode_total=data.get("decode_total", 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "prefill_healthy": self.prefill_healthy,
            "prefill_total": self.prefill_total,
            "decode_healthy": self.decode_healthy,
            "decode_total": self.decode_total,
            "response_type": RequestType.HEALTH,
        }


@dataclass
class EngineWorkerUrlsResponse:
    request_id: str
    prefill_urls: List[str]
    decode_urls: List[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineWorkerUrlsResponse":
        return cls(
            request_id=data["request_id"],
            prefill_urls=list(data.get("prefill_urls") or []),
            decode_urls=list(data.get("decode_urls") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prefill_urls": self.prefill_urls,
            "decode_urls": self.decode_urls,
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
        self._health_refresh_task: Optional[asyncio.Task] = None
        self._background_tasks: set[asyncio.Task] = set()


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
                self._track_background_task(
                    asyncio.create_task(self._handle_health_request(engine_request))
                )

            elif engine_request.request_type == RequestType.WORKERS:
                workers_response = self.get_worker_urls_response(
                    request_id=engine_request.request_id
                )
                await self.send_response(engine_request, workers_response.to_dict())

    async def schedule_loop(self):
        while True:
            request = await self.waiting_queue.get()
            logger.debug(f"Processing prefill for request: {request.request_id}")
            # schedule
            prefill_worker = self.schedule_prefill(request.request_text, request.headers) 
            decode_worker = self.schedule_decode(request.request_text, request.headers)

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

    def _track_background_task(self, task: asyncio.Task) -> asyncio.Task:
        self._background_tasks.add(task)

        def _on_done(done_task: asyncio.Task) -> None:
            self._background_tasks.discard(done_task)
            if done_task.cancelled():
                return
            exc = done_task.exception()
            if exc is not None:
                logger.error(
                    "Background task failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )

        task.add_done_callback(_on_done)
        return task

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
        status = (
            "ok"
            if prefill_healthy > 0
            and prefill_total > 0
            and decode_healthy > 0
            and decode_total > 0
            else "unhealthy"
        )
        return EngineHealthResponse(
            request_id=request_id,
            status=status,
            prefill_healthy=prefill_healthy,
            prefill_total=prefill_total,
            decode_healthy=decode_healthy,
            decode_total=decode_total,
        )

    def get_worker_urls_response(self, request_id: str = "") -> EngineWorkerUrlsResponse:
        return EngineWorkerUrlsResponse(
            request_id=request_id,
            prefill_urls=self.worker_registry.get_base_urls_by_type(WorkerType.PREFILL),
            decode_urls=self.worker_registry.get_base_urls_by_type(WorkerType.DECODE),
        )

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

    def register_worker_group(
        self,
        base_url: str,
        worker_type: WorkerType,
        dp_size: int,
        initial_healthy: bool = True,
    ) -> None:
        if dp_size > 1:
            for rank in range(dp_size):
                worker = DPAwareWorker(
                    base_url,
                    worker_type,
                    self.config,
                    rank,
                    dp_size,
                )
                worker.set_healthy(initial_healthy)
                self.worker_registry.register(worker)
        else:
            worker = BasicWorker(base_url, worker_type, self.config)
            worker.set_healthy(initial_healthy)
            self.worker_registry.register(worker)

    def remove_worker_group(self, base_url: str, worker_type: WorkerType) -> None:
        removed = self.worker_registry.remove_by_base_url(worker_type, base_url)
        if removed:
            logger.info(
                "removed discovered workers: type=%s base_url=%s count=%s",
                worker_type,
                base_url,
                len(removed),
            )

    def _dp_size_for_type(self, worker_type: WorkerType) -> int:
        if worker_type == WorkerType.PREFILL:
            return self.config.prefill_intra_dp_size
        if worker_type == WorkerType.DECODE:
            return self.config.decode_intra_dp_size
        return 1

    async def _upsert_discovered_worker(self, discovered_worker) -> None:
        dp_size = self._dp_size_for_type(discovered_worker.worker_type)
        self.register_worker_group(
            discovered_worker.base_url,
            discovered_worker.worker_type,
            dp_size,
            initial_healthy=False,
        )
        logger.info(
            "registered discovered worker: pod=%s type=%s base_url=%s dp_size=%s",
            discovered_worker.pod_name,
            discovered_worker.worker_type,
            discovered_worker.base_url,
            dp_size,
        )
        self._request_worker_health_refresh()

    async def _delete_discovered_worker(self, discovered_worker) -> None:
        self.remove_worker_group(
            discovered_worker.base_url,
            discovered_worker.worker_type,
        )

    def _request_worker_health_refresh(self) -> None:
        task = self._health_refresh_task
        if task is not None and not task.done():
            return
        loop = asyncio.get_running_loop()
        self._health_refresh_task = self._track_background_task(
            loop.create_task(self._debounced_worker_health_refresh())
        )

    async def _debounced_worker_health_refresh(self) -> None:
        await asyncio.sleep(0.1)
        await self.refresh_worker_health()

    async def k8s_discovery_loop(self) -> None:
        from smart_router.discovery.k8s import K8sPodDiscovery

        discovery = K8sPodDiscovery.from_config(self.config)
        await discovery.run(
            on_upsert=self._upsert_discovered_worker,
            on_delete=self._delete_discovered_worker,
        )

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
        await self.refresh_worker_health()
        tasks = [
            self.receive_loop(),
            self.schedule_loop(),
            self.health_check_loop(),
        ]
        if getattr(getattr(self, "config", None), "enable_k8s_discovery", False):
            tasks.append(self.k8s_discovery_loop())
        await asyncio.gather(*tasks)

    async def shutdown(self):
        """Gracefully shutdown the engine:
        1. stopping receiving new requests;
        2. waiting for in-flight tasks to complete."""
        # stop receiving new requests
        self.input_socket.close(0)

        # closesocket
        self.output_socket.close(0)

    def schedule_prefill(self, request_text: str, headers: Dict[str, str]) -> Optional[Worker]:
        raise NotImplementedError
    
    def schedule_decode(self, request_text: str, headers: Dict[str, str]) -> Optional[Worker]:
        raise NotImplementedError
    
    
