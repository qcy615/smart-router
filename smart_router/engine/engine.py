import logging
import asyncio
import zmq
import zmq.asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import platform

from smart_router.config import SchedulerConfig
from smart_router.engine.utils import make_zmq_socket
from smart_router.worker import Worker, WorkerRegistry

logger = logging.getLogger(__name__)

is_linux = platform.system() == "Linux"
if is_linux:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("enable uvloop event loop poicy!")


class RequestType:
    SCHEDULE = "schedule"
    RELEASE = "release"


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
            request_text=data["request_text"],
            headers=data["headers"],
            worker_url=data["worker_url"],
            worker_rank=data["worker_rank"],
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineResponse":
        return cls(
            request_id=data["request_id"],
            prefill_url=data["prefill_url"],
            prefill_rank=data["prefill_rank"],
            decode_url=data["decode_url"],
            decode_rank=data["decode_rank"],
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prefill_url": self.prefill_url,
            "prefill_rank": self.prefill_rank,
            "decode_url": self.decode_url,
            "decode_rank": self.decode_rank,
        }
    


class Engine:
    def __init__(
        self,
        input_socket_address: str,
        output_socket_address: str,
        scheduler_config: Optional[SchedulerConfig] = None,
    ) -> None:
        # Initialize ZeroMQ context and sockets
        ctx = zmq.Context()
        self.async_ctx = zmq.asyncio.Context(ctx)
        self.input_socket: zmq.Socket = make_zmq_socket(self.async_ctx, input_socket_address, zmq.PULL)
        self.output_socket: zmq.Socket = make_zmq_socket(self.async_ctx, output_socket_address, zmq.ROUTER)

        # queues for scheduling
        self.waiting_queue: asyncio.Queue[EngineRequest] = asyncio.Queue()
        self.scheduler_config = scheduler_config or SchedulerConfig()

        self.worker_registry: WorkerRegistry = WorkerRegistry()


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
                worker.decrement_load()

                logger.debug(f"Received release request: {engine_request.request_id}, worker id: {worker_id}")

    async def schedule_loop(self):
        while True:
            batch = await self._get_next_batch()
            await self._schedule_batch(batch)

    async def _schedule_batch(self, batch: List[EngineRequest]) -> None:
        if not batch:
            return

        logger.debug(
            "Processing schedule batch size=%s request_ids=%s",
            len(batch),
            [request.request_id for request in batch],
        )

        prefill_workers = self.schedule_prefill_batch(batch)
        decode_workers = self.schedule_decode_batch(batch)
        workers_to_increment: List[Optional[Worker]] = []
        response_pairs: List[tuple[EngineRequest, EngineResponse]] = []

        for index, request in enumerate(batch):
            prefill_worker = (
                prefill_workers[index] if index < len(prefill_workers) else None
            )
            decode_worker = decode_workers[index] if index < len(decode_workers) else None

            # Scheduling is all-or-nothing for each request: only commit worker
            # loads when both prefill and decode assignments are available.
            if prefill_worker is not None and decode_worker is not None:
                workers_to_increment.extend([prefill_worker, decode_worker])
                resp = EngineResponse(
                    request_id=request.request_id,
                    prefill_url=self._worker_base_url(prefill_worker),
                    prefill_rank=self._worker_dp_rank(prefill_worker),
                    decode_url=self._worker_base_url(decode_worker),
                    decode_rank=self._worker_dp_rank(decode_worker),
                )
            else:
                logger.warning(
                    "Unable to schedule request_id=%s prefill_assigned=%s decode_assigned=%s",
                    request.request_id,
                    prefill_worker is not None,
                    decode_worker is not None,
                )
                resp = EngineResponse(
                    request_id=request.request_id,
                    prefill_url=None,
                    prefill_rank=-1,
                    decode_url=None,
                    decode_rank=-1,
                )

            response_pairs.append((request, resp))

        self._increment_worker_loads(workers_to_increment)

        for request, resp in response_pairs:
            await self.send_response(request, resp.to_dict())
            logger.debug(
                "Sent schedule response request_id=%s prefill=%s decode=%s",
                request.request_id,
                resp.prefill_url,
                resp.decode_url,
            )

    async def _get_next_batch(self) -> List[EngineRequest]:
        max_batch_size = max(1, self.scheduler_config.max_batch_size)
        timeout_secs = max(0, self.scheduler_config.batch_wait_timeout_ms) / 1000

        first_request = await self.waiting_queue.get()
        batch = [first_request]
        if max_batch_size == 1:
            return batch

        deadline = asyncio.get_running_loop().time() + timeout_secs
        while len(batch) < max_batch_size:
            try:
                batch.append(self.waiting_queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break

            try:
                request = await asyncio.wait_for(
                    self.waiting_queue.get(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break
            batch.append(request)

        return batch

    def _increment_worker_loads(self, workers: List[Optional[Worker]]) -> None:
        load_by_worker: Dict[Worker, int] = {}
        for worker in workers:
            if worker is None:
                continue
            load_by_worker[worker] = load_by_worker.get(worker, 0) + 1

        for worker, load in load_by_worker.items():
            worker.increment_load(load)

    def _worker_base_url(self, worker: Optional[Worker]) -> Optional[str]:
        if worker is None:
            return None
        return worker.base_url()

    def _worker_dp_rank(self, worker: Optional[Worker]) -> int:
        if worker is None:
            return -1
        return worker.dp_rank()

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
        await asyncio.gather(
            self.receive_loop(),
            self.schedule_loop(),
        )

    async def shutdown(self):
        """Gracefully shutdown the engine:
        1. stopping receiving new requests;
        2. waiting for in-flight tasks to complete."""
        # stop receiving new requests
        self.input_socket.close(0)

        # closesocket
        self.output_socket.close(0)

    def schedule_prefill(
        self,
        request_text: str,
        headers: Dict[str, str],
    ) -> Optional[Worker]:
        raise NotImplementedError
    
    def schedule_decode(
        self,
        request_text: str,
        headers: Dict[str, str],
    ) -> Optional[Worker]:
        raise NotImplementedError

    def schedule_prefill_batch(
        self,
        requests: List[EngineRequest],
    ) -> List[Optional[Worker]]:
        return [
            self.schedule_prefill(request.request_text, request.headers)
            for request in requests
        ]

    def schedule_decode_batch(
        self,
        requests: List[EngineRequest],
    ) -> List[Optional[Worker]]:
        return [
            self.schedule_decode(request.request_text, request.headers)
            for request in requests
        ]
    
    
