import logging
import asyncio
import zmq
import zmq.asyncio
from dataclasses import dataclass, field
from typing import Any, Dict
import json
import platform

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
    request_body: Dict[str, Any] = field(default_factory=dict)
    api_kind: str = field(default="")
    prompt_token_ids: list[int] = field(default_factory=list)
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
            request_body=data.get("request_body", {}),
            api_kind=data.get("api_kind", ""),
            prompt_token_ids=[int(token_id) for token_id in data.get("prompt_token_ids", [])],
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
            "request_body": self.request_body,
            "api_kind": self.api_kind,
            "prompt_token_ids": self.prompt_token_ids,
            "worker_url": self.worker_url,
            "worker_rank": self.worker_rank,
        }
    

@dataclass
class EngineResponse:
    request_id: str
    prefill_url: str
    prefill_rank: int
    decode_url: str
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
    ) -> None:
        # Initialize ZeroMQ context and sockets
        ctx = zmq.Context()
        self.async_ctx = zmq.asyncio.Context(ctx)
        self.input_socket: zmq.Socket = make_zmq_socket(self.async_ctx, input_socket_address, zmq.PULL)
        self.output_socket: zmq.Socket = make_zmq_socket(self.async_ctx, output_socket_address, zmq.ROUTER)

        # queues for scheduling
        self.waiting_queue: asyncio.Queue[EngineRequest] = asyncio.Queue()

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
            request = await self.waiting_queue.get()
            logger.debug(f"Processing prefill for request: {request.request_id}")
            # schedule
            prefill_worker = self.schedule_prefill(
                request.request_text,
                request.headers,
                request.request_body,
                request.api_kind,
                request.prompt_token_ids,
            )
            prefill_worker.increment_load()    
            decode_worker = self.schedule_decode(
                request.request_text,
                request.headers,
                request.request_body,
                request.api_kind,
                request.prompt_token_ids,
            )
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
        if type(self) is Engine:
            logger.warning("Base Engine.run() called without a concrete scheduler; returning")
            return

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
        request_body: Dict[str, Any] | None = None,
        api_kind: str | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> Worker:
        raise NotImplementedError
    
    def schedule_decode(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_body: Dict[str, Any] | None = None,
        api_kind: str | None = None,
        prompt_token_ids: list[int] | None = None,
    ) -> Worker:
        raise NotImplementedError
    
    
