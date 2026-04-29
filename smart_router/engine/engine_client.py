import logging
import asyncio
import os
import zmq
import zmq.asyncio
import json
import platform
from typing import Optional

from smart_router.engine.utils import make_zmq_socket
from smart_router.engine.engine import EngineRequest, EngineResponse, RequestType

is_linux = platform.system() == "Linux"
if is_linux:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


logger = logging.getLogger(__name__)

class EngineClient:
    def __init__(
        self, 
        input_socket_address: str,
        output_socket_address: str,
    ) -> None:  
        self.identity = f"worker-{os.getpid()}"
        # Initialize ZeroMQ context and sockets
        ctx = zmq.Context()
        self.async_ctx = zmq.asyncio.Context(ctx)
        self.input_socket = make_zmq_socket(self.async_ctx, input_socket_address, zmq.PUSH)
        self.output_socket = make_zmq_socket(self.async_ctx, output_socket_address, zmq.DEALER, identity=self.identity.encode("utf-8"))

        # pending: request_id -> response_queue
        self.pending: dict[str, asyncio.Future] = {}

        logger.info(f"Initialized EgineClient with identity {self.identity} successfully")

    async def send_request(self, request: EngineRequest) -> Optional[asyncio.Future]:
        fut = None
        if request.request_type == RequestType.SCHEDULE:
            loop = asyncio.get_running_loop()
            fut = loop.create_future()

            self.pending[request.request_id] = fut
        await self.input_socket.send_json(request.to_dict())
        
        return fut

    async def receive_loop(self):
        while True:
            frames = await self.output_socket.recv_multipart()
            engine_resp = EngineResponse.from_dict(json.loads(frames[-1]))
            
            request_id = engine_resp.request_id
            if request_id in self.pending:
                fut = self.pending.pop(request_id, None)
                
                if fut:
                    if not fut.done():
                        fut.set_result(engine_resp)
                    else:
                        logger.warning(
                            "Received late schedule response request_id=%s identity=%s",
                            request_id,
                            self.identity,
                        )

                logger.debug(f"Received response: {engine_resp.request_id}")
            else:
                logger.warning(f"{request_id} not in {self.identity}'s pending!")

    async def shutdown(self):
        logger.info(f"Shutdown EgineClient with identity {self.identity} ...")
        self.input_socket.close()
        self.output_socket.close()
        



