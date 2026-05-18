import asyncio
import logging
from typing import Any, Dict, List, Optional

from smart_router.engine.engine import Engine, EngineResponse
from smart_router.config import SmartRouterConfig
from smart_router.policies import Policy, get_policy_config
from smart_router.worker import BasicWorker, DPAwareWorker, Worker, WorkerRegistry, WorkerType
from smart_router.worker.factory import register_workers_for_url

logger = logging.getLogger(__name__)


class NormalEngine(Engine):
    """Non-PD-disaggregation scheduling engine.

    In normal mode, there is no prefill/decode separation. Workers are
    REGULAR type and each request is forwarded to a single worker chosen
    by the configured policy.
    """

    def __init__(
        self,
        config: SmartRouterConfig,
        input_socket_address: str,
        output_socket_address: str,
    ) -> None:
        super().__init__(
            input_socket_address=input_socket_address,
            output_socket_address=output_socket_address,
        )

        self.config: SmartRouterConfig = config
        self.worker_registry: WorkerRegistry = WorkerRegistry()
        self.policy: Policy = get_policy_config(config.policy_config)

        # Initialize regular workers.
        for url in config.worker_urls or []:
            register_workers_for_url(self.worker_registry, url, WorkerType.REGULAR, config)

        self.configure_worker_discovery(config)

        logger.info("registered workers: %s", self.worker_registry.get_all_urls())

    def schedule_prefill(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_token_ids: Optional[List[int]] = None,
        schedule_context: Optional[Dict[str, Any]] = None,
    ) -> Worker:
        """Not used in normal mode."""
        raise NotImplementedError("NormalEngine does not support schedule_prefill")

    def schedule_decode(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_token_ids: Optional[List[int]] = None,
        schedule_context: Optional[Dict[str, Any]] = None,
    ) -> Worker:
        """Not used in normal mode."""
        raise NotImplementedError("NormalEngine does not support schedule_decode")

    def schedule_worker(self, request_text: str, headers: Dict[str, str]) -> Optional[Worker]:
        """Schedule a single worker for normal (non-PD) mode."""
        workers = self.worker_registry.get_healthy_by_type(WorkerType.REGULAR)
        return self.policy.select_worker(workers, request_text=request_text, headers=headers)

    async def schedule_loop(self):
        while True:
            request = await self.waiting_queue.get()
            logger.debug(f"Processing normal schedule for request: {request.request_id}")
            worker = self.schedule_worker(request.request_text, request.headers)
            if worker is None:
                resp = EngineResponse(
                    request_id=request.request_id,
                    worker_url="",
                    worker_rank=-1,
                    prefill_url="",
                    prefill_rank=-1,
                    decode_url="",
                    decode_rank=-1,
                    error="No available workers",
                )
                await self.send_response(request, resp.to_dict())
                continue

            worker.increment_load()
            resp = EngineResponse(
                request_id=request.request_id,
                worker_url=worker.base_url(),
                worker_rank=worker.dp_rank(),
                prefill_url="",
                prefill_rank=-1,
                decode_url="",
                decode_rank=-1,
            )
            await self.send_response(request, resp.to_dict())


def start_normal_engine(config: SmartRouterConfig, input_addr: str, output_addr: str) -> None:
    engine = NormalEngine(
        config,
        input_socket_address=input_addr,
        output_socket_address=output_addr,
    )
    asyncio.run(engine.run())
