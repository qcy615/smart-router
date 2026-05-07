import asyncio
import logging
from typing import Dict, Optional

from smart_router.engine.engine import Engine
from smart_router.config import SmartRouterConfig
from smart_router.policies import Policy, get_policy_config
from smart_router.worker import BasicWorker, DPAwareWorker, Worker, WorkerRegistry, WorkerType
from smart_router.engine.engine import Engine

DECODE_URL_PLACEHOLDER = "DECODE_URL_PLACEHOLDER"

logger = logging.getLogger(__name__)

class VLLMEngine(Engine):
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
        self.prefill_policy: Policy = get_policy_config(config.prefill_policy_config)
        self.decode_policy: Policy = get_policy_config(config.decode_policy_config)

        # Initialize prefill workers.
        for url in config.prefill_urls:
            if config.prefill_intra_dp_size > 1:
                for rank in range(config.prefill_intra_dp_size):
                    worker = DPAwareWorker(
                        url,
                        WorkerType.PREFILL,
                        config,
                        rank,
                        config.prefill_intra_dp_size,
                    )
                    self.worker_registry.register(worker)
            else:
                worker = BasicWorker(url, WorkerType.PREFILL, config)
                self.worker_registry.register(worker)

        # Initialize decode workers.
        for url in config.decode_urls:
            if config.decode_intra_dp_size > 1:
                for rank in range(config.decode_intra_dp_size):
                    worker = DPAwareWorker(
                        url,
                        WorkerType.DECODE,
                        config,
                        rank,
                        config.decode_intra_dp_size,
                    )
                    self.worker_registry.register(worker)
            else:
                worker = BasicWorker(url, WorkerType.DECODE, config)
                self.worker_registry.register(worker)

        logger.info("registered workers: %s", self.worker_registry.get_all_urls())

    async def run(self):
        self._start_policy(self.prefill_policy)
        self._start_policy(self.decode_policy)
        try:
            await super().run()
        finally:
            await self._stop_policy(self.prefill_policy)
            await self._stop_policy(self.decode_policy)

    def _start_policy(self, policy: Policy) -> None:
        start = getattr(policy, "start", None)
        if callable(start):
            start()

    async def _stop_policy(self, policy: Policy) -> None:
        stop = getattr(policy, "stop", None)
        if not callable(stop):
            return
        result = stop()
        if asyncio.iscoroutine(result):
            await result

    
    def schedule_prefill(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_body: Dict[str, object] | None = None,
        api_kind: str | None = None,
    ) -> Worker:
        workers = self.worker_registry.get_healthy_by_type(WorkerType.PREFILL)
        prefill: Optional[Worker] = self.prefill_policy.select_worker(
            workers,
            request_text=request_text,
            headers=headers,
            request_body=request_body,
            api_kind=api_kind,
        )
        return prefill
    
    def schedule_decode(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_body: Dict[str, object] | None = None,
        api_kind: str | None = None,
    ) -> Worker:
        workers = self.worker_registry.get_healthy_by_type(WorkerType.DECODE)
        decode: Optional[Worker] = self.decode_policy.select_worker(
            workers,
            request_text=request_text,
            headers=headers,
            request_body=request_body,
            api_kind=api_kind,
        )
        return decode
    
def start_engine(config: SmartRouterConfig, input_addr: str, output_addr: str) -> None:
    engine = VLLMEngine(
        config,
        input_socket_address=input_addr,
        output_socket_address=output_addr
    )
    asyncio.run(engine.run())
