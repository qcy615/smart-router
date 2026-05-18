import asyncio
import logging
from typing import Any, Dict, List, Optional

from smart_router.engine.engine import Engine
from smart_router.config import SmartRouterConfig
from smart_router.policies import Policy, get_policy_config
from smart_router.worker import Worker, WorkerRegistry, WorkerType
from smart_router.worker.factory import register_workers_for_url

logger = logging.getLogger(__name__)


class SglangEngine(Engine):
    """SGLang PD-disaggregation scheduling engine.

    Unlike VLLMEngine, SGLang uses bootstrap_host/bootstrap_port/bootstrap_room
    for KV cache transfer between prefill and decode workers. The engine only
    needs to schedule workers; bootstrap info is injected at the route layer.
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
        self.prefill_policy: Policy = get_policy_config(config.prefill_policy_config)
        self.decode_policy: Policy = get_policy_config(config.decode_policy_config)

        # Initialize prefill workers.
        for url in config.prefill_urls or []:
            register_workers_for_url(self.worker_registry, url, WorkerType.PREFILL, config)

        # Initialize decode workers.
        for url in config.decode_urls or []:
            register_workers_for_url(self.worker_registry, url, WorkerType.DECODE, config)

        self.configure_worker_discovery(config)

        logger.info("registered workers: %s", self.worker_registry.get_all_urls())

    def schedule_prefill(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_token_ids: Optional[List[int]] = None,
        schedule_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Worker]:
        _ = request_token_ids
        _ = schedule_context
        workers = self.worker_registry.get_healthy_by_type(WorkerType.PREFILL)
        prefill: Optional[Worker] = self.prefill_policy.select_worker(
            workers, request_text=request_text, headers=headers
        )
        return prefill

    def schedule_decode(
        self,
        request_text: str,
        headers: Dict[str, str],
        request_token_ids: Optional[List[int]] = None,
        schedule_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Worker]:
        _ = request_token_ids
        _ = schedule_context
        workers = self.worker_registry.get_healthy_by_type(WorkerType.DECODE)
        decode: Optional[Worker] = self.decode_policy.select_worker(
            workers, request_text=request_text, headers=headers
        )
        return decode


def start_sglang_engine(config: SmartRouterConfig, input_addr: str, output_addr: str) -> None:
    engine = SglangEngine(
        config,
        input_socket_address=input_addr,
        output_socket_address=output_addr
    )
    asyncio.run(engine.run())
