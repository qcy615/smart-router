from __future__ import annotations

from typing import List

from smart_router.config import SmartRouterConfig
from smart_router.worker.basic_worker import BasicWorker
from smart_router.worker.core import Worker, WorkerType
from smart_router.worker.dp_aware_worker import DPAwareWorker
from smart_router.worker.worker_registry import WorkerRegistry


def build_workers_for_url(
    url: str,
    worker_type: WorkerType,
    config: SmartRouterConfig,
) -> List[Worker]:
    if worker_type == WorkerType.PREFILL:
        dp_size = config.prefill_worker_config.intra_dp_size
    elif worker_type == WorkerType.DECODE:
        dp_size = config.decode_worker_config.intra_dp_size
    else:
        dp_size = config.worker_config.intra_dp_size
    if dp_size > 1:
        return [
            DPAwareWorker(url, worker_type, config, rank, dp_size)
            for rank in range(dp_size)
        ]

    return [BasicWorker(url, worker_type, config)]


def register_workers_for_url(
    registry: WorkerRegistry,
    url: str,
    worker_type: WorkerType,
    config: SmartRouterConfig,
) -> List[str]:
    workers = build_workers_for_url(url, worker_type, config)
    worker_ids = []
    for worker in workers:
        registry.register(worker)
        worker_ids.append(worker.url())
    return worker_ids
