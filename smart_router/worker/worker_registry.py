# -*- coding: utf-8 -*-
"""Worker registry and collection helper functions.

Provides centralized registry for workers with model-based indexing, enabling
efficient multi-router support and worker management.
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple
from smart_router.worker.core import WorkerType, Worker

logger = logging.getLogger()

# ==== Worker Registry =====


class WorkerRegistry:
    """Worker registry with model-based indexing for multi-router support."""

    def __init__(self) -> None:
        """Create a new worker registry."""
        self._workers: Dict[str, Worker] = {}
        self._type_workers: Dict[WorkerType, List[str]] = {}
        self._lock = threading.RLock()

    def register(self, worker: Worker) -> None:
        """Register a new worker and return its unique ID."""
        with self._lock:
            worker_id = worker.url()
            worker_type_key = worker.worker_type()
            old_worker = self._workers.get(worker_id)
            if old_worker is not None:
                old_type_key = old_worker.worker_type()
                if old_type_key == worker_type_key:
                    return

                if old_type_key in self._type_workers:
                    self._type_workers[old_type_key] = [
                        id for id in self._type_workers[old_type_key] if id != worker_id
                    ]
                    if not self._type_workers[old_type_key]:
                        del self._type_workers[old_type_key]

            # Store worker
            self._workers[worker_id] = worker

            # Update type index
            if worker_type_key not in self._type_workers:
                self._type_workers[worker_type_key] = []
            if worker_id not in self._type_workers[worker_type_key]:
                self._type_workers[worker_type_key].append(worker_id)

    def remove(self, worker_id: str) -> Optional[Worker]:
        """Remove a worker by ID."""
        with self._lock:
            if worker_id not in self._workers:
                return None

            worker = self._workers.pop(worker_id)

            # Remove from type index
            worker_type_key = worker.worker_type()
            if worker_type_key in self._type_workers:
                self._type_workers[worker_type_key] = [
                    id for id in self._type_workers[worker_type_key] if id != worker_id
                ]
                if not self._type_workers[worker_type_key]:
                    del self._type_workers[worker_type_key]

            return worker

    def remove_by_base_url(
        self, worker_type: WorkerType, base_url: str
    ) -> List[Worker]:
        """Remove all workers matching a type and base URL."""
        with self._lock:
            worker_ids = [
                worker_id
                for worker_id, worker in self._workers.items()
                if worker.worker_type() == worker_type and worker.base_url() == base_url
            ]
            removed = []
            for worker_id in worker_ids:
                worker = self.remove(worker_id)
                if worker is not None:
                    removed.append(worker)
            return removed

    def get(self, worker_id: str) -> Optional[Worker]:
        """Get a worker by ID."""
        with self._lock:
            return self._workers.get(worker_id)

    def get_by_type(self, worker_type: WorkerType) -> List[Worker]:
        """Get all workers by worker type."""
        with self._lock:
            worker_ids = self._type_workers.get(worker_type, [])
            return [self._workers[id] for id in worker_ids if id in self._workers]
    
    def get_healthy_by_type(self, worker_type: WorkerType) -> List[Worker]:
        with self._lock:
            worker_ids = self._type_workers.get(worker_type, [])
            return [
                self._workers[id]
                for id in worker_ids
                if id in self._workers and self._workers[id].is_healthy()
            ]

    def get_health_check_groups(
        self,
    ) -> List[Tuple[WorkerType, str, List[Worker]]]:
        """Group workers by type and base URL for one health check per DP instance."""
        with self._lock:
            grouped: Dict[Tuple[WorkerType, str], List[Worker]] = {}
            for worker in self._workers.values():
                key = (worker.worker_type(), worker.base_url())
                grouped.setdefault(key, []).append(worker)

            return [
                (worker_type, base_url, list(workers))
                for (worker_type, base_url), workers in grouped.items()
            ]

    def set_group_health(
        self, worker_type: WorkerType, base_url: str, healthy: bool
    ) -> None:
        """Apply one base-URL health result to all matching DP ranks."""
        with self._lock:
            for worker in self._workers.values():
                if worker.worker_type() == worker_type and worker.base_url() == base_url:
                    worker.set_healthy(healthy)

    def health_counts_by_type(self) -> Dict[WorkerType, Tuple[int, int]]:
        """Return healthy and total counts by unique worker base URL."""
        with self._lock:
            counts: Dict[WorkerType, Tuple[int, int]] = {}
            for worker_type, _base_url, workers in self.get_health_check_groups():
                healthy, total = counts.get(worker_type, (0, 0))
                if workers and workers[0].is_healthy():
                    healthy += 1
                total += 1
                counts[worker_type] = (healthy, total)
            return counts

    def get_all(self) -> List[Worker]:
        """Get all workers."""
        with self._lock:
            return list(self._workers.values())

    def get_all_with_ids(self) -> List[tuple[str, Worker]]:
        """Get all workers with their IDs."""
        with self._lock:
            return [(id, worker) for id, worker in self._workers.items()]

    def get_all_urls(self) -> List[str]:
        """Get all worker URLs."""
        with self._lock:
            return [worker.url() for worker in self._workers.values()]

    def get_base_urls_by_type(
        self, worker_type: WorkerType, healthy_only: bool = False
    ) -> List[str]:
        """Return unique base URLs for one worker type, preserving registry order."""
        with self._lock:
            urls: List[str] = []
            for worker in self.get_by_type(worker_type):
                if healthy_only and not worker.is_healthy():
                    continue
                base_url = worker.base_url()
                if base_url not in urls:
                    urls.append(base_url)
            return urls

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            total_workers = len(self._workers)

            healthy_count = 0
            total_load = 0
            regular_count = 0
            prefill_count = 0
            decode_count = 0

            for worker in self._workers.values():
                if worker.is_healthy():
                    healthy_count += 1
                total_load += worker.load()

                worker_type = worker.worker_type()
                if worker_type == WorkerType.REGULAR:
                    regular_count += 1
                elif worker_type == WorkerType.PREFILL:
                    prefill_count += 1
                elif worker_type ==  WorkerType.DECODE:
                    decode_count += 1

            return  (
            f"WorkerRegistryStats("
            f"total={total_workers}, "
            f"load={total_load}, "
            f"healthy={healthy_count},"
            f"regular={regular_count}, "
            f"prefill={prefill_count}, "
            f"decode={decode_count})"
            )


def get_healthy_workers(workers: List[Worker]) -> List[Worker]:
    """Helper function to filter healthy workers."""
    return [worker for worker in workers if worker.is_healthy()]

def get_available_workers(workers: List[Worker]) -> List[Worker]:
    """Helper function to filter available workers (healthy and circuit breaker allows)."""
    return [worker for worker in workers if worker.is_available()]
