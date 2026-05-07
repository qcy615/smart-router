import logging
import threading
from typing import List, Optional
from smart_router.policies.policy import Policy
from smart_router.worker import Worker
from smart_router.config import PolicyConfig

logger = logging.getLogger(__name__)

class RoundRobinPolicy(Policy):
    """
    Round-robin selection policy
    """

    def __init__(self, config: PolicyConfig):
        self._counter = 0
        self._lock = threading.Lock()

    def name(self) -> str:
        return "round_robin"

    def reset(self):
        with self._lock:
            self._counter = 0

    def select_worker(
        self,
        workers: List[Worker],
        request_text: Optional[str] = None,
        headers: Optional[dict] = None,
        request_body: Optional[dict] = None,
        api_kind: Optional[str] = None,
    ) -> Optional[Worker]:
        _ = request_text
        _ = headers
        _ = request_body
        _ = api_kind

        if len(workers) == 0:
            return None

        with self._lock:
            count = self._counter
            self._counter += 1

        selected_idx = count % len(workers)

        logger.debug(f"[POLICY: {self.name()} ] select {workers[selected_idx]}")
        return workers[selected_idx]
