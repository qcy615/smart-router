import logging
from smart_router.policies.policy import Policy, PolicyConfig
from typing import Dict, Optional, List
from smart_router.worker import Worker


class MinimumLoadPolicy(Policy):
    """Policy that selects the worker with the minimum load."""
    
    def __init__(self, config: PolicyConfig):
        pass

    def name(self) -> str:
        return "minimum_load"
    
    def select_worker(
        self,
        workers: List[Worker],
        request_text: Optional[str] = None,
        headers: Optional[dict] = None,
        request_body: Optional[dict] = None,
        api_kind: Optional[str] = None,
        prompt_token_ids: Optional[list[int]] = None,
    ) -> Optional[Worker]:
        _ = request_text
        _ = headers
        _ = request_body
        _ = api_kind
        _ = prompt_token_ids

        if len(workers) == 0:
            return None

        min_load = float('inf')
        selected_worker = None

        for worker in workers:
            load = worker.load()
            if load < min_load:
                min_load = load
                selected_worker = worker

        return selected_worker   
