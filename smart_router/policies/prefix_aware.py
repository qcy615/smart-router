import logging
import random
import time
from typing import List, Optional

from smart_router.config import PolicyConfig
from smart_router.worker import Worker
from smart_router.policies.policy import Policy
from smart_router.policies.prefix_tree import PrefixTree

logger = logging.getLogger(__name__)


class PrefixAwarePolicy(Policy):
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()

        # only one tree
        self.tree = PrefixTree()

        #
        self.worker_urls = set()

    def name(self) -> str:
        return "prefix_aware"

    def select_worker(
        self,
        workers: List[Worker],
        request_text: Optional[str] = None,
        headers: Optional[dict] = None,
        request_body: Optional[dict] = None,
        api_kind: Optional[str] = None,
        prompt_token_ids: Optional[list[int]] = None,
    ) -> Optional[Worker]:
        _ = headers
        _ = request_body
        _ = api_kind
        _ = prompt_token_ids

        if not workers:
            return None

        request_text = request_text or ""
        loads = [worker.load() for worker in workers]
        min_load = min(loads)
        max_load = max(loads)

        # Check whether the system is load-imbalanced.
        is_imbalanced = (
            (max_load - min_load) > self.config.balance_abs_threshold
            and max_load > min_load * self.config.balance_rel_threshold
        )

        if is_imbalanced:
            selected_worker = self._select_min_load(workers)
            self._insert_tree(request_text, selected_worker.url())
            return selected_worker

        if not request_text:
            return self._select_min_load(workers)

        # Prefix-aware routing
        worker_urls = [worker.url() for worker in workers]
        matched_text, matched_tenants = self.tree.prefix_match(request_text, worker_urls)

        # count match rate
        match_rate = len(matched_text) / len(request_text)

        # prefix cache hit
        if match_rate > self.config.cache_threshold and matched_tenants:
            candidate_workers = [
                worker for worker in workers if worker.url() in matched_tenants
            ]
            if candidate_workers:
                selected_worker = self._select_min_load(candidate_workers)
                self._insert_tree(request_text, selected_worker.url())
                return selected_worker

        # cache miss -> least load
        selected_worker = self._select_min_load(workers)
        self._insert_tree(request_text, selected_worker.url())
        return selected_worker

    def _insert_tree(self, request_text: str, worker_url: str) -> None:
        if worker_url not in self.worker_urls:
            self.tree.add_tenants([worker_url], time_s=time.time())
            self.worker_urls.add(worker_url)

        self.tree.insert(request_text, worker_url, time.time())

    def _select_min_load(self, workers: List[Worker]) -> Worker:
        min_load = min(worker.load() for worker in workers)
        candidates = [worker for worker in workers if worker.load() == min_load]
        return random.choice(candidates)
