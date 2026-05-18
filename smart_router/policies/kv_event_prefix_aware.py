import logging
import random
from typing import List, Optional

from smart_router.cache import KVCacheState
from smart_router.config import PolicyConfig
from smart_router.policies.policy import Policy
from smart_router.worker import Worker

logger = logging.getLogger(__name__)


class KVEventPrefixAwarePolicy(Policy):
    """Prefix-aware policy backed by vLLM KV cache events.

    This policy does not update the legacy text PrefixTree. It only consumes the
    KVCacheState mirror populated from vLLM BlockStored/BlockRemoved events.
    """

    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.kv_cache_state: Optional[KVCacheState] = None

    def name(self) -> str:
        return "kv_event_prefix_aware"

    def set_kv_cache_state(self, state: KVCacheState) -> None:
        self.kv_cache_state = state

    def select_worker(
        self,
        workers: List[Worker],
        request_text: Optional[str] = None,
        headers: Optional[dict] = None,
        request_token_ids: Optional[list[int]] = None,
        kv_match_scores: Optional[dict[str, int]] = None,
    ) -> Optional[Worker]:
        _ = request_text
        _ = headers

        if not workers:
            return None

        loads = [worker.load() for worker in workers]
        min_load = min(loads)
        max_load = max(loads)
        is_imbalanced = (
            (max_load - min_load) > self.config.balance_abs_threshold
            and max_load > min_load * self.config.balance_rel_threshold
        )
        logger.info(
            "[KV-POLICY] policy=%s worker_count=%d token_count=%d "
            "loads=%s min_load=%d max_load=%d abs_threshold=%s "
            "rel_threshold=%s cache_threshold=%s imbalanced=%s",
            self.name(),
            len(workers),
            len(request_token_ids or []),
            {worker.url(): worker.load() for worker in workers},
            min_load,
            max_load,
            self.config.balance_abs_threshold,
            self.config.balance_rel_threshold,
            self.config.cache_threshold,
            is_imbalanced,
        )
        if is_imbalanced:
            selected = self._select_min_load(workers)
            logger.info(
                "[KV-POLICY] selected=%s reason=load_imbalanced load=%d",
                selected.url(),
                selected.load(),
            )
            return selected

        if self.kv_cache_state is None or not request_token_ids:
            selected = self._select_min_load(workers)
            logger.info(
                "[KV-CACHE-HIT] policy=%s token_count=%d reason=%s "
                "best_matched_tokens=0 best_match_rate=0.0000 "
                "match_rates=%s",
                self.name(),
                len(request_token_ids or []),
                "missing_kv_state"
                if self.kv_cache_state is None else "missing_request_token_ids",
                {worker.url(): 0.0 for worker in workers},
            )
            logger.info(
                "[KV-POLICY] selected=%s reason=%s load=%d",
                selected.url(),
                "missing_kv_state"
                if self.kv_cache_state is None else "missing_request_token_ids",
                selected.load(),
            )
            return selected

        worker_ids = [worker.url() for worker in workers]
        if kv_match_scores is None:
            scores = self.kv_cache_state.best_workers_by_tokens(
                worker_ids, request_token_ids)
        else:
            scores = {
                worker_id: kv_match_scores.get(worker_id, 0)
                for worker_id in worker_ids
            }
        best_score = max(scores.values(), default=0)
        token_count = len(request_token_ids)
        match_rates = {
            worker_id: round(scores.get(worker_id, 0) / token_count, 4)
            for worker_id in worker_ids
        }
        best_match_rate = best_score / token_count
        logger.info(
            "[KV-CACHE-HIT] policy=%s token_count=%d best_matched_tokens=%d "
            "best_match_rate=%.4f match_rates=%s matched_tokens=%s",
            self.name(),
            token_count,
            best_score,
            best_match_rate,
            match_rates,
            scores,
        )
        if best_score <= 0:
            selected = self._select_min_load(workers)
            logger.info(
                "[KV-POLICY] selected=%s reason=no_cache_match load=%d",
                selected.url(),
                selected.load(),
            )
            self._log_worker_stats(workers, scores, request_token_ids)
            return selected

        if best_match_rate > self.config.cache_threshold:
            candidates = [
                worker for worker in workers
                if scores.get(worker.url(), 0) == best_score
            ]
            selected = self._select_min_load(candidates)
            logger.info(
                "[KV-POLICY] selected=%s reason=kv_cache_match "
                "best_score=%d match_rate=%.4f candidates=%s",
                selected.url(),
                best_score,
                best_match_rate,
                [worker.url() for worker in candidates],
            )
            self._log_worker_stats(workers, scores, request_token_ids)
            return selected

        selected = self._select_min_load(workers)
        logger.info(
            "[KV-POLICY] selected=%s reason=cache_below_threshold "
            "best_score=%d match_rate=%.4f threshold=%s load=%d",
            selected.url(),
            best_score,
            best_match_rate,
            self.config.cache_threshold,
            selected.load(),
        )
        self._log_worker_stats(workers, scores, request_token_ids)
        return selected

    def _select_min_load(self, workers: List[Worker]) -> Worker:
        min_load = min(worker.load() for worker in workers)
        candidates = [worker for worker in workers if worker.load() == min_load]
        return random.choice(candidates)

    def _log_worker_stats(self, workers: List[Worker], scores: dict[str, int],
                          request_token_ids: list[int]) -> None:
        if not logger.isEnabledFor(logging.DEBUG) or self.kv_cache_state is None:
            return
        worker_stats = {}
        token_count = len(request_token_ids)
        for worker in workers:
            worker_id = worker.url()
            matched_tokens = scores.get(worker_id, 0)
            worker_stats[worker_id] = {
                "matched_tokens": matched_tokens,
                "match_rate": round(matched_tokens / token_count, 4)
                if token_count else 0.0,
                "load": worker.load(),
                "block_count": self.kv_cache_state.worker_block_count(worker_id),
                "medium_counts": self.kv_cache_state.worker_medium_block_counts(
                    worker_id),
            }
        logger.debug("[KV-POLICY] stats=%s", worker_stats)
