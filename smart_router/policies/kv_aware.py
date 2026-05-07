from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, List, Optional

from smart_router.config import PolicyConfig
from smart_router.kv.hashing import (
    block_hashes_for_tokens,
    derive_request_extra_keys,
    split_full_blocks,
)
from smart_router.kv.indexer import KvRadixIndexer, WorkerKey
from smart_router.kv.subscriber import KvEventSubscriberGroup, build_prefill_subscribers
from smart_router.kv.tokenizer import RequestTokenizer, TokenizerLike
from smart_router.policies.policy import Policy
from smart_router.worker import Worker

logger = logging.getLogger(__name__)


class KvAwarePolicy(Policy):
    def __init__(
        self,
        config: PolicyConfig,
        indexer: KvRadixIndexer | None = None,
        tokenizer: TokenizerLike | None = None,
        subscriber_group: KvEventSubscriberGroup | None = None,
    ) -> None:
        self.config = config
        self.indexer = indexer or KvRadixIndexer(expected_block_size=config.kv_block_size)
        self._subscriber_group = subscriber_group
        self._started = False

        if tokenizer is not None:
            self.tokenizer: RequestTokenizer | None = RequestTokenizer("", tokenizer=tokenizer)
        elif config.kv_tokenizer_path:
            self.tokenizer = RequestTokenizer(config.kv_tokenizer_path)
        else:
            self.tokenizer = None
            logger.warning("kv_aware policy has no tokenizer path; it will fall back to least load")

        if self._subscriber_group is None:
            self._subscriber_group = build_prefill_subscribers(
                indexer=self.indexer,
                worker_urls=config.kv_worker_urls,
                intra_dp_size=config.kv_intra_dp_size,
                event_endpoints=config.kv_event_endpoints,
                replay_endpoints=config.kv_replay_endpoints,
                topic=config.kv_event_topic,
            )

    def name(self) -> str:
        return "kv_aware"

    def start(self) -> None:
        if self._started:
            return
        self._subscriber_group.start()
        self._started = True

    async def stop(self) -> None:
        if self._started:
            await self._subscriber_group.stop()
            self._started = False

    def select_worker(
        self,
        workers: List[Worker],
        request_text: Optional[str] = None,
        headers: Optional[dict] = None,
        request_body: Optional[dict[str, Any]] = None,
        api_kind: Optional[str] = None,
    ) -> Optional[Worker]:
        _ = request_text
        _ = headers

        if not workers:
            return None
        if len(workers) == 1:
            return workers[0]

        local_hashes = self._request_block_hashes(request_body, api_kind)
        if not local_hashes or not self.indexer.has_events():
            return self._select_min_load(workers)

        candidate_keys = {self._worker_key(worker) for worker in workers}
        overlap_scores = self.indexer.find_matches(local_hashes, candidate_keys)
        block_count = len(local_hashes)

        best_score: float | None = None
        best_workers: list[Worker] = []
        for worker in workers:
            worker_key = self._worker_key(worker)
            overlap_blocks = overlap_scores.get(worker_key, 0)
            miss_blocks = max(block_count - overlap_blocks, 0)
            score = (
                self.config.kv_overlap_weight * miss_blocks
                + self.config.kv_load_weight * worker.load()
            )
            if best_score is None or score < best_score:
                best_score = score
                best_workers = [worker]
            elif score == best_score:
                best_workers.append(worker)

        selected = random.choice(best_workers)
        logger.debug(
            "[POLICY: %s] selected=%s blocks=%s overlap=%s score=%s",
            self.name(),
            selected.url(),
            block_count,
            overlap_scores.get(self._worker_key(selected), 0),
            best_score,
        )
        return selected

    def _request_block_hashes(
        self,
        request_body: dict[str, Any] | None,
        api_kind: str | None,
    ) -> list[bytes] | None:
        if self.tokenizer is None:
            return None

        token_ids = self.tokenizer.tokenize_request(request_body, api_kind)
        if not token_ids:
            return None

        full_tokens = split_full_blocks(token_ids, self.config.kv_block_size)
        if not full_tokens:
            return None

        block_count = len(full_tokens) // self.config.kv_block_size
        extra_keys = derive_request_extra_keys(request_body, block_count)
        return block_hashes_for_tokens(
            full_tokens,
            self.config.kv_block_size,
            extra_keys,
        )

    def _select_min_load(self, workers: List[Worker]) -> Worker:
        min_load = min(worker.load() for worker in workers)
        candidates = [worker for worker in workers if worker.load() == min_load]
        return random.choice(candidates)

    def _worker_key(self, worker: Worker) -> WorkerKey:
        return WorkerKey(worker.base_url(), worker.dp_rank() if worker.is_dp_aware() else -1)
