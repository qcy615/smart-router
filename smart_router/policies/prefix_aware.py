import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

from smart_router.config import PolicyConfig
from smart_router.worker import Worker
from smart_router.policies.policy import Policy, PolicyRequest
from smart_router.policies.prefix_tree import PrefixTree

logger = logging.getLogger(__name__)


@dataclass
class _BatchRequestGroup:
    indexes: List[int] = field(default_factory=list)
    request_texts: List[str] = field(default_factory=list)
    common_prefix: str = ""
    max_text_len: int = 0


@dataclass
class _BatchMatchInfo:
    group: _BatchRequestGroup
    matched_tenants: List[str] = field(default_factory=list)
    match_rate: float = 0.0
    cache_hit: bool = False


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
    ) -> Optional[Worker]:
        _ = headers

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

    def select_worker_batch(
        self,
        workers: List[Worker],
        requests: List[PolicyRequest],
    ) -> List[Optional[Worker]]:
        if not workers:
            return [None for _ in requests]

        if not requests:
            return []

        worker_urls = [worker.url() for worker in workers]
        effective_loads = {worker.url(): worker.load() for worker in workers}
        selected_by_index: List[Optional[Worker]] = [None for _ in requests]

        groups = self._build_batch_request_groups(requests)
        match_infos = self._build_batch_match_infos(groups, worker_urls)

        match_infos.sort(
            key=lambda info: (
                not info.cache_hit,
                -info.match_rate,
                min(info.group.indexes),
            )
        )

        for info in match_infos:
            group_affinity_worker_urls: List[str] = []

            for index in sorted(info.group.indexes):
                selected_worker = self._select_worker_for_batch_group_request(
                    workers=workers,
                    effective_loads=effective_loads,
                    matched_tenants=info.matched_tenants,
                    group_affinity_worker_urls=group_affinity_worker_urls,
                )
                selected_by_index[index] = selected_worker

                if selected_worker is not None:
                    worker_url = selected_worker.url()
                    effective_loads[worker_url] += 1
                    if worker_url not in group_affinity_worker_urls:
                        group_affinity_worker_urls.append(worker_url)

        now = time.time()
        self._ensure_worker_tenants(
            [
                selected_worker.url()
                for selected_worker in selected_by_index
                if selected_worker is not None
            ],
            now,
        )
        self.tree.insert_many(
            [
                (request.request_text or "", selected_worker.url(), now)
                for request, selected_worker in zip(requests, selected_by_index)
                if selected_worker is not None
            ]
        )

        return selected_by_index

    def _build_batch_request_groups(
        self,
        requests: List[PolicyRequest],
    ) -> List[_BatchRequestGroup]:
        groups: List[_BatchRequestGroup] = []
        current_group: Optional[_BatchRequestGroup] = None
        indexed_requests = [
            (index, request.request_text or "")
            for index, request in enumerate(requests)
        ]

        for index, request_text in sorted(
            indexed_requests,
            key=lambda item: (item[1], item[0]),
        ):
            if not request_text:
                groups.append(self._new_batch_request_group(index, request_text))
                continue

            if current_group is None:
                current_group = self._new_batch_request_group(index, request_text)
                groups.append(current_group)
                continue

            candidate_prefix = os.path.commonprefix(
                [current_group.common_prefix, request_text]
            )
            candidate_max_text_len = max(current_group.max_text_len, len(request_text))

            if self._has_batch_affinity(candidate_prefix, candidate_max_text_len):
                current_group.indexes.append(index)
                current_group.request_texts.append(request_text)
                current_group.common_prefix = candidate_prefix
                current_group.max_text_len = candidate_max_text_len
            else:
                current_group = self._new_batch_request_group(index, request_text)
                groups.append(current_group)

        return groups

    def _new_batch_request_group(
        self,
        index: int,
        request_text: str,
    ) -> _BatchRequestGroup:
        return _BatchRequestGroup(
            indexes=[index],
            request_texts=[request_text],
            common_prefix=request_text,
            max_text_len=len(request_text),
        )

    def _has_batch_affinity(
        self,
        shared_prefix: str,
        max_text_len: int,
    ) -> bool:
        if max_text_len == 0:
            return False
        return len(shared_prefix) / max_text_len > self.config.cache_threshold

    def _build_batch_match_infos(
        self,
        groups: List[_BatchRequestGroup],
        worker_urls: List[str],
    ) -> List[_BatchMatchInfo]:
        match_infos: List[_BatchMatchInfo] = []

        for group in groups:
            matched_text = ""
            matched_tenants = None
            if group.common_prefix:
                matched_text, matched_tenants = self.tree.prefix_match(
                    group.common_prefix,
                    worker_urls,
                )

            match_rate = (
                sum(
                    len(matched_text) / len(request_text)
                    for request_text in group.request_texts
                    if request_text
                )
                / len(group.request_texts)
                if group.request_texts
                else 0.0
            )
            cache_hit = bool(
                group.common_prefix
                and match_rate > self.config.cache_threshold
                and matched_tenants
            )
            match_infos.append(
                _BatchMatchInfo(
                    group=group,
                    matched_tenants=(matched_tenants or []) if cache_hit else [],
                    match_rate=match_rate,
                    cache_hit=cache_hit,
                )
            )

        return match_infos

    def _select_worker_for_batch_group_request(
        self,
        workers: List[Worker],
        effective_loads: dict[str, int],
        matched_tenants: List[str],
        group_affinity_worker_urls: List[str],
    ) -> Worker:
        min_load = min(effective_loads.values())
        max_load = max(effective_loads.values())
        is_imbalanced = (
            (max_load - min_load) > self.config.balance_abs_threshold
            and max_load > min_load * self.config.balance_rel_threshold
        )

        if is_imbalanced:
            return self._select_min_effective_load(workers, effective_loads)

        candidate_url_set = set(matched_tenants)
        candidate_url_set.update(group_affinity_worker_urls)
        candidate_workers = [
            worker for worker in workers if worker.url() in candidate_url_set
        ]

        if candidate_workers:
            return self._select_min_effective_load(candidate_workers, effective_loads)

        return self._select_min_effective_load(workers, effective_loads)

    def _ensure_worker_tenants(self, worker_urls: List[str], time_s: float) -> None:
        new_worker_urls = [
            worker_url
            for worker_url in dict.fromkeys(worker_urls)
            if worker_url not in self.worker_urls
        ]
        if not new_worker_urls:
            return

        self.tree.add_tenants(new_worker_urls, time_s=time_s)
        self.worker_urls.update(new_worker_urls)

    def _select_min_load(self, workers: List[Worker]) -> Worker:
        min_load = min(worker.load() for worker in workers)
        candidates = [worker for worker in workers if worker.load() == min_load]
        return random.choice(candidates)

    def _select_min_effective_load(
        self,
        workers: List[Worker],
        effective_loads: dict[str, int],
    ) -> Worker:
        return min(workers, key=lambda worker: effective_loads[worker.url()])
