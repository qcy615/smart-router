from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from smart_router.kv.events import ClearedKvEvent, RemovedKvEvent, StoredKvEvent
from smart_router.kv.hashing import (
    block_hashes_for_tokens,
    normalize_external_hash,
    split_full_blocks,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerKey:
    url: str
    rank: int


class _RadixNode:
    def __init__(self, block_hash: bytes | None = None, parent: Optional["_RadixNode"] = None):
        self.block_hash = block_hash
        self.parent = parent
        self.children: Dict[bytes, _RadixNode] = {}
        self.owners: set[WorkerKey] = set()


class KvRadixIndexer:
    def __init__(self, expected_block_size: int = 16) -> None:
        self.expected_block_size = expected_block_size
        self._root = _RadixNode()
        self._worker_lookup: dict[WorkerKey, dict[tuple[Any, int | None], _RadixNode]] = {}
        self._lock = threading.RLock()

    def apply_stored(self, worker: WorkerKey, event: StoredKvEvent) -> None:
        if event.block_size != self.expected_block_size:
            logger.warning(
                "KV event block_size=%s differs from configured kv_block_size=%s",
                event.block_size,
                self.expected_block_size,
            )

        full_tokens = split_full_blocks(
            event.token_ids,
            event.block_size,
            max_blocks=len(event.block_hashes),
        )
        if len(full_tokens) < len(event.block_hashes) * event.block_size:
            logger.debug("Skipping incomplete KV stored event for worker=%s", worker)
            return

        with self._lock:
            parent_node = self._root
            if event.parent_block_hash is not None:
                parent_key = normalize_external_hash(event.parent_block_hash, event.group_idx)
                parent_node = self._worker_lookup.get(worker, {}).get(parent_key)
                if parent_node is None:
                    logger.debug("Skipping KV event with unknown parent for worker=%s", worker)
                    return

            lookup = self._worker_lookup.setdefault(worker, {})
            parent_local_hash = parent_node.block_hash
            local_hashes = block_hashes_for_tokens(
                full_tokens,
                event.block_size,
                event.extra_keys,
                parent_hash=parent_local_hash,
            )

            node = parent_node
            for local_hash, external_hash in zip(local_hashes, event.block_hashes):
                child = node.children.get(local_hash)
                if child is None:
                    child = _RadixNode(local_hash, parent=node)
                    node.children[local_hash] = child
                child.owners.add(worker)
                lookup[normalize_external_hash(external_hash, event.group_idx)] = child
                node = child

    def apply_removed(self, worker: WorkerKey, event: RemovedKvEvent) -> None:
        with self._lock:
            lookup = self._worker_lookup.get(worker)
            if not lookup:
                return

            for external_hash in event.block_hashes:
                node = lookup.pop(normalize_external_hash(external_hash, event.group_idx), None)
                if node is None:
                    continue
                node.owners.discard(worker)
                self._prune_empty_ancestors(node)

            if not lookup:
                self._worker_lookup.pop(worker, None)

    def apply_cleared(self, worker: WorkerKey) -> None:
        with self._lock:
            lookup = self._worker_lookup.pop(worker, {})
            for node in list(lookup.values()):
                node.owners.discard(worker)
                self._prune_empty_ancestors(node)

    def apply_event(self, worker: WorkerKey, event: StoredKvEvent | RemovedKvEvent | ClearedKvEvent) -> None:
        if isinstance(event, StoredKvEvent):
            self.apply_stored(worker, event)
        elif isinstance(event, RemovedKvEvent):
            self.apply_removed(worker, event)
        elif isinstance(event, ClearedKvEvent):
            self.apply_cleared(worker)

    def find_matches(
        self,
        local_hashes: list[bytes],
        workers: set[WorkerKey] | None = None,
    ) -> dict[WorkerKey, int]:
        with self._lock:
            node = self._root
            active: set[WorkerKey] | None = None
            scores: dict[WorkerKey, int] = {}

            for depth, local_hash in enumerate(local_hashes, start=1):
                node = node.children.get(local_hash)
                if node is None:
                    break

                node_owners = set(node.owners)
                if workers is not None:
                    node_owners &= workers
                active = node_owners if active is None else active & node_owners
                if not active:
                    break

                for worker in active:
                    scores[worker] = depth

            return scores

    def has_events(self) -> bool:
        with self._lock:
            return any(lookup for lookup in self._worker_lookup.values())

    def _prune_empty_ancestors(self, node: _RadixNode) -> None:
        while node.parent is not None and not node.owners and not node.children:
            parent = node.parent
            if node.block_hash is not None:
                parent.children.pop(node.block_hash, None)
            node = parent
