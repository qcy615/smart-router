from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Union

BlockHash = Union[int, bytes]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlockStoredEvent:
    block_hashes: list[BlockHash]
    parent_block_hash: Optional[BlockHash]
    token_ids: list[int]
    block_size: int
    lora_id: Optional[int]
    medium: Optional[str]


@dataclass(frozen=True)
class BlockRemovedEvent:
    block_hashes: list[BlockHash]
    medium: Optional[str]


@dataclass(frozen=True)
class AllBlocksClearedEvent:
    pass


KVCacheEvent = Union[BlockStoredEvent, BlockRemovedEvent, AllBlocksClearedEvent]


@dataclass(frozen=True)
class KVEventBatch:
    ts: float
    events: list[KVCacheEvent]
    data_parallel_rank: Optional[int] = None


@dataclass
class BlockRecord:
    block_hash: BlockHash
    parent_block_hash: Optional[BlockHash]
    token_ids: tuple[int, ...]
    block_size: int
    lora_id: Optional[int]
    medium: str
    updated_at: float


class KVCacheState:
    """Thread-safe mirror of KV block ownership reported by vLLM events."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._blocks: dict[BlockHash, BlockRecord] = {}
        self._worker_medium_blocks: dict[str, dict[str, set[BlockHash]]] = (
            defaultdict(lambda: defaultdict(set)))
        self._block_owners: dict[BlockHash, set[str]] = defaultdict(set)
        self._token_index: dict[
            tuple[str, Optional[int], Optional[BlockHash], int, tuple[int, ...]],
            set[BlockHash],
        ] = defaultdict(set)
        self._block_sizes: set[int] = set()

    def apply_batch(self, worker_id: str, batch: KVEventBatch) -> None:
        should_log = logger.isEnabledFor(logging.INFO)
        before_blocks = self.worker_block_count(worker_id) if should_log else 0
        stored_events = 0
        stored_blocks = 0
        removed_events = 0
        removed_blocks = 0
        cleared_events = 0
        for event in batch.events:
            if isinstance(event, BlockStoredEvent):
                stored_events += 1
                stored_blocks += len(event.block_hashes)
            elif isinstance(event, BlockRemovedEvent):
                removed_events += 1
                removed_blocks += len(event.block_hashes)
            elif isinstance(event, AllBlocksClearedEvent):
                cleared_events += 1
            self.apply_event(worker_id, event, batch.ts)
        if should_log:
            after_blocks = self.worker_block_count(worker_id)
            medium_counts = self.worker_medium_block_counts(worker_id)
            logger.info(
                "[KV-CACHE-APPLY] worker=%s dp_rank=%s events=%d "
                "stored_events=%d stored_blocks=%d removed_events=%d "
                "removed_blocks=%d cleared_events=%d blocks_before=%d "
                "blocks_after=%d medium_counts=%s",
                worker_id,
                batch.data_parallel_rank,
                len(batch.events),
                stored_events,
                stored_blocks,
                removed_events,
                removed_blocks,
                cleared_events,
                before_blocks,
                after_blocks,
                medium_counts,
            )

    def apply_event(self, worker_id: str, event: KVCacheEvent,
                    ts: float) -> None:
        if isinstance(event, BlockStoredEvent):
            self.store_blocks(worker_id, event, ts)
        elif isinstance(event, BlockRemovedEvent):
            self.remove_blocks(worker_id, event.block_hashes, event.medium)
        elif isinstance(event, AllBlocksClearedEvent):
            self.clear_worker(worker_id)

    def store_blocks(self, worker_id: str, event: BlockStoredEvent,
                     ts: float) -> None:
        medium = event.medium or "unknown"
        parent_hash = event.parent_block_hash
        with self._lock:
            self._block_sizes.add(event.block_size)
            for idx, block_hash in enumerate(event.block_hashes):
                start = idx * event.block_size
                end = start + event.block_size
                token_ids = tuple(event.token_ids[start:end])
                record_parent = parent_hash if idx == 0 else event.block_hashes[
                    idx - 1]

                record = BlockRecord(
                    block_hash=block_hash,
                    parent_block_hash=record_parent,
                    token_ids=token_ids,
                    block_size=event.block_size,
                    lora_id=event.lora_id,
                    medium=medium,
                    updated_at=ts,
                )
                self._blocks[block_hash] = record
                self._worker_medium_blocks[worker_id][medium].add(block_hash)
                self._block_owners[block_hash].add(worker_id)

                if len(token_ids) == event.block_size:
                    self._token_index[self._token_key(record)].add(block_hash)

    def remove_blocks(self,
                      worker_id: str,
                      block_hashes: Iterable[BlockHash],
                      medium: Optional[str] = None) -> None:
        with self._lock:
            media = [medium] if medium is not None else list(
                self._worker_medium_blocks[worker_id].keys())
            for block_hash in block_hashes:
                for medium_name in media:
                    self._worker_medium_blocks[worker_id][medium_name].discard(
                        block_hash)
                self._block_owners[block_hash].discard(worker_id)

    def clear_worker(self, worker_id: str) -> None:
        with self._lock:
            worker_blocks = self._worker_medium_blocks.pop(worker_id, {})
            for blocks in worker_blocks.values():
                for block_hash in blocks:
                    self._block_owners[block_hash].discard(worker_id)

    def count_matched_tokens(self,
                             worker_id: str,
                             token_ids: list[int],
                             medium: str = "GPU",
                             lora_id: Optional[int] = None,
                             block_size: Optional[int] = None) -> int:
        if not token_ids:
            return 0
        with self._lock:
            block_sizes = [block_size] if block_size else sorted(
                self._block_sizes, reverse=True)
            return max(
                (self._count_matched_tokens_locked(worker_id, token_ids, medium,
                                                   lora_id, size)
                 for size in block_sizes if size),
                default=0,
            )

    def best_workers_by_tokens(
        self,
        worker_ids: list[str],
        token_ids: list[int],
        medium: str = "GPU",
        lora_id: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> dict[str, int]:
        if not worker_ids:
            return {}
        if not token_ids:
            return {worker_id: 0 for worker_id in worker_ids}
        with self._lock:
            block_sizes = [block_size] if block_size else sorted(
                self._block_sizes, reverse=True)
            best = {worker_id: 0 for worker_id in worker_ids}
            for size in block_sizes:
                if not size:
                    continue
                scores = self._count_all_workers_locked(worker_ids, token_ids,
                                                        medium, lora_id, size)
                for worker_id, matched in scores.items():
                    if matched > best[worker_id]:
                        best[worker_id] = matched
            return best

    def worker_block_count(self,
                           worker_id: str,
                           medium: Optional[str] = None) -> int:
        with self._lock:
            if medium is not None:
                return len(self._worker_medium_blocks[worker_id][medium])
            return sum(
                len(blocks)
                for blocks in self._worker_medium_blocks[worker_id].values())

    def worker_medium_block_counts(self, worker_id: str) -> dict[str, int]:
        with self._lock:
            return {
                medium: len(blocks)
                for medium, blocks in self._worker_medium_blocks[worker_id].items()
            }

    def primary_block_size(self) -> Optional[int]:
        with self._lock:
            return max(self._block_sizes) if self._block_sizes else None

    def _count_all_workers_locked(
        self,
        worker_ids: list[str],
        token_ids: list[int],
        medium: str,
        lora_id: Optional[int],
        block_size: int,
    ) -> dict[str, int]:
        matched_by_worker = {worker_id: 0 for worker_id in worker_ids}
        remaining_workers = set(worker_ids)
        parent_candidates: dict[str, set[Optional[BlockHash]]] = {
            worker_id: {None}
            for worker_id in worker_ids
        }
        offset = 0

        while remaining_workers and offset + block_size <= len(token_ids):
            block_tokens = tuple(token_ids[offset:offset + block_size])
            next_candidates: dict[str, set[BlockHash]] = {
                worker_id: set()
                for worker_id in remaining_workers
            }

            for worker_id in list(remaining_workers):
                for parent_hash in parent_candidates[worker_id]:
                    key = (medium, lora_id, parent_hash, block_size, block_tokens)
                    for block_hash in self._token_index.get(key, set()):
                        if worker_id in self._block_owners.get(block_hash, set()):
                            next_candidates[worker_id].add(block_hash)

            for worker_id in list(remaining_workers):
                worker_next = next_candidates[worker_id]
                if not worker_next:
                    remaining_workers.remove(worker_id)
                    continue
                matched_by_worker[worker_id] += block_size
                parent_candidates[worker_id] = set(worker_next)

            offset += block_size

        return matched_by_worker

    def _count_matched_tokens_locked(self, worker_id: str, token_ids: list[int],
                                     medium: str, lora_id: Optional[int],
                                     block_size: int) -> int:
        matched = 0
        parent_candidates: set[Optional[BlockHash]] = {None}
        offset = 0

        while offset + block_size <= len(token_ids):
            block_tokens = tuple(token_ids[offset:offset + block_size])
            next_candidates: set[BlockHash] = set()
            for parent_hash in parent_candidates:
                key = (medium, lora_id, parent_hash, block_size, block_tokens)
                for block_hash in self._token_index.get(key, set()):
                    if worker_id in self._block_owners.get(block_hash, set()):
                        next_candidates.add(block_hash)

            if not next_candidates:
                break
            matched += block_size
            parent_candidates = set(next_candidates)
            offset += block_size

        return matched

    @staticmethod
    def _token_key(record: BlockRecord) -> tuple[str, Optional[int],
                                                Optional[BlockHash], int,
                                                tuple[int, ...]]:
        return (record.medium, record.lora_id, record.parent_block_hash,
                record.block_size, record.token_ids)


def normalize_batch(raw: Any) -> KVEventBatch:
    if isinstance(raw, dict):
        ts = float(raw.get("ts", 0.0))
        events = [_normalize_event(event) for event in raw.get("events", [])]
        dp_rank = raw.get("data_parallel_rank")
        return KVEventBatch(ts=ts, events=events, data_parallel_rank=dp_rank)

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        ts = float(raw[0])
        events = [_normalize_event(event) for event in raw[1]]
        dp_rank = raw[2] if len(raw) > 2 else None
        return KVEventBatch(ts=ts, events=events, data_parallel_rank=dp_rank)

    raise ValueError(f"Unsupported KV event batch payload: {raw!r}")


def _normalize_event(raw: Any) -> KVCacheEvent:
    if isinstance(raw, dict):
        tag = raw.get("type") or raw.get("__type__") or raw.get("tag")
        if tag == "BlockStored" or "token_ids" in raw:
            return BlockStoredEvent(
                block_hashes=list(raw.get("block_hashes", [])),
                parent_block_hash=raw.get("parent_block_hash"),
                token_ids=list(raw.get("token_ids", [])),
                block_size=int(raw.get("block_size", 0)),
                lora_id=raw.get("lora_id"),
                medium=raw.get("medium"),
            )
        if tag == "BlockRemoved" or "block_hashes" in raw:
            return BlockRemovedEvent(
                block_hashes=list(raw.get("block_hashes", [])),
                medium=raw.get("medium"),
            )
        return AllBlocksClearedEvent()

    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"Unsupported KV event payload: {raw!r}")

    values = list(raw)
    tag = values[0] if values and isinstance(values[0], str) else None
    fields = values[1:] if tag else values

    if tag == "BlockStored" or len(fields) == 6:
        return BlockStoredEvent(
            block_hashes=list(fields[0]),
            parent_block_hash=fields[1],
            token_ids=list(fields[2]),
            block_size=int(fields[3]),
            lora_id=fields[4],
            medium=fields[5],
        )
    if tag == "BlockRemoved" or len(fields) == 2:
        return BlockRemovedEvent(block_hashes=list(fields[0]),
                                 medium=fields[1])
    if tag == "AllBlocksCleared" or not fields:
        return AllBlocksClearedEvent()

    raise ValueError(f"Unsupported KV event payload: {raw!r}")
