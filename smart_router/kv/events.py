from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:
    import msgspec
except ModuleNotFoundError:  # pragma: no cover - exercised only without runtime dep.
    msgspec = None  # type: ignore[assignment]


@dataclass(frozen=True)
class StoredKvEvent:
    block_hashes: list[Any]
    parent_block_hash: Any | None
    token_ids: list[int]
    block_size: int
    lora_id: int | None = None
    medium: str | None = None
    lora_name: str | None = None
    extra_keys: list[Any] | None = None
    group_idx: int | None = None


@dataclass(frozen=True)
class RemovedKvEvent:
    block_hashes: list[Any]
    medium: str | None = None
    group_idx: int | None = None


@dataclass(frozen=True)
class ClearedKvEvent:
    pass


NormalizedKvEvent = StoredKvEvent | RemovedKvEvent | ClearedKvEvent


if msgspec is not None:

    class _EventBatch(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False,  # type: ignore[call-arg]
    ):
        ts: float
        events: list[Any]
        data_parallel_rank: int | None = None


    class _KVCacheEvent(
        msgspec.Struct,
        array_like=True,  # type: ignore[call-arg]
        omit_defaults=True,  # type: ignore[call-arg]
        gc=False,  # type: ignore[call-arg]
        tag=True,
    ):
        pass


    class _BlockStored(_KVCacheEvent, tag="BlockStored"):  # type: ignore[call-arg]
        block_hashes: list[bytes | int]
        parent_block_hash: bytes | int | None
        token_ids: list[int]
        block_size: int
        lora_id: int | None
        medium: str | None
        lora_name: str | None = None
        extra_keys: list[tuple[Any, ...] | None] | None = None
        group_idx: int | None = None


    class _BlockRemoved(_KVCacheEvent, tag="BlockRemoved"):  # type: ignore[call-arg]
        block_hashes: list[bytes | int]
        medium: str | None
        group_idx: int | None = None


    class _AllBlocksCleared(_KVCacheEvent, tag="AllBlocksCleared"):  # type: ignore[call-arg]
        pass


    class _KVEventBatch(_EventBatch):
        events: list[_BlockStored | _BlockRemoved | _AllBlocksCleared]


def parse_event_batch(payload: bytes) -> tuple[int | None, list[NormalizedKvEvent]]:
    if msgspec is None:
        raise RuntimeError(
            "msgspec is required to decode vLLM KV event payloads. "
            "Install smart-router with runtime dependencies."
        )

    batch = msgspec.msgpack.decode(payload, type=_KVEventBatch)
    events: list[NormalizedKvEvent] = []
    for event in batch.events:
        if isinstance(event, _BlockStored):
            events.append(
                StoredKvEvent(
                    block_hashes=list(event.block_hashes),
                    parent_block_hash=event.parent_block_hash,
                    token_ids=list(event.token_ids),
                    block_size=event.block_size,
                    lora_id=event.lora_id,
                    medium=event.medium,
                    lora_name=event.lora_name,
                    extra_keys=list(event.extra_keys) if event.extra_keys else None,
                    group_idx=event.group_idx,
                )
            )
        elif isinstance(event, _BlockRemoved):
            events.append(
                RemovedKvEvent(
                    block_hashes=list(event.block_hashes),
                    medium=event.medium,
                    group_idx=event.group_idx,
                )
            )
        elif isinstance(event, _AllBlocksCleared):
            events.append(ClearedKvEvent())
    return batch.data_parallel_rank, events


def encode_event_batch_for_tests(
    events: list[NormalizedKvEvent],
    data_parallel_rank: int | None = None,
    ts: float = 0.0,
) -> bytes:
    if msgspec is None:
        raise RuntimeError("msgspec is required to encode test KV event payloads.")

    raw_events: list[Any] = []
    for event in events:
        if isinstance(event, StoredKvEvent):
            raw_events.append(
                _BlockStored(
                    event.block_hashes,
                    event.parent_block_hash,
                    event.token_ids,
                    event.block_size,
                    event.lora_id,
                    event.medium,
                    event.lora_name,
                    event.extra_keys,
                    event.group_idx,
                )
            )
        elif isinstance(event, RemovedKvEvent):
            raw_events.append(
                _BlockRemoved(event.block_hashes, event.medium, event.group_idx)
            )
        elif isinstance(event, ClearedKvEvent):
            raw_events.append(_AllBlocksCleared())

    return msgspec.msgpack.encode(
        _KVEventBatch(ts=ts, events=raw_events, data_parallel_rank=data_parallel_rank)
    )
