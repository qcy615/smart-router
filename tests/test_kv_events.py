import asyncio
from typing import Any

import msgspec
from smart_router.kv.events import (
    ClearedKvEvent,
    RemovedKvEvent,
    StoredKvEvent,
    encode_event_batch_for_tests,
    parse_event_batch,
)
from smart_router.kv.indexer import KvRadixIndexer, WorkerKey
from smart_router.kv.subscriber import KvEventSubscriber


def test_parser_accepts_real_vllm_event_tags():
    class EventBatch(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
        ts: float
        events: list[Any]
        data_parallel_rank: int | None = None

    class KVCacheEvent(
        msgspec.Struct,
        array_like=True,
        omit_defaults=True,
        gc=False,
        tag=True,
    ):
        pass

    class BlockStored(KVCacheEvent):
        block_hashes: list[bytes | int]
        parent_block_hash: bytes | int | None
        token_ids: list[int]
        block_size: int
        lora_id: int | None
        medium: str | None
        lora_name: str | None
        extra_keys: list[tuple[Any, ...] | None] | None = None
        group_idx: int | None = None

    class BlockRemoved(KVCacheEvent):
        block_hashes: list[bytes | int]
        medium: str | None
        group_idx: int | None = None

    class AllBlocksCleared(KVCacheEvent):
        pass

    class KVEventBatch(EventBatch):
        events: list[BlockStored | BlockRemoved | AllBlocksCleared]

    payload = msgspec.msgpack.encode(
        KVEventBatch(
            ts=0.0,
            events=[
                BlockStored([b"h1"], None, [1, 2], 2, None, "GPU", None),
                BlockRemoved([b"h1"], "GPU"),
                AllBlocksCleared(),
            ],
            data_parallel_rank=0,
        )
    )

    dp_rank, events = parse_event_batch(payload)

    assert dp_rank == 0
    assert isinstance(events[0], StoredKvEvent)
    assert isinstance(events[1], RemovedKvEvent)
    assert isinstance(events[2], ClearedKvEvent)


def test_vllm_kv_event_msgpack_round_trip():
    payload = encode_event_batch_for_tests(
        [
            StoredKvEvent(
                block_hashes=[b"h1"],
                parent_block_hash=None,
                token_ids=[1, 2],
                block_size=2,
                lora_id=None,
                medium="GPU",
                lora_name=None,
                extra_keys=[("adapter-a",)],
            ),
            RemovedKvEvent(block_hashes=[b"h1"], medium="GPU"),
            ClearedKvEvent(),
        ],
        data_parallel_rank=0,
    )

    dp_rank, events = parse_event_batch(payload)

    assert dp_rank == 0
    assert len(events) == 3
    assert isinstance(events[0], StoredKvEvent)
    assert isinstance(events[1], RemovedKvEvent)
    assert isinstance(events[2], ClearedKvEvent)


def test_subscriber_replays_gap_before_applying_live_event():
    indexer = KvRadixIndexer(expected_block_size=2)
    worker = WorkerKey("http://prefill", -1)
    subscriber = KvEventSubscriber(indexer, worker, "inproc://unused")

    first_payload = encode_event_batch_for_tests(
        [
            StoredKvEvent(
                block_hashes=[b"h1"],
                parent_block_hash=None,
                token_ids=[1, 2],
                block_size=2,
            )
        ],
        data_parallel_rank=0,
    )
    replay_payload = encode_event_batch_for_tests(
        [
            StoredKvEvent(
                block_hashes=[b"h2"],
                parent_block_hash=b"h1",
                token_ids=[3, 4],
                block_size=2,
            )
        ],
        data_parallel_rank=0,
    )
    live_payload = encode_event_batch_for_tests([ClearedKvEvent()], data_parallel_rank=0)

    async def fake_replay(start_seq, live_seq):
        assert (start_seq, live_seq) == (1, 2)
        subscriber._apply_payload(1, replay_payload)

    subscriber._replay_missing = fake_replay

    async def run_test():
        await subscriber._handle_live_frames([b"", (0).to_bytes(8, "big"), first_payload])
        await subscriber._handle_live_frames([b"", (2).to_bytes(8, "big"), live_payload])

    asyncio.run(run_test())

    assert subscriber.last_seq == 2
    assert not indexer.has_events()
