from smart_router.kv.events import ClearedKvEvent, RemovedKvEvent, StoredKvEvent
from smart_router.kv.hashing import block_hashes_for_tokens
from smart_router.kv.hashing import derive_request_extra_keys
from smart_router.kv.indexer import KvRadixIndexer, WorkerKey


def test_kv_indexer_tracks_stored_removed_and_cleared_blocks():
    indexer = KvRadixIndexer(expected_block_size=2)
    worker_a = WorkerKey("http://prefill-a", -1)
    worker_b = WorkerKey("http://prefill-b", -1)

    indexer.apply_stored(
        worker_a,
        StoredKvEvent(
            block_hashes=[b"a1", b"a2"],
            parent_block_hash=None,
            token_ids=[1, 2, 3, 4],
            block_size=2,
        ),
    )
    indexer.apply_stored(
        worker_b,
        StoredKvEvent(
            block_hashes=[b"b1"],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
        ),
    )

    request_hashes = block_hashes_for_tokens([1, 2, 3, 4], 2)
    assert indexer.find_matches(request_hashes) == {worker_a: 2, worker_b: 1}

    indexer.apply_removed(worker_a, RemovedKvEvent(block_hashes=[b"a2"]))
    assert indexer.find_matches(request_hashes) == {worker_a: 1, worker_b: 1}

    indexer.apply_cleared(worker_a)
    assert indexer.find_matches(request_hashes) == {worker_b: 1}


def test_kv_indexer_uses_external_parent_hash_for_later_blocks():
    indexer = KvRadixIndexer(expected_block_size=2)
    worker = WorkerKey("http://prefill", 1)

    indexer.apply_stored(
        worker,
        StoredKvEvent(
            block_hashes=[b"h1"],
            parent_block_hash=None,
            token_ids=[10, 11],
            block_size=2,
        ),
    )
    indexer.apply_stored(
        worker,
        StoredKvEvent(
            block_hashes=[b"h2"],
            parent_block_hash=b"h1",
            token_ids=[12, 13],
            block_size=2,
        ),
    )

    request_hashes = block_hashes_for_tokens([10, 11, 12, 13], 2)
    assert indexer.find_matches(request_hashes, {worker}) == {worker: 2}


def test_kv_indexer_skips_unknown_parent_to_avoid_false_prefix_match():
    indexer = KvRadixIndexer(expected_block_size=2)
    worker = WorkerKey("http://prefill", -1)

    indexer.apply_event(
        worker,
        StoredKvEvent(
            block_hashes=[b"h2"],
            parent_block_hash=b"missing",
            token_ids=[12, 13],
            block_size=2,
        ),
    )

    assert indexer.find_matches(block_hashes_for_tokens([12, 13], 2)) == {}
    indexer.apply_event(worker, ClearedKvEvent())


def test_request_extra_keys_match_lora_and_first_block_cache_salt_shape():
    assert derive_request_extra_keys(
        {"lora_name": "adapter-a", "cache_salt": "tenant-a"},
        block_count=2,
    ) == [("adapter-a", "tenant-a"), ("adapter-a",)]
