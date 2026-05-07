from smart_router.config import PolicyConfig
from smart_router.kv.events import StoredKvEvent
from smart_router.kv.indexer import KvRadixIndexer, WorkerKey
from smart_router.policies.kv_aware import KvAwarePolicy


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        return [int(part) for part in text.split()]


class FakeWorker:
    def __init__(self, url: str, load: int = 0, rank: int = -1):
        self._url = url
        self._load = load
        self._rank = rank

    def url(self):
        return self._url if self._rank < 0 else f"{self._url}@{self._rank}"

    def base_url(self):
        return self._url

    def dp_rank(self):
        return self._rank

    def is_dp_aware(self):
        return self._rank >= 0

    def load(self):
        return self._load


def test_kv_aware_policy_prefers_more_overlap_when_weight_wins():
    indexer = KvRadixIndexer(expected_block_size=2)
    indexer.apply_stored(
        WorkerKey("http://a", -1),
        StoredKvEvent([b"a1"], None, [1, 2], 2),
    )
    indexer.apply_stored(
        WorkerKey("http://b", -1),
        StoredKvEvent([b"b1", b"b2"], None, [1, 2, 3, 4], 2),
    )

    policy = KvAwarePolicy(
        PolicyConfig(
            policy="kv_aware",
            kv_block_size=2,
            kv_overlap_weight=10.0,
            kv_load_weight=1.0,
        ),
        indexer=indexer,
        tokenizer=FakeTokenizer(),
    )

    selected = policy.select_worker(
        [FakeWorker("http://a", load=0), FakeWorker("http://b", load=2)],
        request_body={"prompt": "1 2 3 4"},
        api_kind="completions",
    )

    assert selected.base_url() == "http://b"


def test_kv_aware_policy_falls_back_to_least_load_without_full_blocks():
    policy = KvAwarePolicy(
        PolicyConfig(policy="kv_aware", kv_block_size=4),
        indexer=KvRadixIndexer(expected_block_size=4),
        tokenizer=FakeTokenizer(),
    )

    selected = policy.select_worker(
        [FakeWorker("http://busy", load=7), FakeWorker("http://idle", load=1)],
        request_body={"prompt": "1 2"},
        api_kind="completions",
    )

    assert selected.base_url() == "http://idle"
