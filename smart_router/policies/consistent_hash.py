import bisect
import struct
import threading
from typing import List, Optional

from smart_router.config import PolicyConfig
from smart_router.worker import Worker
from smart_router.policies.policy import Policy

VIRTUAL_NODES_PER_WORKER = 160


class ConsistentHashPolicy(Policy):
    def __init__(self, config: PolicyConfig):
        self.config = config
        self.hash_ring = {}
        self.sorted_keys = []
        self.current_worker_keys = ()
        self.lock = threading.Lock()

    def name(self) -> str:
        return "consistent_hash"

    @staticmethod
    def murmur_hash_64a(key: bytes, seed: int):
        M = 0xC6A4A7935BD1E995
        R = 47

        length = len(key)
        h = (seed & 0xFFFFFFFFFFFFFFFF) ^ ((length * M) & 0xFFFFFFFFFFFFFFFF)

        nblocks = length // 8

        for i in range(nblocks):
            k = struct.unpack_from("<Q", key, i * 8)[0]

            k = (k * M) & 0xFFFFFFFFFFFFFFFF
            k ^= k >> R
            k = (k * M) & 0xFFFFFFFFFFFFFFFF

            h ^= k
            h = (h * M) & 0xFFFFFFFFFFFFFFFF

        remaining = key[nblocks * 8 :]

        if remaining:
            rem = int.from_bytes(remaining, "little")
            h ^= rem
            h = (h * M) & 0xFFFFFFFFFFFFFFFF

        h ^= h >> R
        h = (h * M) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> R

        return h & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def murmur_rehash_64a(k: int):
        M = 0xC6A4A7935BD1E995
        R = 47
        SEED = 4193360111

        h = SEED ^ (8 * M)

        k = (k * M) & 0xFFFFFFFFFFFFFFFF
        k ^= k >> R
        k = (k * M) & 0xFFFFFFFFFFFFFFFF

        h ^= k
        h = (h * M) & 0xFFFFFFFFFFFFFFFF

        h ^= h >> R
        h = (h * M) & 0xFFFFFFFFFFFFFFFF
        h ^= h >> R

        return h & 0xFFFFFFFFFFFFFFFF

    def furc_get_bit(self, key_bytes, idx, cache, old_ord):
        SEED = 4193360111

        ord_ = idx >> 6

        if old_ord[0] < ord_:
            for n in range(old_ord[0] + 1, ord_ + 1):
                if n == 0:
                    hv = self.murmur_hash_64a(key_bytes, SEED)
                else:
                    hv = self.murmur_rehash_64a(cache[n - 1])

                if len(cache) <= n:
                    cache.append(hv)
                else:
                    cache[n] = hv

            old_ord[0] = ord_

        hash_val = cache[ord_]
        bit_pos = idx & 0x3F

        return (hash_val >> bit_pos) & 1

    def furc_hash(self, key: str, m: int):
        MAX_TRIES = 32
        FURC_SHIFT = 23

        if m <= 1:
            return 0

        key_bytes = key.encode()
        cache = []
        old_ord = [-1]

        d = 0
        while m > (1 << d):
            d += 1

        a = d

        for _ in range(MAX_TRIES):
            while not self.furc_get_bit(key_bytes, a, cache, old_ord):
                if d == 0:
                    return 0
                d -= 1
                a = d

            a += FURC_SHIFT
            num = 1

            for _ in range(max(0, d - 1)):
                bit = self.furc_get_bit(key_bytes, a, cache, old_ord)
                num = (num << 1) | bit
                a += FURC_SHIFT

            if num < m:
                return num

        return 0

    def fbi_hash(self, key: str):
        LARGE_MODULUS = (1 << 23) - 1
        furc_result = self.furc_hash(key, LARGE_MODULUS)
        return self.murmur_hash_64a(
            furc_result.to_bytes(4, "little"),
            4193360111,
        )

    def update_hash_ring(self, workers: List[Worker]) -> None:
        worker_keys = tuple(worker.url() for worker in workers)

        with self.lock:
            if worker_keys == self.current_worker_keys:
                return

            ring = {}
            for worker in workers:
                worker_key = worker.url()
                for i in range(VIRTUAL_NODES_PER_WORKER):
                    virtual_key = f"{worker_key}:{i}"
                    h = self.fbi_hash(virtual_key)
                    ring[h] = worker

            self.hash_ring = ring
            self.sorted_keys = sorted(ring.keys())
            self.current_worker_keys = worker_keys

    def select_worker(
        self,
        workers: List[Worker],
        request_text: Optional[str] = None,
        headers: Optional[dict] = None,
        request_body: Optional[dict] = None,
        api_kind: Optional[str] = None,
    ) -> Optional[Worker]:
        _ = headers
        _ = request_body
        _ = api_kind

        if not workers:
            return None

        if len(workers) == 1:
            return workers[0]

        self.update_hash_ring(workers)
        key = request_text or ""
        h = self.fbi_hash(key)

        with self.lock:
            if not self.hash_ring:
                return None
            sorted_keys = self.sorted_keys
            hash_ring = self.hash_ring

        idx = bisect.bisect_left(sorted_keys, h)
        if idx == len(sorted_keys):
            idx = 0

        return hash_ring[sorted_keys[idx]]
