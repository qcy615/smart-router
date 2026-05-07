import logging
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PolicyConfig:

    policy: str = "round_robin"  # default policy

    cache_threshold: float = 0.5

    balance_abs_threshold: int = 5

    balance_rel_threshold: float = 2.0

    kv_tokenizer_path: Optional[str] = None

    kv_block_size: int = 16

    kv_worker_urls: List[str] = field(default_factory=list)

    kv_intra_dp_size: int = 1

    kv_event_endpoints: List[str] = field(default_factory=list)

    kv_replay_endpoints: List[str] = field(default_factory=list)

    kv_overlap_weight: float = 1.0

    kv_load_weight: float = 1.0

    kv_event_topic: str = ""
