from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    max_batch_size: int = 1
    batch_wait_timeout_ms: int = 0
