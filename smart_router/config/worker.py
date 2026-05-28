from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HealthConfig:
    timeout_secs: int = 5
    check_interval_secs: int = 60
    endpoint: str = "/health"


@dataclass
class WorkerGroupConfig:
    urls: Optional[List[str]] = None
    intra_dp_size: int = 1
    bootstrap_ports: Optional[List[int]] = None
