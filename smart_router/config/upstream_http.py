from dataclasses import dataclass
from typing import Optional


@dataclass
class UpstreamHTTPClientConfig:
    connect_timeout_secs: float = 5.0
    read_timeout_secs: Optional[float] = None
    write_timeout_secs: float = 30.0
    pool_timeout_secs: float = 1.0
    max_connections: int = 1024
    max_keepalive_connections: int = 256
    keepalive_expiry_secs: float = 30.0
