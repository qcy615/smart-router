from dataclasses import dataclass
from typing import Optional


@dataclass
class K8SDiscoveryConfig:
    enabled: bool = False
    prefill_port: Optional[int] = None
    decode_port: Optional[int] = None
    regular_port: Optional[int] = None
    namespace: Optional[str] = None
    task_label_key: str = "task_id"
    task_id: Optional[str] = None
    url_scheme: str = "http"
