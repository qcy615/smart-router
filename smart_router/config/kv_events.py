from dataclasses import dataclass
from typing import List, Optional


@dataclass
class KVEventsConfig:
    enabled: bool = False
    port: int = 5557
    topic: str = ""
    endpoints: Optional[List[str]] = None
