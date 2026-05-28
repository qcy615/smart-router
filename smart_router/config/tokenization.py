from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenizationConfig:
    tokenizer: Optional[str] = None
    tokenizer_trust_remote_code: bool = False
    tokenize_url: Optional[str] = None
    tokenize_timeout: float = 10.0
    tokenize_cache_size: int = 4096
    tokenize_cache_ttl: float = 3600.0
