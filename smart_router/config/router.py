from dataclasses import dataclass


@dataclass
class RouterModeConfig:
    router_type: str = "vllm"
    pd_disaggregation: bool = False
