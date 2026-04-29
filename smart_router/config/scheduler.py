from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    max_batch_size: int = 1
    batch_wait_timeout_ms: int = 0
    schedule_response_timeout_ms: int = 5000
    schedule_response_send_margin_ms: int = 1000
    adaptive_interval_enabled: bool = False
    stats_window_size: int = 16
    default_forward_time_ms: float = 100.0
    network_latency_ms: float = 0.0
    min_interval_ms: float = 0.0
    max_interval_ms: float = 1000.0
    watchdog_multiplier: float = 5.0
