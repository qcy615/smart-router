from dataclasses import dataclass, field
from smart_router.config.worker import HealthConfig, CircuitBreakerConfig
from smart_router.config.policy import PolicyConfig
from typing import List
from argparse import Namespace

@dataclass
class SmartRouterConfig:
    router_type: str = "vllm-pd-disagg"

    prefill_urls: List[str] = None
    prefill_intra_dp_size: int = 1

    decode_urls: List[str] = None
    decode_intra_dp_size: int = 1

    health_config: HealthConfig = field(default_factory=HealthConfig)

    ciruit_breaker_config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    prefill_policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    decode_policy_config: PolicyConfig = field(default_factory=PolicyConfig)

def build_config(args: Namespace) -> SmartRouterConfig:
    """
    Build smart router config from args.
    """
    def make_policy_config(
        policy: str,
        cache_threshold: float,
        balance_abs_threshold: int,
        balance_rel_threshold: float,
        worker_urls: List[str] | None,
        intra_dp_size: int,
        kv_event_endpoints: List[str] | None = None,
        kv_replay_endpoints: List[str] | None = None,
    ) -> PolicyConfig:
        return PolicyConfig(
            policy=policy,
            cache_threshold=cache_threshold,
            balance_abs_threshold=balance_abs_threshold,
            balance_rel_threshold=balance_rel_threshold,
            kv_tokenizer_path=getattr(args, "kv_tokenizer_path", None),
            kv_block_size=getattr(args, "kv_block_size", 16),
            kv_worker_urls=worker_urls or [],
            kv_intra_dp_size=intra_dp_size,
            kv_event_endpoints=kv_event_endpoints or [],
            kv_replay_endpoints=kv_replay_endpoints or [],
            kv_overlap_weight=getattr(args, "kv_overlap_weight", 1.0),
            kv_load_weight=getattr(args, "kv_load_weight", 1.0),
            kv_event_topic=getattr(args, "kv_event_topic", ""),
        )

    decode_policy_config = None
    if args.decode_policy != "":
        decode_policy_config = make_policy_config(
            args.decode_policy,
            args.decode_cache_threshold,
            args.decode_balance_abs_threshold,
            args.decode_balance_rel_threshold,
            args.decode_urls,
            args.decode_intra_dp_size,
        )

    prefill_policy_config = None
    if args.prefill_policy != "":
        prefill_policy_config = make_policy_config(
            args.prefill_policy,
            args.prefill_cache_threshold,
            args.prefill_balance_abs_threshold,
            args.prefill_balance_rel_threshold,
            args.prefill_urls,
            args.prefill_intra_dp_size,
            getattr(args, "prefill_kv_event_endpoints", []),
            getattr(args, "prefill_kv_replay_endpoints", []),
        )

    # default policy config
    prefill_default_policy_config = make_policy_config(
        args.policy,
        args.cache_threshold,
        args.balance_abs_threshold,
        args.balance_rel_threshold,
        args.prefill_urls,
        args.prefill_intra_dp_size,
        getattr(args, "prefill_kv_event_endpoints", []),
        getattr(args, "prefill_kv_replay_endpoints", []),
    )
    decode_default_policy_config = make_policy_config(
        args.policy,
        args.cache_threshold,
        args.balance_abs_threshold,
        args.balance_rel_threshold,
        args.decode_urls,
        args.decode_intra_dp_size,
    )
   
    return SmartRouterConfig(
        router_type=args.router_type,
        prefill_urls=args.prefill_urls,
        prefill_intra_dp_size=args.prefill_intra_dp_size,
        decode_urls=args.decode_urls, 
        decode_intra_dp_size=args.decode_intra_dp_size,
        decode_policy_config=decode_policy_config if decode_policy_config else decode_default_policy_config,
        prefill_policy_config=prefill_policy_config if prefill_policy_config else prefill_default_policy_config,
    )
