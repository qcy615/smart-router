from dataclasses import dataclass, field
from smart_router.config.worker import HealthConfig
from smart_router.config.policy import PolicyConfig
from typing import List, Optional
from argparse import Namespace

@dataclass
class SmartRouterConfig:
    router_type: str = "vllm-pd-disagg"

    prefill_urls: List[str] = None
    prefill_intra_dp_size: int = 1
    prefill_bootstrap_ports: List[int] = None

    decode_urls: List[str] = None
    decode_intra_dp_size: int = 1

    health_config: HealthConfig = field(default_factory=HealthConfig)

    prefill_policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    decode_policy_config: PolicyConfig = field(default_factory=PolicyConfig)

    enable_k8s_discovery: bool = False
    prefill_port: Optional[int] = None
    decode_port: Optional[int] = None
    k8s_task_label_key: str = "task_id"
    k8s_namespace: Optional[str] = None


def _validate_config_args(args: Namespace) -> None:
    if args.prefill_intra_dp_size < 1:
        raise ValueError("--prefill-intra-dp-size must be greater than 0")
    if args.decode_intra_dp_size < 1:
        raise ValueError("--decode-intra-dp-size must be greater than 0")

    if not getattr(args, "enable_k8s_discovery", False):
        return

    if args.prefill_urls or args.decode_urls:
        raise ValueError(
            "--enable-k8s-discovery cannot be used with "
            "--prefill-urls or --decode-urls"
        )

    if args.prefill_port is None:
        raise ValueError("--prefill-port is required when --enable-k8s-discovery is set")
    if args.decode_port is None:
        raise ValueError("--decode-port is required when --enable-k8s-discovery is set")
    if args.prefill_port < 1 or args.prefill_port > 65535:
        raise ValueError("--prefill-port must be between 1 and 65535")
    if args.decode_port < 1 or args.decode_port > 65535:
        raise ValueError("--decode-port must be between 1 and 65535")
    if not args.k8s_task_label_key:
        raise ValueError("--k8s-task-label-key cannot be empty")


def build_config(args: Namespace) -> SmartRouterConfig:
    """
    Build smart router config from args.
    """
    _validate_config_args(args)

    decode_policy_config = None
    if args.decode_policy != "":
        decode_policy_config: PolicyConfig = PolicyConfig(
            policy=args.decode_policy,
            cache_threshold=args.decode_cache_threshold,
            balance_abs_threshold=args.decode_balance_abs_threshold,
            balance_rel_threshold=args.decode_balance_rel_threshold,
        )

    prefill_policy_config = None
    if args.prefill_policy != "":
        prefill_policy_config: PolicyConfig = PolicyConfig(
            policy=args.prefill_policy,
            cache_threshold=args.prefill_cache_threshold,
            balance_abs_threshold=args.prefill_balance_abs_threshold,
            balance_rel_threshold=args.prefill_balance_rel_threshold,
        )

    # default policy config
    policy_config = PolicyConfig(
        policy=args.policy,
        cache_threshold=args.cache_threshold,
        balance_abs_threshold=args.balance_abs_threshold,
        balance_rel_threshold=args.balance_rel_threshold,
    )
   
    return SmartRouterConfig(
        router_type=args.router_type,
        prefill_urls=args.prefill_urls,
        prefill_intra_dp_size=args.prefill_intra_dp_size,
        prefill_bootstrap_ports=getattr(args, "prefill_bootstrap_ports", None),
        decode_urls=args.decode_urls, 
        decode_intra_dp_size=args.decode_intra_dp_size,
        health_config=HealthConfig(
            check_interval_secs=getattr(args, "health_check_interval", 60),
        ),
        decode_policy_config=decode_policy_config if decode_policy_config else policy_config,
        prefill_policy_config=prefill_policy_config if prefill_policy_config else policy_config,
        enable_k8s_discovery=getattr(args, "enable_k8s_discovery", False),
        prefill_port=getattr(args, "prefill_port", None),
        decode_port=getattr(args, "decode_port", None),
        k8s_task_label_key=getattr(args, "k8s_task_label_key", "task_id"),
        k8s_namespace=getattr(args, "k8s_namespace", None),
    )
