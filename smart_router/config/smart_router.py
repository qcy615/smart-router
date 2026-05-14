from dataclasses import dataclass, field
from smart_router.config.worker import HealthConfig
from smart_router.config.policy import PolicyConfig
from typing import List, Optional
from argparse import Namespace


@dataclass
class K8SDiscoveryConfig:
    enabled: bool = False
    prefill_port: Optional[int] = None
    decode_port: Optional[int] = None
    namespace: Optional[str] = None
    task_label_key: str = "task_id"
    task_id: Optional[str] = None
    url_scheme: str = "http"

@dataclass
class SmartRouterConfig:
    router_type: str = "vllm"
    pd_disaggregation: bool = False  # Enable PD disaggregated mode

    worker_urls: List[str] = None
    worker_intra_dp_size: int = 1
    policy_config: PolicyConfig = field(default_factory=PolicyConfig)

    prefill_urls: List[str] = None
    prefill_intra_dp_size: int = 1
    prefill_bootstrap_ports: List[int] = None

    decode_urls: List[str] = None
    decode_intra_dp_size: int = 1

    health_config: HealthConfig = field(default_factory=HealthConfig)
    k8s_discovery_config: K8SDiscoveryConfig = field(default_factory=K8SDiscoveryConfig)

    prefill_policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    decode_policy_config: PolicyConfig = field(default_factory=PolicyConfig)

    kv_events_enabled: bool = False
    kv_events_port: int = 5557
    kv_events_topic: str = ""
    kv_events_endpoints: Optional[List[str]] = None

    tokenizer: Optional[str] = None
    tokenizer_trust_remote_code: bool = False
    tokenize_url: Optional[str] = None
    tokenize_timeout: float = 10.0
    tokenize_cache_size: int = 4096
    tokenize_cache_ttl: float = 3600.0


def _validate_k8s_discovery_config(config: K8SDiscoveryConfig) -> None:
    if not config.enabled:
        return

    missing = []
    if config.prefill_port is None:
        missing.append("--k8s-prefill-port")
    if config.decode_port is None:
        missing.append("--k8s-decode-port")
    if missing:
        raise RuntimeError(
            "K8S discovery requires " + " and ".join(missing)
        )

    for name, port in (
        ("--k8s-prefill-port", config.prefill_port),
        ("--k8s-decode-port", config.decode_port),
    ):
        if port is None or port <= 0 or port > 65535:
            raise RuntimeError(f"{name} must be between 1 and 65535")

def build_config(args: Namespace) -> SmartRouterConfig:
    """
    Build smart router config from args.
    """
    decode_policy_config = None
    if args.decode_policy != "":
        decode_policy_config: PolicyConfig = PolicyConfig(
            policy=args.decode_policy,
            cache_threshold=args.decode_cache_threshold,
            balance_abs_threshold=args.decode_balance_abs_threshold,
            balance_rel_threshold=args.decode_balance_rel_threshold,
            prefix_cache_eviction_threshold_chars=args.decode_prefix_cache_eviction_threshold_chars,
            prefix_cache_eviction_target_chars=args.decode_prefix_cache_eviction_target_chars,
            prefix_cache_eviction_interval_secs=args.decode_prefix_cache_eviction_interval_secs,
        )

    prefill_policy_config = None
    if args.prefill_policy != "":
        prefill_policy_config: PolicyConfig = PolicyConfig(
            policy=args.prefill_policy,
            cache_threshold=args.prefill_cache_threshold,
            balance_abs_threshold=args.prefill_balance_abs_threshold,
            balance_rel_threshold=args.prefill_balance_rel_threshold,
            prefix_cache_eviction_threshold_chars=args.prefill_prefix_cache_eviction_threshold_chars,
            prefix_cache_eviction_target_chars=args.prefill_prefix_cache_eviction_target_chars,
            prefix_cache_eviction_interval_secs=args.prefill_prefix_cache_eviction_interval_secs,
        )

    # default policy config
    policy_config = PolicyConfig(
        policy=args.policy,
        cache_threshold=args.cache_threshold,
        balance_abs_threshold=args.balance_abs_threshold,
        balance_rel_threshold=args.balance_rel_threshold,
        prefix_cache_eviction_threshold_chars=args.prefix_cache_eviction_threshold_chars,
        prefix_cache_eviction_target_chars=args.prefix_cache_eviction_target_chars,
        prefix_cache_eviction_interval_secs=args.prefix_cache_eviction_interval_secs,
    )
   
    k8s_discovery_config = K8SDiscoveryConfig(
        enabled=getattr(args, "enable_k8s_discovery", False),
        prefill_port=getattr(args, "k8s_prefill_port", None),
        decode_port=getattr(args, "k8s_decode_port", None),
        namespace=getattr(args, "k8s_namespace", None),
        task_label_key=getattr(args, "k8s_task_label_key", "task_id"),
        task_id=getattr(args, "k8s_task_id", None),
        url_scheme=getattr(args, "k8s_url_scheme", "http"),
    )
    _validate_k8s_discovery_config(k8s_discovery_config)

    return SmartRouterConfig(
        router_type=args.router_type,
        pd_disaggregation=args.pd_disaggregation,
        worker_urls=args.worker_urls,
        worker_intra_dp_size=args.worker_intra_dp_size,
        prefill_urls=args.prefill_urls,
        prefill_intra_dp_size=args.prefill_intra_dp_size,
        prefill_bootstrap_ports=getattr(args, "prefill_bootstrap_ports", None),
        decode_urls=args.decode_urls, 
        decode_intra_dp_size=args.decode_intra_dp_size,
        health_config=HealthConfig(
            check_interval_secs=getattr(args, "health_check_interval", 60),
        ),
        k8s_discovery_config=k8s_discovery_config,
        decode_policy_config=decode_policy_config if decode_policy_config else policy_config,
        prefill_policy_config=prefill_policy_config if prefill_policy_config else policy_config,
        policy_config=policy_config,
        kv_events_enabled=getattr(args, "kv_events_enabled", False),
        kv_events_port=getattr(args, "kv_events_port", 5557),
        kv_events_topic=getattr(args, "kv_events_topic", ""),
        kv_events_endpoints=getattr(args, "kv_events_endpoints", None),
        tokenizer=getattr(args, "tokenizer", None),
        tokenizer_trust_remote_code=getattr(
            args, "tokenizer_trust_remote_code", False),
        tokenize_url=getattr(args, "tokenize_url", None),
        tokenize_timeout=getattr(args, "tokenize_timeout", 10.0),
        tokenize_cache_size=getattr(args, "tokenize_cache_size", 4096),
        tokenize_cache_ttl=getattr(args, "tokenize_cache_ttl", 3600.0),
    )
