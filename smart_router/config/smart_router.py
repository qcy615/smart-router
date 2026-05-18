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
    regular_port: Optional[int] = None
    namespace: Optional[str] = None
    task_label_key: str = "task_id"
    task_id: Optional[str] = None
    url_scheme: str = "http"

@dataclass
class UpstreamHTTPClientConfig:
    connect_timeout_secs: float = 5.0
    read_timeout_secs: Optional[float] = None
    write_timeout_secs: float = 30.0
    pool_timeout_secs: float = 1.0
    max_connections: int = 1024
    max_keepalive_connections: int = 256
    keepalive_expiry_secs: float = 30.0

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
    upstream_http_client_config: UpstreamHTTPClientConfig = field(
        default_factory=UpstreamHTTPClientConfig
    )

    prefill_policy_config: PolicyConfig = field(default_factory=PolicyConfig)
    decode_policy_config: PolicyConfig = field(default_factory=PolicyConfig)


def _validate_k8s_discovery_config(
    config: K8SDiscoveryConfig,
    *,
    pd_disaggregation: bool,
) -> None:
    if not config.enabled:
        return

    missing = []
    if pd_disaggregation:
        if config.prefill_port is None:
            missing.append("--k8s-prefill-port")
        if config.decode_port is None:
            missing.append("--k8s-decode-port")
    elif config.regular_port is None:
        missing.append("--k8s-regular-port")
    if missing:
        raise RuntimeError(
            "K8S discovery requires " + " and ".join(missing)
        )

    for name, port in (
        ("--k8s-prefill-port", config.prefill_port),
        ("--k8s-decode-port", config.decode_port),
        ("--k8s-regular-port", config.regular_port),
    ):
        if port is not None and (port <= 0 or port > 65535):
            raise RuntimeError(f"{name} must be between 1 and 65535")


def _optional_timeout(value: float) -> Optional[float]:
    return value if value > 0 else None


def _validate_upstream_http_client_config(config: UpstreamHTTPClientConfig) -> None:
    for name, value in (
        ("--upstream-connect-timeout-sec", config.connect_timeout_secs),
        ("--upstream-write-timeout-sec", config.write_timeout_secs),
        ("--upstream-pool-timeout-sec", config.pool_timeout_secs),
        ("--upstream-keepalive-expiry-sec", config.keepalive_expiry_secs),
    ):
        if value <= 0:
            raise RuntimeError(f"{name} must be greater than 0")

    if config.read_timeout_secs is not None and config.read_timeout_secs <= 0:
        raise RuntimeError("--upstream-read-timeout-sec must be greater than 0 or 0 to disable")

    if config.max_connections <= 0:
        raise RuntimeError("--upstream-max-connections must be greater than 0")
    if config.max_keepalive_connections < 0:
        raise RuntimeError("--upstream-max-keepalive-connections must be >= 0")
    if config.max_keepalive_connections > config.max_connections:
        raise RuntimeError(
            "--upstream-max-keepalive-connections must be <= --upstream-max-connections"
        )


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
        regular_port=getattr(args, "k8s_regular_port", None),
        namespace=getattr(args, "k8s_namespace", None),
        task_label_key=getattr(args, "k8s_task_label_key", "task_id"),
        task_id=getattr(args, "k8s_task_id", None),
        url_scheme=getattr(args, "k8s_url_scheme", "http"),
    )
    _validate_k8s_discovery_config(
        k8s_discovery_config,
        pd_disaggregation=args.pd_disaggregation,
    )

    upstream_http_client_config = UpstreamHTTPClientConfig(
        connect_timeout_secs=getattr(args, "upstream_connect_timeout_sec", 5.0),
        read_timeout_secs=_optional_timeout(
            getattr(args, "upstream_read_timeout_sec", 0.0)
        ),
        write_timeout_secs=getattr(args, "upstream_write_timeout_sec", 30.0),
        pool_timeout_secs=getattr(args, "upstream_pool_timeout_sec", 1.0),
        max_connections=getattr(args, "upstream_max_connections", 1024),
        max_keepalive_connections=getattr(
            args, "upstream_max_keepalive_connections", 256
        ),
        keepalive_expiry_secs=getattr(args, "upstream_keepalive_expiry_sec", 30.0),
    )
    _validate_upstream_http_client_config(upstream_http_client_config)

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
        upstream_http_client_config=upstream_http_client_config,
        decode_policy_config=decode_policy_config if decode_policy_config else policy_config,
        prefill_policy_config=prefill_policy_config if prefill_policy_config else policy_config,
        policy_config=policy_config,
    )
