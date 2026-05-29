
import argparse


SUPPORT_POLICIES = [
    "round_robin",
    "power_of_two",
    "prefix_aware",
    "kv_event_prefix_aware",
    "consistent_hash",
    "minimum_load",
]


PREFILL_DECODE_INHERITABLE_POLICY_ARGS = (
    "policy",
    "cache_threshold",
    "balance_abs_threshold",
    "balance_rel_threshold",
    "prefix_cache_eviction_threshold_chars",
    "prefix_cache_eviction_target_chars",
    "prefix_cache_eviction_interval_secs",
)

_PREFILL_DECODE_PREFIXES = ("prefill", "decode")
_EXPLICIT_ARG_DESTS = "_explicit_arg_dests"


class _StoreExplicitAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        explicit_arg_dests = getattr(namespace, _EXPLICIT_ARG_DESTS, None)
        if explicit_arg_dests is None:
            explicit_arg_dests = set()
            setattr(namespace, _EXPLICIT_ARG_DESTS, explicit_arg_dests)
        explicit_arg_dests.add(self.dest)
        setattr(namespace, self.dest, values)


class SmartRouterArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        namespace, extras = super().parse_known_args(args, namespace)
        self._apply_prefill_decode_inheritance(namespace)
        return namespace, extras

    @staticmethod
    def _apply_prefill_decode_inheritance(namespace):
        explicit_arg_dests = getattr(namespace, _EXPLICIT_ARG_DESTS, set())
        setattr(namespace, _EXPLICIT_ARG_DESTS, explicit_arg_dests)

        if "worker_intra_dp_size" in explicit_arg_dests:
            for prefix in _PREFILL_DECODE_PREFIXES:
                role_dest = f"{prefix}_intra_dp_size"
                if role_dest not in explicit_arg_dests:
                    setattr(namespace, role_dest, namespace.worker_intra_dp_size)

        for prefix in _PREFILL_DECODE_PREFIXES:
            for dest in PREFILL_DECODE_INHERITABLE_POLICY_ARGS:
                role_dest = f"{prefix}_{dest}"
                if dest in explicit_arg_dests and role_dest not in explicit_arg_dests:
                    setattr(namespace, role_dest, getattr(namespace, dest))

            role_policy_dest = f"{prefix}_policy"
            role_has_explicit_policy_arg = any(
                f"{prefix}_{dest}" in explicit_arg_dests
                for dest in PREFILL_DECODE_INHERITABLE_POLICY_ARGS
            )

            if (
                role_has_explicit_policy_arg
                and getattr(namespace, role_policy_dest, "") == ""
            ):
                setattr(namespace, role_policy_dest, namespace.policy)

            if getattr(namespace, role_policy_dest, "") == "":
                continue

            for dest in PREFILL_DECODE_INHERITABLE_POLICY_ARGS:
                role_dest = f"{prefix}_{dest}"
                if role_dest not in explicit_arg_dests:
                    setattr(namespace, role_dest, getattr(namespace, dest))


def build_parser() -> argparse.ArgumentParser:
    parser = SmartRouterArgumentParser()

    api_group = parser.add_argument_group("API server")
    api_group.add_argument("--host", default="0.0.0.0", help="The host to bind the server to.")
    api_group.add_argument("--port", type=int, default=8000, help="The port to bind the server to.")
    api_group.add_argument("--apiserver-workers", type=int, default=8, help="The number of worker processes for the API server.")
    api_group.add_argument(
        "--health-check-interval",
        type=int,
        default=60,
        help="Seconds between full worker health checks.",
    )

    upstream_group = parser.add_argument_group("Upstream HTTP client")
    upstream_group.add_argument(
        "--upstream-connect-timeout-sec",
        type=float,
        default=5.0,
        help="Seconds to wait when opening an upstream HTTP connection.",
    )
    upstream_group.add_argument(
        "--upstream-read-timeout-sec",
        type=float,
        default=0.0,
        help="Seconds to wait for upstream response data. Set 0 to disable.",
    )
    upstream_group.add_argument(
        "--upstream-write-timeout-sec",
        type=float,
        default=30.0,
        help="Seconds to wait while sending upstream request data.",
    )
    upstream_group.add_argument(
        "--upstream-pool-timeout-sec",
        type=float,
        default=1.0,
        help="Seconds to wait for a free upstream HTTP connection from the pool.",
    )
    upstream_group.add_argument(
        "--upstream-max-connections",
        type=int,
        default=1024,
        help="Maximum concurrent upstream HTTP connections per API server process.",
    )
    upstream_group.add_argument(
        "--upstream-max-keepalive-connections",
        type=int,
        default=256,
        help="Maximum idle upstream HTTP keep-alive connections per API server process.",
    )
    upstream_group.add_argument(
        "--upstream-keepalive-expiry-sec",
        type=float,
        default=30.0,
        help="Seconds before idle upstream HTTP keep-alive connections expire.",
    )

    k8s_group = parser.add_argument_group("Kubernetes discovery")
    k8s_group.add_argument(
        "--enable-k8s-discovery",
        action="store_true",
        help="Discover workers from Kubernetes pods.",
    )
    k8s_group.add_argument(
        "--k8s-prefill-port",
        type=int,
        help="Port used to build discovered prefill worker URLs.",
    )
    k8s_group.add_argument(
        "--k8s-decode-port",
        type=int,
        help="Port used to build discovered decode worker URLs.",
    )
    k8s_group.add_argument(
        "--k8s-regular-port",
        type=int,
        help="Port used to build discovered regular worker URLs in non-PD mode.",
    )
    k8s_group.add_argument(
        "--k8s-namespace",
        help="Kubernetes namespace to watch. Defaults to the service account namespace.",
    )
    k8s_group.add_argument(
        "--k8s-task-label-key",
        default="task_id",
        help="Pod label key used to group router and workers into one inference task.",
    )
    k8s_group.add_argument(
        "--k8s-task-id",
        help="Task id label value to watch. Defaults to the router pod's own label value.",
    )
    k8s_group.add_argument(
        "--k8s-url-scheme",
        default="http",
        choices=["http", "https"],
        help="URL scheme used for discovered worker URLs.",
    )

    routing_group = parser.add_argument_group("Routing")
    routing_group.add_argument(
        "--policy",
        default="round_robin",
        choices=SUPPORT_POLICIES,
        action=_StoreExplicitAction,
        help="The routing policy to use. This can be overridden by --prefill-policy and --decode-policy.",
    )
    routing_group.add_argument("--cache-threshold", type=float, default=0.5, action=_StoreExplicitAction, help="The cache threshold for prefix-aware policy.")
    routing_group.add_argument("--balance-abs-threshold", type=int, default=32, action=_StoreExplicitAction, help="The absolute balance threshold for prefix-aware policy.")
    routing_group.add_argument("--balance-rel-threshold", type=float, default=0.1, action=_StoreExplicitAction, help="The relative balance threshold for prefix-aware policy.")
    routing_group.add_argument("--prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, action=_StoreExplicitAction, help="Per-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    routing_group.add_argument("--prefix-cache-eviction-target-chars", type=int, default=1_600_000, action=_StoreExplicitAction, help="Per-worker prefix tree character target after eviction for prefix-aware policy.")
    routing_group.add_argument("--prefix-cache-eviction-interval-secs", type=float, default=120.0, action=_StoreExplicitAction, help="Seconds between prefix-aware tree eviction checks.")
    routing_group.add_argument(
        "--router-type",
        default="vllm",
        choices=["vllm", "sglang"],
        help="The routing type to use inference framework.",
    )
    routing_group.add_argument(
        "--pd-disaggregation",
        action="store_true",
        default=False,
        help="Enable PD (prefill-decode) disaggregation mode. If not set, requests are forwarded directly to workers.",
    )

    worker_group = parser.add_argument_group("Workers")
    worker_group.add_argument("--worker-urls", nargs="+", help="Worker URLs for non-PD mode.")
    worker_group.add_argument("--worker-intra-dp-size", type=int, default=1, action=_StoreExplicitAction, help="Intra data-parallel size for non-PD mode workers.")

    prefill_group = parser.add_argument_group("Prefill workers")
    prefill_group.add_argument("--prefill-urls", nargs="+")
    prefill_group.add_argument("--prefill-intra-dp-size", type=int, default=1, action=_StoreExplicitAction)
    prefill_group.add_argument("--prefill-policy", default="", choices=[""] + SUPPORT_POLICIES, action=_StoreExplicitAction, help="The routing policy to use for prefill. Overrides --policy if set.")
    prefill_group.add_argument("--prefill-cache-threshold", type=float, default=0.5, action=_StoreExplicitAction, help="The cache threshold for prefix-aware policy for prefill.")
    prefill_group.add_argument("--prefill-balance-abs-threshold", type=int, default=32, action=_StoreExplicitAction, help="The absolute balance threshold for prefix-aware policy for prefill.")
    prefill_group.add_argument("--prefill-balance-rel-threshold", type=float, default=0.1, action=_StoreExplicitAction, help="The relative balance threshold for prefix-aware policy for prefill.")
    prefill_group.add_argument("--prefill-prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, action=_StoreExplicitAction, help="Per-prefill-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    prefill_group.add_argument("--prefill-prefix-cache-eviction-target-chars", type=int, default=1_600_000, action=_StoreExplicitAction, help="Per-prefill-worker prefix tree character target after eviction for prefix-aware policy.")
    prefill_group.add_argument("--prefill-prefix-cache-eviction-interval-secs", type=float, default=120.0, action=_StoreExplicitAction, help="Seconds between prefill prefix-aware tree eviction checks.")

    decode_group = parser.add_argument_group("Decode workers")
    decode_group.add_argument("--decode-urls", nargs="+")
    decode_group.add_argument("--decode-intra-dp-size", type=int, default=1, action=_StoreExplicitAction)
    decode_group.add_argument("--decode-policy", default="", choices=[""] + SUPPORT_POLICIES, action=_StoreExplicitAction, help="The routing policy to use for decode. Overrides --policy if set.")
    decode_group.add_argument("--decode-cache-threshold", type=float, default=0.5, action=_StoreExplicitAction, help="The cache threshold for prefix-aware policy for decode.")
    decode_group.add_argument("--decode-balance-abs-threshold", type=int, default=32, action=_StoreExplicitAction, help="The absolute balance threshold for prefix-aware policy for decode.")
    decode_group.add_argument("--decode-balance-rel-threshold", type=float, default=0.1, action=_StoreExplicitAction, help="The relative balance threshold for prefix-aware policy for decode.")
    decode_group.add_argument("--decode-prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, action=_StoreExplicitAction, help="Per-decode-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    decode_group.add_argument("--decode-prefix-cache-eviction-target-chars", type=int, default=1_600_000, action=_StoreExplicitAction, help="Per-decode-worker prefix tree character target after eviction for prefix-aware policy.")
    decode_group.add_argument("--decode-prefix-cache-eviction-interval-secs", type=float, default=120.0, action=_StoreExplicitAction, help="Seconds between decode prefix-aware tree eviction checks.")

    kv_events_group = parser.add_argument_group("KV cache events")
    kv_events_group.add_argument(
        "--kv-events-enabled",
        action="store_true",
        help="Subscribe to vLLM KV cache events and mirror per-DP prefix cache state.",
    )
    kv_events_group.add_argument(
        "--kv-events-port",
        type=int,
        default=5557,
        help="Base vLLM KV event publisher port. Rank N uses port + N unless endpoints are provided.",
    )
    kv_events_group.add_argument(
        "--kv-events-topic",
        default="",
        help="ZMQ topic for vLLM KV cache events.",
    )
    kv_events_group.add_argument(
        "--kv-events-endpoints",
        nargs="+",
        help=(
            "KV event endpoints. Provide one base endpoint per worker URL, or one "
            "endpoint per DP rank. Base endpoints are expanded by intra-DP size."
        ),
    )

    tokenization_group = parser.add_argument_group("Tokenization")
    tokenization_group.add_argument(
        "--tokenizer",
        default=None,
        help="Local HuggingFace tokenizer name/path used if remote /tokenize is not configured or fails.",
    )
    tokenization_group.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the local tokenizer.",
    )
    tokenization_group.add_argument(
        "--tokenize-url",
        default=None,
        help="vLLM /tokenize endpoint used to compute request token ids.",
    )
    tokenization_group.add_argument(
        "--tokenize-timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for remote /tokenize requests.",
    )
    tokenization_group.add_argument(
        "--tokenize-cache-size",
        type=int,
        default=4096,
        help="Maximum number of tokenized request entries kept in router memory.",
    )
    tokenization_group.add_argument(
        "--tokenize-cache-ttl",
        type=float,
        default=3600.0,
        help="Tokenization cache TTL in seconds.",
    )

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser
