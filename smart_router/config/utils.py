
import argparse


SUPPORT_POLICIES = [
    "round_robin",
    "power_of_two",
    "prefix_aware",
    "kv_event_prefix_aware",
    "consistent_hash",
    "minimum_load",
]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # apis
    parser.add_argument("--host", default="0.0.0.0", help="The host to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="The port to bind the server to.")
    parser.add_argument("--apiserver-workers", type=int, default=8, help="The number of worker processes for the API server.")
    parser.add_argument(
        "--health-check-interval",
        type=int,
        default=60,
        help="Seconds between full worker health checks.",
    )
    parser.add_argument(
        "--upstream-connect-timeout-sec",
        type=float,
        default=5.0,
        help="Seconds to wait when opening an upstream HTTP connection.",
    )
    parser.add_argument(
        "--upstream-read-timeout-sec",
        type=float,
        default=0.0,
        help="Seconds to wait for upstream response data. Set 0 to disable.",
    )
    parser.add_argument(
        "--upstream-write-timeout-sec",
        type=float,
        default=30.0,
        help="Seconds to wait while sending upstream request data.",
    )
    parser.add_argument(
        "--upstream-pool-timeout-sec",
        type=float,
        default=1.0,
        help="Seconds to wait for a free upstream HTTP connection from the pool.",
    )
    parser.add_argument(
        "--upstream-max-connections",
        type=int,
        default=1024,
        help="Maximum concurrent upstream HTTP connections per API server process.",
    )
    parser.add_argument(
        "--upstream-max-keepalive-connections",
        type=int,
        default=256,
        help="Maximum idle upstream HTTP keep-alive connections per API server process.",
    )
    parser.add_argument(
        "--upstream-keepalive-expiry-sec",
        type=float,
        default=30.0,
        help="Seconds before idle upstream HTTP keep-alive connections expire.",
    )
    parser.add_argument(
        "--enable-k8s-discovery",
        action="store_true",
        help="Discover workers from Kubernetes pods.",
    )
    parser.add_argument(
        "--k8s-prefill-port",
        type=int,
        help="Port used to build discovered prefill worker URLs.",
    )
    parser.add_argument(
        "--k8s-decode-port",
        type=int,
        help="Port used to build discovered decode worker URLs.",
    )
    parser.add_argument(
        "--k8s-regular-port",
        type=int,
        help="Port used to build discovered regular worker URLs in non-PD mode.",
    )
    parser.add_argument(
        "--k8s-namespace",
        help="Kubernetes namespace to watch. Defaults to the service account namespace.",
    )
    parser.add_argument(
        "--k8s-task-label-key",
        default="task_id",
        help="Pod label key used to group router and workers into one inference task.",
    )
    parser.add_argument(
        "--k8s-task-id",
        help="Task id label value to watch. Defaults to the router pod's own label value.",
    )
    parser.add_argument(
        "--k8s-url-scheme",
        default="http",
        choices=["http", "https"],
        help="URL scheme used for discovered worker URLs.",
    )

    # overview
    parser.add_argument(
        "--policy", 
        default="round_robin", 
        choices=SUPPORT_POLICIES, 
        help="The routing policy to use. This can be overridden by --prefill-policy and --decode-policy."   )
    parser.add_argument("--cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for prefill.")
    parser.add_argument("--balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, help="Per-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    parser.add_argument("--prefix-cache-eviction-target-chars", type=int, default=1_600_000, help="Per-worker prefix tree character target after eviction for prefix-aware policy.")
    parser.add_argument("--prefix-cache-eviction-interval-secs", type=float, default=120.0, help="Seconds between prefix-aware tree eviction checks.")
    
    parser.add_argument(
        "--router-type",
        default="vllm",
        choices=["vllm", "sglang"],
        help="The routing type to use inference framework.")

    parser.add_argument(
        "--pd-disaggregation",
        action="store_true",
        default=False,
        help="Enable PD (prefill-decode) disaggregation mode. If not set, requests are forwarded directly to workers.",
    )

    # worker urls (non-PD mode)
    parser.add_argument("--worker-urls", nargs="+", help="Worker URLs for non-PD mode.")
    parser.add_argument("--worker-intra-dp-size", type=int, default=1,
                        help="Intra data-parallel size for non-PD mode workers.")

    # prefill
    parser.add_argument("--prefill-urls", nargs="+")
    parser.add_argument("--prefill-intra-dp-size", type=int, default=1)
    parser.add_argument("--prefill-policy", default="", choices=[""]+ SUPPORT_POLICIES, help="The routing policy to use for prefill. Overrides --policy if set.")
    parser.add_argument("--prefill-cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, help="Per-prefill-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    parser.add_argument("--prefill-prefix-cache-eviction-target-chars", type=int, default=1_600_000, help="Per-prefill-worker prefix tree character target after eviction for prefix-aware policy.")
    parser.add_argument("--prefill-prefix-cache-eviction-interval-secs", type=float, default=30.0, help="Seconds between prefill prefix-aware tree eviction checks.")

    # decode
    parser.add_argument("--decode-urls", nargs="+")
    parser.add_argument("--decode-intra-dp-size", type=int, default=1)
    parser.add_argument("--decode-policy", default="", choices=[""]+SUPPORT_POLICIES, help="The routing policy to use for decode. Overrides --policy if set.")
    parser.add_argument("--decode-cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-prefix-cache-eviction-threshold-chars", type=int, default=2_000_000, help="Per-decode-worker prefix tree character high watermark for prefix-aware policy. Set <= 0 to disable eviction.")
    parser.add_argument("--decode-prefix-cache-eviction-target-chars", type=int, default=1_600_000, help="Per-decode-worker prefix tree character target after eviction for prefix-aware policy.")
    parser.add_argument("--decode-prefix-cache-eviction-interval-secs", type=float, default=30.0, help="Seconds between decode prefix-aware tree eviction checks.")

    # vLLM KV cache events
    parser.add_argument(
        "--kv-events-enabled",
        action="store_true",
        help="Subscribe to vLLM KV cache events and mirror per-DP prefix cache state.",
    )
    parser.add_argument(
        "--kv-events-port",
        type=int,
        default=5557,
        help="Base vLLM KV event publisher port. Rank N uses port + N unless endpoints are provided.",
    )
    parser.add_argument(
        "--kv-events-topic",
        default="",
        help="ZMQ topic for vLLM KV cache events.",
    )
    parser.add_argument(
        "--kv-events-endpoints",
        nargs="+",
        help=(
            "KV event endpoints. Provide one base endpoint per worker URL, or one "
            "endpoint per DP rank. Base endpoints are expanded by intra-DP size."
        ),
    )

    # token ids for KV event matching
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Local HuggingFace tokenizer name/path used if remote /tokenize is not configured or fails.",
    )
    parser.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the local tokenizer.",
    )
    parser.add_argument(
        "--tokenize-url",
        default=None,
        help="vLLM /tokenize endpoint used to compute request token ids.",
    )
    parser.add_argument(
        "--tokenize-timeout",
        type=float,
        default=10.0,
        help="Timeout in seconds for remote /tokenize requests.",
    )
    parser.add_argument(
        "--tokenize-cache-size",
        type=int,
        default=4096,
        help="Maximum number of tokenized request entries kept in router memory.",
    )
    parser.add_argument(
        "--tokenize-cache-ttl",
        type=float,
        default=3600.0,
        help="Tokenization cache TTL in seconds.",
    )

    # logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser
