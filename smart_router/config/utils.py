
import argparse


SUPPORT_POLICIES = ["round_robin", "power_of_two", "prefix_aware", "consistent_hash", "minimum_load"]

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

    # overview
    parser.add_argument(
        "--policy", 
        default="round_robin", 
        choices=SUPPORT_POLICIES, 
        help="The routing policy to use. This can be overridden by --prefill-policy and --decode-policy."   )
    parser.add_argument("--cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for prefill.")
    parser.add_argument("--balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for prefill.")
    
    parser.add_argument(
        "--router_type",
        default="vllm-pd-disagg", 
        choices=["vllm-pd-disagg", "sglang-pd-disagg"],
        help="The routing type to use.")

    # prefill
    parser.add_argument("--prefill-urls", nargs="+")
    parser.add_argument("--prefill-intra-dp-size", type=int, default=1)
    parser.add_argument("--prefill-port", type=int, help="Worker HTTP port for Kubernetes-discovered prefill pods.")
    parser.add_argument("--prefill-policy", default="", choices=[""]+ SUPPORT_POLICIES, help="The routing policy to use for prefill. Overrides --policy if set.")
    parser.add_argument("--prefill-cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for prefill.")

    # decode
    parser.add_argument("--decode-urls", nargs="+")
    parser.add_argument("--decode-intra-dp-size", type=int, default=1)
    parser.add_argument("--decode-port", type=int, help="Worker HTTP port for Kubernetes-discovered decode pods.")
    parser.add_argument("--decode-policy", default="", choices=[""]+SUPPORT_POLICIES, help="The routing policy to use for decode. Overrides --policy if set.")
    parser.add_argument("--decode-cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for decode.")

    # Kubernetes discovery
    parser.add_argument(
        "--enable-k8s-discovery",
        action="store_true",
        help="Discover prefill/decode workers by watching Kubernetes pods with the same task label.",
    )
    parser.add_argument(
        "--k8s-task-label-key",
        default="task_id",
        help="Pod label key used to group router and worker pods into one inference task.",
    )
    parser.add_argument(
        "--k8s-namespace",
        help="Kubernetes namespace to watch. Defaults to the service account namespace.",
    )

    # logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser
