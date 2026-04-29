
import argparse


SUPPORT_POLICIES = ["round_robin", "power_of_two", "prefix_aware", "consistent_hash", "minimum_load"]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # apis
    parser.add_argument("--host", default="0.0.0.0", help="The host to bind the server to.")
    parser.add_argument("--port", type=int, default=8000, help="The port to bind the server to.")
    parser.add_argument("--apiserver-workers", type=int, default=8, help="The number of worker processes for the API server.")

    # overview
    parser.add_argument(
        "--policy", 
        default="round_robin", 
        choices=SUPPORT_POLICIES, 
        help="The routing policy to use. This can be overridden by --prefill-policy and --decode-policy."   )
    parser.add_argument("--cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for prefill.")
    parser.add_argument("--balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--scheduler-max-batch-size", type=int, default=1, help="Maximum number of queued requests to schedule as one batch.")
    parser.add_argument("--scheduler-batch-wait-timeout-ms", type=int, default=0, help="Maximum time to wait for a scheduling batch after the first request is dequeued.")
    parser.add_argument("--scheduler-schedule-response-timeout-ms", type=int, default=5000, help="Maximum time an API request waits for a scheduler assignment before returning 503.")
    parser.add_argument("--scheduler-schedule-response-send-margin-ms", type=int, default=1000, help="Safety margin before schedule-response timeout reserved for worker selection and response sending.")
    parser.add_argument("--scheduler-adaptive-interval-enabled", action="store_true", help="Enable feedback-driven adaptive scheduling intervals.")
    parser.add_argument("--scheduler-stats-window-size", type=int, default=16, help="Maximum number of forward-time samples kept for adaptive scheduling.")
    parser.add_argument("--scheduler-default-forward-time-ms", type=float, default=100.0, help="Fallback forward pass time for adaptive scheduling before runtime samples are available.")
    parser.add_argument("--scheduler-network-latency-ms", type=float, default=0.0, help="Estimated network latency added to adaptive scheduling interval calculation.")
    parser.add_argument("--scheduler-min-interval-ms", type=float, default=0.0, help="Lower bound for adaptive scheduling interval.")
    parser.add_argument("--scheduler-max-interval-ms", type=float, default=1000.0, help="Upper bound for adaptive scheduling interval.")
    parser.add_argument("--scheduler-watchdog-multiplier", type=float, default=5.0, help="Multiplier applied to forward time for adaptive scheduling watchdog timeout.")
    
    parser.add_argument(
        "--router_type",
        default="vllm-pd-disagg", 
        choices=["vllm-pd-disagg", "sglang-pd-disagg", "discovery"], 
        help="The routing type to use.")

    # prefill
    parser.add_argument("--prefill-urls", nargs="+")
    parser.add_argument("--prefill-intra-dp-size", type=int, default=1)
    parser.add_argument("--prefill-policy", default="", choices=[""]+ SUPPORT_POLICIES, help="The routing policy to use for prefill. Overrides --policy if set.")
    parser.add_argument("--prefill-cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for prefill.")
    parser.add_argument("--prefill-balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for prefill.")

    # decode
    parser.add_argument("--decode-urls", nargs="+")
    parser.add_argument("--decode-intra-dp-size", type=int, default=1)
    parser.add_argument("--decode-policy", default="", choices=[""]+SUPPORT_POLICIES, help="The routing policy to use for decode. Overrides --policy if set.")
    parser.add_argument("--decode-cache-threshold", type=float, default=0.5, help="The cache threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-balance-abs-threshold", type=int, default=32, help="The absolute balance threshold for prefix-aware policy for decode.")
    parser.add_argument("--decode-balance-rel-threshold", type=float, default=0.1, help="The relative balance threshold for prefix-aware policy for decode.")

    # logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser
