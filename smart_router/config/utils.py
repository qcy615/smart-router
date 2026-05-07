
import argparse


SUPPORT_POLICIES = [
    "round_robin",
    "power_of_two",
    "prefix_aware",
    "consistent_hash",
    "minimum_load",
    "kv_aware",
]

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
    parser.add_argument("--kv-tokenizer-path", default=None, help="Tokenizer/model path used by kv-aware routing.")
    parser.add_argument("--kv-block-size", type=int, default=16, help="KV cache block size used by kv-aware routing.")
    parser.add_argument("--prefill-kv-event-endpoints", nargs="*", default=[], help="vLLM KV event PUB endpoints aligned with --prefill-urls.")
    parser.add_argument("--prefill-kv-replay-endpoints", nargs="*", default=[], help="Optional vLLM KV replay ROUTER endpoints aligned with --prefill-urls.")
    parser.add_argument("--kv-overlap-weight", type=float, default=1.0, help="Weight for kv-aware miss block cost.")
    parser.add_argument("--kv-load-weight", type=float, default=1.0, help="Weight for active worker load in kv-aware policy.")
    parser.add_argument("--kv-event-topic", default="", help="ZMQ topic used by vLLM KV event publisher.")
    parser.add_argument(
        "--enable-openai-local-processing",
        action="store_true",
        help="Enable local OpenAI chat token preprocessing and postprocessing.",
    )
    parser.add_argument(
        "--openai-tokenizer-path",
        default=None,
        help="Tokenizer/model path used by local OpenAI processing. Defaults to --kv-tokenizer-path.",
    )
    parser.add_argument(
        "--reasoning-parser",
        default="think_tags",
        choices=["think_tags", "none"],
        help="Reasoning parser used by local OpenAI postprocessing.",
    )
    parser.add_argument(
        "--tool-call-parser",
        default="hermes",
        choices=["hermes", "none"],
        help="Tool-call parser used by local OpenAI postprocessing.",
    )
    
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
