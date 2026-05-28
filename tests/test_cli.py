import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_router import cli
from smart_router.config import (
    KVEventsConfig,
    RouterModeConfig,
    SmartRouterConfig,
    TokenizationConfig,
    WorkerGroupConfig,
    build_config,
    build_parser,
)


def test_cli_help_lists_available_commands(capsys):
    assert cli.main([]) == 0
    captured = capsys.readouterr()
    assert "serve" in captured.out
    assert "benchmark" in captured.out


def test_cli_rejects_unknown_command(capsys):
    assert cli.main(["unknown-command"]) == 2
    captured = capsys.readouterr()
    assert "Unknown command" in captured.err


def test_cli_dispatches_to_benchmark_handler(monkeypatch):
    benchmark_calls: list[list[str]] = []

    def fake_import_module(name: str):
        assert name == "smart_router.entrypoints.benchmark.benchmark_serving_multi_turn"
        return type(
            "BenchmarkModule",
            (),
            {"cli": staticmethod(lambda argv=None: benchmark_calls.append(argv or []) or 0)},
        )

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert cli.main(["benchmark", "--help"]) == 0
    assert benchmark_calls == [["--help"]]


def test_parser_help_groups_options():
    help_text = build_parser().format_help()

    for group_title in (
        "API server:",
        "Upstream HTTP client:",
        "Kubernetes discovery:",
        "Routing:",
        "Workers:",
        "Prefill workers:",
        "Decode workers:",
        "KV cache events:",
        "Tokenization:",
        "Logging:",
    ):
        assert group_title in help_text


def test_k8s_discovery_cli_builds_config():
    args = build_parser().parse_args(
        [
            "--enable-k8s-discovery",
            "--pd-disaggregation",
            "--k8s-prefill-port",
            "8100",
            "--k8s-decode-port",
            "8200",
            "--k8s-regular-port",
            "8300",
            "--k8s-namespace",
            "inference",
            "--k8s-task-label-key",
            "task",
            "--k8s-task-id",
            "abc",
            "--k8s-url-scheme",
            "https",
        ]
    )

    config = build_config(args)

    assert config.k8s_discovery_config.enabled is True
    assert config.k8s_discovery_config.prefill_port == 8100
    assert config.k8s_discovery_config.decode_port == 8200
    assert config.k8s_discovery_config.regular_port == 8300
    assert config.k8s_discovery_config.namespace == "inference"
    assert config.k8s_discovery_config.task_label_key == "task"
    assert config.k8s_discovery_config.task_id == "abc"
    assert config.k8s_discovery_config.url_scheme == "https"


def test_upstream_http_client_cli_builds_config():
    args = build_parser().parse_args(
        [
            "--upstream-connect-timeout-sec",
            "2.5",
            "--upstream-read-timeout-sec",
            "120",
            "--upstream-write-timeout-sec",
            "15",
            "--upstream-pool-timeout-sec",
            "0.5",
            "--upstream-max-connections",
            "512",
            "--upstream-max-keepalive-connections",
            "128",
            "--upstream-keepalive-expiry-sec",
            "20",
        ]
    )

    config = build_config(args)

    upstream_config = config.upstream_http_client_config
    assert upstream_config.connect_timeout_secs == 2.5
    assert upstream_config.read_timeout_secs == 120
    assert upstream_config.write_timeout_secs == 15
    assert upstream_config.pool_timeout_secs == 0.5
    assert upstream_config.max_connections == 512
    assert upstream_config.max_keepalive_connections == 128
    assert upstream_config.keepalive_expiry_secs == 20


def test_upstream_http_client_default_read_timeout_is_disabled():
    args = build_parser().parse_args([])

    config = build_config(args)

    assert config.upstream_http_client_config.read_timeout_secs is None


def test_k8s_discovery_requires_regular_port_in_normal_mode():
    args = build_parser().parse_args(["--enable-k8s-discovery"])

    try:
        build_config(args)
    except RuntimeError as exc:
        assert "--k8s-regular-port" in str(exc)
    else:
        raise AssertionError("build_config should reject missing discovery ports")


def test_k8s_discovery_requires_prefill_and_decode_ports_in_pd_mode():
    args = build_parser().parse_args(
        ["--enable-k8s-discovery", "--pd-disaggregation"]
    )

    try:
        build_config(args)
    except RuntimeError as exc:
        assert "--k8s-prefill-port" in str(exc)
        assert "--k8s-decode-port" in str(exc)
    else:
        raise AssertionError("build_config should reject missing discovery ports")


def test_k8s_discovery_normal_mode_cli_builds_regular_port_config():
    args = build_parser().parse_args(
        [
            "--enable-k8s-discovery",
            "--k8s-regular-port",
            "8300",
        ]
    )

    config = build_config(args)

    assert config.router_mode_config.pd_disaggregation is False
    assert config.k8s_discovery_config.regular_port == 8300


def test_prefill_decode_intra_dp_size_inherits_worker_value():
    args = build_parser().parse_args(["--worker-intra-dp-size", "4"])

    config = build_config(args)

    assert config.worker_config.intra_dp_size == 4
    assert config.prefill_worker_config.intra_dp_size == 4
    assert config.decode_worker_config.intra_dp_size == 4


def test_prefill_decode_intra_dp_size_explicit_values_override_worker_value():
    args = build_parser().parse_args(
        [
            "--worker-intra-dp-size",
            "4",
            "--prefill-intra-dp-size",
            "2",
            "--decode-intra-dp-size",
            "3",
        ]
    )

    config = build_config(args)

    assert config.worker_config.intra_dp_size == 4
    assert config.prefill_worker_config.intra_dp_size == 2
    assert config.decode_worker_config.intra_dp_size == 3


def test_cli_builds_structured_worker_configs():
    args = build_parser().parse_args(
        [
            "--router-type",
            "sglang",
            "--pd-disaggregation",
            "--worker-urls",
            "http://worker-a",
            "--worker-intra-dp-size",
            "4",
            "--prefill-urls",
            "http://prefill-a",
            "--prefill-intra-dp-size",
            "2",
            "--decode-urls",
            "http://decode-a",
            "--decode-intra-dp-size",
            "3",
        ]
    )

    config = build_config(args)

    assert config.router_mode_config == RouterModeConfig(
        router_type="sglang",
        pd_disaggregation=True,
    )
    assert config.worker_config == WorkerGroupConfig(
        urls=["http://worker-a"],
        intra_dp_size=4,
    )
    assert config.prefill_worker_config == WorkerGroupConfig(
        urls=["http://prefill-a"],
        intra_dp_size=2,
    )
    assert config.decode_worker_config == WorkerGroupConfig(
        urls=["http://decode-a"],
        intra_dp_size=3,
    )


def test_cli_builds_kv_events_and_tokenization_configs():
    args = build_parser().parse_args(
        [
            "--kv-events-enabled",
            "--kv-events-port",
            "6000",
            "--kv-events-topic",
            "prefix-events",
            "--kv-events-endpoints",
            "tcp://127.0.0.1:7000",
            "--tokenizer",
            "local-tokenizer",
            "--tokenizer-trust-remote-code",
            "--tokenize-url",
            "http://prefill/tokenize",
            "--tokenize-timeout",
            "2.5",
            "--tokenize-cache-size",
            "128",
            "--tokenize-cache-ttl",
            "30",
        ]
    )

    config = build_config(args)

    assert config.kv_events_config == KVEventsConfig(
        enabled=True,
        port=6000,
        topic="prefix-events",
        endpoints=["tcp://127.0.0.1:7000"],
    )
    assert config.tokenization_config == TokenizationConfig(
        tokenizer="local-tokenizer",
        tokenizer_trust_remote_code=True,
        tokenize_url="http://prefill/tokenize",
        tokenize_timeout=2.5,
        tokenize_cache_size=128,
        tokenize_cache_ttl=30,
    )


def test_smart_router_config_accepts_structured_configs():
    config = SmartRouterConfig(
        router_mode_config=RouterModeConfig(
            router_type="sglang",
            pd_disaggregation=True,
        ),
        prefill_worker_config=WorkerGroupConfig(
            urls=["http://prefill-a"],
            intra_dp_size=2,
            bootstrap_ports=[9000],
        ),
        decode_worker_config=WorkerGroupConfig(
            urls=["http://decode-a"],
            intra_dp_size=3,
        ),
        kv_events_config=KVEventsConfig(
            enabled=True,
            port=6000,
            topic="prefix-events",
            endpoints=["tcp://127.0.0.1:7000"],
        ),
        tokenization_config=TokenizationConfig(
            tokenizer="local-tokenizer",
            tokenizer_trust_remote_code=True,
            tokenize_url="http://prefill/tokenize",
            tokenize_timeout=2.5,
            tokenize_cache_size=128,
            tokenize_cache_ttl=30,
        ),
    )

    assert config.router_mode_config == RouterModeConfig(
        router_type="sglang",
        pd_disaggregation=True,
    )
    assert config.prefill_worker_config == WorkerGroupConfig(
        urls=["http://prefill-a"],
        intra_dp_size=2,
        bootstrap_ports=[9000],
    )
    assert config.decode_worker_config == WorkerGroupConfig(
        urls=["http://decode-a"],
        intra_dp_size=3,
    )
    assert config.kv_events_config == KVEventsConfig(
        enabled=True,
        port=6000,
        topic="prefix-events",
        endpoints=["tcp://127.0.0.1:7000"],
    )
    assert config.tokenization_config == TokenizationConfig(
        tokenizer="local-tokenizer",
        tokenizer_trust_remote_code=True,
        tokenize_url="http://prefill/tokenize",
        tokenize_timeout=2.5,
        tokenize_cache_size=128,
        tokenize_cache_ttl=30,
    )


def test_prefix_cache_eviction_global_cli_builds_policy_config():
    args = build_parser().parse_args(
        [
            "--policy",
            "prefix_aware",
            "--prefix-cache-eviction-threshold-chars",
            "123",
            "--prefix-cache-eviction-target-chars",
            "45",
            "--prefix-cache-eviction-interval-secs",
            "0.5",
        ]
    )

    config = build_config(args)

    assert config.prefill_policy_config.prefix_cache_eviction_threshold_chars == 123
    assert config.prefill_policy_config.prefix_cache_eviction_target_chars == 45
    assert config.prefill_policy_config.prefix_cache_eviction_interval_secs == 0.5
    assert config.decode_policy_config.prefix_cache_eviction_threshold_chars == 123
    assert config.decode_policy_config.prefix_cache_eviction_target_chars == 45
    assert config.decode_policy_config.prefix_cache_eviction_interval_secs == 0.5


def test_prefix_cache_eviction_prefill_and_decode_cli_overrides():
    args = build_parser().parse_args(
        [
            "--prefill-policy",
            "prefix_aware",
            "--prefill-prefix-cache-eviction-threshold-chars",
            "100",
            "--prefill-prefix-cache-eviction-target-chars",
            "80",
            "--prefill-prefix-cache-eviction-interval-secs",
            "1.5",
            "--decode-policy",
            "prefix_aware",
            "--decode-prefix-cache-eviction-threshold-chars",
            "200",
            "--decode-prefix-cache-eviction-target-chars",
            "160",
            "--decode-prefix-cache-eviction-interval-secs",
            "2.5",
        ]
    )

    config = build_config(args)

    assert config.prefill_policy_config.prefix_cache_eviction_threshold_chars == 100
    assert config.prefill_policy_config.prefix_cache_eviction_target_chars == 80
    assert config.prefill_policy_config.prefix_cache_eviction_interval_secs == 1.5
    assert config.decode_policy_config.prefix_cache_eviction_threshold_chars == 200
    assert config.decode_policy_config.prefix_cache_eviction_target_chars == 160
    assert config.decode_policy_config.prefix_cache_eviction_interval_secs == 2.5


def test_prefill_decode_policy_cli_inherits_global_policy_values():
    args = build_parser().parse_args(
        [
            "--policy",
            "prefix_aware",
            "--cache-threshold",
            "0.7",
            "--balance-abs-threshold",
            "12",
            "--balance-rel-threshold",
            "0.4",
            "--prefix-cache-eviction-threshold-chars",
            "300",
            "--prefix-cache-eviction-target-chars",
            "240",
            "--prefix-cache-eviction-interval-secs",
            "3.5",
            "--prefill-policy",
            "prefix_aware",
            "--decode-policy",
            "kv_event_prefix_aware",
        ]
    )

    config = build_config(args)

    assert config.prefill_policy_config.policy == "prefix_aware"
    assert config.decode_policy_config.policy == "kv_event_prefix_aware"

    for policy_config in (
        config.prefill_policy_config,
        config.decode_policy_config,
    ):
        assert policy_config.cache_threshold == 0.7
        assert policy_config.balance_abs_threshold == 12
        assert policy_config.balance_rel_threshold == 0.4
        assert policy_config.prefix_cache_eviction_threshold_chars == 300
        assert policy_config.prefix_cache_eviction_target_chars == 240
        assert policy_config.prefix_cache_eviction_interval_secs == 3.5


def test_prefill_decode_policy_cli_uses_explicit_prefixed_values():
    args = build_parser().parse_args(
        [
            "--policy",
            "prefix_aware",
            "--cache-threshold",
            "0.2",
            "--prefix-cache-eviction-interval-secs",
            "6.0",
            "--prefill-cache-threshold",
            "0.8",
            "--decode-prefix-cache-eviction-interval-secs",
            "2.5",
        ]
    )

    config = build_config(args)

    assert config.prefill_policy_config.policy == "prefix_aware"
    assert config.prefill_policy_config.cache_threshold == 0.8
    assert config.prefill_policy_config.prefix_cache_eviction_interval_secs == 6.0

    assert config.decode_policy_config.policy == "prefix_aware"
    assert config.decode_policy_config.cache_threshold == 0.2
    assert config.decode_policy_config.prefix_cache_eviction_interval_secs == 2.5
