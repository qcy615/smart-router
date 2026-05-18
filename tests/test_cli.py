import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_router import cli
from smart_router.config import build_config, build_parser


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

    assert config.pd_disaggregation is False
    assert config.k8s_discovery_config.regular_port == 8300


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
