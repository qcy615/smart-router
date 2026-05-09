import pytest

from smart_router.config import build_config, build_parser


def _config_from_args(*args):
    parser = build_parser()
    return build_config(parser.parse_args(list(args)))


def test_k8s_discovery_config_accepts_ports_and_label_key():
    config = _config_from_args(
        "--enable-k8s-discovery",
        "--prefill-port",
        "8100",
        "--decode-port",
        "8200",
        "--k8s-task-label-key",
        "task_id",
        "--k8s-namespace",
        "inference",
    )

    assert config.enable_k8s_discovery is True
    assert config.prefill_port == 8100
    assert config.decode_port == 8200
    assert config.k8s_task_label_key == "task_id"
    assert config.k8s_namespace == "inference"


def test_k8s_discovery_rejects_static_urls():
    with pytest.raises(ValueError, match="cannot be used"):
        _config_from_args(
            "--enable-k8s-discovery",
            "--prefill-port",
            "8100",
            "--decode-port",
            "8200",
            "--prefill-urls",
            "http://prefill",
        )


def test_k8s_discovery_requires_ports():
    with pytest.raises(ValueError, match="--prefill-port"):
        _config_from_args("--enable-k8s-discovery", "--decode-port", "8200")

    with pytest.raises(ValueError, match="--decode-port"):
        _config_from_args("--enable-k8s-discovery", "--prefill-port", "8100")


def test_dp_sizes_must_be_positive():
    with pytest.raises(ValueError, match="prefill"):
        _config_from_args("--prefill-intra-dp-size", "0")

    with pytest.raises(ValueError, match="decode"):
        _config_from_args("--decode-intra-dp-size", "0")
