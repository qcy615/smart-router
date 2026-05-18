from smart_router.config import SmartRouterConfig
from smart_router.entrypoints.serve import api_server


def test_vllm_app_registers_completions_route():
    app = api_server._build_app(
        SmartRouterConfig(router_type="vllm", pd_disaggregation=True)
    )

    route_paths = {route.path for route in app.routes}

    assert "/v1/completions" in route_paths
    assert "/v1/chat/completions" in route_paths
    assert "/v1/models" in route_paths


def test_model_source_urls_round_trip(monkeypatch):
    monkeypatch.delenv(api_server.MODEL_SOURCE_URLS_ENV, raising=False)

    api_server._dump_model_source_urls(
        ["http://prefill-a", "http://shared"],
        ["http://shared", "http://decode-b"],
    )

    assert api_server._load_model_source_urls() == [
        "http://prefill-a",
        "http://shared",
        "http://decode-b",
    ]


def test_load_model_source_urls_handles_invalid_payload(monkeypatch):
    monkeypatch.setenv(api_server.MODEL_SOURCE_URLS_ENV, '{"bad": true}')
    assert api_server._load_model_source_urls() == []

    monkeypatch.setenv(api_server.MODEL_SOURCE_URLS_ENV, "not-json")
    assert api_server._load_model_source_urls() == []
