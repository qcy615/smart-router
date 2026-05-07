import asyncio
from types import SimpleNamespace

import httpx
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from smart_router.engine.engine import EngineResponse, RequestType
from smart_router.entrypoints.serve.vllm_routes import VllmRoutes


class RecordingHttpClient:
    def __init__(self, *, post_responses=None, stream_responses=None):
        self._post_responses = {
            url: list(responses) for url, responses in (post_responses or {}).items()
        }
        self._stream_responses = {
            url: list(responses) for url, responses in (stream_responses or {}).items()
        }
        self.post_calls = []
        self.stream_calls = []

    async def post(self, url: str, json=None, headers=None):
        self.post_calls.append({"url": url, "json": json, "headers": headers})
        return self._post_responses[url].pop(0)

    def stream(self, method: str, url: str, json=None, headers=None):
        self.stream_calls.append(
            {"method": method, "url": url, "json": json, "headers": headers}
        )
        return _FakeStreamContext(self._stream_responses[url].pop(0))

    async def aclose(self):
        return None


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeStreamResponse:
    def __init__(self, status_code: int, chunks: list[bytes]):
        self.status_code = status_code
        self._chunks = chunks

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    async def aread(self) -> bytes:
        return b"".join(self._chunks)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class FakeEngineClient:
    def __init__(self, schedule_response: EngineResponse):
        self.identity = "test-engine-client"
        self._schedule_response = schedule_response
        self.requests = []

    async def send_request(self, request):
        self.requests.append(request)
        if request.request_type == RequestType.SCHEDULE:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(self._schedule_response)
            return future
        return None


def _make_request(engine_client: FakeEngineClient):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine_client=engine_client)))


async def _read_streaming_body(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


def test_get_prefill_body_forces_prefill_only_non_streaming_request():
    routes = VllmRoutes(http_client=RecordingHttpClient())

    original_body = {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    prefill_body = routes._get_prefill_body(original_body)

    assert prefill_body["max_tokens"] == 1
    assert prefill_body["stream"] is False
    assert "stream_options" not in prefill_body
    assert "return_token_ids" not in prefill_body
    assert original_body["stream"] is True


def test_non_stream_chat_route_forwards_kv_params_and_releases_decode_worker():
    prefill_payload = {
        "kv_transfer_params": {"remote_engine_id": "prefill-engine"},
        "prompt_token_ids": [11, 22, 33],
        "choices": [{"message": {"content": "Hello"}}],
    }
    decode_payload = {
        "id": "decode-response",
        "choices": [{"message": {"content": "Hello world"}}],
    }
    http_client = RecordingHttpClient(
        post_responses={
            "http://prefill/v1/chat/completions": [
                httpx.Response(200, json=prefill_payload)
            ],
            "http://decode/v1/chat/completions": [
                httpx.Response(200, json=decode_payload)
            ],
        }
    )
    routes = VllmRoutes(http_client=http_client)
    engine_client = FakeEngineClient(
        EngineResponse(
            request_id="req-1",
            prefill_url="http://prefill",
            prefill_rank=0,
            decode_url="http://decode",
            decode_rank=1,
        )
    )
    app = Starlette(
        routes=[Route("/v1/chat/completions", routes.chat_completions, methods=["POST"])]
    )
    app.state.engine_client = engine_client
    body = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer test"},
        )

    assert response.status_code == 200
    assert "return_token_ids" not in http_client.post_calls[0]["json"]
    assert "prompt_token_ids" not in http_client.post_calls[1]["json"]
    assert http_client.post_calls[1]["json"]["kv_transfer_params"] == {
        "remote_engine_id": "prefill-engine"
    }
    assert http_client.post_calls[1]["headers"]["X-data-parallel-rank"] == "1"
    assert engine_client.requests[0].request_body == body
    assert engine_client.requests[0].api_kind == "chat"

    release_requests = [
        req for req in engine_client.requests if req.request_type == RequestType.RELEASE
    ]
    assert [(req.worker_url, req.worker_rank) for req in release_requests] == [
        ("http://prefill", 0),
        ("http://decode", 1),
    ]


def test_stream_request_forwards_kv_params_without_prompt_token_ids():
    prefill_payload = {
        "kv_transfer_params": {"remote_engine_id": "prefill-engine"},
        "prompt_token_ids": [101, 202],
        "choices": [{"message": {"content": "Hi"}}],
    }
    decode_chunks = [
        b'data: {"choices":[{"delta":{"content":"there"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    http_client = RecordingHttpClient(
        post_responses={
            "http://prefill/v1/chat/completions": [
                httpx.Response(200, json=prefill_payload)
            ],
        },
        stream_responses={
            "http://decode/v1/chat/completions": [
                FakeStreamResponse(200, decode_chunks)
            ],
        },
    )
    routes = VllmRoutes(http_client=http_client)
    engine_client = FakeEngineClient(
        EngineResponse(
            request_id="req-2",
            prefill_url="http://prefill",
            prefill_rank=0,
            decode_url="http://decode",
            decode_rank=1,
        )
    )
    request = _make_request(engine_client)
    body = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }

    async def run_test():
        response = await routes._handle_stream_request(
            request=request,
            body=body,
            headers={"Authorization": "Bearer test"},
            request_text="hello",
            endpoint_path="/v1/chat/completions",
            api_kind="chat",
        )
        streamed_body = await _read_streaming_body(response)
        return streamed_body

    streamed_body = asyncio.run(run_test())

    assert "return_token_ids" not in http_client.post_calls[0]["json"]
    assert "prompt_token_ids" not in http_client.stream_calls[0]["json"]
    assert http_client.stream_calls[0]["json"]["kv_transfer_params"] == {
        "remote_engine_id": "prefill-engine"
    }
    assert b'"content": "Hi"' in streamed_body

    release_requests = [
        req for req in engine_client.requests if req.request_type == RequestType.RELEASE
    ]
    assert [(req.worker_url, req.worker_rank) for req in release_requests] == [
        ("http://prefill", 0),
        ("http://decode", 1),
    ]
