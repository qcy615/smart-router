import asyncio
from types import SimpleNamespace

import httpx

from smart_router.engine.engine import EngineResponse, RequestType
from smart_router.entrypoints.serve.normal_routes import NormalRoutes


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
    def __init__(self):
        self.identity = "test-engine-client"
        self.requests = []

    async def send_request(self, request):
        self.requests.append(request)
        if request.request_type == RequestType.SCHEDULE:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(
                EngineResponse(
                    request_id=request.request_id,
                    prefill_url=None,
                    prefill_rank=-1,
                    decode_url=None,
                    decode_rank=-1,
                    worker_url="http://worker",
                    worker_rank=3,
                )
            )
            return future
        return None


def _make_request(engine_client: FakeEngineClient):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(engine_client=engine_client)))


async def _read_streaming_body(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return b"".join(chunks)


def _release_requests(engine_client: FakeEngineClient):
    return [
        req for req in engine_client.requests if req.request_type == RequestType.RELEASE
    ]


def test_normal_stream_releases_worker_after_stream_body_is_consumed():
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    http_client = RecordingHttpClient(
        stream_responses={
            "http://worker/v1/chat/completions": [
                FakeStreamResponse(200, chunks)
            ],
        },
    )
    routes = NormalRoutes(router_type="vllm", http_client=http_client)
    engine_client = FakeEngineClient()
    request = _make_request(engine_client)

    async def run_test():
        response = await routes._handle_request(
            request=request,
            body={
                "model": "demo-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
            headers={"Authorization": "Bearer test"},
            request_text="hello",
            endpoint_path="/v1/chat/completions",
            stream=True,
        )

        assert _release_requests(engine_client) == []
        streamed_body = await _read_streaming_body(response)
        return streamed_body

    streamed_body = asyncio.run(run_test())

    assert streamed_body == b"".join(chunks)
    release_requests = _release_requests(engine_client)
    assert [(req.worker_url, req.worker_rank) for req in release_requests] == [
        ("http://worker", 3)
    ]


def test_normal_non_stream_releases_worker_before_returning_response():
    payload = {"id": "response-id", "choices": [{"text": "hello"}]}
    http_client = RecordingHttpClient(
        post_responses={
            "http://worker/v1/completions": [
                httpx.Response(200, json=payload)
            ],
        },
    )
    routes = NormalRoutes(router_type="vllm", http_client=http_client)
    engine_client = FakeEngineClient()
    request = _make_request(engine_client)

    async def run_test():
        response = await routes._handle_request(
            request=request,
            body={"model": "demo-model", "prompt": "hello", "stream": False},
            headers={"Authorization": "Bearer test"},
            request_text="hello",
            endpoint_path="/v1/completions",
            stream=False,
        )
        return response

    response = asyncio.run(run_test())

    assert response.status_code == 200
    release_requests = _release_requests(engine_client)
    assert [(req.worker_url, req.worker_rank) for req in release_requests] == [
        ("http://worker", 3)
    ]
