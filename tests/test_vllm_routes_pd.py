import asyncio
from types import SimpleNamespace

import httpx
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from smart_router.engine.engine import EngineResponse, RequestType
from smart_router.entrypoints.serve.vllm_routes import VllmRoutes
from smart_router.openai.processing import OpenAIChatProcessor, OpenAIProcessingConfig


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


class FakeChatTokenizer:
    def __init__(self, token_ids=None):
        self.token_ids = token_ids or [101, 202, 303]
        self.chat_template_calls = []

    def apply_chat_template(self, messages, *args, **kwargs):
        self.chat_template_calls.append(
            {"messages": messages, "args": args, "kwargs": kwargs}
        )
        return list(self.token_ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        return list(self.token_ids)


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
    assert b"there" in streamed_body

    release_requests = [
        req for req in engine_client.requests if req.request_type == RequestType.RELEASE
    ]
    assert [(req.worker_url, req.worker_rank) for req in release_requests] == [
        ("http://prefill", 0),
        ("http://decode", 1),
    ]


def test_local_non_stream_chat_uses_internal_completions_and_aggregates_json():
    prefill_payload = {
        "id": "prefill-response",
        "model": "demo-model",
        "kv_transfer_params": {"remote_engine_id": "prefill-engine"},
        "choices": [{"text": "Hello", "token_ids": [501]}],
    }
    decode_chunks = [
        b'data: {"id":"decode-response","model":"demo-model","choices":[{"text":"Hello","token_ids":[501]}]}\n\n',
        b'data: {"id":"decode-response","model":"demo-model","choices":[{"text":" world","token_ids":[502]}]}\n\n',
        b'data: {"id":"decode-response","model":"demo-model","choices":[{"text":"","finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    http_client = RecordingHttpClient(
        post_responses={
            "http://prefill/v1/completions": [
                httpx.Response(200, json=prefill_payload)
            ],
        },
        stream_responses={
            "http://decode/v1/completions": [
                FakeStreamResponse(200, decode_chunks)
            ],
        },
    )
    config = OpenAIProcessingConfig(enabled=True, tokenizer_path="/models/demo")
    processor = OpenAIChatProcessor(config, tokenizer=FakeChatTokenizer())
    routes = VllmRoutes(
        http_client=http_client,
        openai_processing_config=config,
        openai_processor=processor,
    )
    engine_client = FakeEngineClient(
        EngineResponse(
            request_id="req-local-1",
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
        "max_completion_tokens": 8,
    }

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "Hello world"
    assert payload["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }

    prefill_call = http_client.post_calls[0]
    assert prefill_call["url"] == "http://prefill/v1/completions"
    assert prefill_call["json"]["prompt"] == [101, 202, 303]
    assert prefill_call["json"]["return_token_ids"] is True
    assert prefill_call["json"]["max_tokens"] == 1
    assert prefill_call["json"]["stream"] is False

    decode_call = http_client.stream_calls[0]
    assert decode_call["url"] == "http://decode/v1/completions"
    assert decode_call["json"]["prompt"] == [101, 202, 303]
    assert decode_call["json"]["return_token_ids"] is True
    assert decode_call["json"]["max_tokens"] == 8
    assert decode_call["json"]["stream"] is True
    assert decode_call["json"]["kv_transfer_params"] == {
        "remote_engine_id": "prefill-engine"
    }

    schedule_request = engine_client.requests[0]
    assert schedule_request.request_body["prompt"] == [101, 202, 303]
    assert schedule_request.prompt_token_ids == [101, 202, 303]
    assert schedule_request.api_kind == "completions"


def test_local_stream_chat_transforms_completion_sse():
    prefill_payload = {
        "id": "prefill-response",
        "model": "demo-model",
        "kv_transfer_params": {"remote_engine_id": "prefill-engine"},
        "choices": [{"text": "Hi", "token_ids": [601]}],
    }
    decode_chunks = [
        b'data: {"id":"decode-response","model":"demo-model","choices":[{"text":" there","token_ids":[602]}]}\n\n',
        b'data: {"id":"decode-response","model":"demo-model","choices":[{"text":"","finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    http_client = RecordingHttpClient(
        post_responses={
            "http://prefill/v1/completions": [
                httpx.Response(200, json=prefill_payload)
            ],
        },
        stream_responses={
            "http://decode/v1/completions": [
                FakeStreamResponse(200, decode_chunks)
            ],
        },
    )
    config = OpenAIProcessingConfig(enabled=True, tokenizer_path="/models/demo")
    processor = OpenAIChatProcessor(config, tokenizer=FakeChatTokenizer([11, 22]))
    routes = VllmRoutes(
        http_client=http_client,
        openai_processing_config=config,
        openai_processor=processor,
    )
    engine_client = FakeEngineClient(
        EngineResponse(
            request_id="req-local-2",
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
        response = await routes._handle_local_stream_request(
            request=request,
            original_body=body,
            processed=processor.preprocess_chat(body),
            headers={},
            processor=processor,
        )
        return await _read_streaming_body(response)

    streamed_body = asyncio.run(run_test())

    assert b'"object": "chat.completion.chunk"' in streamed_body
    assert b'"role": "assistant"' in streamed_body
    assert b'"content": "Hi"' in streamed_body
    assert b'"content": " there"' in streamed_body
    assert b"data: [DONE]" in streamed_body
    assert http_client.stream_calls[0]["json"]["stream"] is True
