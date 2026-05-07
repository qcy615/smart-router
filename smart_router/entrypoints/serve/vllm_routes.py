import copy
import json
import logging
import time
import uuid
import asyncio
from typing import Any, Dict, Optional

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from smart_router.engine.engine import EngineRequest, EngineResponse, RequestType
from smart_router.openai.processing import (
    DONE,
    CompletionDelta,
    CompletionSSEDecoder,
    OpenAIChatProcessor,
    OpenAIProcessingConfig,
    OpenAIProcessingError,
    PreprocessedChatRequest,
    TokenInfo,
)


logger = logging.getLogger(__name__)

class VllmRoutes:
    def __init__(
        self,
        http_client: Any | None = None,
        openai_processing_config: OpenAIProcessingConfig | None = None,
        openai_processor: OpenAIChatProcessor | None = None,
    ):
        # Shared async HTTP client for forwarding requests.
        self.http_client = http_client or httpx.AsyncClient(timeout=60 * 60.0)
        self.openai_processing_config = openai_processing_config
        self._openai_processor = openai_processor

    async def close(self) -> None:
        if hasattr(self.http_client, "aclose"):
            await self.http_client.aclose()

    async def models(self, request: Request) -> Response:
        headers = self._sanitize_headers(request)
        source_urls = getattr(request.app.state, "model_source_urls", [])
        if not source_urls:
            return JSONResponse({"object": "list", "data": []})

        responses = await asyncio.gather(
            *[self._fetch_models_from_source(url, headers) for url in source_urls]
        )

        successful_sources = 0
        models_by_id: Dict[str, Dict[str, Any]] = {}
        for response in responses:
            if response is None:
                continue

            successful_sources += 1
            for model in response:
                model_id = model.get("id")
                if not isinstance(model_id, str) or not model_id:
                    continue
                if model_id not in models_by_id:
                    models_by_id[model_id] = model
                    continue

                existing = models_by_id[model_id]
                for key, value in model.items():
                    if key not in existing or existing[key] in (None, "", [], {}):
                        existing[key] = value
                if isinstance(model.get("max_model_len"), int):
                    existing_max_model_len = existing.get("max_model_len")
                    if not isinstance(existing_max_model_len, int):
                        existing_max_model_len = 0
                    existing["max_model_len"] = max(
                        existing_max_model_len,
                        model["max_model_len"],
                    )

        if successful_sources == 0:
            return JSONResponse(
                {"error": "No available upstream /v1/models endpoint"},
                status_code=503,
            )

        return JSONResponse(
            {
                "object": "list",
                "data": [models_by_id[key] for key in sorted(models_by_id)],
            }
        )

    async def _fetch_models_from_source(
        self,
        source_url: str,
        headers: Dict[str, str],
    ) -> Optional[list[Dict[str, Any]]]:
        try:
            response = await self.http_client.get(
                f"{source_url.rstrip('/')}/v1/models",
                headers=headers,
            )
        except Exception:
            logger.exception("Failed to fetch /v1/models from %s", source_url)
            return None

        if not response.is_success:
            logger.warning(
                "Upstream /v1/models request failed url=%s status=%s",
                source_url,
                response.status_code,
            )
            return None

        try:
            payload = response.json()
        except Exception:
            logger.exception("Upstream /v1/models returned invalid JSON from %s", source_url)
            return None

        return self._extract_model_cards(payload, source_url)

    def _extract_model_cards(
        self,
        payload: Any,
        source_url: str,
    ) -> list[Dict[str, Any]]:
        if isinstance(payload, dict):
            raw_models = payload.get("data", [])
        elif isinstance(payload, list):
            raw_models = payload
        else:
            logger.warning("Unexpected /v1/models payload type from %s", source_url)
            return []

        models: list[Dict[str, Any]] = []
        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                continue

            model_id = raw_model.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue

            model = copy.deepcopy(raw_model)
            model.setdefault("id", model_id)
            model.setdefault("object", "model")
            model.setdefault("owned_by", "smart-router")
            models.append(model)

        return models


    async def completions(self, request: Request) -> Response:
        body = await request.json()
        headers = self._sanitize_headers(request)
        stream = bool(body.get("stream", False))
        request_text = self._extract_request_text(body)
        return await self._handle_pd_request(
            request,
            body=body,
            headers=headers,
            request_text=request_text,
            endpoint_path="/v1/completions",
            api_kind="completions",
            stream=stream,
        )

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        headers = self._sanitize_headers(request)
        stream = bool(body.get("stream", False))
        processor_or_response = self._get_openai_chat_processor(request)
        if isinstance(processor_or_response, Response):
            return processor_or_response
        if processor_or_response is not None:
            return await self._handle_local_chat_completions(
                request=request,
                body=body,
                headers=headers,
                stream=stream,
                processor=processor_or_response,
            )

        request_text = self._extract_request_text(body)
        return await self._handle_pd_request(
            request,
            body=body,
            headers=headers,
            request_text=request_text,
            endpoint_path="/v1/chat/completions",
            api_kind="chat",
            stream=stream,
        )

    async def _handle_pd_request(
        self,
        request: Request,
        body: Dict[str, Any],
        headers: Dict[str, str],
        request_text: str,
        endpoint_path: str,
        api_kind: str,
        stream: bool,
    ) -> Response:
        logger.debug(
            "PD request start api_kind=%s stream=%s endpoint=%s",
            api_kind,
            stream,
            endpoint_path,
        )
        if stream:
            return await self._handle_stream_request(
                request,
                body=body,
                headers=headers,
                request_text=request_text,
                endpoint_path=endpoint_path,
                api_kind=api_kind,
            )
        return await self._handle_non_stream_request(
            request=request,
            body=body,
            headers=headers,
            request_text=request_text,
            endpoint_path=endpoint_path,
            api_kind=api_kind,
        )

    async def _prepare_pd_context(
        self,
        request: Request,
        body: Dict[str, Any],
        headers: Dict[str, str],
        request_text: str,
        endpoint_path: str,
        api_kind: str,
        prompt_token_ids: list[int] | None = None,
    ) -> Dict[str, Any] | Response:
        engine_request = EngineRequest(
            request_id=uuid.uuid4().hex,
            identity=request.app.state.engine_client.identity,
            request_text=request_text,
            request_type=RequestType.SCHEDULE,
            headers=headers,
            request_body=body,
            api_kind=api_kind,
            prompt_token_ids=prompt_token_ids or [],
        )
        # send request to engine using engine_client
        fut: asyncio.Future = await request.app.state.engine_client.send_request(engine_request)
        try:
            resp: EngineResponse = await asyncio.wait_for(fut, timeout=5.0)

        except asyncio.TimeoutError:
            logger.error("time out for get schedule result")
            return JSONResponse("time out for selecting workers", status_code=503)
        
        prefill_url, prefill_rank = resp.prefill_url, resp.prefill_rank
        decode_url, decode_rank = resp.decode_url, resp.decode_rank
        
   
        if prefill_url is None:
            return JSONResponse("No available prefill workers", status_code=503)


        request_id = self._generate_vllm_request_id(prefill_url, decode_url)
        logger.debug(
            "Generated vLLM request id=%s prefill=%s decode=%s",
            request_id,
            prefill_url,
            decode_url,
        )

        prefill_response: httpx.Response | None = None
        try:
            prefill_body = self._get_prefill_body(body)
            prefill_headers = self._get_prefill_headers(headers, request_id, prefill_rank)
            logger.debug(
                "Prepared prefill request id=%s body=%s headers=%s",
                request_id,
                prefill_body,
                self._mask_headers_for_log(prefill_headers),
            )
            logger.debug(
                "vLLM Stage1 Prefill start id=%s url=%s endpoint=%s",
                request_id,
                prefill_url,
                endpoint_path,
            )
            prefill_response = await self.http_client.post(
                f"{prefill_url}{endpoint_path}",
                json=prefill_body,
                headers=prefill_headers,
            )
        finally:
            await self._decrement_worker(request, prefill_url, prefill_rank)
            logger.debug("vLLM Stage1 Prefill finish id=%s", request_id)

        if not prefill_response.is_success:
            return await self._build_upstream_error_response("Prefill", prefill_response)

        return {
            "decode_url": decode_url,
            "decode_rank": decode_rank,
            "request_id": request_id,
            "prefill_response": prefill_response,
        }

    def _get_openai_chat_processor(
        self, request: Request
    ) -> OpenAIChatProcessor | Response | None:
        config = getattr(request.app.state, "openai_processing_config", None)
        if config is None:
            config = self.openai_processing_config
        if config is None and self._openai_processor is not None:
            config = self._openai_processor.config
        if config is None or not config.enabled:
            return None

        if self._openai_processor is not None and self._openai_processor.config == config:
            return self._openai_processor

        try:
            self._openai_processor = OpenAIChatProcessor(config)
        except RuntimeError as exc:
            return JSONResponse(
                {"error": {"message": str(exc), "type": "server_error"}},
                status_code=500,
            )
        return self._openai_processor

    async def _handle_local_chat_completions(
        self,
        request: Request,
        body: Dict[str, Any],
        headers: Dict[str, str],
        stream: bool,
        processor: OpenAIChatProcessor,
    ) -> Response:
        try:
            processed = processor.preprocess_chat(body)
        except OpenAIProcessingError as exc:
            return JSONResponse(
                {
                    "error": {
                        "message": exc.message,
                        "type": "invalid_request_error",
                    }
                },
                status_code=exc.status_code,
            )

        if stream:
            return await self._handle_local_stream_request(
                request=request,
                original_body=body,
                processed=processed,
                headers=headers,
                processor=processor,
            )
        return await self._handle_local_non_stream_request(
            request=request,
            original_body=body,
            processed=processed,
            headers=headers,
            processor=processor,
        )

    async def _handle_local_non_stream_request(
        self,
        request: Request,
        original_body: Dict[str, Any],
        processed: PreprocessedChatRequest,
        headers: Dict[str, str],
        processor: OpenAIChatProcessor,
    ) -> Response:
        context_or_response = await self._prepare_pd_context(
            request=request,
            body=processed.completion_body,
            headers=headers,
            request_text=processed.request_text,
            endpoint_path="/v1/completions",
            api_kind="completions",
            prompt_token_ids=processed.prompt_token_ids,
        )
        if isinstance(context_or_response, Response):
            return context_or_response

        decode_url: str = context_or_response["decode_url"]
        decode_rank: int = context_or_response["decode_rank"]
        request_id: str = context_or_response["request_id"]
        prefill_response: httpx.Response = context_or_response["prefill_response"]
        prefill_response_json = prefill_response.json()
        builder = processor.create_chat_builder(processed, original_body)
        self._apply_prefill_delta(builder, processor, prefill_response_json)

        kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
        decode_body = self._get_local_decode_body(processed.completion_body, kv_transfer_params)
        decode_headers = self._get_decode_headers(headers, request_id, decode_rank)
        first_prefill_token = processor.extract_first_completion_token(prefill_response_json)

        try:
            logger.debug(
                "vLLM Stage2 Decode start id=%s url=%s endpoint=/v1/completions mode=local-non-stream",
                request_id,
                decode_url,
            )
            async with self.http_client.stream(
                "POST",
                f"{decode_url}/v1/completions",
                json=decode_body,
                headers=decode_headers,
            ) as decode_response_stream:
                if not decode_response_stream.is_success:
                    return await self._build_stream_upstream_error_response(
                        "Decode",
                        decode_response_stream,
                    )

                await self._consume_local_decode_stream(
                    decode_response_stream=decode_response_stream,
                    builder=builder,
                    processor=processor,
                    first_prefill_token=first_prefill_token,
                )
        finally:
            await self._decrement_worker(request, decode_url, decode_rank)
            logger.debug("vLLM Stage2 Decode finish id=%s mode=local-non-stream", request_id)

        return JSONResponse(builder.to_response())

    async def _handle_local_stream_request(
        self,
        request: Request,
        original_body: Dict[str, Any],
        processed: PreprocessedChatRequest,
        headers: Dict[str, str],
        processor: OpenAIChatProcessor,
    ) -> Response:
        context_or_response = await self._prepare_pd_context(
            request=request,
            body=processed.completion_body,
            headers=headers,
            request_text=processed.request_text,
            endpoint_path="/v1/completions",
            api_kind="completions",
            prompt_token_ids=processed.prompt_token_ids,
        )
        if isinstance(context_or_response, Response):
            return context_or_response

        decode_url = context_or_response["decode_url"]
        decode_rank = context_or_response["decode_rank"]
        request_id: str = context_or_response["request_id"]
        prefill_response: httpx.Response = context_or_response["prefill_response"]

        async def stream_response():
            builder = processor.create_chat_builder(processed, original_body)
            prefill_response_json = prefill_response.json()
            for payload in self._apply_prefill_delta(
                builder,
                processor,
                prefill_response_json,
            ):
                yield self._sse_bytes(payload)

            kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
            decode_body = self._get_local_decode_body(
                processed.completion_body,
                kv_transfer_params,
            )
            decode_headers = self._get_decode_headers(headers, request_id, decode_rank)
            first_prefill_token = processor.extract_first_completion_token(prefill_response_json)

            try:
                logger.debug(
                    "vLLM Stage2 Decode start id=%s url=%s endpoint=/v1/completions mode=local-stream",
                    request_id,
                    decode_url,
                )
                async with self.http_client.stream(
                    "POST",
                    f"{decode_url}/v1/completions",
                    json=decode_body,
                    headers=decode_headers,
                ) as decode_response_stream:
                    if not decode_response_stream.is_success:
                        error_text = await self._read_stream_error_text(decode_response_stream)
                        yield (
                            f"data: {json.dumps({'error': f'Decode server error '
                            f'{decode_response_stream.status_code}: {error_text}'})}\n\n"
                        ).encode("utf-8")
                        return

                    async for payload in self._iter_local_decode_payloads(
                        decode_response_stream=decode_response_stream,
                        builder=builder,
                        processor=processor,
                        first_prefill_token=first_prefill_token,
                    ):
                        yield self._sse_bytes(payload)
                    yield b"data: [DONE]\n\n"
            finally:
                await self._decrement_worker(request, decode_url, decode_rank)
                logger.debug("vLLM Stage2 Decode finish id=%s mode=local-stream", request_id)

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async def _consume_local_decode_stream(
        self,
        decode_response_stream: Any,
        builder: Any,
        processor: OpenAIChatProcessor,
        first_prefill_token: TokenInfo,
    ) -> None:
        async for _ in self._iter_local_decode_payloads(
            decode_response_stream=decode_response_stream,
            builder=builder,
            processor=processor,
            first_prefill_token=first_prefill_token,
        ):
            pass

    async def _iter_local_decode_payloads(
        self,
        decode_response_stream: Any,
        builder: Any,
        processor: OpenAIChatProcessor,
        first_prefill_token: TokenInfo,
    ):
        decoder = CompletionSSEDecoder(processor)
        seen_first_non_empty_token = False
        done = False
        async for chunk in decode_response_stream.aiter_bytes():
            if not chunk:
                continue
            for event in decoder.feed(chunk):
                if event is DONE:
                    done = True
                    break
                delta = event
                if not isinstance(delta, CompletionDelta):
                    continue
                should_skip, seen_first_non_empty_token = self._should_skip_decode_delta(
                    delta,
                    seen_first_non_empty_token,
                    first_prefill_token,
                    processor,
                )
                if should_skip:
                    continue
                for payload in builder.apply_delta(delta):
                    yield payload
            if done:
                break

        if not done:
            for event in decoder.close():
                if event is DONE:
                    break
                if not isinstance(event, CompletionDelta):
                    continue
                should_skip, seen_first_non_empty_token = self._should_skip_decode_delta(
                    event,
                    seen_first_non_empty_token,
                    first_prefill_token,
                    processor,
                )
                if should_skip:
                    continue
                for payload in builder.apply_delta(event):
                    yield payload

        for payload in builder.finish_chunks():
            yield payload

    def _apply_prefill_delta(
        self,
        builder: Any,
        processor: OpenAIChatProcessor,
        prefill_response_json: Dict[str, Any],
    ) -> list[dict[str, Any]]:
        prefill_delta = processor.completion_delta_from_response_json(prefill_response_json)
        if prefill_delta is None:
            return []
        prefill_delta.finish_reason = None
        return builder.apply_delta(prefill_delta)

    def _get_local_decode_body(
        self,
        completion_body: Dict[str, Any],
        kv_transfer_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        decode_body = copy.deepcopy(completion_body)
        decode_body["kv_transfer_params"] = kv_transfer_params
        decode_body["stream"] = True
        return decode_body

    def _should_skip_decode_delta(
        self,
        delta: CompletionDelta,
        seen_first_non_empty_token: bool,
        first_prefill_token: TokenInfo,
        processor: OpenAIChatProcessor,
    ) -> tuple[bool, bool]:
        if seen_first_non_empty_token or not delta.has_non_empty_token():
            return False, seen_first_non_empty_token
        seen_first_non_empty_token = True
        return processor.delta_matches_token(delta, first_prefill_token), seen_first_non_empty_token

    def _sse_bytes(self, payload: Dict[str, Any]) -> bytes:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

    async def _build_stream_upstream_error_response(
        self,
        stage: str,
        response: Any,
    ) -> JSONResponse:
        error_text = await self._read_stream_error_text(response)
        logger.error(
            "%s server error status=%s body=%s",
            stage,
            response.status_code,
            error_text,
        )
        return JSONResponse(
            {"error": f"{stage} server error {response.status_code}: {error_text}"},
            status_code=500,
        )

    async def _read_stream_error_text(self, response: Any) -> str:
        error_body = await response.aread()
        return error_body.decode(errors="replace")
    
    async def _decrement_worker(self, request: Request, url: str, rank: int):
        engine_request = EngineRequest(
            request_id=uuid.uuid4().hex,
            identity=request.app.state.engine_client.identity,
            request_type=RequestType.RELEASE,
            worker_rank=rank,
            worker_url=url,
        )
        await  request.app.state.engine_client.send_request(engine_request)
        

    async def _handle_non_stream_request(
        self,
        request: Request,
        body: Dict[str, Any],
        headers: Dict[str, str],
        request_text: str,
        endpoint_path: str,
        api_kind: str,
    ) -> Response:
        _ = api_kind
        context_or_response = await self._prepare_pd_context(
            request=request,
            body=body,
            headers=headers,
            request_text=request_text,
            endpoint_path=endpoint_path,
            api_kind=api_kind,
        )
        if isinstance(context_or_response, Response):
            return context_or_response

        decode_url: str = context_or_response["decode_url"]
        decode_rank: int = context_or_response["decode_rank"]
        request_id: str = context_or_response["request_id"]
        prefill_response: httpx.Response = context_or_response["prefill_response"]

        prefill_response_json = prefill_response.json()
        kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
        logger.debug(
            "vLLM Stage1 Prefill response id=%s status=%s kv_transfer_params=%s",
            request_id,
            prefill_response.status_code,
            "present" if kv_transfer_params else "empty",
        )
        decode_body = copy.deepcopy(body)
        decode_body["kv_transfer_params"] = kv_transfer_params
        decode_headers = self._get_decode_headers(headers, request_id, decode_rank)

        try:
            logger.debug(
                "vLLM Stage2 Decode start id=%s url=%s endpoint=%s mode=non-stream",
                request_id,
                decode_url,
                endpoint_path,
            )
            decode_response = await self.http_client.post(
                f"{decode_url}{endpoint_path}",
                json=decode_body,
                headers=decode_headers,
            )
        finally:
            await self._decrement_worker(request, decode_url, decode_rank)
            logger.debug("vLLM Stage2 Decode finish id=%s mode=non-stream", request_id)

        if not decode_response.is_success:
            return await self._build_upstream_error_response("Decode", decode_response)

        logger.debug(
            "vLLM Stage2 Decode response id=%s status=%s mode=non-stream",
            request_id,
            decode_response.status_code,
        )
        return JSONResponse(decode_response.json(), status_code=decode_response.status_code)

    async def _handle_stream_request(
        self,
        request: Request,
        body: Dict[str, Any],
        headers: Dict[str, str],
        request_text: str,
        endpoint_path: str,
        api_kind: str,
    ) -> Response:
        context_or_response = await self._prepare_pd_context(
            request=request,
            body=body,
            headers=headers,
            request_text=request_text,
            endpoint_path=endpoint_path,
            api_kind=api_kind,
        )
        if isinstance(context_or_response, Response):
            return context_or_response

        decode_url = context_or_response["decode_url"]
        decode_rank = context_or_response["decode_rank"]
        request_id: str = context_or_response["request_id"]
        prefill_response: httpx.Response = context_or_response["prefill_response"]

        async def stream_response():
            try:
                prefill_response_json = prefill_response.json()

                first_chunk = self._build_prefill_first_token_chunk(
                    api_kind=api_kind,
                    prefill_response_json=prefill_response_json,
                )
                if first_chunk is not None:
                    logger.debug("Prefill first token emitted id=%s", request_id)
                    yield first_chunk
                else:
                    logger.debug("Prefill first token missing id=%s", request_id)
                first_prefill_token = self._extract_first_token_info(
                    api_kind,
                    prefill_response_json,
                )

                kv_transfer_params = prefill_response_json.get("kv_transfer_params", {})
                logger.debug(
                    "vLLM Stage1 Prefill response id=%s status=%s kv_transfer_params=%s",
                    request_id,
                    prefill_response.status_code,
                    "present" if kv_transfer_params else "empty",
                )
                decode_body = copy.deepcopy(body)
                decode_body["kv_transfer_params"] = kv_transfer_params
                decode_headers = self._get_decode_headers(headers, request_id, decode_rank)

                logger.debug(
                    "vLLM Stage2 Decode start id=%s url=%s endpoint=%s mode=stream",
                    request_id,
                    decode_url,
                    endpoint_path,
                )
                async with self.http_client.stream(
                    "POST",
                    f"{decode_url}{endpoint_path}",
                    json=decode_body,
                    headers=decode_headers,
                ) as decode_response_stream:
                    if not decode_response_stream.is_success:
                        error_body = await decode_response_stream.aread()
                        error_text = error_body.decode(errors="replace")
                        logger.error(
                            "vLLM Stage2 Decode response id=%s status=%s mode=stream body=%s",
                            request_id,
                            decode_response_stream.status_code,
                            error_text,
                        )
                        yield (
                            f"data: {json.dumps({'error': f'Decode server error '
                            f'{decode_response_stream.status_code}: {error_text}'})}\n\n"
                        )
                        return

                    seen_first_non_empty_token = False
                    async for chunk in decode_response_stream.aiter_bytes():
                        if not chunk:
                            continue

                        if not seen_first_non_empty_token:
                            should_skip, seen_first_non_empty_token = (
                                self._should_skip_raw_decode_chunk(
                                    chunk,
                                    api_kind,
                                    first_prefill_token,
                                )
                            )
                            if should_skip:
                                logger.debug(
                                    "Decode duplicate first non-empty token skipped id=%s",
                                    request_id,
                                )
                                continue

                        yield chunk
            finally:
                await self._decrement_worker(request, decode_url, decode_rank)
                logger.debug("vLLM Stage2 Decode finish id=%s mode=stream", request_id)

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async def _build_upstream_error_response(
        self, stage: str, response: httpx.Response
    ) -> JSONResponse:
        error_body = await response.aread()
        error_text = error_body.decode(errors="replace")
        logger.error(
            "%s server error status=%s body=%s",
            stage,
            response.status_code,
            error_text,
        )
        return JSONResponse(
            {
                "error": f"{stage} server error {response.status_code}: {error_text}"
            },
            status_code=500,
        )

    def _sanitize_headers(self, request: Request) -> Dict[str, str]:
        headers = dict(request.headers)
        headers.pop("content-length", None)
        headers.pop("host", None)
        return headers

    def _mask_headers_for_log(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        masked_headers = copy.deepcopy(headers)
        for key in list(masked_headers.keys()):
            if key.lower() in {"authorization", "cookie", "set-cookie", "proxy-authorization"}:
                masked_headers[key] = "***"
        return masked_headers

    def _extract_request_text(self, body: Dict[str, Any]) -> str:
        if "messages" in body:
            return str(body["messages"])
        if "prompt" in body:
            return str(body["prompt"])
        return json.dumps(body, ensure_ascii=False)

    def _build_prefill_first_token_chunk(
        self, api_kind: str, prefill_response_json: Dict[str, Any]
    ) -> Optional[bytes]:
        """Build one SSE chunk from prefill non-streaming response."""
        choices = prefill_response_json.get("choices") or []
        if not choices:
            return None

        first_choice = choices[0] or {}
        if api_kind == "chat":
            message = first_choice.get("message") or {}
            token_text = message.get("content")
            if not token_text:
                return None
            payload = {
                "id": prefill_response_json.get("id", f"prefill-{uuid.uuid4()}"),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": prefill_response_json.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": token_text},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

        token_text = first_choice.get("text")
        if not token_text:
            return None
        payload = {
            "id": prefill_response_json.get("id", f"prefill-{uuid.uuid4()}"),
            "object": "text_completion",
            "created": int(time.time()),
            "model": prefill_response_json.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "text": token_text,
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")

    def _extract_first_token_info(
        self,
        api_kind: str,
        payload: Dict[str, Any],
    ) -> TokenInfo:
        choices = payload.get("choices") or []
        if not choices:
            return TokenInfo()
        choice = choices[0] or {}
        text = self._extract_choice_text(choice, api_kind)
        token_ids = self._extract_choice_token_ids(choice)
        return TokenInfo(
            text=text or None,
            token_ids=token_ids[:1],
        )

    def _should_skip_raw_decode_chunk(
        self,
        chunk: bytes,
        api_kind: str,
        first_prefill_token: TokenInfo,
    ) -> tuple[bool, bool]:
        decode_token = self._first_token_from_raw_chunk(chunk, api_kind)
        if decode_token is None:
            return False, False
        if first_prefill_token.token_ids and decode_token.token_ids:
            return first_prefill_token.token_ids[0] == decode_token.token_ids[0], True
        return bool(first_prefill_token.text and decode_token.text == first_prefill_token.text), True

    def _first_token_from_raw_chunk(
        self,
        chunk: bytes,
        api_kind: str,
    ) -> TokenInfo | None:
        try:
            text = chunk.decode("utf-8")
        except Exception:
            return None

        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue

            data_text = line[5:].strip()
            if not data_text or data_text == "[DONE]":
                continue

            try:
                payload = json.loads(data_text)
            except Exception:
                continue

            choices = payload.get("choices") or []
            if not choices:
                continue
            choice = choices[0] or {}
            token_text = self._extract_choice_text(choice, api_kind)
            token_ids = self._extract_choice_token_ids(choice)
            if token_text or token_ids:
                return TokenInfo(text=token_text or None, token_ids=token_ids[:1])
        return None

    def _extract_choice_text(self, choice: Dict[str, Any], api_kind: str) -> Optional[str]:
        if api_kind == "chat":
            delta = choice.get("delta") or {}
            token_text = delta.get("content")
            if token_text is None:
                message = choice.get("message") or {}
                token_text = message.get("content")
            if token_text is None:
                token_text = choice.get("text")
        else:
            token_text = choice.get("text")
            if token_text is None:
                delta = choice.get("delta") or {}
                token_text = delta.get("content")
        return token_text if isinstance(token_text, str) and token_text else None

    def _extract_choice_token_ids(self, choice: Dict[str, Any]) -> list[int]:
        for key in ("token_ids", "output_token_ids"):
            value = choice.get(key)
            if isinstance(value, list):
                return [int(token_id) for token_id in value]
        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            value = logprobs.get("token_ids")
            if isinstance(value, list):
                return [int(token_id) for token_id in value]
        return []

    def _chunk_has_non_empty_token(self, chunk: bytes, api_kind: str) -> bool:
        try:
            text = chunk.decode("utf-8")
        except Exception:
            return False

        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue

            data_text = line[5:].strip()
            if not data_text or data_text == "[DONE]":
                continue

            try:
                payload = json.loads(data_text)
            except Exception:
                continue

            choices = payload.get("choices") or []
            if not choices:
                continue

            choice = choices[0] or {}
            token_text: Optional[str]
            if api_kind == "chat":
                delta = choice.get("delta") or {}
                token_text = delta.get("content")
                if token_text is None:
                    token_text = choice.get("text")
            else:
                token_text = choice.get("text")
                if token_text is None:
                    delta = choice.get("delta") or {}
                    token_text = delta.get("content")

            if isinstance(token_text, str) and token_text:
                return True

        return False

    def _get_prefill_body(self, body: Dict[str, Any]) -> Dict[str, Any]:
        new_body = copy.deepcopy(body)
        # Prepare prefill request (max_tokens=1 to force prefill-only mode)
        new_body["max_tokens"] = 1
        if new_body.get("min_tokens") is not None:
            new_body["min_tokens"] = 1
        if new_body.get("max_completion_tokens") is not None:
            new_body["max_completion_tokens"] = 1

        # Force non-streaming for prefill to get JSON response with kv_transfer_params
        new_body["stream"] = False
        # Remove stream_options since we're setting stream=false
        if "stream_options" in new_body:
            del new_body["stream_options"]

        new_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        return new_body

    def _get_prefill_headers(self, headers: Dict[str, str], request_id: str, rank: int) -> Dict[str, Any]:
        new_headers = copy.deepcopy(headers)
        new_headers["Content-Type"] = "application/json"
        new_headers["X-Request-Id"] = request_id
        if rank > -1:
            new_headers["X-data-parallel-rank"] = str(rank)
        return new_headers

    def _get_decode_headers(self, headers: Dict[str, str], request_id: str, rank: int) -> Dict[str, Any]:
        """Prepare headers for decode request, including vLLM-specific headers."""
        new_headers = copy.deepcopy(headers)
        new_headers["Content-Type"] = "application/json"
        new_headers["X-Request-Id"] = request_id
        if rank > -1:
            new_headers["X-data-parallel-rank"] = str(rank)
        return new_headers

    def _generate_vllm_request_id(self, prefill_url: str, decode_url: str) -> str:
        """Generate a unique request ID for vLLM based on prefill and decode addresses."""
        prefill_addr = prefill_url.replace("http://", "").replace("https://", "")
        decode_addr = decode_url.replace("http://", "").replace("https://", "")
        return f"___prefill_addr_{prefill_addr}___decode_addr_{decode_addr}_{uuid.uuid4()}"
