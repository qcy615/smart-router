import json
import logging
import uuid
import asyncio
from typing import Any, Dict

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from smart_router.config.smart_router import UpstreamHTTPClientConfig
from smart_router.engine.engine import EngineRequest, EngineResponse, RequestType
from smart_router.entrypoints.serve.http_client import build_upstream_http_client

logger = logging.getLogger(__name__)


class NormalRoutes:
    """Route handler for normal (non-PD-disaggregation) mode.

    In normal mode, requests are forwarded directly to a single scheduled
    worker without any prefill/decode split or bootstrap injection.
    """

    def __init__(
        self,
        router_type: str = "sglang",
        http_client: Any | None = None,
        http_client_config: UpstreamHTTPClientConfig | None = None,
    ):
        self.router_type = router_type
        self.http_client = http_client or build_upstream_http_client(http_client_config)

    async def close(self) -> None:
        if hasattr(self.http_client, "aclose"):
            await self.http_client.aclose()

    async def models(self, request: Request) -> Response:
        """Proxy /v1/models request to the first available worker."""
        source_urls = getattr(request.app.state, "model_source_urls", [])
        if not source_urls:
            return JSONResponse({"object": "list", "data": []})

        for url in source_urls:
            try:
                response = await self.http_client.get(f"{url.rstrip('/')}/v1/models")
                if response.is_success:
                    return JSONResponse(response.json())
            except Exception:
                logger.warning("Failed to fetch /v1/models from %s", url)
                continue

        return JSONResponse({"error": "No available upstream /v1/models endpoint"}, status_code=503)

    async def completions(self, request: Request) -> Response:
        body = await request.json()
        headers = self._sanitize_headers(request)
        stream = bool(body.get("stream", False))
        request_text = self._extract_request_text(body)
        return await self._handle_request(
            request,
            body=body,
            headers=headers,
            request_text=request_text,
            endpoint_path="/v1/completions",
            stream=stream,
        )

    async def chat_completions(self, request: Request) -> Response:
        body = await request.json()
        headers = self._sanitize_headers(request)
        stream = bool(body.get("stream", False))
        request_text = self._extract_request_text(body)
        return await self._handle_request(
            request,
            body=body,
            headers=headers,
            request_text=request_text,
            endpoint_path="/v1/chat/completions",
            stream=stream,
        )

    async def _handle_request(
            self,
            request: Request,
            body: Dict[str, Any],
            headers: Dict[str, str],
            request_text: str,
            endpoint_path: str,
            stream: bool,
    ) -> Response:
        # 1. Schedule a worker via engine
        worker_url, worker_rank = await self._schedule_worker(request, request_text, headers)
        if worker_url is None:
            return JSONResponse({"error": "No available workers"}, status_code=503)

        logger.debug(
            "Normal mode scheduled worker: url=%s rank=%s endpoint=%s stream=%s",
            worker_url, worker_rank, endpoint_path, stream,
        )

        if stream:
            return await self._handle_stream_request(
                request, body, headers, worker_url, worker_rank, endpoint_path,
            )

        return await self._handle_non_stream_request(
            request, body, headers, worker_url, worker_rank, endpoint_path,
        )

    async def _schedule_worker(
            self, request: Request, request_text: str, headers: Dict[str, str]
    ) -> tuple[str | None, int]:
        """Schedule a single worker via the engine. Returns (url, rank) or (None, -1)."""
        engine_request = EngineRequest(
            request_id=uuid.uuid4().hex,
            identity=request.app.state.engine_client.identity,
            request_text=request_text,
            request_type=RequestType.SCHEDULE,
            headers=headers,
        )
        fut = await request.app.state.engine_client.send_request(engine_request)
        try:
            resp: EngineResponse = await asyncio.wait_for(fut, timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for schedule result")
            return None, -1

        if not resp.worker_url:
            logger.warning("Normal mode schedule result: no available workers")
            return None, -1

        return resp.worker_url, resp.worker_rank

    def _inject_dp_info(
            self,
            body: Dict[str, Any],
            headers: Dict[str, str],
            worker_rank: int,
    ) -> tuple[Dict[str, Any], Dict[str, str]]:
        """Inject DP rank info based on router_type.

        SGLang: injects 'routed_dp_rank' into request body.
        VLLM:   injects 'X-data-parallel-rank' into HTTP headers.
        """
        forward_body = body.copy()
        forward_headers = dict(headers)
        forward_headers["Content-Type"] = "application/json"

        if worker_rank > -1:
            if self.router_type == "sglang":
                forward_body["routed_dp_rank"] = worker_rank
            else:  # vLLM
                forward_headers["X-data-parallel-rank"] = str(worker_rank)

        return forward_body, forward_headers

    async def _handle_non_stream_request(
            self,
            request: Request,
            body: Dict[str, Any],
            headers: Dict[str, str],
            worker_url: str,
            worker_rank: int,
            endpoint_path: str,
    ) -> Response:
        forward_body, forward_headers = self._inject_dp_info(body, headers, worker_rank)

        try:
            response = await self.http_client.post(
                f"{worker_url}{endpoint_path}",
                json=forward_body,
                headers=forward_headers,
            )
        except httpx.RequestError as e:
            logger.error("Normal mode non-stream: connection to %s failed: %s", worker_url, e)
            return JSONResponse(
                {"error": f"Connection to upstream failed: {e}"},
                status_code=502,
            )
        else:
            if not response.is_success:
                return await self._build_upstream_error_response(response)

            return JSONResponse(response.json(), status_code=response.status_code)
        finally:
            await self._decrement_worker(request, worker_url, worker_rank)

    async def _handle_stream_request(
            self,
            request: Request,
            body: Dict[str, Any],
            headers: Dict[str, str],
            worker_url: str,
            worker_rank: int,
            endpoint_path: str,
    ) -> Response:
        forward_body, forward_headers = self._inject_dp_info(body, headers, worker_rank)

        async def stream_response():
            try:
                async with self.http_client.stream(
                        "POST",
                        f"{worker_url}{endpoint_path}",
                        json=forward_body,
                        headers=forward_headers,
                ) as response:
                    if not response.is_success:
                        error_body = await response.aread()
                        error_text = error_body.decode(errors="replace")
                        logger.error(
                            "Normal mode stream error status=%s body=%s",
                            response.status_code, error_text,
                        )
                        yield f"data: {json.dumps({'error': f'Server error {response.status_code}'})}\n\n".encode(
                            "utf-8")
                        return

                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        yield chunk
            except httpx.RequestError as e:
                logger.error("Normal mode stream: connection to %s failed: %s", worker_url, e)
                yield f"data: {json.dumps({'error': f'Connection to upstream failed: {e}'})}\n\n".encode("utf-8")
            finally:
                await self._decrement_worker(request, worker_url, worker_rank)

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async def _decrement_worker(self, request: Request, url: str, rank: int):
        """Send a RELEASE request to decrement worker load."""
        logger.debug("Normal mode releasing worker: url=%s rank=%s", url, rank)
        engine_request = EngineRequest(
            request_id=uuid.uuid4().hex,
            identity=request.app.state.engine_client.identity,
            request_type=RequestType.RELEASE,
            worker_rank=rank,
            worker_url=url,
        )
        await request.app.state.engine_client.send_request(engine_request)

    async def _build_upstream_error_response(self, response: httpx.Response) -> JSONResponse:
        error_body = await response.aread()
        error_text = error_body.decode(errors="replace")
        logger.error("Upstream error status=%s body=%s", response.status_code, error_text)
        return JSONResponse(
            {"error": f"Server error {response.status_code}: {error_text}"},
            status_code=response.status_code,
        )

    def _sanitize_headers(self, request: Request) -> Dict[str, str]:
        headers = dict(request.headers)
        headers.pop("content-length", None)
        headers.pop("content-type", None)
        headers.pop("host", None)
        return headers

    def _extract_request_text(self, body: Dict[str, Any]) -> str:
        if "messages" in body:
            return str(body["messages"])
        if "prompt" in body:
            return str(body["prompt"])
        if "text" in body:
            text = body["text"]
            return text if isinstance(text, str) else str(text)
        return json.dumps(body, ensure_ascii=False)
