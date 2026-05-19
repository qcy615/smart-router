import copy
import ipaddress
import json
import logging
import random
import urllib.parse
import uuid
from typing import Any, Dict,Tuple, List, Optional
from contextlib import AsyncExitStack
import asyncio
import httpx

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from smart_router.engine.engine import EngineRequest, EngineResponse, RequestType
from smart_router.config.smart_router import SmartRouterConfig
from smart_router.entrypoints.serve.http_client import build_upstream_http_client

logger = logging.getLogger(__name__)

SGLANG_DEFAULT_BOOTSTRAP_PORT = 8998


def _maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


class SGLangRoutes:
    """SGLang PD-disaggregation route handler.

    In SGLang's PD disaggregation mode, the prefill and decode requests are
    sent in parallel (via asyncio.gather). Both receive the same request body
    augmented with bootstrap_host, bootstrap_port, and kv_bootstrap_room, which
    allow the decode worker to retrieve KV cache from the prefill worker.
    """

    def __init__(self, config: SmartRouterConfig, http_client: Any | None = None):
        self.http_client = http_client or build_upstream_http_client(
            getattr(config, "upstream_http_client_config", None)
        )
        # bootstrap_ports: one port per prefill URL, for KV cache bootstrap
        self.bootstrap_ports = config.prefill_bootstrap_ports or []
        self.prefill_urls = config.prefill_urls
        self.prefill_intra_dp_size = config.prefill_intra_dp_size
        self.decode_intra_dp_size = config.decode_intra_dp_size

    async def close(self) -> None:
        if hasattr(self.http_client, "aclose"):
            await self.http_client.aclose()

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

    async def generate(self, request: Request) -> Response:
        body = await request.json()
        headers = self._sanitize_headers(request)
        stream = bool(body.get("stream", False))
        request_text = self._extract_request_text(body)
        return await self._handle_pd_request(
            request,
            body=body,
            headers=headers,
            request_text=request_text,
            endpoint_path="/generate",
            api_kind="generate",
            stream=stream,
        )

    async def get_models(self, request: Request) -> Response:
        """Proxy /v1/models request to a prefill server."""
        if not self.prefill_urls:
            return JSONResponse(
                {"error": "No prefill servers configured"}, status_code=503
            )
        prefill_server = self.prefill_urls[0]
        try:
            response = await self.http_client.get(f"{prefill_server}/v1/models")
            if response.status_code != 200:
                return JSONResponse(
                    {"error": f"Prefill server error: Status {response.status_code}"},
                    status_code=response.status_code,
                )
            return JSONResponse(response.json())
        except Exception as e:
            logger.error("Failed to get models from prefill server: %s", e)
            return JSONResponse({"error": str(e)}, status_code=500)

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
            "SGLang PD request start api_kind=%s stream=%s endpoint=%s",
            api_kind,
            stream,
            endpoint_path,
        )
        # 1. Schedule prefill and decode workers via engine
        schedule_result = await self._schedule_workers(request, request_text, headers)
        if isinstance(schedule_result, Response):
            return schedule_result

        prefill_url = schedule_result["prefill_url"]
        prefill_rank = schedule_result["prefill_rank"]
        decode_url = schedule_result["decode_url"]
        decode_rank = schedule_result["decode_rank"]

        logger.debug(
            "SGLang PD scheduled workers: prefill_url=%s prefill_rank=%s decode_url=%s decode_rank=%s",
            prefill_url, prefill_rank, decode_url, decode_rank,
        )

        # 2. Build prefill and decode req request with bootstrap info
        prefill_request, decode_request = self._build_bootstrap_request(
            body, prefill_url, prefill_rank, decode_rank, self.prefill_intra_dp_size
        )
        logger.debug(
            "SGLang prefill_request, decode_request: prefill_request=%s decode_request=%s",
            prefill_request, decode_request
        )
        # 3. Dispatch based on stream flag
        if stream:
            return await self._handle_stream_request(
                request,
                prefill_request=prefill_request,
                decode_request=decode_request,
                prefill_url=prefill_url,
                prefill_rank=prefill_rank,
                decode_url=decode_url,
                decode_rank=decode_rank,
                endpoint_path=endpoint_path,
                api_kind=api_kind,
            )
        else:
            return await self._handle_non_stream_request(
                request,
                prefill_request=prefill_request,
                decode_request=decode_request,
                prefill_url=prefill_url,
                prefill_rank=prefill_rank,
                decode_url=decode_url,
                decode_rank=decode_rank,
                endpoint_path=endpoint_path,
            )

    async def _schedule_workers(
            self, request: Request, request_text: str, headers: Dict[str, str]
    ) -> Dict[str, Any] | Response:
        """Schedule prefill and decode workers via the engine."""
        logger.debug("SGLang PD scheduling workers for request")
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
            return JSONResponse({"error": "Timeout selecting workers"}, status_code=503)

        if resp.prefill_url is None:
            logger.warning("SGLang PD schedule result: no available prefill workers")
            return JSONResponse({"error": "No available prefill workers"}, status_code=503)
        if resp.decode_url is None:
            logger.warning("SGLang PD schedule result: no available decode workers")
            return JSONResponse({"error": "No available decode workers"}, status_code=503)

        logger.debug(
            "SGLang PD schedule result: prefill_url=%s prefill_rank=%s decode_url=%s decode_rank=%s",
            resp.prefill_url, resp.prefill_rank, resp.decode_url, resp.decode_rank,
        )

        return {
            "prefill_url": resp.prefill_url,
            "prefill_rank": resp.prefill_rank,
            "decode_url": resp.decode_url,
            "decode_rank": resp.decode_rank,
        }

    def _build_bootstrap_request(
            self,
            body: Dict[str, Any],
            prefill_url: str,
            prefill_rank: int,
            decode_rank: int,
            prefill_dp_size: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Build the request body with bootstrap info for SGLang PD disaggregation."""

        # Parse prefill URL to get hostname for bootstrap
        parsed_url = urllib.parse.urlparse(prefill_url)
        hostname = _maybe_wrap_ipv6_address(parsed_url.hostname)

        # Get bootstrap port: if we have per-URL mapping, use it; otherwise use rank-based fallback
        bootstrap_port = self._get_bootstrap_port(prefill_url, prefill_rank)

        # Determine batch size
        batch_size = self._get_request_batch_size(body)

        # Generate bootstrap room(s)
        if batch_size is not None:
            # 生成满足约束的 bootstrap_rooms（每个 room % prefill_dp_size == prefill_rank）
            if prefill_rank > -1:
                bootstrap_rooms = []
                for _ in range(batch_size):
                    base = random.randint(0, (2 ** 63 - 1) // prefill_dp_size)
                    bootstrap_rooms.append(base * prefill_dp_size + prefill_rank)
            else:
                bootstrap_rooms = [random.randint(0, 2 ** 63 - 1) for _ in range(batch_size)]

            base_fields = {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
                "bootstrap_room": bootstrap_rooms,
            }

            prefill_req = copy.deepcopy(body)
            prefill_req.update(base_fields)
            if prefill_rank > -1:
                prefill_req["routed_dp_rank"] = prefill_rank  # 单个值，整个 batch 路由到同一 DP rank

            decode_req = copy.deepcopy(body)
            decode_req.update(base_fields)
            if decode_rank > -1:
                decode_req["routed_dp_rank"] = decode_rank

            # if prefill_rank > -1:
            #    decode_req["disagg_prefill_dp_rank"] = prefill_rank

            logger.debug(
                "SGLang PD bootstrap (batch): host=%s port=%s rooms=%s batch_size=%s prefill_rank=%s decode_rank=%s",
                hostname, bootstrap_port, bootstrap_rooms, batch_size, prefill_rank, decode_rank,
            )
        else:
            # 生成满足约束的 bootstrap_room
            if prefill_rank > -1:
                base = random.randint(0, (2 ** 63 - 1) // prefill_dp_size)
                bootstrap_room = base * prefill_dp_size + prefill_rank
            else:
                bootstrap_room = random.randint(0, 2 ** 63 - 1)
            base_fields = {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
                "bootstrap_room": bootstrap_room,
            }

            prefill_req = copy.deepcopy(body)
            prefill_req.update(base_fields)
            if prefill_rank > -1:
                # prefill_req["disagg_mode"] = "prefill"
                prefill_req["routed_dp_rank"] = prefill_rank  # scheduling to prefill DP rank

            decode_req = copy.deepcopy(body)
            decode_req.update(base_fields)

            if decode_rank > -1:
                # prefill_req["disagg_mode"] = "decode"
                decode_req["routed_dp_rank"] = decode_rank  # scheduling to decode DP rank

            # Remove the parameter to use slow path
            # if prefill_rank > -1:
            #    decode_req["disagg_prefill_dp_rank"] = prefill_rank  # scheduling decode to prefill DP rank

            logger.debug(
                "SGLang PD bootstrap (single): host=%s port=%s room=%s prefill_rank=%s decode_rank=%s",
                hostname, bootstrap_port, bootstrap_room, prefill_rank, decode_rank
            )
        return prefill_req, decode_req


    def _get_bootstrap_port(self, prefill_url: str, prefill_rank: int) -> int:
        """Get the bootstrap port for a given prefill URL.

        The bootstrap port is the port on which the prefill worker listens
        for KV cache bootstrap connections from decode workers.

        Lookup order:
        1. Per-URL mapping from self.bootstrap_ports (indexed by URL order)
        2. Fallback: use the default port 8998
        """
        if self.bootstrap_ports:
            # Try to find by URL index
            # We need the prefill_urls list from config. Since we don't have
            # direct access here, we use a simple heuristic: bootstrap_ports
            # is aligned with prefill_urls in order.
            # For now, if there's only one prefill URL, use the first port.
            # If multiple, we use rank-based index.
            if self.bootstrap_ports and self.prefill_urls:
                try:
                    url_index = self.prefill_urls.index(prefill_url)
                    if url_index < len(self.bootstrap_ports):
                        return self.bootstrap_ports[url_index]
                except ValueError:
                    pass
                return self.bootstrap_ports[0]

        return SGLANG_DEFAULT_BOOTSTRAP_PORT  # fallback to sglang default port

    def _get_request_batch_size(self, request: Dict[str, Any]) -> Optional[int]:
        """Determine if this is a batch request and return the batch size."""
        if (text := request.get("text")) is not None:
            return None if isinstance(text, str) else len(text)
        if (input_ids := request.get("input_ids")) is not None:
            if not input_ids:
                return None
            return None if isinstance(input_ids[0], int) else len(input_ids)
        return None

    async def _handle_non_stream_request(
            self,
            request: Request,
            prefill_request: Dict[str, Any],
            decode_request: Dict[str, Any],
            prefill_url: str,
            prefill_rank: int,
            decode_url: str,
            decode_rank: int,
            endpoint_path: str,
    ) -> Response:
        """Handle non-streaming SGLang PD request.

        In SGLang PD mode, both prefill and decode requests are sent in parallel.
        The prefill request processes the prompt and stores KV cache; the decode
        request retrieves KV cache via bootstrap and generates the full response.
        """
        try:
            # Send both requests in parallel
            logger.debug(
                "SGLang PD non-stream: sending prefill to %s%s and decode to %s%s",
                prefill_url, endpoint_path, decode_url, endpoint_path,
            )
            prefill_task = self.http_client.post(
                f"{prefill_url}{endpoint_path}",
                json=prefill_request,
                headers={"Content-Type": "application/json"},
            )
            decode_task = self.http_client.post(
                f"{decode_url}{endpoint_path}",
                json=decode_request,
                headers={"Content-Type": "application/json"},
            )

            prefill_response, decode_response = await asyncio.gather(
                prefill_task, decode_task, return_exceptions=True
            )

            # Handle exceptions from prefill
            if isinstance(prefill_response, Exception):
                logger.error(
                    "SGLang PD non-stream: prefill request failed: %s", prefill_response
                )
                prefill_response = None

            # Handle exceptions from decode
            if isinstance(decode_response, Exception):
                logger.error(
                    "SGLang PD non-stream: decode request failed: %s", decode_response
                )
                return JSONResponse(
                    {"error": f"Decode request failed: {decode_response}"},
                    status_code=502,
                )

            logger.debug(
                "SGLang PD non-stream: prefill status=%s decode status=%s",
                getattr(prefill_response, 'status_code', None),
                getattr(decode_response, 'status_code', None),
            )
        finally:
            # Release prefill and decode worker load immediately after prefill completes
            await asyncio.gather(
                self._decrement_worker(request, prefill_url, prefill_rank),
                self._decrement_worker(request, decode_url, decode_rank),
            )

        if decode_response is None:
            return JSONResponse({"error": "Request failed before receiving decode response"}, status_code=500)

        if not decode_response.is_success:
            return await self._build_upstream_error_response("Decode", decode_response)

        # Merge logprobs if requested return_logprob（/generate）and return_logprobs（/v1/completions）
        want_logprobs = (
                decode_request.get("return_logprob", False)
                or decode_request.get("return_logprobs", False)
        )
        ret_json = decode_response.json()
        if want_logprobs and prefill_response is not None and prefill_response.is_success:
            prefill_json = prefill_response.json()
            if "meta_info" in ret_json and "meta_info" in prefill_json:
                if (
                        "input_token_logprobs" in ret_json["meta_info"]
                        and "input_token_logprobs" in prefill_json["meta_info"]
                ):
                    ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                    )

        return JSONResponse(ret_json, status_code=decode_response.status_code)

    async def _handle_stream_request(
            self,
            request: Request,
            prefill_request: Dict[str, Any],
            decode_request: Dict[str, Any],
            prefill_url: str,
            prefill_rank: int,
            decode_url: str,
            decode_rank: int,
            endpoint_path: str,
            api_kind: str,
    ) -> Response:
        """Handle streaming SGLang PD request.

        Prefill and decode HTTP connections are opened in parallel via
        AsyncExitStack + asyncio.gather to avoid a deadlock: the prefill server
        waits for the decode side to connect via bootstrap before returning
        response headers, so opening them sequentially would block forever.
        """

        async def stream_response():

            prefill_ctx = self.http_client.stream(
                "POST",
                f"{prefill_url}{endpoint_path}",
                json=prefill_request,
                headers={"Content-Type": "application/json"},
            )
            decode_ctx = self.http_client.stream(
                "POST",
                f"{decode_url}{endpoint_path}",
                json=decode_request,
                headers={"Content-Type": "application/json"},
            )

            # FIX: 两个 worker 都在最外层 finally 释放，保证任何异常路径都能执行
            try:
                # -------------------------------------------------------
                # 关键修复：用 AsyncExitStack + asyncio.gather 并行建立
                # 两个 HTTP 连接，避免顺序 async with 导致的死锁。
                # 原因：prefill 服务器在等待 decode 端 bootstrap 握手时
                # 不会返回响应头，若顺序打开则 decode 请求永远无法发出。
                # -------------------------------------------------------
                async with AsyncExitStack() as stack:
                    logger.debug(
                        "SGLang PD stream: starting prefill %s%s and decode %s%s in parallel",
                        prefill_url, endpoint_path, decode_url, endpoint_path,
                    )
                    results = await asyncio.gather(
                        stack.enter_async_context(prefill_ctx),
                        stack.enter_async_context(decode_ctx),
                        return_exceptions=True,
                    )

                    # Handle connection failures
                    prefill_exc = results[0] if isinstance(results[0], Exception) else None
                    decode_exc = results[1] if isinstance(results[1], Exception) else None

                    if prefill_exc is not None:
                        err_msg = f"Prefill connection failed: {prefill_exc}"
                        logger.error(f"SGLang PD stream: {err_msg}")
                        yield (f"data: {json.dumps({'error': err_msg})}\n\n").encode("utf-8")
                        return
                    if decode_exc is not None:
                        err_msg = f"Decode connection failed: {decode_exc}"
                        logger.error(f"SGLang PD stream: {err_msg}")
                        yield (f"data: {json.dumps({'error': err_msg})}\n\n").encode("utf-8")
                        return

                    prefill_stream, decode_response_stream = results

                    logger.debug(
                        "SGLang PD stream: prefill status=%s decode status=%s",
                        prefill_stream.status_code, decode_response_stream.status_code,
                    )

                    if not prefill_stream.is_success:
                        error_body = await prefill_stream.aread()
                        error_text = error_body.decode(errors="replace")
                        logger.error(
                            "SGLang Prefill stream error status=%s body=%s",
                            prefill_stream.status_code,
                            error_text,
                        )

                    prefill_first_chunk_json = None
                    # FIX:  return_logprob and return_logprobs
                    return_logprob = (
                        decode_request.get("return_logprob", False)
                        or decode_request.get("return_logprobs", False)
                    )

                    async def _consume_prefill():
                        nonlocal prefill_first_chunk_json
                        try:
                            async for chunk in prefill_stream.aiter_bytes():
                                if not chunk:
                                    continue
                                if return_logprob and prefill_first_chunk_json is None:
                                    decoded = chunk.decode("utf-8")
                                    for line in decoded.split("\n"):
                                        line = line.strip()
                                        if line.startswith("data:") and "[DONE]" not in line:
                                            try:
                                                data_text = line[5:].strip()
                                                prefill_first_chunk_json = json.loads(data_text)
                                                break
                                            except Exception:
                                                pass
                        except Exception as e:
                            logger.debug("Prefill stream consumption error (ignored): %s", e)

                    consume_task = asyncio.create_task(_consume_prefill())

                    try:
                        if not decode_response_stream.is_success:
                            error_body = await decode_response_stream.aread()
                            error_text = error_body.decode(errors="replace")
                            logger.error(
                                "SGLang Decode stream error status=%s body=%s",
                                decode_response_stream.status_code,
                                error_text,
                            )
                            err_msg = (
                                f"Decode server error "
                                f"{decode_response_stream.status_code}: {error_text}"
                            )
                            yield (
                                f"data: {json.dumps({'error': err_msg})}\n\n"
                            ).encode("utf-8")
                            return

                        async for chunk in decode_response_stream.aiter_bytes():
                            if not chunk:
                                continue

                            if return_logprob and prefill_first_chunk_json is not None:
                                decoded = chunk.decode("utf-8")
                                if decoded and decoded.startswith("data:") and "[DONE]" not in decoded:
                                    try:
                                        data_text = decoded[5:].strip("\n")
                                        ret_json = json.loads(data_text)
                                        if (
                                                "meta_info" in ret_json
                                                and "input_token_logprobs" in ret_json["meta_info"]
                                                and "meta_info" in prefill_first_chunk_json
                                                and "input_token_logprobs" in prefill_first_chunk_json["meta_info"]
                                        ):
                                            ret_json["meta_info"]["input_token_logprobs"] = (
                                                    prefill_first_chunk_json["meta_info"]["input_token_logprobs"]
                                                    + ret_json["meta_info"]["input_token_logprobs"]
                                            )
                                        yield b"data: " + json.dumps(ret_json).encode("utf-8") + b"\n\n"
                                        continue
                                    except Exception:
                                        pass

                            yield chunk

                    finally:
                        await consume_task

            finally:
                await asyncio.gather(
                    self._decrement_worker(request, prefill_url, prefill_rank),
                    self._decrement_worker(request, decode_url, decode_rank),
                )
                logger.debug("SGLang PD stream request finished")

        return StreamingResponse(stream_response(), media_type="text/event-stream")

    async def _decrement_worker(self, request: Request, url: str, rank: int):
        """Send a RELEASE request to decrement worker load."""
        logger.debug("SGLang PD releasing worker: url=%s rank=%s", url, rank)
        engine_request = EngineRequest(
            request_id=uuid.uuid4().hex,
            identity=request.app.state.engine_client.identity,
            request_type=RequestType.RELEASE,
            worker_rank=rank,
            worker_url=url,
        )
        await request.app.state.engine_client.send_request(engine_request)

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
            {"error": f"{stage} server error {response.status_code}: {error_text}"},
            status_code=500,
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
