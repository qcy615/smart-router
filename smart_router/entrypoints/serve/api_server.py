import asyncio
import logging
import json
import os
import platform
import uuid

import sys
import tempfile
import uvicorn

from multiprocessing import Process
from typing import Optional

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse

from smart_router.config import build_config, build_parser
from smart_router.engine.engine import EngineHealthResponse, EngineRequest, RequestType
from smart_router.engine.engine_client import EngineClient
from smart_router.engine.vllm_engine import start_engine
from smart_router.engine.sglang_engine import start_sglang_engine
from smart_router.engine.normal_engine import start_normal_engine
from smart_router.entrypoints.serve.vllm_routes import VllmRoutes
from smart_router.entrypoints.serve.sglang_routes import SGLangRoutes
from smart_router.entrypoints.serve.normal_routes import NormalRoutes
from smart_router.logger import init_logging

logger =logging.getLogger(__name__)

# Detect 0S
is_linux = platform.system() == "Linux"
MODEL_SOURCE_URLS_ENV = "SMART_ROUTER_MODEL_SOURCE_URLS"


def _dump_model_source_urls(prefill_urls: list[str] | None, decode_urls: list[str] | None) -> None:
    urls = []
    for url in (prefill_urls or []) + (decode_urls or []):
        if url and url not in urls:
            urls.append(url)
    os.environ[MODEL_SOURCE_URLS_ENV] = json.dumps(urls)


def _load_model_source_urls() -> list[str]:
    raw = os.getenv(MODEL_SOURCE_URLS_ENV)
    if not raw:
        return []

    try:
        urls = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if not isinstance(urls, list):
        return []

    return [url for url in urls if isinstance(url, str) and url]


def _get_zmq_addresses():
    """Generate ZMg addresses. Use unique IPc paths on Linux to avoid conflicts."""
    if is_linux:
        # Use temp directory with PID to avoid conflicts between instances
        # ipc_dir = os.path.join(tempfile.gettempdir(),f"smart-router-{os.getpid()}")
        main_pid = os.environ.get("_SMART_ROUTER_MAIN_PID")
        if main_pid is None:
            # We are in the main process - record our PID
            main_pid = str(os.getpid())
            os.environ["_SMART_ROUTER_MAIN_PID"] = main_pid
        ipc_dir = os.path.join(tempfile.gettempdir(), f"smart-router-{main_pid}")
        os.makedirs(ipc_dir, exist_ok=True)
        return (
            f"ipc://{os.path.join(ipc_dir, 'output.ipc')}",
            f"ipc://{os.path.join(ipc_dir, 'input.ipc')}"
        )
    else:
        return "tcp://127.0.0.1:5558", "tcp://127.0.0.1:5557"


# Module-level ZMQ addresses (resolved lazily, same for all workers in same process group)
output_addr: Optional[str] = None
input_addr: Optional[str] = None

# Global reference to receive_loop task for cleanup
_receive_task: Optional[asyncio.Task] = None

# Module-level config and app(populated by _init_app or main)
_config = None
app: Starlette


def _build_app(config):
    """Build Starlette app with routes based on router_type and pd_disaggregation."""
    router_type = config.router_type
    pd_disaggregation = config.pd_disaggregation
    upstream_http_client_config = getattr(config, "upstream_http_client_config", None)

    if router_type == "sglang" and pd_disaggregation:
        sglang_routes = SGLangRoutes(config)
        model_routes = VllmRoutes(
            http_client_config=upstream_http_client_config
        )
        routes = [
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", model_routes.models, methods=["GET"]),
            Route("/v1/chat/completions", sglang_routes.chat_completions, methods=["POST"]),
            Route("/v1/completions", sglang_routes.completions, methods=["POST"]),
            # Route("/generate", sglang_routes.generate, methods=["POST"]),
        ]

        application = Starlette(
            routes=routes,
            on_startup=[startup],
            on_shutdown=[shutdown],
        )
        application.state.sglang_routes = sglang_routes
        application.state.model_routes = model_routes

    elif pd_disaggregation:
        # vLLM PD disaggregation
        vllm_routes = VllmRoutes(
            http_client_config=upstream_http_client_config
        )
        routes = [
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", vllm_routes.models, methods=["GET"]),
            Route("/v1/chat/completions", vllm_routes.chat_completions, methods=["POST"]),
            Route("/v1/completions", vllm_routes.completions, methods=["POST"]),
        ]

        application = Starlette(
            routes=routes,
            on_startup=[startup],
            on_shutdown=[shutdown],
        )
        application.state.vllm_routes = vllm_routes

    else:
        # Non-PD mode: direct forwarding to workers
        normal_routes = NormalRoutes(
            router_type=router_type,
            http_client_config=upstream_http_client_config,
        )
        model_routes = VllmRoutes(
            http_client_config=upstream_http_client_config
        )
        routes = [
            Route("/health", health, methods=["GET"]),
            Route("/v1/models", model_routes.models, methods=["GET"]),
            Route("/v1/chat/completions", normal_routes.chat_completions, methods=["POST"]),
            Route("/v1/completions", normal_routes.completions, methods=["POST"]),
        ]

        application = Starlette(
            routes=routes,
            on_startup=[startup],
            on_shutdown=[shutdown],
        )
        application.state.normal_routes = normal_routes
        application.state.model_routes = model_routes

    health_config = getattr(config, "health_config", None)
    application.state.health_timeout_secs = getattr(health_config, "timeout_secs", 5) + 1
    return application


async def health(request):
    engine_client = getattr(request.app.state, "engine_client", None)
    if engine_client is None:
        return JSONResponse(
            {"status": "unhealthy", "error": "Engine client is not initialized"},
            status_code=503,
        )

    engine_request = EngineRequest(
        request_id=uuid.uuid4().hex,
        identity=engine_client.identity,
        request_type=RequestType.HEALTH,
    )
    fut = await engine_client.send_request(engine_request)
    try:
        timeout_secs = getattr(request.app.state, "health_timeout_secs", 6)
        resp: EngineHealthResponse = await asyncio.wait_for(fut, timeout=timeout_secs)
    except asyncio.TimeoutError:
        logger.error("Timeout waiting for health check result")
        return JSONResponse(
            {"status": "unhealthy", "error": "Timeout checking worker health"},
            status_code=503,
        )
    except Exception:
        logger.exception("Failed to get health check result")
        return JSONResponse(
            {"status": "unhealthy", "error": "Failed checking worker health"},
            status_code=503,
        )

    status_code = 200 if resp.status == "ok" else 503
    body = {
        "status": resp.status,
        "prefill_healthy": resp.prefill_healthy,
        "prefill_total": resp.prefill_total,
        "decode_healthy": resp.decode_healthy,
        "decode_total": resp.decode_total,
    }
    if resp.regular_total > 0:
        body["regular_healthy"] = resp.regular_healthy
        body["regular_total"] = resp.regular_total

    return JSONResponse(body, status_code=status_code)


def _init_app():
    """Initialize app from sys.argv. Called only when needed (not on import)."""
    global app, _config, output_addr, input_addr

    output_addr, input_addr = _get_zmq_addresses()

    _argv = sys.argv[1:]
    if _argv and _argv[0] == "serve":
        _argv = _argv[1:]
    _args = build_parser().parse_args(_argv)
    _config = build_config(_args)

    # Re-init logging in worker processes (uvicorn forks workers that
    # lose the main process's logging config)
    init_logging(_args.log_level)

    app = _build_app(_config)


async def startup():
    """Initialize Engineclient and start receive loop for each worker process."""
    global _receive_task
    app.state.engine_client = EngineClient(input_addr, output_addr)
    app.state.model_source_urls = _load_model_source_urls()
    _receive_task = asyncio.create_task(app.state.engine_client.receive_loop())
    logger.info(f"Engineclient started with identity: {app.state.engine_client.identity}")


async def shutdown():
    """Gracefully shutdown route handlers and EngineClient."""
    global _receive_task

    # Close NormalRoutes HTTP connection pool (normal mode)
    normal_routes = getattr(app.state, "normal_routes", None)
    if normal_routes is not None:
        await normal_routes.close()
        logger.info("NormalRoutes HTTP client closed")

    # Close SGLangRoutes HTTP connection pool (sglang-pd-disagg mode)
    sglang_routes = getattr(app.state, "sglang_routes", None)
    if sglang_routes is not None:
        await sglang_routes.close()
        logger.info("SGLangRoutes HTTP client closed")

    # Close VllmRoutes HTTP connection pool (vllm-pd-disagg mode)
    vllm_routes = getattr(app.state, "vllm_routes", None)
    if vllm_routes is not None:
        await vllm_routes.close()
        logger.info("VllmRoutes HTTP client closed")

    model_routes = getattr(app.state, "model_routes", None)
    if model_routes is not None:
        await model_routes.close()
        logger.info("ModelRoutes HTTP client closed")

    engine_client = getattr(app.state, "engine_client", None)
    if engine_client is None:
        logger.warning("EngineClient not found during shutdown")
        return

    # Close sockets first -this causes receive_loop to exit naturally
    # on the next recv_multipart()instead of losing in-flight messages.
    await engine_client.shutdown()
    logger.info(f"Engineclient sockets closed: {engine_client.identity}")

    # Then cancel the receive task (it should exit on its own after socket close
    # but cancel as a safety net)
    if _receive_task is not None and not _receive_task.done():
        _receive_task.cancel()
        try:
            await _receive_task
        except asyncio.CancelledError:
            pass
        logger.info("Receive loop cancelled")

    logger.info(f"Engineclient shutdown complete: {engine_client.identity}")


def main(argv: list[str]|None = None) -> int:
    global app, _config, output_addr, input_addr
    parser = build_parser()
    args = parser.parse_args(argv)

    # Build config
    config = build_config(args)
    _dump_model_source_urls(config.prefill_urls, config.decode_urls)

    init_logging(args.log_level)

    # Resolve ZMQ addresses
    output_addr, input_addr = _get_zmq_addresses()

    # Select engine based on router_type and pd_disaggregation
    if config.pd_disaggregation:
        if config.router_type == "sglang":
            engine_target = start_sglang_engine
        else:
            engine_target = start_engine
    else:
        engine_target = start_normal_engine

    # Start engine process
    engine_process = Process(
        target=engine_target,
        args=(config, input_addr, output_addr),
        name="Router Engine",
    )
    engine_process.start()
    logger.info(f"Engine process started with PID: {engine_process.pid}")

    # Build app for uvicorn import path
    _config = config
    app = _build_app(config)

    # Track engine process for cleanup, avoid signal handler conflicts with uvicorn.
    # Instead of overriding signal handlers, use atexit + uvicorn's own signal handling.
    import atexit

    def cleanup_engine():
        if engine_process.is_alive():
            logger.info("Terminating engine process...")
            engine_process.terminate()
        engine_process.join(timeout=10)
        if engine_process.is_alive():
            logger.warning("Engine process did not terminate gracefully, killing...")
            engine_process.kill()
            engine_process.join()
        logger.info("Engine process stopped")

        # Cleanup IPc files on Linux
        if is_linux:
            # ipc_dir = os.path.join(tempfile.gettempdir(), f"smart-router-{os.getpid()}")
            main_pid = os.environ.get("_SMART_ROUTER_MAIN_PID", str(os.getpid()))
            ipc_dir = os.path.join(tempfile.gettempdir(), f"smart-router-{main_pid}")
            for fname in ("output.ipc", "input.ipc"):
                fpath = os.path.join(ipc_dir, fname)
                if os.path.exists(fpath):
                    try:
                        os.remove(fpath)
                    except OSError:
                        pass

    atexit.register(cleanup_engine)

    try:
        uvicorn.run(
            "smart_router.entrypoints.serve.api_server:app",
            host=args.host,
            port=args.port,
            workers=args.apiserver_workers,
        )
    finally:
        # Ensure cleanup runs even if atexit doesn't(e.g. signal)
        cleanup_engine()
        atexit.unregister(cleanup_engine)

    return 0


# Lazy initialization: only parse argv and build app when actually needed.
# This prevents crashes on import (e.g. in tests or when importing for main()).
# uvicorn workers will trigger this via the module-level app access below,
# but only after the main process has already set things up.
def __getattr__(name):
    """Lazy module attribute access - build app on first access to 'app'."""
    if name == "app":
        _init_app()
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())
