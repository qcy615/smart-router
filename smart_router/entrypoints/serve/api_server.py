import asyncio
import json
import os
import platform
import uvicorn

from multiprocessing import Event, Process

from starlette.applications import Starlette
from starlette.routing import Route

from smart_router.config import build_config, build_parser
from smart_router.engine.engine_client import EngineClient
from smart_router.engine.vllm_engine import start_engine
from smart_router.entrypoints.serve.vllm_routes import VllmRoutes
from smart_router.logger import init_logging
from smart_router.openai.processing import OpenAIProcessingConfig

MODEL_SOURCE_URLS_ENV = "SMART_ROUTER_MODEL_SOURCE_URLS"
OPENAI_PROCESSING_CONFIG_ENV = "SMART_ROUTER_OPENAI_PROCESSING_CONFIG"

 # 检测操作系统
is_linux = platform.system() == "Linux"

if is_linux:
    # Linux 使用 IPC 协议
    output_addr = "ipc://output.ipc"
    input_addr = "ipc://input.ipc"
else:
    # Windows (或其他) 使用 TCP 协议
    output_addr = "tcp://127.0.0.1:5558"
    input_addr = "tcp://127.0.0.1:5557"

async def startup():
    app.state.engine_client = EngineClient(input_addr, output_addr)
    app.state.model_source_urls = _load_model_source_urls()
    app.state.openai_processing_config = _load_openai_processing_config()
    asyncio.create_task(app.state.engine_client.receive_loop())

async def shutdown():
    await app.state.engine_client.shutdown()
    await vllm_routes.close()


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


def _dump_openai_processing_config(config: OpenAIProcessingConfig) -> None:
    os.environ[OPENAI_PROCESSING_CONFIG_ENV] = json.dumps(
        {
            "enabled": config.enabled,
            "tokenizer_path": config.tokenizer_path,
            "reasoning_parser": config.reasoning_parser,
            "tool_call_parser": config.tool_call_parser,
        }
    )


def _load_openai_processing_config() -> OpenAIProcessingConfig:
    raw = os.getenv(OPENAI_PROCESSING_CONFIG_ENV)
    if not raw:
        return OpenAIProcessingConfig()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return OpenAIProcessingConfig()
    if not isinstance(data, dict):
        return OpenAIProcessingConfig()

    return OpenAIProcessingConfig(
        enabled=bool(data.get("enabled", False)),
        tokenizer_path=data.get("tokenizer_path")
        if isinstance(data.get("tokenizer_path"), str)
        else None,
        reasoning_parser=data.get("reasoning_parser")
        if data.get("reasoning_parser") in {"think_tags", "none"}
        else "think_tags",
        tool_call_parser=data.get("tool_call_parser")
        if data.get("tool_call_parser") in {"hermes", "none"}
        else "hermes",
    )


vllm_routes = VllmRoutes()

routes = [
    Route("/v1/models", vllm_routes.models, methods=["GET"]),
    Route("/v1/chat/completions", vllm_routes.chat_completions, methods=["POST"]),
    # Route("/v1/completions", completions, methods=["POST"]),
]

app = Starlette(
    routes=routes,
    on_startup=[startup],
    on_shutdown=[shutdown],
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # build config
    config = build_config(args)
    _dump_model_source_urls(config.prefill_urls, config.decode_urls)
    _dump_openai_processing_config(config.openai_processing_config)

    init_logging(args.log_level)

    # start engine process
    stop_event = Event()
    engine_process = Process(
        target=start_engine,
        args=(config, input_addr, output_addr),
        name="Router Engine",
    )
    engine_process.start()

    uvicorn.run(
        "smart_router.entrypoints.serve.api_server:app",
        host=args.host,
        port=args.port,
        workers=args.apiserver_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
