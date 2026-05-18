from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Optional
from urllib.parse import urlparse

from smart_router.cache.kv_cache_state import (
    AllBlocksClearedEvent,
    BlockRemovedEvent,
    BlockStoredEvent,
    KVCacheState,
    KVEventBatch,
    normalize_batch,
)
from smart_router.config import SmartRouterConfig

if TYPE_CHECKING:
    from smart_router.worker import Worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerEventEndpoint:
    worker_id: str
    endpoint: str
    topic: str = ""


class KVEventSubscriber:
    def __init__(self, state: KVCacheState,
                 endpoints: Iterable[WorkerEventEndpoint]) -> None:
        self.state = state
        self.endpoints = list(endpoints)
        self._ctx: Optional[Any] = None
        self._tasks: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    def start(self) -> None:
        if self._tasks:
            return
        import zmq.asyncio

        self._ctx = zmq.asyncio.Context()
        self._stopped.clear()
        logger.info("[KV-EVENT-SUBSCRIBER] starting endpoint_count=%d",
                    len(self.endpoints))
        for endpoint in self.endpoints:
            task = asyncio.create_task(self._run_endpoint(endpoint))
            self._tasks.append(task)

    async def stop(self) -> None:
        self._stopped.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None

    async def _run_endpoint(self, endpoint: WorkerEventEndpoint) -> None:
        import zmq

        if self._ctx is None:
            raise RuntimeError("KVEventSubscriber.start() must be called first")

        socket = self._ctx.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, endpoint.topic.encode("utf-8"))
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(endpoint.endpoint)
        logger.info("Subscribed to vLLM KV events worker=%s endpoint=%s topic=%r",
                    endpoint.worker_id, endpoint.endpoint, endpoint.topic)

        try:
            while not self._stopped.is_set():
                try:
                    frames = await socket.recv_multipart()
                    if len(frames) < 3:
                        logger.warning("Invalid KV event frame from %s: %s",
                                       endpoint.endpoint, frames)
                        continue
                    payload = frames[-1]
                    batch = _decode_msgpack(payload)
                    logger.info(
                        "[KV-EVENT-RECV] worker=%s endpoint=%s topic=%r "
                        "seq=%s payload_bytes=%d %s",
                        endpoint.worker_id,
                        endpoint.endpoint,
                        frames[0].decode(errors="replace") if frames else "",
                        int.from_bytes(frames[1], "big")
                        if len(frames) > 1 else None,
                        len(payload),
                        _summarize_batch(batch),
                    )
                    self.state.apply_batch(endpoint.worker_id, batch)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Failed to process KV event from %s",
                                     endpoint.endpoint)
                    await asyncio.sleep(0.1)
        finally:
            socket.close(linger=0)


def _decode_msgpack(payload: bytes):
    try:
        import msgspec
    except ImportError as exc:
        raise RuntimeError(
            "msgspec is required to decode vLLM KV event payloads") from exc
    return normalize_batch(msgspec.msgpack.decode(payload))


def build_worker_event_endpoints(
        config: SmartRouterConfig,
        workers: Iterable["Worker"]) -> list[WorkerEventEndpoint]:
    explicit, base_endpoints = _parse_explicit_endpoints(
        config.kv_events_endpoints)
    endpoints: list[WorkerEventEndpoint] = []
    seen: set[str] = set()
    base_url_to_index: dict[str, int] = {}

    for worker in workers:
        worker_id = worker.url()
        base_url = worker.base_url()
        endpoint = explicit.get(worker_id)
        endpoint_source = "explicit_worker"
        if endpoint is None:
            base_endpoint = explicit.get(base_url)
            endpoint_source = "explicit_base"
            if base_endpoint is None:
                base_index = base_url_to_index.setdefault(
                    base_url, len(base_url_to_index))
                base_endpoint = (base_endpoints[base_index]
                                 if base_index < len(base_endpoints) else None)
                endpoint_source = f"bare_base[{base_index}]"
            endpoint = (_offset_endpoint_port(base_endpoint, worker.dp_rank())
                        if base_endpoint is not None else None)
        if endpoint is None:
            endpoint = _derive_endpoint(config, worker)
            endpoint_source = "derived"
        if worker_id in seen:
            continue
        seen.add(worker_id)
        logger.info(
            "[KV-EVENT-MAP] worker=%s base_url=%s dp_rank=%s endpoint=%s "
            "source=%s",
            worker_id,
            base_url,
            worker.dp_rank(),
            endpoint,
            endpoint_source,
        )
        endpoints.append(
            WorkerEventEndpoint(worker_id=worker_id,
                                endpoint=endpoint,
                                topic=config.kv_events_topic))

    return endpoints


def _derive_endpoint(config: SmartRouterConfig, worker: "Worker") -> str:
    parsed = urlparse(worker.base_url())
    host = parsed.hostname or "127.0.0.1"
    if host in {"0.0.0.0", "*", "::"}:
        host = "127.0.0.1"
    rank_offset = worker.dp_rank() if worker.dp_rank() > 0 else 0
    port = config.kv_events_port + rank_offset
    return f"tcp://{host}:{port}"


def _parse_explicit_endpoints(
        entries: Optional[list[str]]) -> tuple[dict[str, str], list[str]]:
    parsed: dict[str, str] = {}
    base_endpoints: list[str] = []
    for entry in entries or []:
        if "=" not in entry:
            base_endpoints.append(entry.strip())
            continue
        worker_id, endpoint = entry.split("=", 1)
        parsed[worker_id.strip()] = endpoint.strip()
    return parsed, base_endpoints


def _offset_endpoint_port(endpoint: Optional[str], dp_rank: int) -> Optional[str]:
    if endpoint is None or dp_rank <= 0:
        return endpoint
    if endpoint.startswith("inproc://"):
        return f"{endpoint}_dp{dp_rank}"
    if endpoint.startswith("tcp://"):
        last_colon_idx = endpoint.rfind(":")
        if last_colon_idx < 0:
            return endpoint
        base_addr = endpoint[:last_colon_idx]
        base_port = int(endpoint[last_colon_idx + 1:])
        return f"{base_addr}:{base_port + dp_rank}"
    return endpoint


def _summarize_batch(batch: KVEventBatch) -> str:
    stored_events = 0
    stored_blocks = 0
    removed_events = 0
    removed_blocks = 0
    cleared_events = 0
    media: dict[str, int] = {}
    for event in batch.events:
        if isinstance(event, BlockStoredEvent):
            stored_events += 1
            stored_blocks += len(event.block_hashes)
            media[event.medium or "unknown"] = (
                media.get(event.medium or "unknown", 0) + len(event.block_hashes))
        elif isinstance(event, BlockRemovedEvent):
            removed_events += 1
            removed_blocks += len(event.block_hashes)
            media[event.medium or "unknown"] = (
                media.get(event.medium or "unknown", 0) + len(event.block_hashes))
        elif isinstance(event, AllBlocksClearedEvent):
            cleared_events += 1
    return (
        f"batch_ts={batch.ts} dp_rank={batch.data_parallel_rank} "
        f"events={len(batch.events)} stored_events={stored_events} "
        f"stored_blocks={stored_blocks} removed_events={removed_events} "
        f"removed_blocks={removed_blocks} cleared_events={cleared_events} "
        f"media={media}")
