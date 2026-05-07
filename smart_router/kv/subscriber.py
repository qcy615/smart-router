from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse

import zmq
import zmq.asyncio

from smart_router.engine.utils import make_zmq_socket
from smart_router.kv.events import parse_event_batch
from smart_router.kv.indexer import KvRadixIndexer, WorkerKey

logger = logging.getLogger(__name__)

END_SEQ = (-1).to_bytes(8, "big", signed=True)


def offset_endpoint_port(endpoint: str | None, rank: int) -> str | None:
    if not endpoint or rank == 0:
        return endpoint
    if endpoint.startswith("inproc://"):
        return f"{endpoint}_dp{rank}"
    if endpoint.startswith("tcp://"):
        idx = endpoint.rfind(":")
        if idx < 0:
            return endpoint
        return f"{endpoint[:idx]}:{int(endpoint[idx + 1:]) + rank}"
    return endpoint


def default_replay_endpoint(event_endpoint: str | None) -> str | None:
    if not event_endpoint or not event_endpoint.startswith("tcp://"):
        return None
    parsed = urlparse(event_endpoint)
    if parsed.port is None:
        return None
    return event_endpoint[: event_endpoint.rfind(":")] + f":{parsed.port + 1}"


class KvEventSubscriber:
    def __init__(
        self,
        indexer: KvRadixIndexer,
        worker: WorkerKey,
        event_endpoint: str,
        replay_endpoint: str | None = None,
        topic: str = "",
        replay_timeout_secs: float = 1.0,
    ) -> None:
        self.indexer = indexer
        self.worker = worker
        self.event_endpoint = event_endpoint
        self.replay_endpoint = replay_endpoint
        self.topic = topic.encode("utf-8")
        self.replay_timeout_secs = replay_timeout_secs
        self.last_seq: int | None = None
        self._ctx = zmq.asyncio.Context.instance()
        self._socket: zmq.asyncio.Socket | None = None
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()

    def start(self) -> asyncio.Task:
        if self._task is None:
            self._task = asyncio.create_task(self.run())
        return self._task

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._socket is not None:
            self._socket.close(0)

    async def run(self) -> None:
        self._socket = make_zmq_socket(self._ctx, self.event_endpoint, zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        logger.info("Subscribed to KV events endpoint=%s worker=%s", self.event_endpoint, self.worker)

        while not self._stopped.is_set():
            frames = await self._socket.recv_multipart()
            try:
                await self._handle_live_frames(frames)
            except Exception:
                logger.exception("Failed to handle KV event frames from %s", self.event_endpoint)

    async def _handle_live_frames(self, frames: list[bytes]) -> None:
        if len(frames) != 3:
            logger.warning("Invalid KV event multipart frame count=%s", len(frames))
            return

        _topic, seq_bytes, payload = frames
        seq = int.from_bytes(seq_bytes, "big")
        if self.last_seq is not None and seq > self.last_seq + 1:
            await self._replay_missing(self.last_seq + 1, seq)

        self._apply_payload(seq, payload)

    async def _replay_missing(self, start_seq: int, live_seq: int) -> None:
        if not self.replay_endpoint:
            logger.warning("KV event gap for %s but no replay endpoint configured", self.worker)
            return

        logger.warning(
            "KV event gap for %s: last=%s live=%s replay_endpoint=%s",
            self.worker,
            self.last_seq,
            live_seq,
            self.replay_endpoint,
        )

        replay_socket = make_zmq_socket(self._ctx, self.replay_endpoint, zmq.DEALER, linger=0)
        try:
            await replay_socket.send_multipart([b"", start_seq.to_bytes(8, "big")])
            while True:
                frames = await asyncio.wait_for(
                    replay_socket.recv_multipart(),
                    timeout=self.replay_timeout_secs,
                )
                seq_bytes, payload = self._extract_replay_payload(frames)
                if seq_bytes == END_SEQ:
                    break
                seq = int.from_bytes(seq_bytes, "big")
                if seq >= live_seq:
                    break
                self._apply_payload(seq, payload)
        except asyncio.TimeoutError:
            logger.warning("Timed out replaying KV events from %s", self.replay_endpoint)
        finally:
            replay_socket.close(0)

    def _extract_replay_payload(self, frames: list[bytes]) -> tuple[bytes, bytes]:
        if len(frames) == 3 and frames[0] == b"":
            return frames[1], frames[2]
        if len(frames) == 2:
            return frames[0], frames[1]
        raise ValueError(f"Invalid replay frame count={len(frames)}")

    def _apply_payload(self, seq: int, payload: bytes) -> None:
        if self.last_seq is not None and seq <= self.last_seq:
            return

        dp_rank, events = parse_event_batch(payload)
        if dp_rank is not None and self.worker.rank >= 0 and dp_rank != self.worker.rank:
            logger.warning(
                "KV event dp_rank=%s does not match configured worker=%s",
                dp_rank,
                self.worker,
            )

        for event in events:
            self.indexer.apply_event(self.worker, event)
        self.last_seq = seq


class KvEventSubscriberGroup:
    def __init__(self, subscribers: list[KvEventSubscriber]) -> None:
        self.subscribers = subscribers

    def start(self) -> None:
        for subscriber in self.subscribers:
            subscriber.start()

    async def stop(self) -> None:
        await asyncio.gather(*(subscriber.stop() for subscriber in self.subscribers))


def build_prefill_subscribers(
    indexer: KvRadixIndexer,
    worker_urls: list[str],
    intra_dp_size: int,
    event_endpoints: list[str],
    replay_endpoints: list[str],
    topic: str = "",
) -> KvEventSubscriberGroup:
    if not event_endpoints:
        return KvEventSubscriberGroup([])
    if len(event_endpoints) != len(worker_urls):
        raise ValueError(
            "--prefill-kv-event-endpoints must have the same length as --prefill-urls"
        )
    if replay_endpoints and len(replay_endpoints) != len(worker_urls):
        raise ValueError(
            "--prefill-kv-replay-endpoints must have the same length as --prefill-urls"
        )

    subscribers: list[KvEventSubscriber] = []
    for idx, worker_url in enumerate(worker_urls):
        base_event_endpoint = event_endpoints[idx]
        base_replay_endpoint = (
            replay_endpoints[idx]
            if replay_endpoints
            else default_replay_endpoint(base_event_endpoint)
        )
        for rank in range(max(intra_dp_size, 1)):
            worker_rank = rank if intra_dp_size > 1 else -1
            subscribers.append(
                KvEventSubscriber(
                    indexer=indexer,
                    worker=WorkerKey(worker_url, worker_rank),
                    event_endpoint=offset_endpoint_port(base_event_endpoint, rank) or base_event_endpoint,
                    replay_endpoint=offset_endpoint_port(base_replay_endpoint, rank),
                    topic=topic,
                )
            )

    return KvEventSubscriberGroup(subscribers)
