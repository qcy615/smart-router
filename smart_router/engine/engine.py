import logging
import asyncio
import zmq
import zmq.asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import platform

from smart_router.config import SchedulerConfig
from smart_router.engine.utils import make_zmq_socket
from smart_router.worker import Worker, WorkerRegistry, WorkerType

logger = logging.getLogger(__name__)

is_linux = platform.system() == "Linux"
if is_linux:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("enable uvloop event loop poicy!")


class RequestType:
    SCHEDULE = "schedule"
    RELEASE = "release"


@dataclass
class EngineRequest:
    request_id: str
    identity: str
    request_type: RequestType  # "schedule" | "release"
    # request_type: schedule
    request_text: str = field(default="")
    headers: Dict[str, str] = field(default_factory=dict)
    # request_type: release
    worker_url: str = field(default="")
    worker_rank: int = field(default=-1)
    forward_time_ms: Optional[float] = field(default=None)
    enqueue_time: Optional[float] = field(default=None, compare=False)


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineRequest":
        return cls(
            request_id=data["request_id"],
            identity=data["identity"],
            request_type=data["request_type"],
            request_text=data.get("request_text", ""),
            headers=data.get("headers", {}),
            worker_url=data.get("worker_url", ""),
            worker_rank=data.get("worker_rank", -1),
            forward_time_ms=data.get("forward_time_ms"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "identity": self.identity,
            "request_type": self.request_type,
            "request_text": self.request_text,
            "headers": self.headers,
            "worker_url": self.worker_url,
            "worker_rank": self.worker_rank,
            "forward_time_ms": self.forward_time_ms,
        }
    

@dataclass
class EngineResponse:
    request_id: str
    prefill_url: Optional[str]
    prefill_rank: int
    decode_url: Optional[str]
    decode_rank: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineResponse":
        return cls(
            request_id=data["request_id"],
            prefill_url=data["prefill_url"],
            prefill_rank=data["prefill_rank"],
            decode_url=data["decode_url"],
            decode_rank=data["decode_rank"],
        )
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prefill_url": self.prefill_url,
            "prefill_rank": self.prefill_rank,
            "decode_url": self.decode_url,
            "decode_rank": self.decode_rank,
        }
    


class Engine:
    def __init__(
        self,
        input_socket_address: str,
        output_socket_address: str,
        scheduler_config: Optional[SchedulerConfig] = None,
    ) -> None:
        # Initialize ZeroMQ context and sockets
        ctx = zmq.Context()
        self.async_ctx = zmq.asyncio.Context(ctx)
        self.input_socket: zmq.Socket = make_zmq_socket(self.async_ctx, input_socket_address, zmq.PULL)
        self.output_socket: zmq.Socket = make_zmq_socket(self.async_ctx, output_socket_address, zmq.ROUTER)

        # queues for scheduling
        self.waiting_queue: asyncio.Queue[EngineRequest] = asyncio.Queue()
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.worker_registry: WorkerRegistry = WorkerRegistry()

        stats_window_size = max(1, self.scheduler_config.stats_window_size)
        self._forward_time_windows: Dict[WorkerType, deque[float]] = {
            WorkerType.PREFILL: deque(maxlen=stats_window_size),
            WorkerType.DECODE: deque(maxlen=stats_window_size),
        }
        self._current_interval_secs = self._recompute_adaptive_interval_secs()
        self._last_dispatch_time: Optional[float] = None
        self._ready_event = asyncio.Event()
        self._watchdog_deadline: Optional[float] = None


    async def receive_loop(self):
        while True:
            request = await self.input_socket.recv_json()
            engine_request = EngineRequest.from_dict(request)
            if engine_request.request_type == RequestType.SCHEDULE:
                engine_request.enqueue_time = asyncio.get_running_loop().time()
                await self.waiting_queue.put(engine_request)
                logger.debug(f"Received schedule request: {engine_request.request_id}, queue size: {self.waiting_queue.qsize()}")

            elif engine_request.request_type == RequestType.RELEASE:
                worker_id = self._release_worker_id(engine_request)

                worker = self.worker_registry.get(worker_id)
                if worker is None:
                    logger.warning(
                        "Received release for unknown worker request_id=%s worker_id=%s",
                        engine_request.request_id,
                        worker_id,
                    )
                    continue

                worker.decrement_load()
                self._record_forward_time(
                    worker.worker_type(),
                    engine_request.forward_time_ms,
                )
                self._ready_event.set()

                logger.debug(f"Received release request: {engine_request.request_id}, worker id: {worker_id}")

    async def schedule_loop(self):
        while True:
            batch = await self._get_next_batch()
            await self._schedule_batch(batch)

    async def _schedule_batch(self, batch: List[EngineRequest]) -> None:
        if not batch:
            return

        loop = asyncio.get_running_loop()
        schedule_started = loop.time()
        oldest_enqueue_time = self._oldest_enqueue_time(batch)
        logger.debug(
            "Processing schedule batch size=%s oldest_queue_wait_ms=%.2f "
            "queue_remaining=%s %s request_ids=%s",
            len(batch),
            self._elapsed_ms(oldest_enqueue_time, schedule_started),
            self.waiting_queue.qsize(),
            self._adaptive_debug_state(schedule_started),
            [request.request_id for request in batch],
        )

        policy_started = loop.time()
        prefill_workers = self.schedule_prefill_batch(batch)
        decode_workers = self.schedule_decode_batch(batch)
        policy_elapsed_ms = self._elapsed_ms(policy_started, loop.time())
        workers_to_increment: List[Optional[Worker]] = []
        response_pairs: List[tuple[EngineRequest, EngineResponse]] = []

        for index, request in enumerate(batch):
            prefill_worker = (
                prefill_workers[index] if index < len(prefill_workers) else None
            )
            decode_worker = decode_workers[index] if index < len(decode_workers) else None

            # Scheduling is all-or-nothing for each request: only commit worker
            # loads when both prefill and decode assignments are available.
            if prefill_worker is not None and decode_worker is not None:
                workers_to_increment.extend([prefill_worker, decode_worker])
                resp = EngineResponse(
                    request_id=request.request_id,
                    prefill_url=self._worker_base_url(prefill_worker),
                    prefill_rank=self._worker_dp_rank(prefill_worker),
                    decode_url=self._worker_base_url(decode_worker),
                    decode_rank=self._worker_dp_rank(decode_worker),
                )
            else:
                logger.warning(
                    "Unable to schedule request_id=%s prefill_assigned=%s decode_assigned=%s",
                    request.request_id,
                    prefill_worker is not None,
                    decode_worker is not None,
                )
                resp = EngineResponse(
                    request_id=request.request_id,
                    prefill_url=None,
                    prefill_rank=-1,
                    decode_url=None,
                    decode_rank=-1,
                )

            response_pairs.append((request, resp))

        self._increment_worker_loads(workers_to_increment)

        response_send_started = loop.time()
        for request, resp in response_pairs:
            await self.send_response(request, resp.to_dict())
            logger.debug(
                "Sent schedule response request_id=%s prefill=%s decode=%s",
                request.request_id,
                resp.prefill_url,
                resp.decode_url,
            )
        response_send_elapsed_ms = self._elapsed_ms(response_send_started, loop.time())

        self._mark_batch_dispatched()
        logger.debug(
            "Schedule batch complete size=%s policy_ms=%.2f response_send_ms=%.2f "
            "total_ms=%.2f queue_remaining=%s %s",
            len(batch),
            policy_elapsed_ms,
            response_send_elapsed_ms,
            self._elapsed_ms(schedule_started, loop.time()),
            self.waiting_queue.qsize(),
            self._adaptive_debug_state(loop.time()),
        )

    async def _get_next_batch(self) -> List[EngineRequest]:
        if self.scheduler_config.adaptive_interval_enabled:
            return await self._get_next_adaptive_batch()

        return await self._get_next_fixed_batch()

    async def _get_next_fixed_batch(self) -> List[EngineRequest]:
        max_batch_size = max(1, self.scheduler_config.max_batch_size)
        timeout_secs = max(0, self.scheduler_config.batch_wait_timeout_ms) / 1000

        first_request = await self.waiting_queue.get()
        batch = [first_request]
        if max_batch_size == 1:
            return batch

        deadline = asyncio.get_running_loop().time() + timeout_secs
        while len(batch) < max_batch_size:
            try:
                batch.append(self.waiting_queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break

            try:
                request = await asyncio.wait_for(
                    self.waiting_queue.get(),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                break
            batch.append(request)

        return batch

    async def _get_next_adaptive_batch(self) -> List[EngineRequest]:
        max_batch_size = max(1, self.scheduler_config.max_batch_size)
        loop = asyncio.get_running_loop()
        batch_wait_started = loop.time()

        first_request = await self.waiting_queue.get()
        batch = [first_request]

        dispatch_reason = self._adaptive_immediate_dispatch_reason()
        if dispatch_reason is None:
            logger.debug(
                "Adaptive scheduling wait start request_id=%s queue_size=%s %s",
                first_request.request_id,
                self.waiting_queue.qsize() + len(batch),
                self._adaptive_debug_state(loop.time(), first_request),
            )
            dispatch_reason = await self._wait_for_adaptive_dispatch_ready(first_request)

        while len(batch) < max_batch_size:
            try:
                batch.append(self.waiting_queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        logger.debug(
            "Adaptive scheduling batch ready reason=%s batch_size=%s wait_ms=%.2f "
            "queue_remaining=%s %s",
            dispatch_reason,
            len(batch),
            self._elapsed_ms(batch_wait_started, loop.time()),
            self.waiting_queue.qsize(),
            self._adaptive_debug_state(loop.time(), first_request),
        )

        return batch

    def _increment_worker_loads(self, workers: List[Optional[Worker]]) -> None:
        load_by_worker: Dict[Worker, int] = {}
        for worker in workers:
            if worker is None:
                continue
            load_by_worker[worker] = load_by_worker.get(worker, 0) + 1

        for worker, load in load_by_worker.items():
            worker.increment_load(load)

    def _release_worker_id(self, engine_request: EngineRequest) -> str:
        if engine_request.worker_rank == -1:
            return engine_request.worker_url
        return f"{engine_request.worker_url}@{engine_request.worker_rank}"

    def _record_forward_time(
        self,
        worker_type: WorkerType,
        forward_time_ms: Optional[float],
    ) -> None:
        if (
            not self.scheduler_config.adaptive_interval_enabled
            or forward_time_ms is None
            or forward_time_ms <= 0
        ):
            return

        window = self._forward_time_windows.get(worker_type)
        if window is None:
            return

        window.append(float(forward_time_ms))
        self._current_interval_secs = self._recompute_adaptive_interval_secs()
        logger.debug(
            "Adaptive forward sample worker_type=%s sample_ms=%.2f "
            "window=%s/%s mean_ms=%.2f interval_ms=%.2f",
            worker_type.value,
            forward_time_ms,
            len(window),
            window.maxlen,
            self._mean_forward_time_secs(worker_type) * 1000,
            self._current_interval_secs * 1000,
        )

    def _recompute_adaptive_interval_secs(self) -> float:
        network_latency_secs = max(0.0, self.scheduler_config.network_latency_ms) / 1000
        intervals = []

        prefill_active = self._active_worker_count(WorkerType.PREFILL)
        if prefill_active > 0:
            intervals.append(
                (
                    self._mean_forward_time_secs(WorkerType.PREFILL)
                    + network_latency_secs
                )
                / prefill_active
            )

        decode_active = self._active_worker_count(WorkerType.DECODE)
        if decode_active > 0:
            intervals.append(
                (
                    self._mean_forward_time_secs(WorkerType.DECODE)
                    + network_latency_secs
                )
                / decode_active
            )

        if intervals:
            interval_secs = max(intervals)
        else:
            interval_secs = (
                max(0.0, self.scheduler_config.default_forward_time_ms) / 1000
                + network_latency_secs
            )

        min_interval_secs = max(0.0, self.scheduler_config.min_interval_ms) / 1000
        max_interval_ms = self.scheduler_config.max_interval_ms
        if max_interval_ms > 0:
            max_interval_secs = max(min_interval_secs, max_interval_ms / 1000)
            interval_secs = min(interval_secs, max_interval_secs)

        return max(interval_secs, min_interval_secs)

    def _mean_forward_time_secs(self, worker_type: WorkerType) -> float:
        window = self._forward_time_windows.get(worker_type)
        if window:
            return (sum(window) / len(window)) / 1000
        return max(0.0, self.scheduler_config.default_forward_time_ms) / 1000

    def _active_worker_count(self, worker_type: WorkerType) -> int:
        return sum(
            1
            for worker in self.worker_registry.get_by_type(worker_type)
            if worker.is_available()
        )

    def _has_active_worker_load(self) -> bool:
        return any(worker.load() > 0 for worker in self.worker_registry.get_all())

    def _adaptive_immediate_dispatch_reason(self) -> Optional[str]:
        if self._last_dispatch_time is None:
            return "first_dispatch"

        if not self._has_active_worker_load():
            self._ready_event.set()
            self._watchdog_deadline = None
            return "idle"

        return None

    async def _wait_for_adaptive_dispatch_ready(
        self,
        first_request: EngineRequest,
    ) -> str:
        loop = asyncio.get_running_loop()
        wait_started = loop.time()

        while True:
            now = loop.time()
            self._current_interval_secs = self._recompute_adaptive_interval_secs()
            interval_elapsed = (
                self._last_dispatch_time is None
                or now - self._last_dispatch_time >= self._current_interval_secs
            )

            if not self._has_active_worker_load():
                self._ready_event.set()
                self._watchdog_deadline = None

            if (
                self._watchdog_deadline is not None
                and now >= self._watchdog_deadline
            ):
                logger.warning(
                    "Adaptive scheduling watchdog expired; forcing readiness "
                    "wait_ms=%.2f %s",
                    self._elapsed_ms(wait_started, now),
                    self._adaptive_debug_state(now, first_request),
                )
                self._watchdog_deadline = None
                self._ready_event.set()

            schedule_deadline = self._schedule_response_deadline(first_request)
            schedule_deadline_elapsed = False
            if schedule_deadline is not None and now >= schedule_deadline:
                logger.warning(
                    "Adaptive scheduling reached schedule-response deadline; "
                    "forcing readiness wait_ms=%.2f %s",
                    self._elapsed_ms(wait_started, now),
                    self._adaptive_debug_state(now, first_request),
                )
                self._ready_event.set()
                schedule_deadline_elapsed = True

            if schedule_deadline_elapsed:
                self._ready_event.clear()
                return "schedule_deadline"

            if interval_elapsed:
                self._ready_event.clear()
                logger.debug(
                    "Adaptive scheduling wait finish reason=interval_elapsed "
                    "wait_ms=%.2f %s",
                    self._elapsed_ms(wait_started, now),
                    self._adaptive_debug_state(now, first_request),
                )
                return "interval_elapsed"

            timeout_candidates = []
            if not interval_elapsed and self._last_dispatch_time is not None:
                timeout_candidates.append(
                    max(
                        0.0,
                        self._last_dispatch_time
                        + self._current_interval_secs
                        - now,
                    )
                )
            if self._watchdog_deadline is not None:
                timeout_candidates.append(max(0.0, self._watchdog_deadline - now))
            if schedule_deadline is not None:
                timeout_candidates.append(max(0.0, schedule_deadline - now))

            positive_timeouts = [
                timeout for timeout in timeout_candidates if timeout > 0
            ]
            timeout = min(positive_timeouts) if positive_timeouts else 0.05

            # ready_event is only a wake-up signal now, not a dispatch gate.
            # Clear any stale signal before waiting so an already-set event does
            # not spin this loop until the interval deadline.
            self._ready_event.clear()
            try:
                await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    def _adaptive_debug_state(
        self,
        now: Optional[float] = None,
        first_request: Optional[EngineRequest] = None,
    ) -> str:
        if not self.scheduler_config.adaptive_interval_enabled:
            return "adaptive=off"

        if now is None:
            try:
                now = asyncio.get_running_loop().time()
            except RuntimeError:
                now = None

        prefill_window = self._forward_time_windows[WorkerType.PREFILL]
        decode_window = self._forward_time_windows[WorkerType.DECODE]
        since_last_dispatch_ms = (
            self._elapsed_ms(self._last_dispatch_time, now)
            if now is not None
            else -1
        )
        watchdog_remaining_ms = self._remaining_ms(self._watchdog_deadline, now)
        schedule_deadline = (
            self._schedule_response_deadline(first_request)
            if first_request is not None
            else None
        )
        schedule_deadline_remaining_ms = self._remaining_ms(schedule_deadline, now)

        return (
            "adaptive=on "
            f"max_batch_size={max(1, self.scheduler_config.max_batch_size)} "
            f"schedule_timeout_ms={self.scheduler_config.schedule_response_timeout_ms} "
            f"schedule_send_margin_ms={self.scheduler_config.schedule_response_send_margin_ms} "
            f"interval_ms={self._current_interval_secs * 1000:.2f} "
            f"max_interval_ms={self.scheduler_config.max_interval_ms:.2f} "
            f"prefill_window={len(prefill_window)}/{prefill_window.maxlen} "
            f"prefill_mean_ms={self._mean_forward_time_secs(WorkerType.PREFILL) * 1000:.2f} "
            f"prefill_active={self._active_worker_count(WorkerType.PREFILL)} "
            f"decode_window={len(decode_window)}/{decode_window.maxlen} "
            f"decode_mean_ms={self._mean_forward_time_secs(WorkerType.DECODE) * 1000:.2f} "
            f"decode_active={self._active_worker_count(WorkerType.DECODE)} "
            f"ready={self._ready_event.is_set()} "
            f"active_load={self._has_active_worker_load()} "
            f"since_last_dispatch_ms={since_last_dispatch_ms:.2f} "
            f"watchdog_remaining_ms={watchdog_remaining_ms:.2f} "
            f"schedule_deadline_remaining_ms={schedule_deadline_remaining_ms:.2f}"
        )

    def _oldest_enqueue_time(
        self,
        batch: List[EngineRequest],
    ) -> Optional[float]:
        enqueue_times = [
            request.enqueue_time
            for request in batch
            if request.enqueue_time is not None
        ]
        return min(enqueue_times) if enqueue_times else None

    def _elapsed_ms(
        self,
        started: Optional[float],
        ended: Optional[float],
    ) -> float:
        if started is None or ended is None:
            return -1.0
        return max(0.0, ended - started) * 1000

    def _remaining_ms(
        self,
        deadline: Optional[float],
        now: Optional[float],
    ) -> float:
        if deadline is None or now is None:
            return -1.0
        return max(0.0, deadline - now) * 1000

    def _schedule_response_deadline(
        self,
        first_request: EngineRequest,
    ) -> Optional[float]:
        if first_request.enqueue_time is None:
            return None

        timeout_secs = (
            max(1, self.scheduler_config.schedule_response_timeout_ms) / 1000
        )
        send_margin_secs = min(
            max(0.0, self.scheduler_config.schedule_response_send_margin_ms) / 1000,
            timeout_secs * 0.9,
        )
        return first_request.enqueue_time + max(0.0, timeout_secs - send_margin_secs)

    def _mark_batch_dispatched(self) -> None:
        if not self.scheduler_config.adaptive_interval_enabled:
            return

        loop = asyncio.get_running_loop()
        self._last_dispatch_time = loop.time()

        if self._has_active_worker_load():
            self._ready_event.clear()
            watchdog_secs = self._watchdog_timeout_secs()
            self._watchdog_deadline = (
                self._last_dispatch_time + watchdog_secs
                if watchdog_secs > 0
                else None
            )
            return

        self._watchdog_deadline = None
        self._ready_event.set()

    def _watchdog_timeout_secs(self) -> float:
        multiplier = max(0.0, self.scheduler_config.watchdog_multiplier)
        if multiplier == 0:
            return 0.0

        return multiplier * max(
            self._mean_forward_time_secs(WorkerType.PREFILL),
            self._mean_forward_time_secs(WorkerType.DECODE),
        )

    def _worker_base_url(self, worker: Optional[Worker]) -> Optional[str]:
        if worker is None:
            return None
        return worker.base_url()

    def _worker_dp_rank(self, worker: Optional[Worker]) -> int:
        if worker is None:
            return -1
        return worker.dp_rank()

    async def send_response(self, request: EngineRequest, msg: Dict[str, Any]) -> None:
        await self.output_socket.send_multipart([
            request.identity.encode("utf-8"),
            b"",
            json.dumps(msg).encode("utf-8"),
        ])

    async def run(self):
        """
        receive_loop: request  -> prefill_waiting_queue 
        prefill_loop: prefill_waiting_queue -> request -> handle prefill -> decode_waiting_queue
        decode_loop: decode_waiting_queue -> request -> handle decode
        """
        await asyncio.gather(
            self.receive_loop(),
            self.schedule_loop(),
        )

    async def shutdown(self):
        """Gracefully shutdown the engine:
        1. stopping receiving new requests;
        2. waiting for in-flight tasks to complete."""
        # stop receiving new requests
        self.input_socket.close(0)

        # closesocket
        self.output_socket.close(0)

    def schedule_prefill(
        self,
        request_text: str,
        headers: Dict[str, str],
    ) -> Optional[Worker]:
        raise NotImplementedError
    
    def schedule_decode(
        self,
        request_text: str,
        headers: Dict[str, str],
    ) -> Optional[Worker]:
        raise NotImplementedError

    def schedule_prefill_batch(
        self,
        requests: List[EngineRequest],
    ) -> List[Optional[Worker]]:
        return [
            self.schedule_prefill(request.request_text, request.headers)
            for request in requests
        ]

    def schedule_decode_batch(
        self,
        requests: List[EngineRequest],
    ) -> List[Optional[Worker]]:
        return [
            self.schedule_decode(request.request_text, request.headers)
            for request in requests
        ]
    
    
