from smart_router.engine.engine import EngineRequest, RequestType


def test_engine_request_round_trip():
    request = EngineRequest(
        request_id="req-1",
        identity="client-1",
        request_type=RequestType.SCHEDULE,
        request_text="hello",
        headers={"x-test": "true"},
    )

    restored = EngineRequest.from_dict(request.to_dict())

    assert restored == request


def test_engine_request_release_forward_time_round_trip_and_backward_compatibility():
    request = EngineRequest(
        request_id="req-2",
        identity="client-1",
        request_type=RequestType.RELEASE,
        worker_url="http://worker",
        worker_rank=0,
        forward_time_ms=12.5,
    )

    restored = EngineRequest.from_dict(request.to_dict())
    legacy = EngineRequest.from_dict(
        {
            "request_id": "req-3",
            "identity": "client-1",
            "request_type": RequestType.RELEASE,
        }
    )

    assert restored == request
    assert legacy.forward_time_ms is None
    assert legacy.worker_rank == -1
