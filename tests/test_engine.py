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
