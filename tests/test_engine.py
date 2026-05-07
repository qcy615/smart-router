import asyncio
import os
import time
from multiprocessing import Process

from smart_router.engine.engine_client import EngineClient
from smart_router.engine.engine import Engine, EngineRequest, RequestType



def test_engine_and_engine_client():
    addr = "tcp://127.0.0.1:5557"

    engine = Engine(addr, "tcp://127.0.0.1:5558")

    async def run_engine():
        await engine.run()

    asyncio.run(run_engine())


def test_engine_request_serializes_prompt_token_ids():
    request = EngineRequest(
        request_id="req",
        identity="identity",
        request_type=RequestType.SCHEDULE,
        request_text="1 2 3",
        request_body={"prompt": [1, 2, 3]},
        api_kind="completions",
        prompt_token_ids=[1, 2, 3],
    )

    restored = EngineRequest.from_dict(request.to_dict())

    assert restored.prompt_token_ids == [1, 2, 3]
    assert restored.request_body == {"prompt": [1, 2, 3]}

if __name__ == "__main__":
    test_engine_and_engine_client()
