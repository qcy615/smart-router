import asyncio

from smart_router.config import UpstreamHTTPClientConfig
from smart_router.entrypoints.serve.http_client import build_upstream_http_client


def test_build_upstream_http_client_uses_configured_timeouts_and_limits():
    config = UpstreamHTTPClientConfig(
        connect_timeout_secs=2.0,
        read_timeout_secs=120.0,
        write_timeout_secs=15.0,
        pool_timeout_secs=0.5,
        max_connections=12,
        max_keepalive_connections=3,
        keepalive_expiry_secs=4.0,
    )

    client = build_upstream_http_client(config)
    try:
        assert client.timeout.connect == 2.0
        assert client.timeout.read == 120.0
        assert client.timeout.write == 15.0
        assert client.timeout.pool == 0.5

        pool = client._transport._pool
        assert pool._max_connections == 12
        assert pool._max_keepalive_connections == 3
        assert pool._keepalive_expiry == 4.0
    finally:
        asyncio.run(client.aclose())


def test_build_upstream_http_client_disables_read_timeout_by_default():
    client = build_upstream_http_client()
    try:
        assert client.timeout.read is None
    finally:
        asyncio.run(client.aclose())
