import httpx

from smart_router.config.upstream_http import UpstreamHTTPClientConfig


def build_upstream_http_client(
    config: UpstreamHTTPClientConfig | None = None,
) -> httpx.AsyncClient:
    config = config or UpstreamHTTPClientConfig()
    timeout = httpx.Timeout(
        connect=config.connect_timeout_secs,
        read=config.read_timeout_secs,
        write=config.write_timeout_secs,
        pool=config.pool_timeout_secs,
    )
    limits = httpx.Limits(
        max_connections=config.max_connections,
        max_keepalive_connections=config.max_keepalive_connections,
        keepalive_expiry=config.keepalive_expiry_secs,
    )
    return httpx.AsyncClient(timeout=timeout, limits=limits)
