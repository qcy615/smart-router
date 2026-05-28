from __future__ import annotations

import logging
import os
import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any, Optional

from smart_router.config.tokenization import TokenizationConfig

logger = logging.getLogger(__name__)


class RouterTokenizerManager:
    def __init__(self, config: Optional[TokenizationConfig] = None) -> None:
        self.config = config or TokenizationConfig()
        self._tokenizers: dict[str, Any] = {}
        self._http_client: Optional[Any] = None
        self._cache: OrderedDict[str, tuple[float, list[int]]] = OrderedDict()
        self._inflight: dict[str, asyncio.Future[list[int]]] = {}
        self._cache_lock = asyncio.Lock()

    @classmethod
    def from_env(cls) -> "RouterTokenizerManager":
        trust_remote_code = os.getenv(
            "SMART_ROUTER_TOKENIZER_TRUST_REMOTE_CODE", "").lower()
        return cls(
            TokenizationConfig(
                tokenizer=os.getenv("SMART_ROUTER_TOKENIZER") or None,
                tokenizer_trust_remote_code=trust_remote_code in {"1", "true", "yes"},
                tokenize_url=os.getenv("SMART_ROUTER_TOKENIZE_URL") or None,
                tokenize_cache_size=int(
                    os.getenv("SMART_ROUTER_TOKENIZE_CACHE_SIZE", "4096")),
                tokenize_cache_ttl=float(
                    os.getenv("SMART_ROUTER_TOKENIZE_CACHE_TTL", "3600")),
            ))

    async def encode_request(self, body: dict[str, Any],
                             api_kind: str) -> list[int]:
        explicit_token_ids = self._extract_explicit_token_ids(body)
        if explicit_token_ids:
            logger.info(
                "[ROUTER-TOKENIZE] source=explicit_body api_kind=%s token_count=%d",
                api_kind,
                len(explicit_token_ids),
            )
            return explicit_token_ids

        cache_key = self._cache_key(body, api_kind)
        if cache_key is not None:
            return await self._get_or_compute_cached(
                cache_key, lambda: self._encode_request_uncached(body, api_kind))

        return await self._encode_request_uncached(body, api_kind)

    async def _encode_request_uncached(self, body: dict[str, Any],
                                       api_kind: str) -> list[int]:
        if self.config.tokenize_url:
            token_ids = await self._encode_remote(body, api_kind)
            if token_ids:
                return token_ids

        return self.encode_request_local(body, api_kind)

    async def _get_or_compute_cached(self, cache_key: str,
                                     compute) -> list[int]:
        cached = await self._get_cached(cache_key)
        if cached is not None:
            logger.debug("[ROUTER-TOKENIZE] source=cache token_count=%d",
                         len(cached))
            return cached

        loop = asyncio.get_running_loop()
        async with self._cache_lock:
            cached = self._get_cached_locked(cache_key)
            if cached is not None:
                logger.debug("[ROUTER-TOKENIZE] source=cache token_count=%d",
                             len(cached))
                return cached
            fut = self._inflight.get(cache_key)
            if fut is None:
                fut = loop.create_future()
                self._inflight[cache_key] = fut
                leader = True
            else:
                leader = False

        if not leader:
            return list(await fut)

        try:
            token_ids = await compute()
            if token_ids:
                await self._set_cached(cache_key, token_ids)
            fut.set_result(list(token_ids))
            return token_ids
        except Exception as exc:
            fut.set_exception(exc)
            fut.add_done_callback(_consume_future_exception)
            raise
        finally:
            async with self._cache_lock:
                self._inflight.pop(cache_key, None)

    async def _get_cached(self, cache_key: str) -> Optional[list[int]]:
        async with self._cache_lock:
            return self._get_cached_locked(cache_key)

    def _get_cached_locked(self, cache_key: str) -> Optional[list[int]]:
        item = self._cache.get(cache_key)
        if item is None:
            return None
        expires_at, token_ids = item
        if expires_at <= time.monotonic():
            self._cache.pop(cache_key, None)
            return None
        self._cache.move_to_end(cache_key)
        return list(token_ids)

    async def _set_cached(self, cache_key: str, token_ids: list[int]) -> None:
        if self.config.tokenize_cache_size <= 0:
            return
        expires_at = time.monotonic() + self.config.tokenize_cache_ttl
        async with self._cache_lock:
            self._cache[cache_key] = (expires_at, list(token_ids))
            self._cache.move_to_end(cache_key)
            while len(self._cache) > self.config.tokenize_cache_size:
                self._cache.popitem(last=False)

    def encode_request_local(self, body: dict[str, Any],
                             api_kind: str) -> list[int]:
        tokenizer_name = self.config.tokenizer or body.get("model")
        if not tokenizer_name or not isinstance(tokenizer_name, str):
            logger.debug("No tokenizer configured and request model is missing")
            return []

        try:
            tokenizer = self._get_tokenizer(tokenizer_name)
            if api_kind == "chat" and isinstance(body.get("messages"), list):
                token_ids = self._encode_chat(tokenizer, body)
            else:
                token_ids = self._encode_completion(tokenizer, body)
            logger.info(
                "[ROUTER-TOKENIZE] source=router_tokenizer api_kind=%s "
                "tokenizer=%s token_count=%d",
                api_kind,
                tokenizer_name,
                len(token_ids),
            )
            return token_ids
        except Exception:
            logger.warning("Failed to tokenize router request", exc_info=True)
            return []

    async def _encode_remote(self, body: dict[str, Any],
                             api_kind: str) -> list[int]:
        payload = self._build_remote_tokenize_payload(body, api_kind)
        if payload is None:
            return []

        url = self.config.tokenize_url
        assert url is not None
        try:
            response = await self._get_http_client().post(url, json=payload)
            if not response.is_success:
                logger.warning(
                    "[ROUTER-TOKENIZE] source=vllm_tokenize status=%s "
                    "url=%s body=%s",
                    response.status_code,
                    url,
                    response.text[:500],
                )
                return []
            data = response.json()
            tokens = data.get("tokens")
            if isinstance(tokens, list) and all(
                    isinstance(token_id, int) for token_id in tokens):
                logger.info(
                    "[ROUTER-TOKENIZE] source=vllm_tokenize api_kind=%s "
                    "url=%s token_count=%d max_model_len=%s",
                    api_kind,
                    url,
                    len(tokens),
                    data.get("max_model_len"),
                )
                return tokens
            logger.warning(
                "[ROUTER-TOKENIZE] source=vllm_tokenize invalid response url=%s "
                "keys=%s",
                url,
                list(data.keys()) if isinstance(data, dict) else type(data),
            )
        except Exception:
            logger.warning(
                "[ROUTER-TOKENIZE] source=vllm_tokenize failed url=%s",
                url,
                exc_info=True,
            )
        return []

    def _get_http_client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for smart-router remote vLLM tokenization."
            ) from exc
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.tokenize_timeout),
            limits=httpx.Limits(max_connections=512,
                                max_keepalive_connections=128),
        )
        return self._http_client

    def _cache_key(self, body: dict[str, Any], api_kind: str) -> Optional[str]:
        payload = self._build_remote_tokenize_payload(body, api_kind)
        if payload is None:
            payload = self._build_local_tokenize_payload(body, api_kind)
        if payload is None:
            return None
        raw = json.dumps(payload,
                         ensure_ascii=False,
                         sort_keys=True,
                         separators=(",", ":"))
        digest = hashlib.blake2b(raw.encode("utf-8"),
                                 digest_size=16).hexdigest()
        source = self.config.tokenize_url or self.config.tokenizer or body.get(
            "model", "")
        return f"{api_kind}:{source}:{digest}"

    def _build_remote_tokenize_payload(
            self, body: dict[str, Any], api_kind: str) -> Optional[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": body.get("model"),
            "return_token_strs": False,
        }
        if api_kind == "chat":
            messages = body.get("messages")
            if not isinstance(messages, list):
                return None
            payload["messages"] = messages
            self._copy_optional_fields(
                body,
                payload,
                [
                    "add_generation_prompt",
                    "continue_final_message",
                    "add_special_tokens",
                    "chat_template",
                    "chat_template_kwargs",
                    "mm_processor_kwargs",
                    "tools",
                ],
            )
            return payload

        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            return None
        payload["prompt"] = prompt
        self._copy_optional_fields(body, payload, ["add_special_tokens"])
        return payload

    def _copy_optional_fields(self, source: dict[str, Any],
                              target: dict[str, Any],
                              fields: list[str]) -> None:
        for field in fields:
            if field in source:
                target[field] = source[field]

    def _build_local_tokenize_payload(
            self, body: dict[str, Any], api_kind: str) -> Optional[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": body.get("model"),
            "tokenizer": self.config.tokenizer,
        }
        if api_kind == "chat":
            messages = body.get("messages")
            if not isinstance(messages, list):
                return None
            payload["messages"] = messages
            payload["add_generation_prompt"] = body.get("add_generation_prompt",
                                                        True)
            return payload
        prompt = body.get("prompt")
        if isinstance(prompt, (str, list)):
            payload["prompt"] = prompt
            payload["add_special_tokens"] = body.get("add_special_tokens", False)
            return payload
        return None

    def _get_tokenizer(self, tokenizer_name: str) -> Any:
        tokenizer = self._tokenizers.get(tokenizer_name)
        if tokenizer is not None:
            return tokenizer

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for smart-router tokenization. "
                "Install it or pass prompt_token_ids explicitly.") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=self.config.tokenizer_trust_remote_code,
        )
        self._tokenizers[tokenizer_name] = tokenizer
        logger.info("Loaded router tokenizer: %s", tokenizer_name)
        return tokenizer

    def _encode_chat(self, tokenizer: Any, body: dict[str, Any]) -> list[int]:
        messages = body["messages"]
        try:
            token_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=body.get("add_generation_prompt", True),
            )
            return list(token_ids)
        except Exception:
            logger.debug("apply_chat_template failed, falling back to content-only encoding",
                         exc_info=True)

        text_parts: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content", "")
                text_parts.append(self._stringify_content(content))
        return self._encode_text(tokenizer, "\n".join(text_parts))

    def _encode_completion(self, tokenizer: Any,
                           body: dict[str, Any]) -> list[int]:
        prompt = body.get("prompt", "")
        if isinstance(prompt, str):
            return self._encode_text(tokenizer, prompt)
        if isinstance(prompt, list):
            if all(isinstance(item, int) for item in prompt):
                return list(prompt)
            if all(isinstance(item, str) for item in prompt):
                return self._encode_text(tokenizer, "\n".join(prompt))
        return self._encode_text(tokenizer, str(prompt))

    def _encode_text(self, tokenizer: Any, text: str) -> list[int]:
        return list(tokenizer.encode(text, add_special_tokens=False))

    def _extract_explicit_token_ids(self, body: dict[str, Any]) -> list[int]:
        prompt_token_ids = body.get("prompt_token_ids")
        if isinstance(prompt_token_ids, list) and all(
                isinstance(token_id, int) for token_id in prompt_token_ids):
            return list(prompt_token_ids)

        prompt = body.get("prompt")
        if isinstance(prompt, list) and all(
                isinstance(token_id, int) for token_id in prompt):
            return list(prompt)

        return []

    def _stringify_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        return str(content)


def _consume_future_exception(fut: asyncio.Future) -> None:
    try:
        fut.exception()
    except asyncio.CancelledError:
        pass
