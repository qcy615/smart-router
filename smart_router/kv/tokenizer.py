from __future__ import annotations

import json
import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...


class RequestTokenizer:
    def __init__(self, tokenizer_path: str, tokenizer: TokenizerLike | None = None) -> None:
        if tokenizer is not None:
            self._tokenizer = tokenizer
            return

        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "kv_aware routing requires transformers for local tokenization. "
                "Install the benchmark/kv dependencies or provide a tokenizer object in tests."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def tokenize_request(self, body: dict[str, Any] | None, api_kind: str | None) -> list[int] | None:
        body = body or {}
        if self._has_multimodal_payload(body):
            logger.debug("KV-aware routing fallback: multimodal request is unsupported")
            return None

        if api_kind == "chat" or "messages" in body:
            messages = body.get("messages")
            if not isinstance(messages, list):
                return None
            return self._tokenize_messages(messages)

        prompt = body.get("prompt")
        if isinstance(prompt, str):
            return list(self._tokenizer.encode(prompt, add_special_tokens=False))
        if prompt is not None:
            logger.debug("KV-aware routing fallback: batched/non-string prompt is unsupported")
            return None

        return list(self._tokenizer.encode(json.dumps(body, ensure_ascii=False), add_special_tokens=False))

    def _tokenize_messages(self, messages: list[Any]) -> list[int] | None:
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                token_ids = apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                return list(token_ids)
            except Exception:
                logger.exception("Failed to apply tokenizer chat template")
                return None

        text_parts: list[str] = []
        for message in messages:
            if not isinstance(message, dict):
                return None
            content = message.get("content", "")
            if not isinstance(content, str):
                return None
            role = message.get("role", "")
            text_parts.append(f"{role}: {content}")
        text_parts.append("assistant:")
        return list(self._tokenizer.encode("\n".join(text_parts), add_special_tokens=False))

    def _has_multimodal_payload(self, body: dict[str, Any]) -> bool:
        if any(key in body for key in ("multi_modal_data", "mm_processor_kwargs", "image", "images")):
            return True

        messages = body.get("messages")
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                return True
        return False
