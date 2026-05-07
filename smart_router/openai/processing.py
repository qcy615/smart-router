from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


class ChatTokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ...


@dataclass(frozen=True)
class OpenAIProcessingConfig:
    enabled: bool = False
    tokenizer_path: str | None = None
    reasoning_parser: str = "think_tags"
    tool_call_parser: str = "hermes"


@dataclass
class PreprocessedChatRequest:
    prompt_token_ids: list[int]
    completion_body: dict[str, Any]
    request_text: str


@dataclass
class TokenInfo:
    text: str | None = None
    token_ids: list[int] = field(default_factory=list)


@dataclass
class CompletionDelta:
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    logprobs: Any = None
    id: str | None = None
    model: str | None = None
    created: int | None = None

    def has_non_empty_token(self) -> bool:
        return bool(self.token_ids) or bool(self.text)


class _DoneEvent:
    pass


DONE = _DoneEvent()


class OpenAIProcessingError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class OpenAIChatProcessor:
    def __init__(
        self,
        config: OpenAIProcessingConfig,
        tokenizer: ChatTokenizerLike | None = None,
    ) -> None:
        self.config = config
        self._tokenizer = tokenizer or self._load_tokenizer(config.tokenizer_path)

    def preprocess_chat(self, body: dict[str, Any]) -> PreprocessedChatRequest:
        messages = body.get("messages")
        if not isinstance(messages, list):
            raise OpenAIProcessingError("`messages` must be a list")
        if self._has_multimodal_messages(messages):
            raise OpenAIProcessingError("local OpenAI processing does not support multimodal chat content")

        n = body.get("n", 1)
        if n not in (None, 1):
            raise OpenAIProcessingError("local OpenAI processing currently supports only n=1")

        prompt_token_ids = self._apply_chat_template(body, messages)
        completion_body = self._build_completion_body(body, prompt_token_ids)
        return PreprocessedChatRequest(
            prompt_token_ids=prompt_token_ids,
            completion_body=completion_body,
            request_text=" ".join(str(token_id) for token_id in prompt_token_ids),
        )

    def create_chat_builder(
        self,
        processed: PreprocessedChatRequest,
        original_body: dict[str, Any],
    ) -> "ChatResponseBuilder":
        return ChatResponseBuilder(self, processed, original_body)

    def completion_delta_from_payload(self, payload: dict[str, Any]) -> CompletionDelta | None:
        choices = payload.get("choices") or []
        if not choices:
            return None

        choice = choices[0] or {}
        text = choice.get("text")
        if text is None:
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
        if text is None:
            text = ""

        return CompletionDelta(
            text=text if isinstance(text, str) else str(text),
            token_ids=self._extract_choice_token_ids(choice),
            finish_reason=choice.get("finish_reason"),
            usage=payload.get("usage") if isinstance(payload.get("usage"), dict) else None,
            logprobs=choice.get("logprobs"),
            id=payload.get("id") if isinstance(payload.get("id"), str) else None,
            model=payload.get("model") if isinstance(payload.get("model"), str) else None,
            created=payload.get("created") if isinstance(payload.get("created"), int) else None,
        )

    def completion_delta_from_response_json(self, payload: dict[str, Any]) -> CompletionDelta | None:
        return self.completion_delta_from_payload(payload)

    def extract_first_completion_token(self, payload: dict[str, Any]) -> TokenInfo:
        delta = self.completion_delta_from_response_json(payload)
        if delta is None:
            return TokenInfo()
        token_ids = delta.token_ids[:1]
        return TokenInfo(text=delta.text or None, token_ids=token_ids)

    def delta_matches_token(self, delta: CompletionDelta, token: TokenInfo) -> bool:
        if not delta.has_non_empty_token():
            return False
        if token.token_ids and delta.token_ids:
            return token.token_ids[0] == delta.token_ids[0]
        return bool(token.text and delta.text == token.text)

    def _load_tokenizer(self, tokenizer_path: str | None) -> ChatTokenizerLike:
        if not tokenizer_path:
            raise RuntimeError(
                "local OpenAI processing requires --openai-tokenizer-path or --kv-tokenizer-path"
            )
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "local OpenAI processing requires transformers for tokenization"
            ) from exc
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def _apply_chat_template(self, body: dict[str, Any], messages: list[Any]) -> list[int]:
        args = body.get("chat_template_args", [])
        kwargs = body.get("chat_template_kwargs", {})
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        if not isinstance(args, list):
            raise OpenAIProcessingError("`chat_template_args` must be a list")
        if not isinstance(kwargs, dict):
            raise OpenAIProcessingError("`chat_template_kwargs` must be an object")

        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            **kwargs,
        }
        for key in ("tools", "tool_choice", "response_format"):
            if key in body and key not in template_kwargs:
                template_kwargs[key] = body[key]

        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            token_ids = apply_chat_template(messages, *args, **template_kwargs)
            return self._coerce_token_ids(token_ids)

        text = self._fallback_render_messages(messages)
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _build_completion_body(
        self,
        body: dict[str, Any],
        prompt_token_ids: list[int],
    ) -> dict[str, Any]:
        completion_body = {
            key: value
            for key, value in body.items()
            if key
            not in {
                "messages",
                "tools",
                "tool_choice",
                "response_format",
                "chat_template_args",
                "chat_template_kwargs",
                "max_completion_tokens",
                "parallel_tool_calls",
            }
        }
        completion_body["prompt"] = list(prompt_token_ids)
        completion_body["return_token_ids"] = True

        if "max_completion_tokens" in body and "max_tokens" not in completion_body:
            completion_body["max_tokens"] = body["max_completion_tokens"]

        logprobs = body.get("logprobs")
        if isinstance(logprobs, bool):
            if logprobs:
                completion_body["logprobs"] = body.get("top_logprobs") or 1
            else:
                completion_body.pop("logprobs", None)
        elif logprobs is not None:
            completion_body["logprobs"] = logprobs

        return completion_body

    def _coerce_token_ids(self, token_ids: Any) -> list[int]:
        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        if isinstance(token_ids, tuple):
            token_ids = list(token_ids)
        if not isinstance(token_ids, list):
            raise OpenAIProcessingError("chat template did not return token ids")
        if token_ids and isinstance(token_ids[0], list):
            raise OpenAIProcessingError("batched chat template output is unsupported")
        return [int(token_id) for token_id in token_ids]

    def _has_multimodal_messages(self, messages: list[Any]) -> bool:
        for message in messages:
            if not isinstance(message, dict):
                raise OpenAIProcessingError("each chat message must be an object")
            content = message.get("content")
            if isinstance(content, list):
                return True
            if content is not None and not isinstance(content, str):
                raise OpenAIProcessingError("local OpenAI processing supports only text chat content")
        return False

    def _fallback_render_messages(self, messages: list[Any]) -> str:
        parts: list[str] = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content") or ""
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def _extract_choice_token_ids(self, choice: dict[str, Any]) -> list[int]:
        for key in ("token_ids", "output_token_ids"):
            value = choice.get(key)
            if isinstance(value, list):
                return [int(token_id) for token_id in value]

        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            value = logprobs.get("token_ids")
            if isinstance(value, list):
                return [int(token_id) for token_id in value]
        return []


class CompletionSSEDecoder:
    def __init__(self, processor: OpenAIChatProcessor) -> None:
        self.processor = processor
        self._buffer = ""

    def feed(self, chunk: bytes) -> list[CompletionDelta | _DoneEvent]:
        self._buffer += chunk.decode("utf-8", errors="replace")
        events: list[CompletionDelta | _DoneEvent] = []
        while "\n\n" in self._buffer:
            raw_event, self._buffer = self._buffer.split("\n\n", 1)
            event = self._parse_event(raw_event)
            if event is not None:
                events.append(event)
        return events

    def close(self) -> list[CompletionDelta | _DoneEvent]:
        if not self._buffer.strip():
            self._buffer = ""
            return []
        raw_event = self._buffer
        self._buffer = ""
        event = self._parse_event(raw_event)
        return [event] if event is not None else []

    def _parse_event(self, raw_event: str) -> CompletionDelta | _DoneEvent | None:
        data_lines: list[str] = []
        for line in raw_event.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data_lines.append(line[5:].strip())
        if not data_lines:
            return None

        data = "\n".join(data_lines)
        if data == "[DONE]":
            return DONE

        payload = json.loads(data)
        if not isinstance(payload, dict):
            return None
        return self.processor.completion_delta_from_payload(payload)


class ChatResponseBuilder:
    def __init__(
        self,
        processor: OpenAIChatProcessor,
        processed: PreprocessedChatRequest,
        original_body: dict[str, Any],
    ) -> None:
        self.processor = processor
        self.processed = processed
        self.original_body = original_body
        self.id = f"chatcmpl-{uuid.uuid4().hex}"
        self.model = str(original_body.get("model", ""))
        self.created = int(time.time())
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.finish_reason: str | None = None
        self.logprobs: Any = None
        self.backend_usage: dict[str, Any] | None = None
        self.generated_tokens = 0
        self._sent_role = False
        self._finished = False
        self._think_parser = ThinkTagsParser(
            enabled=processor.config.reasoning_parser == "think_tags"
        )
        self._tool_parser = HermesToolParser(
            enabled=processor.config.tool_call_parser == "hermes",
            strict=self._strict_tool_jail(original_body),
        )

    def apply_delta(self, delta: CompletionDelta) -> list[dict[str, Any]]:
        self._update_metadata(delta)
        if delta.has_non_empty_token():
            self.generated_tokens += len(delta.token_ids) if delta.token_ids else 1

        if delta.finish_reason:
            self.finish_reason = delta.finish_reason
        if not delta.text:
            return []

        payloads: list[dict[str, Any]] = []
        for kind, piece in self._think_parser.feed(delta.text):
            if kind == "reasoning":
                self.reasoning_parts.append(piece)
                payloads.append(self._chunk({"reasoning_content": piece}))
                continue

            content, tool_calls = self._tool_parser.feed(piece)
            if content:
                self.content_parts.append(content)
                payloads.append(self._chunk({"content": content}))
            for tool_call in tool_calls:
                tool_index = len(self.tool_calls)
                self.tool_calls.append(tool_call)
                payloads.append(self._chunk({"tool_calls": [self._stream_tool_call(tool_call, tool_index)]}))
        return payloads

    def finish_chunks(self) -> list[dict[str, Any]]:
        if self._finished:
            return []

        payloads: list[dict[str, Any]] = []
        for kind, piece in self._think_parser.finish():
            if kind == "reasoning":
                self.reasoning_parts.append(piece)
                payloads.append(self._chunk({"reasoning_content": piece}))
                continue

            content, tool_calls = self._tool_parser.feed(piece)
            if content:
                self.content_parts.append(content)
                payloads.append(self._chunk({"content": content}))
            for tool_call in tool_calls:
                tool_index = len(self.tool_calls)
                self.tool_calls.append(tool_call)
                payloads.append(self._chunk({"tool_calls": [self._stream_tool_call(tool_call, tool_index)]}))

        content, tool_calls = self._tool_parser.finish()
        if content:
            self.content_parts.append(content)
            payloads.append(self._chunk({"content": content}))
        for tool_call in tool_calls:
            tool_index = len(self.tool_calls)
            self.tool_calls.append(tool_call)
            payloads.append(self._chunk({"tool_calls": [self._stream_tool_call(tool_call, tool_index)]}))

        final_reason = self._final_finish_reason()
        payloads.append(
            self._chunk(
                {},
                finish_reason=final_reason,
                usage=self.usage() if self._include_stream_usage() else None,
            )
        )
        self.finish_reason = final_reason
        self._finished = True
        return payloads

    def to_response(self) -> dict[str, Any]:
        if not self._finished:
            self.finish_chunks()

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self.content_parts),
        }
        if self.reasoning_parts:
            message["reasoning_content"] = "".join(self.reasoning_parts)
        if self.tool_calls:
            message["content"] = None
            message["tool_calls"] = self.tool_calls

        return {
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": self.logprobs,
                    "finish_reason": self.finish_reason or self._final_finish_reason(),
                }
            ],
            "usage": self.usage(),
        }

    def usage(self) -> dict[str, int]:
        prompt_tokens = len(self.processed.prompt_token_ids)
        completion_tokens = self.generated_tokens
        if completion_tokens == 0 and self.backend_usage:
            backend_completion = self.backend_usage.get("completion_tokens")
            if isinstance(backend_completion, int):
                completion_tokens = backend_completion
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _update_metadata(self, delta: CompletionDelta) -> None:
        if delta.id:
            self.id = delta.id
        if delta.model:
            self.model = delta.model
        if delta.created:
            self.created = delta.created
        if delta.logprobs is not None:
            self.logprobs = delta.logprobs
        if delta.usage is not None:
            self.backend_usage = delta.usage

    def _chunk(
        self,
        delta: dict[str, Any],
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        if not self._sent_role:
            delta = {"role": "assistant", **delta}
            self._sent_role = True

        payload = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "logprobs": self.logprobs,
                    "finish_reason": finish_reason,
                }
            ],
        }
        if usage is not None:
            payload["usage"] = usage
        return payload

    def _stream_tool_call(self, tool_call: dict[str, Any], index: int) -> dict[str, Any]:
        stream_call = dict(tool_call)
        stream_call["index"] = index
        return stream_call

    def _final_finish_reason(self) -> str:
        if self.tool_calls:
            return "tool_calls"
        return self.finish_reason or "stop"

    def _include_stream_usage(self) -> bool:
        stream_options = self.original_body.get("stream_options")
        return isinstance(stream_options, dict) and bool(stream_options.get("include_usage"))

    def _strict_tool_jail(self, body: dict[str, Any]) -> bool:
        tool_choice = body.get("tool_choice")
        if tool_choice == "required":
            return True
        if isinstance(tool_choice, dict):
            return True
        return False


class ThinkTagsParser:
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.in_think = False
        self.buffer = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not self.enabled:
            return [("content", text)] if text else []
        self.buffer += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        if not self.enabled:
            return []
        return self._drain(final=True)

    def _drain(self, final: bool) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        while self.buffer:
            marker = self.CLOSE if self.in_think else self.OPEN
            index = self.buffer.find(marker)
            if index >= 0:
                self._emit(outputs, self.buffer[:index])
                self.buffer = self.buffer[index + len(marker) :]
                self.in_think = not self.in_think
                continue

            if final:
                self._emit(outputs, self.buffer)
                self.buffer = ""
                break

            keep = self._marker_prefix_len(self.buffer, marker)
            emit_text = self.buffer[: len(self.buffer) - keep]
            self.buffer = self.buffer[len(self.buffer) - keep :]
            self._emit(outputs, emit_text)
            break
        return outputs

    def _emit(self, outputs: list[tuple[str, str]], text: str) -> None:
        if not text:
            return
        outputs.append(("reasoning" if self.in_think else "content", text))

    def _marker_prefix_len(self, text: str, marker: str) -> int:
        max_len = min(len(marker) - 1, len(text))
        for size in range(max_len, 0, -1):
            if marker.startswith(text[-size:]):
                return size
        return 0


class HermesToolParser:
    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"

    def __init__(self, enabled: bool, strict: bool) -> None:
        self.enabled = enabled
        self.strict = strict
        self.buffer = ""
        self.in_tool = False
        self.tool_buffer = ""

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        if not text:
            return "", []
        if not self.enabled:
            return text, []
        if self.strict:
            self.buffer += text
            return "", []

        self.buffer += text
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        while self.buffer:
            if self.in_tool:
                close_index = self.buffer.find(self.CLOSE)
                if close_index < 0:
                    self.tool_buffer += self.buffer
                    self.buffer = ""
                    break
                self.tool_buffer += self.buffer[:close_index]
                self.buffer = self.buffer[close_index + len(self.CLOSE) :]
                parsed = self._parse_tool_calls(self.tool_buffer)
                if parsed:
                    tool_calls.extend(parsed)
                else:
                    content_parts.append(f"{self.OPEN}{self.tool_buffer}{self.CLOSE}")
                self.tool_buffer = ""
                self.in_tool = False
                continue

            open_index = self.buffer.find(self.OPEN)
            if open_index >= 0:
                content_parts.append(self.buffer[:open_index])
                self.buffer = self.buffer[open_index + len(self.OPEN) :]
                self.in_tool = True
                continue

            keep = self._marker_prefix_len(self.buffer, self.OPEN)
            content_parts.append(self.buffer[: len(self.buffer) - keep])
            self.buffer = self.buffer[len(self.buffer) - keep :]
            break

        return "".join(content_parts), tool_calls

    def finish(self) -> tuple[str, list[dict[str, Any]]]:
        if not self.enabled:
            return "", []
        if self.strict:
            text = self.buffer
            self.buffer = ""
            return self._parse_strict(text)
        if self.in_tool:
            text = f"{self.OPEN}{self.tool_buffer}{self.buffer}"
            self.in_tool = False
            self.tool_buffer = ""
            self.buffer = ""
            return text, []
        text = self.buffer
        self.buffer = ""
        return text, []

    def _parse_strict(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        stripped = text.strip()
        if not stripped:
            return "", []
        start = stripped.find(self.OPEN)
        end = stripped.find(self.CLOSE, start + len(self.OPEN))
        if start >= 0 and end >= 0:
            raw = stripped[start + len(self.OPEN) : end]
            parsed = self._parse_tool_calls(raw)
            if parsed:
                prefix = stripped[:start]
                suffix = stripped[end + len(self.CLOSE) :]
                return prefix + suffix, parsed
            return text, []

        parsed = self._parse_tool_calls(stripped)
        if parsed:
            return "", parsed
        return text, []

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(text)
        except Exception:
            return []

        raw_calls: Any
        if isinstance(payload, dict) and isinstance(payload.get("tool_calls"), list):
            raw_calls = payload["tool_calls"]
        elif isinstance(payload, list):
            raw_calls = payload
        else:
            raw_calls = [payload]

        calls: list[dict[str, Any]] = []
        for raw_call in raw_calls:
            call = self._normalize_tool_call(raw_call)
            if call:
                calls.append(call)
        return calls

    def _normalize_tool_call(self, raw_call: Any) -> dict[str, Any] | None:
        if not isinstance(raw_call, dict):
            return None

        function = raw_call.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            arguments = function.get("arguments", raw_call.get("arguments", {}))
        else:
            name = raw_call.get("name")
            arguments = raw_call.get("arguments", {})

        if not isinstance(name, str) or not name:
            return None
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))

        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{uuid.uuid4().hex}"
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }

    def _marker_prefix_len(self, text: str, marker: str) -> int:
        max_len = min(len(marker) - 1, len(text))
        for size in range(max_len, 0, -1):
            if marker.startswith(text[-size:]):
                return size
        return 0
