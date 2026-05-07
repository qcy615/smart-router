import pytest

from smart_router.openai.processing import (
    CompletionDelta,
    OpenAIChatProcessor,
    OpenAIProcessingConfig,
    OpenAIProcessingError,
)


class FakeChatTokenizer:
    def __init__(self, token_ids=None):
        self.token_ids = token_ids or [10, 20, 30]
        self.chat_template_calls = []

    def apply_chat_template(self, messages, *args, **kwargs):
        self.chat_template_calls.append(
            {"messages": messages, "args": args, "kwargs": kwargs}
        )
        return list(self.token_ids)

    def encode(self, text: str, add_special_tokens: bool = False):
        return list(self.token_ids)


def _processor(tokenizer=None):
    return OpenAIChatProcessor(
        OpenAIProcessingConfig(enabled=True, tokenizer_path="/models/demo"),
        tokenizer=tokenizer or FakeChatTokenizer(),
    )


def test_preprocess_chat_renders_tokens_and_builds_completion_body():
    tokenizer = FakeChatTokenizer([101, 202])
    processor = _processor(tokenizer)
    body = {
        "model": "demo-model",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [{"type": "function", "function": {"name": "search"}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
        "max_completion_tokens": 16,
        "logprobs": True,
        "top_logprobs": 4,
        "temperature": 0.7,
    }

    processed = processor.preprocess_chat(body)

    assert processed.prompt_token_ids == [101, 202]
    assert processed.completion_body["prompt"] == [101, 202]
    assert processed.completion_body["return_token_ids"] is True
    assert processed.completion_body["max_tokens"] == 16
    assert processed.completion_body["logprobs"] == 4
    assert processed.completion_body["temperature"] == 0.7
    assert "messages" not in processed.completion_body
    assert "tools" not in processed.completion_body
    assert "tool_choice" not in processed.completion_body
    assert "response_format" not in processed.completion_body

    call = tokenizer.chat_template_calls[0]
    assert call["kwargs"]["tools"] == body["tools"]
    assert call["kwargs"]["tool_choice"] == "auto"
    assert call["kwargs"]["response_format"] == {"type": "json_object"}
    assert call["kwargs"]["enable_thinking"] is False


def test_preprocess_rejects_multimodal_and_multiple_choices():
    processor = _processor()

    with pytest.raises(OpenAIProcessingError):
        processor.preprocess_chat(
            {
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ],
            }
        )

    with pytest.raises(OpenAIProcessingError):
        processor.preprocess_chat(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": "hello"}],
                "n": 2,
            }
        )


def test_chat_builder_splits_think_tags_into_reasoning_content():
    processor = _processor()
    processed = processor.preprocess_chat(
        {"model": "demo", "messages": [{"role": "user", "content": "hello"}]}
    )
    builder = processor.create_chat_builder(
        processed,
        {"model": "demo", "messages": [{"role": "user", "content": "hello"}]},
    )

    builder.apply_delta(
        CompletionDelta(text="<think>plan</think>answer", token_ids=[1, 2])
    )
    response = builder.to_response()

    message = response["choices"][0]["message"]
    assert message["reasoning_content"] == "plan"
    assert message["content"] == "answer"
    assert response["usage"]["completion_tokens"] == 2


def test_chat_builder_parses_hermes_tool_call_and_sets_finish_reason():
    processor = _processor()
    processed = processor.preprocess_chat(
        {"model": "demo", "messages": [{"role": "user", "content": "hello"}]}
    )
    builder = processor.create_chat_builder(
        processed,
        {"model": "demo", "messages": [{"role": "user", "content": "hello"}]},
    )

    builder.apply_delta(
        CompletionDelta(
            text='<tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>',
            token_ids=[1],
            finish_reason="stop",
        )
    )
    response = builder.to_response()

    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tool_call = choice["message"]["tool_calls"][0]
    assert tool_call["type"] == "function"
    assert tool_call["function"] == {"name": "search", "arguments": '{"q":"x"}'}
    assert choice["message"]["content"] is None


def test_chat_builder_releases_failed_tool_parse_as_content():
    processor = _processor()
    processed = processor.preprocess_chat(
        {"model": "demo", "messages": [{"role": "user", "content": "hello"}]}
    )
    builder = processor.create_chat_builder(
        processed,
        {"model": "demo", "messages": [{"role": "user", "content": "hello"}]},
    )

    builder.apply_delta(CompletionDelta(text="<tool_call>{bad}</tool_call>"))
    response = builder.to_response()

    assert response["choices"][0]["message"]["content"] == "<tool_call>{bad}</tool_call>"
    assert response["choices"][0]["finish_reason"] == "stop"
