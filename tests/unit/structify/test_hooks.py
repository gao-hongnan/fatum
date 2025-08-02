from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from anthropic.types import Message as AnthropicResponse
from google.genai.types import GenerateContentResponse
from instructor.hooks import HookName
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from fatum.structify.hooks import (
    CompletionTrace,
    HookHandler,
    _setup_hooks,
    ahook_instructor,
)


class SampleModel(BaseModel):
    name: str
    value: int


@pytest.mark.unit
class TestCompletionTrace:
    def test_completion_trace_initialization(self) -> None:
        trace: CompletionTrace[Any] = CompletionTrace()

        assert trace.completion_kwargs == {}
        assert trace.messages == []
        assert trace.raw_response is None
        assert trace.parsed_result is None
        assert trace.error is None
        assert trace.last_attempt_error is None
        assert trace.parse_error is None

    def test_completion_trace_with_data(self) -> None:
        kwargs = {"model": "gpt-4", "temperature": 0.7}
        messages = [{"role": "user", "content": "test"}]
        parsed = SampleModel(name="test", value=42)
        error = Exception("test error")

        trace: CompletionTrace[Any] = CompletionTrace(
            completion_kwargs=kwargs,
            messages=messages,
            raw_response=None,
            parsed_result=parsed,
            error=error,
            last_attempt_error=error,
            parse_error=error,
        )

        assert trace.completion_kwargs == kwargs
        assert trace.messages == messages
        assert trace.raw_response is None
        assert trace.parsed_result == parsed
        assert trace.error == error
        assert trace.last_attempt_error == error
        assert trace.parse_error == error

    def test_completion_trace_generic_typing(self) -> None:
        openai_trace: CompletionTrace[ChatCompletion] = CompletionTrace()
        assert openai_trace.raw_response is None

        anthropic_trace: CompletionTrace[AnthropicResponse] = CompletionTrace()
        assert anthropic_trace.raw_response is None

        gemini_trace: CompletionTrace[GenerateContentResponse] = CompletionTrace()
        assert gemini_trace.raw_response is None

    def test_completion_trace_serialization(self) -> None:
        trace: CompletionTrace[Any] = CompletionTrace(
            completion_kwargs={"model": "gpt-4"},
            messages=[{"role": "user", "content": "test"}],
        )

        data = trace.model_dump()
        assert isinstance(data, dict)
        assert data["completion_kwargs"] == {"model": "gpt-4"}
        assert data["messages"] == [{"role": "user", "content": "test"}]

    def test_completion_trace_arbitrary_types(self) -> None:
        custom_object = SampleModel(name="test", value=42)

        trace: CompletionTrace[Any] = CompletionTrace(
            parsed_result=custom_object,
        )

        assert trace.parsed_result == custom_object
        assert isinstance(trace.parsed_result, SampleModel)
        assert trace.parsed_result.name == "test"
        assert trace.parsed_result.value == 42


@pytest.mark.unit
class TestSetupHooks:
    def testsetup_hooks_returns_trace_and_hooks(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()

        trace: CompletionTrace[Any]
        hooks: list[tuple[HookName, HookHandler]]
        trace, hooks = _setup_hooks(mock_client)

        assert isinstance(trace, CompletionTrace)
        assert isinstance(hooks, list)
        assert len(hooks) == 5

        hook_names = [hook[0] for hook in hooks]
        expected_hooks = [
            HookName.COMPLETION_KWARGS,
            HookName.COMPLETION_RESPONSE,
            HookName.COMPLETION_ERROR,
            HookName.COMPLETION_LAST_ATTEMPT,
            HookName.PARSE_ERROR,
        ]
        assert hook_names == expected_hooks

    def testsetup_hooks_registers_handlers(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()

        trace: CompletionTrace[Any]
        hooks: list[tuple[HookName, HookHandler]]
        trace, hooks = _setup_hooks(mock_client)

        assert mock_client.on.call_count == 5

        calls = mock_client.on.call_args_list
        expected_hooks = [
            HookName.COMPLETION_KWARGS,
            HookName.COMPLETION_RESPONSE,
            HookName.COMPLETION_ERROR,
            HookName.COMPLETION_LAST_ATTEMPT,
            HookName.PARSE_ERROR,
        ]

        for i, expected_hook in enumerate(expected_hooks):
            assert calls[i][0][0] == expected_hook
            assert callable(calls[i][0][1])

    def test_hook_handlers_capture_data(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()

        trace: CompletionTrace[Any]
        hooks: list[tuple[HookName, HookHandler]]
        trace, hooks = _setup_hooks(mock_client)

        handlers = {hook[0]: hook[1] for hook in hooks}

        test_kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}
        handlers[HookName.COMPLETION_KWARGS](**test_kwargs)
        assert trace.completion_kwargs == test_kwargs
        assert trace.messages == test_kwargs["messages"]

        test_response: Any = {"id": "test_response"}
        handlers[HookName.COMPLETION_RESPONSE](test_response)
        assert trace.raw_response == test_response

        test_error = Exception("test error")
        handlers[HookName.COMPLETION_ERROR](test_error)
        assert trace.error == test_error

        test_last_error = Exception("last attempt error")
        handlers[HookName.COMPLETION_LAST_ATTEMPT](test_last_error)
        assert trace.last_attempt_error == test_last_error

        test_parse_error = Exception("parse error")
        handlers[HookName.PARSE_ERROR](test_parse_error)
        assert trace.parse_error == test_parse_error

    def test_hook_handlers_safe_with_missing_data(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()

        trace: CompletionTrace[Any]
        hooks: list[tuple[HookName, HookHandler]]
        trace, hooks = _setup_hooks(mock_client)
        handlers = {hook[0]: hook[1] for hook in hooks}

        handlers[HookName.COMPLETION_KWARGS](model="gpt-4")
        assert trace.completion_kwargs == {"model": "gpt-4"}
        assert trace.messages == []

        handlers[HookName.COMPLETION_RESPONSE](None)
        assert trace.raw_response is None

        handlers[HookName.COMPLETION_ERROR](None)
        assert trace.error is None


@pytest.mark.unit
@pytest.mark.asyncio
class TestAhookInstructor:
    async def test_ahook_instructor_disabled(self) -> None:
        mock_client = MagicMock()

        trace: CompletionTrace[Any]
        async with ahook_instructor(mock_client, enable=False) as trace:
            assert isinstance(trace, CompletionTrace)

            assert trace.completion_kwargs == {}
            assert trace.messages == []
            assert trace.raw_response is None

        mock_client.on.assert_not_called()

    async def test_ahook_instructor_enabled(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()
        mock_client.off = MagicMock()

        trace: CompletionTrace[Any]
        async with ahook_instructor(mock_client, enable=True) as trace:
            assert isinstance(trace, CompletionTrace)

            assert mock_client.on.call_count == 5

        assert mock_client.off.call_count == 5

    async def test_ahook_instructor_default_enabled(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()
        mock_client.off = MagicMock()

        trace: CompletionTrace[Any]
        async with ahook_instructor(mock_client) as trace:
            assert isinstance(trace, CompletionTrace)
            assert mock_client.on.call_count == 5

        assert mock_client.off.call_count == 5

    async def test_ahook_instructor_cleanup_on_exception(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()
        mock_client.off = MagicMock()

        with pytest.raises(ValueError, match="test error"):
            trace: CompletionTrace[Any]
            async with ahook_instructor(mock_client, enable=True) as trace:
                assert isinstance(trace, CompletionTrace)
                assert mock_client.on.call_count == 5
                raise ValueError("test error")

        assert mock_client.off.call_count == 5

    async def test_ahook_instructor_hook_deregistration(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()
        mock_client.off = MagicMock()

        _trace: CompletionTrace[Any]
        async with ahook_instructor(mock_client, enable=True) as _trace:
            registered_calls = mock_client.on.call_args_list

        deregistered_calls = mock_client.off.call_args_list

        assert len(registered_calls) == len(deregistered_calls)

        for reg_call, dereg_call in zip(registered_calls, deregistered_calls, strict=False):
            assert reg_call[0][0] == dereg_call[0][0]
            assert reg_call[0][1] == dereg_call[0][1]

    async def test_ahook_instructor_captures_during_context(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()
        mock_client.off = MagicMock()

        trace: CompletionTrace[Any]
        async with ahook_instructor(mock_client, enable=True) as trace:
            registered_calls = mock_client.on.call_args_list
            handlers = {call[0][0]: call[0][1] for call in registered_calls}

            test_kwargs = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}]}
            handlers[HookName.COMPLETION_KWARGS](**test_kwargs)

            test_response = {"id": "response_123"}
            handlers[HookName.COMPLETION_RESPONSE](test_response)

            assert trace.completion_kwargs == test_kwargs
            assert trace.raw_response == test_response

    async def test_ahook_instructor_multiple_contexts(self) -> None:
        mock_client1 = MagicMock()
        mock_client1.on = MagicMock()
        mock_client1.off = MagicMock()

        mock_client2 = MagicMock()
        mock_client2.on = MagicMock()
        mock_client2.off = MagicMock()

        trace1: CompletionTrace[Any]
        trace2: CompletionTrace[Any]
        async with (
            ahook_instructor(mock_client1, enable=True) as trace1,
            ahook_instructor(mock_client2, enable=True) as trace2,
        ):
            assert trace1 is not trace2
            assert isinstance(trace1, CompletionTrace)
            assert isinstance(trace2, CompletionTrace)

            assert mock_client1.on.call_count == 5
            assert mock_client2.on.call_count == 5

        assert mock_client1.off.call_count == 5
        assert mock_client2.off.call_count == 5


@pytest.mark.unit
class TestHookIntegration:
    def test_hook_system_end_to_end(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()
        mock_client.off = MagicMock()

        trace: CompletionTrace[Any]
        hooks: list[tuple[HookName, HookHandler]]
        trace, hooks = _setup_hooks(mock_client)

        assert trace.completion_kwargs == {}
        assert trace.messages == []
        assert trace.raw_response is None
        assert trace.error is None

        handlers = {hook[0]: hook[1] for hook in hooks}

        kwargs = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
        }
        handlers[HookName.COMPLETION_KWARGS](**kwargs)

        response = {"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello!"}}]}
        handlers[HookName.COMPLETION_RESPONSE](response)

        assert trace.completion_kwargs == kwargs
        assert trace.messages == kwargs["messages"]
        assert trace.raw_response == response
        assert trace.error is None
        assert trace.parse_error is None

    def test_hook_system_error_scenarios(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()

        trace: CompletionTrace[Any]
        hooks: list[tuple[HookName, HookHandler]]
        trace, hooks = _setup_hooks(mock_client)
        handlers = {hook[0]: hook[1] for hook in hooks}

        completion_error = Exception("API rate limit exceeded")
        handlers[HookName.COMPLETION_ERROR](completion_error)
        assert trace.error == completion_error

        last_attempt_error = Exception("Final attempt failed")
        handlers[HookName.COMPLETION_LAST_ATTEMPT](last_attempt_error)
        assert trace.last_attempt_error == last_attempt_error

        parse_error = Exception("Invalid JSON in response")
        handlers[HookName.PARSE_ERROR](parse_error)
        assert trace.parse_error == parse_error

        assert trace.error == completion_error
        assert trace.last_attempt_error == last_attempt_error
        assert trace.parse_error == parse_error

    def test_hook_system_edge_cases(self) -> None:
        mock_client = MagicMock()
        mock_client.on = MagicMock()

        trace: CompletionTrace[Any]
        hooks: list[tuple[HookName, HookHandler]]
        trace, hooks = _setup_hooks(mock_client)
        handlers = {hook[0]: hook[1] for hook in hooks}

        handlers[HookName.COMPLETION_KWARGS]()
        assert trace.completion_kwargs == {}
        assert trace.messages == []

        handlers[HookName.COMPLETION_RESPONSE](None)
        assert trace.raw_response is None

        handlers[HookName.COMPLETION_KWARGS](model="gpt-3.5")
        handlers[HookName.COMPLETION_KWARGS](model="gpt-4")
        assert trace.completion_kwargs == {"model": "gpt-4"}


@pytest.mark.unit
class TestHookTypesSafety:
    def test_completion_trace_type_parameters(self) -> None:
        openai_trace: CompletionTrace[ChatCompletion] = CompletionTrace()
        anthropic_trace: CompletionTrace[AnthropicResponse] = CompletionTrace()
        gemini_trace: CompletionTrace[GenerateContentResponse] = CompletionTrace()

        assert isinstance(openai_trace, CompletionTrace)
        assert isinstance(anthropic_trace, CompletionTrace)
        assert isinstance(gemini_trace, CompletionTrace)

    def test_hook_names_type_safety(self) -> None:
        valid_names: list[HookName] = [
            HookName.COMPLETION_KWARGS,
            HookName.COMPLETION_RESPONSE,
            HookName.COMPLETION_ERROR,
            HookName.COMPLETION_LAST_ATTEMPT,
            HookName.PARSE_ERROR,
        ]

        for name in valid_names:
            assert isinstance(name, HookName)
            assert isinstance(name.value, str)

    def test_message_param_type_alias(self) -> None:
        valid_message: dict[str, Any] = {"role": "user", "content": "test"}
        assert isinstance(valid_message, dict)
        assert "role" in valid_message
        assert "content" in valid_message
