from __future__ import annotations

from typing import Any, AsyncIterator, cast
from unittest.mock import AsyncMock, MagicMock, patch

import instructor
import pytest
from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from fatum.structify.adapters.anthropic import AnthropicAdapter
from fatum.structify.adapters.base import BaseAdapter
from fatum.structify.adapters.gemini import GeminiAdapter
from fatum.structify.adapters.openai import OpenAIAdapter
from fatum.structify.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    CompletionResult,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    InstructorConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)


class SampleModel(BaseModel):
    name: str
    value: int


@pytest.mark.unit
class TestBaseAdapter:
    def test_base_adapter_initialization(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert adapter.provider_config == provider_config
        assert adapter.completion_params == completion_params
        assert adapter.instructor_config == instructor_config
        assert hasattr(adapter, "_client") and getattr(adapter, "_client", None) is None
        assert hasattr(adapter, "_instructor") and getattr(adapter, "_instructor", None) is None

    def test_lazy_client_initialization(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert hasattr(adapter, "_client") and getattr(adapter, "_client", None) is None

        with patch.object(adapter, "_create_client", return_value=MagicMock()) as mock_create:
            client = adapter.client
            mock_create.assert_called_once()
            assert hasattr(adapter, "_client") and getattr(adapter, "_client", None) is not None

            client2 = adapter.client
            assert client is client2
            mock_create.assert_called_once()

    def test_lazy_instructor_initialization(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert hasattr(adapter, "_instructor") and getattr(adapter, "_instructor", None) is None

        with patch.object(adapter, "_with_instructor", return_value=MagicMock()) as mock_create:
            instructor_client = adapter.instructor
            mock_create.assert_called_once()
            assert hasattr(adapter, "_instructor") and getattr(adapter, "_instructor", None) is not None

            instructor_client2 = adapter.instructor
            assert instructor_client is instructor_client2
            mock_create.assert_called_once()


@pytest.mark.unit
class TestOpenAIAdapter:
    def test_create_client(self) -> None:
        provider_config = OpenAIProviderConfig(
            api_key="test_key",
        )
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        with patch("fatum.structify.adapters.openai.AsyncOpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = adapter.client

            mock_openai.assert_called_once_with(
                api_key="test_key",
            )
            assert client == mock_client

    def test_with_instructor(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        mock_client = MagicMock(spec=AsyncOpenAI)
        mock_instructor = MagicMock()

        with (
            patch.object(adapter, "_client", mock_client),
            patch(
                "fatum.structify.adapters.openai.instructor.from_openai", return_value=mock_instructor
            ) as mock_from_openai,
        ):
            instructor_client = adapter.instructor

            mock_from_openai.assert_called_once_with(mock_client, mode=instructor.Mode.TOOLS)
            assert instructor_client == mock_instructor

    @pytest.mark.asyncio
    async def test_acreate_without_hooks(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        mock_instructor = AsyncMock()
        mock_result = SampleModel(name="test", value=42)
        mock_instructor.create.return_value = mock_result

        messages = cast(list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}])

        with (
            patch.object(adapter, "_instructor", mock_instructor),
            patch("fatum.structify.adapters.base.ahook_instructor") as mock_hook,
        ):
            mock_hook.return_value = AsyncMock()
            mock_hook.return_value.__aenter__.return_value = MagicMock()
            mock_hook.return_value.__aexit__.return_value = None

            result = await adapter.acreate(
                messages=messages,
                response_model=SampleModel,
                with_hooks=False,
            )

            assert result == mock_result
            mock_instructor.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_acreate_with_hooks(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        mock_instructor = AsyncMock()
        mock_result = SampleModel(name="test", value=42)
        mock_instructor.create.return_value = mock_result

        messages = cast(list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}])

        with (
            patch.object(adapter, "_instructor", mock_instructor),
            patch("fatum.structify.adapters.base.ahook_instructor") as mock_hook,
        ):
            from fatum.structify.hooks import CompletionTrace

            mock_trace: CompletionTrace[Any] = CompletionTrace()
            mock_hook.return_value.__aenter__.return_value = mock_trace
            mock_hook.return_value.__aexit__.return_value = None

            result = await adapter.acreate(
                messages=messages,
                response_model=SampleModel,
                with_hooks=True,
            )

            assert isinstance(result, CompletionResult)
            assert result.data == mock_result
            assert result.trace == mock_trace

    @pytest.mark.asyncio
    async def test_astream(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        mock_instructor = AsyncMock()
        partial_results = [
            SampleModel(name="partial1", value=1),
            SampleModel(name="partial2", value=2),
            SampleModel(name="final", value=3),
        ]

        async def mock_create_partial(*_args: Any, **_kwargs: Any) -> AsyncIterator[SampleModel]:
            for result in partial_results:
                yield result

        mock_instructor.create_partial = mock_create_partial

        messages = cast(list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}])

        with patch.object(adapter, "_instructor", mock_instructor):
            results = [
                partial
                async for partial in adapter.astream(
                    messages=messages,
                    response_model=SampleModel,
                )
            ]

            assert len(results) == 3
            assert results == partial_results


@pytest.mark.unit
class TestAnthropicAdapter:
    def test_create_client(self) -> None:
        provider_config = AnthropicProviderConfig(
            api_key="test_key",
        )
        completion_params = AnthropicCompletionClientParams(model="claude-3-sonnet")
        instructor_config = InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS)

        adapter = AnthropicAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        with patch("fatum.structify.adapters.anthropic.AsyncAnthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.return_value = mock_client

            client = adapter.client

            mock_anthropic.assert_called_once_with(
                api_key="test_key",
            )
            assert client == mock_client

    def test_with_instructor(self) -> None:
        provider_config = AnthropicProviderConfig(api_key="test_key")
        completion_params = AnthropicCompletionClientParams(model="claude-3-sonnet")
        instructor_config = InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS)

        adapter = AnthropicAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        mock_client = MagicMock(spec=AsyncAnthropic)
        mock_instructor = MagicMock()

        with (
            patch.object(adapter, "_client", mock_client),
            patch(
                "fatum.structify.adapters.anthropic.instructor.from_anthropic", return_value=mock_instructor
            ) as mock_from_anthropic,
        ):
            instructor_client = adapter.instructor

            mock_from_anthropic.assert_called_once_with(mock_client, mode=instructor.Mode.ANTHROPIC_TOOLS)
            assert instructor_client == mock_instructor


@pytest.mark.unit
class TestGeminiAdapter:
    def test_create_client(self) -> None:
        provider_config = GeminiProviderConfig(api_key="test_key")
        completion_params = GeminiCompletionClientParams(model="gemini-pro")
        instructor_config = InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)

        adapter = GeminiAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        with patch("fatum.structify.adapters.gemini.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client = adapter.client

            mock_client_class.assert_called_once_with(api_key="test_key")
            assert client == mock_client

    def test_with_instructor(self) -> None:
        provider_config = GeminiProviderConfig(api_key="test_key")
        completion_params = GeminiCompletionClientParams(model="gemini-pro")
        instructor_config = InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)

        adapter = GeminiAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        mock_client = MagicMock(spec=genai.Client)
        mock_instructor = MagicMock(spec=instructor.AsyncInstructor)

        with (
            patch.object(adapter, "_client", mock_client),
            patch(
                "fatum.structify.adapters.gemini.instructor.from_genai", return_value=mock_instructor
            ) as mock_from_genai,
        ):
            instructor_client = adapter.instructor

            mock_from_genai.assert_called_once_with(
                mock_client, use_async=True, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
            )
            assert instructor_client == mock_instructor

    @pytest.mark.asyncio
    async def test_acreate_gemini_specific(self) -> None:
        provider_config = GeminiProviderConfig(api_key="test_key")
        completion_params = GeminiCompletionClientParams(
            model="gemini-pro",
        )
        instructor_config = InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)

        adapter = GeminiAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        mock_instructor = AsyncMock()
        mock_result = SampleModel(name="test", value=42)
        mock_instructor.create.return_value = mock_result

        messages = cast(list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}])

        with (
            patch.object(adapter, "_instructor", mock_instructor),
            patch("fatum.structify.adapters.gemini.GenerateContentConfig") as mock_config,
        ):
            mock_config_instance = MagicMock()
            mock_config.return_value = mock_config_instance

            result = await adapter.acreate(
                messages=messages,
                response_model=SampleModel,
                with_hooks=False,
            )

            assert result == mock_result

            mock_config.assert_called_once()

            mock_instructor.create.assert_called_once_with(
                model="gemini-pro",
                response_model=SampleModel,
                messages=messages,
                config=mock_config_instance,
            )


@pytest.mark.unit
class TestAdapterErrorHandling:
    @pytest.mark.asyncio
    async def test_adapter_handles_instructor_errors(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        mock_instructor = AsyncMock()
        mock_instructor.create.side_effect = Exception("API Error")

        messages = cast(list[ChatCompletionMessageParam], [{"role": "user", "content": "test"}])

        with patch.object(adapter, "_instructor", mock_instructor), pytest.raises(Exception, match="API Error"):
            await adapter.acreate(
                messages=messages,
                response_model=SampleModel,
            )

    def test_adapter_parameter_validation(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert adapter.provider_config == provider_config
        assert adapter.completion_params == completion_params
        assert adapter.instructor_config == instructor_config

    def test_adapter_client_creation_failure(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="invalid_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = OpenAIAdapter(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        with (
            patch("fatum.structify.adapters.openai.AsyncOpenAI", side_effect=Exception("Auth Error")),
            pytest.raises(Exception, match="Auth Error"),
        ):
            _ = adapter.client


@pytest.mark.unit
class TestAdapterTypeConsistency:
    def test_all_adapters_inherit_from_base(self) -> None:
        assert issubclass(OpenAIAdapter, BaseAdapter)
        assert issubclass(AnthropicAdapter, BaseAdapter)
        assert issubclass(GeminiAdapter, BaseAdapter)

    def test_adapter_generic_parameters(self) -> None:
        assert issubclass(OpenAIAdapter, BaseAdapter)
        assert issubclass(AnthropicAdapter, BaseAdapter)
        assert issubclass(GeminiAdapter, BaseAdapter)

    def test_adapter_method_signatures(self) -> None:
        for adapter_class in [OpenAIAdapter, AnthropicAdapter, GeminiAdapter]:
            assert hasattr(adapter_class, "_create_client")
            assert hasattr(adapter_class, "_with_instructor")
            assert hasattr(adapter_class, "acreate")
            assert hasattr(adapter_class, "astream")

            assert hasattr(adapter_class, "_create_client")
            assert hasattr(adapter_class, "_with_instructor")
            assert callable(adapter_class.acreate)
            assert callable(adapter_class.astream)
