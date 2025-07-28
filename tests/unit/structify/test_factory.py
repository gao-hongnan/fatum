from __future__ import annotations

import instructor
import pytest

from fatum.structify.adapters.anthropic import AnthropicAdapter
from fatum.structify.adapters.gemini import GeminiAdapter
from fatum.structify.adapters.openai import OpenAIAdapter
from fatum.structify.config import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    InstructorConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)
from fatum.structify.factory import AdapterFactory


@pytest.mark.unit
class TestAdapterFactory:

    def test_create_openai_adapter(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.provider_config == provider_config
        assert adapter.completion_params == completion_params
        assert adapter.instructor_config == instructor_config

    def test_create_anthropic_adapter(self) -> None:
        provider_config = AnthropicProviderConfig(api_key="test_key")
        completion_params = AnthropicCompletionClientParams(model="claude-3-sonnet-20240229")
        instructor_config = InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert isinstance(adapter, AnthropicAdapter)
        assert adapter.provider_config == provider_config
        assert adapter.completion_params == completion_params
        assert adapter.instructor_config == instructor_config

    def test_create_gemini_adapter(self) -> None:
        provider_config = GeminiProviderConfig(api_key="test_key")
        completion_params = GeminiCompletionClientParams(model="gemini-pro")
        instructor_config = InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert isinstance(adapter, GeminiAdapter)
        assert adapter.provider_config == provider_config
        assert adapter.completion_params == completion_params
        assert adapter.instructor_config == instructor_config

    def test_factory_type_safety_openai(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert hasattr(adapter, "_create_client")
        assert hasattr(adapter, "_with_instructor")

        assert adapter.provider_config.provider == "openai"

    def test_factory_type_safety_anthropic(self) -> None:
        provider_config = AnthropicProviderConfig(api_key="test_key")
        completion_params = AnthropicCompletionClientParams(model="claude-3-sonnet")
        instructor_config = InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert hasattr(adapter, "_create_client")
        assert hasattr(adapter, "_with_instructor")
        assert adapter.provider_config.provider == "anthropic"

    def test_factory_type_safety_gemini(self) -> None:
        provider_config = GeminiProviderConfig(api_key="test_key")
        completion_params = GeminiCompletionClientParams(model="gemini-pro")
        instructor_config = InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert hasattr(adapter, "_create_client")
        assert hasattr(adapter, "_with_instructor")
        assert adapter.provider_config.provider == "gemini"

    def test_factory_preserves_config_attributes(self) -> None:
        provider_config = OpenAIProviderConfig.model_validate(
            {
                "api_key": "test_key",
                "organization": "test_org",
                "base_url": "https://custom.openai.com",
            }
        )
        completion_params = OpenAICompletionClientParams.model_validate(
            {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
            }
        )
        instructor_config = InstructorConfig.model_validate(
            {
                "mode": instructor.Mode.TOOLS,
                "max_retries": 3,
            }
        )

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert adapter.provider_config.api_key == "test_key"
        assert adapter.completion_params.model == "gpt-4"
        assert adapter.instructor_config.mode == instructor.Mode.TOOLS

        provider_data = adapter.provider_config.model_dump()
        assert provider_data["organization"] == "test_org"
        assert provider_data["base_url"] == "https://custom.openai.com"

        completion_data = adapter.completion_params.model_dump()
        assert completion_data["temperature"] == 0.7
        assert completion_data["max_tokens"] == 1000

        instructor_data = adapter.instructor_config.model_dump()
        assert instructor_data["max_retries"] == 3

    def test_factory_immutability(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        original_api_key = provider_config.api_key
        original_model = completion_params.model
        original_mode = instructor_config.mode

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert provider_config.api_key == original_api_key
        assert completion_params.model == original_model
        assert instructor_config.mode == original_mode

        assert adapter.provider_config is provider_config
        assert adapter.completion_params is completion_params
        assert adapter.instructor_config is instructor_config

    def test_factory_with_minimal_config(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert isinstance(adapter, OpenAIAdapter)
        assert adapter.provider_config.api_key == "test_key"
        assert adapter.completion_params.model == "gpt-4"
        assert adapter.instructor_config.mode == instructor.Mode.TOOLS

    def test_factory_with_maximal_config(self) -> None:
        provider_config = OpenAIProviderConfig.model_validate(
            {
                "api_key": "test_key",
                "organization": "test_org",
                "base_url": "https://custom.openai.com",
                "timeout": 30.0,
                "max_retries": 5,
            }
        )
        completion_params = OpenAICompletionClientParams.model_validate(
            {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "stream": False,
            }
        )
        instructor_config = InstructorConfig.model_validate(
            {
                "mode": instructor.Mode.TOOLS,
                "max_retries": 3,
                "validation_context": {"strict": True},
            }
        )

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert isinstance(adapter, OpenAIAdapter)

        provider_data = adapter.provider_config.model_dump()
        assert provider_data["organization"] == "test_org"
        assert provider_data["base_url"] == "https://custom.openai.com"
        assert provider_data["timeout"] == 30.0
        assert provider_data["max_retries"] == 5

        completion_data = adapter.completion_params.model_dump()
        assert completion_data["temperature"] == 0.7
        assert completion_data["max_tokens"] == 1000
        assert completion_data["top_p"] == 0.9
        assert completion_data["frequency_penalty"] == 0.1
        assert completion_data["presence_penalty"] == 0.1
        assert completion_data["stream"] is False

        instructor_data = adapter.instructor_config.model_dump()
        assert instructor_data["max_retries"] == 3
        assert instructor_data["validation_context"]["strict"] is True


@pytest.mark.unit
class TestFactoryErrorHandling:

    def test_factory_all_providers_have_cases(self) -> None:
        from fatum.structify.types import Provider

        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        openai_adapter = AdapterFactory.create(
            provider_config=OpenAIProviderConfig(api_key="test"),
            completion_params=OpenAICompletionClientParams(model="gpt-4"),
            instructor_config=instructor_config,
        )
        assert openai_adapter.provider_config.provider == Provider.OPENAI.value

        anthropic_adapter = AdapterFactory.create(
            provider_config=AnthropicProviderConfig(api_key="test"),
            completion_params=AnthropicCompletionClientParams(model="claude-3"),
            instructor_config=instructor_config,
        )
        assert anthropic_adapter.provider_config.provider == Provider.ANTHROPIC.value

        gemini_adapter = AdapterFactory.create(
            provider_config=GeminiProviderConfig(api_key="test"),
            completion_params=GeminiCompletionClientParams(model="gemini-pro"),
            instructor_config=instructor_config,
        )
        assert gemini_adapter.provider_config.provider == Provider.GEMINI.value

    def test_factory_lazy_initialization(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert adapter._client is None
        assert adapter._instructor is None

        assert adapter.provider_config is not None
        assert adapter.completion_params is not None
        assert adapter.instructor_config is not None

    def test_factory_multiple_instances_independence(self) -> None:
        provider_config1 = OpenAIProviderConfig(api_key="test_key_1")
        completion_params1 = OpenAICompletionClientParams(model="gpt-4")
        instructor_config1 = InstructorConfig(mode=instructor.Mode.TOOLS)

        provider_config2 = OpenAIProviderConfig(api_key="test_key_2")
        completion_params2 = OpenAICompletionClientParams(model="gpt-3.5-turbo")
        instructor_config2 = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter1 = AdapterFactory.create(
            provider_config=provider_config1,
            completion_params=completion_params1,
            instructor_config=instructor_config1,
        )

        adapter2 = AdapterFactory.create(
            provider_config=provider_config2,
            completion_params=completion_params2,
            instructor_config=instructor_config2,
        )

        assert adapter1 is not adapter2
        assert adapter1.provider_config is not adapter2.provider_config
        assert adapter1.completion_params is not adapter2.completion_params

        assert adapter1.provider_config.api_key == "test_key_1"
        assert adapter2.provider_config.api_key == "test_key_2"
        assert adapter1.completion_params.model == "gpt-4"
        assert adapter2.completion_params.model == "gpt-3.5-turbo"


@pytest.mark.unit
class TestFactoryTypeAnnotations:

    def test_factory_return_type_openai(self) -> None:
        provider_config = OpenAIProviderConfig(api_key="test_key")
        completion_params = OpenAICompletionClientParams(model="gpt-4")
        instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert type(adapter).__name__ == "OpenAIAdapter"
        assert hasattr(adapter, "_create_client")
        assert hasattr(adapter, "_with_instructor")

    def test_factory_return_type_anthropic(self) -> None:
        provider_config = AnthropicProviderConfig(api_key="test_key")
        completion_params = AnthropicCompletionClientParams(model="claude-3")
        instructor_config = InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert type(adapter).__name__ == "AnthropicAdapter"
        assert hasattr(adapter, "_create_client")
        assert hasattr(adapter, "_with_instructor")

    def test_factory_return_type_gemini(self) -> None:
        provider_config = GeminiProviderConfig(api_key="test_key")
        completion_params = GeminiCompletionClientParams(model="gemini-pro")
        instructor_config = InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)

        adapter = AdapterFactory.create(
            provider_config=provider_config,
            completion_params=completion_params,
            instructor_config=instructor_config,
        )

        assert type(adapter).__name__ == "GeminiAdapter"
        assert hasattr(adapter, "_create_client")
        assert hasattr(adapter, "_with_instructor")
