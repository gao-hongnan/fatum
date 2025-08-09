from __future__ import annotations

from typing import Any

import instructor
import pytest
from pydantic import BaseModel, ValidationError

from fatum.structify.enums import Capability, Provider
from fatum.structify.hooks import CompletionTrace
from fatum.structify.models import (
    Allowable,
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    CompletionClientParams,
    CompletionResult,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    InstructorConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
    ProviderConfig,
)


@pytest.mark.unit
class TestAllowable:
    def test_allowable_extra_allow(self) -> None:
        class TestModel(Allowable):
            name: str

        instance = TestModel(name="test", extra_field="extra_value")  # type: ignore[call-arg]
        assert instance.name == "test"
        assert hasattr(instance, "extra_field")
        assert instance.extra_field == "extra_value"  # type: ignore[attr-defined]

    def test_allowable_serialization(self) -> None:
        class TestModel(Allowable):
            name: str

        instance = TestModel(name="test", extra_field="extra")  # type: ignore[call-arg]
        data = instance.model_dump()
        assert data["name"] == "test"
        assert data["extra_field"] == "extra"


@pytest.mark.unit
class TestProviderConfigs:
    def test_openai_provider_config(self) -> None:
        config = OpenAIProviderConfig(api_key="test_key")

        assert config.provider == Provider.OPENAI.value
        assert config.api_key == "test_key"

    def test_anthropic_provider_config(self) -> None:
        config = AnthropicProviderConfig(api_key="test_key")

        assert config.provider == Provider.ANTHROPIC.value
        assert config.api_key == "test_key"

    def test_gemini_provider_config(self) -> None:
        config = GeminiProviderConfig(api_key="test_key")

        assert config.provider == Provider.GEMINI.value
        assert config.api_key == "test_key"

    def test_provider_config_extra_fields(self) -> None:
        config = OpenAIProviderConfig(
            api_key="test_key",
            organization="test_org",  # type: ignore[call-arg]
            base_url="https://custom.openai.com",  # type: ignore[call-arg]
        )

        assert config.api_key == "test_key"
        data = config.model_dump()
        assert data["organization"] == "test_org"
        assert data["base_url"] == "https://custom.openai.com"

    def test_provider_config_exclude_provider(self) -> None:
        config = OpenAIProviderConfig(api_key="test_key")
        data = config.model_dump()

        assert "provider" not in data
        assert "api_key" in data

    def test_provider_config_validation(self) -> None:
        with pytest.raises(ValidationError):
            OpenAIProviderConfig()  # type: ignore[call-arg]


@pytest.mark.unit
class TestClientParams:
    def test_openai_completion_client_params(self) -> None:
        params = OpenAICompletionClientParams(model="gpt-4")

        assert params.provider == Provider.OPENAI.value
        assert params.capability == Capability.COMPLETION
        assert params.model == "gpt-4"

    def test_anthropic_completion_client_params(self) -> None:
        params = AnthropicCompletionClientParams(model="claude-3-sonnet-20240229")

        assert params.provider == Provider.ANTHROPIC.value
        assert params.capability == Capability.COMPLETION
        assert params.model == "claude-3-sonnet-20240229"

    def test_gemini_completion_client_params(self) -> None:
        params = GeminiCompletionClientParams(model="gemini-pro")

        assert params.provider == Provider.GEMINI.value
        assert params.capability == Capability.COMPLETION
        assert params.model == "gemini-pro"

    def test_client_params_extra_fields(self) -> None:
        params = OpenAICompletionClientParams(
            model="gpt-4",
            temperature=0.7,  # type: ignore[call-arg]
            max_tokens=1000,  # type: ignore[call-arg]
            top_p=0.9,  # type: ignore[call-arg]
        )

        assert params.model == "gpt-4"
        data = params.model_dump()
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 1000
        assert data["top_p"] == 0.9

    def test_client_params_exclude_fields(self) -> None:
        params = OpenAICompletionClientParams(model="gpt-4")
        data = params.model_dump()

        assert "provider" not in data
        assert "capability" not in data
        assert "model" in data

    def test_client_params_validation(self) -> None:
        with pytest.raises(ValidationError):
            OpenAICompletionClientParams()  # type: ignore[call-arg]


@pytest.mark.unit
class TestDiscriminatedUnions:
    def test_provider_config_union_openai(self) -> None:
        config_data = {"api_key": "test_key"}

        config = OpenAIProviderConfig(api_key=config_data["api_key"])
        assert isinstance(config, OpenAIProviderConfig)
        assert config.provider == Provider.OPENAI.value

    def test_provider_config_union_anthropic(self) -> None:
        config_data = {"api_key": "test_key"}

        config = AnthropicProviderConfig(api_key=config_data["api_key"])
        assert isinstance(config, AnthropicProviderConfig)
        assert config.provider == Provider.ANTHROPIC.value

    def test_provider_config_union_gemini(self) -> None:
        config_data = {"api_key": "test_key"}

        config = GeminiProviderConfig(api_key=config_data["api_key"])
        assert isinstance(config, GeminiProviderConfig)
        assert config.provider == Provider.GEMINI.value

    def test_completion_client_params_union_openai(self) -> None:
        params_data = {"model": "gpt-4"}

        params = OpenAICompletionClientParams(model=params_data["model"])
        assert isinstance(params, OpenAICompletionClientParams)
        assert params.provider == Provider.OPENAI.value

    def test_completion_client_params_union_anthropic(self) -> None:
        params_data = {"model": "claude-3-sonnet"}

        params = AnthropicCompletionClientParams(model=params_data["model"])
        assert isinstance(params, AnthropicCompletionClientParams)
        assert params.provider == Provider.ANTHROPIC.value

    def test_completion_client_params_union_gemini(self) -> None:
        params_data = {"model": "gemini-pro"}

        params = GeminiCompletionClientParams(model=params_data["model"])
        assert isinstance(params, GeminiCompletionClientParams)
        assert params.provider == Provider.GEMINI.value

    def test_provider_config_discriminated_union_parsing(self) -> None:
        from pydantic import TypeAdapter

        provider_adapter: TypeAdapter[ProviderConfig] = TypeAdapter(ProviderConfig)

        openai_data = {"provider": "openai", "api_key": "test-key"}
        parsed_openai = provider_adapter.validate_python(openai_data)
        assert isinstance(parsed_openai, OpenAIProviderConfig)
        assert parsed_openai.provider == "openai"
        assert parsed_openai.api_key == "test-key"

        anthropic_data = {"provider": "anthropic", "api_key": "test-key"}
        parsed_anthropic = provider_adapter.validate_python(anthropic_data)
        assert isinstance(parsed_anthropic, AnthropicProviderConfig)
        assert parsed_anthropic.provider == "anthropic"
        assert parsed_anthropic.api_key == "test-key"

        gemini_data = {"provider": "gemini", "api_key": "test-key"}
        parsed_gemini = provider_adapter.validate_python(gemini_data)
        assert isinstance(parsed_gemini, GeminiProviderConfig)
        assert parsed_gemini.provider == "gemini"
        assert parsed_gemini.api_key == "test-key"

    def test_completion_params_discriminated_union_parsing(self) -> None:
        from pydantic import TypeAdapter

        params_adapter: TypeAdapter[CompletionClientParams] = TypeAdapter(CompletionClientParams)

        openai_params_data = {"provider": "openai", "model": "gpt-4"}
        parsed_openai_params = params_adapter.validate_python(openai_params_data)
        assert isinstance(parsed_openai_params, OpenAICompletionClientParams)
        assert parsed_openai_params.provider == "openai"
        assert parsed_openai_params.model == "gpt-4"

        anthropic_params_data = {"provider": "anthropic", "model": "claude-3-opus"}
        parsed_anthropic_params = params_adapter.validate_python(anthropic_params_data)
        assert isinstance(parsed_anthropic_params, AnthropicCompletionClientParams)
        assert parsed_anthropic_params.provider == "anthropic"
        assert parsed_anthropic_params.model == "claude-3-opus"

        gemini_params_data = {"provider": "gemini", "model": "gemini-pro"}
        parsed_gemini_params = params_adapter.validate_python(gemini_params_data)
        assert isinstance(parsed_gemini_params, GeminiCompletionClientParams)
        assert parsed_gemini_params.provider == "gemini"
        assert parsed_gemini_params.model == "gemini-pro"

    def test_discriminated_union_invalid_provider(self) -> None:
        from pydantic import TypeAdapter

        provider_adapter: TypeAdapter[ProviderConfig] = TypeAdapter(ProviderConfig)

        invalid_data = {"provider": "invalid-provider", "api_key": "test-key"}
        with pytest.raises(ValidationError) as exc_info:
            provider_adapter.validate_python(invalid_data)

        errors = exc_info.value.errors()
        assert any("discriminator" in str(error) for error in errors)


@pytest.mark.unit
class TestInstructorConfig:
    def test_instructor_config_creation(self) -> None:
        modes = [
            instructor.Mode.TOOLS,
            instructor.Mode.ANTHROPIC_TOOLS,
            instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
        ]

        for mode in modes:
            config = InstructorConfig(mode=mode)
            assert config.mode == mode

    def test_instructor_config_extra_fields(self) -> None:
        config = InstructorConfig(
            mode=instructor.Mode.TOOLS,
            max_retries=3,  # type: ignore[call-arg]
            validation_context={"test": True},  # type: ignore[call-arg]
        )

        assert config.mode == instructor.Mode.TOOLS
        data = config.model_dump()
        assert data["max_retries"] == 3
        assert data["validation_context"]["test"] is True

    def test_instructor_config_validation(self) -> None:
        with pytest.raises(ValidationError):
            InstructorConfig()  # type: ignore[call-arg]


@pytest.mark.unit
class TestCompletionResult:
    def test_completion_result_creation(self) -> None:
        class TestModel(BaseModel):
            name: str
            value: int

        test_data = TestModel(name="test", value=42)
        test_trace: CompletionTrace[Any] = CompletionTrace()

        result = CompletionResult(data=test_data, trace=test_trace)

        assert result.data == test_data
        assert result.trace == test_trace
        assert isinstance(result.data, TestModel)
        assert isinstance(result.trace, CompletionTrace)

    def test_completion_result_generic_typing(self) -> None:
        class TestModel(BaseModel):
            name: str

        test_data = TestModel(name="test")
        test_trace: CompletionTrace[Any] = CompletionTrace()

        result: CompletionResult[TestModel, Any] = CompletionResult(data=test_data, trace=test_trace)

        assert isinstance(result.data, TestModel)
        assert result.data.name == "test"

    def test_completion_result_arbitrary_types(self) -> None:
        class TestModel(BaseModel):
            name: str

        test_data = TestModel(name="test")
        test_trace: CompletionTrace[Any] = CompletionTrace()

        result = CompletionResult(data=test_data, trace=test_trace)
        assert result.data.name == "test"


@pytest.mark.unit
class TestConfigSerialization:
    def test_provider_config_round_trip(self) -> None:
        original = OpenAIProviderConfig(
            api_key="test_key",
            organization="test_org",  # type: ignore[call-arg]
        )

        data = original.model_dump()

        restored = OpenAIProviderConfig(**data)

        assert restored.api_key == original.api_key
        assert restored.model_dump() == original.model_dump()

    def test_client_params_round_trip(self) -> None:
        original = OpenAICompletionClientParams(
            model="gpt-4",
            temperature=0.7,  # type: ignore[call-arg]
            max_tokens=1000,  # type: ignore[call-arg]
        )

        data = original.model_dump()

        restored = OpenAICompletionClientParams(**data)

        assert restored.model == original.model
        assert restored.model_dump() == original.model_dump()

    def test_instructor_config_round_trip(self) -> None:
        original = InstructorConfig(
            mode=instructor.Mode.TOOLS,
            max_retries=5,  # type: ignore[call-arg]
        )

        data = original.model_dump()

        restored = InstructorConfig(**data)

        assert restored.mode == original.mode
        assert restored.model_dump() == original.model_dump()


@pytest.mark.unit
class TestConfigValidation:
    def test_empty_api_key_validation(self) -> None:
        config = OpenAIProviderConfig(api_key="")
        assert config.api_key == ""

    def test_empty_model_validation(self) -> None:
        params = OpenAICompletionClientParams(model="")
        assert params.model == ""

    def test_invalid_instructor_mode(self) -> None:
        with pytest.raises(ValidationError):
            InstructorConfig(mode="invalid_mode")  # type: ignore[arg-type]

    def test_config_immutability(self) -> None:
        config = OpenAIProviderConfig(api_key="test_key")

        updated_config = config.model_copy(update={"api_key": "new_key"})

        assert config.api_key == "test_key"
        assert updated_config.api_key == "new_key"
