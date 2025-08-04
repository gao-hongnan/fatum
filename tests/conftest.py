from __future__ import annotations

import instructor
import pytest
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from fatum.structify.adapters.anthropic import AnthropicAdapter
from fatum.structify.adapters.gemini import GeminiAdapter
from fatum.structify.adapters.openai import OpenAIAdapter
from fatum.structify.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    InstructorConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)


class TestSettings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI__API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC__API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI__API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class SimpleTestModel(BaseModel):
    name: str = Field(description="A name")
    age: int = Field(description="An age", ge=0, le=150)
    description: str = Field(description="A brief description")


class ComplexTestModel(BaseModel):
    title: str = Field(description="Title of the content")
    rating: float = Field(description="Rating from 0 to 10", ge=0, le=10)
    tags: list[str] = Field(description="List of relevant tags")
    summary: str = Field(description="Brief summary of the content")
    is_recommended: bool = Field(description="Whether this is recommended")


@pytest.fixture(scope="session")
def test_settings() -> TestSettings:
    return TestSettings()


@pytest.fixture(scope="session")
def skip_if_no_api_keys(test_settings: TestSettings) -> None:
    if not all(
        [
            test_settings.openai_api_key,
            test_settings.anthropic_api_key,
            test_settings.gemini_api_key,
        ]
    ):
        pytest.skip("API keys not available for integration tests")


@pytest.fixture(scope="session")
def _skip_if_no_api_keys(test_settings: TestSettings) -> None:
    if not all(
        [
            test_settings.openai_api_key,
            test_settings.anthropic_api_key,
            test_settings.gemini_api_key,
        ]
    ):
        pytest.skip("API keys not available for integration tests")


@pytest.fixture
def openai_provider_config(test_settings: TestSettings) -> OpenAIProviderConfig:
    return OpenAIProviderConfig(api_key=test_settings.openai_api_key)


@pytest.fixture
def anthropic_provider_config(test_settings: TestSettings) -> AnthropicProviderConfig:
    return AnthropicProviderConfig(api_key=test_settings.anthropic_api_key)


@pytest.fixture
def gemini_provider_config(test_settings: TestSettings) -> GeminiProviderConfig:
    return GeminiProviderConfig(api_key=test_settings.gemini_api_key)


@pytest.fixture
def openai_completion_params() -> OpenAICompletionClientParams:
    return OpenAICompletionClientParams.model_validate(
        {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_completion_tokens": 500,
        }
    )


@pytest.fixture
def anthropic_completion_params() -> AnthropicCompletionClientParams:
    return AnthropicCompletionClientParams.model_validate(
        {
            "model": "claude-3-5-haiku-20241022",
            "temperature": 0.1,
            "max_tokens": 500,
        }
    )


@pytest.fixture
def gemini_completion_params() -> GeminiCompletionClientParams:
    return GeminiCompletionClientParams.model_validate(
        {
            "model": "gemini-2.5-flash",
            "temperature": 0.1,
            "max_output_tokens": 500,
        }
    )


@pytest.fixture
def openai_instructor_config() -> InstructorConfig:
    return InstructorConfig(mode=instructor.Mode.TOOLS)


@pytest.fixture
def anthropic_instructor_config() -> InstructorConfig:
    return InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS)


@pytest.fixture
def gemini_instructor_config() -> InstructorConfig:
    return InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)


@pytest.fixture
def openai_adapter(
    openai_provider_config: OpenAIProviderConfig,
    openai_completion_params: OpenAICompletionClientParams,
    openai_instructor_config: InstructorConfig,
) -> OpenAIAdapter:
    adapter = OpenAIAdapter(
        provider_config=openai_provider_config,
        completion_params=openai_completion_params,
        instructor_config=openai_instructor_config,
    )
    return adapter


@pytest.fixture
def anthropic_adapter(
    anthropic_provider_config: AnthropicProviderConfig,
    anthropic_completion_params: AnthropicCompletionClientParams,
    anthropic_instructor_config: InstructorConfig,
) -> AnthropicAdapter:
    adapter = AnthropicAdapter(
        provider_config=anthropic_provider_config,
        completion_params=anthropic_completion_params,
        instructor_config=anthropic_instructor_config,
    )
    return adapter


@pytest.fixture
def gemini_adapter(
    gemini_provider_config: GeminiProviderConfig,
    gemini_completion_params: GeminiCompletionClientParams,
    gemini_instructor_config: InstructorConfig,
) -> GeminiAdapter:
    adapter = GeminiAdapter(
        provider_config=gemini_provider_config,
        completion_params=gemini_completion_params,
        instructor_config=gemini_instructor_config,
    )
    return adapter


@pytest.fixture
def simple_test_model_class() -> type[SimpleTestModel]:
    return SimpleTestModel


@pytest.fixture
def complex_test_model_class() -> type[ComplexTestModel]:
    return ComplexTestModel


@pytest.fixture
def test_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that provides structured responses.",
        },
        {
            "role": "user",
            "content": "Generate information about a person named Alice who is 25 years old and works as a software engineer.",
        },
    ]


@pytest.fixture
def complex_test_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a helpful movie reviewer that provides detailed structured reviews.",
        },
        {
            "role": "user",
            "content": "Review the movie 'Inception' with a rating, tags, and metadata.",
        },
    ]


pytest_plugins = ["pytest_asyncio"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test requiring API keys",
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test that uses mocking",
    )
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
