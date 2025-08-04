"""Pydantic models for the structify module."""

from __future__ import annotations

from typing import Annotated, Generic, Literal

import instructor
from pydantic import BaseModel, ConfigDict, Field

from fatum.structify.enums import Capability, Provider
from fatum.structify.hooks import CompletionTrace
from fatum.structify.types import ClientResponseT, StructuredResponseT


class Allowable(BaseModel):
    model_config = ConfigDict(extra="allow")


class BaseProviderConfig(Allowable):
    api_key: str  # NOTE: All 3 big providers names this `api_key` - do a drift check if really need rename this or remove this field.


class OpenAIProviderConfig(BaseProviderConfig):
    provider: Literal["openai"] = Field(default=Provider.OPENAI.value, exclude=True)


class AnthropicProviderConfig(BaseProviderConfig):
    provider: Literal["anthropic"] = Field(default=Provider.ANTHROPIC.value, exclude=True)


class GeminiProviderConfig(BaseProviderConfig):
    provider: Literal["gemini"] = Field(default=Provider.GEMINI.value, exclude=True)


class AzureOpenAIProviderConfig(BaseProviderConfig):
    provider: Literal["azure-openai"] = Field(default=Provider.AZURE_OPENAI.value, exclude=True)


ProviderConfig = Annotated[
    OpenAIProviderConfig | AnthropicProviderConfig | GeminiProviderConfig | AzureOpenAIProviderConfig,
    Field(discriminator="provider"),
]


class BaseClientParams(Allowable):
    capability: Capability = Field(exclude=True)
    model: str


class OpenAICompletionClientParams(BaseClientParams):
    provider: Literal["openai"] = Field(default=Provider.OPENAI.value, exclude=True)
    capability: Capability = Field(default=Capability.COMPLETION, exclude=True)


class OpenAIEmbeddingClientParams(BaseClientParams):
    provider: Literal["openai"] = Field(default=Provider.OPENAI.value, exclude=True)
    capability: Capability = Field(default=Capability.EMBEDDING, exclude=True)


class OpenAIVisionClientParams(BaseClientParams):
    provider: Literal["openai"] = Field(default=Provider.OPENAI.value, exclude=True)
    capability: Capability = Field(default=Capability.VISION, exclude=True)


class AnthropicCompletionClientParams(BaseClientParams):
    provider: Literal["anthropic"] = Field(default=Provider.ANTHROPIC.value, exclude=True)
    capability: Capability = Field(default=Capability.COMPLETION, exclude=True)


class GeminiCompletionClientParams(BaseClientParams):
    provider: Literal["gemini"] = Field(default=Provider.GEMINI.value, exclude=True)
    capability: Capability = Field(default=Capability.COMPLETION, exclude=True)


class AzureOpenAICompletionClientParams(BaseClientParams):
    provider: Literal["azure-openai"] = Field(default=Provider.AZURE_OPENAI.value, exclude=True)
    capability: Capability = Field(default=Capability.COMPLETION, exclude=True)


CompletionClientParams = Annotated[
    OpenAICompletionClientParams
    | AnthropicCompletionClientParams
    | GeminiCompletionClientParams
    | AzureOpenAICompletionClientParams,
    Field(discriminator="provider"),
]


class InstructorConfig(Allowable):
    mode: instructor.Mode


class CompletionResult(BaseModel, Generic[StructuredResponseT, ClientResponseT]):
    data: StructuredResponseT
    trace: CompletionTrace[ClientResponseT]

    model_config = ConfigDict(arbitrary_types_allowed=True)
