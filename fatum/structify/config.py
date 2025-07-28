from __future__ import annotations

from typing import Annotated, Generic, Literal

import instructor
from pydantic import BaseModel, ConfigDict, Field

from fatum.structify.hooks import CompletionTrace
from fatum.structify.types import (
    BaseModelT,
    Capability,
    Provider,
    ResponseT,
)


class Allowable(BaseModel):
    model_config = ConfigDict(extra="allow")


class BaseProviderConfig(Allowable):
    api_key: str


class OpenAIProviderConfig(BaseProviderConfig):
    provider: Literal["openai"] = Field(default=Provider.OPENAI.value, exclude=True)


class AnthropicProviderConfig(BaseProviderConfig):
    provider: Literal["anthropic"] = Field(default=Provider.ANTHROPIC.value, exclude=True)


class GeminiProviderConfig(BaseProviderConfig):
    provider: Literal["gemini"] = Field(default=Provider.GEMINI.value, exclude=True)


ProviderConfig = Annotated[
    OpenAIProviderConfig | AnthropicProviderConfig | GeminiProviderConfig,
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


CompletionClientParams = Annotated[
    OpenAICompletionClientParams | AnthropicCompletionClientParams | GeminiCompletionClientParams,
    Field(discriminator="provider"),
]


class InstructorConfig(Allowable):
    mode: instructor.Mode


class CompletionResult(BaseModel, Generic[BaseModelT, ResponseT]):
    data: BaseModelT
    trace: CompletionTrace[ResponseT]

    model_config = ConfigDict(arbitrary_types_allowed=True)
