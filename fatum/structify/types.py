from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from anthropic.types import Message as AnthropicResponse
    from google.genai.types import GenerateContentResponse
    from openai.types.chat import ChatCompletion

    from fatum.structify.config import (
        BaseProviderConfig,
        CompletionClientParams,
    )

StructuredResponseT = TypeVar("StructuredResponseT", bound=BaseModel)
BaseProviderConfigT = TypeVar("BaseProviderConfigT", bound="BaseProviderConfig")
ClientT = TypeVar("ClientT")
CompletionClientParamsT = TypeVar("CompletionClientParamsT", bound="CompletionClientParams")
ClientResponseT = TypeVar("ClientResponseT", bound="ChatCompletion | AnthropicResponse | GenerateContentResponse")
MessageParam: TypeAlias = dict[str, Any]


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Capability(StrEnum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
