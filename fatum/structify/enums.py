from enum import StrEnum


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AZURE_OPENAI = "azure-openai"


class Capability(StrEnum):
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
