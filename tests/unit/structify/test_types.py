from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from fatum.structify.enums import Capability, Provider
from fatum.structify.types import (
    BaseProviderConfigT,
    ClientResponseT,
    ClientT,
    StructuredResponseT,
)

if TYPE_CHECKING:
    pass


class SampleModel(BaseModel):
    name: str
    value: int


@pytest.mark.unit
class TestProviderEnum:
    def test_provider_enum_values(self) -> None:
        assert str(Provider.OPENAI) == "openai"
        assert str(Provider.ANTHROPIC) == "anthropic"
        assert str(Provider.GEMINI) == "gemini"
        assert str(Provider.AZURE_OPENAI) == "azure-openai"

    def test_provider_enum_membership(self) -> None:
        assert "openai" in Provider
        assert "anthropic" in Provider
        assert "gemini" in Provider
        assert "azure-openai" in Provider

        assert "invalid_provider" not in Provider
        assert "chatgpt" not in Provider
        assert "claude" not in Provider

    def test_provider_enum_iteration(self) -> None:
        providers = list(Provider)
        assert len(providers) == 4
        assert Provider.OPENAI in providers
        assert Provider.ANTHROPIC in providers
        assert Provider.GEMINI in providers
        assert Provider.AZURE_OPENAI in providers

    def test_provider_enum_string_operations(self) -> None:
        provider = Provider.OPENAI

        assert str(provider) == "openai"
        assert provider.upper() == "OPENAI"
        assert provider.capitalize() == "Openai"
        assert provider.startswith("open")
        assert provider.endswith("ai")

    def test_provider_enum_comparison(self) -> None:
        assert Provider.OPENAI == Provider.OPENAI
        assert Provider.OPENAI is Provider.OPENAI

        providers = [Provider.OPENAI, Provider.ANTHROPIC, Provider.GEMINI]
        for i, p1 in enumerate(providers):
            for p2 in providers[i + 1 :]:
                assert p1 != p2

        assert Provider.OPENAI.value == "openai"
        assert Provider.ANTHROPIC.value == "anthropic"
        assert Provider.GEMINI.value == "gemini"

        for provider in Provider:
            assert provider.value.islower()
            assert provider.value == provider.value.lower()

    def test_provider_enum_constructors(self) -> None:
        assert Provider("openai") == Provider.OPENAI
        assert Provider("anthropic") == Provider.ANTHROPIC
        assert Provider("gemini") == Provider.GEMINI

        with pytest.raises(ValueError):
            Provider("invalid_provider")


@pytest.mark.unit
class TestCapabilityEnum:
    def test_capability_enum_values(self) -> None:
        assert str(Capability.COMPLETION) == "completion"
        assert str(Capability.EMBEDDING) == "embedding"
        assert str(Capability.VISION) == "vision"

    def test_capability_enum_membership(self) -> None:
        assert "completion" in Capability
        assert "embedding" in Capability
        assert "vision" in Capability

        assert "invalid_capability" not in Capability
        assert "chat" not in Capability
        assert "text_generation" not in Capability

    def test_capability_enum_iteration(self) -> None:
        capabilities = list(Capability)
        assert len(capabilities) == 3
        assert Capability.COMPLETION in capabilities
        assert Capability.EMBEDDING in capabilities
        assert Capability.VISION in capabilities

    def test_capability_enum_string_operations(self) -> None:
        capability = Capability.COMPLETION

        assert str(capability) == "completion"
        assert capability.upper() == "COMPLETION"
        assert capability.capitalize() == "Completion"
        assert capability.startswith("comp")
        assert capability.endswith("tion")

    def test_capability_enum_comparison(self) -> None:
        assert Capability.COMPLETION == Capability.COMPLETION
        assert Capability.COMPLETION is Capability.COMPLETION

        capabilities = [Capability.COMPLETION, Capability.EMBEDDING, Capability.VISION]
        for i, c1 in enumerate(capabilities):
            for c2 in capabilities[i + 1 :]:
                assert c1 != c2

        assert Capability.COMPLETION.value == "completion"
        assert Capability.EMBEDDING.value == "embedding"
        assert Capability.VISION.value == "vision"

        for capability in Capability:
            assert capability.value.islower()
            assert capability.value == capability.value.lower()

    def test_capability_enum_constructors(self) -> None:
        assert Capability("completion") == Capability.COMPLETION
        assert Capability("embedding") == Capability.EMBEDDING
        assert Capability("vision") == Capability.VISION

        with pytest.raises(ValueError):
            Capability("invalid_capability")


@pytest.mark.unit
class TestTypeVars:
    def test_structured_response_t(self) -> None:
        assert StructuredResponseT.__bound__ is BaseModel
        assert StructuredResponseT.__name__ == "StructuredResponseT"

    def test_base_provider_config_t(self) -> None:
        assert BaseProviderConfigT.__name__ == "BaseProviderConfigT"

    def test_client_t(self) -> None:
        assert ClientT.__name__ == "ClientT"
        if hasattr(ClientT, "__bound__"):
            assert ClientT.__bound__ is None
        if hasattr(ClientT, "__constraints__"):
            assert not ClientT.__constraints__

    def test_client_response_t(self) -> None:
        assert ClientResponseT.__name__ == "ClientResponseT"
        if hasattr(ClientResponseT, "__bound__"):
            bound = ClientResponseT.__bound__
            assert bound is not None

    def test_type_var_variance(self) -> None:
        assert not hasattr(StructuredResponseT, "__covariant__") or not StructuredResponseT.__covariant__
        assert not hasattr(StructuredResponseT, "__contravariant__") or not StructuredResponseT.__contravariant__

        assert not hasattr(ClientT, "__covariant__") or not ClientT.__covariant__
        assert not hasattr(ClientT, "__contravariant__") or not ClientT.__contravariant__


@pytest.mark.unit
class TestTypeDefinitions:
    def test_type_checking_imports(self) -> None:
        from fatum.structify import types

        assert hasattr(types, "TYPE_CHECKING")
        assert types.TYPE_CHECKING is False

    def test_response_type_constraints(self) -> None:
        assert hasattr(ClientResponseT, "__name__")
        assert ClientResponseT.__name__ == "ClientResponseT"

    def test_forward_references(self) -> None:
        from fatum.structify.types import BaseProviderConfigT

        assert isinstance(BaseProviderConfigT, type(StructuredResponseT))

    def test_module_level_exports(self) -> None:
        from fatum.structify import types

        expected_type_exports = [
            "StructuredResponseT",
            "BaseProviderConfigT",
            "ClientT",
            "ClientResponseT",
            "CompletionClientParamsT",
            "MessageParam",
        ]

        for export in expected_type_exports:
            assert hasattr(types, export), f"Missing export in types.py: {export}"

        # Check that enums are now in enums.py
        from fatum.structify import enums

        expected_enum_exports = [
            "Provider",
            "Capability",
        ]

        for export in expected_enum_exports:
            assert hasattr(enums, export), f"Missing export in enums.py: {export}"

    def test_enum_base_classes(self) -> None:
        from enum import StrEnum

        assert issubclass(Provider, StrEnum)
        assert issubclass(Capability, StrEnum)

        assert issubclass(Provider, str)
        assert issubclass(Capability, str)


@pytest.mark.unit
class TestTypeAnnotations:
    def test_provider_annotations(self) -> None:
        assert hasattr(Provider, "__annotations__") or Provider.__class__.__annotations__

    def test_capability_annotations(self) -> None:
        assert hasattr(Capability, "__annotations__") or Capability.__class__.__annotations__

    def test_type_var_annotations(self) -> None:
        from typing import TypeVar

        assert isinstance(StructuredResponseT, TypeVar)
        assert isinstance(BaseProviderConfigT, TypeVar)
        assert isinstance(ClientT, TypeVar)
        assert isinstance(ClientResponseT, TypeVar)


@pytest.mark.unit
class TestEnumExtensibility:
    def test_provider_extensibility(self) -> None:
        current_providers = {Provider.OPENAI, Provider.ANTHROPIC, Provider.GEMINI}
        assert len(current_providers) == 3

        for provider in current_providers:
            assert isinstance(provider, str)
            assert len(provider) > 0
            assert provider.islower()

    def test_capability_extensibility(self) -> None:
        current_capabilities = {Capability.COMPLETION, Capability.EMBEDDING, Capability.VISION}
        assert len(current_capabilities) == 3

        for capability in current_capabilities:
            assert isinstance(capability, str)
            assert len(capability) > 0
            assert capability.islower()

    def test_enum_naming_conventions(self) -> None:
        assert str(Provider.OPENAI) == "openai"
        assert str(Provider.ANTHROPIC) == "anthropic"
        assert str(Provider.GEMINI) == "gemini"

        assert str(Capability.COMPLETION) == "completion"
        assert str(Capability.EMBEDDING) == "embedding"
        assert str(Capability.VISION) == "vision"

    def test_enum_serialization(self) -> None:
        import json

        provider_data = {"provider": Provider.OPENAI}
        serialized = json.dumps(provider_data, default=str)
        deserialized = json.loads(serialized)
        assert deserialized["provider"] == "openai"

        capability_data = {"capability": Capability.COMPLETION}
        serialized = json.dumps(capability_data, default=str)
        deserialized = json.loads(serialized)
        assert deserialized["capability"] == "completion"


@pytest.mark.unit
class TestTypeCompatibility:
    def test_provider_string_compatibility(self) -> None:
        provider = Provider.OPENAI

        assert f"Provider: {provider}" == "Provider: openai"
        assert f"The provider is {provider}" == "The provider is openai"

        assert provider.replace("open", "closed") == "closedai"
        assert provider.split("e") == ["op", "nai"]

        assert provider in ["openai", "anthropic", "gemini"]
        assert provider not in ["chatgpt", "claude"]

    def test_capability_string_compatibility(self) -> None:
        capability = Capability.COMPLETION

        assert f"Capability: {capability}" == "Capability: completion"
        assert f"The capability is {capability}" == "The capability is completion"

        assert capability.replace("comp", "exec") == "execletion"
        assert capability.split("t") == ["comple", "ion"]

        assert capability in ["completion", "embedding", "vision"]
        assert capability not in ["generation", "translation"]

    def test_type_var_generic_compatibility(self) -> None:
        from typing import Generic, List

        class GenericContainer(Generic[StructuredResponseT]):
            def __init__(self, items: List[StructuredResponseT]) -> None:
                self.items = items

            def get_first(self) -> StructuredResponseT | None:
                return self.items[0] if self.items else None

        test_items = [SampleModel(name="test1", value=1), SampleModel(name="test2", value=2)]
        container = GenericContainer(test_items)

        assert container.get_first() == test_items[0]
        assert isinstance(container.get_first(), SampleModel)
