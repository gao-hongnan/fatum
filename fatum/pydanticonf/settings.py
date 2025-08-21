from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_settings.sources import PydanticBaseSettingsSource, YamlConfigSettingsSource


class BaseSettingsWithYaml(BaseSettings):
    """Base settings class with YAML configuration support."""

    @classmethod
    def settings_customise_sources(
        cls: type[BaseSettingsWithYaml],
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize settings sources to include YAML support."""
        config: SettingsConfigDict = cls.model_config

        if not (yaml_file := config.get("yaml_file")):
            return super().settings_customise_sources(
                settings_cls=settings_cls,
                init_settings=init_settings,
                env_settings=env_settings,
                dotenv_settings=dotenv_settings,
                file_secret_settings=file_secret_settings,
            )

        yaml_settings = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=yaml_file,
            yaml_file_encoding=config.get("yaml_file_encoding", "utf-8"),
        )

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            yaml_settings,
        )
