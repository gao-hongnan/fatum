from __future__ import annotations

import os
import platform
from typing import Any, Callable

import psutil
from pydantic import BaseModel, Field


def fallback(func: Callable[[], Any], default: Any = None) -> Any:
    """Safely call platform functions with fallback."""
    try:
        result = func()
        return result if result else default
    except Exception:
        return default


class MemoryInfo(BaseModel):
    """Memory information."""

    total_mb: int | None = None
    available_mb: int | None = None

    @classmethod
    def current(cls: type[MemoryInfo]) -> MemoryInfo:
        """Get current memory information."""
        try:
            vm = psutil.virtual_memory()
            return cls(
                total_mb=vm.total // (1024 * 1024),
                available_mb=vm.available // (1024 * 1024),
            )
        except (ImportError, AttributeError):
            return cls()


class SystemInfo(BaseModel):
    """System information for experiment tracking and reproducibility."""

    platform: str = Field(default_factory=lambda: fallback(platform.system, ""))
    platform_release: str = Field(default_factory=lambda: fallback(platform.release, ""))
    platform_version: str = Field(default_factory=lambda: fallback(platform.version, ""))
    architecture: str = Field(default_factory=lambda: fallback(platform.machine, ""))
    hostname: str = Field(default_factory=lambda: fallback(platform.node, ""))

    cpu_count: int | None = Field(default_factory=lambda: fallback(os.cpu_count, None))
    memory: MemoryInfo = Field(default_factory=MemoryInfo.current)


def get_system_info() -> SystemInfo:
    return SystemInfo()
