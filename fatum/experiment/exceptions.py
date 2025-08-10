from __future__ import annotations

from typing import Any


class ExperimentError(Exception):
    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context = context or {}


class ValidationError(ExperimentError):
    def __init__(self, field: str, value: Any, message: str) -> None:
        super().__init__(
            f"Validation failed for '{field}': {message}",
            {"field": field, "value": value},
        )


class StorageError(ExperimentError):
    pass


class NotFoundError(ExperimentError):
    def __init__(self, resource_type: str, identifier: str) -> None:
        super().__init__(
            f"{resource_type} not found: {identifier}",
            {"resource_type": resource_type, "identifier": identifier},
        )


class StateError(ExperimentError):
    def __init__(self, current_state: str, action: str) -> None:
        super().__init__(
            f"Cannot {action} in {current_state} state",
            {"current_state": current_state, "action": action},
        )
