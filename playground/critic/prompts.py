from __future__ import annotations

from functools import lru_cache
from typing import Any, Protocol, TypeAlias

from jinja2 import Environment, FileSystemLoader
from jinja2.exceptions import TemplateNotFound, UndefinedError

PromptVariables: TypeAlias = dict[str, Any]


class PromptRenderer(Protocol):
    def render(self, template_path: str, variables: PromptVariables) -> str: ...


class JinjaPromptRenderer:
    """A simple, stateless Jinja2 template renderer."""

    _jinja_env: Environment

    def __init__(self, template_base_path: str) -> None:
        self._jinja_env = Environment(
            loader=FileSystemLoader(template_base_path),
            autoescape=False,
        )

    def render(self, template_path: str, variables: PromptVariables | None = None) -> str:
        try:
            template = self._jinja_env.get_template(template_path)
            return template.render(variables)
        except TemplateNotFound as e:
            raise TemplateNotFound(f"Prompt template not found at '{template_path}' (looked for: {e.name})") from e
        except UndefinedError as e:
            raise UndefinedError(f"Missing required variable in template '{template_path}': {e.message}") from e


@lru_cache(maxsize=1)
def get_prompt_renderer(base_path: str) -> JinjaPromptRenderer:
    """Provides a cached, singleton instance of the PromptRenderer."""
    return JinjaPromptRenderer(base_path)
