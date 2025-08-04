from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Generic, Protocol

import instructor
from instructor.hooks import HookName
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from fatum.structify.types import ClientResponseT, MessageParam


class HookHandler(Protocol):
    """Protocol for hook handler functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> None: ...


class CompletionTrace(BaseModel, Generic[ClientResponseT]):
    completion_kwargs: dict[str, Any] = Field(default_factory=dict)
    messages: list[MessageParam] = Field(default_factory=list)
    raw_response: ClientResponseT | None = None
    parsed_result: BaseModel | None = None
    error: Exception | None = None
    last_attempt_error: Exception | None = None
    parse_error: Exception | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _setup_hooks(
    client: instructor.AsyncInstructor,
) -> tuple[CompletionTrace[ClientResponseT], list[tuple[HookName, HookHandler]]]:
    captured: CompletionTrace[ClientResponseT] = CompletionTrace()

    def capture_kwargs(*_: Any, **kwargs: Any) -> None:
        captured.completion_kwargs = kwargs
        # NOTE: openai and anthropic uses `messages` and gemini uses `contents`
        messages = kwargs.get("messages", kwargs.get("contents", []))

        # NOTE: Gemini Content objects are not serializable, so we convert them to dicts what......?
        if messages and hasattr(messages[0], "model_dump"):
            messages = [msg.model_dump() if hasattr(msg, "model_dump") else msg for msg in messages]

        captured.messages = messages

    def capture_response_data(response: ClientResponseT) -> None:
        captured.raw_response = response

    def capture_error(error: Exception) -> None:
        captured.error = error

    def capture_last_attempt(error: Exception) -> None:
        captured.last_attempt_error = error

    def capture_parse_error(error: Exception) -> None:
        captured.parse_error = error

    hooks: list[tuple[HookName, HookHandler]] = [
        (HookName.COMPLETION_KWARGS, capture_kwargs),
        (HookName.COMPLETION_RESPONSE, capture_response_data),
        (HookName.COMPLETION_ERROR, capture_error),
        (HookName.COMPLETION_LAST_ATTEMPT, capture_last_attempt),
        (HookName.PARSE_ERROR, capture_parse_error),
    ]

    for hook_name, handler in hooks:
        client.on(hook_name, handler)

    return captured, hooks


@asynccontextmanager
async def ahook_instructor(
    client: instructor.AsyncInstructor,
    enable: bool = True,
) -> AsyncIterator[CompletionTrace[ClientResponseT]]:
    """
    Capture execution details from an asynchronous instructor client.

    Use this async context manager when working with asynchronous instructor
    clients (instructor.AsyncInstructor) in async/await code. This is necessary
    when:
    - Your application uses asyncio for concurrent operations
    - You're working within an async function or coroutine
    - You need to make non-blocking API calls
    - You're handling multiple concurrent LLM requests

    Parameters
    ----------
    client : instructor.AsyncInstructor
        The async instructor client instance to capture events from.
    enable : bool, optional
        Whether to enable hook capture. When False, returns an empty
        CompletionTrace without setting up any hooks. Default is True.

    Yields
    ------
    CompletionTrace
        An object containing:
        - kwargs: The complete kwargs passed to the completion API
        - messages: The list of messages sent to the LLM
        - raw_response: The raw API response
        - parsed_result: The parsed Pydantic model result
        - error: Any completion error that occurred
        - last_attempt_error: Error from the final retry attempt
        - parse_error: Any parsing/validation error

    Examples
    --------
    >>> import asyncio
    >>> import instructor
    >>> from openai import AsyncOpenAI
    >>>
    >>> async def main():
    ...     client = instructor.from_openai(AsyncOpenAI())
    ...     async with ahook_instructor(client) as captured:
    ...         result = await client.create(
    ...             response_model=MyModel,
    ...             messages=[{"role": "user", "content": "Hello"}]
    ...         )
    ...         print(captured.messages)
    ...         print(captured.completion_kwargs)  # Access all completion kwargs
    ...         print(captured.raw_response)
    >>>
    >>> asyncio.run(main())
    >>>
    >>> # Conditionally enable hooks based on debug mode
    >>> async def main_with_toggle():
    ...     debug_mode = os.getenv("DEBUG", "").lower() == "true"
    ...     client = instructor.from_openai(AsyncOpenAI())
    ...     async with ahook_instructor(client, enable=debug_mode) as captured:
    ...         result = await client.create(...)

    Notes
    -----
    The async version is required when using AsyncInstructor to properly
    handle the asynchronous event loop and ensure callbacks are registered
    and unregistered correctly in an async context.
    """
    if not enable:
        captured: CompletionTrace[ClientResponseT] = CompletionTrace()
        yield captured
        return

    captured, hooks = _setup_hooks(client)

    try:
        yield captured
    finally:
        for hook_name, handler in hooks:
            client.off(hook_name, handler)


# NOTE: Import response types for model rebuild - these are needed at runtime when CompletionTrace.model_rebuild() is called!
from anthropic.types import Message as AnthropicResponse  # noqa: E402, F401 # type: ignore
from google.genai.types import GenerateContentResponse  # noqa: E402, F401 # type: ignore
from openai.types.chat import ChatCompletion  # noqa: E402, F401 # type: ignore

# NOTE: Rebuild the model to resolve forward references
CompletionTrace.model_rebuild()
