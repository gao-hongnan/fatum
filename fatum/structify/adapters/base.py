from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Generic, Literal, overload

from instructor import AsyncInstructor
from openai.types.chat import ChatCompletionMessageParam

from fatum.structify.hooks import ahook_instructor
from fatum.structify.models import CompletionResult
from fatum.structify.types import BaseProviderConfigT, ClientResponseT, ClientT, StructuredResponseT

if TYPE_CHECKING:
    from fatum.structify.hooks import CompletionTrace
    from fatum.structify.models import CompletionClientParams, CompletionResult, InstructorConfig


class BaseAdapter(ABC, Generic[BaseProviderConfigT, ClientT, ClientResponseT]):
    def __init__(
        self,
        *,
        provider_config: BaseProviderConfigT,
        completion_params: CompletionClientParams,
        instructor_config: InstructorConfig,
    ) -> None:
        self.provider_config = provider_config
        self.completion_params = completion_params
        self.instructor_config = instructor_config
        self._client: ClientT | None = None
        self._instructor: AsyncInstructor | None = None
        self._client_lock = threading.Lock()
        self._instructor_lock = threading.Lock()

    @property
    def client(self) -> ClientT:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._create_client()
        return self._client

    @property
    def instructor(self) -> AsyncInstructor:
        if self._instructor is None:
            with self._instructor_lock:
                if self._instructor is None:
                    self._instructor = self._with_instructor()
        return self._instructor

    @abstractmethod
    def _create_client(self) -> ClientT: ...

    @abstractmethod
    def _with_instructor(self) -> AsyncInstructor: ...

    @overload
    async def acreate(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        *,
        with_hooks: Literal[False] = False,
        **kwargs: Any,
    ) -> StructuredResponseT: ...

    @overload
    async def acreate(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        *,
        with_hooks: Literal[True],
        **kwargs: Any,
    ) -> CompletionResult[StructuredResponseT, ClientResponseT]: ...

    async def acreate(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        with_hooks: bool = False,
        **kwargs: Any,
    ) -> StructuredResponseT | CompletionResult[StructuredResponseT, ClientResponseT]:
        if not messages:
            raise ValueError("Messages list cannot be empty")

        captured: CompletionTrace[ClientResponseT]
        # NOTE: Merge completion params with kwargs, letting kwargs override
        completion_kwargs = {**self.completion_params.model_dump(), **kwargs}

        async with ahook_instructor(self.instructor, enable=with_hooks) as captured:
            response = await self.instructor.create(
                response_model=response_model,
                messages=messages,
                **self.instructor_config.model_dump(exclude={"mode"}),
                **completion_kwargs,
            )
            return self._assemble(response, captured, with_hooks)

    @overload
    def astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        *,
        with_hooks: Literal[False] = False,
    ) -> AsyncIterator[StructuredResponseT]: ...

    @overload
    def astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        *,
        with_hooks: Literal[True],
    ) -> AsyncIterator[CompletionResult[StructuredResponseT, ClientResponseT]]: ...

    def astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        with_hooks: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StructuredResponseT | CompletionResult[StructuredResponseT, ClientResponseT]]:
        return self._astream(messages, response_model, with_hooks, **kwargs)

    async def _astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        with_hooks: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StructuredResponseT | CompletionResult[StructuredResponseT, ClientResponseT]]:
        if not messages:
            raise ValueError("Messages list cannot be empty")

        captured: CompletionTrace[ClientResponseT]
        completion_kwargs = {**self.completion_params.model_dump(), **kwargs}

        async with ahook_instructor(self.instructor, enable=with_hooks) as captured:
            async for partial in self.instructor.create_partial(
                response_model=response_model,
                messages=messages,
                **self.instructor_config.model_dump(exclude={"mode"}),
                **completion_kwargs,
            ):
                yield self._assemble(partial, captured, with_hooks)

    def _assemble(
        self,
        response: StructuredResponseT,
        captured: CompletionTrace[ClientResponseT],
        with_hooks: bool,
    ) -> StructuredResponseT | CompletionResult[StructuredResponseT, ClientResponseT]:
        if not with_hooks:
            return response

        return CompletionResult(data=response, trace=captured)
