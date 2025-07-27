from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, AsyncIterator, Generic, Literal, overload

from instructor import AsyncInstructor
from openai.types.chat import ChatCompletionMessageParam

from fatum.structify.config import CompletionResult
from fatum.structify.hooks import ahook_instructor
from fatum.structify.types import BaseModelT, BaseProviderConfigT, ClientT, ResponseT

if TYPE_CHECKING:
    from fatum.structify.config import CompletionClientParams, CompletionResult, InstructorConfig
    from fatum.structify.hooks import CompletionTrace


class BaseAdapter(ABC, Generic[BaseProviderConfigT, ClientT, ResponseT]):
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

    @property
    def client(self) -> ClientT:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def instructor(self) -> AsyncInstructor:
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
        response_model: type[BaseModelT],
        *,
        with_hooks: Literal[False] = False,
        **kwargs: Any,
    ) -> BaseModelT: ...

    @overload
    async def acreate(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[BaseModelT],
        *,
        with_hooks: Literal[True],
        **kwargs: Any,
    ) -> CompletionResult[BaseModelT, ResponseT]: ...

    async def acreate(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[BaseModelT],
        with_hooks: bool = False,
        **kwargs: Any,
    ) -> BaseModelT | CompletionResult[BaseModelT, ResponseT]:
        captured: CompletionTrace[ResponseT]
        async with ahook_instructor(self.instructor, enable=with_hooks) as captured:
            response = await self.instructor.create(
                response_model=response_model,
                messages=messages,
                **self.instructor_config.model_dump(exclude={"mode"}),
                **self.completion_params.model_dump(),
                **kwargs,
            )
            return self._assemble(response, captured, with_hooks)

    @overload
    def astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[BaseModelT],
        *,
        with_hooks: Literal[False] = False,
    ) -> AsyncIterator[BaseModelT]: ...

    @overload
    def astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[BaseModelT],
        *,
        with_hooks: Literal[True],
    ) -> AsyncIterator[CompletionResult[BaseModelT, ResponseT]]: ...

    def astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[BaseModelT],
        with_hooks: bool = False,
    ) -> AsyncIterator[BaseModelT | CompletionResult[BaseModelT, ResponseT]]:
        return self._astream(messages, response_model, with_hooks)

    async def _astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[BaseModelT],
        with_hooks: bool = False,
    ) -> AsyncIterator[BaseModelT | CompletionResult[BaseModelT, ResponseT]]:
        captured: CompletionTrace[ResponseT]
        async with ahook_instructor(self.instructor, enable=with_hooks) as captured:
            async for partial in self.instructor.create_partial(
                response_model=response_model,
                messages=messages,
                **self.instructor_config.model_dump(exclude={"mode"}),
                **self.completion_params.model_dump(),
            ):
                yield self._assemble(partial, captured, with_hooks)

    def _assemble(
        self,
        response: BaseModelT,
        captured: CompletionTrace[ResponseT],
        with_hooks: bool,
    ) -> BaseModelT | CompletionResult[BaseModelT, ResponseT]:
        if not with_hooks:
            return response

        return CompletionResult(data=response, trace=captured)
