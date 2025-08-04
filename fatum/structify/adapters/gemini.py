from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, overload

import instructor
from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from openai.types.chat import ChatCompletionMessageParam

from fatum.structify.adapters.base import BaseAdapter
from fatum.structify.hooks import CompletionTrace, ahook_instructor
from fatum.structify.models import GeminiProviderConfig
from fatum.structify.types import StructuredResponseT

if TYPE_CHECKING:
    from fatum.structify.models import CompletionResult


class GeminiAdapter(BaseAdapter[GeminiProviderConfig, genai.Client, GenerateContentResponse]):
    def _create_client(self) -> genai.Client:
        return genai.Client(api_key=self.provider_config.api_key)

    def _with_instructor(self) -> instructor.AsyncInstructor:
        client: genai.Client = self.client
        result: instructor.AsyncInstructor = instructor.from_genai(
            client, use_async=True, mode=self.instructor_config.mode
        )
        assert isinstance(result, instructor.AsyncInstructor)
        return result

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
    ) -> CompletionResult[StructuredResponseT, GenerateContentResponse]: ...

    async def acreate(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        with_hooks: bool = False,
        **kwargs: Any,
    ) -> StructuredResponseT | CompletionResult[StructuredResponseT, GenerateContentResponse]:
        model = self.completion_params.model
        config = GenerateContentConfig(**self.completion_params.model_dump(exclude={"model"}))

        captured: CompletionTrace[GenerateContentResponse]
        async with ahook_instructor(self.instructor, enable=with_hooks) as captured:
            response = await self.instructor.create(
                model=model,
                response_model=response_model,
                messages=messages,
                **self.instructor_config.model_dump(exclude={"mode"}),
                config=config,
                **kwargs,
            )
            return self._assemble(response, captured, with_hooks)

    async def _astream(
        self,
        messages: list[ChatCompletionMessageParam],
        response_model: type[StructuredResponseT],
        with_hooks: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StructuredResponseT | CompletionResult[StructuredResponseT, GenerateContentResponse]]:
        model = self.completion_params.model
        config = GenerateContentConfig(**self.completion_params.model_dump(exclude={"model"}))

        captured: CompletionTrace[GenerateContentResponse]
        async with ahook_instructor(self.instructor, enable=with_hooks) as captured:
            async for partial in self.instructor.create_partial(
                model=model,
                response_model=response_model,
                messages=messages,
                **self.instructor_config.model_dump(exclude={"mode"}),
                config=config,
                **kwargs,
            ):
                yield self._assemble(partial, captured, with_hooks)
