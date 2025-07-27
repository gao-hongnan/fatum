from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, overload

import instructor
from google import genai
from google.genai.types import GenerateContentResponse
from openai.types.chat import ChatCompletionMessageParam

from fatum.structify.adapters.base import BaseAdapter
from fatum.structify.config import GeminiProviderConfig
from fatum.structify.types import BaseModelT

if TYPE_CHECKING:
    from fatum.structify.config import CompletionResult, Message
    from fatum.structify.hooks import CompletionTrace


class GeminiAdapter(BaseAdapter[GeminiProviderConfig, genai.Client, GenerateContentResponse]):
    def _create_client(self) -> genai.Client:
        return genai.Client(api_key=self.provider_config.api_key)

    def _with_instructor(self) -> instructor.AsyncInstructor:
        client: genai.Client = self.client
        return instructor.from_genai(client, use_async=True, mode=self.instructor_config.mode)  # type: ignore[no-any-return]

    # @overload
    # async def acreate(
    #     self,
    #     messages: list[Message],
    #     response_model: type[BaseModelT],
    #     *,
    #     with_hooks: Literal[False] = False,
    #     **kwargs: Any,
    # ) -> BaseModelT: ...

    # @overload
    # async def acreate(
    #     self,
    #     messages: list[Message],
    #     response_model: type[BaseModelT],
    #     *,
    #     with_hooks: Literal[True],
    #     **kwargs: Any,
    # ) -> CompletionResult[BaseModelT, GenerateContentResponse]: ...

    # async def acreate(
    #     self,
    #     messages: list[Message],
    #     response_model: type[BaseModelT],
    #     with_hooks: bool = False,
    #     **kwargs: Any,
    # ) -> BaseModelT | CompletionResult[BaseModelT, GenerateContentResponse]:
    #     from fatum.structify.hooks import ahook_instructor

    #     formatted_messages = self._format_messages(messages)

    #     captured: CompletionTrace[GenerateContentResponse]
    #     async with ahook_instructor(self.instructor, enable=with_hooks) as captured:
    #         response = await self.instructor.create(
    #             model=self.completion_params.model,
    #             response_model=response_model,
    #             messages=formatted_messages,
    #             **self.instructor_config.model_dump(exclude={"mode"}),
    #             **kwargs,
    #         )
    #         return self._assemble(response, captured, with_hooks)

    # async def _astream(
    #     self,
    #     formatted_messages: list[ChatCompletionMessageParam],
    #     response_model: type[BaseModelT],
    #     with_hooks: bool = False,
    # ) -> AsyncIterator[BaseModelT | CompletionResult[BaseModelT, GenerateContentResponse]]:
    #     from fatum.structify.hooks import ahook_instructor

    #     captured: CompletionTrace[GenerateContentResponse]
    #     async with ahook_instructor(self.instructor, enable=with_hooks) as captured:
    #         # NOTE: Don't pass **self.completion_params.model_dump() as they're already configured in the Gemini client - this is a Gemini-specific quirk.
    #         async for partial in self.instructor.create_partial(
    #             model=self.completion_params.model,
    #             response_model=response_model,
    #             messages=formatted_messages,
    #             **self.instructor_config.model_dump(exclude={"mode"}),
    #         ):
    #             yield self._assemble(partial, captured, with_hooks)
