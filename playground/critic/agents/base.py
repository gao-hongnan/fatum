from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

ConfigT = TypeVar("ConfigT", bound=BaseModel)
ResponseT = TypeVar("ResponseT", bound=BaseModel)


class UserQuery(BaseModel):
    question: str = Field(...)


class MaybeOrchestrator(BaseModel, ABC, Generic[ConfigT, ResponseT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: ConfigT

    @abstractmethod
    async def aexecute(self, *args: Any, **kwargs: Any) -> ResponseT: ...
