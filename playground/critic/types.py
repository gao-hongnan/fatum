from pydantic import BaseModel, Field, computed_field

from .agents.base import UserQuery
from .agents.cove import CoVeCandidate


class VerificationResult(BaseModel):
    user_query: UserQuery = Field(...)
    result: CoVeCandidate = Field(...)


class BatchVerificationResult(BaseModel):
    results: list[VerificationResult] = Field(...)
    total_questions: int = Field(...)
    aligned_count: int = Field(...)
    average_confidence: float = Field(...)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def success_rate(self) -> float:
        return self.aligned_count / self.total_questions if self.total_questions > 0 else 0.0
