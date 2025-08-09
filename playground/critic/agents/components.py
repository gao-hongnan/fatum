from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DraftResponse(BaseModel):
    reasoning: str = Field(..., description="Detailed reasoning for the alignment decision")
    is_aligned: bool = Field(..., description="Whether the answer is aligned with facts")


class SkepticQuestions(BaseModel):
    questions: list[str] = Field(
        ...,
        description="List of 3-6 yes/no questions that would help verify or disprove the draft answer",
        min_length=3,
        max_length=6,
    )


class FactCheckAnswer(BaseModel):
    answer: Literal["yes", "no"] = Field(..., description="Factual answer to the verification question")
    brief_explanation: str = Field(..., description="Brief explanation for the answer")


class JudgeVerdict(BaseModel):
    reasoning: str = Field(..., description="Reasoning for the final verdict based on verification results")
    is_aligned: bool = Field(..., description="Final verdict on whether the answer is aligned with facts")
    confidence: float = Field(..., description="Confidence level in the verdict (0.0-1.0)")
    revision_made: bool = Field(..., description="Whether the verdict differs from the initial assessment")
