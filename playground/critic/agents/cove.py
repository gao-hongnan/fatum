from __future__ import annotations

import asyncio
from typing import Any

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, computed_field

from fatum.structify import create_adapter
from fatum.structify.adapters.anthropic import AnthropicAdapter
from fatum.structify.adapters.gemini import GeminiAdapter
from fatum.structify.adapters.openai import OpenAIAdapter
from fatum.structify.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)

from ..config.settings import (
    CoVeVerifierConfig,
    PromptsConfig,
    ProviderAgnosticAgent,
)
from ..prompts import JinjaPromptRenderer, get_prompt_renderer
from .base import MaybeOrchestrator, UserQuery
from .components import (
    DraftResponse,
    FactCheckAnswer,
    JudgeVerdict,
    SkepticQuestions,
)


class CoVeCandidate(BaseModel):
    chain_of_thought: list[str]
    is_aligned: bool
    confidence: float = Field(..., ge=0.0, le=1.0)

    draft_answer: str = Field(...)
    verification_questions: list[str] = Field(...)
    verification_answers: list[str] = Field(...)
    revision_made: bool = Field(default=False)
    verdict: str = Field(...)

    @computed_field
    def reasoning(self) -> str:
        return "\n".join(self.chain_of_thought) if self.chain_of_thought else ""


class CoVeOrchestrator(MaybeOrchestrator[CoVeVerifierConfig, CoVeCandidate]):
    config: CoVeVerifierConfig
    _prompt_renderer: JinjaPromptRenderer

    def __init__(self, config: CoVeVerifierConfig):
        super().__init__(config=config)
        self._prompt_renderer = get_prompt_renderer(self.config.drafter.prompts.base_path)

    def _build_messages(
        self, prompts_config: PromptsConfig, context_variables: dict[str, Any]
    ) -> list[ChatCompletionMessageParam]:
        system_prompt = self._prompt_renderer.render(
            template_path=prompts_config.system_prompt_path,
            variables=prompts_config.system_context_variables,
        )

        user_variables = {
            **prompts_config.user_context_variables,
            **context_variables,
        }
        user_prompt = self._prompt_renderer.render(
            template_path=prompts_config.user_prompt_path,
            variables=user_variables,
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _create_adapter(self, agent_config: ProviderAgnosticAgent) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
        match (agent_config.provider_config, agent_config.completion_params):
            case (OpenAIProviderConfig() as pc, OpenAICompletionClientParams() as cp):
                return create_adapter(
                    provider_config=pc,
                    completion_params=cp,
                    instructor_config=agent_config.instructor_config,
                )
            case (AnthropicProviderConfig() as pc, AnthropicCompletionClientParams() as cp):
                return create_adapter(
                    provider_config=pc,
                    completion_params=cp,
                    instructor_config=agent_config.instructor_config,
                )
            case (GeminiProviderConfig() as pc, GeminiCompletionClientParams() as cp):
                return create_adapter(
                    provider_config=pc,
                    completion_params=cp,
                    instructor_config=agent_config.instructor_config,
                )
            case _:
                raise ValueError(f"Unsupported provider combination: {agent_config.provider_config.provider}")

    async def _run_drafter_phase(self, user_query: UserQuery) -> DraftResponse:
        messages = self._build_messages(self.config.drafter.prompts, user_query.model_dump())
        adapter = self._create_adapter(self.config.drafter)
        return await adapter.acreate(messages, DraftResponse)

    async def _run_skeptic_phase(self, draft: DraftResponse, user_query: UserQuery) -> SkepticQuestions:
        context_variables = {
            "draft_reasoning": draft.reasoning,
            "draft_is_aligned": draft.is_aligned,
            **user_query.model_dump(),
        }
        messages = self._build_messages(self.config.skeptic.prompts, context_variables)
        adapter = self._create_adapter(self.config.skeptic)
        return await adapter.acreate(messages, SkepticQuestions)

    async def _run_fact_checker_phase(
        self,
        questions: list[str],
        user_query: UserQuery,
    ) -> list[FactCheckAnswer]:
        async def _fact_check_question(question: str) -> FactCheckAnswer:
            context_variables = {
                "verification_question": question,
                **user_query.model_dump(),
            }
            messages = self._build_messages(self.config.fact_checker.prompts, context_variables)
            adapter = self._create_adapter(self.config.fact_checker)
            return await adapter.acreate(messages, FactCheckAnswer)

        tasks = [asyncio.create_task(_fact_check_question(question)) for question in questions]
        return await asyncio.gather(*tasks)

    async def _run_judge_phase(
        self,
        draft: DraftResponse,
        questions: list[str],
        fact_checks: list[FactCheckAnswer],
        user_query: UserQuery,
    ) -> JudgeVerdict:
        qa_pairs = list(zip(questions, fact_checks, strict=False))
        context_variables = {
            "draft_reasoning": draft.reasoning,
            "draft_is_aligned": draft.is_aligned,
            "qa_pairs": qa_pairs,
            **user_query.model_dump(),
        }
        messages = self._build_messages(self.config.judge.prompts, context_variables)
        adapter = self._create_adapter(self.config.judge)
        return await adapter.acreate(messages, JudgeVerdict)

    async def aexecute(self, user_query: UserQuery) -> CoVeCandidate:
        draft = await self._run_drafter_phase(user_query)
        questions_obj = await self._run_skeptic_phase(draft, user_query)
        fact_check_answers = await self._run_fact_checker_phase(questions_obj.questions, user_query)
        judge_verdict = await self._run_judge_phase(draft, questions_obj.questions, fact_check_answers, user_query)

        chain_of_thought = [
            f"[Drafter] {draft.reasoning}",
            f"[Skeptic] Generated {len(questions_obj.questions)} verification questions",
            *[
                f"[FactCheck] {q} -> {a.answer}: {a.brief_explanation}"
                for q, a in zip(questions_obj.questions, fact_check_answers, strict=False)
            ],
            f"[Judge] {judge_verdict.reasoning}",
        ]

        return CoVeCandidate(
            chain_of_thought=chain_of_thought,
            is_aligned=judge_verdict.is_aligned,
            confidence=judge_verdict.confidence,
            draft_answer=draft.reasoning,
            verification_questions=questions_obj.questions,
            verification_answers=[f"{fc.answer}: {fc.brief_explanation}" for fc in fact_check_answers],
            revision_made=judge_verdict.revision_made,
            verdict=f"{'Revised' if judge_verdict.revision_made else 'Confirmed'}: {'Aligned' if judge_verdict.is_aligned else 'Not aligned'}",
        )
