"""
```bash
uv run playground/structify/02_conversation_demo.py --stream --provider openai
```
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Literal

import instructor
from instructor.dsl.partial import PartialLiteralMixin
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fatum.structify import create_adapter
from fatum.structify.adapters.anthropic import AnthropicAdapter
from fatum.structify.adapters.gemini import GeminiAdapter
from fatum.structify.adapters.openai import OpenAIAdapter
from fatum.structify.models import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    InstructorConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)

console = Console()


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI__API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC__API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI__API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


class IssueAnalysis(BaseModel, PartialLiteralMixin):
    problem_category: Literal["hardware", "software", "network", "performance", "other"]
    severity: Literal["low", "medium", "high", "critical"]
    symptoms: list[str] = Field(description="List of symptoms reported by user")
    possible_causes: list[str] = Field(description="Potential root causes")
    confidence: float = Field(ge=0, le=1, description="Confidence in diagnosis")


class Solution(BaseModel, PartialLiteralMixin):
    steps: list[str] = Field(description="Step-by-step solution")
    estimated_time: str = Field(description="Estimated time to complete")
    difficulty: Literal["easy", "moderate", "advanced"]
    requires_restart: bool = Field(default=False)
    success_indicators: list[str] = Field(description="How to verify the fix worked")


class ConversationResponse(BaseModel):
    message: str = Field(description="Response to the user")
    needs_more_info: bool = Field(default=False, description="Whether more information is needed")
    follow_up_questions: list[str] = Field(default_factory=list, description="Questions to ask if needed")


def create_provider_adapter(provider: str) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not set")
        return create_adapter(
            provider_config=OpenAIProviderConfig(api_key=settings.openai_api_key),
            completion_params=OpenAICompletionClientParams(
                model="gpt-4o-mini",
            ),
            instructor_config=InstructorConfig(mode=instructor.Mode.TOOLS_STRICT),
        )
    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not set")
        return create_adapter(
            provider_config=AnthropicProviderConfig(api_key=settings.anthropic_api_key),
            completion_params=AnthropicCompletionClientParams(
                model="claude-3-5-haiku-20241022",
            ),
            instructor_config=InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS),
        )
    elif provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("Gemini API key not set")
        return create_adapter(
            provider_config=GeminiProviderConfig(api_key=settings.gemini_api_key),
            completion_params=GeminiCompletionClientParams(
                model="gemini-2.5-flash",
            ),
            instructor_config=InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def format_streaming_response(partial: ConversationResponse) -> Text:
    text = Text()

    if hasattr(partial, "message") and partial.message:
        text.append(partial.message, style="green")

    if (
        hasattr(partial, "needs_more_info")
        and partial.needs_more_info
        and hasattr(partial, "follow_up_questions")
        and partial.follow_up_questions
    ):
        text.append("\n\n[dim]Follow-up questions:[/dim]\n")
        for q in partial.follow_up_questions:
            text.append(f"  • {q}\n", style="dim")

    return text


def format_streaming_analysis(partial: IssueAnalysis) -> Text:
    text = Text()

    text.append("📊 Issue Analysis\n\n", style="bold yellow")

    if hasattr(partial, "problem_category") and partial.problem_category:
        text.append("Category: ", style="cyan")
        text.append(f"{partial.problem_category}\n", style="white")

    if hasattr(partial, "severity") and partial.severity:
        text.append("Severity: ", style="cyan")
        text.append(f"{partial.severity}\n", style="white")

    if hasattr(partial, "confidence") and partial.confidence and partial.confidence > 0:
        text.append("Confidence: ", style="cyan")
        text.append(f"{partial.confidence:.0%}\n", style="white")

    if hasattr(partial, "symptoms") and partial.symptoms:
        text.append("\nSymptoms:\n", style="cyan")
        for s in partial.symptoms:
            text.append(f"  • {s}\n", style="white")

    if hasattr(partial, "possible_causes") and partial.possible_causes:
        text.append("\nPossible Causes:\n", style="cyan")
        for c in partial.possible_causes:
            text.append(f"  • {c}\n", style="white")

    return text


def format_streaming_solution(partial: Solution) -> Text:
    text = Text()

    text.append("💡 Recommended Solution\n\n", style="bold green")

    if hasattr(partial, "difficulty") and partial.difficulty:
        text.append("Difficulty: ", style="bold")
        text.append(f"{partial.difficulty}\n", style="white")

    if hasattr(partial, "estimated_time") and partial.estimated_time:
        text.append("Estimated Time: ", style="bold")
        text.append(f"{partial.estimated_time}\n", style="white")

    if hasattr(partial, "requires_restart") and hasattr(partial.requires_restart, "__bool__"):
        text.append("Requires Restart: ", style="bold")
        text.append(f"{'Yes' if partial.requires_restart else 'No'}\n", style="white")

    if hasattr(partial, "steps") and partial.steps:
        text.append("\nSteps to follow:\n", style="bold")
        for i, step in enumerate(partial.steps, 1):
            text.append(f"  {i}. {step}\n", style="white")

    if hasattr(partial, "success_indicators") and partial.success_indicators:
        text.append("\nSuccess Indicators:\n", style="bold")
        for indicator in partial.success_indicators:
            text.append(f"  ✓ {indicator}\n", style="white")

    return text


async def stream_conversation_response(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
    messages: list[ChatCompletionMessageParam],
) -> ConversationResponse:
    partial_count = 0
    final_response = None

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        async for partial in adapter.astream(
            messages=messages,
            response_model=ConversationResponse,
        ):
            partial_count += 1
            final_response = partial

            formatted = format_streaming_response(partial)
            live.update(formatted)

            await asyncio.sleep(0.01)

    return final_response or ConversationResponse(message="", needs_more_info=False)


async def stream_issue_analysis(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
    messages: list[ChatCompletionMessageParam],
) -> IssueAnalysis:
    console.print("\n[yellow]📊 Analyzing issue...[/yellow]")

    partial_count = 0
    final_analysis = None

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        async for partial in adapter.astream(
            messages=messages,
            response_model=IssueAnalysis,
        ):
            partial_count += 1
            final_analysis = partial

            formatted = format_streaming_analysis(partial)
            live.update(formatted)

            await asyncio.sleep(0.01)

    return final_analysis or IssueAnalysis(
        problem_category="other",
        severity="low",
        symptoms=[],
        possible_causes=[],
        confidence=0.0,
    )


async def stream_solution(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
    messages: list[ChatCompletionMessageParam],
) -> Solution:
    console.print("\n[yellow]💡 Generating solution...[/yellow]")

    partial_count = 0
    final_solution = None

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        async for partial in adapter.astream(
            messages=messages,
            response_model=Solution,
        ):
            partial_count += 1
            final_solution = partial

            formatted = format_streaming_solution(partial)
            live.update(formatted)

            await asyncio.sleep(0.01)

    console.print(f"\n[green]✓ Streaming complete! Received {partial_count} partial updates[/green]")

    return final_solution or Solution(
        steps=[],
        estimated_time="Unknown",
        difficulty="easy",
        requires_restart=False,
        success_indicators=[],
    )


async def tech_support_conversation(provider: str, stream_mode: bool = False) -> None:
    console.print(
        Panel.fit(
            f"🔧 [bold cyan]Tech Support Assistant[/bold cyan] - {provider.upper()}\n"
            "[dim]I'll help you diagnose and solve computer issues[/dim]",
            border_style="cyan",
        )
    )

    adapter = create_provider_adapter(provider)

    conversation_history: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "You are a helpful tech support assistant. "
                "Help users diagnose and solve their computer issues. "
                "Remember the conversation context and build upon previous messages. "
                "Be patient and ask clarifying questions when needed."
            ),
        )
    ]

    user_messages = [
        "My computer has been running really slow lately",
        "It started about a week ago. I notice it especially when opening programs",
        "Yes, the fans are running loudly and it feels warm",
        "I haven't cleaned it in about 6 months",
    ]

    issue_analysis = None

    for i, user_message in enumerate(user_messages, 1):
        console.print(f"\n[cyan]User:[/cyan] {user_message}")

        conversation_history.append(ChatCompletionUserMessageParam(role="user", content=user_message))

        if stream_mode:
            console.print("[green]Assistant:[/green] ", end="")
            response = await stream_conversation_response(adapter, conversation_history)
        else:
            response = await adapter.acreate(
                messages=conversation_history,
                response_model=ConversationResponse,
            )
            console.print(f"[green]Assistant:[/green] {response.message}")

        conversation_history.append(ChatCompletionAssistantMessageParam(role="assistant", content=response.message))

        if i == 2:
            if stream_mode:
                issue_analysis = await stream_issue_analysis(
                    adapter,
                    conversation_history
                    + [
                        ChatCompletionSystemMessageParam(
                            role="system", content="Analyze the technical issue based on the conversation so far."
                        )
                    ],
                )
            else:
                console.print("\n[yellow]📊 Analyzing issue...[/yellow]")
                issue_analysis = await adapter.acreate(
                    messages=conversation_history
                    + [
                        ChatCompletionSystemMessageParam(
                            role="system", content="Analyze the technical issue based on the conversation so far."
                        )
                    ],
                    response_model=IssueAnalysis,
                )

                table = Table(title="Issue Analysis", show_header=False)
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="white")

                table.add_row("Category", issue_analysis.problem_category)
                table.add_row("Severity", issue_analysis.severity)
                table.add_row("Confidence", f"{issue_analysis.confidence:.0%}")
                table.add_row("Symptoms", "\n".join(f"• {s}" for s in issue_analysis.symptoms))
                table.add_row("Possible Causes", "\n".join(f"• {c}" for c in issue_analysis.possible_causes))

                console.print(table)

        if not stream_mode and response.needs_more_info and response.follow_up_questions:
            console.print("\n[dim]Follow-up questions:[/dim]")
            for q in response.follow_up_questions:
                console.print(f"  [dim]• {q}[/dim]")

    if stream_mode:
        solution = await stream_solution(
            adapter,
            conversation_history
            + [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="Based on the conversation, provide a detailed solution to fix the user's computer performance issue.",
                )
            ],
        )
    else:
        console.print("\n[yellow]💡 Generating solution...[/yellow]")

        solution = await adapter.acreate(
            messages=conversation_history
            + [
                ChatCompletionSystemMessageParam(
                    role="system",
                    content="Based on the conversation, provide a detailed solution to fix the user's computer performance issue.",
                )
            ],
            response_model=Solution,
        )

        console.print(Panel.fit("[bold]Recommended Solution[/bold]", style="green"))

        console.print(f"[bold]Difficulty:[/bold] {solution.difficulty}")
        console.print(f"[bold]Estimated Time:[/bold] {solution.estimated_time}")
        console.print(f"[bold]Requires Restart:[/bold] {'Yes' if solution.requires_restart else 'No'}")

        console.print("\n[bold]Steps to follow:[/bold]")
        for i, step in enumerate(solution.steps, 1):
            console.print(f"  {i}. {step}")

        console.print("\n[bold]Success Indicators:[/bold]")
        for indicator in solution.success_indicators:
            console.print(f"  ✓ {indicator}")

    console.print("\n" + "=" * 50)
    console.print(
        Panel.fit(
            f"[green]Conversation Complete![/green]\n"
            f"Messages exchanged: {len(user_messages)}\n"
            f"Issue identified: {issue_analysis.problem_category if issue_analysis else 'N/A'}\n"
            f"Solution provided: ✓",
            style="green",
        )
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Structify conversation demo")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini"],
        default="openai",
        help="LLM provider to use",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming mode for responses",
    )

    args = parser.parse_args()

    try:
        await tech_support_conversation(args.provider, stream_mode=args.stream)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[dim]Please set the appropriate API key in your environment[/dim]")
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")


if __name__ == "__main__":
    asyncio.run(main())
