from __future__ import annotations

import asyncio
from typing import Any

import instructor
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from fatum.structify import AdapterFactory
from fatum.structify.adapters.anthropic import AnthropicAdapter
from fatum.structify.adapters.gemini import GeminiAdapter
from fatum.structify.adapters.openai import OpenAIAdapter
from fatum.structify.config import (
    AnthropicCompletionClientParams,
    AnthropicProviderConfig,
    CompletionResult,
    GeminiCompletionClientParams,
    GeminiProviderConfig,
    InstructorConfig,
    OpenAICompletionClientParams,
    OpenAIProviderConfig,
)

console = Console()


class Settings(BaseSettings):
    """Load settings from environment variables."""

    openai_api_key: str = Field(default="", alias="OPENAI__API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC__API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI__API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


class OpenAIProvider(OpenAIProviderConfig):
    api_key: str


class OpenAICompletion(OpenAICompletionClientParams):
    model: str = Field(default="gpt-4o-mini")
    temperature: float = Field(default=0.7)
    max_completion_tokens: int = Field(default=1000)


class AnthropicProvider(AnthropicProviderConfig):
    api_key: str


class AnthropicCompletion(AnthropicCompletionClientParams):
    model: str = Field(default="claude-3-5-haiku-20241022")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1000)


class GeminiProvider(GeminiProviderConfig):
    api_key: str


class GeminiCompletion(GeminiCompletionClientParams):
    model: str = Field(default="gemini-2.0-flash", exclude=True)
    temperature: float = Field(default=0.7)
    max_output_tokens: int = Field(default=1000)


class MovieReview(BaseModel):
    title: str
    rating: float = Field(ge=0, le=10)
    summary: str
    pros: list[str]
    cons: list[str]


def create_review_table(review: MovieReview) -> Table:
    table = Table(title=f"🎬 {review.title}", show_header=True, header_style="bold magenta")

    table.add_column("Aspect", style="cyan", width=12)
    table.add_column("Details", style="white")

    table.add_row("Rating", f"⭐ {review.rating}/10")
    table.add_row("Summary", review.summary)
    table.add_row("Pros", "\n".join(f"✅ {pro}" for pro in review.pros))
    table.add_row("Cons", "\n".join(f"❌ {con}" for con in review.cons))

    return table


def display_trace_info(result: CompletionResult[Any, Any]) -> None:
    json_display = Syntax(
        code=result.trace.model_dump_json(indent=4, fallback=lambda x: str(x)),
        lexer="json",
        theme="monokai",
        line_numbers=False,
        word_wrap=True,
    )

    console.print(
        Panel(
            json_display,
            title="📊 [bold cyan]Trace Information[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
            expand=False,
        )
    )


def format_streaming_text(partial: MovieReview) -> Text:
    text = Text()

    if hasattr(partial, "title") and partial.title:
        text.append(f"🎬 {partial.title}\n", style="bold cyan")

    if hasattr(partial, "rating") and partial.rating and partial.rating > 0:
        text.append(f"⭐ {partial.rating}/10\n", style="yellow")

    if text.plain:
        text.append("\n")

    if hasattr(partial, "summary") and partial.summary:
        text.append("Summary: ", style="bold magenta")
        text.append(f"{partial.summary}\n", style="white")
        text.append("\n")

    if hasattr(partial, "pros") and partial.pros:
        text.append("✅ Pros:\n", style="bold green")
        for pro in partial.pros:
            text.append(f"   • {pro}\n", style="green")
        text.append("\n")

    if hasattr(partial, "cons") and partial.cons:
        text.append("❌ Cons:\n", style="bold red")
        for con in partial.cons:
            text.append(f"   • {con}\n", style="red")

    return text


async def openai_example() -> tuple[OpenAIAdapter, MovieReview]:
    console.print(Panel.fit("🤖 OpenAI Example", style="bold blue"))

    provider = OpenAIProvider(api_key=settings.openai_api_key)
    completion = OpenAICompletion()
    instructor_config = InstructorConfig(mode=instructor.Mode.TOOLS)

    adapter = AdapterFactory.create(
        provider_config=provider,
        completion_params=completion,
        instructor_config=instructor_config,
    )

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful movie critic."),
        ChatCompletionUserMessageParam(role="user", content="Review the movie 'Inception' for me."),
    ]

    console.print("\n[cyan]1. Regular completion:[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Getting review...", total=None)
        review: MovieReview = await adapter.acreate(
            messages=messages,
            response_model=MovieReview,
        )

    console.print(create_review_table(review))

    console.print("\n[cyan]2. Regular completion with hooks:[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Getting review with trace...", total=None)
        result = await adapter.acreate(
            messages=messages,
            response_model=MovieReview,
            with_hooks=True,
        )

    console.print(create_review_table(result.data))
    display_trace_info(result)

    console.print("\n[cyan]3. Streaming completion:[/cyan]")
    console.print("[dim]Streaming updates...[/dim]\n")

    partial_count = 0
    final_review = None

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        async for partial_review in adapter.astream(
            messages=messages,
            response_model=MovieReview,
        ):
            partial_count += 1
            final_review = partial_review

            formatted = format_streaming_text(partial_review)
            live.update(formatted)

            await asyncio.sleep(0.02)

    console.print(f"\n[green]✓ Streaming complete! Received {partial_count} partial updates[/green]")
    if final_review:
        console.print("\n[bold]Final Result:[/bold]")
        console.print(create_review_table(final_review))

    return adapter, review


async def anthropic_example() -> tuple[AnthropicAdapter, MovieReview]:
    console.print(Panel.fit("🧠 Anthropic Example", style="bold magenta"))

    provider = AnthropicProvider(api_key=settings.anthropic_api_key)
    completion = AnthropicCompletion()
    instructor_config = InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS)

    adapter = AdapterFactory.create(
        provider_config=provider,
        completion_params=completion,
        instructor_config=instructor_config,
    )

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful movie critic."),
        ChatCompletionUserMessageParam(role="user", content="Review the movie 'The Matrix' for me."),
    ]

    console.print("\n[cyan]1. Regular completion:[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Getting review...", total=None)
        review: MovieReview = await adapter.acreate(
            messages=messages,
            response_model=MovieReview,
        )

    console.print(create_review_table(review))

    console.print("\n[cyan]2. Regular completion with hooks:[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Getting review with trace...", total=None)
        result = await adapter.acreate(
            messages=messages,
            response_model=MovieReview,
            with_hooks=True,
        )

    console.print(create_review_table(result.data))
    display_trace_info(result)

    console.print("\n[cyan]3. Streaming completion:[/cyan]")
    console.print("[dim]Streaming updates...[/dim]\n")

    partial_count = 0
    final_review = None

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        async for partial_review in adapter.astream(
            messages=messages,
            response_model=MovieReview,
        ):
            partial_count += 1
            final_review = partial_review

            formatted = format_streaming_text(partial_review)
            live.update(formatted)

            await asyncio.sleep(0.02)

    console.print(f"\n[green]✓ Streaming complete! Received {partial_count} partial updates[/green]")
    if final_review:
        console.print("\n[bold]Final Result:[/bold]")
        console.print(create_review_table(final_review))

    return adapter, review


async def gemini_example() -> tuple[GeminiAdapter, MovieReview]:
    console.print(Panel.fit("✨ Gemini Example", style="bold yellow"))

    provider = GeminiProvider(api_key=settings.gemini_api_key)
    completion = GeminiCompletion()
    instructor_config = InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS)

    adapter = AdapterFactory.create(
        provider_config=provider,
        completion_params=completion,
        instructor_config=instructor_config,
    )

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful movie critic."),
        ChatCompletionUserMessageParam(role="user", content="Review the movie 'John Wick' for me."),
    ]

    console.print("\n[cyan]1. Regular completion:[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Getting review...", total=None)
        review: MovieReview = await adapter.acreate(
            messages=messages,
            response_model=MovieReview,
        )

    console.print(create_review_table(review))

    console.print("\n[cyan]2. Regular completion with hooks:[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Getting review with trace...", total=None)
        result = await adapter.acreate(
            messages=messages,
            response_model=MovieReview,
            with_hooks=True,
        )

    console.print(create_review_table(result.data))
    display_trace_info(result)

    console.print("\n[cyan]3. Streaming completion:[/cyan]")
    console.print("[dim]Streaming updates...[/dim]\n")

    partial_count = 0
    final_review = None

    with Live(console=console, refresh_per_second=30, transient=False) as live:
        async for partial_review in adapter.astream(
            messages=messages,
            response_model=MovieReview,
        ):
            partial_count += 1
            final_review = partial_review

            formatted = format_streaming_text(partial_review)
            live.update(formatted)

            await asyncio.sleep(0.02)

    console.print(f"\n[green]✓ Streaming complete! Received {partial_count} partial updates[/green]")
    if final_review:
        console.print("\n[bold]Final Result:[/bold]")
        console.print(create_review_table(final_review))

    return adapter, review


async def main() -> None:
    console.print(
        Panel.fit(
            "🎬 [bold]Structify Demo[/bold] 🎬\n[dim]Demonstrating structured output with multiple LLM providers[/dim]",
            style="bold green",
        )
    )

    examples = [
        ("OpenAI", openai_example),
        ("Anthropic", anthropic_example),
        ("Gemini", gemini_example),
    ]

    for _, example_func in examples:
        await example_func()

    console.print("\n" + "=" * 50)
    console.print(Panel.fit("✅ [bold green]Demo Complete![/bold green]", style="green"))


if __name__ == "__main__":
    asyncio.run(main())
