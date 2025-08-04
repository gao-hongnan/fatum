"""
Structify Adapter Pattern Demo - Movie Review CLI

```bash
uv run playground/structify/01_demo.py --provider all

uv run playground/structify/01_demo.py --provider openai
uv run playground/structify/01_demo.py --provider anthropic
uv run playground/structify/01_demo.py --provider gemini

uv run playground/structify/01_demo.py --movie "The Matrix"
uv run playground/structify/01_demo.py --provider anthropic --movie "Blade Runner"

uv run playground/structify/01_demo.py --stream
uv run playground/structify/01_demo.py --provider openai --stream
uv run playground/structify/01_demo.py --provider gemini --movie "Dune" --stream

uv run playground/structify/01_demo.py --trace --provider all
uv run playground/structify/01_demo.py --provider openai --trace
uv run playground/structify/01_demo.py --provider openai --movie "Interstellar" --stream --trace
```
"""

from __future__ import annotations

import argparse
import asyncio
from pprint import pprint
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

from fatum.structify import create_adapter
from fatum.structify.adapters.anthropic import AnthropicAdapter
from fatum.structify.adapters.gemini import GeminiAdapter
from fatum.structify.adapters.openai import OpenAIAdapter
from fatum.structify.models import (
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
    model: str = Field(default="gemini-2.5-flash", exclude=True)
    temperature: float = Field(default=1.0)
    max_output_tokens: int = Field(default=1000)


class MovieReview(BaseModel):
    title: str
    rating: float = Field(ge=0, le=10)
    summary: str
    pros: list[str]
    cons: list[str]


def create_review_table(review: MovieReview, provider_name: str) -> Table:
    table = Table(title=f"🎬 {review.title} - via {provider_name}", show_header=True, header_style="bold magenta")

    table.add_column("Aspect", style="cyan", width=12)
    table.add_column("Details", style="white")

    table.add_row("Rating", f"⭐ {review.rating}/10")
    table.add_row("Summary", review.summary)
    table.add_row("Pros", "\n".join(f"✅ {pro}" for pro in review.pros))
    table.add_row("Cons", "\n".join(f"❌ {con}" for con in review.cons))

    return table


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


async def review_movie(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
    messages: list[ChatCompletionMessageParam],
    provider_name: str,
    show_trace: bool = False,
) -> MovieReview:
    console.print(Panel.fit(f"🤖 {provider_name} Example", style="bold blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description=f"Getting review from {provider_name}...", total=None)

        if show_trace:
            result = await adapter.acreate(
                messages=messages,
                response_model=MovieReview,
                with_hooks=True,
            )
            pprint(type(result.trace.raw_response))
            review = result.data
            display_trace_info(result)
        else:
            review = await adapter.acreate(
                messages=messages,
                response_model=MovieReview,
            )

    console.print(create_review_table(review, provider_name))
    return review


async def review_movie_streaming(
    adapter: OpenAIAdapter | AnthropicAdapter | GeminiAdapter,
    messages: list[ChatCompletionMessageParam],
    provider_name: str,
) -> MovieReview:
    console.print(Panel.fit(f"🤖 {provider_name} Streaming Example", style="bold blue"))
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
        console.print(create_review_table(final_review, provider_name))

    return final_review or MovieReview(title="Unknown", rating=0, summary="", pros=[], cons=[])


def create_demo_adapter(provider: str) -> OpenAIAdapter | AnthropicAdapter | GeminiAdapter:
    if provider == "openai":
        return create_adapter(
            provider_config=OpenAIProvider(api_key=settings.openai_api_key),
            completion_params=OpenAICompletion(),
            instructor_config=InstructorConfig(mode=instructor.Mode.TOOLS),
        )
    elif provider == "anthropic":
        return create_adapter(
            provider_config=AnthropicProvider(api_key=settings.anthropic_api_key),
            completion_params=AnthropicCompletion(),
            instructor_config=InstructorConfig(mode=instructor.Mode.ANTHROPIC_TOOLS),
        )
    elif provider == "gemini":
        return create_adapter(
            provider_config=GeminiProvider(api_key=settings.gemini_api_key),
            completion_params=GeminiCompletion(),
            instructor_config=InstructorConfig(mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS),
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Movie review with different LLM providers")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "all"],
        default="all",
        help="LLM provider to use (default: all)",
    )
    parser.add_argument("--movie", default="Inception", help="Movie to review (default: Inception)")
    parser.add_argument("--trace", action="store_true", help="Show trace information")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "🎬 [bold]Structify Demo[/bold] 🎬\n[dim]Unified interface for multiple LLM providers[/dim]",
            style="bold green",
        )
    )

    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionSystemMessageParam(role="system", content="You are a helpful movie critic."),
        ChatCompletionUserMessageParam(role="user", content=f"Review the movie '{args.movie}' for me."),
    ]

    providers = ["openai", "anthropic", "gemini"] if args.provider == "all" else [args.provider]

    for i, provider in enumerate(providers):
        adapter = create_demo_adapter(provider)

        if args.stream:
            await review_movie_streaming(adapter, messages, provider.title())
        else:
            await review_movie(adapter, messages, provider.title(), show_trace=args.trace)

        if i < len(providers) - 1:
            console.print("\n" + "=" * 50 + "\n")

    console.print("\n" + "=" * 50)
    console.print(Panel.fit("✅ [bold green]Demo Complete![/bold green]", style="green"))


if __name__ == "__main__":
    asyncio.run(main())
