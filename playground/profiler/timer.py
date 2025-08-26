from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Callable

import numpy as np
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from fatum.profiler.timer import Timer, timer

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

console = Console()


def section(title: str) -> None:
    """Print a section header."""
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold yellow]{title}[/bold yellow]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]\n")


def demo_basic_timer() -> None:
    """Demonstrate basic Timer usage."""
    section("1. Basic Timer Context Manager")

    console.print("[green]Standard usage with logging:[/green]")
    with Timer(name="Basic operation"):
        time.sleep(0.5)
        console.print("  Doing some work...")

    console.print("\n[green]Silent mode - capture time without logging:[/green]")
    with Timer(name="Silent operation", silent=True) as t:
        time.sleep(0.3)
        console.print("  Working silently...")
    console.print(f"  [yellow]Captured time: {t.elapsed_seconds:.4f} seconds[/yellow]")


def demo_nested_timers() -> None:
    """Demonstrate nested timer usage."""
    section("2. Nested Timers")

    with Timer(name="Outer operation"):
        console.print("Starting outer operation...")
        time.sleep(0.2)

        with Timer(name="  Inner operation 1"):
            console.print("  Processing batch 1...")
            time.sleep(0.15)

        with Timer(name="  Inner operation 2"):
            console.print("  Processing batch 2...")
            time.sleep(0.1)

        console.print("Finishing outer operation...")
        time.sleep(0.1)


@timer(name="fibonacci_recursive")
def fibonacci_recursive(n: int) -> int:
    """Calculate Fibonacci recursively (inefficient for demo)."""
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


@timer(name="fibonacci_iterative", silent=True)
def fibonacci_iterative(n: int) -> int:
    """Calculate Fibonacci iteratively (efficient)."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def demo_sync_decorator() -> None:
    """Demonstrate @timer decorator on sync functions."""
    section("3. Sync Function Decorator")

    console.print("[green]Decorated recursive function (with logging):[/green]")
    result = fibonacci_recursive(10)
    console.print(f"  Result: {result}")

    console.print("\n[green]Decorated iterative function (silent mode):[/green]")
    result = fibonacci_iterative(10)
    console.print(f"  Result: {result}")
    console.print("  [dim](No timing log due to silent mode)[/dim]")


@timer(name="async_fetch_data")
async def fetch_data(delay: float) -> dict[str, Any]:
    """Simulate async data fetching."""
    await asyncio.sleep(delay)
    return {"data": f"Fetched after {delay}s", "items": random.randint(10, 100)}


@timer(name="async_process_data")
async def process_data(data: dict[str, Any]) -> str:
    """Simulate async data processing."""
    await asyncio.sleep(0.1)
    return f"Processed {data['items']} items"


async def demo_async_decorator() -> None:
    """Demonstrate @timer decorator on async functions."""
    section("4. Async Function Decorator")

    console.print("[green]Decorated async functions:[/green]")

    data = await fetch_data(0.3)
    console.print(f"  Fetched: {data}")

    result = await process_data(data)
    console.print(f"  {result}")

    console.print("\n[green]Concurrent async operations:[/green]")
    tasks = [fetch_data(random.uniform(0.1, 0.3)) for _ in range(3)]
    results = await asyncio.gather(*tasks)
    console.print(f"  Fetched {len(results)} results concurrently")


def demo_performance_comparison() -> None:
    """Compare performance of different approaches."""
    section("5. Performance Comparison")

    approaches: dict[str, Callable[[], Any] | None] = {
        "List Comprehension": lambda: [x**2 for x in range(10000)],
        "Map Function": lambda: [x**2 for x in range(10000)],
        "For Loop": lambda: [i**2 for i in range(10000)],
        "NumPy (if available)": None,
    }

    approaches["NumPy (if available)"] = lambda: np.arange(10000) ** 2

    table = Table(title="Performance Comparison")
    table.add_column("Approach", style="cyan")
    table.add_column("Time (seconds)", style="yellow")
    table.add_column("Relative", style="green")

    times: list[tuple[str, float]] = []

    for name, func in approaches.items():
        if func is None:
            continue

        with Timer(silent=True) as t:
            for _ in range(100):
                _ = func()

        times.append((name, t.elapsed_seconds))

    min_time = min(t for _, t in times)

    for name, elapsed in sorted(times, key=lambda x: x[1]):
        relative = f"{elapsed / min_time:.2f}x"
        table.add_row(name, f"{elapsed:.6f}", relative)

    console.print(table)


def demo_practical_example() -> None:
    """Demonstrate practical usage scenarios."""
    section("6. Practical Example: Data Processing Pipeline")

    @timer(name="load_data")
    def load_data() -> list[int]:
        """Simulate loading data."""
        time.sleep(0.2)
        return list(range(1000))

    @timer(name="transform_data")
    def transform_data(data: list[int]) -> list[int]:
        """Simulate data transformation."""
        time.sleep(0.15)
        return [x * 2 for x in data]

    @timer(name="validate_data")
    def validate_data(data: list[int]) -> bool:
        """Simulate data validation."""
        time.sleep(0.1)
        return all(x % 2 == 0 for x in data)

    @timer(name="save_data", silent=True)
    def save_data(data: list[int]) -> int:
        """Simulate saving data."""
        time.sleep(0.25)
        return len(data)

    with (
        Timer(name="Complete Pipeline"),
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress,
    ):
        task = progress.add_task("Loading data...", total=None)
        data = load_data()

        progress.update(task, description="Transforming data...")
        transformed = transform_data(data)

        progress.update(task, description="Validating data...")
        is_valid = validate_data(transformed)

        if is_valid:
            progress.update(task, description="Saving data...")
            count = save_data(transformed)
            console.print(f"  [green]✓ Saved {count} records[/green]")
        else:
            console.print("  [red]✗ Validation failed[/red]")


def demo_collecting_metrics() -> None:
    """Demonstrate collecting timing metrics."""
    section("7. Collecting Metrics")

    console.print("[green]Running multiple iterations and collecting metrics:[/green]\n")

    iterations = 10
    times: list[float] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Running {iterations} iterations...", total=None)

        for i in range(iterations):
            progress.update(task, description=f"Iteration {i + 1}/{iterations}...")

            with Timer(silent=True) as t:
                time.sleep(random.uniform(0.05, 0.15))
                _ = sum(range(10000))

            times.append(t.elapsed_seconds)

    stats = Panel(
        f"[cyan]Iterations:[/cyan] {iterations}\n"
        f"[cyan]Min time:[/cyan] {min(times):.4f}s\n"
        f"[cyan]Max time:[/cyan] {max(times):.4f}s\n"
        f"[cyan]Avg time:[/cyan] {sum(times) / len(times):.4f}s\n"
        f"[cyan]Total time:[/cyan] {sum(times):.4f}s",
        title="[bold yellow]Timing Statistics[/bold yellow]",
        border_style="green",
    )
    console.print(stats)


async def main() -> None:
    """Run all demos."""
    console.print(
        Panel(
            "[bold cyan]Timer Demo Suite[/bold cyan]\nDemonstrating the Timer context manager and @timer decorator",
            border_style="blue",
        )
    )

    demo_basic_timer()
    demo_nested_timers()
    demo_sync_decorator()
    await demo_async_decorator()
    demo_performance_comparison()
    demo_practical_example()
    demo_collecting_metrics()

    console.print(
        Panel(
            "[bold green]✓ Demo Complete![/bold green]\n"
            "The Timer class provides a simple, flexible way to measure execution time.",
            border_style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
