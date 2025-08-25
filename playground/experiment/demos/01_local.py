"""Local experiment tracking demo with clean API and minimal output.

This demo showcases the experiment tracking system with local storage,
demonstrating how to:
- Track single and multiple training runs
- Collect and display metrics in organized tables
- Save artifacts (models, configs, reports) with proper organization
- Access experiment paths and artifacts programmatically

Type Safety Note:
-----------------
This demo uses `exp.run()` instead of `experiment.run()` to preserve type information.

Why the difference?
- `exp.run()`: Returns Run[LocalStorage] - preserves the specific storage type
- `experiment.run()`: Returns Run[Storage] - loses type information due to context variable type erasure

When you use `experiment.run()` (the standalone function), it retrieves the experiment
from a context variable that has been type-erased to Experiment[Storage]. This means
the specific LocalStorage type is lost, and type checkers will complain when accessing
custom methods like save_dict(), log_metrics(), etc.

By using `exp.run()` directly on the experiment instance, we maintain full type safety
because the experiment knows its concrete storage type (LocalStorage).

Usage:
    # Run both demos (default)
    uv run -m experiment.demos.01_local

    # Run single experiment only
    uv run -m experiment.demos.01_local --mode single

    # Run hyperparameter search only
    uv run -m experiment.demos.01_local --mode multiple

    # Specify custom output directory
    uv run -m experiment.demos.01_local --output-dir ./my_experiments

Arguments:
    --mode: Choose between 'single', 'multiple', or 'both' (default: both)
    --output-dir: Output directory for experiments (default: ../outputs)

Output Structure:
    {output_dir}/
    ├── local/                          # Single run experiments
    │   └── {experiment_id}/
    │       ├── experiment.json
    │       └── runs/
    │           └── {run_id}/
    │               ├── config.json
    │               ├── metrics.json
    │               ├── reports/
    │               └── artifacts/
    └── local_multi/                    # Multi-run experiments
        └── {experiment_id}/
            ├── experiment.json
            └── runs/
                └── {run_id}/
                    ├── config.json
                    ├── metrics.json
                    ├── reports/
                    └── artifacts/
                └── {run_id}/
                    ├── config.json
                    ├── metrics.json
                    ├── reports/
                    └── artifacts/
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fatum import experiment
from fatum.experiment.experiment import Run
from fatum.experiment.types import MetricKey
from fatum.reproducibility.git import get_git_info

from ..backends.local import LocalStorage
from .utils import show_directory_tree, simulate_model_training

console = Console()


def _run_and_collect_metrics(run: Run[LocalStorage], config: dict[str, Any], epochs: int = 5) -> dict[str, Any]:
    """Run training and collect metrics without printing."""
    run.storage.save_dict(config, "config.json")

    metrics = simulate_model_training(epochs=epochs)
    for i, metric in enumerate(metrics):
        run.storage.log_metrics(metric, step=i)

    run.storage.save_text(
        f"# Training Report\n\nFinal accuracy: {metrics[-1]['accuracy']:.3f}\nFinal loss: {metrics[-1]['loss']:.3f}",
        "reports/training_report.md",
    )

    model_info = {
        "architecture": "ResNet50",
        "parameters": 25_557_032,
        "input_shape": [224, 224, 3],
        "output_classes": 1000,
    }
    run.storage.save_dict(model_info, "model/architecture.json")

    model_path = Path("dummy_model.txt")
    model_path.write_text("Pretrained model weights (simulated)")
    run.storage.save(model_path, path="model/weights.pkl")
    model_path.unlink()

    return {
        "config": config,
        "metrics": metrics,
        "final_loss": metrics[-1]["loss"],
        "final_accuracy": metrics[-1]["accuracy"],
    }


def run_single(output_dir: Path) -> None:
    storage_dir = output_dir / "local"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    commit_hash = get_git_info().short_commit
    experiment_name = f"model_training_{timestamp}_{commit_hash}"

    with experiment.experiment(
        name=experiment_name,
        storage=LocalStorage(storage_dir),
        id=experiment_name,
        description=f"Training a neural network model with {commit_hash}",
        tags=["demo", "local", "training"],
    ) as exp:
        with exp.run("main") as r:
            config = {
                "model": "resnet50",
                "dataset": "imagenet",
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "epochs": 5,
                "dropout": 0.2,
                "weight_decay": 0.0001,
            }
            results = _run_and_collect_metrics(r, config, epochs=5)

        table = Table(title=f"Single Run Results - {exp.id}")
        table.add_column("Epoch", style="cyan")
        table.add_column("Loss", style="yellow")
        table.add_column("Accuracy", style="green")

        for i, metric in enumerate(results["metrics"]):
            table.add_row(str(i), f"{metric['loss']:.4f}", f"{metric['accuracy']:.4f}")

        console.print(table)

        experiment_path = storage_dir / exp.id
        console.print(f"\n📁 Experiment saved to: [cyan]{experiment_path.resolve()}[/cyan]")
        console.print(f"   ID: [green]{exp.id}[/green]")


def run_multiple(output_dir: Path) -> None:
    """Demonstrate multiple runs - comparing different learning rates."""
    storage_dir = output_dir / "local_multi"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"hyperparameter_search_{timestamp}"

    run_results = []

    with experiment.experiment(
        name=experiment_name,
        storage=LocalStorage(storage_dir),
        id=experiment_name,
        description="Testing different learning rates",
        tags=["demo", "hyperparameter_search", "multiple_runs"],
    ) as exp:
        learning_rates = [0.001, 0.01]

        for lr in learning_rates:
            with exp.run(f"lr_{lr}") as r:
                config = {
                    "model": "resnet50",
                    "dataset": "imagenet",
                    "batch_size": 32,
                    "learning_rate": lr,
                    "optimizer": "adam",
                    "epochs": 3,
                }
                results = _run_and_collect_metrics(r, config, epochs=3)
                results["run_id"] = r.id
                run_results.append(results)

                final_metrics = results["metrics"][-1]
                summary = f"Learning Rate: {lr}\nFinal Loss: {final_metrics['loss']:.3f}\nFinal Accuracy: {final_metrics['accuracy']:.3f}"
                r.storage.save_text(summary, "summary.txt")

                model_path = Path(f"model_lr_{lr}.txt")
                model_path.write_text(f"Model weights trained with lr={lr}")
                r.storage.save(model_path, path="model/weights.pkl")
                model_path.unlink()

        table = Table(title=f"Hyperparameter Search Results - {exp.id}")
        table.add_column("Run ID", style="cyan")
        table.add_column("Learning Rate", style="yellow")
        table.add_column("Final Loss", style="magenta")
        table.add_column("Final Accuracy", style="green")

        for result in run_results:
            table.add_row(
                result["run_id"],
                str(result["config"]["learning_rate"]),
                f"{result['final_loss']:.4f}",
                f"{result['final_accuracy']:.4f}",
            )

        console.print(table)

        experiment_path = storage_dir / exp.id

        console.print(f"\n📁 Experiment saved to: [cyan]{experiment_path.resolve()}[/cyan]")
        console.print(f"   ID: [green]{exp.id}[/green]")


def run_flexible_containers(output_dir: Path) -> None:
    """Demonstrate flexible run container configuration."""
    storage_dir = output_dir / "flexible_containers"

    console.print("\n[yellow]Demo 1: Flat structure (no runs/ folder)[/yellow]")
    storage = LocalStorage(storage_dir / "flat")
    with experiment.experiment(
        name="ml_pipeline",
        storage=storage,
        description="ML pipeline with flat structure",
    ) as exp:
        stages = ["preprocessing", "training", "evaluation"]
        for stage in stages:
            with exp.run(stage) as r:
                r.storage.log_metric(MetricKey("duration_seconds"), random.uniform(10, 100))
                r.storage.log_metric(MetricKey("records_processed"), random.randint(1000, 10000))
                console.print(f"  ✓ {stage}: {r.id}")

    console.print("\n[yellow]Demo 2: Custom container (models/)[/yellow]")
    storage = LocalStorage(storage_dir / "custom")
    with experiment.experiment(
        name="model_iterations",
        storage=storage,
        description="Model iterations under models/ folder",
    ) as exp:
        models = ["baseline", "optimized", "final"]
        for model_name in models:
            with exp.run(model_name) as r:
                r.storage.log_metric(MetricKey("f1_score"), random.uniform(0.7, 0.95))
                r.storage.log_metric(MetricKey("latency_ms"), random.uniform(10, 50))
                console.print(f"  ✓ {model_name}: {r.id}")

    console.print("\n[yellow]Demo 3: Data pipeline stages[/yellow]")
    storage = LocalStorage(storage_dir / "pipeline")
    with experiment.experiment(
        name="etl_pipeline",
        storage=storage,
        description="ETL pipeline with stage-based organization",
    ) as exp:
        pipeline_stages = [
            ("extract", {"sources": 5, "rows": 50000}),
            ("transform", {"transformations": 12, "rows": 48000}),
            ("load", {"tables": 3, "rows": 48000}),
        ]
        for stage_name, metrics in pipeline_stages:
            with exp.run(stage_name) as r:
                for key, value in metrics.items():
                    r.storage.log_metric(MetricKey(key), value)
                console.print(f"  ✓ {stage_name}: {r.id}")

    console.print(f"\n📁 All experiments saved to: [cyan]{storage_dir.resolve()}[/cyan]")


def run() -> None:
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Local experiment tracking demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multiple", "flexible", "all"],
        default="all",
        help="Which demo to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../outputs"),
        help="Output directory for experiments (default: ../outputs)",
    )

    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold cyan]Experiment Tracking Demo[/bold cyan]\nLocal storage with clean API and table-based output",
            border_style="cyan",
        )
    )

    if args.mode in ["single", "all"]:
        console.print("\n[bold]Single Run Demo[/bold]")
        run_single(args.output_dir)

    if args.mode in ["multiple", "all"]:
        console.print("\n[bold]Multiple Runs Demo[/bold]")
        run_multiple(args.output_dir)

    if args.mode in ["flexible", "all"]:
        console.print("\n[bold]Flexible Run Containers Demo[/bold]")
        run_flexible_containers(args.output_dir)

    if args.mode in ["all"]:
        console.print("\n[bold]Directory Structure:[/bold]")
        for subdir in ["local", "local_multi", "flexible_containers"]:
            path = args.output_dir / subdir
            if path.exists():
                show_directory_tree(path, max_depth=2)

    console.print("\n✨ Demo complete!")


if __name__ == "__main__":
    run()
