"""Local experiment tracking demo with clean API and minimal output.

This demo showcases the experiment tracking system with local storage,
demonstrating how to:
- Track single and multiple training runs
- Collect and display metrics in organized tables
- Save artifacts (models, configs, reports) with proper organization
- Access experiment paths and artifacts programmatically

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
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fatum import experiment
from fatum.experiment.experiment import Run
from fatum.reproducibility.git import get_git_info

from .utils import show_directory_tree, simulate_model_training

console = Console()


def _run_and_collect_metrics(run: Run, config: dict[str, Any], epochs: int = 5) -> dict[str, Any]:
    """Run training and collect metrics without printing."""
    run.save_dict(config, "config.json")

    metrics = simulate_model_training(epochs=epochs)
    for i, metric in enumerate(metrics):
        run.log_metrics(metric, step=i)

    run.save_text(
        f"# Training Report\n\nFinal accuracy: {metrics[-1]['accuracy']:.3f}\nFinal loss: {metrics[-1]['loss']:.3f}",
        "reports/training_report.md",
    )

    model_info = {
        "architecture": "ResNet50",
        "parameters": 25_557_032,
        "input_shape": [224, 224, 3],
        "output_classes": 1000,
    }
    run.save_dict(model_info, "model/architecture.json")

    model_path = Path("dummy_model.txt")
    model_path.write_text("Pretrained model weights (simulated)")
    run.save(model_path, path="model/weights.pkl")
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
        id=experiment_name,
        base_path=storage_dir,
        description=f"Training a neural network model with {commit_hash}",
        tags=["demo", "local", "training"],
    ) as exp:
        with experiment.run("main") as r:
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
        id=experiment_name,
        base_path=storage_dir,
        description="Testing different learning rates",
        tags=["demo", "hyperparameter_search", "multiple_runs"],
    ) as exp:
        learning_rates = [0.001, 0.01]

        for lr in learning_rates:
            with experiment.run(f"lr_{lr}", tags=[f"lr={lr}"]) as r:
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
                r.save_text(summary, "summary.txt")

                model_path = Path(f"model_lr_{lr}.txt")
                model_path.write_text(f"Model weights trained with lr={lr}")
                r.save(model_path, path="model/weights.pkl")
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


def run() -> None:
    """Main entry point with argparse."""
    parser = argparse.ArgumentParser(
        description="Local experiment tracking demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multiple", "both"],
        default="both",
        help="Which demo to run (default: both)",
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

    if args.mode in ["single", "both"]:
        console.print("\n[bold]Single Run Demo[/bold]")
        run_single(args.output_dir)

    if args.mode in ["multiple", "both"]:
        console.print("\n[bold]Multiple Runs Demo[/bold]")
        run_multiple(args.output_dir)

    if args.mode == "both":
        console.print("\n[bold]Directory Structure:[/bold]")
        for subdir in ["local", "local_multi"]:
            path = args.output_dir / subdir
            if path.exists():
                show_directory_tree(path, max_depth=2)

    console.print("\n✨ Demo complete!")


if __name__ == "__main__":
    run()
