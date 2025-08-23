"""Common utilities for experiment demos."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.tree import Tree

console = Console()


def show_directory_tree(path: Path, max_depth: int = 3) -> None:
    """Display the experiment directory structure as a tree."""
    tree = Tree(f"[bold cyan]{path.name}/[/bold cyan]")

    def add_items(node: Tree, dir_path: Path, depth: int = 0) -> None:
        if depth >= max_depth:
            return

        try:
            items = sorted(dir_path.iterdir())
            dirs = [d for d in items if d.is_dir()]
            files = [f for f in items if f.is_file()]

            for dir_item in dirs:
                dir_node = node.add(f"📁 [yellow]{dir_item.name}/[/yellow]")
                add_items(dir_node, dir_item, depth + 1)

            for file_item in files:
                size = file_item.stat().st_size
                node.add(f"📄 [green]{file_item.name}[/green] ({size} bytes)")
        except PermissionError:
            pass

    add_items(tree, path)
    console.print(tree)


def simulate_model_training(epochs: int = 5) -> list[dict[str, float]]:
    """Simulate model training and return metrics."""
    metrics = []
    for epoch in range(epochs):
        loss = 2.0 * (0.9**epoch) + random.uniform(-0.1, 0.1)
        accuracy = min(0.95, 0.5 + 0.08 * epoch + random.uniform(-0.02, 0.02))
        metrics.append({"loss": loss, "accuracy": accuracy})
    return metrics


def simulate_hyperparameter_search(n_trials: int = 5) -> list[dict[str, Any]]:
    """Simulate hyperparameter search with multiple trials."""
    results = []

    for trial in range(n_trials):
        lr = random.choice([0.001, 0.01, 0.1])
        batch_size = random.choice([16, 32, 64])
        dropout = random.uniform(0.1, 0.5)

        base_score = 0.6
        if lr == 0.01:
            base_score += 0.15
        if batch_size == 32:
            base_score += 0.1
        if dropout < 0.3:
            base_score += 0.05

        final_score = min(0.95, base_score + random.uniform(-0.05, 0.05))

        results.append(
            {
                "trial": trial,
                "learning_rate": lr,
                "batch_size": batch_size,
                "dropout": dropout,
                "final_score": final_score,
            }
        )

    return results


def simulate_distributed_training(nodes: int = 3) -> list[dict[str, Any]]:
    """Simulate distributed training across multiple nodes."""
    import time

    results = []
    for node_id in range(nodes):
        node_metrics = {
            "node_id": node_id,
            "hostname": f"gpu-node-{node_id:02d}",
            "gpu_util": random.uniform(85, 98),
            "memory_gb": random.uniform(12, 16),
            "throughput": random.uniform(1000, 1500),
            "loss": 2.0 * (0.85 ** (node_id + 1)) + random.uniform(-0.1, 0.1),
        }
        results.append(node_metrics)
        time.sleep(0.05)

    return results
