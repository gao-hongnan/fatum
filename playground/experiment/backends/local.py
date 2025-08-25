from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from fatum.experiment.types import Metric, MetricKey


class LocalStorage:
    def __init__(self, base_path: str | Path) -> None:
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._run_path: Path | None = None
        self._metrics_buffer: list[Metric] = []

    def initialize(self, run_id: str, experiment_id: str) -> None:
        """Initialize storage for a run (required by Storage protocol)."""
        self._run_path = self.base_path / experiment_id / "runs" / run_id
        self._run_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }
        (self._run_path / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def finalize(self, status: str) -> None:
        """Finalize storage when run completes (required by Storage protocol)."""
        if not self._run_path:
            return

        if self._metrics_buffer:
            self.flush_metrics()

        metadata_file = self._run_path / "metadata.json"
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            metadata.update(
                {
                    "status": status,
                    "ended_at": datetime.now().isoformat(),
                }
            )
            metadata_file.write_text(json.dumps(metadata, indent=2))

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        """Log multiple metrics at once."""
        if not self._run_path:
            return

        for key, value in metrics.items():
            metric = Metric(key=MetricKey(key), value=value, step=step)
            self._metrics_buffer.append(metric)

        # Auto-flush if buffer is large
        if len(self._metrics_buffer) >= 100:
            self.flush_metrics()

    def log_metric(self, key: MetricKey, value: float, step: int = 0) -> None:
        """Log a single metric."""
        if not self._run_path:
            return

        metric = Metric(key=key, value=value, step=step)
        self._metrics_buffer.append(metric)

        if len(self._metrics_buffer) >= 100:
            self.flush_metrics()

    def flush_metrics(self) -> None:
        """Flush buffered metrics to disk."""
        if not self._metrics_buffer or not self._run_path:
            return

        metrics_dir = self._run_path / "metrics"
        metrics_dir.mkdir(exist_ok=True)

        timestamp = int(datetime.now().timestamp() * 1000000)
        metrics_file = metrics_dir / f"batch_{timestamp}.jsonl"

        with metrics_file.open("w") as f:
            for metric in self._metrics_buffer:
                f.write(json.dumps(metric.model_dump(mode="json")) + "\n")

        self._metrics_buffer.clear()

    def save_dict(self, data: dict[str, Any], path: str) -> None:
        """Save a dictionary as JSON."""
        if not self._run_path:
            return

        file_path = self._run_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, indent=2))

    def save_text(self, content: str, path: str) -> None:
        """Save text content to a file."""
        if not self._run_path:
            return

        file_path = self._run_path / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    def save(self, source: Path, path: str) -> None:
        """Save a file or directory."""
        if not self._run_path:
            return

        dest = self._run_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)

        if source.is_file():
            shutil.copy2(source, dest)
        else:
            shutil.copytree(source, dest, dirs_exist_ok=True)
