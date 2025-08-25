from __future__ import annotations

import json
import platform
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fatum import experiment
from fatum.reproducibility.git import GitInfo

console = Console()

DEFAULT_BUCKET = "fatum-experiments"
DEFAULT_CACHE_DIR = Path("./cache")
DEFAULT_MAX_WORKERS = 20
DEFAULT_MULTIPART_THRESHOLD_MB = 25
DEFAULT_MULTIPART_CHUNKSIZE_MB = 25
DEFAULT_MINIO_ENDPOINT = "http://localhost:9000"
DEFAULT_MINIO_ACCESS_KEY = "minioadmin"
DEFAULT_MINIO_SECRET_KEY = "minioadmin"
DEFAULT_REGION = "us-east-1"
EXPERIMENTS_PREFIX = "experiments"
METADATA_DIR = "metadata"
ARTIFACTS_DIR = "artifacts"
DOCUMENTS_DIR = "documents"


class SystemInfo(BaseModel):
    platform: str = Field(default_factory=lambda: platform.system())
    platform_release: str = Field(default_factory=lambda: platform.release())
    platform_version: str = Field(default_factory=lambda: platform.version())
    architecture: str = Field(default_factory=lambda: platform.machine())
    python_version: str = Field(default_factory=lambda: platform.python_version())
    node: str = Field(default_factory=lambda: platform.node())


class DocumentS3Storage:
    def __init__(
        self,
        bucket: str,
        max_workers: int = DEFAULT_MAX_WORKERS,
        multipart_threshold_mb: int = DEFAULT_MULTIPART_THRESHOLD_MB,
        multipart_chunksize_mb: int = DEFAULT_MULTIPART_CHUNKSIZE_MB,
        cache_dir: Path | None = None,
        **boto_kwargs: Any,
    ) -> None:
        self.bucket = bucket
        self.max_workers = max_workers
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        config = Config(
            max_pool_connections=max_workers * 2,
            retries={"max_attempts": 2, "mode": "adaptive"},
            s3={"addressing_style": "path"},
        )

        self.s3 = boto3.client("s3", config=config, **boto_kwargs)

        self.transfer_config = TransferConfig(
            multipart_threshold=multipart_threshold_mb * 1024 * 1024,
            multipart_chunksize=multipart_chunksize_mb * 1024 * 1024,
            max_concurrency=max_workers,
            use_threads=True,
        )

        self._run_id: str | None = None
        self._experiment_id: str | None = None
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                with suppress(ClientError):
                    self.s3.create_bucket(Bucket=self.bucket)

    def initialize(self, run_id: str, experiment_id: str) -> None:
        self._run_id = run_id
        self._experiment_id = experiment_id

        metadata = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "started_at": datetime.now().isoformat(),
            "git_info": GitInfo.current().model_dump() if GitInfo.current() else None,
            "system_info": SystemInfo().model_dump(),
        }
        self._save_json(f"{METADATA_DIR}/run_init.json", metadata)

    def finalize(self, status: str) -> None:
        final_metadata = {
            "status": status,
            "ended_at": datetime.now().isoformat(),
        }
        self._save_json(f"{METADATA_DIR}/run_final.json", final_metadata)

    def save_documents(self, document_dir: Path, prefix: str = DOCUMENTS_DIR) -> dict[str, str]:
        if not document_dir.exists():
            raise FileNotFoundError(f"Document directory not found: {document_dir}")

        file_pairs: list[tuple[Path, str]] = []
        base_key = self._build_key(prefix)

        if document_dir.is_dir():
            for file_path in document_dir.rglob("*"):
                if file_path.is_file():
                    relative = file_path.relative_to(document_dir)
                    s3_key = f"{base_key}/{relative}".replace("\\", "/")
                    file_pairs.append((file_path, s3_key))
        else:
            s3_key = f"{base_key}/{document_dir.name}"
            file_pairs.append((document_dir, s3_key))

        if not file_pairs:
            return {}

        results = {}

        def upload_single(local_path: Path, key: str) -> tuple[str, str, bool]:
            try:
                file_size = local_path.stat().st_size
                upload_config = self.transfer_config if file_size > self.transfer_config.multipart_threshold else None

                if upload_config:
                    self.s3.upload_file(str(local_path), self.bucket, key, Config=upload_config)
                else:
                    self.s3.upload_file(str(local_path), self.bucket, key)

                return str(local_path), key, True
            except Exception:
                return str(local_path), key, False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(upload_single, path, key) for path, key in file_pairs]
            for future in as_completed(futures):
                local_path, s3_key, success = future.result()
                if success:
                    results[local_path] = s3_key

        if results:
            stats = {
                "count": len(results),
                "total_size": sum(Path(p).stat().st_size for p in results),
                "uploaded_at": datetime.now().isoformat(),
            }
            self._save_json(f"{METADATA_DIR}/document_stats.json", stats)

        return results

    def save_config(self, config: dict[str, Any]) -> None:
        self._save_json("config.json", config)

    def save_artifact(self, key: str, source: Path) -> str:
        s3_key = self._build_key(f"{ARTIFACTS_DIR}/{key}")

        try:
            file_size = source.stat().st_size
            upload_config = self.transfer_config if file_size > self.transfer_config.multipart_threshold else None

            if upload_config:
                self.s3.upload_file(str(source), self.bucket, s3_key, Config=upload_config)
            else:
                self.s3.upload_file(str(source), self.bucket, s3_key)

            return f"s3://{self.bucket}/{s3_key}"
        except Exception:
            return ""

    def download_directory(self, prefix: str | None = None, local_dir: Path | None = None) -> dict[str, Path]:
        if prefix is None:
            prefix = self._build_key("")

        if local_dir is None:
            if self._experiment_id and self._run_id:
                local_dir = self.cache_dir / self._experiment_id / self._run_id
            else:
                local_dir = self.cache_dir

        local_dir.mkdir(parents=True, exist_ok=True)

        keys: list[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            if "Contents" in page:
                keys.extend(
                    obj.get("Key", "")
                    for obj in page["Contents"]
                    if obj.get("Key", "") and not obj.get("Key", "").endswith("/")
                )

        if not keys:
            return {}

        results = {}

        def download_single(key: str) -> tuple[str, Path | None]:
            try:
                local_path = local_dir / key[len(prefix) :].lstrip("/")
                local_path.parent.mkdir(parents=True, exist_ok=True)

                response = self.s3.head_object(Bucket=self.bucket, Key=key)
                file_size = response.get("ContentLength", 0)

                download_config = self.transfer_config if file_size > self.transfer_config.multipart_threshold else None

                if download_config:
                    self.s3.download_file(self.bucket, key, str(local_path), Config=download_config)
                else:
                    self.s3.download_file(self.bucket, key, str(local_path))

                return key, local_path
            except Exception:
                return key, None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(download_single, key) for key in keys]
            for future in as_completed(futures):
                key, path = future.result()
                if path:
                    results[key] = path

        return results

    def _build_key(self, suffix: str = "") -> str:
        base = f"{EXPERIMENTS_PREFIX}/{self._experiment_id}/runs/{self._run_id}"
        return f"{base}/{suffix}" if suffix else base

    def _save_json(self, relative_path: str, data: dict[str, Any]) -> None:
        if not self._run_id or not self._experiment_id:
            return

        key = self._build_key(relative_path)
        temp_file = Path(tempfile.mktemp(suffix=".json"))
        try:
            temp_file.write_text(json.dumps(data, indent=2, default=str))
            self.s3.upload_file(str(temp_file), self.bucket, key)
        finally:
            temp_file.unlink(missing_ok=True)


def create_minio_storage(
    bucket: str = DEFAULT_BUCKET,
    max_workers: int = DEFAULT_MAX_WORKERS,
    endpoint_url: str = DEFAULT_MINIO_ENDPOINT,
    aws_access_key_id: str = DEFAULT_MINIO_ACCESS_KEY,
    aws_secret_access_key: str = DEFAULT_MINIO_SECRET_KEY,
    region_name: str = DEFAULT_REGION,
) -> DocumentS3Storage:
    return DocumentS3Storage(
        bucket=bucket,
        max_workers=max_workers,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )


def create_sample_documents(output_dir: Path) -> Path:
    docs_dir = output_dir / "sample_documents"
    docs_dir.mkdir(parents=True, exist_ok=True)

    documents = {
        "README.md": "# Sample Project\n\nThis is a sample documentation.",
        "config.yaml": """model:
  name: transformer
  layers: 12
  hidden_size: 768
training:
  batch_size: 32
  learning_rate: 0.001
""",
        "results.json": json.dumps(
            {"accuracy": 0.94, "loss": 0.23, "epochs": 50},
            indent=2,
        ),
        "data/train.csv": "id,feature1,feature2,label\n1,0.5,0.3,1\n2,0.7,0.8,0\n",
        "models/checkpoint.txt": "Model checkpoint placeholder",
        "logs/training.log": """2024-01-01 10:00:00 - Starting training
2024-01-01 10:01:00 - Epoch 1/50 - Loss: 0.5
2024-01-01 10:02:00 - Epoch 2/50 - Loss: 0.4
""",
    }

    for file_path, content in documents.items():
        full_path = docs_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)

    return docs_dir


def run() -> None:
    console.print(
        Panel.fit(
            "[bold yellow]High-Performance Document Storage Demo[/bold yellow]\n"
            "Self-contained S3 implementation with ThreadPoolExecutor",
            border_style="yellow",
        )
    )

    console.print("\n[cyan]Creating optimized S3 storage...[/cyan]")
    storage = create_minio_storage(max_workers=10)

    console.print("[cyan]Creating sample documents...[/cyan]")
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = create_sample_documents(Path(temp_dir))

        files = list(docs_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        total_size = sum(f.stat().st_size for f in files)
        console.print(f"Created {len(files)} sample documents ({total_size:,} bytes)\n")

        console.print("[bold]Using fatum.experiment API:[/bold]\n")

        with experiment.experiment(
            name="document_storage_demo",
            storage=storage,
            description="High-performance document storage with S3",
            tags=["demo", "s3", "documents"],
        ) as exp:
            console.print(f"[bold]Experiment:[/bold] {exp.id}")

            with exp.run("upload_documents") as run:
                console.print(f"[bold]Run:[/bold] {run.id}\n")

                config = {
                    "model": "transformer",
                    "dataset": "sample",
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                }
                run.storage.save_config(config)
                console.print("[cyan]Saved configuration[/cyan]")

                console.print("[cyan]Uploading documents to S3 (concurrent)...[/cyan]")
                uploaded = run.storage.save_documents(docs_dir)

                if uploaded:
                    console.print(f"✅ Uploaded {len(uploaded)} files successfully")

                    table = Table(title="Uploaded Documents")
                    table.add_column("Local File", style="cyan")
                    table.add_column("S3 Key", style="green")

                    for local, s3_key in list(uploaded.items())[:5]:
                        table.add_row(Path(local).name, s3_key.split("/")[-1])

                    if len(uploaded) > 5:
                        table.add_row("...", f"... ({len(uploaded) - 5} more files)")

                    console.print(table)

                artifact_path = Path(temp_dir) / "model.pkl"
                artifact_path.write_text("Model weights placeholder")
                uri = run.storage.save_artifact("final_model.pkl", artifact_path)
                if uri:
                    console.print(f"[cyan]Saved artifact:[/cyan] {uri}")

            with exp.run("analysis") as run:
                console.print(f"\n[bold]Second Run:[/bold] {run.id}")

                analysis_config = {
                    "method": "statistical",
                    "confidence": 0.95,
                    "test_size": 0.2,
                }
                run.storage.save_config(analysis_config)

                analysis_dir = Path(temp_dir) / "analysis"
                analysis_dir.mkdir()
                (analysis_dir / "report.txt").write_text("Analysis complete")
                (analysis_dir / "metrics.json").write_text('{"r2": 0.92, "rmse": 0.15}')

                uploaded = run.storage.save_documents(analysis_dir, prefix="analysis")
                console.print(f"[cyan]Uploaded {len(uploaded)} analysis files[/cyan]")

        console.print("\n[cyan]Downloading run data from S3 (concurrent)...[/cyan]")
        downloaded = storage.download_directory(
            prefix=f"{EXPERIMENTS_PREFIX}/{exp.id}/runs/upload_documents",
            local_dir=Path(temp_dir) / "retrieved",
        )

        if downloaded:
            console.print(f"✅ Downloaded {len(downloaded)} files")
            console.print(f"   Files saved to: {Path(temp_dir) / 'retrieved'}")

    console.print("\n[green]✨ Demo complete![/green]")
    console.print("\n[dim]View your data in MinIO console: http://localhost:9001[/dim]")
    console.print("[dim]Credentials: minioadmin / minioadmin[/dim]")


if __name__ == "__main__":
    run()
