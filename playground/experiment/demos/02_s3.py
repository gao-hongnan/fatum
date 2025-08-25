from __future__ import annotations

import tempfile
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fatum import experiment
from playground.experiment.backends.s3 import S3Config, S3Storage

console = Console()


def create_sample_documents(output_dir: Path) -> Path:
    """Create sample documents for demo."""
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
        "results.json": '{"accuracy": 0.94, "loss": 0.23, "epochs": 50}',
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
    """Run S3 storage demo."""
    console.print(
        Panel.fit(
            "[bold yellow]S3 Storage Configuration Demo[/bold yellow]\n"
            "Demonstrates comprehensive S3Config and storage capabilities",
            border_style="yellow",
        )
    )

    console.print("\n[cyan]Creating S3 storage with configuration...[/cyan]")

    config = S3Config(
        bucket="fatum-experiments",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        addressing_style="path",
        use_ssl=False,
        max_concurrency=10,
        multipart_threshold_mb=25,
        multipart_chunksize_mb=25,
        # server_side_encryption="AES256",
    )
    storage = S3Storage(config)

    console.print("[green]✓ Storage configured with S3Config[/green]")
    console.print(f"  Bucket: {config.bucket}")
    console.print(f"  Endpoint: {config.endpoint_url}")
    console.print(f"  Max concurrency: {config.max_concurrency}")
    console.print(f"  Multipart threshold: {config.multipart_threshold_mb}MB")

    console.print("\n[cyan]Creating sample documents...[/cyan]")
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = create_sample_documents(Path(temp_dir))

        files = list(docs_dir.rglob("*"))
        files = [f for f in files if f.is_file()]
        total_size = sum(f.stat().st_size for f in files)
        console.print(f"Created {len(files)} sample documents ({total_size:,} bytes)\n")

        console.print("[bold]Using fatum.experiment API with S3Storage:[/bold]\n")

        with experiment.experiment(
            name="s3_config_demo",
            storage=storage,
            description="Demo of comprehensive S3 configuration",
            tags=["demo", "s3", "config"],
        ) as exp:
            console.print(f"[bold]Experiment:[/bold] {exp.id}")

            with exp.run("document_upload") as run:
                console.print(f"[bold]Run:[/bold] {run.id}\n")

                app_config = {
                    "model": "transformer",
                    "dataset": "sample",
                    "batch_size": 32,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                }
                run.storage.save_config(app_config)
                console.print("[cyan]Saved experiment configuration[/cyan]")

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
                console.print(f"\n[bold]Analysis Run:[/bold] {run.id}")

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
            prefix=f"{config.experiments_prefix}/{exp.id}/runs/document_upload",
            local_dir=Path(temp_dir) / "retrieved",
        )

        if downloaded:
            console.print(f"✅ Downloaded {len(downloaded)} files")
            console.print(f"   Files saved to: {Path(temp_dir) / 'retrieved'}")

    console.print("\n[green]✨ Demo complete![/green]")
    console.print("\n[dim]Configuration features demonstrated:[/dim]")
    console.print("[dim]- Pydantic-based S3Config with validation[/dim]")
    console.print("[dim]- Server-side encryption support[/dim]")
    console.print("[dim]- Concurrent upload/download with ThreadPoolExecutor[/dim]")
    console.print("[dim]- Transfer configuration (multipart thresholds)[/dim]")
    console.print("[dim]- Factory methods for MinIO and AWS S3[/dim]")
    console.print("\n[dim]View your data in MinIO console: http://localhost:9001[/dim]")
    console.print("[dim]Credentials: minioadmin / minioadmin[/dim]")


if __name__ == "__main__":
    run()
