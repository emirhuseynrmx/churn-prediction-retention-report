from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from churn_retention_report.config import ChurnConfig
from churn_retention_report.pipeline import run_churn_pipeline

app = typer.Typer(help="Train churn model and export retention deliverables.")
console = Console()


@app.command()
def run(
    input_path: Annotated[Path, typer.Argument(help="Customer CSV with churn labels.")],
    config: Annotated[Path | None, typer.Option(help="Optional JSON config path.")] = None,
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("outputs"),
) -> None:
    churn_config = ChurnConfig.load(config)
    result = run_churn_pipeline(input_path, out, churn_config)
    console.print("[green]Churn report complete[/green]")
    console.print(f"Predictions: {result.predictions_path}")
    console.print(f"Recommendations: {result.recommendations_path}")
    console.print(f"Metrics report: {result.metrics_report_path}")
