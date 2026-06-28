from __future__ import annotations

from typer.testing import CliRunner

from churn_retention_report.cli import app


def test_cli_run_command(tmp_path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "data/telco_customers.csv",
            "--config",
            "examples/config.json",
            "--out",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Churn report complete" in result.output
    assert (tmp_path / "predictions.csv").exists()
