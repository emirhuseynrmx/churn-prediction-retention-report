from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline


def write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.4f")
    return path


def write_model(model: Pipeline, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    return path


def write_metrics_report(
    *,
    metrics: dict[str, object],
    output_path: Path,
    rows: int,
    churn_rate: float,
    high_risk_count: int,
    medium_risk_count: int,
    top_features: pd.DataFrame,
    lift_table: pd.DataFrame,
) -> Path:
    sorted_lift = lift_table.sort_values("decile")
    top_decile = sorted_lift.iloc[0]
    top_two_deciles = sorted_lift.head(2).iloc[-1]
    lines = [
        "# Churn Prediction Metrics Report",
        "",
        f"- Rows analyzed: `{rows}`",
        f"- Observed churn rate: `{churn_rate:.2%}`",
        f"- High-risk customers: `{high_risk_count}`",
        f"- Medium-risk customers: `{medium_risk_count}`",
        "- Top decile churn capture: "
        f"`{float(top_decile['cumulative_churn_capture_rate']):.2%}`",
        "- Top 20% churn capture: "
        f"`{float(top_two_deciles['cumulative_churn_capture_rate']):.2%}`",
        "",
        "## Model Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Probability Evidence",
            "",
            "- Calibration is calculated on the holdout set only.",
            "- Confidence intervals are bootstrap estimates from holdout predictions.",
            "- `calibration_table.csv` compares predicted probability with observed churn rate.",
            "- `metric_confidence_intervals.csv` reports uncertainty around ROC-AUC and PR-AUC.",
        ]
    )
    lines.extend(["", "## Top Risk Drivers", ""])
    for _, row in top_features.head(10).iterrows():
        lines.append(
            f"- `{row['feature']}`: `{row['importance']:.4f}` ({row['direction']})"
        )
    lines.extend(
        [
            "",
            "## Recommended Business Read",
            "",
            "Use `retention_recommendations.csv` as the action queue. "
            "High-risk customers should be reviewed first, then medium-risk customers.",
            "",
            "Use `holdout_lift_table.csv` to prove whether the model concentrates "
            "churners near the top of the risk-ranked customer list.",
            "",
            "PR-AUC is included because churn datasets are often imbalanced and "
            "precision/recall quality matters more than accuracy alone.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
