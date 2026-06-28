from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from churn_retention_report.config import ChurnConfig


class DataQualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    rows: int
    columns: int
    churn_rate: float
    duplicate_customer_ids: int
    missing_values: dict[str, int]
    numeric_columns: list[str]
    categorical_columns: list[str]
    leakage_warnings: list[str]

    def to_markdown(self) -> str:
        lines = [
            "# Data Quality Report",
            "",
            f"- Rows: `{self.rows}`",
            f"- Columns: `{self.columns}`",
            f"- Churn rate: `{self.churn_rate:.2%}`",
            f"- Duplicate customer ids: `{self.duplicate_customer_ids}`",
            "",
            "## Missing Values",
            "",
        ]
        if self.missing_values:
            for column, count in self.missing_values.items():
                lines.append(f"- `{column}`: `{count}`")
        else:
            lines.append("- No missing values detected.")
        lines.extend(["", "## Feature Types", ""])
        lines.append(f"- Numeric: `{', '.join(self.numeric_columns)}`")
        lines.append(f"- Categorical: `{', '.join(self.categorical_columns)}`")
        lines.extend(["", "## Leakage Warnings", ""])
        if self.leakage_warnings:
            lines.extend(f"- `{warning}`" for warning in self.leakage_warnings)
        else:
            lines.append("- No obvious post-churn leakage columns detected.")
        return "\n".join(lines) + "\n"


def profile_data_quality(frame: pd.DataFrame, config: ChurnConfig) -> DataQualityReport:
    features = frame.drop(columns=[config.id_column, config.target_column])
    numeric = [
        column for column in features.columns if pd.api.types.is_numeric_dtype(features[column])
    ]
    categorical = [column for column in features.columns if column not in numeric]
    return DataQualityReport(
        rows=len(frame),
        columns=len(frame.columns),
        churn_rate=float(frame[config.target_column].mean()),
        duplicate_customer_ids=int(frame[config.id_column].duplicated().sum()),
        missing_values={
            str(column): int(count)
            for column, count in frame.isna().sum().items()
            if int(count) > 0
        },
        numeric_columns=[str(column) for column in numeric],
        categorical_columns=[str(column) for column in categorical],
        leakage_warnings=_detect_leakage_columns(frame, config),
    )


def _detect_leakage_columns(frame: pd.DataFrame, config: ChurnConfig) -> list[str]:
    suspicious_terms = [
        "cancel",
        "cancellation",
        "churn_date",
        "refund_after",
        "refund_post",
        "closed_at",
        "end_date",
        "ended_at",
        "contract_end",
        "churn_reason",
        "termination",
    ]
    warnings: list[str] = []
    excluded = {config.id_column.lower(), config.target_column.lower()}
    for column in frame.columns:
        normalized = str(column).lower()
        if normalized in excluded:
            continue
        if any(term in normalized for term in suspicious_terms):
            warnings.append(
                f"{column}: possible post-churn leakage column; review before modeling."
            )
    return warnings


def write_data_quality_report(report: DataQualityReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_markdown(), encoding="utf-8")
    return path
