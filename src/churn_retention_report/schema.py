from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from churn_retention_report.config import ChurnConfig


def validate_input_frame(frame: pd.DataFrame, config: ChurnConfig) -> pd.DataFrame:
    prepared = frame.copy()
    if config.id_column in prepared:
        prepared[config.id_column] = prepared[config.id_column].astype("string")
    if config.target_column in prepared:
        prepared[config.target_column] = _normalize_target_labels(
            prepared[config.target_column],
            config.positive_label,
        )
    required = {
        config.id_column: pa.Column(str, nullable=False),
        config.target_column: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
    }
    schema = pa.DataFrameSchema(required, coerce=True, strict=False)
    validated = schema.validate(prepared, lazy=True)
    duplicate_ids = int(validated[config.id_column].duplicated().sum())
    if duplicate_ids:
        raise ValueError(
            "Duplicate customer IDs found. Please deduplicate the file or provide "
            "row-level account IDs before running the churn report."
        )
    if len(validated) < config.minimum_rows:
        raise ValueError(
            f"Churn modeling needs at least {config.minimum_rows} rows for a meaningful "
            "train/test split."
        )
    if validated[config.target_column].nunique() < 2:
        raise ValueError("Target column must contain both churned and retained examples.")
    class_counts = validated[config.target_column].value_counts().to_dict()
    churned_count = int(class_counts.get(1, 0))
    retained_count = int(class_counts.get(0, 0))
    if churned_count < config.minimum_class_count or retained_count < config.minimum_class_count:
        raise ValueError(
            f"Churn modeling needs at least {config.minimum_class_count} churned and "
            f"{config.minimum_class_count} retained examples "
            "for a stable train/test split."
        )
    return validated


def _normalize_target_labels(values: pd.Series, positive_label: str | int | bool) -> pd.Series:
    observed = {_normalize_label(value) for value in values.dropna().unique()}
    if observed.issubset({"0", "1"}):
        return values
    positive = _normalize_label(positive_label)
    return values.map(lambda value: 1 if _normalize_label(value) == positive else 0)


def _normalize_label(value: object) -> str:
    return str(value).strip().lower()
