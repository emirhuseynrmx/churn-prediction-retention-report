from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from churn_retention_report.config import ChurnConfig


def validate_predictions(frame: pd.DataFrame, config: ChurnConfig) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {
            config.id_column: pa.Column(str, nullable=False),
            "churn_probability": pa.Column(float, checks=pa.Check.in_range(0, 1)),
            "risk_segment": pa.Column(str, checks=pa.Check.isin(["high", "medium", "low"])),
        },
        coerce=True,
        strict=True,
    )
    return schema.validate(frame, lazy=True)


def validate_risk_segments(frame: pd.DataFrame) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {
            "risk_segment": pa.Column(str, checks=pa.Check.isin(["high", "medium", "low"])),
            "customers": pa.Column(int, checks=pa.Check.ge(0)),
            "avg_churn_probability": pa.Column(float, checks=pa.Check.in_range(0, 1)),
        },
        coerce=True,
        strict=True,
    )
    return schema.validate(frame, lazy=True)


def validate_feature_importance(frame: pd.DataFrame) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {
            "feature": pa.Column(str, nullable=False),
            "importance": pa.Column(float, checks=pa.Check.ge(0)),
            "direction": pa.Column(
                str,
                checks=pa.Check.isin(
                    ["increases_risk", "decreases_risk", "model_importance"]
                ),
            ),
        },
        coerce=True,
        strict=True,
    )
    return schema.validate(frame, lazy=True)


def validate_lift_table(frame: pd.DataFrame) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {
            "decile": pa.Column(int, checks=pa.Check.ge(1)),
            "customers": pa.Column(int, checks=pa.Check.ge(0)),
            "churners": pa.Column(int, checks=pa.Check.ge(0)),
            "churn_rate": pa.Column(float, checks=pa.Check.in_range(0, 1)),
            "avg_churn_probability": pa.Column(float, checks=pa.Check.in_range(0, 1)),
            "cumulative_churners": pa.Column(int, checks=pa.Check.ge(0)),
            "cumulative_churn_capture_rate": pa.Column(
                float,
                checks=pa.Check.in_range(0, 1),
            ),
        },
        coerce=True,
        strict=True,
    )
    return schema.validate(frame, lazy=True)


def validate_shap_importance(frame: pd.DataFrame) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {
            "feature": pa.Column(str, nullable=False),
            "mean_abs_shap": pa.Column(float, checks=pa.Check.ge(0)),
            "mean_signed_shap": pa.Column(float, nullable=False),
            "direction": pa.Column(
                str,
                checks=pa.Check.isin(["pushes_churn_up", "pushes_churn_down"]),
            ),
        },
        coerce=True,
        strict=True,
    )
    return schema.validate(frame, lazy=True)


def validate_recommendations(frame: pd.DataFrame, config: ChurnConfig) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {
            config.id_column: pa.Column(str, nullable=False),
            "risk_segment": pa.Column(str, checks=pa.Check.isin(["high", "medium", "low"])),
            "churn_probability": pa.Column(float, checks=pa.Check.in_range(0, 1)),
            "likely_drivers": pa.Column(str, nullable=False),
            "recommended_action": pa.Column(str, nullable=False),
            "retention_priority": pa.Column(int, checks=pa.Check.in_range(1, 3)),
        },
        coerce=True,
        strict=True,
    )
    return schema.validate(frame, lazy=True)
