from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from churn_retention_report.config import ChurnConfig


def validate_input_frame(frame: pd.DataFrame, config: ChurnConfig) -> pd.DataFrame:
    required = {
        config.id_column: pa.Column(str, nullable=False),
        config.target_column: pa.Column(int, checks=pa.Check.isin([0, 1]), nullable=False),
    }
    schema = pa.DataFrameSchema(required, coerce=True, strict=False)
    validated = schema.validate(frame, lazy=True)
    if len(validated) < 20:
        raise ValueError("Churn modeling needs at least 20 rows for a meaningful train/test split.")
    if validated[config.target_column].nunique() < 2:
        raise ValueError("Target column must contain both churned and retained examples.")
    class_counts = validated[config.target_column].value_counts().to_dict()
    churned_count = int(class_counts.get(1, 0))
    retained_count = int(class_counts.get(0, 0))
    if churned_count < 5 or retained_count < 5:
        raise ValueError(
            "Churn modeling needs at least 5 churned and 5 retained examples "
            "for a stable train/test split."
        )
    return validated
