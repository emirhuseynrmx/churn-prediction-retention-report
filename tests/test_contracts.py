from __future__ import annotations

import pandas as pd

from churn_retention_report.config import ChurnConfig
from churn_retention_report.contracts import (
    validate_feature_importance,
    validate_lift_table,
    validate_predictions,
    validate_recommendations,
    validate_risk_segments,
    validate_shap_importance,
)


def test_output_contracts_accept_valid_frames() -> None:
    config = ChurnConfig()

    predictions = validate_predictions(
        pd.DataFrame(
            {
                "customer_id": ["C001"],
                "churn_probability": [0.82],
                "risk_segment": ["high"],
            }
        ),
        config,
    )
    segments = validate_risk_segments(
        pd.DataFrame(
            {
                "risk_segment": ["high"],
                "customers": [1],
                "avg_churn_probability": [0.82],
            }
        )
    )
    importance = validate_feature_importance(
        pd.DataFrame(
            {
                "feature": ["numeric__usage"],
                "importance": [0.4],
                "direction": ["decreases_risk"],
            }
        )
    )
    recommendations = validate_recommendations(
        pd.DataFrame(
            {
                "customer_id": ["C001"],
                "risk_segment": ["high"],
                "churn_probability": [0.82],
                "likely_drivers": ["usage"],
                "recommended_action": ["Call customer"],
                "retention_priority": [1],
            }
        ),
        config,
    )
    lift_table = validate_lift_table(
        pd.DataFrame(
            {
                "decile": [1],
                "customers": [10],
                "churners": [4],
                "churn_rate": [0.4],
                "avg_churn_probability": [0.81],
                "cumulative_churners": [4],
                "cumulative_churn_capture_rate": [0.4],
            }
        )
    )
    shap_importance = validate_shap_importance(
        pd.DataFrame(
            {
                "feature": ["numeric__usage"],
                "mean_abs_shap": [0.42],
                "mean_signed_shap": [-0.12],
                "direction": ["pushes_churn_down"],
            }
        )
    )

    assert len(predictions) == 1
    assert len(segments) == 1
    assert len(importance) == 1
    assert len(recommendations) == 1
    assert len(lift_table) == 1
    assert len(shap_importance) == 1
