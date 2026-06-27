from __future__ import annotations

import pandas as pd

from churn_retention_report.config import ChurnConfig
from churn_retention_report.recommendations import (
    assign_risk_segment,
    build_risk_segments,
)


def test_assign_risk_segment() -> None:
    config = ChurnConfig()

    assert assign_risk_segment(0.8, config) == "high"
    assert assign_risk_segment(0.5, config) == "medium"
    assert assign_risk_segment(0.2, config) == "low"


def test_build_risk_segments() -> None:
    predictions = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "churn_probability": [0.9, 0.5, 0.1],
            "risk_segment": ["high", "medium", "low"],
        }
    )

    segments = build_risk_segments(predictions)

    assert set(segments["risk_segment"]) == {"high", "medium", "low"}
