from __future__ import annotations

import pandas as pd

from churn_retention_report.config import ChurnConfig
from churn_retention_report.quality import profile_data_quality


def test_profile_data_quality_reports_core_stats() -> None:
    config = ChurnConfig()
    frame = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B"],
            "plan": ["Basic", "Basic", "Pro"],
            "usage": [1, 2, 3],
            "churned": [1, 0, 0],
        }
    )

    report = profile_data_quality(frame, config)

    assert report.rows == 3
    assert report.duplicate_customer_ids == 1
    assert report.numeric_columns == ["usage"]
    assert report.missing_values == {}


def test_profile_data_quality_warns_about_possible_leakage_columns() -> None:
    config = ChurnConfig()
    frame = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C", "D", "E"],
            "plan": ["Basic", "Basic", "Pro", "Pro", "Basic"],
            "cancel_date": ["", "", "2026-01-01", "", ""],
            "end_date": ["", "", "", "", "2026-01-01"],
            "usage": [1, 2, 3, 4, 5],
            "churned": [0, 0, 1, 0, 1],
        }
    )

    report = profile_data_quality(frame, config)

    assert any("cancel_date" in warning for warning in report.leakage_warnings)
    assert any("end_date" in warning for warning in report.leakage_warnings)
