from __future__ import annotations

import pytest
from pydantic import ValidationError

from churn_retention_report.config import ChurnConfig, RiskThresholds


def test_risk_thresholds_require_high_above_medium() -> None:
    with pytest.raises(ValidationError, match="High risk threshold"):
        RiskThresholds(high=0.3, medium=0.5)


def test_xgboost_parameters_are_configurable() -> None:
    config = ChurnConfig.model_validate(
        {
            "xgboost": {
                "n_estimators": 80,
                "max_depth": 2,
                "learning_rate": 0.08,
            }
        }
    )

    assert config.xgboost.n_estimators == 80
    assert config.xgboost.max_depth == 2
    assert config.xgboost.learning_rate == 0.08
