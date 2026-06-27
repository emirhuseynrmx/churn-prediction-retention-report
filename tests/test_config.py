from __future__ import annotations

import pytest
from pydantic import ValidationError

from churn_retention_report.config import RiskThresholds


def test_risk_thresholds_require_high_above_medium() -> None:
    with pytest.raises(ValidationError, match="High risk threshold"):
        RiskThresholds(high=0.3, medium=0.5)
