from __future__ import annotations

import pandas as pd
import pytest

from churn_retention_report.config import ChurnConfig
from churn_retention_report.schema import validate_input_frame


def test_validate_input_frame_requires_minimum_examples_per_class() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:03d}" for index in range(20)],
            "churned": [1] + [0] * 19,
            "usage": list(range(20)),
        }
    )

    with pytest.raises(ValueError, match="at least 5 churned and 5 retained"):
        validate_input_frame(frame, ChurnConfig())
