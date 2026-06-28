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
        validate_input_frame(frame, ChurnConfig(minimum_rows=20))


def test_validate_input_frame_accepts_numeric_customer_ids() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": list(range(120)),
            "churned": [1] * 60 + [0] * 60,
            "usage": list(range(120)),
        }
    )

    validated = validate_input_frame(frame, ChurnConfig())

    assert validated["customer_id"].iloc[0] == "0"


def test_validate_input_frame_maps_positive_label() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:03d}" for index in range(120)],
            "churned": ["cancelled"] * 60 + ["active"] * 60,
            "usage": list(range(120)),
        }
    )

    validated = validate_input_frame(frame, ChurnConfig(positive_label="cancelled"))

    assert validated["churned"].sum() == 60


def test_validate_input_frame_rejects_duplicate_customer_ids() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": ["C001", "C001", *[f"C{index:03d}" for index in range(118)]],
            "churned": [1] * 60 + [0] * 60,
            "usage": list(range(120)),
        }
    )

    with pytest.raises(ValueError, match="Duplicate customer IDs"):
        validate_input_frame(frame, ChurnConfig())


def test_validate_input_frame_uses_configurable_minimum_rows() -> None:
    frame = pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:03d}" for index in range(40)],
            "churned": [1] * 20 + [0] * 20,
            "usage": list(range(40)),
        }
    )

    with pytest.raises(ValueError, match="at least 100 rows"):
        validate_input_frame(frame, ChurnConfig())
