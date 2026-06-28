from __future__ import annotations

import pandas as pd

from churn_retention_report.kaggle_data import normalize_telco_churn


def test_normalize_telco_churn_maps_real_kaggle_columns() -> None:
    raw = pd.DataFrame(
        {
            "customerID": ["C1", "C2"],
            "Contract": ["Month-to-month", "Two year"],
            "InternetService": ["Fiber optic", "DSL"],
            "PaymentMethod": ["Electronic check", "Credit card"],
            "MonthlyCharges": [89.9, 44.2],
            "TotalCharges": ["89.9", "530.4"],
            "TechSupport": ["No", "Yes"],
            "OnlineSecurity": ["No", "Yes"],
            "DeviceProtection": ["Yes", "No"],
            "tenure": [1, 12],
            "PaperlessBilling": ["Yes", "No"],
            "SeniorCitizen": [1, 0],
            "Churn": ["Yes", "No"],
        }
    )

    normalized = normalize_telco_churn(raw)

    assert normalized["customer_id"].tolist() == ["C1", "C2"]
    assert normalized["churned"].tolist() == [1, 0]
    assert normalized["support_tickets"].tolist() == [2, 1]
