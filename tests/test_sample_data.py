from __future__ import annotations

from churn_retention_report.sample_data import generate_saas_churn_data


def test_generate_saas_churn_data_is_reproducible() -> None:
    first = generate_saas_churn_data(rows=50, seed=7)
    second = generate_saas_churn_data(rows=50, seed=7)

    assert first.equals(second)
    assert set(first["churned"].unique()) <= {0, 1}
    assert "failed_payments" in first.columns
