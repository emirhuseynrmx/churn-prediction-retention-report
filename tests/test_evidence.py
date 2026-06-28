from __future__ import annotations

import pandas as pd

from churn_retention_report.evidence import build_probability_evidence


def test_probability_evidence_uses_holdout_scores_only() -> None:
    y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = pd.Series([0.05, 0.88, 0.22, 0.81, 0.35, 0.74, 0.44, 0.69])

    evidence = build_probability_evidence(
        y_true=y_true,
        probabilities=probabilities,
        bins=4,
        bootstrap_iterations=80,
        random_state=7,
    )

    assert len(evidence.calibration_table) == 4
    assert set(evidence.metric_intervals["metric"]) == {"roc_auc", "pr_auc"}
    assert 0 <= evidence.metrics["brier_score"] <= 1
