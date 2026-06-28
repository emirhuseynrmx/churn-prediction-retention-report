from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class ProbabilityEvidence:
    calibration_table: pd.DataFrame
    metric_intervals: pd.DataFrame
    metrics: dict[str, float]


def build_probability_evidence(
    *,
    y_true: pd.Series,
    probabilities: pd.Series,
    bins: int,
    bootstrap_iterations: int,
    random_state: int,
) -> ProbabilityEvidence:
    y = y_true.astype(int).to_numpy()
    scores = probabilities.astype(float).to_numpy()
    calibration_table = build_calibration_table(y, scores, bins=bins)
    metric_intervals = build_metric_intervals(
        y,
        scores,
        bootstrap_iterations=bootstrap_iterations,
        random_state=random_state,
    )
    metrics = {"brier_score": round(float(brier_score_loss(y, scores)), 4)}
    for row in metric_intervals.itertuples(index=False):
        metrics[f"{row.metric}_ci_low"] = round(float(row.ci_low), 4)
        metrics[f"{row.metric}_ci_high"] = round(float(row.ci_high), 4)
    return ProbabilityEvidence(
        calibration_table=calibration_table,
        metric_intervals=metric_intervals,
        metrics=metrics,
    )


def build_calibration_table(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int,
) -> pd.DataFrame:
    frame = pd.DataFrame({"churned": y_true, "churn_probability": probabilities})
    frame["bin"] = pd.qcut(
        frame["churn_probability"].rank(method="first"),
        q=min(bins, len(frame)),
        labels=False,
        duplicates="drop",
    )
    calibration = (
        frame.groupby("bin", observed=True)
        .agg(
            customers=("churned", "size"),
            mean_predicted_probability=("churn_probability", "mean"),
            observed_churn_rate=("churned", "mean"),
        )
        .reset_index()
        .sort_values("mean_predicted_probability", ascending=False)
    )
    calibration["bin"] = range(1, len(calibration) + 1)
    calibration["absolute_gap"] = (
        calibration["mean_predicted_probability"] - calibration["observed_churn_rate"]
    ).abs()
    return calibration[
        [
            "bin",
            "customers",
            "mean_predicted_probability",
            "observed_churn_rate",
            "absolute_gap",
        ]
    ]


def build_metric_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    bootstrap_iterations: int,
    random_state: int,
) -> pd.DataFrame:
    point_estimates = {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }
    rng = np.random.default_rng(random_state)
    samples: dict[str, list[float]] = {metric: [] for metric in point_estimates}
    row_count = len(y_true)
    for _ in range(bootstrap_iterations):
        indices = rng.integers(0, row_count, row_count)
        sampled_y = y_true[indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        sampled_probabilities = probabilities[indices]
        samples["roc_auc"].append(float(roc_auc_score(sampled_y, sampled_probabilities)))
        samples["pr_auc"].append(
            float(average_precision_score(sampled_y, sampled_probabilities))
        )

    rows = []
    for metric, estimate in point_estimates.items():
        values = np.array(samples[metric], dtype=float)
        if values.size == 0:
            ci_low = estimate
            ci_high = estimate
        else:
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "bootstrap_samples": int(values.size),
            }
        )
    return pd.DataFrame(rows)
