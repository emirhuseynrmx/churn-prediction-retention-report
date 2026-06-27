from __future__ import annotations

import pandas as pd

from churn_retention_report.config import ChurnConfig


def assign_risk_segment(probability: float, config: ChurnConfig) -> str:
    if probability >= config.risk_thresholds.high:
        return "high"
    if probability >= config.risk_thresholds.medium:
        return "medium"
    return "low"


def build_predictions(frame: pd.DataFrame, probabilities, config: ChurnConfig) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            config.id_column: frame[config.id_column],
            "churn_probability": probabilities.round(4),
        }
    )
    output["risk_segment"] = output["churn_probability"].map(
        lambda value: assign_risk_segment(float(value), config)
    )
    return output


def build_risk_segments(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby("risk_segment", as_index=False)
        .agg(
            customers=("risk_segment", "size"),
            avg_churn_probability=("churn_probability", "mean"),
        )
        .sort_values("avg_churn_probability", ascending=False)
    )


def build_lift_table(
    predictions: pd.DataFrame,
    source_frame: pd.DataFrame,
    config: ChurnConfig,
) -> pd.DataFrame:
    scored = predictions.merge(
        source_frame[[config.id_column, config.target_column]],
        on=config.id_column,
        how="left",
    ).sort_values("churn_probability", ascending=False)
    scored["decile"] = (
        pd.qcut(
            scored["churn_probability"].rank(method="first", ascending=False),
            q=config.lift_table_deciles,
            labels=False,
        )
        + 1
    )
    total_churners = max(int(scored[config.target_column].sum()), 1)
    lift = (
        scored.groupby("decile", as_index=False)
        .agg(
            customers=(config.id_column, "size"),
            churners=(config.target_column, "sum"),
            churn_rate=(config.target_column, "mean"),
            avg_churn_probability=("churn_probability", "mean"),
        )
        .sort_values("decile")
    )
    lift["cumulative_churners"] = lift["churners"].cumsum()
    lift["cumulative_churn_capture_rate"] = lift["cumulative_churners"] / total_churners
    return lift


def build_recommendations(
    predictions: pd.DataFrame,
    source_frame: pd.DataFrame,
    feature_importance: pd.DataFrame,
    config: ChurnConfig,
) -> pd.DataFrame:
    top_rule_keys = _match_drivers(
        feature_importance["feature"].head(12).tolist(),
        config.recommendation_rules,
    )
    rows: list[dict[str, object]] = []
    merged = predictions.merge(source_frame, on=config.id_column, how="left")
    benchmarks = _build_driver_benchmarks(merged)
    for _, row in merged.iterrows():
        drivers = _customer_drivers(row, benchmarks, top_rule_keys)
        recommendation = _recommendation_for_segment(
            segment=str(row["risk_segment"]),
            drivers=drivers,
            rules=config.recommendation_rules,
        )
        priority = _priority_for_segment(str(row["risk_segment"]))
        rows.append(
            {
                config.id_column: row[config.id_column],
                "risk_segment": row["risk_segment"],
                "churn_probability": row["churn_probability"],
                "likely_drivers": ", ".join(drivers[:3]) if drivers else "general_engagement",
                "recommended_action": recommendation,
                "retention_priority": priority,
            }
        )
    return pd.DataFrame(rows)


def _match_drivers(top_features: list[str], rules: dict[str, str]) -> list[str]:
    drivers: list[str] = []
    for feature in top_features:
        for key in rules:
            if key in feature and key not in drivers:
                drivers.append(key)
    return drivers


def _build_driver_benchmarks(frame: pd.DataFrame) -> dict[str, float]:
    benchmarks: dict[str, float] = {}
    for column in ["usage", "payments", "support_tickets", "nps_score", "failed_payments"]:
        if column in frame:
            benchmarks[column] = float(frame[column].median())
    if "support_tickets" in frame:
        benchmarks["support_tickets_high"] = float(frame["support_tickets"].quantile(0.75))
    return benchmarks


def _customer_drivers(
    row: pd.Series,
    benchmarks: dict[str, float],
    top_rule_keys: list[str],
) -> list[str]:
    drivers: list[str] = []
    if "usage" in top_rule_keys and float(row.get("usage", 0)) <= benchmarks.get("usage", 0):
        drivers.append("usage")
    has_many_tickets = float(row.get("support_tickets", 0)) >= benchmarks.get(
        "support_tickets_high",
        0,
    )
    if "support_tickets" in top_rule_keys and has_many_tickets:
        drivers.append("support_tickets")
    if "payments" in top_rule_keys and (
        float(row.get("failed_payments", 0)) > 0
        or float(row.get("payments", 0)) <= benchmarks.get("payments", 0)
    ):
        drivers.append("payments")
    if "nps_score" in top_rule_keys and float(row.get("nps_score", 10)) <= benchmarks.get(
        "nps_score",
        10,
    ):
        drivers.append("nps_score")
    if "plan" in top_rule_keys and str(row.get("plan", "")).lower() == "basic":
        drivers.append("plan")
    if not drivers:
        return top_rule_keys[:2] or ["general_engagement"]
    return drivers


def _recommendation_for_segment(
    segment: str,
    drivers: list[str],
    rules: dict[str, str],
) -> str:
    playbooks = [rules[driver] for driver in drivers if driver in rules]
    primary_playbook = playbooks[0] if playbooks else "Run a customer success review."
    if segment == "high":
        return (
            "Contact within 48 hours. Primary playbook: "
            f"{primary_playbook}"
        )
    if segment == "medium":
        return (
            "Send targeted intervention before renewal risk increases. "
            f"Suggested playbook: {primary_playbook}"
        )
    if drivers:
        return f"Monitor monthly and reinforce the strongest positive behavior: {drivers[0]}."
    return "Maintain standard lifecycle communication."


def _priority_for_segment(segment: str) -> int:
    if segment == "high":
        return 1
    if segment == "medium":
        return 2
    return 3
