from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def write_risk_chart(predictions: pd.DataFrame, path: Path) -> Path:
    counts = predictions["risk_segment"].value_counts().reindex(["high", "medium", "low"]).fillna(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color=["#d73a49", "#f2cc60", "#2ea043"])
    ax.set_title("Customer Risk Segments")
    ax.set_xlabel("Risk segment")
    ax.set_ylabel("Customers")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_feature_importance_chart(feature_importance: pd.DataFrame, path: Path) -> Path:
    top = feature_importance.head(10).iloc[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="#0969da")
    ax.set_title("Top Churn Risk Drivers")
    has_coefficients = set(top["direction"]).issubset({"increases_risk", "decreases_risk"})
    ax.set_xlabel("Absolute logistic coefficient" if has_coefficients else "Feature importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def write_shap_summary_chart(shap_importance: pd.DataFrame, path: Path) -> Path:
    top = shap_importance.head(12).iloc[::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = [
        "#d73a49" if direction == "pushes_churn_up" else "#2ea043"
        for direction in top["direction"]
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["mean_abs_shap"], color=colors)
    ax.set_title("SHAP Explainability: Top Churn Drivers")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_probability_distribution_chart(
    *,
    predictions: pd.DataFrame,
    medium_threshold: float,
    high_threshold: float,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(
        predictions["churn_probability"],
        bins=12,
        color="#0969da",
        edgecolor="white",
        alpha=0.88,
    )
    ax.set_title("Probability Sanity Check")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Customers")
    ax.axvline(
        medium_threshold,
        color="#f2cc60",
        linestyle="--",
        linewidth=1.5,
        label="Medium risk",
    )
    ax.axvline(
        high_threshold,
        color="#d73a49",
        linestyle="--",
        linewidth=1.5,
        label="High risk",
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def write_dashboard_image(
    *,
    predictions: pd.DataFrame,
    risk_segments: pd.DataFrame,
    feature_importance: pd.DataFrame,
    recommendations: pd.DataFrame,
    path: Path,
) -> Path:
    high_risk = int((predictions["risk_segment"] == "high").sum())
    medium_risk = int((predictions["risk_segment"] == "medium").sum())
    top_driver = str(feature_importance.iloc[0]["feature"]).replace("numeric__", "")
    top_action = str(recommendations.iloc[0]["recommended_action"])
    avg_high_risk = float(risk_segments.loc[
        risk_segments["risk_segment"] == "high",
        "avg_churn_probability",
    ].max())
    if pd.isna(avg_high_risk):
        avg_high_risk = 0.0

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Churn Retention Executive Snapshot", fontsize=18, fontweight="bold")

    axes[0, 0].axis("off")
    axes[0, 0].text(0.02, 0.68, "High Risk", fontsize=13, color="#57606a")
    axes[0, 0].text(0.02, 0.36, f"{high_risk}", fontsize=38, fontweight="bold")
    axes[0, 0].text(0.02, 0.15, f"Avg risk: {avg_high_risk:.1%}", fontsize=12)

    axes[0, 1].axis("off")
    axes[0, 1].text(0.02, 0.68, "Medium Risk", fontsize=13, color="#57606a")
    axes[0, 1].text(0.02, 0.36, f"{medium_risk}", fontsize=38, fontweight="bold")
    axes[0, 1].text(0.02, 0.15, "Review before renewal", fontsize=12)

    counts = (
        predictions["risk_segment"]
        .value_counts()
        .reindex(["high", "medium", "low"])
        .fillna(0)
    )
    axes[1, 0].bar(counts.index, counts.values, color=["#d73a49", "#f2cc60", "#2ea043"])
    axes[1, 0].set_title("Risk Queue")
    axes[1, 0].set_ylabel("Customers")
    axes[1, 0].grid(axis="y", alpha=0.2)

    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.80, "Top Driver", fontsize=13, color="#57606a")
    axes[1, 1].text(0.02, 0.62, top_driver, fontsize=17, fontweight="bold")
    axes[1, 1].text(0.02, 0.42, "Recommended Action", fontsize=13, color="#57606a")
    axes[1, 1].text(0.02, 0.14, fill(top_action, width=52), fontsize=10, wrap=True)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path
