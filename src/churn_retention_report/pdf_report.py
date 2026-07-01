# ruff: noqa: E501
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from shutil import which

import pandas as pd

from churn_retention_report.quality import DataQualityReport


def write_pdf_report(
    *,
    output_path: Path,
    metrics: dict[str, object],
    quality_report: DataQualityReport,
    risk_segments: pd.DataFrame,
    lift_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
    metric_intervals: pd.DataFrame,
    recommendations: pd.DataFrame,
    risk_chart_path: Path,
    feature_chart_path: Path,
    shap_chart_path: Path,
    probability_chart_path: Path,
    dashboard_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = output_path.with_suffix(".typ")
    source_path.write_text(
        _build_typst_source(
            metrics=metrics,
            quality_report=quality_report,
            risk_segments=risk_segments,
            lift_table=lift_table,
            calibration_table=calibration_table,
            metric_intervals=metric_intervals,
            recommendations=recommendations,
            risk_chart_path=risk_chart_path,
            feature_chart_path=feature_chart_path,
            shap_chart_path=shap_chart_path,
            probability_chart_path=probability_chart_path,
            dashboard_path=dashboard_path,
        ),
        encoding="utf-8",
    )
    _compile_typst(source_path, output_path)
    return output_path


def _build_typst_source(
    *,
    metrics: dict[str, object],
    quality_report: DataQualityReport,
    risk_segments: pd.DataFrame,
    lift_table: pd.DataFrame,
    calibration_table: pd.DataFrame,
    metric_intervals: pd.DataFrame,
    recommendations: pd.DataFrame,
    risk_chart_path: Path,
    feature_chart_path: Path,
    shap_chart_path: Path,
    probability_chart_path: Path,
    dashboard_path: Path,
) -> str:
    sorted_lift = lift_table.sort_values("decile")
    top_decile_capture = float(sorted_lift.iloc[0]["cumulative_churn_capture_rate"])
    top_two_decile_capture = float(sorted_lift.head(2).iloc[-1]["cumulative_churn_capture_rate"])
    high_risk_row = risk_segments.loc[risk_segments["risk_segment"] == "high"]
    high_risk_count = int(high_risk_row["customers"].iloc[0]) if not high_risk_row.empty else 0
    id_column = recommendations.columns[0]
    top_recommendations = recommendations.sort_values(
        ["retention_priority", "churn_probability"],
        ascending=[True, False],
    ).head(5)

    metric_cards = [
        ("Rows analyzed", f"{quality_report.rows:,}", "Validated customer rows", "blue"),
        ("Observed churn", f"{quality_report.churn_rate:.1%}", "Historical target rate", "amber"),
        ("High risk", f"{high_risk_count:,}", "Priority accounts", "red"),
        ("Top 20% capture", f"{top_two_decile_capture:.1%}", "Holdout churn capture", "green"),
    ]
    model_metrics = _pairs_table(
        {
            "roc_auc": metrics.get("roc_auc", "n/a"),
            "pr_auc": metrics.get("pr_auc", "n/a"),
            "precision": metrics.get("precision", "n/a"),
            "recall": metrics.get("recall", "n/a"),
            "f1": metrics.get("f1", "n/a"),
            "brier_score": metrics.get("brier_score", "n/a"),
        }
    )
    dataset_summary = _pairs_table(
        {
            "rows": quality_report.rows,
            "columns": quality_report.columns,
            "observed_churn_rate": f"{quality_report.churn_rate:.2%}",
            "duplicate_customer_ids": quality_report.duplicate_customer_ids,
            "leakage_warnings": len(quality_report.leakage_warnings),
        }
    )
    recommendation_table = _frame_table(
        top_recommendations[
            [
                id_column,
                "risk_segment",
                "churn_probability",
                "likely_drivers",
                "retention_priority",
            ]
        ],
        max_rows=5,
    )

    cards = "\n".join(
        f"  metric({_q(label)}, {_q(value)}, {_q(note)}, color: {color}),"
        for label, value, note, color in metric_cards
    )
    return f'''#set page(paper: "us-letter", margin: (x: 0.62in, y: 0.58in))
#set text(size: 9.1pt, fill: rgb("#152033"))
#set par(leading: 0.62em, spacing: 0.52em)

#let navy = rgb("#102033")
#let blue = rgb("#2563eb")
#let green = rgb("#059669")
#let amber = rgb("#d97706")
#let red = rgb("#dc2626")
#let border = rgb("#d8dee8")
#let soft = rgb("#f6f8fb")
#let muted = rgb("#64748b")

#let metric(label, value, note, color: blue) = block(fill: soft, stroke: 0.6pt + border, radius: 7pt, inset: 10pt, width: 100%)[
  #text(size: 7.5pt, fill: muted, weight: "bold", label)
  #v(3pt)
  #text(size: 18pt, fill: color, weight: "bold", value)
  #v(2pt)
  #text(size: 7.4pt, fill: muted, note)
]

#let section(title) = [
  #v(8pt)
  #text(size: 13.3pt, weight: "bold", fill: navy, title)
  #v(4pt)
  #line(length: 100%, stroke: 0.7pt + border)
  #v(6pt)
]

#let cell(body, header: false) = table.cell(
  fill: if header {{ navy }} else {{ none }},
  inset: 5pt,
)[#text(size: if header {{ 7.3pt }} else {{ 7.0pt }}, fill: if header {{ white }} else {{ rgb("#152033") }}, weight: if header {{ "bold" }} else {{ "regular" }}, body)]

#align(center)[
  #text(size: 20pt, weight: "bold", fill: navy, "Churn Risk and Retention Report")
  #v(3pt)
  #text(size: 8.7pt, fill: muted, "Validated customer churn scoring from public Telco churn data")
]

#v(9pt)
#grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 8pt,
{cards}
)

#section("Executive Read")
#grid(columns: (1.08fr, 0.92fr), gutter: 12pt,
[
  The model ranks customers by churn risk and turns that score into a retention queue. The highest-risk group should be reviewed first because the top 10% captures {top_decile_capture:.1%} of observed churners on the holdout split.

  This report separates model evidence from the full customer action file. Metrics, lift, calibration, and confidence intervals come from holdout customers; the scored queue covers the full customer base.
],
[
  #image({_q(_image_ref(dashboard_path))}, width: 100%)
])

#section("Dataset Summary")
{dataset_summary}

#section("Model Evidence")
#grid(columns: (0.92fr, 1.08fr), gutter: 12pt,
[
{model_metrics}
],
[
  #image({_q(_image_ref(probability_chart_path))}, width: 100%)
])

#section("Lift and Calibration")
#grid(columns: (1fr, 1fr), gutter: 12pt,
[
{_frame_table(_compact_lift_table(lift_table), max_rows=5)}
],
[
{_frame_table(metric_intervals, max_rows=6)}
  #v(6pt)
{_frame_table(calibration_table.head(5), max_rows=5)}
])

#pagebreak()

#section("Risk Queue")
#grid(columns: (0.95fr, 1.05fr), gutter: 12pt,
[
  #image({_q(_image_ref(risk_chart_path))}, width: 100%)
],
[
{_frame_table(risk_segments, max_rows=5)}
  #v(6pt)
  #text(size: 8pt, fill: muted, "The PDF keeps actions short. Full recommended_action text is delivered in retention_recommendations.csv.")
])

#section("Top Recommended Accounts")
{recommendation_table}

#section("Driver Evidence")
#grid(columns: (1fr, 1fr), gutter: 12pt,
[
  #image({_q(_image_ref(feature_chart_path))}, width: 100%)
],
[
  #image({_q(_image_ref(shap_chart_path))}, width: 100%)
])

#section("Scope")
This is a one-time churn analysis package from a supplied CSV. It does not claim guaranteed accuracy, production CRM integration, scheduled retraining, or hosted monitoring. The useful output is the reproducible evidence bundle plus the customer-level retention queue.
'''



def _compact_lift_table(lift_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "decile",
        "customers",
        "churners",
        "churn_rate",
        "avg_churn_probability",
        "cumulative_churn_capture_rate",
    ]
    return lift_table.loc[:, [column for column in columns if column in lift_table.columns]]


def _pairs_table(values: dict[str, object]) -> str:
    rows = pd.DataFrame(
        [(key, _format_value(value)) for key, value in values.items()],
        columns=["Metric", "Value"],
    )
    return _frame_table(rows, max_rows=len(rows))


def _frame_table(frame: pd.DataFrame, *, max_rows: int) -> str:
    display = frame.head(max_rows).copy()
    display.columns = [_short_column_name(column) for column in display.columns]
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    headers = ", ".join(f"cell({_q(column)}, header: true)" for column in display.columns)
    cells = ",\n  ".join(
        f"cell({_q(value)})"
        for row in display.astype(str).itertuples(index=False)
        for value in row
    )
    column_spec = ", ".join(["1fr"] * len(display.columns))
    return f'''#table(
  columns: ({column_spec}),
  stroke: 0.45pt + border,
  table.header({headers}),
  {cells}
)'''


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _image_ref(path: Path) -> str:
    return path.name


def _q(value: object) -> str:
    return json.dumps(str(value))



def _short_column_name(column: object) -> str:
    names = {
        "avg_churn_probability": "avg_prob",
        "cumulative_churners": "cum_churners",
        "cumulative_churn_capture_rate": "capture",
        "mean_predicted_probability": "mean_pred",
        "observed_churn_rate": "observed_rate",
        "absolute_gap": "gap",
        "bootstrap_samples": "samples",
        "monthly_return_pct": "monthly_return",
        "average_trade": "avg_trade",
        "retention_priority": "priority",
        "churn_probability": "risk",
        "risk_segment": "segment",
        "likely_drivers": "drivers",
    }
    return names.get(str(column), str(column))


def _compile_typst(source_path: Path, output_path: Path) -> None:
    if which("typst") is None:
        raise RuntimeError("Typst CLI is required to build the PDF report.")
    subprocess.run(
        ["typst", "compile", source_path.name, output_path.name],
        cwd=source_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
