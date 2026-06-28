from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

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
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )
    top_decile_capture = float(
        lift_table.sort_values("decile").iloc[0]["cumulative_churn_capture_rate"]
    )
    top_two_decile_capture = float(
        lift_table.sort_values("decile").head(2).iloc[-1]["cumulative_churn_capture_rate"]
    )
    high_risk_row = risk_segments.loc[risk_segments["risk_segment"] == "high"]
    high_risk_count = int(high_risk_row["customers"].iloc[0]) if not high_risk_row.empty else 0
    id_column = recommendations.columns[0]
    top_recommendations = recommendations.sort_values(
        ["retention_priority", "churn_probability"],
        ascending=[True, False],
    ).head(5)
    story = [
        Paragraph("Churn Prediction + Retention Report", styles["Title"]),
        Paragraph(
            "Executive summary: this report ranks customers by churn risk and turns "
            "model output into a prioritized retention action queue.",
            styles["BodyText"],
        ),
        Paragraph(
            "The model ranks customers by churn risk. The top-risk segment should be "
            "reviewed first because it concentrates the highest expected churn probability.",
            styles["BodyText"],
        ),
        Spacer(1, 0.15 * inch),
        Image(str(dashboard_path), width=6.7 * inch, height=4.25 * inch),
        Spacer(1, 0.18 * inch),
        Paragraph("Dataset Summary", styles["Heading2"]),
        _table_from_pairs(
            {
                "rows": quality_report.rows,
                "columns": quality_report.columns,
                "observed_churn_rate": f"{quality_report.churn_rate:.2%}",
                "duplicate_customer_ids": quality_report.duplicate_customer_ids,
                "leakage_warnings": len(quality_report.leakage_warnings),
            }
        ),
        Spacer(1, 0.18 * inch),
        Paragraph("Model Metrics", styles["Heading2"]),
        _table_from_pairs(metrics),
        Spacer(1, 0.18 * inch),
        Paragraph("Risk Segment Breakdown", styles["Heading2"]),
        _dataframe_table(risk_segments),
        Spacer(1, 0.18 * inch),
        Paragraph("Lift Table Interpretation", styles["Heading2"]),
        Paragraph(
            f"The highest-risk 10% captures {top_decile_capture:.1%} of observed churners. "
            f"The highest-risk 20% captures {top_two_decile_capture:.1%}. "
            f"Review the {high_risk_count} high-risk customers first.",
            styles["BodyText"],
        ),
        _dataframe_table(lift_table.head(5)),
        Spacer(1, 0.18 * inch),
        Paragraph("Probability Evidence", styles["Heading2"]),
        Paragraph(
            "Calibration and confidence intervals are computed from holdout customers only, "
            "so model evidence stays separate from the full customer action queue.",
            styles["BodyText"],
        ),
        _dataframe_table(metric_intervals),
        Spacer(1, 0.08 * inch),
        _dataframe_table(calibration_table.head(6), font_size=7),
        Spacer(1, 0.18 * inch),
        Paragraph("Top Recommended Actions", styles["Heading2"]),
        _dataframe_table(
            top_recommendations[
                [
                    id_column,
                    "risk_segment",
                    "churn_probability",
                    "likely_drivers",
                    "retention_priority",
                ]
            ],
            font_size=7,
        ),
        Spacer(1, 0.08 * inch),
        Paragraph(
            "Detailed recommended_action text is delivered in "
            "retention_recommendations.csv so the PDF stays readable.",
            styles["BodyText"],
        ),
        Spacer(1, 0.18 * inch),
        Paragraph("Recommended Retention Strategy", styles["Heading2"]),
        Paragraph(
            "Use the retention recommendation CSV as the operating queue. High-risk "
            "customers should receive direct outreach; medium-risk customers should "
            "receive targeted education, usage review, or plan-fit checks.",
            styles["BodyText"],
        ),
        Spacer(1, 0.18 * inch),
        Paragraph("Visual Evidence", styles["Heading2"]),
        Image(str(risk_chart_path), width=3.2 * inch, height=1.8 * inch),
        Spacer(1, 0.1 * inch),
        Image(str(feature_chart_path), width=5.8 * inch, height=3.3 * inch),
        Spacer(1, 0.1 * inch),
        Paragraph("Probability Sanity Check", styles["Heading2"]),
        Image(str(probability_chart_path), width=5.8 * inch, height=3.3 * inch),
        Spacer(1, 0.1 * inch),
        Paragraph("SHAP Explainability", styles["Heading2"]),
        Image(str(shap_chart_path), width=5.8 * inch, height=3.3 * inch),
        Spacer(1, 0.18 * inch),
        Paragraph("Limitations", styles["Heading2"]),
        Paragraph(
            "This report does not guarantee a specific accuracy number. Results depend "
            "on data volume, feature quality, label quality, and whether post-churn "
            "leakage columns were excluded before modeling.",
            styles["BodyText"],
        ),
        Spacer(1, 0.18 * inch),
        Paragraph("Next Steps", styles["Heading2"]),
        Paragraph(
            "Recommended next steps: review leakage warnings, confirm the business "
            "definition of churn, enrich weak features, run the action queue, and compare "
            "retention outcomes after the next billing cycle.",
            styles["BodyText"],
        ),
        Spacer(1, 0.18 * inch),
        Paragraph("Scope and Privacy", styles["Heading2"]),
        Paragraph(
            "This repository keeps raw private datasets out of version control. The "
            "checked-in sample artifacts are reproducible from the included public-data "
            "workflow and are meant to show the report shape without exposing private data.",
            styles["BodyText"],
        ),
    ]
    doc.build(story)
    return output_path


def _table_from_pairs(metrics: dict[str, object]) -> Table:
    rows = [["Metric", "Value"]]
    rows.extend([[key, str(value)] for key, value in metrics.items()])
    return _styled_table(rows)


def _dataframe_table(frame: pd.DataFrame, font_size: int = 8) -> Table:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.4f}")
    rows = [display.columns.tolist()]
    rows.extend(display.astype(str).values.tolist())
    return _styled_table(rows, font_size=font_size)


def _styled_table(rows: list[list[object]], font_size: int = 8) -> Table:
    col_widths = None
    column_count = len(rows[0]) if rows else 0
    if column_count == 2:
        col_widths = [2.3 * inch, 3.2 * inch]
    elif column_count > 2:
        col_widths = [6.5 * inch / column_count] * column_count
    table = Table(rows, repeatRows=1, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0969da")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0d7de")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f8fa")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table
