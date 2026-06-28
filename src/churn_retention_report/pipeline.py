from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from churn_retention_report.artifacts import (
    OutputManifest,
    write_config_snapshot,
    write_manifest,
    write_model_card,
)
from churn_retention_report.config import ChurnConfig
from churn_retention_report.contracts import (
    validate_feature_importance,
    validate_lift_table,
    validate_predictions,
    validate_recommendations,
    validate_risk_segments,
    validate_shap_importance,
)
from churn_retention_report.explainability import build_shap_importance
from churn_retention_report.modeling import train_model
from churn_retention_report.outputs import write_csv, write_metrics_report, write_model
from churn_retention_report.pdf_report import write_pdf_report
from churn_retention_report.quality import profile_data_quality, write_data_quality_report
from churn_retention_report.recommendations import (
    build_lift_table,
    build_predictions,
    build_recommendations,
    build_risk_segments,
)
from churn_retention_report.schema import validate_input_frame
from churn_retention_report.visuals import (
    write_dashboard_image,
    write_feature_importance_chart,
    write_probability_distribution_chart,
    write_risk_chart,
    write_shap_summary_chart,
)


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    metrics: dict[str, object]
    predictions_path: Path
    risk_segments_path: Path
    lift_table_path: Path
    feature_importance_path: Path
    shap_importance_path: Path
    recommendations_path: Path
    metrics_report_path: Path
    data_quality_report_path: Path
    model_card_path: Path
    pdf_report_path: Path
    manifest_path: Path
    config_snapshot_path: Path
    risk_chart_path: Path
    feature_chart_path: Path
    shap_chart_path: Path
    probability_chart_path: Path
    dashboard_path: Path
    model_path: Path


def run_churn_pipeline(input_path: Path, output_dir: Path, config: ChurnConfig) -> PipelineResult:
    frame = validate_input_frame(pd.read_csv(input_path), config)
    artifacts = train_model(frame, config)
    feature_frame = frame.drop(columns=[config.id_column, config.target_column])
    probabilities = artifacts.model.predict_proba(feature_frame)[:, 1]
    predictions = build_predictions(frame, probabilities, config)
    holdout_frame = frame.loc[artifacts.holdout_index]
    holdout_predictions = build_predictions(
        holdout_frame,
        artifacts.holdout_probabilities,
        config,
    )
    risk_segments = build_risk_segments(predictions)
    lift_table = build_lift_table(holdout_predictions, holdout_frame, config)
    recommendations = build_recommendations(
        predictions,
        frame,
        artifacts.feature_importance,
        config,
    )
    shap_importance = build_shap_importance(
        model=artifacts.model,
        feature_frame=feature_frame,
        feature_names=artifacts.feature_names,
        config=config,
    )
    feature_importance = validate_feature_importance(artifacts.feature_importance)
    shap_importance = validate_shap_importance(shap_importance)
    predictions = validate_predictions(predictions, config)
    risk_segments = validate_risk_segments(risk_segments)
    lift_table = validate_lift_table(lift_table)
    recommendations = validate_recommendations(recommendations, config)
    quality_report = profile_data_quality(frame, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = write_csv(predictions, output_dir / "predictions.csv")
    risk_segments_path = write_csv(risk_segments, output_dir / "risk_segments.csv")
    lift_table_path = write_csv(lift_table, output_dir / "holdout_lift_table.csv")
    feature_importance_path = write_csv(
        feature_importance,
        output_dir / "feature_importance.csv",
    )
    shap_importance_path = write_csv(
        shap_importance,
        output_dir / "shap_feature_importance.csv",
    )
    risk_chart_path = write_risk_chart(predictions, output_dir / "risk_segments.png")
    feature_chart_path = write_feature_importance_chart(
        feature_importance,
        output_dir / "feature_importance.png",
    )
    shap_chart_path = write_shap_summary_chart(
        shap_importance,
        output_dir / "shap_summary.png",
    )
    probability_chart_path = write_probability_distribution_chart(
        predictions=predictions,
        medium_threshold=config.risk_thresholds.medium,
        high_threshold=config.risk_thresholds.high,
        path=output_dir / "probability_distribution.png",
    )
    recommendations_path = write_csv(
        recommendations,
        output_dir / "retention_recommendations.csv",
    )
    dashboard_path = write_dashboard_image(
        predictions=predictions,
        risk_segments=risk_segments,
        feature_importance=feature_importance,
        recommendations=recommendations,
        path=output_dir / "executive_dashboard.png",
    )
    model_path = write_model(artifacts.model, output_dir / "model.joblib")
    metrics_report_path = write_metrics_report(
        metrics=artifacts.metrics,
        output_path=output_dir / "metrics_report.md",
        rows=len(frame),
        churn_rate=float(frame[config.target_column].mean()),
        high_risk_count=int((predictions["risk_segment"] == "high").sum()),
        medium_risk_count=int((predictions["risk_segment"] == "medium").sum()),
        top_features=feature_importance,
        lift_table=lift_table,
    )
    data_quality_report_path = write_data_quality_report(
        quality_report,
        output_dir / "data_quality_report.md",
    )
    model_card_path = write_model_card(
        config=config,
        metrics=artifacts.metrics,
        path=output_dir / "model_card.md",
    )
    pdf_report_path = write_pdf_report(
        output_path=output_dir / "client_report.pdf",
        metrics=artifacts.metrics,
        quality_report=quality_report,
        risk_segments=risk_segments,
        lift_table=lift_table,
        recommendations=recommendations,
        risk_chart_path=risk_chart_path,
        feature_chart_path=feature_chart_path,
        shap_chart_path=shap_chart_path,
        probability_chart_path=probability_chart_path,
        dashboard_path=dashboard_path,
    )
    config_snapshot_path = write_config_snapshot(config, output_dir / "config_snapshot.json")
    manifest_path = write_manifest(
        OutputManifest(
            files={
                "predictions": str(predictions_path),
                "risk_segments": str(risk_segments_path),
                "holdout_lift_table": str(lift_table_path),
                "feature_importance": str(feature_importance_path),
                "shap_feature_importance": str(shap_importance_path),
                "retention_recommendations": str(recommendations_path),
                "metrics_report": str(metrics_report_path),
                "data_quality_report": str(data_quality_report_path),
                "model_card": str(model_card_path),
                "pdf_report": str(pdf_report_path),
                "risk_chart": str(risk_chart_path),
                "feature_chart": str(feature_chart_path),
                "shap_summary": str(shap_chart_path),
                "probability_distribution": str(probability_chart_path),
                "executive_dashboard": str(dashboard_path),
                "model": str(model_path),
                "config_snapshot": str(config_snapshot_path),
            },
            model_name=config.model_name,
            validated_outputs=[
                "predictions",
                "risk_segments",
                "holdout_lift_table",
                "feature_importance",
                "shap_feature_importance",
                "retention_recommendations",
            ],
        ),
        output_dir / "manifest.json",
    )
    return PipelineResult(
        output_dir=output_dir,
        metrics=artifacts.metrics,
        predictions_path=predictions_path,
        risk_segments_path=risk_segments_path,
        lift_table_path=lift_table_path,
        feature_importance_path=feature_importance_path,
        shap_importance_path=shap_importance_path,
        recommendations_path=recommendations_path,
        metrics_report_path=metrics_report_path,
        data_quality_report_path=data_quality_report_path,
        model_card_path=model_card_path,
        pdf_report_path=pdf_report_path,
        manifest_path=manifest_path,
        config_snapshot_path=config_snapshot_path,
        risk_chart_path=risk_chart_path,
        feature_chart_path=feature_chart_path,
        shap_chart_path=shap_chart_path,
        probability_chart_path=probability_chart_path,
        dashboard_path=dashboard_path,
        model_path=model_path,
    )
