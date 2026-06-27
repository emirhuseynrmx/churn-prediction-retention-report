from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn_retention_report.config import ChurnConfig
from churn_retention_report.pipeline import run_churn_pipeline
from churn_retention_report.schema import validate_input_frame


def test_validate_input_frame_accepts_sample_data() -> None:
    config = ChurnConfig.load(Path("examples/config.json"))
    frame = pd.read_csv("data/sample_customers.csv")

    validated = validate_input_frame(frame, config)

    assert len(validated) == 30
    assert validated[config.target_column].nunique() == 2


def test_run_churn_pipeline_writes_client_deliverables(tmp_path: Path) -> None:
    config = ChurnConfig.load(Path("examples/config.json"))

    result = run_churn_pipeline(Path("data/sample_customers.csv"), tmp_path, config)

    assert result.predictions_path.exists()
    assert result.risk_segments_path.exists()
    assert result.lift_table_path.exists()
    assert result.feature_importance_path.exists()
    assert result.shap_importance_path.exists()
    assert result.recommendations_path.exists()
    assert result.metrics_report_path.exists()
    assert result.data_quality_report_path.exists()
    assert result.model_card_path.exists()
    assert result.pdf_report_path.exists()
    assert result.manifest_path.exists()
    assert result.config_snapshot_path.exists()
    assert result.risk_chart_path.exists()
    assert result.feature_chart_path.exists()
    assert result.shap_chart_path.exists()
    assert result.probability_chart_path.exists()
    assert result.dashboard_path.exists()
    assert result.model_path.exists()
    assert "roc_auc" in result.metrics


def test_run_churn_pipeline_supports_logistic_explainability(tmp_path: Path) -> None:
    config = ChurnConfig.load(Path("examples/config.json")).model_copy(
        update={"model_name": "balanced_logistic_regression"}
    )

    result = run_churn_pipeline(Path("data/sample_customers.csv"), tmp_path, config)

    assert result.shap_importance_path.exists()
    assert result.shap_chart_path.exists()
