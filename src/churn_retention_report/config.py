from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    high: float = Field(default=0.7, gt=0, lt=1)
    medium: float = Field(default=0.4, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> RiskThresholds:
        if self.high <= self.medium:
            raise ValueError("High risk threshold must be greater than medium risk threshold.")
        return self


class ChurnConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_column: str = "churned"
    id_column: str = "customer_id"
    positive_label: int = 1
    test_size: float = Field(default=0.3, gt=0, lt=1)
    random_state: int = 42
    model_name: Literal["xgboost_classifier", "balanced_logistic_regression"] = (
        "xgboost_classifier"
    )
    decision_threshold: float = Field(default=0.5, gt=0, lt=1)
    top_recommendation_count: int = Field(default=10, ge=1)
    lift_table_deciles: int = Field(default=10, ge=5, le=20)
    shap_sample_size: int = Field(default=300, ge=25, le=5000)
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    recommendation_rules: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> ChurnConfig:
        if path is None:
            return cls()
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
