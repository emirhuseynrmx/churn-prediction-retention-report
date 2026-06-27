from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn_retention_report.config import ChurnConfig


def build_shap_importance(
    *,
    model: Pipeline,
    feature_frame: pd.DataFrame,
    feature_names: list[str],
    config: ChurnConfig,
) -> pd.DataFrame:
    sampled = feature_frame.sample(
        n=min(config.shap_sample_size, len(feature_frame)),
        random_state=config.random_state,
    )
    transformed = model.named_steps["preprocessor"].transform(sampled)
    dense_features = _to_dense_array(transformed)
    classifier = model.named_steps["classifier"]
    values = _build_explanation_values(classifier, dense_features)
    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    return pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "direction": [
                "pushes_churn_up" if value > 0 else "pushes_churn_down"
                for value in mean_signed
            ],
        }
    ).sort_values("mean_abs_shap", ascending=False)


def _build_explanation_values(classifier: object, dense_features: np.ndarray) -> np.ndarray:
    if isinstance(classifier, XGBClassifier):
        explainer = shap.TreeExplainer(classifier)
        return _select_binary_values(explainer.shap_values(dense_features))
    if hasattr(classifier, "coef_"):
        return _linear_shap_values(classifier, dense_features)
    raise TypeError(
        "Unsupported model for explainability. Use xgboost_classifier or "
        "balanced_logistic_regression."
    )


def _linear_shap_values(classifier: object, dense_features: np.ndarray) -> np.ndarray:
    try:
        explainer = shap.LinearExplainer(classifier, dense_features)
        return _select_binary_values(explainer.shap_values(dense_features))
    except Exception:
        coefficients = np.asarray(classifier.coef_)[0]
        return dense_features * coefficients


def _to_dense_array(features: object) -> np.ndarray:
    if hasattr(features, "toarray"):
        return features.toarray()
    return np.asarray(features)


def _select_binary_values(shap_values: object) -> np.ndarray:
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    return values
