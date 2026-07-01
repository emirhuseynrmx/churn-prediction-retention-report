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
    names = list(sampled.columns)
    try:
        values = _build_pipeline_shap_values(model, sampled, config)
    except Exception:
        transformed = model.named_steps["preprocessor"].transform(sampled)
        dense_features = _to_dense_array(transformed)
        classifier = model.named_steps["classifier"]
        values = _build_classifier_shap_values(classifier, dense_features)
        names = feature_names

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    if len(names) != len(mean_abs):
        names = [f"feature_{index}" for index in range(len(mean_abs))]

    return pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "direction": [
                "pushes_churn_up" if value > 0 else "pushes_churn_down"
                for value in mean_signed
            ],
        }
    ).sort_values("mean_abs_shap", ascending=False)


def _build_pipeline_shap_values(
    model: Pipeline,
    sampled: pd.DataFrame,
    config: ChurnConfig,
) -> np.ndarray:
    background = shap.sample(
        sampled,
        nsamples=min(50, len(sampled)),
        random_state=config.random_state,
    )

    def predict_churn_probability(values: np.ndarray | pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(values, columns=sampled.columns)
        return model.predict_proba(frame)[:, 1]

    explainer = shap.Explainer(
        predict_churn_probability,
        background,
        algorithm="permutation",
    )
    explanation = explainer(sampled, max_evals=max(2 * sampled.shape[1] + 1, 10))
    return np.asarray(explanation.values)


def _build_classifier_shap_values(
    classifier: object,
    dense_features: np.ndarray,
) -> np.ndarray:
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
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])
    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values
