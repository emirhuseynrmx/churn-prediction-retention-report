from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
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
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]

    if isinstance(classifier, XGBClassifier):
        preprocessor.transform(sampled)
        importances = np.asarray(classifier.feature_importances_)
        names = feature_names if len(feature_names) == len(importances) else [
            f"feature_{i}" for i in range(len(importances))
        ]
        mean_abs = importances
        mean_signed = importances
    elif hasattr(classifier, "coef_"):
        transformed = preprocessor.transform(sampled)
        dense = _to_dense_array(transformed)
        coef = np.asarray(classifier.coef_)[0]
        names = feature_names if len(feature_names) == len(coef) else [
            f"feature_{i}" for i in range(len(coef))
        ]
        mean_signed = coef
        mean_abs = np.abs(coef)
    else:
        result = permutation_importance(
            model, sampled, sampled.index,
            n_repeats=5, random_state=config.random_state,
        )
        names = list(sampled.columns)
        mean_abs = result.importances_mean
        mean_signed = result.importances_mean

    return pd.DataFrame({
        "feature": names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
        "direction": [
            "pushes_churn_up" if v > 0 else "pushes_churn_down"
            for v in mean_signed
        ],
    }).sort_values("mean_abs_shap", ascending=False)


def _to_dense_array(features: object) -> np.ndarray:
    if hasattr(features, "toarray"):
        return features.toarray()
    return np.asarray(features)
