from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn_retention_report.config import ChurnConfig


def build_shap_importance(
    *,
    model: Pipeline,
    feature_frame: pd.DataFrame,
    feature_names: list[str],
    config: ChurnConfig,
) -> pd.DataFrame:
    import shap

    sampled = feature_frame.sample(
        n=min(config.shap_sample_size, len(feature_frame)),
        random_state=config.random_state,
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_transformed = preprocessor.transform(sampled)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    transformed_names = list(preprocessor.get_feature_names_out())

    # TreeExplainer is ~10x faster than KernelExplainer and handles XGBoost natively
    explainer = shap.TreeExplainer(classifier)
    shap_obj = explainer(X_transformed)

    values = shap_obj.values
    if values.ndim == 3:
        # Binary classification: (n_samples, n_features, 2) — take churn class
        values = values[:, :, 1]

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)

    names = (
        transformed_names
        if len(transformed_names) == values.shape[1]
        else feature_names
    )

    return pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "direction": [
                "pushes_churn_up" if v > 0 else "pushes_churn_down"
                for v in mean_signed
            ],
        }
    ).sort_values("mean_abs_shap", ascending=False)
