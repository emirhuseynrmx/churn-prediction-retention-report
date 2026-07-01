from __future__ import annotations

from dataclasses import dataclass

import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn_retention_report.config import ChurnConfig
from churn_retention_report.features import build_preprocessor, split_feature_types

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(frozen=True)
class ModelArtifacts:
    model: Pipeline
    metrics: dict[str, float | list[list[int]]]
    feature_names: list[str]
    feature_importance: pd.DataFrame
    holdout_index: pd.Index
    holdout_probabilities: pd.Series


def _optuna_objective(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: object,
    random_state: int,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "eval_metric": "logloss",
        "random_state": random_state,
        "verbosity": 0,
    }
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", XGBClassifier(**params))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipe, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def train_model(frame: pd.DataFrame, config: ChurnConfig) -> ModelArtifacts:
    feature_frame = frame.drop(columns=[config.id_column, config.target_column])
    target = frame[config.target_column].astype(int)
    numeric_features, categorical_features = split_feature_types(feature_frame)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=target,
    )

    if config.model_name == "xgboost_classifier":
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.random_state),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
        )
        study.optimize(
            lambda t: _optuna_objective(t, x_train, y_train, preprocessor, config.random_state),
            n_trials=40,
            show_progress_bar=False,
        )
        best = study.best_params
        classifier: LogisticRegression | XGBClassifier = XGBClassifier(
            **best,
            eval_metric="logloss",
            random_state=config.random_state,
            verbosity=0,
        )
    else:
        classifier = _build_classifier(config)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= config.decision_threshold).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "baseline_pr_auc": round(float(y_test.mean()), 4),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }
    feature_names = list(model.named_steps["preprocessor"].get_feature_names_out())
    importance = _build_feature_importance(model, feature_names)
    return ModelArtifacts(
        model=model,
        metrics=metrics,
        feature_names=feature_names,
        feature_importance=importance,
        holdout_index=x_test.index,
        holdout_probabilities=pd.Series(probabilities, index=x_test.index),
    )


def _build_classifier(config: ChurnConfig) -> LogisticRegression | XGBClassifier:
    if config.model_name == "balanced_logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config.random_state,
        )
    xgb = config.xgboost
    return XGBClassifier(
        colsample_bytree=xgb.colsample_bytree,
        eval_metric="logloss",
        learning_rate=xgb.learning_rate,
        max_depth=xgb.max_depth,
        n_estimators=xgb.n_estimators,
        random_state=config.random_state,
        reg_lambda=xgb.reg_lambda,
        subsample=xgb.subsample,
    )


def _build_feature_importance(model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "coef_"):
        coefficients = classifier.coef_[0]
        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": abs(coefficients),
                "direction": [
                    "increases_risk" if value > 0 else "decreases_risk"
                    for value in coefficients
                ],
            }
        ).sort_values("importance", ascending=False)
    importances = classifier.feature_importances_
    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
            "direction": ["model_importance" for _ in feature_names],
        }
    ).sort_values("importance", ascending=False)
