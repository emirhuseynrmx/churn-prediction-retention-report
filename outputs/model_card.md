# Model Card

- Model: `xgboost_classifier`
- Target column: `churned`
- Decision threshold: `0.5`

## Intended Use

Customer churn risk scoring and retention prioritization for business review.

## Limitations

- Accuracy depends on data quality, label quality, and feature relevance.
- Scores should support human retention decisions, not replace them.
- Retrain periodically when product, pricing, or customer behavior changes.

## Metrics

- `roc_auc`: `0.8436`
- `pr_auc`: `0.6562`
- `baseline_pr_auc`: `0.2655`
- `accuracy`: `0.796`
- `precision`: `0.6512`
- `recall`: `0.4991`
- `f1`: `0.5651`
- `confusion_matrix`: `[[1402, 150], [281, 280]]`
- `brier_score`: `0.1361`
- `roc_auc_ci_low`: `0.8267`
- `roc_auc_ci_high`: `0.862`
- `pr_auc_ci_low`: `0.6113`
- `pr_auc_ci_high`: `0.7003`
