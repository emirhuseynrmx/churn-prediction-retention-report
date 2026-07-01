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

- `roc_auc`: `0.8437`
- `pr_auc`: `0.6564`
- `baseline_pr_auc`: `0.2655`
- `accuracy`: `0.7927`
- `precision`: `0.6358`
- `recall`: `0.5134`
- `f1`: `0.568`
- `confusion_matrix`: `[[1387, 165], [273, 288]]`
- `brier_score`: `0.136`
- `roc_auc_ci_low`: `0.8252`
- `roc_auc_ci_high`: `0.862`
- `pr_auc_ci_low`: `0.6105`
- `pr_auc_ci_high`: `0.6995`
