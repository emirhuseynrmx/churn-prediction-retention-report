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

- `roc_auc`: `0.8435`
- `pr_auc`: `0.6564`
- `baseline_pr_auc`: `0.2655`
- `accuracy`: `0.7908`
- `precision`: `0.6319`
- `recall`: `0.508`
- `f1`: `0.5632`
- `confusion_matrix`: `[[1386, 166], [276, 285]]`
