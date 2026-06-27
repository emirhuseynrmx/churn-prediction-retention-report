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

- `roc_auc`: `0.77`
- `pr_auc`: `0.5164`
- `baseline_pr_auc`: `0.2267`
- `accuracy`: `0.7667`
- `precision`: `0.4615`
- `recall`: `0.1765`
- `f1`: `0.2553`
- `confusion_matrix`: `[[109, 7], [28, 6]]`
