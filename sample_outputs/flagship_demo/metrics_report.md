# Churn Prediction Metrics Report

- Rows analyzed: `500`
- Observed churn rate: `23.00%`
- High-risk customers: `33`
- Medium-risk customers: `66`
- Top decile churn capture: `39.13%`
- Top 20% churn capture: `64.35%`

## Model Metrics

- `roc_auc`: `0.77`
- `pr_auc`: `0.5164`
- `baseline_pr_auc`: `0.2267`
- `accuracy`: `0.7667`
- `precision`: `0.4615`
- `recall`: `0.1765`
- `f1`: `0.2553`
- `confusion_matrix`: `[[109, 7], [28, 6]]`

## Top Risk Drivers

- `numeric__usage`: `0.1629` (model_importance)
- `numeric__support_tickets`: `0.1340` (model_importance)
- `numeric__payments`: `0.1307` (model_importance)
- `categorical__plan_Pro`: `0.1115` (model_importance)
- `numeric__nps_score`: `0.1077` (model_importance)
- `numeric__failed_payments`: `0.0919` (model_importance)
- `numeric__months_active`: `0.0885` (model_importance)
- `numeric__discount_percent`: `0.0722` (model_importance)
- `categorical__plan_Basic`: `0.0658` (model_importance)
- `categorical__plan_Enterprise`: `0.0347` (model_importance)

## Recommended Business Read

Use `retention_recommendations.csv` as the action queue. High-risk customers should be reviewed first, then medium-risk customers.

Use `lift_table.csv` to prove whether the model concentrates churners near the top of the risk-ranked customer list.

PR-AUC is included because churn datasets are often imbalanced and precision/recall quality matters more than accuracy alone.
