# Churn Prediction Metrics Report

- Rows analyzed: `7043`
- Observed churn rate: `26.54%`
- High-risk customers: `553`
- Medium-risk customers: `1477`
- Top decile churn capture: `28.88%`
- Top 20% churn capture: `48.13%`

## Model Metrics

- `roc_auc`: `0.8435`
- `pr_auc`: `0.6564`
- `baseline_pr_auc`: `0.2655`
- `accuracy`: `0.7908`
- `precision`: `0.6319`
- `recall`: `0.508`
- `f1`: `0.5632`
- `confusion_matrix`: `[[1386, 166], [276, 285]]`

## Top Risk Drivers

- `categorical__plan_Month-to-month`: `0.5494` (model_importance)
- `categorical__internet_service_Fiber optic`: `0.1439` (model_importance)
- `numeric__support_tickets`: `0.0696` (model_importance)
- `categorical__payment_method_Electronic check`: `0.0495` (model_importance)
- `numeric__months_active`: `0.0338` (model_importance)
- `categorical__plan_Two year`: `0.0315` (model_importance)
- `numeric__paperless_billing`: `0.0220` (model_importance)
- `categorical__plan_One year`: `0.0169` (model_importance)
- `categorical__internet_service_No`: `0.0164` (model_importance)
- `numeric__usage`: `0.0145` (model_importance)

## Recommended Business Read

Use `retention_recommendations.csv` as the action queue. High-risk customers should be reviewed first, then medium-risk customers.

Use `holdout_lift_table.csv` to prove whether the model concentrates churners near the top of the risk-ranked customer list.

PR-AUC is included because churn datasets are often imbalanced and precision/recall quality matters more than accuracy alone.
