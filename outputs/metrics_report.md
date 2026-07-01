# Churn Prediction Metrics Report

- Rows analyzed: `7043`
- Observed churn rate: `26.54%`
- High-risk customers: `456`
- Medium-risk customers: `1586`
- Top decile churn capture: `28.16%`
- Top 20% churn capture: `49.20%`

## Model Metrics

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

## Probability Evidence

- Calibration is calculated on the holdout set only.
- Confidence intervals are bootstrap estimates from holdout predictions.
- `calibration_table.csv` compares predicted probability with observed churn rate.
- `metric_confidence_intervals.csv` reports uncertainty around ROC-AUC and PR-AUC.

## Top Risk Drivers

- `categorical__plan_Month-to-month`: `0.4229` (model_importance)
- `numeric__support_tickets`: `0.1275` (model_importance)
- `categorical__internet_service_Fiber optic`: `0.1117` (model_importance)
- `categorical__payment_method_Electronic check`: `0.0729` (model_importance)
- `numeric__months_active`: `0.0517` (model_importance)
- `categorical__internet_service_No`: `0.0398` (model_importance)
- `categorical__plan_Two year`: `0.0359` (model_importance)
- `numeric__paperless_billing`: `0.0312` (model_importance)
- `categorical__plan_One year`: `0.0299` (model_importance)
- `numeric__usage`: `0.0215` (model_importance)

## Recommended Business Read

Use `retention_recommendations.csv` as the action queue. High-risk customers should be reviewed first, then medium-risk customers.

Use `holdout_lift_table.csv` to prove whether the model concentrates churners near the top of the risk-ranked customer list.

PR-AUC is included because churn datasets are often imbalanced and precision/recall quality matters more than accuracy alone.
