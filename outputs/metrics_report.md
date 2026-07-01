# Churn Prediction Metrics Report

- Rows analyzed: `7043`
- Observed churn rate: `26.54%`
- High-risk customers: `539`
- Medium-risk customers: `1475`
- Top decile churn capture: `29.23%`
- Top 20% churn capture: `48.48%`

## Model Metrics

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

## Probability Evidence

- Calibration is calculated on the holdout set only.
- Confidence intervals are bootstrap estimates from holdout predictions.
- `calibration_table.csv` compares predicted probability with observed churn rate.
- `metric_confidence_intervals.csv` reports uncertainty around ROC-AUC and PR-AUC.

## Top Risk Drivers

- `categorical__plan_Month-to-month`: `0.5483` (model_importance)
- `categorical__internet_service_Fiber optic`: `0.1418` (model_importance)
- `numeric__support_tickets`: `0.0725` (model_importance)
- `categorical__payment_method_Electronic check`: `0.0470` (model_importance)
- `numeric__months_active`: `0.0330` (model_importance)
- `categorical__plan_Two year`: `0.0309` (model_importance)
- `numeric__paperless_billing`: `0.0245` (model_importance)
- `categorical__internet_service_No`: `0.0171` (model_importance)
- `categorical__plan_One year`: `0.0170` (model_importance)
- `numeric__usage`: `0.0150` (model_importance)

## Recommended Business Read

Use `retention_recommendations.csv` as the action queue. High-risk customers should be reviewed first, then medium-risk customers.

Use `holdout_lift_table.csv` to prove whether the model concentrates churners near the top of the risk-ranked customer list.

PR-AUC is included because churn datasets are often imbalanced and precision/recall quality matters more than accuracy alone.
