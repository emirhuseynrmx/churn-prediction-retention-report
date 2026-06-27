# Case Study: SaaS Churn Risk Report

This case study uses a reproducible synthetic SaaS dataset. It is realistic enough to show the full delivery flow without exposing private customer data.

## Client Scenario

A subscription business wants to identify accounts likely to cancel and prioritize retention actions.

Expected input:

```text
customer_id, plan, usage, payments, support_tickets, months_active,
failed_payments, nps_score, discount_percent, churned
```

## Proof Of Work Command

```bash
churn-generate-sample --rows 500 --seed 42 --out data/synthetic_saas_customers.csv
churn-report data/synthetic_saas_customers.csv --config examples/config.json --out outputs/flagship_demo
```

## Delivered Files

- `predictions.csv`: one churn probability per customer.
- `risk_segments.csv`: high, medium, and low risk segment summary.
- `lift_table.csv`: proof that the model concentrates churners in the highest-risk deciles.
- `feature_importance.csv`: model-level risk drivers.
- `shap_feature_importance.csv`: SHAP/XAI attribution table for explainability.
- `retention_recommendations.csv`: account-level action queue.
- `metrics_report.md`: ROC AUC, precision, recall, F1, accuracy, confusion matrix.
- `data_quality_report.md`: missing values, duplicate IDs, churn rate, feature coverage.
- `model_card.md`: model purpose, limits, intended use, and metrics.
- `manifest.json`: generated file inventory and output contracts.
- `client_report.pdf`: business-ready PDF report for non-technical stakeholders.
- `risk_segments.png`: visual risk segment proof.
- `feature_importance.png`: visual risk driver proof.
- `shap_summary.png`: visual model explainability proof.
- `probability_distribution.png`: probability sanity check for score distribution.
- `executive_dashboard.png`: one-page snapshot for sales calls and handoff decks.

## Service Packaging

This is the level of work that can justify a premium offer because the client receives a complete decision package, not a loose script.

- Standard churn report: `$750+`
- Advanced churn and retention package: `$1,500-$3,000`
- Recurring refresh/API/dashboard handoff: custom quote

## Business Read

The model ranks customers by churn risk. The top-risk segment should be reviewed first because it concentrates the highest expected churn probability.

Use `lift_table.csv` to answer the practical retention question: if the team focuses on the highest-risk customers first, how much observed churn does that list capture?

Use `retention_recommendations.csv` as the operating queue. It turns churn probabilities into customer-level actions with likely drivers and retention priority.

## Honest Limits

This project does not guarantee a specific accuracy number. Churn quality depends on the client's historical labels, data completeness, and how strongly the available features explain cancellation behavior.
