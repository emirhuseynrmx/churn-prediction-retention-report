# Churn Prediction Retention Report

[![CI](https://github.com/emirhuseynrmx/churn-prediction-retention-report/actions/workflows/ci.yml/badge.svg)](https://github.com/emirhuseynrmx/churn-prediction-retention-report/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

This is a small production-style churn reporting pipeline.

The goal is simple: take a customer CSV, validate it, train a churn model, score every customer, and export files a business can actually use. It is not trying to be a full SaaS product, live dashboard, or MLOps platform.

It is closer to the kind of one-time client delivery I would build for a churn analysis project: clean inputs, reproducible config, clear outputs, a PDF report, and an action queue for retention work.

## What It Produces

Given a customer file with columns like:

```text
customer_id, plan, usage, payments, support_tickets, churned
```

the pipeline writes:

- `predictions.csv`
- `risk_segments.csv`
- `lift_table.csv`
- `feature_importance.csv`
- `shap_feature_importance.csv`
- `retention_recommendations.csv`
- `metrics_report.md`
- `data_quality_report.md`
- `model_card.md`
- `client_report.pdf`
- `manifest.json`
- `config_snapshot.json`
- `risk_segments.png`
- `feature_importance.png`
- `shap_summary.png`
- `probability_distribution.png`
- `executive_dashboard.png`
- `model.joblib`

## Why I Built It This Way

Most churn projects should not stop at `customer_id, churn_probability`.

The useful output is a ranked retention queue:

```text
customer_id
churn_probability
risk_segment
likely_drivers
recommended_action
retention_priority
```

That is the part a founder, customer success team, or operator can open and act on.

The repo also includes the boring parts that usually make client work safer:

- Pydantic config for column mapping and thresholds
- Pandera validation for inputs and exported CSVs
- minimum retained/churned class count checks
- risk threshold validation
- leakage warnings for suspicious post-churn columns
- XGBoost as the default model
- logistic regression fallback
- SHAP explanations for XGBoost and logistic configs
- ROC AUC, PR-AUC, baseline PR-AUC, precision, recall, F1, confusion matrix
- lift table with top 10% and top 20% churn capture
- probability distribution sanity chart
- model card, config snapshot, output manifest
- Dockerfile for reproducible runs

## Run It

```bash
pip install -e ".[dev]"
churn-report data/sample_customers.csv --config examples/config.json --out outputs/demo
```

Generate a larger synthetic SaaS dataset:

```bash
churn-generate-sample --rows 500 --out data/synthetic_saas_customers.csv
churn-report data/synthetic_saas_customers.csv --config examples/config.json --out outputs/saas_demo
```

Docker:

```bash
docker build -t churn-retention-report .
docker run --rm -v "%cd%/outputs:/app/outputs" churn-retention-report
```

## Example Output

Sample outputs are committed under `sample_outputs/flagship_demo/` so the report shape is visible without running the code first.

![Sample churn retention PDF report](docs/assets/sample-churn-report-preview.png)

![Risk segment distribution](docs/assets/risk_segments.png)

![Feature importance chart](docs/assets/feature_importance.png)

![SHAP explainability chart](docs/assets/shap_summary.png)

![Probability distribution chart](docs/assets/probability_distribution.png)

![Executive dashboard](docs/assets/executive_dashboard.png)

## Output Contract

Every client-facing CSV is validated before export:

- `predictions.csv`: customer id, churn probability, risk segment
- `risk_segments.csv`: segment size and average risk
- `lift_table.csv`: decile-level churn capture and cumulative lift
- `feature_importance.csv`: model feature importance
- `shap_feature_importance.csv`: SHAP attribution table
- `retention_recommendations.csv`: customer action queue with priority

## Project Notes

This package is for a one-time churn analysis and retention report from a provided CSV.

It does not include API deployment, CRM integration, scheduled retraining, database extraction, hosted dashboards, or guaranteed model accuracy. Those are separate scopes.

Client data should not be committed to a public repository. Sensitive columns can be removed or anonymized before modeling.

## Docs

- [Architecture](docs/architecture.md)
- [Case Study](docs/case_study.md)
- [Client Intake Form](docs/client_intake_form.md)
- [Scope And Privacy](docs/scope_and_privacy.md)
