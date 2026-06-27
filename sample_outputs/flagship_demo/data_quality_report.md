# Data Quality Report

- Rows: `500`
- Columns: `10`
- Churn rate: `23.00%`
- Duplicate customer ids: `0`

## Missing Values

- `customer_id`: `0`
- `plan`: `0`
- `usage`: `0`
- `payments`: `0`
- `support_tickets`: `0`
- `months_active`: `0`
- `failed_payments`: `0`
- `nps_score`: `0`
- `discount_percent`: `0`
- `churned`: `0`

## Feature Types

- Numeric: `usage, payments, support_tickets, months_active, failed_payments, nps_score, discount_percent`
- Categorical: `plan`

## Leakage Warnings

- No obvious post-churn leakage columns detected.
