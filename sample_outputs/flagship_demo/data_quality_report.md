# Data Quality Report

- Rows: `7043`
- Columns: `11`
- Churn rate: `26.54%`
- Duplicate customer ids: `0`

## Missing Values

- `customer_id`: `0`
- `plan`: `0`
- `internet_service`: `0`
- `payment_method`: `0`
- `usage`: `0`
- `payments`: `0`
- `support_tickets`: `0`
- `months_active`: `0`
- `paperless_billing`: `0`
- `senior_citizen`: `0`
- `churned`: `0`

## Feature Types

- Numeric: `usage, payments, support_tickets, months_active, paperless_billing, senior_citizen`
- Categorical: `plan, internet_service, payment_method`

## Leakage Warnings

- No obvious post-churn leakage columns detected.
