# Leakage Policy

This pipeline is designed for customer-level churn scoring before the churn event is known.

Allowed inputs should describe account state before the prediction point:

- plan and billing method
- tenure or months active
- usage
- support pressure
- payment friction known before churn
- account attributes known at scoring time

Columns that describe the churn event or later recovery flow are flagged as leakage risks:

- `cancel_date`
- `cancellation_reason`
- `refund_after_cancel`
- `winback_status`
- `closed_reason`
- any post-churn support note or label

Model performance is measured on a holdout split. Customer-level recommendations can be produced
for the full customer base after the holdout metrics are computed.
