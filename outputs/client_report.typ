#set page(paper: "us-letter", margin: (x: 0.62in, y: 0.58in))
#set text(size: 9.1pt, fill: rgb("#152033"))
#set par(leading: 0.62em, spacing: 0.52em)

#let navy = rgb("#102033")
#let blue = rgb("#2563eb")
#let green = rgb("#059669")
#let amber = rgb("#d97706")
#let red = rgb("#dc2626")
#let border = rgb("#d8dee8")
#let soft = rgb("#f6f8fb")
#let muted = rgb("#64748b")

#let metric(label, value, note, color: blue) = block(fill: soft, stroke: 0.6pt + border, radius: 7pt, inset: 10pt, width: 100%)[
  #text(size: 7.5pt, fill: muted, weight: "bold", label)
  #v(3pt)
  #text(size: 18pt, fill: color, weight: "bold", value)
  #v(2pt)
  #text(size: 7.4pt, fill: muted, note)
]

#let section(title) = [
  #v(8pt)
  #text(size: 13.3pt, weight: "bold", fill: navy, title)
  #v(4pt)
  #line(length: 100%, stroke: 0.7pt + border)
  #v(6pt)
]

#let cell(body, header: false) = table.cell(
  fill: if header { navy } else { none },
  inset: 5pt,
)[#text(size: if header { 7.3pt } else { 7.0pt }, fill: if header { white } else { rgb("#152033") }, weight: if header { "bold" } else { "regular" }, body)]

#align(center)[
  #text(size: 20pt, weight: "bold", fill: navy, "Churn Risk and Retention Report")
  #v(3pt)
  #text(size: 8.7pt, fill: muted, "Validated customer churn scoring from public Telco churn data")
]

#v(9pt)
#grid(columns: (1fr, 1fr, 1fr, 1fr), gutter: 8pt,
  metric("Rows analyzed", "7,043", "Validated customer rows", color: blue),
  metric("Observed churn", "26.5%", "Historical target rate", color: amber),
  metric("High risk", "539", "Priority accounts", color: red),
  metric("Top 20% capture", "48.5%", "Holdout churn capture", color: green),
)

#section("Executive Read")
#grid(columns: (1.08fr, 0.92fr), gutter: 12pt,
[
  The model ranks customers by churn risk and turns that score into a retention queue. The highest-risk group should be reviewed first because the top 10% captures 29.2% of observed churners on the holdout split.

  This report separates model evidence from the full customer action file. Metrics, lift, calibration, and confidence intervals come from holdout customers; the scored queue covers the full customer base.
],
[
  #image("executive_dashboard.png", width: 100%)
])

#section("Dataset Summary")
#table(
  columns: (1fr, 1fr),
  stroke: 0.45pt + border,
  table.header(cell("Metric", header: true), cell("Value", header: true)),
  cell("rows"),
  cell("7043"),
  cell("columns"),
  cell("11"),
  cell("observed_churn_rate"),
  cell("26.54%"),
  cell("duplicate_customer_ids"),
  cell("0"),
  cell("leakage_warnings"),
  cell("0")
)

#section("Model Evidence")
#grid(columns: (0.92fr, 1.08fr), gutter: 12pt,
[
#table(
  columns: (1fr, 1fr),
  stroke: 0.45pt + border,
  table.header(cell("Metric", header: true), cell("Value", header: true)),
  cell("roc_auc"),
  cell("0.8437"),
  cell("pr_auc"),
  cell("0.6564"),
  cell("precision"),
  cell("0.6358"),
  cell("recall"),
  cell("0.5134"),
  cell("f1"),
  cell("0.5680"),
  cell("brier_score"),
  cell("0.1360")
)
],
[
  #image("probability_distribution.png", width: 100%)
])

#section("Lift and Calibration")
#grid(columns: (1fr, 1fr), gutter: 12pt,
[
#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr, 1fr),
  stroke: 0.45pt + border,
  table.header(cell("decile", header: true), cell("customers", header: true), cell("churners", header: true), cell("churn_rate", header: true), cell("avg_prob", header: true), cell("capture", header: true)),
  cell("1"),
  cell("212"),
  cell("164"),
  cell("0.7736"),
  cell("0.7440"),
  cell("0.2923"),
  cell("2"),
  cell("211"),
  cell("108"),
  cell("0.5118"),
  cell("0.5796"),
  cell("0.4848"),
  cell("3"),
  cell("211"),
  cell("99"),
  cell("0.4692"),
  cell("0.4534"),
  cell("0.6613"),
  cell("4"),
  cell("211"),
  cell("76"),
  cell("0.3602"),
  cell("0.3327"),
  cell("0.7968"),
  cell("5"),
  cell("212"),
  cell("45"),
  cell("0.2123"),
  cell("0.2302"),
  cell("0.8770")
)
],
[
#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  stroke: 0.45pt + border,
  table.header(cell("metric", header: true), cell("estimate", header: true), cell("ci_low", header: true), cell("ci_high", header: true), cell("samples", header: true)),
  cell("roc_auc"),
  cell("0.8437"),
  cell("0.8252"),
  cell("0.8620"),
  cell("300"),
  cell("pr_auc"),
  cell("0.6564"),
  cell("0.6105"),
  cell("0.6995"),
  cell("300")
)
  #v(6pt)
#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  stroke: 0.45pt + border,
  table.header(cell("bin", header: true), cell("customers", header: true), cell("mean_pred", header: true), cell("observed_rate", header: true), cell("gap", header: true)),
  cell("1"),
  cell("212"),
  cell("0.7440"),
  cell("0.7736"),
  cell("0.0295"),
  cell("2"),
  cell("211"),
  cell("0.5796"),
  cell("0.5118"),
  cell("0.0677"),
  cell("3"),
  cell("211"),
  cell("0.4534"),
  cell("0.4692"),
  cell("0.0158"),
  cell("4"),
  cell("211"),
  cell("0.3327"),
  cell("0.3602"),
  cell("0.0274"),
  cell("5"),
  cell("211"),
  cell("0.2304"),
  cell("0.2133"),
  cell("0.0171")
)
])

#pagebreak()

#section("Risk Queue")
#grid(columns: (0.95fr, 1.05fr), gutter: 12pt,
[
  #image("risk_segments.png", width: 100%)
],
[
#table(
  columns: (1fr, 1fr, 1fr),
  stroke: 0.45pt + border,
  table.header(cell("segment", header: true), cell("customers", header: true), cell("avg_prob", header: true)),
  cell("high"),
  cell("539"),
  cell("0.7801"),
  cell("medium"),
  cell("1475"),
  cell("0.5454"),
  cell("low"),
  cell("5029"),
  cell("0.1281")
)
  #v(6pt)
  #text(size: 8pt, fill: muted, "The PDF keeps actions short. Full recommended_action text is delivered in retention_recommendations.csv.")
])

#section("Top Recommended Accounts")
#table(
  columns: (1fr, 1fr, 1fr, 1fr, 1fr),
  stroke: 0.45pt + border,
  table.header(cell("customer_id", header: true), cell("segment", header: true), cell("risk", header: true), cell("drivers", header: true), cell("priority", header: true)),
  cell("7216-EWTRS"),
  cell("high"),
  cell("0.9119"),
  cell("general_engagement"),
  cell("1"),
  cell("0107-YHINA"),
  cell("high"),
  cell("0.9048"),
  cell("general_engagement"),
  cell("1"),
  cell("9497-QCMMS"),
  cell("high"),
  cell("0.8992"),
  cell("general_engagement"),
  cell("1"),
  cell("9300-AGZNL"),
  cell("high"),
  cell("0.8992"),
  cell("general_engagement"),
  cell("1"),
  cell("0295-PPHDO"),
  cell("high"),
  cell("0.8991"),
  cell("general_engagement"),
  cell("1")
)

#section("Driver Evidence")
#grid(columns: (1fr, 1fr), gutter: 12pt,
[
  #image("feature_importance.png", width: 100%)
],
[
  #image("shap_summary.png", width: 100%)
])

#section("Scope")
This is a one-time churn analysis package from a supplied CSV. It does not claim guaranteed accuracy, production CRM integration, scheduled retraining, or hosted monitoring. The useful output is the reproducible evidence bundle plus the customer-level retention queue.
