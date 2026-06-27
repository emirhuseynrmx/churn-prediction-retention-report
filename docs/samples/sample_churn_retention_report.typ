#set page(
  paper: "us-letter",
  margin: (x: 0.62in, y: 0.58in),
)

#set text(font: "Arial", size: 9.2pt, fill: rgb("#152033"))
#set par(leading: 0.62em, spacing: 0.5em)

#let navy = rgb("#102033")
#let blue = rgb("#2563eb")
#let green = rgb("#059669")
#let amber = rgb("#d97706")
#let red = rgb("#dc2626")
#let border = rgb("#d8dee8")
#let soft = rgb("#f6f8fb")
#let ink = rgb("#152033")
#let muted = rgb("#64748b")

#let pill(label, color) = box(
  inset: (x: 7pt, y: 3pt),
  radius: 999pt,
  fill: color.lighten(88%),
  stroke: 0.5pt + color.lighten(55%),
  text(size: 7.5pt, fill: color.darken(20%), weight: "bold", label),
)

#let metric(label, value, note, color: blue) = block(
  fill: soft,
  stroke: 0.6pt + border,
  radius: 7pt,
  inset: 10pt,
  width: 100%,
)[
  #text(size: 7.5pt, fill: muted, weight: "bold", label)
  #v(3pt)
  #text(size: 18pt, fill: color, weight: "bold", value)
  #v(2pt)
  #text(size: 7.5pt, fill: muted, note)
]

#let section(title) = [
  #v(8pt)
  #text(size: 13.5pt, weight: "bold", fill: navy, title)
  #v(4pt)
  #line(length: 100%, stroke: 0.7pt + border)
  #v(6pt)
]

#let smalltable(cols, rows) = table(
  columns: cols,
  inset: 5.5pt,
  stroke: 0.45pt + border,
  align: horizon,
  table.header(
    ..rows.at(0).map(cell => table.cell(fill: navy, text(size: 7.4pt, fill: white, weight: "bold", cell))),
  ),
  ..rows.slice(1).flatten().map(cell => text(size: 7.2pt, fill: ink, cell)),
)

#align(center)[
  #text(size: 21pt, weight: "bold", fill: navy, "Sample Churn Risk & Retention Report")
  #v(3pt)
  #text(size: 9pt, fill: muted, "Client-ready churn analysis package generated from the demo pipeline")
  #v(7pt)
  #pill("XGBoost", blue)
  #h(5pt)
  #pill("SHAP explainability", green)
  #h(5pt)
  #pill("PDF + CSV deliverables", amber)
]

#v(10pt)

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  metric("Rows analyzed", "500", "Synthetic SaaS customer file"),
  metric("Observed churn", "23.0%", "Historical target rate", color: amber),
  metric("High-risk customers", "33", "Priority accounts", color: red),
  metric("Top 20% capture", "64.35%", "Churners found in top risk bands", color: green),
)

#section("Executive Read")

#grid(
  columns: (1.1fr, 0.9fr),
  gutter: 12pt,
  [
    The model ranks customers by churn risk and turns the score into a retention queue. The highest-risk segment should be reviewed first because it concentrates the largest expected churn probability.

    This sample report is not just a model output. The delivery includes a data quality check, model metrics, lift analysis, SHAP-based driver explanations, and a customer-level action file.

    #v(4pt)
    #text(weight: "bold", "Recommended next action:")
    start with the top-risk customers, review their likely drivers, and assign outreach by priority.
  ],
  block(fill: soft, stroke: 0.6pt + border, radius: 7pt, inset: 10pt)[
    #text(size: 10.5pt, weight: "bold", fill: navy, "Delivered files")
    #v(5pt)
    #text(size: 7.8pt)[
      predictions.csv \
      retention_recommendations.csv \
      risk_segments.csv \
      lift_table.csv \
      feature_importance.csv \
      shap_feature_importance.csv \
      data_quality_report.md \
      metrics_report.md \
      client_report.pdf
    ]
  ],
)

#section("Risk Segment Snapshot")

#grid(
  columns: (1fr, 1.02fr),
  gutter: 12pt,
  [
    #image("/outputs/flagship_demo/risk_segments.png", width: 100%)
  ],
  [
    #smalltable(
      (1.1fr, 0.8fr, 1.2fr),
      (
        ("Segment", "Customers", "Avg. churn probability"),
        ("High", "33", "77.61%"),
        ("Medium", "66", "53.84%"),
        ("Low", "401", "12.43%"),
      ),
    )
    #v(8pt)
    #text(size: 8.2pt, fill: muted)[
      The high-risk group is intentionally small. The point is not to label everyone as risky, but to identify the accounts that deserve immediate retention attention.
    ]
  ],
)

#pagebreak()

#section("Model Proof")

#grid(
  columns: (1fr, 1fr, 1fr, 1fr),
  gutter: 8pt,
  metric("ROC AUC", "0.77", "Ranking quality"),
  metric("PR-AUC", "0.516", "Useful for imbalanced churn"),
  metric("Precision", "0.462", "High-risk predictions"),
  metric("F1", "0.255", "Precision/recall balance"),
)

#v(6pt)

#grid(
  columns: (0.95fr, 1.05fr),
  gutter: 12pt,
  [
    #text(size: 11pt, weight: "bold", fill: navy, "Lift analysis")
    #v(5pt)
    #smalltable(
      (0.5fr, 0.7fr, 0.7fr, 0.9fr),
      (
        ("Decile", "Customers", "Churners", "Capture"),
        ("1", "50", "45", "39.13%"),
        ("2", "50", "29", "64.35%"),
        ("3", "50", "19", "80.87%"),
        ("4", "50", "9", "88.70%"),
        ("5", "50", "5", "93.04%"),
      ),
    )
    #v(7pt)
    #text(size: 8.2pt, fill: muted)[
      The top decile contains 39.13% of all churners. The top 20% contains 64.35%. This is the business case for ranked retention work.
    ]
  ],
  [
    #image("/outputs/flagship_demo/probability_distribution.png", width: 100%)
    #v(4pt)
    #text(size: 7.6pt, fill: muted, "Probability distribution helps catch overconfident or collapsed scoring.")
  ],
)

#section("Top Recommended Accounts")

#smalltable(
  (0.9fr, 0.65fr, 0.8fr, 1.5fr, 0.55fr),
  (
    ("Customer", "Segment", "Risk", "Likely drivers", "Priority"),
    ("CUST-00335", "High", "92.95%", "usage, support tickets, payments", "1"),
    ("CUST-00084", "High", "89.02%", "usage, support tickets, payments", "1"),
    ("CUST-00027", "High", "85.32%", "usage, support tickets, payments", "1"),
    ("CUST-00323", "High", "85.00%", "usage, payments, plan", "1"),
    ("CUST-00039", "High", "83.31%", "usage, support tickets, payments", "1"),
  ),
)

#v(8pt)
#block(fill: rgb("#fff7ed"), stroke: 0.7pt + rgb("#fed7aa"), radius: 7pt, inset: 9pt)[
  #text(weight: "bold", fill: amber.darken(20%), "How to use this queue")
  #v(3pt)
  #text(size: 8.2pt)[
    High-priority accounts should receive targeted intervention before renewal risk increases. The full CSV includes the recommended action text for every scored customer.
  ]
]

#pagebreak()

#section("Explainability")

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #image("/outputs/flagship_demo/feature_importance.png", width: 100%)
  ],
  [
    #image("/outputs/flagship_demo/shap_summary.png", width: 100%)
  ],
)

#v(5pt)
#text(size: 8.2pt, fill: muted)[
  Feature importance shows the model's strongest global signals. SHAP adds directionality so the report can explain why risk moves up or down instead of only returning a score.
]

#section("Data Quality Check")

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  [
    #smalltable(
      (1.25fr, 0.75fr),
      (
        ("Check", "Result"),
        ("Rows", "500"),
        ("Columns", "10"),
        ("Duplicate customer ids", "0"),
        ("Missing values", "0"),
        ("Leakage warnings", "None detected"),
      ),
    )
  ],
  block(fill: soft, stroke: 0.6pt + border, radius: 7pt, inset: 10pt)[
    #text(size: 10.5pt, weight: "bold", fill: navy, "Scope note")
    #v(4pt)
    #text(size: 8.2pt)[
      This package is designed for a one-time churn analysis from a provided CSV. It does not include live CRM integration, scheduled retraining, or guaranteed model accuracy. Those are separate scopes.
    ]
  ],
)

#section("What Makes This Delivery Different")

#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  block(fill: soft, stroke: 0.6pt + border, radius: 7pt, inset: 9pt)[
    #text(weight: "bold", fill: navy, "Business-ready outputs")
    #v(3pt)
    #text(size: 8.2pt, "PDF report, ranked action queue, charts, and CSV exports.")
  ],
  block(fill: soft, stroke: 0.6pt + border, radius: 7pt, inset: 9pt)[
    #text(weight: "bold", fill: navy, "Reproducible package")
    #v(3pt)
    #text(size: 8.2pt, "Config snapshot, model card, manifest, validation, and source code.")
  ],
)
