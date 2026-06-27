from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

app = typer.Typer(help="Generate realistic synthetic SaaS churn data.")


def generate_saas_churn_data(rows: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plans = rng.choice(["Basic", "Pro", "Enterprise"], size=rows, p=[0.45, 0.4, 0.15])
    months_active = rng.integers(1, 48, size=rows)
    support_tickets = rng.poisson(lam=np.where(plans == "Enterprise", 1.2, 2.4), size=rows)
    baseline_usage = np.where(plans == "Enterprise", 145, np.where(plans == "Pro", 76, 28))
    usage = (
        rng.normal(loc=baseline_usage, scale=18)
        + months_active * 0.9
        - support_tickets * 2.2
    ).clip(1)
    payments = (months_active * rng.uniform(0.7, 1.05, size=rows)).round().clip(1)
    failed_payments = rng.binomial(3, p=np.where(plans == "Basic", 0.16, 0.07), size=rows)
    nps_score = (
        rng.normal(loc=7.5, scale=1.6, size=rows)
        + usage / 80
        - support_tickets * 0.35
        - failed_payments * 0.5
    ).clip(0, 10)
    discount_percent = rng.choice([0, 5, 10, 20], size=rows, p=[0.55, 0.2, 0.18, 0.07])
    usage_gap = (baseline_usage - usage) / 40
    risk_logit = (
        -0.35
        + usage_gap * 1.15
        - months_active * 0.018
        - nps_score * 0.18
        + support_tickets * 0.32
        + failed_payments * 0.58
        + np.where(plans == "Basic", 0.45, 0)
        - np.where(plans == "Enterprise", 0.25, 0)
        + np.where(discount_percent >= 20, 0.15, 0)
    )
    churn_probability = 1 / (1 + np.exp(-risk_logit))
    churned = rng.binomial(1, churn_probability)
    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:05d}" for index in range(1, rows + 1)],
            "plan": plans,
            "usage": usage.round(2),
            "payments": payments.astype(int),
            "support_tickets": support_tickets.astype(int),
            "months_active": months_active.astype(int),
            "failed_payments": failed_payments.astype(int),
            "nps_score": nps_score.round(1),
            "discount_percent": discount_percent.astype(int),
            "churned": churned.astype(int),
        }
    )


@app.command()
def generate(
    out: Annotated[Path, typer.Option(help="Output CSV path.")] = Path(
        "data/synthetic_saas_customers.csv"
    ),
    rows: Annotated[int, typer.Option(help="Number of rows.")] = 500,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
) -> None:
    frame = generate_saas_churn_data(rows=rows, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    typer.echo(f"Generated {len(frame)} rows at {out}")
