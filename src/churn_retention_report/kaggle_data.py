from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

DATASET_SLUG = "blastchar/telco-customer-churn"
RAW_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

app = typer.Typer(help="Download and prepare the Kaggle Telco churn dataset.")


def download_telco_dataset(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_SLUG,
            "-p",
            str(data_dir),
            "--unzip",
        ],
        check=True,
    )
    raw_path = data_dir / RAW_FILENAME
    if not raw_path.exists():
        raise FileNotFoundError(f"Kaggle download completed but {RAW_FILENAME} was not found.")
    return raw_path


def normalize_telco_churn(raw: pd.DataFrame) -> pd.DataFrame:
    total_charges = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)
    support_pressure = (
        (raw["TechSupport"].astype(str).str.lower() == "no").astype(int)
        + (raw["OnlineSecurity"].astype(str).str.lower() == "no").astype(int)
        + (raw["DeviceProtection"].astype(str).str.lower() == "no").astype(int)
    )
    return pd.DataFrame(
        {
            "customer_id": raw["customerID"].astype(str),
            "plan": raw["Contract"].astype(str),
            "internet_service": raw["InternetService"].astype(str),
            "payment_method": raw["PaymentMethod"].astype(str),
            "usage": pd.to_numeric(raw["MonthlyCharges"], errors="coerce").fillna(0.0),
            "payments": total_charges,
            "support_tickets": support_pressure.astype(int),
            "months_active": pd.to_numeric(raw["tenure"], errors="coerce").fillna(0).astype(int),
            "paperless_billing": (
                raw["PaperlessBilling"].astype(str).str.lower() == "yes"
            ).astype(int),
            "senior_citizen": pd.to_numeric(
                raw["SeniorCitizen"],
                errors="coerce",
            ).fillna(0).astype(int),
            "churned": (raw["Churn"].astype(str).str.lower() == "yes").astype(int),
        }
    )


def prepare_telco_churn(raw_path: Path, output_path: Path) -> Path:
    normalized = normalize_telco_churn(pd.read_csv(raw_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False, float_format="%.4f")
    return output_path


@app.command()
def prepare(
    data_dir: Annotated[Path, typer.Option(help="Directory for the Kaggle download.")] = Path(
        "data/raw/kaggle/telco-churn"
    ),
    out: Annotated[Path, typer.Option(help="Prepared CSV path.")] = Path(
        "data/telco_customers.csv"
    ),
    skip_download: Annotated[
        bool,
        typer.Option(help="Use an existing raw Kaggle CSV in data_dir."),
    ] = False,
) -> None:
    raw_path = data_dir / RAW_FILENAME if skip_download else download_telco_dataset(data_dir)
    prepared = prepare_telco_churn(raw_path, out)
    typer.echo(f"Prepared Kaggle Telco churn data at {prepared}")
