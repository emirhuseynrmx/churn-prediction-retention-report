from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from churn_retention_report.config import ChurnConfig


class OutputManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    files: dict[str, str]
    model_name: str
    validated_outputs: list[str]


def write_manifest(manifest: OutputManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_model_card(
    *,
    config: ChurnConfig,
    metrics: dict[str, object],
    path: Path,
) -> Path:
    lines = [
        "# Model Card",
        "",
        f"- Model: `{config.model_name}`",
        f"- Target column: `{config.target_column}`",
        f"- Decision threshold: `{config.decision_threshold}`",
        "",
        "## Intended Use",
        "",
        "Customer churn risk scoring and retention prioritization for business review.",
        "",
        "## Limitations",
        "",
        "- Accuracy depends on data quality, label quality, and feature relevance.",
        "- Scores should support human retention decisions, not replace them.",
        "- Retrain periodically when product, pricing, or customer behavior changes.",
        "",
        "## Metrics",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"- `{key}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_config_snapshot(config: ChurnConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path
