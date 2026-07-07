"""Monitoring drift summaries."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_drift_summary(current_metrics: pd.DataFrame, baseline_metrics: pd.DataFrame, out_json: Path) -> None:
    summary = {}
    for col in [c for c in current_metrics.columns if c in baseline_metrics.columns]:
        try:
            summary[col] = {
                "current_mean": float(current_metrics[col].mean()),
                "baseline_mean": float(baseline_metrics[col].mean()),
                "delta": float(current_metrics[col].mean() - baseline_metrics[col].mean()),
            }
        except Exception:
            continue
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

