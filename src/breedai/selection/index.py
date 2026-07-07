"""Selection index computation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def compute_selection_index(predictions_csv: Path, weights_yaml: Path, out_csv: Path) -> None:
    pred = pd.read_csv(predictions_csv)
    weights = yaml.safe_load(weights_yaml.read_text()) or {}
    w = weights.get("trait_weights", {})
    pred["weight"] = pred["trait"].map(w).fillna(0.0)
    pred["weighted"] = pred["GEBV"] * pred["weight"]
    out = pred.groupby("animal_id", as_index=False)["weighted"].sum().rename(columns={"weighted": "index_score"})
    out["rank"] = out["index_score"].rank(method="dense", ascending=False).astype(int)
    out = out.sort_values("rank")
    out.to_csv(out_csv, index=False)

