#!/usr/bin/env python3
"""
Stage 12: Selection index.

Combines GEBVs across traits using economic weights to produce a
single aggregate merit index for each animal.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_index(
    gebv_file: str | Path,
    weights: dict[str, float],
    outdir: str | Path,
) -> pd.DataFrame:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(gebv_file)
    if "trait" not in df.columns:
        logger.warning("GEBV file has no 'trait' column — returning empty index")
        return pd.DataFrame()

    traits = df["trait"].unique()
    pivot = df.pivot_table(index="sample_id", columns="trait", values="gebv")

    if not weights:
        logger.info("No economic weights provided — equal-weight index")
        weights = {t: 1.0 for t in traits}

    for t in traits:
        if t not in weights:
            weights[t] = 0.0

    w = np.array([weights.get(t, 0.0) for t in pivot.columns])
    w = w / np.sum(np.abs(w)) if np.sum(np.abs(w)) > 0 else w

    index_vals = pivot.values @ w
    result = pd.DataFrame({"sample_id": pivot.index, "selection_index": index_vals})
    result = result.sort_values("selection_index", ascending=False).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    result.to_csv(od / "selection_index.csv", index=False)

    meta = {"weights_used": weights, "n_animals": len(result), "top5_mean": round(float(result.head(5)["selection_index"].mean()), 4)}
    (od / "selection_index_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Selection index computed: %d animals, top-5 mean=%.4f", len(result), meta["top5_mean"])
    return result
