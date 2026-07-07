#!/usr/bin/env python3
"""
Stage 10: Default GBLUP training track.

Delegates to the canonical GBLUP implementation in scripts/models/gblup.py.
This wrapper handles the per-trait loop, split creation, and result saving.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from models.gblup import train_and_evaluate, metrics

logger = logging.getLogger(__name__)


def run(
    G: np.ndarray,
    pheno_df: pd.DataFrame,
    sample_ids: list[str],
    outdir: str | Path,
    cv_folds: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Run GBLUP on every trait and save results."""
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    n = len(sample_ids)
    np.random.seed(random_state)
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    all_results = []

    for trait in pheno_df.columns:
        y = pheno_df[trait].values.astype(float)
        mask = ~np.isnan(y)
        ti = np.array([i for i in train_idx if mask[i]])
        vi = np.array([i for i in val_idx if mask[i]])
        tsi = np.array([i for i in test_idx if mask[i]])

        t0 = time.time()
        res = train_and_evaluate(G, y, ti, vi, tsi, use_cv=True)
        elapsed = time.time() - t0

        record = {
            "trait": trait,
            "algorithm": "GBLUP",
            "track": "default",
            "alpha": res["alpha"],
            "train_time_s": round(elapsed, 2),
        }
        record.update({f"val_{k}": v for k, v in res["val_metrics"].items()})
        if "test_metrics" in res:
            record.update({f"test_{k}": v for k, v in res["test_metrics"].items()})
        all_results.append(record)

        trait_dir = od / trait
        trait_dir.mkdir(exist_ok=True)
        save_res = {k: v for k, v in res.items() if k != "model"}
        with open(trait_dir / "gblup_result.json", "w") as f:
            json.dump(save_res, f, indent=2)
        logger.info("GBLUP %s: val_r2=%.4f, val_r=%.4f (%.1fs)", trait, res["val_metrics"]["r2"], res["val_metrics"]["pearson_r"], elapsed)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(od / "gblup_results.csv", index=False)
    logger.info("Default GBLUP results: %s", od / "gblup_results.csv")
    return {"results": all_results, "csv": str(od / "gblup_results.csv")}
