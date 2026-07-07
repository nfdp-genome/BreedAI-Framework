#!/usr/bin/env python3
"""
Stage 11: GEBV prediction and reliability estimation.

Computes Genomic Estimated Breeding Values for all animals and
estimates reliability via prediction error variance (PEV) approximation
or cross-validation-based approach.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


def predict_gebv(
    G: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Fit GBLUP on all data and return GEBV for every animal.
    """
    model = Ridge(alpha=alpha)
    model.fit(G, y)
    gebv = model.predict(G)
    return pd.DataFrame({"sample_id": sample_ids, "gebv": gebv})


def estimate_reliability_cv(
    G: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    alpha: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Per-animal reliability approximated as 1 - PEV/Var(y) via
    leave-one-fold-out prediction residuals.
    """
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    pred = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(G):
        m = Ridge(alpha=alpha)
        m.fit(G[np.ix_(train_idx, train_idx)], y[train_idx])
        pred[test_idx] = m.predict(G[np.ix_(test_idx, train_idx)])

    residuals = y - pred
    var_y = np.var(y)
    pev = residuals ** 2
    reliability = np.clip(1.0 - pev / var_y, 0.0, 1.0) if var_y > 0 else np.zeros(len(y))
    return reliability


def run(
    G: np.ndarray,
    pheno_df: pd.DataFrame,
    sample_ids: list[str],
    outdir: str | Path,
    alpha: float = 1.0,
    n_folds: int = 5,
) -> dict:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    all_gebv = []
    for trait in pheno_df.columns:
        y = pheno_df[trait].values.astype(float)
        mask = ~np.isnan(y)
        ids_valid = [sample_ids[i] for i in range(len(y)) if mask[i]]
        G_valid = G[np.ix_(mask, mask)]
        y_valid = y[mask]

        gebv_df = predict_gebv(G_valid, y_valid, ids_valid, alpha)
        gebv_df["trait"] = trait

        rel = estimate_reliability_cv(G_valid, y_valid, n_folds, alpha)
        gebv_df["reliability"] = rel

        gebv_df.to_csv(od / f"gebv_{trait}.csv", index=False)
        all_gebv.append(gebv_df)
        logger.info("GEBV %s: mean_rel=%.3f, mean_gebv=%.4f", trait, rel.mean(), gebv_df["gebv"].mean())

    combined = pd.concat(all_gebv, ignore_index=True)
    combined.to_csv(od / "gebv_all_traits.csv", index=False)

    summary = {
        "traits": list(pheno_df.columns),
        "n_animals": len(sample_ids),
        "mean_reliability": {t: round(float(combined[combined.trait == t]["reliability"].mean()), 4) for t in pheno_df.columns},
    }
    (od / "gebv_summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "csv": str(od / "gebv_all_traits.csv")}
