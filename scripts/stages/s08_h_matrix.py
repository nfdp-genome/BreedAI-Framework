#!/usr/bin/env python3
"""
Stage 8: H-matrix construction for ssGBLUP.

If a pedigree is available, blends A (numerator relationship matrix)
and G into H.  Otherwise, records a skip with metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_A_matrix(pedigree_df: pd.DataFrame) -> np.ndarray:
    """
    Build numerator relationship matrix (A) from a pedigree DataFrame
    with columns [animal, sire, dam].  Missing parents coded as '0' or NaN.

    Uses tabular method (Henderson, 1976).
    """
    ped = pedigree_df.copy()
    ped.columns = ["animal", "sire", "dam"]
    ped = ped.fillna("0")
    animals = ped["animal"].tolist()
    n = len(animals)
    idx = {a: i for i, a in enumerate(animals)}

    A = np.zeros((n, n))
    for i, row in ped.iterrows():
        aid = row["animal"]
        sid = row["sire"]
        did = row["dam"]
        ai = idx[aid]

        si = idx.get(sid)
        di = idx.get(did)

        if si is not None and di is not None:
            A[ai, ai] = 1.0 + 0.5 * A[si, di]
        elif si is not None:
            A[ai, ai] = 1.0
        elif di is not None:
            A[ai, ai] = 1.0
        else:
            A[ai, ai] = 1.0

        for j in range(ai):
            s_val = A[j, si] if si is not None else 0.0
            d_val = A[j, di] if di is not None else 0.0
            A[ai, j] = 0.5 * (s_val + d_val)
            A[j, ai] = A[ai, j]

    return A


def build_H(G: np.ndarray, A: np.ndarray, omega: float = 0.05) -> np.ndarray:
    """
    Blend G and A into H for ssGBLUP.
    H = (1-omega)*G + omega*A  (simplified Aguilar et al., 2010 form).
    Both matrices must share the same sample ordering.
    """
    return (1.0 - omega) * G + omega * A


def run(
    G: np.ndarray,
    sample_ids: list[str],
    pedigree_file: str | None,
    outdir: str | Path,
) -> dict:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    if not pedigree_file or not Path(pedigree_file).exists():
        logger.info("No pedigree — H-matrix skipped; using G only")
        meta = {"status": "skipped", "reason": "no_pedigree"}
        (od / "h_matrix_skip.json").write_text(json.dumps(meta, indent=2))
        return {"H": G, "status": "skipped"}

    logger.info("Building A-matrix from pedigree: %s", pedigree_file)
    ped = pd.read_csv(pedigree_file)
    A_full = build_A_matrix(ped)

    ped_ids = ped.iloc[:, 0].astype(str).tolist()
    common = [s for s in sample_ids if s in ped_ids]
    if len(common) < len(sample_ids):
        logger.warning("Only %d / %d samples found in pedigree", len(common), len(sample_ids))

    ped_idx = {a: i for i, a in enumerate(ped_ids)}
    reorder = [ped_idx[s] for s in sample_ids if s in ped_idx]
    A = A_full[np.ix_(reorder, reorder)]

    H = build_H(G, A)
    np.save(od / "H_matrix.npy", H)
    pd.DataFrame(H, index=sample_ids, columns=sample_ids).to_csv(od / "H_matrix.csv")
    logger.info("H-matrix written to %s", od)
    return {"H": H, "status": "done"}
