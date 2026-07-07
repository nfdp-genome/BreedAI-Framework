"""GRM utilities (VanRaden)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def vanraden_grm(x: np.ndarray) -> np.ndarray:
    """Compute VanRaden GRM from 0/1/2 genotype matrix."""
    if x.ndim != 2:
        raise ValueError("x must be 2D (animals x SNPs)")
    p = np.nanmean(x, axis=0) / 2.0
    z = x - (2.0 * p)
    denom = 2.0 * np.sum(p * (1.0 - p))
    if denom <= 0:
        raise ValueError("Invalid denominator for VanRaden GRM.")
    return (z @ z.T) / denom


def save_grm_npy(g: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, g)

