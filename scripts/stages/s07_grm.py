#!/usr/bin/env python3
"""
Stage 7: Genomic Relationship Matrix (GRM).

Thin wrapper around the existing 01a_utils_calculate_gmatrix.py so that
the core dataset builder can call it as a function.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPTS_DIR = Path(__file__).resolve().parent.parent


def _import_gmatrix_util():
    """Dynamically import the existing G-matrix calculator."""
    spec_path = SCRIPTS_DIR / "01a_utils_calculate_gmatrix.py"
    if not spec_path.exists():
        raise FileNotFoundError(f"G-matrix utility not found at {spec_path}")
    spec = importlib.util.spec_from_file_location("gmatrix_util", spec_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compute_grm(
    geno_df: pd.DataFrame,
    method: str = "vanRaden",
    outdir: str | Path | None = None,
) -> dict:
    """
    Compute GRM using the existing BreedAI utility.

    Returns dict with 'G' (numpy array), 'info', and optionally
    writes G-matrix + metadata to outdir.
    """
    mod = _import_gmatrix_util()
    G, info = mod.calculate_gmatrix(
        geno_df,
        method=method,
        standardize=True,
        save_intermediate=bool(outdir),
        output_dir=str(outdir) if outdir else None,
    )

    if outdir:
        od = Path(outdir)
        od.mkdir(parents=True, exist_ok=True)
        np.save(od / "Gmatrix.npy", G)
        pd.DataFrame(G, index=geno_df.index, columns=geno_df.index).to_csv(od / "Gmatrix.csv")
        with open(od / "gmatrix_metadata.json", "w") as f:
            serialisable = {k: v for k, v in info.items() if not isinstance(v, np.ndarray)}
            for k, v in serialisable.items():
                if isinstance(v, (np.integer, np.floating)):
                    serialisable[k] = float(v)
            json.dump(serialisable, f, indent=2, default=str)
        logger.info("GRM saved to %s", od)

    return {"G": G, "info": info, "sample_ids": geno_df.index.tolist()}
