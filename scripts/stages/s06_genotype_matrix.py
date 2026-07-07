#!/usr/bin/env python3
"""
Stage 6: Genotype matrix generation and standardisation.

For genotype-level entry: validates the numeric matrix, ensures 0/1/2 coding,
and writes the canonical genotype file for downstream stages.
For VCF entry: delegates to plink2 --make-pgen + export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_geno(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_csv(p, index_col=0)
    df.index = df.index.astype(str)
    logger.info("Loaded genotype matrix: %d samples × %d SNPs from %s", *df.shape, p.name)
    return df


def validate_coding(geno_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure values are in {0, 1, 2, NaN}."""
    X = geno_df.values.astype(float)
    valid = np.isnan(X) | np.isin(X, [0, 1, 2])
    if not valid.all():
        n_bad = (~valid).sum()
        logger.warning("%d cells not in {0,1,2,NaN}; clipping to [0,2]", n_bad)
        X = np.clip(np.nan_to_num(X, nan=np.nan), 0, 2)
        geno_df = pd.DataFrame(X, index=geno_df.index, columns=geno_df.columns)
    return geno_df


def save_canonical(geno_df: pd.DataFrame, outdir: str | Path, filename: str = "genotypes.csv") -> Path:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    out = od / filename
    geno_df.to_csv(out)
    logger.info("Canonical genotype matrix written: %s (%d × %d)", out, *geno_df.shape)
    return out


def build_snp_metadata(geno_df: pd.DataFrame) -> pd.DataFrame:
    X = geno_df.values.astype(float)
    p = np.nanmean(X, axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)
    callrate = 1.0 - np.isnan(X).mean(axis=0) if np.any(np.isnan(X)) else np.ones(X.shape[1])
    var = np.nanvar(X, axis=0)
    meta = pd.DataFrame({
        "snp_id": geno_df.columns,
        "allele_freq": p,
        "maf": maf,
        "callrate": callrate,
        "variance": var,
    })
    return meta
