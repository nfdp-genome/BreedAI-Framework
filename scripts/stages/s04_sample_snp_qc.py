#!/usr/bin/env python3
"""
Stage 4: Sample and SNP quality control.

For genotype-level entry: operates on Geno.csv (numeric 0/1/2 matrix).
For VCF/FASTQ entry: delegates to the Nextflow step4 pipeline or runs
plink2-based QC inline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_genotype_qc(
    geno_df: pd.DataFrame,
    min_sample_callrate: float = 0.95,
    min_snp_callrate: float = 0.98,
    min_maf: float = 0.01,
    min_variance: float = 0.0001,
    enable_hwe: bool = False,
    hwe_p: float = 1e-6,
) -> dict:
    """
    In-memory QC on a numeric genotype DataFrame (samples × SNPs).
    Returns dict with filtered DataFrame, removed lists, and summary.
    """
    n_samples_in, n_snps_in = geno_df.shape
    logger.info("QC input: %d samples × %d SNPs", n_samples_in, n_snps_in)

    removed_samples = []
    removed_snps = []

    X = geno_df.values.astype(float)
    total = X.shape[1]

    # Sample call rate
    sample_callrate = 1.0 - np.isnan(X).mean(axis=1) if np.any(np.isnan(X)) else np.ones(X.shape[0])
    mask_samples = sample_callrate >= min_sample_callrate
    if not mask_samples.all():
        bad = geno_df.index[~mask_samples].tolist()
        removed_samples.extend(bad)
        geno_df = geno_df.loc[mask_samples]
        X = geno_df.values.astype(float)
        logger.info("Removed %d samples (low call rate)", len(bad))

    # SNP call rate
    snp_callrate = 1.0 - np.isnan(X).mean(axis=0) if np.any(np.isnan(X)) else np.ones(X.shape[1])
    mask_snps = snp_callrate >= min_snp_callrate
    if not mask_snps.all():
        bad = geno_df.columns[~mask_snps].tolist()
        removed_snps.extend(bad)
        geno_df = geno_df.loc[:, mask_snps]
        X = geno_df.values.astype(float)

    # Minor allele frequency
    p = np.nanmean(X, axis=0) / 2.0
    maf = np.minimum(p, 1.0 - p)
    mask_maf = maf >= min_maf
    if not mask_maf.all():
        bad = geno_df.columns[~mask_maf].tolist()
        removed_snps.extend(bad)
        geno_df = geno_df.loc[:, mask_maf]
        X = geno_df.values.astype(float)

    # Variance filter
    var = np.nanvar(X, axis=0)
    mask_var = var >= min_variance
    if not mask_var.all():
        bad = geno_df.columns[~mask_var].tolist()
        removed_snps.extend(bad)
        geno_df = geno_df.loc[:, mask_var]

    n_samples_out, n_snps_out = geno_df.shape
    summary = {
        "n_samples_in": n_samples_in,
        "n_snps_in": n_snps_in,
        "n_samples_out": n_samples_out,
        "n_snps_out": n_snps_out,
        "removed_samples": len(removed_samples),
        "removed_snps": len(set(removed_snps)),
        "thresholds": {
            "min_sample_callrate": min_sample_callrate,
            "min_snp_callrate": min_snp_callrate,
            "min_maf": min_maf,
            "min_variance": min_variance,
        },
    }
    logger.info("QC output: %d samples × %d SNPs", n_samples_out, n_snps_out)
    return {
        "geno_df": geno_df,
        "removed_samples": removed_samples,
        "removed_snps": list(set(removed_snps)),
        "summary": summary,
    }


def save_qc_artifacts(qc_result: dict, outdir: str | Path) -> None:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "qc_summary.json", "w") as f:
        json.dump(qc_result["summary"], f, indent=2)
    pd.Series(qc_result["removed_samples"]).to_csv(od / "removed_samples.csv", index=False, header=["sample_id"])
    pd.Series(qc_result["removed_snps"]).to_csv(od / "removed_snps.csv", index=False, header=["snp_id"])
    logger.info("QC artifacts saved to %s", od)
