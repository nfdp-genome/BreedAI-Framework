#!/usr/bin/env python3
"""
Stage 13: Monitoring and drift detection.

Compares the current run's genotype/phenotype distributions against
a baseline to flag drift.  Generates a summary report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compare_distributions(
    current_geno: pd.DataFrame,
    baseline_geno: pd.DataFrame | None,
    current_pheno: pd.DataFrame,
    baseline_pheno: pd.DataFrame | None,
) -> dict:
    report: dict = {"genotype_drift": {}, "phenotype_drift": {}}

    if baseline_geno is not None:
        curr_maf = np.nanmean(current_geno.values.astype(float), axis=0) / 2.0
        base_maf = np.nanmean(baseline_geno.values.astype(float), axis=0) / 2.0
        common = min(len(curr_maf), len(base_maf))
        delta = np.abs(curr_maf[:common] - base_maf[:common])
        report["genotype_drift"] = {
            "mean_maf_delta": round(float(np.mean(delta)), 6),
            "max_maf_delta": round(float(np.max(delta)), 6),
            "n_snps_compared": int(common),
        }

    if baseline_pheno is not None:
        for col in current_pheno.columns:
            if col in baseline_pheno.columns:
                c = current_pheno[col].dropna()
                b = baseline_pheno[col].dropna()
                report["phenotype_drift"][col] = {
                    "mean_shift": round(float(c.mean() - b.mean()), 4),
                    "std_ratio": round(float(c.std() / b.std()) if b.std() > 0 else 0.0, 4),
                }

    return report


def run(
    current_geno: pd.DataFrame,
    current_pheno: pd.DataFrame,
    baseline_run: str | None,
    outdir: str | Path,
) -> dict:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    base_geno = None
    base_pheno = None
    if baseline_run and Path(baseline_run).exists():
        core = Path(baseline_run) / "core_dataset"
        gf = core / "genotypes.csv"
        pf = core / "phenotypes.csv"
        if gf.exists():
            base_geno = pd.read_csv(gf, index_col=0)
        if pf.exists():
            base_pheno = pd.read_csv(pf, index_col=0)

    report = compare_distributions(current_geno, base_geno, current_pheno, base_pheno)

    if not baseline_run:
        report["note"] = "No baseline provided — first run, drift analysis skipped"

    (od / "monitoring_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Monitoring report: %s", od / "monitoring_report.json")
    return report
