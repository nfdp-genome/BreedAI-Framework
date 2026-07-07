#!/usr/bin/env python3
"""
Stage 5: Genotype imputation.

Backends:
  - beagle: Beagle 5.x via java -jar. Requires a reference panel VCF
    from the training set.  Operates through a VCF round-trip.
  - mean_fill: per-SNP mean (2*p) from provided allele frequencies.
    Used as explicit fallback when Beagle is unavailable or disabled.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BEAGLE_JAR_ENV = "BREEDAI_BEAGLE_JAR"


def _find_beagle_jar() -> str | None:
    jar = os.environ.get(BEAGLE_JAR_ENV, "").strip()
    if jar and Path(jar).exists():
        return jar
    for candidate in [
        shutil.which("beagle"),
        shutil.which("beagle.jar"),
        Path.home() / "bin" / "beagle.jar",
        Path("'${BEAGLE_JAR:-beagle.jar}'"),
    ]:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def is_beagle_available() -> bool:
    return _find_beagle_jar() is not None and shutil.which("java") is not None


def mean_fill(
    geno_df: pd.DataFrame,
    allele_freq: np.ndarray | None = None,
) -> pd.DataFrame:
    """Per-SNP mean imputation. If allele_freq is given, uses 2*p; else column mean."""
    X = geno_df.values.astype(float)
    if allele_freq is not None and len(allele_freq) == X.shape[1]:
        fill_vals = 2.0 * allele_freq
    else:
        fill_vals = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        if mask.any():
            X[mask, j] = fill_vals[j]
    return pd.DataFrame(X, index=geno_df.index, columns=geno_df.columns)


def _df_to_vcf(geno_df: pd.DataFrame, vcf_path: Path, chrom: str = "1") -> None:
    gt_map = {0.0: "0/0", 1.0: "0/1", 2.0: "1/1"}
    with vcf_path.open("w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        header += geno_df.index.tolist()
        f.write("\t".join(header) + "\n")
        for j, snp in enumerate(geno_df.columns):
            row = [chrom, str(j + 1), snp, "A", "C", ".", "PASS", ".", "GT"]
            for val in geno_df.iloc[:, j]:
                row.append(gt_map.get(val, "./.") if not np.isnan(val) else "./.")
            f.write("\t".join(row) + "\n")


def _vcf_to_df(vcf_path: Path, sample_ids: list[str]) -> pd.DataFrame:
    gt_map = {"0/0": 0, "0|0": 0, "0/1": 1, "0|1": 1, "1/0": 1, "1|0": 1, "1/1": 2, "1|1": 2}
    snp_ids = []
    data = []
    with vcf_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            snp_ids.append(parts[2])
            gts = [gt_map.get(g.split(":")[0], np.nan) for g in parts[9:]]
            data.append(gts)
    X = np.array(data, dtype=float).T
    return pd.DataFrame(X, index=sample_ids, columns=snp_ids)


def run_beagle(
    geno_df: pd.DataFrame,
    ref_geno_df: pd.DataFrame | None = None,
    outdir: str | Path = "",
    allele_freq: np.ndarray | None = None,
) -> dict:
    """
    Run Beagle imputation through a VCF round-trip.

    geno_df:     target genotypes with NaN for missing
    ref_geno_df: training genotypes (used as reference panel)
    """
    jar = _find_beagle_jar()
    if not jar:
        logger.warning("Beagle jar not found — falling back to mean-fill")
        return {"geno_df": mean_fill(geno_df, allele_freq), "method": "mean_fill", "status": "fallback_no_jar"}

    od = Path(outdir) if outdir else Path(tempfile.mkdtemp(prefix="beagle_"))
    od.mkdir(parents=True, exist_ok=True)

    target_vcf = od / "target.vcf"
    _df_to_vcf(geno_df, target_vcf)

    cmd = ["java", "-Xmx4g", "-jar", jar, f"gt={target_vcf}", f"out={od / 'imputed'}"]

    if ref_geno_df is not None:
        ref_vcf = od / "ref.vcf"
        _df_to_vcf(ref_geno_df, ref_vcf)
        cmd.append(f"ref={ref_vcf}")

    logger.info("Running Beagle: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            logger.warning("Beagle failed (rc=%d): %s", result.returncode, result.stderr[:500])
            return {"geno_df": mean_fill(geno_df, allele_freq), "method": "mean_fill", "status": "fallback_beagle_error"}
    except subprocess.TimeoutExpired:
        logger.warning("Beagle timed out — falling back to mean-fill")
        return {"geno_df": mean_fill(geno_df, allele_freq), "method": "mean_fill", "status": "fallback_timeout"}

    imputed_vcf = od / "imputed.vcf.gz"
    if not imputed_vcf.exists():
        imputed_vcf = od / "imputed.vcf"
    if not imputed_vcf.exists():
        logger.warning("Beagle output not found — falling back to mean-fill")
        return {"geno_df": mean_fill(geno_df, allele_freq), "method": "mean_fill", "status": "fallback_no_output"}

    # Decompress if needed
    if str(imputed_vcf).endswith(".gz"):
        plain = od / "imputed_plain.vcf"
        subprocess.run(f"zcat {imputed_vcf} > {plain}", shell=True, check=True)
        imputed_vcf = plain

    imputed_df = _vcf_to_df(imputed_vcf, geno_df.index.tolist())
    # Reindex to match original column order
    imputed_df = imputed_df.reindex(columns=geno_df.columns)
    # Any remaining NaN after Beagle → mean-fill
    if imputed_df.isna().any().any():
        imputed_df = mean_fill(imputed_df, allele_freq)

    logger.info("Beagle imputation complete: %d samples × %d SNPs", *imputed_df.shape)
    return {"geno_df": imputed_df, "method": "beagle", "status": "done"}


def run(
    geno_df: pd.DataFrame,
    enabled: bool = False,
    method: str = "beagle",
    outdir: str = "",
    dry_run: bool = False,
    allele_freq: np.ndarray | None = None,
    ref_geno_df: pd.DataFrame | None = None,
) -> dict:
    if not enabled:
        logger.info("Imputation disabled — mean-filling NaNs")
        return {"geno_df": mean_fill(geno_df, allele_freq), "method": "mean_fill", "status": "disabled"}

    if dry_run:
        logger.info("Dry run — mean-filling NaNs")
        return {"geno_df": mean_fill(geno_df, allele_freq), "method": "mean_fill", "status": "dry_run"}

    if method == "beagle" and is_beagle_available():
        return run_beagle(geno_df, ref_geno_df=ref_geno_df, outdir=outdir, allele_freq=allele_freq)

    logger.info("Beagle not available — mean-filling NaNs")
    return {"geno_df": mean_fill(geno_df, allele_freq), "method": "mean_fill", "status": "fallback"}
