#!/usr/bin/env python3
"""
Build the shared "core dataset" contract used by both the default and R&D
tracks in BreedAI.

Usage:
    python 05_build_core_dataset.py --config configs/run_configs/cattle_vandenberg_default.yaml
    python 05_build_core_dataset.py --config configs/run_configs/cattle_vandenberg_default.yaml --dry-run

The core dataset is written to:
    Phase1_Learning_Benchmarking/core_dataset/

Artifacts produced:
    genotypes.csv       Canonical genotype matrix (samples × SNPs, QC-passed)
    phenotypes.csv      Aligned phenotype table
    sample_metadata.csv Sample metadata (breed, etc.)
    snp_metadata.csv    Per-SNP allele freq, MAF, call rate, variance
    splits.json         Train / val / test index arrays
    gmatrix/            G-matrix (CSV + NPY + metadata JSON)
    h_matrix/           H-matrix or skip metadata
    qc/                 QC summary + removed sample/SNP lists
    fixed_effects/      Fixed-effects spec + design matrix
    tool_versions.json  Tool / library versions
    core_dataset_manifest.json  Manifest with paths + checksums
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_config import load_config, input_is_fastq, has_pedigree, save_resolved_config
from stages import (
    s04_sample_snp_qc as qc_stage,
    s05_imputation as impute_stage,
    s06_genotype_matrix as geno_stage,
    s07_grm as grm_stage,
    s08_h_matrix as h_stage,
    s09_fixed_effects as fe_stage,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_core_dataset")


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _tool_versions() -> dict:
    import sklearn
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
    }
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        pass
    return versions


def _make_splits(n: int, cfg: dict) -> dict:
    test_size = cfg["cv"]["test_size"]
    val_size = cfg["cv"]["val_size"]
    seed = cfg["cv"]["random_state"]
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    return {
        "test": idx[:n_test].tolist(),
        "val": idx[n_test:n_test + n_val].tolist(),
        "train": idx[n_test + n_val:].tolist(),
        "n_total": n,
        "test_size": test_size,
        "val_size": val_size,
        "random_state": seed,
    }


def build(cfg: dict, outdir: Path, dry_run: bool = False) -> dict:
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)

    # Save resolved config for reproducibility
    save_resolved_config(cfg, outdir / "resolved_config.yaml")

    # ---- Load genotypes ----
    geno_file = cfg["dataset"]["geno_file"]
    pheno_file = cfg["dataset"]["pheno_file"]

    if input_is_fastq(cfg):
        logger.warning(
            "FASTQ entry detected. Stages 1–3 (QC, alignment, calling) "
            "must be run externally before building the core dataset. "
            "Expecting processed Geno.csv at %s", geno_file,
        )

    logger.info("Loading genotypes from %s", geno_file)
    geno_df = geno_stage.load_geno(geno_file)
    geno_df = geno_stage.validate_coding(geno_df)

    logger.info("Loading phenotypes from %s", pheno_file)
    pheno_df = pd.read_csv(pheno_file, index_col=0)
    pheno_df.index = pheno_df.index.astype(str)

    # Align samples
    common = geno_df.index.intersection(pheno_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between genotype and phenotype files")
    logger.info("Common samples: %d (geno=%d, pheno=%d)", len(common), len(geno_df), len(pheno_df))
    geno_df = geno_df.loc[common]
    pheno_df = pheno_df.loc[common]
    sample_ids = common.tolist()

    # ---- Stage 4: QC ----
    qc_cfg = cfg.get("qc", {})
    qc_result = qc_stage.run_genotype_qc(
        geno_df,
        min_sample_callrate=qc_cfg.get("min_sample_callrate", 0.95),
        min_snp_callrate=qc_cfg.get("min_snp_callrate", 0.98),
        min_maf=qc_cfg.get("min_maf", 0.01),
        min_variance=qc_cfg.get("min_variance", 0.0001),
        enable_hwe=qc_cfg.get("enable_hwe", False),
    )
    geno_df = qc_result["geno_df"]
    qc_stage.save_qc_artifacts(qc_result, outdir / "qc")

    # Re-align after QC sample removal
    pheno_df = pheno_df.loc[pheno_df.index.isin(geno_df.index)]
    geno_df = geno_df.loc[geno_df.index.isin(pheno_df.index)]
    sample_ids = geno_df.index.tolist()

    # ---- Stage 5: Imputation ----
    imp_cfg = cfg.get("imputation", {})
    imp_result = impute_stage.run(
        geno_df,
        enabled=imp_cfg.get("enabled", False),
        method=imp_cfg.get("method", "beagle"),
        outdir=str(outdir / "imputation"),
        dry_run=dry_run,
    )
    geno_df = imp_result["geno_df"]

    # ---- Stage 6: Canonical genotype matrix ----
    geno_stage.save_canonical(geno_df, outdir, "genotypes.csv")
    snp_meta = geno_stage.build_snp_metadata(geno_df)
    snp_meta.to_csv(outdir / "snp_metadata.csv", index=False)

    # Save phenotypes
    pheno_df.to_csv(outdir / "phenotypes.csv")

    # Sample metadata
    meta_file = cfg["dataset"].get("metadata_file", "")
    if meta_file and Path(meta_file).exists():
        meta_df = pd.read_csv(meta_file)
        meta_df.to_csv(outdir / "sample_metadata.csv", index=False)
    else:
        pd.DataFrame({"sample_id": sample_ids}).to_csv(outdir / "sample_metadata.csv", index=False)

    # ---- Splits ----
    splits = _make_splits(len(sample_ids), cfg)
    with open(outdir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    # ---- Stage 7: GRM ----
    grm_dir = outdir / "gmatrix"
    grm_result = grm_stage.compute_grm(
        geno_df,
        method=cfg.get("grm", {}).get("method", "vanRaden"),
        outdir=grm_dir,
    )

    # ---- Stage 8: H-matrix ----
    h_dir = outdir / "h_matrix"
    h_stage.run(
        G=grm_result["G"],
        sample_ids=sample_ids,
        pedigree_file=cfg["dataset"].get("pedigree_file", ""),
        outdir=h_dir,
    )

    # ---- Stage 9: Fixed effects ----
    fe_stage.build_fixed_effects_spec(
        metadata_file=meta_file if meta_file else None,
        sample_ids=sample_ids,
        outdir=outdir / "fixed_effects",
    )

    # ---- Tool versions ----
    versions = _tool_versions()
    with open(outdir / "tool_versions.json", "w") as f:
        json.dump(versions, f, indent=2)

    # ---- Manifest ----
    manifest_files = {}
    for p in sorted(outdir.rglob("*")):
        if p.is_file() and p.name != "core_dataset_manifest.json":
            rel = str(p.relative_to(outdir))
            manifest_files[rel] = {"size_bytes": p.stat().st_size, "md5": _md5(p)}

    manifest = {
        "created": datetime.now().isoformat(),
        "species": cfg.get("species"),
        "breeding_goal": cfg.get("breeding_goal"),
        "input_type": cfg.get("input_type"),
        "n_samples": len(sample_ids),
        "n_snps": geno_df.shape[1],
        "n_traits": pheno_df.shape[1],
        "traits": list(pheno_df.columns),
        "elapsed_s": round(time.time() - t0, 1),
        "files": manifest_files,
    }
    with open(outdir / "core_dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Core dataset built: %d samples, %d SNPs, %d traits in %.1fs → %s",
        len(sample_ids), geno_df.shape[1], pheno_df.shape[1], time.time() - t0, outdir,
    )
    return manifest


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build BreedAI core dataset")
    ap.add_argument("--config", required=True, help="Run config YAML path")
    ap.add_argument("--outdir", default=None, help="Override output directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir) if args.outdir else PROJECT_ROOT / "Phase1_Learning_Benchmarking" / "core_dataset"
    build(cfg, outdir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
