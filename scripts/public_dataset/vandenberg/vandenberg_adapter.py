#!/usr/bin/env python3
"""
Dataset adapter: Van den Berg et al. (2020) cattle benchmark.

This adapter maps the Vandenberg public dataset into the BreedAI core
dataset contract so it can be consumed by both default and R&D tracks.

The Vandenberg dataset is already at genotype level (processed CSV or
raw text with numeric 0/1/2 coding). It does NOT require FASTQ/VCF
processing. The full FASTQ-to-GEBV path is implemented in BreedAI for
projects that need it, but this POC uses the genotype-level entry point.

Input layouts supported:
  A) Processed: Geno_QTL300_rg8.csv + Pheno_QTL300_rg8.csv
  B) Raw text:  Genotypes_26503SNPs.txt + Phenotypes_replicate_X.txt + ID_Breed.txt

The adapter auto-detects which layout is available and prepares config.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RAW_DIR = PROJECT_ROOT / "dataset" / "public_datasets" / "cattle" / "raw" / "vandenberg"
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "public_datasets" / "cattle" / "processed" / "vandenberg_QTL300_rg8"


def detect_layout() -> dict:
    """
    Determine what Vandenberg data is available and return the best
    entry-point config fragment.
    """
    result = {"layout": None, "geno_file": None, "pheno_file": None, "metadata_file": None, "notes": []}

    processed_geno = PROCESSED_DIR / "Geno_QTL300_rg8.csv"
    processed_pheno = PROCESSED_DIR / "Pheno_QTL300_rg8.csv"

    if processed_geno.exists() and processed_pheno.exists():
        result["layout"] = "processed_csv"
        result["geno_file"] = str(processed_geno)
        result["pheno_file"] = str(processed_pheno)
        result["notes"].append("Using pre-processed CSV files (recommended for POC).")
        logger.info("Detected processed Vandenberg CSV files")
        return result

    raw_geno = RAW_DIR / "Genotypes_26503SNPs.txt"
    if raw_geno.exists():
        pheno_candidates = sorted(RAW_DIR.glob("Phenotypes_GenCor_0.8/Phenotypes_replicate_1.txt"))
        if not pheno_candidates:
            pheno_candidates = sorted(RAW_DIR.glob("Phenotypes_GenCor_*/Phenotypes_replicate_1.txt"))

        if pheno_candidates:
            result["layout"] = "raw_text"
            result["geno_file"] = str(raw_geno)
            result["pheno_file"] = str(pheno_candidates[0])
            result["notes"].append(
                "Using raw text files. Run 02_prepare_vandenberg.py or "
                "03_prepare_step4_inputs.py first to produce Geno.csv / Pheno.csv."
            )

            breed_file = RAW_DIR / "ID_Breed.txt"
            if breed_file.exists():
                result["metadata_file"] = str(breed_file)
            logger.info("Detected raw Vandenberg text files")
            return result

    result["layout"] = "missing"
    result["notes"].append("Vandenberg data not found. Download it first (see scripts/public_dataset/vandenberg/00_quick_start.md).")
    logger.warning("No Vandenberg data found under %s or %s", RAW_DIR, PROCESSED_DIR)
    return result


def build_run_config(
    mode: str = "default_plus_rnd",
    replicate: int = 1,
    genetic_corr: str = "0.8",
) -> dict:
    """
    Generate a full BreedAI run config dict for the Vandenberg POC.

    Assumptions documented here:
    - Species: cattle (simulated Holsteins in the original study)
    - Breeding goal: growth (TBV = True Breeding Value, 4 traits)
    - No pedigree available in the public dataset
    - Genotype coding is already 0/1/2 (no VCF needed)
    - The full FASTQ-to-GEBV pipeline exists in BreedAI but is NOT
      needed for this POC because data is already at genotype level
    """
    layout = detect_layout()

    if layout["layout"] == "missing":
        raise FileNotFoundError("Vandenberg dataset not found. " + "; ".join(layout["notes"]))

    if layout["layout"] == "raw_text":
        prep_dir = (
            PROJECT_ROOT
            / "dataset"
            / "public_datasets"
            / "cattle"
            / "processed"
            / f"vandenberg_step4_ready"
            / f"replicate_{replicate}"
        )
        geno_csv = prep_dir / "Geno.csv"
        pheno_csv = prep_dir / "Pheno.csv"
        meta_csv = prep_dir / "metadata.csv"

        if geno_csv.exists() and pheno_csv.exists():
            geno_file = str(geno_csv)
            pheno_file = str(pheno_csv)
            meta_file = str(meta_csv) if meta_csv.exists() else ""
        else:
            geno_file = layout["geno_file"]
            pheno_file = layout["pheno_file"]
            meta_file = layout.get("metadata_file", "")
    else:
        geno_file = layout["geno_file"]
        pheno_file = layout["pheno_file"]
        meta_file = layout.get("metadata_file", "")

    cfg = {
        "species": "cattle",
        "breeding_goal": "growth",
        "input_type": "genotype",
        "dataset": {
            "source": f"vandenberg_rg{genetic_corr}_rep{replicate}",
            "geno_file": geno_file,
            "pheno_file": pheno_file,
            "pedigree_file": "",
            "metadata_file": meta_file or "",
        },
        "mode": mode,
        "qc": {
            "min_sample_callrate": 0.95,
            "min_snp_callrate": 0.98,
            "min_maf": 0.01,
            "min_variance": 0.0001,
            "enable_hwe": False,
        },
        "imputation": {"enabled": False},
        "grm": {"method": "vanRaden"},
        "cv": {"n_folds": 5, "test_size": 0.20, "val_size": 0.20, "random_state": 42},
        "models": {
            "default": ["GBLUP"],
            "ssgblup_if_pedigree": True,
            "rnd": [
                "GBLUP_Ridge", "LASSO", "ElasticNet", "BayesianRidge",
                "RandomForest", "SVR_RBF", "NeuralNet_MLP",
                "BayesA", "BayesB", "BayesCpi", "XGBoost", "LightGBM",
            ],
        },
        "ensemble": {
            "simple_average": True,
            "median": True,
            "weighted_average": True,
            "stacking_nonneg_ridge": True,
            "stack_outer_splits": 5,
            "stack_inner_splits": 3,
            "stack_alpha": 1.0,
        },
        "selection_index": {"enabled": False, "weights": {}},
        "monitoring": {"enabled": False, "baseline_run": ""},
        "slurm": {
            "partition": "batch",
            "account": "YOUR_SLURM_ACCOUNT",
            "mem": "64G",
            "cpus_per_task": 32,
            "time": "14-00:00:00",
        },
        "_adapter": {
            "name": "vandenberg",
            "layout_detected": layout["layout"],
            "notes": layout["notes"],
        },
    }
    return cfg


if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    cfg = build_run_config()
    pprint.pprint(cfg)
