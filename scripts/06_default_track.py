#!/usr/bin/env python3
"""
Default literature/production track: GBLUP + optional ssGBLUP.

Consumes the core dataset built by 05_build_core_dataset.py and runs
the default models (GBLUP, ssGBLUP if pedigree exists).

Usage:
    python 06_default_track.py --config configs/run_configs/cattle_vandenberg_default.yaml
    python 06_default_track.py --core-dataset Phase1_Learning_Benchmarking/core_dataset
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from pipeline_config import load_config, has_pedigree, mode_includes_rnd
from stages.s10_default_gblup import run as run_gblup
from stages.s11_gebv_reliability import run as run_gebv
from stages.s12_selection_index import compute_index
from stages.s13_monitoring import run as run_monitoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("default_track")


def load_core(core_dir: Path) -> dict:
    geno = pd.read_csv(core_dir / "genotypes.csv", index_col=0)
    pheno = pd.read_csv(core_dir / "phenotypes.csv", index_col=0)
    G = np.load(core_dir / "gmatrix" / "Gmatrix.npy")
    with open(core_dir / "splits.json") as f:
        splits = json.load(f)
    sample_ids = geno.index.tolist()

    H = None
    h_path = core_dir / "h_matrix" / "H_matrix.npy"
    if h_path.exists():
        H = np.load(h_path)

    return {
        "geno": geno,
        "pheno": pheno,
        "G": G,
        "H": H,
        "splits": splits,
        "sample_ids": sample_ids,
    }


def run(cfg: dict, core_dir: Path, results_dir: Path) -> dict:
    t0 = time.time()
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading core dataset from %s", core_dir)
    data = load_core(core_dir)

    # ---- GBLUP ----
    gblup_dir = results_dir / "default_gblup"
    gblup_result = run_gblup(
        G=data["G"],
        pheno_df=data["pheno"],
        sample_ids=data["sample_ids"],
        outdir=gblup_dir,
        cv_folds=cfg.get("cv", {}).get("n_folds", 5),
        test_size=cfg.get("cv", {}).get("test_size", 0.2),
        val_size=cfg.get("cv", {}).get("val_size", 0.2),
        random_state=cfg.get("cv", {}).get("random_state", 42),
    )

    # ---- GEBV + Reliability ----
    gebv_dir = results_dir / "gebv"
    best_alpha = gblup_result["results"][0].get("alpha", 1.0) if gblup_result["results"] else 1.0
    gebv_result = run_gebv(
        G=data["G"],
        pheno_df=data["pheno"],
        sample_ids=data["sample_ids"],
        outdir=gebv_dir,
        alpha=best_alpha,
    )

    # ---- Selection index ----
    idx_cfg = cfg.get("selection_index", {})
    if idx_cfg.get("enabled", False):
        compute_index(
            gebv_file=gebv_dir / "gebv_all_traits.csv",
            weights=idx_cfg.get("weights", {}),
            outdir=results_dir / "selection_index",
        )

    # ---- Monitoring ----
    mon_cfg = cfg.get("monitoring", {})
    run_monitoring(
        current_geno=data["geno"],
        current_pheno=data["pheno"],
        baseline_run=mon_cfg.get("baseline_run", ""),
        outdir=results_dir / "monitoring",
    )

    # ---- Model card ----
    model_card = {
        "track": "default",
        "models": ["GBLUP"],
        "n_samples": len(data["sample_ids"]),
        "n_snps": data["geno"].shape[1],
        "traits": list(data["pheno"].columns),
        "gblup_val_metrics": {r["trait"]: r.get("val_r2") for r in gblup_result["results"]},
        "gebv_mean_reliability": gebv_result["summary"].get("mean_reliability", {}),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(results_dir / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    logger.info("Default track complete in %.1fs → %s", time.time() - t0, results_dir)
    return model_card


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run BreedAI default GBLUP track")
    ap.add_argument("--config", default=None, help="Run config YAML")
    ap.add_argument("--core-dataset", default=None, help="Path to core_dataset/ folder")
    ap.add_argument("--results-dir", default=None, help="Where to write results")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    core = Path(args.core_dataset) if args.core_dataset else PROJECT_ROOT / "Phase1_Learning_Benchmarking" / "core_dataset"
    rdir = Path(args.results_dir) if args.results_dir else PROJECT_ROOT / "Phase1_Learning_Benchmarking" / "training_validation" / "default_track"

    if not core.exists():
        logger.error("Core dataset not found at %s — run 05_build_core_dataset.py first", core)
        sys.exit(1)

    run(cfg, core, rdir)


if __name__ == "__main__":
    main()
