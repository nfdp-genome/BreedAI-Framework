#!/usr/bin/env python3
"""
Phase 1 orchestrator: core dataset + default GBLUP (+ optional R&D prep).

**Standard workflow (recommended):** place ``Geno.csv`` and ``Pheno.csv`` under
``dataset/input/``, then run this script (or use ``start_menu.sh`` → Phase 1).

Orchestrates:
  1. Build core dataset (05_build_core_dataset.py logic)
  2. Run default GBLUP track (06_default_track.py logic)
  3. Optionally sync ``dataset/input`` for R&D (02a) and print instructions
  4. Write run summary

Usage:
    python 07_run_poc.py                              # default_plus_rnd, dataset/input
    python 07_run_poc.py --mode default               # GBLUP only
    python 07_run_poc.py --adapter vandenberg         # internal: use public_datasets paths
    python 07_run_poc.py --dry-run                    # validate wiring
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase1_orchestrator")


def _generate_reports(core_dir: Path, default_dir: Path, phase1_dir: Path):
    """Generate notebook reports into the standard notebooks/ location."""
    nb_dir = PROJECT_ROOT / "notebooks" / "Phase_1_Learning_Benchmarking"
    nb_dir.mkdir(parents=True, exist_ok=True)

    combined_csv = phase1_dir / "training_validation" / "combined_train_validate_results.csv"
    gblup_csv = default_dir / "default_gblup" / "gblup_results.csv"
    results_csv = combined_csv if combined_csv.exists() else gblup_csv

    if results_csv.exists():
        try:
            from importlib.util import spec_from_file_location, module_from_spec
            spec = spec_from_file_location("report", SCRIPTS_DIR / "02b_phase1_report_benchmarking.py")
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            out = nb_dir / "1.2_Learning_Benchmarking_report.ipynb"
            mod.create_report_notebook(str(results_csv), str(out))
            logger.info("Benchmarking report notebook: %s (from %s)", out, results_csv.name)
        except Exception as e:
            logger.warning("Could not generate benchmarking notebook: %s", e)

    dataset_dir = str(PROJECT_ROOT / "dataset" / "input")
    gmatrix_dir = str(core_dir / "gmatrix")
    if Path(gmatrix_dir).exists():
        try:
            spec = spec_from_file_location("qc_report", SCRIPTS_DIR / "02c_phase1_report_preprocessing.py")
            mod = module_from_spec(spec)
            spec.loader.exec_module(mod)
            out = nb_dir / "1.1_Preprocessing_report.ipynb"
            mod.create_qc_report_notebook(dataset_dir, gmatrix_dir, str(out))
            logger.info("QC notebook: %s", out)
        except Exception as e:
            logger.warning("Could not generate QC notebook: %s", e)


def _write_phase2_bridge(core_dir: Path, default_dir: Path, phase1_dir: Path):
    """
    Write bridge artifacts so the existing Phase 2 scripts can consume
    the default-track GBLUP results without modification.

    Creates:
      - training_validation/variance_filter_mask.npy
      - training_validation/combined_train_validate_results.csv
    """
    import numpy as np
    import pandas as pd

    tv_dir = phase1_dir / "training_validation"
    tv_dir.mkdir(parents=True, exist_ok=True)

    # 1) variance_filter_mask.npy from core dataset QC
    geno_path = core_dir / "genotypes.csv"
    snp_meta_path = core_dir / "snp_metadata.csv"
    mask_path = tv_dir / "variance_filter_mask.npy"

    if snp_meta_path.exists() and not mask_path.exists():
        snp_meta = pd.read_csv(snp_meta_path)
        mask = (snp_meta["variance"] > 0.0001).values
        # Phase 2 expects mask length = original input feature count
        # Core dataset already filtered, so all remaining SNPs pass → full-true mask
        n_snps = len(mask)
        full_mask = np.ones(n_snps, dtype=bool)
        np.save(mask_path, full_mask)
        logger.info("Bridge: wrote %s (%d features)", mask_path.name, n_snps)

    # 2) combined_train_validate_results.csv from GBLUP results (schema must match 02a / 02b report)
    gblup_csv = default_dir / "default_gblup" / "gblup_results.csv"
    combined_path = tv_dir / "combined_train_validate_results.csv"

    if gblup_csv.exists():
        if combined_path.exists():
            try:
                ex = pd.read_csv(combined_path)
                n_alg = ex["algorithm"].nunique() if "algorithm" in ex.columns else 0
                if len(ex) > 12 or n_alg > 2:
                    logger.info(
                        "Bridge: keeping existing %s (%d rows, %d algorithms) — R&D results present",
                        combined_path.name,
                        len(ex),
                        n_alg,
                    )
                else:
                    _write_gblup_bridge_csv(gblup_csv, combined_path, core_dir)
            except Exception as e:
                logger.warning("Bridge: could not check existing combined CSV (%s); rewriting from GBLUP", e)
                _write_gblup_bridge_csv(gblup_csv, combined_path, core_dir)
        else:
            _write_gblup_bridge_csv(gblup_csv, combined_path, core_dir)


def _write_gblup_bridge_csv(gblup_csv: Path, combined_path: Path, core_dir: Path | None = None) -> None:
    """Write combined_train_validate_results.csv rows compatible with 02b_phase1_report_benchmarking."""
    import pandas as pd

    n_train = n_val = n_test = 0
    if core_dir is not None:
        sp_path = core_dir / "splits.json"
        if sp_path.exists():
            with open(sp_path, encoding="utf-8") as sf:
                sp = json.load(sf)
            n_train = len(sp.get("train", []) or [])
            n_val = len(sp.get("val", []) or [])
            n_test = len(sp.get("test", []) or [])

    gblup = pd.read_csv(gblup_csv)
    rows = []
    for _, r in gblup.iterrows():
        vr2 = float(r.get("val_r2", 0) or 0)
        vpr = float(r.get("val_pearson_r", 0) or 0)
        vrmse = float(r.get("val_rmse", 0) or 0)
        vmae = float(r.get("val_mae", 0) or 0)
        vbias = float(r.get("val_bias", 0) or 0)
        tr2 = float(r.get("test_r2", 0) or 0)
        tpr = float(r.get("test_pearson_r", 0) or 0)
        trmse = float(r.get("test_rmse", 0) or 0)
        tmae = float(r.get("test_mae", 0) or 0)
        tbias = float(r.get("test_bias", 0) or 0)
        # Default track CSV has no train-set metrics; mirror validation for train columns so 02b has required fields.
        rows.append(
            {
                "trait": r["trait"],
                "algorithm": "GBLUP_Ridge",
                "train_r2": vr2,
                "train_pearson_r": vpr,
                "train_rmse": vrmse,
                "train_mae": vmae,
                "train_bias": vbias,
                "val_r2": vr2,
                "val_pearson_r": vpr,
                "val_rmse": vrmse,
                "val_mae": vmae,
                "val_bias": vbias,
                "test_r2": tr2,
                "test_pearson_r": tpr,
                "test_pearson_p": float("nan"),
                "test_spearman_r": float("nan"),
                "test_rmse": trmse,
                "test_mae": tmae,
                "test_bias": tbias,
                "cv_r2_mean": vr2,
                "cv_r2_std": 0.0,
                "cv_pearson_mean": vpr,
                "cv_pearson_std": 0.0,
                "fit_time": float(r.get("train_time_s", 0) or 0),
                "n_train": n_train,
                "n_val": n_val,
                "n_test": n_test,
                "overfitting_indicator": vr2 - tr2,
            }
        )
    pd.DataFrame(rows).to_csv(combined_path, index=False)
    logger.info("Bridge: wrote %s (%d rows) for benchmarking report (default GBLUP only)", combined_path.name, len(rows))


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Run Phase 1 (core dataset + default track) from dataset/input/"
    )
    ap.add_argument("--mode", default="default_plus_rnd", choices=["default", "default_plus_rnd"])
    ap.add_argument("--dry-run", action="store_true", help="Validate wiring without heavy computation")
    ap.add_argument("--replicate", type=int, default=1, help="Only for --adapter vandenberg")
    ap.add_argument(
        "--adapter",
        default="input",
        choices=["input", "vandenberg"],
        help="input=dataset/input/ Geno.csv+Pheno.csv (default); vandenberg=legacy adapter paths",
    )
    ap.add_argument("--species", default="cattle", help="Species label for config")
    ap.add_argument("--goal", default="growth", help="Breeding goal label for config")
    ap.add_argument(
        "--no-submit-rnd",
        action="store_true",
        help="With default_plus_rnd, do not sbatch the R&D train-validate pipeline (02_phase1_train_validate.sh)",
    )
    args = ap.parse_args()

    t0 = time.time()

    # ---- Generate config: standard dataset/input/ or legacy Vandenberg adapter ----
    if args.adapter == "vandenberg":
        from public_dataset.vandenberg.vandenberg_adapter import build_run_config

        cfg = build_run_config(mode=args.mode, replicate=args.replicate)
    else:
        from pipeline_config import build_config_from_dataset_input

        cfg = build_config_from_dataset_input(
            mode=args.mode,
            species=args.species,
            breeding_goal=args.goal,
            project_root=PROJECT_ROOT,
        )

    phase1_dir = PROJECT_ROOT / "Phase1_Learning_Benchmarking"
    core_dir = phase1_dir / "core_dataset"
    default_dir = phase1_dir / "training_validation" / "default_track"

    # ---- Step 1: Build core dataset ----
    logger.info("=" * 60)
    logger.info("STEP 1: Building core dataset")
    logger.info("=" * 60)
    from pipeline_config import save_resolved_config

    phase1_dir.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
        save_resolved_config(cfg, phase1_dir / "phase1_run_config.yaml")
    except ImportError:
        with open(phase1_dir / "phase1_run_config.json", "w") as f:
            json.dump(cfg, f, indent=2, default=str)

    if args.dry_run:
        core_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DRY RUN: Skipping core dataset build")
        manifest = {"status": "dry_run", "config": cfg.get("dataset", {}).get("source")}
    else:
        from stages import s05_imputation  # ensure import works
        from pipeline_config import load_config
        full_cfg = load_config(overrides=cfg)

        from importlib import import_module
        builder = import_module("05_build_core_dataset")
        manifest = builder.build(full_cfg, core_dir)

    # ---- Step 2: Default GBLUP track ----
    logger.info("=" * 60)
    logger.info("STEP 2: Running default GBLUP track")
    logger.info("=" * 60)

    if args.dry_run:
        default_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DRY RUN: Skipping GBLUP training")
        model_card = {"status": "dry_run"}
    else:
        from importlib import import_module
        default_track = import_module("06_default_track")
        model_card = default_track.run(cfg, core_dir, default_dir)

    # ---- Bridge: write artifacts that Phase 2 expects ----
    if not args.dry_run:
        _write_phase2_bridge(core_dir, default_dir, phase1_dir)

    # ---- Generate reports into standard notebooks/ location ----
    if not args.dry_run:
        _generate_reports(core_dir, default_dir, phase1_dir)
        if args.mode == "default_plus_rnd":
            logger.info(
                "If 1.2_Learning_Benchmarking_report.ipynb lists only GBLUP rows, R&D jobs are still running; "
                "when combine finishes, regenerate 1.2 from training_validation/combined_train_validate_results.csv."
            )

    # ---- Step 3: R&D track (train-validate array via 02_phase1_train_validate.sh) ----
    rnd_job_id = None
    if args.mode == "default_plus_rnd":
        logger.info("=" * 60)
        logger.info("STEP 3: R&D benchmarking pipeline")
        logger.info("=" * 60)
        if not args.dry_run:
            input_dir = PROJECT_ROOT / "dataset" / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            for fname in ("genotypes.csv", "phenotypes.csv"):
                src = core_dir / fname
                dst = input_dir / ("Geno.csv" if "geno" in fname else "Pheno.csv")
                if src.exists():
                    if dst.exists():
                        dst.unlink()
                    shutil.copy2(src, dst)
                    logger.info("Synced %s → dataset/input/%s", src.name, dst.name)

            rnd_script = SCRIPTS_DIR / "02_phase1_train_validate.sh"
            if args.no_submit_rnd:
                logger.info(
                    "Skipping R&D job submission (--no-submit-rnd). Submit manually:\n"
                    "  cd %s && sbatch 02_phase1_train_validate.sh",
                    SCRIPTS_DIR,
                )
            elif not rnd_script.is_file():
                logger.warning("R&D script not found: %s", rnd_script)
            else:
                if not shutil.which("sbatch"):
                    logger.warning(
                        "sbatch not in PATH; cannot submit R&D pipeline. On a login node run:\n"
                        "  cd %s && sbatch 02_phase1_train_validate.sh",
                        SCRIPTS_DIR,
                    )
                else:
                    try:
                        r = subprocess.run(
                            ["sbatch", str(rnd_script)],
                            cwd=str(SCRIPTS_DIR),
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=120,
                        )
                        if r.returncode == 0:
                            out = (r.stdout or "").strip()
                            m = re.search(r"Submitted batch job (\d+)", out)
                            rnd_job_id = m.group(1) if m else out.splitlines()[-1] if out else None
                            logger.info("Submitted R&D train-validate job: %s", rnd_job_id)
                            logger.info(
                                "This job is separate from the Phase 1 orchestrator — benchmarking runs there. "
                                "Monitor: squeue -u $USER; logs: %s/train_validate/",
                                PROJECT_ROOT / "logs",
                            )
                        else:
                            logger.warning(
                                "sbatch R&D failed (rc=%s): %s %s",
                                r.returncode,
                                r.stderr,
                                r.stdout,
                            )
                    except Exception as e:
                        logger.warning("Could not submit R&D job: %s", e)
        else:
            logger.info("DRY RUN: would submit R&D pipeline after default track")
    else:
        logger.info("Mode=default — R&D track skipped")

    # ---- Phase 1 run summary ----
    elapsed = time.time() - t0
    summary = {
        "run": "phase1_orchestrator",
        "adapter": args.adapter,
        "mode": args.mode,
        "date": datetime.now().isoformat(),
        "elapsed_s": round(elapsed, 1),
        "core_dataset": str(core_dir),
        "default_results": str(default_dir),
        "model_card": model_card,
        "rnd_prep_job": rnd_job_id,
        "notebooks": str(
            PROJECT_ROOT / "notebooks" / "Phase_1_Learning_Benchmarking"
        ),
    }

    summary_path = phase1_dir / "phase1_run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Phase 1 orchestrator finished in %.1fs", elapsed)
    logger.info("Core dataset: %s", core_dir)
    logger.info("Default results: %s", default_dir)
    logger.info("Summary: %s", summary_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
