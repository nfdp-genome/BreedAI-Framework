#!/usr/bin/env python3
"""
Deploy the default-track GBLUP model and predict on new animals.

Guardrails:
  --min-overlap PCT   Reject if SNP overlap < PCT (default 50)
  --warn-overlap PCT  Warn  if SNP overlap < PCT (default 80)
  --impute METHOD     beagle | mean_fill (default mean_fill)

Missing-SNP handling priority:
  1. Beagle (if --impute beagle and jar available)
  2. Mean-fill from training allele frequencies (explicit fallback)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from stages.s05_imputation import (
    is_beagle_available,
    mean_fill,
    run_beagle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("deploy_default")

DEFAULT_WARN_OVERLAP = 80.0
DEFAULT_MIN_OVERLAP = 50.0


def _load_best_alpha(default_dir: Path) -> float:
    csv_path = default_dir / "default_gblup" / "gblup_results.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return float(df["alpha"].iloc[0]) if "alpha" in df.columns else 1.0
    return 1.0


def _align_and_impute(
    new_geno: pd.DataFrame,
    train_geno: pd.DataFrame,
    allele_freq: np.ndarray,
    impute_method: str,
    outdir: Path,
) -> tuple[pd.DataFrame, dict]:
    train_snps = train_geno.columns
    new_snps = set(new_geno.columns)
    train_snps_set = set(train_snps)

    common = sorted(train_snps_set & new_snps, key=lambda s: list(train_snps).index(s))
    missing = sorted(train_snps_set - new_snps)
    extra = sorted(new_snps - train_snps_set)

    pct_overlap = round(100.0 * len(common) / len(train_snps), 2) if len(train_snps) > 0 else 0.0

    diag = {
        "n_train_snps": len(train_snps),
        "n_new_snps": len(new_snps),
        "n_common": len(common),
        "n_missing": len(missing),
        "n_extra_dropped": len(extra),
        "pct_overlap": pct_overlap,
    }

    if len(common) == 0:
        raise ValueError("Zero SNPs overlap between new genotypes and training data")

    # Build aligned matrix with common SNPs filled, missing as NaN
    aligned = pd.DataFrame(np.nan, index=new_geno.index, columns=train_snps, dtype=float)
    aligned[common] = new_geno[common].values

    n_nan_before = int(np.isnan(aligned.values).sum())

    # Impute missing values
    impute_status = "not_needed"
    if n_nan_before > 0:
        if impute_method == "beagle" and is_beagle_available():
            imp_dir = outdir / "imputation"
            imp_result = run_beagle(
                aligned,
                ref_geno_df=train_geno,
                outdir=str(imp_dir),
                allele_freq=allele_freq,
            )
            aligned = imp_result["geno_df"]
            impute_status = imp_result["status"]
            diag["imputation_method"] = imp_result["method"]
        else:
            if impute_method == "beagle":
                logger.warning("Beagle requested but not available — falling back to mean-fill")
            aligned = mean_fill(aligned, allele_freq)
            impute_status = "mean_fill"
            diag["imputation_method"] = "mean_fill"
    else:
        diag["imputation_method"] = "none_needed"

    n_nan_after = int(np.isnan(aligned.values).sum())
    diag["n_nan_before_impute"] = n_nan_before
    diag["n_nan_after_impute"] = n_nan_after
    diag["imputation_status"] = impute_status

    logger.info(
        "SNP alignment: %d common, %d missing, %d extra dropped | impute=%s | NaN %d→%d",
        len(common), len(missing), len(extra), diag["imputation_method"], n_nan_before, n_nan_after,
    )
    return aligned, diag


def deploy_and_predict(
    core_dir: Path,
    default_dir: Path,
    new_geno_path: Path | None,
    outdir: Path,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    warn_overlap: float = DEFAULT_WARN_OVERLAP,
    impute_method: str = "mean_fill",
) -> dict:
    t0 = time.time()
    outdir.mkdir(parents=True, exist_ok=True)
    pred_dir = outdir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    geno = pd.read_csv(core_dir / "genotypes.csv", index_col=0)
    geno.index = geno.index.astype(str)
    pheno = pd.read_csv(core_dir / "phenotypes.csv", index_col=0)
    pheno.index = pheno.index.astype(str)
    G = np.load(core_dir / "gmatrix" / "Gmatrix.npy")
    sample_ids = geno.index.tolist()
    alpha = _load_best_alpha(default_dir)
    allele_freq = np.nanmean(geno.values.astype(float), axis=0) / 2.0

    logger.info("Training: %d samples, %d SNPs, alpha=%.4f", len(sample_ids), geno.shape[1], alpha)

    overlap_policy = {
        "warn_threshold_pct": warn_overlap,
        "reject_threshold_pct": min_overlap,
    }
    assumptions = []
    alignment_diag = None
    overlap_decision = "ok"

    if new_geno_path and new_geno_path.exists():
        new_geno_raw = pd.read_csv(new_geno_path, index_col=0)
        new_geno_raw.index = new_geno_raw.index.astype(str)

        if new_geno_raw.index.duplicated().any():
            dups = new_geno_raw.index[new_geno_raw.index.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate sample IDs in new genotype file: {dups[:10]}")

        new_geno_aligned, alignment_diag = _align_and_impute(
            new_geno_raw, geno, allele_freq, impute_method, outdir,
        )

        pct = alignment_diag["pct_overlap"]

        # Overlap guardrails
        if pct < min_overlap:
            overlap_decision = "rejected"
            msg = (
                f"REJECTED: SNP overlap {pct:.1f}% is below minimum threshold "
                f"{min_overlap:.1f}%. Predictions would be unreliable."
            )
            logger.error(msg)
            alignment_diag["overlap_decision"] = overlap_decision
            alignment_diag["overlap_policy"] = overlap_policy

            reject_report = {
                "status": "rejected",
                "reason": msg,
                "snp_alignment": alignment_diag,
                "overlap_policy": overlap_policy,
                "date": datetime.now().isoformat(),
            }
            with open(outdir / "prediction_report.json", "w") as f:
                json.dump(reject_report, f, indent=2)
            with open(outdir / "model_card.json", "w") as f:
                json.dump({"status": "rejected", **reject_report}, f, indent=2)

            raise ValueError(msg)

        if pct < warn_overlap:
            overlap_decision = "warning"
            logger.warning(
                "SNP overlap %.1f%% is below warning threshold %.1f%% — predictions may be degraded",
                pct, warn_overlap,
            )
            assumptions.append(
                f"SNP overlap ({pct:.1f}%) is below the warning threshold ({warn_overlap:.1f}%). "
                f"Predictions may be degraded due to heavy imputation."
            )

        alignment_diag["overlap_decision"] = overlap_decision
        alignment_diag["overlap_policy"] = overlap_policy

        if alignment_diag.get("n_missing", 0) > 0:
            method_used = alignment_diag.get("imputation_method", "mean_fill")
            assumptions.append(
                f"{alignment_diag['n_missing']} missing SNPs handled via {method_used}."
            )
        if alignment_diag.get("n_extra_dropped", 0) > 0:
            assumptions.append(
                f"{alignment_diag['n_extra_dropped']} extra SNPs in new file were dropped."
            )

        # Cross-kernel
        X_train = geno.values.astype(float)
        X_new = new_geno_aligned.values.astype(float)
        p = allele_freq
        P = 2.0 * p
        X_train_c = X_train - P
        X_new_c = X_new - P
        scale = np.sum(2.0 * p * (1.0 - p))
        if scale <= 0:
            scale = X_train.shape[1]

        G_train = X_train_c @ X_train_c.T / scale
        G_new_train = X_new_c @ X_train_c.T / scale

        pred_ids = new_geno_aligned.index.tolist()
        is_new = True
        logger.info("New animals: %d", len(pred_ids))
    else:
        G_train = G
        G_new_train = G
        pred_ids = sample_ids
        is_new = False
        assumptions.append("No new genotype file provided. Predicting on training animals (test mode).")
        logger.info("Self-prediction mode: %d animals", len(pred_ids))

    # Train + predict per trait
    all_preds = []
    model_info = {}

    for trait in pheno.columns:
        y = pheno[trait].values.astype(float)
        mask = ~np.isnan(y)
        G_t = G_train[np.ix_(mask, mask)]
        y_t = y[mask]

        model = Ridge(alpha=alpha)
        model.fit(G_t, y_t)

        G_pred = G_new_train[:, mask] if is_new else G_train[:, mask]
        preds = model.predict(G_pred)

        all_preds.append(pd.DataFrame({"sample_id": pred_ids, "trait": trait, "gebv": preds}))
        model_info[trait] = {
            "alpha": alpha,
            "n_train": int(mask.sum()),
            "pred_mean": round(float(preds.mean()), 6),
            "pred_std": round(float(preds.std()), 6),
            "pred_min": round(float(preds.min()), 6),
            "pred_max": round(float(preds.max()), 6),
        }
        logger.info("GBLUP %s: n_train=%d, pred_mean=%.4f, pred_std=%.4f", trait, mask.sum(), preds.mean(), preds.std())

    predictions = pd.concat(all_preds, ignore_index=True)
    predictions.to_csv(pred_dir / "predictions.csv", index=False)
    predictions.pivot(index="sample_id", columns="trait", values="gebv").to_csv(pred_dir / "predictions_wide.csv")

    card = {
        "track": "default_deployment",
        "model": "GBLUP_Ridge",
        "alpha": alpha,
        "n_training_samples": len(sample_ids),
        "n_prediction_samples": len(pred_ids),
        "n_training_snps": len(geno.columns),
        "traits": list(pheno.columns),
        "is_new_animals": is_new,
        "per_trait": model_info,
        "snp_alignment": alignment_diag,
        "overlap_policy": overlap_policy,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(outdir / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)

    report = {
        "title": "BreedAI Default Track Prediction Report",
        "date": datetime.now().isoformat(),
        "model": "GBLUP (Ridge on VanRaden G-matrix kernel)",
        "training_data": {"n_samples": len(sample_ids), "n_snps": len(geno.columns), "n_traits": len(pheno.columns)},
        "prediction_data": {"n_samples": len(pred_ids), "source": str(new_geno_path) if new_geno_path else "self", "is_new_animals": is_new},
        "snp_alignment": alignment_diag or {"note": "self-prediction"},
        "overlap_policy": overlap_policy,
        "per_trait_summary": model_info,
        "assumptions": assumptions + [
            "GBLUP uses VanRaden (2008) G-matrix as the kernel.",
            "Ridge alpha selected via RidgeCV during Phase 1.",
            "No fixed effects (intercept-only).",
            "No pedigree (G-matrix only, no ssGBLUP).",
        ],
        "output_files": {
            "predictions_long": str(pred_dir / "predictions.csv"),
            "predictions_wide": str(pred_dir / "predictions_wide.csv"),
            "model_card": str(outdir / "model_card.json"),
        },
    }
    with open(outdir / "prediction_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Prediction report: %s", outdir / "prediction_report.json")
    logger.info("Deployment complete in %.1fs → %s", time.time() - t0, outdir)
    return card


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Deploy default GBLUP and predict")
    ap.add_argument("--core-dataset", default=None)
    ap.add_argument("--default-results", default=None)
    ap.add_argument("--new-geno", default=None, help="New genotype CSV for prediction")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--min-overlap", type=float, default=DEFAULT_MIN_OVERLAP, help="Reject below this %% (default 50)")
    ap.add_argument("--warn-overlap", type=float, default=DEFAULT_WARN_OVERLAP, help="Warn below this %% (default 80)")
    ap.add_argument("--impute", default="mean_fill", choices=["beagle", "mean_fill"], help="Imputation backend")
    args = ap.parse_args()

    core = Path(args.core_dataset) if args.core_dataset else PROJECT_ROOT / "Phase1_Learning_Benchmarking" / "core_dataset"
    default = Path(args.default_results) if args.default_results else PROJECT_ROOT / "Phase1_Learning_Benchmarking" / "training_validation" / "default_track"
    outdir = Path(args.outdir) if args.outdir else PROJECT_ROOT / "Phase2_Deployment_Prediction" / "deployment" / "default_track"
    new_geno = Path(args.new_geno) if args.new_geno else None

    if not core.exists():
        logger.error("Core dataset not found: %s", core)
        sys.exit(1)

    deploy_and_predict(core, default, new_geno, outdir, args.min_overlap, args.warn_overlap, args.impute)


if __name__ == "__main__":
    main()
