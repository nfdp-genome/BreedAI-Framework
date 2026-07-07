#!/usr/bin/env python3
"""
Test new-animal prediction with realistic edge cases.

Tests:
  1. Normal overlap (~98%) — should pass cleanly
  2. Low overlap (~60%) — should warn but still predict
  3. Very low overlap (~30%) — should be rejected (below 50% default)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_prediction")

CORE_DIR = PROJECT_ROOT / "Phase1_Learning_Benchmarking" / "core_dataset"
DEFAULT_DIR = PROJECT_ROOT / "Phase1_Learning_Benchmarking" / "training_validation" / "default_track"
TEST_DIR = PROJECT_ROOT / "Phase2_Deployment_Prediction" / "test_new_animal"


def _create_geno_file(
    name: str,
    n_holdout: int,
    n_drop: int,
    n_extra: int,
    nan_rate: float = 0.005,
    seed: int = 99,
) -> Path:
    geno = pd.read_csv(CORE_DIR / "genotypes.csv", index_col=0)
    geno.index = geno.index.astype(str)
    np.random.seed(seed)

    holdout_idx = np.random.choice(len(geno), size=n_holdout, replace=False)
    new_geno = geno.iloc[holdout_idx].copy()
    new_geno.index = [f"NEW_{i+1}" for i in range(n_holdout)]

    original_snps = list(new_geno.columns)

    if n_drop > 0:
        drop_idx = np.random.choice(len(original_snps), size=min(n_drop, len(original_snps) - 1), replace=False)
        keep_snps = [s for i, s in enumerate(original_snps) if i not in drop_idx]
        new_geno = new_geno[keep_snps]

    if n_extra > 0:
        extra_data = np.random.choice([0, 1, 2], size=(n_holdout, n_extra))
        extra_df = pd.DataFrame(extra_data, index=new_geno.index, columns=[f"EXTRA_{i}" for i in range(n_extra)])
        new_geno = pd.concat([new_geno, extra_df], axis=1)

    cols = list(new_geno.columns)
    np.random.shuffle(cols)
    new_geno = new_geno[cols]

    new_geno = new_geno.astype(float)
    if nan_rate > 0:
        mask = np.random.random(new_geno.shape) < nan_rate
        new_geno.values[mask] = np.nan

    TEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TEST_DIR / f"{name}.csv"
    new_geno.to_csv(out_path)

    n_common = len(set(original_snps) & set(new_geno.columns))
    logger.info(
        "Created %s: %d animals, %d SNPs, %d common (%.1f%%), %d NaN",
        name, n_holdout, len(new_geno.columns), n_common,
        100.0 * n_common / len(original_snps),
        int(np.isnan(new_geno.values).sum()),
    )
    return out_path


def _run_prediction(new_geno_path: Path, outdir: Path, min_overlap: float = 50.0, warn_overlap: float = 80.0):
    from importlib import import_module
    deploy = import_module("08_deploy_default_track")
    return deploy.deploy_and_predict(
        CORE_DIR, DEFAULT_DIR, new_geno_path, outdir,
        min_overlap=min_overlap, warn_overlap=warn_overlap,
    )


def _validate(outdir: Path, n_holdout: int) -> list[str]:
    errors = []
    for f in ["predictions/predictions.csv", "predictions/predictions_wide.csv", "model_card.json", "prediction_report.json"]:
        if not (outdir / f).exists():
            errors.append(f"Missing: {f}")

    pred_csv = outdir / "predictions" / "predictions.csv"
    if pred_csv.exists():
        df = pd.read_csv(pred_csv)
        if df["sample_id"].nunique() != n_holdout:
            errors.append(f"Expected {n_holdout} samples, got {df['sample_id'].nunique()}")
        if df["gebv"].isna().any():
            errors.append("NaN in predictions")

    card_path = outdir / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
        if "overlap_policy" not in card:
            errors.append("model_card missing overlap_policy")
        alignment = card.get("snp_alignment", {})
        if "overlap_decision" not in alignment:
            errors.append("model_card missing overlap_decision")

    report_path = outdir / "prediction_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if "overlap_policy" not in report:
            errors.append("prediction_report missing overlap_policy")

    return errors


def main():
    t0 = time.time()
    if not CORE_DIR.exists():
        logger.error("Core dataset not found — run POC first")
        sys.exit(1)

    results = {}

    # ---- Test 1: Normal overlap (~98%) ----
    logger.info("=" * 60)
    logger.info("TEST 1: Normal overlap (~98%%)")
    logger.info("=" * 60)
    f1 = _create_geno_file("test_normal", n_holdout=50, n_drop=500, n_extra=200, seed=42)
    out1 = TEST_DIR / "test_normal_output"
    _run_prediction(f1, out1)
    errs1 = _validate(out1, 50)
    card1 = json.loads((out1 / "model_card.json").read_text())
    overlap1 = card1.get("snp_alignment", {}).get("pct_overlap", 0)
    decision1 = card1.get("snp_alignment", {}).get("overlap_decision", "")
    results["test_normal"] = {"errors": errs1, "overlap": overlap1, "decision": decision1}
    logger.info("Result: %d errors, overlap=%.1f%%, decision=%s", len(errs1), overlap1, decision1)

    # ---- Test 2: Low overlap (~60%) — should warn ----
    logger.info("=" * 60)
    logger.info("TEST 2: Low overlap (~60%%) — expect warning")
    logger.info("=" * 60)
    n_train_snps = len(pd.read_csv(CORE_DIR / "genotypes.csv", index_col=0, nrows=0).columns)
    n_drop_60 = int(n_train_snps * 0.40)
    f2 = _create_geno_file("test_low_overlap", n_holdout=30, n_drop=n_drop_60, n_extra=100, seed=77)
    out2 = TEST_DIR / "test_low_overlap_output"
    _run_prediction(f2, out2, min_overlap=50.0, warn_overlap=80.0)
    errs2 = _validate(out2, 30)
    card2 = json.loads((out2 / "model_card.json").read_text())
    overlap2 = card2.get("snp_alignment", {}).get("pct_overlap", 0)
    decision2 = card2.get("snp_alignment", {}).get("overlap_decision", "")
    results["test_low_overlap"] = {"errors": errs2, "overlap": overlap2, "decision": decision2}

    if decision2 != "warning":
        errs2.append(f"Expected decision=warning, got {decision2}")
    logger.info("Result: %d errors, overlap=%.1f%%, decision=%s", len(errs2), overlap2, decision2)

    # ---- Test 3: Very low overlap (~30%) — should be rejected ----
    logger.info("=" * 60)
    logger.info("TEST 3: Very low overlap (~30%%) — expect rejection")
    logger.info("=" * 60)
    n_drop_70 = int(n_train_snps * 0.70)
    f3 = _create_geno_file("test_reject", n_holdout=20, n_drop=n_drop_70, n_extra=50, seed=55)
    out3 = TEST_DIR / "test_reject_output"
    rejected = False
    try:
        _run_prediction(f3, out3, min_overlap=50.0, warn_overlap=80.0)
    except ValueError as e:
        if "REJECTED" in str(e):
            rejected = True
            logger.info("Correctly rejected: %s", e)
        else:
            raise

    results["test_reject"] = {"rejected": rejected}
    if not rejected:
        results["test_reject"]["errors"] = ["Expected rejection but prediction succeeded"]
        logger.error("FAIL: expected rejection for ~30%% overlap")

    # Verify rejection report was written
    reject_report = out3 / "prediction_report.json"
    if reject_report.exists():
        rr = json.loads(reject_report.read_text())
        if rr.get("status") != "rejected":
            results["test_reject"].setdefault("errors", []).append("Rejection report status not 'rejected'")
    else:
        results["test_reject"].setdefault("errors", []).append("No rejection report written")

    # ---- Summary ----
    elapsed = time.time() - t0
    all_pass = True
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, res in results.items():
        errs = res.get("errors", [])
        if name == "test_reject":
            ok = res.get("rejected", False) and not errs
        else:
            ok = len(errs) == 0
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        logger.info("  %s: %s (overlap=%.1f%%, decision=%s)", name, status, res.get("overlap", 0), res.get("decision", res.get("rejected", "")))
        for e in errs:
            logger.error("    ✗ %s", e)

    logger.info("Total: %.1fs | %s", elapsed, "ALL PASSED" if all_pass else "SOME FAILED")

    val_path = TEST_DIR / "validation_result.json"
    with open(val_path, "w") as f:
        json.dump({"all_passed": all_pass, "tests": results, "elapsed_s": round(elapsed, 1)}, f, indent=2)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
