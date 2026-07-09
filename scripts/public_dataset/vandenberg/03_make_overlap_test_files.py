#!/usr/bin/env python3
"""
Generate the three SNP-overlap "safeguard" test files for Phase 2.

These are synthetic new-animal genotype files derived from input/Geno.csv, with
markers dropped/added and columns shuffled to simulate animals genotyped on a
different chip. They demonstrate BreedAI's deployment SNP-overlap guardrails:

    Geno_test1_normal_overlap.csv   ~98% overlap -> pass cleanly
    Geno_test2_low_overlap.csv      ~60% overlap -> warn but still predict
    Geno_test3_very_low_overlap.csv ~30% overlap -> reject (< 50% threshold)

Runs from any directory; paths resolve against the repo root. By default it reads
input/Geno.csv and writes the three files into input/.

    python scripts/public_dataset/vandenberg/03_make_overlap_test_files.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root: .../scripts/public_dataset/vandenberg/03_make_overlap_test_files.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p
    return (p if p.exists() else (REPO_ROOT / p)).resolve() if p.parts else REPO_ROOT / p


def make_test_file(geno: pd.DataFrame, name: str, n_holdout: int, n_drop: int,
                   n_extra: int, seed: int, out_dir: Path) -> Path:
    """Build one overlap-test file from the training genotypes."""
    np.random.seed(seed)
    original_snps = list(geno.columns)
    n_total_snps = len(original_snps)

    holdout_idx = np.random.choice(len(geno), size=n_holdout, replace=False)
    new_geno = geno.iloc[holdout_idx].copy()
    new_geno.index = [f"NEW_{i + 1}" for i in range(n_holdout)]

    # Drop some of the original SNPs (reduces overlap)
    if n_drop > 0:
        drop_idx = set(np.random.choice(n_total_snps, size=min(n_drop, n_total_snps - 1),
                                        replace=False).tolist())
        keep_snps = [s for i, s in enumerate(original_snps) if i not in drop_idx]
        new_geno = new_geno[keep_snps]

    # Add fake extra SNPs the reference panel doesn't have
    if n_extra > 0:
        extra = np.random.choice([0, 1, 2], size=(n_holdout, n_extra))
        extra_df = pd.DataFrame(extra, index=new_geno.index,
                                columns=[f"EXTRA_{i}" for i in range(n_extra)])
        new_geno = pd.concat([new_geno, extra_df], axis=1)

    # Shuffle column order (order must not matter to the pipeline)
    cols = list(new_geno.columns)
    np.random.shuffle(cols)
    out = out_dir / f"{name}.csv"
    new_geno[cols].to_csv(out)
    common = len(set(new_geno.columns) & set(original_snps))
    print(f"  {out.name:34s} {n_holdout:>3d} animals, {new_geno.shape[1]:>6d} SNPs, "
          f"{common:>6d} shared (~{100 * common / n_total_snps:.0f}% overlap)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geno-file", default="input/Geno.csv",
                    help="Reference genotypes to derive from (default: input/Geno.csv)")
    ap.add_argument("--output-dir", default="input",
                    help="Where to write the test files (default: input)")
    args = ap.parse_args()

    geno_path = _resolve(args.geno_file)
    out_dir = _resolve(args.output_dir)
    if not geno_path.exists():
        raise SystemExit(f"ERROR: reference genotypes not found: {geno_path}\n"
                         f"Build input/Geno.csv first (see cattle_dataset/README.md).")
    out_dir.mkdir(parents=True, exist_ok=True)

    geno = pd.read_csv(geno_path, index_col=0)
    print(f"Reference: {geno_path} ({geno.shape[0]} animals x {geno.shape[1]} SNPs)")
    print("Writing SNP-overlap test files:")
    n = geno.shape[1]
    make_test_file(geno, "Geno_test1_normal_overlap", 50, 500, 200, 42, out_dir)
    make_test_file(geno, "Geno_test2_low_overlap", 30, int(n * 0.40), 100, 77, out_dir)
    make_test_file(geno, "Geno_test3_very_low_overlap", 20, int(n * 0.70), 50, 55, out_dir)
    print("Done. Feed each to Phase 2 to see pass / warn / reject.")


if __name__ == "__main__":
    main()
