#!/usr/bin/env python3
"""Create core_dataset contract placeholders from available artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    core = run_dir / "artifacts" / "core_dataset"
    core.mkdir(parents=True, exist_ok=True)

    # Optional passthrough from step4/step5 outputs if present.
    step5 = run_dir / "artifacts" / "step5"
    step4 = run_dir / "artifacts" / "step4"
    src = step5 if (step5 / "grm_ready.pgen").exists() else step4

    for ext in ("pgen", "pvar", "psam"):
        src_file = src / f"grm_ready.{ext}"
        dst_file = core / f"genotypes.{ext}"
        if src_file.exists():
            dst_file.write_bytes(src_file.read_bytes())
        else:
            dst_file.write_text("")

    # Contract placeholders (safe defaults; downstream fills/overwrites).
    np.save(core / "X.npy", np.empty((0, 0), dtype=np.float32))
    pd.DataFrame(columns=["chr", "pos", "id", "a1", "a2", "maf", "callrate"]).to_csv(
        core / "snp_info.tsv", sep="\t", index=False
    )
    pd.DataFrame(columns=["sample_id", "animal_id", "batch", "breed", "herd", "sex"]).to_csv(
        core / "samples.tsv", sep="\t", index=False
    )
    pd.DataFrame(columns=["animal_id"]).to_csv(core / "phenotypes.csv", index=False)
    (core / "splits.json").write_text(json.dumps({"mode": "unassigned", "folds": []}, indent=2))
    np.save(core / "G.npy", np.empty((0, 0), dtype=np.float32))
    (core / "H.meta.json").write_text(json.dumps({"status": "skipped", "reason": "pedigree not provided"}, indent=2))
    (core / "qc_summary.json").write_text(json.dumps({"status": "placeholder"}, indent=2))
    (core / "versions.json").write_text(json.dumps({"status": "placeholder"}, indent=2))


if __name__ == "__main__":
    main()

