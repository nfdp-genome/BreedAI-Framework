#!/usr/bin/env python3
"""
Stage 9: Fixed-effects model setup.

Identifies and prepares fixed-effect covariates (herd, year-season,
age, sex, breed) for the mixed-model.  When metadata is unavailable,
generates a minimal spec with intercept only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CATEGORICAL = ["herd", "breed", "sex", "year_season"]
DEFAULT_CONTINUOUS = ["age", "parity", "days_in_milk"]


def build_fixed_effects_spec(
    metadata_file: str | None,
    sample_ids: list[str],
    outdir: str | Path,
) -> dict:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    spec = {
        "intercept": True,
        "categorical": [],
        "continuous": [],
        "n_samples": len(sample_ids),
    }

    if not metadata_file or not Path(metadata_file).exists():
        logger.info("No metadata — intercept-only fixed-effects spec")
        (od / "fixed_effects.json").write_text(json.dumps(spec, indent=2))
        return spec

    meta = pd.read_csv(metadata_file)
    meta_cols = [c.lower() for c in meta.columns]
    for col in DEFAULT_CATEGORICAL:
        if col in meta_cols:
            spec["categorical"].append(col)
    for col in DEFAULT_CONTINUOUS:
        if col in meta_cols:
            spec["continuous"].append(col)

    logger.info("Fixed effects: cat=%s, cont=%s", spec["categorical"], spec["continuous"])
    (od / "fixed_effects.json").write_text(json.dumps(spec, indent=2))

    # Build design matrix X_fixed if effects exist
    if spec["categorical"] or spec["continuous"]:
        meta.index = meta.iloc[:, 0].astype(str)
        meta = meta.reindex(sample_ids)
        X_fixed = pd.get_dummies(meta[spec["categorical"]], drop_first=True) if spec["categorical"] else pd.DataFrame(index=sample_ids)
        for col in spec["continuous"]:
            if col in meta.columns:
                X_fixed[col] = meta[col].values
        X_fixed.to_csv(od / "X_fixed.csv")
        spec["n_effects"] = X_fixed.shape[1]

    return spec
