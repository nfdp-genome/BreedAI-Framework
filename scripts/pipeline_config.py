#!/usr/bin/env python3
"""
Load and validate BreedAI run configuration.

Reads a YAML config, merges with defaults, resolves paths relative to
PROJECT_ROOT, and returns a plain dict usable by every stage script.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "run_config_template.yaml"

REQUIRED_KEYS = ["species", "breeding_goal", "input_type", "mode", "dataset"]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


def _resolve_paths(cfg: dict, root: Path) -> dict:
    """Resolve dataset file paths relative to PROJECT_ROOT."""
    ds = cfg.get("dataset", {})
    for key in ("geno_file", "pheno_file", "pedigree_file", "metadata_file"):
        val = ds.get(key, "")
        if val and not Path(val).is_absolute():
            ds[key] = str(root / val)
    cfg["dataset"] = ds
    return cfg


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    """Load run config YAML, merge with defaults, resolve paths."""
    if not YAML_AVAILABLE:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")

    base = {}
    if DEFAULT_CONFIG.exists():
        with open(DEFAULT_CONFIG) as f:
            base = yaml.safe_load(f) or {}

    user = {}
    if config_path:
        p = Path(config_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {p}")
        with open(p) as f:
            user = yaml.safe_load(f) or {}

    cfg = _deep_merge(base, user)

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    for key in REQUIRED_KEYS:
        if key not in cfg or not cfg[key]:
            raise ValueError(f"Missing required config key: {key}")

    cfg = _resolve_paths(cfg, PROJECT_ROOT)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def save_resolved_config(cfg: dict, out_path: str | Path) -> None:
    """Dump the fully resolved config to YAML for reproducibility."""
    if not YAML_AVAILABLE:
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        return
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def input_is_fastq(cfg: dict) -> bool:
    return cfg.get("input_type", "").lower() == "fastq"


def input_is_genotype(cfg: dict) -> bool:
    return cfg.get("input_type", "").lower() in ("genotype", "plink", "vcf")


def mode_includes_rnd(cfg: dict) -> bool:
    return cfg.get("mode", "default") == "default_plus_rnd"


def has_pedigree(cfg: dict) -> bool:
    p = cfg.get("dataset", {}).get("pedigree_file", "")
    return bool(p) and Path(p).exists()


def build_config_from_dataset_input(
    mode: str = "default_plus_rnd",
    species: str = "cattle",
    breeding_goal: str = "growth",
    project_root: Optional[Path] = None,
) -> dict:
    """
    Build a run config that reads genotype/phenotype from the standard runtime folder:
      <project_root>/input/Geno.csv
      <project_root>/input/Pheno.csv

    Optional files (used when present):
      input/metadata.csv
      input/pedigree.csv
    """
    root = project_root or PROJECT_ROOT
    inp = root / "input"
    geno = inp / "Geno.csv"
    pheno = inp / "Pheno.csv"
    missing = [str(p) for p in (geno, pheno) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Standard input files not found. Place Geno.csv and Pheno.csv under "
            f"{inp}. Missing: {missing}"
        )
    meta = inp / "metadata.csv"
    ped = inp / "pedigree.csv"
    cfg = {
        "species": species,
        "breeding_goal": breeding_goal,
        "input_type": "genotype",
        "dataset": {
            "source": "dataset_input",
            "geno_file": str(geno.resolve()),
            "pheno_file": str(pheno.resolve()),
            "pedigree_file": str(ped.resolve()) if ped.exists() else "",
            "metadata_file": str(meta.resolve()) if meta.exists() else "",
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
                "GBLUP_Ridge",
                "LASSO",
                "ElasticNet",
                "BayesianRidge",
                "RandomForest",
                "SVR_RBF",
                "NeuralNet_MLP",
                "BayesA",
                "BayesB",
                "BayesCpi",
                "XGBoost",
                "LightGBM",
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
            # sbatch reads SBATCH_ACCOUNT from the environment; leave empty to use the
            # cluster default. Set it via: export SBATCH_ACCOUNT=<your-account>
            "account": os.environ.get("SBATCH_ACCOUNT", ""),
            "mem": "64G",
            "cpus_per_task": 32,
            "time": "14-00:00:00",
        },
        "_input_workflow": {
            "source": "dataset_input",
            "input_dir": str(inp.resolve()),
        },
    }
    return cfg


if __name__ == "__main__":
    import argparse, pprint
    ap = argparse.ArgumentParser(description="Validate a BreedAI run config")
    ap.add_argument("config", nargs="?", help="Path to run config YAML")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pprint.pprint(cfg)
