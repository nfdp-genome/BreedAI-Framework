"""Genotype IO helpers (PLINK2-centric)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_psam(psam_path: Path) -> pd.DataFrame:
    return pd.read_csv(psam_path, delim_whitespace=True)


def load_pvar(pvar_path: Path) -> pd.DataFrame:
    return pd.read_csv(pvar_path, delim_whitespace=True)


def save_x_cache(x: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, x)


def load_x_cache(x_path: Path) -> np.ndarray:
    return np.load(x_path)


def allele_freq_from_matrix(x: np.ndarray) -> np.ndarray:
    """Estimate allele frequency from 0/1/2 genotype matrix."""
    if x.size == 0:
        return np.array([])
    return x.mean(axis=0) / 2.0


def create_empty_contract_tables(out_dir: Path) -> Tuple[Path, Path]:
    snp_path = out_dir / "snp_info.tsv"
    samples_path = out_dir / "samples.tsv"
    pd.DataFrame(columns=["chr", "pos", "id", "a1", "a2", "maf", "callrate"]).to_csv(
        snp_path, sep="\t", index=False
    )
    pd.DataFrame(columns=["sample_id", "animal_id", "batch", "breed", "herd", "sex"]).to_csv(
        samples_path, sep="\t", index=False
    )
    return snp_path, samples_path

