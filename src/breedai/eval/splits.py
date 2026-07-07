"""Cross-validation split registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def random_kfold_indices(n: int, k: int, seed: int = 42) -> List[Dict[str, List[int]]]:
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [{"train": tr.tolist(), "test": te.tolist()} for tr, te in kf.split(np.arange(n))]


def forward_time_indices(df: pd.DataFrame, time_col: str, k: int) -> List[Dict[str, List[int]]]:
    sdf = df.sort_values(time_col).reset_index(drop=True)
    fold_sizes = np.full(k, len(sdf) // k, dtype=int)
    fold_sizes[: len(sdf) % k] += 1
    starts = np.cumsum(np.concatenate([[0], fold_sizes[:-1]]))
    splits = []
    for i, st in enumerate(starts):
        te_idx = np.arange(st, st + fold_sizes[i])
        tr_idx = np.arange(0, st)
        if len(tr_idx) == 0:
            tr_idx = np.arange(st + fold_sizes[i], len(sdf))
        splits.append({"train": tr_idx.tolist(), "test": te_idx.tolist()})
    return splits


def save_splits(splits: List[Dict[str, List[int]]], out_path: Path, mode: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"mode": mode, "folds": splits}, indent=2))

