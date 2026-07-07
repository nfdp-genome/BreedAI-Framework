"""Fixed-effects matrix builder."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def build_fixed_effects_spec(df: pd.DataFrame, template: Dict[str, List[str]]) -> Dict[str, object]:
    cols = [c for c in template.get("fixed_effects", []) if c in df.columns]
    sparse = {}
    for c in cols:
        if df[c].dtype == object:
            vc = df[c].value_counts()
            sparse[c] = int((vc < 5).sum())
    return {"columns_used": cols, "sparsity_warnings": sparse}


def write_model_card_fixed_effects(spec: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(spec).to_json(out_path, indent=2)

