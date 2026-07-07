#!/usr/bin/env python3
"""Default literature/production track placeholder (GBLUP/ssGBLUP)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--phase", type=int, required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out = run_dir / "reports" / "default"
    out.mkdir(parents=True, exist_ok=True)
    pred = run_dir / "artifacts" / "predictions.csv"
    model_card = run_dir / "artifacts" / "model_card.json"
    bundle = run_dir / "reports" / "default" / "step8_10.html"

    pd.DataFrame(columns=["animal_id", "trait", "GEBV", "rank", "zscore"]).to_csv(pred, index=False)
    model_card.write_text(
        json.dumps(
            {
                "track": "default",
                "phase": args.phase,
                "models": ["GBLUP", "ssGBLUP_if_pedigree"],
                "note": "Scaffold track. Integrate production solver next.",
            },
            indent=2,
        )
    )
    bundle.write_text(
        "<html><body><h1>Default Track Report</h1><p>Scaffold generated. Integrate GBLUP/ssGBLUP execution next.</p></body></html>"
    )


if __name__ == "__main__":
    main()

