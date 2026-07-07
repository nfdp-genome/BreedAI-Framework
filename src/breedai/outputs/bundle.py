"""Standard output bundle writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


def write_bundle(run_dir: Path, model_card: Dict[str, object]) -> None:
    artifacts = run_dir / "artifacts"
    reports = run_dir / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    (artifacts / "predictions.csv").write_text("animal_id,trait,GEBV,rank,zscore\n")
    (artifacts / "qc_flags.csv").write_text("entity_id,entity_type,reason\n")
    (artifacts / "model_card.json").write_text(json.dumps(model_card, indent=2))

    index = reports / "index.html"
    index.write_text(
        "<html><body><h1>BreedAI Report Index</h1><ul>"
        "<li><a href='default/step8_10.html'>Default track</a></li>"
        "<li><a href='rnd/step8_10.html'>R&D track</a></li>"
        "</ul></body></html>"
    )

