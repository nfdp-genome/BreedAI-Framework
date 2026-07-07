"""ssGBLUP scaffolding utilities."""

from __future__ import annotations

import json
from pathlib import Path


def write_h_skipped(out_path: Path, reason: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"status": "skipped", "reason": reason}, indent=2))

