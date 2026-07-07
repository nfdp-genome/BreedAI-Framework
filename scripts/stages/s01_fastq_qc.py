#!/usr/bin/env python3
"""
Stage 1: FASTQ quality control.

Wraps fastp / FastQC for raw read filtering and adapter trimming.
Only used when input_type == 'fastq'.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

TOOL = "fastp"


def is_available() -> bool:
    return shutil.which(TOOL) is not None


def run(
    fastq_r1: str,
    fastq_r2: str | None,
    outdir: str,
    threads: int = 8,
    dry_run: bool = False,
) -> dict:
    """Run FASTQ QC. Returns dict with output paths and basic stats."""
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)

    out_r1 = od / "cleaned_R1.fastq.gz"
    html_report = od / "fastp_report.html"
    json_report = od / "fastp_report.json"

    if dry_run or not is_available():
        logger.warning("fastp not available or dry-run; writing placeholders")
        json_report.write_text(json.dumps({"status": "placeholder", "tool": TOOL}))
        html_report.write_text("<html><body>Placeholder FASTQ QC report</body></html>")
        return {"status": "placeholder", "json_report": str(json_report)}

    cmd = [
        TOOL,
        "-i", fastq_r1,
        "-o", str(out_r1),
        "--html", str(html_report),
        "--json", str(json_report),
        "--thread", str(threads),
    ]
    if fastq_r2:
        out_r2 = od / "cleaned_R2.fastq.gz"
        cmd.extend(["-I", fastq_r2, "-O", str(out_r2)])

    logger.info("Running FASTQ QC: %s", " ".join(cmd))
    subprocess.check_call(cmd)

    with open(json_report) as f:
        stats = json.load(f)
    return {"status": "done", "json_report": str(json_report), "summary": stats.get("summary", {})}
