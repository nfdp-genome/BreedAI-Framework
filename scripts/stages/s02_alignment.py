#!/usr/bin/env python3
"""
Stage 2: Read alignment to reference genome.

Wraps bwa-mem2 + samtools for FASTQ → sorted BAM.
Only used when input_type == 'fastq'.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def is_available() -> bool:
    return shutil.which("bwa-mem2") is not None and shutil.which("samtools") is not None


def run(
    fastq_r1: str,
    fastq_r2: str | None,
    reference: str,
    outdir: str,
    sample_id: str = "sample",
    threads: int = 8,
    dry_run: bool = False,
) -> dict:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    bam = od / f"{sample_id}.sorted.bam"

    if dry_run or not is_available():
        logger.warning("bwa-mem2/samtools not available or dry-run; writing placeholder")
        meta = {"status": "placeholder", "tool": "bwa-mem2", "bam": str(bam)}
        (od / "alignment_meta.json").write_text(json.dumps(meta))
        return meta

    rg = f"@RG\\tID:{sample_id}\\tSM:{sample_id}\\tPL:ILLUMINA"
    reads = [fastq_r1] + ([fastq_r2] if fastq_r2 else [])
    align_cmd = f'bwa-mem2 mem -t {threads} -R "{rg}" {reference} {" ".join(reads)}'
    sort_cmd = f"samtools sort -@ {threads} -o {bam}"
    full_cmd = f"{align_cmd} | {sort_cmd}"

    logger.info("Aligning: %s", full_cmd)
    subprocess.check_call(full_cmd, shell=True)
    subprocess.check_call(["samtools", "index", "-@", str(threads), str(bam)])

    return {"status": "done", "bam": str(bam)}
