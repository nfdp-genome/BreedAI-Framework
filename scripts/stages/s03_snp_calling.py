#!/usr/bin/env python3
"""
Stage 3: Joint SNP calling.

Wraps GATK HaplotypeCaller (gVCF mode) + GenomicsDBImport + GenotypeGVCFs,
or a simpler bcftools mpileup/call path.
Only used when input_type == 'fastq'.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def is_available(caller: str = "gatk_gvcf_joint") -> bool:
    if caller == "gatk_gvcf_joint":
        return shutil.which("gatk") is not None
    return shutil.which("bcftools") is not None


def run(
    bam_list: list[str],
    reference: str,
    outdir: str,
    caller: str = "gatk_gvcf_joint",
    known_sites: str | None = None,
    threads: int = 8,
    dry_run: bool = False,
) -> dict:
    od = Path(outdir)
    od.mkdir(parents=True, exist_ok=True)
    vcf_out = od / "cohort.filtered_snps.vcf.gz"

    if dry_run or not is_available(caller):
        logger.warning("Caller '%s' not available or dry-run; writing placeholder VCF", caller)
        vcf_out.write_bytes(b"")
        meta = {"status": "placeholder", "caller": caller, "vcf": str(vcf_out)}
        (od / "calling_meta.json").write_text(json.dumps(meta))
        return meta

    if caller == "bcftools":
        bam_args = " ".join(bam_list)
        cmd = (
            f"bcftools mpileup -Ou -f {reference} {bam_args} | "
            f"bcftools call -mv -Oz -o {vcf_out}"
        )
        logger.info("Calling SNPs (bcftools): %s", cmd)
        subprocess.check_call(cmd, shell=True)
    else:
        gvcf_dir = od / "gvcfs"
        gvcf_dir.mkdir(exist_ok=True)
        gvcfs = []
        for bam in bam_list:
            sid = Path(bam).stem.replace(".sorted", "")
            g = gvcf_dir / f"{sid}.g.vcf.gz"
            cmd = [
                "gatk", "HaplotypeCaller",
                "-R", reference, "-I", bam,
                "-O", str(g), "-ERC", "GVCF",
                "--native-pair-hmm-threads", str(threads),
            ]
            if known_sites:
                cmd.extend(["--dbsnp", known_sites])
            logger.info("HaplotypeCaller: %s", sid)
            subprocess.check_call(cmd)
            gvcfs.append(str(g))

        db_path = od / "genomicsdb"
        sample_map = od / "sample_map.txt"
        with open(sample_map, "w") as f:
            for g in gvcfs:
                sid = Path(g).stem.replace(".g.vcf", "")
                f.write(f"{sid}\t{g}\n")
        subprocess.check_call([
            "gatk", "GenomicsDBImport",
            "--sample-name-map", str(sample_map),
            "--genomicsdb-workspace-path", str(db_path),
            "-L", "1",
        ])
        subprocess.check_call([
            "gatk", "GenotypeGVCFs",
            "-R", reference,
            "-V", f"gendb://{db_path}",
            "-O", str(vcf_out),
        ])

    subprocess.check_call(["tabix", "-f", "-p", "vcf", str(vcf_out)])
    return {"status": "done", "vcf": str(vcf_out)}
