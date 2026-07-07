"""Run planner for backbone + tracks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PlanStep:
    name: str
    command: str
    required: bool = True


@dataclass
class RunPlan:
    run_id: str
    phase: int
    mode: str
    input_type: str
    outdir: Path
    steps: List[PlanStep] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "phase": self.phase,
            "mode": self.mode,
            "input_type": self.input_type,
            "outdir": str(self.outdir),
            "steps": [
                {"name": s.name, "command": s.command, "required": s.required}
                for s in self.steps
            ],
            "notes": self.notes,
        }


def build_plan(
    run_id: str,
    phase: int,
    mode: str,
    input_type: str,
    repo_root: Path,
    run_dir: Path,
    workdir: Optional[Path] = None,
) -> RunPlan:
    outdir = run_dir / "artifacts"
    report_dir = run_dir / "reports"
    wf_step2 = repo_root / "pipelines" / "step2_fastq_qc" / "main.nf"
    wf_step3 = repo_root / "pipelines" / "step3_variant_calling_standard" / "main.nf"
    wf_step4 = repo_root / "pipelines" / "step4_genotype_qc_grm_ready" / "main.nf"
    wf_step5 = repo_root / "pipelines" / "step5_imputation" / "main.nf"
    profile = "docker"
    ibex_cfg = repo_root / "configs" / "nextflow" / "ibex.config"
    work_flag = f"-work-dir {workdir}" if workdir else ""
    ibex_flag = f"-c {ibex_cfg}" if ibex_cfg.exists() else ""

    plan = RunPlan(
        run_id=run_id,
        phase=phase,
        mode=mode,
        input_type=input_type,
        outdir=outdir,
    )

    if input_type == "fastq":
        plan.steps.append(
            PlanStep(
                "step2_fastq_qc",
                f"nextflow run {wf_step2} -profile {profile} {ibex_flag} {work_flag} --outdir {outdir/'step2'}",
            )
        )
        plan.steps.append(
            PlanStep(
                "step3_variant_calling",
                f"nextflow run {wf_step3} -profile {profile} {ibex_flag} {work_flag} --outdir {outdir/'step3'}",
            )
        )

    if input_type in {"fastq", "vcf"}:
        plan.steps.append(
            PlanStep(
                "step4_genotype_qc",
                f"nextflow run {wf_step4} -profile {profile} {ibex_flag} {work_flag} --vcf {outdir/'step3'/'cohort.filtered_snps.vcf.gz'} --outdir {outdir/'step4'} --report_dir {report_dir}",
            )
        )
        plan.steps.append(
            PlanStep(
                "step5_imputation",
                f"nextflow run {wf_step5} -profile {profile} {ibex_flag} {work_flag} --vcf {outdir/'step4'/'grm_ready.vcf.gz'} --outdir {outdir/'step5'} --report_dir {report_dir}",
            )
        )

    plan.steps.append(
        PlanStep(
            "core_dataset_build",
            f"python3 {repo_root/'src'/'breedai'/'runner'/'write_core_dataset.py'} --run-dir {run_dir}",
        )
    )

    plan.steps.append(
        PlanStep(
            "default_track",
            f"python3 {repo_root/'src'/'breedai'/'runner'/'default_track.py'} --run-dir {run_dir} --phase {phase}",
        )
    )

    if mode == "default_plus_rnd":
        plan.steps.append(
            PlanStep(
                "rnd_track",
                f"python3 {repo_root/'src'/'breedai'/'runner'/'rnd_track_bridge.py'} --run-dir {run_dir} --phase {phase}",
            )
        )
    else:
        plan.notes.append("R&D track disabled (mode=default).")

    return plan

