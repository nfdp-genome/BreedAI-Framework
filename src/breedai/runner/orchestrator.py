"""Run orchestrator for breedai run/status."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from breedai.policy.resolve import resolve_policy, write_resolved_policy
from breedai.runner.planner import RunPlan, build_plan


@dataclass
class RunContext:
    repo_root: Path
    run_id: str
    run_dir: Path
    phase: int
    mode: str
    species: str
    goal: str
    input_type: str
    submit: bool
    slurm: bool
    resume: bool
    workdir: Optional[Path]


def _timestamp_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _git_commit(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root, text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _write_manifest(ctx: RunContext, policy_path: Path, plan_path: Path) -> Path:
    manifest = {
        "run_id": ctx.run_id,
        "run_dir": str(ctx.run_dir),
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(ctx.repo_root),
        "phase": ctx.phase,
        "mode": ctx.mode,
        "species": ctx.species,
        "goal": ctx.goal,
        "input_type": ctx.input_type,
        "submit": ctx.submit,
        "slurm": ctx.slurm,
        "resume": ctx.resume,
        "policy": str(policy_path),
        "plan": str(plan_path),
    }
    out = ctx.run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def _write_plan(plan: RunPlan, path: Path) -> None:
    path.write_text(json.dumps(plan.to_dict(), indent=2))


def _sbatch_script(ctx: RunContext, plan: RunPlan) -> Path:
    script = ctx.run_dir / "submit.sbatch"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=breedai_{ctx.phase}_{ctx.mode}",
        "#SBATCH --time=14-00:00:00",
        "#SBATCH --mem=64G",
        "#SBATCH --cpus-per-task=16",
        "#SBATCH --partition=batch",
        "#SBATCH --account=YOUR_SLURM_ACCOUNT",
        f"#SBATCH --output={ctx.run_dir}/slurm_%j.out",
        f"#SBATCH --error={ctx.run_dir}/slurm_%j.err",
        "set -euo pipefail",
        f"cd \"{ctx.repo_root}\"",
        "",
    ]
    for step in plan.steps:
        cmd = step.command
        if ctx.resume and cmd.startswith("nextflow run"):
            cmd = cmd + " -resume"
        lines += [f"echo '>> {step.name}'", cmd, ""]
    script.write_text("\n".join(lines) + "\n")
    return script


def create_run(
    repo_root: Path,
    phase: int,
    mode: str,
    species: str,
    goal: str,
    input_type: str,
    run_id: Optional[str] = None,
    submit: bool = False,
    slurm: bool = True,
    dry_run: bool = False,
    resume: bool = False,
    workdir: Optional[Path] = None,
    overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    run_id = run_id or _timestamp_run_id()
    run_dir = repo_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)

    ctx = RunContext(
        repo_root=repo_root,
        run_id=run_id,
        run_dir=run_dir,
        phase=phase,
        mode=mode,
        species=species,
        goal=goal,
        input_type=input_type,
        submit=submit,
        slurm=slurm,
        resume=resume,
        workdir=workdir,
    )

    policy = resolve_policy(repo_root, species, goal, overrides=overrides or {})
    policy_path = run_dir / "resolved_policy.yaml"
    write_resolved_policy(policy, policy_path)

    plan = build_plan(
        run_id=run_id,
        phase=phase,
        mode=mode,
        input_type=input_type,
        repo_root=repo_root,
        run_dir=run_dir,
        workdir=workdir,
    )
    plan_path = run_dir / "plan.json"
    _write_plan(plan, plan_path)
    manifest_path = _write_manifest(ctx, policy_path, plan_path)
    sbatch_path = _sbatch_script(ctx, plan)

    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "policy": str(policy_path),
        "plan": str(plan_path),
        "sbatch": str(sbatch_path),
        "steps": [s.name for s in plan.steps],
    }

    if dry_run:
        result["status"] = "dry_run_prepared"
        return result

    if submit and slurm:
        job_id = subprocess.check_output(["sbatch", "--parsable", str(sbatch_path)], text=True).strip()
        result["status"] = "submitted"
        result["job_id"] = job_id
        return result

    # Local non-slurm execution
    for step in plan.steps:
        cmd = step.command + (" -resume" if (resume and step.command.startswith("nextflow run")) else "")
        subprocess.check_call(cmd, shell=True, cwd=repo_root)
    result["status"] = "completed_local"
    return result


def status(repo_root: Path, run_id: str) -> Dict[str, object]:
    run_dir = repo_root / "runs" / run_id
    manifest = run_dir / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Run not found: {run_id}")
    data = json.loads(manifest.read_text())
    data["exists"] = True
    data["artifacts_dir"] = str(run_dir / "artifacts")
    data["reports_dir"] = str(run_dir / "reports")
    return data

