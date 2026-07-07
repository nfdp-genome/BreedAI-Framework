#!/usr/bin/env python3
"""BreedAI run/status CLI for backbone orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from breedai.runner.orchestrator import create_run, status


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_overrides(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set override: {item}. Expected dotted.key=value")
        k, v = item.split("=", 1)
        if v.lower() in {"true", "false"}:
            parsed: object = v.lower() == "true"
        else:
            try:
                parsed = int(v)
            except ValueError:
                try:
                    parsed = float(v)
                except ValueError:
                    parsed = v
        out[k] = parsed
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="BreedAI CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Create/submit breeding pipeline run")
    run_p.add_argument("--phase", type=int, choices=[1, 2], required=True)
    run_p.add_argument("--mode", choices=["default", "default_plus_rnd"], default="default")
    run_p.add_argument("--species", required=True)
    run_p.add_argument("--goal", required=True)
    run_p.add_argument("--input-type", choices=["fastq", "vcf", "plink"], required=True)
    run_p.add_argument("--run-id", default=None)
    run_p.add_argument("--submit", action="store_true")
    run_p.add_argument("--slurm", action="store_true", default=True)
    run_p.add_argument("--dry-run", action="store_true")
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--workdir", default=None)
    run_p.add_argument("--set", action="append", default=[], help="Policy override dotted.key=value")

    status_p = sub.add_parser("status", help="Show run metadata")
    status_p.add_argument("--run", required=True)

    args = parser.parse_args()
    repo_root = _repo_root()

    if args.cmd == "run":
        overrides = _parse_overrides(args.set)
        result = create_run(
            repo_root=repo_root,
            phase=args.phase,
            mode=args.mode,
            species=args.species,
            goal=args.goal,
            input_type=args.input_type,
            run_id=args.run_id,
            submit=args.submit,
            slurm=args.slurm,
            dry_run=args.dry_run,
            resume=args.resume,
            workdir=Path(args.workdir) if args.workdir else None,
            overrides=overrides,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "status":
        data = status(repo_root, args.run)
        print(json.dumps(data, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())

