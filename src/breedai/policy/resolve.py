"""Policy resolver: merge base + species/goal + CLI overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from .schema import PolicyModel


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_nested(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def resolve_policy(
    repo_root: Path,
    species: str,
    goal: str,
    overrides: Dict[str, Any] | None = None,
) -> PolicyModel:
    policy_dir = repo_root / "configs" / "policies"
    base_cfg = load_yaml(policy_dir / "_base.yaml")
    species_cfg = load_yaml(policy_dir / f"{species}_{goal}.yaml")
    merged = _deep_merge(base_cfg, species_cfg)
    merged["species"] = species
    merged["goal"] = goal

    for k, v in (overrides or {}).items():
        _set_nested(merged, k, v)

    return PolicyModel(**merged)


def write_resolved_policy(policy: PolicyModel, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(policy.model_dump(), f, sort_keys=False)

