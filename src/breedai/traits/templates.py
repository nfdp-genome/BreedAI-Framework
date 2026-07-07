"""Species/goal fixed-effect templates."""

from __future__ import annotations

from typing import Dict, List


def template_for(species: str, goal: str) -> Dict[str, List[str]]:
    common = ["sex", "herd", "batch"]
    if species == "cattle" and goal == "milk":
        return {"fixed_effects": common + ["parity", "days_in_milk", "year_season", "pc1", "pc2"]}
    if species == "cattle" and goal == "growth":
        return {"fixed_effects": common + ["age", "year_season", "pc1", "pc2"]}
    return {"fixed_effects": common}

