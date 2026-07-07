from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from breedai.policy.resolve import resolve_policy


def test_policy_resolver_merges_base_and_species_goal():
    p = resolve_policy(REPO_ROOT, "cattle", "growth")
    assert p.species == "cattle"
    assert p.goal == "growth"
    assert p.reference_build
    assert p.step3.caller == "gatk_gvcf_joint"
    assert p.step5.method == "beagle"


def test_policy_resolver_applies_overrides():
    p = resolve_policy(
        REPO_ROOT,
        "cattle",
        "milk",
        overrides={"step4.min_maf": 0.05, "modeling.cv_mode_default": "random_kfold"},
    )
    assert abs(p.step4.min_maf - 0.05) < 1e-12
    assert p.modeling.cv_mode_default == "random_kfold"

