from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from breedai.runner.planner import build_plan


def test_fastq_route_includes_step2_step3_step4_step5():
    run_dir = REPO_ROOT / "tmp_test_plan"
    run_dir.mkdir(exist_ok=True)
    plan = build_plan(
        run_id="test",
        phase=1,
        mode="default",
        input_type="fastq",
        repo_root=REPO_ROOT,
        run_dir=run_dir,
    )
    names = [s.name for s in plan.steps]
    assert "step2_fastq_qc" in names
    assert "step3_variant_calling" in names
    assert "step4_genotype_qc" in names
    assert "step5_imputation" in names
    assert "default_track" in names


def test_default_plus_rnd_includes_rnd_track():
    run_dir = REPO_ROOT / "tmp_test_plan_rnd"
    run_dir.mkdir(exist_ok=True)
    plan = build_plan(
        run_id="test",
        phase=1,
        mode="default_plus_rnd",
        input_type="vcf",
        repo_root=REPO_ROOT,
        run_dir=run_dir,
    )
    names = [s.name for s in plan.steps]
    assert "rnd_track" in names

